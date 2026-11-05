"""One bounded supervised demo from confirmed counterfactual order labels.

The frozen RLOO actor is copied into a new, independent optimizer. Four response
sequences per update balance winning-order labels with normal-request replay.
FP32 weights and GPU Adam moments persist for all sixteen updates; the final
BF16 inference actor is streamed to disk without another large resume snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from src.agents.rloo_terminal_train import export_bf16, optimizer_steps, token_logprobs
from src.agents.same_prefix_training_demo import append, probe_delta, probes, require, write

PROFILE = "counterfactual_winner_supervised_replay_fixed16_v1"
UPDATES = 16
ROWS_PER_KIND = 2
LEARNING_RATE = 2e-6


def load_rows(path: Path, kind: str) -> list[dict]:
    """Keep actual token sequences; never truncate or regenerate a label."""
    rows = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            source = json.loads(line)
            prompt = source.get("prompt_token_ids", source.get("prompt_ids"))
            response = source.get("response_token_ids", source.get("response_ids"))
            for name, values, limit in (
                ("prompt", prompt, 8192),
                ("response", response, 4096),
            ):
                require(
                    isinstance(values, list)
                    and 0 < len(values) <= limit
                    and all(type(value) is int and value >= 0 for value in values),
                    f"{path.name} row {index}: invalid {name} tokens",
                )
            rows.append(
                {
                    **source,
                    "prompt_token_ids": prompt,
                    "response_token_ids": response,
                    "kind": kind,
                    "source_row": index,
                }
            )
    return rows


def make_schedule(winners: list[dict], replay: list[dict], seed: int) -> list[list[dict]]:
    require(bool(winners) and bool(replay), "winner labels and normal-request replay required")
    rng = random.Random(seed)
    winners, replay = list(winners), list(replay)
    rng.shuffle(winners)
    rng.shuffle(replay)
    return [
        [
            rows[(step * ROWS_PER_KIND + offset) % len(rows)]
            for rows in (winners, replay)
            for offset in range(ROWS_PER_KIND)
        ]
        for step in range(UPDATES)
    ]


def fit_nll(model, rows: list[dict], torch) -> dict:
    """Describe fitting on fixed training examples, not a held-out metric."""
    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            value = float(-token_logprobs(model, row, torch).mean())
            require(math.isfinite(value), "nonfinite diagnostic label NLL")
            losses.append(value)
    return dict(
        rows=len(rows),
        mean_response_token_NLL=math.fsum(losses) / len(losses),
        per_row_response_token_NLL=losses,
        held_out=False,
    )


def run(args) -> None:
    require(args.updates == UPDATES, "the approved demo is exactly sixteen optimizer updates")
    output, model_path, collection = (
        args.output.resolve(),
        args.model.resolve(),
        args.collection.resolve(),
    )
    require(
        model_path != output and not model_path.is_relative_to(output),
        "training output must not contain the frozen input actor",
    )
    output.mkdir(parents=True, exist_ok=True)
    actor = output / "current_actor"
    require(
        not (output / "training_summary.json").exists() and not actor.exists(),
        "fresh supervised run required; no silent restart or actor overwrite",
    )
    winners = load_rows(collection / "training_samples.jsonl", "counterfactual_winner")
    replay = load_rows(collection / "replay_samples.jsonl", "replay")
    schedule = make_schedule(winners, replay, args.seed)
    diagnostic_replay = list(replay)
    random.Random(args.seed).shuffle(diagnostic_replay)
    diagnostic_replay = diagnostic_replay[:32]
    started = time.monotonic()
    summary = dict(
        profile=PROFILE,
        status="STARTING",
        seed=args.seed,
        model_path=str(model_path),
        collection_path=str(collection),
        training_samples=str(collection / "training_samples.jsonl"),
        replay_samples=str(collection / "replay_samples.jsonl"),
        winner_labels=len(winners),
        normal_replay_rows=len(replay),
        planned_optimizer_steps=UPDATES,
        optimizer_updates=0,
        completed_updates=0,
        actor_updates=0,
        training_rows=0,
        winner_training_rows=0,
        replay_training_rows=0,
        critic_updates=0,
        learning_rate=LEARNING_RATE,
        loss="response-token mean cross entropy, four sequences equally weighted",
        per_update_winner_rows=ROWS_PER_KIND,
        per_update_replay_rows=ROWS_PER_KIND,
        per_update_sequences=2 * ROWS_PER_KIND,
        sample_schedule="seeded shuffle per kind; deterministic cycling, no label resampling",
        optimizer_independent_of_RLOO=True,
        optimizer_reset_between_updates=False,
        FP32_master_preserved_during_updates=True,
        optimizer_moments_device="cuda",
        persistent_full_training_state_saved=False,
        on_policy=False,
        PPO=False,
        critic=False,
        held_out_training_metric=False,
        future_labels_in_inference_prompt=False,
        actor_checkpoint=str(actor),
        updates=[],
    )
    write(output / "training_summary.json", summary)
    try:
        os.environ.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.eval()
        before_bf16 = probes(model)
        summary["status"] = "INITIAL_LABEL_NLL"
        write(output / "training_summary.json", summary)
        summary["before_NLL"] = dict(
            winners=fit_nll(model, winners, torch),
            normal_replay=fit_nll(model, diagnostic_replay, torch),
        )
        model.float()
        before_fp32 = probes(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        summary["status"] = "SUPERVISED_UPDATES"
        write(output / "training_summary.json", summary)
        with (output / "optimizer_updates.jsonl").open("x") as stream:
            for step, rows in enumerate(schedule, 1):
                tick = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for row in rows:
                    nll = -token_logprobs(model, row, torch).mean()
                    require(bool(torch.isfinite(nll)), "nonfinite supervised response loss")
                    (nll / len(rows)).backward()
                    losses.append(float(nll.detach()))
                    del nll
                norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                )
                require(norm > 0, "nonzero gradient required for an effective update")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                require(optimizer_steps(optimizer) == (step, step), "Adam update count reset")
                event = dict(
                    optimizer_step=step,
                    sequences=len(rows),
                    winner_sequences=ROWS_PER_KIND,
                    replay_sequences=ROWS_PER_KIND,
                    mean_response_token_NLL=math.fsum(losses) / len(losses),
                    winner_NLL=math.fsum(losses[:ROWS_PER_KIND]) / ROWS_PER_KIND,
                    replay_NLL=math.fsum(losses[ROWS_PER_KIND:]) / ROWS_PER_KIND,
                    gradient_norm_before_clip=norm,
                    prompt_tokens=sum(len(row["prompt_token_ids"]) for row in rows),
                    response_tokens=sum(len(row["response_token_ids"]) for row in rows),
                    selected_rows=[
                        dict(
                            kind=row["kind"],
                            source_row=row["source_row"],
                            anchor_id=row.get("anchor_id"),
                        )
                        for row in rows
                    ],
                    seconds=time.monotonic() - tick,
                    Adam_step_min=step,
                    Adam_step_max=step,
                )
                append(stream, event)
                summary["updates"].append(event)
                summary.update(
                    optimizer_updates=step,
                    completed_updates=step,
                    actor_updates=step,
                    training_rows=step * 2 * ROWS_PER_KIND,
                    winner_training_rows=step * ROWS_PER_KIND,
                    replay_training_rows=step * ROWS_PER_KIND,
                )
                write(output / "training_summary.json", summary)
                print(json.dumps(dict(event="SUPERVISED_UPDATE_COMPLETE", **event)), flush=True)
        model.gradient_checkpointing_disable()
        model.eval()
        summary.update(status="FINAL_LABEL_NLL")
        write(output / "training_summary.json", summary)
        summary["after_NLL"] = dict(
            winners=fit_nll(model, winners, torch),
            normal_replay=fit_nll(model, diagnostic_replay, torch),
        )
        after_fp32 = probes(model)
        expected_bf16 = {
            key: torch.tensor(value, dtype=torch.bfloat16).float().tolist()
            for key, value in after_fp32.items()
        }
        summary.update(
            status="EXPORTING",
            FP32_parameter_delta=probe_delta(before_fp32, after_fp32),
            BF16_parameter_delta=probe_delta(before_bf16, expected_bf16),
        )
        write(output / "training_summary.json", summary)
        summary["checkpoint_readback"] = export_bf16(
            model, tokenizer, actor, expected_bf16, output, torch
        )
        summary.update(
            status="COMPLETE",
            elapsed_seconds=time.monotonic() - started,
            training_sequences=sum(event["sequences"] for event in summary["updates"]),
            training_prompt_tokens=sum(event["prompt_tokens"] for event in summary["updates"]),
            training_response_tokens=sum(event["response_tokens"] for event in summary["updates"]),
            optimizer_step_min=optimizer_steps(optimizer)[0],
            optimizer_step_max=optimizer_steps(optimizer)[1],
            peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            peak_GPU_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
            original_RLOO_actor_unchanged=True,
            original_RLOO_optimizer_unchanged=True,
            training_state_retention=(
                "transient FP32/Adam released at process exit; final BF16 retained"
            ),
        )
        write(output / "training_summary.json", summary)
        print(json.dumps(dict(event="SUPERVISED_TRAINING_COMPLETE", **summary)), flush=True)
    except BaseException as error:
        summary.update(
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.monotonic() - started,
        )
        write(output / "training_summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--seed", type=int, default=43)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
