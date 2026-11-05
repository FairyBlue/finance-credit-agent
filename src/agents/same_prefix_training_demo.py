"""Two-to-four full-parameter cached-EST head updates; not an on-policy result.

This small engineering entry deliberately bypasses the critic/GAE rollout loop.
It uses signed raw exact residuals already built from EST only. REF is never
opened. The established bounded AdamW implementation keeps FP32 moments on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
PROFILE = "cached_est_exact_head_full_parameter_demo_v1"


def write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(stream: Any, value: Any) -> None:
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_groups(path: Path, steps: int) -> tuple[list[list[dict]], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    source_rows = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            source_rows += 1
            if row["estimate_replicate"] != 0:
                continue
            require(row["sample_mask"] is True, "only complete EST labels enter training")
            for name in ("alpha_S_exact", "raw_S_exact", "financial_Y_est"):
                require(
                    type(row[name]) in (int, float) and math.isfinite(row[name]),
                    f"finite {name} required",
                )
            require(
                row["alpha"] == 0.5
                and math.isclose(
                    row["alpha_S_exact"], 0.5 * row["raw_S_exact"], abs_tol=1e-12, rel_tol=1e-12
                ),
                "cached alpha=.5 label differs",
            )
            for key, limit in (("prompt_token_ids", 8192), ("response_token_ids", 4096)):
                ids = row[key]
                require(
                    isinstance(ids, list)
                    and 0 < len(ids) <= limit
                    and all(type(v) is int and v >= 0 for v in ids),
                    f"invalid {key}",
                )
            groups[row["group_id"]].append(row)
    eligible = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: row["slot_id"])
        require(
            len(members) == 4 and [r["slot_id"] for r in members] == [0, 1, 2, 3],
            "every cached group needs all original four heads",
        )
        for field in (
            "anchor_id",
            "policy_snapshot_sha256",
            "financial_state_sha256",
            "prompt_sha256",
        ):
            require(len({r[field] for r in members}) == 1, f"group {field} differs")
        require(
            all(r["prompt_token_ids"] == members[0]["prompt_token_ids"] for r in members),
            "same-prefix head prompts differ",
        )
        require(
            abs(math.fsum(r["alpha_S_exact"] for r in members)) < 1e-10,
            "cached group residual is not centered",
        )
        if any(r["alpha_S_exact"] != 0 for r in members):
            eligible.append(members)
    require(bool(eligible), "no nonzero complete EST group available")
    selected = eligible[:steps]
    schedule = [selected[i % len(selected)] for i in range(steps)]
    return schedule, dict(
        source_rows=source_rows,
        replicate_used=0,
        complete_groups=len(groups),
        nonzero_groups=len(eligible),
        selected_group_ids=[g[0]["group_id"] for g in selected],
        step_group_ids=[g[0]["group_id"] for g in schedule],
        selection="first sorted nonzero groups, cycle only if fewer than steps",
        training_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        REF_labels_used=False,
    )


def probes(model: Any) -> dict[str, list[float]]:
    import torch

    result = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel():
                count = min(17, parameter.numel())
                positions = torch.tensor(
                    [i * (parameter.numel() - 1) // max(1, count - 1) for i in range(count)],
                    device=parameter.device,
                )
                result[name] = parameter.detach().view(-1)[positions].float().cpu().tolist()
    return result


def probe_delta(before: dict, after: dict) -> dict:
    differences = [b - a for name in before for a, b in zip(before[name], after[name], strict=True)]
    return dict(
        sampled_elements=len(differences),
        changed_elements=sum(x != 0 for x in differences),
        maximum_absolute_delta=max(map(abs, differences)),
        RMS_delta=math.sqrt(math.fsum(x * x for x in differences) / len(differences)),
    )


def response_logprob(model: Any, row: dict, torch: Any) -> Any:
    response = row["response_token_ids"]
    ids = torch.tensor([row["prompt_token_ids"] + response], device="cuda", dtype=torch.long)
    # Qwen3 supports logits_to_keep: the first returned position predicts the
    # first response token. The last input token has no supervised successor.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=len(response) + 1,
        ).logits[0, :-1, :]
        target = torch.tensor(response, device="cuda", dtype=torch.long)
        return -torch.nn.functional.cross_entropy(logits.float(), target, reduction="mean")


def generations(model: Any, tokenizer: Any, examples: list[dict], torch: Any) -> list[dict]:
    output = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in examples:
            ids = torch.tensor([row["prompt_token_ids"]], device="cuda", dtype=torch.long)
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=64,
                use_cache=True,
                eos_token_id=[151645, 151643],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )[0, ids.shape[1] :].tolist()
            output.append(
                dict(
                    head_key=row["head_key"],
                    anchor_id=row["anchor_id"],
                    response_token_ids=generated,
                    response_text=tokenizer.decode(generated, skip_special_tokens=True),
                    complete_eos=bool(generated) and generated[-1] in (151645, 151643),
                    do_sample=False,
                    maximum_new_tokens=64,
                )
            )
    return output


def checkpoint_probes(directory: Path, expected: dict) -> dict:
    """Read only a few mmap-backed values per written tensor, not an 8GB hash."""
    import torch
    from safetensors import safe_open

    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as handle:
            weight_map = {name: "model.safetensors" for name in handle.keys()}  # noqa: SIM118
    result = {}
    shards: dict[str, list[str]] = defaultdict(list)
    for name in expected:
        require(name in weight_map, f"checkpoint parameter absent: {name}")
        shards[weight_map[name]].append(name)
    for shard, names in shards.items():
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                require(tensor.dtype == torch.bfloat16, "checkpoint must contain BF16 tensors")
                count = min(17, tensor.numel())
                positions = torch.tensor(
                    [i * (tensor.numel() - 1) // max(1, count - 1) for i in range(count)]
                )
                result[name] = tensor.view(-1)[positions].float().tolist()
                require(result[name] == expected[name], "written checkpoint probe differs")
    return dict(
        verified_parameter_tensors=len(result),
        verified_sampled_elements=sum(map(len, result.values())),
        dtype="bfloat16",
    )


def train(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "training_state.json").exists(), "fresh demo required; no retry/resume")
    require(not (output / "checkpoint").exists(), "checkpoint already exists")
    require(2 <= args.steps <= 4, "demo steps must be 2..4")
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0, "positive LR required")
    schedule, selection = load_groups(args.data.resolve(), args.steps)
    state = dict(
        profile=PROFILE,
        status="STARTING",
        purpose="engineering cached-head update only",
        model_path=str(args.model.resolve()),
        steps_requested=args.steps,
        optimizer_updates=0,
        learning_rate=args.learning_rate,
        seed=args.seed,
        selection=selection,
        loss="minus alpha_S_exact times response mean logprob, four heads equally weighted",
        alpha=0.5,
        eta_applied=False,
        whitening=False,
        GAE=False,
        critic=False,
        on_policy=False,
        REF_labels_used=False,
        automatic_retry=False,
        optimizer_saved=False,
    )
    write(output / "training_state.json", state)
    if args.check_data:
        state["status"] = "DATA_CHECKED_NOT_TRAINED"
        write(output / "training_state.json", state)
        print(json.dumps(state, sort_keys=True), flush=True)
        return
    try:
        for name in ("USE_TORCH", "USE_TF", "USE_FLAX"):
            os.environ.pop(name, None)
        os.environ.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
        import psutil
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.agents.chunked_adam import chunked_adamw_step

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False
        )
        state["status"] = "LOADING"
        write(output / "training_state.json", state)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        before_probes = probes(model)
        examples = []
        for group in schedule:
            if group[0]["anchor_id"] not in {r["anchor_id"] for r in examples}:
                examples.append(group[0])
            if len(examples) == 2:
                break
        before = generations(model, tokenizer, examples, torch)
        with torch.no_grad():
            initial_logprobs = [
                float(response_logprob(model, row, torch).item()) for row in examples
            ]
        model.float()
        require(
            all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters()),
            "all model parameters must be trainable FP32",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        state.update(
            status="TRAINING",
            parameters=sum(p.numel() for p in model.parameters()),
            parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            optimizer="existing chunked_adamw_step, FP32 pageable CPU moments",
        )
        write(output / "training_state.json", state)

        def admit_gpu(required: int) -> None:
            require(torch.cuda.mem_get_info()[0] >= required + 2 * GIB, "GPU chunk headroom low")

        def admit_host(required: int) -> None:
            require(
                psutil.virtual_memory().available >= required + 4 * GIB, "CPU moment headroom low"
            )

        with (output / "train_steps.jsonl").open("x") as log:
            for step, group in enumerate(schedule, 1):
                started = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                records = []
                for row in group:
                    logprob = response_logprob(model, row, torch)
                    loss = -float(row["alpha_S_exact"]) * logprob / 4
                    require(bool(torch.isfinite(loss)), "nonfinite head policy-gradient loss")
                    loss.backward()
                    records.append(
                        dict(
                            head_key=row["head_key"],
                            label=row["alpha_S_exact"],
                            response_mean_logprob=float(logprob.detach().item()),
                            scaled_loss=float(loss.detach().item()),
                        )
                    )
                    del loss, logprob
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    ).item()
                )
                require(gradient_norm > 0, "nonzero gradient required for a real update")
                events = []
                chunked_adamw_step(
                    optimizer,
                    chunk_bytes=64 * 1024**2,
                    admit_gpu=admit_gpu,
                    admit_host=admit_host,
                    emit=events.append,
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                delta = probe_delta(before_probes, probes(model))
                require(delta["changed_elements"] > 0, "no FP32 parameter change observed")
                append(
                    log,
                    dict(
                        step=step,
                        group_id=group[0]["group_id"],
                        records=records,
                        gradient_norm_before_clip=gradient_norm,
                        parameter_probe_delta=delta,
                        optimizer_events=events,
                        elapsed_seconds=time.monotonic() - started,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    ),
                )
                state.update(optimizer_updates=step, last_parameter_probe_delta=delta)
                write(output / "training_state.json", state)
                print(
                    f"updated {step}/{args.steps} "
                    f"group={group[0]['group_id']} grad={gradient_norm}",
                    flush=True,
                )
        del optimizer
        gc.collect()
        model.eval()
        model.gradient_checkpointing_disable()
        model.bfloat16()
        torch.cuda.empty_cache()
        final_probes = probes(model)
        exported_delta = probe_delta(before_probes, final_probes)
        require(
            exported_delta["changed_elements"] > 0, "no BF16 parameter change observed; no retry"
        )
        with torch.no_grad():
            final_logprobs = [float(response_logprob(model, row, torch).item()) for row in examples]
        after = generations(model, tokenizer, examples, torch)
        checkpoint = output / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(checkpoint)
        verified = checkpoint_probes(checkpoint, final_probes)
        write(
            output / "before_after.json",
            dict(
                before_greedy=before,
                after_greedy=after,
                before_response_mean_logprobs=initial_logprobs,
                after_response_mean_logprobs=final_logprobs,
                BF16_parameter_probe_delta=exported_delta,
                checkpoint_probe_readback=verified,
                performance_claim=False,
            ),
        )
        if args.rollouts:
            from src.agents.same_prefix_demo_rollout import run_rollouts

            run_rollouts(model, tokenizer, output_directory=output, count=args.rollouts)
        state.update(
            status="COMPLETE",
            checkpoint=str(checkpoint),
            checkpoint_dtype="bfloat16",
            BF16_parameter_probe_delta=exported_delta,
            checkpoint_probe_readback=verified,
            postupdate_rollouts=args.rollouts,
            performance_claim=False,
        )
        write(output / "training_state.json", state)
        print(json.dumps(state, sort_keys=True), flush=True)
    except BaseException as exc:
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write(output / "training_state.json", state)
        raise


def evaluate_only(args: argparse.Namespace) -> None:
    """Evaluate an existing checkpoint in a fresh run; never create an optimizer."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "evaluation_state.json").exists(), "fresh evaluation required")
    require(not (output / "training_state.json").exists(), "do not overwrite a training run")
    state = dict(
        profile=PROFILE,
        status="LOADING",
        evaluate_only=True,
        model_path=str(args.model.resolve()),
        optimizer_updates=0,
        rollouts_requested=args.rollouts,
        performance_claim=False,
    )
    write(output / "evaluation_state.json", state)
    try:
        os.environ.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.agents.same_prefix_demo_rollout import run_rollouts

        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.eval()
        run_rollouts(model, tokenizer, output_directory=output, count=args.rollouts)
        state.update(status="COMPLETE", postupdate_rollouts=args.rollouts)
        write(output / "evaluation_state.json", state)
    except BaseException as exc:
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write(output / "evaluation_state.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollouts", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluate_only(args)
    else:
        if args.data is None:
            parser.error("--data is required for training")
        train(args)


if __name__ == "__main__":
    main()
