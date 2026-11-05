"""Short head-only learned-critic/GAE demo on fresh same-prefix branches.

Gamma=lambda=1 makes the head GAE exactly remaining M1 return minus its
pre-update learned value. Tail requests define that return but are not extra
actor examples. All optimizer states are fresh each round and never exported.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.agents.same_prefix_training_demo import (
    GIB,
    append,
    checkpoint_probes,
    probe_delta,
    probes,
    require,
    response_logprob,
    write,
)

PROFILE = "same_prefix_learned_head_gae_demo_v1"


def retirement_inputs(args: argparse.Namespace) -> dict[str, Path]:
    """Retire only approved runs' own completed immediately preceding exports."""
    if not getattr(args, "retire_input_checkpoints", False):
        return {}
    output = args.output.absolute()
    require(".." not in output.parts, "retirement output must not contain parent traversal")
    arm_directory = "zero" if args.arm == "S0" else args.arm
    require(
        output.name == "train" and output.parent.parent.name == arm_directory,
        "retirement requires this arm roundN/train",
    )
    run = output.parents[2]
    round_name = output.parent.name
    curve_rounds = {f"round{number}": number for number in range(2, 9)}
    curve_run = (
        args.seed == 42
        and args.arm == "S0"
        and run.name == "m2-s0-learning-curve-20261024-v1"
        and round_name in curve_rounds
    )
    legacy_run = round_name == "round2" and (
        (args.seed == 43 and run.name.startswith("m2-same-prefix-online-demo-seed43-"))
        or (args.seed == 42 and run.name == "m2-same-prefix-online-8window-demo-20261024-v1")
        or (args.seed == 42 and args.arm == "S0" and run.name == "m2-s0-multistep-demo-20261024-v1")
    )
    require(
        (curve_run or legacy_run) and run.parent == Path(__file__).resolve().parents[2] / "runs",
        "retirement is restricted to approved new demo runs, never common SFT or other history",
    )
    round_number = curve_rounds[round_name] if curve_run else 2
    require(args.critic is not None, "retirement requires its previous critic")
    original = run / arm_directory / f"round{round_number - 1}" / "train"
    source = json.loads((original / "training_summary.json").read_text())
    require(
        source.get("status") == "COMPLETE"
        and source.get("seed") == args.seed
        and source.get("arm") == args.arm,
        "completed same-arm previous-round source required",
    )
    result = {}
    for role, argument in (("actor", args.model), ("critic", args.critic)):
        expected = original / role
        require(
            argument.absolute() == expected and source.get(f"{role}_checkpoint") == str(expected),
            f"retirement {role} is not this run's immediately previous checkpoint",
        )
        require(expected.is_dir(), f"retirement {role} checkpoint absent")
        for path in (output, *output.parents, expected, *expected.parents):
            require(not path.is_symlink(), "symlinks are forbidden for checkpoint retirement")
        require(
            not any(path.is_symlink() for path in expected.rglob("*")),
            "symlink inside input checkpoint",
        )
        result[role] = expected
    return result


def retire_loaded_input(
    role: str, paths: dict[str, Path], model: Any, torch: Any, output: Path, summary: dict
) -> None:
    if role not in paths:
        return
    path = paths[role]
    require(
        path.is_dir()
        and not any(p.is_symlink() for p in (path, *path.parents))
        and not any(p.is_symlink() for p in path.rglob("*")),
        "input path changed before retirement",
    )
    require(
        all(not p.is_meta and p.device.type == "cuda" for p in model.parameters()),
        "all input weights must be fully loaded on GPU before retirement",
    )
    torch.cuda.synchronize()
    event = dict(
        role=role, path=str(path), phase="loaded_on_GPU_before_new_export", status="DELETING"
    )
    summary.setdefault("input_checkpoint_retirement", []).append(event)
    write(output / "training_summary.json", summary)
    shutil.rmtree(path)
    require(not path.exists(), "input checkpoint retirement incomplete")
    event["status"] = "RETIRED"
    summary.setdefault("retired_input_checkpoints", []).append(str(path))
    write(output / "training_summary.json", summary)


def records(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def field(row: dict, *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"missing one of {names}")


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def economic_key(value: str) -> tuple:
    key = json.loads(value)
    require(isinstance(key, list) and bool(key), "economic order array required")
    if key[0] in {"hold", "abstain"}:
        require(len(key) == 1, "parameter-free hold/abstain economic key")
        return (key[0],)
    require(
        len(key) == 3
        and key[0] in {"buy", "sell"}
        and isinstance(key[1], str)
        and bool(key[1].strip())
        and finite(key[2])
        and 0 <= key[2] <= 1,
        "invalid economic order",
    )
    return key[0], key[1].strip().upper(), float(key[2]) or 0.0


def build_batch(directory: Path, arm: str, penalty: float = 0.05) -> tuple[list[dict], dict]:
    """Read heads and EST only. Partial four-slot groups remain explicitly NA."""
    episode_ids = json.loads((directory / "collection_summary.json").read_text())["episode_ids"]
    require(
        isinstance(episode_ids, list)
        and len(episode_ids) in (4, 8)
        and len(episode_ids) == len(set(episode_ids)),
        "collection must declare four or eight unique Train windows",
    )
    planned_groups, planned_heads = len(episode_ids), 4 * len(episode_ids)
    heads = records(directory / "head_results.jsonl")
    paths = records(directory / "trajectory_results.jsonl")
    groups: dict[str, list[dict]] = defaultdict(list)
    estimates = {}
    for path in paths:
        if path.get("role") != "estimate":
            continue
        identity = (
            path["anchor_id"],
            int(field(path, "slot_id", "slot")),
            int(field(path, "estimate_replicate", "replicate")),
        )
        require(identity not in estimates, "duplicate EST identity")
        estimates[identity] = path
    for head in heads:
        groups[head["anchor_id"]].append(head)
    require(
        len(heads) == planned_heads
        and len(groups) == planned_groups
        and {head["episode_id"] for head in heads} == set(episode_ids),
        "all declared windows and their four original head slots must be retained",
    )
    batch, group_rows = [], []
    for anchor, members in sorted(groups.items()):
        members.sort(key=lambda row: field(row, "slot_id", "slot"))
        require(
            [field(h, "slot_id", "slot") for h in members] == [0, 1, 2, 3],
            "each anchor retains all original four head slots",
        )
        reasons, prepared = [], []
        for slot, head in enumerate(members):
            if field(head, "status", "env_status") != "ADVANCING":
                reasons.append(f"head-{slot}-not-advancing")
                continue
            draws = [estimates.get((anchor, slot, rep)) for rep in (0, 1)]
            if any(draw is None or draw["status"] != "COMPLETE" for draw in draws):
                reasons.append(f"head-{slot}-incomplete-EST")
                continue
            returns, outcomes = [], []
            for draw in draws:
                score = field(draw, "terminal_financial_Y", "Y")
                prefix = draw["prefix_requests"]
                tail = draw["tail_requests"]
                total = draw["total_requests"]
                invalid = draw["remaining_invalid_count"]
                require(finite(score), "finite original financial Y required")
                require(
                    all(type(v) is int and v >= 0 for v in (prefix, tail, total, invalid)),
                    "integer request accounting required",
                )
                require(
                    total == prefix + 1 + tail and 0 <= invalid <= 1 + tail,
                    "M1 original timeline length differs",
                )
                if "timesteps" in draw:
                    steps = draw["timesteps"]
                    require(len(steps) == 1 + tail, "head plus actual tail timeline differs")
                    require(
                        sum(bool(s.get("invalid", s.get("invalid_action", False))) for s in steps)
                        == invalid,
                        "remaining invalid count differs",
                    )
                # Uniform M1 reward conserves Y over the ORIGINAL full timeline.
                # Its last residual-corrected share only changes roundoff.
                share = score / total
                final_share = score - share * (total - 1)
                remaining = share * tail + final_share - penalty * invalid
                returns.append(remaining)
                outcomes.append(score)
            prepared.append(
                dict(
                    anchor_id=anchor,
                    slot_id=slot,
                    head_key=head.get("head_key", f"{anchor}/s{slot}"),
                    prompt_token_ids=head["prompt_token_ids"],
                    response_token_ids=head["response_token_ids"],
                    economic_order_key=head["economic_order_key"],
                    order=economic_key(head["economic_order_key"]),
                    mean_EST_financial_Y=math.fsum(outcomes) / 2,
                    head_return=math.fsum(returns) / 2,
                    EST_financial_Y=outcomes,
                    EST_head_returns=returns,
                )
            )
        if reasons:
            group_rows.append(
                dict(
                    anchor_id=anchor,
                    supported=False,
                    reasons=reasons,
                    planned_heads=4,
                    admitted_heads=0,
                )
            )
            continue
        require(len(prepared) == 4, "supported group needs all four complete heads")
        prompt = prepared[0]["prompt_token_ids"]
        require(
            all(row["prompt_token_ids"] == prompt for row in prepared),
            "head critic states must have exactly the same visible prompt",
        )
        for row in prepared:
            for name, limit in (("prompt_token_ids", 8192), ("response_token_ids", 4096)):
                ids = row[name]
                require(
                    isinstance(ids, list)
                    and 0 < len(ids) <= limit
                    and all(type(v) is int and v >= 0 for v in ids),
                    f"invalid {name}",
                )
        mean_y = math.fsum(row["mean_EST_financial_Y"] for row in prepared) / 4
        diverse = len({row["order"] for row in prepared}) > 1
        for row in prepared:
            raw = 0.0
            if diverse and arm == "plain":
                raw = row["mean_EST_financial_Y"] - mean_y
            if diverse and arm == "exact":
                same = [r["mean_EST_financial_Y"] for r in prepared if r["order"] == row["order"]]
                raw = math.fsum(same) / len(same) - mean_y
            row["raw_S"] = raw
        group_rows.append(
            dict(
                anchor_id=anchor,
                supported=True,
                planned_heads=4,
                admitted_heads=4,
                order_gate=diverse,
            )
        )
        batch.extend(prepared)
    return batch, dict(
        planned_groups=planned_groups,
        planned_heads=planned_heads,
        supported_groups=len(batch) // 4,
        supported_heads=len(batch),
        NA_heads=planned_heads - len(batch),
        groups=group_rows,
        EST_repetitions_per_head=2,
        head_actor_weight_repetitions=1,
        REF_used=False,
        penalty=penalty,
    )


def scalar_value(model: Any, row: dict, torch: Any) -> Any:
    ids = torch.tensor([row["prompt_token_ids"]], device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).last_hidden_state[:, -1, :]
        return model.score(hidden).float().reshape(())


def adam_update(
    model: Any, items: list, loss_function: Any, learning_rate: float, torch: Any
) -> dict:
    """One accumulated full-parameter update, with one pageable CPU moment set."""
    import psutil

    from src.agents.chunked_adam import chunked_adamw_step

    model.float()
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    require(
        all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()),
        "all backbone/head parameters must remain trainable FP32",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for item in items:
        loss = loss_function(item) / len(items)
        require(bool(torch.isfinite(loss)), "nonfinite training loss")
        loss.backward()
        losses.append(float(loss.detach().item()))
        del loss
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    require(norm > 0, "real gradient required")
    events = []

    def admit_host(required: int) -> None:
        require(psutil.virtual_memory().available >= required + 4 * GIB, "CPU moment headroom low")

    def admit_gpu(required: int) -> None:
        require(torch.cuda.mem_get_info()[0] >= required + 2 * GIB, "GPU moment chunk headroom low")

    chunked_adamw_step(
        optimizer,
        chunk_bytes=64 * 1024**2,
        admit_host=admit_host,
        admit_gpu=admit_gpu,
        emit=events.append,
    )
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    model.gradient_checkpointing_disable()
    model.eval()
    model.bfloat16()
    torch.cuda.empty_cache()
    return dict(
        optimizer_updates=1,
        loss=math.fsum(losses),
        gradient_norm_before_clip=norm,
        optimizer_events=events,
        parameters=sum(p.numel() for p in model.parameters()),
        fresh_optimizer_each_role_each_round=True,
        optimizer_state_saved=False,
    )


def run(args: argparse.Namespace) -> None:
    retire_paths = retirement_inputs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(
        not (output / "training_summary.json").exists(), "fresh round required; no retry/resume"
    )
    require(
        not (output / "actor").exists() and not (output / "critic").exists(),
        "actor/critic export already exists",
    )
    require(args.alpha == 0.5 and args.eta == 0.25, "fixed independent alpha=.5 and eta=.25")
    batch, coverage = build_batch(args.data.resolve(), args.arm)
    summary = dict(
        profile=PROFILE,
        status="STARTING",
        arm=args.arm,
        seed=args.seed,
        gamma=1.0,
        lam=1.0,
        alpha=args.alpha,
        eta=args.eta,
        coverage=coverage,
        actor_updates=0,
        critic_updates=0,
        REF_used=False,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        actor_scope="one head per original slot, EST2 averaged; no tail actor rows",
        critic_scope="same-prefix scalar pre-action value; full backbone plus score head",
        actor_value_timing="frozen critic values before the current critic update",
        optimizer_state_reset_each_round=True,
        performance_claim=False,
    )
    write(output / "training_summary.json", summary)
    try:
        require(bool(batch), "no complete four-head EST group; do not fill missing examples")
        os.environ.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False
        )
        common = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        critic_path = args.critic or args.model
        critic = AutoModelForSequenceClassification.from_pretrained(
            critic_path, num_labels=1, **common
        )
        retire_loaded_input("critic", retire_paths, critic, torch, output, summary)
        initial_critic = probes(critic)
        anchor_rows: dict[str, list[dict]] = defaultdict(list)
        for row in batch:
            anchor_rows[row["anchor_id"]].append(row)
        critic.eval()
        with torch.no_grad():
            values = {
                anchor: float(scalar_value(critic, group[0], torch))
                for anchor, group in anchor_rows.items()
            }
        for row in batch:
            row["value_before_critic_update"] = values[row["anchor_id"]]
            row["U_head"] = row["head_return"] - row["value_before_critic_update"]
            row["eta_alpha_S"] = args.eta * args.alpha * row["raw_S"]
            row["combined_unwhitened"] = row["U_head"] + row["eta_alpha_S"]
        combined = torch.tensor([row["combined_unwhitened"] for row in batch], dtype=torch.float32)
        whitened = (combined - combined.mean()) / (combined.std(correction=1) + 1e-8)
        require(
            bool(torch.isfinite(whitened).all()), "finite shared-batch actor whitening required"
        )
        for row, advantage in zip(batch, whitened.tolist(), strict=True):
            row["actor_advantage"] = advantage
        with (output / "head_advantages.jsonl").open("x") as stream:
            for row in batch:
                append(stream, row)
        targets = {
            anchor: math.fsum(r["head_return"] for r in group) / 4
            for anchor, group in anchor_rows.items()
        }
        critic_before_mse = math.fsum(
            (values[r["anchor_id"]] - r["head_return"]) ** 2 for r in batch
        ) / len(batch)
        summary["status"] = "CRITIC_UPDATE"
        write(output / "training_summary.json", summary)
        critic_update = adam_update(
            critic,
            list(anchor_rows),
            lambda anchor, model=critic: (
                0.5 * (scalar_value(model, anchor_rows[anchor][0], torch) - targets[anchor]) ** 2
            ),
            args.critic_learning_rate,
            torch,
        )
        with torch.no_grad():
            new_values = {
                anchor: float(scalar_value(critic, group[0], torch))
                for anchor, group in anchor_rows.items()
            }
        critic_update.update(
            before_full_head_MSE=critic_before_mse,
            after_full_head_MSE=math.fsum(
                (new_values[r["anchor_id"]] - r["head_return"]) ** 2 for r in batch
            )
            / len(batch),
            value_before=values,
            value_after=new_values,
            BF16_parameter_delta=probe_delta(initial_critic, probes(critic)),
            anchor_loss_collapse=(
                "four identical prompt-state losses have the exact same MSE gradient "
                "as their mean target"
            ),
        )
        critic.save_pretrained(output / "critic", safe_serialization=True, max_shard_size="4GB")
        critic_update["checkpoint_readback"] = checkpoint_probes(output / "critic", probes(critic))
        del critic
        gc.collect()
        torch.cuda.empty_cache()
        summary.update(
            critic_updates=1,
            critic=critic_update,
            status="ACTOR_UPDATE",
            critic_checkpoint=str(output / "critic"),
        )
        write(output / "training_summary.json", summary)
        # No critic model, gradients or optimizer moments remain resident here.
        actor = AutoModelForCausalLM.from_pretrained(args.model, **common)
        retire_loaded_input("actor", retire_paths, actor, torch, output, summary)
        initial_actor = probes(actor)
        actor_update = adam_update(
            actor,
            batch,
            lambda row: -row["actor_advantage"] * response_logprob(actor, row, torch),
            args.actor_learning_rate,
            torch,
        )
        actor_update["BF16_parameter_delta"] = probe_delta(initial_actor, probes(actor))
        actor.save_pretrained(output / "actor", safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output / "actor")
        actor_update["checkpoint_readback"] = checkpoint_probes(output / "actor", probes(actor))
        summary.update(
            status="COMPLETE",
            actor_updates=1,
            actor=actor_update,
            actor_checkpoint=str(output / "actor"),
            supported_groups=coverage["supported_groups"],
            supported_heads=len(batch),
            whitening_mean=float(combined.mean()),
            whitening_std_sample=float(combined.std(correction=1)),
        )
        write(output / "training_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as exc:
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write(output / "training_summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--critic", type=Path)
    parser.add_argument("--arm", choices=("S0", "plain", "exact"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--actor-learning-rate", type=float, default=2e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-6)
    parser.add_argument("--retire-input-checkpoints", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    run(args)
    print(f"training round elapsed_seconds={time.monotonic() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
