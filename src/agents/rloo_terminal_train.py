"""Four PPO minibatch updates on grouped full-parent RLOO trajectories.

FP32 master parameters and GPU Adam moments survive every stage.
Only an independent BF16 copy is exported for the rollout worker. Financial Y
is paid at the real terminal request; rejected requests retain immediate costs.
This is a bounded critic-free training demo, not a complete PPO reproduction.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from src.agents.same_prefix_online_train import finite, records
from src.agents.same_prefix_training_demo import (
    ROOT,
    append,
    checkpoint_probes,
    probe_delta,
    probes,
    require,
    write,
)

PROFILE = "terminal_Y_grouped_rloo_persistent_gpu_adam_sharded_demo_v2"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 65, 73)]
EIGHT_EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157)]
PENALTY = 0.05
LOCAL_RUN = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
AUX_RUN = ROOT / "runs/m2-rloo-local-aux-demo-20261026-v1"
AUX_PROFILE = "terminal_Y_RLOO_separate_fixed8_local_PPO_persistent_v1"
COMMON_SFT = ROOT / "runs/m2-order-recovery-sft-20261011-v1/merged_model"
OBJECTIVES = {
    "RLOO": "whitened_RLOO_request_PPO",
    "plain": "joint_whitened_RLOO_plus_local_request_PPO",
    "exact": "joint_whitened_RLOO_plus_local_request_PPO",
    "exact_aux": "separate_RLOO_and_fixed8_local_PPO",
}


def profile_for(args) -> str:
    return AUX_PROFILE if getattr(args, "arm", "RLOO") == "exact_aux" else PROFILE


def recipe(args) -> dict:
    arm = getattr(args, "arm", "RLOO")
    return dict(
        arm=arm,
        alpha=getattr(args, "alpha", 0.5) if arm != "RLOO" else 0.0,
        eta=getattr(args, "eta", 0.25) if arm != "RLOO" else 0.0,
        objective=OBJECTIVES[arm],
    )


def require_recipe(saved: dict, args) -> None:
    expected = recipe(args)
    actual = {
        key: saved.get(key, value if expected["arm"] == "RLOO" else None)
        for key, value in dict(arm="RLOO", alpha=0.0, eta=0.0).items()
    }
    actual["objective"] = saved.get(
        "objective", OBJECTIVES[expected["arm"]] if expected["arm"] != "exact_aux" else None
    )
    require(actual == expected, "resume arm/alpha/eta changed")


def trajectory_rows(trajectory: dict) -> tuple[list[dict], dict]:
    steps = trajectory["timesteps"]
    score = trajectory.get("terminal_financial_Y", trajectory.get("Y"))
    require(finite(score), "complete parent needs finite financial Y")
    total = len(steps)
    require(
        1 < total <= 80
        and trajectory["prefix_requests"] == 0
        and trajectory["total_requests"] == total
        and trajectory["last_session_index"] == 20,
        "complete full-reset parent timeline required",
    )
    replicate = trajectory["replicate_index"]
    require(type(replicate) is int and 0 <= replicate < 4, "original replica index 0..3")
    batch, sessions = [], Counter()
    expected_session = 0
    for index, step in enumerate(steps):
        prompt, response = step["prompt_token_ids"], step["response_token_ids"]
        require(
            isinstance(prompt, list)
            and isinstance(response, list)
            and 0 < len(prompt) <= 8192
            and 0 < len(response) <= 4096
            and all(type(token) is int and token >= 0 for token in prompt + response),
            "nonempty original prompt/response tokens within fixed limits required",
        )
        before, after = step["decision_session_index"], step["result_session_index"]
        invalid, advancing, done = step["invalid"], step["advancing"], step["done"]
        require(
            type(before) is int
            and type(after) is int
            and before == expected_session
            and 0 <= before < 20
            and type(invalid) is bool
            and type(advancing) is bool
            and type(done) is bool
            and after == before + int(advancing)
            and (not invalid or not advancing)
            and done == (index == total - 1)
            and (not done or after == 20),
            "actual request/session timeline inconsistent",
        )
        sessions[before] += 1
        expected_session = after
        financial_reward = float(score) if done else 0.0
        batch.append(
            dict(
                episode_id=trajectory["episode_id"],
                trajectory_key=trajectory["trajectory_key"],
                replicate_index=replicate,
                request_index=index,
                decision_session_index=before,
                result_session_index=after,
                invalid=invalid,
                advancing=advancing,
                done=done,
                prompt_token_ids=prompt,
                response_token_ids=response,
                response_text=step["response_text"],
                terminal_financial_Y=float(score),
                financial_reward=financial_reward,
                training_reward=financial_reward - PENALTY * int(invalid),
            )
        )
    require(
        expected_session == 20 and set(sessions) == set(range(20)) and max(sessions.values()) <= 4,
        "full original 20-session coverage within four requests/session required",
    )
    running = 0.0
    for row in reversed(batch):
        running = math.fsum((row["training_reward"], running))
        row["return_G"] = running
    invalids = sum(row["invalid"] for row in batch)
    require(
        math.isclose(batch[0]["return_G"], score - PENALTY * invalids, abs_tol=1e-10),
        "terminal Y and all immediate invalid costs must be conserved",
    )
    return batch, dict(
        episode_id=trajectory["episode_id"],
        replicate_index=replicate,
        trajectory_key=trajectory["trajectory_key"],
        training_rows=total,
        invalid_rows=invalids,
        terminal_financial_Y=float(score),
        total_training_return=batch[0]["return_G"],
    )


def build_batch(
    directory: Path,
    expected_seed: int | None = None,
    arm: str = "RLOO",
    alpha: float = 0.5,
    eta: float = 0.25,
) -> tuple[list[dict], dict]:
    collection = json.loads((directory / "collection_summary.json").read_text())
    episodes, seed = collection["episode_ids"], collection["seed"]
    require(
        (episodes == EPISODES and seed == 42) or (episodes == EIGHT_EPISODES and seed in (42, 43)),
        "original four seed42 or fixed eight seed42/43 Train profile required",
    )
    require(expected_seed is None or seed == expected_seed, "collection/training seed mismatch")
    local = arm in {"plain", "exact", "exact_aux"}
    require(
        (
            local
            and collection.get("local_credit") is True
            and collection["parents_only"] is False
            and episodes == EPISODES
            and seed == 42
        )
        or (not local and arm == "RLOO" and collection["parents_only"] is True),
        "matching parent-only RLOO or declared four-window local-credit collection required",
    )
    require(collection.get("parent_repetitions") == 4, "four original replicas/window required")
    indexed: dict[str, dict[int, dict]] = defaultdict(dict)
    for trajectory in records(directory / "trajectory_results.jsonl"):
        if trajectory.get("role") != "parent":
            require(local, "head/EST/REF never train in legacy parent-only collection")
            continue
        episode, replicate = trajectory["episode_id"], trajectory["replicate_index"]
        require(
            episode in episodes
            and type(replicate) is int
            and 0 <= replicate < 4
            and replicate not in indexed[episode],
            "unexpected or duplicate parent slot",
        )
        indexed[episode][replicate] = trajectory
    batch, groups = [], []
    for episode in episodes:
        parents = indexed.get(episode, {})
        complete = len(parents) == 4 and all(p["status"] == "COMPLETE" for p in parents.values())
        if not complete:
            groups.append(
                dict(
                    episode_id=episode,
                    status="NA",
                    planned_trajectories=4,
                    complete_trajectories=sum(p["status"] == "COMPLETE" for p in parents.values()),
                    reason="whole window NA: all original four peers required, no filling",
                )
            )
            continue
        rows_by_replica, details = [], []
        for replicate in range(4):
            rows, detail = trajectory_rows(parents[replicate])
            rows_by_replica.append(rows)
            details.append(detail)
        totals = [detail["total_training_return"] for detail in details]
        for replicate, rows in enumerate(rows_by_replica):
            baseline = math.fsum(v for j, v in enumerate(totals) if j != replicate) / 3
            for row in rows:
                row["rloo_baseline"] = baseline
                row["trajectory_total_return"] = totals[replicate]
                row["unwhitened_advantage"] = row["return_G"] - baseline
            batch.extend(rows)
        groups.append(
            dict(
                episode_id=episode,
                status="COMPLETE",
                planned_trajectories=4,
                complete_trajectories=4,
                total_returns=totals,
                trajectories=details,
            )
        )
    coverage = dict(
        episode_ids=episodes,
        seed=seed,
        planned_windows=len(episodes),
        planned_trajectories=4 * len(episodes),
        admitted_windows=sum(g["status"] == "COMPLETE" for g in groups),
        admitted_trajectories=4 * sum(g["status"] == "COMPLETE" for g in groups),
        training_rows=len(batch),
        invalid_rows=sum(row["invalid"] for row in batch),
        groups=groups,
        head_EST_REF_used=False,
        original_collection_model_requests=collection.get("model_requests"),
    )
    if local:
        from src.agents.rloo_local_credit import apply_from_directory

        coverage["local_credit"] = apply_from_directory(
            batch, episodes, directory, "exact" if arm == "exact_aux" else arm, alpha, eta
        )
        if arm == "exact_aux":
            require(
                coverage["local_credit"]["planned_anchors"] == 8,
                "auxiliary denominator is the fixed eight planned anchors",
            )
            coverage["local_credit"].update(
                actor_objective_arm="exact_aux",
                mean_abs_applied_credit_scope="legacy eta*alpha*S, before objective loss weighting",
            )
            for row in batch:
                row["legacy_mixed_unwhitened_advantage"] = row["unwhitened_advantage"]
                row["unwhitened_advantage"] = row["raw_RLOO"]
    return batch, coverage


def auxiliary_request_loss(
    ratio, base_advantage, base_weight, local_advantage, local_weight, clip_epsilon, torch
):
    """One ratio/graph, independently clipped base and fixed-denominator local PPO."""
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    base = -torch.minimum(ratio * base_advantage, clipped * base_advantage).mean() * base_weight
    local = (
        (-torch.minimum(ratio * local_advantage, clipped * local_advantage).mean() * local_weight)
        if local_advantage != 0
        else base.new_zeros(())
    )
    return base + local, base, local


def token_logprobs(model, row: dict, torch):
    response = row["response_token_ids"]
    ids = torch.tensor([row["prompt_token_ids"] + response], device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=len(response) + 1,
        ).logits[0, :-1, :]
        labels = torch.tensor(response, device="cuda", dtype=torch.long)
        return -torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")


def optimizer_steps(optimizer) -> tuple[int, int]:
    values = [int(state["step"].item()) for state in optimizer.state.values()]
    return (min(values), max(values)) if values else (0, 0)


def _sample(tensor, torch, count=3):
    count = min(count, tensor.numel())
    if not count:
        return []
    positions = torch.tensor(
        [i * (tensor.numel() - 1) // max(1, count - 1) for i in range(count)],
        device=tensor.device,
    )
    return tensor.detach().reshape(-1)[positions].float().cpu().tolist()


def _write_component(path, name, kind, tensor, completed, torch):
    temporary = path.with_name(f".{path.name}.saving-{uuid.uuid4().hex}")
    expected = _sample(tensor, torch)
    torch.save(
        dict(name=name, kind=kind, completed_updates=completed, tensor=tensor.detach()), temporary
    )
    checked = torch.load(temporary, map_location="cpu", weights_only=False, mmap=True)
    require(
        checked["name"] == name
        and checked["kind"] == kind
        and checked["completed_updates"] == completed
        and checked["tensor"].dtype == tensor.dtype
        and checked["tensor"].shape == tensor.shape
        and _sample(checked["tensor"], torch) == expected,
        "checkpoint component readback differs",
    )
    del checked
    temporary.replace(path)
    return path.stat().st_size


def _read_component(directory, entry, kind, completed, torch):
    filename = entry["files"][kind]
    require(Path(filename).name == filename, "checkpoint shard must be a local filename")
    value = torch.load(directory / filename, map_location="cpu", weights_only=False, mmap=True)
    require(
        value["name"] == entry["name"]
        and value["kind"] == kind
        and value["completed_updates"] == completed,
        "incomplete/mixed checkpoint",
    )
    return value["tensor"]


def save_master(model, optimizer, args, completed: int, exported_probes: dict, torch) -> dict:
    """Replace one <=1.45-GiB component at a time, invalidate manifest until commit.

    The host cgroup high watermark is 57.6 GiB. Never create another complete
    CPU model/moment set or a second 45-GiB checkpoint. Interrupted saves are
    marked incomplete and must not be mistaken for the previous generation.
    """
    directory = args.state_output.resolve()
    require(not directory.exists() or directory.is_dir(), "checkpoint output must be directory")
    directory.mkdir(parents=True, exist_ok=True)
    for path in (directory, *directory.parents):
        require(not path.is_symlink(), "checkpoint symlink forbidden")
    manifest_path = directory / "manifest.json"
    write(
        manifest_path,
        dict(
            profile=profile_for(args),
            status="SAVING",
            target_updates=completed,
            **recipe(args),
            seed=args.seed,
            episode_ids=args.episode_ids,
        ),
    )
    parameters = dict(model.named_parameters())
    entries, total_bytes = [], 0
    group = optimizer.param_groups[0]
    hyper = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
    for index, (name, value) in enumerate(model.state_dict().items()):
        entry = dict(name=name, files={"master": f"tensor-{index:04d}.pt"})
        total_bytes += _write_component(
            directory / entry["files"]["master"], name, "master", value, completed, torch
        )
        if name in parameters:
            require(value.dtype == torch.float32, "master must remain exact FP32")
            state = optimizer.state[parameters[name]]
            require(int(state["step"].item()) == completed, "Adam step counter lost")
            entry["optimizer_step"] = completed
            entry["step_dtype"] = str(state["step"].dtype).removeprefix("torch.")
            for kind in ("exp_avg", "exp_avg_sq"):
                moment = state[kind]
                require(
                    moment.device == value.device and moment.dtype == torch.float32,
                    "FP32 Adam moments must stay with master on its device",
                )
                entry["files"][kind] = f"{kind}-{index:04d}.pt"
                total_bytes += _write_component(
                    directory / entry["files"][kind], name, kind, moment, completed, torch
                )
        entries.append(entry)
        if index % 50 == 0:
            print(
                json.dumps(
                    dict(
                        event="CHECKPOINT_SHARD_PROGRESS",
                        tensor_index=index,
                        bytes_written=total_bytes,
                    )
                ),
                flush=True,
            )
    metadata = dict(
        **recipe(args),
        profile=profile_for(args),
        completed_optimizer_steps=completed,
        seed=args.seed,
        episode_ids=args.episode_ids,
        learning_rate=args.learning_rate,
        parameter_names=list(parameters),
        master_probes=probes(model),
        bf16_export_probes=exported_probes,
        optimizer_hyperparameters=hyper,
        actor_path=str(args.actor_output.resolve()),
        torch_rng_state=torch.get_rng_state(),
        python_rng_state=random.getstate(),
        cuda_rng_state=torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
    )
    temporary = directory / ".metadata.pt.saving"
    torch.save(metadata, temporary)
    temporary.replace(directory / "metadata.pt")
    volatile = directory.is_relative_to(Path("/dev/shm"))
    write(
        manifest_path,
        dict(
            profile=profile_for(args),
            status="COMPLETE",
            **recipe(args),
            completed_optimizer_steps=completed,
            bytes=total_bytes,
            seed=args.seed,
            episode_ids=args.episode_ids,
            entries=entries,
            storage=(
                "RAM filesystem; host reboot volatile" if volatile else "persistent project disk"
            ),
        ),
    )
    return dict(
        path=str(directory),
        bytes=total_bytes,
        shard_files=sum(len(e["files"]) for e in entries),
        master_dtype="float32",
        moments_device=str(next(model.parameters()).device),
        completed_optimizer_steps=completed,
        FP32_readback_probes_verified=True,
        optimizer_steps_verified=True,
        checkpoint_format="per-component sharded manifest",
        interrupted_save_is_resumable=False,
        host_reboot_volatile=volatile,
    )


def restore_master(model, optimizer, args, sampled_bf16_probes, torch):
    directory = args.resume_state.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest.get("status") == "COMPLETE" and manifest.get("profile") == profile_for(args),
        "incomplete or incompatible checkpoint manifest",
    )
    require_recipe(manifest, args)
    saved = torch.load(directory / "metadata.pt", map_location="cpu", weights_only=False)
    require_recipe(saved, args)
    require(saved["profile"] == profile_for(args) and saved["seed"] == args.seed, "resume profile")
    require(
        saved.get("episode_ids", EPISODES) == args.episode_ids,
        "resume Train-window profile changed",
    )
    require(
        saved["bf16_export_probes"] == sampled_bf16_probes,
        "master and BF16 sampled snapshot must correspond",
    )
    completed = saved["completed_optimizer_steps"]
    if args.arm in {"plain", "exact", "exact_aux"}:
        expected_completed = 4 * (int(args.output.parent.name.removeprefix("round")) - 1)
        require(completed == expected_completed, "own arm must resume immediately prior round")
    require(
        completed == manifest["completed_optimizer_steps"]
        and completed % 4 == 0
        and 0 < completed <= 28,
        "resume within 32-step cap",
    )
    parameters, tensors = dict(model.named_parameters()), model.state_dict()
    require(saved["parameter_names"] == list(parameters), "resume parameter ordering")
    require(
        [entry["name"] for entry in manifest["entries"]] == list(tensors), "resume tensor ordering"
    )
    group = optimizer.param_groups[0]
    for key, value in saved["optimizer_hyperparameters"].items():
        require(value == group[key], f"resume Adam {key} changed")
    with torch.no_grad():
        for entry in manifest["entries"]:
            name = entry["name"]
            source = _read_component(directory, entry, "master", completed, torch)
            target = tensors[name]
            require(
                source.dtype == target.dtype and source.shape == target.shape,
                "master tensor dtype/shape changed",
            )
            target.copy_(source)
            del source
            if name in parameters:
                parameter = parameters[name]
                require(entry["optimizer_step"] == completed, "per-parameter step mismatch")
                state = {
                    "step": torch.tensor(
                        completed, dtype=getattr(torch, entry["step_dtype"]), device="cpu"
                    )
                }
                for kind in ("exp_avg", "exp_avg_sq"):
                    source = _read_component(directory, entry, kind, completed, torch)
                    require(
                        source.dtype == torch.float32 and source.shape == parameter.shape,
                        "Adam moment dtype/shape changed",
                    )
                    state[kind] = source.to(device=parameter.device, copy=True)
                    del source
                optimizer.state[parameter] = state
    require(probes(model) == saved["master_probes"], "FP32 resume lost precision")
    require(optimizer_steps(optimizer) == (completed, completed), "Adam counters reset")
    torch.set_rng_state(saved["torch_rng_state"])
    if saved["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
    random.setstate(saved["python_rng_state"])
    return completed


def export_bf16(model, tokenizer, directory: Path, expected: dict, scratch: Path, torch) -> dict:
    """Stream <=4-GB CPU shards directly to disk, never stage a RAM actor copy."""
    from safetensors.torch import save_file

    del scratch
    directory.mkdir(parents=True, exist_ok=True)
    require(directory.is_dir() and not directory.is_symlink(), "owned actor must be directory")
    marker = directory / "EXPORT_INCOMPLETE.json"
    write(marker, dict(status="EXPORTING", master_checkpoint_already_complete=True))
    state = model.state_dict()
    plans, current, current_size = [], [], 0
    for name, value in state.items():
        item_size = value.numel() * (2 if value.is_floating_point() else value.element_size())
        if current and current_size + item_size > 4_000_000_000:
            plans.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += item_size
    if current:
        plans.append(current)
    weight_map, total_size = {}, 0
    for index, names in enumerate(plans, 1):
        filename = f"model-{index:05d}-of-{len(plans):05d}.safetensors"
        temporary = directory / f".{filename}.exporting-{uuid.uuid4().hex}"
        copied = {}
        for name in names:
            value = state[name]
            copied[name] = (
                value.detach()
                .to(
                    device="cpu", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype
                )
                .contiguous()
            )
            total_size += copied[name].numel() * copied[name].element_size()
            weight_map[name] = filename
        save_file(copied, str(temporary), metadata={"format": "pt"})
        del copied
        gc.collect()
        temporary.replace(directory / filename)
        print(
            json.dumps(
                dict(event="BF16_EXPORT_SHARD_COMPLETE", shard=index, total_shards=len(plans))
            ),
            flush=True,
        )
    model.config.save_pretrained(directory)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["torch_dtype"] = "bfloat16"
    config["architectures"] = [type(model).__name__]
    write(config_path, config)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    write(
        directory / "model.safetensors.index.json",
        dict(metadata={"total_size": total_size}, weight_map=weight_map),
    )
    readback = checkpoint_probes(directory, expected)
    require(all(p.dtype == torch.float32 for p in model.parameters()), "FP32 master was cast")
    marker.unlink()
    return dict(
        **readback,
        master_never_cast=True,
        consumed_own_actor_replaced=True,
        export_format="streamed BF16 shards; no full RAM actor copy",
    )


def validate_paths(args) -> Path:
    output = args.output.resolve()
    run = output.parent.parent
    original_run = (
        run.parent == ROOT / "runs"
        and run.name.startswith("m2-rloo-terminal-demo-20261025-v")
        and args.seed == 42
    )
    eight_window_run = (
        run.parent == ROOT / "runs/m2-rloo-eight-window-20261025-v1"
        and args.seed in (42, 43)
        and run.name == f"seed{args.seed}"
        and output.parent.name in {f"round{i}" for i in range(1, 5)}
    )
    local_run = (
        run.parent == LOCAL_RUN
        and args.arm in {"plain", "exact"}
        and run.name == args.arm
        and args.seed == 42
        and output.parent.name in {f"round{i}" for i in range(1, 5)}
    )
    aux_run = (
        run.parent == AUX_RUN
        and args.arm == "exact_aux"
        and run.name == "exact_aux"
        and args.seed == 42
        and output.parent.name in {f"round{i}" for i in range(1, 5)}
    )
    require(
        output.name == "train"
        and output.parent.name.startswith("round")
        and ((args.arm == "RLOO" and (original_run or eight_window_run)) or local_run or aux_run),
        "output must belong to the approved new RLOO demo",
    )
    require(args.actor_output.absolute() == run / "current_actor", "only own current_actor export")
    if local_run or aux_run:
        require(args.alpha == 0.5 and args.eta == 0.25, "fixed local-credit alpha/eta")
        require(
            args.state_output.absolute() == run / "training_state",
            "only own arm persistent training_state",
        )
        require(
            args.resume_state is None or args.resume_state.absolute() == run / "training_state",
            "resume only own arm state",
        )
        round_number = int(output.parent.name.removeprefix("round"))
        require(
            (round_number == 1) == (args.resume_state is None),
            "first stage starts SFT; later stages preserve own optimizer state",
        )
        if aux_run:
            require(
                args.model.resolve()
                == (COMMON_SFT if round_number == 1 else run / "current_actor"),
                "aux starts common SFT then samples its own current actor",
            )
            require(
                args.data.resolve() == LOCAL_RUN / "common_initial"
                if round_number == 1
                else args.data.resolve().is_relative_to(run),
                "aux first batch reuses shared initial; later batches belong to this arm",
            )
    for path in (args.actor_output, *args.actor_output.parents):
        require(not path.is_symlink(), "symlink forbidden in actor export path")
    require(args.updates == 4, "fixed four-minibatch stage")
    require(args.learning_rate == 2e-6 and args.clip_epsilon == 0.2, "fixed demo LR and clip")
    return output


def run(args) -> None:
    output = validate_paths(args)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "training_summary.json").exists(), "fresh stage; no silent retry")
    batch, coverage = build_batch(
        args.data.resolve(), expected_seed=args.seed, arm=args.arm, alpha=args.alpha, eta=args.eta
    )
    args.episode_ids = coverage["episode_ids"]
    expected_episodes = (
        EIGHT_EPISODES if output.parent.parent.name in {"seed42", "seed43"} else EPISODES
    )
    require(args.episode_ids == expected_episodes, "run/data Train-window profile mismatch")
    summary = dict(
        **recipe(args),
        profile=profile_for(args),
        status="STARTING",
        seed=args.seed,
        coverage=coverage,
        training_rows=len(batch),
        invalid_rows=coverage["invalid_rows"],
        admitted_groups=coverage["admitted_windows"],
        planned_groups=coverage["planned_windows"],
        actor_updates=0,
        optimizer_updates=0,
        critic_updates=0,
        planned_optimizer_steps=4,
        actor_learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        terminal_financial_reward=True,
        invalid_penalty=PENALTY,
        head_EST_REF_used=False,
        EST_used_for_local_credit=args.arm != "RLOO",
        optimizer_state_reset_each_round=False,
        FP32_master_retained=True,
        training_state_storage=(
            "RAM-backed /dev/shm; host reboot volatile"
            if args.state_output.resolve().is_relative_to(Path("/dev/shm"))
            else "persistent project disk; sharded resume directory"
        ),
        BF16_export_storage="streamed CPU BF16 shards to own disk actor; no RAM actor stage",
        optimizer_placement="FP32 master and Adam moments on GPU",
        baseline="same-window other-three FULL trajectory returns, independent of this action",
        advantage="terminal-Y reward-to-go minus same-window leave-one-out total-return baseline",
        loss="mean response-token clipped PPO surrogate; four disjoint trajectory minibatches",
        weighting="each actual request weight=1/(minibatch trajectories*20 fixed sessions)",
        probability_reference="actual BF16 rollout snapshot before loading persistent FP32 master",
        objective_scope="critic-free bounded demo, not a complete PPO implementation",
        updates=[],
    )
    if "local_credit" in coverage:
        summary["local_credit"] = coverage["local_credit"]
        if args.arm == "exact_aux":
            summary.update(
                local_aux_weight=args.eta * args.alpha / 8,
                local_aux_fixed_denominator=8,
                local_advantage="raw exact_S / (raw_RLOO_sample_std + 1e-8); no centering",
                loss="base request PPO + separately clipped fixed-eight-anchor local PPO",
                weighting="base 1/(minibatch trajectories*20); aux eta*alpha/8 per local row",
                local_scale_note="four-window nominal local coefficient is 10x old mixed term; "
                "actual gradient effect is not inferred",
            )
        else:
            summary["advantage"] += " + eta*alpha*local_S on original replica0 j5/j15 requests"
        with (output / "local_credit_anchors.jsonl").open("x") as stream:
            for anchor in coverage["local_credit"]["anchors"]:
                append(stream, anchor)
    write(output / "training_summary.json", summary)
    try:
        require(coverage["admitted_windows"] > 0 and len(batch) > 1, "no complete RLOO group")
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
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
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
        sampled_bf16_probes = probes(model)
        summary["status"] = "OLD_LOGPROBS"
        write(output / "training_summary.json", summary)
        with torch.no_grad():
            for row in batch:
                row["old_token_logprobs"] = token_logprobs(model, row, torch).cpu().tolist()
        model.float()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
        prior_steps = 0
        if args.resume_state is not None:
            prior_steps = restore_master(model, optimizer, args, sampled_bf16_probes, torch)
        initial_master_probes = probes(model)
        advantages = torch.tensor([r["unwhitened_advantage"] for r in batch], dtype=torch.float32)
        deviation = float(advantages.std(correction=1))
        require(math.isfinite(deviation) and deviation > 1e-8, "nonzero RLOO advantage batch")
        normalized = (advantages - advantages.mean()) / (deviation + 1e-8)
        for row, value in zip(batch, normalized.tolist(), strict=True):
            row["actor_advantage"] = value
            if args.arm == "exact_aux":
                row.update(
                    base_RLOO_actor_advantage=value,
                    base_RLOO_whitening_mean=float(advantages.mean()),
                    base_RLOO_whitening_std_sample=deviation,
                    local_aux_actor_advantage=row["local_credit_S"] / (deviation + 1e-8),
                    local_aux_loss_weight=args.eta * args.alpha / 8,
                    local_aux_fixed_denominator=8,
                )
        with (output / "request_advantages.jsonl").open("x") as stream:
            for row in batch:
                append(stream, row)
        summary.update(
            status="ACTOR_UPDATES",
            optimizer_steps_before=prior_steps,
            resumed_FP32_master=args.resume_state is not None,
            resumed_Adam_moments=args.resume_state is not None,
            whitening="whole admitted-request batch; centered/sample std, correction=1",
            whitening_mean=float(advantages.mean()),
            whitening_std_sample=deviation,
        )
        if args.arm == "exact_aux":
            summary.update(
                base_RLOO_whitening_mean=float(advantages.mean()),
                base_RLOO_whitening_std_sample=deviation,
                whitening="raw RLOO only over whole admitted-request batch; "
                "local S uses its std without centering",
            )
        write(output / "training_summary.json", summary)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        admitted = [g["episode_id"] for g in coverage["groups"] if g["status"] == "COMPLETE"]
        replica_orders = {}
        for index, episode in enumerate(admitted):
            order = list(range(4))
            random.Random(args.seed + prior_steps + index).shuffle(order)
            replica_orders[episode] = order
        update_stream = (output / "optimizer_updates.jsonl").open("x")
        for minibatch in range(4):
            selected = {episode: replica_orders[episode][minibatch] for episode in admitted}
            rows = [r for r in batch if selected[r["episode_id"]] == r["replicate_index"]]
            sizes = Counter(r["trajectory_key"] for r in rows)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            losses, clipped, ratios = [], [], []
            base_losses, auxiliary_losses = [], []
            started = time.monotonic()
            for row in rows:
                new = token_logprobs(model, row, torch)
                old = torch.tensor(row["old_token_logprobs"], device="cuda", dtype=torch.float32)
                ratio = (new - old).exp()
                require(bool(torch.isfinite(ratio).all()), "nonfinite PPO probability ratio")
                advantage = row["actor_advantage"]
                surrogate = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantage,
                )
                # Fixed session count, not the realized/action-dependent request
                # count: recovery requests must not dilute an entire trajectory.
                weight = 1.0 / (len(sizes) * 20)
                if args.arm == "exact_aux":
                    loss, base_loss, auxiliary_loss = auxiliary_request_loss(
                        ratio,
                        advantage,
                        weight,
                        row["local_aux_actor_advantage"],
                        args.eta * args.alpha / 8,
                        args.clip_epsilon,
                        torch,
                    )
                    base_losses.append(float(base_loss.detach()))
                    auxiliary_losses.append(float(auxiliary_loss.detach()))
                    del base_loss, auxiliary_loss
                else:
                    loss = -surrogate.mean() * weight
                require(bool(torch.isfinite(loss)), "nonfinite PPO loss")
                loss.backward()
                losses.append(float(loss.detach()))
                ratios.append(float(ratio.detach().mean()))
                clipped.append(
                    float(((ratio.detach() - 1).abs() > args.clip_epsilon).float().mean())
                )
                del new, old, ratio, surrogate, loss
            norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            )
            require(norm > 0, "nonzero gradient required for an effective optimizer step")
            optimizer.step()
            events = [
                dict(
                    event="stock_adamw_step_complete",
                    moments_device="cuda",
                    foreach=False,
                    fused=False,
                )
            ]
            optimizer.zero_grad(set_to_none=True)
            completed = prior_steps + minibatch + 1
            require(
                optimizer_steps(optimizer) == (completed, completed), "Adam steps must accumulate"
            )
            event = dict(
                optimizer_step=completed,
                stage_minibatch=minibatch + 1,
                trajectories=len(sizes),
                requests=len(rows),
                invalid_requests=sum(r["invalid"] for r in rows),
                loss=math.fsum(losses),
                gradient_norm_before_clip=norm,
                mean_token_probability_ratio=math.fsum(ratios) / len(ratios),
                mean_request_clip_fraction=math.fsum(clipped) / len(clipped),
                seconds=time.monotonic() - started,
                optimizer_events=events,
            )
            if args.arm == "exact_aux":
                event.update(
                    base_loss=math.fsum(base_losses),
                    auxiliary_loss=math.fsum(auxiliary_losses),
                    base_request_loss_weight=1.0 / (len(sizes) * 20),
                    local_aux_loss_weight=args.eta * args.alpha / 8,
                    local_aux_fixed_denominator=8,
                    local_anchor_rows=sum(r["local_credit_status"] != "NOT_ANCHOR" for r in rows),
                    local_supported_rows=sum(r["local_credit_status"] == "COMPLETE" for r in rows),
                    local_nonzero_rows=sum(r["local_credit_S"] != 0 for r in rows),
                    base_RLOO_whitening_mean=float(advantages.mean()),
                    base_RLOO_whitening_std_sample=deviation,
                )
            append(update_stream, event)
            summary["updates"].append(event)
            summary.update(
                actor_updates=minibatch + 1,
                optimizer_updates=minibatch + 1,
                optimizer_steps_after=completed,
                completed_updates=completed,
            )
            write(output / "training_summary.json", summary)
            print(json.dumps(dict(event="OPTIMIZER_STEP_COMPLETE", **event)), flush=True)
        update_stream.close()
        model.gradient_checkpointing_disable()
        model.eval()
        positive = [r for r in batch if r["actor_advantage"] > 0][:8]
        negative = [r for r in batch if r["actor_advantage"] < 0][:8]
        diagnostic = []
        with torch.no_grad():
            for row in positive + negative:
                old = math.fsum(row["old_token_logprobs"]) / len(row["old_token_logprobs"])
                new = float(token_logprobs(model, row, torch).mean())
                diagnostic.append(
                    dict(
                        trajectory_key=row["trajectory_key"],
                        request_index=row["request_index"],
                        advantage=row["actor_advantage"],
                        old_mean_logprob=old,
                        new_mean_logprob=new,
                        delta=new - old,
                        advantage_weighted_delta=row["actor_advantage"] * (new - old),
                    )
                )
        final_master_probes = probes(model)
        expected_bf16 = {
            key: torch.tensor(value, dtype=torch.bfloat16).float().tolist()
            for key, value in final_master_probes.items()
        }
        summary.update(
            status="SAVING_FP32_MASTER",
            master_parameter_delta=probe_delta(initial_master_probes, final_master_probes),
            BF16_parameter_delta=probe_delta(sampled_bf16_probes, expected_bf16),
            logprob_diagnostics=diagnostic,
        )
        write(output / "training_summary.json", summary)
        summary["persistent_state"] = save_master(
            model, optimizer, args, prior_steps + 4, expected_bf16, torch
        )
        summary.update(status="EXPORTING")
        write(output / "training_summary.json", summary)
        summary["checkpoint_readback"] = export_bf16(
            model,
            tokenizer,
            args.actor_output,
            expected_bf16,
            args.state_output.resolve().parent,
            torch,
        )
        summary.update(
            status="COMPLETE",
            actor_checkpoint=str(args.actor_output.resolve()),
            training_state=str(args.state_output.resolve()),
            optimizer_steps_after=prior_steps + 4,
        )
        write(output / "training_summary.json", summary)
        print(json.dumps(dict(event="TRAINING_STAGE_COMPLETE", **summary)), flush=True)
    except BaseException as exc:
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        write(output / "training_summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--arm", choices=("RLOO", "plain", "exact", "exact_aux"), default="RLOO")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.25)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
