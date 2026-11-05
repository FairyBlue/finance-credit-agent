"""Two explicitly migrated 32-update RLOO stages over 32 Train windows.

Each update covers four distinct complete trajectories. FP32 master parameters
and GPU Adam moments migrate once from the completed seed43 sixteen-step run,
then remain in this experiment's own state. Only terminal BF16 actors export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.agents.rloo_terminal_train import (
    EIGHT_EPISODES,
    _read_component,
    _write_component,
    export_bf16,
    optimizer_steps,
    recipe,
    require_recipe,
    token_logprobs,
    trajectory_rows,
)
from src.agents.rloo_terminal_train import (
    PROFILE as LEGACY_PROFILE,
)
from src.agents.same_prefix_online_train import records
from src.agents.same_prefix_training_demo import ROOT, append, probe_delta, probes, require, write

PROFILE = "broad_32_window_terminal_RLOO_persistent_v1"
EXPERIMENT_PROFILE = "broad_rloo_order_main_v1"
RUN = ROOT / "runs/m2-broad-rloo-order-main-20261028-v1"
LEGACY_RUN = ROOT / "runs/m2-rloo-eight-window-20261025-v1/seed43"
UPDATES = 32


def configuration(args):
    path = args.experiment_config.resolve()
    require(path == RUN / "config.json", "explicit approved experiment config required")
    payload = path.read_bytes()
    config = json.loads(payload)
    episodes = config["train_episode_ids"]
    require(config["profile"] == EXPERIMENT_PROFILE, "declared broad experiment profile")
    require(config.get("seed", 43) == args.seed == 43, "fixed seed43")
    require(
        len(episodes) == len(set(episodes)) == 32
        and all(e.startswith("formal-train-") for e in episodes),
        "32 declared Train windows only",
    )
    return config, hashlib.sha256(payload).hexdigest()


def validate_paths(args):
    config, digest = configuration(args)
    args.episode_ids = config["train_episode_ids"]
    args.arm, args.alpha, args.eta = "RLOO", 0.0, 0.0
    output = args.output.resolve()
    require(output in {RUN / "round1/train", RUN / "round2/train"}, "owned broad train stage")
    number = int(output.parent.name.removeprefix("round"))
    require(args.updates == UPDATES, "each broad stage has exactly32 updates")
    require(args.learning_rate == 2e-6 and args.clip_epsilon == 0.2, "unchanged LR and clip")
    require(args.state_output.resolve() == RUN / "training_state", "own persistent state output")
    require(args.actor_output.resolve() == RUN / "current_actor", "own final actor output")
    require(
        args.model.resolve()
        == (LEGACY_RUN / "current_actor" if number == 1 else RUN / "current_actor"),
        "first stage samples frozen old actor, second its own committed actor",
    )
    require(
        args.resume_state.resolve()
        == (LEGACY_RUN / "training_state" if number == 1 else RUN / "training_state"),
        "first explicit legacy migration then own immediately previous state",
    )
    require(args.data.resolve().is_relative_to(RUN), "training samples belong to this run")
    for path in (args.state_output, args.actor_output):
        require(not path.is_symlink(), "owned checkpoint path must not be symlink")
    return output, number, digest


def build_batch(directory: Path, episodes: list[str], model_path: Path):
    source = json.loads((directory / "collection_summary.json").read_text())
    require(
        source["seed"] == 43
        and source["episode_ids"] == episodes
        and source.get("parents_only") is True
        and source.get("parent_repetitions") == 4
        and source.get("planned_parent_trajectories") == 128
        and Path(source["model_path"]).resolve() == model_path.resolve()
        and source.get("decision_mode", "native") == "native",
        "actual frozen native parent collection for the declared32 Train windows required",
    )
    indexed = defaultdict(dict)
    for parent in records(directory / "trajectory_results.jsonl"):
        require(parent["role"] == "parent", "only full parent trajectories train")
        episode, rep = parent["episode_id"], parent["replicate_index"]
        require(
            episode in episodes
            and type(rep) is int
            and 0 <= rep < 4
            and rep not in indexed[episode],
            "unique original episode/replica slot",
        )
        indexed[episode][rep] = parent
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
                    reason="complete four-peer RLOO group unavailable; no filling or resampling",
                )
            )
            continue
        rows_and_details = [trajectory_rows(parents[rep]) for rep in range(4)]
        totals = [detail["total_training_return"] for _, detail in rows_and_details]
        for rep, (rows, _) in enumerate(rows_and_details):
            baseline = math.fsum(v for j, v in enumerate(totals) if j != rep) / 3
            for row in rows:
                row.update(
                    rloo_baseline=baseline,
                    trajectory_total_return=totals[rep],
                    unwhitened_advantage=row["return_G"] - baseline,
                )
            batch.extend(rows)
        groups.append(
            dict(
                episode_id=episode,
                status="COMPLETE",
                planned_trajectories=4,
                complete_trajectories=4,
                total_returns=totals,
                trajectories=[detail for _, detail in rows_and_details],
            )
        )
    return batch, dict(
        episode_ids=episodes,
        seed=43,
        planned_windows=32,
        planned_trajectories=128,
        admitted_windows=sum(g["status"] == "COMPLETE" for g in groups),
        admitted_trajectories=4 * sum(g["status"] == "COMPLETE" for g in groups),
        training_rows=len(batch),
        invalid_rows=sum(r["invalid"] for r in batch),
        groups=groups,
        head_EST_REF_used=False,
    )


def minibatch_schedule(episodes: list[str], seed: int, prior_steps: int):
    """Use128 parent slots, cycling supported parents only when windows failed."""
    require(4 <= len(episodes) <= 32, "at least four complete windows needed")
    rng = random.Random(seed + prior_steps)
    order = list(episodes)
    rng.shuffle(order)
    replica_orders = {episode: list(range(4)) for episode in order}
    for replicas in replica_orders.values():
        rng.shuffle(replicas)
    schedule, visits = [], Counter()
    for step in range(32):
        selected = {}
        for offset in range(4):
            episode = order[(4 * step + offset) % len(order)]
            selected[episode] = replica_orders[episode][visits[episode] % 4]
            visits[episode] += 1
        schedule.append(selected)
    require(
        len(schedule) == 32 and all(len(s) == 4 for s in schedule), "32 four-parent minibatches"
    )
    slots = [(episode, rep) for s in schedule for episode, rep in s.items()]
    require(
        len(slots) == 128 and len(set(slots)) == 4 * len(episodes),
        "all admitted original parents used; only deterministic supported-parent reuse",
    )
    return schedule


def restore(model, optimizer, args, number, digest, bf16_probes, torch):
    directory = args.resume_state.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    saved = torch.load(directory / "metadata.pt", map_location="cpu", weights_only=False)
    require(manifest["status"] == "COMPLETE", "committed persistent input state required")
    require_recipe(manifest, args)
    require_recipe(saved, args)
    expected = 16 if number == 1 else 48
    expected_profile = LEGACY_PROFILE if number == 1 else PROFILE
    require(manifest["profile"] == saved["profile"] == expected_profile, "explicit source profile")
    require(
        manifest["completed_optimizer_steps"] == saved["completed_optimizer_steps"] == expected,
        "resume immediately previous16 or48 optimizer steps",
    )
    require(saved["seed"] == args.seed, "resume seed unchanged")
    if number == 1:
        require(
            manifest["episode_ids"] == saved["episode_ids"] == EIGHT_EPISODES,
            "only the completed original eight-window seed43 source can migrate",
        )
    else:
        require(
            manifest["episode_ids"] == saved["episode_ids"] == args.episode_ids
            and manifest["experiment_config_sha256"] == saved["experiment_config_sha256"] == digest,
            "own32-window universe and explicit experiment config unchanged",
        )
    require(saved["bf16_export_probes"] == bf16_probes, "sampling actor and FP32 source correspond")
    parameters, tensors = dict(model.named_parameters()), model.state_dict()
    require(saved["parameter_names"] == list(parameters), "unchanged parameter order")
    require([e["name"] for e in manifest["entries"]] == list(tensors), "unchanged tensor order")
    for key, value in saved["optimizer_hyperparameters"].items():
        require(value == optimizer.param_groups[0][key], f"unchanged Adam hyperparameter {key}")
    with torch.no_grad():
        for index, entry in enumerate(manifest["entries"]):
            name = entry["name"]
            source = _read_component(directory, entry, "master", expected, torch)
            require(
                source.dtype == tensors[name].dtype and source.shape == tensors[name].shape,
                "master shape/dtype unchanged",
            )
            tensors[name].copy_(source)
            del source
            if name in parameters:
                parameter = parameters[name]
                require(entry["optimizer_step"] == expected, "per-parameter source step")
                state = {
                    "step": torch.tensor(
                        expected, dtype=getattr(torch, entry["step_dtype"]), device="cpu"
                    )
                }
                for kind in ("exp_avg", "exp_avg_sq"):
                    source = _read_component(directory, entry, kind, expected, torch)
                    require(
                        source.dtype == torch.float32 and source.shape == parameter.shape,
                        "Adam moment shape/dtype unchanged",
                    )
                    state[kind] = source.to(device=parameter.device, copy=True)
                    del source
                optimizer.state[parameter] = state
            if index % 50 == 0:
                print(
                    json.dumps(
                        dict(
                            event="RESUME_COMPONENT_PROGRESS",
                            tensor_index=index,
                            source_updates=expected,
                        )
                    ),
                    flush=True,
                )
    require(probes(model) == saved["master_probes"], "FP32 master precision preserved")
    require(optimizer_steps(optimizer) == (expected, expected), "Adam counts preserved")
    torch.set_rng_state(saved["torch_rng_state"])
    if saved["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
    random.setstate(saved["python_rng_state"])
    return expected


def save_state(model, optimizer, args, completed, digest, bf16_probes, torch):
    directory = args.state_output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    common = dict(
        profile=PROFILE,
        **recipe(args),
        seed=args.seed,
        episode_ids=args.episode_ids,
        experiment_config_sha256=digest,
        source_migration="legacy seed43 step16 to32 Train windows",
    )
    write(manifest_path, dict(**common, status="SAVING", target_updates=completed))
    parameters = dict(model.named_parameters())
    entries, total = [], 0
    for index, (name, value) in enumerate(model.state_dict().items()):
        entry = dict(name=name, files={"master": f"tensor-{index:04d}.pt"})
        total += _write_component(
            directory / entry["files"]["master"], name, "master", value, completed, torch
        )
        if name in parameters:
            state = optimizer.state[parameters[name]]
            require(
                value.dtype == torch.float32 and int(state["step"].item()) == completed,
                "persistent FP32 master and Adam step",
            )
            entry.update(
                optimizer_step=completed, step_dtype=str(state["step"].dtype).removeprefix("torch.")
            )
            for kind in ("exp_avg", "exp_avg_sq"):
                moment = state[kind]
                require(
                    moment.device == value.device and moment.dtype == torch.float32,
                    "FP32 moments remain on master GPU",
                )
                entry["files"][kind] = f"{kind}-{index:04d}.pt"
                total += _write_component(
                    directory / entry["files"][kind], name, kind, moment, completed, torch
                )
        entries.append(entry)
        if index % 50 == 0:
            print(
                json.dumps(
                    dict(
                        event="CHECKPOINT_COMPONENT_PROGRESS",
                        tensor_index=index,
                        bytes_written=total,
                        completed_updates=completed,
                    )
                ),
                flush=True,
            )
    group = optimizer.param_groups[0]
    metadata = dict(
        **common,
        completed_optimizer_steps=completed,
        learning_rate=args.learning_rate,
        parameter_names=list(parameters),
        master_probes=probes(model),
        bf16_export_probes=bf16_probes,
        optimizer_hyperparameters={k: group[k] for k in ("lr", "betas", "eps", "weight_decay")},
        actor_path=str(args.actor_output.resolve()),
        torch_rng_state=torch.get_rng_state(),
        python_rng_state=random.getstate(),
        cuda_rng_state=torch.cuda.get_rng_state(),
    )
    temporary = directory / ".metadata.pt.saving"
    torch.save(metadata, temporary)
    temporary.replace(directory / "metadata.pt")
    write(
        manifest_path,
        dict(
            **common,
            status="COMPLETE",
            completed_optimizer_steps=completed,
            bytes=total,
            entries=entries,
            storage="persistent project disk",
        ),
    )
    return dict(
        path=str(directory),
        bytes=total,
        shard_files=sum(len(e["files"]) for e in entries),
        completed_optimizer_steps=completed,
        master_dtype="float32",
        moments_device="cuda",
        interrupted_save_is_resumable=False,
    )


def run(args):
    output, number, digest = validate_paths(args)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "training_summary.json").exists(), "fresh stage; no implicit retry")
    batch, coverage = build_batch(args.data.resolve(), args.episode_ids, args.model.resolve())
    summary = dict(
        profile=PROFILE,
        status="STARTING",
        experiment_config_sha256=digest,
        seed=args.seed,
        round=number,
        arm="RLOO",
        coverage=coverage,
        planned_optimizer_steps=32,
        optimizer_updates=0,
        actor_updates=0,
        critic_updates=0,
        training_rows=len(batch),
        invalid_rows=coverage["invalid_rows"],
        planned_groups=32,
        admitted_groups=coverage["admitted_windows"],
        resume_state=str(args.resume_state.resolve()),
        source_updates=16 if number == 1 else 48,
        completed_updates=16 if number == 1 else 48,
        updates=[],
        actor_learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        terminal_financial_reward=True,
        invalid_penalty=0.05,
        head_EST_REF_used=False,
        baseline="same-window other-three complete trajectory total returns",
        advantage="terminal-Y reward-to-go minus leave-one-out full-return baseline",
        loss="mean response-token clipped PPO;32 four-trajectory minibatches",
        weighting="each real request1/(4 trajectories*20 fixed sessions)",
        optimizer_state_reset_each_round=False,
        FP32_master_retained=True,
        probability_reference="actual BF16 sampled policy before exact FP32 state restore",
        training_state_storage="persistent project disk; one owned component-sharded state",
    )
    write(output / "training_summary.json", summary)
    try:
        require(
            coverage["admitted_windows"] >= 4,
            "at least four complete four-peer windows required; no failed-path filling",
        )
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
        initial_bf16 = probes(model)
        summary.update(status="OLD_LOGPROBS", old_logprob_rows_complete=0)
        write(output / "training_summary.json", summary)
        with torch.no_grad():
            for index, row in enumerate(batch, 1):
                row["old_token_logprobs"] = token_logprobs(model, row, torch).cpu().tolist()
                if index % 128 == 0 or index == len(batch):
                    summary["old_logprob_rows_complete"] = index
                    write(output / "training_summary.json", summary)
                    print(
                        json.dumps(
                            dict(event="OLD_LOGPROB_PROGRESS", completed=index, total=len(batch))
                        ),
                        flush=True,
                    )
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
        summary.update(status="RESTORING_FP32_ADAM")
        write(output / "training_summary.json", summary)
        prior = restore(model, optimizer, args, number, digest, initial_bf16, torch)
        before_master = probes(model)
        values = torch.tensor([r["unwhitened_advantage"] for r in batch], dtype=torch.float32)
        std = float(values.std(correction=1))
        require(math.isfinite(std) and std > 1e-8, "nonzero finite RLOO advantage batch")
        for row, advantage in zip(
            batch, ((values - values.mean()) / (std + 1e-8)).tolist(), strict=True
        ):
            row["actor_advantage"] = advantage
        with (output / "request_advantages.jsonl").open("x") as stream:
            for row in batch:
                append(stream, row)
        admitted = [g["episode_id"] for g in coverage["groups"] if g["status"] == "COMPLETE"]
        schedule = minibatch_schedule(admitted, args.seed, prior)
        indexed = defaultdict(list)
        for row in batch:
            indexed[(row["episode_id"], row["replicate_index"])].append(row)
        summary.update(
            status="ACTOR_UPDATES",
            optimizer_steps_before=prior,
            resumed_FP32_master=True,
            resumed_Adam_moments=True,
            whitening="whole128-parent admitted request batch; centered/sample std",
            whitening_mean=float(values.mean()),
            whitening_std_sample=std,
            parent_uses=128,
            unique_parent_uses=4 * len(admitted),
            repeated_parent_uses=128 - 4 * len(admitted),
            repeat_rule="none with32 complete windows; otherwise cycle admitted parents only",
        )
        write(output / "training_summary.json", summary)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        with (output / "optimizer_updates.jsonl").open("x") as stream:
            for step, selected in enumerate(schedule, 1):
                rows = [r for e, rep in selected.items() for r in indexed[(e, rep)]]
                sizes = Counter(r["trajectory_key"] for r in rows)
                require(len(sizes) == 4, "four complete trajectories per optimizer step")
                optimizer.zero_grad(set_to_none=True)
                model.train()
                tick = time.monotonic()
                losses, ratios, clips = [], [], []
                for row in rows:
                    new = token_logprobs(model, row, torch)
                    old = torch.tensor(
                        row["old_token_logprobs"], device="cuda", dtype=torch.float32
                    )
                    ratio = (new - old).exp()
                    require(bool(torch.isfinite(ratio).all()), "finite PPO ratio")
                    advantage = row["actor_advantage"]
                    surrogate = torch.minimum(
                        ratio * advantage,
                        ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantage,
                    )
                    loss = -surrogate.mean() / (4 * 20)
                    require(bool(torch.isfinite(loss)), "finite PPO loss")
                    loss.backward()
                    losses.append(float(loss.detach()))
                    ratios.append(float(ratio.detach().mean()))
                    clips.append(
                        float(((ratio.detach() - 1).abs() > args.clip_epsilon).float().mean())
                    )
                    del new, old, ratio, surrogate, loss
                norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                )
                require(norm > 0, "nonzero gradient required for effective optimizer step")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed = prior + step
                require(
                    optimizer_steps(optimizer) == (completed, completed), "persistent Adam counters"
                )
                event = dict(
                    optimizer_step=completed,
                    stage_minibatch=step,
                    trajectories=4,
                    requests=len(rows),
                    invalid_requests=sum(r["invalid"] for r in rows),
                    prompt_tokens=sum(len(r["prompt_token_ids"]) for r in rows),
                    response_tokens=sum(len(r["response_token_ids"]) for r in rows),
                    loss=math.fsum(losses),
                    gradient_norm_before_clip=norm,
                    mean_token_probability_ratio=math.fsum(ratios) / len(ratios),
                    mean_request_clip_fraction=math.fsum(clips) / len(clips),
                    seconds=time.monotonic() - tick,
                    parent_slots=selected,
                )
                append(stream, event)
                summary["updates"].append(event)
                summary.update(
                    actor_updates=step,
                    optimizer_updates=step,
                    completed_updates=completed,
                    optimizer_steps_after=completed,
                )
                write(output / "training_summary.json", summary)
                print(json.dumps(dict(event="OPTIMIZER_STEP_COMPLETE", **event)), flush=True)
        model.gradient_checkpointing_disable()
        model.eval()
        diagnostics = []
        chosen = [r for r in batch if r["actor_advantage"] > 0][:8] + [
            r for r in batch if r["actor_advantage"] < 0
        ][:8]
        with torch.no_grad():
            for row in chosen:
                before = math.fsum(row["old_token_logprobs"]) / len(row["old_token_logprobs"])
                after = float(token_logprobs(model, row, torch).mean())
                diagnostics.append(
                    dict(
                        trajectory_key=row["trajectory_key"],
                        request_index=row["request_index"],
                        advantage=row["actor_advantage"],
                        old_mean_logprob=before,
                        new_mean_logprob=after,
                        delta=after - before,
                    )
                )
        final_master = probes(model)
        bf16 = {
            k: torch.tensor(v, dtype=torch.bfloat16).float().tolist()
            for k, v in final_master.items()
        }
        summary.update(
            status="SAVING_FP32_MASTER",
            logprob_diagnostics=diagnostics,
            master_parameter_delta=probe_delta(before_master, final_master),
            BF16_parameter_delta=probe_delta(initial_bf16, bf16),
        )
        write(output / "training_summary.json", summary)
        summary["persistent_state"] = save_state(
            model, optimizer, args, prior + 32, digest, bf16, torch
        )
        summary.update(status="EXPORTING")
        write(output / "training_summary.json", summary)
        summary["checkpoint_readback"] = export_bf16(
            model, tokenizer, args.actor_output, bf16, RUN, torch
        )
        summary.update(
            status="COMPLETE",
            actor_checkpoint=str(args.actor_output.resolve()),
            training_state=str(args.state_output.resolve()),
            optimizer_steps_after=prior + 32,
            training_request_uses=sum(e["requests"] for e in summary["updates"]),
            repeated_request_uses=sum(e["requests"] for e in summary["updates"]) - len(batch),
            training_prompt_tokens=sum(e["prompt_tokens"] for e in summary["updates"]),
            training_response_tokens=sum(e["response_tokens"] for e in summary["updates"]),
            peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            peak_GPU_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
        )
        write(output / "training_summary.json", summary)
        print(
            json.dumps(
                dict(
                    event="BROAD_TRAIN_STAGE_COMPLETE",
                    completed_updates=prior + 32,
                    optimizer_updates=32,
                )
            ),
            flush=True,
        )
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        write(output / "training_summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--resume-state", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
