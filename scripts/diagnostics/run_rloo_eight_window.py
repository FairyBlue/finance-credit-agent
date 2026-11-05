#!/usr/bin/env python3
"""Run the fixed eight-Train-window RLOO replication for seeds 42 and 43."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import signal
import uuid
from pathlib import Path
from types import SimpleNamespace

from run_same_prefix_online_demo import (
    COLLECTOR,
    MODEL,
    ROOT,
    Controller,
    collection,
    gpu,
    read,
    save,
)

DEFAULT = ROOT / "runs/m2-rloo-eight-window-20261025-v1"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157)]
TRAINER = "src.agents.rloo_terminal_train"
SEEDS = (42, 43)


def observation(summary):
    values, metrics = summary["parent_terminal_Y"], summary["parent_terminal_metrics"]
    complete = (
        len(values) == len(metrics) == 32
        and summary["parents_complete"] == 32
        and all(v is not None and math.isfinite(v) for v in values)
        and all(m is not None for m in metrics)
    )
    return dict(
        mean_Y=math.fsum(values) / 32 if complete else None,
        mean_raw_net_return=math.fsum(m["raw_net_return"] for m in metrics) / 32
        if complete
        else None,
        invalid_requests=sum(m["invalid_count"] for m in metrics) if complete else None,
        mean_transaction_cost=math.fsum(m["transaction_cost"] for m in metrics) / 32
        if complete
        else None,
        parents_complete=summary["parents_complete"],
        parents_failed=summary["parents_failed"],
        model_requests=summary["model_requests"],
    )


class SeedRun(Controller):
    def __init__(self, suite, seed):
        self.suite, self.seed = suite, seed
        self.initial_only = suite.args.collect_only
        args = SimpleNamespace(
            output=suite.output / f"seed{seed}",
            model=MODEL,
            collect_only=self.initial_only,
            seed=seed,
            episode_ids=EPISODES,
            alpha=0.0,
            eta=0.0,
            stage_timeout_seconds=suite.args.stage_timeout_seconds,
            coordinator_request_timeout_seconds=60,
        )
        super().__init__(args)
        self.config = suite.config
        self.state_path = self.output / (
            "initial_state.json" if self.initial_only else "rloo_state.json"
        )
        self.scratch = self.output / "training_state"
        self.attempt = 1
        self.state.update(
            phase="eight_window_rloo",
            rounds=4,
            arms=["rloo"],
            planned_updates=16,
            parent_repetitions=4,
        )
        self.result = dict(
            profile="eight_window_two_seed_terminal_RLOO_v1",
            status="RUNNING",
            planned_updates=16,
            updates_per_round=4,
            seed=seed,
            episode_ids=EPISODES,
            parent_repetitions=4,
            planned_parent_trajectories=32,
            baseline=None,
            baseline_path=str(self.output / "common_initial"),
            rounds=[],
            training_state=str(self.scratch),
            training_state_storage="persistent project disk: FP32 weights and Adam moments",
            actor_checkpoint=str(self.output / "current_actor"),
            critic_updates=0,
            best_round_selection=False,
            failed_run_updates_replayed_from_SFT=0,
        )

    def update(self, **values):
        super().update(**values)
        self.suite.state.update(
            current_seed=self.seed, current_stage=self.state.get("current_stage")
        )
        self.suite.persist()

    def persist(self):
        save(
            self.output / ("initial_results.json" if self.initial_only else "rloo_results.json"),
            self.result,
        )
        if self.initial_only or self.result["baseline"] is None:
            return
        rows = [dict(updates=0, round=0, status="BASELINE", **observation(self.result["baseline"]))]
        keys = list(observation(self.result["baseline"]))
        for entry in self.result["rounds"]:
            rows.append(
                dict(
                    updates=entry["round"] * 4,
                    round=entry["round"],
                    status=entry["status"],
                    **(
                        observation(entry["evaluation"])
                        if entry.get("evaluation")
                        else dict.fromkeys(keys)
                    ),
                )
            )
        with (self.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def checked_collection(self, path, actor, namespace):
        result = collection(path, actor, EPISODES, self.seed)
        if (
            result.get("parent_repetitions") != 4
            or result.get("planned_parent_trajectories") != 32
            or result["parents_only"] is not True
            or result["round_index"] != namespace
        ):
            raise RuntimeError("collection is not this fixed eight-window grouped batch")
        return result

    def fresh_collection_path(self, destination):
        if not destination.exists() or not any(destination.iterdir()):
            return destination
        if (destination / "collection_summary.json").exists():
            return destination
        if not self.suite.args.resume_completed_stages:
            raise FileExistsError("existing incomplete collection; explicit resume required")
        for name in ("requests.jsonl", "trajectory_results.jsonl", "head_results.jsonl"):
            path = destination / name
            if path.exists() and path.stat().st_size:
                raise RuntimeError("generated partial collection cannot be resampled")
        destination = destination.with_name(f"{destination.name}-attempt{self.attempt}")
        if destination.exists():
            raise FileExistsError("fresh pre-generation retry directory required")
        return destination

    def grouped_collect(self, name, actor, path, namespace):
        if not (path / "collection_summary.json").exists():
            suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
            self.command(
                name + suffix,
                COLLECTOR,
                [
                    "--model",
                    str(actor),
                    "--output",
                    str(path),
                    "--episode-ids",
                    *EPISODES,
                    "--seed",
                    str(self.seed),
                    "--round-index",
                    str(namespace),
                    "--parents-only",
                    "--parent-repetitions",
                    "4",
                ],
            )
        return self.checked_collection(path, actor, namespace)

    def prepare_resume(self):
        previous, result = read(self.state_path), read(self.output / "rloo_results.json")
        if (
            previous["status"] not in {"FAILED", "PAUSED"}
            or result["seed"] != self.seed
            or result["episode_ids"] != EPISODES
            or Path(result["training_state"]) != self.scratch
        ):
            raise RuntimeError("resume requires this stopped seed profile")
        entries = result["rounds"]
        if [e["round"] for e in entries] != list(range(1, len(entries) + 1)):
            raise RuntimeError("noncontiguous existing training rounds")
        for entry in entries:
            training = read(self.output / f"round{entry['round']}/train/training_summary.json")
            if (
                training["status"] != "COMPLETE"
                or training["optimizer_updates"] != 4
                or training["completed_updates"] != entry["round"] * 4
            ):
                raise RuntimeError("partial training cannot be silently replayed")
            entry["training"] = training
        if entries:
            manifest = read(self.scratch / "manifest.json")
            actor = self.output / "current_actor"
            index = read(actor / "model.safetensors.index.json")
            if (
                manifest["status"] != "COMPLETE"
                or manifest["completed_optimizer_steps"] != len(entries) * 4
                or (actor / "EXPORT_INCOMPLETE.json").exists()
                or not all((actor / f).is_file() for f in set(index["weight_map"].values()))
            ):
                raise RuntimeError("resume requires committed FP32/Adam and complete BF16 actor")
        elif self.scratch.exists():
            raise RuntimeError("uncommitted scratch state cannot resume")
        self.attempt = previous.get("controller_attempt", 1) + 1
        archive = self.output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for filename in ("rloo_state.json", "rloo_results.json", "completion.json", "results.csv"):
            source, target = (
                self.output / filename,
                archive / f"attempt{self.attempt - 1}_{filename}",
            )
            if source.exists():
                if target.exists():
                    raise FileExistsError("previous attempt snapshot already exists")
                shutil.copy2(source, target)
        self.state, self.result = previous, result
        self.state.update(status="RUNNING", current_stage=None, controller_attempt=self.attempt)
        self.result.update(status="RUNNING", controller_attempt=self.attempt)
        self.state.pop("error", None)
        self.result.pop("error", None)

    def execute_seed(self):
        if self.state_path.exists():
            if not self.suite.args.resume_completed_stages or self.initial_only:
                raise FileExistsError("fresh phase required; no implicit retry")
            self.prepare_resume()
        elif self.scratch.exists():
            raise FileExistsError("fresh training needs unused state directory")
        entry = None
        try:
            self.update()
            self.persist()
            data = self.fresh_collection_path(Path(self.result["baseline_path"]))
            baseline_reused = (data / "collection_summary.json").exists()
            baseline = self.grouped_collect("common_initial", MODEL, data, 0)
            self.result.update(
                baseline=baseline,
                baseline_path=str(data),
                baseline_reused_in_training_phase=baseline_reused,
            )
            self.persist()
            if self.initial_only:
                self.result["status"] = "COLLECTED"
                self.update(status="COMPLETE", current_stage=None)
                return
            actor = MODEL
            existing = {e["round"]: e for e in self.result["rounds"]}
            for number in range(1, 5):
                entry = existing.get(number)
                if entry is None:
                    entry = dict(
                        round=number,
                        status="TRAINING",
                        training=None,
                        evaluation=None,
                        training_data_path=str(data),
                    )
                    self.result["rounds"].append(entry)
                    self.persist()
                    arguments = [
                        "--data",
                        str(data),
                        "--output",
                        str(self.output / f"round{number}/train"),
                        "--model",
                        str(actor),
                        "--state-output",
                        str(self.scratch),
                        "--actor-output",
                        str(self.output / "current_actor"),
                        "--updates",
                        "4",
                        "--seed",
                        str(self.seed),
                        "--learning-rate",
                        "2e-6",
                        "--clip-epsilon",
                        "0.2",
                    ]
                    if number > 1:
                        arguments += ["--resume-state", str(self.scratch)]
                    suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
                    self.command(f"round{number}_train{suffix}", TRAINER, arguments)
                    entry["training"] = read(
                        self.output / f"round{number}/train/training_summary.json"
                    )
                training = entry["training"]
                if (
                    training["status"] != "COMPLETE"
                    or training["optimizer_updates"] != 4
                    or training["completed_updates"] != number * 4
                ):
                    raise RuntimeError("exactly four persistent optimizer steps required")
                actor = self.output / "current_actor"
                if entry.get("evaluation") is not None:
                    data = Path(entry["collection_path"])
                    continue
                entry.update(status="COLLECTING")
                entry.pop("error", None)
                self.persist()
                data = self.fresh_collection_path(self.output / f"round{number}/eval")
                namespace = 0 if number == 4 else number
                evaluation = self.grouped_collect(f"round{number}_parents", actor, data, namespace)
                entry.update(
                    status="COMPLETE",
                    evaluation=evaluation,
                    collection_path=str(data),
                    evaluation_seed_namespace=namespace,
                    paired_with_initial_sampling_seeds=number == 4,
                )
                self.persist()
            self.result.update(status="COMPLETE", completed_updates=16)
            self.update(status="COMPLETE", current_stage=None)
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if entry is not None:
                entry.update(status="FAILED", error=message)
                partial = self.output / f"round{entry['round']}/train/training_summary.json"
                if entry["training"] is None and partial.exists():
                    entry["training"] = read(partial)
            self.result.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            raise
        finally:
            self.stop_child()
            self.persist()
            evaluations = [e["evaluation"] for e in self.result["rounds"] if e.get("evaluation")]
            save(
                self.output
                / ("initial_completion.json" if self.initial_only else "completion.json"),
                dict(
                    status=self.result["status"],
                    seed=self.seed,
                    planned_updates=16,
                    completed_updates=sum(
                        e["training"].get("optimizer_updates", 0)
                        for e in self.result["rounds"]
                        if e.get("training")
                    ),
                    completed_rounds=sum(e["status"] == "COMPLETE" for e in self.result["rounds"]),
                    stage_wall_seconds=sum(s.get("seconds", 0) for s in self.state["stages"]),
                    training_state=str(self.scratch),
                    training_state_is_host_reboot_volatile=False,
                    actor_checkpoint=str(self.output / "current_actor"),
                    critic_updates=0,
                    baseline_reused_in_training_phase=self.result.get(
                        "baseline_reused_in_training_phase", False
                    ),
                    baseline_collection_cost={
                        k: self.result["baseline"].get(k, 0)
                        for k in ("model_requests", "prompt_tokens", "response_tokens")
                    }
                    if self.result["baseline"]
                    else None,
                    new_post_update_collection_cost={
                        k: sum(e.get(k, 0) for e in evaluations)
                        for k in ("model_requests", "prompt_tokens", "response_tokens")
                    },
                    final_evaluation_shares_initial_sampling_seed_namespace=True,
                ),
            )


class Suite:
    def __init__(self, args):
        self.args, self.output = args, args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.active, self.config, self.claimed = None, None, False
        self.seeds = (args.seed,) if args.collect_only else SEEDS
        self.prefix = f"initial_seed{args.seed}" if args.collect_only else "suite"
        self.state = dict(
            status="RUNNING",
            seeds=list(self.seeds),
            current_seed=None,
            current_stage=None,
            planned_updates=0 if args.collect_only else 32,
        )
        self.result = dict(
            profile="eight_window_two_seed_RLOO_replication_v1",
            status="RUNNING",
            seeds={
                str(s): dict(status="NOT_RUN", directory=str(self.output / f"seed{s}"))
                for s in self.seeds
            },
        )

    def persist(self):
        save(self.output / f"{self.prefix}_state.json", self.state)
        save(self.output / f"{self.prefix}_results.json", self.result)

    def execute(self):
        state_path = self.output / f"{self.prefix}_state.json"
        if state_path.exists() and not self.args.resume_completed_stages:
            raise FileExistsError("fresh suite phase required; explicit resume only")
        if state_path.exists():
            old = read(state_path)
            if old["status"] not in {"FAILED", "PAUSED"}:
                raise RuntimeError("suite resume requires a failed/paused previous attempt")
            attempt = old.get("controller_attempt", 1) + 1
            self.state["controller_attempt"] = attempt
            archive = self.output / "controller_attempts"
            archive.mkdir(exist_ok=True)
            for name in (
                f"{self.prefix}_state.json",
                f"{self.prefix}_results.json",
                f"{self.prefix}_completion.json",
            ):
                source, target = self.output / name, archive / f"attempt{attempt - 1}_{name}"
                if source.exists():
                    if target.exists():
                        raise FileExistsError("suite snapshot already exists")
                    shutil.copy2(source, target)
        holder = self.args.holder_directory or Path(
            read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"]
        )
        attempt = self.state.get("controller_attempt", 1)
        self.config = gpu.config_for_holder(
            holder,
            lease_token=f"rloo-eight-{uuid.uuid4().hex}",
            evidence_directory=self.output / f"{self.prefix}_gpu_attempt{attempt}",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(
            self.output / f"{self.prefix}_gpu_config_attempt{attempt}.json", self.config
        )
        self.persist()

        def forward(signum, _frame):
            if self.active is not None and self.active.child is not None:
                self.active.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            state = gpu.controller_claim(self.config)
            self.claimed = True
            gpu.request(
                str(holder),
                dict(
                    op="release",
                    lease_token=self.config["lease_token"],
                    bytes=state["maximum_bytes"],
                    request_id=f"release-{uuid.uuid4().hex}",
                ),
                timeout=60,
            )
            checked = gpu.reservation_status(self.config)
            if checked["paused"] is not True or any(
                checked[k]
                for k in (
                    "payload_bytes",
                    "allocator_allocated_bytes",
                    "allocator_reserved_bytes",
                )
            ):
                raise RuntimeError("holder must be paused and empty")
            for seed in self.seeds:
                previous = self.output / f"seed{seed}/rloo_results.json"
                if not self.args.collect_only and previous.exists():
                    result = read(previous)
                    if result["status"] == "COMPLETE":
                        if not self.args.resume_completed_stages:
                            raise FileExistsError("existing completed seed; explicit suite resume")
                        self.result["seeds"][str(seed)].update(
                            status="COMPLETE", reused_complete=True
                        )
                        continue
                self.active = SeedRun(self, seed)
                self.result["seeds"][str(seed)]["status"] = "RUNNING"
                self.persist()
                self.active.execute_seed()
                self.result["seeds"][str(seed)].update(
                    status=self.active.result["status"],
                    result_path=str(
                        self.active.output
                        / (
                            "initial_results.json"
                            if self.args.collect_only
                            else "rloo_results.json"
                        )
                    ),
                )
                self.persist()
            self.state.update(status="COMPLETE", current_seed=None, current_stage=None)
            self.result["status"] = "COMPLETE"
            return 0
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            self.state.update(status="FAILED", error=message)
            self.result.update(status="FAILED", error=message)
            if self.active is not None:
                self.result["seeds"][str(self.active.seed)]["status"] = "FAILED"
            return 1
        finally:
            if self.active is not None:
                self.active.stop_child()
            self.persist()
            completed = []
            for seed in self.seeds:
                path = self.output / f"seed{seed}/completion.json"
                if path.exists():
                    completed.append(read(path))
            save(
                self.output / f"{self.prefix}_completion.json",
                dict(
                    status=self.state["status"],
                    seeds=list(self.seeds),
                    completed_seeds=[c["seed"] for c in completed if c["status"] == "COMPLETE"],
                    completed_updates=sum(c["completed_updates"] for c in completed),
                    planned_updates=0 if self.args.collect_only else 32,
                    training_state_is_host_reboot_volatile=False,
                    costs_scope="each seed baseline once plus post-update collections; "
                    "collect-only baseline reused, never counted as a second generation",
                ),
            )
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.state["holder_restore"] = "REFILLING_76_GIB"
                except (OSError, RuntimeError, ValueError) as error:
                    self.state.update(
                        holder_restore="DEFERRED_TO_EMPTY_CGROUP_WATCHDOG",
                        restore_reason=str(error),
                    )
                self.persist()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        default=42,
        help="seed used only by --collect-only; full suite always runs 42 then 43",
    )
    parser.add_argument("--resume-completed-stages", action="store_true")
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=3600)
    raise SystemExit(Suite(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
