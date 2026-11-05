#!/usr/bin/env python3
"""Run four grouped on-policy batches and 16 persistent-state RLOO updates."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import signal
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from run_same_prefix_online_demo import (
    COLLECTOR,
    EPISODES,
    MODEL,
    ROOT,
    Controller,
    collection,
    gpu,
    read,
    save,
)

DEFAULT = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
INITIAL = ROOT / "runs/m2-rloo-terminal-demo-20261025-v1/common_initial"
SCRATCH = DEFAULT / "training_state"
TRAINER = "src.agents.rloo_terminal_train"


def observation(summary):
    ys, metrics = summary["parent_terminal_Y"], summary["parent_terminal_metrics"]
    complete = (
        len(ys) == len(metrics) == 16
        and summary["parents_complete"] == 16
        and all(y is not None and math.isfinite(y) for y in ys)
        and all(m is not None for m in metrics)
    )
    return {
        "mean_Y": math.fsum(ys) / 16 if complete else None,
        "mean_raw_net_return": math.fsum(m["raw_net_return"] for m in metrics) / 16
        if complete else None,
        "invalid_requests": sum(m["invalid_count"] for m in metrics)
        if complete else None,
        "mean_transaction_cost": math.fsum(m["transaction_cost"] for m in metrics) / 16
        if complete else None,
        "parents_complete": summary["parents_complete"],
        "parents_failed": summary["parents_failed"],
        "model_requests": summary["model_requests"],
    }


class Demo(Controller):
    def __init__(self, args):
        base = SimpleNamespace(
            output=args.output,
            model=MODEL,
            collect_only=args.collect_only,
            seed=42,
            episode_ids=EPISODES,
            alpha=0.0,
            eta=0.0,
            stage_timeout_seconds=args.stage_timeout_seconds,
            coordinator_request_timeout_seconds=60,
        )
        super().__init__(base)
        self.holder = args.holder_directory
        self.initial_only = args.collect_only
        self.resume_completed = args.resume_completed_stages
        self.attempt = 1
        self.initial_collection = args.initial_collection.resolve()
        self.scratch = args.state_directory.resolve()
        self.volatile_state = self.scratch.is_relative_to(Path('/dev/shm'))
        self.state_path = self.output / (
            "initial_state.json" if self.initial_only else "rloo_state.json"
        )
        self.state.update(phase="rloo_terminal", rounds=4, arms=["rloo"], planned_updates=16)
        self.result = {
            "profile": "persistent_gpu_fp32_terminal_Y_sharded_RLOO_demo_v2",
            "status": "RUNNING",
            "planned_updates": 16,
            "updates_per_round": 4,
            "seed": 42,
            "episode_ids": EPISODES,
            "parent_repetitions": 4,
            "baseline": None,
            "baseline_path": str(self.initial_collection),
            "rounds": [],
            "training_state": str(self.scratch),
            "checkpoint_placement": "GPU FP32 master and Adam; streaming disk shards",
            "restarts_failed_run": str(INITIAL.parent),
            "failed_run_updates_replayed_from_SFT": 4,
            "resumes_paused_run": str(ROOT / "runs/m2-rloo-terminal-demo-20261025-v2"),
            "training_state_storage": (
                "RAM filesystem: host reboot volatile" if self.volatile_state
                else "persistent project disk: FP32 weights and Adam moments"
            ),
            "actor_checkpoint": str(self.output / "current_actor"),
            "critic_updates": 0,
            "best_round_selection": False,
        }

    def persist(self):
        path = "initial_results.json" if self.initial_only else "rloo_results.json"
        save(self.output / path, self.result)
        if self.initial_only:
            return
        rows = []
        if self.result["baseline"] is not None:
            rows.append({"updates": 0, "round": 0, "status": "BASELINE",
                         **observation(self.result["baseline"])})
        for entry in self.result["rounds"]:
            rows.append({
                "updates": entry["round"] * 4,
                "round": entry["round"],
                "status": entry["status"],
                **(observation(entry["evaluation"]) if entry.get("evaluation") else {
                    key: None for key in (
                        "mean_Y", "mean_raw_net_return", "invalid_requests",
                        "mean_transaction_cost", "parents_complete", "parents_failed",
                        "model_requests",
                    )
                }),
            })
        if rows:
            with (self.output / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def grouped_collect(self, name, actor, destination, namespace):
        self.command(name, COLLECTOR, [
            "--model", str(actor), "--output", str(destination),
            "--episode-ids", *EPISODES, "--seed", "42",
            "--round-index", str(namespace), "--parents-only", "--parent-repetitions", "4",
        ])
        result = collection(destination, actor, EPISODES, 42)
        if (
            result.get("parent_repetitions") != 4
            or result.get("planned_parent_trajectories") != 16
            or result["parents_only"] is not True
            or result["round_index"] != namespace
        ):
            raise RuntimeError("collection does not match the fixed four-window RLOO batch")
        return result

    def prepare_resume(self):
        """Resume only a committed training boundary, retaining failed attempts."""
        if self.initial_only:
            raise RuntimeError("completed-stage resume is a training-only operation")
        previous = read(self.state_path)
        result = read(self.output / "rloo_results.json")
        if previous["status"] not in {"FAILED", "PAUSED"}:
            raise RuntimeError("resume requires an explicitly stopped failed/paused controller")
        if (result["seed"] != 42 or result["episode_ids"] != EPISODES
                or Path(result["training_state"]).resolve() != self.scratch
                or Path(result["baseline_path"]).resolve() != self.initial_collection):
            raise RuntimeError("resume identity or storage path changed")
        entries = result["rounds"]
        if not entries or [e["round"] for e in entries] != list(range(1, len(entries) + 1)):
            raise RuntimeError("resume needs contiguous existing rounds")
        for entry in entries:
            training = read(self.output / f"round{entry['round']}/train/training_summary.json")
            if (training["status"] != "COMPLETE" or training["optimizer_updates"] != 4
                    or training["completed_updates"] != entry["round"] * 4):
                raise RuntimeError("partial training cannot be silently replayed or resumed")
            entry["training"] = training
        completed = entries[-1]["round"] * 4
        manifest = read(self.scratch / "manifest.json")
        if (manifest["status"] != "COMPLETE"
                or manifest["completed_optimizer_steps"] != completed):
            raise RuntimeError("last committed FP32/Adam boundary does not match training")
        actor = self.output / "current_actor"
        index = read(actor / "model.safetensors.index.json")
        if ((actor / "EXPORT_INCOMPLETE.json").exists()
                or not all((actor / name).is_file() for name in set(index["weight_map"].values()))):
            raise RuntimeError("committed BF16 rollout actor is incomplete")
        self.attempt = previous.get("controller_attempt", 1) + 1
        archive = self.output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for filename in ("rloo_state.json", "rloo_results.json", "completion.json", "results.csv"):
            source = self.output / filename
            target = archive / f"attempt{self.attempt - 1}_{filename}"
            if source.exists():
                if target.exists():
                    raise FileExistsError("previous controller snapshot already exists")
                shutil.copy2(source, target)
        self.state, self.result = previous, result
        self.state.update(status="RUNNING", controller_attempt=self.attempt, current_stage=None)
        self.state.pop("error", None)
        self.result.update(status="RUNNING", controller_attempt=self.attempt,
                           resumed_committed_updates=completed)
        self.result.pop("error", None)

    def evaluation_path(self, number):
        destination = self.output / f"round{number}" / "eval"
        if (destination / "collection_summary.json").exists():
            return destination
        if destination.exists() and any(destination.iterdir()):
            if not self.resume_completed:
                raise FileExistsError("unexpected existing evaluation")
            # Only infrastructure failures before ANY generated request may retry.
            for name in ("requests.jsonl", "trajectory_results.jsonl", "head_results.jsonl"):
                path = destination / name
                if path.exists() and path.stat().st_size:
                    raise RuntimeError("partially generated evaluation cannot be resampled")
            destination = self.output / f"round{number}" / f"eval-attempt{self.attempt}"
            if destination.exists():
                raise FileExistsError("fresh pre-generation retry directory required")
        return destination

    def execute(self):
        if self.resume_completed:
            self.prepare_resume()
        elif self.state_path.exists():
            raise FileExistsError("fresh controller phase required; no implicit retry")
        holder = self.holder or Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder, lease_token=f"rloo-terminal-{uuid.uuid4().hex}",
            evidence_directory=self.output / (
                "initial_gpu" if self.initial_only else f"training_gpu_attempt{self.attempt}"
            ),
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(
            self.output / ("initial_gpu_config.json" if self.initial_only
                           else f"gpu_config_attempt{self.attempt}.json"),
            self.config,
        )
        self.update()
        self.persist()

        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        entry = None
        try:
            state = gpu.controller_claim(self.config)
            self.claimed = True
            gpu.request(str(holder), {
                "op": "release", "lease_token": self.config["lease_token"],
                "bytes": state["maximum_bytes"], "request_id": f"release-{uuid.uuid4().hex}",
            })
            checked = gpu.reservation_status(self.config)
            if checked["paused"] is not True or any(checked[k] for k in (
                "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes",
            )):
                raise RuntimeError("holder must be paused and empty")
            data = self.initial_collection
            if (data / "collection_summary.json").exists():
                baseline = collection(data, MODEL, EPISODES, 42)
                if baseline.get("parent_repetitions") != 4 or baseline["round_index"] != 0:
                    raise RuntimeError("initial collection is not this grouped SFT batch")
            else:
                baseline = self.grouped_collect("common_initial", MODEL, data, 0)
            self.result["baseline"] = baseline
            self.persist()
            if self.initial_only:
                self.result["status"] = "COLLECTED"
                self.update(status="COMPLETE", current_stage=None)
                return 0
            if self.scratch.exists() and not self.resume_completed:
                raise FileExistsError("new run requires unused scratch state")
            actor = MODEL
            existing = {item["round"]: item for item in self.result["rounds"]}
            for number in range(1, 5):
                entry = existing.get(number)
                if entry is None:
                    entry = {"round": number, "status": "TRAINING", "training": None,
                             "evaluation": None, "training_data_path": str(data)}
                    self.result["rounds"].append(entry)
                    self.persist()
                    destination = self.output / f"round{number}" / "train"
                    arguments = [
                        "--data", str(data), "--output", str(destination), "--model", str(actor),
                        "--state-output", str(self.scratch),
                        "--actor-output", str(self.output / "current_actor"),
                        "--updates", "4", "--seed", "42", "--learning-rate", "2e-6",
                        "--clip-epsilon", "0.2",
                    ]
                    if number > 1:
                        arguments += ["--resume-state", str(self.scratch)]
                    suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
                    self.command(f"round{number}_train{suffix}", TRAINER, arguments)
                    entry["training"] = read(destination / "training_summary.json")
                training = entry["training"]
                if (training["status"] != "COMPLETE"
                        or training["optimizer_updates"] != 4
                        or training["completed_updates"] != number * 4):
                    raise RuntimeError("four genuine persistent optimizer updates required")
                actor = self.output / "current_actor"
                if entry.get("evaluation") is not None:
                    data = Path(entry["collection_path"])
                    continue
                entry.update(status="COLLECTING", training=training)
                entry.pop("error", None)
                self.persist()
                data = self.evaluation_path(number)
                namespace = 0 if number == 4 else number
                if (data / "collection_summary.json").exists():
                    evaluation = collection(data, actor, EPISODES, 42)
                else:
                    suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
                    evaluation = self.grouped_collect(
                        f"round{number}_parents{suffix}", actor, data, namespace
                    )
                entry.update(
                    status="COMPLETE", evaluation=evaluation, collection_path=str(data),
                    evaluation_seed_namespace=namespace,
                    paired_with_initial_sampling_seeds=number == 4,
                )
                self.persist()
            self.result.update(status="COMPLETE", completed_updates=16)
            self.update(status="COMPLETE", current_stage=None)
            return 0
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if entry is not None:
                entry.update(status="FAILED", error=message)
                partial = self.output / f"round{entry['round']}" / "train/training_summary.json"
                if entry["training"] is None and partial.exists():
                    entry["training"] = read(partial)
            if self.state["stages"] and self.state["stages"][-1]["status"] == "RUNNING":
                stage = self.state["stages"][-1]
                stage.update(status="FAILED", error=message,
                             seconds=time.monotonic() - stage["started_monotonic"])
            self.result.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            return 1
        finally:
            self.stop_child()
            self.persist()
            completion_name = "initial_completion.json" if self.initial_only else "completion.json"
            save(self.output / completion_name, {
                "status": self.result["status"],
                "completed_updates": self.result.get("completed_updates", sum(
                    e["training"]["optimizer_updates"] for e in self.result["rounds"]
                    if e.get("training")
                )),
                "completed_rounds": sum(e["status"] == "COMPLETE" for e in self.result["rounds"]),
                "stage_wall_seconds": sum(s.get("seconds", 0) for s in self.state["stages"]),
                "training_state": str(self.scratch),
                "training_state_is_host_reboot_volatile": self.volatile_state,
                "actor_checkpoint": str(self.output / "current_actor"),
                "baseline_reused_path": str(self.initial_collection),
                "baseline_new_generation_requests": 0,
                "new_post_update_collections": True,
                "final_evaluation_shares_initial_sampling_seed_namespace": True,
                "intermediate_evaluations_use_fresh_training_seed_namespaces": [1, 2, 3],
                "planned_updates": 16, "critic_updates": 0,
            })
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.update(holder_restore="REFILLING_76_GIB")
                except (OSError, RuntimeError, ValueError) as error:
                    self.update(holder_restore="DEFERRED_TO_EMPTY_CGROUP_WATCHDOG",
                                restore_reason=str(error))
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume-completed-stages", action="store_true")
    parser.add_argument("--initial-collection", type=Path, default=INITIAL)
    parser.add_argument("--state-directory", type=Path, default=SCRATCH)
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=3600)
    raise SystemExit(Demo(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
