#!/usr/bin/env python3
"""Run two fixed S0 updates using only complete parent request trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import signal
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from run_same_prefix_online_demo import MODEL, ROOT, Controller, collection, gpu, read, save

DEFAULT = ROOT / "runs/m2-s0-multistep-demo-20261024-v1"
INITIAL = ROOT / "runs/m2-same-prefix-online-8window-demo-20261024-v1/common_initial"
TRAINER = "src.agents.multistep_s0_train"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157)]


class Demo(Controller):
    def __init__(self, args):
        base = SimpleNamespace(
            output=args.output,
            model=MODEL,
            collect_only=False,
            seed=42,
            episode_ids=EPISODES,
            alpha=0.5,
            eta=0.25,
            stage_timeout_seconds=args.stage_timeout_seconds,
        )
        super().__init__(base)
        self.holder = args.holder_directory
        self.initial = args.initial_collection.resolve()
        baseline = collection(self.initial, MODEL, EPISODES, 42)
        if baseline["round_index"] != 0 or baseline["parents_only"] is not False:
            raise RuntimeError("reuse requires the original completed SFT initial collection")
        with (self.initial / "trajectory_results.jsonl").open() as stream:
            parents = [r for line in stream if (r := json.loads(line))["role"] == "parent"]
        if (
            len(parents) != 8
            or {r["episode_id"] for r in parents} != set(EPISODES)
            or any(r["status"] != "COMPLETE" for r in parents)
        ):
            raise RuntimeError("initial training requires exactly eight complete original parents")
        timesteps = [step for parent in parents for step in parent["timesteps"]]
        if any(len(r["timesteps"]) != r["total_requests"] for r in parents):
            raise RuntimeError("parent request rows do not cover the recorded original requests")
        self.state_path = self.output / "multistep_state.json"
        self.state.update(phase="multistep_s0", rounds=2, arms=["zero"])
        self.result = {
            "profile": "s0_two_round_full_parent_request_demo_v1",
            "status": "RUNNING",
            "planned_rounds": 2,
            "seed": 42,
            "episode_ids": EPISODES,
            "initial_collection_path": str(self.initial),
            "baseline": baseline,
            "baseline_reused": True,
            "baseline_reused_source_model_requests": baseline["model_requests"],
            "baseline_reused_parent_requests": len(timesteps),
            "baseline_reused_parent_invalid_requests": sum(s["invalid"] for s in timesteps),
            "baseline_new_model_requests": 0,
            "rounds": [
                {
                    "round": n,
                    "status": "NOT_RUN",
                    "training": None,
                    "evaluation": None,
                    "actor_checkpoint": None,
                    "critic_checkpoint": None,
                    "collection_path": None,
                    "training_wall_seconds": None,
                    "collection_wall_seconds": None,
                    "error": None,
                }
                for n in (1, 2)
            ],
        }

    def persist(self):
        save(self.output / "demo_results.json", self.result)
        rows = []
        baseline = {"round": 0, "status": "REUSED_BASELINE", "evaluation": self.result["baseline"]}
        for entry in (baseline, *self.result["rounds"]):
            evaluation, training = entry.get("evaluation"), entry.get("training")
            values = evaluation["parent_terminal_Y"] if evaluation else []
            complete = len(values) == 8 and all(v is not None for v in values)
            metrics = evaluation.get("parent_terminal_metrics") if evaluation else None
            rows.append(
                {
                    "round": entry["round"],
                    "status": entry["status"],
                    "mean_Y": math.fsum(values) / 8 if complete else None,
                    "mean_raw_net_return": math.fsum(m["raw_net_return"] for m in metrics) / 8
                    if complete and metrics and all(m is not None for m in metrics)
                    else None,
                    "parents_complete": evaluation["parents_complete"] if evaluation else None,
                    "parents_failed": evaluation["parents_failed"] if evaluation else None,
                    "training_rows": training.get("training_rows") if training else None,
                    "invalid_rows": training.get("invalid_rows") if training else None,
                    "actor_updates": training.get("actor_updates") if training else None,
                    "critic_updates": training.get("critic_updates") if training else None,
                    "new_model_requests": evaluation["model_requests"]
                    if entry["round"] and evaluation
                    else 0
                    if entry["round"] == 0
                    else None,
                }
            )
        with (self.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def train_round(self, number, data, actor, critic):
        destination = self.output / "zero" / f"round{number}" / "train"
        arguments = [
            "--data",
            str(data),
            "--output",
            str(destination),
            "--model",
            str(actor),
            "--arm",
            "S0",
            "--seed",
            "42",
            "--alpha",
            "0.5",
            "--eta",
            "0.25",
            "--actor-learning-rate",
            "2e-6",
            "--critic-learning-rate",
            "1e-6",
        ]
        if critic is not None:
            arguments += ["--critic", str(critic), "--retire-input-checkpoints"]
        seconds = self.command(f"round{number}_train", TRAINER, arguments)
        training = read(destination / "training_summary.json")
        if (
            training["status"] != "COMPLETE"
            or training["actor_updates"] != 1
            or training["critic_updates"] != 1
        ):
            raise RuntimeError("each round requires exactly one completed actor and critic update")
        new_actor, new_critic = destination / "actor", destination / "critic"
        if not new_actor.is_dir() or not new_critic.is_dir():
            raise RuntimeError("new actor/critic checkpoint missing")
        if number == 2 and (
            actor != self.output / "zero/round1/train/actor"
            or critic != self.output / "zero/round1/train/critic"
            or training.get("retired_input_checkpoints") != [str(critic), str(actor)]
            or actor.exists()
            or critic.exists()
        ):
            raise RuntimeError("previous-round actor/critic were not loaded and retired correctly")
        return new_actor, new_critic, training, seconds

    def execute(self):
        if self.state_path.exists() or (self.output / "completion.json").exists():
            raise FileExistsError("fresh run required; no implicit retry or resume")
        holder = self.holder or Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder,
            lease_token=f"multistep-s0-{uuid.uuid4().hex}",
            evidence_directory=self.output / "gpu_coordinator",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(self.output / "gpu_config.json", self.config)
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
            gpu.request(
                str(holder),
                {
                    "op": "release",
                    "lease_token": self.config["lease_token"],
                    "bytes": state["maximum_bytes"],
                    "request_id": f"release-{uuid.uuid4().hex}",
                },
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
            actor, critic, data = MODEL, None, self.initial
            for entry in self.result["rounds"]:
                number = entry["round"]
                entry["status"] = "TRAINING"
                self.persist()
                actor, critic, training, seconds = self.train_round(number, data, actor, critic)
                entry.update(
                    training=training,
                    actor_checkpoint=str(actor),
                    critic_checkpoint=str(critic),
                    training_wall_seconds=seconds,
                    status="COLLECTING",
                )
                self.persist()
                data = self.output / "zero" / f"round{number}" / "eval"
                start = time.monotonic()
                evaluation = self.collect(f"round{number}_parents", actor, data, number, True)
                if evaluation["parents_only"] is not True:
                    raise RuntimeError(
                        "post-update data must contain only actual parent trajectories"
                    )
                entry.update(
                    evaluation=evaluation,
                    collection_path=str(data),
                    collection_wall_seconds=time.monotonic() - start,
                    status="COMPLETE",
                )
                self.persist()
            expected = self.output / "zero/round2/train/critic"
            if (
                critic != expected
                or critic.is_symlink()
                or not critic.resolve().is_relative_to(self.output)
            ):
                raise RuntimeError("refusing to prune any non-demo final critic")
            shutil.rmtree(critic)
            self.result.update(
                status="COMPLETE",
                final_actor=str(actor),
                retired_final_critic=str(critic),
                best_round_selection=False,
            )
            self.update(status="COMPLETE", current_stage=None)
            return 0
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if entry is not None:
                entry.update(status="FAILED", error=message)
                partial = (
                    self.output / "zero" / f"round{entry['round']}" / "train/training_summary.json"
                )
                if entry["training"] is None and partial.exists():
                    entry["training"] = read(partial)
            self.result.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            return 1
        finally:
            self.stop_child()
            self.persist()
            evaluations = [
                e["evaluation"] for e in self.result["rounds"] if e["evaluation"] is not None
            ]
            completion = {
                "status": self.result["status"],
                "planned_rounds": 2,
                "completed_rounds": sum(e["status"] == "COMPLETE" for e in self.result["rounds"]),
                "baseline_reused_source_model_requests": self.result[
                    "baseline_reused_source_model_requests"
                ],
                "baseline_reused_parent_requests": self.result["baseline_reused_parent_requests"],
                "baseline_new_model_requests": 0,
                "known_completed_new_collection_totals": {
                    k: sum(e[k] for e in evaluations)
                    for k in ("model_requests", "prompt_tokens", "response_tokens")
                },
                "all_planned_new_collections_complete": len(evaluations) == 2,
                "stage_wall_seconds": sum(s.get("seconds", 0) for s in self.state["stages"]),
                "final_actor": self.result.get("final_actor"),
                "limits": [
                    "S0 only; two fixed rounds, eight Train windows and seed42; "
                    "no best-round selection.",
                    "All actual parent requests, including invalid requests, are trained; "
                    "no heads/EST/REF.",
                    "Actor and critic continue from round1 to round2; Adam resets each round.",
                    "Initial parent requests are reused; "
                    "no new initial sampling cost or Valid/OOS.",
                ],
            }
            save(self.output / "completion.json", completion)
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.update(holder_restore="REFILLING_76_GIB")
                except (OSError, RuntimeError, ValueError) as error:
                    self.update(
                        holder_restore="DEFERRED_TO_EMPTY_CGROUP_WATCHDOG",
                        restore_reason=str(error),
                    )
            for s, handler in handlers.items():
                signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--initial-collection", type=Path, default=INITIAL)
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=1800)
    raise SystemExit(Demo(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
