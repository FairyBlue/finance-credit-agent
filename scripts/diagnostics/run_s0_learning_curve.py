#!/usr/bin/env python3
"""Eight fixed S0 updates on eight reused Train windows, without best-round selection."""

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
    MODEL,
    ROOT,
    TRAINER,
    Controller,
    collection,
    gpu,
    read,
    save,
)

DEFAULT = ROOT / "runs/m2-s0-learning-curve-20261024-v1"
INITIAL = ROOT / "runs/m2-same-prefix-online-8window-demo-20261024-v1/common_initial"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157)]


class Curve(Controller):
    def __init__(self, output: Path, initial: Path, holder: Path | None, timeout: int):
        args = SimpleNamespace(
            output=output,
            model=MODEL,
            collect_only=False,
            seed=42,
            episode_ids=EPISODES,
            alpha=0.5,
            eta=0.25,
            actor_lr=2e-6,
            critic_lr=1e-6,
            holder_directory=holder,
            stage_timeout_seconds=timeout,
            retain_final_actors_only=True,
        )
        super().__init__(args)
        self.phase = "learning_curve"
        self.state_path = self.output / "learning_curve_state.json"
        self.state.update(phase=self.phase, rounds=8, arms=["zero"])
        self.initial_path = initial.resolve()
        baseline = collection(self.initial_path, MODEL, EPISODES, 42)
        if baseline["round_index"] != 0 or baseline["parents_only"] is not False:
            raise RuntimeError("reuse requires the completed full initial SFT collection")
        self.curve = {
            "profile": "s0_fixed_eight_round_train_curve_v1",
            "status": "RUNNING",
            "planned_rounds": 8,
            "seed": 42,
            "episode_ids": EPISODES,
            "alpha": 0.5,
            "eta": 0.25,
            "initial_collection_path": str(self.initial_path),
            "baseline": baseline,
            "baseline_reused": True,
            "baseline_reused_model_requests": baseline["model_requests"],
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
                for n in range(1, 9)
            ],
        }

    def persist_curve(self):
        save(self.output / "curve_results.json", self.curve)
        rows = []
        for entry in [
            {
                "round": 0,
                "status": "REUSED_BASELINE",
                "evaluation": self.curve["baseline"],
                "training": None,
            },
            *self.curve["rounds"],
        ]:
            evaluation, training = entry["evaluation"], entry["training"]
            complete = evaluation is not None and evaluation["parents_failed"] == 0
            values = evaluation["parent_terminal_Y"] if evaluation else []
            complete = complete and len(values) == 8 and all(v is not None for v in values)
            metrics = evaluation.get("parent_terminal_metrics") if evaluation else None
            groups = training.get("coverage", {}).get("groups") if training else None
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
                    "supported_heads": training.get("supported_heads") if training else None,
                    "actor_updates": training.get("actor_updates") if training else None,
                    "critic_updates": training.get("critic_updates") if training else None,
                    "order_diverse_groups": sum(g.get("order_gate") is True for g in groups)
                    if groups is not None
                    else None,
                    "new_model_requests": evaluation["model_requests"]
                    if entry["round"] and evaluation
                    else 0
                    if entry["round"] == 0
                    else None,
                    "new_prompt_tokens": evaluation["prompt_tokens"]
                    if entry["round"] and evaluation
                    else 0
                    if entry["round"] == 0
                    else None,
                    "new_response_tokens": evaluation["response_tokens"]
                    if entry["round"] and evaluation
                    else 0
                    if entry["round"] == 0
                    else None,
                }
            )
        with (self.output / "curve_results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def train_round(self, number: int, data: Path, actor: Path, critic: Path | None) -> tuple:
        destination = self.output / "zero" / f"round{number}" / "train"
        args = [
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
            args += ["--critic", str(critic), "--retire-input-checkpoints"]
        seconds = self.command(f"round{number}_train", TRAINER, args)
        summary = read(destination / "training_summary.json")
        if (
            summary["status"] != "COMPLETE"
            or summary["actor_updates"] != 1
            or summary["critic_updates"] != 1
        ):
            raise RuntimeError("each round requires one completed critic and actor update")
        new_actor, new_critic = destination / "actor", destination / "critic"
        if not new_actor.is_dir() or not new_critic.is_dir():
            raise RuntimeError("new actor/critic HF checkpoint missing")
        if number > 1:
            expected = self.output / "zero" / f"round{number - 1}" / "train"
            if (
                actor != expected / "actor"
                or critic != expected / "critic"
                or summary.get("retired_input_checkpoints") != [str(critic), str(actor)]
                or critic.exists()
                or actor.exists()
            ):
                raise RuntimeError("previous-round inputs were not correctly loaded and retired")
        return new_actor, new_critic, summary, seconds

    def execute(self):
        if self.state_path.exists() or (self.output / "completion.json").exists():
            raise FileExistsError("fresh curve required; no implicit retry/resume")
        holder = self.args.holder_directory
        if holder is None:
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder,
            lease_token=f"s0-curve-{uuid.uuid4().hex}",
            evidence_directory=self.output / "learning_curve_gpu_coordinator",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(self.output / "learning_curve_gpu_config.json", self.config)
        self.update()
        self.persist_curve()

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
                for k in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
            ):
                raise RuntimeError("holder must be paused and empty")
            actor, critic, data = MODEL, None, self.initial_path
            for entry in self.curve["rounds"]:
                number = entry["round"]
                entry["status"] = "TRAINING"
                self.persist_curve()
                actor, critic, training, seconds = self.train_round(number, data, actor, critic)
                entry.update(
                    training=training,
                    actor_checkpoint=str(actor),
                    critic_checkpoint=str(critic),
                    training_wall_seconds=seconds,
                    status="COLLECTING",
                )
                self.persist_curve()
                data = (
                    self.output / "zero" / f"round{number}" / ("eval" if number == 8 else "collect")
                )
                start = time.monotonic()
                evaluation = self.collect(
                    f"round{number}_collect", actor, data, number, number == 8
                )
                entry.update(
                    evaluation=evaluation,
                    collection_path=str(data),
                    collection_wall_seconds=time.monotonic() - start,
                    status="COMPLETE",
                )
                self.persist_curve()
            expected = self.output / "zero/round8/train/critic"
            if (
                critic != expected
                or critic.is_symlink()
                or not critic.resolve().is_relative_to(self.output)
            ):
                raise RuntimeError("refusing to prune any non-curve final critic")
            shutil.rmtree(critic)
            self.curve.update(
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
            self.curve.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            return 1
        finally:
            self.stop_child()
            self.persist_curve()
            evaluations = [
                e["evaluation"] for e in self.curve["rounds"] if e["evaluation"] is not None
            ]
            known = {
                k: sum(e[k] for e in evaluations)
                for k in ("model_requests", "prompt_tokens", "response_tokens")
            }
            summary = {
                "status": self.curve["status"],
                "planned_rounds": 8,
                "completed_rounds": sum(e["status"] == "COMPLETE" for e in self.curve["rounds"]),
                "baseline_reused_model_requests": self.curve["baseline_reused_model_requests"],
                "baseline_new_model_requests": 0,
                "known_completed_new_collection_totals": known,
                "all_planned_new_collections_complete": len(evaluations) == 8,
                "stage_wall_seconds": sum(s.get("seconds", 0) for s in self.state["stages"]),
                "final_actor": self.curve.get("final_actor"),
                "limits": [
                    "S0 only, eight fixed Train windows, seed42; no best-round selection.",
                    "Each round continues previous actor and learned critic; "
                    "Adam is reset each round.",
                    "Head-only gamma=lambda=1 demo, no REF and no formal Valid/OOS.",
                    "Historical initial collection is reused; "
                    "only post-update collections are new cost.",
                ],
            }
            save(self.output / "completion.json", summary)
            save(self.output / "demo_summary.json", summary)
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
    args = parser.parse_args()
    raise SystemExit(
        Curve(
            args.output, args.initial_collection, args.holder_directory, args.stage_timeout_seconds
        ).execute()
    )


if __name__ == "__main__":
    main()
