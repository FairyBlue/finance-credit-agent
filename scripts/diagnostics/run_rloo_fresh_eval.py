#!/usr/bin/env python3
"""Evaluate four frozen policies in sampling namespace4 inside a systemd service."""

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
    EPISODES,
    MODEL,
    ROOT,
    Controller,
    collection,
    gpu,
    read,
    save,
)

DEFAULT = ROOT / "runs/m2-rloo-local-credit-fresh-eval-20261026-v1"
RLOO = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
LOCAL = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
MODELS = {
    "SFT": MODEL,
    "RLOO": RLOO / "current_actor",
    "plain": LOCAL / "plain/current_actor",
    "exact": LOCAL / "exact/current_actor",
}
ROUND_INDEX = 4
COMPLETE = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES"}
COST_KEYS = ("model_requests", "prompt_tokens", "response_tokens")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def observation(summary):
    """Keep a fixed sixteen-parent denominator; failed or absent slots remain NA."""
    values = summary.get("parent_terminal_Y", [])
    metrics = summary.get("parent_terminal_metrics", [])
    complete = (
        len(values) == len(metrics) == 16
        and summary.get("parents_complete") == 16
        and summary.get("parents_failed") == 0
        and all(finite(value) for value in values)
        and all(
            isinstance(metric, dict)
            and all(
                finite(metric.get(key))
                for key in ("raw_net_return", "transaction_cost", "invalid_count")
            )
            for metric in metrics
        )
    )
    invalid = sum(metric["invalid_count"] for metric in metrics) if complete else None
    requests = summary.get("model_requests", 0)
    return dict(
        planned_parent_trajectories=16,
        mean_Y=math.fsum(values) / 16 if complete else None,
        mean_raw_net_return=math.fsum(m["raw_net_return"] for m in metrics) / 16
        if complete
        else None,
        mean_transaction_cost=math.fsum(m["transaction_cost"] for m in metrics) / 16
        if complete
        else None,
        invalid_requests=invalid,
        rejection_rate=invalid / requests if complete and requests else None,
        parents_complete=summary.get("parents_complete", 0),
        parents_failed=summary.get("parents_failed"),
        model_requests=requests,
        complete_fixed_denominator=complete,
    )


class Evaluation(Controller):
    def __init__(self, args):
        self.cli, self.attempt = args, 1
        super().__init__(
            SimpleNamespace(
                output=args.output,
                model=MODEL,
                collect_only=True,
                seed=42,
                episode_ids=EPISODES,
                alpha=0.5,
                eta=0.25,
                stage_timeout_seconds=args.stage_timeout_seconds,
                coordinator_request_timeout_seconds=60,
            )
        )
        self.state_path = self.output / "state.json"
        self.state.update(
            phase="fresh_sampling_evaluation",
            round_index=ROUND_INDEX,
            policy_order=list(MODELS),
            optimizer_updates=0,
            planned_updates=0,
            rounds=0,
            arms=list(MODELS),
        )
        self.result = dict(
            profile="fixed_four_policy_fresh_sampling_evaluation_v1",
            status="RUNNING",
            seed=42,
            episode_ids=EPISODES,
            round_index=ROUND_INDEX,
            parent_repetitions=4,
            planned_parent_trajectories_per_policy=16,
            planned_parent_trajectories=64,
            policy_order=list(MODELS),
            common_random_numbers=True,
            optimizer_updates=0,
            EST_used=False,
            REF_used=False,
            reward_or_request_cap_changed=False,
            formal_valid_oos_access=False,
            training_namespaces=[0, 1, 2, 3],
            fresh_sampling_namespace=ROUND_INDEX,
            policies={
                name: dict(
                    status="NOT_RUN",
                    model_path=str(model),
                    collection_path=str(self.output / name),
                    evaluation=None,
                    observation=observation({}),
                )
                for name, model in MODELS.items()
            },
        )

    def persist(self):
        self.update()
        save(self.output / "results.json", self.result)
        rows = [
            dict(policy=name, status=entry["status"], **entry["observation"])
            for name, entry in self.result["policies"].items()
        ]
        with (self.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def validate_inputs(self):
        receipts = {}
        for name, directory in (
            ("RLOO", RLOO),
            ("plain", LOCAL / "plain"),
            ("exact", LOCAL / "exact"),
        ):
            result = read(directory / "rloo_results.json")
            rounds = result.get("rounds", [])
            if (
                result.get("status") != "COMPLETE"
                or result.get("completed_updates") != 16
                or result.get("seed") != 42
                or result.get("episode_ids") != EPISODES
                or len(rounds) != 4
                or rounds[-1].get("status") != "COMPLETE"
                or rounds[-1].get("training", {}).get("completed_updates") != 16
                or observation(rounds[-1].get("evaluation") or {})["mean_Y"] is None
            ):
                raise RuntimeError(f"{name} needs its completed16-update original result")
            receipts[name] = dict(
                result_path=str(directory / "rloo_results.json"),
                completed_updates=16,
                original_status=result["status"],
            )
        for name, model in MODELS.items():
            if not (model / "config.json").is_file() or (model / "EXPORT_INCOMPLETE.json").exists():
                raise RuntimeError(f"{name} has no complete frozen HF actor")
            index = model / "model.safetensors.index.json"
            if index.exists():
                files = set(read(index)["weight_map"].values())
                if not files or not all((model / filename).is_file() for filename in files):
                    raise RuntimeError(f"{name} actor shard missing")
            elif not (model / "model.safetensors").is_file():
                raise RuntimeError(f"{name} actor weights missing")
        self.result["input_receipts"] = receipts

    def prepare(self):
        if self.state_path.exists():
            if not self.cli.resume_completed_stages:
                raise FileExistsError("existing evaluation requires explicit resume")
            state, result = read(self.state_path), read(self.output / "results.json")
            if (
                state["status"] not in {"FAILED", "PAUSED"}
                or result["seed"] != 42
                or result["episode_ids"] != EPISODES
                or result["round_index"] != ROUND_INDEX
                or {k: v["model_path"] for k, v in result["policies"].items()}
                != {k: str(v) for k, v in MODELS.items()}
            ):
                raise RuntimeError("only this failed/paused fixed evaluation may resume")
            self.attempt = state.get("controller_attempt", 1) + 1
            archive = self.output / "controller_attempts"
            archive.mkdir(exist_ok=True)
            for filename in ("state.json", "results.json", "results.csv", "completion.json"):
                source, target = (
                    self.output / filename,
                    archive / f"attempt{self.attempt - 1}_{filename}",
                )
                if source.exists():
                    if target.exists():
                        raise FileExistsError("previous attempt snapshot already exists")
                    shutil.copy2(source, target)
            self.state, self.result = state, result
            self.state.update(status="RUNNING", current_stage=None, controller_attempt=self.attempt)
            self.result.update(status="RUNNING", controller_attempt=self.attempt)
            self.state.pop("error", None)
            self.result.pop("error", None)

    def checked_collection(self, path, model):
        summary = collection(path, model, EPISODES, 42)
        if (
            summary.get("parents_only") is not True
            or summary.get("local_credit", False)
            or summary.get("parent_repetitions") != 4
            or summary.get("planned_parent_trajectories") != 16
            or summary.get("round_index") != ROUND_INDEX
            or summary.get("estimate_paths") != 0
            or summary.get("optimizer_updates") != 0
        ):
            raise RuntimeError("collection is not the fixed fresh namespace4 parents-only batch")
        return summary

    def collection_path(self, entry):
        path = Path(entry["collection_path"])
        if (
            not path.exists()
            or not any(path.iterdir())
            or (path / "collection_summary.json").exists()
        ):
            return path
        if not self.cli.resume_completed_stages:
            raise FileExistsError("incomplete evaluation retained; explicit resume required")
        for filename in ("requests.jsonl", "trajectory_results.jsonl", "head_results.jsonl"):
            artifact = path / filename
            if artifact.exists() and artifact.stat().st_size:
                raise RuntimeError(
                    "generated partial evaluation remains NA and cannot be resampled"
                )
        path = path.with_name(f"{path.name}-attempt{self.attempt}")
        if path.exists():
            raise FileExistsError("pre-generation retry requires a fresh attempt directory")
        return path

    def execute(self):
        self.prepare()
        self.persist()

        def forward(signum, _frame):
            if self.child is not None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        current = None
        try:
            self.validate_inputs()
            holder = self.cli.holder_directory or Path(
                read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"]
            )
            self.config = gpu.config_for_holder(
                holder,
                lease_token=f"rloo-fresh-{uuid.uuid4().hex}",
                evidence_directory=self.output / f"gpu_attempt{self.attempt}",
                portfolio_target_bytes=76 * 1024**3,
            )
            gpu.freeze_config(self.output / f"gpu_config_attempt{self.attempt}.json", self.config)
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
                for k in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
            ):
                raise RuntimeError("holder must be paused and empty")
            for name, model in MODELS.items():
                current = self.result["policies"][name]
                if current["status"] in COMPLETE:
                    self.checked_collection(Path(current["collection_path"]), model)
                    continue
                path = self.collection_path(current)
                current.update(status="RUNNING", collection_path=str(path))
                current.pop("error", None)
                self.update(current_policy=name)
                self.persist()
                if not (path / "collection_summary.json").exists():
                    suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
                    self.command(
                        f"eval_{name}{suffix}",
                        COLLECTOR,
                        [
                            "--model",
                            str(model),
                            "--output",
                            str(path),
                            "--episode-ids",
                            *EPISODES,
                            "--seed",
                            "42",
                            "--round-index",
                            str(ROUND_INDEX),
                            "--parents-only",
                            "--parent-repetitions",
                            "4",
                        ],
                    )
                summary = self.checked_collection(path, model)
                observed = observation(summary)
                current.update(
                    status="COMPLETE"
                    if observed["complete_fixed_denominator"]
                    else "COMPLETE_WITH_MODEL_FAILURES",
                    evaluation=summary,
                    observation=observed,
                    collection_cost={k: summary.get(k, 0) for k in COST_KEYS},
                )
                self.persist()
            status = (
                "COMPLETE"
                if all(e["status"] == "COMPLETE" for e in self.result["policies"].values())
                else "COMPLETE_WITH_MODEL_FAILURES"
            )
            self.result["status"] = status
            self.update(status=status, current_stage=None, current_policy=None)
            return 0
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if current is not None:
                current.update(status="FAILED", error=message)
            self.result.update(status="FAILED", error=message)
            for stage in self.state["stages"]:
                if stage["status"] == "RUNNING":
                    stage.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            return 1
        finally:
            self.stop_child()
            self.persist()
            save(
                self.output / "completion.json",
                dict(
                    status=self.result["status"],
                    seed=42,
                    round_index=ROUND_INDEX,
                    planned_parent_trajectories=64,
                    policy_order=list(MODELS),
                    completed_policy_collections=[
                        name
                        for name, e in self.result["policies"].items()
                        if e["status"] in COMPLETE
                    ],
                    policies_with_model_failures=[
                        name
                        for name, e in self.result["policies"].items()
                        if e["status"] == "COMPLETE_WITH_MODEL_FAILURES"
                    ],
                    observed_complete_parent_trajectories=sum(
                        e["observation"]["parents_complete"]
                        for e in self.result["policies"].values()
                    ),
                    generation_cost={
                        k: sum(
                            e.get("collection_cost", {}).get(k, 0)
                            for e in self.result["policies"].values()
                        )
                        for k in COST_KEYS
                    },
                    generation_cost_scope=(
                        "completed collection summaries; partial failed files remain available"
                    ),
                    policies_without_complete_cost_summary=[
                        name
                        for name, entry in self.result["policies"].items()
                        if entry["status"] not in COMPLETE
                    ],
                    stage_wall_seconds=sum(s.get("seconds", 0) for s in self.state["stages"]),
                    optimizer_updates=0,
                    new_checkpoints=0,
                    common_random_numbers=True,
                    model_failure_resampling=False,
                    training_or_reward_changes=False,
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
    parser.add_argument("--resume-completed-stages", action="store_true")
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=3600)
    raise SystemExit(Evaluation(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
