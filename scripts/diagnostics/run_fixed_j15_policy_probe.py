#!/usr/bin/env python3
"""Compare two fixed policies from the same eight stored j15 prefixes."""

from __future__ import annotations

import argparse
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

DEFAULT = ROOT / "runs/m2-fixed-j15-policy-probe-20261024-v1"
SOURCE = ROOT / "runs/m2-same-prefix-online-8window-demo-20261024-v1/common_initial"
TRAINED = ROOT / "runs/m2-s0-learning-curve-20261024-v1/zero/round8/train/actor"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157)]


class Probe(Controller):
    def __init__(self, args):
        source = args.source.resolve()
        initial = collection(source, args.sft_model.resolve(), EPISODES, 42)
        if initial["round_index"] != 0 or initial["parents_only"] is not False:
            raise RuntimeError("source must be the completed original SFT initial collection")
        base = SimpleNamespace(
            output=args.output,
            model=args.sft_model.resolve(),
            seed=42,
            episode_ids=EPISODES,
            alpha=0.5,
            eta=0.25,
            collect_only=False,
            stage_timeout_seconds=args.stage_timeout_seconds,
        )
        super().__init__(base)
        self.holder = args.holder_directory
        self.source = source
        self.phase = "fixed_j15_probe"
        self.state_path = self.output / "probe_state.json"
        self.state = {
            "status": "RUNNING",
            "phase": self.phase,
            "stages": [],
            "source": str(source),
            "seed": 42,
            "episode_ids": EPISODES,
        }
        self.result = {
            "profile": "fixed_j15_two_policy_probe_v1",
            "status": "RUNNING",
            "source": str(source),
            "seed": 42,
            "episode_ids": EPISODES,
            "continuations_per_anchor": 4,
            "planned_per_model": 32,
            "models": [
                {
                    "name": name,
                    "model_path": str(model.resolve()),
                    "status": "NOT_RUN",
                    "summary": None,
                    "wall_seconds": None,
                    "error": None,
                }
                for name, model in (("sft", args.sft_model), ("trained", args.trained_model))
            ],
            "limits": [
                "Fixed original eight Train j15 prefixes; four continuations per policy and state.",
                "Common random seed namespace across policies; "
                "no new parents, heads, EST or training.",
                "Failed continuations retain their original status and missing outcomes; "
                "no retries.",
            ],
        }

    def persist(self):
        summaries = [e["summary"] for e in self.result["models"] if e["summary"] is not None]
        self.result["known_reported_costs"] = {
            k: sum(s.get(k, 0) for s in summaries)
            for k in ("model_requests", "prompt_tokens", "response_tokens", "invalid_requests")
        }
        self.result["all_stage_summaries_present"] = len(summaries) == 2
        save(self.output / "probe_result.json", self.result)

    def collect_policy(self, entry):
        destination = self.output / entry["name"]
        entry["collection_path"] = str(destination)
        entry["status"] = "RUNNING"
        self.persist()
        arguments = [
            "--model",
            entry["model_path"],
            "--output",
            str(destination),
            "--episode-ids",
            *EPISODES,
            "--seed",
            "42",
            "--round",
            "0",
            "--fixed-prefix-source",
            str(self.source),
            "--continuations-per-anchor",
            "4",
        ]
        seconds = self.command(f"{entry['name']}_fixed_j15", COLLECTOR, arguments)
        summary = read(destination / "fixed_prefix_summary.json")
        expected_source = self.source / "anchor_snapshots.jsonl"
        if (
            summary["status"] != "COMPLETE"
            or Path(summary["model_path"]).resolve() != Path(entry["model_path"])
            or Path(summary["fixed_prefix_source"]).resolve() != expected_source.resolve()
            or summary["episode_ids"] != EPISODES
            or summary["seed"] != 42
            or summary["round_index"] != 0
            or summary["continuations_per_anchor"] != 4
            or summary["planned_trajectories"] != 32
            or summary["complete_trajectories"] + summary["failed_trajectories"] != 32
            or summary["optimizer_updates"] != 0
            or summary["REF_used"] is not False
        ):
            raise RuntimeError("fixed-prefix collection identity or planned denominator mismatch")
        entry.update(summary=summary, wall_seconds=seconds, status="COMPLETE")
        self.persist()

    def execute(self):
        if self.state_path.exists() or (self.output / "completion.json").exists():
            raise FileExistsError("fresh probe required; no implicit retry or resume")
        holder = self.holder
        if holder is None:
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder,
            lease_token=f"fixed-j15-{uuid.uuid4().hex}",
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
                for k in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
            ):
                raise RuntimeError("holder must be paused and empty")
            for entry in self.result["models"]:
                self.collect_policy(entry)
            self.result["status"] = "COMPLETE"
            self.update(status="COMPLETE", current_stage=None)
            return 0
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if entry is not None:
                entry.update(status="FAILED", error=message)
                partial = self.output / entry["name"] / "fixed_prefix_summary.json"
                if partial.exists():
                    entry["summary"] = read(partial)
            self.result.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            return 1
        finally:
            self.stop_child()
            self.persist()
            save(self.output / "completion.json", self.result)
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
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--sft-model", type=Path, default=MODEL)
    parser.add_argument("--trained-model", type=Path, default=TRAINED)
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=1800)
    raise SystemExit(Probe(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
