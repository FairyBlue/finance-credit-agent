#!/usr/bin/env python3
"""Run one bounded order-contrast, supervised-update and full-trajectory demo."""

from __future__ import annotations

import argparse
import signal
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

from run_rloo_eight_window import EPISODES, observation
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

DEFAULT = ROOT / "runs/m2-counterfactual-order-demo-20261027-v1"
REFERENCE = ROOT / "runs/m2-rloo-eight-window-20261025-v1/seed43"
NAMESPACES = (5, 6)
COMPLETE = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES"}


def sample_count(path):
    return sum(bool(line.strip()) for line in path.read_text().splitlines())


class Demo(Controller):
    def __init__(self, args):
        self.cli = args
        super().__init__(
            SimpleNamespace(
                output=args.output,
                model=REFERENCE / "current_actor",
                collect_only=False,
                seed=43,
                episode_ids=EPISODES,
                alpha=0.0,
                eta=0.0,
                stage_timeout_seconds=args.stage_timeout_seconds,
                coordinator_request_timeout_seconds=60,
            )
        )
        self.state_path = self.output / "state.json"
        self.state.update(phase="counterfactual_order_demo", planned_updates=16, arms=[])
        self.result = dict(
            profile="active_legal_order_supervised_demo_v1",
            status="RUNNING",
            seed=43,
            episode_ids=EPISODES,
            source_collection=str(REFERENCE / "round4/eval"),
            frozen_RLOO_model=str(REFERENCE / "current_actor"),
            planned_anchors=32,
            maximum_branches=512,
            planned_updates=16,
            maximum_evaluation_parents=192,
            evaluation_namespaces=list(NAMESPACES),
            formal_valid_oos_access=False,
            stage1=None,
            stage2=None,
            evaluations={},
        )

    def persist(self):
        self.update()
        save(self.output / "results.json", self.result)

    def validate_inputs(self):
        source = collection(REFERENCE / "round4/eval", REFERENCE / "current_actor", EPISODES, 43)
        if source.get("parents_complete") != 32 or source.get("round_index") != 0:
            raise RuntimeError("complete original frozen RLOO parent source required")
        for model in (MODEL, REFERENCE / "current_actor"):
            index = read(model / "model.safetensors.index.json")
            if not all((model / name).is_file() for name in set(index["weight_map"].values())):
                raise RuntimeError(f"missing frozen model shards: {model}")

    def prepare(self):
        if not self.state_path.exists():
            return
        if not self.cli.resume_completed_stages:
            raise FileExistsError("existing run needs explicit completed-stage resume")
        previous = read(self.state_path)
        if previous["status"] not in {"FAILED", "PAUSED"}:
            raise RuntimeError("only a stopped controller can resume")
        self.result = read(self.output / "results.json")
        self.state = previous
        attempt = self.state.get("attempt", 1) + 1
        archive = self.output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for name in ("state.json", "results.json", "completion.json"):
            path = self.output / name
            if path.exists():
                (archive / f"attempt{attempt - 1}_{name}").write_bytes(path.read_bytes())
        self.state.update(status="RUNNING", attempt=attempt, current_stage=None)
        self.result.update(status="RUNNING")
        self.state.pop("error", None)
        self.result.pop("error", None)

    def stage(self, name, module, arguments):
        suffix = self.state.get("attempt", 1)
        self.command(f"{name}_attempt{suffix}", module, arguments)

    def run_steps(self):
        first = self.output / "stage1"
        if not (first / "collection_summary.json").is_file():
            self.stage(
                "order_contrasts",
                "src.agents.counterfactual_order_collect",
                [
                    "--model",
                    str(REFERENCE / "current_actor"),
                    "--source",
                    str(REFERENCE / "round4/eval"),
                    "--output",
                    str(first),
                    "--seed",
                    "43",
                ],
            )
        self.result["stage1"] = read(first / "collection_summary.json")
        if self.result["stage1"]["status"] not in COMPLETE:
            raise RuntimeError("order contrast collection did not complete")
        count = sample_count(first / "training_samples.jsonl")
        self.result["label_count"] = count
        self.persist()
        if not count:
            self.result.update(
                status="COMPLETE_NO_USABLE_LABELS",
                stage2={"status": "SKIPPED_NO_USABLE_LABELS", "optimizer_updates": 0},
                decision="STOP_NO_USABLE_COUNTERFACTUAL_LABELS",
            )
            return
        second = self.output / "stage2"
        if not (second / "training_summary.json").is_file():
            self.stage(
                "supervised_training",
                "src.agents.counterfactual_order_train",
                [
                    "--model",
                    str(REFERENCE / "current_actor"),
                    "--collection",
                    str(first),
                    "--output",
                    str(second),
                    "--updates",
                    "16",
                    "--seed",
                    "43",
                ],
            )
        train = read(second / "training_summary.json")
        if train["status"] != "COMPLETE" or train["optimizer_updates"] != 16:
            raise RuntimeError("sixteen completed supervised updates required")
        self.result["stage2"] = train
        self.persist()
        models = {
            "SFT": MODEL,
            "RLOO": REFERENCE / "current_actor",
            "distilled": second / "current_actor",
        }
        for name, actor in models.items():
            for namespace in NAMESPACES:
                path = self.output / "eval" / name / f"namespace{namespace}"
                if not (path / "collection_summary.json").is_file():
                    self.stage(
                        f"eval_{name}_ns{namespace}",
                        COLLECTOR,
                        [
                            "--model",
                            str(actor),
                            "--output",
                            str(path),
                            "--episode-ids",
                            *EPISODES,
                            "--seed",
                            "43",
                            "--round-index",
                            str(namespace),
                            "--parents-only",
                            "--parent-repetitions",
                            "4",
                        ],
                    )
                result = collection(path, actor, EPISODES, 43)
                if (
                    result.get("round_index") != namespace
                    or result.get("parents_only") is not True
                    or result.get("parent_repetitions") != 4
                    or result.get("planned_parent_trajectories") != 32
                ):
                    raise RuntimeError("fixed eight-window evaluation required")
                self.result["evaluations"][f"{name}/namespace{namespace}"] = dict(
                    path=str(path), summary=result, observation=observation(result)
                )
                self.persist()
        model_failures = any(
            entry["summary"].get("parents_failed", 0)
            for entry in self.result["evaluations"].values()
        )
        self.result["status"] = "COMPLETE_WITH_MODEL_FAILURES" if model_failures else "COMPLETE"

    def execute(self):
        self.prepare()
        self.persist()

        def forward(signum, _frame):
            if self.child is not None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.validate_inputs()
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
            attempt = self.state.get("attempt", 1)
            self.config = gpu.config_for_holder(
                holder,
                lease_token=f"order-demo-{uuid.uuid4().hex}",
                evidence_directory=self.output / f"gpu_attempt{attempt}",
                portfolio_target_bytes=76 * 1024**3,
            )
            gpu.freeze_config(self.output / f"gpu_config_attempt{attempt}.json", self.config)
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
            if checked["paused"] is not True or checked["payload_bytes"]:
                raise RuntimeError("holder must release before GPU work")
            self.run_steps()
            self.update(status=self.result["status"], current_stage=None)
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            self.result.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            code = 1
        finally:
            self.stop_child()
            self.persist()
            save(
                self.output / "completion.json",
                dict(
                    status=self.result["status"],
                    label_count=self.result.get("label_count"),
                    optimizer_updates=(self.result.get("stage2") or {}).get("optimizer_updates", 0),
                    completed_evaluation_cells=len(self.result["evaluations"]),
                    planned_evaluation_cells=6,
                    formal_valid_oos_access=False,
                    stage_wall_seconds=sum(s.get("seconds", 0) for s in self.state["stages"]),
                    source_parents_reused=True,
                    new_parent_generation_in_stage1=0,
                ),
            )
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.state["holder_restore"] = "REFILLING_76_GIB"
                except (OSError, RuntimeError, ValueError) as error:
                    self.state.update(
                        holder_restore="DEFERRED_TO_CGROUP_WATCHDOG", restore_reason=str(error)
                    )
                self.persist()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume-completed-stages", action="store_true")
    parser.add_argument("--stage-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    code = Demo(args).execute()
    if code == 0:
        code = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/diagnostics/read_counterfactual_order_demo.py"),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=False,
        ).returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()
