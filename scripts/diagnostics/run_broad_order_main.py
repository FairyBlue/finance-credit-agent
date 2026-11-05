#!/usr/bin/env python3
"""Run the authorized broad RLOO, frozen order scorer and full Valid experiment."""

from __future__ import annotations

import argparse
import hashlib
import signal
import subprocess
import time
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

DEFAULT = ROOT / "runs/m2-broad-rloo-order-main-20261028-v1"
TERMINAL = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES"}


def observed(summary):
    values = summary["parent_terminal_Y"]
    metrics = summary["parent_terminal_metrics"]
    n = summary["planned_parent_trajectories"]
    good = (
        len(values) == len(metrics) == n
        and summary["parents_complete"] == n
        and all(v is not None for v in values)
        and all(m is not None for m in metrics)
    )
    return dict(
        planned_parents=n,
        complete_parents=summary["parents_complete"],
        failed_parents=summary["parents_failed"],
        mean_net_return=sum(m["raw_net_return"] for m in metrics) / n if good else None,
        mean_Y=sum(values) / n if good else None,
        mean_transaction_cost=sum(m["transaction_cost"] for m in metrics) / n if good else None,
        invalid_requests=summary["invalid_requests"],
        model_requests=summary["model_requests"],
    )


class Main(Controller):
    def __init__(self, args):
        self.cli = args
        plan = args.output.resolve() / "config.json"
        self.plan_path, self.plan = plan, read(plan)
        self.digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        super().__init__(
            SimpleNamespace(
                output=args.output,
                model=Path(self.plan["source_model"]),
                collect_only=False,
                seed=43,
                episode_ids=self.plan["train_episode_ids"],
                alpha=0.0,
                eta=0.0,
                stage_timeout_seconds=args.stage_timeout_seconds,
                coordinator_request_timeout_seconds=60,
            )
        )
        self.state_path = self.output / "state.json"
        self.state.update(phase="broad_order_main", planned_added_updates=64, attempt=1)
        self.result = dict(
            profile=self.plan["profile"],
            status="RUNNING",
            config_sha256=self.digest,
            seed=43,
            train_episode_ids=self.plan["train_episode_ids"],
            valid_episode_ids=self.plan["valid_episode_ids"],
            source_model=self.plan["source_model"],
            source_training_state=self.plan["source_training_state"],
            added_updates=0,
            cumulative_updates=16,
            train_collections={},
            training_rounds=[],
            stage1_status="NOT_RUN",
            stage2_status="NOT_RUN",
            stage3_status="NOT_RUN",
            stage1_failure=None,
            selected_model=None,
            order_collection=None,
            ranker=None,
            evaluations={},
            cpu_baselines=None,
            formal_valid_access=False,
            formal_final_oos_access=False,
        )

    def persist(self):
        self.update()
        save(self.output / "results.json", self.result)

    def prepare(self):
        if not self.state_path.exists():
            return
        if not self.cli.resume_completed_stages:
            raise FileExistsError("explicit completed-stage resume required")
        old, result = read(self.state_path), read(self.output / "results.json")
        if old["status"] not in {"FAILED", "PAUSED"} or result["config_sha256"] != self.digest:
            raise RuntimeError("resume requires same config and stopped controller")
        attempt = old.get("attempt", 1) + 1
        archive = self.output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for name in ("state.json", "results.json", "completion.json"):
            p = self.output / name
            if p.exists():
                (archive / f"attempt{attempt - 1}_{name}").write_bytes(p.read_bytes())
        (self.output / "completion.json").unlink(missing_ok=True)
        self.state, self.result = old, result
        self.state.update(status="RUNNING", current_stage=None, attempt=attempt)
        self.result["status"] = "RUNNING"
        for d in (self.state, self.result):
            d.pop("error", None)

    def stage(self, name, module, arguments):
        self.command(f"{name}_attempt{self.state.get('attempt', 1)}", module, arguments)

    def wait_bootstrap(self):
        self.update(current_stage="WAITING_FOR_INITIAL_TRAIN_COLLECTION")
        receipt = self.output / "bootstrap_completion.json"
        while not receipt.exists():
            time.sleep(2)
        if read(receipt)["status"] != "COMPLETE":
            raise RuntimeError("initial collection service failed; original files retained")
        holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        deadline = time.monotonic() + 180
        while read(holder / "status.json").get("claim") is not None:
            if time.monotonic() > deadline:
                raise RuntimeError("bootstrap GPU claim not released")
            time.sleep(1)

    def parents(self, label, model, split, namespace, reps, mode="native", ranker=None):
        path = self.output / (f"stage1/{label}" if split == "train" else f"stage3/eval/{label}")
        if not (path / "collection_summary.json").exists():
            arguments = [
                "--experiment-config",
                str(self.plan_path),
                "--split",
                split,
                "--model",
                str(model),
                "--output",
                str(path),
                "--seed",
                "43",
                "--round-index",
                str(namespace),
                "--parents-only",
                "--parent-repetitions",
                str(reps),
                "--decision-mode",
                mode,
            ]
            if ranker is not None:
                arguments += ["--ranker-path", str(ranker)]
            self.stage(label, COLLECTOR, arguments)
        ids = self.plan[f"{split}_episode_ids"]
        summary = collection(path, model, ids, 43)
        if (
            summary["parent_repetitions"] != reps
            or summary["round_index"] != namespace
            or summary["planned_parent_trajectories"] != len(ids) * reps
        ):
            raise RuntimeError("collection identity/budget mismatch")
        return path, summary

    def train(self, number, data, actor, state):
        path = self.output / f"round{number}/train"
        summary_path = path / "training_summary.json"
        if not summary_path.exists():
            self.stage(
                f"train_round{number}",
                "src.agents.broad_rloo_train",
                [
                    "--experiment-config",
                    str(self.plan_path),
                    "--data",
                    str(data),
                    "--output",
                    str(path),
                    "--model",
                    str(actor),
                    "--resume-state",
                    str(state),
                    "--state-output",
                    str(self.output / "training_state"),
                    "--actor-output",
                    str(self.output / "current_actor"),
                    "--updates",
                    "32",
                    "--seed",
                    "43",
                ],
            )
        result = read(summary_path)
        expected = 16 + number * 32
        if (
            result["status"] != "COMPLETE"
            or result["optimizer_updates"] != 32
            or result["completed_updates"] != expected
        ):
            raise RuntimeError(f"round{number} did not commit its32 updates")
        return result

    def training_phase(self):
        old = Path(self.plan["source_model"])
        initial = self.output / "stage1/initial_train"
        initial_summary = collection(initial, old, self.plan["train_episode_ids"], 43)
        self.result["train_collections"]["initial_train"] = dict(
            path=str(initial),
            summary=initial_summary,
            observation=observed(initial_summary),
            updates=0,
        )
        self.persist()
        if self.result["stage1_status"] not in {"COMPLETE", "FAILED_FALLBACK"}:
            try:
                first = self.train(1, initial, old, Path(self.plan["source_training_state"]))
                self.result.update(added_updates=32, cumulative_updates=48)
                self.result["training_rounds"] = [first]
                self.persist()
                path, summary = self.parents(
                    "after32_train", self.output / "current_actor", "train", 11, 4
                )
                self.result["train_collections"]["after32_train"] = dict(
                    path=str(path), summary=summary, observation=observed(summary), updates=32
                )
                self.persist()
                second = self.train(
                    2, path, self.output / "current_actor", self.output / "training_state"
                )
                self.result["training_rounds"] = [first, second]
                self.result.update(
                    added_updates=64, cumulative_updates=80, stage1_status="COMPLETE"
                )
            except Exception as error:
                self.stop_child()
                latest = self.output / (
                    "round2/train/training_summary.json"
                    if len(self.result["training_rounds"]) == 1
                    else "round1/train/training_summary.json"
                )
                failed = read(latest) if latest.exists() else {}
                if failed.get("optimizer_updates", 0) == 0:
                    # A loader/wiring failure is repaired and resumed, rather
                    # than silently reported as an unsuccessful learning run.
                    raise
                self.result.update(
                    stage1_status="FAILED_FALLBACK",
                    stage1_failure=f"{type(error).__name__}: {error}",
                )
                self.event(
                    "BROAD_TRAINING_FAILED_CONTINUE_WITH_FROZEN_RLOO",
                    error=self.result["stage1_failure"],
                )
            self.persist()
        if self.result["stage1_status"] == "COMPLETE":
            actor = self.output / "current_actor"
            path, summary = self.parents("final_train", actor, "train", 10, 4)
            self.result["train_collections"]["final_train"] = dict(
                path=str(path), summary=summary, observation=observed(summary), updates=64
            )
        else:
            actor, path = old, initial
        self.result.update(selected_model=str(actor), selected_parent_source=str(path))
        self.persist()

    def order_phase(self):
        branch = self.output / "stage2/branches"
        if not (branch / "collection_summary.json").exists():
            self.stage(
                "order_branches",
                "src.agents.broad_order_collect",
                [
                    "--experiment-config",
                    str(self.plan_path),
                    "--model",
                    self.result["selected_model"],
                    "--source",
                    self.result["selected_parent_source"],
                    "--output",
                    str(branch),
                    "--seed",
                    "43",
                    "--prepared",
                    self.plan["prepared_train_path"],
                ],
            )
        self.result["order_collection"] = read(branch / "collection_summary.json")
        self.persist()
        output = self.output / "stage2/ranker"
        if not (output / "training_summary.json").exists():
            self.stage(
                "fit_order_ranker",
                "src.agents.order_ranker_train",
                [
                    "--collection",
                    str(branch),
                    "--output",
                    str(output),
                    "--seed",
                    "43",
                ],
            )
        trained = read(output / "training_summary.json")
        self.result.update(ranker=trained, stage2_status=trained["status"])
        self.persist()

    def valid_phase(self):
        selected = Path(self.result["selected_model"])
        ranker = self.result["ranker"]
        ranker_path = Path(ranker.get("model_path") or self.output / "stage2/ranker/ranker.json")
        models = {
            "SFT": (MODEL, "native", True),
            "old_RLOO": (Path(self.plan["source_model"]), "native", True),
            "broad_RLOO": (selected, "native", self.result["stage1_status"] == "COMPLETE"),
            "menu_control": (selected, "menu", True),
            "learned_ranker": (
                selected,
                "ranker",
                ranker["status"] == "COMPLETE" and ranker_path.is_file(),
            ),
        }
        save(
            self.output / "models_frozen.json",
            dict(
                config_sha256=self.digest,
                stage1_status=self.result["stage1_status"],
                selected_model=str(selected),
                ranker=str(ranker_path),
                models={
                    name: dict(path=str(path), decision_mode=mode, available=available)
                    for name, (path, mode, available) in models.items()
                },
                no_more_training_in_this_experiment=True,
                formal_final_oos_access=False,
            ),
        )
        if not Path(self.plan["prepared_valid_path"]).exists():
            self.stage(
                "prepare_valid",
                "src.agents.broad_prepare",
                ["--experiment-config", str(self.plan_path), "--split", "valid"],
            )
        self.result["formal_valid_access"] = True
        self.persist()
        for name, (path, mode, available) in models.items():
            if not available:
                self.result["evaluations"][name] = dict(
                    status="UNAVAILABLE",
                    planned_parents=120,
                    reason="upstream training has no complete final model",
                )
                self.persist()
                continue
            folder, summary = self.parents(
                name, path, "valid", 20, 2, mode, ranker_path if mode == "ranker" else None
            )
            self.result["evaluations"][name] = dict(
                status="COMPLETE"
                if summary["parents_failed"] == 0
                else "COMPLETE_WITH_MODEL_FAILURES",
                path=str(folder),
                summary=summary,
                observation=observed(summary),
            )
            self.persist()
        baseline = self.output / "stage3/cpu_baselines"
        if not (baseline / "summary.json").exists():
            self.stage(
                "cpu_baselines",
                "src.agents.broad_cpu_baselines",
                [
                    "--experiment-config",
                    str(self.plan_path),
                    "--prepared",
                    self.plan["prepared_valid_path"],
                    "--output",
                    str(baseline),
                ],
            )
        self.result.update(cpu_baselines=read(baseline / "summary.json"), stage3_status="COMPLETE")
        self.persist()

    def execute(self):
        self.prepare()
        self.persist()

        def forward(signum, _frame):
            if self.child is not None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal{signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.wait_bootstrap()
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
            attempt = self.state.get("attempt", 1)
            self.config = gpu.config_for_holder(
                holder,
                lease_token=f"broad-main-{uuid.uuid4().hex}",
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
                raise RuntimeError("GPU holder did not release")
            self.training_phase()
            self.order_phase()
            self.valid_phase()
            failed = (
                self.result["stage1_status"] != "COMPLETE"
                or self.result["stage2_status"] != "COMPLETE"
                or any(v["status"] != "COMPLETE" for v in self.result["evaluations"].values())
            )
            status = "COMPLETE_WITH_MODEL_FAILURES" if failed else "COMPLETE"
            self.result["status"] = status
            self.update(status=status, current_stage=None)
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
                    added_optimizer_updates=self.result["added_updates"],
                    cumulative_updates=self.result["cumulative_updates"],
                    completed_valid_arms=sum(
                        v["status"] in TERMINAL for v in self.result["evaluations"].values()
                    ),
                    observed_complete_valid_parents=sum(
                        v.get("observation", {}).get("complete_parents", 0)
                        for v in self.result["evaluations"].values()
                    ),
                    planned_valid_parents=600,
                    formal_valid_access=self.result["formal_valid_access"],
                    formal_final_oos_access=False,
                    stage_wall_seconds=sum(s.get("seconds", 0) for s in self.state["stages"]),
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
    parser.add_argument("--stage-timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    code = Main(args).execute()
    if code == 0:
        code = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/diagnostics/read_broad_order_main.py"),
                "--output",
                str(args.output.resolve()),
            ],
            cwd=ROOT,
            check=False,
        ).returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()
