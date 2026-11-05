#!/usr/bin/env python3
"""Persistent controller for the authorized three-step mandate-repair demo."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_same_prefix_online_demo import MODEL, ROOT, gpu, read, save

DEFAULT = ROOT / "runs/m3-mandate-repair-20261029-v1"
COLLECTOR = "src.agents.mandate_repair_collect"
TRAINER = "src.agents.mandate_repair_train"
BUDGET = 3_000_000
AUX_BUDGET = BUDGET // 3
GROUP_RESERVATION = 4 * 8 * 8192


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


class Controller:
    def __init__(self, output, resume):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "state.json"
        self.child, self.config, self.claimed = None, None, False
        self.attempt = 1
        if self.path.exists():
            previous = read(self.path)
            if not resume or previous["status"] != "FAILED":
                raise RuntimeError("new run or explicit resume of a failed controller required")
            self.attempt = previous["attempt"] + 1
            archive = self.output / "controller_attempts"
            archive.mkdir(exist_ok=True)
            save(archive / f"attempt{self.attempt - 1}.json", previous)
            self.state = previous
            self.state.pop("error", None)
        else:
            self.state = dict(
                authorized_steps=[1, 2, 3],
                stages=[],
                arms={},
                started_at=now(),
                formal_valid_access=False,
                formal_final_oos_access=False,
                synthetic_mandates=True,
                known_quote_primary=True,
            )
        self.state.update(status="RUNNING", attempt=self.attempt, updated_at=now())
        self.environment = dict(os.environ)
        self.environment.update(
            PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "third_party/agent-r1")]),
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )

    def persist(self, **values):
        self.state.update(**values, updated_at=now())
        save(self.path, self.state)

    def stage(self, name, module, arguments, gpu_worker=True):
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{name}_attempt{self.attempt}.log"
        entry = dict(
            name=name, status="RUNNING", log=str(log), started_at=now(), gpu_worker=gpu_worker
        )
        self.state["stages"].append(entry)
        self.persist(current_stage=name)
        started = time.monotonic()
        with log.open("x") as stream:
            self.child = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-u", "-m", module, *map(str, arguments)],
                cwd=ROOT,
                env=self.environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            if gpu_worker:
                gpu.request(
                    self.config["directory"],
                    dict(
                        op="attach",
                        lease_token=self.config["lease_token"],
                        workers=[gpu.process_identity(self.child.pid)],
                        request_id=uuid.uuid4().hex,
                    ),
                    timeout=60,
                )
            code = self.child.wait(timeout=8 * 3600)
        self.child = None
        entry.update(
            status="COMPLETE" if code == 0 else "FAILED",
            seconds=time.monotonic() - started,
            finished_at=now(),
        )
        self.persist()
        if code:
            raise RuntimeError(f"{name} exited {code}; see {log}")

    def fresh(self, directory, marker="summary.json"):
        if (directory / marker).exists():
            return False
        if directory.exists() and any(directory.iterdir()):
            archive = directory.parent / "failed_attempts"
            archive.mkdir(exist_ok=True)
            directory.rename(archive / f"{directory.name}_{uuid.uuid4().hex[:8]}")
        return True

    def evaluate(self, key, adapter=None, split="dev", metadata=None):
        destination = self.output / "eval" / f"{key}_{split}"
        data = self.output / "data" / ("development.jsonl" if split == "dev" else "test.jsonl")
        if split == "test" and not (self.output / "frozen_models.json").exists():
            raise RuntimeError("reserved test cannot open before all endpoints freeze")
        if self.fresh(destination):
            args = [
                "--data",
                data,
                "--base-model",
                MODEL,
                "--output",
                destination,
                "--seed",
                43,
                "--temperature",
                0,
                "--repetitions",
                1,
            ]
            if adapter is not None:
                args += ["--adapter", adapter]
            self.stage(f"eval_{key}_{split}", COLLECTOR, args)
        summary = read(destination / "summary.json")
        summary.update(split=split, policy=key, **(metadata or {}))
        save(destination / "summary.json", summary)
        if summary["planned_trajectories"] != 100:
            raise RuntimeError("predefined evaluation denominator changed")
        return summary

    def report(self):
        script = ROOT / "scripts/diagnostics/read_mandate_repair.py"
        if script.exists():
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(script), "--output", str(self.output)],
                cwd=ROOT,
                env=self.environment,
                capture_output=True,
                text=True,
            )
            (self.output / "readout.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                self.persist(readout_error=result.stderr[-3000:])

    def train_arm(self, arm, seed, common):
        key = f"{arm}_seed{seed}"
        directory = self.output / "training" / key
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = (
            read(progress_path)
            if progress_path.exists()
            else dict(
                arm=arm,
                seed=seed,
                status="RUNNING",
                token_budget=BUDGET,
                auxiliary_token_cap=AUX_BUDGET if arm != "rloo" else 0,
                cumulative_tokens=0,
                parent_tokens=0,
                auxiliary_tokens=0,
                rounds=[],
                final_adapter=str(common),
                completed_optimizer_steps=0,
            )
        )
        tasks = [
            json.loads(line) for line in (self.output / "data/train.jsonl").read_text().splitlines()
        ]
        # Same seed fixes the task stream across arms; method-specific branches
        # may consume the budget sooner, as intended by the total-token control.
        random.Random(seed).shuffle(tasks)
        actor = Path(progress["final_adapter"])
        if progress["status"] == "COMPLETE":
            return actor
        round_index = len(progress["rounds"])
        while BUDGET - progress["cumulative_tokens"] >= GROUP_RESERVATION:
            rd = directory / f"round_{round_index:03d}"
            rd.mkdir(parents=True, exist_ok=True)
            selected = rd / "selected.jsonl"
            # Sixteen K4 groups per frozen sampling policy; no repeated actor
            # updates on a cached whole corpus and no per-round fresh Adam.
            count = 16
            offset = round_index * count
            if not selected.exists():
                selected.write_text(
                    "".join(
                        json.dumps(tasks[(offset + i) % len(tasks)]) + "\n" for i in range(count)
                    )
                )
            collection = rd / "collect"
            remaining = BUDGET - progress["cumulative_tokens"]
            if self.fresh(collection):
                args = [
                    "--data",
                    selected,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    collection,
                    "--seed",
                    seed,
                    "--temperature",
                    1,
                    "--repetitions",
                    4,
                    "--token-budget",
                    remaining,
                    "--max-groups",
                    count,
                ]
                if arm != "rloo":
                    args += [
                        "--arm",
                        arm,
                        "--aux-token-budget",
                        max(0, AUX_BUDGET - progress["auxiliary_tokens"]),
                    ]
                self.stage(f"{key}_round{round_index:03d}_collect", COLLECTOR, args)
            collected = read(collection / "summary.json")
            spent = collected["total_tokens"]
            if spent <= 0:
                progress["budget_stop_reason"] = "NO_FULL_TASK_GROUP_ADMITTED"
                break
            destination = rd / "train"
            next_actor = destination / "adapter"
            if self.fresh(destination):
                args = [
                    "rl",
                    "--data",
                    collection,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    destination,
                    "--seed",
                    seed,
                    "--arm",
                    arm,
                ]
                if progress["rounds"]:
                    previous = Path(progress["rounds"][-1]["train"]) / "training_state.pt"
                    args += ["--resume-state", previous]
                if arm != "rloo":
                    args += ["--local-credit", collection / "local_credit.jsonl"]
                self.stage(f"{key}_round{round_index:03d}_train", TRAINER, args)
            trained = read(destination / "summary.json")
            aux = collected.get("auxiliary_tokens", collected.get("aux_tokens", 0))
            progress["cumulative_tokens"] += spent
            progress["auxiliary_tokens"] += aux
            progress["parent_tokens"] += spent - aux
            progress["completed_optimizer_steps"] = trained["completed_optimizer_steps"]
            progress["final_adapter"] = str(next_actor)
            progress["rounds"].append(
                dict(
                    round=round_index,
                    collect=str(collection),
                    train=str(destination),
                    cumulative_tokens=progress["cumulative_tokens"],
                    tokens=spent,
                    auxiliary_tokens=aux,
                    optimizer_updates=trained["optimizer_updates"],
                    completed_optimizer_steps=trained["completed_optimizer_steps"],
                )
            )
            if progress["cumulative_tokens"] > BUDGET or progress["auxiliary_tokens"] > AUX_BUDGET:
                raise RuntimeError("actual model tokens exceeded the fixed arm budget")
            save(progress_path, progress)
            self.state["arms"][key] = progress
            self.persist()
            actor = next_actor
            # Every exported endpoint is evaluated, so quality-versus-actual-cost
            # curves do not silently select a favorable intermediate policy.
            self.evaluate(
                f"{key}_curve{round_index:03d}",
                actor,
                metadata=dict(
                    arm=arm,
                    training_seed=seed,
                    cumulative_training_tokens=progress["cumulative_tokens"],
                    curve_checkpoint=True,
                ),
            )
            self.report()
            round_index += 1
        progress.update(
            status="COMPLETE",
            finished_at=now(),
            final_adapter=str(actor),
            unused_budget_tokens=BUDGET - progress["cumulative_tokens"],
            endpoints_not_strictly_equal_actual_tokens=True,
        )
        save(progress_path, progress)
        self.state["arms"][key] = progress
        self.persist()
        return actor

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            manifest = read(self.output / "data/manifest.json")
            if manifest["status"] != "COMPLETE" or manifest.get("data_build_version") != 2:
                raise RuntimeError("final version-two task data required")
            self.persist(current_step=1)
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
            self.config = gpu.config_for_holder(
                holder,
                lease_token=f"mandate-repair-{uuid.uuid4().hex}",
                evidence_directory=self.output / f"gpu_attempt{self.attempt}",
                portfolio_target_bytes=76 * 1024**3,
            )
            gpu.freeze_config(self.output / f"gpu_config_attempt{self.attempt}.json", self.config)
            hs = gpu.controller_claim(self.config)
            self.claimed = True
            gpu.request(
                str(holder),
                dict(
                    op="release",
                    lease_token=self.config["lease_token"],
                    bytes=hs["maximum_bytes"],
                    request_id=uuid.uuid4().hex,
                ),
                timeout=60,
            )
            released = gpu.reservation_status(self.config)
            if not released["paused"] or released["payload_bytes"]:
                raise RuntimeError("holder did not release for the real worker")
            self.evaluate("original")
            self.report()
            teacher = self.output / "teacher/summary.json"
            deadline = time.monotonic() + 1800
            while not teacher.exists():
                self.persist(current_stage="WAITING_TEACHER")
                if time.monotonic() > deadline:
                    raise RuntimeError("teacher data has not become ready")
                time.sleep(2)
            if read(teacher)["status"] != "COMPLETE":
                raise RuntimeError("teacher failed")
            self.persist(current_step=2, step1_status="COMPLETE")
            common = self.output / "common_sft"
            common_resume = None
            if (common / "training_state.pt").is_file() and not (common / "summary.json").exists():
                archive = self.output / "failed_attempts"
                archive.mkdir(exist_ok=True)
                retained = archive / f"common_sft_resume_{uuid.uuid4().hex[:8]}"
                common.rename(retained)
                common_resume = retained / "training_state.pt"
                save(
                    self.output / f"sft_resume_attempt{self.attempt}.json",
                    dict(
                        checkpoint=str(common_resume),
                        same_data_order=True,
                        same_fp32_weights_and_adam=True,
                        execution_change="disable activation checkpointing; global batch unchanged",
                    ),
                )
            if self.fresh(common):
                arguments = [
                    "sft",
                    "--data",
                    self.output / "teacher/train_requests.jsonl",
                    "--model",
                    MODEL,
                    "--output",
                    common,
                    "--seed",
                    43,
                ]
                if common_resume is not None:
                    arguments += ["--resume-state", common_resume]
                self.stage("common_sft", TRAINER, arguments)
            common_actor = common / "adapter"
            self.evaluate("sft", common_actor)
            endpoints = {"original": None, "sft": common_actor}
            endpoints["rloo_seed42"] = self.train_arm("rloo", 42, common_actor)
            self.evaluate(
                "rloo_seed42",
                endpoints["rloo_seed42"],
                metadata=dict(
                    arm="rloo",
                    training_seed=42,
                    cumulative_training_tokens=self.state["arms"]["rloo_seed42"][
                        "cumulative_tokens"
                    ],
                ),
            )
            self.persist(current_step=3, step2_status="COMPLETE")
            self.report()
            for arm, seed in [("rloo", 43), ("sham", 42), ("m3", 42), ("sham", 43), ("m3", 43)]:
                key = f"{arm}_seed{seed}"
                endpoints[key] = self.train_arm(arm, seed, common_actor)
                self.evaluate(
                    key,
                    endpoints[key],
                    metadata=dict(
                        arm=arm,
                        training_seed=seed,
                        cumulative_training_tokens=self.state["arms"][key]["cumulative_tokens"],
                    ),
                )
                self.report()
            from src.agents.mandate_repair_train import adapter_digest

            frozen = self.output / "frozen_models.json"
            if not frozen.exists():
                save(
                    frozen,
                    dict(
                        status="COMPLETE",
                        frozen_at=now(),
                        models={
                            key: dict(
                                base_model=str(MODEL),
                                adapter=str(path) if path else None,
                                adapter_sha256=adapter_digest(path) if path else None,
                            )
                            for key, path in endpoints.items()
                        },
                        no_endpoint_selected_by_dev_score=True,
                    ),
                )
            for key, adapter in endpoints.items():
                self.evaluate(key, adapter, split="test")
            self.persist(
                status="COMPLETE",
                current_step=3,
                step3_status="COMPLETE",
                current_stage=None,
                finished_at=now(),
            )
        except BaseException as error:
            code = 1
            self.persist(status="FAILED", error=f"{type(error).__name__}: {error}")
        finally:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.state["holder_restore"] = "REFILLING_76_GIB"
                except Exception as error:
                    self.state.update(holder_restore="WATCHDOG_PENDING", restore_error=str(error))
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            self.persist()
            save(self.output / "completion.json", self.state)
            self.report()
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(Controller(args.output, args.resume).run())


if __name__ == "__main__":
    main()
