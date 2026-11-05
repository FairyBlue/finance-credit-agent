#!/usr/bin/env python3
"""Run the authorized teacher/SFT step only, with persistent GPU coordination."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from run_same_prefix_online_demo import MODEL, ROOT, gpu, read, save

DEFAULT = ROOT / "runs/m2-portfolio-teacher-sft-20261028-v1"


def execute(output, resume):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    attempt = 1
    if state_path.exists():
        old = read(state_path)
        if not resume or old["status"] != "FAILED":
            raise RuntimeError("fresh output or explicit failed-stage resume required")
        attempt = old["attempt"] + 1
        archive = output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for name in ("state.json", "completion.json"):
            path = output / name
            if path.exists():
                path.rename(archive / f"attempt{attempt - 1}_{name}")
    state = dict(
        status="RUNNING", attempt=attempt, current_stage=None, stages=[],
        authorized_step=2, rl_optimizer_updates=0, step3_started=False,
        formal_valid_access=False, formal_final_oos_access=False,
        started_at_UTC=datetime.now(timezone.utc).isoformat(),
    )
    child, config, claimed = None, None, False
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "third_party/agent-r1")]),
        USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
    )

    def persist():
        save(state_path, state)

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        else:
            raise KeyboardInterrupt(f"signal {signum}")

    handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}

    def stage(name, module, arguments, gpu_worker=False):
        nonlocal child
        log = output / "logs" / f"{name}_attempt{attempt}.log"
        log.parent.mkdir(exist_ok=True)
        entry = dict(name=name, status="RUNNING", log=str(log))
        state["stages"].append(entry)
        state["current_stage"] = name
        persist()
        started = time.monotonic()
        with log.open("x") as stream:
            child = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-u", "-m", module, *arguments],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            )
            if gpu_worker:
                gpu.request(config["directory"], dict(
                    op="attach", lease_token=config["lease_token"],
                    workers=[gpu.process_identity(child.pid)], request_id=uuid.uuid4().hex,
                ), timeout=60)
            code = child.wait(timeout=14400)
        child = None
        entry.update(status="COMPLETE" if code == 0 else "FAILED",
                     seconds=time.monotonic() - started)
        persist()
        if code:
            raise RuntimeError(f"{name} exited {code}; see {log}")

    code = 0
    try:
        persist()
        if not (output / "data/split_manifest.json").is_file():
            raise RuntimeError("Train learning/development manifest is not ready")
        holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        config = gpu.config_for_holder(
            holder, lease_token=f"portfolio-teacher-{uuid.uuid4().hex}",
            evidence_directory=output / f"gpu_attempt{attempt}",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(output / f"gpu_config_attempt{attempt}.json", config)
        holder_state = gpu.controller_claim(config)
        claimed = True
        gpu.request(str(holder), dict(
            op="release", lease_token=config["lease_token"],
            bytes=holder_state["maximum_bytes"], request_id=uuid.uuid4().hex,
        ), timeout=60)
        released = gpu.reservation_status(config)
        if not released["paused"] or released["payload_bytes"]:
            raise RuntimeError("GPU holder has not released")
        manifest = output / "data/split_manifest.json"
        common = output / "eval/common_sft"
        if not (common / "summary.json").exists():
            stage("common_sft_eval", "src.agents.portfolio_teacher_eval", [
                "--split-manifest", str(manifest), "--subset", "both",
                "--policy-role", "common_sft", "--model", str(MODEL),
                "--output", str(common), "--seed", "43",
            ], gpu_worker=True)
        deadline = time.monotonic() + 3600
        while not (output / "data/dataset_ready.json").exists():
            state["current_stage"] = "WAITING_FOR_TEACHER_DATA"
            persist()
            if time.monotonic() > deadline:
                raise RuntimeError("teacher data did not finish")
            time.sleep(2)
        data_ready = read(output / "data/dataset_ready.json")
        if data_ready["status"] != "COMPLETE":
            raise RuntimeError("teacher data failed")
        training = output / "training"
        if not (training / "summary.json").exists():
            stage("teacher_lora_sft", "src.agents.portfolio_teacher_sft", [
                "--data", str(output / "data/train.jsonl"), "--model", str(MODEL),
                "--output", str(training), "--seed", "43",
            ], gpu_worker=True)
        state["sft_optimizer_updates"] = read(training / "summary.json")["optimizer_steps"]
        persist()
        actor = output / "current_actor"
        if not (actor / "merge_receipt.json").exists():
            stage("merge_teacher_actor", "src.agents.coldstart_sft", [
                "merge", "--model", str(MODEL), "--adapter", str(training / "adapter"),
                "--output", str(actor),
            ])
        evaluation = output / "eval/teacher_sft"
        if not (evaluation / "summary.json").exists():
            stage("teacher_sft_eval", "src.agents.portfolio_teacher_eval", [
                "--split-manifest", str(manifest), "--subset", "both",
                "--policy-role", "teacher_sft", "--model", str(actor),
                "--output", str(evaluation), "--reference", str(common), "--seed", "43",
            ], gpu_worker=True)
        state.update(status="COMPLETE", current_stage=None)
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        code = 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if claimed:
            try:
                gpu.controller_idle(config)
                state["holder_restore"] = "REFILLING_76_GIB"
            except (RuntimeError, OSError, ValueError) as error:
                state.update(holder_restore="DEFERRED_TO_WATCHDOG", restore_error=str(error))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        persist()
        save(output / "completion.json", state)
    if code == 0:
        code = subprocess.run([
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/diagnostics/read_portfolio_teacher_sft.py"),
            "--output", str(output),
        ], cwd=ROOT, env=environment, check=False).returncode
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(args.output, args.resume))


if __name__ == "__main__":
    main()
