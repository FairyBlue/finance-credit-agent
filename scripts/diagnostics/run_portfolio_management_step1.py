#!/usr/bin/env python3
"""Run only the authorized first portfolio-management step, surviving disconnects."""

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

DEFAULT = ROOT / "runs/m2-portfolio-management-step1-20261028-v1"
PREPARED = ROOT / "runs/m2-broad-rloo-order-main-20261028-v1/prepared/train.parquet"


def execute(output, resume):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "state.json"
    attempt = 1
    if path.exists():
        previous = read(path)
        if not resume or previous["status"] != "FAILED":
            raise RuntimeError("fresh output or explicit failed-stage resume required")
        attempt = previous["attempt"] + 1
        archive = output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for name in ("state.json", "completion.json"):
            p = output / name
            if p.exists():
                p.rename(archive / f"attempt{attempt - 1}_{name}")
    state = dict(
        status="RUNNING", attempt=attempt, current_stage=None, stages=[],
        authorized_step=1, optimizer_updates=0, formal_valid_access=False,
        formal_final_oos_access=False,
        started_at_UTC=datetime.now(timezone.utc).isoformat(),
    )
    child, config, claimed = None, None, False
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "third_party/agent-r1")]),
        USE_TORCH="1", USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
        FCA_ONLINE_ADAPTER_CACHE=str(
            ROOT / "runs/m2-same-prefix-online-demo-20261024-v2/shared_adapter.pkl"
        ),
    )

    def persist():
        save(path, state)

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        else:
            raise KeyboardInterrupt(f"signal {signum}")

    handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}

    def stage(name, module, arguments, uses_gpu=False):
        nonlocal child
        log = output / "logs" / f"{name}_attempt{attempt}.log"
        log.parent.mkdir(exist_ok=True)
        record = dict(name=name, status="RUNNING", log=str(log))
        state["stages"].append(record)
        state["current_stage"] = name
        persist()
        started = time.monotonic()
        with log.open("x") as stream:
            child = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-u", "-m", module, *arguments],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
            if uses_gpu:
                gpu.request(config["directory"], dict(
                    op="attach", lease_token=config["lease_token"],
                    workers=[gpu.process_identity(child.pid)], request_id=uuid.uuid4().hex,
                ), timeout=60)
            code = child.wait(timeout=3600)
        child = None
        record.update(
            status="COMPLETE" if code == 0 else "FAILED", seconds=time.monotonic() - started
        )
        persist()
        if code:
            raise RuntimeError(f"{name} exited {code}; see {log}")

    code = 0
    try:
        persist()
        if not (output / "rules/summary.json").exists():
            stage("fixed_rules", "src.agents.portfolio_management_rules", [
                "--prepared", str(PREPARED), "--output", str(output / "rules"),
            ])
        if not (output / "llm_demo/summary.json").exists():
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
            config = gpu.config_for_holder(
                holder, lease_token=f"portfolio-step1-{uuid.uuid4().hex}",
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
            stage("zero_shot_llm", "src.agents.portfolio_management_llm_demo", [
                "--prepared", str(PREPARED), "--model", str(MODEL),
                "--output", str(output / "llm_demo"), "--seed", "43",
            ], uses_gpu=True)
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
        command = [str(ROOT / ".venv/bin/python"),
                   str(ROOT / "scripts/diagnostics/read_portfolio_management_step1.py"),
                   "--output", str(output)]
        code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(args.output, args.resume))


if __name__ == "__main__":
    main()
