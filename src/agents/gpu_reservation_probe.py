"""Model-free CUDA handoff probes for the persistent holder.

Run only as an explicitly launched diagnostic in a dedicated systemd unit.
Workers retain empty tensors without inference, training, or arithmetic kernels.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.agents.gpu_memory_coordinator import (
    GIB,
    GpuMemoryCoordinator,
    controller_claim,
    controller_idle,
    load_config,
    reservation_status,
)
from src.agents.gpu_reservation_service import atomic_json, now, process_identity


def worker(args, config):
    coordinator = GpuMemoryCoordinator(config, identity="f" * 64)
    admitted = coordinator.before_phase("init_actor")
    import torch

    payload = torch.empty(2 * GIB, dtype=torch.uint8, device="cuda:0")
    coordinator.after_phase("init_actor")
    atomic_json(args.directory / f"worker-{args.index}-ready.json", {
        "at_SGT": now(), "worker": process_identity(os.getpid()),
        "payload_bytes": payload.numel(), "admission": admitted,
        "holder": reservation_status(config),
        "model_loaded": False, "training_updates": 0,
    })
    time.sleep(args.hold_seconds)
    del payload
    torch.cuda.empty_cache()
    atomic_json(args.directory / f"worker-{args.index}-finished.json", {
        "at_SGT": now(), "status": "PASS", "model_loaded": False,
        "training_updates": 0,
    })


def controller(args, config):
    initial = controller_claim(config)
    rounds = 2 if args.command == "normal" else 1
    records = []
    for index in range(rounds):
        process = subprocess.Popen([
            sys.executable, "-u", "-m", "src.agents.gpu_reservation_probe", "worker",
            "--config", str(args.config), "--sha256", args.sha256,
            "--directory", str(args.directory), "--index", str(index),
            "--hold-seconds", str(args.hold_seconds),
        ], start_new_session=True)
        atomic_json(args.directory / f"controller-worker-{index}.json", {
            "at_SGT": now(), "owner": process_identity(os.getpid()),
            "worker_pid": process.pid, "different_process_group": True,
            "mode": args.command,
        })
        if args.command == "orphan":
            atomic_json(args.directory / "controller_exit.json", {
                "at_SGT": now(), "status": "EXITED_WITH_LIVE_CHILD", "worker_pid": process.pid,
                "initial_holder_payload_bytes": initial["payload_bytes"],
            })
            return
        code = process.wait(timeout=90)
        if code:
            raise RuntimeError(f"model-free probe worker exited {code}")
        controller_idle(config)
        deadline = time.monotonic() + 15
        while True:
            state = reservation_status(config)
            if state["payload_bytes"] > 0 and state["paused"] is False:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("holder did not refill between probe tasks")
            time.sleep(0.25)
        records.append({"index": index, "returncode": code,
                        "refilled_payload_bytes": state["payload_bytes"]})
        controller_claim(config)
    atomic_json(args.directory / "controller_complete.json", {
        "at_SGT": now(), "status": "PASS", "rounds": records,
        "model_loaded": False, "training_updates": 0,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("normal", "orphan", "worker"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--hold-seconds", type=float, default=10)
    args = parser.parse_args(argv)
    if not 0 < args.hold_seconds <= 30:
        parser.error("hold-seconds must be in (0, 30]")
    args.directory = args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config.resolve(), args.sha256)
    if args.command == "worker":
        worker(args, config)
    else:
        controller(args, config)


if __name__ == "__main__":
    main()
