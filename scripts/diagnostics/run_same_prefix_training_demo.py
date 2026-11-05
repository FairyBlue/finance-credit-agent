#!/usr/bin/env python3
"""Run the training demo inside a dedicated systemd service using our GPU lease."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.gpu_memory_coordinator import (  # noqa: E402
    config_for_holder,
    controller_claim,
    controller_idle,
    freeze_config,
    reservation_status,
)
from src.agents.gpu_reservation_service import process_identity, request  # noqa: E402


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def run(output: Path, holder: Path, trainer_args: list[str], lease: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "demo_gpu_wrapper.json"
    config_path = output / "demo_gpu_coordinator_config.json"
    log_path = output / "train.log"
    if any(path.exists() for path in (receipt_path, config_path, log_path)):
        raise FileExistsError("fresh wrapper files required; no automatic retry")
    config = config_for_holder(
        holder,
        lease_token=lease,
        evidence_directory=output / "demo_gpu_coordinator",
        portfolio_target_bytes=76 * 1024**3,
    )
    digest = freeze_config(config_path, config)
    command = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "src.agents.same_prefix_training_demo",
        *trainer_args,
    ]
    record = {
        "started_at_UTC": datetime.now(timezone.utc).isoformat(),
        "controller_pid": os.getpid(),
        "command": command,
        "lease_token": lease,
        "coordinator_config_sha256": digest,
        "child_returncode": None,
        "holder_restore": "NOT_CLAIMED",
        "unclaim": "WATCHDOG_AFTER_SERVICE_CGROUP_EMPTY",
    }
    child = None
    claimed = False

    def forward(signum, _frame):
        record["stop_signal"] = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        else:
            raise KeyboardInterrupt(f"controller received signal {signum}")

    previous_handlers = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        state = controller_claim(config)
        claimed = True
        state = request(
            str(holder),
            {
                "op": "release",
                "lease_token": lease,
                "bytes": state["maximum_bytes"],
                "request_id": f"demo-release-{lease}",
            },
        )
        checked = reservation_status(config)
        if (
            checked["claim"]["lease_token"] != lease
            or checked["paused"] is not True
            or any(
                checked[k]
                for k in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
            )
        ):
            raise RuntimeError("our holder did not release to paused zero residency")
        emit(
            "GPU_HOLDER_RELEASED",
            holder_payload_bytes=checked["payload_bytes"],
            controller_pid=os.getpid(),
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            path
            for path in (
                str(ROOT),
                str(ROOT / "third_party/agent-r1"),
                environment.get("PYTHONPATH", ""),
            )
            if path
        )
        environment.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            CUDA_VISIBLE_DEVICES=str(config["cuda_device"]),
        )
        with log_path.open("x") as log:
            child = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            record["child_pid"] = child.pid
            try:
                request(
                    str(holder),
                    {
                        "op": "attach",
                        "lease_token": lease,
                        "workers": [process_identity(child.pid)],
                        "request_id": f"demo-attach-{lease}",
                    },
                )
            except (FileNotFoundError, ProcessLookupError):
                if child.poll() is None:
                    raise
            emit("TRAINER_STARTED", pid=child.pid, log=str(log_path))
            record["child_returncode"] = child.wait()
        returncode = record["child_returncode"]
        return returncode if returncode >= 0 else 128 - returncode
    except BaseException as error:
        record["controller_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Only our direct child is signalled. Remaining cgroup workers prevent
        # refill, and the holder watchdog restores occupancy after service exit.
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            record["child_returncode"] = child.returncode
        if claimed:
            try:
                restored = controller_idle(config)
                record["holder_restore"] = "REFILLING_76_GIB"
                record["holder_payload_bytes_at_refill"] = restored["payload_bytes"]
            except (OSError, RuntimeError, ValueError) as error:
                record["holder_restore"] = "DEFERRED_TO_CGROUP_WATCHDOG"
                record["restore_reason"] = f"{type(error).__name__}: {error}"
        record["finished_at_UTC"] = datetime.now(timezone.utc).isoformat()
        with receipt_path.open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        emit(
            "TRAINER_EXIT",
            child_returncode=record["child_returncode"],
            holder_restore=record["holder_restore"],
            receipt=str(receipt_path),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--lease-token", default=None)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    holder = args.holder_directory
    if holder is None:
        holder = Path(
            json.loads((ROOT / "runs/GPU_RESERVATION_CURRENT.json").read_text())["directory"]
        )
    trainer_args = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
    lease = args.lease_token or f"same-prefix-demo-{uuid.uuid4().hex}"
    raise SystemExit(run(args.output.resolve(), holder.resolve(), trainer_args, lease))


if __name__ == "__main__":
    main()
