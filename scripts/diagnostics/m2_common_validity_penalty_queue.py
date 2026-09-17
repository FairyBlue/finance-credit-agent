#!/usr/bin/env python3
"""Wait for the shared H100, then exec the sealed, fixed common-penalty run once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-common-validity-penalty-20260917"
WAIT_SECONDS = 12 * 3600
POLL_SECONDS = 45
FREE_BELOW_MIB = 1024


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def save(base, record):
    target = base / "gpu_queue_state.json"
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def memory_used():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or not lines[0].strip().isdigit():
        raise RuntimeError("expected exactly one GPU with an integer memory.used reading")
    return int(lines[0])


def queue(base, *, wait_seconds=WAIT_SECONDS, poll_seconds=POLL_SECONDS):
    base = Path(base).resolve()
    if not (base / "run_contract.json").is_file():
        raise RuntimeError("prepare, preflight, and seal this run before queueing")
    if (base / "state.json").exists() or (base / "gpu_queue_state.json").exists():
        raise RuntimeError("run or queue already exists; no implicit retry")
    if wait_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("queue time bounds must be positive")
    started = time.monotonic()
    record = dict(
        schema_version=1, status="WAITING_FOR_GPU", started_at_SGT=now(),
        directory=str(base), maximum_wait_seconds=wait_seconds,
        poll_seconds=poll_seconds, free_below_MiB=FREE_BELOW_MIB,
        memory_used_MiB=None, waited_seconds=0.0,
        automatic_retry=False, stops_other_processes=False, telemetry_error_count=0,
    )

    def update(status, **extra):
        record.update(
            status=status, at_SGT=now(),
            waited_seconds=round(time.monotonic() - started, 3), **extra,
        )
        save(base, record)

    try:
        update("WAITING_FOR_GPU")
        while time.monotonic() - started < wait_seconds:
            try:
                used = memory_used()
            except (
                subprocess.TimeoutExpired, subprocess.CalledProcessError, RuntimeError
            ) as error:
                # An unknown GPU reading never admits training. Keep the same
                # bounded queue alive through transient NVML/telemetry failures.
                remaining = wait_seconds - (time.monotonic() - started)
                update(
                    "WAITING_FOR_GPU", memory_used_MiB=None,
                    last_telemetry_error=f"{type(error).__name__}: {error}",
                    telemetry_error_count=record["telemetry_error_count"] + 1,
                )
                if remaining > 0:
                    time.sleep(min(poll_seconds, remaining))
                continue
            record["memory_used_MiB"] = used
            record.pop("last_telemetry_error", None)
            remaining = wait_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            if used < FREE_BELOW_MIB:
                executable = str(ROOT / ".venv/bin/python")
                arguments = [
                    executable, "-u", "-m", "src.agents.common_validity_penalty",
                    "run", "--directory", str(base),
                ]
                update("STARTING", executable=executable, arguments=arguments)
                os.chdir(ROOT)
                os.execve(executable, arguments, os.environ.copy())
                raise RuntimeError("execve unexpectedly returned")
            update("WAITING_FOR_GPU")
            time.sleep(min(poll_seconds, remaining))
        update("WAIT_TIMEOUT")
        return 2
    except Exception as error:
        update("FAILED", error=f"{type(error).__name__}: {error}")
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--wait-seconds", type=int, default=WAIT_SECONDS)
    args = parser.parse_args(argv)
    return queue(args.directory, wait_seconds=args.wait_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
