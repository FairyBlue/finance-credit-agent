#!/usr/bin/env python3
"""Start the frozen fresh-sampling probe after its training suite releases GPU."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
OUTPUT = ROOT / "runs/m2-rloo-local-credit-fresh-eval-20261026-v1"


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def record(status, **values):
    OUTPUT.mkdir(exist_ok=True)
    value = dict(status=status, training_suite=str(LOCAL), **values)
    temporary = OUTPUT / "continuation_status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUTPUT / "continuation_status.json")
    print(json.dumps(value), flush=True)


def main():
    deadline = time.monotonic() + 3600
    record("WAITING_FOR_TRAINING_AND_GPU_RELEASE")
    while time.monotonic() < deadline:
        state = read(LOCAL / "suite_state.json")
        if state.get("status") in {"FAILED", "PAUSED"}:
            record("STOPPED_TRAINING_NOT_COMPLETE", training_status=state["status"])
            return 1
        completion = read(LOCAL / "suite_completion.json")
        if state.get("status") == completion.get("status") == "COMPLETE":
            if completion.get("completed_new_updates") != 32:
                record("STOPPED_UPDATE_COUNT_MISMATCH")
                return 1
            holder = read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")
            status = read(Path(holder["directory"]) / "status.json") if holder else {}
            if status.get("status") == "HOLDING" and status.get("claim") is None:
                record("STARTING_FRESH_EVALUATION", completed_new_updates=32)
                script = ROOT / "scripts/diagnostics/run_rloo_fresh_eval.py"
                os.execv(sys.executable, [sys.executable, "-u", str(script)])
        time.sleep(5)
    record("STOPPED_WAIT_TIMEOUT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
