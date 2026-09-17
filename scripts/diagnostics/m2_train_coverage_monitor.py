#!/usr/bin/env python3
"""Refresh CPU evidence for the fixed coverage run; never restart its service."""

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
DEFAULT = ROOT / "runs/m2-train-coverage-20260915"
READER = ROOT / "scripts/diagnostics/m2_train_coverage_readout.py"
UNIT = "m2-train-coverage-20260915.service"
INTERVAL_SECONDS = 45
TIMEOUT_SECONDS = 7 * 3600
EXPECTED_TRAJECTORIES = 308


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def save(base, record):
    target = base / "monitor_state.json"
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def refresh(base):
    return subprocess.run(
        ["/usr/bin/python3", str(READER), "--directory", str(base), "--write"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )


def service_status(unit):
    result = subprocess.run(
        [
            "systemctl", "--user", "show", unit,
            "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def record_refresh(record, result, prefix="readout"):
    record.update({
        f"{prefix}_returncode": result.returncode,
        f"{prefix}_output": result.stdout[-2000:],
        f"{prefix}_error": result.stderr[-2000:],
    })


def completion_problems(base, service, readout_returncode):
    state = json.loads((base / "state.json").read_text())
    outcome = json.loads((base / "readout.json").read_text())
    totals = outcome.get("totals", {})
    requirements = {
        "final readout command failed": readout_returncode == 0,
        "main service is not inactive": service.get("ActiveState") == "inactive",
        "main service Result is not success": service.get("Result") == "success",
        "main service exit status is not zero": service.get("ExecMainStatus") == "0",
        "controller is not COMPLETE": state.get("status") == "COMPLETE",
        "readout execution is not COMPLETE": outcome.get("execution_status") == "COMPLETE",
        "readout evidence is not COMPLETE": outcome.get("status") == "COMPLETE",
        "trajectory counts differ from the fixed 308 complete grid": (
            all(totals.get(key) == EXPECTED_TRAJECTORIES for key in (
                "planned", "observed_unique_expected", "terminal_complete",
            ))
            and totals.get("incomplete_observed") == 0 and totals.get("missing") == 0
        ),
        "readout problems are missing or nonempty": outcome.get("problems") == [],
        "readout pending items are missing or nonempty": outcome.get("pending") == [],
        "FINAL_RESULTS.md is missing": (base / "FINAL_RESULTS.md").is_file(),
    }
    return [problem for problem, passed in requirements.items() if not passed]


def monitor(base, unit):
    base = Path(base).resolve()
    if not base.is_dir():
        raise ValueError(f"prepared run directory does not exist: {base}")
    started, started_at = time.monotonic(), now()
    record = {}
    while time.monotonic() - started < TIMEOUT_SECONDS:
        record = dict(
            schema_version=1, status="MONITORING", at_SGT=now(),
            started_at_SGT=started_at, directory=str(base), unit=unit,
            interval_seconds=INTERVAL_SECONDS, timeout_seconds=TIMEOUT_SECONDS,
            expected_trajectories=EXPECTED_TRAJECTORIES,
        )
        try:
            result = refresh(base)
            record_refresh(record, result)
            service = service_status(unit)
            record["service"] = service
            if service.get("ActiveState") in ("inactive", "failed"):
                # The controller may have finished after the preceding readout snapshot.
                final = refresh(base)
                record_refresh(record, final, "final_readout")
                problems = completion_problems(base, service, final.returncode)
                record.update(
                    status="STOPPED_EARLY" if problems else "COMPLETE",
                    completion_problems=problems, finished_at_SGT=now(),
                )
                save(base, record)
                return 2 if problems else 0
            if service.get("ActiveState") not in (
                "active", "activating", "deactivating", "reloading",
            ):
                raise RuntimeError(f"unexpected main service state: {service}")
            if result.returncode:
                raise RuntimeError(f"readout command failed with exit status {result.returncode}")
        except Exception as error:
            record.update(status="STOPPED_EARLY", error=repr(error), finished_at_SGT=now())
            save(base, record)
            return 2
        record["at_SGT"] = now()
        save(base, record)
        remaining = TIMEOUT_SECONDS - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(min(INTERVAL_SECONDS, remaining))
    record.update(
        status="STOPPED_EARLY", error="monitor reached its fixed seven-hour timeout",
        finished_at_SGT=now(),
    )
    save(base, record)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--unit", default=UNIT)
    args = parser.parse_args(argv)
    return monitor(args.directory.expanduser(), args.unit)


if __name__ == "__main__":
    raise SystemExit(main())
