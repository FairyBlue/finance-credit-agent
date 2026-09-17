#!/usr/bin/env python3
"""Refresh CPU evidence for the fixed shared-validity-penalty run; never restart its service."""

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
READER = ROOT / "scripts/diagnostics/m2_common_validity_penalty_readout.py"
UNIT = "m2-common-validity-penalty-20260917.service"
INTERVAL_SECONDS = 45
TIMEOUT_SECONDS = 17 * 3600
EXPECTED_TRAJECTORIES = 164


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def save(base, record, name="monitor_state.json"):
    target = base / name
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
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def record_refresh(record, result, prefix="readout"):
    record.update(
        {
            f"{prefix}_returncode": result.returncode,
            f"{prefix}_output": result.stdout[-2000:],
            f"{prefix}_error": result.stderr[-2000:],
        }
    )


def completion_problems(base, service, readout_returncode):
    state = json.loads((base / "state.json").read_text()) if (base / "state.json").is_file() else {}
    outcome = (
        json.loads((base / "readout.json").read_text()) if (base / "readout.json").is_file() else {}
    )
    totals = outcome.get("totals", {})
    requirements = {
        "final readout command failed": readout_returncode == 0,
        "main service is not inactive": service.get("ActiveState") == "inactive",
        "main service Result is not success": service.get("Result") == "success",
        "main service exit status is not zero": service.get("ExecMainStatus") == "0",
        "controller is not COMPLETE": state.get("status") == "COMPLETE",
        "readout execution is not COMPLETE": outcome.get("execution_status") == "COMPLETE",
        "readout evidence is not COMPLETE": outcome.get("status") == "COMPLETE",
        "trajectory counts differ from the fixed 164 complete grid": (
            all(
                totals.get(key) == EXPECTED_TRAJECTORIES
                for key in (
                    "planned",
                    "observed_unique_expected",
                    "terminal_complete",
                )
            )
            and totals.get("incomplete_observed") == 0
            and totals.get("missing") == 0
        ),
        "readout problems are missing or nonempty": outcome.get("problems") == [],
        "readout pending items are missing or nonempty": outcome.get("pending") == [],
        "common initial critic heads were not verified": outcome.get(
            "initial_critic_heads", {}
        ).get("all_equal")
        is True,
        "common training reward evidence was not complete": all(
            outcome.get("cells", {}).get(name, {}).get("common_validity_penalty", {}).get("status")
            == "COMPLETE"
            for name in ("sft0", "m1", "m2_eta025")
        ),
        "FINAL_RESULTS.md is missing": (base / "FINAL_RESULTS.md").is_file(),
    }
    return [problem for problem, passed in requirements.items() if not passed]


def finish(base, record):
    save(base, record)
    save(
        base,
        {
            "schema_version": 1,
            "status": "PASS" if record["status"] == "COMPLETE" else "INCOMPLETE",
            "checked_at_SGT": now(),
            "service": record.get("service", {}),
            "fixed_trajectory_budget": EXPECTED_TRAJECTORIES,
            "final_readout_returncode": record.get("final_readout_returncode"),
            "problems": record.get("completion_problems", [record.get("error", "monitor stopped")]),
            "scope": "Main service exit and final CPU evidence refresh; no automatic restart.",
        },
        "completion_validation.json",
    )


def monitor(base, unit):
    base = Path(base).resolve()
    if not base.is_dir():
        raise ValueError(f"prepared run directory does not exist: {base}")
    started, started_at = time.monotonic(), now()
    record = {}
    while time.monotonic() - started < TIMEOUT_SECONDS:
        record = dict(
            schema_version=1,
            status="MONITORING",
            at_SGT=now(),
            started_at_SGT=started_at,
            directory=str(base),
            unit=unit,
            interval_seconds=INTERVAL_SECONDS,
            timeout_seconds=TIMEOUT_SECONDS,
            expected_trajectories=EXPECTED_TRAJECTORIES,
        )
        try:
            result = refresh(base)
            record_refresh(record, result)
            queue_path = base / "gpu_queue_state.json"
            queue = json.loads(queue_path.read_text()) if queue_path.is_file() else {}
            record["gpu_queue"] = queue
            if not (base / "state.json").is_file() and queue.get("status") in {
                "WAITING_FOR_GPU",
                "STARTING",
            }:
                record["status"] = queue["status"]
                record["training_has_started"] = False
            else:
                record["training_has_started"] = (base / "state.json").is_file()
            service = service_status(unit)
            record["service"] = service
            if service.get("ActiveState") in ("inactive", "failed"):
                # The controller may have finished after the preceding readout snapshot.
                final = refresh(base)
                record_refresh(record, final, "final_readout")
                problems = completion_problems(base, service, final.returncode)
                record.update(
                    status="STOPPED_EARLY" if problems else "COMPLETE",
                    completion_problems=problems,
                    finished_at_SGT=now(),
                )
                finish(base, record)
                return 2 if problems else 0
            if service.get("ActiveState") not in (
                "active",
                "activating",
                "deactivating",
                "reloading",
            ):
                raise RuntimeError(f"unexpected main service state: {service}")
            if result.returncode:
                raise RuntimeError(f"readout command failed with exit status {result.returncode}")
        except Exception as error:
            record.update(status="STOPPED_EARLY", error=repr(error), finished_at_SGT=now())
            finish(base, record)
            return 2
        record["at_SGT"] = now()
        save(base, record)
        remaining = TIMEOUT_SECONDS - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(min(INTERVAL_SECONDS, remaining))
    record.update(
        status="STOPPED_EARLY",
        error="monitor reached its fixed seventeen-hour timeout (including queue)",
        finished_at_SGT=now(),
    )
    finish(base, record)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--unit", default=UNIT)
    args = parser.parse_args(argv)
    return monitor(args.directory.expanduser(), args.unit)


if __name__ == "__main__":
    raise SystemExit(main())
