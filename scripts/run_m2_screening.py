#!/usr/bin/env python3
"""Durable serial M2 screening supervisor; run as a lingering systemd user service."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.screening_archive import archive_completed_cell  # noqa: E402
from src.agents.screening_launcher import ARMS, atomic_json, utc_now  # noqa: E402
from src.evaluation.screening import build_readout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "artifacts/m2_screening")
    parser.add_argument("--settings", type=Path, default=ROOT / "configs/m2_screening.yaml")
    parser.add_argument("--stage", choices=("wiring", "pilot", "screening", "all"), default="all")
    parser.add_argument("--only-arm", choices=ARMS)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "supervisor.lock"
    import fcntl

    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {
            "status": "RUNNING", "pid": os.getpid(), "started_at": utc_now(),
            "expected_commit": args.expected_commit, "requested_stage": args.stage,
            "ssh_independent": True, "cells": [], "final_oos_accessed": False,
        }
        atomic_json(root / "supervisor.json", status)
        queue: list[tuple[str, str, int]] = []
        if args.stage in {"all", "wiring"}:
            queue.extend(("wiring", arm, 42) for arm in ARMS)
        if args.stage in {"all", "pilot"}:
            queue.extend(("pilot", arm, 42) for arm in ("M1", "M2"))
        if args.stage in {"all", "screening"}:
            queue.extend(
                ("screening", arm, seed)
                for seed, arms in ((42, ARMS), (43, tuple(reversed(ARMS)))) for arm in arms
            )
            queue.append(("zero-shot", "A2", 42))
        if args.only_arm:
            queue = [cell for cell in queue if cell[1] == args.only_arm]
        try:
            for stage, arm, seed in queue:
                name = f"{stage}-{arm.lower()}-s{seed}"
                cell = root / name
                status["current_cell"] = name
                status["updated_at"] = utc_now()
                atomic_json(root / "supervisor.json", status)
                if cell.exists():
                    previous = json.loads((cell / "summary.json").read_text())
                    if previous.get("expected_commit") != args.expected_commit:
                        raise RuntimeError(f"refusing to mix code identities: {cell}")
                    if previous["status"] == "PASSED":
                        archive_completed_cell(cell, stage)
                        print(f"Preserving completed cell: {name}", flush=True)
                        status["cells"].append({"cell": name, "status": "PASSED"})
                        continue
                    raise RuntimeError(f"existing incomplete cell needs explicit review: {cell}")
                if shutil.disk_usage(root).free < 112 * 1024**3:
                    raise RuntimeError("less than 112 GiB free for checkpoints and actor export")
                command = [
                    sys.executable, "-m", "src.agents.screening_launcher",
                    "--cell-dir", str(cell), "--stage", stage, "--arm", arm,
                    "--seed", str(seed), "--expected-commit", args.expected_commit,
                    "--prepared-dir", str(args.prepared_dir.resolve()),
                    "--settings", str(args.settings.resolve()),
                ]
                print(f"{utc_now()} Starting {name}", flush=True)
                with (root / f"{name}.launcher.log").open("a") as logfile:
                    outcome = subprocess.run(
                        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=logfile,
                        stderr=subprocess.STDOUT, check=False,
                    )
                state = "PASSED" if outcome.returncode == 0 else "FAILED"
                status["cells"].append({
                    "cell": name, "status": state, "exit_code": outcome.returncode,
                })
                atomic_json(root / "supervisor.json", status)
                print(f"{utc_now()} {name}: {state}", flush=True)
                if outcome.returncode:
                    # Do not let unknown worker/infrastructure failures affect subsequent cells.
                    # Preserve all successes; agent/operator resolves this before advancing.
                    raise RuntimeError(f"{name} failed; inspect durable launcher/trainer logs")
                status["current_phase"] = "archive"
                atomic_json(root / "supervisor.json", status)
                receipt = archive_completed_cell(cell, stage)
                print(
                    f"{utc_now()} {name}: archived, reclaimed {receipt['reclaimed_bytes']} bytes",
                    flush=True,
                )
                status["current_phase"] = "training"
                time.sleep(2)
            status["status"] = "PASSED"
            if args.stage in {"all", "screening"} and not args.only_arm:
                from src.data.screening import evaluate_screening_baselines

                baselines = evaluate_screening_baselines(args.prepared_dir, repo_root=ROOT)
                atomic_json(root / "baselines.json", baselines)
                report = build_readout(root)
                status["screening_verdict"] = report["verdict"]
            return 0
        except BaseException as error:
            status.update(status="STOPPED", error=f"{type(error).__name__}: {error}")
            if args.stage in {"all", "screening"}:
                build_readout(root)
            raise
        finally:
            status["updated_at"] = utc_now()
            atomic_json(root / "supervisor.json", status)


if __name__ == "__main__":
    raise SystemExit(main())
