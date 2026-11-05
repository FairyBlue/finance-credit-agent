#!/usr/bin/env python3
"""Reverify recorded actions without generating or replacing model responses."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from src.agents.mandate_repair_env import MandateRepairEnv, canonical


def replay(directory):
    directory = Path(directory).resolve()
    receipt = directory / "verification_replay.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    summary = json.loads((directory / "summary.json").read_text())
    tasks = {
        t["task_id"]: t
        for t in map(json.loads, Path(summary["data_path"]).read_text().splitlines())
    }
    requests = defaultdict(list)
    for row in map(json.loads, (directory / "requests.jsonl").read_text().splitlines()):
        requests[row["trajectory_key"]].append(row)
    raw = [
        json.loads(line)
        for line in (directory / "trajectory_results.jsonl").read_text().splitlines()
    ]
    results, changes = [], []
    for previous in raw:
        env = MandateRepairEnv(tasks[previous["task_id"]])
        for request in sorted(
            requests[previous["trajectory_key"]], key=lambda row: row["call_index"]
        ):
            if request.get("tool_result") is not None and not env.done:
                env.step(request["response_text"])
        current = dict(previous, **env.result())
        # Keep genuine generation/environment failures; replay does not create
        # an observation or substitute a completion for a missing request.
        current["status"] = previous["status"]
        current["environment_status"] = env.result()["status"]
        current["terminal_reward"] = (
            env.result()["reward"] if previous["collector_failure"] is None else None
        )
        if previous["collector_failure"] is not None:
            current.update(success=False, reward=None)
        if current["success"] != previous["success"]:
            changes.append(
                dict(
                    trajectory_key=previous["trajectory_key"],
                    old_success=previous["success"],
                    new_success=current["success"],
                )
            )
        results.append(current)
    backup = directory / "before_certificate_validation"
    backup.mkdir()
    for name in ("trajectory_results.jsonl", "summary.json", "collection_summary.json"):
        shutil.copy2(directory / name, backup / name)
    (directory / "trajectory_results.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in results)
    )
    summary["results"] = results
    summary["successes"] = sum(r["success"] for r in results)
    summary["success_rate_all_planned"] = summary["successes"] / summary["planned_trajectories"]
    summary["verification_protocol"] = "explicit-infeasibility-certificate-v2"
    for name in ("summary.json", "collection_summary.json"):
        (directory / name).write_text(json.dumps(summary, indent=2) + "\n")
    value = dict(
        status="COMPLETE",
        method="replay_exact_recorded_actions_no_generation",
        trajectories=len(results),
        generated_tokens=0,
        optimizer_updates=0,
        outcome_changes=changes,
        raw_backup=str(backup),
    )
    receipt.write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(canonical(replay(parser.parse_args().directory)))


if __name__ == "__main__":
    main()
