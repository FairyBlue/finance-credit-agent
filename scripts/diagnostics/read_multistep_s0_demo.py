#!/usr/bin/env python3
"""Describe parent-request S0 updates and actual parent evaluations, without replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-s0-multistep-demo-20261024-v1"
HELPER = Path(__file__).with_name("read_same_prefix_online_8window.py")
spec = importlib.util.spec_from_file_location("multistep_eight_window_stats", HELPER)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def parent_activity(path):
    """Count applied parent requests and fills, never head/EST costs."""
    observed = {}
    if path is not None and (file := Path(path) / "trajectory_results.jsonl").exists():
        with file.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("role") != "parent":
                    continue
                eid = row["episode_id"]
                if eid not in stats.EPISODES or eid in observed:
                    raise ValueError("duplicate or unexpected parent activity")
                steps = row["timesteps"]
                fills = [
                    fill for step in steps for fill in (step.get("info", {}).get("fills") or [])
                ]
                costs = [fill.get("commission") for fill in fills] + [
                    fill.get("slippage_cost") for fill in fills
                ]
                invalid = sum(step.get("invalid") is True for step in steps)
                observed[eid] = dict(
                    status=row["status"],
                    applied_parent_requests=len(steps),
                    invalid_parent_requests=invalid,
                    parent_rejection_rate=invalid / len(steps) if steps else None,
                    advancing_parent_requests=sum(step.get("advancing") is True for step in steps),
                    fill_count=len(fills),
                    transaction_cost=math.fsum(costs)
                    if all(stats.finite(value) for value in costs)
                    else None,
                    action_family_counts=dict(Counter(step.get("action_family") for step in steps)),
                )
                if len(observed) == 8:
                    break
    windows = {eid: observed.get(eid) for eid in stats.EPISODES}
    strata = {}
    for name, ids in stats.STRATA.items():
        selected = [windows[eid] for eid in ids]
        complete = all(item is not None and item["status"] == "COMPLETE" for item in selected)
        requests = sum(item["applied_parent_requests"] for item in selected) if complete else None
        invalid = sum(item["invalid_parent_requests"] for item in selected) if complete else None
        costs = [item["transaction_cost"] for item in selected] if complete else []
        strata[name] = dict(
            status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
            observed_parent_trajectories=sum(item is not None for item in selected),
            planned_parent_trajectories=len(ids),
            applied_parent_requests=requests,
            invalid_parent_requests=invalid,
            parent_rejection_rate=invalid / requests if requests else None,
            transaction_cost_sum=math.fsum(costs)
            if costs and all(stats.finite(value) for value in costs)
            else None,
            transaction_cost_mean=stats.mean(costs) if costs else None,
        )
    return dict(
        source_directory=str(path) if path is not None else None, windows=windows, strata=strata
    )


def training_metrics(training):
    if training is None:
        return dict(
            status="PENDING",
            training_rows=None,
            invalid_rows=None,
            invalid_row_fraction=None,
            actor_updates=None,
            critic_updates=None,
        )
    rows, invalid = training.get("training_rows"), training.get("invalid_rows")
    return dict(
        status=training.get("status"),
        training_rows=rows,
        invalid_rows=invalid,
        invalid_row_fraction=invalid / rows if stats.finite(invalid) and rows else None,
        actor_updates=training.get("actor_updates"),
        critic_updates=training.get("critic_updates"),
        planned_trajectories=training.get("planned_trajectories"),
        complete_trajectories=training.get("complete_trajectories"),
        missing_or_incomplete_trajectories=training.get("missing_or_incomplete_trajectories"),
        critic_mse_before=training.get("critic_mse_before"),
        critic_mse_after=training.get("critic_mse_after"),
        critic_fit={
            key: value
            for key, value in training.get("critic", {}).items()
            if "MSE" in key or key == "loss"
        },
    )


def summarize(directory=DEFAULT):
    directory = Path(directory).resolve()
    source_path = directory / "demo_results.json"
    payload = source_path.read_bytes() if source_path.exists() else None
    source = json.loads(payload) if payload is not None else {}
    if source and (source.get("planned_rounds") != 2 or source.get("seed") != 42):
        raise ValueError("reader expects the fixed seed42 two-update multistep demo")
    entries = source.get("rounds", [])
    indexed = {entry["round"]: entry for entry in entries}
    if len(entries) != len(indexed) or any(n not in (1, 2) for n in indexed):
        raise ValueError("unexpected or duplicate round")
    baseline = stats.evaluate(source.get("baseline"))
    curve, csv_rows = [], []
    for number in (0, 1, 2):
        entry = indexed.get(number, {})
        summary = source.get("baseline") if number == 0 else entry.get("evaluation")
        evaluation = baseline if number == 0 else stats.evaluate(summary)
        difference = stats.compare(evaluation, baseline)
        activity = parent_activity(
            (source.get("initial_collection_path") if number == 0 else entry.get("collection_path"))
            if summary is not None
            else None
        )
        training = training_metrics(entry.get("training")) if number else None
        point = dict(
            round=number,
            execution_status="REUSED_BASELINE" if number == 0 else entry.get("status", "PENDING"),
            evaluation=evaluation,
            versus_common_SFT=difference,
            parent_activity=activity,
            training=training,
            error=entry.get("error"),
        )
        curve.append(point)
        for window, delta in zip(evaluation["windows"], difference["windows"], strict=True):
            actual = activity["windows"][window["episode_id"]] or {}
            csv_rows.append(
                dict(
                    round=number,
                    episode_id=window["episode_id"],
                    stratum="common4" if window["episode_id"] in stats.COMMON else "added4",
                    evaluation_status=window["status"],
                    net_return=window["net_return"],
                    net_return_pct=window["net_return_pct"],
                    Y=window["Y"],
                    net_return_delta_pp=delta["net_return_delta_pp"],
                    Y_delta=delta["Y_delta"],
                    applied_parent_requests=actual.get("applied_parent_requests"),
                    invalid_parent_requests=actual.get("invalid_parent_requests"),
                    parent_rejection_rate=actual.get("parent_rejection_rate"),
                    transaction_cost=actual.get("transaction_cost"),
                    fill_count=actual.get("fill_count"),
                    action_family_counts=json.dumps(actual["action_family_counts"], sort_keys=True)
                    if "action_family_counts" in actual
                    else None,
                    training_rows=training.get("training_rows") if training else None,
                    training_invalid_rows=training.get("invalid_rows") if training else None,
                )
            )
    complete = (
        source.get("status") == "COMPLETE"
        and len(indexed) == 2
        and all(p["evaluation"]["strata"]["all8"]["status"] == "OBSERVED" for p in curve)
        and all(p["training"]["status"] == "COMPLETE" for p in curve[1:])
    )
    report = dict(
        profile="multistep_parent_request_s0_train_demo_description_v1",
        status="DESCRIBED" if complete else ("PARTIAL" if source else "PENDING"),
        execution_status=source.get("status", "PENDING"),
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        source_results=str(source_path),
        source_results_sha256=hashlib.sha256(payload).hexdigest() if payload is not None else None,
        episode_ids=stats.EPISODES,
        strata=stats.STRATA,
        curve=curve,
        round2_versus_common_SFT=curve[2]["versus_common_SFT"],
        baseline_reused_source_model_requests=source.get("baseline_reused_source_model_requests"),
        baseline_reused_parent_requests=source.get("baseline_reused_parent_requests"),
        baseline_reused_parent_invalid_requests=source.get(
            "baseline_reused_parent_invalid_requests"
        ),
        baseline_new_model_requests=source.get("baseline_new_model_requests"),
        limits=[
            "Eight reused Train windows, seed42, two full-parent-request updates; no Valid/OOS.",
            "Only role=parent rows train; the source's head/EST requests do not train.",
            "All eight-window means retain their planned denominator; missing values remain NA.",
            "The fixed second round and full curve are reported, without best selection.",
            "Rejection rates use applied parent requests; fees sum actual parent fill costs.",
            "common4/added4 are reused ID strata, not independent new-date evidence.",
            "Training mode changes; old head-only means are not a strict single-factor control.",
        ],
    )
    return report, csv_rows


def write_readout(directory=DEFAULT):
    directory = Path(directory).resolve()
    report, records = summarize(directory)
    paths = [directory / name for name in ("multistep_readout.json", "window_results.csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError("multistep readout exists; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    with paths[1].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report["csv_sha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    with paths[0].open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    report = summarize(args.directory)[0] if args.preview else write_readout(args.directory)
    print(
        json.dumps(
            dict(
                status=report["status"],
                execution_status=report["execution_status"],
                round2=report["round2_versus_common_SFT"]["strata"],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
