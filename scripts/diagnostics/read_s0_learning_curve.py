#!/usr/bin/env python3
"""Describe the fixed S0 Train learning curve without selecting a best round."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-s0-learning-curve-20261024-v1"
HELPER = Path(__file__).with_name("read_same_prefix_online_8window.py")
spec = importlib.util.spec_from_file_location("s0_curve_eight_window_stats", HELPER)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
PLANNED_ROUNDS = 8


def training_metrics(training, round_index):
    coverage = training.get("coverage", {}) if training else {}
    groups = coverage.get("groups")
    critic = training.get("critic", {}) if training else {}
    before, after = (critic.get(name) for name in ("before_full_head_MSE", "after_full_head_MSE"))
    return dict(
        status=training.get("status", "UNKNOWN")
        if training
        else ("NOT_APPLICABLE" if round_index == 0 else "PENDING"),
        supported_heads=training.get("supported_heads") if training else None,
        supported_groups=training.get("supported_groups") if training else None,
        planned_heads=coverage.get("planned_heads"),
        NA_heads=coverage.get("NA_heads"),
        order_gate_open_groups=sum(g.get("order_gate") is True for g in groups)
        if groups is not None
        else None,
        order_gate_closed_groups=sum(g.get("order_gate") is False for g in groups)
        if groups is not None
        else None,
        groups_without_gate=sum(type(g.get("order_gate")) is not bool for g in groups)
        if groups is not None
        else None,
        actor_updates=training.get("actor_updates") if training else None,
        critic_updates=training.get("critic_updates") if training else None,
        critic_before_full_head_MSE=before,
        critic_after_full_head_MSE=after,
        critic_MSE_delta=stats.subtract(after, before),
    )


def summarize(directory=DEFAULT):
    directory = Path(directory).resolve()
    path = directory / "curve_results.json"
    payload = path.read_bytes() if path.exists() else None
    source = json.loads(payload) if payload is not None else {}
    if source and source.get("planned_rounds") != PLANNED_ROUNDS:
        raise ValueError("this reader describes the fixed eight-update S0 curve")
    entries = source.get("rounds", [])
    indexed = {entry["round"]: entry for entry in entries}
    if len(indexed) != len(entries) or any(type(n) is not int or not 1 <= n <= 8 for n in indexed):
        raise ValueError("duplicate or unexpected learning-curve round")
    baseline = stats.evaluate(source.get("baseline"))
    curve, csv_rows = [], []
    for round_index in range(PLANNED_ROUNDS + 1):
        entry = indexed.get(round_index, {})
        evaluation = baseline if round_index == 0 else stats.evaluate(entry.get("evaluation"))
        difference = stats.compare(evaluation, baseline)
        training = training_metrics(entry.get("training"), round_index)
        point = dict(
            round=round_index,
            execution_status="BASELINE" if round_index == 0 else entry.get("status", "PENDING"),
            evaluation=evaluation,
            versus_common_SFT=difference,
            training=training,
            actor_checkpoint=entry.get("actor_checkpoint"),
            error=entry.get("error"),
        )
        curve.append(point)
        for window, delta, base in zip(
            evaluation["windows"], difference["windows"], baseline["windows"], strict=True
        ):
            csv_rows.append(
                dict(
                    round=round_index,
                    episode_id=window["episode_id"],
                    stratum="common4" if window["episode_id"] in stats.COMMON else "added4",
                    execution_status=point["execution_status"],
                    evaluation_status=window["status"],
                    net_return=window["net_return"],
                    net_return_pct=window["net_return_pct"],
                    Y=window["Y"],
                    baseline_net_return=base["net_return"],
                    baseline_Y=base["Y"],
                    net_return_delta_pp=delta["net_return_delta_pp"],
                    Y_delta=delta["Y_delta"],
                    **{f"training_{k}": v for k, v in training.items()},
                )
            )
    complete = (
        source.get("status") == "COMPLETE"
        and len(entries) == 8
        and all(p["evaluation"]["strata"]["all8"]["status"] == "OBSERVED" for p in curve)
        and all(p["training"]["status"] == "COMPLETE" for p in curve[1:])
    )
    source_status = source.get("status", "PENDING")
    status = "DESCRIBED" if complete else ("PENDING" if not source else "PARTIAL")
    report = dict(
        profile="fixed_s0_eight_round_train_curve_description_v1",
        status=status,
        execution_status=source_status,
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        source_results=str(path),
        source_results_sha256=hashlib.sha256(payload).hexdigest() if payload is not None else None,
        statistical_helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        planned_rounds=PLANNED_ROUNDS,
        planned_windows=8,
        episode_ids=stats.EPISODES,
        strata=stats.STRATA,
        initial_collection_path=source.get("initial_collection_path"),
        baseline_reused=source.get("baseline_reused"),
        baseline_reused_model_requests=source.get("baseline_reused_model_requests"),
        observed_evaluation_rounds=sum(
            p["evaluation"]["strata"]["all8"]["status"] == "OBSERVED" for p in curve[1:]
        ),
        completed_training_rounds=sum(p["training"]["status"] == "COMPLETE" for p in curve[1:]),
        curve=curve,
        round8_versus_common_SFT=curve[8]["versus_common_SFT"],
        round8_evaluation=curve[8]["evaluation"],
        limits=[
            "Fixed round 8 and the entire 0..8 curve are reported; no best-round selection.",
            "Eight reused Train windows, seed42; not Valid/OOS, new dates, or a method comparison.",
            "Fresh sampled evaluations are descriptive, without a monotonic-return guarantee.",
            "common4/added4 are predeclared ID strata, not independent market samples.",
            "Missing points stay NA; means retain the full planned stratum denominator.",
            "Critic MSE is its recorded current training-batch fit before/after one update.",
            "Fresh targets differ by round; MSE levels are not a fixed validation curve.",
        ],
    )
    return report, csv_rows


def write_readout(directory=DEFAULT):
    directory = Path(directory).resolve()
    report, records = summarize(directory)
    paths = [directory / name for name in ("learning_curve_readout.json", "window_curve.csv")]
    if any(path.exists() for path in paths):
        raise FileExistsError("learning-curve readout exists; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    with paths[1].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report["window_curve_csv_sha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
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
                completed_training_rounds=report["completed_training_rounds"],
                observed_evaluation_rounds=report["observed_evaluation_rounds"],
                round8=report["round8_versus_common_SFT"]["strata"],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
