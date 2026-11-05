#!/usr/bin/env python3
"""Describe the existing eight-window online demo; no model or replay imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "runs/m2-same-prefix-online-8window-demo-20261024-v1"
EPISODES = tuple(f"formal-train-{i:04d}" for i in (1, 9, 17, 25, 65, 73, 149, 157))
COMMON = tuple(f"formal-train-{i:04d}" for i in (1, 9, 65, 73))
STRATA = {
    "all8": EPISODES,
    "common4": COMMON,
    "added4": tuple(e for e in EPISODES if e not in COMMON),
}
ARMS = ("zero", "plain", "exact")
STAGES = {"baseline": "baseline", "r1": "after_round1", "r2": "after_round2"}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values and all(finite(v) for v in values) else None


def subtract(left, right, scale=1):
    return scale * (left - right) if finite(left) and finite(right) else None


def evaluate(summary):
    windows = {
        e: dict(episode_id=e, net_return=None, net_return_pct=None, Y=None, status="PENDING")
        for e in EPISODES
    }
    if summary is not None:
        ids, ys, metrics = (
            summary[k] for k in ("episode_ids", "parent_terminal_Y", "parent_terminal_metrics")
        )
        if len(ids) != 8 or set(ids) != set(EPISODES) or len(ys) != 8 or len(metrics) != 8:
            raise ValueError("eight-window evaluation identity differs")
        for eid, y, metric in zip(ids, ys, metrics, strict=True):
            net = metric.get("raw_net_return") if metric is not None else None
            if y is not None and not finite(y) or net is not None and not finite(net):
                raise ValueError("nonfinite observed evaluation")
            windows[eid].update(
                net_return=net,
                net_return_pct=100 * net if finite(net) else None,
                Y=y,
                status="OBSERVED" if finite(y) and finite(net) else "INCOMPLETE",
            )
    strata = {}
    for name, ids in STRATA.items():
        subset = [windows[e] for e in ids]
        complete = all(row["status"] == "OBSERVED" for row in subset)
        strata[name] = dict(
            status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
            observed_windows=sum(row["status"] == "OBSERVED" for row in subset),
            planned_windows=len(ids),
            mean_net_return=mean([r["net_return"] for r in subset]),
            mean_net_return_pct=mean([r["net_return_pct"] for r in subset]),
            mean_Y=mean([r["Y"] for r in subset]),
        )
    return dict(
        strata=strata,
        windows=[windows[e] for e in EPISODES],
        collection_invalid_requests=summary.get("invalid_requests") if summary else None,
        parent_failures=summary.get("parents_failed") if summary else None,
    )


def compare(left, right):
    windows = []
    for a, b in zip(left["windows"], right["windows"], strict=True):
        if a["episode_id"] != b["episode_id"]:
            raise ValueError("unpaired episode")
        pp = subtract(a["net_return"], b["net_return"], 100)
        windows.append(
            dict(
                episode_id=a["episode_id"],
                net_return_delta_pp=pp,
                Y_delta=subtract(a["Y"], b["Y"]),
                status="OBSERVED" if pp is not None else "PENDING_OR_INCOMPLETE",
            )
        )
    strata = {}
    for name, ids in STRATA.items():
        selected = [w for w in windows if w["episode_id"] in ids]
        values = [w["net_return_delta_pp"] for w in selected]
        complete = all(finite(v) for v in values)
        observed = [v for v in values if finite(v)]
        strata[name] = dict(
            status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
            planned_windows=len(ids),
            paired_windows=len(observed),
            net_return_delta_pp=mean(values),
            Y_delta=mean([w["Y_delta"] for w in selected]),
            positive_windows=sum(v > 1e-12 for v in observed),
            negative_windows=sum(v < -1e-12 for v in observed),
            tied_windows=sum(abs(v) <= 1e-12 for v in observed),
        )
    return dict(strata=strata, windows=windows)


def training_counts(directory, arm, number, training, collection):
    path = directory / ("common_initial" if number == 1 else f"{arm}/round2/collect")
    heads_path = path / "head_results.jsonl"
    heads = (
        [json.loads(line) for line in heads_path.read_text().splitlines() if line.strip()]
        if collection is not None and heads_path.exists()
        else None
    )
    coverage = training.get("coverage", {}) if training else {}
    return dict(
        arm=arm,
        round=number,
        status=training.get("status") if training else "PENDING",
        planned_heads=32,
        supported_heads=training.get("supported_heads") if training else None,
        supported_groups=training.get("supported_groups") if training else None,
        NA_heads=coverage.get("NA_heads"),
        actor_updates=training.get("actor_updates") if training else None,
        critic_updates=training.get("critic_updates") if training else None,
        order_diverse_groups=sum(g.get("order_gate") is True for g in coverage["groups"])
        if "groups" in coverage
        else None,
        observed_head_records=len(heads) if heads is not None else None,
        legal_heads=sum(h["legal_head"] for h in heads) if heads is not None else None,
        rejected_heads=sum(
            any(s.get("invalid") is True for s in h.get("timesteps", [])) for h in heads
        )
        if heads is not None
        else None,
        nonadvancing_heads=sum(h.get("status") == "NONADVANCING" for h in heads)
        if heads is not None
        else None,
        collection_invalid_requests=collection.get("invalid_requests") if collection else None,
        head_source=str(heads_path),
    )


def summarize(directory=DEFAULT):
    directory = Path(directory).resolve()
    path = directory / "results.json"
    payload = path.read_bytes() if path.exists() else None
    source = json.loads(payload) if payload else {"results": []}
    by_arm = {row["arm"]: row for row in source["results"]}
    if len(by_arm) != len(source["results"]) or not set(by_arm) <= set(ARMS):
        raise ValueError("duplicate or unexpected arm")
    evaluations, rows, training = {}, [], []
    for arm in ARMS:
        result = by_arm.get(arm, {})
        evaluations[arm] = {}
        for stage, key in STAGES.items():
            panel = evaluate(result.get(key))
            evaluations[arm][stage] = panel
            rows.extend(
                dict(kind="evaluation_mean", arm=arm, stage=stage, stratum=name, **values)
                for name, values in panel["strata"].items()
            )
            rows.extend(
                dict(kind="window", arm=arm, stage=stage, **window) for window in panel["windows"]
            )
        for number, collection_key in ((1, "baseline"), (2, "after_round1")):
            counts = training_counts(
                directory,
                arm,
                number,
                result.get(f"round{number}_training"),
                result.get(collection_key),
            )
            training.append(counts)
            rows.append(dict(kind="training_counts", **counts))
    contrasts = {}
    for other in ("zero", "plain"):
        name = f"exact_minus_{other}"
        contrasts[name] = {}
        for stage in STAGES:
            difference = compare(evaluations["exact"][stage], evaluations[other][stage])
            contrasts[name][stage] = difference
            rows.extend(
                dict(kind="contrast_mean", arm=name, stage=stage, stratum=s, **v)
                for s, v in difference["strata"].items()
            )
            rows.extend(
                dict(kind="window_contrast", arm=name, stage=stage, **window)
                for window in difference["windows"]
            )
    complete = (
        len(by_arm) == 3
        and all(r["status"] == "COMPLETE" for r in by_arm.values())
        and all(
            p["strata"]["all8"]["status"] == "OBSERVED"
            for arm in evaluations.values()
            for p in arm.values()
        )
    )
    report = dict(
        profile="same_prefix_eight_window_train_demo_description_v1",
        status="DESCRIBED" if complete else "PENDING",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        source_results=str(path),
        source_results_sha256=hashlib.sha256(payload).hexdigest() if payload else None,
        episode_ids=EPISODES,
        strata=STRATA,
        observed_arms=len(by_arm),
        evaluations=evaluations,
        contrasts=contrasts,
        training=training,
        limits=[
            "Eight previously used Train windows; one seed and two head-update rounds.",
            "Within-run descriptions; no causal comparison with the old four-window run.",
            "common4/added4 are fixed ID strata, not independent or unseen date samples.",
            "Incomplete values stay NA; no reduced-denominator means or imputed zero.",
            "Head rejections use head timesteps; collection invalid counts cover all roles.",
            "Net-return deltas are percentage points; no significance or advantage gate.",
        ],
    )
    return report, rows


def write_readout(directory=DEFAULT):
    directory = Path(directory).resolve()
    report, records = summarize(directory)
    paths = [directory / name for name in ("window_results.csv", "eight_window_readout.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("eight-window readout exists; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in records for k in r))
        )
        writer.writeheader()
        writer.writerows(records)
    report["csv_sha256"] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    with paths[1].open("x") as stream:
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
                observed_arms=report["observed_arms"],
                r2={name: values["r2"]["strata"] for name, values in report["contrasts"].items()},
            )
        )
    )


if __name__ == "__main__":
    main()
