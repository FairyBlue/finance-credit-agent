#!/usr/bin/env python3
"""Describe two completed or pending Train demos from their existing results.

No models, environments, raw trajectories or source inventories are loaded.
Missing results stay pending/NA. Four reused windows and two seeds are not an
independent market sample, and the shared baseline is not counted three times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    42: ROOT / "runs/m2-same-prefix-online-demo-20261024-v2",
    43: ROOT / "runs/m2-same-prefix-online-demo-seed43-20261024-v1",
}
ARMS = ("zero", "plain", "exact")
STAGES = {"baseline": "baseline", "r1": "after_round1", "r2": "after_round2"}
EPISODES = tuple(f"formal-train-{i:04d}" for i in (1, 9, 65, 73))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values and all(finite(v) for v in values) else None


def subtract(left, right, scale=1.0):
    return scale * (left - right) if finite(left) and finite(right) else None


def panel(summary, seed):
    if summary is None:
        return dict(
            status="PENDING",
            complete_windows=0,
            total_windows=4,
            parents_failed=None,
            mean_net_return=None,
            mean_net_return_pct=None,
            mean_Y=None,
            windows=[
                dict(episode_id=e, status="PENDING", net_return=None, net_return_pct=None, Y=None)
                for e in EPISODES
            ],
        )
    if summary["seed"] != seed or set(summary["episode_ids"]) != set(EPISODES):
        raise ValueError("collection seed/Train-window identity mismatch")
    ids, ys, metrics = (
        summary[k] for k in ("episode_ids", "parent_terminal_Y", "parent_terminal_metrics")
    )
    if len(ids) != len(set(ids)) or len(ids) != len(ys) or len(ids) != len(metrics):
        raise ValueError("collection window arrays differ")
    windows = {}
    for eid, y, metric in zip(ids, ys, metrics, strict=True):
        net = metric.get("raw_net_return") if metric is not None else None
        if y is not None and not finite(y) or net is not None and not finite(net):
            raise ValueError("nonfinite observed evaluation")
        observed = finite(y) and finite(net)
        windows[eid] = dict(
            episode_id=eid,
            status="OBSERVED" if observed else "INCOMPLETE",
            net_return=net,
            net_return_pct=100 * net if finite(net) else None,
            Y=y,
        )
    ordered = [windows[e] for e in EPISODES]
    count = sum(row["status"] == "OBSERVED" for row in ordered)
    complete = (
        count == 4
        and summary["parents_complete"] == 4
        and summary["parents_failed"] == 0
        and summary["status"] == "COMPLETE"
    )
    net = mean([row["net_return"] for row in ordered]) if complete else None
    y = mean([row["Y"] for row in ordered]) if complete else None
    return dict(
        status="OBSERVED" if complete else "INCOMPLETE",
        complete_windows=count,
        total_windows=4,
        parents_failed=summary["parents_failed"],
        mean_net_return=net,
        mean_net_return_pct=100 * net if net is not None else None,
        mean_Y=y,
        windows=ordered,
    )


def contrast(exact, other):
    windows = []
    for left, right in zip(exact["windows"], other["windows"], strict=True):
        if left["episode_id"] != right["episode_id"]:
            raise ValueError("unpaired window identity")
        pp = subtract(left["net_return"], right["net_return"], 100)
        windows.append(
            dict(
                episode_id=left["episode_id"],
                status="OBSERVED" if pp is not None else "PENDING_OR_INCOMPLETE",
                net_return_delta_pp=pp,
                Y_delta=subtract(left["Y"], right["Y"]),
            )
        )
    paired = sum(row["net_return_delta_pp"] is not None for row in windows)
    complete = exact["status"] == other["status"] == "OBSERVED"
    deltas = [
        row["net_return_delta_pp"] for row in windows if row["net_return_delta_pp"] is not None
    ]
    return dict(
        status="OBSERVED" if complete else "PENDING_OR_INCOMPLETE",
        paired_windows=paired,
        total_windows=4,
        net_return_delta_pp=mean(deltas) if complete else None,
        Y_delta=mean([row["Y_delta"] for row in windows]) if complete else None,
        positive_windows=sum(v > 1e-12 for v in deltas),
        negative_windows=sum(v < -1e-12 for v in deltas),
        tied_windows=sum(abs(v) <= 1e-12 for v in deltas),
        windows=windows,
    )


def summarize(runs=None):
    seeds, csv_rows, inputs = [], [], []
    for seed, directory in (runs or RUNS).items():
        path = Path(directory).resolve() / "results.json"
        payload = path.read_bytes() if path.exists() else None
        source = json.loads(payload) if payload is not None else {"results": []}
        inputs.append(
            dict(
                seed=seed,
                path=str(path),
                exists=payload is not None,
                sha256=hashlib.sha256(payload).hexdigest() if payload else None,
            )
        )
        by_arm = {row["arm"]: row for row in source["results"]}
        if len(by_arm) != len(source["results"]) or not set(by_arm) <= set(ARMS):
            raise ValueError("duplicate or unexpected result arm")
        panels = {}
        for arm in ARMS:
            result = by_arm.get(arm)
            panels[arm] = {
                stage: panel(result.get(key) if result else None, seed)
                for stage, key in STAGES.items()
            }
            for stage, observed in panels[arm].items():
                csv_rows.append(
                    dict(
                        seed=seed,
                        kind="arm_mean",
                        arm=arm,
                        stage=stage,
                        status=observed["status"],
                        observed_windows=observed["complete_windows"],
                        total_windows=4,
                        net_return=observed["mean_net_return"],
                        net_return_pct=observed["mean_net_return_pct"],
                        Y=observed["mean_Y"],
                    )
                )
                csv_rows.extend(
                    dict(seed=seed, kind="window", arm=arm, stage=stage, **window)
                    for window in observed["windows"]
                )
        contrasts = {}
        for other in ("zero", "plain"):
            name = f"exact_minus_{other}"
            contrasts[name] = {}
            for stage in STAGES:
                difference = contrast(panels["exact"][stage], panels[other][stage])
                contrasts[name][stage] = difference
                csv_rows.append(
                    dict(
                        seed=seed,
                        kind="contrast_mean",
                        arm=name,
                        stage=stage,
                        **{k: v for k, v in difference.items() if k != "windows"},
                    )
                )
                csv_rows.extend(
                    dict(seed=seed, kind="window_contrast", arm=name, stage=stage, **window)
                    for window in difference["windows"]
                )
        complete = (
            len(by_arm) == 3
            and all(by_arm[a]["status"] == "COMPLETE" for a in ARMS)
            and all(p["status"] == "OBSERVED" for a in panels.values() for p in a.values())
        )
        terminal_failure = any(row["status"] == "FAILED" for row in by_arm.values())
        seeds.append(
            dict(
                seed=seed,
                status="COMPLETE"
                if complete
                else ("INCOMPLETE_WITH_FAILURES" if terminal_failure else "PENDING"),
                observed_arms=len(by_arm),
                planned_arms=3,
                panels=panels,
                contrasts=contrasts,
            )
        )
    all_complete = all(item["status"] == "COMPLETE" for item in seeds)
    across = {}
    for name in ("exact_minus_zero", "exact_minus_plain"):
        across[name] = {}
        for stage in STAGES:
            values = [s["contrasts"][name][stage] for s in seeds]
            across[name][stage] = dict(
                status="OBSERVED" if all_complete else "PENDING",
                planned_seeds=len(seeds),
                complete_seeds=sum(s["status"] == "COMPLETE" for s in seeds),
                mean_net_return_delta_pp=mean([v["net_return_delta_pp"] for v in values])
                if all_complete
                else None,
                mean_Y_delta=mean([v["Y_delta"] for v in values]) if all_complete else None,
                positive_seed_count=sum(v["net_return_delta_pp"] > 1e-12 for v in values)
                if all_complete
                else None,
                positive_window_seed_pairs=sum(v["positive_windows"] for v in values)
                if all_complete
                else None,
                planned_window_seed_pairs=4 * len(seeds),
            )
    report = dict(
        profile="same_prefix_online_two_seed_description_v1",
        status="DESCRIBED" if all_complete else "PENDING",
        generated_at_UTC=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        complete_seeds=sum(s["status"] == "COMPLETE" for s in seeds),
        planned_seeds=len(seeds),
        episode_ids=EPISODES,
        seeds=seeds,
        cross_seed_contrasts=across,
        limits=[
            "Four reused Train windows; no new-date or Valid/OOS evaluation.",
            "Descriptive head-update demo; no significance or method-advantage gate.",
            "Missing/incomplete results are NA, not zero or a smaller-denominator mean.",
            "Baseline is shared across arms within each seed, not independent data.",
            "pp means percentage points: 100*(exact raw_net_return - comparator).",
            "Seed-window pairs are reported counts, not independent market samples.",
        ],
    )
    return report, csv_rows


def write_summary(output, runs=None):
    report, records = summarize(runs)
    output = Path(output).resolve()
    names = ("cross_seed_summary.json", "cross_seed_summary.csv")
    if any((output / name).exists() for name in names):
        raise FileExistsError("cross-seed output already exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in records for key in row))
    with (output / names[1]).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    report["csv_sha256"] = hashlib.sha256((output / names[1]).read_bytes()).hexdigest()
    with (output / names[0]).open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42", type=Path, default=RUNS[42])
    parser.add_argument("--seed43", type=Path, default=RUNS[43])
    parser.add_argument("--output", type=Path, default=RUNS[43])
    parser.add_argument("--preview", action="store_true", help="print only; do not write artifacts")
    args = parser.parse_args()
    runs = {42: args.seed42, 43: args.seed43}
    report = summarize(runs)[0] if args.preview else write_summary(args.output, runs)
    print(
        json.dumps(
            dict(
                status=report["status"],
                complete_seeds=report["complete_seeds"],
                planned_seeds=report["planned_seeds"],
                per_seed=[
                    dict(
                        seed=s["seed"],
                        status=s["status"],
                        r2_exact_minus_zero=s["contrasts"]["exact_minus_zero"]["r2"][
                            "net_return_delta_pp"
                        ],
                        r2_exact_minus_plain=s["contrasts"]["exact_minus_plain"]["r2"][
                            "net_return_delta_pp"
                        ],
                    )
                    for s in report["seeds"]
                ],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
