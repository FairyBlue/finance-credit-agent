"""CPU-only sampling spread audit of the completed coverage experiment.

Intervals resample recorded siblings within each of the four fixed Train
windows. They describe finite sampling uncertainty, not out-of-sample or
training-seed uncertainty. No model is loaded and no new trajectories are made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def resample_mean(values, rng):
    return statistics.mean(rng.choices(values, k=len(values)))


def compare(groups, target, reference, *, paired, repetitions=20000):
    a, b = groups[target], groups[reference]
    assert set(a) == set(b) and len(a) == 4
    rng = random.Random(20260917)
    windows = sorted(a)
    deltas = {w: statistics.mean(a[w]) - statistics.mean(b[w]) for w in windows}
    draws = []
    for _ in range(repetitions):
        if paired:
            parts = [
                resample_mean([x - y for x, y in zip(a[w], b[w], strict=True)], rng)
                for w in windows
            ]
        else:
            parts = [resample_mean(a[w], rng) - resample_mean(b[w], rng) for w in windows]
        draws.append(statistics.mean(parts))
    return {
        "target": target,
        "reference": reference,
        "mean_delta_pp": statistics.mean(deltas.values()),
        "window_delta_pp": deltas,
        "resampling": "paired lanes within window"
        if paired
        else "independent groups within window",
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": 20260917,
        "conditional_percentile_95_interval_pp": [
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ],
        "bootstrap_fraction_positive_descriptive_only": sum(x > 0 for x in draws) / repetitions,
        "interval_scope": (
            "Fixed four windows, observed policies and four recorded siblings per "
            "window; no market/window or training-seed resampling. Small-sample "
            "descriptive interval, not a formal method-performance test."
        ),
    }


def analyze(source, output):
    source, output = source.resolve(), output.resolve()
    readout_path = source / "readout.json"
    readout = json.loads(readout_path.read_text())
    raw_audit = json.loads((source / "raw_trajectory_audit.json").read_text())
    assert raw_audit["status"] == "PASS"
    assert digest(readout_path) == raw_audit["source_files"]["readout.json"]["sha256"]
    assert readout["status"] == "COMPLETE" and readout["totals"]["terminal_complete"] == 308
    provenance = {
        "readout.json": digest(readout_path),
        "raw_trajectory_audit.json": digest(source / "raw_trajectory_audit.json"),
    }
    groups = defaultdict(dict)
    spread = []
    all_rows = []
    seed_maps = {}
    window_indices = {}
    for cell_name, cell in readout["cells"].items():
        seeds = {}
        for path in sorted((source / cell_name / "artifacts/rollout_seed").glob("*.jsonl")):
            provenance[str(path.relative_to(source))] = digest(path)
            for line in path.read_text().splitlines():
                record = json.loads(line)
                assert record["base_seed"] == 42
                if not record["validate"]:
                    continue
                key = tuple(record[k] for k in ("step", "sample_index", "rollout_n", "env_step"))
                assert key not in seeds
                payload = {
                    k: record[k]
                    for k in (
                        "base_seed",
                        "env_step",
                        "rollout_n",
                        "sample_index",
                        "scheme",
                        "step",
                        "validate",
                    )
                }
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
                expected = (
                    int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF
                )
                assert expected == record["request_seed"]
                seeds[key] = expected
        seed_maps[cell_name] = seeds
        rows = defaultdict(list)
        for row in cell["trajectory_rows"]:
            identity = row["identity"]
            if not identity["validate"] or identity["rollout_n"] == 0:
                continue
            assert row["terminal_eligible"] and row["reported_status"] == "COMPLETE"
            key = (identity["step"], row["episode_id"])
            known_index = window_indices.setdefault(row["episode_id"], identity["sample_index"])
            assert known_index == identity["sample_index"]
            rows[key].append((identity["rollout_n"], 100 * row["net_return"]))
            all_rows.append(
                {
                    "cell": cell_name,
                    "policy_step": identity["step"],
                    "window": row["episode_id"],
                    "lane": identity["rollout_n"],
                    "net_return_pct": 100 * row["net_return"],
                    "source": row["source"],
                }
            )
        for (step, window), lane_values in sorted(rows.items()):
            assert sorted(x[0] for x in lane_values) == [1, 2, 3, 4]
            values = [v for _, v in sorted(lane_values)]
            groups[f"{cell_name}:step{step}"][window] = values
            spread.append(
                {
                    "cell": cell_name,
                    "policy_step": step,
                    "window": window,
                    "n": 4,
                    "mean_net_return_pct": statistics.mean(values),
                    "sample_sd_pp": statistics.stdev(values),
                    "min_net_return_pct": min(values),
                    "max_net_return_pct": max(values),
                }
            )
    shared = set(seed_maps["m1"]) & set(seed_maps["m2_eta025"])
    assert len(all_rows) == 144 and len(groups) == 9
    assert sorted(window_indices.values()) == [0, 1, 2, 3]
    assert shared and all(seed_maps["m1"][k] == seed_maps["m2_eta025"][k] for k in shared)
    changed = 0
    for cell_name in ("m1", "m2_eta025"):
        for window_index in range(4):
            for lane in range(1, 5):
                first = [
                    seed_maps[cell_name][(step, window_index, lane, 0)] for step in (2, 4, 6, 8)
                ]
                assert len(set(first)) == 4
                changed += 1
    contrasts = []
    for cell in ("m1", "m2_eta025"):
        for baseline in (2, 4):
            contrasts.append(
                compare(groups, f"{cell}:step8", f"{cell}:step{baseline}", paired=False)
            )
        contrasts.append(compare(groups, f"{cell}:step8", "sft0:step0", paired=False))
    contrasts.append(compare(groups, "m2_eta025:step8", "m1:step8", paired=True))
    summaries = []
    for name, windows in groups.items():
        assert len(windows) == 4
        summaries.append(
            {
                "group": name,
                "mean_net_return_pct": statistics.mean(
                    statistics.mean(v) for v in windows.values()
                ),
                "conditional_monte_carlo_standard_error_pp": math.sqrt(
                    sum(statistics.variance(v) / len(v) for v in windows.values())
                )
                / 4,
                "scope": (
                    "SD of the estimated equal-window mean from finite independent "
                    "sampled siblings, conditional on fixed windows and policy; not "
                    "market or training uncertainty."
                ),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("sampled_trajectories.csv", all_rows),
        ("within_window_spread.csv", spread),
    ):
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = {
        "status": "PASS",
        "at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "source_run": str(source),
        "source_sha256": provenance,
        "script_sha256": digest(Path(__file__)),
        "sampled_trajectories": len(all_rows),
        "seed_checks": {
            "shared_M1_M2_eval_request_identity_count": len(shared),
            "all_shared_request_seeds_equal": True,
            "common_episode_to_sample_index": window_indices,
            "sampled_cell_window_lanes_with_changed_seeds_across_checkpoints": changed,
            "note": (
                "Same-step M1/M2 lane pairing is justified by equal request seeds at "
                "shared indices, not identical states/actions. Seeds change with "
                "policy step, so checkpoint contrasts resample independently."
            ),
        },
        "groups": summaries,
        "within_window_spread": spread,
        "contrasts": contrasts,
        "limits": [
            (
                "Only four sampled siblings per window; bootstrap support is small "
                "and tails may be underestimated."
            ),
            (
                "Intervals condition on the same four, partly overlapping windows; "
                "they do not measure out-of-sample performance or training-seed "
                "robustness."
            ),
            (
                "Changing policy and evaluation seed together prevents attributing "
                "the observed checkpoint difference solely to learning."
            ),
            (
                "No new GPU trajectories; source outcomes are taken from the "
                "completed readout verified against its independent raw audit."
            ),
        ],
    }
    (output / "sampling_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"status": "PASS", "groups": summaries, "contrasts": contrasts},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/m2-train-coverage-20260916-v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/m2-train-coverage-diagnosis-20260917/sampling")
    )
    args = parser.parse_args()
    analyze(args.source, args.output)
