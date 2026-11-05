#!/usr/bin/env python3
"""Read the completed SFT/M1/M2 four-round comparison without mixing policy steps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROFILE = "common_sft_low_memory_comparison_v1"
SEED43_PROFILE = "common_sft_low_memory_comparison_seed43_v1"
LANES = {"greedy": 4, "sampled": 16}
METRICS = (
    "raw_net_return", "terminal_score", "transaction_cost", "max_drawdown",
    "agent_volatility", "benchmark_excess", "turnover", "maximum_position_weight",
)


def comparison(state, *, expected_seed=42):
    profiles = {42: PROFILE, 43: SEED43_PROFILE}
    if type(expected_seed) is not int or expected_seed not in profiles:
        raise ValueError("only the reviewed seed42/43 comparisons are supported")
    if (state.get("status") != "COMPLETE" or state.get("profile") != profiles[expected_seed]
            or state.get("seed", 42) != expected_seed
            or state.get("planned_trajectories") != 164):
        raise ValueError("requires the complete fixed 164-trajectory comparison")
    cells = {row["cell"]["cell_id"]: row for row in state["cells"]}
    if len(state["cells"]) != 3 or set(cells) != {"sft0", "m1", "m2_eta025"}:
        raise ValueError("SFT/M1/M2 population is incomplete or duplicated")
    curves = []
    for cell_id, model, steps, rounds in (
        ("sft0", "SFT", {"0"}, 0), ("m1", "M1", {"2", "4"}, 4),
        ("m2_eta025", "M2", {"2", "4"}, 4),
    ):
        cell = cells[cell_id]
        if cell.get("status") != "COMPLETE" or cell.get("actual_rounds") != rounds:
            raise ValueError("a cell did not complete its fixed update budget")
        evaluations = cell["evaluation_by_step"]
        if set(evaluations) != steps:
            raise ValueError("evaluation policy steps differ from the fixed grid")
        for step in sorted(evaluations, key=int):
            if set(evaluations[step]) != set(LANES):
                raise ValueError("greedy/sample lanes must remain separate")
            for lane, expected_n in LANES.items():
                row = evaluations[step][lane]
                if row["trajectories"] != expected_n:
                    raise ValueError("evaluation lane has the wrong trajectory count")
                if (type(row["decisions"]) is not int or row["decisions"] < expected_n
                        or type(row["invalid_decisions"]) is not int
                        or not 0 <= row["invalid_decisions"] <= row["decisions"]):
                    raise ValueError("invalid decision counters")
                for metric in METRICS:
                    value = row["mean_" + metric]
                    if type(value) not in (float, int) or not math.isfinite(value):
                        raise ValueError("nonfinite or missing outcome metric")
                curves.append({"model": model, "policy_step": int(step), "lane": lane, **row})
    comparisons = []
    for step in (2, 4):
        for lane in LANES:
            baseline = cells["sft0"]["evaluation_by_step"]["0"][lane]
            m1 = cells["m1"]["evaluation_by_step"][str(step)][lane]
            m2 = cells["m2_eta025"]["evaluation_by_step"][str(step)][lane]
            outcome = {"policy_step": step, "lane": lane}
            for name, row in (("sft", baseline), ("m1", m1), ("m2", m2)):
                outcome[name + "_net_return_percent"] = 100 * row["mean_raw_net_return"]
                outcome[name + "_invalid"] = row["invalid_decisions"]
                outcome[name + "_decisions"] = row["decisions"]
            for name, left, right in (
                ("m1_minus_sft", m1, baseline), ("m2_minus_sft", m2, baseline),
                ("m2_minus_m1", m2, m1),
            ):
                outcome[name + "_percentage_points"] = 100 * (
                    left["mean_raw_net_return"] - right["mean_raw_net_return"]
                )
            comparisons.append(outcome)
    return curves, comparisons


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=ROOT / "runs/m2-sft-four-round-comparison-20261004")
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    args = parser.parse_args()
    base = args.directory.resolve()
    from src.agents.learning_restart import now
    if args.seed == 43:
        from src.agents.low_memory_comparison_seed43 import PROFILE as trial_profile
    else:
        from src.agents.low_memory_comparison import PROFILE as trial_profile
    from src.agents.low_memory_pilot import check_contract

    check_contract(base, profile=trial_profile)
    payload = (base / "state.json").read_bytes()
    state = json.loads(payload)
    curves, comparisons = comparison(state, expected_seed=args.seed)
    write_csv(base / "evaluation_curve.csv", curves)
    write_csv(base / "comparison_summary.csv", comparisons)
    readout = {
        "status": "PASS", "at_SGT": now(), "profile": trial_profile.name, "seed": args.seed,
        "complete_trajectories": 164, "state_sha256": hashlib.sha256(payload).hexdigest(),
        "comparison": comparisons, "evaluation_curve": curves,
        "formal_performance_claim": False,
        "sampled_rng_note": "Existing request seed includes policy step; steps 0/2/4 differ",
    }
    (base / "comparison_readout.json").write_text(
        json.dumps(readout, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "PASS", "comparisons": len(comparisons)}))


if __name__ == "__main__":
    main()
