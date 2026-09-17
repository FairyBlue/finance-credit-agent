#!/usr/bin/env python3
"""Descriptive cold-start readout; never selects prompts, samples, or actions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/m2-coldstart-20260913"


def mean(values):
    return sum(values) / len(values) if values else None


def describe(rows):
    groups = defaultdict(list)
    for row in rows:
        key = "_".join(
            (row["horizon"], row["sampling"], "cash" if row["state_kind"] == "cash" else "held")
        )
        groups[key].append(row)
    result = {}
    for key, selected in groups.items():
        completed = [r for r in selected if r["status"] == "COMPLETE"]
        fills = [f for r in selected for f in r["model_fills"]]
        actions = Counter()
        for row in selected:
            actions.update(row["actions"])
        result[key] = {
            "planned": len(selected),
            "complete": len(completed),
            "requests": sum(r["requests"] for r in selected),
            "invalid": sum(r["invalid"] for r in selected),
            "actions": dict(actions),
            "trajectories_with_model_fills": sum(bool(r["model_fills"]) for r in selected),
            "trajectories_trading_at_least_two_sessions": sum(
                len({f["executed_at"] for f in r["model_fills"]}) >= 2 for r in selected
            ),
            "fill_sides": dict(Counter(f["side"] for f in fills)),
            "fill_symbols": dict(Counter(f["symbol"] for f in fills)),
            "mean_takeover_return": mean([r["takeover_net_return"] for r in completed]),
            "model_transaction_cost": sum(f["commission"] + f["slippage_cost"] for f in fills),
            "mean_final_cash_fraction": mean(
                [r["final_state"]["cash"] / r["final_state"]["portfolio_equity"] for r in completed]
            ),
            "mean_final_position_count": mean(
                [len(r["final_state"]["positions"]) for r in completed]
            ),
            "max_position_weight": max(
                (r["max_model_position_weight"] for r in selected), default=0
            ),
            "max_drawdown_including_prefix": max(
                (r["terminal_metrics"]["max_drawdown"] for r in completed), default=None
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-eval", default="sft_eval")
    args = parser.parse_args()
    base = json.loads((RUN / "base_eval/result.json").read_text())
    after = json.loads((RUN / args.after_eval / "result.json").read_text())
    before_by_id = {r["id"]: r for r in base["grid"]}
    assert set(before_by_id) == {r["id"] for r in after["grid"]}
    pairs = []
    for later in after["grid"]:
        earlier = before_by_id[later["id"]]
        assert earlier["context"] == later["context"]
        assert earlier["identity"] == later["identity"]
        complete = earlier["status"] == later["status"] == "COMPLETE"
        pairs.append(
            {
                "id": later["id"],
                "horizon": later["horizon"],
                "sampling": later["sampling"],
                "both_complete": complete,
                "takeover_return_change": (
                    later["takeover_net_return"] - earlier["takeover_net_return"]
                    if complete
                    else None
                ),
                "base_status": earlier["status"],
                "sft_status": later["status"],
            }
        )
    data = {
        "status": "COMPLETE",
        "base": describe(base["grid"]),
        "after_sft": describe(after["grid"]),
        "pairs": pairs,
        "limits": [
            "Train diagnostics, overlapping reserved windows, no independent OOS claim.",
            "Short risk/scorer include fixed prefix; model fills exclude prefix.",
            "Same context/seed identities do not imply identical response trajectories.",
        ],
    }
    (RUN / "behavior_readout.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in data.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
