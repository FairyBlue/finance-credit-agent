"""Validate comparison units, policy-step separation, and incomplete populations."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

_helpers = runpy.run_path(
    str(Path(__file__).resolve().parents[1]
        / "scripts/diagnostics/m2_low_memory_comparison_readout.py")
)
PROFILE, comparison = _helpers["PROFILE"], _helpers["comparison"]


def state():
    def evaluation(greedy, sampled):
        return {
            lane: {
                "trajectories": n, "decisions": n * 20, "invalid_decisions": 0,
                "mean_raw_net_return": value, "mean_terminal_score": -1.0,
                "mean_transaction_cost": 100.0, "mean_max_drawdown": 0.01,
                "mean_agent_volatility": 0.001, "mean_benchmark_excess": 0.0,
                "mean_turnover": 1.0, "mean_maximum_position_weight": 0.2,
            }
            for lane, n, value in (("greedy", 4, greedy), ("sampled", 16, sampled))
        }

    cells = [
        {"cell": {"cell_id": cell_id}, "status": "COMPLETE", "actual_rounds": rounds,
         "evaluation_by_step": grid}
        for cell_id, rounds, grid in (
            ("sft0", 0, {"0": evaluation(0.01, 0.02)}),
            ("m1", 4, {"2": evaluation(0.02, 0.01), "4": evaluation(0.03, 0.02)}),
            ("m2_eta025", 4, {"2": evaluation(0.025, 0.015), "4": evaluation(0.02, 0.01)}),
        )
    ]
    return {"status": "COMPLETE", "profile": PROFILE, "planned_trajectories": 164,
            "cells": cells}


def test_curves_preserve_reversing_learning_and_decode_effects_in_percentage_points():
    curves, rows = comparison(state())
    assert len(curves) == 10
    grouped = {(row["policy_step"], row["lane"]): row for row in rows}
    assert grouped[(2, "greedy")]["m2_minus_m1_percentage_points"] == pytest.approx(0.5)
    assert grouped[(4, "greedy")]["m2_minus_m1_percentage_points"] == pytest.approx(-1.0)
    assert grouped[(2, "sampled")]["m2_minus_sft_percentage_points"] == pytest.approx(-0.5)
    assert grouped[(4, "greedy")]["m1_minus_sft_percentage_points"] == pytest.approx(2.0)
    assert grouped[(4, "sampled")]["sft_net_return_percent"] == 2.0


def test_seed43_readout_requires_its_own_sealed_identity():
    value = state()
    value.update(profile=_helpers["SEED43_PROFILE"], seed=43)
    assert len(comparison(value, expected_seed=43)[0]) == 10
    with pytest.raises(ValueError):
        comparison(value)
    value["seed"] = 42
    with pytest.raises(ValueError):
        comparison(value, expected_seed=43)
    value = state()
    with pytest.raises(ValueError):
        comparison(value, expected_seed=43)


@pytest.mark.parametrize("damage", ["failed", "duplicate", "missing_step", "lane_count",
                                    "nonfinite", "accidental_sft_update"])
def test_partial_or_wrong_comparisons_are_not_published(damage):
    value = copy.deepcopy(state())
    if damage == "failed":
        value["status"] = "FAILED"
    elif damage == "duplicate":
        value["cells"].append(copy.deepcopy(value["cells"][1]))
    elif damage == "missing_step":
        del value["cells"][2]["evaluation_by_step"]["2"]
    elif damage == "lane_count":
        value["cells"][0]["evaluation_by_step"]["0"]["sampled"]["trajectories"] = 15
    elif damage == "nonfinite":
        value["cells"][1]["evaluation_by_step"]["4"]["greedy"]["mean_raw_net_return"] = float(
            "nan"
        )
    else:
        value["cells"][0]["actual_rounds"] = 1
    with pytest.raises(ValueError):
        comparison(value)
