"""Readiness gates use autonomous evidence and preserve incomplete paired outcomes."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.agents.coldstart_eval import grid

SPEC = importlib.util.spec_from_file_location(
    "coldstart_v2_readout",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_coldstart_v2_readout.py",
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

SCENARIOS = [
    "cash", "single_low", "single_high", "pair_low", "pair_mixed",
    "triple_below", "triple_above", "triple_mixed",
]


def observed(suite):
    contexts = {"short": [], "full": [], "teacher_labels_included": False}
    for index in range(16):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        error = suite == "fresh" and scenario in report.RECOVERY_SCENARIOS
        contexts["short"].append({
            "id": f"short-{index}", "state_kind": scenario,
            "prefix": ["synthetic error"] if error else [],
            "prefix_expected_invalid": [True] if error else [],
            "prefix_expected_errors": ["wrong direction"] if error else [],
        })
    for index in range(4 if suite == "regression" else 1):
        contexts["full"].append({"id": f"full-{index}", "state_kind": "cash", "prefix": []})
    rows = grid(contexts)
    for row in rows:
        row.update(
            status="COMPLETE", requests=20, invalid=0, actions={"buy": 1, "sell": 1, "hold": 18},
            takeover_net_return=0.01 + row["identity"]["rollout_n"] * 0.001,
            model_fills=[
                {"executed_at": date + "T13:30:00+00:00", "side": side, "symbol": "GS",
                 "commission": 1.0, "slippage_cost": 0.5}
                for date, side in (("2023-07-03", "buy"), ("2023-07-05", "sell"))
            ],
            final_state={"cash": 1000.0, "portfolio_equity": 1100.0, "positions": {"GS": {}}},
            max_model_position_weight=0.2,
            terminal_metrics={"terminal_score": 1.0, "max_drawdown": 0.01},
        )
    return {"status": "OBSERVED", "action_protocol": report.PROTOCOL, "grid": rows}, contexts


def test_pairs_exclude_incomplete_financial_outcomes_and_check_seed_identity():
    before, _ = observed("fresh")
    after = copy.deepcopy(before)
    after["grid"][0]["takeover_net_return"] += 0.03
    after["grid"][1]["status"] = "INCOMPLETE"
    del after["grid"][1]["takeover_net_return"]
    value = report.pair_results(before, after)
    assert value["pairs"][0]["takeover_return_change"] == pytest.approx(0.03)
    assert value["pairs"][1]["takeover_return_change"] is None
    assert value["paired_return_change"]["short_sample"]["both_complete"] == 63
    assert value["after"]["overall"]["complete"] == 84
    after["grid"][2]["identity"]["rollout_n"] = 999
    with pytest.raises(ValueError, match="identity"):
        report.pair_results(before, after)


def test_complete_varied_legal_trading_passes_with_declared_prefix_errors_excluded():
    regression, _ = observed("regression")
    fresh, _ = observed("fresh")
    value = report.candidate_gates({"regression": regression["grid"], "fresh": fresh["grid"]})
    assert value["status"] == "PASS"
    assert value["automatic_rl_launch"] is False
    recovery = value["criteria"]["fresh_error_recovery"]
    assert recovery["planned"] == recovery["complete"] == 40
    assert recovery["invalid_per_request"] == 0
    description = report.describe(fresh["grid"])
    assert description["model_fill_count"] == 170
    assert description["model_transaction_cost"] == 255.0
    assert description["legal_model_calls"] == 1700


def test_all_hold_cannot_pass_only_by_being_legal_and_complete():
    suites = {name: observed(name)[0]["grid"] for name in ("regression", "fresh")}
    for rows in suites.values():
        for row in rows:
            row.update(model_fills=[], takeover_net_return=0.0, actions={"hold": 20})
    value = report.candidate_gates(suites)
    assert value["status"] == "FAIL"
    assert value["criteria"]["fresh_complete"]["pass"]
    assert value["criteria"]["fresh_invalid"]["pass"]
    assert not value["criteria"]["fresh_siblings"]["pass"]
    assert not value["criteria"]["regression_siblings"]["pass"]
    assert not value["criteria"]["full_sample_sustained_trading"]["pass"]


def test_invalid_threshold_and_distinct_dates_are_counted_exactly():
    suites = {name: observed(name)[0]["grid"] for name in ("regression", "fresh")}
    for rows in suites.values():
        for row in rows:
            row.update(invalid=1, actions={"buy": 1, "sell": 1, "invalid": 1, "hold": 17})
    assert report.candidate_gates(suites)["status"] == "PASS"
    suites["fresh"][0]["invalid"] = 2
    value = report.candidate_gates(suites)
    assert not value["criteria"]["fresh_invalid"]["pass"]
    assert not value["criteria"]["fresh_error_recovery"]["pass"]
    for rows in suites.values():
        for row in rows:
            row["model_fills"][1]["executed_at"] = "2023-07-03T15:30:00+00:00"
    assert not report.candidate_gates(suites)["criteria"]["full_sample_sustained_trading"]["pass"]


def test_incomplete_sibling_group_does_not_count_as_diverse_and_population_is_checked():
    result, contexts = observed("fresh")
    report.validate_population(result, contexts)
    result["grid"][1]["status"] = "INCOMPLETE"
    detail = next(item for item in report.siblings(result["grid"])["details"]
                  if item["context_id"] == "short-0")
    assert detail["complete_siblings"] == 3
    assert detail["takeover_return_range"] is None
    assert detail["distinct_outcomes"] is False
    result["grid"].pop()
    with pytest.raises(ValueError, match="frozen dataset population"):
        report.validate_population(result, contexts)


def test_negative_returns_do_not_fail_readiness_or_select_better_outcomes():
    suites = {name: observed(name)[0]["grid"] for name in ("regression", "fresh")}
    for rows in suites.values():
        for row in rows:
            row["takeover_net_return"] -= 0.5
    assert report.candidate_gates(suites)["status"] == "PASS"
    assert report.describe(suites["fresh"])["mean_takeover_return_complete_only"] < 0


def test_invalid_error_readout_streams_model_requests_and_excludes_prefix(tmp_path):
    records = [
        {"info": {"invalid_action": True, "error": "wrong direction"}},
        {"info": {"invalid_action": True, "error": "wrong direction"}},
        {"info": {"invalid_action": False},
         "prefix_info": [{"invalid_action": True, "error": "prefix only"}]},
        {"finish_reason": "length"},
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    counts, digest = report.model_error_counts(path)
    assert counts == {"wrong direction": 2}
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
