"""Independent accounting checks on actual CPU environments with synthetic prices."""

from __future__ import annotations

import copy
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.data import MarketObservation, StockBenchAdapter

_PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_reward_probe.py"
_SPEC = importlib.util.spec_from_file_location("reward_probe_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)

_DAYS = (
    "2023-01-03",
    "2023-01-04",
    "2023-01-05",
    "2023-01-06",
    "2023-01-09",
    "2023-01-10",
    "2023-01-11",
    "2023-01-12",
    "2023-01-13",
    "2023-01-17",
    "2023-01-18",
    "2023-01-19",
    "2023-01-20",
    "2023-01-23",
    "2023-01-24",
    "2023-01-25",
    "2023-01-26",
    "2023-01-27",
    "2023-01-30",
    "2023-01-31",
    "2023-02-01",
)
_SYMBOLS = ["EEE", "DDD", "AAPL", "BBB", "CCC"]


@pytest.fixture(autouse=True)
def isolate_cpu_probe_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # The standalone probe deliberately selects CPU/offline imports. Undo its
    # environment changes after every test so later real-trainer subprocess
    # checks retain their original Torch import contract.
    for name in (
        "USE_TORCH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.setenv(name, os.environ.get(name, ""))


@pytest.fixture
def window() -> dict[str, Any]:
    dates = [datetime.fromisoformat(day + "T21:00:00+00:00") for day in _DAYS]
    bars = [
        MarketObservation(
            observation_id=f"market:{symbol}:{day}",
            kind="market",
            symbol=symbol,
            event_at=stamp,
            available_at=stamp,
            payload={
                "date": day,
                "open": 100.0,
                "close": 100.0,
                "high": 100.0,
                "low": 100.0,
                "volume": 1000,
            },
        )
        for symbol in _SYMBOLS
        for day, stamp in zip(_DAYS, dates, strict=True)
    ]
    return {
        "episode_id": probe.EPISODES[0],
        "source_index": 0,
        "raw_prompt": [],
        "env_kwargs": {
            "adapter": StockBenchAdapter(bars, data_version="synthetic-reward-probe"),
            "episode_dates": dates,
            "episode_open_times": [stamp.replace(hour=14, minute=30) for stamp in dates],
            "symbols": _SYMBOLS,
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
            "drawdown_coefficient": 0.0,
            "turnover_coefficient": 0.0,
            "benchmark_closes": [1000.0] * 21,
            "screening": True,
            "finance_scorer_id": "benchmark_scale_v2",
            "finance_volatility_floor": 0.001,
            "risk_lambda": 2.0,
            "turnover_limit": 1.0,
            "position_limit": 0.3,
            "screening_calibration_sha256": "a" * 64,
        },
    }


def execute(window: dict[str, Any], strategy: str) -> dict[str, Any]:
    return probe.execute_strategy(
        window,
        {
            "grid_index": 0,
            "episode_id": window["episode_id"],
            "window_index": 0,
            "strategy_id": strategy,
            "actions": probe.policy_actions(strategy, window["env_kwargs"]["symbols"]),
        },
    )


def test_registry_and_grid_keep_all_prespecified_controls(window: dict[str, Any]) -> None:
    registry = probe.strategy_registry()
    assert registry["trajectories"] == 12
    assert registry["total_environment_steps"] == 240
    assert registry["model_requests"] == 0
    assert registry["reward_or_cost_or_risk_changes"] is False
    windows = [window, {**window, "episode_id": probe.EPISODES[1], "source_index": 4}]
    grid = probe.fixed_grid(windows)
    assert [cell["grid_index"] for cell in grid] == list(range(12))
    assert {(cell["episode_id"], cell["strategy_id"]) for cell in grid} == {
        (episode, strategy) for episode in probe.EPISODES for strategy in probe.STRATEGIES
    }
    assert all(len(cell["actions"]) == 20 for cell in grid)
    staggered = grid[3]["actions"]
    assert [action["arguments"]["symbol"] for action in staggered[:5]] == _SYMBOLS
    assert all(action["name"] == "hold" for action in staggered[5:])
    with pytest.raises(ValueError, match="window identity"):
        probe.fixed_grid([window, window])


@pytest.mark.parametrize("strategy", probe.STRATEGIES)
def test_actual_env_twenty_advances_match_independent_tape(
    window: dict[str, Any], strategy: str
) -> None:
    result = execute(window, strategy)
    metrics = result["independent_metrics"]
    assert result["status"] == "VERIFIED"
    assert len(result["steps"]) == len(result["credit_step_trace"]) == 20
    assert set(result["reconciliation"]) == {
        "terminal_score",
        "raw_net_return",
        "benchmark_return",
        "benchmark_excess",
        "agent_volatility",
        "benchmark_volatility",
        "score_denominator",
        "max_drawdown",
        "turnover",
        "max_daily_turnover",
        "maximum_position_weight",
        "risk_penalty",
        "transaction_cost",
        "commission",
        "slippage_cost",
    }
    assert all(row["reward"] == 0 for row in result["steps"][:-1])
    assert result["steps"][-1]["reward"] == pytest.approx(metrics["terminal_score"])
    assert metrics["net_return_without_costs_same_filled_shares"] == pytest.approx(0)
    assert metrics["raw_net_return"] == pytest.approx(-metrics["transaction_cost"] / 1000)
    assert metrics["score_denominator"] == 0.001
    if strategy == "cash":
        assert result["fills"] == []
        assert metrics["terminal_score"] == metrics["transaction_cost"] == 0
    else:
        assert metrics["terminal_score"] < 0
    if strategy == "aapl_buy_hold_60pct":
        assert len(result["fills"]) == 1  # A soft limit does not reject the trade.
        assert metrics["position_excess"] > 0.3
        assert metrics["weighted_risk_deduction"] > 0
    if strategy == "aapl_buy_hold_30pct":
        assert metrics["position_excess"] > 0  # Fees alone move the marked weight above target.
    if strategy == "first5_staggered_10pct":
        assert [fill["symbol"] for fill in result["fills"]] == _SYMBOLS
        assert len({fill["executed_at"] for fill in result["fills"]}) == 5


def test_real_roundtrip_uses_two_distinct_next_opens_and_charges_both_sides(
    window: dict[str, Any],
) -> None:
    bought = execute(window, "aapl_buy_hold_10pct")
    roundtrip = execute(window, "aapl_roundtrip_10pct")
    assert bought["independent_metrics"]["transaction_cost"] == pytest.approx(0.15005)
    assert roundtrip["independent_metrics"]["transaction_cost"] == pytest.approx(0.3)
    assert roundtrip["independent_metrics"]["raw_net_return"] == pytest.approx(-0.0003)
    buy, sell = roundtrip["fills"]
    assert buy["shares"] == sell["shares"] == 1.0
    assert buy["execution_price"] == pytest.approx(100.05)
    assert sell["execution_price"] == pytest.approx(99.95)
    assert buy["executed_at"] == window["env_kwargs"]["episode_open_times"][1]
    assert sell["executed_at"] == window["env_kwargs"]["episode_open_times"][2]
    assert roundtrip["independent_metrics"]["daily_reconstruction"][-1]["positions"]["AAPL"] == 0


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("commission", 0.0, "commission"),
        ("raw_open_price", 101.0, "raw opening"),
        ("executed_at", "2023-01-03T14:30:00+00:00", "next open"),
        ("shares", -1.0, "filled shares"),
    ],
)
def test_independent_tape_rejects_corrupt_fill_evidence(
    window: dict[str, Any], field: str, value: Any, error: str
) -> None:
    result = execute(window, "aapl_buy_hold_10pct")
    steps = copy.deepcopy(result["steps"])
    steps[0]["info"]["fills"][0][field] = value
    with pytest.raises(ValueError, match=error):
        probe.recompute_from_tape(
            kwargs=window["env_kwargs"],
            actions=result["actions"],
            marks=result["marks"],
            steps=steps,
        )


def test_incomplete_tape_and_changed_registered_action_are_rejected(window: dict[str, Any]) -> None:
    result = execute(window, "cash")
    with pytest.raises(ValueError, match="Incomplete accounting tape"):
        probe.recompute_from_tape(
            kwargs=window["env_kwargs"],
            actions=result["actions"],
            marks=result["marks"],
            steps=result["steps"][:-1],
        )
    result["actions"][0] = {"name": "abstain", "arguments": {}}
    with pytest.raises(ValueError, match="Registered fixed action sequence"):
        probe.execute_strategy(window, result)


def test_controlled_reward_sign_checks_are_supplementary() -> None:
    result = probe.mathematical_properties()
    assert result["status"] == "PASSED"
    low, high = result["higher_return_increases_score_other_inputs_fixed"]
    assert high > low
    before, after = result["dominant_positive_position_excess_decreases_score"]
    assert after < before


def test_validation_failure_preserves_observed_complete_tape(
    window: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**kwargs: Any) -> None:
        raise ValueError("injected reconciliation failure")

    monkeypatch.setattr(probe, "recompute_from_tape", fail)
    with pytest.raises(probe.RewardProbeFailure, match="injected") as caught:
        execute(window, "aapl_buy_hold_10pct")
    evidence = caught.value.evidence
    assert len(evidence["steps"]) == len(evidence["credit_step_trace"]) == 20
    assert len(evidence["marks"]) == 21 and len(evidence["fills"]) == 1
    assert evidence["observed_advances"] == 20
    assert evidence["environment_metrics"]["session_count"] == 21


def test_frozen_registry_files_are_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    probe.write_new(path, {"frozen": 1})
    with pytest.raises(FileExistsError):
        probe.write_new(path, {"frozen": 2})
    assert '"frozen": 1' in path.read_text()
