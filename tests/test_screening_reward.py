"""Behavioral checks for the new scorer and independent 32 x 44 calibration."""

from copy import deepcopy
from dataclasses import replace
from datetime import date
from typing import Any

import exchange_calendars as xcals
import pytest

from src.env import ExecutionPolicy
from src.rewards.calibration import CalibrationMarketData, _replay
from src.rewards.finance import RiskComponents
from src.rewards.screening import (
    calibrate_screening_reward,
    score_benchmark_scaled,
    screening_proxy_targets,
    verify_screening_calibration,
)


def synthetic_market() -> CalibrationMarketData:
    symbols = tuple(f"S{index:02d}" for index in range(20))
    calendar = xcals.get_calendar("XNYS")
    labels = tuple(calendar.sessions_in_range("2023-01-03", "2023-02-01"))
    close_rows = tuple(
        {
            symbol: 100 + j + i * (j - 8.5) * 0.15 + (i % 4 - 1.5) * (j % 3 + 1) * 0.4
            for j, symbol in enumerate(symbols)
        }
        for i in range(21)
    )
    open_rows = tuple(
        {symbol: value * (1 + 0.001 * (i % 3 - 1)) for symbol, value in row.items()}
        for i, row in enumerate(close_rows)
    )
    return CalibrationMarketData(
        symbols=symbols,
        sessions=tuple(label.date() for label in labels),
        session_open_times=tuple(calendar.session_open(label).to_pydatetime() for label in labels),
        session_close_times=tuple(
            calendar.session_close(label).to_pydatetime() for label in labels
        ),
        open_prices=open_rows,
        close_prices=close_rows,
        train_start=date(2023, 1, 3),
        train_end=date(2023, 2, 1),
        data_version="synthetic-screen-train",
        source_manifest_sha256="a" * 64,
    )


def calibration_inputs() -> tuple[dict, dict]:
    market = synthetic_market()
    policy = ExecutionPolicy(10.0, 5.0)
    benchmark = _replay(
        market=market,
        start=0,
        targets=screening_proxy_targets(market.symbols)[-1][1],
        initial_cash=1_000_000.0,
        execution_policy=policy,
    ).equity_curve
    ids = [f"formal-train-{index + 1:04d}" for index in range(0, 125, 4)]
    return {key: market for key in ids}, {key: benchmark for key in ids}


@pytest.fixture(scope="module")
def calibrated() -> dict:
    markets, benchmarks = calibration_inputs()
    return calibrate_screening_reward(
        markets=markets,
        expected_benchmarks=benchmarks,
        initial_cash=1_000_000.0,
        execution_policy=ExecutionPolicy(10.0, 5.0),
        source={},
    )


def test_external_denominator_cash_floor_and_risk_semantics() -> None:
    risk = RiskComponents(0.02, 0.1, 0.0)
    score = score_benchmark_scaled(
        episode_return=0.01,
        benchmark_return=0.03,
        benchmark_volatility=0.01,
        volatility_floor=0.001,
        risk_lambda=2.0,
        risk=risk,
    )
    assert score.advantage == pytest.approx(-2.2)
    assert score.risk_adjusted_excess_return == pytest.approx(-2.0)
    cash = score_benchmark_scaled(
        episode_return=0.0,
        benchmark_return=0.03,
        benchmark_volatility=0.0,
        volatility_floor=0.001,
        risk_lambda=2.0,
        risk=RiskComponents(0.0, 0.0, 0.0),
    )
    assert cash.advantage == pytest.approx(-30.0)  # No hidden [-5,5] clipping.
    # Agent volatility is deliberately absent from the accepted input contract.
    with pytest.raises(TypeError, match="volatility"):
        score_benchmark_scaled(  # type: ignore[call-arg]
            episode_return=0.01,
            benchmark_return=0.03,
            benchmark_volatility=0.01,
            volatility_floor=0.001,
            risk_lambda=2.0,
            risk=risk,
            volatility=0.1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("volatility_floor", 0.0),
        ("volatility_floor", float("inf")),
        ("benchmark_volatility", -0.01),
        ("risk_lambda", -1.0),
        ("episode_return", float("nan")),
    ],
)
def test_scorer_rejects_invalid_inputs(field: str, value: float) -> None:
    kwargs: dict[str, Any] = dict(
        episode_return=0.1,
        benchmark_return=0.02,
        benchmark_volatility=0.01,
        volatility_floor=0.001,
        risk_lambda=1.0,
        risk=RiskComponents(0, 0, 0),
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        score_benchmark_scaled(**kwargs)


def test_calibration_budget_weighting_and_benchmark_identity(calibrated: dict) -> None:
    statistics, reward = calibrated["statistics"], calibrated["reward"]
    assert statistics["calibration_window_count"] == 32
    assert statistics["trajectory_count"] == 1408
    assert statistics["step_count"] == 28160
    assert reward["volatility_floor"] == pytest.approx(
        0.1 * statistics["benchmark_volatility"]["median"]
    )
    assert reward["risk_lambda"] * statistics["mean_risk_penalty"] == pytest.approx(
        0.25 * statistics["mean_absolute_finance_core"]
    )
    rows = calibrated["proxy_evidence"]
    cash = [row for row in rows if row["proxy_id"] == "cash"]
    assert len(cash) == 32
    assert all(row["episode_return"] == row["risk_penalty"] == 0.0 for row in cash)
    benchmark = [row for row in rows if row["proxy_id"] == "equal_weight_100"]
    assert all(row["core"] == pytest.approx(0.0, abs=1e-12) for row in benchmark)
    assert any(row["Y"] < 0.0 for row in benchmark)
    verify_screening_calibration(calibrated)


def test_legacy_parameters_are_diagnostics_only(calibrated: dict) -> None:
    markets, benchmarks = calibration_inputs()
    result = calibrate_screening_reward(
        markets=markets,
        expected_benchmarks=benchmarks,
        initial_cash=1_000_000.0,
        execution_policy=ExecutionPolicy(10.0, 5.0),
        source={},
        legacy_reward={"epsilon": 0.8, "risk_lambda": 100.0},
    )
    assert result["reward"] == calibrated["reward"]
    assert result["statistics"] == calibrated["statistics"]
    assert "legacy_Y" in result["proxy_evidence"][0]


def test_payload_tampering_and_zero_volatility_fail_closed(calibrated: dict) -> None:
    changed = deepcopy(calibrated)
    changed["reward"]["volatility_floor"] *= 2
    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        verify_screening_calibration(changed)
    market = synthetic_market()
    flat = tuple({symbol: 100.0 for symbol in market.symbols} for _ in market.sessions)
    market = replace(market, open_prices=flat, close_prices=flat)
    markets = {key: market for key in calibrated["source"]["calibration_episode_ids"]}
    benchmarks = {key: [1_000_000.0] * 21 for key in markets}
    with pytest.raises(ValueError, match="volatility floor"):
        calibrate_screening_reward(
            markets=markets,
            expected_benchmarks=benchmarks,
            initial_cash=1_000_000.0,
            execution_policy=ExecutionPolicy(0.0, 0.0),
            source={},
        )
