"""Shared metric behavior for trained arms and accounting-engine baselines."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.env import ExecutionPolicy
from src.evaluation import (
    agent_metrics_by_regime,
    all_cash_targets,
    compute_agent_metrics,
    compute_financial_metrics,
    metrics_by_regime,
    plot_equity_vs_buy_and_hold,
    plot_reward_vs_step,
    plot_tool_call_success,
    replay_target_weights,
)


def test_financial_metrics_use_one_equity_curve_and_benchmark_path() -> None:
    metrics = compute_financial_metrics(
        equity_curve=[100.0, 110.0, 99.0, 120.0],
        benchmark_equity_curve=[100.0, 102.0, 103.0, 105.0],
        transaction_costs=[0.1, 0.2, 0.0],
        traded_notionals=[50.0, 20.0, 0.0],
        annualization_periods=252,
    )

    assert metrics.cumulative_return == pytest.approx(0.20)
    assert metrics.maximum_drawdown == pytest.approx(0.10)
    assert metrics.benchmark_adjusted_return == pytest.approx(0.15)
    assert metrics.cost_ratio == pytest.approx(0.003)
    assert metrics.turnover == pytest.approx(0.70)


def test_agent_metrics_count_invalid_calls_abstentions_and_tokens() -> None:
    metrics = compute_agent_metrics(
        [
            {"tool": "retrieve", "valid": True, "tokens": 100},
            {"tool": "buy", "valid": False, "tokens": 50},
            {"tool": "abstain", "valid": True, "tokens": 25},
        ]
    )
    assert metrics.tool_calls == 3
    assert metrics.invalid_call_rate == pytest.approx(1 / 3)
    assert metrics.abstain_rate == pytest.approx(1 / 3)
    assert metrics.token_consumption == 175


def test_regime_scorecards_include_finance_and_agent_metrics() -> None:
    finance = metrics_by_regime(
        regime_labels=["up", "up", "down"],
        portfolio_returns=[0.10, -0.05, -0.10],
        benchmark_returns=[0.02, 0.01, -0.02],
        transaction_cost_fractions=[0.001, 0.002, 0.003],
        turnover_fractions=[0.1, 0.2, 0.3],
    )
    agent = agent_metrics_by_regime(
        [
            {"regime": "up", "tool": "buy", "valid": True, "tokens": 10},
            {"regime": "down", "tool": "abstain", "valid": True, "tokens": 5},
        ]
    )

    assert finance["up"]["turnover"] == pytest.approx(0.3)
    assert finance["up"]["cost_ratio"] == pytest.approx(0.003)
    assert "sortino" in finance["down"]
    assert agent["down"].abstain_rate == 1.0


def test_three_standard_plots_render_to_files(tmp_path: Path) -> None:
    outputs = [
        plot_reward_vs_step(
            {"A1": [(0, 0.0), (1, 0.1)], "A3": [(0, 0.0), (1, 0.2)]},
            output_path=tmp_path / "reward.png",
        ),
        plot_tool_call_success(
            {"A1": 0.8, "A2": 0.9, "A3": 1.0},
            output_path=tmp_path / "tools.png",
        ),
        plot_equity_vs_buy_and_hold(
            dates=["2025-01-02", "2025-01-03"],
            equity_curves={"A3": [100.0, 102.0], "buy-and-hold": [100.0, 101.0]},
            output_path=tmp_path / "equity.png",
        ),
    ]
    assert all(output.is_file() and output.stat().st_size > 0 for output in outputs)


def test_all_cash_baseline_runs_through_shared_next_open_ledger() -> None:
    closes = [
        datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 3, 21, 0, tzinfo=timezone.utc),
    ]
    opens = [
        datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 3, 14, 30, tzinfo=timezone.utc),
    ]
    result = replay_target_weights(
        session_close_times=closes,
        session_open_times=opens,
        open_prices=[{"AAA": 100.0}, {"AAA": 110.0}],
        close_prices=[{"AAA": 105.0}, {"AAA": 112.0}],
        target_weights_by_decision=all_cash_targets(decision_count=1),
        initial_cash=1000.0,
        execution_policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )
    assert result.equity_curve == (1000.0, 1000.0)
    assert result.fills == ()
