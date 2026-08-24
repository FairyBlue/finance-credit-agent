"""Numerical specification for finance-native rewards and step allocation."""

import pytest

from src.rewards import RiskComponents, StepContext, UniformBroadcastCredit, finance_advantage


def test_finance_advantage_matches_pre_registered_formula() -> None:
    result = finance_advantage(
        episode_return=0.12,
        benchmark_return=0.04,
        volatility=0.20,
        epsilon=0.01,
        risk_lambda=0.5,
        risk=RiskComponents(max_drawdown=0.10, turnover_excess=0.20, position_excess=0.05),
    )

    assert result.risk_penalty == 0.20
    assert result.advantage == pytest.approx((0.12 - 0.04) / (0.20 + 0.01) - 0.5 * 0.20)


def test_near_zero_volatility_is_stabilized_by_explicit_epsilon() -> None:
    result = finance_advantage(
        episode_return=0.01,
        benchmark_return=0.0,
        volatility=1e-15,
        epsilon=1e-3,
        risk_lambda=0.0,
        risk=RiskComponents(0.0, 0.0, 0.0),
    )
    assert result.advantage == pytest.approx(0.01 / (1e-15 + 1e-3))


def test_all_cash_episode_is_finite_and_retains_benchmark_opportunity_cost() -> None:
    result = finance_advantage(
        episode_return=0.0,
        benchmark_return=0.02,
        volatility=0.0,
        epsilon=0.01,
        risk_lambda=1.0,
        risk=RiskComponents(0.0, 0.0, 0.0),
    )
    assert result.advantage == pytest.approx(-2.0)
    assert result.risk_penalty == 0.0


def test_m1_uniform_broadcast_conserves_trajectory_advantage() -> None:
    steps = [
        StepContext(step_id="s1", action="retrieve"),
        StepContext(step_id="s2", action="compute"),
        StepContext(step_id="s3", action="buy"),
    ]

    credits = UniformBroadcastCredit().allocate(trajectory_advantage=0.9, steps=steps)

    assert [credit.credit for credit in credits] == pytest.approx([0.3, 0.3, 0.3])
    assert sum(credit.credit for credit in credits) == pytest.approx(0.9)
