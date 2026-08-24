"""Pre-registered finance-native trajectory advantage."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite_non_negative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class RiskComponents:
    """Normalized components used by the v1 `max(...)` risk penalty."""

    max_drawdown: float
    turnover_excess: float
    position_excess: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_drawdown",
            _finite_non_negative(self.max_drawdown, name="max_drawdown"),
        )
        object.__setattr__(
            self,
            "turnover_excess",
            _finite_non_negative(self.turnover_excess, name="turnover_excess"),
        )
        object.__setattr__(
            self,
            "position_excess",
            _finite_non_negative(self.position_excess, name="position_excess"),
        )

    @property
    def penalty(self) -> float:
        return max(self.max_drawdown, self.turnover_excess, self.position_excess)


@dataclass(frozen=True, slots=True)
class FinanceAdvantage:
    advantage: float
    excess_return: float
    risk_adjusted_excess_return: float
    risk_penalty: float


def finance_advantage(
    *,
    episode_return: float,
    benchmark_return: float,
    volatility: float,
    epsilon: float,
    risk_lambda: float,
    risk: RiskComponents,
) -> FinanceAdvantage:
    """Compute ``(R-B)/(sigma+epsilon) - lambda*RiskPenalty`` exactly.

    `epsilon` and `risk_lambda` are mandatory because the frozen brief does not
    assign them numerical values.
    """

    numeric = {
        "episode_return": float(episode_return),
        "benchmark_return": float(benchmark_return),
        "volatility": float(volatility),
        "epsilon": float(epsilon),
        "risk_lambda": float(risk_lambda),
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("finance advantage inputs must be finite")
    if numeric["volatility"] < 0.0:
        raise ValueError("volatility must be non-negative")
    if numeric["epsilon"] <= 0.0:
        raise ValueError("epsilon must be strictly positive")
    if numeric["risk_lambda"] < 0.0:
        raise ValueError("risk_lambda must be non-negative")

    excess = numeric["episode_return"] - numeric["benchmark_return"]
    adjusted = excess / (numeric["volatility"] + numeric["epsilon"])
    advantage = adjusted - numeric["risk_lambda"] * risk.penalty
    return FinanceAdvantage(
        advantage=advantage,
        excess_return=excess,
        risk_adjusted_excess_return=adjusted,
        risk_penalty=risk.penalty,
    )
