"""One metric implementation shared by arms and every baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FinancialMetrics:
    cumulative_return: float
    annualized_return: float
    sharpe: float
    sortino: float
    maximum_drawdown: float
    turnover: float
    cost_ratio: float
    benchmark_adjusted_return: float


@dataclass(frozen=True, slots=True)
class AgentMetrics:
    tool_calls: int
    successful_tool_call_rate: float
    invalid_call_rate: float
    abstain_rate: float
    token_consumption: int
    average_reasoning_rounds: float


def _equity_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def compute_financial_metrics(
    *,
    equity_curve: Sequence[float],
    benchmark_equity_curve: Sequence[float],
    transaction_costs: Sequence[float],
    traded_notionals: Sequence[float],
    annualization_periods: int = 252,
) -> FinancialMetrics:
    """Compute the protocol's financial metrics from accounting-engine outputs.

    Turnover and cost ratio use initial equity as their auditable denominator;
    the same convention is applied to every arm and baseline.
    """

    equity = _equity_array(equity_curve, name="equity_curve")
    benchmark = _equity_array(benchmark_equity_curve, name="benchmark_equity_curve")
    if equity.size != benchmark.size:
        raise ValueError("portfolio and benchmark equity curves must align")
    if annualization_periods <= 0:
        raise ValueError("annualization_periods must be positive")
    costs = np.asarray(transaction_costs, dtype=np.float64)
    notionals = np.asarray(traded_notionals, dtype=np.float64)
    if np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("transaction_costs must be finite and non-negative")
    if np.any(~np.isfinite(notionals)) or np.any(notionals < 0.0):
        raise ValueError("traded_notionals must be finite and non-negative")

    returns = equity[1:] / equity[:-1] - 1.0
    cumulative = float(equity[-1] / equity[0] - 1.0)
    benchmark_cumulative = float(benchmark[-1] / benchmark[0] - 1.0)
    annualized = float((equity[-1] / equity[0]) ** (annualization_periods / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / volatility * math.sqrt(annualization_periods))
        if volatility > 0
        else 0.0
    )
    downside = returns[returns < 0.0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
    sortino = (
        float(returns.mean() / downside_deviation * math.sqrt(annualization_periods))
        if downside_deviation > 0.0
        else 0.0
    )
    peaks = np.maximum.accumulate(equity)
    maximum_drawdown = float(np.max((peaks - equity) / peaks))
    return FinancialMetrics(
        cumulative_return=cumulative,
        annualized_return=annualized,
        sharpe=sharpe,
        sortino=sortino,
        maximum_drawdown=maximum_drawdown,
        turnover=float(notionals.sum() / equity[0]),
        cost_ratio=float(costs.sum() / equity[0]),
        benchmark_adjusted_return=cumulative - benchmark_cumulative,
    )


def compute_agent_metrics(records: Sequence[Mapping[str, object]]) -> AgentMetrics:
    tool_calls = len(records)
    invalid = sum(not bool(record.get("valid", True)) for record in records)
    abstentions = sum(str(record.get("tool", "")) == "abstain" for record in records)
    tokens = 0
    for record in records:
        raw_tokens = record.get("tokens", 0)
        if not isinstance(raw_tokens, (int, float, str)):
            raise ValueError("tokens must be numeric")
        tokens += int(raw_tokens)
    episode_ids = {record.get("episode_id") for record in records if "episode_id" in record}
    episode_count = len(episode_ids) if episode_ids else (1 if records else 0)
    average_rounds = tool_calls / episode_count if episode_count else 0.0
    denominator = tool_calls or 1
    return AgentMetrics(
        tool_calls=tool_calls,
        successful_tool_call_rate=(tool_calls - invalid) / denominator,
        invalid_call_rate=invalid / denominator,
        abstain_rate=abstentions / denominator,
        token_consumption=tokens,
        average_reasoning_rounds=average_rounds,
    )


def agent_metrics_by_regime(
    records: Sequence[Mapping[str, object]],
) -> dict[str, AgentMetrics]:
    """Compute the same agent scorecard within every frozen regime label."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        regime = record.get("regime")
        if not isinstance(regime, str) or not regime:
            raise ValueError("every agent record needs a non-empty regime label")
        grouped.setdefault(regime, []).append(record)
    return {regime: compute_agent_metrics(group) for regime, group in grouped.items()}


def metrics_by_regime(
    *,
    regime_labels: Sequence[str],
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    transaction_cost_fractions: Sequence[float] | None = None,
    turnover_fractions: Sequence[float] | None = None,
    annualization_periods: int = 252,
) -> dict[str, dict[str, float]]:
    """Report the complete aligned finance scorecard for every frozen regime."""

    labels = np.asarray(regime_labels, dtype=object)
    portfolio = np.asarray(portfolio_returns, dtype=np.float64)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64)
    if not (len(labels) == len(portfolio) == len(benchmark)):
        raise ValueError("regime labels and return arrays must align")
    if (
        np.any(~np.isfinite(portfolio))
        or np.any(~np.isfinite(benchmark))
        or np.any(portfolio <= -1.0)
        or np.any(benchmark <= -1.0)
    ):
        raise ValueError("regime returns must be finite and greater than -1")
    costs = np.asarray(
        transaction_cost_fractions
        if transaction_cost_fractions is not None
        else np.zeros(len(labels)),
        dtype=np.float64,
    )
    turnover = np.asarray(
        turnover_fractions if turnover_fractions is not None else np.zeros(len(labels)),
        dtype=np.float64,
    )
    if not (len(costs) == len(turnover) == len(labels)):
        raise ValueError("regime cost/turnover fractions must align with returns")
    if (
        np.any(~np.isfinite(costs))
        or np.any(~np.isfinite(turnover))
        or np.any(costs < 0.0)
        or np.any(turnover < 0.0)
    ):
        raise ValueError("regime cost/turnover fractions must be finite and non-negative")
    output: dict[str, dict[str, float]] = {}
    for label in dict.fromkeys(str(value) for value in labels):
        mask = labels == label
        values = portfolio[mask]
        benchmark_values = benchmark[mask]
        cumulative = float(np.prod(1.0 + values) - 1.0)
        benchmark_cumulative = float(np.prod(1.0 + benchmark_values) - 1.0)
        volatility = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        downside = values[values < 0.0]
        downside_deviation = (
            float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
        )
        regime_equity = np.concatenate(([1.0], np.cumprod(1.0 + values)))
        peaks = np.maximum.accumulate(regime_equity)
        output[label] = {
            "cumulative_return": cumulative,
            "annualized_return": float(
                (1.0 + cumulative) ** (annualization_periods / len(values)) - 1.0
            ),
            "benchmark_adjusted_return": cumulative - benchmark_cumulative,
            "sharpe": (
                float(values.mean() / volatility * math.sqrt(annualization_periods))
                if volatility > 0.0
                else 0.0
            ),
            "sortino": (
                float(values.mean() / downside_deviation * math.sqrt(annualization_periods))
                if downside_deviation > 0.0
                else 0.0
            ),
            "maximum_drawdown": float(np.max((peaks - regime_equity) / peaks)),
            "turnover": float(turnover[mask].sum()),
            "cost_ratio": float(costs[mask].sum()),
            "periods": float(len(values)),
        }
    return output
