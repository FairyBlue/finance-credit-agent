"""Baseline replay through the exact same `PortfolioLedger` as trained agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.env import ExecutionPolicy, Fill, PortfolioLedger

from .metrics import FinancialMetrics, compute_financial_metrics


@dataclass(frozen=True, slots=True)
class ReplayResult:
    equity_curve: tuple[float, ...]
    fills: tuple[Fill, ...]
    transaction_costs: tuple[float, ...]
    traded_notionals: tuple[float, ...]

    def metrics(
        self,
        *,
        benchmark_equity_curve: Sequence[float],
        annualization_periods: int = 252,
    ) -> FinancialMetrics:
        return compute_financial_metrics(
            equity_curve=self.equity_curve,
            benchmark_equity_curve=benchmark_equity_curve,
            transaction_costs=self.transaction_costs,
            traded_notionals=self.traded_notionals,
            annualization_periods=annualization_periods,
        )


def replay_target_weights(
    *,
    session_close_times: Sequence[datetime],
    session_open_times: Sequence[datetime],
    open_prices: Sequence[Mapping[str, float]],
    close_prices: Sequence[Mapping[str, float]],
    target_weights_by_decision: Sequence[Mapping[str, float]],
    initial_cash: float,
    execution_policy: ExecutionPolicy,
) -> ReplayResult:
    """Replay any baseline/agent target schedule with after-close/next-open fills."""

    if not (
        len(session_close_times)
        == len(session_open_times)
        == len(open_prices)
        == len(close_prices)
    ):
        raise ValueError("session open/close times and price sequences must align")
    if len(session_close_times) < 2:
        raise ValueError("replay needs at least two sessions")
    if len(target_weights_by_decision) != len(session_close_times) - 1:
        raise ValueError("provide exactly one target mapping per decision session")
    if any(
        open_time >= close_time
        for open_time, close_time in zip(
            session_open_times, session_close_times, strict=True
        )
    ):
        raise ValueError("every session open must precede its close")
    if any(
        previous_close >= next_open
        for previous_close, next_open in zip(
            session_close_times[:-1], session_open_times[1:], strict=True
        )
    ):
        raise ValueError("each execution open must follow the prior decision close")
    ledger = PortfolioLedger(initial_cash=initial_cash, policy=execution_policy)
    equities = [ledger.equity(close_prices[0])]
    costs: list[float] = []
    notionals: list[float] = []
    for index, targets in enumerate(target_weights_by_decision):
        for symbol, target_weight in targets.items():
            ledger.queue_target_order(
                symbol,
                target_weight=float(target_weight),
                decision_at=session_close_times[index],
            )
        fills = ledger.execute_next_open(
            session_open_times[index + 1], open_prices[index + 1]
        )
        costs.append(sum(fill.total_cost for fill in fills))
        notionals.append(sum(fill.notional for fill in fills))
        equities.append(ledger.equity(close_prices[index + 1]))
    return ReplayResult(
        equity_curve=tuple(equities),
        fills=ledger.fills,
        transaction_costs=tuple(costs),
        traded_notionals=tuple(notionals),
    )


def buy_and_hold_equal_weight_targets(
    *, symbols: Sequence[str], decision_count: int
) -> tuple[dict[str, float], ...]:
    normalized = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not normalized:
        raise ValueError("symbols must not be empty")
    if decision_count < 1:
        raise ValueError("decision_count must be positive")
    initial_targets = {symbol: 1.0 / len(normalized) for symbol in normalized}
    return (initial_targets, *({} for _ in range(decision_count - 1)))


def all_cash_targets(*, decision_count: int) -> tuple[dict[str, float], ...]:
    if decision_count < 1:
        raise ValueError("decision_count must be positive")
    return tuple({} for _ in range(decision_count))
