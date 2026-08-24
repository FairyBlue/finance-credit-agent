"""Behavioral tests for the shared next-open portfolio ledger."""

from datetime import datetime, timezone

import pytest

from src.env import ExecutionPolicy, PortfolioLedger

UTC = timezone.utc


def test_cash_plus_holdings_value_is_conserved_net_of_explicit_costs() -> None:
    ledger = PortfolioLedger(
        initial_cash=1000.0,
        policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )
    decision_at = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)
    next_open = datetime(2025, 1, 3, 14, 30, tzinfo=UTC)
    ledger.queue_target_order("AAA", target_weight=0.5, decision_at=decision_at)

    equity_before = ledger.equity({"AAA": 100.0})
    fills = ledger.execute_next_open(next_open, {"AAA": 100.0})
    equity_after = ledger.equity({"AAA": 100.0})

    assert len(fills) == 1
    assert equity_before - equity_after == pytest.approx(
        fills[0].commission + fills[0].slippage_cost
    )
    assert ledger.cash + ledger.positions["AAA"] * 100.0 == pytest.approx(equity_after)


def test_commission_and_slippage_are_deducted_on_buys_and_sells() -> None:
    ledger = PortfolioLedger(
        initial_cash=1000.0,
        policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )
    ledger.queue_target_order(
        "AAA", target_weight=0.5, decision_at=datetime(2025, 1, 2, 21, 0, tzinfo=UTC)
    )
    buy = ledger.execute_next_open(
        datetime(2025, 1, 3, 14, 30, tzinfo=UTC), {"AAA": 100.0}
    )[0]

    assert buy.execution_price == pytest.approx(100.05)
    assert buy.commission == pytest.approx(buy.notional * 0.001)
    assert buy.slippage_cost == pytest.approx(buy.shares * 0.05)

    ledger.queue_target_order(
        "AAA", target_weight=0.0, decision_at=datetime(2025, 1, 3, 21, 0, tzinfo=UTC)
    )
    sell = ledger.execute_next_open(
        datetime(2025, 1, 6, 14, 30, tzinfo=UTC), {"AAA": 120.0}
    )[0]

    assert sell.execution_price == pytest.approx(119.94)
    assert sell.commission == pytest.approx(sell.notional * 0.001)
    assert sell.slippage_cost == pytest.approx(sell.shares * 0.06)
    assert ledger.positions["AAA"] == pytest.approx(0.0)


def test_decision_cannot_fill_on_the_same_bar_and_uses_the_next_open_price() -> None:
    ledger = PortfolioLedger(initial_cash=1000.0, policy=ExecutionPolicy(10.0, 5.0))
    decision_at = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)
    ledger.queue_target_order("AAA", target_weight=0.25, decision_at=decision_at)

    with pytest.raises(ValueError, match="strictly after"):
        ledger.execute_next_open(decision_at, {"AAA": 999.0})
    assert ledger.cash == 1000.0
    assert "AAA" not in ledger.positions

    fill = ledger.execute_next_open(
        datetime(2025, 1, 3, 14, 30, tzinfo=UTC), {"AAA": 80.0}
    )[0]
    assert fill.raw_open_price == 80.0


def test_long_only_unlevered_policy_never_allows_negative_cash_or_short_positions() -> None:
    ledger = PortfolioLedger(initial_cash=1000.0, policy=ExecutionPolicy(10.0, 5.0))
    decision_at = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="shorting/leverage"):
        ledger.queue_target_order("AAA", target_weight=-0.01, decision_at=decision_at)
    with pytest.raises(ValueError, match="shorting/leverage"):
        ledger.queue_target_order("AAA", target_weight=1.01, decision_at=decision_at)

    ledger.queue_target_order("AAA", target_weight=1.0, decision_at=decision_at)
    ledger.execute_next_open(datetime(2025, 1, 3, 14, 30, tzinfo=UTC), {"AAA": 100.0})
    assert ledger.cash >= 0.0
    assert all(shares >= 0.0 for shares in ledger.positions.values())


def test_exact_full_weight_target_tolerates_only_cash_roundoff_at_large_notional() -> None:
    ledger = PortfolioLedger(
        initial_cash=1_000_000.0,
        policy=ExecutionPolicy(commission_bps=10.0, slippage_bps=5.0),
    )
    ledger.queue_target_order(
        "AAA",
        target_weight=1.0,
        decision_at=datetime(2025, 1, 2, 21, 0, tzinfo=UTC),
    )

    ledger.execute_next_open(
        datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
        {"AAA": 98.375},
    )

    assert ledger.cash == 0.0
    assert ledger.equity({"AAA": 98.375}) < 1_000_000.0
