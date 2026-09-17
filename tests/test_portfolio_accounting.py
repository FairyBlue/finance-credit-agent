"""Behavioral tests for the shared next-open portfolio ledger."""

from datetime import datetime, timezone
from typing import Literal

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


@pytest.mark.parametrize(
    ("direction", "target", "next_price", "should_skip"),
    [
        ("buy", 0.55, 200.0, True),
        ("sell", 0.45, 50.0, True),
        ("buy", 0.55, 80.0, False),
        ("sell", 0.45, 120.0, False),
    ],
)
def test_directional_target_never_fills_the_opposite_side_after_a_price_gap(
    direction: Literal["buy", "sell"], target: float, next_price: float, should_skip: bool,
) -> None:
    ledger = PortfolioLedger(initial_cash=1000.0, policy=ExecutionPolicy(10.0, 5.0))
    ledger.queue_target_order(
        "AAA", target_weight=0.5, decision_at=datetime(2025, 1, 2, 21, tzinfo=UTC),
    )
    ledger.execute_next_open(datetime(2025, 1, 3, 14, 30, tzinfo=UTC), {"AAA": 100.0})
    cash_before = ledger.cash
    positions_before = dict(ledger.positions)
    fills_before = ledger.fills
    equity_at_open = ledger.equity({"AAA": next_price})
    order = ledger.queue_target_order(
        "AAA", target_weight=target, decision_at=datetime(2025, 1, 3, 21, tzinfo=UTC),
        direction=direction,
    )
    assert order.direction == direction

    fills = ledger.execute_next_open(
        datetime(2025, 1, 6, 14, 30, tzinfo=UTC), {"AAA": next_price},
    )

    assert ledger.pending_orders == ()
    if should_skip:
        assert fills == ()
        assert ledger.fills == fills_before
        assert ledger.cash == cash_before
        assert dict(ledger.positions) == positions_before
        assert ledger.equity({"AAA": next_price}) == equity_at_open
        skip, = ledger.last_execution_skips
        assert skip.requested_side == direction
        assert skip.implied_side == ("sell" if direction == "buy" else "buy")
        assert skip.reason == "opposite_direction_at_next_open"
        assert skip.current_weight_at_open == pytest.approx(
            positions_before["AAA"] * next_price / equity_at_open
        )
        # It is a consumed one-open order, never a delayed opposite-side fill.
        assert ledger.execute_next_open(
            datetime(2025, 1, 7, 14, 30, tzinfo=UTC), {"AAA": 100.0},
        ) == ()
        assert ledger.last_execution_skips == ()
    else:
        fill, = fills
        assert fill.side == direction
        assert fill.total_cost > 0
        assert ledger.last_execution_skips == ()
        assert equity_at_open - ledger.equity({"AAA": next_price}) == pytest.approx(
            fill.total_cost
        )


@pytest.mark.parametrize(
    ("target", "next_price", "expected_side"),
    [(0.55, 200.0, "sell"), (0.45, 50.0, "buy")],
)
def test_generic_baseline_targets_keep_bidirectional_rebalancing(
    target: float, next_price: float, expected_side: str,
) -> None:
    ledger = PortfolioLedger(initial_cash=1000.0, policy=ExecutionPolicy(10.0, 5.0))
    ledger.queue_target_order(
        "AAA", target_weight=0.5, decision_at=datetime(2025, 1, 2, 21, tzinfo=UTC),
    )
    ledger.execute_next_open(datetime(2025, 1, 3, 14, 30, tzinfo=UTC), {"AAA": 100.0})
    order = ledger.queue_target_order(
        "AAA", target_weight=target, decision_at=datetime(2025, 1, 3, 21, tzinfo=UTC),
    )

    fill, = ledger.execute_next_open(
        datetime(2025, 1, 6, 14, 30, tzinfo=UTC), {"AAA": next_price},
    )

    assert order.direction is None
    assert fill.side == expected_side
    assert ledger.last_execution_skips == ()
