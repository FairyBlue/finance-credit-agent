"""Long-only, unlevered portfolio accounting with next-open execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    commission_bps: float
    slippage_bps: float
    allow_short: bool = False
    allow_leverage: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.commission_bps) or self.commission_bps < 0.0:
            raise ValueError("commission_bps must be finite and non-negative")
        if not math.isfinite(self.slippage_bps) or self.slippage_bps < 0.0:
            raise ValueError("slippage_bps must be finite and non-negative")
        if self.allow_short or self.allow_leverage:
            raise ValueError("the v1 accounting engine forbids shorting and leverage")

    @property
    def commission_rate(self) -> float:
        return self.commission_bps / 10_000.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0


@dataclass(frozen=True, slots=True)
class PendingTargetOrder:
    symbol: str
    target_weight: float
    decision_at: datetime


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: Literal["buy", "sell"]
    shares: float
    target_weight: float
    raw_open_price: float
    execution_price: float
    notional: float
    commission: float
    slippage_cost: float
    decision_at: datetime
    executed_at: datetime

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost


class PortfolioLedger:
    """Stateful cash/position ledger shared by agents and all baselines."""

    _TOLERANCE = 1e-10

    def __init__(self, *, initial_cash: float, policy: ExecutionPolicy) -> None:
        if not math.isfinite(initial_cash) or initial_cash <= 0.0:
            raise ValueError("initial_cash must be finite and positive")
        self._initial_cash = float(initial_cash)
        self._cash = float(initial_cash)
        self._positions: dict[str, float] = {}
        self._pending: list[PendingTargetOrder] = []
        self.policy = policy
        self._fills: list[Fill] = []

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Mapping[str, float]:
        return MappingProxyType(self._positions)

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def pending_orders(self) -> tuple[PendingTargetOrder, ...]:
        return tuple(self._pending)

    def equity(self, mark_prices: Mapping[str, float]) -> float:
        holdings = 0.0
        for symbol, shares in self._positions.items():
            if shares <= self._TOLERANCE:
                continue
            price = self._validated_price(mark_prices, symbol)
            holdings += shares * price
        return self._cash + holdings

    def queue_target_order(
        self,
        symbol: str,
        *,
        target_weight: float,
        decision_at: datetime,
    ) -> PendingTargetOrder:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must not be empty")
        if not math.isfinite(target_weight) or not 0.0 <= target_weight <= 1.0:
            raise ValueError("target_weight must be within [0, 1]; shorting/leverage are forbidden")
        timestamp = _aware_utc(decision_at, name="decision_at")
        if any(order.symbol == normalized_symbol for order in self._pending):
            raise ValueError(f"a pending order already exists for {normalized_symbol}")
        order = PendingTargetOrder(normalized_symbol, float(target_weight), timestamp)
        self._pending.append(order)
        return order

    def execute_next_open(
        self,
        executed_at: datetime,
        open_prices: Mapping[str, float],
    ) -> tuple[Fill, ...]:
        """Execute queued target weights at the first supplied open after each decision."""

        timestamp = _aware_utc(executed_at, name="executed_at")
        if any(timestamp <= order.decision_at for order in self._pending):
            raise ValueError("next-open execution must occur strictly after the decision timestamp")

        # Validate every required price before mutating the ledger.
        for order in self._pending:
            self._validated_price(open_prices, order.symbol)
        for symbol, shares in self._positions.items():
            if shares > self._TOLERANCE:
                self._validated_price(open_prices, symbol)

        new_fills: list[Fill] = []
        pending = tuple(self._pending)
        for order in pending:
            raw_price = self._validated_price(open_prices, order.symbol)
            portfolio_equity = self.equity(open_prices)
            current_shares = self._positions.get(order.symbol, 0.0)
            target_shares = portfolio_equity * order.target_weight / raw_price
            delta = target_shares - current_shares
            if abs(delta) <= self._TOLERANCE:
                continue

            side: Literal["buy", "sell"] = "buy" if delta > 0.0 else "sell"
            direction = 1.0 if side == "buy" else -1.0
            execution_price = raw_price * (1.0 + direction * self.policy.slippage_rate)

            if side == "buy":
                max_affordable = self._cash / (
                    execution_price * (1.0 + self.policy.commission_rate)
                )
                delta = min(delta, max_affordable)
            else:
                delta = max(delta, -current_shares)

            absolute_shares = abs(delta)
            if absolute_shares <= self._TOLERANCE:
                continue
            notional = absolute_shares * execution_price
            commission = notional * self.policy.commission_rate
            self._cash -= delta * execution_price + commission
            updated_shares = current_shares + delta
            if updated_shares < -self._TOLERANCE:
                raise RuntimeError("accounting invariant violated: negative shares")
            self._positions[order.symbol] = max(0.0, updated_shares)
            if self._cash < -self._TOLERANCE:
                raise RuntimeError("accounting invariant violated: negative cash")
            self._cash = max(0.0, self._cash)

            fill = Fill(
                symbol=order.symbol,
                side=side,
                shares=absolute_shares,
                target_weight=order.target_weight,
                raw_open_price=raw_price,
                execution_price=execution_price,
                notional=notional,
                commission=commission,
                slippage_cost=absolute_shares * abs(execution_price - raw_price),
                decision_at=order.decision_at,
                executed_at=timestamp,
            )
            self._fills.append(fill)
            new_fills.append(fill)

        self._pending.clear()
        # TODO(human): line-review this next-open boundary and fill accounting.
        return tuple(new_fills)

    @staticmethod
    def _validated_price(prices: Mapping[str, float], symbol: str) -> float:
        if symbol not in prices:
            raise KeyError(f"missing price for held or ordered symbol {symbol}")
        price = float(prices[symbol])
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"price for {symbol} must be finite and positive")
        return price
