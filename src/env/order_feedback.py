"""Optional direction-rejection context; this module never chooses an action."""

from __future__ import annotations

ORDER_DIRECTION_TOLERANCE = 1e-12
MAX_CONSECUTIVE_REJECTIONS = 1000


class OrderDirectionError(ValueError):
    """Retain the checked values without changing the historical error text."""

    def __init__(
        self,
        *,
        name: str,
        symbol: str,
        target_weight: float,
        current_weight: float,
    ) -> None:
        self.name = name
        self.symbol = symbol
        self.target_weight = target_weight
        self.current_weight = current_weight
        relation = "above" if name == "buy" else "below"
        super().__init__(f"{name} target_weight must be {relation} the current weight")

    def request_key(self, session_index: int) -> tuple[object, ...]:
        return session_index, self.name, self.symbol, self.target_weight

    def feedback(self, *, session_index: int, consecutive_same_request: int) -> dict:
        is_buy = self.name == "buy"
        threshold = self.current_weight + (
            ORDER_DIRECTION_TOLERANCE if is_buy else -ORDER_DIRECTION_TOLERANCE
        )
        return {
            "version": "order_rejection_v1",
            "rejected_call": {
                "name": self.name,
                "arguments": {"symbol": self.symbol, "target_weight": self.target_weight},
            },
            "current_weight": self.current_weight,
            "current_weight_precision": "unrounded",
            "comparison_tolerance": ORDER_DIRECTION_TOLERANCE,
            "required_target": {
                "operator": ">" if is_buy else "<",
                "threshold": threshold,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "same_action_has_valid_target": threshold < 1.0 if is_buy else threshold > 0.0,
            "session_index": session_index,
            "session_advanced": False,
            "trade_executed": False,
            "consecutive_same_request": consecutive_same_request,
            "consecutive_same_request_limit": MAX_CONSECUTIVE_REJECTIONS,
            "possible_next_actions": [
                "Choose a revised target satisfying required_target within [0, 1], if possible.",
                "Choose hold to preserve the current portfolio and advance the session.",
                "Choose abstain to decline trading and advance the session.",
            ],
        }

    def feedback_v2(self, *, session_index: int, consecutive_same_request: int) -> dict:
        """Explain the requested endpoint without submitting or selecting an order.

        The direction is a current-state API fact, not an investment target.
        Compare against the exact validator thresholds rather than subtracting
        first: floating subtraction can differ at a representable boundary.
        """
        if self.target_weight > self.current_weight + ORDER_DIRECTION_TOLERANCE:
            direction = "buy"
        elif self.target_weight < self.current_weight - ORDER_DIRECTION_TOLERANCE:
            direction = "sell"
        else:
            direction = "no_trade"
        example = (
            {
                "name": direction,
                "arguments": {"symbol": self.symbol, "target_weight": self.target_weight},
            }
            if direction != "no_trade"
            else {"name": "hold", "arguments": {}}
        )
        result = self.feedback(
            session_index=session_index,
            consecutive_same_request=consecutive_same_request,
        )
        result.update(
            version="order_rejection_v2",
            target_weight_semantics=(
                "desired TOTAL portfolio fraction, not an additional trade amount"
            ),
            requested_weight_delta=self.target_weight - self.current_weight,
            direction_for_requested_target=direction,
            directionally_valid_call_example=example,
            example_scope=(
                "Read-only API illustration for the same requested target, not a selected or "
                "executed action. Next-open prices, cash, costs and execution checks still apply; "
                "a fill or exact resulting weight is not guaranteed. For no_trade, hold preserves "
                "the current portfolio; it does not rebalance to an exact requested target."
            ),
            possible_next_actions=[
                "If the same TOTAL target is still desired, its direction is shown by "
                "direction_for_requested_target; the example is not automatically submitted.",
                *result["possible_next_actions"],
            ],
        )
        return result
