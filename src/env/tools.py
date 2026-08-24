"""Agent-R1 `BaseTool` registrations for the trading action space."""

from __future__ import annotations

from typing import Any

from agent_r1.tool import BaseTool, ToolResponse


class _IntentTool(BaseTool):
    """Schema-bearing tool; `TradingEnv` performs stateful execution serially."""

    async def execute(
        self, args: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float | None, dict]:
        return ToolResponse(text=f"{self.name} request accepted"), None, {"intent": dict(args)}


@BaseTool.register("retrieve")
class RetrieveTool(_IntentTool):
    name = "retrieve"
    description = (
        "Retrieve only market, audited PIT fundamental, or causal technical observations "
        "available at the current replay time. News is disabled in v1."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Case-insensitive text query."},
            "symbol": {"type": "string", "description": "Optional ticker filter."},
            "limit": {"type": "integer", "description": "Maximum results (1-50)."},
        },
        "required": ["query"],
    }


@BaseTool.register("compute")
class ComputeTool(_IntentTool):
    name = "compute"
    description = "Compute a point-in-time statistic from already available close prices."
    parameters = {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "Statistic to compute.",
                "enum": ["price_change", "mean_return", "volatility"],
            },
            "symbol": {"type": "string", "description": "Ticker."},
            "lookback": {"type": "integer", "description": "Visible bars to use (2-60)."},
        },
        "required": ["metric", "symbol", "lookback"],
    }


_TARGET_PARAMETERS = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string", "description": "Ticker."},
        "target_weight": {
            "type": "number",
            "description": "Long-only target portfolio weight in [0, 1].",
        },
    },
    "required": ["symbol", "target_weight"],
}


@BaseTool.register("buy")
class BuyTool(_IntentTool):
    name = "buy"
    description = "Raise a symbol to a target long-only weight; fill at the next open."
    parameters = _TARGET_PARAMETERS


@BaseTool.register("sell")
class SellTool(_IntentTool):
    name = "sell"
    description = "Lower a symbol to a target long-only weight; fill at the next open."
    parameters = _TARGET_PARAMETERS


@BaseTool.register("hold")
class HoldTool(_IntentTool):
    name = "hold"
    description = "Make no trade and advance to the next market session."
    parameters = {"type": "object", "properties": {}, "required": []}


@BaseTool.register("abstain")
class AbstainTool(_IntentTool):
    name = "abstain"
    description = "Explicitly decline to trade because available evidence is insufficient."
    parameters = {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Short reason for abstaining."},
        },
        "required": [],
    }
