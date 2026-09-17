"""Versioned, optional action instructions shared by all screening flows."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any

SINGLE_TOOL_V1_RULES = (
    "You control a portfolio through a serial tool interface. "
    "On every turn, return exactly one tool call in the required <tool_call> JSON format, "
    "then wait for the tool result. "
    "Do not return a plan, commentary, a list of future calls, or multiple tool calls. "
    "Select the next action using the visible state and available tools. "
    "The session index counts this episode's decision steps; "
    "it does not limit the historical bars available to compute. "
    "compute uses only close prices available as of the current replay time "
    "and requires lookback between 2 and 60. "
    "Use the tool result before choosing another action. "
    "Orders fill at the next open; short selling and leverage are forbidden."
)

PORTFOLIO_INTERFACE_RULES = (
    "target_weight is the desired TOTAL fraction of portfolio equity in one symbol, "
    "not an additional amount to buy or sell. "
    "The current fraction is positions[symbol].weight; a missing position has zero weight. "
    "buy requires a target strictly greater than the CURRENT weight. "
    "sell requires an existing position and a target strictly smaller than the CURRENT weight. "
    "Market prices can move weights after an order; compare with the latest visible weight, "
    "not a previous requested target. "
    "To preserve the current portfolio, use hold; existing positions need not be resubmitted. "
    "A rejected call makes no trade and does not advance the market session. "
    "Read the error and current state, then choose a valid next action; "
    "repeating a rejected request does not complete it. "
    "Use only symbols listed in the current market state and the declared tool names."
)

_PROTOCOL_RULES = {
    "single_tool_v1": SINGLE_TOOL_V1_RULES,
    "single_tool_portfolio_v1": SINGLE_TOOL_V1_RULES + "\n\n" + PORTFOLIO_INTERFACE_RULES,
}


def validate_action_protocol(value: object) -> str:
    """Reject implicit booleans and unregistered prompt protocols."""
    if type(value) is not str or value not in {"legacy", *_PROTOCOL_RULES}:
        raise ValueError("unknown screening action protocol")
    return value


def action_protocol_sha256(protocol: object) -> str | None:
    """Identify the exact appended text; legacy adds no instruction."""
    if validate_action_protocol(protocol) == "legacy":
        return None
    return sha256(_PROTOCOL_RULES[str(protocol)].encode("ascii")).hexdigest()


def apply_screening_action_protocol(
    messages: list[dict[str, Any]], protocol: object
) -> list[dict[str, Any]]:
    """Apply one fixed instruction without mutating the environment's messages.

    Legacy returns the original input. The optional protocol copies messages,
    appends to the first system message (or inserts one), and is idempotent.
    No user/tool message, schema, trading state or tool result is rewritten.
    """
    if validate_action_protocol(protocol) == "legacy":
        return messages
    rules = _PROTOCOL_RULES[str(protocol)]
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("screening action protocol requires a list of message dictionaries")
    result = deepcopy(messages)
    for message in result:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("screening action protocol requires textual system content")
        if content != rules and not content.endswith("\n\n" + rules):
            if protocol == "single_tool_portfolio_v1" and content.endswith(SINGLE_TOOL_V1_RULES):
                message["content"] = content[: -len(SINGLE_TOOL_V1_RULES)] + rules
            else:
                message["content"] = content + ("\n\n" if content else "") + rules
        return result
    result.insert(0, {"role": "system", "content": rules})
    return result
