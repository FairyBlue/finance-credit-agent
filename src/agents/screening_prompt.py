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


def validate_action_protocol(value: object) -> str:
    """Reject implicit booleans and unregistered prompt protocols."""
    if type(value) is not str or value not in {"legacy", "single_tool_v1"}:
        raise ValueError("screening action protocol must be 'legacy' or 'single_tool_v1'")
    return value


def action_protocol_sha256(protocol: object) -> str | None:
    """Identify the exact appended text; legacy adds no instruction."""
    if validate_action_protocol(protocol) == "legacy":
        return None
    return sha256(SINGLE_TOOL_V1_RULES.encode("ascii")).hexdigest()


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
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("screening action protocol requires a list of message dictionaries")
    result = deepcopy(messages)
    for message in result:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("screening action protocol requires textual system content")
        if content != SINGLE_TOOL_V1_RULES and not content.endswith("\n\n" + SINGLE_TOOL_V1_RULES):
            message["content"] = content + ("\n\n" if content else "") + SINGLE_TOOL_V1_RULES
        return result
    result.insert(0, {"role": "system", "content": SINGLE_TOOL_V1_RULES})
    return result
