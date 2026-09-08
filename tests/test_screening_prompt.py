"""The optional serial-action instruction must not rewrite conversation state."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from src.agents.screening_prompt import (
    SINGLE_TOOL_V1_RULES,
    action_protocol_sha256,
    apply_screening_action_protocol,
    validate_action_protocol,
)


@pytest.mark.parametrize("value", [None, True, False, 0, 1, "", "single_tool", {}, []])
def test_unregistered_or_implicit_protocols_are_rejected(value):
    for function in (validate_action_protocol, action_protocol_sha256):
        with pytest.raises(ValueError, match="action protocol"):
            function(value)
    with pytest.raises(ValueError, match="action protocol"):
        apply_screening_action_protocol([], value)


def test_legacy_preserves_the_exact_input_and_adds_no_text_identity():
    messages = [{"role": "user", "content": "Original state", "metadata": {"step": 3}}]
    assert apply_screening_action_protocol(messages, "legacy") is messages
    assert validate_action_protocol("legacy") == "legacy"
    assert action_protocol_sha256("legacy") is None


@pytest.mark.parametrize("system_content", ["Original system rules.", ""])
def test_first_system_is_extended_once_without_losing_other_messages(system_content):
    messages = [
        {"role": "user", "content": "Earlier state"},
        {"role": "system", "content": system_content, "name": "original"},
        {"role": "assistant", "content": "<tool_call>old call</tool_call>"},
        {"role": "tool", "content": [{"type": "text", "text": "Original result"}]},
        {"role": "system", "content": "Additional original rules"},
        {"role": "user", "content": "Current state"},
    ]
    before = deepcopy(messages)
    result = apply_screening_action_protocol(messages, "single_tool_v1")
    assert messages == before
    assert len(result) == len(messages)
    assert result[1] == {
        **messages[1],
        "content": system_content + ("\n\n" if system_content else "") + SINGLE_TOOL_V1_RULES,
    }
    assert result[:1] + result[2:] == messages[:1] + messages[2:]
    assert apply_screening_action_protocol(result, "single_tool_v1") == result
    result[3]["content"][0]["text"] = "Changed copy"
    assert messages == before


def test_user_only_observation_gets_one_system_message_without_accumulation():
    messages = [
        {"role": "user", "content": "Manage portfolio"},
        {"role": "user", "content": "State"},
    ]
    first = apply_screening_action_protocol(messages, "single_tool_v1")
    assert first == [{"role": "system", "content": SINGLE_TOOL_V1_RULES}, *messages]
    for _ in range(3):
        first = apply_screening_action_protocol(first, "single_tool_v1")
    assert len(first) == 3
    assert len(messages) == 2


def test_protocol_hash_binds_ascii_instruction_text_and_not_the_conversation():
    assert SINGLE_TOOL_V1_RULES.isascii()
    assert action_protocol_sha256("single_tool_v1") == sha256(
        SINGLE_TOOL_V1_RULES.encode("ascii")
    ).hexdigest()
    assert "this episode's decision steps" in SINGLE_TOOL_V1_RULES


@pytest.mark.parametrize("content", [None, ["system text"], {"text": "system text"}])
def test_nontext_system_is_rejected_instead_of_dropping_or_stringifying_it(content):
    messages = [{"role": "system", "content": content}]
    with pytest.raises(ValueError, match="textual system"):
        apply_screening_action_protocol(messages, "single_tool_v1")
    assert messages == [{"role": "system", "content": content}]
