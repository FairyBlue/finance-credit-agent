"""The common interface explanation preserves market data and old protocols."""

from copy import deepcopy

from src.agents.screening_prompt import (
    PORTFOLIO_INTERFACE_RULES,
    SINGLE_TOOL_V1_RULES,
    action_protocol_sha256,
    apply_screening_action_protocol,
)


def test_portfolio_contract_is_idempotent_and_preserves_user_observations():
    messages = [
        {"role": "system", "content": "Manage a portfolio."},
        {"role": "user", "content": '{"positions":{"GS":{"weight":0.201}}}'},
    ]
    before = deepcopy(messages)
    rendered = apply_screening_action_protocol(messages, "single_tool_portfolio_v1")
    assert rendered == apply_screening_action_protocol(rendered, "single_tool_portfolio_v1")
    assert messages == before and rendered[1:] == before[1:]
    assert rendered[0]["content"].endswith(PORTFOLIO_INTERFACE_RULES)
    assert "0.2" not in PORTFOLIO_INTERFACE_RULES


def test_upgrading_v1_has_one_copy_of_shared_rules_and_preserves_v1_identity():
    from hashlib import sha256

    old = apply_screening_action_protocol([{"role": "user", "content": "state"}], "single_tool_v1")
    upgraded = apply_screening_action_protocol(old, "single_tool_portfolio_v1")
    assert upgraded[0]["content"].count(SINGLE_TOOL_V1_RULES) == 1
    assert (
        action_protocol_sha256("single_tool_v1")
        == sha256(SINGLE_TOOL_V1_RULES.encode()).hexdigest()
    )
    assert action_protocol_sha256("single_tool_portfolio_v1") != action_protocol_sha256(
        "single_tool_v1"
    )
