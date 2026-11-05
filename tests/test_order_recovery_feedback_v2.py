"""Endpoint explanations are read-only API facts with unchanged execution rules."""

from __future__ import annotations

import asyncio
import json
import math
from hashlib import sha256

import pytest
from agent_r1.env.base import Action
from test_order_recovery_feedback import _call, _env, _ledger_state, _state

from src.env.observation_memory import (
    FEATURE_OBSERVATION_PROFILES,
    MEMORY_OBSERVATION_PROFILES,
    RECOVERY_OBSERVATION_PROFILE,
    RECOVERY_V2_OBSERVATION_PROFILE,
    validate_observation_profile,
)
from src.env.order_feedback import ORDER_DIRECTION_TOLERANCE, OrderDirectionError


@pytest.mark.parametrize(
    ("name", "current", "target", "direction", "example_name"),
    [
        ("buy", 0.3, 0.2, "sell", "sell"),
        ("sell", 0.3, 0.4, "buy", "buy"),
        ("buy", 0.3, 0.3, "no_trade", "hold"),
        ("sell", 0.0, 0.0, "no_trade", "hold"),
        ("sell", 0.0, 0.2, "buy", "buy"),
        ("buy", 1.0, 1.0, "no_trade", "hold"),
    ],
)
def test_requested_endpoint_direction_is_symmetric_and_does_not_choose_a_target(
    name,
    current,
    target,
    direction,
    example_name,
):
    error = OrderDirectionError(
        name=name,
        symbol="UNSEEN_SYMBOL",
        target_weight=target,
        current_weight=current,
    )
    old = error.feedback(session_index=17, consecutive_same_request=2)
    feedback = error.feedback_v2(session_index=17, consecutive_same_request=2)
    assert feedback["version"] == "order_rejection_v2"
    assert feedback["direction_for_requested_target"] == direction
    assert feedback["requested_weight_delta"] == target - current
    assert "TOTAL" in feedback["target_weight_semantics"]
    assert feedback["session_advanced"] is feedback["trade_executed"] is False
    example = feedback["directionally_valid_call_example"]
    assert example["name"] == example_name
    assert example["arguments"] == (
        {} if direction == "no_trade" else {"symbol": "UNSEEN_SYMBOL", "target_weight": target}
    )
    assert "not a selected or executed action" in feedback["example_scope"]
    assert "not guaranteed" in feedback["example_scope"]
    for key, value in old.items():
        if key not in {"version", "possible_next_actions"}:
            assert feedback[key] == value
    assert error.feedback(session_index=17, consecutive_same_request=2) == old


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_direction_uses_validator_thresholds_including_representable_tolerance_boundary(side):
    current = 0.5
    boundary = current + side * ORDER_DIRECTION_TOLERANCE
    name = "sell" if side > 0 else "buy"
    on_boundary = OrderDirectionError(
        name=name,
        symbol="AAA",
        target_weight=boundary,
        current_weight=current,
    ).feedback_v2(session_index=0, consecutive_same_request=1)
    assert on_boundary["direction_for_requested_target"] == "no_trade"
    beyond = math.nextafter(boundary, 1.0 if side > 0 else 0.0)
    outside_boundary = OrderDirectionError(
        name=name,
        symbol="AAA",
        target_weight=beyond,
        current_weight=current,
    ).feedback_v2(session_index=0, consecutive_same_request=1)
    assert outside_boundary["direction_for_requested_target"] == ("buy" if side > 0 else "sell")


@pytest.mark.parametrize(
    ("name", "target", "direction"), [("buy", 0.4, "sell"), ("sell", 0.6, "buy")]
)
def test_real_rejection_never_executes_the_example_until_explicitly_chosen(name, target, direction):
    async def exercise():
        env = _env(RECOVERY_V2_OBSERVATION_PROFILE)
        env.reset()
        await env.step(_call("buy", 0.5))
        before = _ledger_state(env)
        observation, reward, done, info = await env.step(_call(name, target))
        feedback = _state(observation)["order_rejection"]
        assert info["invalid_action"] and reward == 0.0 and not done
        assert _ledger_state(env) == before
        assert feedback["direction_for_requested_target"] == direction
        assert feedback["requested_weight_delta"] == target - env._current_weight("AAA")
        assert env.credit_step_trace[-1].action == "invalid"
        assert env.ledger.pending_orders == ()
        # A separate policy action is required; the error handler does not
        # manufacture, queue or apply the example order.
        chosen = feedback["directionally_valid_call_example"]
        observation, _, _, info = await env.step(
            Action(
                text="<tool_call>" + json.dumps(chosen) + "</tool_call>",
            )
        )
        assert not info["invalid_action"]
        assert env.current_session_index == before[0] + 1
        assert info["fills"][0]["side"] == direction
        assert "order_rejection" not in _state(observation)

    asyncio.run(exercise())


def test_tolerance_no_trade_example_preserves_current_portfolio_only_when_explicitly_chosen():
    async def exercise():
        env = _env(RECOVERY_V2_OBSERVATION_PROFILE, close_after_buy=100.0)
        env.reset()
        await env.step(_call("buy", 0.5))
        before = _ledger_state(env)
        observation, _, _, info = await env.step(_call("buy", 0.5))
        feedback = _state(observation)["order_rejection"]
        assert info["invalid_action"] and _ledger_state(env) == before
        assert feedback["direction_for_requested_target"] == "no_trade"
        assert feedback["directionally_valid_call_example"] == {"name": "hold", "arguments": {}}
        assert "does not rebalance" in feedback["example_scope"]
        assert env.current_session_index == before[0]
        # This test explicitly selects hold; no fallback selected it for us.
        observation, _, _, info = await env.step(_call("hold"))
        assert env.current_session_index == before[0] + 1
        assert info["fills"] == []
        assert env.ledger.cash == before[2] and dict(env.ledger.positions) == before[3]
        assert "order_rejection" not in _state(observation)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("name", "target", "next_open"), [("buy", 0.4, 50.0), ("sell", 0.6, 200.0)]
)
def test_read_only_direction_example_does_not_override_next_open_skip(name, target, next_open):
    async def exercise():
        env = _env(RECOVERY_V2_OBSERVATION_PROFILE, close_after_buy=100.0, next_open=next_open)
        env.reset()
        await env.step(_call("buy", 0.5))
        observation, *_ = await env.step(_call(name, target))
        before = _ledger_state(env)
        example = _state(observation)["order_rejection"]["directionally_valid_call_example"]
        _, _, _, info = await env.step(
            Action(text="<tool_call>" + json.dumps(example) + "</tool_call>")
        )
        assert not info["invalid_action"]
        assert info["fills"] == []
        assert info["skipped_orders"][0]["reason"] == "opposite_direction_at_next_open"
        assert env.current_session_index == before[0] + 1
        assert env.ledger.cash == before[2] and dict(env.ledger.positions) == before[3]

    asyncio.run(exercise())


def test_v2_inherits_features_memory_and_economic_state_while_only_feedback_changes():
    async def exercise():
        new, old = _env(RECOVERY_V2_OBSERVATION_PROFILE), _env(RECOVERY_OBSERVATION_PROFILE)
        assert (
            validate_observation_profile(RECOVERY_V2_OBSERVATION_PROFILE) == new.observation_profile
        )
        assert RECOVERY_V2_OBSERVATION_PROFILE in FEATURE_OBSERVATION_PROFILES
        assert RECOVERY_V2_OBSERVATION_PROFILE in MEMORY_OBSERVATION_PROFILES
        assert new.reset().messages == old.reset().messages
        for action in (
            _call("buy", 0.5),
            Action(
                text='<tool_call>{"name":"compute",'
                '"arguments":{"metric":"mean_return",'
                '"symbol":"AAA","lookback":2}}</tool_call>'
            ),
        ):
            assert await new.step(action) == await old.step(action)
        new_obs, *_ = await new.step(_call("buy", 0.4))
        old_obs, *_ = await old.step(_call("buy", 0.4))
        new_state, old_state = _state(new_obs), _state(old_obs)
        new_state.pop("order_rejection")
        old_state.pop("order_rejection")
        assert new_state == old_state
        assert _ledger_state(new) == _ledger_state(old)

    asyncio.run(exercise())


def test_v1_observation_bytes_remain_at_the_pre_v2_identity():
    async def exercise():
        env = _env(RECOVERY_OBSERVATION_PROFILE)
        observations = [env.reset().messages]
        for action in (
            _call("buy", 0.5),
            _call("buy", 0.4),
            _call("buy", 0.4),
            _call("sell", 0.6),
            _call("hold"),
        ):
            observation, *_ = await env.step(action)
            observations.append(observation.messages)
        observations.append(env.reset().messages)
        payload = json.dumps(observations, ensure_ascii=True, sort_keys=True).encode("ascii")
        assert (
            sha256(payload).hexdigest()
            == "1d0627359c7938da7056285dc18e457804bf3eb0d2f711fb2e795bd87e909578"
        )

    asyncio.run(exercise())
