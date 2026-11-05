"""CPU contracts for optional rejection context; policy actions remain explicit."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from agent_r1.env.base import Action, Observation

from src.data import MarketObservation, StockBenchAdapter
from src.env.observation_memory import RECOVERY_OBSERVATION_PROFILE, validate_observation_profile
from src.env.order_feedback import MAX_CONSECUTIVE_REJECTIONS, ORDER_DIRECTION_TOLERANCE
from src.env.trading import TradingEnv

_DAYS = (2, 3, 6, 7)


def _env(
    profile: str = RECOVERY_OBSERVATION_PROFILE, *,
    close_after_buy: float = 100.0000004,
    next_open: float = 100.0,
    commission_bps: float = 0.0,
) -> TradingEnv:
    dates = [datetime(2025, 1, day, 21, tzinfo=timezone.utc) for day in _DAYS]
    opens = (100.0, 100.0, next_open, 100.0)
    closes = (100.0, close_after_buy, next_open, 100.0)
    bars = [
        MarketObservation(
            observation_id=f"recovery-{day}", kind="market", symbol="AAA",
            event_at=stamp, available_at=stamp,
            payload={"date": stamp.date().isoformat(), "open": open_price, "close": close},
        )
        for day, stamp, open_price, close in zip(_DAYS, dates, opens, closes, strict=True)
    ]
    return TradingEnv(
        adapter=StockBenchAdapter(bars, data_version="order-recovery-synthetic-v1"),
        episode_dates=dates,
        episode_open_times=[stamp.replace(hour=14, minute=30) for stamp in dates],
        symbols=["AAA"], initial_cash=1000.0,
        commission_bps=commission_bps, slippage_bps=0.0,
        drawdown_coefficient=0.0, turnover_coefficient=0.0,
        benchmark_closes=[100.0] * len(dates), observation_profile=profile,
    )


def _call(name: str, target: float | None = None) -> Action:
    arguments = {} if target is None else {"symbol": "AAA", "target_weight": target}
    return Action(text="<tool_call>" + json.dumps({"name": name, "arguments": arguments})
                  + "</tool_call>")


def _state(observation: Observation) -> dict:
    assert observation.messages is not None
    text = observation.messages[-1]["content"]
    if text.startswith("<tool_response>"):
        text = text.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
    return json.loads(text)


def _ledger_state(env: TradingEnv) -> tuple:
    return (
        env.current_session_index, env.current_date, env.ledger.cash,
        dict(env.ledger.positions), env.ledger.fills, env.ledger.pending_orders,
    )


@pytest.mark.parametrize(("name", "close"), [("buy", 100.0000004), ("sell", 99.9999996)])
def test_rounding_boundary_reports_the_exact_comparison_without_changing_it(name, close):
    async def exercise():
        env = _env(close_after_buy=close)
        env.reset()
        observation, *_ = await env.step(_call("buy", 0.5))
        displayed = _state(observation)["positions"]["AAA"]["weight"]
        raw_weight = env._current_weight("AAA")
        target = (displayed + raw_weight) / 2
        assert target > displayed if name == "buy" else target < displayed
        before = _ledger_state(env)

        observation, reward, done, info = await env.step(_call(name, target))

        assert info["invalid_action"] is True and reward == 0.0 and not done
        assert _ledger_state(env) == before
        state = _state(observation)
        context = state["order_rejection"]
        assert context["version"] == "order_rejection_v1"
        assert context["rejected_call"] == {
            "name": name, "arguments": {"symbol": "AAA", "target_weight": target},
        }
        assert context["current_weight"] == raw_weight != displayed
        assert context["comparison_tolerance"] == ORDER_DIRECTION_TOLERANCE
        required = context["required_target"]
        assert required["operator"] == (">" if name == "buy" else "<")
        assert required["threshold"] == raw_weight + (
            ORDER_DIRECTION_TOLERANCE if name == "buy" else -ORDER_DIRECTION_TOLERANCE
        )
        assert context["session_advanced"] is context["trade_executed"] is False
        assert context["consecutive_same_request"] == 1
        assert state["positions"]["AAA"]["weight"] == displayed
        assert env.credit_step_trace[-1].action == "invalid"
        assert env.credit_step_trace[-1].decision_session_index == env.current_session_index
        assert env.credit_step_trace[-1].result_session_index == env.current_session_index

        # A policy chooses a new target using the returned raw threshold. The
        # environment never manufactures or submits the correction itself.
        corrected = required["threshold"] + (0.05 if name == "buy" else -0.05)
        observation, _, _, info = await env.step(_call(name, corrected))
        assert not info["invalid_action"]
        assert env.current_session_index == before[0] + 1
        assert info["fills"][0]["side"] == name
        assert "order_rejection" not in _state(observation)

    asyncio.run(exercise())


@pytest.mark.parametrize("name", ["buy", "sell"])
def test_reported_strict_threshold_matches_the_existing_acceptance_boundary(name):
    async def exercise():
        env = _env(close_after_buy=100.0)
        env.reset()
        await env.step(_call("buy", 0.5))
        weight = env._current_weight("AAA")
        boundary = weight + (ORDER_DIRECTION_TOLERANCE if name == "buy"
                             else -ORDER_DIRECTION_TOLERANCE)
        before = _ledger_state(env)
        observation, _, _, info = await env.step(_call(name, boundary))
        assert info["invalid_action"] and _ledger_state(env) == before
        assert _state(observation)["order_rejection"]["required_target"]["threshold"] == boundary
        # The nearest representable target on the permitted side passes the
        # original rule, even if its tiny share delta produces no actual fill.
        target = math.nextafter(boundary, 1.0 if name == "buy" else 0.0)
        observation, _, _, info = await env.step(_call(name, target))
        assert not info["invalid_action"]
        assert env.current_session_index == before[0] + 1
        assert "order_rejection" not in _state(observation)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("name", "target", "with_position"),
    [("buy", 0.0, False), ("sell", 0.0, False), ("sell", 0.1, False),
     ("buy", 0.4, True), ("sell", 0.6, True)],
)
def test_rejection_repeats_leave_the_ledger_untouched_until_policy_chooses_hold(
    name, target, with_position,
):
    async def exercise():
        env = _env(commission_bps=10.0)
        env.reset()
        if with_position:
            await env.step(_call("buy", 0.5))
        before = _ledger_state(env)
        for count in range(1, 5):
            observation, reward, done, info = await env.step(_call(name, target))
            state = _state(observation)
            assert _ledger_state(env) == before
            assert reward == 0.0 and not done and info["invalid_action"]
            context = state["order_rejection"]
            assert context["consecutive_same_request"] == count
            assert context["session_index"] == before[0]
            assert len(state["recent_events"]) <= env.MAX_RECENT_EVENTS
            assert state["recent_events"][-1]["arguments"] == {
                "error": info["error"], "name": name, "symbol": "AAA", "target_weight": target,
            }
            if name == "sell" and not with_position:
                assert context["same_action_has_valid_target"] is False
        observation, _, _, info = await env.step(_call("hold"))
        assert env.current_session_index == before[0] + 1
        assert info["fills"] == [] and info["cost_fraction"] == 0.0
        assert env.ledger.cash == before[2]
        assert dict(env.ledger.positions) == before[3]
        assert env.ledger.fills == before[4]
        assert "order_rejection" not in _state(observation)

    asyncio.run(exercise())


def test_streak_is_bounded_and_reset_by_other_requests_and_episode_reset():
    async def exercise():
        env = _env()
        env.reset()
        await env.step(_call("sell", 0.0))
        env._consecutive_order_rejections = MAX_CONSECUTIVE_REJECTIONS - 1
        for _ in range(2):
            observation, *_ = await env.step(_call("sell", 0.0))
            context = _state(observation)["order_rejection"]
            assert context["consecutive_same_request"] == MAX_CONSECUTIVE_REJECTIONS
            assert context["consecutive_same_request_limit"] == MAX_CONSECUTIVE_REJECTIONS
        observation, *_ = await env.step(_call("sell", 0.1))
        assert _state(observation)["order_rejection"]["consecutive_same_request"] == 1
        observation, *_ = await env.step(_call("unknown_tool"))
        assert "order_rejection" not in _state(observation)
        observation, *_ = await env.step(_call("sell", 0.1))
        assert _state(observation)["order_rejection"]["consecutive_same_request"] == 1
        observation, *_ = await env.step(Action(text='<tool_call>{"name":"retrieve",'
                                                   '"arguments":{"query":"AAA"}}</tool_call>'))
        assert "order_rejection" not in _state(observation)
        observation, *_ = await env.step(_call("sell", 0.1))
        assert _state(observation)["order_rejection"]["consecutive_same_request"] == 1
        observation = env.reset()
        assert "order_rejection" not in _state(observation)
        assert _state(observation)["recent_events"] == []
        observation, *_ = await env.step(_call("sell", 0.1))
        assert _state(observation)["order_rejection"]["consecutive_same_request"] == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(("name", "target", "next_open"), [("buy", 0.55, 200.0),
                                                            ("sell", 0.45, 50.0)])
def test_valid_next_open_direction_skip_is_not_reported_as_a_rejection(name, target, next_open):
    async def exercise():
        env = _env(close_after_buy=100.0, next_open=next_open, commission_bps=10.0)
        env.reset()
        await env.step(_call("buy", 0.5))
        # First produce real feedback, then submit a valid directional target.
        await env.step(_call(name, 0.0 if name == "buy" else 1.0))
        before = _ledger_state(env)
        observation, _, _, info = await env.step(_call(name, target))
        assert not info["invalid_action"]
        assert env.current_session_index == before[0] + 1
        assert info["fills"] == [] and info["cost_fraction"] == 0.0
        assert info["skipped_orders"][0]["reason"] == "opposite_direction_at_next_open"
        assert env.ledger.cash == before[2] and dict(env.ledger.positions) == before[3]
        assert "order_rejection" not in _state(observation)

    asyncio.run(exercise())


def test_recovery_profile_inherits_features_and_memory_without_changing_market_state():
    async def exercise():
        new, old = _env(), _env("history_features_v1")
        assert validate_observation_profile(RECOVERY_OBSERVATION_PROFILE) == new.observation_profile
        assert new.reset().messages == old.reset().messages
        for action in (
            _call("buy", 0.5),
            Action(text='<tool_call>{"name":"compute","arguments":'
                        '{"metric":"mean_return","symbol":"AAA","lookback":2}}</tool_call>'),
        ):
            new_result, old_result = await new.step(action), await old.step(action)
            assert new_result == old_result
        new_observation, *_ = await new.step(_call("buy", 0.0))
        old_observation, *_ = await old.step(_call("buy", 0.0))
        new_state, old_state = _state(new_observation), _state(old_observation)
        new_state.pop("order_rejection")
        for field in ("name", "symbol", "target_weight"):
            new_state["recent_events"][-1]["arguments"].pop(field)
        assert new_state == old_state

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("profile", "expected_hash"),
    [
        # Captured from HEAD trading.py before the optional feedback patch,
        # including initial state, fill, rejection, hold, and episode reset.
        ("legacy", "4b98fe2c599d65c0eb4af2a0a5ff28559e2712fa6a9f5c6c185d167a4a9b4163"),
        ("history_memory_v1", "0926ab91015b50e6b69c84f599f807bd41f0c445cbf2945a3b749d2711aab655"),
        ("history_features_v1", "776d9ba9f2bf631130c4a596a82d1110896ca8e774003a0f5c0402a7ef15d0a7"),
    ],
)
def test_historical_profiles_keep_their_pre_recovery_observation_bytes(profile, expected_hash):
    async def exercise():
        env = _env(profile)
        observations = [env.reset().messages]
        for action in (_call("buy", 0.5), _call("buy", 0.0), _call("hold")):
            observation, *_ = await env.step(action)
            observations.append(observation.messages)
        observations.append(env.reset().messages)
        payload = json.dumps(observations, ensure_ascii=True, sort_keys=True).encode("ascii")
        assert sha256(payload).hexdigest() == expected_hash

    asyncio.run(exercise())
