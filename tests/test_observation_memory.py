"""Real CPU environment checks for PIT history visibility and bounded tool memory.

All observations are synthetic. These tests do not open prepared Train/Dev data,
load model weights, or change generation settings.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Literal

import pytest
from agent_r1.env.base import Action, Observation

from src.data import MarketObservation, StockBenchAdapter
from src.env.observation_memory import ToolResultMemory
from src.env.trading import ComputeArgs, TradingEnv

_DAYS = (
    "2023-01-03",
    "2023-01-04",
    "2023-01-05",
    "2023-01-06",
    "2023-01-09",
    "2023-01-10",
    "2023-01-11",
    "2023-01-12",
    "2023-01-13",
    "2023-01-17",
    "2023-01-18",
    "2023-01-19",
    "2023-01-20",
    "2023-01-23",
    "2023-01-24",
    "2023-01-25",
    "2023-01-26",
    "2023-01-27",
    "2023-01-30",
    "2023-01-31",
    "2023-02-01",
    "2023-02-02",
    "2023-02-03",
    "2023-02-06",
    "2023-02-07",
)
_DATES = tuple(datetime.fromisoformat(day + "T21:00:00+00:00") for day in _DAYS)
_COMPUTE_COLUMNS = [
    "symbol",
    "metric",
    "lookback_requested",
    "lookback_used",
    "as_of",
    "value",
]


def _bars() -> list[MarketObservation]:
    return [
        MarketObservation(
            observation_id=f"market:{symbol}:{day}",
            kind="market",
            symbol=symbol,
            event_at=stamp,
            available_at=stamp,
            payload={
                "date": day,
                "open": base + index - 0.5,
                "high": base + index + 1,
                "low": base + index - 1,
                "close": base + index,
                "volume": 1000 + index,
            },
        )
        for symbol, base in (("AAA", 100.0), ("BBB", 200.0))
        for index, (day, stamp) in enumerate(zip(_DAYS, _DATES, strict=True))
    ]


def _env(
    *,
    profile: str | None = "history_memory_v1",
    offset: int = 4,
    observations: list[MarketObservation] | None = None,
    adapter: StockBenchAdapter | None = None,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    reward_mode: str = "outcome_only",
) -> TradingEnv:
    dates = _DATES[offset : offset + 21]
    kwargs: dict[str, Any] = {
        "adapter": adapter
        or StockBenchAdapter(
            _bars() if observations is None else observations,
            data_version="synthetic-history-memory-v1",
        ),
        "episode_dates": dates,
        "episode_open_times": [stamp.replace(hour=14, minute=30) for stamp in dates],
        "symbols": list(symbols),
        "initial_cash": 1000.0,
        "commission_bps": 10.0,
        "slippage_bps": 5.0,
        "drawdown_coefficient": 0.0,
        "turnover_coefficient": 0.0,
        "benchmark_closes": [1000.0 + index for index in range(21)],
        "screening": True,
        "finance_scorer_id": "benchmark_scale_v2",
        "finance_volatility_floor": 0.001,
        "risk_lambda": 0.5,
        "turnover_limit": 1.0,
        "position_limit": 0.3,
        "screening_calibration_sha256": "a" * 64,
        "reward_mode": reward_mode,
    }
    if profile is not None:
        kwargs["observation_profile"] = profile
    return TradingEnv(**kwargs)


def _state(observation: Observation) -> dict[str, Any]:
    assert observation.messages is not None
    text = observation.messages[-1]["content"]
    if text.startswith("<tool_response>"):
        assert text.endswith("</tool_response>")
        text = text.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
    return json.loads(text)


def _action(tool: str, arguments: dict[str, Any] | None = None) -> Action:
    return Action(
        text="<tool_call>"
        + json.dumps(
            {
                "name": tool,
                "arguments": {} if arguments is None else arguments,
            }
        )
        + "</tool_call>"
    )


def _step(env: TradingEnv, tool: str, arguments: dict[str, Any] | None = None):
    observation, reward, done, info = asyncio.run(env.step(_action(tool, arguments)))
    return _state(observation), reward, done, info


def _compute(
    env: TradingEnv,
    *,
    lookback: int = 2,
    metric: str = "price_change",
    symbol: str = "AAA",
):
    state, reward, done, info = _step(
        env,
        "compute",
        {"metric": metric, "symbol": symbol, "lookback": lookback},
    )
    assert not info.get("invalid_action") and reward == 0 and not done
    return state


def _memory(state: dict[str, Any]) -> dict[str, Any]:
    value = state["tool_result_memory"]
    assert value["compute_columns"] == _COMPUTE_COLUMNS
    assert value["capacity"] == 20 and value["byte_limit"] == 4096
    assert len(value["retrievals"]) <= 2
    assert 0 <= value["retained_count"] <= 20 and value["evicted_count"] >= 0
    assert type(value["current_result_in_last_tool_result"]) is bool
    assert value["retained_count"] == (
        len(value["compute_rows"])
        + len(value["retrievals"])
        + int(value["current_result_in_last_tool_result"])
    )
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert len(encoded) <= value["byte_limit"]
    for row in value["compute_rows"]:
        assert len(row) == len(_COMPUTE_COLUMNS)
    return value


def _compute_rows(memory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(zip(memory["compute_columns"], row, strict=True)) for row in memory["compute_rows"]
    ]


def _technical(query: str) -> MarketObservation:
    return MarketObservation(
        observation_id=f"technical:{query}",
        kind="technical",
        symbol="AAA",
        event_at=_DATES[0],
        available_at=_DATES[0],
        payload={"text": query},
    )


def test_default_legacy_has_no_new_observation_fields():
    default, explicit = _env(profile=None), _env(profile="legacy")
    assert default.observation_profile == explicit.observation_profile == "legacy"
    assert default.reset() == explicit.reset()
    state = _compute(default)
    assert "compute_history" not in state and "tool_result_memory" not in state
    assert state == _compute(explicit)


@pytest.mark.parametrize("profile", ["", "unknown", True])
def test_unknown_observation_profile_is_rejected(profile):
    with pytest.raises((TypeError, ValueError), match="observation_profile"):
        _env(profile=profile)


@pytest.mark.parametrize("offset,visible", [(0, 1), (4, 5)])
def test_history_counts_cross_episode_boundary_and_match_compute_used(offset, visible):
    env = _env(offset=offset)
    state = _state(env.reset())
    assert env.observation_profile == "history_memory_v1"
    assert state["session_index"] == 0
    assert state["compute_history"] == {
        "minimum_closes": 2,
        "max_lookback": 60,
        "available_close_records": {"AAA": visible, "BBB": visible},
    }
    _memory(state)
    state, _, _, info = _step(
        env,
        "compute",
        {"metric": "price_change", "symbol": "AAA", "lookback": 60},
    )
    if visible == 1:
        assert info["invalid_action"] and "at least two" in info["error"]
        assert _memory(state)["retained_count"] == 0
    else:
        assert not info.get("invalid_action")
        assert state["last_tool_result"]["lookback_used"] == 5
        assert state["last_tool_result"]["value"] == pytest.approx(104.0 / 100.0 - 1)


def test_history_excludes_missing_close_other_kinds_and_other_symbols():
    other_kinds: tuple[Literal["technical", "fundamental"], ...] = (
        "technical",
        "fundamental",
    )
    extra = [
        MarketObservation(
            observation_id="missing-historical-close",
            kind="market",
            symbol="AAA",
            event_at=_DATES[0],
            available_at=_DATES[0],
            payload={"date": _DAYS[0], "open": 999.0},
        ),
        MarketObservation(
            observation_id="foreign-symbol",
            kind="market",
            symbol="CCC",
            event_at=_DATES[0],
            available_at=_DATES[0],
            payload={"date": _DAYS[0], "close": 999.0},
        ),
        *[
            MarketObservation(
                observation_id=f"nonmarket-{kind}",
                kind=kind,
                symbol="AAA",
                event_at=_DATES[0],
                available_at=_DATES[0],
                payload={"close": 999.0},
            )
            for kind in other_kinds
        ],
    ]
    env = _env(observations=[*_bars(), *extra])
    state = _state(env.reset())
    assert state["compute_history"]["available_close_records"] == {"AAA": 5, "BBB": 5}
    state = _compute(env, lookback=60)
    assert state["last_tool_result"]["lookback_used"] == 5
    assert state["last_tool_result"]["value"] == pytest.approx(0.04)


def test_history_record_count_can_exceed_compute_request_limit():
    older = [
        MarketObservation(
            observation_id=f"earlier-visible-version-{index:02d}",
            kind="market",
            symbol="AAA",
            event_at=_DATES[0],
            available_at=_DATES[0] + timedelta(microseconds=index),
            payload={"date": _DAYS[0], "close": 500.0 + index},
        )
        for index in range(60)
    ]
    env = _env(observations=[*_bars(), *older])
    state = _state(env.reset())
    assert state["compute_history"]["available_close_records"] == {"AAA": 65, "BBB": 5}
    assert state["compute_history"]["max_lookback"] == 60
    state = _compute(env, lookback=60)
    assert state["last_tool_result"]["lookback_used"] == 60


def test_delayed_versions_use_availability_order_and_are_not_silently_deduplicated():
    # Both versions concern an old market day; the current-session bar remains
    # unique. One version is visible exactly at as_of, the other one microsecond later.
    versions = [
        MarketObservation(
            observation_id="zz-old-version-visible",
            kind="market",
            symbol="AAA",
            event_at=_DATES[0],
            available_at=_DATES[4],
            payload={"date": _DAYS[0], "close": 200.0},
        ),
        MarketObservation(
            observation_id="zz-old-version-not-yet-visible",
            kind="market",
            symbol="AAA",
            event_at=_DATES[0],
            available_at=_DATES[4] + timedelta(microseconds=1),
            payload={"date": _DAYS[0], "close": 300.0},
        ),
    ]
    env = _env(observations=[*_bars(), *versions])
    state = _state(env.reset())
    assert state["compute_history"]["available_close_records"] == {"AAA": 6, "BBB": 5}
    result = _compute(env)["last_tool_result"]
    assert result["value"] == pytest.approx(200.0 / 104.0 - 1)
    state, _, _, _ = _step(env, "hold")
    assert state["compute_history"]["available_close_records"] == {"AAA": 8, "BBB": 6}
    result = _compute(env)["last_tool_result"]
    assert result["value"] == pytest.approx(105.0 / 300.0 - 1)


def test_future_market_values_do_not_change_visible_history_results_or_memory():
    original = _bars()
    changed = [
        replace(item, payload={**item.payload, "open": 9876543.0, "close": 8765432.0})
        if item.available_at > _DATES[4]
        else item
        for item in original
    ]
    first, second = _env(observations=original), _env(observations=changed)
    assert first.reset() == second.reset()
    for tool, args in (
        ("compute", {"metric": "price_change", "symbol": "AAA", "lookback": 60}),
        ("retrieve", {"query": "8765432", "limit": 1}),
    ):
        left, right = _step(first, tool, args), _step(second, tool, args)
        assert left == right
        _memory(left[0])
    assert left[0]["last_tool_result"]["returned"] == 0


def test_successful_compute_is_not_duplicated_then_survives_hold_and_invalid_unchanged():
    env = _env()
    env.reset()
    state = _compute(env, lookback=60)
    original = dict(state["last_tool_result"])
    memory = _memory(state)
    assert memory["retained_count"] == 1 and memory["compute_rows"] == []
    assert memory["current_result_in_last_tool_result"] is True
    state, _, _, _ = _step(env, "hold")
    memory = _memory(state)
    assert not memory["current_result_in_last_tool_result"]
    (saved,) = _compute_rows(memory)
    assert saved == {**original, "lookback_requested": 60}
    assert saved["as_of"] == _DATES[4].isoformat() != state["current_date"]
    observation, _, _, info = asyncio.run(env.step(Action(text="malformed response")))
    assert info["invalid_action"]
    assert _compute_rows(_memory(_state(observation))) == [saved]
    # The same arguments at a later as_of are a distinct observation, not a rewrite.
    state = _compute(env, lookback=60)
    assert _memory(state)["retained_count"] == 2
    assert _compute_rows(_memory(state)) == [saved]


def test_same_asof_normalized_compute_arguments_refresh_without_duplicate():
    env = _env()
    env.reset()
    _compute(env, lookback=2)
    _compute(env, lookback=3)
    state, _, _, info = _step(
        env,
        "compute",
        {"lookback": 2.0, "symbol": "AAA", "metric": "price_change"},
    )
    assert not info.get("invalid_action")
    memory = _memory(state)
    assert memory["retained_count"] == 2 and memory["evicted_count"] == 0
    assert memory["refreshed_count"] == 1
    state, _, _, _ = _step(env, "hold")
    assert [row["lookback_requested"] for row in _compute_rows(_memory(state))] == [3, 2]


def test_retrieval_preserves_original_result_and_explicit_defaults_deduplicate():
    env = _env(observations=[*_bars(), _technical("alpha")])
    env.reset()
    state, _, _, _ = _step(env, "retrieve", {"query": "alpha"})
    original = state["last_tool_result"]
    assert original["returned"] == 1
    assert _memory(state)["retrievals"] == []
    state, _, _, _ = _step(env, "retrieve", {"limit": 3, "query": "alpha", "symbol": None})
    assert _memory(state)["retained_count"] == 1
    state, _, _, _ = _step(env, "hold")
    (saved,) = _memory(state)["retrievals"]
    assert saved["result"] == original
    assert saved["arguments"]["query"] == "alpha"
    assert saved["arguments"]["limit"] == 3
    assert saved["result"]["as_of"] != state["current_date"]


def test_distinct_full_queries_do_not_collide_after_display_truncation():
    env = _env()
    env.reset()
    for query in ("x" * 100 + "a", "x" * 100 + "b"):
        state, _, _, info = _step(env, "retrieve", {"query": query, "limit": 1})
        assert not info.get("invalid_action")
    assert _memory(state)["retained_count"] == 2
    state, _, _, _ = _step(env, "hold")
    retrievals = _memory(state)["retrievals"]
    assert len(retrievals) == 2
    for entry in retrievals:
        assert len(entry["arguments"]["query"]) <= 80
        assert entry["arguments"]["query_truncated"] is True


def test_retrieval_bank_keeps_at_most_two_latest_results():
    env = _env(observations=[*_bars(), *[_technical(q) for q in ("alpha", "beta", "gamma")]])
    env.reset()
    for query in ("alpha", "beta", "gamma"):
        state, _, _, _ = _step(env, "retrieve", {"query": query, "limit": 1})
        _memory(state)
    state, _, _, _ = _step(env, "hold")
    memory = _memory(state)
    assert [row["arguments"]["query"] for row in memory["retrievals"]] == ["beta", "gamma"]
    assert memory["retained_count"] == 2 and memory["evicted_count"] == 1


def test_compute_bank_is_fifo_and_compact_bytes_remain_bounded_under_repeated_calls():
    env = _env()
    env.reset()
    for index in range(50):
        state = _compute(
            env,
            lookback=index + 2,
            metric=("price_change", "mean_return", "volatility")[index % 3],
        )
        _memory(state)
    state, _, _, _ = _step(env, "hold")
    memory = _memory(state)
    assert memory["retained_count"] > 0
    assert memory["evicted_count"] + memory["retained_count"] == 50
    assert memory["evicted_count"] >= 30
    assert [row["lookback_requested"] for row in _compute_rows(memory)] == list(
        range(52 - memory["retained_count"], 52)
    )


def test_full_unicode_retrievals_force_byte_eviction_before_result_capacity():
    # Use a legal long synthetic ticker to stress bytes independently of the
    # ordinary twenty-entry limit; no production universe is being changed.
    symbol = "LONG_SYNTHETIC_TICKER_AAA"
    bars = [replace(item, symbol=symbol) if item.symbol == "AAA" else item for item in _bars()]
    queries = ("alpha" + "x" * 75, "beta" + "y" * 76)
    records = [
        replace(
            _technical(query),
            payload={
                "a": "证" * 25 + "x" * 10,
                "b": "据" * 25 + "x" * 10,
                "c": "保" * 25 + "x" * 10,
                "d": "留" * 25 + "x" * 10,
                "q": query,
                "r": "x" * 70,
            },
        )
        for query in queries
    ]
    legacy, memory_env = (
        _env(profile=profile, observations=[*bars, *records], symbols=(symbol, "BBB"))
        for profile in ("legacy", "history_memory_v1")
    )
    legacy.reset()
    memory_env.reset()
    for lookback in range(2, 22):
        _compute(memory_env, lookback=lookback, metric="mean_return", symbol=symbol)
    original_results = []
    for query in queries:
        state, reward, done, info = _step(memory_env, "retrieve", {"query": query})
        original_state, old_reward, old_done, old_info = _step(legacy, "retrieve", {"query": query})
        assert (reward, done, info) == (old_reward, old_done, old_info)
        result = state["last_tool_result"]
        assert result == original_state["last_tool_result"]
        assert result["returned"] == 1
        assert len(json.dumps(result, ensure_ascii=True, separators=(",", ":"))) >= 950
        original_results.append(result)
        _memory(state)
    state, _, _, _ = _step(memory_env, "hold")
    memory = _memory(state)
    assert [entry["result"] for entry in memory["retrievals"]] == original_results
    # Only two entries would be removed by the count limit. These real Unicode
    # results require additional FIFO removals under the full-object byte limit.
    assert memory["retained_count"] < 20 and memory["evicted_count"] > 2
    assert memory["retained_count"] + memory["evicted_count"] == 22
    assert [row["lookback_requested"] for row in _compute_rows(memory)] == list(
        range(22 - len(memory["compute_rows"]), 22)
    )


@pytest.mark.parametrize("lookback", [1, 61])
def test_invalid_compute_lookback_does_not_enter_or_clear_successful_memory(lookback):
    env = _env()
    env.reset()
    original = _compute(env)["last_tool_result"]
    state, _, _, info = _step(
        env, "compute", {"symbol": "AAA", "metric": "price_change", "lookback": lookback}
    )
    assert info["invalid_action"]
    memory = _memory(state)
    assert memory["retained_count"] == 1 and memory["evicted_count"] == 0
    assert _compute_rows(memory) == [{**original, "lookback_requested": 2}]


def test_long_retrieval_symbol_keeps_original_behavior_and_bounded_public_arguments():
    plain, memory_env = _env(profile="legacy"), _env()
    plain.reset()
    memory_env.reset()
    arguments = {"query": "x", "symbol": "A" * 5000}
    old, new = _step(plain, "retrieve", arguments), _step(memory_env, "retrieve", arguments)
    assert old[1:] == new[1:]
    assert not new[3].get("invalid_action")
    result = new[0]["last_tool_result"]
    assert result == old[0]["last_tool_result"] and result["returned"] == 0
    _memory(new[0])
    state, _, _, _ = _step(memory_env, "hold")
    (saved,) = _memory(state)["retrievals"]
    assert saved["result"] == result
    assert saved["arguments"]["symbol"] == "A" * 80
    assert saved["arguments"]["symbol_truncated"] is True


def test_oversized_memory_insertion_failure_restores_prior_entries_and_counters():
    bank = ToolResultMemory()
    arguments = {"symbol": "AAA", "metric": "price_change", "lookback": 2}
    result = {
        "symbol": "AAA",
        "metric": "price_change",
        "lookback_used": 2,
        "as_of": _DATES[4].isoformat(),
        "value": 0.1,
    }
    bank.remember("compute", arguments, result)
    bank.remember("compute", arguments, result)
    before = bank.render()
    assert before["retained_count"] == 1 and before["refreshed_count"] == 1
    with pytest.raises(ValueError, match="byte budget"):
        bank.remember(
            "retrieve",
            {"query": "x", "limit": 1},
            {
                "as_of": _DATES[4].isoformat(),
                "results": [{"text": "x" * 5000}],
                "returned": 1,
                "truncated": False,
            },
        )
    assert bank.render() == before
    # The original result remains a valid duplicate after the rejected insert.
    bank.remember("compute", arguments, result)
    after = bank.render()
    assert after["retained_count"] == 1 and after["evicted_count"] == 0
    assert after["refreshed_count"] == 2


def test_reset_and_separate_environments_do_not_reuse_tool_results():
    adapter = StockBenchAdapter(_bars(), data_version="shared-synthetic-memory-fixture")
    first, second = _env(adapter=adapter), _env(adapter=adapter)
    first.reset()
    _compute(first)
    second_state = _state(second.reset())
    assert _memory(second_state)["retained_count"] == 0
    reset_state = _state(first.reset())
    assert reset_state == second_state
    assert _memory(reset_state)["evicted_count"] == 0
    assert reset_state["compute_history"]["available_close_records"] == {"AAA": 5, "BBB": 5}


@pytest.mark.parametrize("reward_mode", ["outcome_only", "finance_trajectory"])
@pytest.mark.parametrize("profile", ["history_memory_v1", "history_features_v1"])
def test_profile_preserves_actual_trading_rewards_ledger_and_old_observation_fields(
    reward_mode, profile,
):
    plain = _env(profile="legacy", reward_mode=reward_mode)
    memory = _env(profile=profile, reward_mode=reward_mode)
    plain.reset()
    memory.reset()
    actions = [
        Action(text="malformed response"),
        _action("compute", {"metric": "price_change", "symbol": "AAA", "lookback": 2}),
        _action("retrieve", {"query": "AAA", "limit": 1}),
        _action("buy", {"symbol": "AAA", "target_weight": 0.2}),
        _action("compute", {"metric": "mean_return", "symbol": "AAA", "lookback": 60}),
        _action("sell", {"symbol": "AAA", "target_weight": 0.0}),
        *[_action("hold") for _ in range(18)],
    ]
    for action in actions:
        left, right = asyncio.run(plain.step(action)), asyncio.run(memory.step(action))
        assert left[1:] == right[1:]
        old_state, new_state = _state(left[0]), _state(right[0])
        _memory(new_state)
        new_state.pop("compute_history")
        new_state.pop("tool_result_memory")
        new_state.pop("decision_features", None)
        assert old_state == new_state
    assert plain.episode_complete and memory.episode_complete
    assert plain.credit_step_trace == memory.credit_step_trace
    assert plain.screening_terminal_metrics() == memory.screening_terminal_metrics()


@pytest.mark.parametrize("profile", ["history_memory_v1", "history_features_v1"])
def test_memory_observations_remain_reward_neutral_across_screening_reward_modes(profile):
    first = _env(profile=profile, reward_mode="outcome_only")
    second = _env(profile=profile, reward_mode="finance_trajectory")
    assert first.reset() == second.reset()
    actions = [
        _action("compute", {"metric": "price_change", "symbol": "AAA", "lookback": 5}),
        _action("buy", {"symbol": "AAA", "target_weight": 0.2}),
        *[_action("hold") for _ in range(19)],
    ]
    for action in actions:
        left, right = asyncio.run(first.step(action)), asyncio.run(second.step(action))
        assert left[0] == right[0]
        state = _state(left[0])
        for forbidden in ("reward", "last_reward", "terminal_score", "risk_lambda", "credit"):
            assert forbidden not in state
            assert forbidden not in _memory(state)
    assert first.episode_complete and second.episode_complete


@pytest.mark.parametrize("offset,used", [(0, 1), (1, 2), (4, 5)])
def test_features_display_real_compute_values_or_explicit_missing_history(offset, used):
    env = _env(profile="history_features_v1", offset=offset)
    initial = _state(env.reset())
    table = initial["decision_features"]
    assert table["as_of"] == _DATES[offset].isoformat()
    assert table["lookback_requested"] == 5
    assert table["columns"] == [
        "symbol", "lookback_used", "price_change", "mean_return", "volatility",
    ]
    assert [row[0] for row in table["rows"]] == ["AAA", "BBB"]
    for symbol, lookback_used, *values in table["rows"]:
        assert lookback_used == used
        if used == 1:
            assert values == [None, None, None]
        else:
            for metric, value in zip(table["columns"][2:], values, strict=True):
                assert value == env._compute(ComputeArgs(
                    symbol=symbol, metric=metric, lookback=5,
                ))["value"]
    # Rendering and prepared features are read-only, never model tool actions.
    assert env._state_text() == json.dumps(
        initial, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    )
    assert not env.credit_step_trace and env._tool_calls == env._invalid_calls == 0
    assert env.ledger.cash == 1000.0 and not env.ledger.fills
    assert _memory(initial)["retained_count"] == 0
    if used >= 2:
        state = _compute(env, symbol="AAA", lookback=5)
        original = state["last_tool_result"]
        state = _compute(env, symbol="BBB", metric="mean_return", lookback=5)
        assert state["decision_features"] == table
        assert _compute_rows(_memory(state)) == [{**original, "lookback_requested": 5}]


def test_features_respect_availability_order_and_exclude_unavailable_values():
    visible_revision = MarketObservation(
        observation_id="zz-old-visible-revision",
        kind="market", symbol="AAA", event_at=_DATES[0], available_at=_DATES[4],
        payload={"date": _DAYS[0], "close": 200.0},
    )
    unavailable_revision = replace(
        visible_revision, observation_id="zz-old-unavailable-revision",
        available_at=_DATES[4] + timedelta(microseconds=1),
        payload={"date": _DAYS[0], "close": 300.0},
    )
    original = [*_bars(), visible_revision, unavailable_revision]
    perturbed = [
        replace(item, payload={**item.payload, "open": 999999.0, "close": 888888.0})
        if item.available_at > _DATES[4] else item
        for item in original
    ]
    first = _env(profile="history_features_v1", observations=original)
    second = _env(profile="history_features_v1", observations=perturbed)
    assert first.reset() == second.reset()
    state = json.loads(first._state_text())
    aaa = state["decision_features"]["rows"][0]
    assert aaa[:2] == ["AAA", 5]
    assert aaa[2] == pytest.approx(200.0 / 101.0 - 1.0)
    assert state["compute_history"]["available_close_records"]["AAA"] == 6
    advanced, _, _, _ = _step(first, "hold")
    assert advanced["decision_features"]["as_of"] == _DATES[5].isoformat()
    assert advanced["decision_features"]["rows"][0][2] == pytest.approx(105.0 / 103.0 - 1.0)
    assert _state(first.reset()) == state
