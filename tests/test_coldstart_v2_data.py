"""Real portfolio construction, error recovery, and historical label isolation."""

import asyncio
import copy
import json
from datetime import datetime

import pytest

from src.agents.coldstart_data import candidate_calls, parse_state, source_dates, tool_call
from src.agents.coldstart_v2_data import (
    RECOVERY_SCENARIOS,
    SCENARIOS,
    api_targets,
    evenly_spaced,
    find_span,
    fixed_prefix,
    recovery_context,
    render,
    replay_context,
    score_candidate,
    training_contexts,
)
from src.data import MarketObservation, StockBenchAdapter

DAYS = [
    "2023-01-03",
    "2023-01-04",
    "2023-01-05",
    "2023-01-06",
    "2023-01-09",
    "2023-01-10",
    "2023-01-11",
]
SYMBOLS = ("GS", "MSFT", "HD")


def context(scenario="cash", future=100.0):
    bars = []
    for symbol in SYMBOLS:
        for index, day in enumerate(DAYS):
            timestamp = datetime.fromisoformat(day + "T21:00:00+00:00")
            price = 100.0 if index <= 3 else future
            bars.append(
                MarketObservation(
                    observation_id=f"{symbol}-{day}",
                    kind="market",
                    symbol=symbol,
                    event_at=timestamp,
                    available_at=timestamp,
                    payload={"date": day, "open": price, "close": price, "volume": 1000},
                )
            )
    return {
        "id": "test-" + scenario,
        "state_kind": scenario,
        "source_episode": "synthetic-train",
        "decision_date": DAYS[3] + "T21:00:00+00:00",
        "env_kwargs": {
            "adapter": StockBenchAdapter(bars, data_version="synthetic-coldstart-v2"),
            "episode_dates": [day + "T21:00:00+00:00" for day in DAYS],
            "episode_open_times": [day + "T14:30:00+00:00" for day in DAYS],
            "symbols": list(SYMBOLS),
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
            "drawdown_coefficient": 0.0,
            "turnover_coefficient": 0.0,
            "benchmark_closes": [1000.0] * 7,
            "reward_mode": "outcome_only",
            "observation_profile": "history_features_v1",
            "screening": True,
            "finance_scorer_id": "benchmark_scale_v2",
            "finance_volatility_floor": 0.001,
            "risk_lambda": 0.5,
            "turnover_limit": 1.0,
            "position_limit": 0.3,
            "screening_calibration_sha256": "a" * 64,
        },
        "raw_prompt": [{"role": "user", "content": "Manage the portfolio."}],
        "prefix": fixed_prefix(scenario),
        "prefix_expected_invalid": [False] * 3,
        "prefix_expected_errors": [None] * 3,
        "expected_start_session_index": 3,
    }


class JsonTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages, sort_keys=True)


def test_real_prefixes_create_cash_single_pair_and_three_stock_states():
    expected = (0, 1, 1, 2, 2, 3, 3, 3)
    for scenario, count in zip(SCENARIOS, expected, strict=True):
        env, obs, info = asyncio.run(replay_context(context(scenario)))
        assert env.current_session_index == 3
        assert len(parse_state(obs)["positions"]) == count
        assert len(env.ledger.fills) == count
        assert len(info) == 3
    _, below, _ = asyncio.run(replay_context(context("triple_below")))
    _, above, _ = asyncio.run(replay_context(context("triple_above")))
    assert all(p["weight"] < 0.2 for p in parse_state(below)["positions"].values())
    assert all(p["weight"] > 0.2 for p in parse_state(above)["positions"].values())


def test_recovery_is_real_rejection_without_state_change_and_candidates_remain_legal():
    for scenario in RECOVERY_SCENARIOS:
        clean = context(scenario)
        clean_env, clean_obs, _ = asyncio.run(replay_context(clean))
        failed = asyncio.run(recovery_context(clean))
        env, obs, info = asyncio.run(replay_context(failed))
        assert info[-1]["invalid_action"]
        assert "error" in parse_state(obs)
        assert len(env.ledger.fills) == len(clean_env.ledger.fills)
        for key in ("cash", "positions", "portfolio_equity", "session_index"):
            assert parse_state(obs)[key] == parse_state(clean_obs)[key]
        for call in candidate_calls(parse_state(obs)):
            asyncio.run(score_candidate(failed, call))


def test_future_changes_labels_economics_but_never_state_prompt_or_error_prefix():
    a = context("triple_above", future=120.0)
    b = context("triple_above", future=60.0)
    aa, bb = asyncio.run(recovery_context(a)), asyncio.run(recovery_context(b))
    ea, oa, _ = asyncio.run(replay_context(aa))
    eb, ob, _ = asyncio.run(replay_context(bb))
    assert aa["prefix"] == bb["prefix"]
    assert parse_state(oa) == parse_state(ob)
    assert render(JsonTokenizer(), ea, oa) == render(JsonTokenizer(), eb, ob)
    assert (
        asyncio.run(score_candidate(aa, tool_call("hold")))["terminal_equity"]
        != asyncio.run(score_candidate(bb, tool_call("hold")))["terminal_equity"]
    )


def test_api_direction_and_preservation_hold_use_actual_current_weights():
    from agent_r1.env.base import Action

    async def verify():
        directions = set()
        for date_index in range(3):
            for scenario_index, scenario in enumerate(SCENARIOS):
                c = context(scenario)
                _, obs, _ = await replay_context(c)
                for _, completion in api_targets(parse_state(obs), date_index, scenario_index):
                    env, before, _ = await replay_context(c)
                    old_fills = len(env.ledger.fills)
                    _, _, _, info = await env.step(Action(text=completion))
                    assert not info.get("invalid_action")
                    name = json.loads(completion[11:-12])["name"]
                    directions.add(name)
                    if name == "hold":
                        assert len(env.ledger.fills) == old_fills
                        assert env.current_session_index == parse_state(before)["session_index"] + 1
        assert directions == {"hold", "buy", "sell"}

    asyncio.run(verify())


def test_outcome_score_uses_next_open_and_actual_costs():
    score = asyncio.run(
        score_candidate(
            context(future=200.0), tool_call("buy", {"symbol": "GS", "target_weight": 0.2})
        )
    )
    assert score["candidate_fills"][0]["raw_open_price"] == 200.0
    assert score["candidate_fills"][0]["executed_at"][:10] == DAYS[4]
    assert score["candidate_cost"] > 0
    assert score["terminal_equity"] == pytest.approx(1000.0 - score["candidate_cost"])


def test_dates_deduplicate_before_fixed_stride_and_spans_do_not_claim_full_21_sessions():
    c = context()
    kwargs = {k: v for k, v in c["env_kwargs"].items() if k != "adapter"}
    row = {
        "env_kwargs": json.dumps(kwargs),
        "prompt": c["raw_prompt"],
        "extra_info": {"episode_id": "formal-train-0001", "source_split": "train"},
    }
    contexts = training_contexts([row, copy.deepcopy(row)])
    assert len(contexts) == 8
    assert source_dates(contexts) == set(kwargs["episode_dates"])
    gap_days = [
        "2023-05-23",
        "2023-05-24",
        "2023-05-25",
        "2023-05-26",
        "2023-05-30",
        "2023-05-31",
        "2023-06-01",
        "2023-06-02",
        "2023-06-05",
        "2023-06-06",
        "2023-06-07",
        "2023-06-08",
        "2023-06-09",
        "2023-06-12",
        "2023-06-13",
    ]
    gap_kwargs = {"episode_dates": [d + "T20:00:00+00:00" for d in gap_days]}
    gap_row = {**row, "env_kwargs": json.dumps(gap_kwargs)}
    _, start, stop = find_span([gap_row], "2023-05-23", "2023-06-13")
    assert stop - start == 15
    assert not source_dates(contexts) & set(gap_kwargs["episode_dates"])
    assert all("2023-05-22" < day < "2023-06-14" for day in gap_days)
    assert evenly_spaced(list(range(372)), 76)[0] == 0
    assert evenly_spaced(list(range(372)), 76)[-1] == 371
