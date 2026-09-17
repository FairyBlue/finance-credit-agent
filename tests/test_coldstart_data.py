"""Real ledger and PIT checks for the offline common cold-start curriculum."""

import asyncio
import copy
import json
from datetime import datetime

import pytest

from src.agents.coldstart_data import (
    TEACHER_SYMBOLS,
    candidate_calls,
    contexts_from_rows,
    parse_state,
    prefix_for,
    render_prompt,
    replay_context,
    score_candidate,
    select_label,
    source_dates,
    tool_call,
)
from src.data import MarketObservation, StockBenchAdapter

DAYS = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]


def context(future=110.0, state_kind="cash"):
    bars = []
    for symbol in TEACHER_SYMBOLS:
        for index, day in enumerate(DAYS):
            timestamp = datetime.fromisoformat(day + "T21:00:00+00:00")
            price = 100.0 if index <= 1 else future
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
        "env_kwargs": {
            "adapter": StockBenchAdapter(bars, data_version="synthetic-coldstart"),
            "episode_dates": [day + "T21:00:00+00:00" for day in DAYS],
            "episode_open_times": [day + "T14:30:00+00:00" for day in DAYS],
            "symbols": list(TEACHER_SYMBOLS),
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
            "drawdown_coefficient": 0.0,
            "turnover_coefficient": 0.0,
            "benchmark_closes": [1000] * 5,
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
        "state_kind": state_kind,
        "prefix": prefix_for(state_kind),
    }


class JsonTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages, sort_keys=True)


def test_future_price_changes_outcomes_not_input_or_prefix():
    a, b = context(110, "GS"), context(60, "GS")
    ea, oa = asyncio.run(replay_context(a))
    eb, ob = asyncio.run(replay_context(b))
    assert a["prefix"] == b["prefix"] == prefix_for("GS")
    assert parse_state(oa) == parse_state(ob)
    assert render_prompt(JsonTokenizer(), ea, oa) == render_prompt(JsonTokenizer(), eb, ob)
    calls = candidate_calls(parse_state(oa))
    sa = [asyncio.run(score_candidate(a, call)) for call in calls]
    sb = [asyncio.run(score_candidate(b, call)) for call in calls]
    # All information before the candidate action is identical; only labels may change.
    assert sa[0]["terminal_equity"] != sb[0]["terminal_equity"]


def test_teacher_scores_actual_next_open_and_costs_not_decision_close():
    c = context(future=200)
    score = asyncio.run(
        score_candidate(c, tool_call("buy", {"symbol": "GS", "target_weight": 0.2}))
    )
    fill = score["candidate_fills"][0]
    assert fill["raw_open_price"] == 200
    assert fill["executed_at"][:10] == DAYS[2]
    assert score["candidate_cost"] > 0
    # Price doubled before the order could execute; no 100% capital gain exists.
    assert score["terminal_equity"] == pytest.approx(1000 - score["candidate_cost"])
    hold = asyncio.run(score_candidate(c, tool_call("hold")))
    assert select_label([hold, score]) is hold


def test_prefix_is_real_trade_and_candidate_direction_is_legal():
    c = context(state_kind="GS")
    env, obs = asyncio.run(replay_context(c))
    assert len(env.ledger.fills) == 1
    assert env.ledger.fills[0].side == "buy"
    assert env.current_session_index == 1
    calls = candidate_calls(parse_state(obs))
    assert tool_call("sell", {"symbol": "GS", "target_weight": 0.0}) in calls
    for call in calls:
        asyncio.run(score_candidate(c, call))


def test_hold_wins_economic_ties():
    hold = {"completion": tool_call("hold"), "terminal_equity": 1000.0}
    other = {
        "completion": tool_call("buy", {"symbol": "GS", "target_weight": 0.1}),
        "terminal_equity": 1000.0 + 1e-8,
    }
    assert select_label([hold, other]) is hold


def test_source_windows_deduplicate_dates_and_keep_label_horizon_isolated():
    c = context()
    kw = {k: v for k, v in c["env_kwargs"].items() if k != "adapter"}
    row = {
        "env_kwargs": json.dumps(kw),
        "prompt": c["raw_prompt"],
        "extra_info": {"episode_id": "formal-train-0001", "source_split": "train"},
    }
    contexts = contexts_from_rows([row, copy.deepcopy(row)])
    assert len(contexts) == 4
    assert {ctx["decision_date"] for ctx in contexts} == {DAYS[1] + "T21:00:00+00:00"}
    assert source_dates(contexts) == set(kw["episode_dates"])
    assert not source_dates(contexts) & {"2023-06-14T20:00:00+00:00"}
