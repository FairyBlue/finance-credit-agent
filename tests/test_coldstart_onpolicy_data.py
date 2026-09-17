"""Teacher branch isolation and exact replay of policy-visited Train states."""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime

import pytest
from agent_r1.env.base import Action

from src.agents.coldstart_data import parse_state, render_prompt, tool_call
from src.agents.coldstart_onpolicy_data import (
    PROTOCOL,
    complete_request,
    normalized,
    replay_to_request,
    score_candidate,
    select_states,
)
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv

DAYS = [
    "2023-01-03",
    "2023-01-04",
    "2023-01-05",
    "2023-01-06",
    "2023-01-09",
    "2023-01-10",
    "2023-01-11",
    "2023-01-12",
    "2023-01-13",
]
DATES = [day + "T21:00:00+00:00" for day in DAYS]


class JsonTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages, sort_keys=True)

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


def context(late_price=100.0, horizon_price=100.0):
    observations = []
    for symbol in ("GS", "MSFT", "HD"):
        for index, day in enumerate(DAYS):
            timestamp = datetime.fromisoformat(DATES[index])
            price = 100.0 if index <= 2 else horizon_price if index <= 5 else late_price
            observations.append(
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
        "id": "train",
        "source_episode": "train",
        "raw_prompt": [{"role": "user", "content": "Manage the portfolio."}],
        "prefix": [],
        "env_kwargs": {
            "adapter": StockBenchAdapter(observations, data_version="synthetic-onpolicy"),
            "episode_dates": DATES,
            "episode_open_times": [d + "T14:30:00+00:00" for d in DAYS],
            "symbols": ["GS", "MSFT", "HD"],
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 5.0,
            "drawdown_coefficient": 0.0,
            "turnover_coefficient": 0.0,
            "benchmark_closes": [1000.0] * len(DAYS),
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
    }


async def real_source(ctx):
    tokenizer = JsonTokenizer()
    env = TradingEnv(**ctx["env_kwargs"])
    obs = env.reset(raw_prompt=ctx["raw_prompt"])
    records = []
    calls = [
        tool_call("buy", {"symbol": "GS", "target_weight": 0.2}),
        tool_call("hold"),
        tool_call("buy", {"symbol": "GS", "target_weight": 0.1}),
        tool_call("hold"),
    ]
    for index, call in enumerate(calls):
        prompt = render_prompt(tokenizer, env, obs, action_protocol=PROTOCOL)
        record = {
            "trajectory_id": "t",
            "request_index": index,
            "prompt": prompt,
            "prompt_token_ids": tokenizer.encode(prompt),
            "response": call,
            "response_token_ids": [1, 2],
            "finish_reason": "stop",
            "sampling_params": {"max_tokens": 4096},
        }
        obs, reward, _, info = await env.step(Action(text=call, token_ids=[1, 2]))
        record.update(
            trace=normalized(asdict(env.credit_step_trace[-1])),
            info=normalized(info),
            reward=reward,
        )
        records.append(record)
    row = {
        "id": "t",
        "context": ctx,
        "status": "INCOMPLETE",
        "failure": "REQUEST_CAP",
        "requests": 4,
        "invalid": 1,
    }
    return {"row": row, "record": records[3], "records": records, "aliases": []}, tokenizer


def test_real_invalid_prefix_replays_without_advancing_and_exact_prompt_tokens_match():
    source, tokenizer = asyncio.run(real_source(context()))
    env, obs = asyncio.run(replay_to_request(source, tokenizer))
    state = parse_state(obs)
    assert env.current_session_index == 2
    assert state["sessions_remaining"] == 6
    assert len(env.ledger.fills) == 1
    assert state["error"].startswith("Error: buy target_weight")
    assert source["records"][2]["trace"]["invalid_action"]
    assert len(env.credit_step_trace) == 3
    assert source["row"]["status"] == "INCOMPLETE"  # Prefix can still be used for Train.


def test_prefix_prompt_or_token_tampering_is_rejected():
    source, tokenizer = asyncio.run(real_source(context()))
    source["records"][1]["prompt"] += " changed"
    with pytest.raises(ValueError, match="prompt mismatch"):
        asyncio.run(replay_to_request(source, tokenizer))
    source, tokenizer = asyncio.run(real_source(context()))
    source["records"][0]["prompt_token_ids"] = [999]
    with pytest.raises(ValueError, match="token IDs"):
        asyncio.run(replay_to_request(source, tokenizer))


def test_each_teacher_branch_restarts_same_prefix_and_stops_at_three_sessions():
    source, tokenizer = asyncio.run(real_source(context(late_price=1000.0)))
    hold = asyncio.run(score_candidate(source, tokenizer, tool_call("hold")))
    sell = asyncio.run(
        score_candidate(
            source, tokenizer, tool_call("sell", {"symbol": "GS", "target_weight": 0.0})
        )
    )
    hold_again = asyncio.run(score_candidate(source, tokenizer, tool_call("hold")))
    assert hold == hold_again
    assert hold["valuation_session_index"] == 5
    assert hold["valuation_date"] == DATES[5]
    assert not hold["full_episode_terminal_at_valuation"]
    assert hold["terminal_equity"] < 1000  # No gain from the later 10x price is used.
    assert sell["candidate_cost"] > 0
    assert sell["candidate_fills"][0]["executed_at"][:10] == DAYS[3]
    assert source["records"][2]["info"]["invalid_action"]  # Original refusal never repaired.


def test_future_changes_teacher_outcome_not_model_visible_state_or_prefix():
    a, tokenizer = asyncio.run(real_source(context(horizon_price=120.0)))
    b, _ = asyncio.run(real_source(context(horizon_price=60.0)))
    assert a["record"]["prompt"] == b["record"]["prompt"]
    assert a["record"]["prompt_token_ids"] == b["record"]["prompt_token_ids"]
    assert [r["response"] for r in a["records"][:3]] == [r["response"] for r in b["records"][:3]]
    sa = asyncio.run(score_candidate(a, tokenizer, tool_call("hold")))
    sb = asyncio.run(score_candidate(b, tokenizer, tool_call("hold")))
    assert sa["terminal_equity"] != sb["terminal_equity"]


def fake_record(tid, index, prompt, *, invalid=False, session=0, incomplete=False):
    record = {
        "trajectory_id": tid,
        "request_index": index,
        "prompt": prompt,
        "prompt_token_ids": [1],
        "response": tool_call("hold"),
        "response_token_ids": [1],
        "finish_reason": "stop",
        "sampling_params": {"max_tokens": 4096},
        "info": {"invalid_action": invalid},
        "trace": {
            "invalid_action": invalid,
            "decision_session_index": session,
            "decision_at": DATES[session],
        },
    }
    if incomplete:
        record["finish_reason"] = "length"
        record.pop("trace")
        record.pop("info")
    return record


def fake_result(records):
    tids = list(dict.fromkeys(record["trajectory_id"] for record in records))
    return {
        "status": "OBSERVED",
        "grid": [
            {
                "id": tid,
                "context": {"source_episode": "train", "env_kwargs": {"episode_dates": DATES}},
                "status": "INCOMPLETE",
                "failure": "REQUEST_CAP",
                "requests": sum(record["trajectory_id"] == tid for record in records),
                "invalid": sum(
                    record["trajectory_id"] == tid
                    and record.get("trace", {}).get("invalid_action", False)
                    for record in records
                ),
            }
            for tid in tids
        ],
    }


def test_prompt_aliases_promote_errors_before_clean_dedup_and_budget_never_drops_them():
    records = [
        fake_record("a", 0, "same-clean-or-error"),
        fake_record("a", 1, "odd-clean"),
        fake_record("a", 2, "clean-even-a"),
        fake_record("b", 0, "same-clean-or-error", invalid=True),
        fake_record("b", 1, "post-error-b"),
    ]
    selected, receipt = select_states(fake_result(records), records, cap=3)
    assert receipt["priority_unique_prompts"] == 2
    assert [source["record"]["prompt"] for source in selected] == [
        "same-clean-or-error",
        "post-error-b",
        "clean-even-a",
    ]
    assert selected[0]["record"]["trajectory_id"] == "b"
    assert len(selected[0]["aliases"]) == 2
    assert {alias["strata"][0] for alias in selected[0]["aliases"]} == {"invalid", "clean_even"}
    _, blocked = select_states(fake_result(records), records, cap=1)
    assert blocked["status"] == "BLOCKED_PRIORITY_EXCEEDS_CAP"
    assert blocked["priority_unique_prompts"] == 2
    assert blocked["selected_unique_prompts"] == 2  # Evidence is retained, never truncated to one.


def test_final_two_sessions_and_incomplete_requests_are_excluded_explicitly():
    records = [
        fake_record("a", 0, "eligible", session=5),
        fake_record("a", 1, "tail-error", invalid=True, session=7),
        fake_record("a", 2, "tail-next", session=7),
        fake_record("a", 3, "incomplete", incomplete=True),
    ]
    selected, receipt = select_states(fake_result(records), records)
    assert len(selected) == 1
    assert receipt["excluded_request_counts"] == {
        "fewer_than_three_future_sessions": 2,
        "incomplete_or_unexecuted_request": 1,
    }
    assert receipt["excluded_invalid_or_post_error_counts"]["fewer_than_three_future_sessions"] == 2
    assert not complete_request(records[-1])
