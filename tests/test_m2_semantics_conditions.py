"""Actual CPU environment tests for independent availability/time display factors."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from agent_r1.env.base import Action

from src.data import MarketObservation, StockBenchAdapter

_PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_semantics_conditions.py"
_SPEC = importlib.util.spec_from_file_location("semantics_condition_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
S = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(S)
L = S.LEGACY
SYMBOLS = tuple(f"S{i:02}" for i in reversed(range(20)))
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
]
DATES = tuple(datetime.fromisoformat(day + "T21:00:00+00:00") for day in DAYS)
RAW_PROMPT = [{"role": "user", "content": "Manage the portfolio using only point-in-time tools."}]


def observations() -> list[MarketObservation]:
    return [
        MarketObservation(
            observation_id=f"bar:{symbol}:{day}",
            kind="market",
            symbol=symbol,
            event_at=at,
            available_at=at,
            payload={
                "date": day,
                "open": 100.0 + si * 2 + i / 2,
                "close": 100.123 + si * 2 + i / 2,
            },
        )
        for si, symbol in enumerate(SYMBOLS)
        for i, (day, at) in enumerate(zip(DAYS, DATES, strict=True))
    ]


def kwargs(*, offset: int = 0, records: list[MarketObservation] | None = None) -> dict[str, Any]:
    dates = DATES[offset : offset + 21]
    return {
        "adapter": StockBenchAdapter(
            observations() if records is None else records, data_version="synthetic-semantics-v1"
        ),
        "episode_dates": dates,
        "episode_open_times": [at.replace(hour=14, minute=30) for at in dates],
        "symbols": SYMBOLS,
        "initial_cash": 1000.0,
        "commission_bps": 10.0,
        "slippage_bps": 5.0,
        "drawdown_coefficient": 0.0,
        "turnover_coefficient": 0.0,
        "benchmark_closes": [100.0 + i for i in range(21)],
        "screening": True,
        "finance_scorer_id": "benchmark_scale_v2",
        "finance_volatility_floor": 0.001,
        "risk_lambda": 0.5,
        "turnover_limit": 1.0,
        "position_limit": 0.3,
        "screening_calibration_sha256": "a" * 64,
        "raw_prompt": copy.deepcopy(RAW_PROMPT),
    }


def state(observation: Any) -> dict[str, Any]:
    return json.loads(
        observation.messages[-1]["content"]
        .removeprefix("<tool_response>\n")
        .removesuffix("\n</tool_response>")
    )


def action(name: str, args: dict[str, Any] | None = None) -> Action:
    return Action(
        text="<tool_call>" + json.dumps({"name": name, "arguments": args or {}}) + "</tool_call>"
    )


def step(env: Any, name: str, args: dict[str, Any] | None = None):
    return asyncio.run(env.step(action(name, args)))


@pytest.mark.parametrize(
    "condition,start",
    [
        (None, 0),
        (True, 0),
        (False, 0),
        ("raw", 0),
        ("features", 0),
        ("both", 0),
        ("memory_only", False),
        ("availability", 0.0),
        ("time_semantics", 4),
        ("availability_time", -1),
    ],
)
def test_only_registered_factor_grid_and_integer_zero_start(condition, start):
    with pytest.raises(ValueError):
        S.validate_condition(condition, start)


@pytest.mark.parametrize("condition", S.CONDITIONS)
def test_factor_structure_is_exact_independent_and_keeps_old_memory(condition):
    env = S.SemanticsConditionEnv(condition=condition, **kwargs())
    observation, receipt = asyncio.run(S.reset_with_prefix(env))
    document = state(observation)
    assert receipt["prefix_action_count"] == receipt["model_requests"] == 0
    assert receipt["is_model_request"] is False
    assert env.observation_profile == "history_memory_v1"
    assert "tool_result_memory" in document and "compute_history" not in document
    assert "decision_features" not in document
    has_availability = condition in {"availability", "availability_time"}
    has_time = condition in {"time_semantics", "availability_time"}
    assert (S.AVAILABILITY_KEY in document) == has_availability
    assert (S.TIME_KEY in document) == has_time
    if has_availability:
        assert document[S.AVAILABILITY_KEY] == {
            "as_of": DATES[0].isoformat(),
            "explanation": S.AVAILABILITY_TEXT,
            "columns": ["symbol", "status"],
            "rows": [[symbol, "unavailable"] for symbol in SYMBOLS],
        }
    if has_time:
        assert document[S.TIME_KEY] == S.TIME_TEXT
    proof = env.semantic_evidence
    assert proof == env.feature_evidence and proof["kind"] == "semantic_display"
    assert proof["transport_field"] == "feature_evidence"
    assert proof["meaning"] == "semantics diagnostic evidence, not precomputed features"
    assert proof["factors"] == {"availability": has_availability, "time_semantics": has_time}
    assert proof["displayed_state_fields"] == {
        key: document[key] for key in (S.AVAILABILITY_KEY, S.TIME_KEY) if key in document
    }
    assert proof["displayed_fields_sha256"] == S._sha(proof["displayed_state_fields"])
    assert proof["memory_display_sha256"] == S._sha(document["tool_result_memory"])


def test_status_is_per_symbol_uses_PIT_boundary_and_not_episode_session_count():
    # At episode session 0, S19 has one visible close; the other 19 symbols
    # already have five. No history is padded or clipped to episode age.
    records = [r for r in observations() if r.symbol != SYMBOLS[0] or r.event_at >= DATES[4]]
    old = next(r for r in observations() if r.symbol == SYMBOLS[0] and r.event_at == DATES[0])
    delayed = replace(
        old, observation_id="delayed-close", available_at=DATES[4] + timedelta(microseconds=1)
    )
    records.append(delayed)
    env = S.SemanticsConditionEnv(condition="availability", **kwargs(offset=4, records=records))
    initial = state(env.reset())
    assert env.current_session_index == 0
    assert initial[S.AVAILABILITY_KEY]["rows"] == [
        [s, "unavailable" if s == SYMBOLS[0] else "ready"] for s in SYMBOLS
    ]
    assert env.diagnostic_pit_counts[SYMBOLS[0]] == 1
    assert all(env.diagnostic_pit_counts[s] == 5 for s in SYMBOLS[1:])
    records[-1] = replace(delayed, available_at=DATES[4])
    inclusive = S.SemanticsConditionEnv(
        condition="availability", **kwargs(offset=4, records=records)
    )
    inclusive.reset()
    assert inclusive.diagnostic_pit_counts[SYMBOLS[0]] == 2
    assert inclusive.semantic_evidence["availability_by_symbol"][SYMBOLS[0]] == "ready"


def test_unavailable_does_not_disable_schema_or_change_errors_and_long_lookback_is_valid():
    old = L.DecisionConditionEnv(condition="memory_only", start_session=0, **kwargs())
    new = S.SemanticsConditionEnv(condition="availability_time", **kwargs())
    old.reset()
    new.reset()
    assert old.tool_schemas == new.tool_schemas
    arguments = {"symbol": SYMBOLS[0], "metric": "mean_return", "lookback": 60}
    for env in (old, new):
        bad = step(env, "compute", arguments)
        assert bad[3]["invalid_action"] is True and env.current_session_index == 0
        assert "at least two visible close prices" in bad[3]["error"]
        step(env, "hold")
        success = step(env, "compute", arguments)
        assert not success[3].get("invalid_action")
        assert state(success[0])["last_tool_result"]["lookback_used"] == 2
        first_result = state(success[0])["last_tool_result"]
        repeated = step(env, "compute", arguments)
        assert env.current_session_index == 1
        assert state(repeated[0])["last_tool_result"] == first_result
    assert old.credit_step_trace == new.credit_step_trace
    assert old._tool_result_memory.render() == new._tool_result_memory.render()
    assert new.semantic_evidence["availability_by_symbol"][SYMBOLS[0]] == "ready"


def scripted_calls() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("compute", {"symbol": SYMBOLS[0], "metric": "mean_return", "lookback": 5}),
        ("unknown", {}),
        ("retrieve", {"query": "missing"}),
        ("hold", {}),
        ("compute", {"symbol": SYMBOLS[0], "metric": "mean_return", "lookback": 60}),
        ("compute", {"symbol": SYMBOLS[1], "metric": "volatility", "lookback": 5}),
        ("compute", {"symbol": SYMBOLS[1], "metric": "volatility", "lookback": 5}),
        ("abstain", {"reason": "scripted CPU parity check"}),
        ("retrieve", {"query": "still missing", "symbol": SYMBOLS[1]}),
        ("buy", {"symbol": SYMBOLS[0], "target_weight": 0.2}),
        ("sell", {"symbol": SYMBOLS[0], "target_weight": 0.0}),
        *[("hold", {}) for _ in range(16)],
    ]


@pytest.mark.parametrize("offset", [0, 4])
def test_all_scripted_observations_match_old_baseline_and_economics(offset):
    config = kwargs(offset=offset)
    old = L.DecisionConditionEnv(condition="memory_only", start_session=0, **config)
    environments = {c: S.SemanticsConditionEnv(condition=c, **config) for c in S.CONDITIONS}
    source_before = config["adapter"].get_observations(DATES[-1]).canonical_bytes()
    baseline = old.reset()
    initial = {c: env.reset() for c, env in environments.items()}
    assert initial["memory_only"].messages == baseline.messages
    for name, arguments in scripted_calls():
        reference = step(old, name, arguments)
        results = {c: step(env, name, arguments) for c, env in environments.items()}
        assert results["memory_only"][0].messages == reference[0].messages
        for c, result in results.items():
            assert result[1:] == reference[1:]
            document = state(result[0])
            assert {
                k: v
                for k, v in document.items()
                if k
                not in {
                    S.AVAILABILITY_KEY,
                    S.TIME_KEY,
                }
            } == state(reference[0])
            assert environments[c].credit_step_trace == old.credit_step_trace
            assert environments[c]._tool_result_memory.render() == old._tool_result_memory.render()
            assert environments[c].ledger.cash == old.ledger.cash
            assert environments[c].ledger.positions == old.ledger.positions
            assert environments[c].ledger.fills == old.ledger.fills
            assert environments[c].tool_schemas == old.tool_schemas
        if name == "abstain":
            memory = state(results["memory_only"][0])["tool_result_memory"]
            assert {r[0] for r in memory["compute_rows"]} == set(SYMBOLS[:2])
            assert all(
                r[4] != environments["memory_only"].current_date.isoformat()
                for r in memory["compute_rows"]
            )
    assert old.current_session_index == 20 and old.episode_complete
    assert (
        sum(s.result_session_index > s.decision_session_index for s in old.credit_step_trace) == 20
    )
    for env in environments.values():
        assert env.screening_terminal_metrics() == old.screening_terminal_metrics()
    assert config["adapter"].get_observations(DATES[-1]).canonical_bytes() == source_before


def test_evidence_detached_zero_compute_scan_and_reset_does_not_leak_previous_episode():
    env = S.SemanticsConditionEnv(condition="availability_time", **kwargs())
    with patch.object(env, "_compute", side_effect=AssertionError("display must not compute")):
        env.reset()
        step(env, "hold")
    with patch.object(env, "_snapshot", side_effect=AssertionError("property must not scan")):
        evidence = env.semantic_evidence
        counts = env.diagnostic_pit_counts
        assert evidence["session_index"] == 1 and set(counts.values()) == {2}
        evidence["displayed_state_fields"].clear()
        evidence["diagnostic_pit_counts"].clear()
        counts.clear()
        assert env.feature_evidence["displayed_state_fields"]
        assert env.diagnostic_pit_counts
    step(env, "compute", {"symbol": SYMBOLS[0], "metric": "price_change", "lookback": 2})
    assert env._tool_result_memory.render()["retained_count"] == 1
    reset = state(env.reset())
    assert reset["tool_result_memory"]["retained_count"] == 0
    assert env.semantic_evidence["session_index"] == 0
    assert set(env.semantic_evidence["availability_by_symbol"].values()) == {"unavailable"}


def test_renderer_is_exact_frozen_function_preserves_task_and_does_not_truncate():
    assert S.render_prompt is L.render_prompt
    env = S.SemanticsConditionEnv(condition="availability_time", **kwargs())
    observation = env.reset()
    original = copy.deepcopy(observation.messages)

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert options == {
                "tools": env.tool_schemas,
                "add_generation_prompt": True,
                "tokenize": True,
                "enable_thinking": False,
            }
            assert messages[1:] == original
            assert "exactly one tool call" in messages[0]["content"]
            return [42] * 8193

    rendered = S.render_prompt(Tokenizer(), observation, env.tool_schemas)
    assert rendered["prompt_token_count"] == len(rendered["prompt_ids"]) == 8193
    assert observation.messages == original


def test_create_env_keeps_input_and_rejects_other_split():
    rows: list[dict[str, Any]] = [
        {
            "extra_info": {
                "episode_id": episode,
                "source_index": i * 4,
                "index": i,
                "source_split": "train",
                "screening_role": "pilot_train",
            },
            "env_kwargs": json.dumps({"screening": True}),
            "prompt": RAW_PROMPT,
        }
        for i, episode in enumerate(S.EPISODES)
    ]
    with patch.object(S, "SemanticsConditionEnv") as constructor:
        S.create_env(rows, 1, "availability_time", 0)
        assert constructor.call_args.kwargs == {
            "condition": "availability_time",
            "start_session": 0,
            "screening": True,
            "raw_prompt": RAW_PROMPT,
        }
        assert constructor.call_args.kwargs["raw_prompt"] is not RAW_PROMPT
    rows[0]["extra_info"]["source_split"] = "dev"
    with pytest.raises(ValueError, match="Train window identity"):
        S.create_env(rows, 0, "memory_only")
