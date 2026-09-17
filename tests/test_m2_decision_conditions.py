"""Actual CPU environment contracts for the diagnostic-only observation module."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from agent_r1.env.base import Action, Observation

from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import ComputeArgs, TradingEnv

_PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_decision_conditions.py"
_SPEC = importlib.util.spec_from_file_location("decision_conditions_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)
CONDITIONS, FEATURE_COLUMNS, FEATURE_KEY, HOLD_TEXT, METRICS = (
    D.CONDITIONS, D.FEATURE_COLUMNS, D.FEATURE_KEY, D.HOLD_TEXT, D.METRICS,
)
DecisionConditionEnv, create_env, render_prompt = (
    D.DecisionConditionEnv, D.create_env, D.render_prompt,
)
reset_with_prefix, validate_condition = D.reset_with_prefix, D.validate_condition

DAYS = [
    "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09",
    "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-17",
    "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-23", "2023-01-24",
    "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-30", "2023-01-31", "2023-02-01",
]
DATES = tuple(datetime.fromisoformat(day + "T21:00:00+00:00") for day in DAYS)
SYMBOLS = tuple(f"T{index:02}" for index in reversed(range(20)))
RAW_PROMPT = [{"role": "user", "content": "Manage the portfolio using only point-in-time tools."}]


def bars() -> list[MarketObservation]:
    return [MarketObservation(
        observation_id=f"bar:{symbol}:{day}", kind="market", symbol=symbol,
        event_at=stamp, available_at=stamp,
        payload={"date": day, "open": 100.0 + si * 3 + index / 2,
                 "close": 100.1 + si * 3 + index / 2},
    ) for si, symbol in enumerate(SYMBOLS)
        for index, (day, stamp) in enumerate(zip(DAYS, DATES, strict=True))]


def kwargs(observations: list[MarketObservation] | None = None) -> dict[str, Any]:
    return {
        "adapter": StockBenchAdapter(bars() if observations is None else observations,
                                     data_version="synthetic-decision-conditions"),
        "episode_dates": DATES,
        "episode_open_times": [stamp.replace(hour=14, minute=30) for stamp in DATES],
        "symbols": SYMBOLS, "initial_cash": 1000.0,
        "commission_bps": 10.0, "slippage_bps": 5.0,
        "drawdown_coefficient": 0.0, "turnover_coefficient": 0.0,
        "benchmark_closes": [100.0 + i for i in range(21)],
        "screening": True, "finance_scorer_id": "benchmark_scale_v2",
        "finance_volatility_floor": 0.001, "risk_lambda": 0.5,
        "turnover_limit": 1.0, "position_limit": 0.3,
        "screening_calibration_sha256": "a" * 64,
        "reward_mode": "outcome_only", "capture_interactions": True,
        "raw_prompt": copy.deepcopy(RAW_PROMPT),
    }


def env(condition: str = "raw", start: int = 0, **changes: Any) -> Any:
    return DecisionConditionEnv(condition=condition, start_session=start,
                                **{**kwargs(), **changes})


def state(observation: Observation) -> dict[str, Any]:
    assert observation.messages is not None
    content = observation.messages[-1]["content"]
    return json.loads(content.removeprefix("<tool_response>\n")
                     .removesuffix("\n</tool_response>"))


def action(name: str, arguments: dict[str, Any] | None = None) -> Action:
    return Action(text="<tool_call>" + json.dumps({
        "name": name, "arguments": arguments or {},
    }) + "</tool_call>")


def step(environment: TradingEnv, name: str, arguments: dict[str, Any] | None = None):
    return asyncio.run(environment.step(action(name, arguments)))


@pytest.mark.parametrize("condition,start", [
    (True, 0), (False, 0), (None, 0), ("legacy", 0), ("unknown", 0),
    ("raw", True), ("raw", 4.0), ("raw", -1), ("raw", 1),
    ("features", 0), ("history_only", 4), ("memory_only", 4), ("both", 4),
])
def test_rejects_unregistered_conditions_and_start_pairs(condition, start):
    with pytest.raises(ValueError):
        validate_condition(condition, start)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_only_registered_display_fields_change_at_reset(condition):
    actual = env(condition, 4 if condition == "features" else 0)
    observation = actual.reset()
    result = state(observation)
    assert ("compute_history" in result) == (condition in {"history_only", "both"})
    assert ("tool_result_memory" in result) == (condition in {"memory_only", "both"})
    assert FEATURE_KEY not in result  # features begin only at the registered session 4
    original = TradingEnv(**kwargs(), observation_profile="legacy")
    baseline = original.reset()
    assert {k: v for k, v in result.items() if k not in {
        "compute_history", "tool_result_memory", FEATURE_KEY,
    }} == state(baseline)
    assert actual.tool_schemas == original.tool_schemas
    assert actual.observation_profile == "history_memory_v1"


def test_history_shortage_is_retained_and_never_padded():
    for condition in ("raw", "history_only", "memory_only", "both"):
        actual = env(condition)
        initial, receipt = asyncio.run(reset_with_prefix(actual))
        assert receipt["prefix_action_count"] == 0
        if condition in {"history_only", "both"}:
            assert state(initial)["compute_history"]["available_close_records"][SYMBOLS[0]] == 1
        observation, reward, done, info = step(actual, "compute", {
            "symbol": SYMBOLS[0], "metric": "price_change", "lookback": 5,
        })
        assert reward == 0.0 and not done and info["invalid_action"]
        assert "at least two visible close prices" in json.dumps(state(observation))
        assert actual.credit_step_trace[-1].result_session_index == 0
        assert actual._tool_result_memory.render()["retained_count"] == 0


def test_ablation_hides_fields_only_and_keeps_successful_results_across_symbols():
    all_envs = [env(condition) for condition in ("raw", "history_only", "memory_only", "both")]
    for actual in all_envs:
        actual.reset()
    calls = [
        ("hold", {}),
        ("compute", {"symbol": SYMBOLS[0], "metric": "mean_return", "lookback": 5}),
        ("compute", {"symbol": SYMBOLS[1], "metric": "volatility", "lookback": 5}),
        ("hold", {}), ("unknown", {}),
    ]
    for name, arguments in calls:
        results = [step(actual, name, arguments) for actual in all_envs]
        assert all(result[1:] == results[0][1:] for result in results)
        states = [state(result[0]) for result in results]
        assert all({k: v for k, v in item.items() if k not in {
            "compute_history", "tool_result_memory",
        }} == states[0] for item in states)
    banks = [actual._tool_result_memory.render() for actual in all_envs]
    assert all(bank == banks[0] for bank in banks)
    assert {row[0] for row in banks[0]["compute_rows"]} == set(SYMBOLS[:2])
    assert all(row[4] == DATES[1].isoformat() for row in banks[0]["compute_rows"])
    for actual in all_envs:
        actual.reset()
        assert actual._tool_result_memory.render()["retained_count"] == 0


def mechanism_fingerprint(actual: TradingEnv) -> dict[str, Any]:
    return copy.deepcopy({
        "ledger": vars(actual.ledger), "recent_events": actual._recent_events,
        "credit": actual.credit_step_trace, "interactions": actual.interaction_trace,
        "bank": actual._tool_result_memory.render(), "index": actual._index,
        "tool_calls": actual._tool_calls, "invalid": actual._invalid_calls,
        "equity": actual._equity_curve, "returns": actual._net_returns,
    })


def test_prefix_and_feature_rows_are_real_ordered_exact_and_read_only():
    actual = env("features", 4)
    observation, receipt = asyncio.run(reset_with_prefix(actual))
    assert receipt["prefix_action_count"] == receipt["start_session"] == 4
    assert receipt["model_requests"] == 0 and receipt["is_model_request"] is False
    assert len(receipt["credit_step_trace"]) == len(receipt["interaction_trace"]) == 4
    assert [row["action"] for row in receipt["credit_step_trace"]] == ["hold"] * 4
    assert receipt["cash"] == 1000 and receipt["positions"] == {}
    table = state(observation)[FEATURE_KEY]
    assert table["columns"] == list(FEATURE_COLUMNS)
    assert [row[0] for row in table["rows"]] == list(SYMBOLS)
    assert table["as_of"] == DATES[4].isoformat() and table["lookback_requested"] == 5
    before = mechanism_fingerprint(actual)
    for symbol, used, *values in table["rows"]:
        assert used == 5
        for metric, value in zip(METRICS, values, strict=True):
            original = actual._compute(ComputeArgs(
                symbol=symbol, metric=metric, lookback=5,  # type: ignore[arg-type]
            ))
            assert value == original["value"]
    assert json.loads(actual._state_text())[FEATURE_KEY] == table
    assert mechanism_fingerprint(actual) == before
    proof = actual.feature_evidence
    assert proof is not None and proof["source_compute_calls"] == 60
    assert proof["cache_hit"] is True and proof["actual_compute_calls_this_render"] == 0
    assert len(proof["sources"]) == 20
    for source in proof["sources"]:
        assert len(source["selected_close_records"]) == 5
        assert len(source["compute_results"]) == 3
        assert all(row["available_at"] <= proof["as_of"]
                   for row in source["selected_close_records"])
    proof["sources"].clear()
    assert len(actual.feature_evidence["sources"]) == 20  # type: ignore[index]


def test_feature_cache_is_per_env_resets_and_refreshes_on_date_advance():
    actual = env("features", 4)
    with patch.object(actual, "_compute", wraps=actual._compute) as compute:
        initial, _ = asyncio.run(reset_with_prefix(actual))
        assert compute.call_count == 60
        assert actual.feature_evidence["actual_compute_calls_this_render"] == 60
        first = state(initial)[FEATURE_KEY]
        actual._state_text()
        invalid = step(actual, "unknown")[0]
        assert compute.call_count == 60
        assert actual.feature_evidence["cache_hit"] is True
        assert state(invalid)[FEATURE_KEY] == first
        advanced = step(actual, "hold")[0]
        assert compute.call_count == 120
        assert actual.feature_evidence["actual_compute_calls_this_render"] == 60
        assert state(advanced)[FEATURE_KEY]["as_of"] == DATES[5].isoformat()
        assert state(advanced)[FEATURE_KEY]["rows"] != first["rows"]
        asyncio.run(reset_with_prefix(actual))
        assert compute.call_count == 180
    other = env("features", 4)
    assert other._feature_cache is None


def test_future_close_perturbation_does_not_change_current_prepared_features():
    original = bars()
    modified = [replace(item, payload={**item.payload, "close": 99999999.0})
                if item.available_at > DATES[4] else item for item in original]
    first = env("features", 4)
    second = env("features", 4, adapter=StockBenchAdapter(
        modified, data_version="synthetic-future-perturbation",
    ))
    obs1, _ = asyncio.run(reset_with_prefix(first))
    obs2, _ = asyncio.run(reset_with_prefix(second))
    assert state(obs1)[FEATURE_KEY] == state(obs2)[FEATURE_KEY]
    assert first.feature_evidence == second.feature_evidence


def test_raw_and_feature_terminal_ledger_and_scorer_identical_with_prefix_counted_once():
    environments = [env("raw", 4), env("features", 4)]
    initial = [asyncio.run(reset_with_prefix(actual))[0] for actual in environments]
    assert state(initial[0]) == {k: v for k, v in state(initial[1]).items() if k != FEATURE_KEY}
    for index in range(16):
        name, args = ("hold", {})
        if index == 0:
            name, args = "buy", {"symbol": SYMBOLS[1], "target_weight": 0.2}
        if index == 8:
            name, args = "sell", {"symbol": SYMBOLS[1], "target_weight": 0.0}
        results = [step(actual, name, args) for actual in environments]
        assert results[0][1:] == results[1][1:]
        assert results[0][2] == (index == 15)
    assert environments[0].ledger.fills == environments[1].ledger.fills
    assert len(environments[0].ledger.fills) == 2
    assert environments[0].credit_step_trace == environments[1].credit_step_trace
    assert len(environments[0].credit_step_trace) == 20
    assert (
        environments[0].screening_terminal_metrics() == environments[1].screening_terminal_metrics()
    )
    assert environments[0].screening_terminal_metrics()["session_count"] == 21
    assert [asdict(row)["result_session_index"] for row in environments[0].credit_step_trace] == (
        list(range(1, 21))
    )


def test_render_uses_real_helper_preserves_message_and_does_not_truncate():
    actual = env("raw")
    observation = actual.reset()
    before = copy.deepcopy(observation)

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert options == {"tools": actual.tool_schemas, "add_generation_prompt": True,
                               "tokenize": True, "enable_thinking": False}
            assert messages[0]["role"] == "system"
            assert messages[1:] == observation.messages
            return [1] * 8193

    result = render_prompt(Tokenizer(), observation, actual.tool_schemas)
    assert result["prompt_token_count"] == len(result["prompt_ids"]) == 8193
    assert observation == before
    assert result["template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(ValueError, match="message observation"):
        render_prompt(Tokenizer(), Observation(text="not a message"), [])


def test_create_env_requires_original_train_identity_and_preserves_raw_prompt():
    rows: list[dict[str, Any]] = [{"extra_info": {
        "episode_id": episode, "source_index": i * 4, "index": i,
        "source_split": "train", "screening_role": "pilot_train",
    }, "env_kwargs": json.dumps({"screening": True}), "prompt": RAW_PROMPT}
        for i, episode in enumerate(("formal-train-0001", "formal-train-0005"))]
    with patch.object(D, "DecisionConditionEnv") as constructor:
        create_env(rows, 1, "features", 4)
        assert constructor.call_args.kwargs == {
            "condition": "features", "start_session": 4, "screening": True,
            "raw_prompt": RAW_PROMPT,
        }
        assert constructor.call_args.kwargs["raw_prompt"] is not RAW_PROMPT
    rows[0]["extra_info"]["source_split"] = "dev"
    with pytest.raises(ValueError, match="Train window identity"):
        create_env(rows, 0, "raw", 0)
    with pytest.raises(ValueError, match="index 0 or 1"):
        create_env(rows, True, "raw", 0)


def test_scripted_prefix_is_explicit_and_never_counts_as_a_model_response():
    actual = env("raw", 4)
    _, receipt = asyncio.run(reset_with_prefix(actual))
    assert all(row["response"]["text"] == HOLD_TEXT for row in receipt["interaction_trace"])
    assert all(row["reward"] == 0.0 for row in receipt["credit_step_trace"])


def test_hidden_diagnostic_pit_counts_are_detached_without_extra_snapshot():
    actual = env("raw", 4)
    observation, _ = asyncio.run(reset_with_prefix(actual))
    assert "compute_history" not in state(observation)
    with patch.object(actual, "_snapshot", side_effect=AssertionError("unexpected scan")):
        counts = actual.diagnostic_pit_counts
        assert counts == dict.fromkeys(SYMBOLS, 5)
        counts.clear()
        assert actual.diagnostic_pit_counts == dict.fromkeys(SYMBOLS, 5)
