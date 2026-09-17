"""Exercise the model-visible contract and the unchanged serial tool executor."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from agent_r1.env.base import Action
from agent_r1.tool import BaseTool
from pydantic import ValidationError

from src.agents.screening_prompt import apply_screening_action_protocol
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import _ARG_MODELS, TradingEnv

NAMES = ["retrieve", "compute", "buy", "sell", "hold", "abstain"]


def _parameters(name: str) -> dict[str, Any]:
    return BaseTool.from_name(name).tool_schema["function"]["parameters"]


@pytest.fixture
def env() -> TradingEnv:
    dates = [datetime(2025, 1, day, 21, tzinfo=timezone.utc) for day in (2, 3, 6, 7)]
    observations = [
        MarketObservation(
            observation_id=f"bar-{index}",
            kind="market",
            symbol="AAA",
            event_at=when,
            available_at=when,
            payload={"date": when.date().isoformat(), "open": 100.0, "close": 100.0},
        )
        for index, when in enumerate(dates)
    ]
    return TradingEnv(
        adapter=StockBenchAdapter(observations, data_version="tool-contract-test"),
        episode_dates=dates,
        episode_open_times=[when.replace(hour=14, minute=30) for when in dates],
        symbols=["AAA"],
        initial_cash=1000.0,
        commission_bps=10.0,
        slippage_bps=5.0,
        drawdown_coefficient=0.0,
        turnover_coefficient=0.0,
        benchmark_closes=[100.0] * len(dates),
    )


@pytest.mark.parametrize("name", NAMES)
def test_actual_exports_forbid_extra_arguments_and_have_independent_copies(name: str) -> None:
    tool = BaseTool.from_name(name)
    first = tool.tool_schema
    parameters = first["function"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert parameters == tool.function_schema["parameters"] == tool.parameters
    first["function"]["parameters"]["additionalProperties"] = True
    first["function"]["parameters"]["properties"]["injected"] = {"type": "string"}
    assert tool.tool_schema["function"]["parameters"]["additionalProperties"] is False
    assert "injected" not in BaseTool.from_name(name).parameters["properties"]
    assert "injected" not in tool.function_schema["parameters"]["properties"]


def test_visible_bounds_required_nullable_and_defaults_match_executor() -> None:
    retrieve = _parameters("retrieve")
    assert retrieve["required"] == ["query"]
    assert retrieve["properties"]["query"]["minLength"] == 1
    assert retrieve["properties"]["query"]["maxLength"] == 200
    limit = retrieve["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 3, 3)
    symbol = retrieve["properties"]["symbol"]
    assert symbol["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert symbol["default"] is None

    compute = _parameters("compute")
    assert compute["required"] == ["metric", "symbol", "lookback"]
    assert compute["properties"]["symbol"]["minLength"] == 1
    assert compute["properties"]["metric"]["enum"] == ["price_change", "mean_return", "volatility"]
    lookback = compute["properties"]["lookback"]
    assert (lookback["minimum"], lookback["maximum"]) == (2, 60)
    assert "default" not in lookback
    for name in ("buy", "sell"):
        target = _parameters(name)
        assert target["required"] == ["symbol", "target_weight"]
        assert target["properties"]["symbol"]["minLength"] == 1
        weight = target["properties"]["target_weight"]
        assert (weight["minimum"], weight["maximum"]) == (0, 1)
    assert _parameters("hold")["properties"] == {}
    assert _parameters("hold")["required"] == []
    abstain = _parameters("abstain")
    assert abstain["required"] == []
    reason = abstain["properties"]["reason"]
    assert reason["anyOf"] == [{"maxLength": 500, "type": "string"}, {"type": "null"}]
    assert reason["default"] is None


@pytest.mark.parametrize(
    ("name", "arguments", "valid"),
    [
        ("retrieve", {"query": "a"}, True),
        ("retrieve", {"query": "a" * 200, "limit": 1, "symbol": None}, True),
        ("retrieve", {"query": "a", "limit": 3, "symbol": ""}, True),
        ("retrieve", {"query": ""}, False),
        ("retrieve", {"query": "a" * 201}, False),
        ("retrieve", {"query": "a", "limit": 0}, False),
        ("retrieve", {"query": "a", "limit": 4}, False),
        ("retrieve", {}, False),
        ("compute", {"metric": "mean_return", "symbol": "AAA", "lookback": 2}, True),
        ("compute", {"metric": "volatility", "symbol": "AAA", "lookback": 60}, True),
        ("compute", {"metric": "mean_return", "symbol": "AAA", "lookback": 1}, False),
        ("compute", {"metric": "price_change", "symbol": "AAA", "lookback": 61}, False),
        ("compute", {"metric": "rsi", "symbol": "AAA", "lookback": 2}, False),
        ("compute", {"metric": "mean_return", "symbol": "", "lookback": 2}, False),
        ("compute", {"metric": "mean_return", "symbol": "AAA"}, False),
        ("buy", {"symbol": "AAA", "target_weight": 0}, True),
        ("buy", {"symbol": "AAA", "target_weight": 1}, True),
        ("buy", {"symbol": "AAA", "target_weight": -0.001}, False),
        ("buy", {"symbol": "AAA", "target_weight": 1.001}, False),
        ("buy", {"symbol": "", "target_weight": 0.5}, False),
        ("buy", {"symbol": "AAA", "target_weight": 0.5, "quantity": 1}, False),
        ("buy", {"symbol": "AAA"}, False),
        ("sell", {"symbol": "AAA", "target_weight": 0}, True),
        ("sell", {"symbol": "AAA", "target_weight": 1}, True),
        ("sell", {"symbol": "AAA", "target_weight": -0.001}, False),
        ("sell", {"symbol": "AAA", "target_weight": 1.001}, False),
        ("hold", {}, True),
        ("hold", {"reason": "wait"}, False),
        ("abstain", {}, True),
        ("abstain", {"reason": None}, True),
        ("abstain", {"reason": "a" * 500}, True),
        ("abstain", {"reason": "a" * 501}, False),
        ("abstain", {"reason": "a", "symbol": "AAA"}, False),
    ],
)
def test_canonical_json_arguments_agree_with_actual_runtime_validation(
    name: str, arguments: dict[str, Any], valid: bool
) -> None:
    # Optional in a minimal CPU install; available in the pinned runtime.
    validator = pytest.importorskip("jsonschema").Draft202012Validator(_parameters(name))
    assert validator.is_valid(arguments) is valid
    if valid:
        _ARG_MODELS[name].model_validate(arguments)
    else:
        with pytest.raises(ValidationError):
            _ARG_MODELS[name].model_validate(arguments)


def test_real_hermes_execution_still_accepts_buy_and_sell(env: TradingEnv) -> None:
    async def exercise() -> None:
        env.reset()
        for name, weight in (("buy", "0.25"), ("sell", 0.0)):
            # Runtime coercion of numeric strings is intentionally unchanged;
            # the public schema documents canonical typed JSON arguments.
            arguments = {"symbol": "AAA", "target_weight": weight}
            call = json.dumps({"name": name, "arguments": arguments})
            raw = f"<tool_call>{call}</tool_call>"
            parsed, _ = env.parse_response(raw)
            assert len(parsed) == 1 and parsed[0].arguments == arguments
            _, _, done, info = await env.step(Action(text=raw))
            assert not info.get("invalid_action", False)
            assert len(info["fills"]) == 1
            assert not done
        assert env.ledger.positions.get("AAA", 0.0) == 0.0

    asyncio.run(exercise())


@pytest.mark.parametrize("model", ["Qwen3-1.7B", "Qwen3-4B"])
def test_real_env_schemas_survive_cached_tokenizer_and_hermes_prompt(
    env: TradingEnv, model: str
) -> None:
    snapshots = Path.home() / ".cache/huggingface/hub" / f"models--Qwen--{model}/snapshots"
    available = sorted(snapshots.glob("*/tokenizer_config.json"))
    if not available:
        pytest.skip("requires the already cached tokenizer; never downloads")
    tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        str(available[0].parent), local_files_only=True
    )
    messages = apply_screening_action_protocol(env.reset().messages, "single_tool_v1")
    schemas = env.tool_schemas
    assert [schema["function"]["name"] for schema in schemas] == NAMES
    rendered = tokenizer.apply_chat_template(
        messages, tools=schemas, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # The actual Qwen template writes one complete JSON schema per line.
    tool_text = rendered.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    assert [json.loads(line) for line in tool_text.splitlines()] == schemas
    assert all(
        schema["function"]["parameters"]["additionalProperties"] is False for schema in schemas
    )
    assert "<tool_call>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = tokenizer.apply_chat_template(
        messages, tools=schemas, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    assert tokenizer.decode(ids, skip_special_tokens=False) == rendered
