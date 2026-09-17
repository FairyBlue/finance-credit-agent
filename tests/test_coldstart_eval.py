"""Exercise actual ledger transitions through the new evaluator's persistence."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from test_coldstart_data import context

from src.agents import coldstart_eval as evaluation
from src.agents.coldstart_data import tool_call
from src.env.trading import TradingEnv


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return list(range(len(text) // 10))

    def decode(self, tokens, **kwargs):
        return {
            1: tool_call("sell", {"symbol": "GS", "target_weight": 0}),
            2: tool_call("buy", {"symbol": "MSFT", "target_weight": 0.1}),
            3: tool_call("hold"),
        }[tokens[0]]


class Engine:
    def __init__(self, finish="stop"):
        self.round = 0
        self.finish = finish

    def generate(self, prompts, params, **kwargs):
        self.round += 1
        return [
            SimpleNamespace(
                prompt_token_ids=p["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(token_ids=[min(self.round, 3), 999], finish_reason=self.finish)
                ],
            )
            for p in prompts
        ]


def real_context(monkeypatch):
    ctx = context(state_kind="GS")
    adapter = ctx["env_kwargs"].pop("adapter")
    ctx["id"] = "synthetic-held"
    monkeypatch.setattr(evaluation, "TradingEnv", lambda **kw: TradingEnv(adapter=adapter, **kw))
    return {"short": [ctx], "full": [], "teacher_labels_included": False}


def test_real_prefix_and_autonomous_sell_buy_persist_without_counting_prefix(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    result = asyncio.run(
        evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path)
    )
    disk = json.loads((tmp_path / "result.json").read_text())
    assert disk["status"] == "OBSERVED"
    assert len(disk["grid"]) == 5
    for row in disk["grid"]:
        assert row["status"] == "COMPLETE"
        assert row["prefix_fill_count"] == 1
        assert [f["side"] for f in row["model_fills"]] == ["sell", "buy"]
        assert isinstance(row["model_fills"][0]["executed_at"], str)
        assert row["actions"] == {"sell": 1, "buy": 1, "hold": 1}
        assert len(row["steps_including_prefix"]) == 4
    assert result["summary"]["short_sample"]["fill_sides"] == {"sell": 4, "buy": 4}
    requests = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(requests) == 15
    assert all("API execution exercise" not in record["prompt"] for record in requests)


def test_incomplete_generation_has_no_fake_zero_financial_outcome(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    result = asyncio.run(
        evaluation.evaluate(Engine("length"), Tokenizer(), SimpleNamespace, contexts, tmp_path)
    )
    assert all(r["status"] == "INCOMPLETE" and not r["model_fills"] for r in result["grid"])
    assert result["summary"]["short_sample"]["mean_takeover_net_return_complete_only"] is None
    assert result["summary"]["short_sample"]["fill_sides"] == {}


def test_evidence_rejects_unknown_object():
    with pytest.raises(TypeError, match="Unsupported evidence type"):
        evaluation.json_default(object())


def test_declared_invalid_prefix_is_visible_and_autonomously_recovered(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    ctx = contexts["short"][0]
    ctx["prefix"].append(tool_call("sell", {"symbol": "MSFT", "target_weight": 0}))
    ctx["prefix_expected_invalid"] = [False, True]
    ctx["prefix_expected_errors"] = [
        None, "sell target_weight must be below the current weight",
    ]
    ctx["expected_start_session_index"] = 1
    result = asyncio.run(
        evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path)
    )
    for row in result["grid"]:
        assert row["status"] == "COMPLETE"
        assert row["prefix_info"][1]["invalid_action"] is True
        assert row["prefix_fill_count"] == 1
        assert row["invalid"] == 0
        assert row["requests"] == 3
        assert row["actions"] == {"sell": 1, "buy": 1, "hold": 1}
        assert [f["side"] for f in row["model_fills"]] == ["sell", "buy"]
        assert len(row["steps_including_prefix"]) == 5
        assert row["steps_including_prefix"][1]["invalid_action"] is True
        assert row["takeover_state"]["positions"]["GS"]["weight"] > 0
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    first = [r for r in records if r["request_index"] == 0]
    assert len(first) == 5
    # The genuine environment error reaches the autonomous policy; it is not
    # replaced by an execution-intention prompt or a synthetic success result.
    for record in first:
        visible = json.loads(record["prompt"])[-1]["content"]
        assert "Error: sell target_weight must be below the current weight" in visible
        assert "MSFT" in visible


def test_prefix_error_reason_is_checked(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    ctx = contexts["short"][0]
    ctx["prefix"].append(tool_call("sell", {"symbol": "MSFT", "target_weight": 0}))
    ctx["prefix_expected_invalid"] = [False, True]
    ctx["prefix_expected_errors"] = [None, "different invalid reason"]
    with pytest.raises(RuntimeError, match="declared reason"):
        asyncio.run(evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path))


def test_prefix_start_session_is_checked(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    contexts["short"][0]["expected_start_session_index"] = 2
    with pytest.raises(RuntimeError, match="declared start session"):
        asyncio.run(evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path))


@pytest.mark.parametrize("expected", [None, [False, False]])
def test_undeclared_or_wrongly_declared_invalid_prefix_fails(tmp_path, monkeypatch, expected):
    contexts = real_context(monkeypatch)
    ctx = contexts["short"][0]
    ctx["prefix"].append(tool_call("sell", {"symbol": "MSFT", "target_weight": 0}))
    if expected is not None:
        ctx["prefix_expected_invalid"] = expected
    with pytest.raises(RuntimeError, match="declared outcome"):
        asyncio.run(evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path))


@pytest.mark.parametrize("expected", [[True], [0], [], None])
def test_prefix_expectation_types_and_actual_outcome_are_checked(tmp_path, monkeypatch, expected):
    contexts = real_context(monkeypatch)
    contexts["short"][0]["prefix_expected_invalid"] = expected
    if expected == [True]:
        with pytest.raises(RuntimeError, match="declared outcome"):
            asyncio.run(
                evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path)
            )
    else:
        with pytest.raises(ValueError, match="boolean expectation"):
            evaluation.prefix_expectations(contexts["short"][0])


def test_declared_prefix_cannot_end_the_episode(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    ctx = contexts["short"][0]
    ctx["prefix"].extend([tool_call("hold")] * 3)
    ctx["prefix_expected_invalid"] = [False] * 4
    with pytest.raises(RuntimeError, match="declared outcome"):
        asyncio.run(evaluation.evaluate(Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path))


def test_parameterized_population_preserves_old_request_identities():
    contexts = {
        "short": [{"id": "short-" + str(i)} for i in range(16)],
        "full": [{"id": "full-" + str(i)} for i in range(4)],
        "teacher_labels_included": False,
    }
    evaluation.validate_contexts(contexts)
    rows = evaluation.grid(contexts)
    assert len(rows) == 100
    assert rows[80]["identity"] == {
        "step": 1, "sample_index": 16, "rollout_n": 0, "validate": True,
    }
    assert rows[99]["identity"] == {
        "step": 1, "sample_index": 19, "rollout_n": 3, "validate": False,
    }
    new = copy.deepcopy(contexts)
    new["full"] = new["full"][:1]
    evaluation.validate_contexts(new, expected_short=16, expected_full=1)
    assert len(evaluation.grid(new)) == 85
    with pytest.raises(ValueError, match="full evaluation population"):
        evaluation.validate_contexts(new)
    new["short"][1]["id"] = new["short"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.validate_contexts(new, expected_short=16, expected_full=1)


def test_evaluate_forwards_selected_protocol(tmp_path, monkeypatch):
    contexts = real_context(monkeypatch)
    captured = []
    original = evaluation.render_prompt

    def render(tokenizer, env, obs, *, action_protocol):
        captured.append(action_protocol)
        return original(tokenizer, env, obs, action_protocol="single_tool_v1")

    monkeypatch.setattr(evaluation, "render_prompt", render)
    monkeypatch.setattr(evaluation, "validate_action_protocol", lambda value: value)
    result = asyncio.run(
        evaluation.evaluate(
            Engine(), Tokenizer(), SimpleNamespace, contexts, tmp_path,
            action_protocol="single_tool_portfolio_v1",
        )
    )
    assert captured == ["single_tool_portfolio_v1"] * 15
    assert result["action_protocol"] == "single_tool_portfolio_v1"
