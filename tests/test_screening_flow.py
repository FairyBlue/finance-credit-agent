"""CPU integration of screening observations, reward timing and failure evidence."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from agent_r1.env.base import Action, Observation
from omegaconf import OmegaConf

from src.agents import rollout_seed
from src.agents.screening_prompt import (
    SINGLE_TOOL_V1_RULES,
    action_protocol_sha256,
    apply_screening_action_protocol,
)
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv


@pytest.fixture
def environment_kwargs() -> dict[str, Any]:
    days = [
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
    ]
    dates = [datetime.fromisoformat(day + "T21:00:00+00:00") for day in days]
    bars = [
        MarketObservation(
            observation_id=f"bar-{index}",
            kind="market",
            symbol="AAA",
            event_at=timestamp,
            available_at=timestamp,
            payload={
                "date": timestamp.date().isoformat(),
                "open": 100.0 + index,
                "close": 100.5 + index,
            },
        )
        for index, timestamp in enumerate(dates)
    ]
    return {
        "episode_dates": dates,
        "episode_open_times": [stamp.replace(hour=14, minute=30) for stamp in dates],
        "symbols": ["AAA"],
        "initial_cash": 1000.0,
        "commission_bps": 10.0,
        "slippage_bps": 5.0,
        "drawdown_coefficient": 0.0,
        "turnover_coefficient": 0.0,
        "benchmark_closes": [1000.0 * 1.001**index for index in range(21)],
        "adapter": StockBenchAdapter(bars, data_version="screening-fixture"),
        "screening": True,
        "finance_scorer_id": "benchmark_scale_v2",
        "finance_volatility_floor": 0.001,
        "risk_lambda": 0.5,
        "turnover_limit": 1.0,
        "position_limit": 0.3,
        "screening_calibration_sha256": "a" * 64,
    }


def _action(name: str, arguments: dict[str, Any] | None = None) -> Action:
    return Action(
        text="<tool_call>"
        + json.dumps({"name": name, "arguments": arguments or {}})
        + "</tool_call>"
    )


def test_screening_rewards_are_neutral_for_the_entire_action_history(environment_kwargs):
    actions = [
        _action("buy", {"symbol": "AAA", "target_weight": 0.3}),
        _action("compute", {"metric": "price_change", "symbol": "AAA", "lookback": 2}),
        _action("sell", {"symbol": "AAA", "target_weight": 0.1}),
        _action("abstain", {"reason": "wait"}),
    ] + [_action("hold")] * 17
    observations, results = [], []
    for mode in ("outcome_only", "finance_trajectory"):
        env = TradingEnv(**environment_kwargs, reward_mode=mode)
        history = [env.reset().messages]
        for action in actions:
            obs, _reward, _done, _info = asyncio.run(env.step(action))
            history.append(obs.messages)
        assert env.episode_complete
        observations.append(history)
        results.append(env.screening_terminal_metrics())
        assert all("last_reward" not in json.dumps(obs) for obs in history)
    assert observations[0] == observations[1]
    assert results[0] == results[1]
    assert results[0]["transaction_cost"] > 0
    assert results[0]["agent_volatility"] > 0


@pytest.mark.parametrize("mode", ["outcome_only", "finance_trajectory"])
def test_screening_never_scores_partial_nav(environment_kwargs, mode):
    env = TradingEnv(**environment_kwargs, reward_mode=mode)
    env.reset()
    asyncio.run(env.step(_action("hold")))
    with pytest.raises(RuntimeError, match="complete screening trajectory"):
        env.screening_terminal_metrics()


def test_screening_rejects_valid_and_legacy_dense_inputs(environment_kwargs):
    with pytest.raises(ValueError, match="dense"):
        TradingEnv(**environment_kwargs, reward_mode="generic_step")
    kwargs = dict(environment_kwargs)
    kwargs["episode_dates"] = [datetime(2024, 11, 1, 21, tzinfo=timezone.utc)]
    with pytest.raises(ValueError, match="historical Train"):
        TradingEnv(**kwargs, reward_mode="outcome_only")


@pytest.fixture
def screening_module(monkeypatch):
    package = ModuleType("agent_r1.agent_flow")
    package.__path__ = []
    module = ModuleType("agent_r1.agent_flow.agent_env_loop")

    class FakeAgentEnvLoop:
        def __init__(self, *, config, manager, **kwargs):
            self.config = config
            self.server_manager = manager
            self.env_kwargs = kwargs
            rollout = config.actor_rollout_ref.rollout
            self.prompt_length = rollout.prompt_length
            self.response_length = rollout.response_length
            self.max_steps = rollout.agent.max_steps

        def _create_env(self, **kwargs):
            combined = {**self.env_kwargs, **kwargs.get("env_kwargs", {})}
            combined.pop("env_type", None)
            return TradingEnv(**combined)

        async def _obs_to_prompt(self, obs, tools=None):
            return list(json.dumps(obs.messages).encode("ascii"))

        async def run(self, sampling_params, **kwargs):
            env = self._create_env(**kwargs)
            obs = env.reset()
            steps = []
            for index in range(self.max_steps):
                ids = await self._obs_to_prompt(obs)
                response = await self.server_manager.generate(
                    str(index),
                    prompt_ids=ids,
                    sampling_params=sampling_params,
                )
                obs, reward, done, _info = await env.step(response.action)
                steps.append(SimpleNamespace(reward_score=reward, extra_fields={}))
                if done:
                    break
            return SimpleNamespace(steps=steps)

    module.AgentEnvLoop = FakeAgentEnvLoop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent_r1.agent_flow", package)
    monkeypatch.setitem(sys.modules, "agent_r1.agent_flow.agent_env_loop", module)
    name = "src.agents.agent_flow"
    original = sys.modules.pop(name, None)
    imported = importlib.import_module(name)
    try:
        yield imported
    finally:
        sys.modules.pop(name, None)
        if original is not None:
            sys.modules[name] = original


class FakeManager:
    def __init__(self, *, failure=None, invalid_first=False, trade_first=False):
        self.failure = failure
        self.invalid_first = invalid_first
        self.trade_first = trade_first
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append({"prompt_ids": prompt_ids, "sampling_params": dict(sampling_params)})
        count = 32 if self.failure == "cap" else 2
        return SimpleNamespace(
            token_ids=list(range(count)),
            stop_reason="aborted" if self.failure == "abort" else "completed",
            finish_reason="length" if self.failure == "length" else None,
            action=Action(text="invalid")
            if self.invalid_first and len(self.calls) == 1
            else (
                _action("buy", {"symbol": "AAA", "target_weight": 0.3})
                if self.trade_first and len(self.calls) == 1
                else _action("hold")
            ),
        )


def _run_flow(
    module,
    monkeypatch,
    tmp_path,
    environment_kwargs,
    *,
    arm="M2",
    validate=False,
    failure=None,
    max_steps=32,
    invalid_first=False,
    trade_first=False,
    action_protocol="legacy",
    prompt_length=4096,
):
    monkeypatch.setenv(module.SCREENING_EVIDENCE_DIR_ENV, str(tmp_path / "evidence"))
    monkeypatch.setattr(rollout_seed, "_configured_evidence_root", tmp_path / "seeds")
    config = OmegaConf.create(
        {
            "data": {
                "seed": 42,
                **(
                    {"screening_action_protocol": action_protocol}
                    if action_protocol != "legacy"
                    else {}
                ),
            },
            "actor_rollout_ref": {
                "rollout": {
                    "n": 4,
                    "val_kwargs": {"n": 1},
                    "prompt_length": prompt_length,
                    "response_length": 32,
                    "max_model_len": prompt_length + 32,
                    "agent": {"max_steps": max_steps},
                }
            },
        }
    )
    manager = FakeManager(failure=failure, invalid_first=invalid_first, trade_first=trade_first)
    flow = module.ScreeningTradingAgentEnvLoop(
        config=config,
        manager=manager,
        screening_arm=arm,
        env_type="trading",
    )
    output = asyncio.run(
        flow.run(
            {"temperature": 0.0 if validate else 1.0},
            uid="fixed-upstream-uid",
            extra_info={
                "episode_id": "formal-train-0001",
                "source_split": "train",
                "scientific_use": "screening",
            },
            env_kwargs=environment_kwargs,
            _finance_rollout_identity_v1={
                "step": 1,
                "sample_index": 0,
                "rollout_n": 0,
                "validate": validate,
            },
        )
    )
    return output, manager


def _records(root: Path):
    return [
        json.loads(line)
        for path in (root / "evidence").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize("arm", ["A2", "M1", "M2", "M2-plain"])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("action_protocol", ["legacy", "single_tool_v1"])
def test_screening_arms_share_scorer_but_preserve_training_reward_timing(
    screening_module,
    monkeypatch,
    tmp_path,
    environment_kwargs,
    arm,
    validate,
    action_protocol,
):
    output, manager = _run_flow(
        screening_module,
        monkeypatch,
        tmp_path,
        environment_kwargs,
        arm=arm,
        validate=validate,
        action_protocol=action_protocol,
    )
    (record,) = _records(tmp_path)
    assert record["complete"] and record["status"] == "COMPLETE"
    assert record["metrics"]["raw_net_return"] == 0.0
    assert record["terminal_score"] < 0.0
    credits = [step.reward_score for step in output.steps]
    if validate:
        assert credits[:-1] == [0.0] * 19
        assert credits[-1] == record["terminal_score"]
    elif arm == "A2":
        assert credits == [0.0] * 20
    else:
        assert sum(credits) == record["terminal_score"]
        assert credits[:-1] == [record["terminal_score"] / 20] * 19
    assert len(manager.calls) == 20
    assert len({call["sampling_params"]["seed"] for call in manager.calls}) == 20
    assert record["generated_response_tokens"] == 40
    assert record["action_protocol"] == action_protocol
    assert record["action_protocol_sha256"] == action_protocol_sha256(action_protocol)
    assert record["identity"]["validate"] is validate
    for call in manager.calls:
        messages = json.loads(bytes(call["prompt_ids"]).decode("ascii"))
        assert sum(SINGLE_TOOL_V1_RULES in message["content"] for message in messages) == (
            0 if action_protocol == "legacy" else 1
        )
        assert call["sampling_params"]["temperature"] == (0.0 if validate else 1.0)
    assert all("response_token_ids" not in request for request in record["requests"])
    assert all(
        step.extra_fields["finance_screening_group_uid"] == "fixed-upstream-uid"
        and step.extra_fields["finance_screening_terminal_score"] == record["terminal_score"]
        for step in output.steps
    )


@pytest.mark.parametrize("failure", ["length", "cap", "abort"])
@pytest.mark.parametrize("arm", ["A2", "M2"])
def test_failed_response_never_reaches_environment_even_if_action_is_parseable(
    screening_module,
    monkeypatch,
    tmp_path,
    environment_kwargs,
    failure,
    arm,
):
    with pytest.raises(screening_module.ScreeningIncompleteError, match="truncated or aborted"):
        _run_flow(
            screening_module, monkeypatch, tmp_path, environment_kwargs, arm=arm, failure=failure
        )
    (record,) = _records(tmp_path)
    assert record["status"] == "INCOMPLETE" and not record["complete"]
    assert record["steps"] == []
    assert "terminal_score" not in record
    assert record["requests"][0]["response_cap"] == 32
    request = record["requests"][0]
    assert request["response_token_ids"] == list(range(32 if failure == "cap" else 2))
    assert request["response_token_ids_truncated"] is False
    assert request["response_last_token_id"] == request["response_token_ids"][-1]
    assert record["generated_response_tokens"] > 0


@pytest.mark.parametrize("response_cap", [2048, 4096])
@pytest.mark.parametrize("overrun", [0, 7])
def test_real_response_cap_preserves_bounded_failure_tokens_even_with_eos(
    screening_module, response_cap, overrun
):
    tokens = [1] * (response_cap + overrun - 1) + [151645]

    async def generate(*args, **kwargs):
        assert kwargs["sampling_params"]["max_tokens"] == response_cap
        return SimpleNamespace(token_ids=tokens, stop_reason="completed", finish_reason=None)

    proxy = screening_module.ScreeningResponseManager(
        SimpleNamespace(generate=generate),
        prompt_limit=4096,
        response_limit=response_cap,
        context_limit=4096 + response_cap,
    )
    with pytest.raises(screening_module.ScreeningIncompleteError, match="truncated or aborted"):
        asyncio.run(proxy.generate("request", prompt_ids=[1] * 2259, sampling_params={}))
    (record,) = proxy.requests
    assert record["status"] == "RESPONSE_INCOMPLETE"
    assert record["response_token_count"] == response_cap + overrun
    assert record["response_token_ids"] == tokens[:response_cap]
    assert record["response_token_ids_truncated"] is (overrun > 0)
    assert record["response_last_token_id"] == 151645
    assert record["stop_reason"] == "completed"
    assert record["finish_reason"] is None


@pytest.mark.parametrize("response_cap", [2048, 4096])
def test_response_below_real_cap_remains_hash_only(screening_module, response_cap):
    output = SimpleNamespace(
        token_ids=[1] * (response_cap - 2) + [151645],
        stop_reason="completed",
        finish_reason="stop",
    )

    async def generate(*args, **kwargs):
        return output

    proxy = screening_module.ScreeningResponseManager(
        SimpleNamespace(generate=generate),
        prompt_limit=4096,
        response_limit=response_cap,
        context_limit=4096 + response_cap,
    )
    assert asyncio.run(proxy.generate("request", prompt_ids=[1], sampling_params={})) is output
    (record,) = proxy.requests
    assert record["status"] == "COMPLETE"
    assert record["response_token_count"] == response_cap - 1
    assert "response_sha256" in record
    assert "response_token_ids" not in record
    assert "response_token_ids_truncated" not in record
    assert "response_last_token_id" not in record


def test_agent_limit_retains_raw_evidence_without_partial_score(
    screening_module,
    monkeypatch,
    tmp_path,
    environment_kwargs,
):
    with pytest.raises(screening_module.ScreeningIncompleteError, match="exact terminal"):
        _run_flow(screening_module, monkeypatch, tmp_path, environment_kwargs, max_steps=19)
    (record,) = _records(tmp_path)
    assert record["status"] == "INCOMPLETE"
    assert len(record["steps"]) == 19
    assert "metrics" not in record and "credits" not in record


def test_invalid_action_is_retained_when_episode_still_completes(
    screening_module,
    monkeypatch,
    tmp_path,
    environment_kwargs,
):
    output, _ = _run_flow(
        screening_module, monkeypatch, tmp_path, environment_kwargs, invalid_first=True
    )
    (record,) = _records(tmp_path)
    assert record["complete"] and len(output.steps) == 21
    assert record["metrics"]["invalid_count"] == 1
    assert output.steps[0].extra_fields["trading_invalid_action"] is True


def test_context_overflow_is_rejected_before_generation(screening_module):
    manager = FakeManager()
    proxy = screening_module.ScreeningResponseManager(
        manager,
        prompt_limit=100,
        response_limit=32,
        context_limit=110,
    )
    with pytest.raises(screening_module.ScreeningIncompleteError, match="context"):
        asyncio.run(proxy.generate("request", prompt_ids=[1] * 90, sampling_params={}))
    assert manager.calls == []
    assert proxy.requests[0]["status"] == "CONTEXT_OVERFLOW"


def test_a2_trade_has_nonzero_terminal_rnet_and_separate_common_score(
    screening_module,
    monkeypatch,
    tmp_path,
    environment_kwargs,
):
    output, _ = _run_flow(
        screening_module, monkeypatch, tmp_path, environment_kwargs, arm="A2", trade_first=True
    )
    (record,) = _records(tmp_path)
    credits = [step.reward_score for step in output.steps]
    assert credits[:-1] == [0.0] * 19
    assert credits[-1] == record["metrics"]["raw_net_return"] > 0
    assert credits[-1] != record["terminal_score"]
    assert all(
        step.extra_fields["finance_screening_net_return"] == credits[-1] for step in output.steps
    )


def test_historical_observation_keeps_its_registered_schema(environment_kwargs):
    kwargs = {**environment_kwargs, "screening": False}
    env = TradingEnv(**kwargs, reward_mode="outcome_only")
    env.reset()
    obs, _, _, _ = asyncio.run(env.step(_action("hold")))
    assert "last_reward" in json.dumps(obs.messages)


def test_protocol_copies_observation_before_parent_template_and_preserves_environment(
    screening_module, monkeypatch, tmp_path, environment_kwargs
):
    received = []
    original = screening_module.TradingAgentEnvLoop._obs_to_prompt

    async def observe_parent(self, obs, tools=None):
        env_messages = self._active_trading_env._messages
        assert all(SINGLE_TOOL_V1_RULES not in message["content"] for message in env_messages)
        assert obs.messages == apply_screening_action_protocol(env_messages, "single_tool_v1")
        assert obs.messages is not env_messages
        before = deepcopy(env_messages)
        result = await original(self, obs, tools=tools)
        assert env_messages == before
        received.append(obs)
        return result

    monkeypatch.setattr(screening_module.TradingAgentEnvLoop, "_obs_to_prompt", observe_parent)
    output, manager = _run_flow(
        screening_module,
        monkeypatch,
        tmp_path,
        environment_kwargs,
        action_protocol="single_tool_v1",
    )
    assert len(received) == len(output.steps) == len(manager.calls) == 20


@pytest.mark.parametrize(
    "obs",
    [
        Observation(text="Already rendered"),
        Observation(token_ids=[1, 2, 3]),
        Observation(messages=[{"role": "user", "content": "state"}], token_ids=[1]),
        Observation(),
    ],
)
def test_enabled_protocol_rejects_non_message_observations_before_tokenization(
    screening_module, obs
):
    flow = object.__new__(screening_module.ScreeningTradingAgentEnvLoop)
    flow._action_protocol = "single_tool_v1"
    with pytest.raises(RuntimeError, match="requires message observations"):
        asyncio.run(flow._obs_to_prompt(obs))


@pytest.mark.parametrize("protocol", [True, False, 1, "unregistered"])
def test_flow_rejects_invalid_shared_protocol_before_generation(
    screening_module, monkeypatch, tmp_path, environment_kwargs, protocol
):
    with pytest.raises(ValueError, match="action protocol"):
        _run_flow(
            screening_module,
            monkeypatch,
            tmp_path,
            environment_kwargs,
            action_protocol=protocol,
        )
    assert _records(tmp_path) == []


def test_added_protocol_text_cannot_bypass_the_parent_prompt_length_guard(
    screening_module, monkeypatch, tmp_path, environment_kwargs
):
    obs = TradingEnv(**environment_kwargs, reward_mode="outcome_only").reset()
    original_messages = deepcopy(obs.messages)
    legacy_length = len(json.dumps(obs.messages).encode("ascii"))
    flow = object.__new__(screening_module.ScreeningTradingAgentEnvLoop)
    flow._action_protocol = "legacy"
    flow.prompt_length = legacy_length
    assert len(asyncio.run(flow._obs_to_prompt(obs))) == legacy_length
    flow._action_protocol = "single_tool_v1"
    with pytest.raises(screening_module.ScreeningIncompleteError, match="prompt limit"):
        asyncio.run(flow._obs_to_prompt(obs))
    assert obs.messages == original_messages

    with pytest.raises(screening_module.ScreeningIncompleteError, match="prompt limit"):
        _run_flow(
            screening_module,
            monkeypatch,
            tmp_path,
            environment_kwargs,
            action_protocol="single_tool_v1",
            prompt_length=legacy_length,
        )
    (record,) = _records(tmp_path)
    assert record["status"] == "INCOMPLETE" and record["complete"] is False
    assert record["requests"] == [] and record["steps"] == []
    assert "terminal_score" not in record
    assert record["action_protocol_sha256"] == action_protocol_sha256("single_tool_v1")
