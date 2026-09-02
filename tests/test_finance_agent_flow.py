"""CPU-only integration tests for the local A3 finance-credit flow."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.agents.credit_evidence import (
    STEP_CREDIT_EVIDENCE_DIR_ENV,
    summarize_step_credit_evidence,
)
from src.rewards import STEP_CREDIT_OUTPUT_SEMANTICS, UniformBroadcastCredit


@pytest.fixture
def flow_module(monkeypatch: pytest.MonkeyPatch):
    """Import the local flow with a fake parent, then restore the module cache."""

    fake_package = ModuleType("agent_r1.agent_flow")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_module = ModuleType("agent_r1.agent_flow.agent_env_loop")

    class FakeAgentEnvLoop:
        pass

    fake_module.AgentEnvLoop = FakeAgentEnvLoop  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent_r1.agent_flow", fake_package)
    monkeypatch.setitem(sys.modules, "agent_r1.agent_flow.agent_env_loop", fake_module)
    module_name = "src.agents.agent_flow"
    original_module = sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def _trace(
    rewards: list[float],
    *,
    complete: bool,
) -> tuple[SimpleNamespace, ...]:
    actions = ["retrieve", "compute", "hold"]
    return tuple(
        SimpleNamespace(
            action=actions[index],
            decision_at=f"2025-01-0{index + 2}T21:00:00+00:00",
            decision_session_index=index,
            done=complete and index == len(rewards) - 1,
            invalid_action=False,
            result_session_index=min(index + 1, len(rewards) - 1),
            reward=reward,
        )
        for index, reward in enumerate(rewards)
    )


def _output(rewards: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        steps=[
            SimpleNamespace(reward_score=reward, extra_fields={}) for reward in rewards
        ]
    )


def _run_finance_flow(
    monkeypatch: pytest.MonkeyPatch,
    flow_module: Any,
    *,
    rewards: list[float],
    complete: bool,
) -> SimpleNamespace:
    output = _output(rewards)
    env = SimpleNamespace(
        credit_step_trace=_trace(rewards, complete=complete),
        episode_complete=complete,
        reward_mode="finance_trajectory",
    )

    async def upstream_run(_self: object, _sampling_params: dict[str, Any], **_kwargs: Any):
        _self._active_trading_env = env  # type: ignore[attr-defined]
        for step, record in zip(output.steps, env.credit_step_trace, strict=True):
            step.extra_fields["trading_action"] = record.action
        return output

    monkeypatch.setattr(flow_module.TradingAgentEnvLoop, "run", upstream_run)
    flow = object.__new__(flow_module.FinanceCreditAgentEnvLoop)
    flow._credit_strategy = UniformBroadcastCredit()
    flow.config = SimpleNamespace(data=SimpleNamespace(seed=42))
    return asyncio.run(
        flow.run(
            {},
            _finance_rollout_identity_v1={
                "step": 0,
                "sample_index": 0,
                "rollout_n": 0,
                "validate": False,
            },
        )
    )


@pytest.mark.parametrize("trajectory_advantage", [0.9, 0.0, -0.9])
def test_m1_flow_allocates_completed_trajectory_as_uniform_immediate_rewards(
    monkeypatch: pytest.MonkeyPatch,
    flow_module: Any,
    tmp_path: Path,
    trajectory_advantage: float,
) -> None:
    monkeypatch.setenv(STEP_CREDIT_EVIDENCE_DIR_ENV, str(tmp_path))
    result = _run_finance_flow(
        monkeypatch,
        flow_module,
        rewards=[0.0, 0.0, trajectory_advantage],
        complete=True,
    )

    allocated = [step.reward_score for step in result.steps]
    assert allocated == pytest.approx([trajectory_advantage / 3.0] * 3)
    assert sum(allocated) == trajectory_advantage
    assert [step.extra_fields["trading_action"] for step in result.steps] == [
        "retrieve",
        "compute",
        "hold",
    ]
    assert all(
        step.extra_fields["finance_credit_mechanism"] == "m1_uniform"
        and step.extra_fields["finance_credit_output_semantics"]
        == STEP_CREDIT_OUTPUT_SEMANTICS
        and step.extra_fields["finance_credit_step_count"] == 3
        and step.extra_fields["finance_trajectory_complete"] is True
        for step in result.steps
    )
    evidence = summarize_step_credit_evidence(
        tmp_path,
        expected_mechanism="m1_uniform",
        expected_output_semantics=STEP_CREDIT_OUTPUT_SEMANTICS,
        expected_base_seed=42,
    )
    representative = evidence["representative_trajectory"]
    assert isinstance(representative, dict)
    assert representative["credits"] == allocated
    assert representative["trajectory_advantage"] == trajectory_advantage


def test_m1_flow_rejects_truncated_trajectory_without_rewriting_rewards(
    monkeypatch: pytest.MonkeyPatch,
    flow_module: Any,
) -> None:
    original = [0.0, 0.0, 0.0]
    output = _output(original)
    env = SimpleNamespace(
        credit_step_trace=_trace(original, complete=False),
        episode_complete=False,
        reward_mode="finance_trajectory",
    )

    async def upstream_run(_self: object, _sampling_params: dict[str, Any], **_kwargs: Any):
        _self._active_trading_env = env  # type: ignore[attr-defined]
        return output

    monkeypatch.setattr(flow_module.TradingAgentEnvLoop, "run", upstream_run)
    flow = object.__new__(flow_module.FinanceCreditAgentEnvLoop)
    flow._credit_strategy = UniformBroadcastCredit()

    with pytest.raises(RuntimeError, match="completed trajectory"):
        asyncio.run(flow.run({}))

    assert [step.reward_score for step in output.steps] == original
    assert all(not step.extra_fields for step in output.steps)


def test_m1_flow_rejects_empty_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    flow_module: Any,
) -> None:
    output = _output([])

    async def upstream_run(_self: object, _sampling_params: dict[str, Any], **_kwargs: Any):
        return output

    monkeypatch.setattr(flow_module.TradingAgentEnvLoop, "run", upstream_run)
    flow = object.__new__(flow_module.FinanceCreditAgentEnvLoop)
    flow._credit_strategy = UniformBroadcastCredit()

    with pytest.raises(RuntimeError, match="at least one completed"):
        asyncio.run(flow.run({}))
