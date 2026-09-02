"""External Agent-R1 flows; imported only inside the GPU training runtime."""

from __future__ import annotations

import math
from typing import Any

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

# Importing TradingEnv registers `env_type: trading` without touching upstream files.
from src.env.trading import TradingEnv as _TradingEnv
from src.rewards import STEP_CREDIT_OUTPUT_SEMANTICS, StepContext, build_credit_strategy

from .credit_evidence import record_step_credit_evidence
from .rollout_seed import (
    ROLLOUT_IDENTITY_KWARG,
    ROLLOUT_SEED_SCHEME,
    build_seeded_server_manager,
    normalize_rollout_identity,
)


class TradingAgentEnvLoop(AgentEnvLoop):
    """Agent-R1's generic step-MDP loop with the external trading env registered."""

    server_manager: Any
    _active_trading_env: _TradingEnv

    def _create_env(self, **kwargs: Any) -> _TradingEnv:
        env = super()._create_env(**kwargs)
        if not isinstance(env, _TradingEnv):
            raise RuntimeError("trading flow resolved a non-trading environment")
        self._active_trading_env = env
        return env

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        if ROLLOUT_IDENTITY_KWARG not in kwargs:
            raise RuntimeError(
                "deterministic rollout identity was not supplied by the runtime hook"
            )
        identity = normalize_rollout_identity(kwargs.pop(ROLLOUT_IDENTITY_KWARG))
        original_manager = self.server_manager
        seeded_manager = build_seeded_server_manager(
            original_manager,
            base_seed=self.config.data.seed,
            trajectory=identity,
        )
        self.server_manager = seeded_manager
        try:
            output = await super().run(sampling_params, **kwargs)
        finally:
            self.server_manager = original_manager
        if len(output.steps) != len(seeded_manager.request_seeds):
            raise RuntimeError(
                "rollout seed evidence count does not match generated environment steps"
            )
        for step, request_seed in zip(
            output.steps, seeded_manager.request_seeds, strict=True
        ):
            step.extra_fields["finance_rollout_seed"] = request_seed
            step.extra_fields["finance_rollout_seed_scheme"] = ROLLOUT_SEED_SCHEME
        env = getattr(self, "_active_trading_env", None)
        if not isinstance(env, _TradingEnv):
            raise RuntimeError("trading rollout did not retain its environment evidence")
        trace = env.credit_step_trace
        if len(output.steps) != len(trace):
            raise RuntimeError("trading step trace count does not match generated steps")
        for step, record in zip(output.steps, trace, strict=True):
            step.extra_fields["trading_action"] = record.action
            step.extra_fields["trading_decision_session_index"] = (
                record.decision_session_index
            )
            step.extra_fields["trading_decision_at"] = record.decision_at
            step.extra_fields["trading_result_session_index"] = record.result_session_index
            step.extra_fields["trading_env_reward"] = record.reward
            step.extra_fields["trading_env_done"] = record.done
            step.extra_fields["trading_invalid_action"] = record.invalid_action
            step.extra_fields["trading_episode_complete"] = env.episode_complete
        return output

    async def _obs_to_prompt(
        self, obs: Any, tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        prompt_ids = await super()._obs_to_prompt(obs, tools=tools)
        if len(prompt_ids) > self.prompt_length:
            raise RuntimeError(
                f"tokenized trading prompt uses {len(prompt_ids)} tokens, exceeding "
                f"configured prompt limit {self.prompt_length}; rollout aborted"
            )
        return prompt_ids


class FinanceCreditAgentEnvLoop(TradingAgentEnvLoop):
    """A3 loop that retroactively allocates the final finance trajectory signal."""

    def __init__(
        self,
        *args: Any,
        credit_strategy: str = "m1_uniform",
        credit_semantics: str = STEP_CREDIT_OUTPUT_SEMANTICS,
        **kwargs: Any,
    ) -> None:
        self._credit_strategy = build_credit_strategy(credit_strategy)
        if credit_semantics != self._credit_strategy.output_semantics:
            raise ValueError(
                "configured credit semantics do not match the selected strategy: "
                f"{credit_semantics!r} != {self._credit_strategy.output_semantics!r}"
            )
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        output = await super().run(sampling_params, **kwargs)
        if not output.steps:
            raise RuntimeError("finance credit requires at least one completed environment step")
        env = self._active_trading_env
        if env.reward_mode != "finance_trajectory":
            raise RuntimeError("finance credit requires reward_mode=finance_trajectory")
        if not env.episode_complete:
            raise RuntimeError(
                "finance credit requires a completed trajectory; rollout stopped before A_fin"
            )
        trace = env.credit_step_trace
        if not trace[-1].done or any(record.done for record in trace[:-1]):
            raise RuntimeError("finance trajectory has an invalid terminal-step trace")
        observed_rewards = tuple(float(step.reward_score or 0.0) for step in output.steps)
        trace_rewards = tuple(record.reward for record in trace)
        if observed_rewards != trace_rewards:
            raise RuntimeError("Agent-R1 rewards do not match retained trading evidence")
        trajectory_advantage = math.fsum(observed_rewards)
        contexts = [
            StepContext(
                step_id=f"step-{index}",
                action=record.action,
                metadata={
                    "decision_at": record.decision_at,
                    "decision_session_index": record.decision_session_index,
                    "invalid_action": record.invalid_action,
                    "result_session_index": record.result_session_index,
                },
            )
            for index, record in enumerate(trace)
        ]
        credits = self._credit_strategy.allocate(
            trajectory_advantage=trajectory_advantage,
            steps=contexts,
        )
        if sum(credit.credit for credit in credits) != trajectory_advantage:
            raise RuntimeError("step-credit allocation did not conserve A_fin exactly")
        for step, credit in zip(output.steps, credits, strict=True):
            step.reward_score = credit.credit
            step.extra_fields["finance_credit"] = credit.credit
            step.extra_fields["finance_credit_mechanism"] = credit.mechanism
            step.extra_fields["finance_credit_output_semantics"] = credit.output_semantics
            step.extra_fields["finance_credit_step_count"] = len(credits)
            step.extra_fields["finance_trajectory_advantage"] = trajectory_advantage
            step.extra_fields["finance_trajectory_complete"] = True
        record_step_credit_evidence(
            identity=normalize_rollout_identity(kwargs[ROLLOUT_IDENTITY_KWARG]),
            base_seed=self.config.data.seed,
            mechanism=self._credit_strategy.name,
            output_semantics=self._credit_strategy.output_semantics,
            trajectory_advantage=trajectory_advantage,
            credits=[float(step.reward_score) for step in output.steps],
            actions=[context.action for context in contexts],
        )
        return output
