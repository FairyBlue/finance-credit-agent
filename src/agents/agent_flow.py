"""External Agent-R1 flows; imported only inside the GPU training runtime."""

from __future__ import annotations

from typing import Any

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

# Importing TradingEnv registers `env_type: trading` without touching upstream files.
from src.env.trading import TradingEnv as _TradingEnv  # noqa: F401
from src.rewards import StepContext, build_credit_strategy

from .rollout_seed import (
    ROLLOUT_IDENTITY_KWARG,
    ROLLOUT_SEED_SCHEME,
    build_seeded_server_manager,
    normalize_rollout_identity,
)


class TradingAgentEnvLoop(AgentEnvLoop):
    """Agent-R1's generic step-MDP loop with the external trading env registered."""

    server_manager: Any

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

    def __init__(self, *args: Any, credit_strategy: str = "m1_uniform", **kwargs: Any) -> None:
        self._credit_strategy = build_credit_strategy(credit_strategy)
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        output = await super().run(sampling_params, **kwargs)
        if not output.steps:
            return output
        trajectory_advantage = sum(float(step.reward_score or 0.0) for step in output.steps)
        contexts = [
            StepContext(step_id=f"step-{index}", action="agent_env_step")
            for index, _ in enumerate(output.steps)
        ]
        credits = self._credit_strategy.allocate(
            trajectory_advantage=trajectory_advantage,
            steps=contexts,
        )
        for step, credit in zip(output.steps, credits, strict=True):
            step.reward_score = credit.credit
            step.extra_fields["finance_credit_mechanism"] = credit.mechanism
            step.extra_fields["finance_trajectory_advantage"] = trajectory_advantage
        return output
