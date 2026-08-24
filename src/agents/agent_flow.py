"""External Agent-R1 flows; imported only inside the GPU training runtime."""

from __future__ import annotations

from typing import Any

from agent_r1.agent_flow.agent_env_loop import AgentEnvLoop

# Importing TradingEnv registers `env_type: trading` without touching upstream files.
from src.env.trading import TradingEnv as _TradingEnv  # noqa: F401
from src.rewards import StepContext, build_credit_strategy


class TradingAgentEnvLoop(AgentEnvLoop):
    """Agent-R1's generic step-MDP loop with the external trading env registered."""


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
