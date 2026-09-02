"""Synthetic contract from M1 immediate rewards through Agent-R1 step GAE."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from agent_r1.trainer.ppo.core_algos import compute_gae_advantage_return

from src.rewards import StepContext, UniformBroadcastCredit


def test_uniform_m1_rewards_become_nonuniform_step_gae_advantages() -> None:
    steps = [
        StepContext(step_id="s1", action="retrieve"),
        StepContext(step_id="s2", action="compute"),
        StepContext(step_id="s3", action="hold"),
    ]
    credits = UniformBroadcastCredit().allocate(
        trajectory_advantage=0.9,
        steps=steps,
    )
    token_level_rewards = torch.tensor(
        [[credit.credit] for credit in credits], dtype=torch.float32
    )
    response_mask = torch.ones_like(token_level_rewards)

    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=torch.zeros_like(token_level_rewards),
        response_mask=response_mask,
        trajectory_uids=np.array(["trajectory"] * 3, dtype=object),
        step_indices=np.array([0, 1, 2], dtype=np.int32),
        gamma=torch.tensor(1.0),
        lam=torch.tensor(1.0),
    )

    assert token_level_rewards[:, 0].tolist() == pytest.approx([0.3, 0.3, 0.3])
    assert returns[:, 0].tolist() == pytest.approx([0.9, 0.6, 0.3])
    assert advantages[:, 0].tolist() == pytest.approx([1.0, 0.0, -1.0], abs=1e-6)
