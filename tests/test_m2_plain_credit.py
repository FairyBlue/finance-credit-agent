"""Paired actor-only pooling ablation on real flattened CPU StepPO batches."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
import torch
from test_screening_runtime import _batch, _rows

from src.agents.screening_runtime import compute_screening_advantage
from src.rewards.sibling_credit import compute_sibling_credit


def test_plain_keeps_session_local_diversity_gate_despite_unequal_step_positions():
    actions = {
        1: ("buy", "buy", "hold", "sell"),
        2: ("buy", "sell", "hold", "abstain"),
    }
    rows = [
        replace(row, action=actions.get(row.decision_session_index, ("hold",) * 4)[row.rollout_id])
        if row.result_session_index > row.decision_session_index
        else row
        for row in _rows()
    ]
    credit = compute_sibling_credit(rows)
    expected = {
        1: ((-3.5, -1.5, 0.5, 4.5), (-2.5, -2.5, 0.5, 4.5)),
        2: ((-3.5, -1.5, 0.5, 4.5), (-3.5, -1.5, 0.5, 4.5)),
    }
    for index, row in enumerate(rows):
        session = row.decision_session_index
        advancing = row.result_session_index > session
        plain, pooled = expected.get(session, ((0.0,) * 4, (0.0,) * 4))
        assert credit.plain[index] == (plain[row.rollout_id] if advancing else 0.0)
        assert credit.pooled[index] == (pooled[row.rollout_id] if advancing else 0.0)
    assert credit.diagnostics["action_diversity_rate"] == 2 / 20
    assert credit.diagnostics["pooling_opportunity_rate"] == 1 / 20


def test_plain_and_pooled_change_only_actor_residual_under_common_penalty():
    batches = {arm: _batch(arm) for arm in ("M1", "M2-plain", "M2")}
    for arm, batch in batches.items():
        compute_screening_advantage(batch, arm=arm, eta=0.25, invalid_action_penalty=0.05)
    m1, plain, pooled = (batches[arm] for arm in ("M1", "M2-plain", "M2"))
    for field in ("returns", "token_level_rewards", "rm_scores"):
        assert torch.equal(plain.batch[field], m1.batch[field])
        assert torch.equal(pooled.batch[field], m1.batch[field])
    assert not torch.equal(plain.batch["advantages"], pooled.batch["advantages"])

    expected_by_arm = {
        "M2-plain": (-0.875, -0.375, 0.125, 1.125),
        "M2": (-0.625, -0.625, 0.125, 1.125),
    }
    for arm, expected in expected_by_arm.items():
        diagnostics = batches[arm].meta_info["finance_screening_diagnostics"]
        evidence = diagnostics["reward_evidence"]
        residuals = []
        for row in evidence:
            advancing = row["action"] in {"buy", "sell", "hold", "abstain"}
            residual = expected[row["rollout_id"]] if advancing else 0.0
            assert row["sibling_residual"] == residual
            residuals.append(residual)
        # Reported S already includes eta; a second scaling would fail this.
        assert diagnostics["all_real_steps"]["std_S"] == pytest.approx(
            torch.tensor(residuals, dtype=torch.float64).std(correction=1).item()
        )
        assert Counter(row["invalid_penalty"] for row in evidence) == {0.0: 170, 0.05: 2}


@pytest.mark.parametrize("arm", ["M2-plain", "M2"])
def test_zero_eta_keeps_exact_shared_penalized_m1_tensors(arm):
    m1, candidate = _batch("M1"), _batch(arm)
    compute_screening_advantage(m1, arm="M1", invalid_action_penalty=0.05)
    compute_screening_advantage(candidate, arm=arm, eta=0, invalid_action_penalty=0.05)
    for field in ("advantages", "returns", "token_level_rewards"):
        assert torch.equal(candidate.batch[field], m1.batch[field])
