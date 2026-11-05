"""Shared environment rejection target through real flattened StepPO batches."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
from test_actor_update_probe import make_actor, make_batch, read_record
from test_screening_runtime import _batch, _rows, _upstream
from verl import DataProto

from src.agents import actor_update_probe as probe
from src.agents import screening_runtime as runtime
from src.agents.screening_runtime import (
    compute_screening_advantage,
    unwhitened_step_gae,
    validate_invalid_action_penalty,
)
from src.rewards.sibling_credit import compute_sibling_credit


@pytest.mark.parametrize("arm", ["M1", "M2-plain", "M2"])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.bfloat16])
def test_actual_gae_critic_and_actor_share_penalty_but_residual_keeps_financial_y(
    arm, value_dtype, tmp_path
):
    batch = _batch(arm, value_dtype=value_dtype)
    original = copy.deepcopy(batch)
    expected = copy.deepcopy(batch)
    invalid = torch.tensor(batch.non_tensor_batch["trading_invalid_action"].tolist())
    response = batch.batch["response_mask"]
    last_tokens = response.sum(dim=1).long() - 1
    expected.batch["token_level_rewards"][torch.arange(len(batch)), last_tokens] -= (
        invalid.to(torch.float32) * 0.05
    )
    native_a, native_returns = _upstream(expected)
    u, _ = unwhitened_step_gae(
        token_level_rewards=expected.batch["token_level_rewards"],
        values=expected.batch["values"],
        response_mask=response,
        trajectory_uids=expected.non_tensor_batch["trajectory_uids"],
        step_indices=expected.non_tensor_batch["step_indices"],
        gamma=1,
        lam=1,
    )
    # These are the original financial scores (not Y minus invalid penalties).
    credit = compute_sibling_credit(_rows(groups=2))
    s = torch.tensor(credit.plain if arm == "M2-plain" else credit.pooled, dtype=u.dtype) * 0.25
    if arm == "M1":
        s.zero_()
    combined = u + s
    final_a = (combined - combined.mean()) / (combined.std(correction=1) + 1e-8)
    compute_screening_advantage(
        batch, arm=arm, eta=0.25, invalid_action_penalty=0.05, diagnostics_dir=tmp_path
    )
    assert torch.equal(batch.batch["returns"], native_returns)
    assert torch.equal(batch.batch["advantages"][:, 0], final_a)
    if arm == "M1":
        assert torch.equal(batch.batch["advantages"], native_a)
    assert torch.equal(batch.batch["token_level_rewards"], expected.batch["token_level_rewards"])
    assert torch.equal(batch.batch["rm_scores"], original.batch["rm_scores"])
    assert np.array_equal(
        batch.non_tensor_batch["finance_screening_terminal_score"],
        original.non_tensor_batch["finance_screening_terminal_score"],
    )
    assert not torch.equal(batch.batch["returns"], _upstream(original)[1])
    diagnostics = json.loads(next(tmp_path.glob("advantage-*.jsonl")).read_text())
    common = diagnostics["common_validity_penalty"]
    assert common["coefficient"] == 0.05
    assert common["invalid_count"] == 2
    assert common["penalty_sum"] == pytest.approx(0.1)
    for trajectory in common["trajectories"]:
        assert trajectory["training_reward_sum"] == pytest.approx(
            trajectory["terminal_Y"] - 0.05 * trajectory["invalid_count"], abs=1e-6
        )
    assert len(diagnostics["reward_evidence"]) == len(batch)
    for index, row in enumerate(diagnostics["reward_evidence"]):
        assert row["invalid_penalty"] == (0.05 if bool(invalid[index]) else 0.0)
        assert row["sibling_residual"] == float(s[index])
        assert row["final_actor_advantage"] == float(final_a[index])
        assert row["critic_return"] == float(native_returns[index, 0])
        assert row["financial_step_reward"] - row["invalid_penalty"] == pytest.approx(
            row["training_step_reward"], abs=1e-7
        )
        assert (
            batch.non_tensor_batch["finance_common_training_step_reward"][index]
            == (row["training_step_reward"])
        )


@pytest.mark.parametrize("arm", ["A2", "M1", "M2-plain", "M2"])
def test_default_and_explicit_off_preserve_historical_tensors_and_metadata(arm):
    default = _batch(arm)
    explicit = copy.deepcopy(default)
    compute_screening_advantage(default, arm=arm)
    compute_screening_advantage(explicit, arm=arm, invalid_action_penalty=0)
    assert set(default.batch.keys()) == set(explicit.batch.keys())
    for key, tensor in default.batch.items():
        assert torch.equal(tensor, explicit.batch[key])
    assert default.meta_info == explicit.meta_info
    assert not any(key.startswith("finance_common_") for key in default.non_tensor_batch)
    assert "common_validity_penalty" not in default.meta_info["finance_screening_diagnostics"]


@pytest.mark.parametrize("arm", ["M2-plain", "M2"])
def test_penalty_ignores_padding_and_survives_trainer_reordering(arm):
    clean = _batch(arm)
    data = copy.deepcopy(clean)
    padding = data.select_idxs([0, 4])
    padding.batch["sample_mask"][:] = False
    padding.batch["response_mask"][:] = 0
    padding.batch["token_level_rewards"][:] = float("nan")
    padding.batch["values"][:] = float("nan")
    for key in padding.non_tensor_batch:
        padding.non_tensor_batch[key] = np.array([None, None], dtype=object)
    data = DataProto.concat([data, padding])
    order = np.random.default_rng(13).permutation(len(data))
    data = data.select_idxs(order)
    compute_screening_advantage(clean, arm=arm, invalid_action_penalty=0.05)
    compute_screening_advantage(data, arm=arm, invalid_action_penalty=0.05)
    restored = data.select_idxs(np.argsort(order))
    for field in ("advantages", "returns", "token_level_rewards"):
        torch.testing.assert_close(restored.batch[field][: len(clean)], clean.batch[field])
    assert torch.count_nonzero(restored.batch["advantages"][len(clean) :]) == 0
    assert restored.non_tensor_batch["finance_common_invalid_penalty"][-2:].tolist() == [None, None]
    assert (
        data.meta_info["finance_screening_diagnostics"]["common_validity_penalty"]["invalid_count"]
        == 2
    )


@pytest.mark.parametrize("arm", ["A2"])
def test_unregistered_arm_cannot_silently_gain_common_penalty(arm):
    with pytest.raises(ValueError, match="only for M1, M2-plain, and M2"):
        compute_screening_advantage(_batch(arm), arm=arm, invalid_action_penalty=0.05)


@pytest.mark.parametrize("coefficient", [True, -0.05, "0.05", 0.1, float("nan")])
def test_unregistered_coefficient_rejected(coefficient):
    with pytest.raises(ValueError, match="penalty"):
        validate_invalid_action_penalty(coefficient, "M1")


@pytest.mark.parametrize("arm", ["M2-plain", "M2"])
def test_environment_installs_explicit_coefficient_and_default_off(tmp_path, monkeypatch, arm):
    monkeypatch.setenv("FINANCE_SCREENING_ARM", arm)
    monkeypatch.setenv("FINANCE_SCREENING_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCE_SCREENING_RUN_SEED", "42")
    monkeypatch.setenv("FINANCE_SCREENING_RESIDUAL_ETA", "0.25")
    installed = []
    monkeypatch.setattr(
        runtime, "patch_trainer_module", lambda _module, settings: installed.append(settings)
    )
    monkeypatch.setattr(runtime, "_configured_settings", None)
    monkeypatch.setenv("FINANCE_COMMON_INVALID_PENALTY", "0.05")
    runtime.install_from_environment()
    assert installed[-1].invalid_action_penalty == 0.05
    assert installed[-1].eta == 0.25
    monkeypatch.setattr(runtime, "_configured_settings", None)
    monkeypatch.delenv("FINANCE_COMMON_INVALID_PENALTY")
    runtime.install_from_environment()
    assert installed[-1].invalid_action_penalty == 0


def test_evaluation_and_double_application_are_rejected_without_mutation():
    batch = _batch("M1")
    original_rewards = batch.batch["token_level_rewards"].clone()
    batch.meta_info["validate"] = True
    with pytest.raises(ValueError, match="evaluation"):
        compute_screening_advantage(batch, arm="M1", invalid_action_penalty=0.05)
    assert torch.equal(batch.batch["token_level_rewards"], original_rewards)
    batch.meta_info.pop("validate")
    compute_screening_advantage(batch, arm="M1", invalid_action_penalty=0.05)
    once_rewards = batch.batch["token_level_rewards"].clone()
    with pytest.raises(ValueError, match="already been computed"):
        compute_screening_advantage(batch, arm="M1", invalid_action_penalty=0.05)
    assert torch.equal(batch.batch["token_level_rewards"], once_rewards)


def _add_reward_components(data):
    size = len(data)
    for field, value in (
        ("finance_common_financial_step_reward", 1.0),
        ("finance_common_invalid_penalty", 0.0),
        ("finance_common_training_step_reward", 1.0),
    ):
        data.non_tensor_batch[field] = np.full(size, value, dtype=object)
    data.non_tensor_batch["finance_common_invalid_penalty"][1] = 0.05
    data.non_tensor_batch["finance_common_training_step_reward"][1] = 0.95
    data.batch["token_level_rewards"][1, 0] = 0.95


def test_reward_components_persist_with_real_gspo_loss_advantage(tmp_path, monkeypatch):
    from agent_r1.workers.actor import dp_actor as agent_actor

    actor = make_actor(monkeypatch, loss_mode="gspo")
    data = make_batch(padding=True)
    _add_reward_components(data)
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    actor.update_policy(data)
    invalid = read_record(tmp_path)["rows"][1]
    assert invalid["financial_step_reward"] == 1
    assert invalid["invalid_penalty"] == 0.05
    assert invalid["training_step_reward"] == 0.95
    assert invalid["assigned_step_reward"] == pytest.approx(0.95)
    assert invalid["advantage"] == [-1.0] * 3


@pytest.mark.parametrize("damage", ["missing", "wrong_reward", "wrong_penalty"])
def test_actor_rejects_component_mismatch_before_any_update(tmp_path, monkeypatch, damage):
    from agent_r1.workers.actor import dp_actor as agent_actor

    actor = make_actor(monkeypatch, loss_mode="gspo")
    data = make_batch()
    _add_reward_components(data)
    if damage == "missing":
        data.non_tensor_batch.pop("finance_common_invalid_penalty")
    elif damage == "wrong_reward":
        data.batch["token_level_rewards"][1, 0] = 1.0
    else:
        data.non_tensor_batch["finance_common_invalid_penalty"][1] = 0.0
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    with pytest.raises(ValueError, match="common reward"):
        actor.update_policy(data)
    assert actor.pre_calls == 0
