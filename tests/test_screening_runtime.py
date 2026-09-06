"""M2 formula and real Agent-R1 flatten/driver/GAE contracts on CPU tensors."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from agent_r1.agent_flow.agent_flow import (
    AgentFlowMetrics,
    AgentFlowOutput,
    AgentFlowWorkerBase,
    _InternalAgentFlowStep,
)
from agent_r1.trainer.ppo import ray_trainer
from agent_r1.trainer.ppo.core_algos import compute_gae_advantage_return
from tensordict import TensorDict  # type: ignore[import-untyped]
from verl import DataProto

from src.agents.screening_runtime import (
    ScreeningRuntimeSettings,
    compute_screening_advantage,
    patch_trainer_module,
    unwhitened_step_gae,
)
from src.rewards.sibling_credit import SiblingStep, compute_sibling_credit


def _rows(*, groups: int = 1, actions: tuple[str, ...] = ("buy", "buy", "hold", "sell")):
    result = []
    for group in range(groups):
        for rollout, score in enumerate((0.0, 2.0, 4.0, 8.0)):
            # Unequal trajectory lengths and non-advancing steps exercise actual GAE.
            prefix = ("retrieve", "compute", "invalid")[:rollout]
            count = 20 + len(prefix)
            for step, action in enumerate((*prefix, *([actions[rollout]] * 20))):
                decision = max(0, step - len(prefix))
                advancing = step >= len(prefix)
                result.append(
                    SiblingStep(
                        trajectory_uid=f"t{group}-{rollout}",
                        group_uid=f"g{group}",
                        episode_id=f"formal-train-{group + 1:04}",
                        policy_step=7,
                        rollout_id=rollout,
                        rollout_n=4,
                        step_index=step,
                        step_count=count,
                        decision_session_index=decision,
                        result_session_index=decision + int(advancing),
                        action=action,
                        invalid_action=action == "invalid",
                        done=step == count - 1,
                        episode_complete=True,
                        terminal_score=score + group * 0.7,
                    )
                )
    return result


def _batch(arm: str = "M2", *, groups: int = 2, value_dtype=torch.float32):
    """Use upstream _postprocess and fit's actual repeat/union sequence."""
    source_rows = _rows(groups=groups)
    outputs = []
    for trajectory in dict.fromkeys(row.trajectory_uid for row in source_rows):
        rows = [row for row in source_rows if row.trajectory_uid == trajectory]
        score = rows[0].terminal_score
        raw_return = score / 100 - 0.03
        if arm == "A2":
            rewards = [0.0] * (len(rows) - 1) + [raw_return]
        else:
            rewards = [score / len(rows)] * (len(rows) - 1)
            rewards.append(score - sum(rewards))
        steps = []
        for row, reward in zip(rows, rewards, strict=True):
            length = 1 + row.step_index % 4
            mask = torch.tensor([[int(i < length) for i in range(4)]])
            steps.append(
                _InternalAgentFlowStep(
                    prompt_ids=torch.tensor([[1, 2]]),
                    response_ids=torch.tensor([[3, 4, 5, 6]]),
                    response_mask=mask,
                    input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
                    position_ids=torch.arange(6).unsqueeze(0),
                    attention_mask=torch.cat([torch.ones(1, 2, dtype=torch.long), mask], dim=1),
                    reward_score=reward,
                    extra_fields={
                        "finance_screening_arm": arm,
                        "finance_screening_group_uid": row.group_uid,
                        "finance_screening_episode_id": row.episode_id,
                        "finance_screening_policy_step": row.policy_step,
                        "finance_screening_rollout_id": row.rollout_id,
                        "finance_screening_rollout_n": 4,
                        "finance_screening_terminal_score": row.terminal_score,
                        "finance_screening_net_return": raw_return,
                        "finance_screening_trajectory_complete": True,
                        "finance_screening_step_count": row.step_count,
                        "trading_action": row.action,
                        "trading_decision_session_index": row.decision_session_index,
                        "trading_result_session_index": row.result_session_index,
                        "trading_env_done": row.done,
                        "trading_invalid_action": row.invalid_action,
                        "trading_episode_complete": True,
                    },
                )
            )
        outputs.append(AgentFlowOutput(steps=steps, metrics=AgentFlowMetrics()))
    flow_worker = object.__new__(AgentFlowWorkerBase)
    flattened = flow_worker._postprocess(outputs)
    prompt_batch = DataProto(
        batch=TensorDict({"fixture": torch.zeros(groups, 1)}, batch_size=[groups]),
        non_tensor_batch={"uid": np.array([f"g{i}" for i in range(groups)], dtype=object)},
    )
    batch = prompt_batch.repeat(repeat_times=4, interleave=True)
    batch = batch.sample_level_repeat([len(output.steps) for output in outputs])
    batch = batch.union(flattened)
    batch.batch["token_level_rewards"] = batch.batch["rm_scores"].clone()
    values = torch.sin(torch.arange(len(batch), dtype=torch.float32) / 9).unsqueeze(1)
    batch.batch["values"] = values.expand(-1, 4).to(value_dtype).clone()
    batch.batch["sample_mask"] = torch.ones(len(batch), dtype=torch.bool)
    return batch


def _upstream(batch):
    valid = batch.batch["sample_mask"]
    return compute_gae_advantage_return(
        token_level_rewards=batch.batch["token_level_rewards"][valid],
        values=batch.batch["values"][valid],
        response_mask=batch.batch["response_mask"][valid],
        trajectory_uids=batch.non_tensor_batch["trajectory_uids"][valid.numpy()],
        step_indices=batch.non_tensor_batch["step_indices"][valid.numpy()],
        gamma=1.0,
        lam=1.0,
    )


def test_same_gate_formula_and_information_steps():
    rows = _rows()
    result = compute_sibling_credit(rows)
    first_decision = [i for i, row in enumerate(rows) if row.result_session_index == 1]
    assert [result.plain[i] for i in first_decision] == [-3.5, -1.5, 0.5, 4.5]
    assert [result.pooled[i] for i in first_decision] == [-2.5, -2.5, 0.5, 4.5]
    for i, row in enumerate(rows):
        if row.action in {"retrieve", "compute", "invalid"}:
            assert result.plain[i] == result.pooled[i] == 0
    assert sum(result.plain) == sum(result.pooled) == 0
    assert result.diagnostics["action_diversity_rate"] == 1
    assert result.diagnostics["pooling_opportunity_rate"] == 1
    assert result.diagnostics["pooling_changed_row_rate"] == 0.5


@pytest.mark.parametrize("actions", [("hold",) * 4, ("buy", "sell", "hold", "abstain")])
def test_same_family_gate_and_all_distinct_reduction(actions):
    result = compute_sibling_credit(_rows(actions=actions))
    assert result.plain == result.pooled
    if len(set(actions)) == 1:
        assert not any(result.plain)
    assert result.diagnostics["pooling_opportunity_rate"] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows[:-1], "step indices"),
        (lambda rows: [*rows, rows[0]], "step indices"),
        (
            lambda rows: [
                replace(row, rollout_id=0) if row.rollout_id == 1 else row for row in rows
            ],
            "duplicate rollout",
        ),
        (lambda rows: [row for row in rows if row.rollout_id != 3], "missing sibling"),
        (lambda rows: [replace(rows[0], policy_step=8), *rows[1:]], "policy snapshot"),
        (lambda rows: [replace(rows[0], episode_id="wrong"), *rows[1:]], "episode occurrence"),
        (lambda rows: [replace(rows[0], episode_complete=False), *rows[1:]], "incomplete"),
        (lambda rows: [replace(rows[0], done=True), *rows[1:]], "final retained"),
        (lambda rows: [replace(rows[0], result_session_index=2), *rows[1:]], "session transition"),
        (lambda rows: [replace(rows[0], terminal_score=float("nan")), *rows[1:]], "finite"),
    ],
)
def test_malformed_siblings_fail_closed(mutation, match):
    with pytest.raises(ValueError, match=match):
        compute_sibling_credit(mutation(_rows()))


@pytest.mark.parametrize("arm", ["A2", "M1", "M2-plain", "M2"])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.bfloat16])
def test_real_postprocess_driver_keeps_upstream_critic_returns(
    arm, value_dtype, tmp_path, monkeypatch
):
    batch = _batch(arm, value_dtype=value_dtype)
    upstream_advantage, upstream_returns = _upstream(batch)
    original = ray_trainer.compute_advantage
    monkeypatch.setattr(ray_trainer, "compute_advantage", original)
    trainer_class = ray_trainer.RayAgentTrainer
    monkeypatch.setattr(trainer_class, "_save_checkpoint", trainer_class._save_checkpoint)
    monkeypatch.setattr(trainer_class, "_load_checkpoint", trainer_class._load_checkpoint)
    monkeypatch.setattr(trainer_class, "_validate", trainer_class._validate)
    monkeypatch.setattr(trainer_class, "_finance_screening_checkpoint_hooks", False, raising=False)
    settings = ScreeningRuntimeSettings(arm, tmp_path.resolve(), 42)
    assert patch_trainer_module(ray_trainer, settings)
    assert not patch_trainer_module(ray_trainer, settings)
    assert trainer_class._finance_screening_checkpoint_hooks
    output = ray_trainer.compute_advantage(
        batch,
        adv_estimator=ray_trainer.AgentAdvantageEstimator.GAE,
        gamma=1,
        lam=1,
        num_repeat=4,
        config={"use_kl_in_reward": False},
    )
    assert torch.equal(output.batch["returns"], upstream_returns)
    if arm in {"A2", "M1"}:
        assert torch.equal(output.batch["advantages"], upstream_advantage)
    else:
        assert not torch.allclose(output.batch["advantages"], upstream_advantage)
    files = list(tmp_path.glob("advantage-*.jsonl"))
    record = json.loads(files[0].read_text())
    assert record["trajectory_count"] == 8
    assert record["policy_step"] == 7
    assert record["run_seed"] == 42
    assert record["whitening_scope"] == "logical_rollout_batch_real_step_rows"
    assert record["advancing_row_count"] == 160
    assert record["real_step_count"] == 172
    assert record["phase"] == "advantage_pre_update"
    assert output.meta_info["finance_screening_diagnostics"] == record


@pytest.mark.parametrize("arm", ["M2", "M2-plain"])
def test_zero_eta_is_bitwise_m1_on_same_real_batch(arm):
    batch = _batch(arm)
    expected, expected_returns = _upstream(batch)
    compute_screening_advantage(batch, arm=arm, eta=0)
    assert torch.equal(batch.batch["advantages"], expected)
    assert torch.equal(batch.batch["returns"], expected_returns)


def test_padding_reordering_and_token_lengths_do_not_change_whitening():
    clean = _batch("M2")
    data = copy.deepcopy(clean)
    # The pinned trainer's padding adds repeated rows and masks them out.
    padding = data.select_idxs(np.array([0, 4, 9]))
    padding.batch["sample_mask"][:] = False
    padding.batch["response_mask"][:] = 0
    padding.batch["token_level_rewards"][:] = float("nan")
    padding.batch["values"][:] = float("nan")
    for key in padding.non_tensor_batch:
        padding.non_tensor_batch[key] = np.array([None] * 3, dtype=object)
    data = DataProto.concat([data, padding])
    order = np.random.default_rng(13).permutation(len(data))
    data = data.select_idxs(order)
    compute_screening_advantage(data, arm="M2")
    compute_screening_advantage(clean, arm="M2")
    restored_advantages = data.batch["advantages"][np.argsort(order)]
    restored_returns = data.batch["returns"][np.argsort(order)]
    torch.testing.assert_close(restored_advantages[: len(clean)], clean.batch["advantages"])
    torch.testing.assert_close(restored_returns[: len(clean)], clean.batch["returns"])
    assert torch.count_nonzero(restored_advantages[len(clean) :]) == 0
    assert torch.count_nonzero(restored_returns[len(clean) :]) == 0
    # One scalar per step has sample std 1, regardless of response length.
    scalars = clean.batch["advantages"][:, 0]
    assert scalars.mean().item() == pytest.approx(0, abs=1e-6)
    assert scalars.std(correction=1).item() == pytest.approx(1, abs=1e-6)


def test_actor_residual_added_after_gae_with_one_global_whitening():
    batch = _batch("M2")
    raw_u, returns = unwhitened_step_gae(
        token_level_rewards=batch.batch["token_level_rewards"],
        values=batch.batch["values"],
        response_mask=batch.batch["response_mask"],
        trajectory_uids=batch.non_tensor_batch["trajectory_uids"],
        step_indices=batch.non_tensor_batch["step_indices"],
        gamma=1,
        lam=1,
    )
    raw_s = torch.tensor(compute_sibling_credit(_rows(groups=2)).pooled, dtype=raw_u.dtype)
    combined = raw_u + raw_s
    expected = (combined - combined.mean()) / (combined.std(correction=1) + 1e-8)
    rewards_before = batch.batch["token_level_rewards"].clone()
    compute_screening_advantage(batch, arm="M2")
    assert torch.equal(batch.batch["advantages"][:, 0], expected)
    assert torch.equal(batch.batch["returns"][:, 0], returns)
    assert torch.equal(batch.batch["token_level_rewards"], rewards_before)


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda batch: batch.non_tensor_batch.pop("finance_screening_policy_step"),
            "evidence missing",
        ),
        (lambda batch: batch.non_tensor_batch["uid"].__setitem__(0, "wrong"), "occurrence UID"),
        (
            lambda batch: batch.non_tensor_batch["finance_screening_arm"].__setitem__(0, "A2"),
            "runtime arm",
        ),
        (
            lambda batch: batch.batch["token_level_rewards"].__setitem__((0, 0), 99),
            "reward timeline",
        ),
        (lambda batch: batch.batch["values"].__setitem__((0, 0), float("nan")), "non-finite"),
        (lambda batch: batch.batch["response_mask"].__setitem__((0, 0), 0), "valid response token"),
    ],
)
def test_real_batch_evidence_corruption_is_rejected(change, match):
    batch = _batch()
    change(batch)
    with pytest.raises(ValueError, match=match):
        compute_screening_advantage(batch, arm="M2")
    assert "advantages" not in batch.batch


def test_repeated_whitening_and_unapproved_contract_are_rejected():
    batch = _batch()
    with pytest.raises(ValueError, match="gamma=1"):
        compute_screening_advantage(batch, arm="M2", gamma=0.99)
    compute_screening_advantage(batch, arm="M2")
    with pytest.raises(ValueError, match="already been computed"):
        compute_screening_advantage(batch, arm="M2")


def test_sitecustomize_installs_screening_hook_on_real_trainer_import(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    for name in (
        "FINANCE_UPDATE_PROBE_DIR",
        "FINANCE_VLLM_DRAIN_COMPAT",
        "FINANCE_ROLLOUT_SEED_DIR",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(repo / "runtime_hooks"), str(repo), str(repo / "third_party/agent-r1")]
            ),
            "FINANCE_SCREENING_ARM": "M2",
            "FINANCE_SCREENING_DIAGNOSTICS_DIR": str(tmp_path),
            "FINANCE_SCREENING_RUN_SEED": "43",
        }
    )
    script = (
        "import sys; assert 'agent_r1.trainer.ppo.ray_trainer' not in sys.modules; "
        "from agent_r1.trainer.ppo import ray_trainer; "
        "s=ray_trainer.compute_advantage._finance_screening_settings; "
        "assert ray_trainer.RayAgentTrainer._finance_screening_checkpoint_hooks; "
        "assert s.arm == 'M2' and s.run_seed == 43; print('SCREENING_HOOK_OK')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SCREENING_HOOK_OK" in completed.stdout


def test_sitecustomize_missing_screening_contract_fails_before_user_code(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    for name in (
        "FINANCE_UPDATE_PROBE_DIR",
        "FINANCE_VLLM_DRAIN_COMPAT",
        "FINANCE_ROLLOUT_SEED_DIR",
        "FINANCE_SCREENING_DIAGNOSTICS_DIR",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join([str(repo / "runtime_hooks"), str(repo)]),
            "FINANCE_SCREENING_ARM": "M2",
            "FINANCE_SCREENING_RUN_SEED": "42",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", "print('BODY_RAN')"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "BODY_RAN" not in completed.stdout
    assert "FINANCE_SCREENING_DIAGNOSTICS_DIR" in completed.stderr
