"""Same-batch fit instrumentation over the real Agent-R1 CPU critic loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from agent_r1.trainer.ppo.trajectory_batching import prepare_trajectory_mini_batch
from agent_r1.workers.critic import dp_critic as agent_critic
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.profiler import performance
from verl.workers.critic import dp_critic as verl_critic

from src.agents import critic_fit_probe as probe


def make_critic(monkeypatch, *, fail_post=False):
    monkeypatch.setattr(agent_critic, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(verl_critic, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0", "0", "0", "0"))

    class ToyCritic(agent_critic.DataParallelPPOCritic):
        def _forward_micro_batch(self, micro_batch):
            return self.critic_module(micro_batch["input_ids"][:, :1].float()).expand(-1, 2)

        def compute_values(self, data):
            self.forward_metadata = dict(data.meta_info)
            self.post_forward_grad_enabled = torch.is_grad_enabled()
            # Exercise RNG preservation, including the failure path.
            torch.rand(3)
            if fail_post:
                self.critic_module.eval()
                raise RuntimeError("fixture post failure")
            result = super().compute_values(data)
            self.post_forward_training_mode = self.critic_module.training
            return result

    module = torch.nn.Linear(1, 1)
    with torch.no_grad():
        module.weight.zero_()
        module.bias.zero_()
    config = OmegaConf.create(
        {
            "model": {"use_remove_padding": False},
            "ppo_epochs": 1,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_mini_batch_size": 4,
            "use_dynamic_bsz": False,
            "forward_micro_batch_size_per_gpu": 1,
            "forward_max_token_len_per_gpu": 32768,
            "grad_clip": 1.0,
            "cliprange_value": 0.5,
            "loss_agg_mode": "token-mean",
        }
    )
    return ToyCritic(config, module, torch.optim.SGD(module.parameters(), lr=0.1))


def make_batch(*, padding=False):
    response_mask = torch.tensor([[1, 0], [1, 0], [1, 0]])
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]),
            "responses": torch.zeros(3, 2, dtype=torch.long),
            "response_mask": response_mask,
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "position_ids": torch.arange(4).expand(3, -1),
            "values": torch.zeros(3, 2),
            "returns": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        },
        non_tensors={
            "trajectory_uids": np.array(["a", "a", "b"], dtype=object),
            "finance_screening_policy_step": np.array([1, 1, 1]),
        },
        meta_info={"untouched": "yes"},
    )
    planned = prepare_trajectory_mini_batch(data, mini_batch_size=4, dp_size=1)
    if padding:
        planned = planned.select_idxs([0, 1, 2, 0])
        planned.batch["sample_mask"][-1] = False
        planned.batch["response_mask"][-1] = 0
        planned.batch["values"][-1] = float("nan")
        planned.batch["returns"][-1] = float("nan")
    return planned


def read_record(directory: Path):
    files = list(directory.glob("critic-fit-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_real_upstream_update_stays_identical_and_measures_same_batch(tmp_path, monkeypatch):
    baseline = make_critic(monkeypatch)
    wrapped = make_critic(monkeypatch)
    assert probe.patch_critic_class(type(wrapped), tmp_path)
    assert not probe.patch_critic_class(type(wrapped), tmp_path)
    baseline_batch, wrapped_batch = make_batch(), make_batch()
    saved_batch = {key: value.clone() for key, value in wrapped_batch.batch.items()}
    saved_meta = dict(wrapped_batch.meta_info)
    expected = baseline.update_critic(baseline_batch)
    rng = torch.get_rng_state().clone()
    observed = wrapped.update_critic(wrapped_batch)
    assert expected == observed
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            baseline.critic_module.parameters(), wrapped.critic_module.parameters(), strict=True
        )
    )
    assert all(torch.equal(saved_batch[key], value) for key, value in wrapped_batch.batch.items())
    assert wrapped_batch.meta_info == saved_meta
    assert wrapped.critic_module.training
    assert not wrapped.post_forward_grad_enabled
    assert not wrapped.post_forward_training_mode
    record = read_record(tmp_path)
    assert record["policy_step"] == 1
    assert record["real_step_count"] == record["value_token_count"] == 3
    assert record["padding_row_count"] == 0
    assert record["planned_mini_batch_count"] == 1
    assert record["pre"]["half_mse"] == pytest.approx(7 / 3)
    assert record["post"]["half_mse"] < record["pre"]["half_mse"]
    assert record["optimizer_grad_norms_before_clip"] == expected["critic/grad_norm"]
    assert record["extra_forward_passes"] == 1
    assert record["extra_forward_input_tokens"] == 12
    assert record["extra_forward_micro_batches"] == 3
    assert record["forward_meta_info"] == {
        "micro_batch_size": 1,
        "max_token_len": 32768,
        "use_dynamic_bsz": False,
    }
    assert record["dropout"]["any_nonzero"] is False
    assert record["pre"]["returns"] == record["post"]["returns"]


def test_padding_is_excluded_from_fit_and_post_forward(monkeypatch):
    critic = make_critic(monkeypatch)
    data = make_batch(padding=True)
    real_data, pre, returns, metadata = probe._prepare(critic, data)
    assert len(real_data) == len(pre) == len(returns) == 3
    assert metadata["padding_row_count"] == 1
    assert metadata["extra_forward_input_tokens"] == 12
    assert torch.isfinite(pre).all() and torch.isfinite(returns).all()


def test_post_failure_restores_mode_and_rng_without_repeating_update(tmp_path, monkeypatch):
    critic = make_critic(monkeypatch, fail_post=True)
    probe.patch_critic_class(type(critic), tmp_path)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="fixture post failure"):
        critic.update_critic(make_batch())
    assert torch.equal(torch.get_rng_state(), rng)
    assert critic.critic_module.training
    assert critic.critic_module.weight.item() > 0  # ordinary update happened once
    assert not list(tmp_path.glob("*.jsonl"))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.batch["response_mask"].__setitem__((0, 1), 1), "first-token value mask"),
        (lambda data: data.batch["response_mask"].__setitem__((0, 0), 0), "disagrees"),
        (lambda data: data.batch["values"].__setitem__((0, 0), float("nan")), "finite"),
    ],
)
def test_bad_step_masks_and_values_fail_before_optimizer(tmp_path, monkeypatch, mutation, message):
    critic = make_critic(monkeypatch)
    probe.patch_critic_class(type(critic), tmp_path)
    data = make_batch()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        critic.update_critic(data)
    assert critic.critic_module.weight.item() == 0


def test_install_disabled_and_paths_are_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv(probe.DIRECTORY_ENV, raising=False)
    assert not probe.install_from_environment()
    monkeypatch.setenv(probe.DIRECTORY_ENV, "relative")
    with pytest.raises(ValueError, match="absolute"):
        probe.install_from_environment()
    critic = make_critic(monkeypatch)
    probe.patch_critic_class(type(critic), tmp_path)
    with pytest.raises(RuntimeError, match="cannot change"):
        probe.patch_critic_class(type(critic), tmp_path / "other")


def test_dropout_evidence_does_not_hide_train_eval_difference():
    module = torch.nn.Sequential(torch.nn.Dropout(0.2))
    evidence = probe._dropout_evidence(module)
    assert evidence["any_nonzero"]
    assert evidence["checked_probabilities"]["0.p"] == 0.2
    assert evidence["fit_comparison_uses_same_mode"]
