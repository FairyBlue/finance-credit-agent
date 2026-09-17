"""Exercise real Agent-R1 PPO/backward/optimizer with a tiny CPU actor."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from agent_r1.trainer.ppo.trajectory_batching import prepare_trajectory_mini_batch
from agent_r1.workers.actor import dp_actor as agent_actor
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.profiler import performance
from verl.workers.config import FSDPActorConfig, PolicyLossConfig

from src.agents import actor_update_probe as probe


def make_actor(monkeypatch, *, fail_post=False, rollout_anchor=False, loss_mode="vanilla"):
    monkeypatch.setattr(agent_actor, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0", "0", "0", "0"))

    class ToyActor(agent_actor.DataParallelPPOActor):
        def __init__(self):
            self.config = OmegaConf.create(
                {
                    "ppo_epochs": 1,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "ppo_mini_batch_size": 4,
                    "use_dynamic_bsz": False,
                    "loss_agg_mode": "token-mean",
                    "entropy_coeff": 0,
                    "calculate_entropy": False,
                    "use_kl_loss": False,
                    "policy_loss": {"loss_mode": loss_mode},
                    "clip_ratio": 0.2,
                    "clip_ratio_low": None,
                    "clip_ratio_high": None,
                    "clip_ratio_c": 3.0,
                    "grad_clip": 1.0,
                    "loss_scale_factor": 1.0,
                    "use_rollout_log_probs": rollout_anchor,
                }
            )
            if loss_mode == "gspo":
                # Native GSPO asserts real ActorConfig, as used by the worker.
                values = OmegaConf.to_container(self.config, resolve=True)
                values["policy_loss"] = PolicyLossConfig(loss_mode="gspo")
                self.config = FSDPActorConfig(**values, rollout_n=4)
            self.actor_module = torch.nn.Embedding(3, 4)
            torch.nn.init.zeros_(self.actor_module.weight)
            self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
            self.ulysses_sequence_parallel_size = 1
            self.scaler = None
            self.pre_calls = self.post_calls = 0

        def _forward_micro_batch(self, inputs, temperature, calculate_entropy=False):
            if torch.is_grad_enabled():
                self.pre_calls += 1
            else:
                self.post_calls += 1
                torch.rand(3)  # Deliberately exercise restoration, including exceptions.
                if fail_post:
                    self.actor_module.eval()
                    raise RuntimeError("fixture post failure")
            logits = self.actor_module(inputs["input_ids"][:, 0]) / temperature
            logp = torch.log_softmax(logits, dim=-1)
            return None, logp.gather(1, inputs["responses"])

    return ToyActor()


def make_batch(*, padding=False):
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[0, 4, 0, 0, 0], [1, 4, 1, 1, 1], [2, 4, 2, 2, 2]]),
            "responses": torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            "attention_mask": torch.ones(3, 5, dtype=torch.long),
            "position_ids": torch.arange(5).expand(3, -1),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]]),
            "old_log_probs": torch.full((3, 3), -3.0),
            "advantages": torch.tensor([[2.0, 2.0, 0], [-1.0, -1.0, -1.0], [0.5, 0, 0]]),
            "token_level_rewards": torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]),
        },
        non_tensors={
            column: np.array(values, dtype=object)
            for column, values in {
                "trajectory_uids": ["a", "a", "b"],
                "step_indices": [0, 1, 0],
                "finance_screening_episode_id": ["window0", "window0", "window1"],
                "finance_screening_policy_step": [1, 1, 1],
                "finance_screening_rollout_id": ["r0", "r0", "r1"],
                "finance_screening_rollout_n": [0, 0, 1],
                "finance_screening_group_uid": ["g0", "g0", "g1"],
                "trading_action": ["buy", "invalid", "hold"],
                "trading_invalid_action": [False, True, False],
                "trading_decision_session_index": [0, 1, 0],
                "trading_result_session_index": [1, 1, 1],
                "finance_screening_terminal_score": [3.0, 3.0, 3.0],
            }.items()
        },
        meta_info={"temperature": 1.0, "unchanged": True},
    )
    planned = prepare_trajectory_mini_batch(data, mini_batch_size=4, dp_size=1)
    if padding:
        planned = planned.select_idxs([0, 1, 2, 0])
        planned.batch["sample_mask"][-1] = False
        planned.batch["response_mask"][-1] = 0
        planned.batch["advantages"][-1] = 0
    return planned


def make_agent_flow_batch():
    """Produce text-only metadata with the real postprocess and flatten helpers."""
    from agent_r1.agent_flow.agent_flow import (
        AgentFlowBase,
        AgentFlowMetrics,
        AgentFlowOutput,
        AgentFlowStep,
        AgentFlowWorkerBase,
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

    class TextFlow(AgentFlowBase):
        def __init__(self):
            self.processor = None
            self.config = OmegaConf.create(
                {"actor_rollout_ref": {"rollout": {"prompt_length": 2, "response_length": 3}}}
            )
            tokenizer = Tokenizer(WordLevel({str(i): i for i in range(5)}, unk_token="4"))
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, pad_token="0", unk_token="4"
            )

        async def run(self, sampling_params, **kwargs):
            raise AssertionError("CPU schema fixture must not generate")

    flow = TextFlow()
    source = make_batch()

    async def process():
        outputs = []
        for index, response in enumerate(([1, 1], [2, 2, 2], [3])):
            extra_fields = {
                key: probe._scalar(values[index])
                for key, values in source.non_tensor_batch.items()
                if key not in {"trajectory_uids", "step_indices"}
            }
            step = AgentFlowStep(
                prompt_ids=[index + 1, 4],
                response_ids=response,
                response_logprobs=[-math.log(5)] * len(response),
                reward_score=1.0,
                extra_fields=extra_fields,
            )
            outputs.append(await flow._postprocess(step, raw_prompt=[{"role": "user"}]))
        return outputs

    steps = asyncio.run(process())
    assert all(step.multi_modal_inputs == {} for step in steps)
    # Real flatten owns trajectory UUIDs/step indices, extra-field dtypes and MM.
    data = AgentFlowWorkerBase._postprocess(
        None,
        [
            AgentFlowOutput(steps=steps[:2], metrics=AgentFlowMetrics()),
            AgentFlowOutput(steps=steps[2:], metrics=AgentFlowMetrics()),
        ],
    )
    assert data.non_tensor_batch["multi_modal_inputs"].tolist() == [{}, {}, {}]
    data.batch["old_log_probs"] = torch.full_like(data.batch["rm_scores"], -math.log(5))
    data.batch["ref_log_prob"] = data.batch["old_log_probs"].clone()
    data.batch["advantages"] = source.batch["advantages"].clone()
    data.batch["token_level_rewards"] = data.batch["rm_scores"].clone()
    data.meta_info["temperature"] = 0.7  # Driver's actual runtime metadata channel.
    # Exercise the actual enabled correction helper: no IS/RS/veto produces no
    # IS tensor, not a fabricated empty tensor that the probe must silently drop.
    data, _ = compute_rollout_correction_and_add_to_batch(
        data,
        OmegaConf.create(
            {"rollout_is": None, "rollout_rs": None, "rollout_token_veto_threshold": None}
        ),
    )
    assert "rollout_is_weights" not in data.batch
    return prepare_trajectory_mini_batch(data, mini_batch_size=8, dp_size=1)


def make_native_forward_actor(monkeypatch):
    """Keep native verl MM extraction/forward and native Agent-R1 GSPO update."""
    from verl.utils import torch_functional

    # Select native torch fallback: installed FlashAttention otherwise attempts
    # a Triton CUDA kernel even when every fixture tensor lives on CPU.
    monkeypatch.setattr(torch_functional, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    monkeypatch.setattr(torch_functional, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    actor = make_actor(monkeypatch, loss_mode="gspo")

    class TinyTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 5)
            torch.nn.init.zeros_(self.embedding.weight)
            self.forward_calls = []

        def forward(self, input_ids, attention_mask, position_ids, **kwargs):
            self.forward_calls.append((torch.is_grad_enabled(), dict(kwargs)))
            if not torch.is_grad_enabled():
                torch.rand(2)  # A real native post forward still preserves RNG.
            return SimpleNamespace(logits=self.embedding(input_ids))

    actor.actor_module = TinyTextModel()
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor.device_name = "cpu"
    actor.param_dtype = torch.bfloat16
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.config = replace(
        actor.config, use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type="low_var_kl"
    )
    type(actor)._forward_micro_batch = agent_actor.VerlDataParallelPPOActor._forward_micro_batch
    return actor


@pytest.mark.parametrize("mixed_none", [False, True])
def test_real_agent_flow_text_schema_and_native_gspo_forward(tmp_path, monkeypatch, mixed_none):
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M1")
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "agent-flow-schema-fixture")
    data = make_agent_flow_batch()
    if mixed_none:
        data.non_tensor_batch["multi_modal_inputs"][1] = None
    snapshot = copy.deepcopy(data)
    baseline = make_native_forward_actor(monkeypatch)
    actor = make_native_forward_actor(monkeypatch)
    # CPU uses the original native padded forward (no CUDA/FlashAttention).
    # Its real MM extractor must produce the same kwargs as the omitted column.
    inputs = {**data.batch, **data.non_tensor_batch}
    _, with_placeholder = actor._forward_micro_batch(inputs, temperature=0.7)
    without_inputs = {key: value for key, value in inputs.items() if key != "multi_modal_inputs"}
    _, without_placeholder = actor._forward_micro_batch(without_inputs, temperature=0.7)
    assert torch.equal(with_placeholder, without_placeholder)
    actor.actor_module.forward_calls.clear()
    expected = baseline.update_policy(copy.deepcopy(data))
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    rng = torch.get_rng_state().clone()
    observed = actor.update_policy(data)
    assert observed == expected
    assert torch.equal(actor.actor_module.embedding.weight, baseline.actor_module.embedding.weight)
    assert torch.equal(torch.get_rng_state(), rng)
    assert actor.actor_module.training
    assert all(torch.equal(snapshot.batch[key], value) for key, value in data.batch.items())
    assert data.meta_info == snapshot.meta_info
    assert all(
        np.array_equal(snapshot.non_tensor_batch[key], value)
        for key, value in data.non_tensor_batch.items()
    )
    assert actor.actor_module.forward_calls == (
        [(True, {"use_cache": False})] * 3 + [(False, {"use_cache": False})] * 3
    )
    record = read_record(tmp_path)
    assert record["actual_policy_loss_functions"] == [
        "verl.trainer.ppo.core_algos.compute_policy_loss_gspo"
    ]
    assert record["real_step_count"] == record["global_real_step_denominator"] == 3
    assert record["response_token_count"] == record["global_response_token_denominator"] == 6
    assert record["rows"][1]["invalid_action"] is True
    assert record["rows"][1]["assigned_step_reward"] == 1.0
    assert record["rows"][1]["advantage"] == [-1.0] * 3
    assert record["rows"][0]["trajectory_uid"] == record["rows"][1]["trajectory_uid"]
    assert record["rows"][2]["trajectory_uid"] != record["rows"][1]["trajectory_uid"]
    assert [row["step_index"] for row in record["rows"]] == [0, 1, 0]
    assert all(math.isfinite(value) for row in record["rows"] for value in row["post_logp"])
    probe.validate_run_evidence(
        tmp_path, expected_updates=1, arm="M1", execution_identity="agent-flow-schema-fixture"
    )


@pytest.mark.parametrize("unsupported", ["multimodal", "importance"])
def test_real_multimodal_or_importance_inputs_still_rejected(tmp_path, monkeypatch, unsupported):
    actor = make_actor(monkeypatch, loss_mode="gspo")
    data = make_batch()
    if unsupported == "multimodal":
        data.non_tensor_batch["multi_modal_inputs"] = np.array(
            [{}, {"pixel_values": torch.ones(1, 2)}, None], dtype=object
        )
        message = "nonempty multimodal inputs"
    else:
        data.batch["rollout_is_weights"] = torch.ones_like(data.batch["advantages"])
        message = "rollout importance weights"
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    with pytest.raises(ValueError, match=message):
        actor.update_policy(data)
    assert actor.pre_calls == actor.post_calls == 0
    assert actor.actor_module.weight.abs().sum() == 0
    assert not list(tmp_path.glob("*.jsonl"))


def read_record(directory):
    files = list(directory.glob("actor-update-*.jsonl"))
    assert len(files) == 1
    records = files[0].read_text().splitlines()
    assert len(records) == 1
    return json.loads(records[0])


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("rollout_anchor", [False, True])
@pytest.mark.parametrize("loss_mode", ["vanilla", "gspo"])
def test_real_update_equivalence_and_actual_loss_advantage(
    tmp_path, monkeypatch, padding, rollout_anchor, loss_mode
):
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M1")
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "native-loss-fixture")
    baseline = make_actor(monkeypatch, rollout_anchor=rollout_anchor, loss_mode=loss_mode)
    wrapped = make_actor(monkeypatch, rollout_anchor=rollout_anchor, loss_mode=loss_mode)
    original_getter = agent_actor.get_policy_loss_fn
    assert probe.patch_actor_class(type(wrapped), tmp_path, agent_actor)
    assert not probe.patch_actor_class(type(wrapped), tmp_path, agent_actor)
    expected = baseline.update_policy(make_batch(padding=padding))
    data = make_batch(padding=padding)
    tensors = {k: v.clone() for k, v in data.batch.items()}
    metadata = dict(data.meta_info)
    rng = torch.get_rng_state().clone()
    observed = wrapped.update_policy(data)
    assert observed == expected
    assert torch.equal(baseline.actor_module.weight, wrapped.actor_module.weight)
    assert torch.equal(torch.get_rng_state(), rng)
    assert agent_actor.get_policy_loss_fn is original_getter
    assert "_forward_micro_batch" not in wrapped.__dict__
    assert all(torch.equal(tensors[k], v) for k, v in data.batch.items())
    assert data.meta_info == metadata
    assert wrapped.actor_module.training
    assert wrapped.pre_calls == 3 + padding
    assert wrapped.post_calls == 3
    record = read_record(tmp_path)
    assert record["real_step_count"] == 3
    assert record["padding_row_count"] == int(padding)
    assert record["response_token_count"] == 6
    assert record["global_real_step_denominator"] == 3
    assert record["policy_loss_mode"] == loss_mode
    if loss_mode == "gspo":
        assert record["actual_policy_loss_functions"] == [
            "verl.trainer.ppo.core_algos.compute_policy_loss_gspo"
        ]
        assert record["configured_loss_aggregation"] == "token-mean"
        assert record["effective_policy_loss_aggregation"] == "seq-mean-token-mean"
        if not rollout_anchor:
            # Native outer loop adds each response's mean loss / global_steps
            # with scale=1. Unequal response lengths distinguish token weighting.
            assert observed["actor/pg_loss"] == pytest.approx((-2 + 1 - 0.5) / 3)
    elif not rollout_anchor:
        assert observed["actor/pg_loss"] == pytest.approx((-2 * 2 + 1 * 3 - 0.5) / 6)
    invalid = record["rows"][1]
    assert invalid["trajectory_uid"] == "a" and invalid["step_index"] == 1
    assert invalid["invalid_action"] is True
    assert invalid["assigned_step_reward"] == 1.0  # Positive direct reward is not A.
    assert invalid["advantage"] == [-1.0] * 3
    assert invalid["response_token_indices"] == [0, 1, 2]
    if rollout_anchor and loss_mode == "vanilla":
        # The deliberately stale -3 anchor puts all toy tokens beyond vanilla
        # positive clipping / negative dual clipping: unchanged, not forced drift.
        assert all(row["delta_logp_mean"] == 0 for row in record["rows"])
        assert observed["actor/grad_norm"] == [0.0]
    elif not rollout_anchor:
        assert invalid["delta_logp_mean"] < 0
        assert record["rows"][0]["delta_logp_mean"] > 0
    else:
        # GSPO has response clipping but no vanilla negative dual clip.
        assert invalid["delta_logp_mean"] < 0
        assert record["rows"][0]["delta_logp_mean"] == 0
    assert record["groups"]["invalid"]["negative_advantage_tokens"] == 3
    assert record["groups"]["legal"]["positive_advantage_tokens"] == 3
    for row in record["rows"]:
        assert row["old_logp_used_by_loss"] == (
            row["driver_old_logp"] if rollout_anchor else row["pre_logp"]
        )
        assert row["driver_old_minus_pre_abs_max"] > 1
        assert row["advantage_times_delta_logp_sum"] >= 0
        if loss_mode == "gspo":
            log_ratio = sum(
                p - old
                for p, old in zip(row["post_logp"], row["old_logp_used_by_loss"], strict=True)
            ) / len(row["advantage"])
            assert row["post_ratio_to_loss_anchor"] == pytest.approx(
                [math.exp(min(log_ratio, 10))] * len(row["advantage"])
            )
    evidence = probe.validate_run_evidence(
        tmp_path, expected_updates=1, arm="M1", execution_identity="native-loss-fixture"
    )
    assert evidence["observed_updates"] == 1


def test_gspo_ratio_uses_response_mean_not_each_token(tmp_path, monkeypatch):
    actor = make_actor(monkeypatch, loss_mode="gspo")
    data = make_batch()
    # Unequal token frequencies yield distinct post-update conditional drifts.
    data.batch["responses"][1] = torch.tensor([0, 1, 1])
    data.batch["input_ids"][1, -3:] = data.batch["responses"][1]
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    actor.update_policy(data)
    row = read_record(tmp_path)["rows"][1]
    assert max(row["delta_logp"]) - min(row["delta_logp"]) > 1e-5
    token_ratios = [math.exp(delta) for delta in row["delta_logp"]]
    expected = math.prod(token_ratios) ** (1 / 3)
    assert row["post_ratio_to_loss_anchor"] == pytest.approx([expected] * 3)
    assert row["post_ratio_to_loss_anchor"] != pytest.approx(token_ratios)
    assert row["loss_ratio_scope"] == "per_response_masked_mean_log_ratio"


def test_post_failure_preserves_rng_mode_and_original_update(tmp_path, monkeypatch):
    actor = make_actor(monkeypatch, fail_post=True)
    original_getter = agent_actor.get_policy_loss_fn
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="fixture post failure"):
        actor.update_policy(make_batch())
    assert torch.equal(torch.get_rng_state(), rng)
    assert actor.actor_module.training
    assert agent_actor.get_policy_loss_fn is original_getter
    assert "_forward_micro_batch" not in actor.__dict__
    assert actor.actor_module.weight.abs().sum() > 0
    assert not list(tmp_path.glob("*.jsonl"))
    assert not probe._UPDATE_LOCK.locked()


@pytest.mark.parametrize("change", ["epochs", "dynamic", "minibatches", "identity", "mask"])
def test_unsupported_or_ambiguous_inputs_rejected_before_update(tmp_path, monkeypatch, change):
    actor = make_actor(monkeypatch)
    data = make_batch()
    if change == "epochs":
        actor.config.ppo_epochs = 2
    elif change == "dynamic":
        actor.config.use_dynamic_bsz = True
    elif change == "minibatches":
        data.batch["mini_batch_id"][-1] = 1
    elif change == "identity":
        data.non_tensor_batch["trading_invalid_action"][1] = False
    else:
        data.batch["response_mask"][1] = 0
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    with pytest.raises(ValueError, match="actor probe"):
        actor.update_policy(data)
    assert actor.actor_module.weight.abs().sum() == 0
    assert not list(tmp_path.glob("*.jsonl"))


def test_disabled_install_and_explicit_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(probe.DIRECTORY_ENV, raising=False)
    assert not probe.install_from_environment()
    monkeypatch.setenv(probe.DIRECTORY_ENV, "relative")
    with pytest.raises(ValueError, match="absolute"):
        probe.install_from_environment()
    actor = make_actor(monkeypatch)
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    with pytest.raises(RuntimeError, match="cannot change"):
        probe.patch_actor_class(type(actor), tmp_path / "changed", agent_actor)


@pytest.mark.parametrize(
    "damage", [None, "identity", "tokens", "summary", "partial", "duplicate", "loss_dispatch"]
)
def test_final_evidence_checks_actual_records(tmp_path, monkeypatch, damage):
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M1")
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "fixture-id")
    actor = make_actor(monkeypatch)
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    actor.update_policy(make_batch())
    path = next(tmp_path.glob("actor-update-*.jsonl"))
    record = json.loads(path.read_text())
    if damage == "identity":
        record["execution_identity_sha256"] = "foreign"
    elif damage == "tokens":
        record["rows"][0]["advantage"].pop()
    elif damage == "summary":
        record["groups"]["invalid"]["negative_advantage_tokens"] = 0
    elif damage == "loss_dispatch":
        record["actual_policy_loss_functions"] = ["unobserved.loss"]
    payload = json.dumps(record) + "\n"
    if damage == "partial":
        payload = payload[:-1]
    elif damage == "duplicate":
        payload *= 2
    path.write_text(payload)
    arguments = dict(expected_updates=1, arm="M1", execution_identity="fixture-id")
    if damage:
        with pytest.raises(ValueError, match="actor probe"):
            probe.validate_run_evidence(tmp_path, **arguments)
    else:
        summary = probe.validate_run_evidence(tmp_path, **arguments)
        assert summary["status"] == "COMPLETE"
        assert summary["observed_updates"] == 1 and summary["policy_steps"] == [1]
        assert summary["real_step_count"] == 3 and summary["response_token_count"] == 6
        assert summary["invalid_step_count"] == 1
        assert summary["invalid_positive_advantage_tokens"] == 0


def test_absent_probe_evidence_never_means_success(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        probe.validate_run_evidence(tmp_path, expected_updates=8, arm="M1", execution_identity="id")


def test_zero_update_baseline_accepts_no_actor_records(tmp_path):
    summary = probe.validate_run_evidence(
        tmp_path / "not-created", expected_updates=0, arm="M1", execution_identity="id"
    )
    assert summary["status"] == "COMPLETE"
    assert summary["observed_updates"] == summary["real_step_count"] == 0
    assert summary["policy_steps"] == [] and summary["source_sha256"] == {}


def test_zero_update_baseline_rejects_any_actor_record(tmp_path, monkeypatch):
    actor = make_actor(monkeypatch)
    probe.patch_actor_class(type(actor), tmp_path, agent_actor)
    actor.update_policy(make_batch())
    with pytest.raises(ValueError, match="expected 0 updates, found 1"):
        probe.validate_run_evidence(tmp_path, expected_updates=0, arm="M1", execution_identity="id")
