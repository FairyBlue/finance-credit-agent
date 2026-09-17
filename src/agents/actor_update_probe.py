"""Opt-in actor loss-input and same-response post-update measurement.

Capture the arguments of the real vanilla PPO or GSPO loss, not a second pre forward.
Only the existing single-minibatch/epoch, single-device path is supported. The
ordinary optimizer/loss runs unchanged; post measurement preserves RNG and mode.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
import sys
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any

from src.agents.critic_fit_probe import _dropout_evidence, _get

DIRECTORY_ENV = "FINANCE_ACTOR_UPDATE_PROBE_DIR"
TARGET_MODULE = "agent_r1.workers.actor.dp_actor"
_MARKER = "_finance_actor_update_probe_directory"
_UPDATE_LOCK = threading.Lock()
_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids", "responses")
_IDENTITY_KEYS = {
    "trajectory_uid": "trajectory_uids",
    "step_index": "step_indices",
    "episode_id": "finance_screening_episode_id",
    "policy_step": "finance_screening_policy_step",
    "rollout_id": "finance_screening_rollout_id",
    "rollout_n": "finance_screening_rollout_n",
    "group_uid": "finance_screening_group_uid",
    "action": "trading_action",
    "invalid_action": "trading_invalid_action",
    "decision_session_index": "trading_decision_session_index",
    "result_session_index": "trading_result_session_index",
    "terminal_Y": "finance_screening_terminal_score",
}
_COMMON_REWARD_KEYS = {
    "financial_step_reward": "finance_common_financial_step_reward",
    "invalid_penalty": "finance_common_invalid_penalty",
    "training_step_reward": "finance_common_training_step_reward",
}


def _cpu(tensor: Any) -> Any:
    return tensor.detach().cpu().clone()


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _digest(tensor: Any) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def policy_loss_contract(config: Any) -> dict[str, str]:
    """Describe supported actual loss semantics without importing a model or torch."""
    mode = _get(config.policy_loss, "loss_mode", "vanilla")
    if mode not in {"vanilla", "gspo"}:
        raise ValueError("actor probe supports vanilla PPO or GSPO without rollout correction")
    configured = _get(config, "loss_agg_mode")
    if configured != "token-mean":
        raise ValueError("actor probe requires the registered configured token-mean loss mode")
    return {
        "policy_loss_mode": mode,
        "configured_loss_aggregation": configured,
        "effective_policy_loss_aggregation": "seq-mean-token-mean"
        if mode == "gspo"
        else configured,
        "loss_ratio_scope": "per_response_masked_mean_log_ratio" if mode == "gspo" else "per_token",
        "loss_ratio_numerical_clamp": "max_log_ratio_10"
        if mode == "gspo"
        else "log_ratio_minus20_plus20",
    }


def _prepare(instance: Any, data: Any) -> dict[str, Any]:
    import torch

    config = instance.config
    if (
        _get(config, "ppo_epochs") != 1
        or _get(config, "use_dynamic_bsz", False)
        or _get(config, "ppo_micro_batch_size_per_gpu") != 1
        or instance.ulysses_sequence_parallel_size != 1
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1)
    ):
        raise ValueError("actor probe requires one epoch, fixed microbatch=1 and one device")
    mini_ids = data.batch.get("mini_batch_id")
    if mini_ids is None or mini_ids.unique().numel() != 1:
        raise ValueError("actor probe requires exactly one planned minibatch")
    loss_contract = policy_loss_contract(config)
    if "rollout_is_weights" in data.batch:
        raise ValueError("actor probe does not support rollout importance weights")
    multi_modal = data.non_tensor_batch.get("multi_modal_inputs")
    if multi_modal is not None:
        if len(multi_modal) != len(data):
            raise ValueError("actor probe multimodal placeholder column does not align with rows")
        # AgentFlow emits {} for processor=None and retains that object column.
        # Native verl extract_multi_modal_inputs maps only None/empty dict rows
        # to {}, identical to omitting the column in our post-update forward.
        # Keep the original batch/forward untouched; actual MM stays unsupported.
        if any(
            value is not None and not (isinstance(value, dict) and not value)
            for value in multi_modal
        ):
            raise ValueError("actor probe does not support nonempty multimodal inputs")
    dropout = _dropout_evidence(instance.actor_module)
    if dropout["any_nonzero"]:
        raise ValueError("actor probe requires zero model dropout for matched pre/post mode")
    dropout.update(pre_mode="train (actual loss forward)", post_mode="train")
    mask = _cpu(data.batch["response_mask"])
    sample = _cpu(data.batch["sample_mask"])
    if mask.ndim != 2 or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("actor probe requires a binary response mask")
    if sample.shape != (len(data),) or not bool(((sample == 0) | (sample == 1)).all()):
        raise ValueError("actor probe requires a binary sample mask")
    if not torch.equal(mask.bool().any(dim=1), sample.bool()):
        raise ValueError("actor probe response mask disagrees with real/padding rows")
    global_tokens = data.batch["mini_batch_global_response_token_num"].unique()
    if global_tokens.numel() != 1 or int(global_tokens.item()) != int(mask.sum()):
        raise ValueError("actor probe global token denominator differs from actual loss mask")
    global_steps = data.batch["mini_batch_global_size"].unique()
    if global_steps.numel() != 1 or int(global_steps.item()) != int(sample.sum()):
        raise ValueError("actor probe global step denominator differs from real loss rows")
    if any(key not in data.non_tensor_batch for key in _IDENTITY_KEYS.values()):
        raise ValueError("actor probe missing trajectory/step/invalid identity metadata")
    if any(len(data.non_tensor_batch[key]) != len(data) for key in _IDENTITY_KEYS.values()):
        raise ValueError("actor probe identity columns do not align with rows")
    common_reward_columns = set(_COMMON_REWARD_KEYS.values()) & data.non_tensor_batch.keys()
    if common_reward_columns and (
        common_reward_columns != set(_COMMON_REWARD_KEYS.values())
        or any(len(data.non_tensor_batch[key]) != len(data) for key in common_reward_columns)
        or "token_level_rewards" not in data.batch
    ):
        raise ValueError("actor probe common reward components are missing or misaligned")
    rows = []
    for i in range(len(data)):
        row = {
            key: _scalar(data.non_tensor_batch[column][i]) for key, column in _IDENTITY_KEYS.items()
        }
        row["batch_row_index"] = i
        row["sample_mask"] = bool(sample[i])
        if row["sample_mask"] and row["invalid_action"] != (row["action"] == "invalid"):
            raise ValueError("actor probe invalid identity disagrees with action")
        if "token_level_rewards" in data.batch:
            row["assigned_step_reward"] = float(
                (data.batch["token_level_rewards"][i] * data.batch["response_mask"][i]).sum()
            )
        if common_reward_columns and row["sample_mask"]:
            components = {
                key: _scalar(data.non_tensor_batch[column][i])
                for key, column in _COMMON_REWARD_KEYS.items()
            }
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in components.values()
            ):
                raise ValueError("actor probe common reward components must be finite")
            expected_penalty = 0.05 if row["invalid_action"] else 0.0
            if (
                components["invalid_penalty"] != expected_penalty
                or not math.isclose(
                    components["training_step_reward"],
                    components["financial_step_reward"] - expected_penalty,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    components["training_step_reward"],
                    row["assigned_step_reward"],
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                )
            ):
                raise ValueError("actor probe common reward does not match actual training tensors")
            row.update(components)
        rows.append(row)
    real = [row for row in rows if row["sample_mask"]]
    if not real or len({row["policy_step"] for row in real}) != 1:
        raise ValueError("actor probe requires real rows from one policy step")
    identities = [(row["trajectory_uid"], row["step_index"]) for row in real]
    if len(set(identities)) != len(identities):
        raise ValueError("actor probe real trajectory/step identities must be unique")
    return {
        "rows": rows,
        "response_mask": mask,
        "policy_step": real[0]["policy_step"],
        "dropout": dropout,
        "padding_row_count": len(rows) - len(real),
        "global_response_token_denominator": int(global_tokens.item()),
        "global_real_step_denominator": int(global_steps.item()),
        **loss_contract,
    }


class _Capture:
    def __init__(self, instance: Any, data: Any, metadata: dict[str, Any]) -> None:
        self.instance, self.data, self.metadata = instance, data, metadata
        self.pending: Any = None
        self.captured: list[dict[str, Any]] = []
        self.cursor = 0
        self.loss_functions: set[str] = set()

    def forward(self, original: Any, model_inputs: Any, **kwargs: Any) -> Any:
        if self.pending is not None:
            raise RuntimeError("actor probe saw another forward before the matching loss")
        output = original(model_inputs, **kwargs)
        # Keep the actual tensor identity only until its immediately following loss.
        self.pending = (model_inputs, kwargs, output[1])
        return output

    def loss(self, arguments: dict[str, Any]) -> None:
        import torch

        if self.pending is None:
            raise RuntimeError("actor probe loss has no matching training forward")
        inputs, forward_kwargs, forward_logp = self.pending
        if arguments["log_prob"] is not forward_logp:
            raise RuntimeError("actor probe loss did not use the captured training forward")
        index = self.cursor
        if inputs["responses"].shape[0] != 1 or index >= len(self.data):
            raise ValueError("actor probe training microbatch order or size changed")
        snapshots = {key: _cpu(inputs[key]) for key in _INPUT_KEYS}
        for key in _INPUT_KEYS:
            if not torch.equal(snapshots[key], _cpu(self.data.batch[key][index : index + 1])):
                raise ValueError(f"actor probe row identity mismatch for {key}")
        for loss_key, batch_key in (
            ("response_mask", "response_mask"),
            ("advantages", "advantages"),
        ):
            observed = _cpu(arguments[loss_key])
            expected = _cpu(self.data.batch[batch_key][index : index + 1])
            if not torch.equal(observed, expected):
                raise ValueError(f"actor probe actual loss {loss_key} differs from row")
        mask = _cpu(arguments["response_mask"])[0].bool()
        record = {
            "identity": self.metadata["rows"][index],
            "inputs": snapshots,
            "device": inputs["input_ids"].device,
            "forward_kwargs": dict(forward_kwargs),
            "mask": mask,
            "response_token_indices": torch.nonzero(mask).flatten().tolist(),
            "response_token_ids": snapshots["responses"][0, mask].tolist(),
        }
        for target, tensor in (
            ("advantage", arguments["advantages"]),
            ("old_logp_used_by_loss", arguments["old_log_prob"]),
            ("pre_logp", arguments["log_prob"]),
            ("driver_old_logp", inputs["old_log_probs"]),
        ):
            selected = _cpu(tensor)[0, mask].double()
            if not bool(torch.isfinite(selected).all()):
                raise ValueError(f"actor probe nonfinite real-token {target}")
            record[target] = selected
        self.captured.append(record)
        self.cursor += 1
        self.pending = None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return math.fsum(values) / len(values) if values else None

    advantages = [x for row in rows for x in row["advantage"]]
    deltas = [x for row in rows for x in row["delta_logp"]]
    ratios = [x for row in rows for x in row["post_ratio_to_loss_anchor"]]
    signed = [a * d for a, d in zip(advantages, deltas, strict=True)]
    return {
        "step_count": len(rows),
        "response_tokens": len(advantages),
        "positive_advantage_tokens": sum(a > 0 for a in advantages),
        "negative_advantage_tokens": sum(a < 0 for a in advantages),
        "zero_advantage_tokens": sum(a == 0 for a in advantages),
        "positive_advantage_steps": sum(all(a > 0 for a in row["advantage"]) for row in rows),
        "negative_advantage_steps": sum(all(a < 0 for a in row["advantage"]) for row in rows),
        "mean_advantage_token_weighted": mean(advantages),
        "mean_delta_logp_token_weighted": mean(deltas),
        "mean_advantage_times_delta_logp_token_weighted": mean(signed),
        "mean_advantage_times_delta_logp_step_weighted": mean(
            [row["advantage_times_delta_logp_sum"] / len(row["advantage"]) for row in rows]
        ),
        "advantage_aligned_delta_token_fraction": mean([float(x > 0) for x in signed]),
        "mean_post_ratio_to_loss_anchor": mean(ratios),
        "post_ratio_outside_clip_fraction": mean(
            [
                float(x < row["clip_ratio_lower"] or x > row["clip_ratio_upper"])
                for row in rows
                for x in row["post_ratio_to_loss_anchor"]
            ]
        ),
    }


def _measure(instance: Any, capture: _Capture, original_forward: Any) -> list[dict[str, Any]]:
    import torch

    module = instance.actor_module
    modes = [(child, child.training) for child in module.modules()]
    devices = sorted({p.device.index for p in module.parameters() if p.device.type == "cuda"})
    config = instance.config
    mode = capture.metadata["policy_loss_mode"]
    low = _get(config, "clip_ratio_low")
    high = _get(config, "clip_ratio_high")
    clip_low = 1 - float(_get(config, "clip_ratio") if low is None else low)
    clip_high = 1 + float(_get(config, "clip_ratio") if high is None else high)
    rows = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            # Same mode as the real training pre forward, with dropout verified zero.
            module.train()
            for record in capture.captured:
                if not record["identity"]["sample_mask"]:
                    continue
                inputs = {
                    key: value.to(record["device"]) for key, value in record["inputs"].items()
                }
                post_kwargs = {**record["forward_kwargs"], "calculate_entropy": False}
                _, post = original_forward(inputs, **post_kwargs)
                post = _cpu(post)[0, record["mask"]].double()
                if not bool(torch.isfinite(post).all()):
                    raise ValueError("actor probe nonfinite post log probabilities")
                pre, old, advantage = (
                    record[key] for key in ("pre_logp", "old_logp_used_by_loss", "advantage")
                )
                delta = post - pre
                # GSPO uses one length-normalized ratio per response. Its
                # straight-through token term is numerically zero in a no-grad
                # readout; broadcasting the scalar reproduces the native value.
                if mode == "gspo":
                    log_ratio = (post - old).mean()
                    ratio = torch.exp(torch.clamp(log_ratio, max=10)).expand_as(post)
                    clamp_count = int(log_ratio > 10) * post.numel()
                else:
                    ratio = torch.exp(torch.clamp(post - old, -20, 20))
                    clamp_count = int(((post - old).abs() > 20).sum())
                row = {
                    **record["identity"],
                    "response_token_indices": record["response_token_indices"],
                    "response_token_ids": record["response_token_ids"],
                    "response_width": int(record["mask"].numel()),
                    "loss_mask_encoding": "one at response_token_indices, zero elsewhere",
                    **{
                        key: record[key].tolist()
                        for key in (
                            "advantage",
                            "old_logp_used_by_loss",
                            "pre_logp",
                            "driver_old_logp",
                        )
                    },
                    "post_logp": post.tolist(),
                    "delta_logp": delta.tolist(),
                    "post_ratio_to_loss_anchor": ratio.tolist(),
                    "ratio_numerical_clamp_count": clamp_count,
                    "policy_loss_mode": mode,
                    "loss_ratio_scope": capture.metadata["loss_ratio_scope"],
                    "advantage_times_delta_logp_sum": float((advantage * delta).sum()),
                    "delta_logp_sum": float(delta.sum()),
                    "delta_logp_mean": float(delta.mean()),
                    "old_minus_pre_abs_max": float((old - pre).abs().max()),
                    "driver_old_minus_pre_abs_max": float(
                        (record["driver_old_logp"] - pre).abs().max()
                    ),
                    "input_sha256": {
                        key: _digest(value) for key, value in record["inputs"].items()
                    },
                    "temperature": record["forward_kwargs"]["temperature"],
                    "clip_ratio_lower": clip_low,
                    "clip_ratio_upper": clip_high,
                }
                rows.append(row)
                del inputs, post
    finally:
        # Preserve every child flag, not just the root's mode.
        for child, training in modes:
            child.training = training
    return rows


def patch_actor_class(actor_class: type[Any], directory: Path, policy_module: Any) -> bool:
    """Wrap only our actor instance; temporary loss interception is thread-scoped."""
    if not directory.is_absolute():
        raise ValueError("actor probe directory must be absolute")
    directory = directory.resolve()
    existing = actor_class.__dict__.get(_MARKER)
    if existing is not None:
        if existing != str(directory):
            raise RuntimeError("actor probe directory cannot change after installation")
        return False
    original = actor_class.update_policy
    if tuple(inspect.signature(original).parameters) not in {("self", "data"), ("args", "kwargs")}:
        raise RuntimeError("actor probe pinned update_policy signature changed")

    @wraps(original)
    def update(instance: Any, data: Any) -> Any:
        metadata = _prepare(instance, data)
        if not _UPDATE_LOCK.acquire(blocking=False):
            raise RuntimeError(
                "actor probe does not support concurrent actor updates in one process"
            )
        capture = _Capture(instance, data, metadata)
        original_forward = instance._forward_micro_batch
        prior_instance_forward = instance.__dict__.get("_forward_micro_batch")
        original_getter = policy_module.get_policy_loss_fn
        thread_id = threading.get_ident()

        def get_loss(name: str) -> Any:
            loss_fn = original_getter(name)
            if threading.get_ident() != thread_id:
                return loss_fn
            if name != metadata["policy_loss_mode"]:
                raise ValueError("actor probe loss dispatch differs from prepared contract")
            capture.loss_functions.add(f"{loss_fn.__module__}.{loss_fn.__name__}")

            @wraps(loss_fn)
            def loss(*args: Any, **kwargs: Any) -> Any:
                arguments = inspect.signature(loss_fn).bind(*args, **kwargs).arguments
                capture.loss(arguments)
                return loss_fn(*args, **kwargs)

            return loss

        try:
            instance._forward_micro_batch = lambda inputs, **kw: capture.forward(
                original_forward, inputs, **kw
            )
            policy_module.get_policy_loss_fn = get_loss
            metrics = original(instance, data)
            if capture.pending is not None or capture.cursor != len(data):
                raise RuntimeError("actor probe did not observe every training loss row once")
            norms = metrics.get("actor/grad_norm", [])
            if not isinstance(norms, (list, tuple)) or len(norms) != 1:
                raise RuntimeError("actor probe expected exactly one ordinary optimizer update")
            started = time.perf_counter()
            rows = _measure(instance, capture, original_forward)
            elapsed = time.perf_counter() - started
        finally:
            policy_module.get_policy_loss_fn = original_getter
            if prior_instance_forward is None:
                del instance._forward_micro_batch
            else:
                instance._forward_micro_batch = prior_instance_forward
            _UPDATE_LOCK.release()
        counter = getattr(instance, "_finance_actor_probe_update_count", 0) + 1
        instance._finance_actor_probe_update_count = counter
        record = {
            "schema_version": 1,
            "status": "COMPLETE",
            "phase": "actor_same_batch_pre_post",
            "pid": os.getpid(),
            "arm": os.environ.get("FINANCE_SCREENING_ARM"),
            "execution_identity_sha256": os.environ.get("FINANCE_SCREENING_IDENTITY_SHA256"),
            "policy_step": metadata["policy_step"],
            "update_call_index": counter,
            "real_step_count": len(rows),
            "padding_row_count": metadata["padding_row_count"],
            "response_token_count": sum(len(row["advantage"]) for row in rows),
            "global_response_token_denominator": metadata["global_response_token_denominator"],
            "global_real_step_denominator": metadata["global_real_step_denominator"],
            **policy_loss_contract(instance.config),
            "actual_policy_loss_functions": sorted(capture.loss_functions),
            "optimizer_step_count": 1,
            "ppo_epochs": 1,
            "planned_minibatch_count": 1,
            "outer_microbatch_loss_scale": 1.0,
            "actor_learning_rates": [
                float(group["lr"]) for group in instance.actor_optimizer.param_groups
            ],
            "optimizer_grad_norms_before_clip": [float(x) for x in norms],
            "original_training_pg_loss": float(metrics["actor/pg_loss"]),
            "extra_post_forward_micro_batches": len(rows),
            "extra_post_forward_seconds": elapsed,
            "dropout": metadata["dropout"],
            "pre_source": (
                "actual arguments passed to the original dispatched policy loss; "
                "no extra pre forward"
            ),
            "post_source": (
                "same input_ids/attention/positions/response, after the sole optimizer step; "
                "train mode, no_grad"
            ),
            "rng_preservation": "CPU and model CUDA states restored around post forward only",
            "aggregation": (
                "response tokens selected by the actual loss mask; padding excluded, "
                "invalid requests retained; token-weighted summaries are descriptive, "
                "not GSPO policy-loss contribution weights"
            ),
            "policy_loss_readout_semantics": (
                "original_training_pg_loss is the unchanged native returned metric; "
                "GSPO forces response/step means despite configured token-mean. "
                "The required planned-minibatch outer scale is 1; reference KL remains "
                "separately aggregated by configured token-mean. "
                "Post ratios follow actual loss_mode; outside-clip fraction measures ratio range, "
                "not native pg_clipfrac or actual clipped-gradient fraction. "
                "A times delta_logp is a directional readout, not a finite policy-loss change "
                "or a contribution including clipping, KL and optimizer effects."
            ),
            "interpretation": (
                "delta logp is teacher-forced conditional token drift, not generated action "
                "probability, expected return or proof of causal reinforcement"
            ),
            "advantage_source": (
                "actual loss advantages after GAE/M2 residual/whitening, "
                "not immediate assigned reward"
            ),
            "groups": {
                name: _summary(selected)
                for name, selected in (
                    ("all_real", rows),
                    ("legal", [row for row in rows if not row["invalid_action"]]),
                    ("invalid", [row for row in rows if row["invalid_action"]]),
                )
            },
            "rows": rows,
        }
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        with (directory / f"actor-update-{os.getpid()}.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return metrics

    actor_class.update_policy = update
    setattr(actor_class, _MARKER, str(directory))
    return True


def validate_run_evidence(
    directory: Path,
    *,
    expected_updates: int,
    arm: str,
    execution_identity: str,
) -> dict[str, Any]:
    """Fail closed on absent/partial/foreign probe records; never infer updates."""
    if type(expected_updates) is not int or expected_updates < 0:
        raise ValueError("actor probe expected_updates must be nonnegative")
    files = sorted(Path(directory).glob("actor-update-*.jsonl"))
    records, hashes = [], {}
    for path in files:
        payload = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(payload).hexdigest()
        if not payload.endswith(b"\n"):
            raise ValueError("actor probe evidence ends with a partial record")
        for line in payload.decode("utf-8").splitlines():
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError("actor probe evidence contains invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError("actor probe evidence record is not an object")
            records.append(record)
    if len(records) != expected_updates:
        raise ValueError(f"actor probe expected {expected_updates} updates, found {len(records)}")
    if expected_updates == 0:
        return {
            "status": "COMPLETE",
            "schema_version": 1,
            "arm": arm,
            "execution_identity_sha256": execution_identity,
            "observed_updates": 0,
            "expected_updates": 0,
            "policy_steps": [],
            "real_step_count": 0,
            "response_token_count": 0,
            "invalid_step_count": 0,
            "invalid_positive_advantage_tokens": 0,
            "source_sha256": hashes,
        }

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"actor probe evidence {message}")

    def finite(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    records.sort(key=lambda row: row.get("policy_step", -1))
    require(
        [row.get("policy_step") for row in records] == list(range(1, expected_updates + 1)),
        "policy steps are missing or duplicated",
    )
    require(len({row.get("pid") for row in records}) == 1, "spans multiple worker processes")
    for update_index, record in enumerate(records, 1):
        require(
            all(
                record.get(key) == value
                for key, value in {
                    "schema_version": 1,
                    "status": "COMPLETE",
                    "phase": "actor_same_batch_pre_post",
                    "arm": arm,
                    "execution_identity_sha256": execution_identity,
                    "update_call_index": update_index,
                    "optimizer_step_count": 1,
                    "ppo_epochs": 1,
                    "planned_minibatch_count": 1,
                }.items()
            ),
            "identity/status/update contract mismatch",
        )
        rows = record.get("rows", [])
        require(
            bool(rows) and len(rows) == record.get("real_step_count"), "real step count differs"
        )
        mode = record.get("policy_loss_mode")
        require(mode in {"vanilla", "gspo"}, "actual policy loss mode is missing or unsupported")
        scope = "per_response_masked_mean_log_ratio" if mode == "gspo" else "per_token"
        aggregation = "seq-mean-token-mean" if mode == "gspo" else "token-mean"
        callable_module = "verl" if mode == "gspo" else "agent_r1"
        require(
            record.get("configured_loss_aggregation") == "token-mean"
            and record.get("effective_policy_loss_aggregation") == aggregation
            and record.get("loss_ratio_scope") == scope
            and record.get("outer_microbatch_loss_scale") == 1.0
            and record.get("actual_policy_loss_functions")
            == [f"{callable_module}.trainer.ppo.core_algos.compute_policy_loss_{mode}"],
            "actual loss dispatch/aggregation contract differs",
        )
        require(
            record.get("global_real_step_denominator") == len(rows),
            "global real step denominator differs",
        )
        require(
            len(rows) == record.get("extra_post_forward_micro_batches"),
            "post forward count differs",
        )
        require(
            type(record.get("padding_row_count")) is int and record["padding_row_count"] >= 0,
            "invalid padding count",
        )
        identities, tokens = set(), 0
        for row in rows:
            require(
                row.get("policy_loss_mode") == mode and row.get("loss_ratio_scope") == scope,
                "row ratio scope differs from dispatched loss",
            )
            require(
                row.get("sample_mask") is True and row.get("policy_step") == update_index,
                "row is padding or belongs to a different policy step",
            )
            require(
                row.get("invalid_action") == (row.get("action") == "invalid"),
                "row illegal flag differs",
            )
            identity = (row.get("trajectory_uid"), row.get("step_index"))
            require(identity not in identities, "duplicate trajectory/step identity")
            identities.add(identity)
            indices = row.get("response_token_indices", [])
            width = row.get("response_width")
            require(type(width) is int and width > 0 and bool(indices), "missing actual loss mask")
            require(
                all(type(i) is int and 0 <= i < width for i in indices)
                and indices == sorted(set(indices)),
                "invalid actual token mask indices",
            )
            token_ids = row.get("response_token_ids", [])
            require(
                len(token_ids) == len(indices)
                and all(type(i) is int and i >= 0 for i in token_ids),
                "response tokens differ",
            )
            for key in (
                "advantage",
                "old_logp_used_by_loss",
                "pre_logp",
                "driver_old_logp",
                "post_logp",
                "delta_logp",
                "post_ratio_to_loss_anchor",
            ):
                values = row.get(key, [])
                require(
                    len(values) == len(indices) and all(finite(value) for value in values),
                    f"missing/nonfinite real-token {key}",
                )
            tokens += len(indices)
        require(
            tokens
            == record.get("response_token_count")
            == record.get("global_response_token_denominator"),
            "token denominator differs",
        )
        groups = record.get("groups", {})
        for name, selected in (
            ("all_real", rows),
            ("legal", [row for row in rows if not row["invalid_action"]]),
            ("invalid", [row for row in rows if row["invalid_action"]]),
        ):
            group = groups.get(name, {})
            count = sum(len(row["advantage"]) for row in selected)
            require(
                group.get("step_count") == len(selected) and group.get("response_tokens") == count,
                f"{name} step/token count differs",
            )
            for sign, predicate in (
                ("positive", lambda a: a > 0),
                ("negative", lambda a: a < 0),
                ("zero", lambda a: a == 0),
            ):
                expected = sum(predicate(a) for row in selected for a in row["advantage"])
                require(
                    group.get(f"{sign}_advantage_tokens") == expected,
                    f"{name} actual advantage sign counts differ",
                )
            for key in (
                "mean_advantage_token_weighted",
                "mean_delta_logp_token_weighted",
                "mean_advantage_times_delta_logp_token_weighted",
                "advantage_aligned_delta_token_fraction",
                "mean_post_ratio_to_loss_anchor",
                "post_ratio_outside_clip_fraction",
            ):
                require(
                    finite(group.get(key)) if selected else group.get(key) is None,
                    f"{name} {key} missing/nonfinite",
                )
    return {
        "status": "COMPLETE",
        "schema_version": 1,
        "arm": arm,
        "execution_identity_sha256": execution_identity,
        "observed_updates": len(records),
        "expected_updates": expected_updates,
        "policy_steps": [record["policy_step"] for record in records],
        "real_step_count": sum(record["real_step_count"] for record in records),
        "response_token_count": sum(record["response_token_count"] for record in records),
        "invalid_step_count": sum(record["groups"]["invalid"]["step_count"] for record in records),
        "invalid_positive_advantage_tokens": sum(
            record["groups"]["invalid"]["positive_advantage_tokens"] for record in records
        ),
        "source_sha256": hashes,
    }


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: Any, finder: Any) -> None:
        self.wrapped, self.finder = wrapped, finder

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self.wrapped.exec_module(module)
        patch_actor_class(module.DataParallelPPOActor, self.finder.directory, module)
        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            raise ImportError(f"cannot install actor update probe for {fullname}")
        spec.loader = _Loader(spec.loader, self)
        return spec


def install_from_environment() -> bool:
    value = os.environ.get(DIRECTORY_ENV)
    if not value:
        return False
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        raise ValueError(f"{DIRECTORY_ENV} must be an absolute path")
    directory = directory.resolve()
    loaded = sys.modules.get(TARGET_MODULE)
    if loaded is not None:
        return patch_actor_class(loaded.DataParallelPPOActor, directory, loaded)
    for finder in sys.meta_path:
        if isinstance(finder, _Finder):
            if finder.directory != directory:
                raise RuntimeError("actor probe directory cannot change after installation")
            return False
    sys.meta_path.insert(0, _Finder(directory))
    return True
