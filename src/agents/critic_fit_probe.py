"""Opt-in same-batch critic fit readout inside the active critic update boundary.

The pre snapshot is the driver's existing old value forward. After the ordinary
update, one extra eval/no-grad forward measures the same real step rows. This
module never changes the optimizer, rewards, masks, or returned training metrics.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any

DIRECTORY_ENV = "FINANCE_CRITIC_FIT_PROBE_DIR"
TARGET_MODULE = "agent_r1.workers.critic.dp_critic"
_MARKER = "_finance_critic_fit_probe_directory"


def _get(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def _stats(values: Any) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "std_population": float(values.std(correction=0).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _fit(values: Any, returns: Any) -> dict[str, Any]:
    error = values - returns
    mse = float(error.square().mean().item())
    bias = float(error.mean().item())
    return_variance = float(returns.var(correction=0).item())
    return {
        "values": _stats(values),
        "returns": _stats(returns),
        "mse": mse,
        "half_mse": 0.5 * mse,
        "rmse": math.sqrt(mse),
        "mean_bias_value_minus_return": bias,
        "half_squared_mean_bias": 0.5 * bias * bias,
        "error_std_population": float(error.std(correction=0).item()),
        "explained_variance": (
            1.0 - float(error.var(correction=0).item()) / return_variance
            if return_variance > 0
            else None
        ),
    }


def _dropout_evidence(module: Any) -> dict[str, Any]:
    import torch

    probabilities: dict[str, float] = {}
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.modules.dropout._DropoutNd):
            probabilities[f"{name}.p"] = float(child.p)
        for attribute in ("attention_dropout",):
            value = getattr(child, attribute, None)
            if isinstance(value, (int, float)):
                probabilities[f"{name}.{attribute}"] = float(value)
    model_config = getattr(module, "config", None)
    for key in (
        "attention_dropout",
        "classifier_dropout",
        "hidden_dropout",
        "summary_dropout_prob",
    ):
        value = _get(model_config, key) if model_config is not None else None
        if value is not None:
            try:
                probabilities[f"config.{key}"] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"unrecognized critic dropout setting {key}={value!r}") from None
    return {
        "checked_probabilities": probabilities,
        "any_nonzero": any(value != 0.0 for value in probabilities.values()),
        "pre_mode": "eval (driver compute_values)",
        "post_mode": "eval",
        "training_loss_mode": "train",
        "fit_comparison_uses_same_mode": True,
    }


def _prepare(instance: Any, data: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    response_mask = data.batch["response_mask"]
    if response_mask.ndim != 2 or response_mask.shape[1] == 0:
        raise ValueError("critic fit probe requires nonempty two-dimensional response_mask")
    if not bool(((response_mask == 0) | (response_mask == 1)).all()):
        raise ValueError("critic fit probe requires a binary response_mask")
    if bool(response_mask[:, 1:].any()):
        raise ValueError("critic fit probe requires the step-level first-token value mask")
    sample_mask = data.batch.get("sample_mask")
    if sample_mask is None:
        sample_mask = torch.ones(len(data), dtype=torch.bool, device=response_mask.device)
    if sample_mask.shape != (len(data),) or not bool(
        ((sample_mask == 0) | (sample_mask == 1)).all()
    ):
        raise ValueError("critic fit probe requires a binary per-row sample_mask")
    real_mask = sample_mask.to(dtype=torch.bool)
    if bool((response_mask[:, 0].bool() != real_mask).any()):
        raise ValueError("critic fit probe first-token mask disagrees with real step rows")
    indices = torch.nonzero(real_mask, as_tuple=False).flatten().cpu()
    if indices.numel() == 0:
        raise ValueError("critic fit probe requires at least one real step")
    real_data = data.select_idxs(indices)
    pre = real_data.batch["values"][:, 0].detach().to(device="cpu", dtype=torch.float64).clone()
    returns = (
        real_data.batch["returns"][:, 0].detach().to(device="cpu", dtype=torch.float64).clone()
    )
    if not bool(torch.isfinite(pre).all() and torch.isfinite(returns).all()):
        raise ValueError("critic fit probe pre values and returns must be finite")
    real_data.meta_info = dict(data.meta_info)
    micro_batch_size = _get(instance.config, "forward_micro_batch_size_per_gpu")
    if type(micro_batch_size) is not int or micro_batch_size <= 0:
        raise ValueError("critic fit probe requires positive forward_micro_batch_size_per_gpu")
    dynamic = bool(_get(instance.config, "use_dynamic_bsz", False))
    max_token_len = _get(instance.config, "forward_max_token_len_per_gpu")
    if type(max_token_len) is not int or max_token_len <= 0:
        raise ValueError("critic fit probe requires positive forward_max_token_len_per_gpu")
    real_data.meta_info.update(
        {
            "micro_batch_size": micro_batch_size,
            "max_token_len": max_token_len,
            "use_dynamic_bsz": dynamic,
        }
    )
    policy_steps = real_data.non_tensor_batch.get("finance_screening_policy_step")
    policy_step_values = (
        sorted({int(value) for value in policy_steps}) if policy_steps is not None else []
    )
    if len(policy_step_values) > 1:
        raise ValueError("critic fit probe batch spans multiple policy snapshots")
    mini_ids = data.batch.get("mini_batch_id")
    metadata = {
        "policy_step": policy_step_values[0]
        if policy_step_values
        else data.meta_info.get("global_steps"),
        "real_step_count": len(real_data),
        "padding_row_count": len(data) - len(real_data),
        "value_token_count": int(response_mask.sum().item()),
        "planned_mini_batch_count": int(mini_ids.unique().numel())
        if mini_ids is not None
        else None,
        "ppo_epochs": int(_get(instance.config, "ppo_epochs", 1)),
        "extra_forward_input_tokens": int(real_data.batch["attention_mask"].sum().item()),
        "extra_forward_micro_batches": math.ceil(len(real_data) / micro_batch_size)
        if not dynamic
        else None,
        "forward_meta_info": {
            key: real_data.meta_info[key]
            for key in ("micro_batch_size", "max_token_len", "use_dynamic_bsz")
        },
        "old_value_dtype": str(real_data.batch["values"].dtype),
        "return_dtype": str(real_data.batch["returns"].dtype),
    }
    return real_data, pre, returns, metadata


def patch_critic_class(critic_class: type[Any], directory: Path) -> bool:
    """Wrap the Agent-R1 critic, whose caller already owns active placement."""
    if not directory.is_absolute():
        raise ValueError("critic fit probe directory must be absolute")
    directory = directory.resolve()
    existing = critic_class.__dict__.get(_MARKER)
    if existing is not None:
        if existing != str(directory):
            raise RuntimeError("critic fit probe directory cannot change after installation")
        return False
    original = critic_class.update_critic
    # The pinned GPUMemoryLogger uses an args/kwargs closure without __wrapped__.
    if tuple(inspect.signature(original).parameters) not in {("self", "data"), ("args", "kwargs")}:
        raise RuntimeError("critic fit probe update signature changed")

    @wraps(original)
    def update(instance: Any, data: Any) -> Any:
        import torch

        real_data, pre, returns, metadata = _prepare(instance, data)
        lrs = [float(group["lr"]) for group in instance.critic_optimizer.param_groups]
        dropout = _dropout_evidence(instance.critic_module)
        metrics = original(instance, data)
        # All post operations remain inside the worker's active critic boundary.
        # The ordinary optimizer update is never included in the RNG save/restore.
        module = instance.critic_module
        prior_mode = bool(module.training)
        cuda_devices = sorted(
            {
                parameter.device.index
                for parameter in module.parameters()
                if parameter.device.type == "cuda"
            }
        )
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                post_values = instance.compute_values(real_data)
                post = post_values[:, 0].detach().to(device="cpu", dtype=torch.float64).clone()
                del post_values
        finally:
            module.train(prior_mode)
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(post).all()) or post.shape != pre.shape:
            raise ValueError("critic fit probe post values must be finite and aligned")
        delta = post - pre
        cliprange = float(_get(instance.config, "cliprange_value"))
        grad_clip = float(_get(instance.config, "grad_clip"))
        raw_norms = metrics.get("critic/grad_norm", [])
        grad_norms = [
            float(value)
            for value in (raw_norms if isinstance(raw_norms, (list, tuple)) else [raw_norms])
        ]
        counter = getattr(instance, "_finance_critic_fit_probe_update_count", 0) + 1
        instance._finance_critic_fit_probe_update_count = counter
        record = {
            "schema_version": 1,
            "status": "COMPLETE",
            "phase": "critic_same_batch_pre_post",
            "arm": os.environ.get("FINANCE_SCREENING_ARM"),
            "pid": os.getpid(),
            "update_call_index": counter,
            "execution_identity_sha256": os.environ.get("FINANCE_SCREENING_IDENTITY_SHA256"),
            "pre_source": "driver old values[:,0], before this update call; no extra pre forward",
            "post_source": "same real rows after ordinary update, eval/no_grad compute_values[:,0]",
            "aggregation": (
                "unweighted real step rows; one first-token value per step; float64 readout"
            ),
            "returns_recomputed": False,
            "extra_forward_passes": 1,
            "extra_forward_seconds_synchronized": elapsed,
            "rng_preservation": (
                "CPU and model CUDA device states restored around post forward only"
            ),
            "mode_restored": prior_mode,
            "dropout": dropout,
            "pre": _fit(pre, returns),
            "post": _fit(post, returns),
            "delta_values": {**_stats(delta), "rms": float(delta.square().mean().sqrt().item())},
            "post_delta_exceeds_cliprange_fraction": float(
                (delta.abs() > cliprange).double().mean().item()
            ),
            "cliprange_value": cliprange,
            "critic_learning_rates": lrs,
            "optimizer_grad_norms_before_clip": grad_norms,
            "grad_clip": grad_clip,
            "estimated_grad_clip_coefficients": [
                min(1.0, grad_clip / (value + 1e-6)) for value in grad_norms
            ],
            "original_training_vf_loss": float(metrics["critic/vf_loss"]),
            "original_vf_loss_semantics": (
                "sum across minibatches and PPO epochs, each globally normalized"
            ),
            **metadata,
        }
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        with (directory / f"critic-fit-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return metrics

    critic_class.update_critic = update
    setattr(critic_class, _MARKER, str(directory))
    return True


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped: Any, finder: Any) -> None:
        self.wrapped = wrapped
        self.finder = finder

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self.wrapped.exec_module(module)
        patch_critic_class(module.DataParallelPPOCritic, self.finder.directory)
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
            raise ImportError(f"cannot install critic fit probe for {fullname}")
        spec.loader = _Loader(spec.loader, self)
        return spec


def install_from_environment() -> bool:
    """Install lazily; unset switch leaves imports and runtime unchanged."""
    value = os.environ.get(DIRECTORY_ENV)
    if not value:
        return False
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        raise ValueError(f"{DIRECTORY_ENV} must be an absolute path")
    directory = directory.resolve()
    loaded = sys.modules.get(TARGET_MODULE)
    if loaded is not None:
        return patch_critic_class(loaded.DataParallelPPOCritic, directory)
    for finder in sys.meta_path:
        if isinstance(finder, _Finder):
            if finder.directory != directory:
                raise RuntimeError("critic fit probe directory cannot change after installation")
            return False
    sys.meta_path.insert(0, _Finder(directory))
    return True
