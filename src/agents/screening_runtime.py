"""Opt-in M2 screening advantage hook for the pinned Agent-R1 driver.

The driver holds the complete logical rollout batch before PPO minibatching.
The hook exposes step GAE before whitening, adds an actor-only sibling residual,
and whitens once over all sample_mask=true step rows. Critic returns retain the
unmodified reward/GAE definition. No upstream files or legacy runs are changed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

from src.rewards.sibling_credit import (
    DIAGNOSTIC_ATOL,
    SiblingStep,
    compute_sibling_credit,
)

SCREENING_ARM_ENV = "FINANCE_SCREENING_ARM"
SCREENING_DIAGNOSTICS_ENV = "FINANCE_SCREENING_DIAGNOSTICS_DIR"
SCREENING_SEED_ENV = "FINANCE_SCREENING_RUN_SEED"
SCREENING_ARMS = frozenset({"A2", "M1", "M2-plain", "M2"})
_TRAINER_MODULE = "agent_r1.trainer.ppo.ray_trainer"
_configured_settings: ScreeningRuntimeSettings | None = None


@dataclass(frozen=True)
class ScreeningRuntimeSettings:
    arm: str
    diagnostics_dir: Path
    run_seed: int


def _python_scalar(value: Any) -> Any:
    # np scalar object columns are produced by AgentFlowWorkerBase._postprocess.
    import numpy as np

    return value.item() if isinstance(value, np.generic) else value


def _step_rows(data: Any, indices: list[int], arm: str) -> list[SiblingStep]:
    arrays = data.non_tensor_batch
    field_names = {
        "trajectory_uid": "trajectory_uids",
        "group_uid": "finance_screening_group_uid",
        "episode_id": "finance_screening_episode_id",
        "policy_step": "finance_screening_policy_step",
        "rollout_id": "finance_screening_rollout_id",
        "rollout_n": "finance_screening_rollout_n",
        "step_index": "step_indices",
        "step_count": "finance_screening_step_count",
        "decision_session_index": "trading_decision_session_index",
        "result_session_index": "trading_result_session_index",
        "action": "trading_action",
        "invalid_action": "trading_invalid_action",
        "done": "trading_env_done",
        "episode_complete": "finance_screening_trajectory_complete",
        "terminal_score": "finance_screening_terminal_score",
    }
    required = {
        *field_names.values(),
        "uid",
        "finance_screening_arm",
        "trading_episode_complete",
        "finance_screening_net_return",
    }
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"screening step evidence missing: {sorted(missing)}")
    for key in required:
        if len(arrays[key]) != len(data):
            raise ValueError(f"screening metadata {key!r} is not aligned to trainer rows")
    result = []
    for index in indices:
        if arrays["finance_screening_arm"][index] != arm:
            raise ValueError("screening flow arm does not match trainer runtime arm")
        if arrays["finance_screening_group_uid"][index] != arrays["uid"][index]:
            raise ValueError("screening sibling UID does not match upstream prompt occurrence UID")
        if _python_scalar(arrays["trading_episode_complete"][index]) is not True:
            raise ValueError("screening requires completed environment evidence")
        result.append(
            SiblingStep(
                **{name: _python_scalar(arrays[key][index]) for name, key in field_names.items()}
            )
        )
    return result


def _validate_reward_timeline(
    data: Any,
    indices: list[int],
    rows: list[SiblingStep],
    step_rewards: Any,
    arm: str,
) -> None:
    """Check the actual optimizer reward against the flow's objective evidence."""
    import torch

    trajectories: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        trajectories[row.trajectory_uid].append(position)
    for positions in trajectories.values():
        positions.sort(key=lambda i: rows[i].step_index)
        net_returns = [
            _python_scalar(data.non_tensor_batch["finance_screening_net_return"][indices[i]])
            for i in positions
        ]
        if any(isinstance(value, bool) or not math.isfinite(value) for value in net_returns):
            raise ValueError("screening raw net return must be finite")
        if any(value != net_returns[0] for value in net_returns):
            raise ValueError("raw net return changes across a trajectory")
        if arm == "A2":
            expected = [0.0] * (len(positions) - 1) + [net_returns[0]]
        else:
            score = rows[positions[0]].terminal_score
            expected = [score / len(positions)] * (len(positions) - 1)
            expected.append(score - sum(expected))
        expected_tensor = step_rewards.new_tensor(expected)
        if not torch.allclose(step_rewards[positions], expected_tensor, rtol=1e-6, atol=1e-8):
            raise ValueError("trainer reward timeline does not match the screening arm contract")


def unwhitened_step_gae(
    *,
    token_level_rewards: Any,
    values: Any,
    response_mask: Any,
    trajectory_uids: Any,
    step_indices: Any,
    gamma: float,
    lam: float,
) -> tuple[Any, Any]:
    """Pinned StepPO GAE up to its whitening line, returning scalar U/returns.

    The map construction, reward sum, first-token pre-action value, recurrence,
    and dtype arithmetic match Agent-R1 b124aa4 core_algos. Completeness and
    contiguous step indices must be checked by the caller before this function.
    """
    import numpy as np
    import torch

    with torch.no_grad():
        rewards = (token_level_rewards * response_mask).sum(dim=1)
        step_values = values[:, 0]
        unique_traj, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=rewards.device)
        step_ids = torch.as_tensor(step_indices, device=rewards.device)
        max_step = int(step_ids.max().item()) + 1
        shape = (len(unique_traj), max_step)
        rewards_map = torch.zeros(shape, dtype=rewards.dtype, device=rewards.device)
        values_map = torch.zeros(shape, dtype=step_values.dtype, device=rewards.device)
        rewards_map[traj_inv, step_ids] = rewards
        values_map[traj_inv, step_ids] = step_values
        last_gae: Any = 0
        reversed_advantages: list[Any] = []
        for step in reversed(range(max_step)):
            next_value = values_map[:, step + 1] if step < max_step - 1 else 0.0
            delta = rewards_map[:, step] + gamma * next_value - values_map[:, step]
            last_gae = delta + gamma * lam * last_gae
            reversed_advantages.append(last_gae)
        advantage_map = torch.stack(reversed_advantages[::-1], dim=1)
        unwhitened = advantage_map[traj_inv, step_ids]
        returns = unwhitened + step_values
    return unwhitened, returns


def _scope_diagnostics(unwhitened: Any, residual: Any) -> dict[str, object]:
    import torch

    # Diagnostic arithmetic uses float64 but does not enter optimizer tensors.
    u = unwhitened.detach().double()
    s = residual.detach().double()
    u_std = u.std(correction=1)
    s_std = s.std(correction=1)
    correlation = None
    if u_std > 0 and s_std > 0:
        correlation = float(
            (((u - u.mean()) * (s - s.mean())).sum() / ((u.numel() - 1) * u_std * s_std)).item()
        )
    return {
        "row_count": u.numel(),
        "std_U": float(u_std.item()),
        "std_S": float(s_std.item()),
        "corr_U_S": correlation,
        "mean_abs_S": float(s.abs().mean().item()),
        "nonzero_S_count": int(torch.count_nonzero(s.abs() > DIAGNOSTIC_ATOL).item()),
        "nonzero_S_rate": float((s.abs() > DIAGNOSTIC_ATOL).double().mean().item()),
        "exact_nonzero_S_count": int(torch.count_nonzero(s).item()),
    }


def compute_screening_advantage(
    data: Any,
    *,
    arm: str,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 4,
    eta: float = 1.0,
    diagnostics_dir: Path | None = None,
    run_seed: int | None = None,
) -> Any:
    """Compute real trainer tensors and pre-update diagnostics for four arms."""
    import numpy as np
    import torch

    if arm not in SCREENING_ARMS:
        raise ValueError(f"unsupported screening arm: {arm!r}")
    if gamma != 1.0 or lam != 1.0 or type(num_repeat) is not int or num_repeat != 4:
        raise ValueError("screening requires gamma=1, lambda=1, and rollout.n=4")
    if not math.isfinite(eta) or eta not in (0.0, 1.0):
        raise ValueError("screening eta must be 1 (0 is reserved for matched-null verification)")
    if data.meta_info.get("validate", False):
        raise ValueError("evaluation rollouts must not enter the training advantage hook")
    if "finance_screening_diagnostics" in data.meta_info:
        raise ValueError("screening advantages have already been computed for this batch")
    required_tensors = {"token_level_rewards", "values", "response_mask"}
    if not required_tensors <= data.batch.keys():
        raise ValueError(
            f"missing screening tensors: {sorted(required_tensors - data.batch.keys())}"
        )
    rewards = data.batch["token_level_rewards"]
    values = data.batch["values"]
    mask = data.batch["response_mask"]
    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("screening reward, value and response-mask shapes must agree")
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise ValueError("screening rewards and critic values must be floating point")
    sample_mask = data.batch.get("sample_mask")
    if sample_mask is None:
        sample_mask = torch.ones(len(data), dtype=torch.bool, device=rewards.device)
    if sample_mask.shape != (len(data),) or not bool(
        ((sample_mask == 0) | (sample_mask == 1)).all()
    ):
        raise ValueError("sample_mask must contain one binary entry per step row")
    valid_mask = sample_mask.to(dtype=torch.bool)
    indices = torch.nonzero(valid_mask, as_tuple=False).flatten().cpu().tolist()
    rows = _step_rows(data, indices, arm)
    credits = compute_sibling_credit(rows)
    with torch.no_grad():
        valid_rewards, valid_values, valid_response = (
            rewards[valid_mask],
            values[valid_mask],
            mask[valid_mask],
        )
        if not bool(((valid_response == 0) | (valid_response == 1)).all()):
            raise ValueError("response_mask must be binary")
        if not bool((valid_response[:, 0] == 1).all()):
            raise ValueError("every real step must begin with a valid response token")
        if not bool(torch.isfinite(valid_rewards).all()) or not bool(
            torch.isfinite(valid_values).all()
        ):
            raise ValueError("screening reward or critic value contains non-finite numbers")
        _validate_reward_timeline(
            data,
            indices,
            rows,
            (valid_rewards * valid_response).sum(dim=1),
            arm,
        )
        u, step_returns = unwhitened_step_gae(
            token_level_rewards=valid_rewards,
            values=valid_values,
            response_mask=valid_response,
            trajectory_uids=np.array([row.trajectory_uid for row in rows], dtype=object),
            step_indices=np.array([row.step_index for row in rows], dtype=np.int32),
            gamma=gamma,
            lam=lam,
        )
        raw_residual = credits.pooled if arm == "M2" else credits.plain
        residual = torch.tensor(raw_residual, dtype=u.dtype, device=u.device)
        if arm in {"A2", "M1"}:
            residual.zero_()
        residual = residual * eta
        combined = u + residual
        std = combined.std(correction=1)
        scalar_advantages = (combined - combined.mean()) / (std + 1e-8)
        if not bool(torch.isfinite(scalar_advantages).all()) or not bool(
            torch.isfinite(step_returns).all()
        ):
            raise ValueError("screening GAE/whitening produced non-finite tensors")
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        advantages[valid_mask] = scalar_advantages.unsqueeze(1) * valid_response
        returns[valid_mask] = step_returns.unsqueeze(1) * valid_response
        advancing_mask = torch.tensor(credits.advancing, dtype=torch.bool, device=u.device)
        diagnostics = {
            "schema_version": 1,
            "phase": "advantage_pre_update",
            "arm": arm,
            "run_seed": run_seed,
            "pid": os.getpid(),
            "eta": eta,
            **credits.diagnostics,
            "all_real_steps": _scope_diagnostics(u, residual),
            "advancing_steps": _scope_diagnostics(u[advancing_mask], residual[advancing_mask]),
            "whitening_std": float(std.item()),
            "whitening_mean": float(combined.mean().item()),
            "whitening_epsilon": 1e-8,
            "whitening_correction": 1,
            "whitening_scope": "logical_rollout_batch_real_step_rows",
            "response_tokens": int(valid_response.sum().item()),
            "padding_row_count": len(data) - len(rows),
            "episode_ids": sorted({row.episode_id for row in rows}),
            "sibling_group_uids": sorted({row.group_uid for row in rows}),
        }
    # Persist before returning anything to the optimizer; logging failure is fatal.
    if diagnostics_dir is not None:
        directory = Path(diagnostics_dir)
        if not directory.is_absolute():
            raise ValueError("screening diagnostics directory must be absolute")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"advantage-{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostics, sort_keys=True, allow_nan=False) + "\n")
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    data.meta_info["finance_screening_diagnostics"] = diagnostics
    return data


def patch_trainer_module(module: ModuleType, settings: ScreeningRuntimeSettings) -> bool:
    """Replace only the driver entry point, preserving all legacy imports."""
    original = module.compute_advantage
    existing = getattr(original, "_finance_screening_settings", None)
    if existing is not None:
        if existing != settings:
            raise RuntimeError("screening hook cannot change configuration after installation")
        return False
    parameters = tuple(inspect.signature(original).parameters)
    if parameters != (
        "data",
        "adv_estimator",
        "gamma",
        "lam",
        "num_repeat",
        "norm_adv_by_std_in_grpo",
        "config",
    ):
        raise RuntimeError("pinned Agent-R1 compute_advantage signature changed")
    from src.agents.screening_checkpoint import install_checkpoint_hooks

    install_checkpoint_hooks(module.RayAgentTrainer)

    @wraps(original)
    def screening_advantage(
        data: Any,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
    ) -> Any:
        if not isinstance(adv_estimator, module.AgentAdvantageEstimator) or (
            adv_estimator != module.AgentAdvantageEstimator.GAE
        ):
            raise ValueError("all screening arms require the step-level GAE estimator")
        if config is not None and config.get("use_kl_in_reward", False):
            raise ValueError("screening residual contract forbids extra KL reward shaping")
        return compute_screening_advantage(
            data,
            arm=settings.arm,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            diagnostics_dir=settings.diagnostics_dir,
            run_seed=settings.run_seed,
        )

    screening_advantage._finance_screening_settings = settings  # type: ignore[attr-defined]
    module.compute_advantage = screening_advantage  # type: ignore[attr-defined]
    return True


class _ScreeningLoader(importlib.abc.Loader):
    def __init__(self, wrapped: Any, finder: _ScreeningFinder) -> None:
        self.wrapped = wrapped
        self.finder = finder

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        patch_trainer_module(module, self.finder.settings)
        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)


class _ScreeningFinder(importlib.abc.MetaPathFinder):
    def __init__(self, settings: ScreeningRuntimeSettings) -> None:
        self.settings = settings

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != _TRAINER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            raise ImportError(f"cannot install screening hook for {fullname}")
        spec.loader = _ScreeningLoader(spec.loader, self)
        return spec


def install_from_environment() -> None:
    """Install lazily only when the dedicated screening arm switch is present."""
    arm = os.environ.get(SCREENING_ARM_ENV)
    if not arm:
        return
    if arm not in SCREENING_ARMS:
        raise ValueError(f"{SCREENING_ARM_ENV} must be one of {sorted(SCREENING_ARMS)}")
    directory_value = os.environ.get(SCREENING_DIAGNOSTICS_ENV, "")
    directory = Path(directory_value).expanduser()
    if not directory_value or not directory.is_absolute():
        raise ValueError(f"{SCREENING_DIAGNOSTICS_ENV} must be an absolute path")
    seed_value = os.environ.get(SCREENING_SEED_ENV, "")
    if not seed_value.isascii() or not seed_value.isdigit():
        raise ValueError(f"{SCREENING_SEED_ENV} must be a non-negative integer")
    settings = ScreeningRuntimeSettings(arm, directory.resolve(), int(seed_value))
    global _configured_settings
    if _configured_settings is not None and _configured_settings != settings:
        raise RuntimeError("screening runtime cannot change settings after installation")
    _configured_settings = settings
    loaded = sys.modules.get(_TRAINER_MODULE)
    if loaded is not None:
        patch_trainer_module(loaded, settings)
    elif not any(isinstance(finder, _ScreeningFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ScreeningFinder(settings))
