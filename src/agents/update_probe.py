"""Runtime-only proof that a real backward pass changed model parameters."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any


def summarize_update_evidence(probe_root: str | Path) -> dict[str, object]:
    """Merge per-worker update records and fail closed when no change was detected."""

    root = Path(probe_root)
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")) if root.is_dir() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                document = json.loads(line)
                if not isinstance(document, dict):
                    raise RuntimeError(f"invalid optimizer evidence in {path}")
                records.append(document)
    per_process_backward: dict[int, int] = defaultdict(int)
    for record in records:
        process = int(record.get("pid", -1))
        per_process_backward[process] = max(
            per_process_backward[process], int(record.get("backward_count", 0))
        )
    detected_records = [record for record in records if record.get("detected") is True]
    representative = detected_records[0] if detected_records else (records[0] if records else None)
    detected_role_counts: dict[str, int] = defaultdict(int)
    for record in detected_records:
        role = record.get("role")
        if isinstance(role, str) and role:
            detected_role_counts[role] += 1
    peak_gpu_memory_bytes = max(
        (int(record.get("peak_gpu_memory_bytes", 0)) for record in records),
        default=0,
    )
    return {
        "detected": bool(detected_records),
        "backward_count": sum(per_process_backward.values()),
        "optimizer_step_count": len(records),
        "global_update_count": len(records),
        "detected_role_counts": dict(sorted(detected_role_counts.items())),
        "detected_roles": sorted(detected_role_counts),
        "worker_files": len(list(root.glob("*.jsonl"))) if root.is_dir() else 0,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "representative_update": representative,
    }


def install_from_environment() -> None:
    """Patch verl optimizer steps inside launcher and Ray worker processes."""

    output_value = os.environ.get("FINANCE_UPDATE_PROBE_DIR")
    if not output_value:
        return
    import torch

    output = Path(output_value).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {"backward_count": 0, "optimizer_step_count": 0}
    original_backward = torch.autograd.backward
    if not getattr(original_backward, "_finance_update_probe", False):

        @wraps(original_backward)
        def counted_backward(*args: Any, **kwargs: Any) -> Any:
            state["backward_count"] += 1
            return original_backward(*args, **kwargs)

        counted_backward._finance_update_probe = True  # type: ignore[attr-defined]
        torch.autograd.backward = counted_backward

    def sample_parameter(module: Any) -> tuple[str, Any, Any] | None:
        for name, parameter in module.named_parameters():
            gradient = parameter.grad
            if gradient is None or parameter.numel() == 0:
                continue
            flat_gradient = gradient.detach().reshape(-1)
            step = max(1, flat_gradient.numel() // 4096)
            sampled_gradient = flat_gradient[::step][:4096].float()
            if not bool(torch.isfinite(sampled_gradient).all()) or not bool(
                sampled_gradient.abs().max() > 0
            ):
                continue
            flat_parameter = parameter.detach().reshape(-1)
            sampled_parameter = flat_parameter[::step][:4096].float().cpu().clone()
            return name, sampled_parameter, sampled_gradient.cpu().clone()
        return None

    def patch_class(cls: type[Any], module_attribute: str, role: str) -> None:
        original: Callable[..., Any] = cls._optimizer_step
        if getattr(original, "_finance_update_probe", False):
            return

        @wraps(original)
        def probed_step(instance: Any, *args: Any, **kwargs: Any) -> Any:
            selected = sample_parameter(getattr(instance, module_attribute))
            result = original(instance, *args, **kwargs)
            state["optimizer_step_count"] += 1
            record: dict[str, object] = {
                "pid": os.getpid(),
                "role": role,
                "backward_count": state["backward_count"],
                "optimizer_step_count": state["optimizer_step_count"],
                "global_update_count": state["optimizer_step_count"],
                "detected": False,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
            try:
                record["grad_norm"] = float(result.detach().float().cpu().item())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                record["grad_norm"] = None
            if selected is not None:
                name, before, sampled_gradient = selected
                current = dict(getattr(instance, module_attribute).named_parameters())[name]
                flat = current.detach().reshape(-1)
                step = max(1, flat.numel() // 4096)
                after = flat[::step][:4096].float().cpu()
                delta = after - before
                delta_l2 = float(torch.linalg.vector_norm(delta).item())
                delta_max = float(delta.abs().max().item())
                finite = all(math.isfinite(value) for value in (delta_l2, delta_max))
                record.update(
                    {
                        "parameter_name": name,
                        "parameter_sample_size": int(before.numel()),
                        "before_sha256": sha256(before.numpy().tobytes()).hexdigest(),
                        "after_sha256": sha256(after.numpy().tobytes()).hexdigest(),
                        "sampled_gradient_l2": float(
                            torch.linalg.vector_norm(sampled_gradient).item()
                        ),
                        "parameter_delta_l2": delta_l2,
                        "parameter_delta_max": delta_max,
                        "detected": finite and delta_max > 0.0,
                    }
                )
            path = output / f"optimizer-update-{os.getpid()}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            return result

        probed_step._finance_update_probe = True  # type: ignore[attr-defined]
        cls._optimizer_step = probed_step

    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.critic.dp_critic import DataParallelPPOCritic

    patch_class(DataParallelPPOActor, "actor_module", "actor")
    patch_class(DataParallelPPOCritic, "critic_module", "critic")
