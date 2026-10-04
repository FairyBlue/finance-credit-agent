"""Bounded AdamW updates with pageable CPU moments and unchanged optimizer identity.

No torch import or device work occurs at import time. The caller owns phase
placement, gradient accumulation/clipping, disk exchange, and checkpointing.
This module never copies a whole parameter or gradient to another device.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

FAILED_ATTRIBUTE = "_finance_chunked_adam_failed"
_STATE_KEYS = {"step", "exp_avg", "exp_avg_sq"}


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite Python scalar")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite Python scalar")
    return float(value)


def _validate(optimizer: Any, chunk_bytes: int, torch: Any) -> list[tuple[Any, Any]]:
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("chunked updates require an exact torch.optim.AdamW instance")
    if torch.__version__.split("+")[0] != "2.7.0":
        raise ValueError("chunked AdamW requires the pinned torch 2.7.0 functional path")
    if type(chunk_bytes) is not int or chunk_bytes < 20:
        raise ValueError("chunk_bytes must be an integer of at least 20 bytes")
    if getattr(optimizer, FAILED_ATTRIBUTE, False):
        raise RuntimeError("chunked AdamW previously failed; discard this optimizer and run")
    for name in ("grad_scale", "found_inf"):
        if getattr(optimizer, name, None) is not None:
            raise ValueError(f"{name} is unsupported in the chunked AdamW path")
    active = []
    seen = set()
    for group in optimizer.param_groups:
        for name in ("amsgrad", "capturable", "differentiable"):
            if group.get(name, False) is not False:
                raise ValueError(f"{name}=True is unsupported in the chunked AdamW path")
        for name in ("foreach", "fused"):
            if group.get(name) is not None and group.get(name) is not False:
                raise ValueError(f"{name}=True is unsupported in the chunked AdamW path")
        if group.get("decoupled_weight_decay", True) is not True:
            raise ValueError("decoupled_weight_decay must remain True")
        if type(group.get("maximize", False)) is not bool:
            raise ValueError("maximize must be bool")
        for name in ("lr", "eps", "weight_decay"):
            if _scalar(group[name], name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        betas = group["betas"]
        # verl forwards OmegaConf ListConfig directly to torch AdamW. Keep the
        # existing object and scalar values, accepting its standard Sequence API.
        if not isinstance(betas, Sequence) or len(betas) != 2:
            raise ValueError("betas must contain two Python scalars")
        for index, beta in enumerate(betas):
            if not 0 <= _scalar(beta, f"beta{index + 1}") < 1:
                raise ValueError("betas must lie in [0, 1)")
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise ValueError("duplicate parameters are unsupported")
            seen.add(id(parameter))
            _dense_float32(parameter, torch, "parameter")
            if parameter.device.type not in {"cpu", "cuda"}:
                raise ValueError("only CPU tests and CUDA production parameters are supported")
            state = optimizer.state.get(parameter)
            if state:
                if set(state) != _STATE_KEYS:
                    raise ValueError("unsupported AdamW state keys")
                step = state["step"]
                if (
                    not isinstance(step, torch.Tensor)
                    or step.device.type != "cpu"
                    or step.shape != torch.Size([])
                    or step.dtype not in {torch.float32, torch.float64}
                    or step.requires_grad
                    or step.is_pinned()
                ):
                    raise ValueError("step must be a pageable CPU floating-point scalar")
                step_value = step.item()
                if not math.isfinite(step_value) or step_value < 0 or step_value != int(step_value):
                    raise ValueError("step must be a finite nonnegative integer")
                for name in ("exp_avg", "exp_avg_sq"):
                    moment = state[name]
                    _dense_float32(moment, torch, name)
                    if (
                        moment.device.type != "cpu"
                        or moment.is_pinned()
                        or moment.shape != parameter.shape
                        or moment.requires_grad
                    ):
                        raise ValueError(f"{name} must be pageable CPU state matching parameter")
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            _dense_float32(gradient, torch, "gradient")
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise ValueError("gradient shape and device must match parameter")
            if gradient.requires_grad:
                raise ValueError("differentiable gradients are unsupported")
            active.append((parameter, group))
    if any(id(parameter) not in seen for parameter in optimizer.state):
        raise ValueError("optimizer state contains a parameter outside its groups")
    return active


def _dense_float32(tensor: Any, torch: Any, name: str) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a dense contiguous FP32 tensor")


def chunked_adamw_step(
    optimizer: Any,
    *,
    chunk_bytes: int,
    admit_gpu: Callable[[int], None] | None = None,
    admit_host: Callable[[int], None] | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Apply one stock AdamW-equivalent step using bounded moment slices.

    ``chunk_bytes`` bounds temporary CUDA tensor payload, including both moments,
    both simultaneously live sqrt/div intermediates, and an optional negated
    gradient for maximize. Allocator granularity and unrelated live tensors need
    an additional reserve in the caller's resource admission policy. CPU tests
    use the same slice sizes, with moment views directly updated in place.

    ``admit_host(n)`` precedes lazy CPU state allocation; ``admit_gpu(n)`` precedes
    each CUDA chunk. Hooks must raise to deny admission. ``emit`` receives one
    completion event or a best-effort error event. An exception permanently marks
    this optimizer failed: partial parameter/state updates cannot be resumed.
    """
    import torch
    from torch.optim.adamw import adamw
    from torch.optim.optimizer import _get_scalar_dtype

    started = time.monotonic()
    chunks = 0
    updated_parameters = 0
    maximum_temporary_bytes = 0
    moment_transfer_bytes = 0
    host_allocated_bytes = 0
    try:
        active = _validate(optimizer, chunk_bytes, torch)
        with torch.no_grad():
            for parameter, group in active:
                state = optimizer.state[parameter]
                if not state:
                    step_dtype = _get_scalar_dtype()
                    scalar_bytes = 8 if step_dtype == torch.float64 else 4
                    required_host_bytes = 2 * parameter.numel() * 4 + scalar_bytes
                    if admit_host is not None:
                        admit_host(required_host_bytes)
                    state["step"] = torch.zeros((), dtype=step_dtype, device="cpu")
                    state["exp_avg"] = torch.zeros(
                        parameter.shape, dtype=torch.float32, device="cpu", pin_memory=False,
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        parameter.shape, dtype=torch.float32, device="cpu", pin_memory=False,
                    )
                    host_allocated_bytes += required_host_bytes

                # Each chunk sees the ORIGINAL step, not the previous chunk's step.
                original_step = state["step"].clone()
                flat_parameter = parameter.view(-1)
                flat_gradient = parameter.grad.view(-1)
                flat_average = state["exp_avg"].view(-1)
                flat_square = state["exp_avg_sq"].view(-1)
                temporary_arrays = 4 + int(group.get("maximize", False))
                elements_per_chunk = chunk_bytes // (temporary_arrays * 4)
                for start in range(0, parameter.numel(), elements_per_chunk):
                    stop = min(start + elements_per_chunk, parameter.numel())
                    required_gpu_bytes = (stop - start) * temporary_arrays * 4
                    if parameter.device.type == "cuda" and admit_gpu is not None:
                        admit_gpu(required_gpu_bytes)
                    average_cpu = flat_average[start:stop]
                    square_cpu = flat_square[start:stop]
                    # Blocking pageable copies make ownership/lifetime explicit.
                    average = average_cpu.to(device=parameter.device, non_blocking=False)
                    square = square_cpu.to(device=parameter.device, non_blocking=False)
                    chunk_step = original_step.clone()
                    adamw(
                        [flat_parameter[start:stop]], [flat_gradient[start:stop]],
                        [average], [square], [], [chunk_step],
                        foreach=False, capturable=False, differentiable=False, fused=False,
                        amsgrad=False, beta1=group["betas"][0], beta2=group["betas"][1],
                        lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"],
                        maximize=group.get("maximize", False),
                    )
                    if parameter.device.type == "cuda":
                        average_cpu.copy_(average, non_blocking=False)
                        square_cpu.copy_(square, non_blocking=False)
                        moment_transfer_bytes += (stop - start) * 4 * 4
                    del average, square, chunk_step
                    maximum_temporary_bytes = max(maximum_temporary_bytes, required_gpu_bytes)
                    chunks += 1
                # Empty parameters also increment exactly once, as stock AdamW does.
                state["step"].add_(1)
                updated_parameters += 1
        if emit is not None:
            emit({
                "event": "chunked_adamw_step_complete",
                "updated_parameters": updated_parameters,
                "chunks": chunks,
                "chunk_bytes": chunk_bytes,
                "maximum_temporary_bytes": maximum_temporary_bytes,
                "moment_transfer_bytes": moment_transfer_bytes,
                "host_allocated_bytes": host_allocated_bytes,
                "elapsed_seconds": time.monotonic() - started,
            })
    except BaseException as error:
        setattr(optimizer, FAILED_ATTRIBUTE, True)
        if emit is not None:
            with suppress(BaseException):
                emit({
                    "event": "chunked_adamw_step_failed",
                    "error_type": type(error).__name__,
                    "completed_chunks": chunks,
                    "updated_parameters": updated_parameters,
                    "elapsed_seconds": time.monotonic() - started,
                })
        raise
