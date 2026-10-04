"""Fresh-run, single-GPU FP32 Adam with one CPU state and one disk state.

Model parameters never move. The existing worker boundary and global gradient
clipping paths are retained; only the optimizer's state and scalar update move.
"""

from __future__ import annotations

import ctypes
import gc
import os
import time
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any

from src.agents.screening_exchange import ExchangeError
from src.agents.screening_placement import WorkerRegistry

PROFILE = "chunked_adam_disk_v1"
_GIB = 1024**3
_PHASES = frozenset({
    "update_actor", "update_critic", "compute_log_prob", "compute_ref_log_prob",
    "compute_values", "rollout_mode", "save_checkpoint",
})


def _release_pageable_allocator() -> dict[str, Any]:
    """Release unused process-owned pages, never evict another process's cache.

    Torch's host_emptyCache covers pinned allocations, not glibc's pageable
    malloc arenas. MomentStore has already issued DONTNEED for each verified raw
    file; collection and malloc_trim now release freed tensor/buffer allocations.
    A missing trim facility is reported, and actual cgroup admission still gates
    every incoming moment rather than assuming those pages became available.
    """
    collected = gc.collect()
    libc = ctypes.CDLL(None)
    trim = getattr(libc, "malloc_trim", None)
    if trim is None:
        return {"gc_collected": collected, "malloc_trim_available": False}
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return {
        "gc_collected": collected, "malloc_trim_available": True,
        "malloc_trim_released": bool(trim(0)),
    }


def environment_options() -> dict[str, int]:
    defaults = dict(
        chunk_bytes=64 * 1024**2,
        gpu_reserve_bytes=2 * _GIB,
        host_reserve_bytes=2 * _GIB,
        disk_reserve_bytes=8 * _GIB,
    )
    result = {}
    for key, default in defaults.items():
        value = os.environ.get("FINANCE_CHUNKED_ADAM_" + key.upper(), str(default))
        try:
            parsed = int(value)
        except ValueError as error:
            raise ExchangeError("invalid chunked Adam option: " + key) from error
        if parsed <= 0 or (
            key == "chunk_bytes" and (parsed < 20 or parsed % 4 or parsed > 256 * 1024**2)
        ):
            raise ExchangeError("out-of-bound chunked Adam option: " + key)
        result[key] = parsed
    return result


class DiskChunkCoordinator:
    """All errors permanently stop this execution identity; never retry a partial step."""

    def __init__(self, *, identity: str, backend: Any, boundary_check: Any, allow_step: Any,
                 directory: Path, chunk_bytes: int, disk_reserve_bytes: int):
        self.identity, self.backend = identity, backend
        self.boundary_check, self.directory = boundary_check, directory
        self.allow_step = allow_step
        self.chunk_bytes, self.disk_reserve_bytes = chunk_bytes, disk_reserve_bytes
        self.pid = os.getpid()
        self.active_role = "actor"
        self.failed = False
        self.transitioning = False
        self.stepping = False
        self.events: list[dict[str, Any]] = []
        self._roles: dict[str, tuple[Any, Any]] = {}
        self.stores: dict[str, Any] = {}
        self.step_counts = {"actor": 0, "critic": 0}

    def _emit(self, event: str, **fields: Any) -> None:
        self.events.append(dict(event=event, time_ns=time.time_ns(),
                                active_role=self.active_role, **fields))

    def _fail(self, error: BaseException) -> None:
        self.failed = True
        self._emit("failed", error_type=type(error).__name__)

    def _guard(self, identity: str) -> None:
        if (identity != self.identity or os.getpid() != self.pid or self.failed
                or self.transitioning or self.stepping):
            raise ExchangeError("chunked placement identity, phase or failure guard")

    def register(self, role: str, module: Any, optimizer: Any, *, identity: str) -> None:
        from src.agents.optimizer_state_store import MomentStore

        self._guard(identity)
        if role not in self.step_counts or role in self._roles or optimizer.state:
            raise ExchangeError("chunked placement requires two distinct fresh optimizers")
        if any(module is m or optimizer is o for m, o in self._roles.values()):
            raise ExchangeError("actor and critic cannot share training state")
        if getattr(optimizer, "_finance_chunked_step", False):
            raise ExchangeError("optimizer already has a chunked step")
        for group in optimizer.param_groups:
            if any(group.get(key) is not False for key in ("foreach", "fused")):
                raise ExchangeError("chunked Adam requires scalar, non-fused updates")
        if (getattr(optimizer, "_optimizer_step_pre_hooks", {})
                or getattr(optimizer, "_optimizer_step_post_hooks", {})):
            raise ExchangeError("optimizer step hooks are outside this execution profile")
        self._roles[role] = (module, optimizer)
        self.stores[role] = MomentStore(
            self.directory / role, identity, role, chunk_bytes=self.chunk_bytes,
            disk_reserve_bytes=self.disk_reserve_bytes,
            admit_host=lambda size: self.backend.admit("cpu", size),
            emit=lambda row: self._emit("disk_state", role=role, detail=row),
        )
        original_step = optimizer.step

        @wraps(original_step)
        def step(instance: Any, closure: Any = None) -> None:
            from src.agents.chunked_adam import chunked_adamw_step

            try:
                self._guard(identity)
                if closure is not None or instance is not optimizer or role != self.active_role:
                    raise ExchangeError("optimizer step is outside its active role")
                if not self.allow_step(role):
                    raise ExchangeError("optimizer step requires the matching guarded update phase")
                if self.stores[role].status != "resident":
                    raise ExchangeError("optimizer step requires resident CPU moments")
                # This method runs inside the worker's original _optimizer_step,
                # after whole-model gradient clipping. Scheduler semantics stay
                # one step per original worker update, regardless of chunk count.
                self.stepping = True
                instance._opt_called = True
                chunked_adamw_step(
                    instance, chunk_bytes=self.chunk_bytes,
                    admit_gpu=lambda size: self.backend.admit(self.backend.gpu_device, size),
                    admit_host=lambda size: self.backend.admit("cpu", size),
                    emit=lambda row: self._emit("optimizer_chunked", role=role, detail=row),
                )
                self.step_counts[role] += 1
                self._emit("optimizer_complete", role=role, role_step=self.step_counts[role])
            except BaseException as error:
                self._fail(error)
                raise
            finally:
                self.stepping = False

        step._wrapped_by_lr_sched = True
        optimizer.step = MethodType(step, optimizer)
        optimizer._finance_chunked_step = True
        self._emit("registered", role=role)

    def _summary(self) -> dict[str, Any]:
        import torch

        if set(self._roles) != {"actor", "critic"}:
            raise ExchangeError("both model roles must be registered")
        counts = {}
        for role, (_, optimizer) in self._roles.items():
            status = self.stores[role].status
            if status == "failed" or (role == self.active_role and status != "resident"):
                raise ExchangeError("optimizer moment store lifecycle is inconsistent")
            if (
                role != self.active_role and status != "offloaded"
                and (status != "resident" or self.step_counts[role] != 0)
            ):
                raise ExchangeError("inactive optimizer state was not offloaded")
            total = 0
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        raise ExchangeError("live gradient outside full update boundary")
                    state = optimizer.state.get(parameter, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            value = state[key]
                            if (role != self.active_role or value.device.type != "cpu"
                                    or value.dtype != torch.float32 or value.is_pinned()):
                                raise ExchangeError("optimizer moment residency changed")
                            total += value.numel() * value.element_size()
            counts[role] = total
        return dict(moment_cpu_bytes_by_role=counts, role_optimizer_steps=dict(self.step_counts),
                    disk_status={role: store.status for role, store in self.stores.items()})

    def assert_ready(self, role: str, *, identity: str) -> dict[str, Any]:
        try:
            self._guard(identity)
            if role != self.active_role:
                raise ExchangeError("requested optimizer role is not active")
            self.boundary_check()
            return self._summary()
        except BaseException as error:
            self._fail(error)
            raise

    def activate(self, role: str, *, identity: str) -> dict[str, Any]:
        try:
            self._guard(identity)
            if role not in self._roles:
                raise ExchangeError("unknown optimizer role")
            self.boundary_check()
            before = self._summary()
            if role == self.active_role:
                return before
            self.transitioning = True
            outgoing = self.active_role
            self._emit("exchange_start", target_role=role, **before)
            # Release every outgoing CPU moment before allocating incoming state.
            self.stores[outgoing].offload(self._roles[outgoing][1])
            if self.stores[outgoing].status != "offloaded":
                raise ExchangeError("outgoing optimizer moments were not fully offloaded")
            self.backend.synchronize()
            self._emit("host_reclaimed", outgoing_role=outgoing,
                       target_role=role, **_release_pageable_allocator())
            if self.stores[role].status == "offloaded":
                self.stores[role].restore(self._roles[role][1])
            if self.stores[role].status != "resident":
                raise ExchangeError("incoming optimizer moments are not resident")
            self.active_role = role
            result = self._summary()
            self._emit("exchange_complete", **result)
            return result
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            self.transitioning = False


class ChunkedWorkerRegistry(WorkerRegistry):
    def __init__(self, *, identity: str, backend: Any, emit: Any, directory: Path,
                 chunk_bytes: int, disk_reserve_bytes: int):
        super().__init__(identity=identity, backend=backend, emit=emit, profile=PROFILE)
        self.current_phase: str | None = None
        self.coordinator = DiskChunkCoordinator(
            identity=identity, backend=backend, boundary_check=self.boundary_check,
            allow_step=self._allow_optimizer_step,
            directory=directory / "moment_scratch" / str(os.getpid()),
            chunk_bytes=chunk_bytes, disk_reserve_bytes=disk_reserve_bytes,
        )

    def _allow_optimizer_step(self, role: str) -> bool:
        self.check_identity()
        return self.busy and self.current_phase == "update_" + role

    def begin(self, phase: str, *, activate: str | None = None) -> None:
        try:
            if self.current_phase is not None or self.busy:
                raise ExchangeError("another full worker phase is in progress")
            if phase not in _PHASES:
                raise ExchangeError("unsupported chunked worker phase")
            expected = phase.removeprefix("update_") if phase.startswith("update_") else None
            if activate != expected:
                raise ExchangeError("worker phase and requested active role disagree")
            if (
                phase == "save_checkpoint"
                and os.environ.get("FINANCE_SCREENING_ARTIFACT_PROFILE") != "actor_bf16_hf_v1"
            ):
                raise ExchangeError("chunked moment scratch permits only actor inference exports")
            super().begin(phase, activate=activate)
            # No optimizer update is allowed during pre-entry validation/exchange.
            self.current_phase = phase
        except BaseException as error:
            self.fail(error, phase)
            raise

    def finish(self, phase: str, role: str) -> None:
        try:
            if not self.busy or self.current_phase != phase:
                raise ExchangeError("worker finish does not match the active phase")
            expected = phase.removeprefix("update_") if phase.startswith("update_") else "actor"
            if role != expected:
                raise ExchangeError("worker finish role differs from its active phase")
            super().finish(phase, role)
            self.current_phase = None
        except BaseException as error:
            self.fail(error, phase)
            raise

    def fail(self, error: BaseException, phase: str) -> None:
        self.current_phase = None
        super().fail(error, phase)
