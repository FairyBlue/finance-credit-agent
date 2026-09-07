"""Byte-balanced synchronous Adam moment exchange for the opt-in screening runtime.

Uses injected device operations and retains metadata/slots, never old tensor lists.
"""

from __future__ import annotations

import os
import time
import weakref
from collections.abc import Callable, MutableMapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

ROLES = ("actor", "critic")
MOMENTS = ("exp_avg", "exp_avg_sq")


class ExchangeError(RuntimeError):
    """The placement contract failed; this coordinator must not be reused."""


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    device: str
    storage_id: int
    storage_bytes: int
    payload_bytes: int
    contiguous: bool = True
    storage_offset: int = 0
    pinned: bool = False
    is_view: bool = False


@dataclass(frozen=True)
class Operations:
    """Injected operations; ``move`` must obey non_blocking=False and preserve dtype.

    ``boundary_check`` is where the future adapter checks reference CPU residency,
    FSDP hidden gradients and completion of prior full update frames. Admission
    must check destination free memory plus measured reserve, before allocation.
    """

    describe: Callable[[Any], TensorInfo]
    scalar_value: Callable[[Any], int | float]
    move: Callable[..., Any]
    synchronize: Callable[[], None]
    no_grad: Callable[[], AbstractContextManager[Any]]
    boundary_check: Callable[[], None]
    admit_transfer: Callable[[str, int], None]


@dataclass(frozen=True)
class Slot:
    # A state dictionary/key descriptor, never a captured moment tensor.
    role: str
    parameter_index: int
    state: MutableMapping[str, Any]
    key: str
    info: TensorInfo


@dataclass
class Inventory:
    slots: list[Slot]
    signature: tuple[Any, ...]
    moments: dict[str, int]
    steps: dict[str, int]
    parameters: dict[str, int]
    largest_storage: int

    def summary(self) -> dict[str, Any]:
        return {
            "moment_bytes_by_device": dict(self.moments),
            "step_bytes_by_device": dict(self.steps),
            "parameter_bytes_by_device": dict(self.parameters),
            "largest_single_storage_bytes": self.largest_storage,
            "moment_slots": len(self.slots),
        }


class ExchangeCoordinator:
    """One PID/rank, one actor and critic; failure is permanent, without rollback."""

    def __init__(
        self,
        *,
        identity: str,
        operations: Operations,
        active_role: str = "actor",
        gpu_device: str = "cuda:0",
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        if not identity or active_role not in ROLES or (world_size, rank) != (1, 0):
            raise ExchangeError("identity, role or single-rank contract invalid")
        if not gpu_device.startswith("cuda:"):
            raise ExchangeError("an explicit CUDA device index is required")
        self.identity = identity
        self.operations = operations
        self.active_role = active_role
        self.gpu_device = gpu_device
        self.pid = os.getpid()
        self.failed = False
        self.transitioning = False
        self.failure_type: str | None = None
        self.events: list[dict[str, Any]] = []
        self._roles: dict[str, tuple[Any, Any]] = {}

    def _emit(self, event: str, **fields: Any) -> None:
        # Callers supply scalars/containers only; no tensor/exception references.
        self.events.append(
            {
                "event": event,
                "time_ns": time.time_ns(),
                "pid": self.pid,
                "active_role": self.active_role,
                **fields,
            }
        )

    def _fail(self, error: BaseException) -> None:
        self.failed = True
        self.failure_type = type(error).__name__
        # Do not retain the exception/traceback, which may own old tensors.
        self.events.append(
            {
                "event": "failed",
                "time_ns": time.time_ns(),
                "pid": self.pid,
                "active_role": self.active_role,
                "error_type": self.failure_type,
            }
        )

    def _guard(self, identity: str) -> None:
        if self.failed:
            raise ExchangeError("coordinator failed; compute/save/retry is prohibited")
        if self.pid != os.getpid() or identity != self.identity or self.transitioning:
            raise ExchangeError("PID, identity or non-reentrant boundary contract invalid")

    def register(self, role: str, module: Any, optimizer: Any, *, identity: str) -> None:
        try:
            self._guard(identity)
            if role not in ROLES:
                raise ExchangeError("only actor/critic may register")
            if role in self._roles:
                old_module, old_optimizer = self._roles[role]
                if old_module is module and old_optimizer is optimizer:
                    return
                raise ExchangeError("a distinct object already owns this role")
            if any(module is m or optimizer is o for m, o in self._roles.values()):
                raise ExchangeError("actor and critic cannot share module/optimizer")
            self._roles[role] = (module, optimizer)
            self._emit("registered", role=role)
        except BaseException as error:
            self._fail(error)
            raise

    def _inventory(self, active_role: str) -> Inventory:
        if set(self._roles) != set(ROLES):
            raise ExchangeError("both roles must register before use")
        slots: list[Slot] = []
        signature: list[Any] = []
        moments = {"cpu": 0, self.gpu_device: 0}
        steps = dict(moments)
        parameters = dict(moments)
        storages: set[tuple[str, int]] = set()
        parameter_ids: set[int] = set()
        largest = 0

        def independent(info: TensorInfo) -> None:
            key = (info.device, info.storage_id)
            if key in storages:
                raise ExchangeError("unsupported shared storage")
            storages.add(key)

        for role in ROLES:
            module, optimizer = self._roles[role]
            params = list(module.parameters())  # Parameter references are allowed.
            expected_ids = {id(parameter) for parameter in params}
            if len(expected_ids) != len(params) or parameter_ids & expected_ids:
                raise ExchangeError("duplicate/shared training parameter")
            parameter_ids.update(expected_ids)
            grouped_ids: list[int] = []
            for group in optimizer.param_groups:
                for option in ("foreach", "fused", "capturable", "amsgrad"):
                    if group.get(option) is not False:
                        raise ExchangeError(f"{option} must be explicitly false")
                grouped_ids.extend(id(parameter) for parameter in group["params"])
            if set(grouped_ids) != expected_ids or len(grouped_ids) != len(params):
                raise ExchangeError("full fine-tuning optimizer parameter membership changed")
            if any(id(parameter) not in expected_ids for parameter in optimizer.state):
                raise ExchangeError("optimizer state contains an unknown parameter")
            for index, parameter in enumerate(params):
                if parameter.grad is not None:
                    raise ExchangeError("training gradient remains; never clear it to pass")
                param_info = self.operations.describe(parameter)
                if param_info.device != self.gpu_device or param_info.dtype != "float32":
                    raise ExchangeError("training parameters must be FP32 on the bound GPU")
                independent(param_info)
                parameters[param_info.device] += param_info.storage_bytes
                state = optimizer.state.get(parameter)
                signature.append((role, id(parameter), tuple(sorted(state)) if state else ()))
                if not state:
                    continue  # Adam's first-use state is legitimately empty.
                if not isinstance(state, MutableMapping) or set(state) != {*MOMENTS, "step"}:
                    raise ExchangeError("only complete Adam exp_avg/exp_avg_sq/step supported")
                step_info = self.operations.describe(state["step"])
                if step_info.shape != () or step_info.device != "cpu":
                    raise ExchangeError("Adam step must remain its original CPU scalar")
                independent(step_info)
                steps[step_info.device] += step_info.storage_bytes
                signature.append(
                    (
                        role,
                        index,
                        "step",
                        id(state["step"]),
                        step_info,
                        self.operations.scalar_value(state["step"]),
                    )
                )
                for key in MOMENTS:
                    info = self.operations.describe(state[key])
                    expected_device = self.gpu_device if role == active_role else "cpu"
                    self._validate_moment(info, expected_device, param_info.shape)
                    independent(info)
                    largest = max(largest, info.storage_bytes)
                    moments[info.device] += info.storage_bytes
                    slots.append(Slot(role, index, state, key, info))
                    signature.append(
                        (
                            role,
                            index,
                            key,
                            info.shape,
                            info.dtype,
                            info.storage_bytes,
                        )
                    )
        return Inventory(slots, tuple(signature), moments, steps, parameters, largest)

    @staticmethod
    def _validate_moment(info: TensorInfo, device: str, shape: tuple[int, ...]) -> None:
        if info.dtype != "float32" or info.shape != shape or info.device != device:
            raise ExchangeError("moment dtype, shape or device mismatch")
        if (
            not info.contiguous
            or info.storage_offset != 0
            or info.is_view
            or info.storage_bytes != info.payload_bytes
            or info.storage_bytes <= 0
        ):
            raise ExchangeError("moment must own its dense, contiguous, complete storage")
        if info.device == "cpu" and info.pinned:
            raise ExchangeError("CPU moments must be pageable, never pinned")

    def assert_ready(self, role: str, *, identity: str) -> dict[str, Any]:
        """Required admission gate for subsequent compute/save; never repairs state."""
        try:
            self._guard(identity)
            if role != self.active_role:
                raise ExchangeError("requested role is not active")
            self.operations.boundary_check()
            return self._inventory(self.active_role).summary()
        except BaseException as error:
            self._fail(error)
            raise

    def _move_one(self, slot: Slot, destination: str, live: dict[str, int]) -> None:
        size, source = slot.info.storage_bytes, slot.info.device
        self.operations.admit_transfer(destination, size)
        if self.failed:
            raise ExchangeError("coordinator was failed during admission")
        self._emit(
            "move_start",
            role=slot.role,
            parameter_index=slot.parameter_index,
            key=slot.key,
            source=source,
            destination=destination,
            bytes=size,
            live_moment_bytes=dict(live),
        )
        old = slot.state[slot.key]
        replacement = None
        try:
            if self.operations.describe(old) != slot.info:
                raise ExchangeError("source moment changed after inventory")
            old_weak = weakref.ref(old)
            replacement = self.operations.move(old, destination, non_blocking=False)
            if self.failed:
                raise ExchangeError("coordinator was failed during copy")
            if replacement is old:
                raise ExchangeError("move retained source tensor identity")
            replacement_info = self.operations.describe(replacement)
            self._validate_moment(replacement_info, destination, slot.info.shape)
            if replacement_info.storage_bytes != size:
                raise ExchangeError("move changed storage size")
            transient = dict(live)
            transient[destination] += size
            self._emit("copy_complete", live_moment_bytes=transient)
            slot.state[slot.key] = replacement
            # Release locals before the next copy, and test object lifetime.
            del old
            old = None
            del replacement
            replacement = None
            if old_weak() is not None:
                raise ExchangeError("replaced moment retained externally; live bound invalid")
            live[source] -= size
            live[destination] += size
            self._emit(
                "move_released",
                role=slot.role,
                parameter_index=slot.parameter_index,
                key=slot.key,
                source=source,
                destination=destination,
                bytes=size,
                live_moment_bytes=dict(live),
            )
        finally:
            # Tracebacks from our own failure must not keep old/new tensor locals.
            del old
            del replacement

    def activate(self, role: str, *, identity: str) -> dict[str, Any]:
        try:
            self._guard(identity)
            if role not in ROLES:
                raise ExchangeError("unknown target role")
            self.transitioning = True
            self.operations.boundary_check()
            before = self._inventory(self.active_role)
            if role == self.active_role:
                return before.summary()
            self.operations.synchronize()
            self._emit("exchange_start", target_role=role, **before.summary())
            outgoing = [slot for slot in before.slots if slot.role == self.active_role]
            incoming = [slot for slot in before.slots if slot.role == role]
            out_index = in_index = delta = 0
            live = dict(before.moments)
            with self.operations.no_grad():
                while out_index < len(outgoing) or in_index < len(incoming):
                    if out_index < len(outgoing) and (in_index == len(incoming) or delta <= 0):
                        slot = outgoing[out_index]
                        self._move_one(slot, "cpu", live)
                        delta += slot.info.storage_bytes
                        out_index += 1
                    else:
                        slot = incoming[in_index]
                        self._move_one(slot, self.gpu_device, live)
                        delta -= slot.info.storage_bytes
                        in_index += 1
                    if self.failed:
                        raise ExchangeError("transition failed during callback")
            self.operations.synchronize()
            self.operations.boundary_check()
            after = self._inventory(role)
            if after.signature != before.signature or after.moments != live:
                raise ExchangeError("state keys, scalar steps, moment metadata or bytes changed")
            self.active_role = role
            self._emit("exchange_complete", delta_bytes=delta, **after.summary())
            return after.summary()
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            self.transitioning = False
