"""Leased, incremental GPU allocation without kernels or model work.

Only this process's allocator is inspected/reclaimed. CUDA context memory is
outside the allocator budget and must be allowed for by the caller. Importing
this module, or constructing TorchBackend, does not import torch or touch CUDA.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

GIB = 1024**3
MIB = 1024**2
HARD_MAXIMUM_BYTES = 60 * GIB
PERSISTENT_MAXIMUM_BYTES = 80 * GIB


@dataclass
class Allocation:
    nbytes: int
    payload: Any


class TorchBackend:
    """Dedicated-process allocator backend; retain empty uint8 tensors only."""

    def __init__(self, device: int = 0):
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative CUDA index")
        self.device = device
        self._torch: Any = None

    def _runtime(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def snapshot(self) -> dict[str, int]:
        torch = self._runtime()
        free, total = torch.cuda.mem_get_info(self.device)
        return dict(
            free_bytes=free,
            total_bytes=total,
            allocator_allocated_bytes=torch.cuda.memory_allocated(self.device),
            allocator_reserved_bytes=torch.cuda.memory_reserved(self.device),
        )

    def allocation_bytes(self, nbytes: int) -> int:
        # Conservative native caching-allocator segment sizes. GiB-sized blocks
        # are exact multiples; small probes may reserve a larger segment.
        if nbytes < MIB:
            return 2 * MIB
        if nbytes < 10 * MIB:
            return 20 * MIB
        return ((nbytes + 2 * MIB - 1) // (2 * MIB)) * (2 * MIB)

    def allocate(self, nbytes: int) -> Allocation:
        torch = self._runtime()
        return Allocation(nbytes, torch.empty(nbytes, dtype=torch.uint8,
                                              device=f"cuda:{self.device}"))

    def release(self, blocks: list[Allocation]) -> None:
        # Clear every tensor reference before empty_cache; merely deleting a
        # local variable or clearing the caller's list leaves allocator cache.
        for block in blocks:
            block.payload = None
        blocks.clear()
        torch = self._runtime()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def is_oom(self, error: BaseException) -> bool:
        return isinstance(error, self._runtime().cuda.OutOfMemoryError)


class Reservation:
    """Incremental reservation with a finite lease and explicit growth controls.

    ``tick`` allocates at most one block. Driver free memory must cover the
    allocator's estimated increment plus ``reserve_bytes``. CUDA OOM retains
    existing blocks and backs off. Other errors close/release and re-raise.
    ``release`` pauses growth; use ``set_target(..., resume=True)`` to restart.
    Released whole blocks can exceed the requested amount. Lease expiry cannot
    be reset by target changes or pauses. ``close`` is idempotent.
    """

    def __init__(
        self,
        backend: Any,
        target_bytes: int,
        maximum_bytes: int = HARD_MAXIMUM_BYTES,
        block_bytes: int = GIB,
        reserve_bytes: int = 2 * GIB,
        clock: Any = time.monotonic,
        ttl_seconds: float = 6 * 3600,
        oom_backoff_seconds: float = 30,
        persistent: bool = False,
    ):
        for name, value in (("target_bytes", target_bytes), ("maximum_bytes", maximum_bytes),
                            ("block_bytes", block_bytes), ("reserve_bytes", reserve_bytes)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(persistent) is not bool:
            raise ValueError("persistent must be explicitly boolean")
        limit = PERSISTENT_MAXIMUM_BYTES if persistent else HARD_MAXIMUM_BYTES
        if not 0 < maximum_bytes <= limit or target_bytes > maximum_bytes:
            raise ValueError(f"target/maximum exceeds the {limit // GIB} GiB allocator limit")
        if persistent and reserve_bytes < 2 * GIB:
            raise ValueError("persistent reservation requires at least 2 GiB free headroom")
        if not 0 < block_bytes <= maximum_bytes:
            raise ValueError("block_bytes must be positive and no greater than maximum")
        for name, value in (("ttl_seconds", ttl_seconds),
                            ("oom_backoff_seconds", oom_backoff_seconds)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.backend = backend
        self.target_bytes, self.maximum_bytes = target_bytes, maximum_bytes
        self.block_bytes, self.reserve_bytes = block_bytes, reserve_bytes
        self.clock, self.ttl_seconds = clock, ttl_seconds
        self.persistent = persistent
        self.oom_backoff_seconds = oom_backoff_seconds
        self.started = clock()
        self.deadline = None if persistent else self.started + ttl_seconds
        self._blocks: list[Allocation] = []
        self._snapshot: dict[str, int] = {}
        self._state = "GROWING"
        self._closed = False
        self._paused = False
        self._retry_at = 0.0
        self._oom_count = 0
        self._error: str | None = None
        self._cleanup_error: str | None = None

    @property
    def payload_bytes(self) -> int:
        return sum(block.nbytes for block in self._blocks)

    @property
    def closed(self) -> bool:
        return self._closed

    def _read_snapshot(self) -> dict[str, int]:
        row = self.backend.snapshot()
        required = {"free_bytes", "total_bytes", "allocator_allocated_bytes",
                    "allocator_reserved_bytes"}
        if not required <= row.keys() or any(type(row[k]) is not int or row[k] < 0
                                           for k in required):
            raise RuntimeError("invalid allocator/driver telemetry")
        if row["free_bytes"] > row["total_bytes"] or (
            row["allocator_allocated_bytes"] > row["allocator_reserved_bytes"]
        ):
            raise RuntimeError("inconsistent allocator/driver telemetry")
        self._snapshot = {key: row[key] for key in required}
        return self._snapshot

    def _fail(self, error: BaseException) -> None:
        self._state, self._closed, self._paused = "FAILED", True, True
        self._error = f"{type(error).__name__}: {error}"
        blocks, self._blocks = self._blocks, []
        try:
            self.backend.release(blocks)
            self._read_snapshot()
        except BaseException as cleanup_error:
            self._cleanup_error = f"{type(cleanup_error).__name__}: {cleanup_error}"
        finally:
            # A conforming backend already cleared payloads before cache reclaim.
            for block in blocks:
                block.payload = None
            blocks.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reservation is closed; create a new explicit lease")
        if self._expired():
            self._close("EXPIRED")
            raise RuntimeError("reservation lease expired")

    def status(self) -> dict[str, Any]:
        """Return cached telemetry; tick/control operations refresh CUDA reads."""
        return dict(
            status=self._state, closed=self._closed, paused=self._paused,
            target_bytes=self.target_bytes, maximum_bytes=self.maximum_bytes,
            payload_bytes=self.payload_bytes, blocks=len(self._blocks),
            block_bytes=self.block_bytes, reserve_bytes=self.reserve_bytes,
            **self._snapshot,
            elapsed_seconds=max(0.0, self.clock() - self.started),
            remaining_seconds=None if self.persistent else max(0.0, self.deadline - self.clock()),
            ttl_seconds=None if self.persistent else self.ttl_seconds,
            persistent=self.persistent,
            oom_count=self._oom_count,
            retry_in_seconds=max(0.0, self._retry_at - self.clock()),
            error=self._error, cleanup_error=self._cleanup_error,
            cuda_context_in_allocator_budget=False,
        )

    def tick(self) -> dict[str, Any]:
        if self._closed:
            return self.status()
        if self._expired():
            return self._close("EXPIRED")
        try:
            # Discard only this process's unoccupied cache before budgeting.
            self.backend.release([])
            telemetry = self._read_snapshot()
            # Lazy CUDA initialization/driver calls can outlast a short lease.
            if self._expired():
                return self._close("EXPIRED")
            if telemetry["allocator_reserved_bytes"] > self.maximum_bytes:
                raise RuntimeError("existing allocator residency exceeds maximum")
            if self._paused:
                self._state = "PAUSED"
                return self.status()
            if self.payload_bytes >= self.target_bytes:
                self._state = "HOLDING"
                return self.status()
            if self.clock() < self._retry_at:
                self._state = "BACKOFF"
                return self.status()
            nbytes = min(self.block_bytes, self.target_bytes - self.payload_bytes)
            estimate = getattr(self.backend, "allocation_bytes", lambda n: n)(nbytes)
            if type(estimate) is not int or estimate < nbytes:
                raise RuntimeError("invalid allocation budget estimate")
            if (
                telemetry["allocator_reserved_bytes"] + estimate > self.maximum_bytes
                or telemetry["free_bytes"] < estimate + self.reserve_bytes
            ):
                self._state = "WAITING_FOR_MEMORY"
                return self.status()
            if self._expired():
                return self._close("EXPIRED")
            try:
                self._blocks.append(self.backend.allocate(nbytes))
            except BaseException as error:
                if not self.backend.is_oom(error):
                    raise
                self._oom_count += 1
                self._retry_at = self.clock() + self.oom_backoff_seconds
                self.backend.release([])
                self._read_snapshot()
                self._state = "BACKOFF"
                return self.status()
            if self._blocks[-1].nbytes != nbytes or self._blocks[-1].payload is None:
                raise RuntimeError("backend returned an invalid allocation")
            telemetry = self._read_snapshot()
            if self._expired():
                return self._close("EXPIRED")
            if telemetry["allocator_reserved_bytes"] > self.maximum_bytes:
                raise RuntimeError("actual allocator residency exceeded its estimated maximum")
            self._state = "HOLDING" if self.payload_bytes >= self.target_bytes else "GROWING"
            self._error = None
            return self.status()
        except BaseException as error:
            self._fail(error)
            raise

    def pause(self) -> dict[str, Any]:
        self._require_open()
        self._paused, self._state = True, "PAUSED"
        return self.status()

    def _expired(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline

    def release(self, bytes_to_release: int) -> dict[str, Any]:
        self._require_open()
        if type(bytes_to_release) is not int or bytes_to_release < 0:
            raise ValueError("bytes_to_release must be a nonnegative integer")
        self._paused, self._state = True, "PAUSED"
        selected = []
        released = 0
        while self._blocks and released < bytes_to_release:
            selected.append(self._blocks.pop())
            released += selected[-1].nbytes
        try:
            self.backend.release(selected)
            self._read_snapshot()
            return {**self.status(), "released_payload_bytes": released}
        except BaseException as error:
            for block in selected:
                block.payload = None
            selected.clear()
            self._fail(error)
            raise

    def set_target(self, target_bytes: int, *, resume: bool = False) -> dict[str, Any]:
        self._require_open()
        if type(target_bytes) is not int or not 0 <= target_bytes <= self.maximum_bytes:
            raise ValueError("target_bytes must be within the allocator maximum")
        if type(resume) is not bool:
            raise ValueError("resume must be explicitly boolean")
        self.target_bytes = target_bytes
        if self.payload_bytes > target_bytes:
            self.release(self.payload_bytes - target_bytes)
        if resume:
            self._paused = False
        self._state = "PAUSED" if self._paused else (
            "HOLDING" if self.payload_bytes >= self.target_bytes else "GROWING"
        )
        return self.status()

    def _close(self, state: str) -> dict[str, Any]:
        if self._closed:
            return self.status()
        self._state, self._closed, self._paused = state, True, True
        blocks, self._blocks = self._blocks, []
        try:
            self.backend.release(blocks)
            self._read_snapshot()
        except BaseException as error:
            for block in blocks:
                block.payload = None
            blocks.clear()
            self._fail(error)
            raise
        return self.status()

    def close(self) -> dict[str, Any]:
        return self._close("CLOSED")
