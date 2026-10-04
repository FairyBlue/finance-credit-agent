"""CPU-only tests for the explicitly leased GPU reservation controller."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.agents.gpu_reservation import GIB, Allocation, Reservation, TorchBackend


class FakeOOM(RuntimeError):
    pass


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Backend:
    def __init__(self, *, total=100 * GIB, external=20 * GIB, rounding=1):
        self.total, self.external, self.rounding = total, external, rounding
        self.live = {}
        self.cache = 0
        self.requests = []
        self.reclaimed = []
        self.error = None

    def allocation_bytes(self, size):
        return ((size + self.rounding - 1) // self.rounding) * self.rounding

    def snapshot(self):
        allocated = sum(block.nbytes for block in self.live.values())
        reserved = sum(self.allocation_bytes(block.nbytes) for block in self.live.values())
        reserved += self.cache
        return dict(free_bytes=max(0, self.total - self.external - reserved),
                    total_bytes=self.total, allocator_allocated_bytes=allocated,
                    allocator_reserved_bytes=reserved)

    def allocate(self, size):
        self.requests.append(size)
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        if self.snapshot()["free_bytes"] < self.allocation_bytes(size):
            raise FakeOOM("concurrent allocation")
        block = Allocation(size, object())
        self.live[id(block)] = block
        return block

    def release(self, blocks):
        for block in blocks:
            block.payload = None
            self.live.pop(id(block))
        self.reclaimed.append([block.nbytes for block in blocks])
        assert all(block.payload is None for block in blocks)
        blocks.clear()
        self.cache = 0  # models empty_cache after dropping every selected reference

    @staticmethod
    def is_oom(error):
        return isinstance(error, FakeOOM)


def make(**kwargs):
    backend = Backend()
    clock = Clock()
    reservation = Reservation(backend, target_bytes=3 * GIB, clock=clock, **kwargs)
    return reservation, backend, clock


def test_one_block_per_tick_and_no_growth_when_at_target():
    reservation, backend, _ = make()
    assert not backend.requests
    for count in range(1, 4):
        row = reservation.tick()
        assert row["payload_bytes"] == count * GIB and len(backend.requests) == count
    assert row["status"] == "HOLDING"
    for _ in range(3):
        assert reservation.tick()["status"] == "HOLDING"
    assert len(backend.requests) == 3


def test_free_fluctuation_waits_with_reserve_then_grows():
    reservation, backend, _ = make()
    backend.external = 98 * GIB
    assert reservation.tick()["status"] == "WAITING_FOR_MEMORY"
    assert not backend.requests
    backend.external = 97 * GIB
    assert reservation.tick()["payload_bytes"] == GIB
    assert reservation.tick()["status"] == "WAITING_FOR_MEMORY"
    backend.external = 94 * GIB
    assert reservation.tick()["payload_bytes"] == 2 * GIB
    assert reservation.tick()["payload_bytes"] == 3 * GIB


def test_partial_final_block_and_allocator_rounding_cap():
    backend, clock = Backend(rounding=GIB), Clock()
    reservation = Reservation(backend, target_bytes=GIB + GIB // 2,
                              maximum_bytes=2 * GIB, clock=clock)
    reservation.tick()
    row = reservation.tick()
    assert row["payload_bytes"] == GIB + GIB // 2
    assert row["allocator_reserved_bytes"] == 2 * GIB
    assert row["status"] == "HOLDING"
    reservation.set_target(2 * GIB)
    assert reservation.tick()["status"] == "WAITING_FOR_MEMORY"
    assert backend.requests == [GIB, GIB // 2]


def test_cached_allocator_bytes_are_trimmed_before_cap_and_driver_admission():
    reservation, backend, _ = make(maximum_bytes=3 * GIB)
    backend.cache = 50 * GIB
    row = reservation.tick()
    assert row["payload_bytes"] == GIB
    assert row["allocator_reserved_bytes"] == GIB and backend.cache == 0


def test_pause_release_does_not_immediately_reacquire_and_resume_is_explicit():
    reservation, backend, _ = make()
    for _ in range(3):
        reservation.tick()
    assert reservation.pause()["status"] == "PAUSED"
    count = len(backend.requests)
    assert reservation.tick()["payload_bytes"] == 3 * GIB
    row = reservation.release(GIB // 2)
    assert row["released_payload_bytes"] == GIB and row["payload_bytes"] == 2 * GIB
    assert row["allocator_reserved_bytes"] == 2 * GIB
    reservation.set_target(3 * GIB)
    assert reservation.tick()["status"] == "PAUSED" and len(backend.requests) == count
    reservation.set_target(3 * GIB, resume=True)
    assert reservation.tick()["status"] == "HOLDING" and len(backend.requests) == count + 1


def test_target_lowering_releases_whole_blocks_and_pauses():
    reservation, _, _ = make()
    for _ in range(3):
        reservation.tick()
    row = reservation.set_target(GIB + GIB // 2)
    assert row["status"] == "PAUSED" and row["payload_bytes"] == GIB
    reservation.set_target(GIB + GIB // 2, resume=True)
    assert reservation.tick()["payload_bytes"] == GIB + GIB // 2


def test_expiry_releases_all_and_target_changes_cannot_extend_lease():
    reservation, backend, clock = make(ttl_seconds=20)
    reservation.tick()
    clock.now = 10
    reservation.set_target(4 * GIB, resume=True)
    assert reservation.status()["remaining_seconds"] == 10
    clock.now = 20
    row = reservation.tick()
    assert row["status"] == "EXPIRED" and row["closed"]
    assert row["payload_bytes"] == row["allocator_reserved_bytes"] == 0
    assert not backend.live
    with pytest.raises(RuntimeError, match="closed"):
        reservation.set_target(GIB, resume=True)
    assert reservation.close()["status"] == "EXPIRED"


def test_expired_lease_cannot_resume_without_tick():
    reservation, backend, clock = make(ttl_seconds=10)
    reservation.tick()
    clock.now = 11
    with pytest.raises(RuntimeError, match="expired"):
        reservation.set_target(4 * GIB, resume=True)
    assert not backend.live and reservation.closed


def test_slow_lazy_initialization_cannot_allocate_after_lease_expiry():
    reservation, backend, clock = make(ttl_seconds=10)
    snapshot = backend.snapshot

    def slow_initialization():
        clock.now = 15
        return snapshot()

    backend.snapshot = slow_initialization
    row = reservation.tick()
    assert row["status"] == "EXPIRED" and row["closed"]
    assert not backend.requests and not backend.live


def test_allocation_finishing_after_lease_expiry_is_immediately_released():
    reservation, backend, clock = make(ttl_seconds=10)
    allocate = backend.allocate

    def slow_allocate(size):
        block = allocate(size)
        clock.now = 15
        return block

    backend.allocate = slow_allocate
    row = reservation.tick()
    assert row["status"] == "EXPIRED" and row["payload_bytes"] == 0
    assert backend.requests == [GIB] and not backend.live


def test_close_and_release_all_are_idempotent_and_zero_release_pauses():
    reservation, backend, _ = make()
    reservation.tick()
    assert reservation.release(0)["status"] == "PAUSED"
    assert reservation.release(100 * GIB)["payload_bytes"] == 0
    assert not backend.live
    assert reservation.close()["status"] == "CLOSED"
    reclaim_count = len(backend.reclaimed)
    assert reservation.close()["closed"]
    assert len(backend.reclaimed) == reclaim_count
    assert reservation.tick()["status"] == "CLOSED"


def test_only_cuda_oom_backs_off_and_retains_existing_blocks():
    reservation, backend, clock = make(oom_backoff_seconds=10)
    reservation.tick()
    backend.error = FakeOOM("allocation race")
    row = reservation.tick()
    assert row["status"] == "BACKOFF" and row["payload_bytes"] == GIB
    assert row["oom_count"] == 1 and not row["closed"]
    for now in (1, 5, 9):
        clock.now = now
        assert reservation.tick()["status"] == "BACKOFF"
    assert len(backend.requests) == 2
    clock.now = 10
    assert reservation.tick()["payload_bytes"] == 2 * GIB


def test_unknown_error_fails_closed_and_releases_prior_allocations():
    reservation, backend, _ = make()
    reservation.tick()
    backend.error = RuntimeError("CUDA illegal access, not OOM")
    with pytest.raises(RuntimeError, match="illegal access"):
        reservation.tick()
    row = reservation.status()
    assert row["status"] == "FAILED" and row["closed"]
    assert row["payload_bytes"] == row["allocator_reserved_bytes"] == 0
    assert not backend.live and "illegal access" in row["error"]


def test_actual_allocator_overrun_releases_new_and_existing_blocks():
    reservation, backend, _ = make(maximum_bytes=3 * GIB)
    reservation.tick()
    original = backend.allocate

    def underestimated(size):
        block = original(size)
        backend.cache = 2 * GIB
        return block

    backend.allocate = underestimated
    with pytest.raises(RuntimeError, match="actual allocator"):
        reservation.tick()
    assert reservation.closed and not backend.live and backend.cache == 0


@pytest.mark.parametrize("kwargs", [
    {"target_bytes": 61 * GIB}, {"maximum_bytes": 61 * GIB}, {"maximum_bytes": 0},
    {"block_bytes": 0}, {"block_bytes": True}, {"reserve_bytes": -1},
    {"ttl_seconds": float("inf")}, {"ttl_seconds": 0}, {"oom_backoff_seconds": 0},
])
def test_invalid_budgets_do_not_touch_backend(kwargs):
    backend = Backend()
    options = {"target_bytes": GIB, **kwargs}
    with pytest.raises(ValueError):
        Reservation(backend, **options)
    assert not backend.requests and not backend.reclaimed


def test_torch_backend_drops_all_references_before_empty_cache():
    backend = TorchBackend()
    blocks = [Allocation(4, object()), Allocation(8, object())]
    retained = list(blocks)

    class DeviceContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class CUDA:
        calls = 0

        @staticmethod
        def device(index):
            assert index == 0
            return DeviceContext()

        @classmethod
        def empty_cache(cls):
            assert blocks == [] and all(block.payload is None for block in retained)
            cls.calls += 1

    class Torch:
        cuda = CUDA

    backend._torch = Torch()
    backend.release(blocks)
    assert CUDA.calls == 1


def test_import_and_backend_construction_do_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys,json; from src.agents.gpu_reservation import "
         "TorchBackend; backend=TorchBackend(); print(json.dumps('torch' in sys.modules))"],
        cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True,
    )
    assert json.loads(result.stdout) is False
