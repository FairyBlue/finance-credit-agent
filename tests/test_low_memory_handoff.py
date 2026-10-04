"""Controller-side admission must account for the actual, releasable holder."""

from __future__ import annotations

import json

import pytest

from src.agents import gpu_handoff
from src.agents import low_memory_pilot as entry

GIB = 1024**3


class Clock:
    def __init__(self):
        self.value = 0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def telemetry(free, held):
    return dict(driver_free_bytes=free * GIB, holder_payload_bytes=held * GIB,
                available_bytes=(free + held) * GIB, minimum_free_bytes=60 * GIB)


def test_holder_is_counted_without_releasing_in_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "handoff_config", lambda base: {"fixture": True})
    monkeypatch.setattr(gpu_handoff, "available_bytes", lambda config: telemetry(3, 59))
    monkeypatch.setattr(entry, "gpu_free_mib",
                        lambda: pytest.fail("use verified handoff telemetry"))
    entry.wait_gpu(tmp_path, "m1", timeout=10)
    result = json.loads((tmp_path / "gpu_queue_state.json").read_text())
    assert result["status"] == "ADMITTED"
    assert result["free_MiB"] == 3 * 1024
    assert result["holder_payload_bytes"] == 59 * GIB


def test_actual_small_hold_does_not_admit_large_target(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "handoff_config", lambda base: {"target_bytes": 59 * GIB})
    monkeypatch.setattr(gpu_handoff, "available_bytes", lambda config: telemetry(3, 4))
    monkeypatch.setattr(entry, "time", Clock())
    with pytest.raises(RuntimeError, match="queue limit"):
        entry.wait_gpu(tmp_path, "m1", timeout=5)
    result = json.loads((tmp_path / "gpu_queue_state.json").read_text())
    assert result["status"] == "WAIT_TIMEOUT" and result["available_bytes"] == 7 * GIB


def test_dead_holder_fails_without_twelve_hour_polling(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "handoff_config", lambda base: {"fixture": True})
    def unavailable(config):
        raise RuntimeError("holder expired")
    monkeypatch.setattr(gpu_handoff, "available_bytes", unavailable)
    with pytest.raises(RuntimeError, match="expired"):
        entry.wait_gpu(tmp_path, "m2_eta025", timeout=100)
    result = json.loads((tmp_path / "gpu_queue_state.json").read_text())
    assert result["status"] == "HANDOFF_UNAVAILABLE"


def test_paused_empty_holder_admits_next_arm_on_actual_free(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "handoff_config", lambda base: {"fixture": True})
    monkeypatch.setattr(gpu_handoff, "available_bytes", lambda config: telemetry(61, 0))
    entry.wait_gpu(tmp_path, "m2_eta025", timeout=10)
    result = json.loads((tmp_path / "gpu_queue_state.json").read_text())
    assert result["status"] == "ADMITTED" and result["holder_payload_bytes"] == 0


def test_completed_run_requires_both_later_model_role_checks(tmp_path, monkeypatch):
    config = {"fixture": True}
    monkeypatch.setattr(entry, "handoff_config", lambda base: config)
    cell = tmp_path / "m1"
    directory = cell / "artifacts/gpu_handoff"
    directory.mkdir(parents=True)
    inactive = dict(paused=True, payload_bytes=0,
                    allocator_allocated_bytes=0, allocator_reserved_bytes=0)
    receipt = dict(status="PASSED", identity="a" * 64, holder_config=config,
                   first_role="critic", worker_pid=123,
                   worker_memory={"driver_free_bytes": 61 * GIB}, after=inactive,
                   role_checks=[dict(role=role, holder_status=inactive)
                                for role in ("reference", "actor")])
    path = directory / "receipt-123.json"
    entry.atomic_json(path, receipt)
    assert entry.validate_handoff(cell, "a" * 64)["status"] == "PASSED"
    receipt["role_checks"].pop()
    entry.atomic_json(path, receipt)
    with pytest.raises(RuntimeError, match="initialization evidence"):
        entry.validate_handoff(cell, "a" * 64)
