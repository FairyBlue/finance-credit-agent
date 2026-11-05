"""GPU occupancy violations stop only the launched inference worker."""

import json
import subprocess

import pytest

from src.agents import order_recovery_gpu_monitor as monitor


class Process:
    def __init__(self, *, complete=False):
        self.complete = complete
        self.terminated = False

    def poll(self):
        return 0 if self.complete or self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        return 0


def snapshot(used):
    return {
        "gpu_uuid": "GPU-test",
        "driver_total_bytes": 94 * monitor.GIB,
        "driver_free_bytes": (94 - used) * monitor.GIB,
    }


def test_success_records_sample_without_claiming_process_attribution(tmp_path, monkeypatch):
    process = Process(complete=True)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(monitor, "_driver_snapshot", lambda: snapshot(33))
    output = tmp_path / "gpu.jsonl"
    result = monitor.run_monitored(
        ["worker"], config={"gpu_uuid": "GPU-test"}, evidence_path=output, timeout=10
    )
    assert result.returncode == 0
    row = json.loads(output.read_text())
    assert row["sampled_peak_bytes"] == 33 * monitor.GIB
    assert row["process_attribution"] is False


def test_budget_violation_terminates_own_worker_and_preserves_samples(tmp_path, monkeypatch):
    process = Process()
    snapshots = iter([snapshot(33), snapshot(81)])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(monitor, "_driver_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    output = tmp_path / "gpu.jsonl"
    with pytest.raises(RuntimeError, match="80 GiB"):
        monitor.run_monitored(
            ["worker"], config={"gpu_uuid": "GPU-test"}, evidence_path=output, timeout=10
        )
    assert process.terminated
    assert len(output.read_text().splitlines()) == 2


def test_timeout_terminates_own_worker(tmp_path, monkeypatch):
    process = Process()
    times = iter([0, 0, 11])
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(monitor, "_driver_snapshot", lambda: snapshot(33))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)
    monkeypatch.setattr(monitor.time, "monotonic", lambda: next(times))
    with pytest.raises(subprocess.TimeoutExpired):
        monitor.run_monitored(
            ["worker"],
            config={"gpu_uuid": "GPU-test"},
            evidence_path=tmp_path / "gpu.jsonl",
            timeout=10,
        )
    assert process.terminated


def test_unavailable_budget_does_not_start_worker(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(monitor, "_driver_snapshot", lambda: snapshot(81))
    with pytest.raises(RuntimeError, match="80 GiB"):
        monitor.run_monitored(
            ["worker"],
            config={"gpu_uuid": "GPU-test"},
            evidence_path=tmp_path / "gpu.jsonl",
            timeout=10,
        )
    assert not calls
