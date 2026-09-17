"""A shared-GPU queue must preserve other jobs and never retry training."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "common_penalty_queue",
    Path(__file__).resolve().parents[1]
    / "scripts/diagnostics/m2_common_validity_penalty_queue.py",
)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


@pytest.fixture
def prepared(tmp_path):
    (tmp_path / "run_contract.json").write_text("{}")
    return tmp_path


def test_busy_gpu_times_out_without_launch_or_kill(prepared, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        queue.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(queue, "memory_used", lambda: 17093)
    monkeypatch.setattr(queue.os, "execve", lambda *args: pytest.fail("busy GPU must not launch"))
    assert queue.queue(prepared, wait_seconds=3, poll_seconds=1) == 2
    state = json.loads((prepared / "gpu_queue_state.json").read_text())
    assert state["status"] == "WAIT_TIMEOUT" and state["waited_seconds"] == 3
    assert state["memory_used_MiB"] == 17093
    assert not state["automatic_retry"] and not (prepared / "state.json").exists()


def test_idle_gpu_execs_fixed_run_once(prepared, monkeypatch):
    class Executed(BaseException):
        pass

    calls = []
    monkeypatch.setattr(queue, "memory_used", lambda: 4)
    monkeypatch.setattr(queue.os, "chdir", lambda path: None)

    def execute(executable, arguments, environment):
        calls.append((executable, arguments, environment))
        raise Executed

    monkeypatch.setattr(queue.os, "execve", execute)
    with pytest.raises(Executed):
        queue.queue(prepared)
    state = json.loads((prepared / "gpu_queue_state.json").read_text())
    assert state["status"] == "STARTING" and len(calls) == 1
    assert calls[0][1][3:5] == ["src.agents.common_validity_penalty", "run"]
    assert calls[0][1][-1] == str(prepared)
    with pytest.raises(RuntimeError, match="no implicit retry"):
        queue.queue(prepared)


def test_gpu_query_failure_is_recorded_and_does_not_launch(prepared, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(queue.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        queue.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def fail():
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(queue, "memory_used", fail)
    monkeypatch.setattr(
        queue.os, "execve", lambda *args: pytest.fail("unknown GPU must not launch")
    )
    assert queue.queue(prepared, wait_seconds=3, poll_seconds=1) == 2
    state = json.loads((prepared / "gpu_queue_state.json").read_text())
    assert state["status"] == "WAIT_TIMEOUT"
    assert state["telemetry_error_count"] == 3
    assert "telemetry unavailable" in state["last_telemetry_error"]


def test_missing_seal_is_rejected_before_polling(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "memory_used", lambda: pytest.fail("unsealed run must not poll"))
    with pytest.raises(RuntimeError, match="seal"):
        queue.queue(tmp_path)
