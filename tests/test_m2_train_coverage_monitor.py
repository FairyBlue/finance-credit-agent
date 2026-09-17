"""The service outcome and fresh complete evidence must independently agree."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "coverage_monitor_test", ROOT / "scripts/diagnostics/m2_train_coverage_monitor.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def outcome():
    return {
        "status": "COMPLETE", "execution_status": "COMPLETE", "problems": [], "pending": [],
        "totals": {
            "planned": 308, "observed_unique_expected": 308, "terminal_complete": 308,
            "incomplete_observed": 0, "missing": 0,
        },
    }


def service():
    # ExecMainCode is deliberately irrelevant: systemd exposes it differently across versions.
    return dict(ActiveState="inactive", Result="success", ExecMainStatus="0", ExecMainCode="0")


def write_evidence(base, readout):
    (base / "state.json").write_text('{"status":"COMPLETE"}')
    (base / "readout.json").write_text(json.dumps(readout))
    (base / "FINAL_RESULTS.md").write_text("Complete fixed population\n")


@pytest.mark.parametrize("final_refresh_complete", [True, False])
def test_monitor_refreshes_after_service_finishes(tmp_path, monkeypatch, final_refresh_complete):
    calls = []

    def refresh(base):
        calls.append(base)
        readout = outcome()
        if len(calls) == 1 or not final_refresh_complete:
            readout.update(status="INCOMPLETE", execution_status="RUNNING", pending=["running"])
            readout["totals"]["terminal_complete"] = 300
        write_evidence(base, readout)
        return SimpleNamespace(returncode=0, stdout=f"snapshot {len(calls)}", stderr="")

    monkeypatch.setattr(entry, "refresh", refresh)
    monkeypatch.setattr(entry, "service_status", lambda unit: service())
    monkeypatch.setattr(entry.time, "sleep", lambda seconds: pytest.fail("must terminate now"))
    assert entry.monitor(tmp_path, "mock.service") == (0 if final_refresh_complete else 2)
    assert calls == [tmp_path, tmp_path]
    state = json.loads((tmp_path / "monitor_state.json").read_text())
    assert state["status"] == ("COMPLETE" if final_refresh_complete else "STOPPED_EARLY")
    assert state["final_readout_output"] == "snapshot 2"
    assert state["started_at_SGT"].endswith("+08:00")


@pytest.mark.parametrize("damage", [
    "failed_service", "exit_status", "readout_command", "controller", "readout_status",
    "count", "problems", "missing_pending", "missing_final",
])
def test_complete_artifacts_do_not_override_failure(tmp_path, monkeypatch, damage):
    readout, main = outcome(), service()
    rc = 0
    if damage == "failed_service":
        main["ActiveState"] = "failed"
    elif damage == "exit_status":
        main["ExecMainStatus"] = "2"
    elif damage == "readout_command":
        rc = 1
    elif damage == "readout_status":
        readout["status"] = "INCOMPLETE"
    elif damage == "count":
        readout["totals"]["terminal_complete"] = 307
    elif damage == "problems":
        readout["problems"] = ["bad identity"]
    elif damage == "missing_pending":
        del readout["pending"]
    write_evidence(tmp_path, readout)
    if damage == "controller":
        (tmp_path / "state.json").write_text('{"status":"FAILED"}')
    if damage == "missing_final":
        (tmp_path / "FINAL_RESULTS.md").unlink()
    calls = []

    def refresh(base):
        calls.append(base)
        return SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(entry, "refresh", refresh)
    monkeypatch.setattr(entry, "service_status", lambda unit: main)
    assert entry.monitor(tmp_path, "mock.service") == 2
    assert len(calls) == 2
    state = json.loads((tmp_path / "monitor_state.json").read_text())
    assert state["status"] == "STOPPED_EARLY" and state["completion_problems"]


def test_active_service_waits_45_seconds_and_timeout_does_not_restart(tmp_path, monkeypatch):
    elapsed, sleeps, commands = [0], [], []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "/usr/bin/python3":
            assert command[1:] == [str(entry.READER), "--directory", str(tmp_path), "--write"]
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert command[:4] == ["systemctl", "--user", "show", "mock.service"]
        return SimpleNamespace(returncode=0, stdout="ActiveState=active\n", stderr="")

    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] = entry.TIMEOUT_SECONDS

    monkeypatch.setattr(entry.subprocess, "run", run)
    monkeypatch.setattr(entry.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(entry.time, "sleep", sleep)
    assert entry.monitor(tmp_path, "mock.service") == 2
    assert sleeps == [45] and len(commands) == 2
    assert entry.TIMEOUT_SECONDS == 7 * 3600
    state = json.loads((tmp_path / "monitor_state.json").read_text())
    assert state["status"] == "STOPPED_EARLY"
    assert "timeout" in state["error"]


def test_readout_exception_stops_without_retry(tmp_path, monkeypatch):
    calls = []

    def refresh(base):
        calls.append(base)
        raise RuntimeError("readout unavailable")

    monkeypatch.setattr(entry, "refresh", refresh)
    assert entry.monitor(tmp_path, "mock.service") == 2
    assert calls == [tmp_path]
    state = json.loads((tmp_path / "monitor_state.json").read_text())
    assert state["status"] == "STOPPED_EARLY"
    assert "readout unavailable" in state["error"]
