"""CPU simulations of the device-wide stop guard and native capture contract."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from src.agents import action_pooling_gpu_monitor as monitor
from src.agents import gpu_memory_coordinator as memory
from src.agents import screening_launcher

GIB = memory.GIB
OWNER = dict(
    pid=os.getpid(),
    start_time_ticks=11,
    cgroup_path="/user.slice/test.service",
    process_group_id=22,
    session_id=33,
)
CHILD = dict(
    pid=9001,
    start_time_ticks=44,
    cgroup_path=OWNER["cgroup_path"],
    process_group_id=9001,
    session_id=9001,
)


def driver(used):
    return dict(
        gpu_uuid="GPU-test", driver_total_bytes=192 * GIB, driver_free_bytes=(192 - used) * GIB
    )


def holder(payload=0):
    return dict(
        status="PAUSED",
        paused=True,
        closed=False,
        training_connected=True,
        payload_bytes=payload * GIB,
        allocator_allocated_bytes=payload * GIB,
        allocator_reserved_bytes=payload * GIB,
        maximum_bytes=80 * GIB,
        error=None,
        cleanup_error=None,
        claim=dict(
            lease_token="test-lease",
            owner=copy.deepcopy(OWNER),
            cgroup_populated=True,
            watchdog_error=None,
        ),
    )


@pytest.fixture
def config(tmp_path):
    directory = tmp_path / "holder"
    directory.mkdir()
    owner = dict(
        pid=123,
        directory=str(directory),
        cuda_device=0,
        started_at_SGT="2026-10-07T12:00:00+08:00",
        expires_at_SGT=None,
        persistent=True,
        allocation_only=True,
        automatic_restart=False,
    )
    raw = json.dumps(owner).encode()
    (directory / "owner.json").write_bytes(raw)
    return dict(
        profile=memory.PROFILE,
        directory=str(directory),
        owner_sha256=hashlib.sha256(raw).hexdigest(),
        holder_pid=123,
        holder_start_time_ticks=456,
        gpu_uuid="GPU-test",
        cuda_device=0,
        lease_token="test-lease",
        evidence_directory=str(tmp_path / "coordinator"),
        portfolio_target_bytes=76 * GIB,
        phase_peak_bytes={name: 76 * GIB for name in memory.PHASES},
        minimum_free_bytes=2 * GIB,
        wait_timeout_seconds=1800,
        poll_seconds=1,
        active_arm_holder_policy="paused_zero",
    )


def test_verified_holder_explains_only_bounded_startup_and_activation_latches(config):
    guard = monitor.AggregateGuard(config, OWNER)
    row = guard.observe(driver(87), holder(76), holder(76), 1799)
    assert row["phase"] == "WAITING_HOLDER_RELEASE"
    assert row["aggregate_used_bytes"] == 87 * GIB
    assert row["startup_nonholder_used_bytes"] == 11 * GIB
    assert guard.observe(driver(80), holder(), holder(), 1799.5)["phase"] == "ACTIVE"
    assert guard.observe(driver(79), holder(), holder(), 9999)["phase"] == "ACTIVE"
    with pytest.raises(monitor.GuardViolation, match="allocated again"):
        guard.observe(driver(87), holder(76), holder(76), 10000)


@pytest.mark.parametrize(
    "payload,used,elapsed,message",
    [
        (76, 157, 0, "excluding verified holder"),
        (76, 87, 1800, "release wait expired"),
        (0, 20, 1800, "release wait expired"),
        (0, 81, 0, "80 GiB stop threshold"),
        (76, 11, 0, "consistent positive"),
    ],
)
def test_stop_conditions(config, payload, used, elapsed, message):
    with pytest.raises(monitor.GuardViolation, match=message):
        monitor.AggregateGuard(config, OWNER).observe(
            driver(used), holder(payload), holder(payload), elapsed
        )


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("state", "paused", False),
        ("state", "status", "FAILED"),
        ("state", "training_connected", False),
        ("state", "cleanup_error", "bad"),
        ("state", "allocator_reserved_bytes", -1),
        ("state", "payload_bytes", True),
        ("claim", "lease_token", "foreign-lease"),
        ("claim", "owner", {**OWNER, "pid": 123}),
        ("claim", "cgroup_populated", False),
        ("claim", "watchdog_error", "unreadable"),
        ("driver", "gpu_uuid", "GPU-other"),
        ("driver", "driver_free_bytes", -1),
        ("driver", "driver_total_bytes", True),
    ],
)
def test_invalid_identity_state_or_metrics_are_fatal(config, target, key, value):
    state, sample = holder(76), driver(87)
    (state["claim"] if target == "claim" else sample if target == "driver" else state)[key] = value
    with pytest.raises(monitor.GuardViolation):
        monitor.AggregateGuard(config, OWNER).observe(sample, state, holder(76), 1)


def test_partial_allocator_release_does_not_arm(config):
    partial = holder()
    partial["allocator_reserved_bytes"] = GIB
    with pytest.raises(monitor.GuardViolation, match="consistent positive"):
        monitor.AggregateGuard(config, OWNER).observe(driver(20), partial, partial, 1)


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now


class Process:
    pid = CHILD["pid"]

    def __init__(self, clock, *, exit_after=1, returncode=0, ignore_term=False):
        self.clock, self.exit_after, self.code = clock, exit_after, returncode
        self.waits = 0
        self.returncode = None
        self.ignore_term = ignore_term

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.clock.now += timeout
        self.waits += 1
        if self.returncode is not None:
            return self.returncode
        if self.exit_after is not None and self.waits >= self.exit_after:
            self.returncode = self.code
            return self.returncode
        raise subprocess.TimeoutExpired("trainer", timeout)


@pytest.fixture
def harness(tmp_path, config, monkeypatch):
    path = tmp_path / "config.json"
    digest = memory.freeze_config(path, config)
    cell = tmp_path / "cell"
    cell.mkdir()
    clock = Clock()
    process = Process(clock)
    rows = [driver(25)]
    states = [holder()]
    calls, signals = [], []

    def identity(pid):
        return copy.deepcopy(
            OWNER
            if pid == OWNER["pid"]
            else CHILD
            if pid == CHILD["pid"]
            else dict(pid=123, start_time_ticks=456)
        )

    def status(directory, payload):
        assert directory == config["directory"]
        assert payload == {"op": "status"}
        state = copy.deepcopy(states.pop(0) if len(states) > 1 else states[0])
        state.update(
            json.loads((Path(directory) / "owner.json").read_text()),
            coordination_profile="persistent_claim_v1",
        )
        return state

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    def killpg(pid, sig):
        assert pid == CHILD["pid"]
        signals.append(sig)
        if sig == signal.SIGKILL or not process.ignore_term:
            process.returncode = -sig

    monkeypatch.setattr(monitor, "process_identity", identity)
    monkeypatch.setattr(monitor, "request", status)
    monkeypatch.setattr(
        memory, "_driver_snapshot", lambda: rows.pop(0) if len(rows) > 1 else rows[0]
    )
    monkeypatch.setattr(monitor.time, "monotonic", clock)
    monkeypatch.setattr(monitor.subprocess, "Popen", popen)
    monkeypatch.setattr(monitor.os, "killpg", killpg)
    return dict(
        cell=cell,
        config=config,
        path=path,
        clock=clock,
        process=process,
        rows=rows,
        states=states,
        calls=calls,
        signals=signals,
        environment={
            "FINANCE_GPU_COORDINATOR_CONFIG": str(path),
            "FINANCE_GPU_COORDINATOR_CONFIG_SHA256": digest,
        },
    )


def run(h):
    return monitor.capture_process(
        ["trainer", "arg"], environment=h["environment"], cell_dir=h["cell"]
    )


def receipt(h):
    return json.loads((h["cell"] / "gpu_monitor_receipt.json").read_text())


def test_capture_signature_and_native_logs_are_compatible(harness):
    h = harness
    assert inspect.signature(monitor.capture_process) == inspect.signature(
        screening_launcher.capture_process
    )
    assert run(h) == 0
    assert {
        "trainer.log",
        "process.json",
        "resources.jsonl",
        "gpu_monitor.jsonl",
        "gpu_monitor_receipt.json",
    } <= {p.name for p in h["cell"].iterdir()}
    command, kwargs = h["calls"][0]
    assert command == ["trainer", "arg"]
    assert kwargs["start_new_session"] is True
    assert kwargs["env"] == h["environment"]
    assert kwargs["cwd"] == screening_launcher.REPO_ROOT
    assert not h["signals"]
    assert receipt(h)["status"] == "COMPLETE"
    assert receipt(h)["active_sampled_peak_bytes"] == 25 * GIB
    assert receipt(h)["instantaneous_memory_cap"] is False


def test_prestart_budget_failure_never_launches(harness):
    h = harness
    h["rows"][:] = [driver(81)]
    with pytest.raises(RuntimeError, match="80 GiB"):
        run(h)
    assert not h["calls"] and not h["signals"]
    assert receipt(h)["status"] == "FAILED"


def test_startup_handover_refreshes_stale_driver_then_checks_full_device(harness):
    h = harness
    # Initial sample retains 76 GiB. The next bracket sees release after its
    # stale 87 GiB driver read, so only a fresh 25 GiB read may arm the guard.
    h["states"][:] = [holder(76), holder(76), holder(76), holder(), holder()]
    h["rows"][:] = [driver(87), driver(87), driver(25)]
    assert run(h) == 0
    assert receipt(h)["sampled_peak_bytes"] == 87 * GIB
    assert receipt(h)["active_sampled_peak_bytes"] == 25 * GIB
    assert receipt(h)["startup_nonholder_sampled_peak_bytes"] == 11 * GIB


def test_active_cap_breach_stops_only_own_group_and_raises(harness):
    h = harness
    h["process"].exit_after = None
    h["rows"][:] = [driver(25), driver(81)]
    with pytest.raises(RuntimeError, match="aggregate GPU occupancy exceeded"):
        run(h)
    assert h["signals"] == [signal.SIGTERM]
    assert receipt(h)["status"] == "FAILED"
    assert receipt(h)["active_sampled_peak_bytes"] == 81 * GIB


def test_owned_group_cleanup_escalates_when_sigterm_is_ignored(harness):
    h = harness
    h["process"].exit_after = None
    h["process"].ignore_term = True
    h["rows"][:] = [driver(25), driver(81)]
    with pytest.raises(RuntimeError, match="80 GiB"):
        run(h)
    assert h["signals"] == [signal.SIGTERM, signal.SIGKILL]
    assert receipt(h)["cleanup"]["signals"] == ["SIGTERM", "SIGKILL"]


def test_release_timeout_cannot_skip_guard_forever(harness):
    h = harness
    h["config"]["wait_timeout_seconds"] = 2
    h["path"].unlink()
    h["environment"]["FINANCE_GPU_COORDINATOR_CONFIG_SHA256"] = memory.freeze_config(
        h["path"], h["config"]
    )
    h["states"][:] = [holder(76)]
    h["rows"][:] = [driver(87)]
    h["process"].exit_after = None
    with pytest.raises(RuntimeError, match="release wait expired"):
        run(h)
    assert h["clock"].now < 10
    assert h["signals"] == [signal.SIGTERM]
    assert receipt(h)["holder_release_wait_seconds"] == 2


def test_blocked_telemetry_expiring_budget_cannot_activate(harness, monkeypatch):
    h = harness
    original = monitor._sample

    def slow(*args):
        result = original(*args)
        h["clock"].now = 1801
        return result

    monkeypatch.setattr(monitor, "_sample", slow)
    with pytest.raises(RuntimeError, match="release wait expired"):
        run(h)
    assert not h["calls"]


def test_zero_exit_before_release_is_not_success(harness):
    h = harness
    h["states"][:] = [holder(76)]
    h["rows"][:] = [driver(87)]
    with pytest.raises(RuntimeError, match="without activating"):
        run(h)
    assert receipt(h)["status"] == "FAILED"


def test_native_nonzero_returncode_is_preserved(harness):
    h = harness
    h["process"].code = 7
    assert run(h) == 7
    assert receipt(h)["returncode"] == 7
    assert receipt(h)["status"] == "FAILED"


def test_frozen_config_tamper_is_fatal_before_launch(harness):
    h = harness
    h["path"].write_text("{}")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        run(h)
    assert not h["calls"]


@pytest.mark.parametrize(
    "filename", ["gpu_monitor_receipt.json", "gpu_monitor.jsonl", "process.json"]
)
def test_existing_evidence_is_not_overwritten(harness, filename):
    h = harness
    path = h["cell"] / filename
    path.write_text("original")
    with pytest.raises((RuntimeError, FileExistsError)):
        run(h)
    assert path.read_text() == "original"
    assert not h["calls"]


def test_cleanup_refuses_changed_child_identity_without_signalling(harness, monkeypatch):
    h = harness
    h["process"].exit_after = None
    h["rows"][:] = [driver(25), driver(81)]
    original = monitor.process_identity
    counts = {}

    def identity(pid):
        counts[pid] = counts.get(pid, 0) + 1
        actual = original(pid)
        if pid == CHILD["pid"] and counts[pid] > 1:
            actual["start_time_ticks"] += 1
        return actual

    monkeypatch.setattr(monitor, "process_identity", identity)
    with pytest.raises(RuntimeError, match="refusing to signal"):
        run(h)
    assert not h["signals"]
    assert "unverified" in receipt(h)["cleanup"]["error"]["message"]


def test_owner_file_change_or_holder_identity_change_is_fatal(harness):
    h = harness
    (Path(h["config"]["directory"]) / "owner.json").write_text("{}")
    with pytest.raises(RuntimeError, match="owner.json changed"):
        run(h)
    assert not h["calls"]


def test_release_refresh_cannot_discard_bad_preceding_claim(harness):
    h = harness
    invalid = holder(76)
    invalid["claim"]["lease_token"] = "another-controller"
    h["states"][:] = [invalid, holder(), holder()]
    h["rows"][:] = [driver(87), driver(25)]
    with pytest.raises(RuntimeError, match="holder claim"):
        run(h)
    assert not h["calls"]


def test_status_transport_failure_after_launch_stops_own_group(harness, monkeypatch):
    h = harness
    original = monitor.request
    h["process"].exit_after = None

    def request(*args):
        if h["calls"]:
            raise TimeoutError("holder status unavailable")
        return original(*args)

    monkeypatch.setattr(monitor, "request", request)
    with pytest.raises(TimeoutError, match="status unavailable"):
        run(h)
    assert h["signals"] == [signal.SIGTERM]
    assert receipt(h)["failure"]["type"] == "TimeoutError"


def test_frozen_config_change_after_launch_stops_own_group(harness, monkeypatch):
    h = harness
    original = monitor.subprocess.Popen
    h["process"].exit_after = None

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        h["path"].write_text("{}")
        return process

    monkeypatch.setattr(monitor.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        run(h)
    assert h["signals"] == [signal.SIGTERM]


def test_gpu_uuid_change_after_launch_stops_own_group(harness):
    h = harness
    h["process"].exit_after = None
    h["rows"][:] = [driver(25), {**driver(25), "gpu_uuid": "GPU-different"}]
    with pytest.raises(RuntimeError, match="changed GPU UUID"):
        run(h)
    assert h["signals"] == [signal.SIGTERM]
