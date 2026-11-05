"""CPU fakes for persistent holder admission and fail-closed phase transitions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import gpu_memory_coordinator as memory

GIB = memory.GIB
IDENTITY = "a" * 64


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


@pytest.fixture
def config(tmp_path):
    directory = tmp_path / "holder"
    directory.mkdir()
    owner = dict(pid=123, directory=str(directory), cuda_device=0,
                 started_at_SGT="2026-10-07T12:00:00+08:00", expires_at_SGT=None,
                 persistent=True, allocation_only=True, automatic_restart=False)
    content = json.dumps(owner).encode()
    (directory / "owner.json").write_bytes(content)
    return dict(
        profile=memory.PROFILE, directory=str(directory),
        owner_sha256=hashlib.sha256(content).hexdigest(),
        holder_pid=123, holder_start_time_ticks=456, gpu_uuid="GPU-test", cuda_device=0,
        lease_token="fresh-run-lease", evidence_directory=str(tmp_path / "evidence"),
        portfolio_target_bytes=76 * GIB,
        phase_peak_bytes={phase: 76 * GIB for phase in memory.PHASES},
        minimum_free_bytes=2 * GIB, wait_timeout_seconds=3, poll_seconds=1,
        active_arm_holder_policy="paused_zero",
    )


def worker_identity(pid=None):
    return dict(pid=os.getpid() if pid is None else pid, start_time_ticks=111,
                cgroup_path="/user.slice/experiment.service", process_group_id=222, session_id=333)


class Broker:
    def __init__(self, config):
        owner = json.loads((Path(config["directory"]) / "owner.json").read_text())
        self.state = {**owner, "coordination_profile": "persistent_claim_v1", "closed": False,
                      "paused": False, "status": "HOLDING", "maximum_bytes": 80 * GIB,
                      "payload_bytes": 50 * GIB, "allocator_allocated_bytes": 50 * GIB,
                      "allocator_reserved_bytes": 50 * GIB, "error": None, "cleanup_error": None,
                      "claim": dict(lease_token=config["lease_token"], owner=worker_identity(999),
                                    cgroup_populated=True, watchdog_error=None, workers=[])}
        self.calls = []
        self.release_callback = None

    def __call__(self, directory, command):
        self.calls.append(dict(command))
        op = command["op"]
        if op != "status":
            assert command["lease_token"] == self.state["claim"]["lease_token"]
        if op == "attach":
            self.state["claim"]["workers"].extend(command["workers"])
        elif op == "pause":
            self.state.update(paused=True, status="PAUSED")
        elif op == "release":
            assert command["bytes"] == 80 * GIB
            self.state.update(paused=True, status="PAUSED", payload_bytes=0,
                              allocator_allocated_bytes=0, allocator_reserved_bytes=0)
            if self.release_callback:
                self.release_callback()
        elif op == "claim":
            self.state["claim"]["owner"] = command["owner"]
        elif op == "refill":
            self.state.update(paused=False, status="GROWING", target_bytes=command["target_bytes"])
        else:
            assert op == "status"
        return copy.deepcopy(self.state)


@pytest.fixture
def harness(config):
    broker = Broker(config)
    clock = Clock()
    samples = SimpleNamespace(foreign_bytes=18 * GIB, own_bytes=0)
    events = []

    def sampler(scope):
        assert scope == worker_identity()["cgroup_path"]
        return dict(free_bytes=96 * GIB - samples.foreign_bytes - samples.own_bytes
                    - broker.state["payload_bytes"], total_bytes=96 * GIB,
                    own_resident_bytes=samples.own_bytes, gpu_uuid="GPU-test")

    coordinator = memory.GpuMemoryCoordinator(
        config, identity=IDENTITY, requester=broker, sampler=sampler,
        identity_reader=worker_identity, holder_validator=lambda state: state,
        clock=clock, sleep=clock.sleep, emit=events.append,
    )
    return SimpleNamespace(coordinator=coordinator, broker=broker, clock=clock,
                           samples=samples, events=events, sampler=sampler)


def test_retain_occupied_payload_until_capacity_then_release(harness):
    original = harness.sampler
    released = []

    def sampler(scope):
        released.append(harness.broker.state["payload_bytes"])
        return original(scope)

    harness.coordinator.sampler = sampler
    receipt = harness.coordinator.before_phase("init_critic")
    assert released[:2] == [50 * GIB, 50 * GIB]
    assert released[2:] == [0, 0]
    assert receipt["before_release"]["holder_payload_bytes"] == 50 * GIB
    assert receipt["after_release"]["free_bytes"] == 78 * GIB
    assert receipt["atomic_transfer_guaranteed"] is False
    harness.coordinator.after_phase("init_critic")
    assert harness.coordinator.active_phase is None


def test_wait_shortage_preserves_payload_and_permanently_fails(harness):
    harness.samples.foreign_bytes = 24 * GIB
    with pytest.raises(TimeoutError, match="init_critic"):
        harness.coordinator.before_phase("init_critic")
    assert harness.broker.state["payload_bytes"] == 50 * GIB
    assert harness.broker.state["paused"] is True
    assert not any(row["op"] == "release" for row in harness.broker.calls)
    assert harness.coordinator.failed
    with pytest.raises(RuntimeError, match="previously failed"):
        harness.coordinator.before_phase("init_critic")


def test_wait_can_finish_when_foreign_process_releases(harness):
    harness.samples.foreign_bytes = 24 * GIB

    def sleep(seconds):
        assert harness.broker.state["payload_bytes"] == 50 * GIB
        harness.clock.sleep(seconds)
        harness.samples.foreign_bytes = 18 * GIB

    harness.coordinator.sleep = sleep
    harness.coordinator.before_phase("init_critic")
    assert harness.clock.value == 1
    assert harness.broker.state["payload_bytes"] == 0


def test_capacity_lost_after_release_waits_without_new_allocation(harness):
    harness.broker.release_callback = lambda: setattr(harness.samples, "foreign_bytes", 24 * GIB)
    with pytest.raises(TimeoutError):
        harness.coordinator.before_phase("rollout_mode")
    assert harness.broker.state["payload_bytes"] == 0
    assert harness.broker.state["paused"] is True
    assert not any(row["op"] == "refill" for row in harness.broker.calls)


def test_high_first_actual_read_does_not_hide_low_confirmation(harness):
    original = harness.sampler
    count = 0

    def sampler(scope):
        nonlocal count
        count += 1
        if count == 2:
            row = original(scope)
            return {**row, "free_bytes": 10 * GIB}
        return original(scope)

    harness.coordinator.sampler = sampler
    harness.coordinator.before_phase("compute_values")
    assert harness.clock.value == 1
    assert count >= 6


def test_own_verified_residency_reduces_additional_capacity(harness):
    harness.samples.own_bytes = 48 * GIB
    harness.broker.state.update(payload_bytes=0, allocator_allocated_bytes=0,
                               allocator_reserved_bytes=0)
    result = harness.coordinator.before_phase("rollout_mode")
    assert result["after_release"]["required_free_bytes"] == 28 * GIB
    assert result["after_release"]["free_bytes"] == 30 * GIB
    harness.coordinator.after_phase("rollout_mode")


def test_zero_holder_through_all_generation_and_training_phases(harness):
    for phase in memory.PHASES:
        harness.coordinator.before_phase(phase)
        harness.coordinator.after_phase(phase)
    assert not any(row["op"] == "refill" for row in harness.broker.calls)
    assert harness.broker.state["paused"] is True
    assert harness.broker.state["payload_bytes"] == 0


@pytest.mark.parametrize("mutation", [
    lambda state: state.update(claim=None),
    lambda state: state["claim"].update(lease_token="different-run"),
    lambda state: state["claim"].update(cgroup_populated=False),
    lambda state: state["claim"].update(watchdog_error="cannot read kernel scope"),
    lambda state: state["claim"]["owner"].update(start_time_ticks=222),
    lambda state: state["claim"].update(workers=[]),
])
def test_lost_dead_or_changed_claim_is_never_admitted(harness, mutation):
    mutation(harness.broker.state)
    with pytest.raises((RuntimeError, TypeError, AssertionError)):
        harness.coordinator.before_phase("update_actor")
    assert not any(row["op"] == "release" for row in harness.broker.calls)


@pytest.mark.parametrize("field,value", [
    ("gpu_uuid", "GPU-other"), ("free_bytes", -1), ("own_resident_bytes", None),
    ("free_bytes", 1000 * GIB), ("own_resident_bytes", 1000 * GIB),
])
def test_invalid_gpu_telemetry_stops_before_release(harness, field, value):
    original = harness.sampler
    harness.coordinator.sampler = lambda scope: {**original(scope), field: value}
    with pytest.raises(RuntimeError, match="telemetry"):
        harness.coordinator.before_phase("update_critic")
    assert not any(row["op"] == "release" for row in harness.broker.calls)


def test_no_overlapping_phase_or_wrong_completion(harness):
    harness.coordinator.before_phase("compute_values")
    with pytest.raises(RuntimeError, match="overlapping"):
        harness.coordinator.before_phase("update_actor")
    with pytest.raises(RuntimeError, match="does not match"):
        harness.coordinator.after_phase("compute_log_prob")


def test_refill_during_active_generation_is_rejected(harness):
    harness.coordinator.before_phase("rollout_mode")
    harness.broker.state.update(paused=False, payload_bytes=GIB, allocator_allocated_bytes=GIB,
                               allocator_reserved_bytes=GIB)
    with pytest.raises(RuntimeError, match="paused and empty"):
        harness.coordinator.after_phase("rollout_mode")
    assert harness.coordinator.failed
    assert harness.broker.state["paused"] is True


def test_worker_start_tick_change_fails_closed(harness):
    harness.coordinator.identity_reader = lambda pid: {
        **worker_identity(pid), "start_time_ticks": 999}
    with pytest.raises(RuntimeError, match="identity changed"):
        harness.coordinator.before_phase("save_checkpoint")


def test_attach_cannot_cross_cgroup(config):
    broker = Broker(config)
    with pytest.raises(RuntimeError, match="outside"):
        memory.GpuMemoryCoordinator(
            config, identity=IDENTITY, requester=broker, sampler=lambda scope: None,
            identity_reader=lambda pid: {**worker_identity(pid), "cgroup_path": "/other.service"},
            holder_validator=lambda state: state,
        )
    assert not any(row["op"] == "attach" for row in broker.calls)


def test_frozen_config_identity_and_no_overwrite(config, tmp_path):
    path = tmp_path / "config.json"
    digest = memory.freeze_config(path, config)
    assert memory.load_config(path, digest) == config
    with pytest.raises(FileExistsError):
        memory.freeze_config(path, config)
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        memory.load_config(path, digest)


def test_environment_binding_changes_stop_existing_worker(harness, monkeypatch, tmp_path):
    path = tmp_path / "configuration.json"
    digest = memory.freeze_config(path, harness.coordinator.config)
    monkeypatch.setenv("FINANCE_GPU_COORDINATOR_CONFIG", str(path))
    monkeypatch.setenv("FINANCE_GPU_COORDINATOR_CONFIG_SHA256", digest)
    harness.coordinator._environment_signature = (str(path), digest)
    harness.coordinator.before_phase("compute_values")
    harness.coordinator.after_phase("compute_values")
    monkeypatch.setenv("FINANCE_GPU_COORDINATOR_CONFIG_SHA256", "b" * 64)
    with pytest.raises(RuntimeError, match="configuration changed"):
        harness.coordinator.before_phase("update_actor")


def test_validate_holder_detects_owner_or_process_replacement(config):
    broker = Broker(config)

    def reader(pid):
        return {**worker_identity(pid), "start_time_ticks": 456}

    assert memory._validate_holder(config, broker.state, identity_reader=reader) == broker.state
    with pytest.raises(RuntimeError, match="process or persistent profile changed"):
        memory._validate_holder(config, broker.state, identity_reader=worker_identity)
    path = Path(config["directory"]) / "owner.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="owner.json changed"):
        memory._validate_holder(config, broker.state, identity_reader=reader)


def test_validate_holder_rejects_too_small_maximum_for_idle_target(config):
    broker = Broker(config)
    broker.state["maximum_bytes"] = 60 * GIB
    with pytest.raises(RuntimeError, match="inconsistent"):
        memory._validate_holder(config, broker.state, identity_reader=lambda pid: {
            **worker_identity(pid), "start_time_ticks": 456})


@pytest.mark.parametrize("field,value", [
    ("active_arm_holder_policy", "adaptive"), ("minimum_free_bytes", GIB),
    ("phase_peak_bytes", {"rollout_mode": 76 * GIB}), ("portfolio_target_bytes", 81 * GIB),
    ("cuda_device", True), ("wait_timeout_seconds", float("nan")),
    ("lease_token", ""), ("poll_seconds", 4),
])
def test_invalid_config_is_rejected(config, field, value):
    config[field] = value
    with pytest.raises(RuntimeError):
        memory._validate_config(config)


def patch_controller(monkeypatch, config):
    broker = Broker(config)
    broker.state["claim"]["owner"] = worker_identity()
    broker.state["claim"]["workers"] = []
    monkeypatch.setattr(memory, "request", broker)
    monkeypatch.setattr(memory, "process_identity", worker_identity)
    monkeypatch.setattr(memory, "reservation_status", lambda config: copy.deepcopy(broker.state))
    monkeypatch.setattr(memory, "_validate_holder", lambda *args: None)
    monkeypatch.setattr(memory, "_scope_pids", lambda scope: {os.getpid()})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    return broker


def test_controller_idle_refill_and_next_claim_pause_retain_payload(monkeypatch, config):
    broker = patch_controller(monkeypatch, config)
    memory.controller_idle(config)
    assert broker.calls[-1]["op"] == "refill"
    assert broker.calls[-1]["target_bytes"] == 76 * GIB
    memory.controller_claim(config)
    assert [row["op"] for row in broker.calls[-2:]] == ["claim", "pause"]
    assert broker.state["paused"] is True
    assert broker.state["payload_bytes"] == 50 * GIB


def test_controller_idle_rejects_surviving_orphan_child(monkeypatch, config):
    broker = patch_controller(monkeypatch, config)
    monkeypatch.setattr(memory, "_scope_pids", lambda scope: {os.getpid(), 777})
    with pytest.raises(RuntimeError, match="child processes remain"):
        memory.controller_idle(config)
    assert not broker.calls


def test_controller_idle_rejects_registered_worker_moved_out_of_scope(monkeypatch, config):
    broker = patch_controller(monkeypatch, config)
    broker.state["claim"]["workers"] = [worker_identity(777)]
    with pytest.raises(RuntimeError, match="worker is still alive"):
        memory.controller_idle(config)
    assert not broker.calls


def test_cpu_controller_query_device_fd_does_not_block_idle_refill(monkeypatch, config):
    broker = patch_controller(monkeypatch, config)
    monkeypatch.setattr(memory, "_process_has_gpu_device", lambda pid: True)
    memory.controller_idle(config)
    assert broker.calls[-1]["op"] == "refill"


@pytest.mark.parametrize("initialized", [True, None, 0])
def test_initialized_or_unknown_controller_cuda_state_blocks_refill(
    monkeypatch, config, initialized,
):
    broker = patch_controller(monkeypatch, config)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: initialized)))
    with pytest.raises(RuntimeError, match="CUDA initialization state"):
        memory.controller_idle(config)
    assert not broker.calls


def test_controller_without_torch_can_refill_without_importing_it(monkeypatch, config):
    broker = patch_controller(monkeypatch, config)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    memory.controller_idle(config)
    assert "torch" not in sys.modules
    assert broker.calls[-1]["op"] == "refill"


def test_controller_idle_rechecks_scope_after_cpu_state_query(monkeypatch, config):
    broker = patch_controller(monkeypatch, config)
    reads = iter([{os.getpid()}, {os.getpid(), 888}])
    monkeypatch.setattr(memory, "_scope_pids", lambda scope: next(reads))
    with pytest.raises(RuntimeError, match="changed during"):
        memory.controller_idle(config)
    assert not broker.calls


def patch_device_scope(monkeypatch, tmp_path, descriptors):
    """Use real directory enumeration and synthetic character-device stat results."""
    proc = tmp_path / "proc"
    descriptor_stats = {}
    for pid, devices in descriptors.items():
        folder = proc / str(pid) / "fd"
        folder.mkdir(parents=True)
        for index, device in enumerate(devices):
            path = folder / str(index)
            path.touch()
            descriptor_stats[path] = device
    original_stat = Path.stat

    def fake_stat(path, *args, **kwargs):
        if path in descriptor_stats:
            device = descriptor_stats[path]
            if isinstance(device, BaseException):
                raise device
            return SimpleNamespace(st_mode=device[0], st_rdev=os.makedev(*device[1:]))
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    return proc


def test_local_gpu_device_detection_does_not_depend_on_host_pids(monkeypatch, tmp_path):
    proc = patch_device_scope(monkeypatch, tmp_path, {
        123: [(stat.S_IFCHR, 195, 0), (stat.S_IFCHR, 195, 0)],
        456: [(stat.S_IFCHR, 195, 1)],
        789: [(stat.S_IFCHR, 195, 255), (stat.S_IFCHR, 511, 0), (stat.S_IFREG, 195, 0)],
    })
    assert memory._process_has_gpu_device(123, proc_root=proc)
    assert memory._process_has_gpu_device(456, proc_root=proc)
    assert not memory._process_has_gpu_device(789, proc_root=proc)


def test_unreadable_gpu_descriptor_cannot_be_treated_as_no_residency(monkeypatch, tmp_path):
    proc = patch_device_scope(monkeypatch, tmp_path, {123: [PermissionError("unreadable fd")]})
    with pytest.raises(PermissionError, match="unreadable fd"):
        memory._process_has_gpu_device(123, proc_root=proc)


def test_unreadable_fd_directory_fails_closed(monkeypatch, tmp_path):
    proc = patch_device_scope(monkeypatch, tmp_path, {123: []})
    original_iterdir = Path.iterdir

    def iterdir(path):
        if path == proc / "123" / "fd":
            raise PermissionError("unreadable process")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    with pytest.raises(PermissionError, match="unreadable process"):
        memory._process_has_gpu_device(123, proc_root=proc)


def native_segment(size=48 * GIB, *, address=GIB, pool=(0, 0), expandable=False):
    return dict(device=0, total_size=size, address=address,
                segment_pool_id=pool, is_expandable=expandable)


def patch_cuda_sampler(monkeypatch, *, device_present=True, reserved=48 * GIB):
    fake_cuda = SimpleNamespace(
        device_count=lambda: 1, current_device=lambda: 0, synchronize=lambda device: None,
        mem_get_info=lambda device: (30 * GIB, 96 * GIB),
        memory_reserved=lambda device: reserved,
        memory_snapshot=lambda: [native_segment()],
        memory=SimpleNamespace(get_allocator_backend=lambda: "native"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(memory, "process_identity", worker_identity)
    def own_device(pid):
        assert pid == os.getpid(), "only this worker may corroborate its allocator sample"
        return device_present

    monkeypatch.setattr(memory, "_process_has_gpu_device", own_device)
    monkeypatch.setattr(memory, "_driver_snapshot", lambda: {"gpu_uuid": "GPU-test"})
    return fake_cuda


def test_cuda_sampler_credits_current_verified_worker_only(monkeypatch, harness):
    patch_cuda_sampler(monkeypatch)
    sampler = memory.TorchGpuSampler(harness.coordinator.config)
    result = sampler(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 48 * GIB
    assert harness.coordinator._required("rollout_mode", result) == 28 * GIB
    assert result["own_gpu_pids"] == [os.getpid()]
    assert result["own_resident_source"] == (
        "pytorch_native_default_pool_mapped_snapshot_lower_bound_v3")
    assert result["unattributed_driver_residency_credited_bytes"] == 0


@pytest.mark.parametrize("other_bytes", [0, 10 * GIB])
def test_other_scope_gpu_users_cannot_pay_for_this_worker_peak(monkeypatch, harness, other_bytes):
    patch_cuda_sampler(monkeypatch)
    monkeypatch.setattr(memory, "_driver_snapshot", lambda: {
        "gpu_uuid": "GPU-test", "other_scope_pid": 777, "other_resident_bytes": other_bytes})
    monkeypatch.setattr(memory, "_scope_pids", lambda scope: pytest.fail(
        "worker admission must not enumerate unrelated cgroup processes"))
    harness.coordinator.sampler = memory.TorchGpuSampler(harness.coordinator.config)
    receipt = harness.coordinator.before_phase("rollout_mode")
    assert receipt["after_release"]["own_resident_bytes"] == 48 * GIB
    assert receipt["after_release"]["required_free_bytes"] == 28 * GIB
    assert receipt["after_release"]["own_gpu_pids"] == [os.getpid()]
    assert receipt["after_release"]["unattributed_driver_residency_credited_bytes"] == 0


def test_other_scope_residency_cannot_turn_insufficient_free_memory_into_admission(
    monkeypatch, harness,
):
    cuda = patch_cuda_sampler(monkeypatch)
    cuda.mem_get_info = lambda device: (20 * GIB, 96 * GIB)
    monkeypatch.setattr(memory, "_driver_snapshot", lambda: {
        "gpu_uuid": "GPU-test", "other_scope_pid": 777, "other_resident_bytes": 10 * GIB})
    harness.broker.state.update(payload_bytes=0, allocator_allocated_bytes=0,
                               allocator_reserved_bytes=0)
    harness.coordinator.sampler = memory.TorchGpuSampler(harness.coordinator.config)
    # Own 48 GiB still requires 28 GiB free. Adding someone else's 10 GiB would
    # incorrectly reduce that threshold to 18 GiB and admit this 20 GiB sample.
    with pytest.raises(TimeoutError):
        harness.coordinator.before_phase("rollout_mode")
    assert not any(row["op"] == "release" for row in harness.broker.calls)


def test_cuda_sampler_rechecks_current_worker_start_identity(monkeypatch, config):
    patch_cuda_sampler(monkeypatch)
    sampler = memory.TorchGpuSampler(config)
    monkeypatch.setattr(memory, "process_identity", lambda pid: {
        **worker_identity(pid), "start_time_ticks": 555})
    with pytest.raises(RuntimeError, match="PID identity"):
        sampler(worker_identity()["cgroup_path"])


def test_allocator_credit_cannot_include_unattributed_driver_memory(monkeypatch, config):
    patch_cuda_sampler(monkeypatch)
    monkeypatch.setattr(memory, "_driver_snapshot", lambda: {
        "gpu_uuid": "GPU-test", "driver_free_bytes": 30 * GIB,
        "foreign_host_pid": os.getpid(), "unattributed_used_bytes": 66 * GIB})
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 48 * GIB


def test_allocator_credit_uses_lower_counter_across_sampling(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    readings = iter([48 * GIB, 44 * GIB])
    cuda.memory_reserved = lambda device: next(readings)
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 44 * GIB
    assert result["allocator_before"]["reserved_bytes"] == 48 * GIB
    assert result["allocator_after"]["reserved_bytes"] == 44 * GIB


@pytest.mark.parametrize("reserved", [-1, True, 1.5])
def test_invalid_or_impossible_allocator_residency_fails_closed(monkeypatch, config, reserved):
    patch_cuda_sampler(monkeypatch, reserved=reserved)
    with pytest.raises(RuntimeError, match="allocator"):
        memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])


def test_native_pool_residency_cannot_exceed_observed_device_use(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch, reserved=100 * GIB)
    cuda.memory_snapshot = lambda: [native_segment(100 * GIB)]
    with pytest.raises(RuntimeError, match="allocator exceeds"):
        memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])


def test_unmapped_private_pool_is_never_credited_as_physical_gpu_memory(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch, reserved=80 * GIB)
    # vLLM can keep 60 GiB of private virtual segments while asleep. Its raw
    # allocator statistics may still include these bytes despite cuMemUnmap.
    cuda.memory_snapshot = lambda: [
        native_segment(20 * GIB),
        native_segment(60 * GIB, address=21 * GIB, pool=(0, 1)),
    ]
    cuda.mem_get_info = lambda device: (60 * GIB, 96 * GIB)
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 20 * GIB
    assert result["allocator_before"]["reserved_bytes"] == 80 * GIB
    assert result["allocator_before"]["private_pool_excluded_bytes"] == 60 * GIB


def test_native_expandable_ranges_credited_but_private_pools_excluded(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    cuda.memory_snapshot = lambda: [
        native_segment(20 * GIB, pool=(0, 1)),
        native_segment(28 * GIB, address=21 * GIB, expandable=True),
    ]
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 28 * GIB
    assert result["allocator_before"]["private_pool_excluded_bytes"] == 20 * GIB
    assert result["allocator_before"]["native_expandable_mapped_bytes"] == 28 * GIB


def test_private_expandable_segments_are_not_credited(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    cuda.memory_snapshot = lambda: [native_segment(48 * GIB, pool=(0, 2), expandable=True)]
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 0
    assert result["allocator_before"]["private_pool_excluded_bytes"] == 48 * GIB
    assert result["allocator_before"]["native_expandable_mapped_bytes"] == 0


def test_native_expandable_unmapped_ranges_do_not_survive_snapshot_credit(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    # Native allocator snapshot emits only mapped ranges; after unmapping a
    # cache range the next snapshot contains the smaller physical allocation.
    snapshots = iter([[native_segment(48 * GIB, expandable=True)],
                      [native_segment(16 * GIB, expandable=True)]])
    cuda.memory_snapshot = lambda: next(snapshots)
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 16 * GIB
    assert result["allocator_before"]["native_expandable_mapped_bytes"] == 48 * GIB
    assert result["allocator_after"]["native_expandable_mapped_bytes"] == 16 * GIB


def test_lower_native_pool_snapshot_is_used_after_cache_release(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    readings = iter([[native_segment(48 * GIB)], [native_segment(30 * GIB)]])
    cuda.memory_snapshot = lambda: next(readings)
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 30 * GIB


@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("segment_pool_id"),
    lambda row: row.update(segment_pool_id=(False, 0)),
    lambda row: row.update(segment_pool_id=(-1, 0)),
    lambda row: row.update(is_expandable=None),
    lambda row: row.update(total_size=-1),
    lambda row: row.update(address=0),
    lambda row: row.update(device=True),
])
def test_unknown_allocator_snapshot_cannot_be_credited(monkeypatch, config, mutation):
    cuda = patch_cuda_sampler(monkeypatch)
    segment = native_segment()
    mutation(segment)
    cuda.memory_snapshot = lambda: [segment]
    with pytest.raises(RuntimeError, match="snapshot"):
        memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])


def test_non_native_allocator_cannot_be_credited(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    cuda.memory.get_allocator_backend = lambda: "cudaMallocAsync"
    with pytest.raises(RuntimeError, match="native CUDA allocator"):
        memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])


def test_duplicate_snapshot_segments_cannot_double_gpu_credit(monkeypatch, config):
    cuda = patch_cuda_sampler(monkeypatch)
    cuda.memory_snapshot = lambda: [native_segment(), native_segment()]
    with pytest.raises(RuntimeError, match="overlapping"):
        memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])


def test_missing_current_gpu_device_cannot_validate_nonzero_allocator(monkeypatch, config):
    patch_cuda_sampler(monkeypatch, device_present=False)
    with pytest.raises(RuntimeError, match="no GPU device descriptor"):
        memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])


def test_initial_zero_allocator_does_not_claim_context_or_driver_memory(monkeypatch, config):
    patch_cuda_sampler(monkeypatch, device_present=False, reserved=0)
    result = memory.TorchGpuSampler(config)(worker_identity()["cgroup_path"])
    assert result["own_resident_bytes"] == 0
    assert result["own_gpu_pids"] == []


def test_worker_identity_change_during_driver_query_is_blocked(monkeypatch, config):
    patch_cuda_sampler(monkeypatch)
    sampler = memory.TorchGpuSampler(config)
    readings = iter([worker_identity(), {**worker_identity(), "start_time_ticks": 555}])
    monkeypatch.setattr(memory, "process_identity", lambda pid: next(readings))
    with pytest.raises(RuntimeError, match="identity changed during driver sampling"):
        sampler(worker_identity()["cgroup_path"])


def test_worker_outside_claimed_scope_is_blocked(monkeypatch, config):
    patch_cuda_sampler(monkeypatch)
    with pytest.raises(RuntimeError, match="claimed cgroup changed"):
        memory.TorchGpuSampler(config)("/other.service")


def test_singular_config_import_does_not_load_torch_or_cuda():
    result = subprocess.run([sys.executable, "-c", "import sys; "
                             "from src.agents import gpu_memory_coordinator; "
                             "assert 'torch' not in sys.modules"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_unconfigured_environment_is_noop(monkeypatch):
    monkeypatch.delenv("FINANCE_GPU_COORDINATOR_CONFIG", raising=False)
    assert memory.coordinator_from_environment("") is None


def test_persistent_and_old_handoff_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("FINANCE_GPU_COORDINATOR_CONFIG", "/tmp/new-config.json")
    monkeypatch.setenv("FINANCE_GPU_HANDOFF_CONFIG", "/tmp/old-config.json")
    with pytest.raises(RuntimeError, match="both be enabled"):
        memory.coordinator_from_environment(IDENTITY)
