"""CPU tests of the holder/worker admission boundary and failure evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.agents import gpu_handoff as handoff

GIB = 1024**3
IDENTITY = "a" * 64
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def holder(tmp_path, monkeypatch):
    directory = tmp_path / "holder"
    directory.mkdir()
    now = datetime.now(ZoneInfo("Asia/Singapore"))
    owner = {
        "pid": os.getpid(), "cuda_device": 0, "directory": str(directory),
        "started_at_SGT": now.isoformat(),
        "expires_at_SGT": (now + timedelta(hours=2)).isoformat(),
        "allocation_only": True, "automatic_restart": False,
    }
    (directory / "owner.json").write_text(json.dumps(owner))
    state = dict(owner, status="HOLDING", closed=False, paused=False,
                 payload_bytes=30 * GIB, allocator_allocated_bytes=30 * GIB,
                 allocator_reserved_bytes=31 * GIB, maximum_bytes=59 * GIB,
                 target_bytes=59 * GIB, remaining_seconds=7200,
                 error=None, cleanup_error=None)
    calls = []

    def request(path, command):
        assert path == str(directory)
        calls.append(copy.deepcopy(command))
        if command["op"] == "pause":
            state.update(status="PAUSED", paused=True)
        elif command["op"] == "release":
            assert command["bytes"] == state["maximum_bytes"]
            state.update(payload_bytes=0, allocator_allocated_bytes=0,
                         allocator_reserved_bytes=0, paused=True, status="PAUSED")
        else:
            assert command["op"] == "status"
        return copy.deepcopy(state)

    driver = {"gpu_uuid": GPU_UUID, "driver_free_bytes": 32 * GIB,
              "driver_total_bytes": 94 * GIB}
    monkeypatch.setattr(handoff, "request", request)
    monkeypatch.setattr(handoff, "_driver_snapshot", lambda: dict(driver))
    monkeypatch.setattr(handoff, "_worker_free_bytes", lambda config: {
        **driver, "driver_free_bytes": 61 * GIB, "source": "fake CUDA free reading",
    })
    monkeypatch.setattr(handoff, "_ATTEMPTS", {})
    config = handoff.config_for_holder(directory)
    config_path = tmp_path / "handoff.json"
    config_path.write_text(json.dumps(config))
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("FINANCE_GPU_HANDOFF_CONFIG", str(config_path))
    monkeypatch.setenv("FINANCE_GPU_HANDOFF_CONFIG_SHA256", digest)
    monkeypatch.setenv("FINANCE_GPU_HANDOFF_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("FINANCE_SCREENING_RESOURCE_PROFILE", "chunked_adam_disk_v1")
    calls.clear()
    return SimpleNamespace(directory=directory, owner=owner, state=state, calls=calls,
                           driver=driver, config=config, config_path=config_path, digest=digest,
                           evidence=evidence, request=request)


def receipt(holder):
    return json.loads((holder.evidence / f"receipt-{os.getpid()}.json").read_text())


def mutations(holder):
    return [row for row in holder.calls if row["op"] != "status"]


def test_disabled_hook_and_import_do_not_import_torch(monkeypatch):
    monkeypatch.delenv("FINANCE_GPU_HANDOFF_CONFIG", raising=False)
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from src.agents import gpu_handoff as h; "
         "assert h.worker_handoff_from_environment('actor', '') is None; "
         "assert 'torch' not in sys.modules"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_config_pins_owner_process_gpu_and_exact_bytes(holder):
    config = handoff.load_config(holder.config_path, holder.digest)
    assert config == holder.config
    assert config["owner_sha256"] == hashlib.sha256(
        (holder.directory / "owner.json").read_bytes()).hexdigest()
    assert config["pid_start_ticks"] == handoff._process_start_ticks(os.getpid())
    assert config["gpu_uuid"] == GPU_UUID
    assert config["minimum_free_bytes"] == 60 * GIB
    holder.config_path.write_text(holder.config_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        handoff.load_config(holder.config_path, holder.digest)


@pytest.mark.parametrize("field,value", [
    ("minimum_free_bytes", 59 * GIB), ("cuda_device", 1),
    ("pid", True), ("profile", "other"), ("owner_sha256", "a"),
    ("directory", "relative/holder"),
])
def test_config_rejects_weakened_or_ambiguous_values(holder, field, value):
    config = {**holder.config, field: value}
    holder.config_path.write_text(json.dumps(config))
    digest = hashlib.sha256(holder.config_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        handoff.load_config(holder.config_path, digest)


def test_queue_counts_actual_payload_not_target_or_allocator_reservation(holder):
    result = handoff.available_bytes(holder.config)
    assert result["available_bytes"] == 62 * GIB
    assert result["holder_payload_bytes"] == 30 * GIB
    assert result["driver_free_bytes"] == 32 * GIB
    assert result["holder_status"]["target_bytes"] == 59 * GIB
    assert mutations(holder) == []


def test_queue_observation_uses_smaller_payload_when_holder_changes(holder, monkeypatch):
    calls = 0

    def moving_request(path, command):
        nonlocal calls
        calls += 1
        if calls == 2:
            holder.state.update(payload_bytes=20 * GIB, allocator_allocated_bytes=20 * GIB,
                                allocator_reserved_bytes=20 * GIB)
        return holder.request(path, command)

    monkeypatch.setattr(handoff, "request", moving_request)
    assert handoff.available_bytes(holder.config)["available_bytes"] == 52 * GIB


def test_queue_accepts_paused_empty_holder_between_arms(holder):
    holder.state.update(status="PAUSED", paused=True, payload_bytes=0,
                        allocator_allocated_bytes=0, allocator_reserved_bytes=0)
    assert handoff.available_bytes(holder.config)["available_bytes"] == 32 * GIB
    assert mutations(holder) == []
    holder.state["allocator_reserved_bytes"] = GIB
    with pytest.raises(RuntimeError, match="zero payload"):
        handoff.available_bytes(holder.config)


@pytest.mark.parametrize("change,match", [
    ({"pid": 1}, "pid mismatch"),
    ({"status": "CLOSED", "closed": True}, "closed or failed"),
    ({"remaining_seconds": 299}, "insufficient time"),
    ({"remaining_seconds": float("nan")}, "insufficient time"),
    ({"allocator_reserved_bytes": 29 * GIB}, "inconsistent"),
    ({"error": "fault"}, "closed or failed"),
])
def test_live_holder_validation_rejects_wrong_or_stale_evidence(holder, change, match):
    holder.state.update(change)
    with pytest.raises(RuntimeError, match=match):
        handoff.reservation_status(holder.config)


def test_owner_change_pid_reuse_and_gpu_change_are_rejected(holder, monkeypatch):
    monkeypatch.setattr(handoff, "_process_start_ticks", lambda pid: 1)
    with pytest.raises(RuntimeError, match="PID was reused"):
        handoff.reservation_status(holder.config)
    monkeypatch.setattr(handoff, "_process_start_ticks",
                        lambda pid: holder.config["pid_start_ticks"])
    holder.driver["gpu_uuid"] = "GPU-other"
    with pytest.raises(RuntimeError, match="GPU UUID changed"):
        handoff.reservation_status(holder.config)
    holder.driver["gpu_uuid"] = GPU_UUID
    path = holder.directory / "owner.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="owner.json changed"):
        handoff.reservation_status(holder.config)


def test_stale_status_file_never_replaces_live_controller_status(holder, monkeypatch):
    (holder.directory / "status.json").write_text(json.dumps(holder.state))

    def disconnected(*args, **kwargs):
        raise ConnectionRefusedError("not live")

    monkeypatch.setattr(handoff, "request", disconnected)
    with pytest.raises(ConnectionRefusedError):
        handoff.available_bytes(holder.config)
    assert mutations(holder) == []


def test_worker_releases_once_then_only_checks_reference_and_actor(holder):
    result = handoff.worker_handoff_from_environment("critic", IDENTITY)
    assert result["status"] == "PASSED"
    assert result["before"]["payload_bytes"] == 30 * GIB
    assert result["after"]["allocator_reserved_bytes"] == 0
    assert result["worker_memory"]["driver_free_bytes"] == 61 * GIB
    assert result["atomic_transfer_guaranteed"] is False
    assert result["completed_at_SGT"].endswith("+08:00")
    handoff.worker_handoff_from_environment("reference", IDENTITY)
    handoff.worker_handoff_from_environment("actor", IDENTITY)
    controls = mutations(holder)
    assert [row["op"] for row in controls] == ["pause", "release"]
    assert controls[1]["bytes"] == 59 * GIB
    assert all(IDENTITY in row["request_id"] for row in controls)
    assert len({row["request_id"] for row in controls}) == 2
    assert [row["role"] for row in receipt(holder)["role_checks"]] == ["reference", "actor"]


def test_new_worker_can_handoff_an_already_paused_empty_holder(holder):
    holder.state.update(status="PAUSED", paused=True, payload_bytes=0,
                        allocator_allocated_bytes=0, allocator_reserved_bytes=0)
    result = handoff.worker_handoff_from_environment("critic", IDENTITY)
    assert result["status"] == "PASSED" and result["before"]["payload_bytes"] == 0


@pytest.mark.parametrize("role", ["actor", "reference", "unknown"])
def test_wrong_first_role_fails_before_mutation_and_cannot_retry(holder, role):
    with pytest.raises(RuntimeError):
        handoff.worker_handoff_from_environment(role, IDENTITY)
    assert mutations(holder) == []
    assert receipt(holder)["status"] == "FAILED"
    with pytest.raises(RuntimeError, match="automatic retry is forbidden"):
        handoff.worker_handoff_from_environment("critic", IDENTITY)


def test_existing_receipt_is_preserved_instead_of_restarting(holder):
    holder.evidence.mkdir()
    path = holder.evidence / f"receipt-{os.getpid()}.json"
    path.write_text('{"old":true}\n')
    with pytest.raises(RuntimeError, match="already exists"):
        handoff.worker_handoff_from_environment("critic", IDENTITY)
    assert path.read_text() == '{"old":true}\n'
    assert mutations(holder) == []


def test_wrong_resource_profile_is_rejected_before_release(holder, monkeypatch):
    monkeypatch.setenv("FINANCE_SCREENING_RESOURCE_PROFILE", "pageable_exchange_v1")
    with pytest.raises(RuntimeError, match="chunked_adam_disk_v1"):
        handoff.worker_handoff_from_environment("critic", IDENTITY)
    assert mutations(holder) == []


def test_lost_release_response_does_not_retry_or_load_models(holder, monkeypatch):
    def lost_reply(path, command):
        state = holder.request(path, command)
        if command["op"] == "release":
            raise TimeoutError("reply lost after release")
        return state

    monkeypatch.setattr(handoff, "request", lost_reply)
    with pytest.raises(TimeoutError, match="reply lost"):
        handoff.worker_handoff_from_environment("critic", IDENTITY)
    assert holder.state["payload_bytes"] == 0 and holder.state["paused"]
    assert receipt(holder)["status"] == "FAILED"
    with pytest.raises(RuntimeError, match="automatic retry"):
        handoff.worker_handoff_from_environment("critic", IDENTITY)
    assert [row["op"] for row in mutations(holder)] == ["pause", "release"]


def test_lost_gpu_capacity_after_release_fails_with_measured_evidence(holder, monkeypatch):
    monkeypatch.setattr(handoff, "_worker_free_bytes", lambda config: {
        "driver_free_bytes": 60 * GIB - 1, "driver_total_bytes": 94 * GIB,
    })
    with pytest.raises(RuntimeError, match="below the required 60 GiB"):
        handoff.worker_handoff_from_environment("critic", IDENTITY)
    report = receipt(holder)
    assert report["status"] == "FAILED"
    assert report["after"]["payload_bytes"] == 0
    assert report["worker_memory"]["driver_free_bytes"] == 60 * GIB - 1
    assert holder.state["paused"]


def test_later_role_rejects_refill_without_second_release(holder):
    handoff.worker_handoff_from_environment("critic", IDENTITY)
    holder.state.update(status="HOLDING", paused=False, payload_bytes=GIB,
                        allocator_allocated_bytes=GIB, allocator_reserved_bytes=GIB)
    with pytest.raises(RuntimeError, match="remain paused"):
        handoff.worker_handoff_from_environment("reference", IDENTITY)
    assert len(mutations(holder)) == 2
    assert receipt(holder)["status"] == "FAILED"


def test_later_role_rejects_identity_change(holder):
    handoff.worker_handoff_from_environment("critic", IDENTITY)
    with pytest.raises(RuntimeError, match="identity or configuration changed"):
        handoff.worker_handoff_from_environment("actor", "b" * 64)
    assert len(mutations(holder)) == 2


def test_later_role_accepts_clean_expiry_only_after_original_process_is_gone(holder, monkeypatch):
    handoff.worker_handoff_from_environment("critic", IDENTITY)
    holder.state.update(status="EXPIRED", closed=True)
    (holder.directory / "status.json").write_text(json.dumps(holder.state))

    def exited(pid):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(handoff, "_process_start_ticks", exited)
    assert handoff.worker_handoff_from_environment("reference", IDENTITY)["status"] == "PASSED"
    assert len(mutations(holder)) == 2


def test_later_role_does_not_trust_clean_file_while_holder_process_is_alive(holder, monkeypatch):
    handoff.worker_handoff_from_environment("critic", IDENTITY)
    holder.state.update(status="EXPIRED", closed=True)
    (holder.directory / "status.json").write_text(json.dumps(holder.state))

    def disconnected(*args, **kwargs):
        raise ConnectionRefusedError("socket gone")

    monkeypatch.setattr(handoff, "request", disconnected)
    with pytest.raises(RuntimeError, match="PID still exists"):
        handoff.worker_handoff_from_environment("reference", IDENTITY)


def test_driver_parser_requires_one_card_and_converts_mib(monkeypatch):
    result = SimpleNamespace(stdout=f"{GPU_UUID}, 61440, 95360\n")

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi" and kwargs["timeout"] == 10
        return result

    monkeypatch.setattr(handoff.subprocess, "run", run)
    assert handoff._driver_snapshot()["driver_free_bytes"] == 60 * GIB
    result.stdout += f"{GPU_UUID}, 61440, 95360\n"
    with pytest.raises(RuntimeError, match="exactly one"):
        handoff._driver_snapshot()
