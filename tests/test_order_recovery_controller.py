"""Inference failures stop the fixed schedule and preserve their receipts."""

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.agents import order_recovery_controller as controller

MANIFEST_SHA256 = "a" * 64
OWNER = {
    "pid": 123, "start_time_ticks": 456,
    "cgroup_path": "/user.slice/recovery-test.service",
    "process_group_id": 123, "session_id": 123,
}


def setup_models(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    for key in controller.MODEL_KEYS:
        model = model_root / key
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
    monkeypatch.setattr(controller, "SFT", model_root / "sft")
    directory = tmp_path / "run"
    directory.mkdir()
    monkeypatch.setattr(controller, "verify_prepared", lambda path: MANIFEST_SHA256)
    monkeypatch.setattr(controller, "config_for_holder", lambda *a, **kw: {})
    monkeypatch.setattr(controller, "freeze_config", lambda *a: None)
    monkeypatch.setattr(controller, "release_for_inference", lambda *a: {"driver_free_bytes": 1})
    monkeypatch.setattr(controller, "controller_idle", lambda *a: None)
    return directory, model_root


def test_worker_failure_stops_without_retry_or_starting_remaining_models(tmp_path, monkeypatch):
    directory, models = setup_models(tmp_path, monkeypatch)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(controller, "run_monitored", execute)
    monkeypatch.setattr(controller.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="remaining slots unstarted"):
        controller.run(directory, models, tmp_path / "holder")
    state = json.loads((directory / "controller_state.json").read_text())
    assert state["status"] == "FAILED"
    assert len(commands) == len(state["models"]) == 1
    assert state["models"][0]["returncode"] == 3
    with pytest.raises(RuntimeError, match="no retry or overwrite"):
        controller.run(directory, models, tmp_path / "holder")
    assert len(commands) == 1


def test_success_runs_each_model_once_and_then_readout(tmp_path, monkeypatch):
    directory, models = setup_models(tmp_path, monkeypatch)
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller, "run_monitored", execute)
    monkeypatch.setattr(controller.subprocess, "run", execute)
    result = controller.run(directory, models, tmp_path / "holder")
    assert result["status"] == "COMPLETE"
    assert [row["model_key"] for row in result["models"]] == list(controller.MODEL_KEYS)
    assert len(commands) == len(controller.MODEL_KEYS) + 1
    assert "readout" in commands[-1]
    assert result["manifest_sha256"] == MANIFEST_SHA256
    for command in commands[:-1]:
        index = command.index("--expected-manifest-sha256")
        assert command[index + 1] == MANIFEST_SHA256


def test_combined_evaluation_binds_both_manifests_and_reads_both_results(tmp_path, monkeypatch):
    from src.agents import order_recovery_episode as episode

    directory, models = setup_models(tmp_path, monkeypatch)
    complete = tmp_path / "complete"
    complete.mkdir()
    digest = "b" * 64
    (complete / "state.json").write_text(json.dumps({
        "status": "PREPARED", "manifest_sha256": digest,
    }))
    verified, commands = [], []
    monkeypatch.setattr(episode, "verify", lambda path, expected: verified.append((path, expected)))

    def execute(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller, "run_monitored", execute)
    monkeypatch.setattr(controller.subprocess, "run", execute)
    result = controller.run(directory, models, tmp_path / "holder", complete)
    assert verified == [(complete.resolve(), digest)]
    assert result["episode_manifest_sha256"] == digest
    assert len(commands) == 7
    for command in commands[:5]:
        assert command[command.index("--episode-directory") + 1] == str(complete)
        assert command[command.index("--expected-episode-sha256") + 1] == digest
    assert "src.agents.order_recovery_probe" in commands[-2]
    assert "src.agents.order_recovery_episode" in commands[-1]


def test_capacity_rejection_does_not_release_held_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "controller_claim", lambda config: calls.append("claim"))
    monkeypatch.setattr(controller, "admission_snapshot", lambda config: {"available_bytes": 1})
    monkeypatch.setattr(controller, "request", lambda *args: calls.append("release"))
    with pytest.raises(RuntimeError, match="reclaimable inference capacity"):
        controller.release_for_inference({})
    assert calls == ["claim"]


def holder_state(payload=0):
    return {
        "payload_bytes": payload,
        "allocator_allocated_bytes": payload,
        "allocator_reserved_bytes": payload,
        "paused": True,
        "claim": {
            "lease_token": "test-lease", "owner": deepcopy(OWNER),
            "cgroup_populated": True, "watchdog_error": None,
        },
    }


def admission(state, *, free=45):
    return {
        "available_bytes": free * controller.GIB + state["payload_bytes"],
        "driver_free_bytes": free * controller.GIB,
        "holder_status": state,
    }


def test_release_checks_the_final_healthy_claim_and_records_controller_scope(monkeypatch):
    held = holder_state(30 * controller.GIB)
    empty = holder_state()
    calls = []
    snapshots = iter([admission(held, free=15), admission(empty)])
    monkeypatch.setattr(controller, "controller_claim", lambda config: calls.append("claim"))
    monkeypatch.setattr(controller, "process_identity", lambda pid: deepcopy(OWNER))
    monkeypatch.setattr(controller, "reservation_status", lambda config: held)
    monkeypatch.setattr(controller, "admission_snapshot", lambda config: next(snapshots))
    monkeypatch.setattr(controller, "request", lambda directory, value: calls.append(value))
    result = controller.release_for_inference({"directory": "/holder", "lease_token": "test-lease"})
    assert calls[0] == "claim"
    assert calls[1]["op"] == "release" and calls[1]["bytes"] == 30 * controller.GIB
    assert calls[1]["lease_token"] == "test-lease"
    assert result["controller_owner"] == OWNER
    assert "KillMode=control-group" in result["descendant_cleanup_contract"]


@pytest.mark.parametrize("change", ["token", "owner", "unknown_cgroup", "empty_cgroup", "watchdog"])
def test_final_claim_uncertainty_never_admits_a_worker(monkeypatch, change):
    held = holder_state(30 * controller.GIB)
    final = holder_state()
    claim = final["claim"]
    if change == "token":
        claim["lease_token"] = "another-lease"
    elif change == "owner":
        claim["owner"]["start_time_ticks"] += 1
    elif change == "unknown_cgroup":
        claim["cgroup_populated"] = None
    elif change == "empty_cgroup":
        claim["cgroup_populated"] = False
    else:
        claim["watchdog_error"] = "PermissionError: cgroup.events unreadable"
    snapshots = iter([admission(held), admission(final)])
    monkeypatch.setattr(controller, "controller_claim", lambda config: None)
    monkeypatch.setattr(controller, "process_identity", lambda pid: deepcopy(OWNER))
    monkeypatch.setattr(controller, "reservation_status", lambda config: held)
    monkeypatch.setattr(controller, "admission_snapshot", lambda config: next(snapshots))
    monkeypatch.setattr(controller, "request", lambda *args: None)
    with pytest.raises(RuntimeError, match="healthy live service claim"):
        controller.release_for_inference({"directory": "/holder", "lease_token": "test-lease"})


def test_nonservice_controller_scope_is_rejected_before_release(monkeypatch):
    owner = {**OWNER, "cgroup_path": "/user.slice/session.scope"}
    state = holder_state(30 * controller.GIB)
    state["claim"]["owner"] = owner
    calls = []
    monkeypatch.setattr(controller, "controller_claim", lambda config: None)
    monkeypatch.setattr(controller, "process_identity", lambda pid: owner)
    monkeypatch.setattr(controller, "reservation_status", lambda config: state)
    monkeypatch.setattr(controller, "admission_snapshot", lambda config: admission(state))
    monkeypatch.setattr(controller, "request", lambda *args: calls.append("release"))
    with pytest.raises(RuntimeError, match="healthy live service claim"):
        controller.release_for_inference({"directory": "/holder", "lease_token": "test-lease"})
    assert not calls


def test_released_holder_with_insufficient_actual_capacity_times_out(monkeypatch):
    state = holder_state()
    times = iter([0.0, 121.0])
    snapshots = iter([admission(state), admission(state, free=20)])
    monkeypatch.setattr(controller, "controller_claim", lambda config: None)
    monkeypatch.setattr(controller, "process_identity", lambda pid: deepcopy(OWNER))
    monkeypatch.setattr(controller, "reservation_status", lambda config: state)
    monkeypatch.setattr(controller, "admission_snapshot", lambda config: next(snapshots))
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="unavailable after holder release"):
        controller.release_for_inference({"directory": "/holder", "lease_token": "test-lease"})


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("worker", 1800), KeyboardInterrupt()])
def test_worker_exceptions_mark_the_active_slot_failed_and_leave_remaining_slots_unstarted(
    tmp_path, monkeypatch, error,
):
    directory, models = setup_models(tmp_path, monkeypatch)
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        raise error

    monkeypatch.setattr(controller, "run_monitored", execute)
    monkeypatch.setattr(controller.subprocess, "run", execute)
    with pytest.raises(type(error)):
        controller.run(directory, models, tmp_path / "holder")
    state = json.loads((directory / "controller_state.json").read_text())
    assert state["status"] == "FAILED"
    assert len(calls) == len(state["models"]) == 1
    assert state["models"][0]["status"] == "FAILED"
    assert type(error).__name__ in state["models"][0]["error"]
    assert calls[0][1]["evidence_path"] == directory / "gpu-sft.jsonl"
    assert calls[0][1]["timeout"] == 1800


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "G" * 64, True])
def test_prepared_digest_must_exist_and_be_valid_before_verify(tmp_path, monkeypatch, digest):
    from src.agents import order_recovery_probe as probe

    (tmp_path / "state.json").write_text(json.dumps({
        "status": "PREPARED", "manifest_sha256": digest,
    }))
    calls = []
    monkeypatch.setattr(probe, "verify", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="prepared manifest SHA-256"):
        controller.verify_prepared(tmp_path)
    assert calls == []


def test_prepared_manifest_digest_is_passed_to_source_verification(tmp_path, monkeypatch):
    from src.agents import order_recovery_probe as probe

    (tmp_path / "state.json").write_text(json.dumps({
        "status": "PREPARED", "manifest_sha256": MANIFEST_SHA256,
    }))
    calls = []
    monkeypatch.setattr(probe, "verify", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert controller.verify_prepared(tmp_path) == MANIFEST_SHA256
    assert calls == [((tmp_path,), {"expected_sha256": MANIFEST_SHA256})]


def test_changed_frozen_manifest_stops_before_holder_configuration_or_worker_launch(
    tmp_path, monkeypatch,
):
    directory, models = setup_models(tmp_path, monkeypatch)
    calls = []

    def reject(path):
        raise ValueError("Recovery manifest digest differs")

    monkeypatch.setattr(controller, "verify_prepared", reject)
    monkeypatch.setattr(controller, "config_for_holder", lambda *a, **kw: calls.append("holder"))
    monkeypatch.setattr(controller, "run_monitored", lambda *a, **kw: calls.append("worker"))
    with pytest.raises(ValueError, match="manifest digest differs"):
        controller.run(directory, models, tmp_path / "holder")
    state = json.loads((directory / "controller_state.json").read_text())
    assert state["status"] == "FAILED" and not state["models"]
    assert calls == []
