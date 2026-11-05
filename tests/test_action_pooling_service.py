"""CPU launch contracts and actual ExecStopPost evidence, including failed mains."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import action_pooling_service as service


def write(path, value):
    path.write_text(json.dumps(value))


def bind_continuation(directory):
    recovery = directory / "recovery_contract.json"
    write(recovery, {"profile": service.CONTINUATION_PROFILE,
                     "planned_trajectories": 1336, "remaining_trajectories": 1124})
    contract = json.loads((directory / "run_contract.json").read_text())
    contract["continuation"] = {"profile": service.CONTINUATION_PROFILE,
                                "recovery_contract_sha256": service.digest(recovery)}
    write(directory / "run_contract.json", contract)


@pytest.fixture
def prepared(tmp_path):
    source = Path(service.__file__).resolve()
    relative = str(source.relative_to(service.ROOT))
    snapshot = tmp_path / "sources" / relative
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    write(tmp_path / "source_identity.json", {relative: service.digest(source)})
    write(tmp_path / "comparison_settings.json", {"planned_trajectories": 1336})
    write(tmp_path / "run_contract.json", {
        "profile": service.PROFILE, "planned_trajectories": 1336,
        "inputs": {name: service.digest(tmp_path / name)
                   for name in ("comparison_settings.json", "source_identity.json")},
        "child_contracts": {},
    })
    return tmp_path


@pytest.fixture
def launch_mocks(monkeypatch):
    calls = []
    checks = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "loginctl":
            return subprocess.CompletedProcess(command, 0, "yes\n", "")
        if command[0] == "systemctl":
            return subprocess.CompletedProcess(command, 1, "not-found\n", "")
        return subprocess.CompletedProcess(command, 0, "", "Running as unit\n")

    monkeypatch.setattr(service.subprocess, "run", run)
    monkeypatch.setattr(service, "check_contract", lambda base: checks.append(base))
    monkeypatch.setattr(service.shutil, "disk_usage",
                        lambda base: SimpleNamespace(free=107 * service.GIB))
    return calls, checks


def test_launch_direct_main_and_stop_post_have_frozen_bindings(prepared, launch_mocks):
    calls, checks = launch_mocks
    result = service.launch(prepared)
    spec = json.loads((prepared / "launch_spec.json").read_text())
    assert checks == [prepared]
    assert result["status"] == "LAUNCHED"
    assert result["systemd_run_returncode"] == 0
    assert result["launch_spec_sha256"] == service.digest(prepared / "launch_spec.json")
    assert spec["run_contract_sha256"] == service.digest(prepared / "run_contract.json")
    assert spec["recorder_sha256"] == service.digest(Path(service.__file__).resolve())
    command = calls[-1]
    assert command[-7:] == service.main_command(prepared)
    assert "--wait" not in command and "--collect" not in command
    assert "--property=Type=exec" in command
    assert "--property=Restart=no" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=MemoryMax=64G" in command
    assert "--property=RuntimeMaxSec=172800" in command
    assert "--property=TimeoutStopSec=30" in command
    assert spec["hook_command"][3:5] == ["src.agents.action_pooling_service", "record-exit"]
    assert spec["automatic_retry"] is False
    assert not (prepared / "service_exit_receipt.json").exists()


def test_continuation_launch_dispatches_directly_and_keeps_the_full_budget(prepared, launch_mocks):
    assert service.main_command(prepared)[3] == "src.agents.action_pooling_comparison"
    bind_continuation(prepared)
    calls, checks = launch_mocks
    service.launch(prepared)
    command = service.main_command(prepared)
    assert command == [str(service.PYTHON), "-u", "-m", "src.agents.action_pooling_recovery",
                       "run", "--directory", str(prepared)]
    assert calls[-1][-7:] == command
    assert checks == [prepared]
    contract, _, _, smoke = service._contract(prepared)
    assert contract["profile"] == service.PROFILE
    assert contract["planned_trajectories"] == 1336
    assert not smoke


@pytest.mark.parametrize("damage", [
    "unknown_profile", "null", "invalid_hash", "wrong_hash", "missing", "tamper", "profile",
])
def test_continuation_contract_damage_rejects_dispatch_and_launch(prepared, launch_mocks, damage):
    bind_continuation(prepared)
    contract = json.loads((prepared / "run_contract.json").read_text())
    recovery = prepared / "recovery_contract.json"
    if damage == "unknown_profile":
        contract["continuation"]["profile"] = "unapproved_continuation"
    elif damage == "null":
        contract["continuation"] = None
    elif damage == "invalid_hash":
        contract["continuation"]["recovery_contract_sha256"] = "not-a-sha256"
    elif damage == "wrong_hash":
        contract["continuation"]["recovery_contract_sha256"] = "0" * 64
    elif damage == "missing":
        recovery.unlink()
    else:
        value = json.loads(recovery.read_text())
        if damage == "profile":
            value["profile"] = "unapproved_recovery"
        else:
            value["remaining_trajectories"] = 1123
        write(recovery, value)
        if damage == "profile":
            contract["continuation"]["recovery_contract_sha256"] = service.digest(recovery)
    write(prepared / "run_contract.json", contract)
    with pytest.raises(ValueError):
        service.main_command(prepared)
    with pytest.raises(ValueError):
        service.launch(prepared)
    assert not any(call[0] == "systemd-run" for call in launch_mocks[0])


@pytest.mark.parametrize("damage", ["disk", "profile", "budget", "settings", "source", "receipt"])
def test_launch_rejects_invalid_prepared_contract_without_starting_unit(
    prepared, launch_mocks, monkeypatch, damage
):
    calls, _ = launch_mocks
    if damage == "disk":
        monkeypatch.setattr(service.shutil, "disk_usage",
                            lambda base: SimpleNamespace(free=107 * service.GIB - 1))
    elif damage == "receipt":
        write(prepared / "launch_receipt.json", {"preserved": True})
    elif damage == "settings":
        write(prepared / "comparison_settings.json", {"planned_trajectories": 1335})
    else:
        contract = json.loads((prepared / "run_contract.json").read_text())
        if damage == "profile":
            contract["profile"] = "old"
        elif damage == "budget":
            contract["planned_trajectories"] = 1335
        else:
            contract["inputs"]["source_identity.json"] = "0" * 64
        write(prepared / "run_contract.json", contract)
    with pytest.raises(ValueError):
        service.launch(prepared)
    assert not any(call[0] == "systemd-run" for call in calls)


def test_launch_failure_is_preserved_and_cannot_be_retried(prepared, launch_mocks, monkeypatch):
    original = service.subprocess.run

    def run(command, **kwargs):
        if command[0] == "systemd-run":
            return subprocess.CompletedProcess(command, 1, "", "actual launch failure")
        return original(command, **kwargs)

    monkeypatch.setattr(service.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        service.launch(prepared)
    receipt = json.loads((prepared / "launch_receipt.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["systemd_run_returncode"] == 1
    with pytest.raises(ValueError, match="Fresh launch"):
        service.launch(prepared)


@pytest.mark.parametrize("damage", ["linger", "existing_unit"])
def test_launch_requires_linger_and_unused_unit(prepared, launch_mocks, monkeypatch, damage):
    calls, _ = launch_mocks
    original = service.subprocess.run

    def run(command, **kwargs):
        if damage == "linger" and command[0] == "loginctl":
            return subprocess.CompletedProcess(command, 0, "no\n", "")
        if damage == "existing_unit" and command[0] == "systemctl":
            return subprocess.CompletedProcess(command, 0, "loaded\n", "")
        return original(command, **kwargs)

    monkeypatch.setattr(service.subprocess, "run", run)
    with pytest.raises(ValueError):
        service.launch(prepared)
    assert not any(call[0] == "systemd-run" for call in calls)


@pytest.fixture
def exiting(prepared, launch_mocks, monkeypatch, request):
    if getattr(request, "param", False):
        bind_continuation(prepared)
    service.launch(prepared)
    spec = json.loads((prepared / "launch_spec.json").read_text())
    unit = spec["unit"]
    invocation = "a" * 32
    identity = {"pid": 424200, "invocation_id": invocation,
                "cgroup_path": "/user.slice/test/" + unit, "uid": os.getuid()}
    props = {
        "Id": unit, "LoadState": "loaded", "ActiveState": "deactivating",
        "SubState": "stop-post", "Result": "success", "InvocationID": invocation,
        "MainPID": "0", "ExecMainPID": "99999999", "ExecMainCode": "1",
        "ExecMainStatus": "0", "ControlPID": str(identity["pid"]),
        "ControlGroup": identity["cgroup_path"], "ExecMainStartTimestampMonotonic": "100",
        "ExecMainExitTimestampMonotonic": "200", "Type": "exec", "Restart": "no",
        "KillMode": "control-group", "MemoryMax": str(64 * service.GIB),
        "RuntimeMaxUSec": "2d", "TimeoutStopUSec": "30s", "WorkingDirectory": str(service.ROOT),
    }
    for field, name in (("ExecStart", "main_command"), ("ExecStopPost", "hook_command")):
        props[field] = "{ path=python ; argv[]=" + " ".join(spec[name]) + " ; ignore_errors=no ; }"
    monkeypatch.setattr(service, "process_identity", lambda unit: identity)
    monkeypatch.setattr(service, "unit_properties", lambda unit: props)
    monkeypatch.setattr(service.time, "monotonic_ns", lambda: 1000 * 1000)
    for key, value in {"SERVICE_RESULT": "success", "EXIT_CODE": "exited", "EXIT_STATUS": "0",
                       "INVOCATION_ID": invocation}.items():
        monkeypatch.setenv(key, value)
    return prepared, spec, props


def record(exiting):
    directory, spec, _ = exiting
    return service.record_exit(directory, spec["unit"], spec["run_contract_sha256"],
                               spec["recorder_sha256"])


@pytest.mark.parametrize("code", [0, 7])
def test_actual_status_is_recorded_without_inventing_completion(exiting, monkeypatch, code):
    directory, _, props = exiting
    monkeypatch.setenv("EXIT_STATUS", str(code))
    monkeypatch.setenv("SERVICE_RESULT", "success" if code == 0 else "exit-code")
    props.update(ExecMainStatus=str(code), Result="success" if code == 0 else "exit-code")
    result = record(exiting)
    assert result["status"] == "RECORDED"
    assert result["controller_returncode"] == code
    assert result["successful_main_exit"] is (code == 0)
    assert "completion" not in result and "acceptance" not in result
    preserved = (directory / "service_exit_receipt.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        record(exiting)
    assert (directory / "service_exit_receipt.json").read_bytes() == preserved


@pytest.mark.parametrize("exiting", [True], indirect=True)
def test_continuation_exit_uses_the_same_numeric_status_proof(exiting):
    directory, spec, _ = exiting
    result = record(exiting)
    assert spec["main_command"][3] == "src.agents.action_pooling_recovery"
    assert result["controller_returncode"] == 0
    assert result["numeric_exit_status_recorded"] is True
    assert result["successful_main_exit"] is True
    assert result["run_contract_sha256"] == service.digest(directory / "run_contract.json")


@pytest.mark.parametrize("exiting", [True], indirect=True)
def test_continuation_tamper_before_exit_is_not_recorded(exiting):
    directory, _, _ = exiting
    recovery = directory / "recovery_contract.json"
    value = json.loads(recovery.read_text())
    value["remaining_trajectories"] = 1123
    write(recovery, value)
    with pytest.raises(ValueError, match="recovery contract hash differs"):
        record(exiting)
    assert not (directory / "service_exit_receipt.json").exists()


@pytest.mark.parametrize("field,value", [
    ("MainPID", "123"), ("ControlPID", "123"), ("InvocationID", "b" * 32),
    ("ControlGroup", "/wrong"), ("SubState", "running"), ("ActiveState", "active"),
    ("ExecMainPID", str(os.getpid())), ("ExecMainExitTimestampMonotonic", "0"),
    ("ExecMainExitTimestampMonotonic", "1001"), ("ExecMainStartTimestampMonotonic", "201"),
    ("ExecMainStatus", "7"), ("Result", "exit-code"), ("Restart", "always"),
    ("KillMode", "process"), ("MemoryMax", "infinity"), ("RuntimeMaxUSec", "infinity"),
    ("TimeoutStopUSec", "1min"), ("ExecStart", "wrapper"), ("ExecStopPost", "wrong hook"),
])
def test_wrong_exit_phase_identity_limits_or_chronology_is_not_recorded(exiting, field, value):
    directory, _, props = exiting
    props[field] = value
    with pytest.raises(ValueError):
        record(exiting)
    assert not (directory / "service_exit_receipt.json").exists()


@pytest.mark.parametrize("field", ["SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS", "INVOCATION_ID"])
def test_missing_real_systemd_environment_is_not_recorded(exiting, monkeypatch, field):
    directory, _, _ = exiting
    monkeypatch.delenv(field)
    with pytest.raises(ValueError, match="environment"):
        record(exiting)
    assert not (directory / "service_exit_receipt.json").exists()


@pytest.mark.parametrize(
    "damage", ["contract", "source", "main", "inventory", "snapshot", "settings"]
)
def test_changed_contract_source_or_direct_main_binding_is_rejected(exiting, damage):
    directory, spec, _ = exiting
    if damage == "contract":
        contract = json.loads((directory / "run_contract.json").read_text())
        contract["later_change"] = True
        write(directory / "run_contract.json", contract)
    elif damage == "source":
        spec["recorder_sha256"] = "0" * 64
        write(directory / "launch_spec.json", spec)
    elif damage == "main":
        spec["main_command"] = ["foreground-wrapper"]
        write(directory / "launch_spec.json", spec)
    elif damage == "inventory":
        write(directory / "source_identity.json", {})
    elif damage == "snapshot":
        relative = Path(service.__file__).resolve().relative_to(service.ROOT)
        (directory / "sources" / relative).write_text("changed frozen snapshot")
    else:
        write(directory / "comparison_settings.json",
              {"planned_trajectories": 1336, "changed": True})
    with pytest.raises(ValueError):
        record(exiting)
    assert not (directory / "service_exit_receipt.json").exists()


def test_signal_exit_is_preserved_without_a_fake_numeric_zero(exiting, monkeypatch):
    _, _, props = exiting
    monkeypatch.setenv("SERVICE_RESULT", "signal")
    monkeypatch.setenv("EXIT_CODE", "killed")
    monkeypatch.setenv("EXIT_STATUS", "TERM")
    props.update(Result="signal", ExecMainCode="2", ExecMainStatus="15")
    result = record(exiting)
    assert result["controller_returncode"] is None
    assert result["successful_main_exit"] is False
    assert result["systemd_environment"]["EXIT_STATUS"] == "TERM"


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ACTION_POOLING_SERVICE_REAL_SMOKE") != "1",
                    reason="Explicit opt-in starts only two CPU fixture units")
@pytest.mark.parametrize("code", [0, 7])
def test_real_cpu_systemd_main_zero_and_seven(tmp_path, code):
    unit = f"fca-action-pooling-smoke-{uuid.uuid4().hex[:12]}-main{code}.service"
    source = Path(service.__file__).resolve()
    write(tmp_path / "run_contract.json", {
        "profile": service.CPU_SMOKE_PROFILE, "cpu_smoke_only": True, "gpu_requests": 0,
        "sources": {str(source): service.digest(source)}, "children": {},
    })
    marker_path = tmp_path / "main_marker.json"
    snippet = (
        "import json,os,pathlib,sys,time; "
        f"pathlib.Path({str(marker_path)!r}).write_text(json.dumps("
        "{'pid':os.getpid(),'invocation_id':os.environ.get('INVOCATION_ID'),"
        "'monotonic_us':time.monotonic_ns()//1000})); time.sleep(0.2); "
        f"sys.exit({code})"
    )
    service._launch(tmp_path, unit, cpu_smoke_command=[str(service.PYTHON), "-c", snippet])
    receipt_path = tmp_path / "service_exit_receipt.json"
    deadline = time.monotonic() + 30
    receipt = None
    while time.monotonic() < deadline:
        if receipt_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text())
                break
            except json.JSONDecodeError:
                pass
        time.sleep(0.1)
    if receipt is None:
        log = subprocess.run(["journalctl", "--user", "--unit", unit, "--no-pager", "-n", "30"],
                             capture_output=True, text=True, check=False, timeout=15)
        subprocess.run(["systemctl", "--user", "stop", unit], check=False, timeout=35)
        pytest.fail("Actual stop-post receipt missing: " + log.stdout + log.stderr)
    marker = json.loads(marker_path.read_text())
    assert receipt["controller_returncode"] == code
    assert receipt["main_pid"] == marker["pid"]
    assert receipt["identity"]["invocation_id"] == marker["invocation_id"]
    assert receipt["main_process_absent_when_hook_ran"] is True
    assert receipt["main_exit_monotonic_us"] >= marker["monotonic_us"]
    assert receipt["recorded_monotonic_us"] >= receipt["main_exit_monotonic_us"]
    assert receipt["successful_main_exit"] is (code == 0)
    assert receipt["cpu_smoke_only"] is True
    assert receipt["run_contract_sha256"] == service.digest(tmp_path / "run_contract.json")
    assert receipt["recorder_sha256"] == service.digest(source)
