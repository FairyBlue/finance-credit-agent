"""Launch the direct comparison controller and record its actual systemd exit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
PROFILE = "shared_recovery_sft_action_pooling_comparison_v1"
CONTINUATION_PROFILE = "action_pooling_explicit_remaining_v1"
LAUNCH_PROFILE = "action_pooling_systemd_service_v1"
EXIT_PROFILE = "action_pooling_systemd_exit_status_v1"
CPU_SMOKE_PROFILE = "action_pooling_systemd_cpu_smoke_v1"
DEFAULT_UNIT = "m2-action-pooling-comparison-20261012-v1.service"
GIB = 1024**3
MINIMUM_FREE_BYTES = 107 * GIB
PROPERTIES = (
    "Id", "LoadState", "ActiveState", "SubState", "Result", "InvocationID",
    "MainPID", "ExecMainPID", "ExecMainCode", "ExecMainStatus", "ControlPID",
    "ControlGroup", "ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic",
    "ExecStart", "ExecStopPost", "Type", "Restart", "KillMode", "MemoryMax",
    "RuntimeMaxUSec", "TimeoutStopUSec", "WorkingDirectory",
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked evidence: " + str(path))
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "Evidence must be an object: " + str(path))
    return value


def exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalize_unit(value: str) -> str:
    unit = value if value.endswith(".service") else value + ".service"
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:@-]*\.service", unit), "Invalid service unit")
    return unit


def _controller_module(directory: Path, contract: dict[str, Any]) -> str:
    if "continuation" not in contract:
        return "src.agents.action_pooling_comparison"
    continuation = contract["continuation"]
    require(isinstance(continuation, dict)
            and continuation.get("profile") == CONTINUATION_PROFILE,
            "Unknown continuation profile")
    expected = continuation.get("recovery_contract_sha256")
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Continuation recovery contract SHA256 is missing or invalid")
    path = directory / "recovery_contract.json"
    recovery = read(path)
    require(digest(path) == expected, "Continuation recovery contract hash differs")
    require(recovery.get("profile") == CONTINUATION_PROFILE,
            "Continuation recovery contract profile differs")
    return "src.agents.action_pooling_recovery"


def main_command(directory: Path) -> list[str]:
    module = _controller_module(directory, read(directory / "run_contract.json"))
    return [str(PYTHON), "-u", "-m", module, "run",
            "--directory", str(directory)]


def _contract(directory: Path) -> tuple[dict[str, Any], str, str, bool]:
    contract = read(directory / "run_contract.json")
    smoke = contract.get("profile") == CPU_SMOKE_PROFILE
    source = Path(__file__).resolve()
    source_sha = digest(source)
    if smoke:
        require(contract.get("cpu_smoke_only") is True and contract.get("gpu_requests") == 0,
                "CPU fixture must explicitly exclude GPU requests")
        require(contract.get("sources", {}).get(str(source)) == source_sha,
                "CPU recorder source binding differs")
    else:
        settings = read(directory / "comparison_settings.json")
        require(contract.get("profile") == PROFILE
                and type(contract.get("planned_trajectories")) is int
                and contract.get("planned_trajectories") == 1336
                and type(settings.get("planned_trajectories")) is int
                and settings.get("planned_trajectories") == 1336,
                "Service requires the fixed 1336-trajectory comparison")
        for name in ("comparison_settings.json", "source_identity.json"):
            require(contract.get("inputs", {}).get(name) == digest(directory / name),
                    "Frozen service input differs: " + name)
        sources = read(directory / "source_identity.json")
        relative = str(source.relative_to(ROOT))
        require(sources.get(relative) == source_sha
                and digest(directory / "sources" / relative) == source_sha,
                "Exit recorder is not bound by the root source contract and snapshot")
        _controller_module(directory, contract)
    return contract, digest(directory / "run_contract.json"), source_sha, smoke


def check_contract(directory: Path) -> None:
    # Delay importing the training orchestrator until an explicit launch.
    from src.agents.action_pooling_comparison import check_contract as check

    check(directory)


def _systemd_command(command: list[str]) -> str:
    require(all(isinstance(arg, str) and arg and not any(c in arg for c in "\0\n\r")
                for arg in command), "Invalid systemd command argument")
    # ':' disables systemd environment expansion; '%%' preserves literal specifiers.
    quoted = ["\"" + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
              + "\"" for arg in command]
    return ":" + " ".join(quoted)


def unit_properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--no-pager", "--property=" + ",".join(PROPERTIES)],
        capture_output=True, text=True, check=False, timeout=15,
    )
    require(result.returncode == 0, "Cannot inspect the actual systemd unit")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _launch(directory: Path, unit: str, *, cpu_smoke_command: list[str] | None = None) -> dict:
    directory = directory.resolve()
    unit = normalize_unit(unit)
    require(directory.is_dir(), "An existing prepared directory is required")
    for name in ("launch_spec.json", "launch_receipt.json", "service_exit_receipt.json"):
        require(not (directory / name).exists(), "Fresh launch required; existing " + name)
    _, contract_sha, source_sha, smoke = _contract(directory)
    require(smoke == (cpu_smoke_command is not None), "CPU fixture command/profile mismatch")
    if not smoke:
        check_contract(directory)
        require(shutil.disk_usage(directory).free >= MINIMUM_FREE_BYTES,
                "Comparison launch requires at least 107 GiB actual free disk")
        require(_contract(directory)[1:3] == (contract_sha, source_sha),
                "Root contract or recorder changed during launch validation")
    linger = subprocess.run(
        ["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    require(linger.returncode == 0 and linger.stdout.strip() == "yes",
            "User linger must be enabled")
    existing = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=LoadState", "--value"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    require(existing.stdout.strip() == "not-found", "A fresh dedicated service name is required")
    controller = cpu_smoke_command if smoke else main_command(directory)
    hook = [str(PYTHON), "-u", "-m", "src.agents.action_pooling_service", "record-exit",
            "--directory", str(directory), "--unit", unit,
            "--expected-contract-sha256", contract_sha,
            "--expected-recorder-sha256", source_sha]
    properties = {
        "Type": "exec", "Restart": "no", "KillMode": "control-group", "MemoryMax": "64G",
        "RuntimeMaxSec": "172800", "TimeoutStopSec": "30", "StandardInput": "null",
        "StandardOutput": "append:" + str(directory / "service.log"),
        "StandardError": "append:" + str(directory / "service.log"),
        "ExecStopPost": _systemd_command(hook),
    }
    environment = {
        "PYTHONPATH": str(ROOT) + ":" + str(ROOT / "third_party/agent-r1"),
        "CC": "/home/tianming/micromamba/envs/finance-credit/bin/x86_64-conda-linux-gnu-cc",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    command = ["systemd-run", "--user", "--unit=" + unit, "--working-directory=" + str(ROOT)]
    command += ["--property=" + key + "=" + value for key, value in properties.items()]
    command += ["--setenv=" + key + "=" + value for key, value in environment.items()]
    command += controller
    spec = {
        "schema_version": 1, "profile": LAUNCH_PROFILE, "unit": unit,
        "created_at_UTC": datetime.now(timezone.utc).isoformat(), "uid": os.getuid(),
        "run_contract_sha256": contract_sha, "recorder_path": str(Path(__file__).resolve()),
        "recorder_sha256": source_sha, "main_command": controller, "hook_command": hook,
        "systemd_properties": properties, "environment": environment, "command": command,
        "cpu_smoke_only": smoke, "automatic_retry": False,
        "controller_ran_without_recorder_parent": True,
    }
    exclusive(directory / "launch_spec.json", spec)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        exclusive(directory / "launch_receipt.json", {
            "profile": LAUNCH_PROFILE, "status": "LAUNCH_ERROR", "unit": unit,
            "run_contract_sha256": contract_sha,
            "launch_spec_sha256": digest(directory / "launch_spec.json"),
            "error_type": type(error).__name__, "error": str(error), "automatic_retry": False,
        })
        raise
    receipt = {
        "profile": LAUNCH_PROFILE, "status": "LAUNCHED" if result.returncode == 0 else "FAILED",
        "unit": unit, "recorded_at_UTC": datetime.now(timezone.utc).isoformat(),
        "run_contract_sha256": contract_sha, "recorder_sha256": source_sha,
        "launch_spec_sha256": digest(directory / "launch_spec.json"),
        "systemd_run_returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "automatic_retry": False, "cpu_smoke_only": smoke,
    }
    exclusive(directory / "launch_receipt.json", receipt)
    result.check_returncode()
    return receipt


def launch(directory: Path, unit: str = DEFAULT_UNIT) -> dict:
    return _launch(directory, unit)


def process_identity(unit: str) -> dict[str, Any]:
    pid = os.getpid()
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    groups = [line.split(":", 2)[2]
              for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
              if line.startswith("0::")]
    require(len(groups) == 1 and groups[0].split("/")[-1] == unit,
            "Exit recorder must run in the actual service stop-post cgroup")
    return {
        "pid": pid, "start_time_ticks": int(fields[19]), "uid": os.getuid(),
        "process_group_id": os.getpgid(pid), "session_id": os.getsid(pid),
        "cgroup_path": groups[0], "unit": unit,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip().replace("-", ""),
        "invocation_id": os.environ.get("INVOCATION_ID"),
    }


def _duration_us(value: str) -> int:
    if value.isdecimal():
        return int(value)
    units = {"us": 1, "ms": 1000, "s": 10**6, "min": 60 * 10**6,
             "h": 3600 * 10**6, "d": 86400 * 10**6}
    match = re.fullmatch(r"([0-9]+)(us|ms|s|min|h|d)", value)
    require(match, "Unexpected systemd time property")
    return int(match[1]) * units[match[2]]


def record_exit(directory: Path, unit: str, expected_contract_sha256: str,
                expected_recorder_sha256: str) -> dict:
    directory = directory.resolve()
    unit = normalize_unit(unit)
    require(not (directory / "service_exit_receipt.json").exists(), "Exit receipt already exists")
    _, contract_sha, source_sha, smoke = _contract(directory)
    require((contract_sha, source_sha) == (expected_contract_sha256, expected_recorder_sha256),
            "Frozen root contract or recorder changed")
    spec = read(directory / "launch_spec.json")
    require(spec.get("profile") == LAUNCH_PROFILE and spec.get("unit") == unit
            and spec.get("run_contract_sha256") == contract_sha
            and spec.get("recorder_sha256") == source_sha
            and spec.get("recorder_path") == str(Path(__file__).resolve())
            and spec.get("cpu_smoke_only") is smoke and spec.get("uid") == os.getuid(),
            "Launch specification bindings differ")
    if not smoke:
        require(spec.get("main_command") == main_command(directory),
                "Controller must run directly as the main process")
    environment = {key: os.environ.get(key) for key in
                   ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS", "INVOCATION_ID")}
    require(all(isinstance(value, str) and value for value in environment.values())
            and re.fullmatch(r"[0-9a-f]{32}", environment["INVOCATION_ID"]),
            "Actual ExecStopPost status environment is missing")
    process, props = process_identity(unit), unit_properties(unit)
    require(props.get("Id") == unit and props.get("LoadState") == "loaded"
            and props.get("ActiveState") == "deactivating" and props.get("SubState") == "stop-post"
            and props.get("MainPID") == "0" and props.get("ControlPID") == str(process["pid"])
            and props.get("InvocationID") == environment["INVOCATION_ID"]
            == process["invocation_id"]
            and props.get("ControlGroup") == process["cgroup_path"],
            "Recorder is not the actual stop-post process after main exit")
    require(props.get("Type") == "exec" and props.get("Restart") == "no"
            and props.get("KillMode") == "control-group" and props.get("MemoryMax") == str(64 * GIB)
            and _duration_us(props.get("RuntimeMaxUSec", "")) == 172800 * 10**6
            and _duration_us(props.get("TimeoutStopUSec", "")) == 30 * 10**6
            and props.get("WorkingDirectory") == str(ROOT), "Actual service limits differ")
    commands = (("ExecStart", "main_command"), ("ExecStopPost", "hook_command"))
    for property_name, command_name in commands:
        expected = "argv[]=" + " ".join(spec[command_name]) + " ; ignore_errors=no ;"
        require(expected in props.get(property_name, "")
                and props[property_name].count("{ path=") == 1,
                "Actual systemd command differs from launch specification")
    main_pid = int(props.get("ExecMainPID", "0"))
    require(main_pid > 0 and not Path(f"/proc/{main_pid}").exists(),
            "Main process is still present")
    main_start = int(props.get("ExecMainStartTimestampMonotonic", "0"))
    main_exit = int(props.get("ExecMainExitTimestampMonotonic", "0"))
    now = time.monotonic_ns() // 1000
    require(0 < main_start <= main_exit <= now, "Actual main exit chronology is missing")
    code, status = environment["EXIT_CODE"], environment["EXIT_STATUS"]
    require(props.get("Result") == environment["SERVICE_RESULT"], "Systemd results differ")
    returncode = None
    if code == "exited":
        require(status.isdecimal() and 0 <= int(status) <= 255, "Invalid numeric main exit status")
        returncode = int(status)
        require(props.get("ExecMainCode") == "1" and props.get("ExecMainStatus") == status,
                "Numeric main wait status differs from ExecStopPost")
    else:
        require(code in {"killed", "dumped"}, "Unknown actual main exit code")
        signal_number = int(status) if status.isdecimal() else signal.Signals["SIG" + status].value
        require(props.get("ExecMainCode") == ("2" if code == "killed" else "3")
                and props.get("ExecMainStatus") == str(signal_number), "Main signal status differs")
    receipt = {
        "schema_version": 1, "profile": EXIT_PROFILE, "status": "RECORDED", "unit": unit,
        "recorded_at_UTC": datetime.now(timezone.utc).isoformat(), "recorded_monotonic_us": now,
        "run_contract_sha256": contract_sha, "recorder_sha256": source_sha,
        "launch_spec_sha256": digest(directory / "launch_spec.json"),
        "source_sha256": {str(Path(__file__).resolve()): source_sha},
        "identity": process, "systemd_environment": environment, "actual_unit_properties": props,
        "main_pid": main_pid, "main_process_absent_when_hook_ran": True,
        "main_start_monotonic_us": main_start, "main_exit_monotonic_us": main_exit,
        "controller_returncode": returncode, "numeric_exit_status_recorded": returncode is not None,
        "successful_main_exit": environment["SERVICE_RESULT"] == "success" and returncode == 0,
        "exit_status_source": "systemd ExecStopPost SERVICE_RESULT/EXIT_CODE/EXIT_STATUS",
        "automatic_retry": False, "controller_ran_without_recorder_parent": True,
        "cpu_smoke_only": smoke,
    }
    exclusive(directory / "service_exit_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launcher = sub.add_parser("launch")
    launcher.add_argument("--directory", type=Path, required=True)
    launcher.add_argument("--unit", default=DEFAULT_UNIT)
    recorder = sub.add_parser("record-exit")
    recorder.add_argument("--directory", type=Path, required=True)
    recorder.add_argument("--unit", required=True)
    recorder.add_argument("--expected-contract-sha256", required=True)
    recorder.add_argument("--expected-recorder-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "launch":
        result = launch(args.directory, args.unit)
    else:
        result = record_exit(args.directory, args.unit, args.expected_contract_sha256,
                             args.expected_recorder_sha256)
    print(json.dumps({key: result[key] for key in ("profile", "status", "unit")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
