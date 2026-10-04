"""One-time holder release before model initialization; never an atomic GPU transfer.

The queue estimates free driver memory plus actual held tensors. The worker
pauses/releases the identified holder, then admits from a fresh CUDA free-memory
reading. Another process can still allocate before model loading. No retries,
model loading, allocations, or automatic holder refills are performed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.agents.gpu_reservation_service import request

PROFILE = "bounded_holder_handoff_v1"
GIB = 1024**3
MINIMUM_FREE_BYTES = 60 * GIB
MINIMUM_LEASE_SECONDS = 300
_ATTEMPTS: dict[int, dict] = {}
_CONFIG_KEYS = {"profile", "directory", "owner_sha256", "pid", "pid_start_ticks",
                "gpu_uuid", "cuda_device", "minimum_free_bytes"}
_GROWTH_STATES = {"GROWING", "WAITING_FOR_MEMORY", "HOLDING"}


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="milliseconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _absolute(value) -> Path:
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise RuntimeError("handoff paths must be absolute and resolved")
    return path


def _process_start_ticks(pid: int) -> int:
    directory = Path(f"/proc/{pid}")
    if directory.stat().st_uid != os.getuid():
        raise RuntimeError("holder process belongs to a different user")
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ProcessLookupError("holder process is a zombie")
    return int(fields[19])  # /proc stat field 22; field 3 follows the command name.


def _driver_snapshot() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    rows = [row.strip().split(",") for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError("handoff requires exactly one visible nvidia-smi GPU")
    gpu_uuid, free, total = (part.strip() for part in rows[0])
    free_bytes, total_bytes = int(free) * 1024**2, int(total) * 1024**2
    if not gpu_uuid.startswith("GPU-") or not 0 <= free_bytes <= total_bytes:
        raise RuntimeError("invalid driver memory/UUID telemetry")
    return {"gpu_uuid": gpu_uuid, "driver_free_bytes": free_bytes,
            "driver_total_bytes": total_bytes}


def _validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise RuntimeError("invalid handoff configuration fields")
    if config["profile"] != PROFILE or not _digest(config["owner_sha256"]):
        raise RuntimeError("invalid handoff profile or owner digest")
    _absolute(config["directory"])
    for field in ("pid", "pid_start_ticks"):
        if type(config[field]) is not int or config[field] <= 0:
            raise RuntimeError(f"invalid holder {field}")
    if (type(config["cuda_device"]) is not int or config["cuda_device"] != 0
            or type(config["minimum_free_bytes"]) is not int
            or config["minimum_free_bytes"] != MINIMUM_FREE_BYTES
            or not isinstance(config["gpu_uuid"], str)
            or not config["gpu_uuid"].startswith("GPU-")):
        raise RuntimeError("handoff requires the fixed single-GPU 60 GiB admission profile")
    return dict(config)


def _owner(config: dict) -> dict:
    content = (Path(config["directory"]) / "owner.json").read_bytes()
    if _sha(content) != config["owner_sha256"]:
        raise RuntimeError("holder owner.json changed")
    owner = json.loads(content)
    for key in ("pid", "directory", "cuda_device"):
        if owner.get(key) != config[key]:
            raise RuntimeError(f"holder owner {key} mismatch")
    if (owner.get("allocation_only") is not True
            or owner.get("automatic_restart") is not False):
        raise RuntimeError("holder must be allocation-only with automatic restart disabled")
    return owner


def _same_gpu(config: dict) -> dict:
    snapshot = _driver_snapshot()
    if snapshot["gpu_uuid"] != config["gpu_uuid"]:
        raise RuntimeError("holder/worker GPU UUID changed")
    return snapshot


def _validate_state(config: dict, state: dict, owner: dict) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("invalid live holder status")
    for key in ("pid", "directory", "cuda_device", "started_at_SGT", "expires_at_SGT"):
        if state.get(key) != owner.get(key):
            raise RuntimeError(f"live holder {key} mismatch")
    for key in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes",
                "maximum_bytes"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise RuntimeError(f"invalid holder {key}")
    if not (state["payload_bytes"] <= state["allocator_allocated_bytes"]
            <= state["allocator_reserved_bytes"] <= state["maximum_bytes"] <= 60 * GIB):
        raise RuntimeError("inconsistent holder allocation telemetry")
    if state["maximum_bytes"] <= 0:
        raise RuntimeError("invalid holder maximum")
    if config["cuda_device"] != state["cuda_device"]:
        raise RuntimeError("holder CUDA device changed")


def config_for_holder(directory) -> dict:
    directory = Path(directory).resolve()
    content = (directory / "owner.json").read_bytes()
    owner = json.loads(content)
    config = _validate_config(dict(
        profile=PROFILE, directory=str(directory), owner_sha256=_sha(content),
        pid=owner["pid"], pid_start_ticks=_process_start_ticks(owner["pid"]),
        gpu_uuid=_driver_snapshot()["gpu_uuid"], cuda_device=owner["cuda_device"],
        minimum_free_bytes=MINIMUM_FREE_BYTES,
    ))
    reservation_status(config)
    return config


def load_config(path, expected_sha256: str) -> dict:
    if not _digest(expected_sha256):
        raise RuntimeError("a frozen handoff configuration SHA256 is required")
    content = _absolute(path).read_bytes()
    if _sha(content) != expected_sha256:
        raise RuntimeError("handoff configuration digest mismatch")
    return _validate_config(json.loads(content))


def reservation_status(
    config: dict, *, minimum_lease_seconds: float = MINIMUM_LEASE_SECONDS,
) -> dict:
    """Read the live control socket, checking the frozen process and GPU identity."""
    config = _validate_config(config)
    if (type(minimum_lease_seconds) not in (int, float)
            or not math.isfinite(minimum_lease_seconds) or minimum_lease_seconds < 0):
        raise ValueError("minimum lease must be finite and nonnegative")
    owner = _owner(config)
    if _process_start_ticks(config["pid"]) != config["pid_start_ticks"]:
        raise RuntimeError("holder PID was reused")
    _same_gpu(config)
    state = request(config["directory"], {"op": "status"})
    _validate_state(config, state, owner)
    if state.get("closed") is not False or state.get("status") in {
        "CLOSED", "EXPIRED", "FAILED"
    } or state.get("error") or state.get("cleanup_error"):
        raise RuntimeError("holder is closed or failed")
    remaining = state.get("remaining_seconds")
    expiry = datetime.fromisoformat(owner["expires_at_SGT"])
    if expiry.tzinfo is None:
        raise RuntimeError("holder expiry must include a timezone")
    wall_remaining = (expiry - datetime.now(expiry.tzinfo)).total_seconds()
    if (type(remaining) not in (int, float) or not math.isfinite(remaining)
            or remaining <= 0 or min(remaining, wall_remaining) < minimum_lease_seconds
            or wall_remaining <= 0):
        raise RuntimeError("holder lease has expired or has insufficient time remaining")
    # Detect a process/owner replacement during the socket round trip as well.
    _owner(config)
    if _process_start_ticks(config["pid"]) != config["pid_start_ticks"]:
        raise RuntimeError("holder process changed during status request")
    return state


def _zero_paused(state: dict) -> None:
    if (state.get("paused") is not True or state.get("status") != "PAUSED"
            or any(state.get(key) != 0 for key in (
                "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"))):
        raise RuntimeError("holder must remain paused with zero payload and allocator residency")


def available_bytes(config: dict) -> dict:
    """Conservative queue estimate, never a promise that subsequent allocation wins."""
    before = reservation_status(config)
    driver = _same_gpu(config)
    after = reservation_status(config)
    for state in (before, after):
        if state["status"] == "PAUSED":
            _zero_paused(state)  # Between training arms: no automatic refilling.
        elif state["status"] not in _GROWTH_STATES or state.get("paused") is not False:
            raise RuntimeError("holder is not in an admissible queue state")
    payload = min(before["payload_bytes"], after["payload_bytes"])
    return {**driver, "holder_payload_bytes": payload,
            "available_bytes": driver["driver_free_bytes"] + payload,
            "minimum_free_bytes": config["minimum_free_bytes"], "holder_status": after}


def _worker_free_bytes(config: dict) -> dict:
    import torch

    _same_gpu(config)
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("handoff worker must expose exactly the identified single CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = getattr(properties, "uuid", None)
    if gpu_uuid is not None:
        if isinstance(gpu_uuid, bytes):
            gpu_uuid = gpu_uuid.decode()
        if str(gpu_uuid).removeprefix("GPU-") != config["gpu_uuid"].removeprefix("GPU-"):
            raise RuntimeError("worker CUDA GPU UUID differs from the holder")
    free, total = torch.cuda.mem_get_info(0)
    return {"driver_free_bytes": int(free), "driver_total_bytes": int(total),
            "gpu_uuid": config["gpu_uuid"], "source": "torch.cuda.mem_get_info"}


def _verify_inactive(config: dict) -> dict:
    try:
        state = reservation_status(config, minimum_lease_seconds=0)
    except (OSError, RuntimeError):
        # After a successful handoff only: a naturally ended, fully cleaned lease
        # is safe if the original process is gone. Static files never admit a run.
        owner = _owner(config)
        _same_gpu(config)
        try:
            _process_start_ticks(config["pid"])
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("cannot verify inactive holder while its PID still exists") from None
        state = json.loads((Path(config["directory"]) / "status.json").read_text())
        _validate_state(config, state, owner)
        if (state.get("status") not in {"EXPIRED", "CLOSED"}
                or state.get("closed") is not True or state.get("paused") is not True
                or state.get("error") or state.get("cleanup_error")
                or any(state.get(key) != 0 for key in (
                    "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"))):
            raise RuntimeError("exited holder lacks clean terminal evidence") from None
        return state
    _zero_paused(state)
    return state


def _write_receipt(path: Path, receipt: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def worker_handoff_from_environment(role: str, identity: str) -> dict | None:
    """Gate critic→reference→actor initialization in one PID; fail closed after errors."""
    config_path = os.environ.get("FINANCE_GPU_HANDOFF_CONFIG")
    if not config_path:
        return None
    pid = os.getpid()
    previous = _ATTEMPTS.get(pid)
    if previous is not None and previous["failed"]:
        raise RuntimeError("this worker handoff previously failed; automatic retry is forbidden")
    attempt = previous or {"failed": True, "identity": identity, "config_path": config_path,
                           "config_sha256": os.environ.get("FINANCE_GPU_HANDOFF_CONFIG_SHA256")}
    _ATTEMPTS[pid] = attempt
    receipt = attempt.get("receipt", {
        "profile": PROFILE, "worker_pid": pid, "identity": identity, "first_role": role,
        "started_at_SGT": _now(), "status": "STARTED", "role_checks": [],
        "atomic_transfer_guaranteed": False,
    })
    path = attempt.get("receipt_path")
    try:
        if path is None:
            directory = _absolute(os.environ["FINANCE_GPU_HANDOFF_EVIDENCE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"receipt-{pid}.json"
            if path.exists():
                raise RuntimeError("worker handoff receipt already exists; no retry is allowed")
            attempt["receipt_path"] = path
        if not _digest(identity) or role not in {"critic", "reference", "actor"}:
            raise RuntimeError("a frozen identity and a known model role are required")
        if os.environ.get("FINANCE_SCREENING_RESOURCE_PROFILE") != "chunked_adam_disk_v1":
            raise RuntimeError("GPU handoff requires the chunked_adam_disk_v1 resource profile")
        if (attempt["identity"] != identity or attempt["config_path"] != config_path
                or attempt["config_sha256"]
                != os.environ.get("FINANCE_GPU_HANDOFF_CONFIG_SHA256")):
            raise RuntimeError("worker handoff identity or configuration changed")
        config = load_config(config_path, attempt["config_sha256"])
        receipt["holder_config"] = config
        if previous is not None:
            if role == "critic":
                raise RuntimeError("critic initialization cannot repeat in a handed-off worker")
            state = _verify_inactive(config)
            receipt["role_checks"].append({"role": role, "at_SGT": _now(),
                                           "holder_status": state})
            _write_receipt(path, receipt)
            return receipt
        if role != "critic":
            raise RuntimeError("first handoff role must be critic, before any model allocation")
        receipt["before"] = reservation_status(config)
        if receipt["before"]["status"] == "PAUSED":
            _zero_paused(receipt["before"])
        elif receipt["before"]["status"] not in _GROWTH_STATES:
            raise RuntimeError("holder cannot be handed off from its current state")
        _write_receipt(path, receipt)
        key = f"handoff-{identity}-{pid}"
        request(config["directory"], {"op": "pause", "request_id": f"{key}-pause"})
        paused = reservation_status(config, minimum_lease_seconds=0)
        if paused.get("paused") is not True or paused["status"] != "PAUSED":
            raise RuntimeError("holder did not acknowledge paused growth")
        receipt["paused"] = paused
        _write_receipt(path, receipt)
        request(config["directory"], {
            "op": "release", "bytes": paused["maximum_bytes"],
            "request_id": f"{key}-release",
        })
        receipt["after"] = reservation_status(config, minimum_lease_seconds=0)
        _zero_paused(receipt["after"])
        receipt["worker_memory"] = _worker_free_bytes(config)
        if receipt["worker_memory"]["driver_free_bytes"] < config["minimum_free_bytes"]:
            raise RuntimeError("GPU free memory after handoff is below the required 60 GiB")
        # Check again after CUDA initialization, before returning to model loading.
        _verify_inactive(config)
        receipt.update(status="PASSED", completed_at_SGT=_now(), error=None)
        _write_receipt(path, receipt)
        attempt.update(failed=False, receipt=receipt)
        return receipt
    except BaseException as error:
        attempt["failed"] = True
        receipt.update(status="FAILED", completed_at_SGT=_now(),
                       error=f"{type(error).__name__}: {error}")
        if path is not None and attempt.get("receipt_path") == path:
            _write_receipt(path, receipt)
        raise
