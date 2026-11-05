"""Cgroup-bound GPU admission for the persistent project holder.

The first execution profile keeps the holder paused and empty for the entire
active training arm. The holder resumes idle occupancy only after the kernel
reports that the controller's dedicated systemd cgroup is empty. Every worker
allocation boundary still checks current CUDA capacity; a released allocation
is never an exclusive or atomic reservation against another GPU user.

Importing this module and preparing a configuration do not import torch or
initialize CUDA. Only the actual worker sampler uses its existing CUDA runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.agents.gpu_reservation_service import process_identity, request

PROFILE = "persistent_gpu_memory_coordinator_v1"
GIB = 1024**3
MIB = 1024**2
PHASES = frozenset({
    "init_critic", "init_reference", "init_actor", "rollout_mode", "trainer_mode",
    "compute_values", "compute_log_prob", "compute_ref_log_prob",
    "update_actor", "update_critic", "save_checkpoint",
})
_CONFIG_KEYS = {
    "profile", "directory", "owner_sha256", "holder_pid", "holder_start_time_ticks",
    "gpu_uuid", "cuda_device", "lease_token", "evidence_directory",
    "portfolio_target_bytes", "phase_peak_bytes", "minimum_free_bytes",
    "wait_timeout_seconds", "poll_seconds", "active_arm_holder_policy",
}
_COORDINATORS: dict[int, GpuMemoryCoordinator] = {}


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="milliseconds")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _absolute(value: Any) -> Path:
    if not isinstance(value, (str, Path)):
        raise RuntimeError("coordinator paths must be absolute and resolved")
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError("coordinator paths must be absolute and resolved")
    return path


def _identity_equal(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key)
               for key in ("pid", "start_time_ticks", "cgroup_path"))


def _same_scope(path: str, scope: str) -> bool:
    return path == scope.rstrip("/") or path.startswith(scope.rstrip("/") + "/")


def _validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise RuntimeError("invalid coordinator configuration fields")
    if (config["profile"] != PROFILE
            or config["active_arm_holder_policy"] != "paused_zero"
            or not isinstance(config["owner_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", config["owner_sha256"]) is None):
        raise RuntimeError("invalid coordinator profile, policy or owner digest")
    _absolute(config["directory"])
    _absolute(config["evidence_directory"])
    for key in ("holder_pid", "holder_start_time_ticks"):
        if type(config[key]) is not int or config[key] <= 0:
            raise RuntimeError("invalid holder process identity")
    if (config["cuda_device"] != 0 or type(config["cuda_device"]) is not int
            or not isinstance(config["gpu_uuid"], str)
            or not config["gpu_uuid"].startswith("GPU-")
            or not isinstance(config["lease_token"], str)
            or not 1 <= len(config["lease_token"]) <= 128):
        raise RuntimeError("invalid coordinator GPU or lease identity")
    for key in ("portfolio_target_bytes", "minimum_free_bytes"):
        if type(config[key]) is not int or not 0 < config[key] <= 80 * GIB:
            raise RuntimeError("invalid coordinator memory budget")
    if config["minimum_free_bytes"] < 2 * GIB:
        raise RuntimeError("at least 2 GiB of allocation headroom is required")
    peaks = config["phase_peak_bytes"]
    if (not isinstance(peaks, dict) or set(peaks) != PHASES
            or any(type(value) is not int or not config["minimum_free_bytes"] <= value <= 80 * GIB
                   for value in peaks.values())):
        raise RuntimeError("all worker phases require bounded memory budgets")
    for key in ("wait_timeout_seconds", "poll_seconds"):
        if (type(config[key]) not in (int, float) or not math.isfinite(config[key])
                or not 0 < config[key] <= 24 * 3600):
            raise RuntimeError("invalid coordinator wait timing")
    if config["poll_seconds"] > min(60, config["wait_timeout_seconds"]):
        raise RuntimeError("coordinator poll interval exceeds the wait budget")
    return {**config, "phase_peak_bytes": dict(peaks)}


def _driver_snapshot() -> dict:
    from src.agents.gpu_handoff import _driver_snapshot as sample

    return sample()


def _validate_holder(config: dict, state: dict, *, identity_reader=process_identity) -> dict:
    content = (Path(config["directory"]) / "owner.json").read_bytes()
    if _sha(content) != config["owner_sha256"]:
        raise RuntimeError("coordinator holder owner.json changed")
    owner = json.loads(content)
    actual = identity_reader(config["holder_pid"])
    if (actual["start_time_ticks"] != config["holder_start_time_ticks"]
            or owner.get("pid") != config["holder_pid"]
            or owner.get("directory") != config["directory"]
            or owner.get("cuda_device") != config["cuda_device"]
            or owner.get("persistent") is not True
            or owner.get("allocation_only") is not True
            or owner.get("automatic_restart") is not False):
        raise RuntimeError("coordinator holder process or persistent profile changed")
    if not isinstance(state, dict):
        raise RuntimeError("invalid holder status")
    for key in ("pid", "directory", "cuda_device", "started_at_SGT", "expires_at_SGT"):
        if state.get(key) != owner.get(key):
            raise RuntimeError("coordinator holder live identity changed")
    if (state.get("coordination_profile") != "persistent_claim_v1"
            or state.get("closed") is not False or state.get("persistent") is not True
            or state.get("error") or state.get("cleanup_error")):
        raise RuntimeError("persistent holder is not live and healthy")
    for key in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes",
                "maximum_bytes"):
        if type(state.get(key)) is not int or state[key] < 0:
            raise RuntimeError("invalid holder allocation telemetry")
    if (config["portfolio_target_bytes"] > state["maximum_bytes"]
            or state["maximum_bytes"] <= 0 or not (
            state["payload_bytes"] <= state["allocator_allocated_bytes"]
            <= state["allocator_reserved_bytes"] <= state["maximum_bytes"] <= 80 * GIB)):
        raise RuntimeError("inconsistent holder allocation telemetry")
    return state


def config_for_holder(directory, *, lease_token: str, evidence_directory,
                      portfolio_target_bytes: int = 76 * GIB,
                      phase_peak_bytes: dict[str, int] | None = None,
                      minimum_free_bytes: int = 2 * GIB,
                      wait_timeout_seconds: float = 1800, poll_seconds: float = 1) -> dict:
    """Bind a future controller to the identified persistent holder, without CUDA."""
    directory = Path(directory).resolve()
    content = (directory / "owner.json").read_bytes()
    owner = json.loads(content)
    actual = process_identity(owner["pid"])
    driver = _driver_snapshot()
    config = _validate_config(dict(
        profile=PROFILE, directory=str(directory), owner_sha256=_sha(content),
        holder_pid=owner["pid"], holder_start_time_ticks=actual["start_time_ticks"],
        gpu_uuid=driver["gpu_uuid"], cuda_device=owner["cuda_device"],
        lease_token=lease_token, evidence_directory=str(Path(evidence_directory).resolve()),
        portfolio_target_bytes=portfolio_target_bytes,
        phase_peak_bytes=({phase: portfolio_target_bytes for phase in PHASES}
                          if phase_peak_bytes is None else phase_peak_bytes),
        minimum_free_bytes=minimum_free_bytes, wait_timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds, active_arm_holder_policy="paused_zero",
    ))
    reservation_status(config)
    return config


def freeze_config(path, config: dict) -> str:
    """Create a fresh configuration; existing execution identities cannot be overwritten."""
    config = _validate_config(config)
    path = _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(content)


def load_config(path, expected_sha256: str) -> dict:
    if (not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        raise RuntimeError("a frozen coordinator configuration SHA256 is required")
    content = _absolute(path).read_bytes()
    if _sha(content) != expected_sha256:
        raise RuntimeError("coordinator configuration digest mismatch")
    return _validate_config(json.loads(content))


def reservation_status(config: dict) -> dict:
    config = _validate_config(config)
    state = request(config["directory"], {"op": "status"})
    _validate_holder(config, state)
    if _driver_snapshot()["gpu_uuid"] != config["gpu_uuid"]:
        raise RuntimeError("coordinator GPU UUID changed")
    return state


def admission_snapshot(config: dict) -> dict:
    """Estimate retained, reclaimable capacity; never an admission guarantee."""
    before = reservation_status(config)
    driver = _driver_snapshot()
    after = reservation_status(config)
    if driver["gpu_uuid"] != config["gpu_uuid"]:
        raise RuntimeError("coordinator GPU changed during admission sampling")
    payload = min(before["payload_bytes"], after["payload_bytes"])
    return {**driver, "holder_payload_bytes": payload,
            "available_bytes": driver["driver_free_bytes"] + payload,
            "required_peak_bytes": max(config["phase_peak_bytes"].values()),
            "holder_status": after, "atomic_transfer_guaranteed": False}


def _append(config: dict, name: str, record: dict) -> None:
    directory = _absolute(config["evidence_directory"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / name).open("a") as stream:
        stream.write(json.dumps({"at_SGT": _now(), **record}, allow_nan=False) + "\n")
        stream.flush()


def controller_claim(config: dict) -> dict:
    """Call inside the new dedicated controller service before starting any worker."""
    config = _validate_config(config)
    reservation_status(config)
    owner = process_identity(os.getpid())
    state = request(config["directory"], {
        "op": "claim", "lease_token": config["lease_token"], "owner": owner,
        "request_id": f"claim-{config['lease_token']}-{os.getpid()}",
    })
    _validate_holder(config, state)
    # An existing controller may have deliberately refilled between arms.
    # Re-claiming that same lease is idempotent, so explicitly pause it again.
    state = request(config["directory"], {
        "op": "pause", "lease_token": config["lease_token"],
        "request_id": f"controller-pause-{os.getpid()}-{uuid.uuid4().hex}",
    })
    _validate_holder(config, state)
    claim = state.get("claim")
    if (not isinstance(claim, dict) or claim.get("lease_token") != config["lease_token"]
            or not _identity_equal(claim.get("owner", {}), owner)
            or claim.get("cgroup_populated") is not True or claim.get("watchdog_error")
            or state.get("paused") is not True):
        raise RuntimeError("controller claim was not acknowledged as paused and live")
    _append(config, "controller_claims.jsonl", {
        "event": "controller_claimed", "owner": owner,
        "holder_payload_bytes": state["payload_bytes"],
        "active_arm_holder_policy": "paused_zero", "atomic_transfer_guaranteed": False,
    })
    return state


def _scope_pids(scope: str) -> set[int]:
    if (not isinstance(scope, str) or not scope.startswith("/")
            or ".." in Path(scope).parts or not Path(scope).name.endswith(".service")):
        raise RuntimeError("idle refill requires a dedicated service cgroup")
    from src.agents.gpu_reservation_service import CGROUP_ROOT

    root = CGROUP_ROOT / scope.lstrip("/")
    if not root.is_dir():
        raise RuntimeError("claimed controller cgroup disappeared")
    result: set[int] = set()
    for path in (root / "cgroup.procs", *root.rglob("*/cgroup.procs")):
        values = path.read_text().split()
        if any(not value.isdigit() or int(value) <= 0 for value in values):
            raise RuntimeError("invalid controller cgroup process telemetry")
        result.update(int(value) for value in values)
    return result


def _process_has_gpu_device(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
    """Corroborate this worker's allocator sample with its own GPU device fd.

    An fd alone does not establish residency: CPU control/query processes can
    open the same device. No other process is sampled or credited here. NVIDIA
    per-GPU devices use character major 195 and minors below 255; minor 255 is
    the control device. Unreadable worker telemetry is not treated as absence.
    """
    present = False
    for descriptor in (proc_root / str(pid) / "fd").iterdir():
        try:
            info = descriptor.stat()
        except FileNotFoundError:
            # An individual fd may close while this live worker is sampled.
            continue
        if (stat.S_ISCHR(info.st_mode) and os.major(info.st_rdev) == 195
                and os.minor(info.st_rdev) < 255):
            present = True
    return present


def controller_idle(config: dict) -> dict:
    """Refill between arms only when the controller is the sole cgroup process.

    Registered worker disappearance alone is insufficient: Ray or vLLM children
    can survive a worker. No process except the CPU controller may remain in its
    dedicated cgroup, and this CPU controller must not have initialized CUDA.
    The controller pauses this growth again before attempting the next arm.
    """
    config = _validate_config(config)
    state = reservation_status(config)
    claim = state.get("claim")
    owner = process_identity(os.getpid())
    if (not isinstance(claim, dict) or claim.get("lease_token") != config["lease_token"]
            or not _identity_equal(claim.get("owner", {}), owner)
            or claim.get("cgroup_populated") is not True or claim.get("watchdog_error")):
        raise RuntimeError("idle refill requires this controller's verified live claim")
    scope = owner["cgroup_path"]
    for worker in claim.get("workers", []):
        try:
            live = process_identity(worker["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        except RuntimeError as error:
            if str(error) == "claim process is no longer running":
                continue
            raise
        if live["start_time_ticks"] == worker["start_time_ticks"]:
            raise RuntimeError("an attached training worker is still alive")
    if _scope_pids(scope) != {owner["pid"]}:
        raise RuntimeError("training or other child processes remain in the controller cgroup")
    # Device-query fds do not imply GPU allocations. Check the existing module
    # without importing torch or creating a CUDA context in this CPU controller.
    torch = sys.modules.get("torch")
    if torch is not None:
        initialized = getattr(getattr(torch, "cuda", None), "is_initialized", None)
        if not callable(initialized) or initialized() is not False:
            raise RuntimeError("CPU controller CUDA initialization state is not verified false")
    # Check the full process set again after verifying CPU-only controller state.
    if _scope_pids(scope) != {owner["pid"]}:
        raise RuntimeError("controller cgroup changed during idle refill checks")
    state = request(config["directory"], {
        "op": "refill", "lease_token": config["lease_token"],
        "target_bytes": config["portfolio_target_bytes"],
        "request_id": f"controller-idle-{os.getpid()}-{uuid.uuid4().hex}",
    })
    _validate_holder(config, state)
    if state.get("claim", {}).get("lease_token") != config["lease_token"]:
        raise RuntimeError("holder claim changed during idle refill")
    _append(config, "controller_claims.jsonl", {
        "event": "controller_idle_refill", "owner": owner,
        "target_bytes": config["portfolio_target_bytes"],
        "active_workers_present": False, "atomic_transfer_guaranteed": False,
    })
    return state


class TorchGpuSampler:
    """Credit this worker's allocator lower bound; all other use stays unavailable.

    This fixed colocated execution profile places actor, critic and inference
    allocations in the same worker PID. Only mapped native default-pool segments
    are credited, including mapped ranges of native expandable segments. Private
    pools are excluded: vLLM's sleep allocator can retain
    their virtual segments after releasing the physical GPU memory. CUDA contexts
    and other allocations are not added. Host-namespace driver PIDs are never
    matched to container PIDs, nor can another process pay for this worker's peak.
    """

    def __init__(self, config: dict):
        self.config = config
        self.worker = process_identity(os.getpid())

    @staticmethod
    def _native_allocator_sample(torch) -> dict:
        """Credit mapped native default-pool ranges, excluding all private pools.

        PyTorch 2.7's snapshot sums only blocks whose mapped flag is true, and
        native expandable unmaps update that flag and reserved-byte statistics.
        CuMem private-pool unmaps happen outside this accounting, so those pools
        remain entirely excluded regardless of their expandable flag.
        """
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("GPU residency credit requires the native CUDA allocator")
        reserved = torch.cuda.memory_reserved(0)
        segments = torch.cuda.memory_snapshot()
        if type(reserved) is not int or reserved < 0 or not isinstance(segments, list):
            raise RuntimeError("invalid current-worker CUDA allocator telemetry")
        native_bytes = private_bytes = native_expandable_bytes = 0
        ranges: list[tuple[int, int]] = []
        for segment in segments:
            if (not isinstance(segment, dict) or type(segment.get("device")) is not int
                    or segment["device"] < 0):
                raise RuntimeError("invalid CUDA allocator snapshot device")
            if segment["device"] != 0:
                continue
            pool = segment.get("segment_pool_id")
            size, address = segment.get("total_size"), segment.get("address")
            if (not isinstance(pool, (tuple, list)) or len(pool) != 2
                    or any(type(value) is not int or value < 0 for value in pool)
                    or type(segment.get("is_expandable")) is not bool
                    or type(size) is not int or size <= 0
                    or type(address) is not int or address <= 0):
                raise RuntimeError("unrecognized CUDA allocator snapshot segment")
            ranges.append((address, address + size))
            if tuple(pool) != (0, 0):
                private_bytes += size
            else:
                native_bytes += size
                if segment["is_expandable"]:
                    native_expandable_bytes += size
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:], strict=False)):
            raise RuntimeError("overlapping CUDA allocator snapshot segments")
        return {"reserved_bytes": reserved, "native_default_pool_bytes": native_bytes,
                "private_pool_excluded_bytes": private_bytes,
                "native_expandable_mapped_bytes": native_expandable_bytes,
                "credited_bytes": min(reserved, native_bytes)}

    def __call__(self, scope: str) -> dict:
        import torch

        if (not _identity_equal(self.worker, process_identity(os.getpid()))
                or not _same_scope(self.worker["cgroup_path"], scope)):
            raise RuntimeError("GPU sampler worker PID identity or claimed cgroup changed")
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("GPU coordinator requires the identified single CUDA device")
        torch.cuda.synchronize(0)
        allocator_before = self._native_allocator_sample(torch)
        own_device_before = _process_has_gpu_device(self.worker["pid"])
        driver = _driver_snapshot()
        if driver["gpu_uuid"] != self.config["gpu_uuid"]:
            raise RuntimeError("worker/holder GPU UUID changed")
        own_device_after = _process_has_gpu_device(self.worker["pid"])
        if not _identity_equal(self.worker, process_identity(os.getpid())):
            raise RuntimeError("GPU sampler worker identity changed during driver sampling")
        # Only native default-pool physical segments pay for existing residency.
        # Take the smaller sample around the queries and exclude private vLLM
        # pools even when awake; their reserved counter can survive cuMemUnmap.
        free, total = torch.cuda.mem_get_info(0)
        allocator_after = self._native_allocator_sample(torch)
        if (type(total) is not int or type(free) is not int
                or not 0 <= free <= total):
            raise RuntimeError("invalid current-worker CUDA allocator telemetry")
        residency = min(allocator_before["credited_bytes"], allocator_after["credited_bytes"])
        if residency > total - free + MIB:
            raise RuntimeError("current-worker allocator exceeds device GPU residency")
        if residency and not (own_device_before and own_device_after):
            raise RuntimeError("current-worker CUDA allocator has no GPU device descriptor")
        return {"free_bytes": free, "total_bytes": total, "own_resident_bytes": residency,
                "own_gpu_pids": [self.worker["pid"]] if own_device_after else [],
                "gpu_uuid": driver["gpu_uuid"],
                "own_resident_source": "pytorch_native_default_pool_mapped_snapshot_lower_bound_v3",
                "allocator_before": allocator_before, "allocator_after": allocator_after,
                "unattributed_driver_residency_credited_bytes": 0}


class GpuMemoryCoordinator:
    """Worker phase guard with injectable CPU-only control, telemetry and clock seams."""

    def __init__(self, config: dict, *, identity: str, requester=request, sampler=None,
                 identity_reader=process_identity, clock=time.monotonic, sleep=time.sleep,
                 emit=None, holder_validator=None):
        self.config = _validate_config(config)
        if not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise RuntimeError("a frozen worker execution identity is required")
        self.identity, self.pid = identity, os.getpid()
        self.requester, self.identity_reader = requester, identity_reader
        self.sampler = sampler or TorchGpuSampler(self.config)
        self.clock, self.sleep = clock, sleep
        self.emit = emit or (lambda record: _append(
            self.config, f"worker-{self.pid}.jsonl", record))
        self.holder_validator = holder_validator or (
            lambda state: _validate_holder(
                self.config, state, identity_reader=self.identity_reader))
        self.worker = identity_reader(self.pid)
        self.sequence = 0
        self.active_phase: str | None = None
        self.failed = False
        state = self._command("status")
        claim = self._claim(state, require_attached=False)
        self.controller_owner = dict(claim["owner"])
        self.scope = claim["owner"]["cgroup_path"]
        if not _same_scope(self.worker["cgroup_path"], self.scope):
            raise RuntimeError("worker is outside the controller's dedicated cgroup")
        state = self._command("attach", workers=[self.worker])
        self._claim(state)
        self.emit({"event": "worker_attached", "worker": self.worker,
                   "identity": identity, "active_arm_holder_policy": "paused_zero"})

    def _command(self, op: str, **fields: Any) -> dict:
        self.sequence += 1
        command = {"op": op, "request_id": f"coordinator-{self.pid}-{uuid.uuid4().hex}"}
        if op != "status":
            command["lease_token"] = self.config["lease_token"]
        state = self.requester(self.config["directory"], {**command, **fields})
        self.holder_validator(state)
        return state

    def _claim(self, state: dict, *, require_attached: bool = True) -> dict:
        claim = state.get("claim")
        if (not isinstance(claim, dict) or claim.get("lease_token") != self.config["lease_token"]
                or claim.get("cgroup_populated") is not True or claim.get("watchdog_error")
                or not isinstance(claim.get("owner"), dict)):
            raise RuntimeError("worker has no healthy matching live holder claim")
        if require_attached and not any(_identity_equal(worker, self.worker)
                                        for worker in claim.get("workers", [])):
            raise RuntimeError("worker identity is not attached to the holder claim")
        if (hasattr(self, "controller_owner")
                and not _identity_equal(claim["owner"], self.controller_owner)):
            raise RuntimeError("holder controller identity changed during this worker")
        return claim

    def _guard(self) -> None:
        if (self.failed or self.pid != os.getpid()
                or not _identity_equal(self.worker, self.identity_reader(self.pid))):
            raise RuntimeError("GPU coordinator worker identity changed or previously failed")
        signature = getattr(self, "_environment_signature", None)
        if signature is not None:
            path, digest = signature
            if (os.environ.get("FINANCE_GPU_COORDINATOR_CONFIG") != path
                    or os.environ.get("FINANCE_GPU_COORDINATOR_CONFIG_SHA256") != digest
                    or load_config(path, digest) != self.config):
                raise RuntimeError("GPU coordinator frozen environment configuration changed")

    def _sample(self) -> dict:
        row = self.sampler(self.scope)
        if (not isinstance(row, dict) or row.get("gpu_uuid") != self.config["gpu_uuid"]
                or any(type(row.get(key)) is not int or row[key] < 0
                       for key in ("free_bytes", "total_bytes", "own_resident_bytes"))
                or row["free_bytes"] > row["total_bytes"]
                or row["own_resident_bytes"] > row["total_bytes"]):
            raise RuntimeError("invalid worker GPU capacity telemetry")
        return row

    def _required(self, phase: str, row: dict) -> int:
        return max(self.config["minimum_free_bytes"],
                   self.config["phase_peak_bytes"][phase] - row["own_resident_bytes"])

    def _wait(self, phase: str, deadline: float, *, retain_payload: bool) -> dict:
        while True:
            self._guard()
            state = self._command("status")
            self._claim(state)
            if state.get("paused") is not True:
                raise RuntimeError("holder growth resumed during a guarded worker phase")
            if not retain_payload and any(state.get(key) != 0 for key in (
                    "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")):
                raise RuntimeError("active arm holder is not paused with zero allocator residency")
            row = self._sample()
            payload = state["payload_bytes"] if retain_payload else 0
            required = self._required(phase, row)
            if row["free_bytes"] + payload >= required:
                # Refresh after the control/status round trip; a changing foreign
                # allocation must not be admitted from the preceding high read.
                confirmed = self._command("status")
                self._claim(confirmed)
                fresh = self._sample()
                if confirmed.get("paused") is not True:
                    raise RuntimeError("holder resumed while checking GPU capacity")
                confirmed_payload = (min(payload, confirmed["payload_bytes"])
                                     if retain_payload else 0)
                if (not retain_payload and any(confirmed.get(key) != 0 for key in (
                        "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"))):
                    raise RuntimeError("holder allocated during zero-residency admission")
                if fresh["free_bytes"] + confirmed_payload >= self._required(phase, fresh):
                    return {**fresh, "required_free_bytes": self._required(phase, fresh),
                            "holder_payload_bytes": confirmed_payload}
            if self.clock() >= deadline:
                raise TimeoutError(f"GPU capacity wait expired before {phase}")
            self.emit({"event": "waiting_for_capacity", "phase": phase,
                       "retained_holder_payload": retain_payload, "holder_payload_bytes": payload,
                       "required_free_bytes": required, **row})
            self.sleep(min(self.config["poll_seconds"], max(0, deadline - self.clock())))

    def before_phase(self, phase: str) -> dict:
        """Preserve held tensors while waiting, then release and recheck actual free bytes."""
        self._guard()
        if phase not in PHASES or self.active_phase is not None:
            raise RuntimeError("unknown or overlapping GPU allocation phase")
        try:
            self.active_phase = phase
            deadline = self.clock() + self.config["wait_timeout_seconds"]
            paused = self._command("pause")
            self._claim(paused)
            if paused.get("paused") is not True:
                raise RuntimeError("holder did not acknowledge paused growth")
            before = self._wait(phase, deadline, retain_payload=True)
            state = self._command("release", bytes=paused["maximum_bytes"])
            self._claim(state)
            if state.get("paused") is not True:
                raise RuntimeError("holder did not acknowledge paused release")
            admitted = self._wait(phase, deadline, retain_payload=False)
            result = {"phase": phase, "before_release": before, "after_release": admitted,
                      "atomic_transfer_guaranteed": False,
                      "active_arm_holder_policy": "paused_zero"}
            self.emit({"event": "phase_admitted", **result})
            return result
        except BaseException as error:
            self.fail(error, phase)
            raise

    def after_phase(self, phase: str) -> None:
        """Keep occupancy empty through all compute and generation in this first profile."""
        self._guard()
        if self.active_phase != phase:
            raise RuntimeError("GPU phase completion does not match its admission")
        try:
            state = self._command("status")
            self._claim(state)
            if state.get("paused") is not True or any(state.get(key) != 0 for key in (
                    "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")):
                raise RuntimeError("holder must remain paused and empty throughout the active arm")
            self.active_phase = None
            self.emit({"event": "phase_completed", "phase": phase, "holder_refill": False})
        except BaseException as error:
            self.fail(error, phase)
            raise

    def fail(self, error: BaseException, phase: str) -> None:
        """Never resume growth after uncertain or partial work; cgroup exit owns recovery."""
        self.failed = True
        self.active_phase = None
        with suppress(BaseException):
            self._command("pause")
        # Preserve the original allocation/control failure if evidence fails.
        with suppress(BaseException):
            self.emit({"event": "phase_failed", "phase": phase,
                       "error": f"{type(error).__name__}: {error}",
                       "automatic_retry": False, "holder_refill": False})


def coordinator_from_environment(identity: str) -> GpuMemoryCoordinator | None:
    path = os.environ.get("FINANCE_GPU_COORDINATOR_CONFIG")
    if not path:
        return None
    if os.environ.get("FINANCE_GPU_HANDOFF_CONFIG"):
        raise RuntimeError("one-time handoff and persistent coordination cannot both be enabled")
    if os.environ.get("FINANCE_SCREENING_RESOURCE_PROFILE") != "chunked_adam_disk_v1":
        raise RuntimeError("GPU coordination requires the chunked_adam_disk_v1 execution profile")
    digest = os.environ.get("FINANCE_GPU_COORDINATOR_CONFIG_SHA256", "")
    config = load_config(path, digest)
    pid = os.getpid()
    previous = _COORDINATORS.get(pid)
    if previous is not None:
        previous._guard()
        if previous.identity != identity or previous.config != config:
            raise RuntimeError("GPU coordinator configuration or execution identity changed")
        return previous
    coordinator = GpuMemoryCoordinator(config, identity=identity)
    coordinator._environment_signature = (path, digest)
    _COORDINATORS[pid] = coordinator
    return coordinator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "claim", "status"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--lease-token")
    parser.add_argument("--evidence-directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256")
    args = parser.parse_args(argv)
    if args.command == "configure":
        if args.directory is None or args.evidence_directory is None or not args.lease_token:
            parser.error("configure requires directory, lease-token and evidence-directory")
        config = config_for_holder(args.directory, lease_token=args.lease_token,
                                   evidence_directory=args.evidence_directory)
        digest = freeze_config(args.config.resolve(), config)
        result = {"profile": PROFILE, "config": str(args.config.resolve()), "sha256": digest}
    else:
        config = load_config(args.config.resolve(), args.config_sha256)
        result = controller_claim(config) if args.command == "claim" else admission_snapshot(config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
