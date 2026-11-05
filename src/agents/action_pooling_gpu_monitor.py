"""Cell-scoped, sampled aggregate GPU stop guard for the pooling comparison.

The claimed holder may retain its payload during the existing bounded worker
startup. Once its allocator is observed empty, the guard permanently checks
the whole device against 80 GiB. Only this call's child process group is stopped;
the dedicated systemd unit owns cleanup of descendants outside that group.
No CUDA runtime is imported and the holder receives status requests only.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from src.agents import gpu_memory_coordinator as memory
from src.agents.gpu_reservation_service import process_identity, request
from src.agents.screening_launcher import REPO_ROOT, atomic_json, utc_now

PROFILE = "aggregate_80GiB_after_verified_holder_release_v1"
LIMIT_BYTES = 80 * memory.GIB
POLL_SECONDS = 1.0
ALLOCATION_FIELDS = (
    "payload_bytes",
    "allocator_allocated_bytes",
    "allocator_reserved_bytes",
)


def _matching_paused_claim(config: dict, owner: dict, state: dict) -> bool:
    claim = state.get("claim")
    return (
        isinstance(claim, dict)
        and claim.get("lease_token") == config["lease_token"]
        and claim.get("owner") == owner
        and claim.get("cgroup_populated") is True
        and not claim.get("watchdog_error")
        and state.get("training_connected") is True
        and state.get("paused") is True
        and state.get("status") == "PAUSED"
        and state.get("closed") is False
        and not state.get("error")
        and not state.get("cleanup_error")
    )


class GuardViolation(RuntimeError):
    """A failed sample, retained in the append-only measurement stream."""

    def __init__(self, message: str, measurement: dict):
        super().__init__(message)
        self.measurement = {**measurement, "guard_error": message}


class AggregateGuard:
    """Pure state machine; callers validate the live holder's provenance first."""

    def __init__(self, config: dict, owner: dict):
        self.config, self.owner = config, dict(owner)
        self.phase = "WAITING_HOLDER_RELEASE"
        self.peak = self.active_peak = self.startup_nonholder_peak = 0
        self.samples = 0

    def observe(self, driver: dict, before: dict, after: dict, elapsed: float) -> dict:
        row = {
            "time": utc_now(),
            "elapsed_seconds": elapsed,
            "phase": self.phase,
            "driver": driver,
            "holder_before": before,
            "holder_after": after,
            "stop_threshold_bytes": LIMIT_BYTES,
            "process_attribution": False,
        }

        def fail(message):
            raise GuardViolation(message, row)

        if (
            not isinstance(driver, dict)
            or driver.get("gpu_uuid") != self.config["gpu_uuid"]
            or any(
                type(driver.get(key)) is not int
                for key in ("driver_total_bytes", "driver_free_bytes")
            )
            or not 0 <= driver["driver_free_bytes"] <= driver["driver_total_bytes"]
            or driver["driver_total_bytes"] <= 0
        ):
            fail("invalid driver telemetry or changed GPU UUID")
        used = driver["driver_total_bytes"] - driver["driver_free_bytes"]
        self.peak = max(self.peak, used)
        row.update(aggregate_used_bytes=used, sampled_peak_bytes=self.peak)
        for state in (before, after):
            if not isinstance(state, dict):
                fail("invalid holder status")
            if not _matching_paused_claim(self.config, self.owner, state):
                fail("holder claim, controller identity or paused status changed")
            if any(
                type(state.get(key)) is not int or state[key] < 0
                for key in (*ALLOCATION_FIELDS, "maximum_bytes")
            ) or not (
                state["payload_bytes"]
                <= state["allocator_allocated_bytes"]
                <= state["allocator_reserved_bytes"]
                <= state["maximum_bytes"]
                <= LIMIT_BYTES
            ):
                fail("invalid holder allocator telemetry")
        empty = all(state[key] == 0 for state in (before, after) for key in ALLOCATION_FIELDS)
        if self.phase == "ACTIVE":
            if not empty:
                fail("holder allocated again after aggregate GPU guard activation")
        else:
            # Check after all potentially blocking telemetry, including on the
            # very sample that would otherwise activate the aggregate guard.
            if elapsed >= self.config["wait_timeout_seconds"]:
                fail("holder release wait expired before aggregate GPU guard activation")
            if empty:
                self.phase = "ACTIVE"
            else:
                payload = min(state["payload_bytes"] for state in (before, after))
                if payload <= 0 or payload > used:
                    fail("startup occupancy has no consistent positive verified holder payload")
                nonholder = used - payload
                self.startup_nonholder_peak = max(self.startup_nonholder_peak, nonholder)
                row.update(
                    startup_releasable_payload_bytes=payload, startup_nonholder_used_bytes=nonholder
                )
                if nonholder > LIMIT_BYTES:
                    fail("startup GPU occupancy excluding verified holder payload exceeded 80 GiB")
        row["phase"] = self.phase
        if self.phase == "ACTIVE":
            self.active_peak = max(self.active_peak, used)
            row.update(
                active_sampled_peak_bytes=self.active_peak, startup_releasable_payload_bytes=0
            )
            if used > LIMIT_BYTES:
                fail("aggregate GPU occupancy exceeded 80 GiB stop threshold")
        self.samples += 1
        return row


def _holder_status(config: dict, owner: dict) -> dict:
    if process_identity(os.getpid()) != owner:
        raise RuntimeError("GPU monitor controller process identity changed")
    state = request(config["directory"], {"op": "status"})
    memory._validate_holder(config, state, identity_reader=process_identity)
    if not _matching_paused_claim(config, owner, state):
        raise RuntimeError("holder claim, controller identity or paused status changed")
    return state


def _sample(config: dict, owner: dict) -> tuple[dict, dict, dict]:
    before = _holder_status(config, owner)
    driver = memory._driver_snapshot()
    after = _holder_status(config, owner)
    # A release between these reads must not arm from a pre-release driver
    # sample. Refresh the driver bracket when the last state is already empty.
    if all(after.get(key) == 0 for key in ALLOCATION_FIELDS) and any(
        before.get(key) != 0 for key in ALLOCATION_FIELDS
    ):
        before = after
        driver = memory._driver_snapshot()
        after = _holder_status(config, owner)
    return driver, before, after


def _stop_owned_group(process, child: dict | None, owner: dict, holder_pid: int) -> dict:
    result = {"scope": "launched_trainer_process_group_only", "signals": []}
    if process is None or process.poll() is not None:
        return result
    actual = process_identity(process.pid)
    if (
        child is None
        or actual != child
        or process.pid in {owner["pid"], holder_pid}
        or actual.get("process_group_id") != process.pid
        or actual.get("session_id") != process.pid
        or not memory._same_scope(actual["cgroup_path"], owner["cgroup_path"])
    ):
        raise RuntimeError("refusing to signal an unverified trainer process group")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        result["signals"].append(sig.name)
        try:
            process.wait(timeout=5)
            return result
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("launched trainer process group did not exit after SIGKILL")


def _exclusive_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture_process(command: list[str], *, environment: dict[str, str], cell_dir: Path) -> int:
    """Keep the native capture API, enforcing the new profile's sampled stop line."""
    cell_dir = Path(cell_dir)
    receipt_path = cell_dir / "gpu_monitor_receipt.json"
    if receipt_path.exists() or (cell_dir / "process.json").exists():
        raise RuntimeError("GPU-monitored cell execution must be fresh")
    config_path = environment.get("FINANCE_GPU_COORDINATOR_CONFIG", "")
    config_sha = environment.get("FINANCE_GPU_COORDINATOR_CONFIG_SHA256", "")
    config = memory.load_config(config_path, config_sha)
    owner = process_identity(os.getpid())
    if not Path(owner["cgroup_path"]).name.endswith(".service"):
        raise RuntimeError("GPU monitor requires the dedicated controller service cgroup")
    guard = AggregateGuard(config, owner)
    started = time.monotonic()
    process = child = None
    failure = None
    cleanup = {}
    returncode = None
    with (cell_dir / "gpu_monitor.jsonl").open("x", encoding="utf-8") as evidence:
        try:
            with (
                (cell_dir / "trainer.log").open("a", encoding="utf-8", buffering=1) as output,
                (cell_dir / "resources.jsonl").open("a", encoding="utf-8") as resources,
            ):
                while True:
                    # Re-read the frozen config as well as the live claim on
                    # every iteration; a changed file is never silently used.
                    if memory.load_config(config_path, config_sha) != config:
                        raise RuntimeError("GPU monitor coordinator configuration changed")
                    driver, before, after = _sample(config, owner)
                    try:
                        row = guard.observe(driver, before, after, time.monotonic() - started)
                    except GuardViolation as error:
                        evidence.write(json.dumps(error.measurement, allow_nan=False) + "\n")
                        evidence.flush()
                        raise
                    row["pid"] = None if process is None else process.pid
                    evidence.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    evidence.flush()
                    resource = {"time": row["time"], "pid": row["pid"], "gpu_monitor": row}
                    for filename in ("memory.current", "memory.peak", "memory.events"):
                        path = Path("/sys/fs/cgroup") / filename
                        if path.exists():
                            resource[filename] = path.read_text().strip()
                    resources.write(json.dumps(resource, sort_keys=True, allow_nan=False) + "\n")
                    resources.flush()
                    if process is None:
                        process = subprocess.Popen(
                            command,
                            cwd=REPO_ROOT,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        atomic_json(
                            cell_dir / "process.json",
                            {
                                "pid": process.pid,
                                "pgid": process.pid,
                                "started_at": utc_now(),
                            },
                        )
                        if process.poll() is None:
                            child = process_identity(process.pid)
                            if (
                                child["process_group_id"] != process.pid
                                or child["session_id"] != process.pid
                                or process.pid in {owner["pid"], config["holder_pid"]}
                                or not memory._same_scope(
                                    child["cgroup_path"], owner["cgroup_path"]
                                )
                            ):
                                raise RuntimeError(
                                    "trainer was not launched in its own group "
                                    "and controller cgroup"
                                )
                    returncode = process.poll()
                    if returncode is not None:
                        if returncode == 0 and guard.phase != "ACTIVE":
                            raise RuntimeError(
                                "trainer exited successfully without activating aggregate GPU guard"
                            )
                        return int(returncode)
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(
                            timeout=min(
                                POLL_SECONDS,
                                max(
                                    0, config["wait_timeout_seconds"] - (time.monotonic() - started)
                                ),
                            )
                            if guard.phase != "ACTIVE"
                            else POLL_SECONDS
                        )
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            evidence.write(
                json.dumps(
                    {"time": utc_now(), "phase": guard.phase, "failure": failure}, allow_nan=False
                )
                + "\n"
            )
            evidence.flush()
            raise
        finally:
            try:
                cleanup = _stop_owned_group(process, child, owner, config["holder_pid"])
            except BaseException as error:
                cleanup = {
                    "scope": "launched_trainer_process_group_only",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
                if failure is None:
                    failure = cleanup["error"]
                # Propagate uncertain cleanup: systemd owns the remaining
                # dedicated cgroup. No foreign/holder process is signalled.
                raise
            finally:
                _exclusive_json(
                    receipt_path,
                    {
                        "profile": PROFILE,
                        "status": "COMPLETE" if failure is None and returncode == 0 else "FAILED",
                        "finished_at": utc_now(),
                        "returncode": returncode,
                        "failure": failure,
                        "config_path": config_path,
                        "config_sha256": config_sha,
                        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "controller_identity": owner,
                        "trainer_identity": child,
                        "phase": guard.phase,
                        "monitoring_activated": guard.phase == "ACTIVE",
                        "stop_threshold_bytes": LIMIT_BYTES,
                        "holder_release_wait_seconds": config["wait_timeout_seconds"],
                        "sampled_peak_bytes": guard.peak,
                        "active_sampled_peak_bytes": guard.active_peak,
                        "startup_nonholder_sampled_peak_bytes": guard.startup_nonholder_peak,
                        "accepted_samples": guard.samples,
                        "poll_target_seconds": POLL_SECONDS,
                        "instantaneous_memory_cap": False,
                        "external_bytes_subtracted_when_active": False,
                        "holder_operations": ["status"],
                        "cleanup": cleanup,
                    },
                )
