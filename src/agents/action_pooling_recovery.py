"""Explicit execution of remaining cells after recorded engineering failures.

The failed invocations remain immutable. Completed cells are imported as
historical evidence, with their original execution identities and summaries.
One zero-output native initialization failure is bound separately. No optimizer,
trajectory, or completed policy evaluation is resumed or rerolled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from src.agents import action_pooling_comparison as comparison
from src.agents import action_pooling_gpu_monitor as gpu_monitor
from src.agents import action_pooling_readout as readout
from src.agents import actor_export
from src.agents import gpu_memory_coordinator as memory
from src.agents import parameter_grid_archive as archive
from src.agents.gpu_reservation_service import process_identity, request

runner, restart = comparison.runner, comparison.restart
PROFILE = "action_pooling_explicit_remaining_v1"
ORIGINAL = runner.REPO_ROOT / "runs/m2-action-pooling-comparison-20261012-v2"
DEFAULT = runner.REPO_ROOT / "runs/m2-action-pooling-comparison-20261012-v4"
TIMEOUT_ORIGINAL = DEFAULT
TIMEOUT_DEFAULT = runner.REPO_ROOT / "runs/m2-action-pooling-comparison-20261013-v5"
TIMEOUT_KIND = "cpu_budget_timeout_after_four_cells_v1"
LATEST_ORIGINAL = TIMEOUT_DEFAULT
LATEST_DEFAULT = runner.REPO_ROOT / "runs/m2-action-pooling-comparison-20261013-v6"
LATEST_KIND = "cpu_budget_timeout_after_seven_cells_v1"
PREFIX = ((42, "sft0"), (42, "m1_lr1e7"))
ERROR = "RuntimeError: partial readout did not validate every completed cell"
TIMEOUT_PREFIX = PREFIX + ((42, "m1_lr2e7"), (42, "m2_plain_lr1e7_eta025"))
REMAINING = tuple(
    (seed, spec.cell_id) for seed, profile in comparison.PROFILES.items()
    for spec in runner.cells(profile) if (seed, spec.cell_id) not in PREFIX
)
TIMEOUT_REMAINING = tuple(row for row in REMAINING if row not in TIMEOUT_PREFIX)
LATEST_PREFIX = TIMEOUT_PREFIX + ((42, "m2_lr1e7_eta025"), (43, "sft0"),
                                  (43, "m2_lr1e7_eta025"))
LATEST_REMAINING = tuple(row for row in REMAINING if row not in LATEST_PREFIX)
TIMEOUT_ERROR = "TimeoutError: native GPU budget did not become available within 1800 seconds"
TIMEOUT_SNAPSHOT_PROFILE = "action_pooling_timeout_failed_run_readonly_snapshot_v1"
LATEST_SNAPSHOT_PROFILE = "action_pooling_latest_timeout_readonly_snapshot_v1"
RUNTIME_PREFLIGHT = dict(profile="isolated_original_runtime_preflight_v1", timeout_seconds=300,
                         verifier="src.agents.launcher.verify_gpu_runtime_and_model",
                         parent_cuda_initialized=False, automatic_retry=False)
ALLOWED_CHANGES = {
    "src/agents/action_pooling_readout.py", "src/agents/action_pooling_comparison.py",
    "src/agents/action_pooling_service.py",
}
SELF = "src/agents/action_pooling_recovery.py"
DATA_FILES = ("train.parquet", "dev.parquet", "reward_calibration.json")
GPU_BUDGET = dict(profile="cpu_native_budget_wait_v1", native_phase_bytes=76 * memory.GIB,
                  aggregate_limit_bytes=80 * memory.GIB, maximum_seconds=1800,
                  poll_seconds=5, required_ready_samples=2,
                  holder_operations=["status"], automatic_retry=False)
FAILED_LAUNCH_PROFILE = "action_pooling_failed_zero_output_launch_v1"


def _verify_failed_launch(path, original):
    """Bind the failed initialization without treating a proof's zero counts as evidence."""
    path = _safe(path)
    proof = restart.read(path)
    _require(proof.get("profile") == FAILED_LAUNCH_PROFILE and proof.get("status") == "PASS"
             and proof.get("actual_exit1") is True and proof.get("zero_new_trajectories") is True
             and proof.get("zero_actor_critic_updates") is True
             and proof.get("new_trajectories") == proof.get("actor_updates")
             == proof.get("critic_updates") == 0
             and proof.get("failure_kind") == "cuda_context_before_holder_release_oom"
             and proof.get("guard_phase") == "WAITING_HOLDER_RELEASE"
             and proof.get("old_failed_launch_preserved") is True
             and proof.get("automatic_retry") is False, "failed-launch proof scope differs")
    failed = _safe(proof["original_v3"])
    _require(failed != original, "failed initialization must be distinct from completed prefix")
    _require(_inventory(failed) == proof["file_inventory"], "failed-launch history changed")
    _require(restart.digest(failed / "run_contract.json") == proof["run_contract_sha256"]
             and restart.read(failed / "source_identity.json") == proof["source_sha256"],
             "failed-launch source/contract binding differs")
    root = restart.read(failed / "run_contract.json")
    for directory in (failed, *(failed / f"seed{s}" for s in comparison.SEEDS)):
        contract = restart.read(directory / "run_contract.json")
        for name, digest in contract["inputs"].items():
            _require(restart.digest(_safe(directory / name)) == digest,
                     "failed-launch sealed input changed")
        for name, digest in restart.read(directory / "source_identity.json").items():
            _require(restart.digest(_safe(directory / "sources" / name)) == digest,
                     "failed-launch frozen source changed")
        if directory != failed:
            _require(restart.digest(directory / "run_contract.json")
                     == root["child_contracts"][directory.name[4:]],
                     "failed-launch child contract changed")
    previous = restart.read(failed / "recovery_contract.json")
    _require(previous.get("original") == str(original)
             and root.get("continuation") == dict(
                 profile=PROFILE,
                 recovery_contract_sha256=restart.digest(failed / "recovery_contract.json"),
             ), "failed launch did not continue the same completed prefix")
    expected_error = "RuntimeError: m1_lr2e7 trainer exited 1; evidence retained"
    state, child = (restart.read(failed / name) for name in ("state.json", "seed42/state.json"))
    cell = failed / "seed42/m1_lr2e7"
    summary = restart.read(cell / "summary.json")
    exit_receipt = restart.read(failed / "service_exit_receipt.json")
    guard = restart.read(cell / "gpu_monitor_receipt.json")
    _require(state.get("status") == child.get("status") == summary.get("status") == "FAILED"
             and state.get("error") == child.get("error") == summary.get("error") == expected_error
             and summary.get("returncode") == guard.get("returncode") == 1
             and exit_receipt.get("numeric_exit_status_recorded") is True
             and type(exit_receipt.get("controller_returncode")) is int
             and exit_receipt["controller_returncode"] == 1
             and exit_receipt.get("successful_main_exit") is False
             and exit_receipt.get("main_process_absent_when_hook_ran") is True
             and exit_receipt.get("run_contract_sha256") == proof["run_contract_sha256"]
             and exit_receipt.get("systemd_environment", {}).get("EXIT_STATUS") == "1"
             and exit_receipt.get("systemd_environment", {}).get("EXIT_CODE") == "exited"
             and guard.get("status") == "FAILED"
             and guard.get("phase") == "WAITING_HOLDER_RELEASE"
             and guard.get("monitoring_activated") is False
             and guard.get("active_sampled_peak_bytes") == 0,
             "failed launch was not an actual exit-1 pre-release initialization failure")
    log = (cell / "trainer.log").read_text()
    _require("gpu_memory_coordinator.py" in log and "torch.cuda.synchronize(0)" in log
             and "CUDA error: out of memory" in log, "failed launch OOM trace differs")
    _require(not (failed / "seed43/state.json").exists(), "failed launch started future seed")
    for seed, cell_id in REMAINING:
        candidate = failed / f"seed{seed}" / cell_id
        if (seed, cell_id) != REMAINING[0]:
            _require(not candidate.exists(), "failed launch attempted additional cells")
    _require([r.get("cell", {}).get("cell_id") for r in child.get("cells", [])]
             == [cell_id for _, cell_id in PREFIX], "failed launch returned a new completed cell")
    for seed, cell_id in PREFIX:
        relative = f"seed{seed}/{cell_id}"
        _require(_inventory(failed / relative) == _inventory(original / relative),
                 "failed launch altered imported prefix")
    _require(not any(p.is_file() for p in (cell / "checkpoints").rglob("*")),
             "failed initialization contains exported weights or checkpoints")
    for evidence in (cell / "artifacts").rglob("*"):
        if not evidence.is_file():
            continue
        _require(evidence.relative_to(cell / "artifacts").parts[0] == "placement"
                 and evidence.suffix == ".jsonl", "failed launch contains new science evidence")
        for line in evidence.read_text().splitlines():
            row = json.loads(line)
            _require(row.get("event") == "installed" or (
                row.get("event") == "exchange" and row.get("phase") == "init_model:error"
                and row.get("detail", {}).get("event") == "failed"
            ), "failed launch contains model/update work beyond failed initialization")
    return dict(path=str(path), sha256=restart.digest(path))


def _budget_remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("native GPU budget did not become available within 1800 seconds")
    return remaining


def _budget_holder_status(config, owner, deadline):
    _require(process_identity(os.getpid()) == owner, "budget controller identity changed")
    state = request(config["directory"], {"op": "status"},
                    timeout=min(15, _budget_remaining(deadline)))
    memory._validate_holder(config, state, identity_reader=process_identity)
    _require(type(state.get("paused")) is bool and state.get("status") in {
        "HOLDING", "GROWING", "WAITING_FOR_MEMORY", "BACKOFF", "PAUSED",
    } and state["paused"] == (state["status"] == "PAUSED"),
             "budget holder idle/paused telemetry is inconsistent")
    if state.get("claim") is None:
        _require(state.get("training_connected") is False,
                 "unclaimed holder telemetry is inconsistent")
    else:
        # controller_idle may resume this same claim's healthy idle holder.
        # The native coordinator pauses it again; this CPU reader never does.
        claim = state["claim"]
        _require(isinstance(claim, dict) and claim.get("lease_token") == config["lease_token"]
                 and claim.get("owner") == owner and claim.get("cgroup_populated") is True
                 and not claim.get("watchdog_error") and state.get("training_connected") is True,
                 "budget holder claim belongs to another controller or is unhealthy")
    return state


def _budget_driver_snapshot(deadline):
    # Keep the subprocess timeout inside the same absolute CPU-only deadline.
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
        timeout=min(10, _budget_remaining(deadline)),
    )
    rows = [row.strip().split(",") for row in result.stdout.splitlines() if row.strip()]
    _require(len(rows) == 1 and len(rows[0]) == 3, "expected exactly one driver GPU row")
    uuid, free, total = (part.strip() for part in rows[0])
    return dict(gpu_uuid=uuid, driver_free_bytes=int(free) * 1024**2,
                driver_total_bytes=int(total) * 1024**2)


def wait_native_budget(child, cell_id):
    """Wait before native CUDA initialization, without claiming or releasing memory."""
    config_path = child / "gpu_coordinator_config.json"
    config_sha = restart.read(child / "run_contract.json")["inputs"][config_path.name]
    config = memory.load_config(config_path, config_sha)
    _require(set(config["phase_peak_bytes"].values()) == {GPU_BUDGET["native_phase_bytes"]}
             and config["wait_timeout_seconds"] == GPU_BUDGET["maximum_seconds"],
             "CPU budget wait requires the frozen 76 GiB native allowance and 1800-second limit")
    owner = process_identity(os.getpid())
    _require(Path(owner["cgroup_path"]).name.endswith(".service"),
             "native budget wait requires the dedicated controller service")
    journal = child / f"native_budget-{cell_id}.jsonl"
    receipt = child / f"native_budget-{cell_id}.json"
    _require(not receipt.exists(), "native budget wait already attempted; no retry")
    started = time.monotonic()
    deadline = started + GPU_BUDGET["maximum_seconds"]
    status, failure, samples, ready_streak = "FAILED", None, 0, 0
    with journal.open("x", encoding="utf-8") as stream:
        try:
            runner.atomic_json(receipt, dict(**GPU_BUDGET, status="WAITING", cell_id=cell_id,
                                             native_started_by_wait=False, samples=0))
            while True:
                _budget_remaining(deadline)
                _require(memory.load_config(config_path, config_sha) == config,
                         "native budget coordinator configuration changed")
                before = _budget_holder_status(config, owner, deadline)
                driver = _budget_driver_snapshot(deadline)
                after = _budget_holder_status(config, owner, deadline)
                _require(driver.get("gpu_uuid") == config["gpu_uuid"]
                         and all(type(driver.get(k)) is int for k in (
                             "driver_free_bytes", "driver_total_bytes",
                         )) and 0 <= driver["driver_free_bytes"] <= driver["driver_total_bytes"]
                         and driver["driver_total_bytes"] > 0,
                         "native budget driver UUID or memory telemetry differs")
                used = driver["driver_total_bytes"] - driver["driver_free_bytes"]
                coherent = all(before.get(k) == after.get(k) for k in (
                    *gpu_monitor.ALLOCATION_FIELDS, "claim", "paused", "status", "maximum_bytes",
                ))
                credit = before["payload_bytes"] if coherent else 0
                _require(credit <= used, "verified holder payload exceeds aggregate driver use")
                nonholder = used - credit
                ready = coherent and nonholder + GPU_BUDGET["native_phase_bytes"] <= (
                    GPU_BUDGET["aggregate_limit_bytes"]
                )
                ready_streak = ready_streak + 1 if ready else 0
                admitted = ready_streak >= GPU_BUDGET["required_ready_samples"]
                elapsed = time.monotonic() - started
                row = dict(time=restart.now(), cell_id=cell_id, elapsed_seconds=elapsed,
                           driver=driver, holder_before=before, holder_after=after,
                           coherent=coherent, holder_payload_credit_bytes=credit,
                           observed_nonholder_bytes=nonholder,
                           native_phase_bytes=GPU_BUDGET["native_phase_bytes"],
                           aggregate_limit_bytes=GPU_BUDGET["aggregate_limit_bytes"],
                           ready_sample=ready, ready_streak=ready_streak,
                           ready=admitted and elapsed < GPU_BUDGET["maximum_seconds"])
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                samples += 1
                runner.atomic_json(receipt, dict(
                    **GPU_BUDGET, status="READY" if row["ready"] else "WAITING",
                    cell_id=cell_id, samples=samples, latest_sample=row,
                    native_started_by_wait=False, config_sha256=config_sha,
                ))
                runner.atomic_json(child / "gpu_queue_state.json", dict(
                    status="NATIVE_BUDGET_READY" if row["ready"] else "WAITING_FOR_GPU",
                    cell=cell_id, at_SGT=restart.now(), waited_seconds=elapsed,
                    free_MiB=driver["driver_free_bytes"] // 1024**2,
                    native_phase_bytes=GPU_BUDGET["native_phase_bytes"],
                    aggregate_limit_bytes=GPU_BUDGET["aggregate_limit_bytes"],
                    observed_nonholder_bytes=nonholder, native_started=False,
                    holder_operations=["status"], automatic_retry=False,
                ))
                _budget_remaining(deadline)
                if admitted:
                    status = "READY"
                    return
                time.sleep(min(GPU_BUDGET["poll_seconds"], _budget_remaining(deadline)))
        except BaseException as error:
            status = "TIMEOUT" if isinstance(error, TimeoutError) else "FAILED"
            failure = dict(type=type(error).__name__, message=str(error))
            stream.write(json.dumps(dict(
                time=restart.now(), status=status, failure=failure,
            )) + "\n")
            stream.flush()
            raise
        finally:
            runner.atomic_json(receipt, dict(
                **GPU_BUDGET, status=status, failure=failure, cell_id=cell_id, samples=samples,
                elapsed_seconds=time.monotonic() - started, config_sha256=config_sha,
                controller_identity=owner, native_started_by_wait=False,
                atomic_reservation_guaranteed=False, active_aggregate_guard_unchanged=True,
            ))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _safe(path):
    path = Path(path).absolute()
    _require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
             "recovery evidence must not use symlinks")
    return path


def _inventory(directory):
    directory = _safe(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "recovery evidence must not use symlinks")
        if not path.is_file():
            _require(path.is_dir(), "recovery evidence must contain regular files")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
        result[str(path.relative_to(directory))] = dict(
            bytes=path.stat().st_size, sha256=digest.hexdigest(),
        )
    return result


def _known_failure(original):
    state = restart.read(original / "state.json")
    child = restart.read(original / "seed42/state.json")
    exit_receipt = restart.read(original / "service_exit_receipt.json")
    _require(
        state.get("status") == child.get("status") == "FAILED"
        and state.get("error") == child.get("error") == ERROR
        and state.get("profile") == comparison.NAME and state.get("current_seed") == 42
        and state.get("completed_seeds") == [] and child.get("seed") == 42
        and child.get("current_cell") == "m1_lr1e7"
        and [r.get("cell", {}).get("cell_id") for r in child.get("cells", [])]
        == [cell for _, cell in PREFIX]
        and exit_receipt.get("numeric_exit_status_recorded") is True
        and type(exit_receipt.get("controller_returncode")) is int
        and exit_receipt["controller_returncode"] == 1
        and exit_receipt.get("successful_main_exit") is False
        and exit_receipt.get("main_process_absent_when_hook_ran") is True
        and exit_receipt.get("run_contract_sha256")
        == restart.digest(original / "run_contract.json")
        and exit_receipt.get("systemd_environment", {}).get("EXIT_CODE") == "exited"
        and exit_receipt.get("systemd_environment", {}).get("EXIT_STATUS") == "1",
        "only the known post-cell readout failure with actual numeric exit 1 is eligible",
    )
    _require(not (original / "seed43/state.json").exists(), "future seed already started")
    for seed, cell in REMAINING:
        _require(not (original / f"seed{seed}" / cell).exists(),
                 "a remaining cell was already attempted")
    for row, (_, cell) in zip(child["cells"], PREFIX, strict=True):
        _require(row == restart.read(original / "seed42" / cell / "summary.json")
                 and row.get("status") == "COMPLETE" and row.get("returncode") == 0,
                 "imported cell is not an actually completed unchanged prefix")
    return child["cells"]


def _frozen_contracts(original):
    root = restart.read(original / "run_contract.json")
    _require(root.get("profile") == comparison.NAME
             and root.get("planned_trajectories") == 1336
             and root.get("automatic_retry") is False and root.get("resume_supported") is False
             and "continuation" not in root
             and set(root.get("inputs", {})) == set(comparison.ROOT_INPUTS)
             and set(root.get("child_contracts", {})) == {"42", "43"},
             "original root contract differs")
    _require(comparison._same(restart.read(original / "comparison_settings.json"),
                              comparison.settings()), "original science settings differ")
    current = runner.source_files()
    changes = {}
    for base in (original, *(original / f"seed{s}" for s in comparison.SEEDS)):
        contract = restart.read(base / "run_contract.json")
        for name, digest in contract["inputs"].items():
            _require(restart.digest(_safe(base / name)) == digest, "original sealed input changed")
        sources = restart.read(base / "source_identity.json")
        _require(set(current) - set(sources) <= {SELF} and not set(sources) - set(current),
                 "unreviewed recovery source inventory changes")
        for name, digest in sources.items():
            _require(restart.digest(_safe(base / "sources" / name)) == digest,
                     "original frozen source copy changed")
            actual = restart.digest(current[name])
            _require(actual == digest or name in ALLOWED_CHANGES,
                     "science source changed outside the readout recovery allowlist: " + name)
            if actual != digest:
                changes[name] = dict(before_sha256=digest, after_sha256=actual)
        if base != original:
            seed = int(base.name[4:])
            _require(restart.digest(base / "run_contract.json")
                     == root["child_contracts"][str(seed)],
                     "original child contract changed")
            manifest = restart.read(base / "prepared/manifest.json")
            for row in (*manifest["datasets"].values(), manifest["calibration"]):
                _require(restart.digest(base / "prepared" / row["path"]) == row["sha256"],
                         "original prepared bytes changed")
            for name, digest in restart.read(base / "selection.json")["source_sha256"].items():
                _require(restart.digest(Path(name)) == digest, "original selection source changed")
            runner.read_settings(base, profile=comparison.PROFILES[seed])
    return changes


def _archive_evidence(cell, summary):
    """Verify retired raw weights via exact archive replay and retained receipts."""
    identity = summary["execution_identity_sha256"]
    receipt = restart.read(cell / "actor_archive_receipt.json")
    pruning = restart.read(cell / "moment_pruning_receipt.json")
    for row in (receipt, pruning):
        _require(row.get("status") == "COMPLETE"
                 and row.get("execution_identity_sha256") == identity,
                 "completed prefix retirement identity differs")
    _require(receipt.get("manifest_sha256") == restart.digest(cell / "actor_archive/manifest.json"),
             "prefix archive manifest changed")
    for row in pruning["files"]:
        _require(not (cell / row["path"]).exists(), "retired moment raw was recreated")
    kept = pruning["kept_manifests"]
    _require(len(kept) == 2 and {row["role"] for row in kept} == {"actor", "critic"},
             "prefix requires two retained owned moment manifests")
    for row in kept:
        path = _safe(cell / row["path"])
        manifest, owner = restart.read(path), restart.read(path.parent / "owner.json")
        _require(restart.digest(path) == row["sha256"]
                 and manifest.get("identity") == owner.get("identity") == identity
                 and manifest.get("owner") == owner.get("owner") == row["owner"]
                 and manifest.get("role") == owner.get("role") == row["role"]
                 and manifest.get("generation") == row["generation"],
                 "retained moment manifest or ownership changed")
    archive._worker_stopped(cell, identity)
    observed = archive.validate_archive(cell / "actor_archive")
    _require(observed.get("manifest_sha256") == receipt["manifest_sha256"],
             "prefix archive replay differs")
    checkpoint = cell / "checkpoints/global_step_4"
    actor = restart.read(checkpoint / "actor" / actor_export.ACTOR_RECEIPT)
    marker = restart.read(checkpoint / actor_export.EXPORT_COMPLETION_MARKER)
    manifest = restart.read(cell / "actor_archive/manifest.json")
    _require(actor.get("hf") == manifest["original_hf"]
             and actor.get("identity_sha256") == marker.get("identity_sha256") == identity
             and actor.get("global_step") == marker.get("global_step") == 4
             and actor.get("artifact_profile") == marker.get("artifact_profile")
             == actor_export.ACTOR_EXPORT_PROFILE
             and actor.get("optimizer_included") is False
             and actor.get("resume_supported") is False
             and marker.get("actor_receipt_sha256")
             == restart.digest(checkpoint / "actor" / actor_export.ACTOR_RECEIPT),
             "archive original HF does not bind the actual export receipt")
    return observed


def audit_original(original=ORIGINAL):
    """Read-only revalidation of exactly the known immutable two-cell prefix."""
    original = _safe(original)
    before = _inventory(original)
    summaries = _known_failure(original)
    changes = _frozen_contracts(original)
    sources, counts, heads = {}, [], set()
    for summary, (_, cell_id) in zip(summaries, PREFIX, strict=True):
        cell = original / "seed42" / cell_id
        spec = next(s for s in comparison.PROFILES[42].cell_specs if s.cell_id == cell_id)
        result = readout.shared._cell(
            cell, 42, cell_id, summary, sources, retain_prefix=True,
            cells=readout.CELLS, record_validator=readout._identity,
        )
        _require(not result[-1], "completed prefix has incomplete evaluation evidence")
        counts.append(len(result[-2]))
        identity = summary["execution_identity_sha256"]
        extra = runner.validate_extra_evidence(
            cell, spec, identity, profile=comparison.PROFILES[42],
        )
        extra["critic_fit"] = runner.tuning.validate_fit_evidence(
            cell, spec=spec, execution_identity=identity,
        )
        extra["placement"] = runner.validate_placement(cell, spec, identity)
        extra["gpu_handoff"] = runner.validate_handoff(cell, identity)
        _require(all(summary.get(k) == v for k, v in extra.items()),
                 "prefix runtime evidence disagrees with persisted summary")
        head = restart.read(cell / "artifacts/critic_initialization/head-rank0.json")
        _require(head.get("seed") == 42 and head.get("execution_identity_sha256") == identity
                 and head.get("arm") == spec.arm and head.get("mode") == "cpu_seed_v1"
                 and head.get("head_sha256") == summary.get("critic_initial_head_sha256")
                 and head.get("model_path") == str(comparison.MODEL),
                 "prefix actual initial critic head differs")
        heads.add(head["head_sha256"])
        if not spec.zero:
            _archive_evidence(cell, summary)
    _require(counts == [60, 152] and len(heads) == 1, "completed prefix budget/head differs")
    _require(_inventory(original) == before, "original changed during read-only audit")
    return dict(profile=PROFILE, status="COMPLETE", original=str(original),
                original_files=before, source_changes=changes,
                imported_cells=[list(row) for row in PREFIX], imported_trajectories=212,
                remaining_cells=[list(row) for row in REMAINING], new_trajectories=1124,
                complete_comparison_budget=1336, retrained=False, reevaluated=False)


def _timeout_failure(original):
    """The M2 request waited on CPU; the preceding four cells actually completed."""
    root, child = (restart.read(original / name)
                   for name in ("state.json", "seed42/state.json"))
    receipt = restart.read(original / "service_exit_receipt.json")
    _require(root.get("status") == child.get("status") == "FAILED"
             and root.get("error") == child.get("error") == TIMEOUT_ERROR
             and root.get("profile") == comparison.NAME and root.get("current_seed") == 42
             and root.get("completed_seeds") == [] and child.get("seed") == 42
             and child.get("current_cell") == TIMEOUT_REMAINING[0][1]
             and [r.get("cell", {}).get("cell_id") for r in child.get("cells", [])]
             == [cell for _, cell in TIMEOUT_PREFIX]
             and receipt.get("numeric_exit_status_recorded") is True
             and type(receipt.get("controller_returncode")) is int
             and receipt["controller_returncode"] == 1
             and receipt.get("successful_main_exit") is False
             and receipt.get("main_process_absent_when_hook_ran") is True
             and receipt.get("run_contract_sha256")
             == restart.digest(original / "run_contract.json")
             and receipt.get("systemd_environment", {}).get("EXIT_CODE") == "exited"
             and receipt.get("systemd_environment", {}).get("EXIT_STATUS") == "1",
             "only the recorded four-cell CPU budget timeout with actual exit 1 is eligible")
    _require(not (original / "seed43/state.json").exists(), "timeout history started future seed")
    for seed, cell in TIMEOUT_REMAINING:
        _require(not (original / f"seed{seed}" / cell).exists(),
                 "timeout pending cell entered native execution")
    summaries = child["cells"]
    for row, (seed, cell) in zip(summaries, TIMEOUT_PREFIX, strict=True):
        _require(row == restart.read(original / f"seed{seed}" / cell / "summary.json")
                 and row.get("status") == "COMPLETE" and type(row.get("returncode")) is int
                 and row["returncode"] == 0, "timeout prefix has an incomplete or changed cell")
    _budget_timeout_evidence(original, 42, TIMEOUT_REMAINING[0][1])
    # Its v2 prefix and the distinct v3 zero-output initialization remain independently bound.
    old_contract = restart.read(original / "run_contract.json")
    old_recovery = restart.read(original / "recovery_contract.json")
    _require("continuation_kind" not in old_recovery,
             "timeout continuation must follow the fixed legacy v4 scope")
    verify_bindings(original, old_contract)
    return summaries


def _budget_timeout_evidence(original, seed, pending):
    budget = restart.read(original / f"seed{seed}/native_budget-{pending}.json")
    journal = original / f"seed{seed}/native_budget-{pending}.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    config_sha = restart.read(original / f"seed{seed}/run_contract.json")["inputs"].get(
        "gpu_coordinator_config.json"
    )
    _require(all(budget.get(k) == v for k, v in GPU_BUDGET.items())
             and budget.get("status") == "TIMEOUT"
             and budget.get("cell_id") == pending
             and budget.get("native_started_by_wait") is False
             and budget.get("elapsed_seconds", 0) >= GPU_BUDGET["maximum_seconds"]
             and type(budget.get("samples")) is int and budget["samples"] > 0
             and budget.get("config_sha256") == config_sha and config_sha is not None
             and rows and rows[-1].get("status") == "TIMEOUT"
             and rows[-1].get("failure", {}).get("type") == "TimeoutError"
             and len(rows) == budget["samples"] + 1
             and all(row.get("ready") is False for row in rows[:-1]),
             "timeout history does not prove a completed CPU-only budget wait")


def _timeout_snapshot(path, original, *, kind=TIMEOUT_KIND):
    path = _safe(path)
    snapshot = restart.read(path)
    pre = snapshot.get("pre_edit_contract_check", {})
    latest = kind == LATEST_KIND
    prefix, remaining = (LATEST_PREFIX, LATEST_REMAINING) if latest else (
        TIMEOUT_PREFIX, TIMEOUT_REMAINING,
    )
    _require(kind in {TIMEOUT_KIND, LATEST_KIND}, "unknown timeout snapshot scope")
    _require(snapshot.get("profile") == (
        LATEST_SNAPSHOT_PROFILE if latest else TIMEOUT_SNAPSHOT_PROFILE)
             and snapshot.get("status") == "PASS" and snapshot.get("original") == str(original)
             and snapshot.get("original_status") == "FAILED"
             and type(snapshot.get("controller_returncode")) is int
             and snapshot["controller_returncode"] == 1
             and snapshot.get("completed_cells") == (
                 [list(row) for row in prefix] if latest else [cell for _, cell in prefix])
             and snapshot.get("completed_trajectories") == (880 if latest else 516)
             and snapshot.get("remaining_trajectories") == (456 if latest else 820)
             and (not latest or (snapshot.get("completed_seeds") == [42]
                  and snapshot.get("remaining_cells") == [list(row) for row in remaining]))
             and snapshot.get("unstarted_pending_cell") == remaining[0][1]
             and snapshot.get("pending_cell_trajectory_count") == 0
             and snapshot.get("old_failed_history_preserved") is True
             and snapshot.get("automatic_retry") is False
             and type(pre.get("returncode")) is int and pre["returncode"] == 0
             and pre.get("command", [])[-4:] == [
                 "src.agents.action_pooling_comparison", "check", "--directory", str(original),
             ] and snapshot.get("run_contract_sha256")
             == restart.digest(original / "run_contract.json")
             and snapshot.get("source_sha256") == restart.read(original / "source_identity.json")
             and snapshot.get("files") == _inventory(original),
             "original timeout snapshot or pre-edit contract check changed")
    log = _safe(pre["log"])
    return dict(path=str(path), sha256=restart.digest(path),
                pre_edit_log=dict(path=str(log), sha256=restart.digest(log)))


def _latest_failure(original):
    root = restart.read(original / "state.json")
    receipt = restart.read(original / "service_exit_receipt.json")
    _require(root.get("status") == "FAILED" and root.get("error") == TIMEOUT_ERROR
             and root.get("profile") == comparison.NAME and root.get("current_seed") == 43
             and root.get("completed_seeds") == [42]
             and receipt.get("numeric_exit_status_recorded") is True
             and type(receipt.get("controller_returncode")) is int
             and receipt["controller_returncode"] == 1
             and receipt.get("successful_main_exit") is False
             and receipt.get("main_process_absent_when_hook_ran") is True
             and receipt.get("run_contract_sha256")
             == restart.digest(original / "run_contract.json")
             and receipt.get("systemd_environment", {}).get("EXIT_CODE") == "exited"
             and receipt.get("systemd_environment", {}).get("EXIT_STATUS") == "1",
             "only the recorded seven-cell queue timeout with actual exit 1 is eligible")
    summaries = []
    for seed in comparison.SEEDS:
        child = restart.read(original / f"seed{seed}/state.json")
        cells = [(s, c) for s, c in LATEST_PREFIX if s == seed]
        _require(child.get("status") == ("COMPLETE" if seed == 42 else "FAILED")
                 and child.get("seed") == seed
                 and [r.get("cell", {}).get("cell_id") for r in child.get("cells", [])]
                 == [cell for _, cell in cells]
                 and (seed == 42 or (child.get("error") == TIMEOUT_ERROR
                      and child.get("current_cell") == LATEST_REMAINING[0][1])),
                 "latest timeout completed seed or prefix differs")
        for row, (_, cell) in zip(child["cells"], cells, strict=True):
            _require(row == restart.read(original / f"seed{seed}" / cell / "summary.json")
                     and row.get("status") == "COMPLETE" and type(row.get("returncode")) is int
                     and row["returncode"] == 0, "latest completed prefix changed")
            summaries.append(row)
    for seed, cell in LATEST_REMAINING:
        _require(not (original / f"seed{seed}" / cell).exists(),
                 "latest pending cell entered native execution")
    _budget_timeout_evidence(original, 43, LATEST_REMAINING[0][1])
    # Validate historical v5 using its frozen runtime, not the current SELF hash.
    old = restart.read(original / "recovery_contract.json")
    _require(_timeout_population(old)[0] == TIMEOUT_PREFIX,
             "latest continuation must follow the fixed four-cell continuation")
    contract = restart.read(original / "run_contract.json")
    _require(contract.get("continuation") == dict(
        profile=PROFILE,
        recovery_contract_sha256=restart.digest(original / "recovery_contract.json"),
    ), "latest historical recovery contract binding changed")
    ancestor = _safe(old["original"])
    _timeout_failure(ancestor)
    _require(_inventory(ancestor) == old["original_files"]
             and _timeout_snapshot(old["original_snapshot"]["path"], ancestor)
             == old["original_snapshot"], "latest timeout ancestor or prior audit changed")
    _timeout_sources(ancestor)
    _timeout_sources(original)
    before, after = (restart.read(base / "source_identity.json") for base in (ancestor, original))
    _require(set(before) == set(after), "historical continuation changed source inventory")
    changes = {name: dict(before_sha256=digest, after_sha256=after[name])
               for name, digest in before.items() if digest != after[name]}
    _require(set(changes) <= {SELF} and changes == old["source_changes"],
             "historical continuation changed science or repair provenance")
    _copied_timeout_inputs(original, ancestor, TIMEOUT_PREFIX, old["imported_files"])
    return summaries


def _timeout_sources(original):
    """Only this recovery engineering module may differ from the frozen v4 runtime."""
    current, changes = runner.source_files(), {}
    root = restart.read(original / "run_contract.json")
    _require(root.get("profile") == comparison.NAME and root.get("planned_trajectories") == 1336
             and set(root.get("inputs", {})) == set(comparison.ROOT_INPUTS)
             and set(root.get("child_contracts", {})) == {"42", "43"}
             and root.get("automatic_retry") is False and root.get("resume_supported") is False
             and comparison._same(restart.read(original / "comparison_settings.json"),
                                  comparison.settings()), "timeout history science scope differs")
    for directory in (original, *(original / f"seed{s}" for s in comparison.SEEDS)):
        contract = restart.read(directory / "run_contract.json")
        for name, digest in contract["inputs"].items():
            _require(restart.digest(_safe(directory / name)) == digest,
                     "timeout history sealed input changed")
        sources = restart.read(directory / "source_identity.json")
        _require(set(sources) == set(current), "timeout recovery changed source inventory")
        for name, digest in sources.items():
            _require(restart.digest(_safe(directory / "sources" / name)) == digest,
                     "timeout frozen source copy changed")
            actual = restart.digest(current[name])
            _require(actual == digest or name == SELF,
                     "timeout recovery changed science or GPU guard: " + name)
            if actual != digest:
                changes[name] = dict(before_sha256=digest, after_sha256=actual)
        if directory != original:
            seed = int(directory.name[4:])
            _require(restart.digest(directory / "run_contract.json")
                     == root["child_contracts"][str(seed)], "timeout child contract changed")
            runner.read_settings(directory, profile=comparison.PROFILES[seed])
    return changes


def _canonical_cell(original, seed, cell_id):
    """Follow only sealed import inventories, retaining the original absolute provenance."""
    seen, relative = set(), f"seed{seed}/{cell_id}"
    for _ in range(8):
        _require(original not in seen, "cyclic historical evidence origin")
        seen.add(original)
        cell = original / relative
        path = original / "recovery_contract.json"
        if not path.exists():
            return cell
        contract = restart.read(path)
        expected = contract.get("imported_files", {}).get(relative)
        if expected is None:
            return cell
        ancestor = _safe(contract["original"])
        _require(_inventory(cell) == expected == _inventory(ancestor / relative),
                 "timeout imported prefix differs from its canonical evidence origin")
        original = ancestor
    raise RuntimeError("historical evidence origin chain exceeded the fixed bound")


def audit_timeout(original=TIMEOUT_ORIGINAL, *, snapshot, kind=TIMEOUT_KIND):
    """Read-only full scientific and archive validation of the four completed cells."""
    original = _safe(original)
    before = _inventory(original)
    latest = kind == LATEST_KIND
    prefix, remaining = (LATEST_PREFIX, LATEST_REMAINING) if latest else (
        TIMEOUT_PREFIX, TIMEOUT_REMAINING,
    )
    snapshot_binding = _timeout_snapshot(snapshot, original, kind=kind)
    summaries = _latest_failure(original) if latest else _timeout_failure(original)
    changes = _timeout_sources(original)
    sources, counts, heads = {}, [], {}
    replayed = []
    for summary, (seed, cell_id) in zip(summaries, prefix, strict=True):
        cell, profile = original / f"seed{seed}" / cell_id, comparison.PROFILES[seed]
        spec = next(s for s in runner.cells(profile) if s.cell_id == cell_id)
        result = readout.shared._cell(
            cell, seed, cell_id, summary, sources, retain_prefix=True,
            cells=readout.CELLS, record_validator=readout._identity,
        )
        _require(not result[-1], "timeout completed prefix has incomplete evaluation evidence")
        counts.append(len(result[-2]))
        cell = _canonical_cell(original, seed, cell_id)
        identity = summary["execution_identity_sha256"]
        extra = runner.validate_extra_evidence(cell, spec, identity, profile=profile)
        extra["critic_fit"] = runner.tuning.validate_fit_evidence(
            cell, spec=spec, execution_identity=identity,
        )
        extra["placement"] = runner.validate_placement(cell, spec, identity)
        extra["gpu_handoff"] = runner.validate_handoff(cell, identity)
        _require(all(summary.get(k) == v for k, v in extra.items()),
                 "timeout prefix update/fit/placement evidence differs from summary")
        head = restart.read(cell / "artifacts/critic_initialization/head-rank0.json")
        _require(head.get("seed") == seed and head.get("execution_identity_sha256") == identity
                 and head.get("arm") == spec.arm and head.get("mode") == "cpu_seed_v1"
                 and head.get("head_sha256") == summary.get("critic_initial_head_sha256")
                 and head.get("model_path") == str(comparison.MODEL),
                 "timeout prefix actual initial critic head differs")
        heads.setdefault(seed, set()).add(head["head_sha256"])
        if not spec.zero and (not latest or (seed, cell_id) not in TIMEOUT_PREFIX):
            _archive_evidence(cell, summary)
            replayed.append([seed, cell_id])
    _require(counts == [60 if cell == "sft0" else 152 for _, cell in prefix]
             and set(heads) == {seed for seed, _ in prefix}
             and all(len(values) == 1 for values in heads.values()),
             "timeout prefix trajectory budget or shared critic head differs")
    _require(_inventory(original) == before, "timeout history changed during read-only audit")
    return dict(
        profile=PROFILE, status="COMPLETE", continuation_kind=kind, original=str(original),
        original_files=before, source_changes=changes, original_snapshot=snapshot_binding,
        imported_cells=[list(row) for row in prefix], imported_trajectories=880 if latest else 516,
        remaining_cells=[list(row) for row in remaining], new_trajectories=456 if latest else 820,
        completed_new_trajectories_in_original=364 if latest else 304,
        complete_comparison_budget=1336,
        scientific_prefix_audit="raw_update_fit_placement_head_and_lossless_archive_v1",
        runtime_preflight=RUNTIME_PREFLIGHT,
        **(dict(archive_replay_cells=replayed, inherited_archive_audit=dict(
            path=str(original / "recovery_contract.json"),
            sha256=restart.digest(original / "recovery_contract.json"),
        )) if latest else {}),
        retrained=False, reevaluated=False,
    )


def prepare(base, *, original=ORIGINAL, coordinated_reservation_directory, failed_launch_audit):
    base, original = _safe(base), _safe(original)
    _require(not base.exists() and not base.is_relative_to(original),
             "fresh recovery root required")
    audit = audit_original(original)
    failed_launch = _verify_failed_launch(failed_launch_audit, original)
    comparison.prepare(base, coordinated_reservation_directory=coordinated_reservation_directory)
    for seed in comparison.SEEDS:
        child, source = base / f"seed{seed}", original / f"seed{seed}"
        for name in DATA_FILES:
            _require(restart.digest(child / "prepared" / name)
                     == restart.digest(source / "prepared" / name),
                     "recovery prepared data differs")
        for name in ("model_identity.json", "settings.json", "selection.json"):
            _require(restart.read(child / name) == restart.read(source / name),
                     "recovery model, data or scientific settings differ")
        shutil.copy2(source / "input_preflight.json", child / "input_preflight.json")
    imported = {}
    for seed, cell in PREFIX:
        relative = f"seed{seed}/{cell}"
        shutil.copytree(original / relative, base / relative)
        imported[relative] = _inventory(base / relative)
        _require(imported[relative] == _inventory(original / relative), "import copy differs")
    contract = {**audit, "status": "SEALED", "imported_files": imported,
                "automatic_retry": False, "optimizer_resume": False,
                "original_failure_preserved": True, "failed_launch": failed_launch,
                "gpu_budget": GPU_BUDGET}
    runner.atomic_json(base / "recovery_contract.json", contract)
    comparison.seal(base, continuation=dict(
        profile=PROFILE, recovery_contract_sha256=restart.digest(base / "recovery_contract.json"),
    ))
    return contract


def prepare_timeout(base, *, original=TIMEOUT_ORIGINAL,
                    coordinated_reservation_directory, snapshot, kind=TIMEOUT_KIND):
    base, original = _safe(base), _safe(original)
    _require(not base.exists() and not base.is_relative_to(original),
             "fresh timeout continuation root required")
    audit = audit_timeout(original, snapshot=snapshot, kind=kind)
    prefix = LATEST_PREFIX if kind == LATEST_KIND else TIMEOUT_PREFIX
    comparison.prepare(base, coordinated_reservation_directory=coordinated_reservation_directory)
    for seed in comparison.SEEDS:
        child, source = base / f"seed{seed}", original / f"seed{seed}"
        for name in DATA_FILES:
            _require(restart.digest(child / "prepared" / name)
                     == restart.digest(source / "prepared" / name),
                     "timeout recovery prepared data differs")
        for name in ("model_identity.json", "settings.json", "selection.json"):
            _require(restart.read(child / name) == restart.read(source / name),
                     "timeout recovery scientific inputs differ")
        shutil.copy2(source / "input_preflight.json", child / "input_preflight.json")
    imported = {}
    for seed, cell in prefix:
        relative = f"seed{seed}/{cell}"
        shutil.copytree(original / relative, base / relative)
        imported[relative] = _inventory(base / relative)
        _require(imported[relative] == _inventory(original / relative),
                 "timeout import copy differs")
    contract = {**audit, "status": "SEALED", "imported_files": imported,
                "automatic_retry": False, "optimizer_resume": False,
                "original_failure_preserved": True, "gpu_budget": GPU_BUDGET}
    runner.atomic_json(base / "recovery_contract.json", contract)
    comparison.seal(base, continuation=dict(
        profile=PROFILE, recovery_contract_sha256=restart.digest(base / "recovery_contract.json"),
    ))
    return contract


def _timeout_population(recovery):
    latest = recovery.get("continuation_kind") == LATEST_KIND
    prefix, remaining = (LATEST_PREFIX, LATEST_REMAINING) if latest else (
        TIMEOUT_PREFIX, TIMEOUT_REMAINING,
    )
    _require(recovery.get("profile") == PROFILE and recovery.get("status") == "SEALED"
             and recovery.get("continuation_kind") in {TIMEOUT_KIND, LATEST_KIND}
             and recovery.get("imported_cells") == [list(row) for row in prefix]
             and recovery.get("remaining_cells") == [list(row) for row in remaining]
             and recovery.get("imported_trajectories") == (880 if latest else 516)
             and recovery.get("new_trajectories") == (456 if latest else 820)
             and recovery.get("completed_new_trajectories_in_original") == (364 if latest else 304)
             and recovery.get("complete_comparison_budget") == 1336
             and recovery.get("scientific_prefix_audit")
             == "raw_update_fit_placement_head_and_lossless_archive_v1"
             and recovery.get("automatic_retry") is False
             and recovery.get("optimizer_resume") is False
             and recovery.get("retrained") is False and recovery.get("reevaluated") is False
             and recovery.get("original_failure_preserved") is True
             and recovery.get("gpu_budget") == GPU_BUDGET,
             "timeout continuation population, science audit or budget differs")
    _require(recovery.get("runtime_preflight") == RUNTIME_PREFLIGHT,
             "timeout isolated runtime preflight scope differs")
    return prefix, remaining


def _copied_timeout_inputs(base, original, prefix, imported_files):
    _require(set(imported_files) == {f"seed{s}/{c}" for s, c in prefix},
             "timeout import population changed")
    for relative, expected in imported_files.items():
        _require(_inventory(base / relative) == expected == _inventory(original / relative),
                 "timeout imported evidence changed")
    for seed in comparison.SEEDS:
        for name in DATA_FILES:
            _require(restart.digest(base / f"seed{seed}/prepared" / name)
                     == restart.digest(original / f"seed{seed}/prepared" / name),
                     "timeout prepared data changed")
        for name in ("model_identity.json", "settings.json", "selection.json",
                     "input_preflight.json"):
            _require(restart.digest(base / f"seed{seed}" / name)
                     == restart.digest(original / f"seed{seed}" / name),
                     "timeout preflight or scientific input changed")


def verify_timeout(base, recovery):
    """Recheck pinned evidence; full archive replay was performed before first seal."""
    prefix, _ = _timeout_population(recovery)
    latest = recovery["continuation_kind"] == LATEST_KIND
    original = _safe(recovery["original"])
    _require(original != base and not base.is_relative_to(original),
             "timeout continuation root aliases history")
    snapshot = recovery["original_snapshot"]
    _require(_timeout_snapshot(snapshot["path"], original,
                               kind=recovery["continuation_kind"]) == snapshot,
             "timeout snapshot or pre-edit check log changed")
    if latest:
        _latest_failure(original)
        _require(recovery.get("archive_replay_cells") == [
            [42, "m2_lr1e7_eta025"], [43, "m2_lr1e7_eta025"],
        ] and recovery.get("inherited_archive_audit") == dict(
            path=str(original / "recovery_contract.json"),
            sha256=restart.digest(original / "recovery_contract.json"),
        ), "latest archive replay or inherited audit binding changed")
    else:
        _timeout_failure(original)
    _require(_inventory(original) == recovery["original_files"]
             and _timeout_sources(original) == recovery["source_changes"],
             "timeout history or recovery engineering repair scope changed")
    _copied_timeout_inputs(base, original, prefix, recovery["imported_files"])


def verify_bindings(base, contract):
    """Verify sealed historical imports without calling comparison.check_contract."""
    base = _safe(base)
    continuation = contract.get("continuation")
    _require(isinstance(continuation, dict) and set(continuation)
             == {"profile", "recovery_contract_sha256"} and continuation["profile"] == PROFILE
             and continuation["recovery_contract_sha256"]
             == restart.digest(base / "recovery_contract.json"),
             "recovery contract binding differs")
    recovery = restart.read(base / "recovery_contract.json")
    if recovery.get("continuation_kind") in {TIMEOUT_KIND, LATEST_KIND}:
        return verify_timeout(base, recovery)
    _require("continuation_kind" not in recovery, "unknown explicit continuation kind")
    _require(recovery.get("profile") == PROFILE and recovery.get("status") == "SEALED"
             and recovery.get("imported_cells") == [list(row) for row in PREFIX]
             and recovery.get("remaining_cells") == [list(row) for row in REMAINING]
             and recovery.get("imported_trajectories") == 212
             and recovery.get("new_trajectories") == 1124
             and recovery.get("complete_comparison_budget") == 1336
             and recovery.get("automatic_retry") is False
             and recovery.get("optimizer_resume") is False
             and recovery.get("retrained") is False and recovery.get("reevaluated") is False
             and recovery.get("original_failure_preserved") is True,
             "recovery population or no-resume scope differs")
    original = _safe(recovery["original"])
    _require(original != base and not base.is_relative_to(original),
             "recovery root aliases history")
    _known_failure(original)
    _require(recovery.get("gpu_budget") == GPU_BUDGET,
             "native prelaunch budget wait scope changed")
    failed_launch = recovery.get("failed_launch", {})
    _require(_verify_failed_launch(failed_launch.get("path", ""), original) == failed_launch,
             "failed initialization audit binding changed")
    _require(_inventory(original) == recovery["original_files"],
             "original failure/evidence changed")
    _require(_frozen_contracts(original) == recovery["source_changes"],
             "recovery source repair scope changed")
    _require(set(recovery["imported_files"]) == {f"seed{s}/{c}" for s, c in PREFIX},
             "recovery import inventory differs")
    for relative, expected in recovery["imported_files"].items():
        _require(_inventory(base / relative) == expected == _inventory(original / relative),
                 "imported original evidence changed")
    for seed in comparison.SEEDS:
        for name in DATA_FILES:
            _require(restart.digest(base / f"seed{seed}/prepared" / name)
                     == restart.digest(original / f"seed{seed}/prepared" / name),
                     "recovery prepared bytes changed")
        for name in ("model_identity.json", "settings.json", "selection.json",
                     "input_preflight.json"):
            _require(restart.digest(base / f"seed{seed}" / name)
                     == restart.digest(original / f"seed{seed}" / name),
                     "recovery preflight or scientific input changed")


def _parent_cuda_initialized():
    import torch

    return torch.cuda.is_initialized()


def _isolated_runtime_preflight(child, cell_id, profile):
    """Run the unchanged CUDA/model verifier in a short-lived child, never this controller."""
    _require(_parent_cuda_initialized() is False,
             "controller CUDA was initialized before preflight")
    runtime_path = child / "runtime_preflight.json"
    evidence = child / "runtime_preflight_probe"
    receipt_path = evidence / "receipt.json"
    if runtime_path.exists():
        receipt = restart.read(receipt_path)
        _require(all(receipt.get(k) == v for k, v in RUNTIME_PREFLIGHT.items())
                 and receipt.get("status") == "COMPLETE"
                 and type(receipt.get("returncode")) is int and receipt["returncode"] == 0
                 and receipt.get("parent_cuda_before") is receipt.get("parent_cuda_after") is False
                 and receipt.get("runtime_sha256") == restart.digest(runtime_path)
                 and receipt.get("config_sha256") == restart.digest(evidence / "config.json")
                 and all(receipt.get(f"{name}_sha256") == restart.digest(evidence / f"{name}.txt")
                         for name in ("stdout", "stderr")),
                 "isolated runtime preflight result changed")
        return
    _require(not evidence.exists(), "isolated runtime preflight already attempted; no retry")
    config = runner.make_config(child, restart.read(child / "prepared/manifest.json"),
                                cell_id, profile=profile)
    command = [sys.executable, "-u", "-c", (
        "import json,sys; from src.agents.launcher import verify_gpu_runtime_and_model; "
        "print(json.dumps(verify_gpu_runtime_and_model(json.load(sys.stdin)),allow_nan=False))"
    )]
    evidence.mkdir()
    runner.atomic_json(evidence / "config.json", config)
    receipt = dict(**RUNTIME_PREFLIGHT, status="RUNNING", command=command, cell_id=cell_id,
                   config_sha256=restart.digest(evidence / "config.json"),
                   started_at_SGT=restart.now(), parent_cuda_before=False, returncode=None)
    runner.atomic_json(receipt_path, receipt)
    try:
        result = subprocess.run(
            command, input=json.dumps(config, allow_nan=False), cwd=runner.REPO_ROOT,
            capture_output=True, text=True, timeout=RUNTIME_PREFLIGHT["timeout_seconds"],
            check=False,
        )
        (evidence / "stdout.txt").write_text(result.stdout)
        (evidence / "stderr.txt").write_text(result.stderr)
        receipt["returncode"] = result.returncode
        receipt["parent_cuda_after"] = _parent_cuda_initialized()
        _require(receipt["parent_cuda_after"] is False,
                 "controller CUDA was initialized during isolated preflight")
        _require(type(result.returncode) is int and result.returncode == 0,
                 "isolated runtime preflight child failed")
        runtime = json.loads(result.stdout)
        recorded = restart.read(child / "model_identity.json")
        _require(all(runtime.get(k) == recorded.get(k) for k in (
            "model_revision", "model_snapshot_path",
        )) and runtime.get("cuda_available") is True,
                 "isolated runtime preflight does not belong to the shared SFT")
        runner.atomic_json(runtime_path, runtime)
        receipt.update(status="COMPLETE", runtime_sha256=restart.digest(runtime_path))
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            for name, value in (("stdout", error.stdout), ("stderr", error.stderr)):
                (evidence / f"{name}.txt").write_text(
                    value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                )
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if "parent_cuda_after" not in receipt:
            receipt["parent_cuda_after"] = _parent_cuda_initialized()
        for name in ("stdout", "stderr"):
            if (evidence / f"{name}.txt").exists():
                receipt[f"{name}_sha256"] = restart.digest(evidence / f"{name}.txt")
        receipt.update(finished_at_SGT=restart.now())
        runner.atomic_json(receipt_path, receipt)


def _run_remaining_seed(base, seed, *, prefix=PREFIX):
    """The normal seed lifecycle, skipping only its explicitly imported cells."""
    profile, child = comparison.PROFILES[seed], base / f"seed{seed}"
    imported = [cell for imported_seed, cell in prefix if imported_seed == seed]
    imported_count = sum(60 if cell == "sft0" else 152 for cell in imported)
    _require(not (child / "state.json").exists(), "one-shot seed continuation already attempted")
    runner.check_contract(child, profile=profile)
    state = dict(status="RUNNING", started_at_SGT=restart.now(), current_cell=None,
                 cells=[restart.read(child / cell / "summary.json") for cell in imported],
                 profile=profile.name, seed=seed, planned_trajectories=668,
                 automatic_retry=False, resume_supported=False,
                 train_episode_ids=profile.train_ids, evaluation_episode_ids=profile.eval_ids,
                 evaluation_steps=list(profile.evaluation_steps),
                 imported_trajectories=imported_count, new_trajectories=668 - imported_count)
    runner.atomic_json(child / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(
        target=runner.monitor_status, args=(child, stop, profile), daemon=True,
    )
    monitor.start()
    try:
        remaining = runner.cells(profile)[len(imported):]
        for index, spec in enumerate(remaining):
            exports = int(any(not s.zero for s in remaining[index:]))
            required = runner.disk_requirement_bytes(profile, exports)
            free = shutil.disk_usage(child).free
            _require(free >= required,
                     "remaining four-round cell requires 107 GiB actual free disk")
            state.update(current_cell=spec.cell_id, free_disk_bytes=free)
            runner.atomic_json(child / "state.json", state)
            wait_native_budget(child, spec.cell_id)
            if prefix in (TIMEOUT_PREFIX, LATEST_PREFIX):
                _isolated_runtime_preflight(child, spec.cell_id, profile)
            _require(shutil.disk_usage(child).free >= required,
                     "disk capacity changed during native GPU budget wait")
            summary = runner.run_cell(child, spec.cell_id, profile=profile)
            state["cells"].append(summary)
            runner.atomic_json(child / "state.json", state)
            config = runner.memory_coordinator_config(child)
            if config is not None:
                from src.agents.gpu_memory_coordinator import controller_idle

                try:
                    idle = controller_idle(config)
                    idle_record = dict(status="REFILLING", at_SGT=restart.now(),
                                       payload_bytes=idle["payload_bytes"])
                except (OSError, RuntimeError) as error:
                    idle_record = dict(status="DEFERRED", at_SGT=restart.now(),
                                       reason=f"{type(error).__name__}: {error}")
                runner.atomic_json(child / "gpu_coordinator_idle.json", idle_record)
            if not spec.zero:
                summary["moment_pruning"] = runner.retire_completed_moments(
                    child, spec, summary, profile=profile,
                )
                runner.atomic_json(child / spec.cell_id / "summary.json", summary)
                runner.atomic_json(child / "state.json", state)
            comparison.after_cell(child, spec, summary)
            runner.atomic_json(child / spec.cell_id / "summary.json", summary)
            runner.atomic_json(child / "state.json", state)
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        runner.atomic_json(child / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        runner.stop_reservation(child)
        runner.refresh_status(child, profile=profile)


def run(base):
    base = _safe(base)
    _require(not (base / "state.json").exists(), "one-shot recovery already attempted; no resume")
    comparison.check_contract(base)
    _require("continuation" in restart.read(base / "run_contract.json"),
             "explicit sealed continuation required")
    recovery = restart.read(base / "recovery_contract.json")
    if recovery.get("continuation_kind") == LATEST_KIND:
        prefix, remaining = LATEST_PREFIX, LATEST_REMAINING
    elif recovery.get("continuation_kind") == TIMEOUT_KIND:
        prefix, remaining = TIMEOUT_PREFIX, TIMEOUT_REMAINING
    else:
        prefix, remaining = PREFIX, REMAINING
    imported_count = sum(60 if cell == "sft0" else 152 for _, cell in prefix)
    for seed in comparison.SEEDS:
        child = base / f"seed{seed}"
        _require(not (child / "state.json").exists(), "seed controller already attempted")
        _require(runner.memory_coordinator_config(child) is not None, "sealed holder required")
    for seed, cell in remaining:
        _require(not (base / f"seed{seed}" / cell).exists(), "remaining cell already attempted")
    _require(shutil.disk_usage(base).free >= comparison.settings()["minimum_free_disk_bytes"],
             "recovery launch requires 107 GiB actual free disk")
    state = dict(profile=comparison.NAME, status="RUNNING", started_at_SGT=restart.now(),
                 current_seed=None, completed_seeds=[], planned_trajectories=1336,
                 imported_trajectories=imported_count, new_trajectories=1336 - imported_count,
                 automatic_retry=False, resume_supported=False, continuation_profile=PROFILE)
    runner.atomic_json(base / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(target=comparison._monitor, args=(base, stop), daemon=True)
    monitor.start()
    try:
        for seed in comparison.SEEDS:
            comparison.check_contract(base)
            state["current_seed"] = seed
            runner.atomic_json(base / "state.json", state)
            _run_remaining_seed(base, seed, prefix=prefix)
            _require(restart.read(base / f"seed{seed}/state.json").get("status") == "COMPLETE",
                     "remaining seed failed its fixed population")
            state["completed_seeds"].append(seed)
            runner.atomic_json(base / "state.json", state)
        result = comparison._readout(base, allow_partial=False)
        _require(result.get("execution_status") in {
            "COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES",
        }, "recovery fixed comparison did not validate")
        state.update(status=result["execution_status"], financial_status=result["financial_status"],
                     current_seed=None)
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        runner.atomic_json(base / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        comparison.refresh_status(base)
        runner.atomic_json(base / "service_completion.json", dict(
            **state, original_failure_preserved=True, method_advantage_claimed=False,
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare", "audit-timeout", "prepare-timeout",
                                            "audit-latest", "prepare-latest", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--coordinated-reservation-directory", type=Path)
    parser.add_argument("--failed-launch-audit", type=Path)
    parser.add_argument("--original-snapshot", type=Path)
    args = parser.parse_args()
    base = args.directory.absolute()
    if args.command == "audit":
        result = audit_original(args.original.absolute())
        print(json.dumps({k: v for k, v in result.items() if k != "original_files"}))
    elif args.command == "prepare":
        _require(args.failed_launch_audit is not None, "--failed-launch-audit is required")
        holder = args.coordinated_reservation_directory or runner.current_reservation_directory()
        prepare(base, original=args.original.absolute(), coordinated_reservation_directory=holder,
                failed_launch_audit=args.failed_launch_audit)
    elif args.command in {"audit-timeout", "prepare-timeout", "audit-latest", "prepare-latest"}:
        _require(args.original_snapshot is not None, "--original-snapshot is required")
        latest = args.command.endswith("latest")
        kind = LATEST_KIND if latest else TIMEOUT_KIND
        original = (LATEST_ORIGINAL if latest else TIMEOUT_ORIGINAL) if (
            args.original == ORIGINAL
        ) else args.original
        original = original.absolute()
        if args.command.startswith("audit"):
            result = audit_timeout(original, snapshot=args.original_snapshot, kind=kind)
            print(json.dumps({k: v for k, v in result.items() if k != "original_files"}))
        else:
            target = (LATEST_DEFAULT if latest else TIMEOUT_DEFAULT) if base == DEFAULT else base
            holder = (args.coordinated_reservation_directory
                      or runner.current_reservation_directory())
            prepare_timeout(target, original=original, coordinated_reservation_directory=holder,
                            snapshot=args.original_snapshot, kind=kind)
    elif args.command == "check":
        comparison.check_contract(base)
    else:
        run(base)


if __name__ == "__main__":
    main()
