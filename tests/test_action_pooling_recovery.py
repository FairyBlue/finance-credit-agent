"""Immutable prefix imports and exact one-shot execution of the eight remaining cells."""

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import action_pooling_recovery as recovery
from src.agents import gpu_memory_coordinator

comparison, runner = recovery.comparison, recovery.runner
PRODUCTION_WAIT = recovery.wait_native_budget
PRODUCTION_DRIVER_SNAPSHOT = recovery._budget_driver_snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.atomic_json(path, value)


@pytest.fixture
def original(tmp_path, monkeypatch):
    base = tmp_path / "original"
    current = {}
    science = {
        "src/science.py": b"unchanged science",
        "src/agents/action_pooling_readout.py": b"old reader",
    }
    for relative, payload in {**science, recovery.SELF: b"new recovery"}.items():
        target = tmp_path / "live" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixed reader" if "readout" in relative else payload)
        current[relative] = target
    monkeypatch.setattr(runner, "source_files", lambda: current)
    monkeypatch.setattr(runner, "read_settings", lambda *args, **kwargs: {})
    summaries = []
    for index, (_, cell) in enumerate(recovery.PREFIX):
        summary = dict(status="COMPLETE", returncode=0, cell={"cell_id": cell},
                       execution_identity_sha256=str(index) * 64)
        summaries.append(summary)
        write(base / "seed42" / cell / "summary.json", summary)
        (base / "seed42" / cell / "raw.jsonl").write_bytes(b'{"preserved":true}\n')
    write(base / "state.json", dict(
        status="FAILED", error=recovery.ERROR, profile=comparison.NAME,
        current_seed=42, completed_seeds=[],
    ))
    write(base / "seed42/state.json", dict(
        status="FAILED", error=recovery.ERROR, seed=42, current_cell="m1_lr1e7", cells=summaries,
    ))
    sources = {}
    for directory in (base, base / "seed42", base / "seed43"):
        for relative, payload in science.items():
            path = directory / "sources" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            sources[relative] = runner.restart.digest(path)
        write(directory / "source_identity.json", sources)
    for seed in comparison.SEEDS:
        child = base / f"seed{seed}"
        rows = {}
        for name in recovery.DATA_FILES:
            path = child / "prepared" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("fixed " + name).encode())
            rows[name] = dict(path=name, sha256=runner.restart.digest(path))
        write(child / "prepared/manifest.json", dict(
            datasets={"train": rows["train.parquet"], "dev": rows["dev.parquet"]},
            calibration=rows["reward_calibration.json"],
        ))
        write(child / "selection.json", dict(source_sha256={}))
        write(child / "model_identity.json", {"model": "same model"})
        write(child / "settings.json", {"profile": f"seed{seed}"})
        write(child / "input_preflight.json", {"status": "PASSED", "seed": seed})
        write(child / "run_contract.json", dict(inputs={
            name: runner.restart.digest(child / name) for name in (
                "source_identity.json", "prepared/manifest.json", "selection.json",
                "settings.json", "model_identity.json", "input_preflight.json",
            )
        }))
    for name in comparison.ROOT_INPUTS:
        if name == "source_identity.json":
            continue
        write(base / name, comparison.settings() if name == "comparison_settings.json" else {})
    write(base / "run_contract.json", dict(
        profile=comparison.NAME, planned_trajectories=1336,
        automatic_retry=False, resume_supported=False,
        inputs={name: runner.restart.digest(base / name) for name in comparison.ROOT_INPUTS},
        child_contracts={str(seed): runner.restart.digest(base / f"seed{seed}/run_contract.json")
                         for seed in comparison.SEEDS},
    ))
    write(base / "service_exit_receipt.json", dict(
        numeric_exit_status_recorded=True, controller_returncode=1, successful_main_exit=False,
        main_process_absent_when_hook_ran=True,
        run_contract_sha256=runner.restart.digest(base / "run_contract.json"),
        systemd_environment=dict(EXIT_CODE="exited", EXIT_STATUS="1"),
    ))
    return base


@pytest.fixture
def failed_launch(original, tmp_path):
    failed = tmp_path / "failed-initialization"
    shutil.copytree(original, failed)
    write(failed / "recovery_contract.json", {"original": str(original)})
    root = runner.restart.read(failed / "run_contract.json")
    root["continuation"] = dict(
        profile=recovery.PROFILE,
        recovery_contract_sha256=runner.restart.digest(failed / "recovery_contract.json"),
    )
    write(failed / "run_contract.json", root)
    error = "RuntimeError: m1_lr2e7 trainer exited 1; evidence retained"
    for name in ("state.json", "seed42/state.json"):
        state = runner.restart.read(failed / name)
        state.update(status="FAILED", error=error)
        write(failed / name, state)
    cell = failed / "seed42/m1_lr2e7"
    write(cell / "summary.json", dict(status="FAILED", error=error, returncode=1))
    write(cell / "gpu_monitor_receipt.json", dict(
        status="FAILED", phase="WAITING_HOLDER_RELEASE", returncode=1,
        monitoring_activated=False, active_sampled_peak_bytes=0,
    ))
    (cell / "trainer.log").write_text(
        "gpu_memory_coordinator.py torch.cuda.synchronize(0) CUDA error: out of memory\n"
    )
    placement = cell / "artifacts/placement/worker.jsonl"
    placement.parent.mkdir(parents=True)
    placement.write_text(json.dumps(dict(event="exchange", phase="init_model:error",
                                         detail={"event": "failed"})) + "\n")
    exit_receipt = runner.restart.read(failed / "service_exit_receipt.json")
    exit_receipt["run_contract_sha256"] = runner.restart.digest(failed / "run_contract.json")
    write(failed / "service_exit_receipt.json", exit_receipt)
    path = tmp_path / "failed_launch_audit.json"
    write(path, dict(
        profile=recovery.FAILED_LAUNCH_PROFILE, status="PASS", original_v3=str(failed),
        actual_exit1=True, zero_new_trajectories=True, zero_actor_critic_updates=True,
        new_trajectories=0, actor_updates=0, critic_updates=0,
        failure_kind="cuda_context_before_holder_release_oom", guard_phase="WAITING_HOLDER_RELEASE",
        run_contract_sha256=runner.restart.digest(failed / "run_contract.json"),
        source_sha256=runner.restart.read(failed / "source_identity.json"),
        file_inventory=recovery._inventory(failed), old_failed_launch_preserved=True,
        automatic_retry=False,
    ))
    return path, failed


def binding_fixture(original, base, failed_launch):
    files = {}
    for seed, cell in recovery.PREFIX:
        relative = f"seed{seed}/{cell}"
        shutil.copytree(original / relative, base / relative)
        files[relative] = recovery._inventory(base / relative)
    for seed in comparison.SEEDS:
        shutil.copytree(original / f"seed{seed}/prepared", base / f"seed{seed}/prepared")
        for name in ("model_identity.json", "settings.json", "selection.json",
                     "input_preflight.json"):
            shutil.copy2(original / f"seed{seed}" / name, base / f"seed{seed}" / name)
    contract = dict(
        profile=recovery.PROFILE, status="SEALED", original=str(original),
        original_files=recovery._inventory(original),
        source_changes=recovery._frozen_contracts(original), imported_files=files,
        imported_cells=[list(row) for row in recovery.PREFIX], imported_trajectories=212,
        remaining_cells=[list(row) for row in recovery.REMAINING], new_trajectories=1124,
        complete_comparison_budget=1336, retrained=False, reevaluated=False,
        automatic_retry=False, optimizer_resume=False, original_failure_preserved=True,
        gpu_budget=recovery.GPU_BUDGET,
        failed_launch=recovery._verify_failed_launch(failed_launch[0], original),
    )
    write(base / "recovery_contract.json", contract)
    root = dict(continuation=dict(
        profile=recovery.PROFILE,
        recovery_contract_sha256=runner.restart.digest(base / "recovery_contract.json"),
    ))
    write(base / "run_contract.json", root)
    return root


def test_bindings_keep_imported_summaries_and_entire_failed_history_unchanged(
    original, tmp_path, failed_launch,
):
    before = recovery._inventory(original)
    base = tmp_path / "new"
    root = binding_fixture(original, base, failed_launch)
    recovery.verify_bindings(base, root)
    assert recovery._inventory(original) == before
    for seed, cell in recovery.PREFIX:
        assert (base / f"seed{seed}" / cell / "summary.json").read_bytes() == (
            original / f"seed{seed}" / cell / "summary.json"
        ).read_bytes()


@pytest.mark.parametrize("damage", [
    "import_raw", "import_summary", "old_log", "old_source", "science", "new_data",
    "recovery_contract", "extra_import", "wrong_binding", "old_remaining_started",
    "preflight", "model", "seed_settings", "selection",
])
def test_binding_mutations_are_rejected(original, tmp_path, failed_launch, damage):
    base = tmp_path / "new"
    root = binding_fixture(original, base, failed_launch)
    if damage in {"import_raw", "import_summary"}:
        name = "raw.jsonl" if damage == "import_raw" else "summary.json"
        (base / "seed42/sft0" / name).write_bytes(b"tampered")
    elif damage == "old_log":
        (original / "extra.log").write_bytes(b"later modification")
    elif damage == "old_source":
        (original / "sources/src/science.py").write_bytes(b"changed")
    elif damage == "science":
        (tmp_path / "live/src/science.py").write_bytes(b"changed algorithm")
    elif damage == "new_data":
        (base / "seed43/prepared/dev.parquet").write_bytes(b"changed")
    elif damage == "recovery_contract":
        write(base / "recovery_contract.json", {"new_trajectories": 1336})
    elif damage == "extra_import":
        (base / "seed42/sft0/extra.log").write_bytes(b"unbound")
    elif damage == "wrong_binding":
        root["continuation"]["recovery_contract_sha256"] = "a" * 64
    elif damage in {"preflight", "model", "seed_settings", "selection"}:
        name = dict(preflight="input_preflight.json", model="model_identity.json",
                    seed_settings="settings.json", selection="selection.json")[damage]
        write(base / "seed43" / name, {"tampered": True})
    else:
        (original / "seed43/sft0").mkdir()
    with pytest.raises((RuntimeError, ValueError)):
        recovery.verify_bindings(base, root)


@pytest.mark.parametrize("damage", ["exit0", "bool_exit", "worker_failed", "other_error"])
def test_only_known_post_readout_failure_is_eligible(original, damage):
    path = original / "service_exit_receipt.json"
    value = runner.restart.read(path)
    if damage == "exit0":
        value["controller_returncode"] = 0
    elif damage == "bool_exit":
        value["controller_returncode"] = True
    elif damage == "worker_failed":
        path = original / "seed42/m1_lr1e7/summary.json"
        value = runner.restart.read(path)
        value["returncode"] = 1
    else:
        path = original / "state.json"
        value = runner.restart.read(path)
        value["error"] = "CUDA OOM"
    write(path, value)
    with pytest.raises(RuntimeError):
        recovery._known_failure(original)


@pytest.fixture
def runnable(tmp_path, monkeypatch):
    base = tmp_path / "new"
    for seed, cell in recovery.PREFIX:
        write(base / f"seed{seed}" / cell / "summary.json", dict(
            cell={"cell_id": cell}, status="COMPLETE", historical=True,
        ))
    (base / "seed43").mkdir()
    write(base / "run_contract.json", dict(continuation={"profile": recovery.PROFILE}))
    write(base / "recovery_contract.json", {})
    monkeypatch.setattr(comparison, "check_contract", lambda path: None)
    monkeypatch.setattr(runner, "check_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparison, "refresh_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "refresh_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(comparison, "_monitor", lambda *args: None)
    monkeypatch.setattr(runner, "monitor_status", lambda *args: None)
    monkeypatch.setattr(runner, "memory_coordinator_config", lambda path: {"fixture": True})
    monkeypatch.setattr(recovery, "wait_native_budget", lambda child, cell: None)
    monkeypatch.setattr(gpu_memory_coordinator, "controller_idle",
                        lambda config: {"payload_bytes": 0})
    monkeypatch.setattr(recovery.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=108 * 2**30))
    calls, retired, post, stopped = [], [], [], []

    def run_cell(child, cell_id, *, profile):
        calls.append((profile.seed, cell_id))
        summary = dict(status="COMPLETE", cell={"cell_id": cell_id}, seed=profile.seed)
        write(child / cell_id / "summary.json", summary)
        return summary

    def retire(child, spec, summary, **kwargs):
        retired.append((summary["seed"], spec.cell_id))
        return dict(status="COMPLETE")

    monkeypatch.setattr(runner, "run_cell", run_cell)
    monkeypatch.setattr(runner, "retire_completed_moments", retire)
    monkeypatch.setattr(comparison, "after_cell",
                        lambda child, spec, summary: post.append((summary["seed"], spec.cell_id)))
    monkeypatch.setattr(runner, "stop_reservation", lambda child: stopped.append(child.name))
    monkeypatch.setattr(comparison, "_readout", lambda *args, **kwargs: dict(
        execution_status="COMPLETE", financial_status="COMPLETE",
    ))
    return base, calls, retired, post, stopped


def test_exact_eight_cell_order_no_imported_requests_and_real_seed43_loop(runnable):
    base, calls, retired, post, stopped = runnable
    originals = {cell: (base / "seed42" / cell / "summary.json").read_bytes()
                 for _, cell in recovery.PREFIX}
    recovery.run(base)
    assert calls == post == list(recovery.REMAINING)
    assert retired == [row for row in recovery.REMAINING if row[1] != "sft0"]
    assert stopped == ["seed42", "seed43"]
    state = runner.restart.read(base / "state.json")
    assert state["status"] == "COMPLETE" and state["completed_seeds"] == [42, 43]
    assert (state["imported_trajectories"], state["new_trajectories"],
            state["planned_trajectories"]) == (212, 1124, 1336)
    for cell, payload in originals.items():
        assert (base / "seed42" / cell / "summary.json").read_bytes() == payload
    assert len(runner.restart.read(base / "seed42/state.json")["cells"]) == 5
    with pytest.raises(RuntimeError, match="one-shot"):
        recovery.run(base)


def test_worker_failure_preserves_failure_and_never_starts_following_cell_or_seed(
    runnable, monkeypatch,
):
    base, calls, retired, post, stopped = runnable

    def fail(child, cell, **kwargs):
        calls.append((42, cell))
        raise RuntimeError("actual failed attempt")

    monkeypatch.setattr(runner, "run_cell", fail)
    with pytest.raises(RuntimeError, match="actual failed attempt"):
        recovery.run(base)
    assert calls == [recovery.REMAINING[0]] and not retired and not post
    assert stopped == ["seed42"]
    assert runner.restart.read(base / "seed42/state.json")["status"] == "FAILED"
    assert runner.restart.read(base / "state.json")["status"] == "FAILED"
    assert not (base / "seed43/state.json").exists()


@pytest.mark.parametrize("already", ["state.json", "seed42/state.json", "seed43/sft0"])
def test_preexisting_attempt_rejected_before_any_worker(runnable, already):
    base, calls, *_ = runnable
    path = base / already
    if already.endswith("json"):
        write(path, {})
    else:
        path.mkdir()
    with pytest.raises(RuntimeError, match="attempted|resume"):
        recovery.run(base)
    assert not calls


def test_actual_disk_shortage_never_becomes_training_retry(runnable, monkeypatch):
    base, calls, *_ = runnable
    monkeypatch.setattr(recovery.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=106 * 2**30))
    with pytest.raises(RuntimeError, match="107 GiB"):
        recovery.run(base)
    assert not calls and not (base / "state.json").exists()


def test_standard_comparison_run_rejects_continuation_before_state_or_worker(runnable):
    base, calls, *_ = runnable
    with pytest.raises(RuntimeError, match="action_pooling_recovery"):
        comparison.run(base)
    assert not calls and not (base / "state.json").exists()


def test_import_symlink_is_rejected_without_following_it(original, tmp_path):
    (original / "seed42/sft0/redirect").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="symlink"):
        recovery._inventory(original)


@pytest.mark.parametrize("damage", [None, "data", "model", "preflight"])
def test_prepare_imports_prefix_and_reuses_only_matching_original_preflight(
    original, tmp_path, monkeypatch, failed_launch, damage,
):
    before = recovery._inventory(original)
    base = tmp_path / "prepared-new"
    audit = dict(
        profile=recovery.PROFILE, status="COMPLETE", original=str(original),
        original_files=before, source_changes=recovery._frozen_contracts(original),
        imported_cells=[list(row) for row in recovery.PREFIX], imported_trajectories=212,
        remaining_cells=[list(row) for row in recovery.REMAINING], new_trajectories=1124,
        complete_comparison_budget=1336, retrained=False, reevaluated=False,
    )
    monkeypatch.setattr(recovery, "audit_original", lambda path: audit)
    calls = []

    def prepare(target, **kwargs):
        calls.append(("prepare", kwargs["coordinated_reservation_directory"]))
        target.mkdir()
        for seed in comparison.SEEDS:
            source, child = original / f"seed{seed}", target / f"seed{seed}"
            shutil.copytree(source / "prepared", child / "prepared")
            for name in ("model_identity.json", "settings.json", "selection.json"):
                shutil.copy2(source / name, child / name)
        if damage == "data":
            (target / "seed43/prepared/dev.parquet").write_bytes(b"new data")
        elif damage == "model":
            write(target / "seed43/model_identity.json", {"model": "different"})

    def seal(target, *, continuation):
        calls.append(("seal", continuation["profile"]))
        if damage == "preflight":
            write(target / "seed43/input_preflight.json", {"status": "PASSED", "tampered": True})
        recovery.verify_bindings(target, {"continuation": continuation})

    monkeypatch.setattr(comparison, "prepare", prepare)
    monkeypatch.setattr(comparison, "seal", seal)
    if damage is None:
        result = recovery.prepare(
            base, original=original, coordinated_reservation_directory="holder",
            failed_launch_audit=failed_launch[0],
        )
        assert result["new_trajectories"] == 1124
        assert calls == [("prepare", "holder"), ("seal", recovery.PROFILE)]
        assert (base / "seed43/input_preflight.json").read_bytes() == (
            original / "seed43/input_preflight.json"
        ).read_bytes()
        for seed, cell in recovery.PREFIX:
            assert recovery._inventory(base / f"seed{seed}" / cell) == recovery._inventory(
                original / f"seed{seed}" / cell
            )
    else:
        with pytest.raises(RuntimeError, match="differ|changed"):
            recovery.prepare(base, original=original, coordinated_reservation_directory="holder",
                             failed_launch_audit=failed_launch[0])
    assert recovery._inventory(original) == before


def test_failed_initialization_binds_actual_exit_and_zero_outputs(original, failed_launch):
    proof, failed = failed_launch
    before = recovery._inventory(failed)
    assert recovery._verify_failed_launch(proof, original) == dict(
        path=str(proof), sha256=runner.restart.digest(proof),
    )
    assert recovery._inventory(failed) == before


@pytest.mark.parametrize("damage", [
    "unbound_file", "raw_rollout", "empty_science_file", "weight", "update_event",
    "exit0", "activated", "other_trace", "next_cell", "imported_summary", "source_copy",
])
def test_failed_launch_rejects_outputs_even_when_proof_claims_zero(
    original, failed_launch, damage,
):
    proof, failed = failed_launch
    cell = failed / "seed42/m1_lr2e7"
    if damage == "unbound_file":
        (failed / "later.log").write_text("changed after audit")
    elif damage in {"raw_rollout", "empty_science_file"}:
        path = cell / "artifacts/rollout_seed/actual.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"new_trajectory":true}\n' if damage == "raw_rollout" else "")
    elif damage == "weight":
        path = cell / "checkpoints/actor/weights.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"new actor weight")
    elif damage == "update_event":
        (cell / "artifacts/placement/worker.jsonl").write_text(
            json.dumps(dict(event="exchange", phase="actor_update")) + "\n"
        )
    elif damage == "exit0":
        path = failed / "service_exit_receipt.json"
        value = runner.restart.read(path)
        value.update(controller_returncode=0)
        write(path, value)
    elif damage == "activated":
        path = cell / "gpu_monitor_receipt.json"
        value = runner.restart.read(path)
        value["monitoring_activated"] = True
        write(path, value)
    elif damage == "other_trace":
        (cell / "trainer.log").write_text("ordinary trainer error")
    elif damage == "next_cell":
        (failed / "seed43/sft0").mkdir()
    elif damage == "imported_summary":
        write(failed / "seed42/sft0/summary.json", {"status": "COMPLETE", "changed": True})
    else:
        (failed / "sources/src/science.py").write_bytes(b"tampered frozen source")
    if damage != "unbound_file":
        # A self-declared zero proof cannot replace the independent actual-file checks.
        value = runner.restart.read(proof)
        value["file_inventory"] = recovery._inventory(failed)
        write(proof, value)
    with pytest.raises(RuntimeError):
        recovery._verify_failed_launch(proof, original)


class Clock:
    def __init__(self):
        self.now, self.sleeps, self.jump = 0, [], None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds if self.jump is None else self.jump


def budget_holder(payload=76, *, claimed=False, paused=False, owner=None):
    return dict(
        status="PAUSED" if paused else "HOLDING", paused=paused, closed=False,
        training_connected=claimed, persistent=True, coordination_profile="persistent_claim_v1",
        payload_bytes=payload * 2**30, allocator_allocated_bytes=payload * 2**30,
        allocator_reserved_bytes=payload * 2**30, maximum_bytes=80 * 2**30,
        error=None, cleanup_error=None,
        claim=(dict(lease_token="budget-lease", owner=copy.deepcopy(owner),
                    cgroup_populated=True, watchdog_error=None) if claimed else None),
    )


@pytest.fixture
def budget(tmp_path, monkeypatch):
    child, directory = tmp_path / "budget-child", tmp_path / "holder"
    child.mkdir()
    directory.mkdir()
    owner = dict(pid=os.getpid(), start_time_ticks=11, cgroup_path="/test/controller.service",
                 process_group_id=22, session_id=33)
    holder_owner = dict(
        pid=123, directory=str(directory), cuda_device=0, started_at_SGT="start",
        expires_at_SGT=None, persistent=True, allocation_only=True, automatic_restart=False,
    )
    raw = json.dumps(holder_owner).encode()
    (directory / "owner.json").write_bytes(raw)
    config = dict(
        profile=gpu_memory_coordinator.PROFILE, directory=str(directory),
        owner_sha256=hashlib.sha256(raw).hexdigest(), holder_pid=123,
        holder_start_time_ticks=456, gpu_uuid="GPU-test", cuda_device=0,
        lease_token="budget-lease", evidence_directory=str(tmp_path / "coordinator"),
        portfolio_target_bytes=76 * 2**30,
        phase_peak_bytes={name: 76 * 2**30 for name in gpu_memory_coordinator.PHASES},
        minimum_free_bytes=2**31, wait_timeout_seconds=1800, poll_seconds=1,
        active_arm_holder_policy="paused_zero",
    )
    digest = gpu_memory_coordinator.freeze_config(child / "gpu_coordinator_config.json", config)
    write(child / "run_contract.json", {"inputs": {"gpu_coordinator_config.json": digest}})
    clock, states, used, operations = Clock(), [budget_holder()], [80], []

    def identity(pid):
        return copy.deepcopy(owner if pid == os.getpid()
                             else dict(pid=123, start_time_ticks=456))

    def status(request_directory, payload, *, timeout):
        assert request_directory == str(directory) and 0 < timeout <= 15
        operations.append(payload)
        value = copy.deepcopy(states.pop(0) if len(states) > 1 else states[0])
        return {**holder_owner, **value}

    def driver(deadline):
        assert deadline > clock.now
        value = used.pop(0) if len(used) > 1 else used[0]
        return dict(gpu_uuid="GPU-test", driver_total_bytes=192 * 2**30,
                    driver_free_bytes=(192 - value) * 2**30)

    monkeypatch.setattr(recovery, "process_identity", identity)
    monkeypatch.setattr(recovery, "request", status)
    monkeypatch.setattr(recovery, "_budget_driver_snapshot", driver)
    monkeypatch.setattr(recovery.time, "monotonic", clock)
    monkeypatch.setattr(recovery.time, "sleep", clock.sleep)
    return SimpleNamespace(child=child, config=config, owner=owner, clock=clock,
                           states=states, used=used, operations=operations, driver=driver,
                           status=status)


def budget_rows(budget, cell="next"):
    return [json.loads(line) for line in
            (budget.child / f"native_budget-{cell}.jsonl").read_text().splitlines()]


def test_cpu_budget_boundary_requires_two_fresh_samples_and_never_starts_native(budget):
    recovery.wait_native_budget(budget.child, "next")
    rows = budget_rows(budget)
    assert [row["ready"] for row in rows] == [False, True]
    assert [row["ready_streak"] for row in rows] == [1, 2]
    assert all(row["observed_nonholder_bytes"] == 4 * 2**30 for row in rows)
    assert budget.clock.sleeps == [5]
    assert budget.operations == [{"op": "status"}] * 4
    assert not (budget.child / "next").exists()
    receipt = runner.restart.read(budget.child / "native_budget-next.json")
    assert receipt["status"] == "READY" and receipt["samples"] == 2
    assert receipt["active_aggregate_guard_unchanged"] is True
    assert receipt["native_started_by_wait"] is False
    with pytest.raises(RuntimeError, match="no retry"):
        recovery.wait_native_budget(budget.child, "next")


def test_budget_delayed_capacity_resets_streak_and_publishes_real_waiting(budget, monkeypatch):
    budget.used[:] = [87, 80, 81, 80, 80]
    receipts = []
    original_sleep = budget.clock.sleep

    def sleep(seconds):
        receipts.append(runner.restart.read(budget.child / "native_budget-next.json"))
        queue = runner.restart.read(budget.child / "gpu_queue_state.json")
        assert queue["status"] == "WAITING_FOR_GPU" and queue["native_started"] is False
        assert not (budget.child / "next").exists()
        original_sleep(seconds)

    monkeypatch.setattr(recovery.time, "sleep", sleep)
    recovery.wait_native_budget(budget.child, "next")
    rows = budget_rows(budget)
    assert [row["ready_streak"] for row in rows] == [0, 1, 0, 1, 2]
    assert rows[0]["observed_nonholder_bytes"] == 11 * 2**30
    assert rows[-1]["ready"] is True and budget.clock.now == 20
    assert all(receipt["status"] == "WAITING" for receipt in receipts)


def test_incoherent_holder_bracket_gets_no_payload_credit(budget):
    budget.states[:] = [budget_holder(76), budget_holder(75), budget_holder(76)]
    recovery.wait_native_budget(budget.child, "next")
    rows = budget_rows(budget)
    assert rows[0]["coherent"] is False and rows[0]["holder_payload_credit_bytes"] == 0
    assert rows[0]["observed_nonholder_bytes"] == 80 * 2**30
    assert [row["ready_streak"] for row in rows] == [0, 1, 2]


def test_timeout_keeps_cell_unstarted_and_retains_samples(budget):
    budget.used[:] = [87]
    budget.clock.jump = 1800
    with pytest.raises(TimeoutError, match="1800 seconds"):
        recovery.wait_native_budget(budget.child, "next")
    receipt = runner.restart.read(budget.child / "native_budget-next.json")
    assert receipt["status"] == "TIMEOUT" and receipt["samples"] == 1
    assert budget.operations == [{"op": "status"}] * 2
    assert not (budget.child / "next").exists()
    assert budget_rows(budget)[-1]["status"] == "TIMEOUT"


@pytest.mark.parametrize("claimed,paused", [(False, False), (True, True), (True, False)])
def test_budget_allows_verified_unclaimed_or_same_owner_paused_or_idle_holder(
    budget, claimed, paused,
):
    budget.states[:] = [budget_holder(claimed=claimed, paused=paused, owner=budget.owner)]
    recovery.wait_native_budget(budget.child, "next")
    assert budget_rows(budget)[-1]["ready"] is True
    assert budget.operations == [{"op": "status"}] * 4


@pytest.mark.parametrize("damage", [
    "foreign_token", "foreign_owner", "unpopulated", "watchdog", "disconnected_claim",
    "unclaimed_connected", "paused_status", "bad_payload", "allocator_order", "owner_sha",
    "holder_ticks", "controller_identity", "rpc_unavailable",
])
def test_budget_never_credits_unverified_or_unhealthy_holder(budget, monkeypatch, damage):
    state = budget_holder(claimed=True, owner=budget.owner)
    budget.states[:] = [state]
    if damage == "foreign_token":
        state["claim"]["lease_token"] = "another-controller"
    elif damage == "foreign_owner":
        state["claim"]["owner"]["start_time_ticks"] += 1
    elif damage == "unpopulated":
        state["claim"]["cgroup_populated"] = False
    elif damage == "watchdog":
        state["claim"]["watchdog_error"] = "failed cgroup check"
    elif damage == "disconnected_claim":
        state["training_connected"] = False
    elif damage == "unclaimed_connected":
        state["claim"] = None
    elif damage == "paused_status":
        state["paused"] = True
    elif damage == "bad_payload":
        state["payload_bytes"] = True
    elif damage == "allocator_order":
        state["allocator_reserved_bytes"] = 75 * 2**30
    elif damage == "owner_sha":
        (Path(budget.config["directory"]) / "owner.json").write_text("tampered")
    elif damage == "holder_ticks":
        monkeypatch.setattr(recovery, "process_identity", lambda pid: copy.deepcopy(
            budget.owner if pid == os.getpid() else dict(pid=123, start_time_ticks=457)
        ))
    elif damage == "controller_identity":
        identities = iter([copy.deepcopy(budget.owner), {**budget.owner, "session_id": 99}])
        monkeypatch.setattr(recovery, "process_identity", lambda pid: next(identities))
    else:
        def unavailable(*args, **kwargs):
            raise OSError("status socket unavailable")
        monkeypatch.setattr(recovery, "request", unavailable)
    with pytest.raises((RuntimeError, OSError)):
        recovery.wait_native_budget(budget.child, "next")
    receipt = runner.restart.read(budget.child / "native_budget-next.json")
    assert receipt["status"] == "FAILED" and receipt["native_started_by_wait"] is False
    assert not (budget.child / "next").exists()


@pytest.mark.parametrize("damage", ["uuid", "free_negative", "free_excess", "total_bool",
                                     "payload_exceeds_used"])
def test_invalid_driver_telemetry_fails_closed(budget, monkeypatch, damage):
    sample = budget.driver(1800)
    if damage == "uuid":
        sample["gpu_uuid"] = "GPU-foreign"
    elif damage == "free_negative":
        sample["driver_free_bytes"] = -1
    elif damage == "free_excess":
        sample["driver_free_bytes"] = sample["driver_total_bytes"] + 1
    elif damage == "total_bool":
        sample["driver_total_bytes"] = True
    else:
        sample["driver_free_bytes"] = sample["driver_total_bytes"] - 75 * 2**30
    monkeypatch.setattr(recovery, "_budget_driver_snapshot", lambda deadline: sample)
    with pytest.raises(RuntimeError):
        recovery.wait_native_budget(budget.child, "next")
    assert runner.restart.read(budget.child / "native_budget-next.json")["status"] == "FAILED"
    assert not (budget.child / "next").exists()


def test_deadline_covers_status_rpc_time_and_stops_without_driver_query(budget, monkeypatch):
    driver_queries = []

    def slow_status(*args, **kwargs):
        value = budget.status(*args, **kwargs)
        budget.clock.now = 1800
        return value

    monkeypatch.setattr(recovery, "request", slow_status)
    monkeypatch.setattr(recovery, "_budget_driver_snapshot", lambda deadline: (
        recovery._budget_remaining(deadline), driver_queries.append(True)
    ))
    with pytest.raises(TimeoutError):
        recovery.wait_native_budget(budget.child, "next")
    assert not driver_queries and not (budget.child / "next").exists()


def test_driver_query_is_cpu_subprocess_with_remaining_timeout(budget, monkeypatch):
    calls = []

    def sample(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="GPU-test, 114688, 196608\n")

    monkeypatch.setattr(recovery.subprocess, "run", sample)
    budget.clock.now = 1798
    result = PRODUCTION_DRIVER_SNAPSHOT(1800)
    assert result["driver_free_bytes"] == 112 * 2**30
    assert calls[0][0][0] == "nvidia-smi" and calls[0][1]["timeout"] == 2


def test_native_budget_failure_stops_before_any_worker_and_future_seed(runnable, monkeypatch):
    base, calls, retired, post, stopped = runnable

    def unavailable(child, cell):
        assert not (child / cell).exists()
        raise TimeoutError("CPU budget deadline")

    monkeypatch.setattr(recovery, "wait_native_budget", unavailable)
    with pytest.raises(TimeoutError, match="CPU budget deadline"):
        recovery.run(base)
    assert not calls and not retired and not post and stopped == ["seed42"]
    assert runner.restart.read(base / "state.json")["status"] == "FAILED"
    assert not (base / "seed43/state.json").exists()


def test_each_cell_budget_runs_before_native_and_survives_same_owner_idle_refill(
    runnable, budget, monkeypatch,
):
    base, calls, *_ = runnable
    events = []
    for seed in comparison.SEEDS:
        child = base / f"seed{seed}"
        shutil.copy2(budget.child / "gpu_coordinator_config.json",
                     child / "gpu_coordinator_config.json")
        shutil.copy2(budget.child / "run_contract.json", child / "run_contract.json")
    monkeypatch.setattr(runner, "memory_coordinator_config", lambda child: budget.config)
    native = runner.run_cell

    def wait(child, cell):
        events.append(("wait", child.name, cell))
        assert not (child / cell).exists()
        PRODUCTION_WAIT(child, cell)

    def run_cell(child, cell, **kwargs):
        assert runner.restart.read(child / f"native_budget-{cell}.json")["status"] == "READY"
        events.append(("native", child.name, cell))
        # The native coordinator owns this lifecycle transition, not the CPU wait.
        budget.states[:] = [budget_holder(0, claimed=True, paused=True, owner=budget.owner)]
        return native(child, cell, **kwargs)

    def refill(config):
        budget.states[:] = [budget_holder(76, claimed=True, paused=False, owner=budget.owner)]
        return {"payload_bytes": 76 * 2**30}

    monkeypatch.setattr(recovery, "wait_native_budget", wait)
    monkeypatch.setattr(runner, "run_cell", run_cell)
    monkeypatch.setattr(gpu_memory_coordinator, "controller_idle", refill)
    recovery.run(base)
    assert calls == list(recovery.REMAINING)
    assert events == [(phase, f"seed{seed}", cell) for seed, cell in recovery.REMAINING
                      for phase in ("wait", "native")]
    first, second = recovery.REMAINING[:2]
    first_rows = [json.loads(line) for line in (base / f"seed{first[0]}" /
                  f"native_budget-{first[1]}.jsonl").read_text().splitlines()]
    second_rows = [json.loads(line) for line in (base / f"seed{second[0]}" /
                   f"native_budget-{second[1]}.jsonl").read_text().splitlines()]
    assert first_rows[0]["holder_before"]["claim"] is None
    assert second_rows[0]["holder_before"]["claim"]["owner"] == budget.owner
    assert second_rows[0]["holder_before"]["status"] == "HOLDING"
    assert second_rows[0]["holder_before"]["paused"] is False
    assert budget.operations == [{"op": "status"}] * 32


@pytest.fixture
def timeout_history(original, failed_launch, tmp_path):
    base = tmp_path / "timeout-v4"
    binding_fixture(original, base, failed_launch)
    for directory in (base, *(base / f"seed{s}" for s in comparison.SEEDS)):
        source = original if directory == base else original / directory.name
        shutil.copytree(source / "sources", directory / "sources")
        for relative, current in runner.source_files().items():
            target = directory / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(current.read_bytes())
        self_path = directory / "sources" / recovery.SELF
        self_path.write_bytes(b"new recovery")
        sources = {relative: runner.restart.digest(directory / "sources" / relative)
                   for relative in runner.source_files()}
        write(directory / "source_identity.json", sources)
        if directory != base:
            shutil.copy2(source / "run_contract.json", directory / "run_contract.json")
            contract = runner.restart.read(directory / "run_contract.json")
            contract["inputs"]["source_identity.json"] = runner.restart.digest(
                directory / "source_identity.json"
            )
            write(directory / "gpu_coordinator_config.json", {"frozen": "config"})
            contract["inputs"]["gpu_coordinator_config.json"] = runner.restart.digest(
                directory / "gpu_coordinator_config.json"
            )
            write(directory / "run_contract.json", contract)
    for name in comparison.ROOT_INPUTS:
        if name != "source_identity.json":
            shutil.copy2(original / name, base / name)
    root = runner.restart.read(original / "run_contract.json")
    root.update(
        inputs={name: runner.restart.digest(base / name) for name in comparison.ROOT_INPUTS},
        child_contracts={str(s): runner.restart.digest(base / f"seed{s}/run_contract.json")
                         for s in comparison.SEEDS},
        continuation=dict(profile=recovery.PROFILE,
                          recovery_contract_sha256=runner.restart.digest(
                              base / "recovery_contract.json")),
    )
    write(base / "run_contract.json", root)
    summaries = [runner.restart.read(base / f"seed{s}" / c / "summary.json")
                 for s, c in recovery.PREFIX]
    for index, (seed, cell) in enumerate(recovery.TIMEOUT_PREFIX[len(recovery.PREFIX):], start=2):
        summary = dict(status="COMPLETE", returncode=0, cell={"cell_id": cell},
                       execution_identity_sha256=str(index) * 64)
        summaries.append(summary)
        write(base / f"seed{seed}" / cell / "summary.json", summary)
        (base / f"seed{seed}" / cell / "raw.jsonl").write_text('{"completed":true}\n')
    write(base / "state.json", dict(status="FAILED", profile=comparison.NAME,
          error=recovery.TIMEOUT_ERROR, current_seed=42, completed_seeds=[]))
    write(base / "seed42/state.json", dict(status="FAILED", seed=42,
          error=recovery.TIMEOUT_ERROR, current_cell=recovery.TIMEOUT_REMAINING[0][1],
          cells=summaries))
    exit_receipt = runner.restart.read(original / "service_exit_receipt.json")
    exit_receipt["run_contract_sha256"] = runner.restart.digest(base / "run_contract.json")
    write(base / "service_exit_receipt.json", exit_receipt)
    pending = recovery.TIMEOUT_REMAINING[0][1]
    write(base / f"seed42/native_budget-{pending}.json", dict(
        **recovery.GPU_BUDGET, status="TIMEOUT", cell_id=pending, native_started_by_wait=False,
        elapsed_seconds=1800.01, samples=1,
        config_sha256=runner.restart.digest(base / "seed42/gpu_coordinator_config.json"),
    ))
    (base / f"seed42/native_budget-{pending}.jsonl").write_text(
        json.dumps({"ready": False}) + "\n" +
        json.dumps(dict(status="TIMEOUT", failure={"type": "TimeoutError"})) + "\n"
    )
    snapshot = tmp_path / "v4_snapshot.json"
    log = tmp_path / "pre_edit_contract.log"
    log.write_text("actual pre-edit check completed with code0\n")
    write(snapshot, dict(
        profile=recovery.TIMEOUT_SNAPSHOT_PROFILE, status="PASS", original=str(base),
        original_status="FAILED", controller_returncode=1,
        pre_edit_contract_check=dict(returncode=0, log=str(log), command=[
            "python", "-m", "src.agents.action_pooling_comparison", "check",
            "--directory", str(base),
        ]), run_contract_sha256=runner.restart.digest(base / "run_contract.json"),
        source_sha256=runner.restart.read(base / "source_identity.json"),
        completed_cells=[c for _, c in recovery.TIMEOUT_PREFIX], completed_trajectories=516,
        remaining_trajectories=820, unstarted_pending_cell=pending, pending_cell_trajectory_count=0,
        old_failed_history_preserved=True, automatic_retry=False, files=recovery._inventory(base),
    ))
    return base, snapshot


def timeout_binding(timeout_history, base):
    original, snapshot = timeout_history
    files = {}
    for seed, cell in recovery.TIMEOUT_PREFIX:
        relative = f"seed{seed}/{cell}"
        shutil.copytree(original / relative, base / relative)
        files[relative] = recovery._inventory(base / relative)
    for seed in comparison.SEEDS:
        shutil.copytree(original / f"seed{seed}/prepared", base / f"seed{seed}/prepared")
        for name in ("model_identity.json", "settings.json", "selection.json",
                     "input_preflight.json"):
            shutil.copy2(original / f"seed{seed}" / name, base / f"seed{seed}" / name)
    contract = dict(
        profile=recovery.PROFILE, status="SEALED", continuation_kind=recovery.TIMEOUT_KIND,
        original=str(original), original_files=recovery._inventory(original),
        source_changes=recovery._timeout_sources(original),
        original_snapshot=recovery._timeout_snapshot(snapshot, original),
        imported_cells=[list(r) for r in recovery.TIMEOUT_PREFIX], imported_trajectories=516,
        remaining_cells=[list(r) for r in recovery.TIMEOUT_REMAINING], new_trajectories=820,
        completed_new_trajectories_in_original=304, complete_comparison_budget=1336,
        scientific_prefix_audit="raw_update_fit_placement_head_and_lossless_archive_v1",
        retrained=False, reevaluated=False, automatic_retry=False, optimizer_resume=False,
        original_failure_preserved=True, gpu_budget=recovery.GPU_BUDGET, imported_files=files,
        runtime_preflight=recovery.RUNTIME_PREFLIGHT,
    )
    write(base / "recovery_contract.json", contract)
    root = dict(continuation=dict(profile=recovery.PROFILE,
                                 recovery_contract_sha256=runner.restart.digest(
                                     base / "recovery_contract.json")))
    write(base / "run_contract.json", root)
    return root


def test_timeout_scope_preserves_all_four_cells_and_prior_failures(timeout_history, tmp_path):
    original, snapshot = timeout_history
    before = recovery._inventory(original)
    base = tmp_path / "v5"
    root = timeout_binding(timeout_history, base)
    recovery.verify_bindings(base, root)
    assert recovery._inventory(original) == before
    assert len(recovery._timeout_failure(original)) == 4
    assert recovery._timeout_snapshot(snapshot, original)["sha256"] == (
        runner.restart.digest(snapshot))
    for seed, cell in recovery.TIMEOUT_PREFIX:
        assert (base / f"seed{seed}" / cell / "summary.json").read_bytes() == (
            original / f"seed{seed}" / cell / "summary.json").read_bytes()


@pytest.mark.parametrize("damage", [
    "exit0", "other_error", "pending_dir", "future_seed", "changed_summary", "early_timeout",
    "ready_budget", "budget95", "journal_ready", "snapshot", "pre_edit_log", "imported_raw",
    "model", "science", "self_after_seal", "old_v3",
])
def test_timeout_continuation_rejects_tampered_or_started_history(
    timeout_history, tmp_path, damage,
):
    original, snapshot = timeout_history
    base = tmp_path / "v5"
    root = timeout_binding(timeout_history, base)
    pending = recovery.TIMEOUT_REMAINING[0][1]
    if damage == "exit0":
        path = original / "service_exit_receipt.json"
        value = runner.restart.read(path)
        value["controller_returncode"] = 0
        write(path, value)
    elif damage == "other_error":
        path = original / "state.json"
        value = runner.restart.read(path)
        value["error"] = "CUDA OOM"
        write(path, value)
    elif damage == "pending_dir":
        (original / "seed42" / pending).mkdir()
    elif damage == "future_seed":
        write(original / "seed43/state.json", {"status": "RUNNING"})
    elif damage == "changed_summary":
        write(original / "seed42/m2_plain_lr1e7_eta025/summary.json", {"status": "FAILED"})
    elif damage in {"early_timeout", "ready_budget", "budget95"}:
        path = original / f"seed42/native_budget-{pending}.json"
        value = runner.restart.read(path)
        key, new_value = dict(early_timeout=("elapsed_seconds", 30),
                              ready_budget=("status", "READY"),
                              budget95=("aggregate_limit_bytes", 95 * 2**30))[damage]
        value[key] = new_value
        write(path, value)
    elif damage == "journal_ready":
        (original / f"seed42/native_budget-{pending}.jsonl").write_text('{"ready":true}\n')
    elif damage == "snapshot":
        value = runner.restart.read(snapshot)
        value["remaining_trajectories"] = 1124
        write(snapshot, value)
    elif damage == "pre_edit_log":
        Path(runner.restart.read(snapshot)["pre_edit_contract_check"]["log"]).write_text("changed")
    elif damage == "imported_raw":
        (base / "seed42/m1_lr2e7/raw.jsonl").write_text("changed")
    elif damage == "model":
        write(base / "seed43/model_identity.json", {"model": "different"})
    elif damage in {"science", "self_after_seal"}:
        relative = "src/science.py" if damage == "science" else recovery.SELF
        (tmp_path / "live" / relative).write_bytes(b"unapproved later change")
    else:
        (tmp_path / "failed-initialization/later.log").write_text("old v3 changed")
    with pytest.raises((RuntimeError, ValueError)):
        recovery.verify_bindings(base, root)


def test_timeout_failure_is_independent_of_snapshot_claims(timeout_history):
    original, _ = timeout_history
    pending = original / "seed42" / recovery.TIMEOUT_REMAINING[0][1]
    pending.mkdir()
    with pytest.raises(RuntimeError, match="native execution"):
        recovery._timeout_failure(original)


def test_timeout_audit_validates_every_cell_and_each_rl_archive_once(
    timeout_history, monkeypatch,
):
    original, snapshot = timeout_history
    validations, archives = [], []
    canonical_origin = Path(runner.restart.read(original / "recovery_contract.json")["original"])
    evidence_paths = []

    def raw(cell, seed, cell_id, summary, *args, **kwargs):
        validations.append((seed, cell_id))
        return ([None] * (60 if cell_id == "sft0" else 152), [])

    for seed, cell_id in recovery.TIMEOUT_PREFIX:
        cell = original / f"seed{seed}" / cell_id
        canonical = (canonical_origin / f"seed{seed}" / cell_id
                     if (seed, cell_id) in recovery.PREFIX else cell)
        summary = runner.restart.read(cell / "summary.json")
        summary["actor_update"] = dict(source_sha256={str(
            canonical / "artifacts/actor_update/probe.jsonl"): "original-digest"})
        summary.update(critic_fit={}, placement={}, gpu_handoff={},
                       critic_initial_head_sha256="head")
        write(cell / "summary.json", summary)
        profile = comparison.PROFILES[seed]
        spec = next(s for s in runner.cells(profile) if s.cell_id == cell_id)
        write(cell / "artifacts/critic_initialization/head-rank0.json", dict(
            seed=seed, execution_identity_sha256=summary["execution_identity_sha256"], arm=spec.arm,
            mode="cpu_seed_v1", head_sha256="head", model_path=str(comparison.MODEL),
        ))
        if canonical != cell:
            shutil.copy2(cell / "summary.json", canonical / "summary.json")
            shutil.copytree(cell / "artifacts", canonical / "artifacts")
    state = runner.restart.read(original / "seed42/state.json")
    state["cells"] = [runner.restart.read(original / f"seed{s}" / c / "summary.json")
                      for s, c in recovery.TIMEOUT_PREFIX]
    write(original / "seed42/state.json", state)
    prior = runner.restart.read(original / "recovery_contract.json")
    prior["imported_files"] = {f"seed{s}/{c}": recovery._inventory(original / f"seed{s}" / c)
                               for s, c in recovery.PREFIX}
    write(original / "recovery_contract.json", prior)
    # The original v2 prefix remains immutable: fake science probes only for this audit unit test.
    monkeypatch.setattr(recovery, "_timeout_failure", lambda path: state["cells"])
    value = runner.restart.read(snapshot)
    value["files"] = recovery._inventory(original)
    write(snapshot, value)
    monkeypatch.setattr(recovery.readout.shared, "_cell", raw)
    def extra(cell, *args, **kwargs):
        evidence_paths.append(cell)
        return dict(actor_update=dict(source_sha256={str(
            cell / "artifacts/actor_update/probe.jsonl"): "original-digest"}))

    monkeypatch.setattr(runner, "validate_extra_evidence", extra)
    monkeypatch.setattr(runner.tuning, "validate_fit_evidence", lambda *a, **k: {})
    monkeypatch.setattr(runner, "validate_placement", lambda *a, **k: {})
    monkeypatch.setattr(runner, "validate_handoff", lambda *a, **k: {})
    monkeypatch.setattr(recovery, "_archive_evidence",
                        lambda cell, summary: archives.append(cell.name))
    before = recovery._inventory(original)
    result = recovery.audit_timeout(original, snapshot=snapshot)
    assert result["imported_trajectories"] == 516 and result["new_trajectories"] == 820
    assert result["completed_new_trajectories_in_original"] == 304
    assert validations == list(recovery.TIMEOUT_PREFIX)
    assert evidence_paths == [(canonical_origin if row in recovery.PREFIX else original) /
                              f"seed{row[0]}" / row[1] for row in recovery.TIMEOUT_PREFIX]
    assert evidence_paths[1] != original / "seed42/m1_lr1e7"
    assert archives == [cell for _, cell in recovery.TIMEOUT_PREFIX if cell != "sft0"]
    assert recovery._inventory(original) == before


def test_timeout_run_executes_only_six_cells_with_516_plus_820_budget(runnable, monkeypatch):
    base, calls, retired, post, stopped = runnable
    for seed, cell in recovery.TIMEOUT_PREFIX[len(recovery.PREFIX):]:
        write(base / f"seed{seed}" / cell / "summary.json", dict(
            cell={"cell_id": cell}, status="COMPLETE", historical=True,
        ))
    write(base / "recovery_contract.json", {"continuation_kind": recovery.TIMEOUT_KIND})
    probes = []
    monkeypatch.setattr(recovery, "_isolated_runtime_preflight",
                        lambda child, cell, profile: probes.append((profile.seed, cell)))
    before = {(seed, cell): (base / f"seed{seed}" / cell / "summary.json").read_bytes()
              for seed, cell in recovery.TIMEOUT_PREFIX}
    recovery.run(base)
    assert calls == post == probes == list(recovery.TIMEOUT_REMAINING)
    assert retired == [row for row in recovery.TIMEOUT_REMAINING if row[1] != "sft0"]
    assert stopped == ["seed42", "seed43"]
    state = runner.restart.read(base / "state.json")
    assert (state["imported_trajectories"], state["new_trajectories"]) == (516, 820)
    seed42 = runner.restart.read(base / "seed42/state.json")
    seed43 = runner.restart.read(base / "seed43/state.json")
    assert (seed42["imported_trajectories"], seed42["new_trajectories"]) == (516, 152)
    assert (seed43["imported_trajectories"], seed43["new_trajectories"]) == (0, 668)
    assert all((base / f"seed{seed}" / cell / "summary.json").read_bytes() == payload
               for (seed, cell), payload in before.items())


@pytest.fixture
def runtime_probe(tmp_path, monkeypatch):
    child = tmp_path / "probe-child"
    child.mkdir()
    model = dict(model_revision="common-SFT-sha", model_snapshot_path="/same/shared-SFT")
    write(child / "model_identity.json", model)
    write(child / "prepared/manifest.json", {})
    monkeypatch.setattr(recovery, "_parent_cuda_initialized", lambda: False)
    monkeypatch.setattr(runner, "make_config", lambda *a, **k: {"exact_config": "unchanged"})
    result = SimpleNamespace(returncode=0, stdout=json.dumps(dict(
        **model, cuda_available=True, c_compiler="cc",
    )), stderr="")
    calls = []

    def run(command, **kwargs):
        assert kwargs["timeout"] == 300 and kwargs["cwd"] == runner.REPO_ROOT
        assert json.loads(kwargs["input"]) == {"exact_config": "unchanged"}
        assert "verify_gpu_runtime_and_model" in command[-1]
        calls.append(command)
        return result

    monkeypatch.setattr(recovery.subprocess, "run", run)
    return SimpleNamespace(child=child, calls=calls, result=result)


def test_isolated_original_runtime_preflight_runs_once_and_binds_output(runtime_probe):
    probe = runtime_probe
    recovery._isolated_runtime_preflight(probe.child, "first", comparison.PROFILES[42])
    recovery._isolated_runtime_preflight(probe.child, "second", comparison.PROFILES[42])
    assert len(probe.calls) == 1
    receipt = runner.restart.read(probe.child / "runtime_preflight_probe/receipt.json")
    assert receipt["status"] == "COMPLETE" and receipt["returncode"] == 0
    assert receipt["parent_cuda_before"] is receipt["parent_cuda_after"] is False
    assert receipt["runtime_sha256"] == runner.restart.digest(
        probe.child / "runtime_preflight.json")
    assert (probe.child / "runtime_preflight_probe/stdout.txt").read_text() == probe.result.stdout


@pytest.mark.parametrize("damage", ["child_failed", "model", "json", "parent_before",
                                     "parent_after", "timeout", "output_tampered"])
def test_runtime_preflight_failures_never_succeed_or_retry(runtime_probe, monkeypatch, damage):
    probe = runtime_probe
    if damage == "child_failed":
        probe.result.returncode = 7
        probe.result.stderr = "actual child error"
    elif damage == "model":
        probe.result.stdout = json.dumps(dict(cuda_available=True, model_revision="wrong"))
    elif damage == "json":
        probe.result.stdout = "not JSON"
    elif damage == "parent_before":
        monkeypatch.setattr(recovery, "_parent_cuda_initialized", lambda: True)
    elif damage == "parent_after":
        states = iter([False, True])
        monkeypatch.setattr(recovery, "_parent_cuda_initialized", lambda: next(states))
    elif damage == "timeout":
        def timeout(*args, **kwargs):
            raise recovery.subprocess.TimeoutExpired("probe", 300, output=b"partial",
                                                      stderr=b"timeout evidence")
        monkeypatch.setattr(recovery.subprocess, "run", timeout)
    else:
        recovery._isolated_runtime_preflight(probe.child, "first", comparison.PROFILES[42])
        write(probe.child / "runtime_preflight.json", {"tampered": True})
    with pytest.raises((RuntimeError, ValueError, recovery.subprocess.TimeoutExpired)):
        recovery._isolated_runtime_preflight(probe.child, "first", comparison.PROFILES[42])
    if damage == "parent_before":
        assert not probe.calls and not (probe.child / "runtime_preflight_probe").exists()
    elif damage != "output_tampered":
        receipt = runner.restart.read(probe.child / "runtime_preflight_probe/receipt.json")
        assert receipt["status"] == "FAILED"
        assert not (probe.child / "runtime_preflight.json").exists()
        monkeypatch.setattr(recovery, "_parent_cuda_initialized", lambda: False)
        with pytest.raises(RuntimeError, match="no retry"):
            recovery._isolated_runtime_preflight(probe.child, "first", comparison.PROFILES[42])


@pytest.mark.parametrize("fail_at", [None, "budget", "probe"])
def test_timeout_seed_lifecycle_gates_probe_before_native_and_fails_closed(
    runnable, monkeypatch, fail_at,
):
    base, calls, *_ = runnable
    for seed, cell in recovery.TIMEOUT_PREFIX[len(recovery.PREFIX):]:
        write(base / f"seed{seed}" / cell / "summary.json", dict(
            cell={"cell_id": cell}, status="COMPLETE", historical=True,
        ))
    write(base / "recovery_contract.json", {"continuation_kind": recovery.TIMEOUT_KIND})
    events, checked = [], set()
    native = runner.run_cell

    def gate(child, cell):
        events.append(("budget", child.name, cell))
        if fail_at == "budget":
            raise TimeoutError("unavailable GPU")

    def probe(child, cell, profile):
        if child not in checked:
            events.append(("probe", child.name, cell))
            if fail_at == "probe":
                raise RuntimeError("runtime child failed")
            checked.add(child)

    def run_cell(child, cell, **kwargs):
        events.append(("native", child.name, cell))
        return native(child, cell, **kwargs)

    monkeypatch.setattr(recovery, "wait_native_budget", gate)
    monkeypatch.setattr(recovery, "_isolated_runtime_preflight", probe)
    monkeypatch.setattr(runner, "run_cell", run_cell)
    if fail_at is not None:
        with pytest.raises((RuntimeError, TimeoutError)):
            recovery.run(base)
        assert not calls
        assert [row[0] for row in events] == (["budget"] if fail_at == "budget"
                                             else ["budget", "probe"])
        assert runner.restart.read(base / "state.json")["status"] == "FAILED"
        assert not (base / "seed43/state.json").exists()
    else:
        recovery.run(base)
        expected = []
        for seed, cell in recovery.TIMEOUT_REMAINING:
            expected.append(("budget", f"seed{seed}", cell))
            if (seed, cell) in (recovery.TIMEOUT_REMAINING[0], (43, "sft0")):
                expected.append(("probe", f"seed{seed}", cell))
            expected.append(("native", f"seed{seed}", cell))
        assert events == expected and calls == list(recovery.TIMEOUT_REMAINING)


@pytest.mark.parametrize("damage", ["returncode", "parent_flag", "config", "stdout"])
def test_cached_runtime_proof_tampering_is_fatal_without_new_probe(runtime_probe, damage):
    probe = runtime_probe
    recovery._isolated_runtime_preflight(probe.child, "first", comparison.PROFILES[42])
    evidence = probe.child / "runtime_preflight_probe"
    if damage in {"returncode", "parent_flag"}:
        receipt = runner.restart.read(evidence / "receipt.json")
        receipt["returncode" if damage == "returncode" else "parent_cuda_after"] = (
            7 if damage == "returncode" else True)
        write(evidence / "receipt.json", receipt)
    else:
        (evidence / ("config.json" if damage == "config" else "stdout.txt")).write_text("changed")
    with pytest.raises(RuntimeError, match="result changed"):
        recovery._isolated_runtime_preflight(probe.child, "second", comparison.PROFILES[42])
    assert len(probe.calls) == 1


@pytest.mark.parametrize("damage", [None, "data", "model", "preflight"])
def test_prepare_timeout_copies_four_cells_only_from_matching_science_inputs(
    timeout_history, tmp_path, monkeypatch, damage,
):
    original, snapshot = timeout_history
    before = recovery._inventory(original)
    audit_base = tmp_path / "synthetic-audit"
    timeout_binding(timeout_history, audit_base)
    audit = runner.restart.read(audit_base / "recovery_contract.json")
    audit["status"] = "COMPLETE"
    monkeypatch.setattr(recovery, "audit_timeout", lambda *a, **k: audit)
    base = tmp_path / "prepared-timeout"

    def prepare(target, **kwargs):
        target.mkdir()
        for seed in comparison.SEEDS:
            child, source = target / f"seed{seed}", original / f"seed{seed}"
            shutil.copytree(source / "prepared", child / "prepared")
            for name in ("model_identity.json", "settings.json", "selection.json"):
                shutil.copy2(source / name, child / name)
        if damage == "data":
            (target / "seed43/prepared/dev.parquet").write_bytes(b"changed")
        elif damage == "model":
            write(target / "seed43/model_identity.json", {"model": "different"})

    def seal(target, *, continuation):
        if damage == "preflight":
            write(target / "seed43/input_preflight.json", {"status": "PASSED", "changed": True})
        recovery.verify_bindings(target, {"continuation": continuation})

    monkeypatch.setattr(comparison, "prepare", prepare)
    monkeypatch.setattr(comparison, "seal", seal)
    if damage is not None:
        with pytest.raises(RuntimeError, match="differ|changed"):
            recovery.prepare_timeout(base, original=original,
                                     coordinated_reservation_directory="holder", snapshot=snapshot)
    else:
        result = recovery.prepare_timeout(
            base, original=original, coordinated_reservation_directory="holder", snapshot=snapshot,
        )
        assert (result["imported_trajectories"], result["new_trajectories"]) == (516, 820)
        assert result["continuation_kind"] == recovery.TIMEOUT_KIND
        for seed, cell in recovery.TIMEOUT_PREFIX:
            assert recovery._inventory(base / f"seed{seed}" / cell) == recovery._inventory(
                original / f"seed{seed}" / cell)
        assert all(not (base / f"seed{seed}" / cell).exists()
                   for seed, cell in recovery.TIMEOUT_REMAINING)
    assert recovery._inventory(original) == before


@pytest.fixture
def latest_history(timeout_history, tmp_path):
    ancestor, _ = timeout_history
    base = tmp_path / "latest-v5"
    timeout_binding(timeout_history, base)
    for directory in (base, *(base / f"seed{s}" for s in comparison.SEEDS)):
        source = ancestor if directory == base else ancestor / directory.name
        shutil.copytree(source / "sources", directory / "sources")
        shutil.copy2(source / "source_identity.json", directory / "source_identity.json")
        if directory != base:
            for name in ("run_contract.json", "gpu_coordinator_config.json"):
                shutil.copy2(source / name, directory / name)
    for name in comparison.ROOT_INPUTS:
        if name != "source_identity.json":
            shutil.copy2(ancestor / name, base / name)
    root = runner.restart.read(ancestor / "run_contract.json")
    root["continuation"] = dict(profile=recovery.PROFILE, recovery_contract_sha256=(
        runner.restart.digest(base / "recovery_contract.json")))
    write(base / "run_contract.json", root)
    for index, (seed, cell) in enumerate(recovery.LATEST_PREFIX[len(recovery.TIMEOUT_PREFIX):], 4):
        write(base / f"seed{seed}" / cell / "summary.json", dict(
            status="COMPLETE", returncode=0, cell={"cell_id": cell},
            execution_identity_sha256=str(index) * 64,
        ))
    for seed in comparison.SEEDS:
        cells = [runner.restart.read(base / f"seed{s}" / c / "summary.json")
                 for s, c in recovery.LATEST_PREFIX if s == seed]
        write(base / f"seed{seed}/state.json", dict(
            status="COMPLETE" if seed == 42 else "FAILED", seed=seed, cells=cells,
            error=None if seed == 42 else recovery.TIMEOUT_ERROR,
            current_cell="m2_lr1e7_eta025" if seed == 42 else recovery.LATEST_REMAINING[0][1],
        ))
    write(base / "state.json", dict(status="FAILED", profile=comparison.NAME,
          error=recovery.TIMEOUT_ERROR, current_seed=43, completed_seeds=[42]))
    receipt = runner.restart.read(ancestor / "service_exit_receipt.json")
    receipt["run_contract_sha256"] = runner.restart.digest(base / "run_contract.json")
    write(base / "service_exit_receipt.json", receipt)
    pending = recovery.LATEST_REMAINING[0][1]
    write(base / f"seed43/native_budget-{pending}.json", dict(
        **recovery.GPU_BUDGET, status="TIMEOUT", cell_id=pending, native_started_by_wait=False,
        elapsed_seconds=1800.01, samples=1,
        config_sha256=runner.restart.digest(base / "seed43/gpu_coordinator_config.json"),
    ))
    (base / f"seed43/native_budget-{pending}.jsonl").write_text(
        json.dumps({"ready": False}) + "\n" +
        json.dumps(dict(status="TIMEOUT", failure={"type": "TimeoutError"})) + "\n"
    )
    snapshot, log = tmp_path / "v5_snapshot.json", tmp_path / "v5_pre_edit_check.log"
    log.write_text("actual pre-edit check code0\n")
    write(snapshot, dict(
        profile=recovery.LATEST_SNAPSHOT_PROFILE, status="PASS", original=str(base),
        original_status="FAILED", controller_returncode=1,
        pre_edit_contract_check=dict(returncode=0, log=str(log), command=[
            "python", "-m", "src.agents.action_pooling_comparison", "check",
            "--directory", str(base),
        ]), run_contract_sha256=runner.restart.digest(base / "run_contract.json"),
        source_sha256=runner.restart.read(base / "source_identity.json"),
        completed_cells=[list(r) for r in recovery.LATEST_PREFIX], completed_seeds=[42],
        completed_trajectories=880, remaining_cells=[list(r) for r in recovery.LATEST_REMAINING],
        remaining_trajectories=456, unstarted_pending_cell=pending, pending_cell_trajectory_count=0,
        old_failed_history_preserved=True, automatic_retry=False, files=recovery._inventory(base),
    ))
    return base, snapshot


def latest_binding(history, base):
    original, snapshot = history
    files = {}
    for seed, cell in recovery.LATEST_PREFIX:
        relative = f"seed{seed}/{cell}"
        shutil.copytree(original / relative, base / relative)
        files[relative] = recovery._inventory(base / relative)
    for seed in comparison.SEEDS:
        shutil.copytree(original / f"seed{seed}/prepared", base / f"seed{seed}/prepared")
        for name in ("model_identity.json", "settings.json", "selection.json",
                     "input_preflight.json"):
            shutil.copy2(original / f"seed{seed}" / name, base / f"seed{seed}" / name)
    contract = runner.restart.read(original / "recovery_contract.json")
    contract.update(
        continuation_kind=recovery.LATEST_KIND, original=str(original),
        original_files=recovery._inventory(original),
        source_changes=recovery._timeout_sources(original),
        original_snapshot=recovery._timeout_snapshot(snapshot, original, kind=recovery.LATEST_KIND),
        imported_cells=[list(r) for r in recovery.LATEST_PREFIX], imported_trajectories=880,
        imported_files=files, remaining_cells=[list(r) for r in recovery.LATEST_REMAINING],
        new_trajectories=456, completed_new_trajectories_in_original=364,
        archive_replay_cells=[[42, "m2_lr1e7_eta025"], [43, "m2_lr1e7_eta025"]],
        inherited_archive_audit=dict(
            path=str(original / "recovery_contract.json"),
            sha256=runner.restart.digest(original / "recovery_contract.json")),
    )
    write(base / "recovery_contract.json", contract)
    root = dict(continuation=dict(profile=recovery.PROFILE, recovery_contract_sha256=(
        runner.restart.digest(base / "recovery_contract.json"))))
    write(base / "run_contract.json", root)
    return root


def test_latest_history_validates_frozen_chain_and_canonical_origins(latest_history):
    original, snapshot = latest_history
    before = recovery._inventory(original)
    assert len(recovery._latest_failure(original)) == 7
    assert recovery._timeout_snapshot(snapshot, original, kind=recovery.LATEST_KIND)["sha256"] == (
        runner.restart.digest(snapshot))
    v4 = Path(runner.restart.read(original / "recovery_contract.json")["original"])
    v2 = Path(runner.restart.read(v4 / "recovery_contract.json")["original"])
    for row in recovery.LATEST_PREFIX:
        seed, cell = row
        expected = (v2 if row in recovery.PREFIX else
                    v4 if row in recovery.TIMEOUT_PREFIX else original)
        assert recovery._canonical_cell(original, seed, cell) == expected / f"seed{seed}" / cell
    assert recovery._inventory(original) == before


def test_latest_binding_accepts_only_self_engineering_change(latest_history, tmp_path):
    original, _ = latest_history
    # A fresh implementation may differ from both archived SELF versions without rewriting history.
    (tmp_path / "live" / recovery.SELF).write_bytes(b"latest engineering implementation")
    before = recovery._inventory(original)
    base = tmp_path / "v6"
    root = latest_binding(latest_history, base)
    recovery.verify_bindings(base, root)
    assert recovery._inventory(original) == before


@pytest.mark.parametrize("damage", ["exit0", "seed42_failed", "wrong_completed_seeds",
                                     "pending_dir", "budget_ready", "imported_raw", "old_v4",
                                     "old_v3", "science", "source_changes", "canonical_cycle"])
def test_latest_scope_rejects_history_or_provenance_damage(latest_history, tmp_path, damage):
    original, _ = latest_history
    base = tmp_path / "v6"
    root = latest_binding(latest_history, base)
    if damage == "exit0":
        path = original / "service_exit_receipt.json"
        value = runner.restart.read(path)
        value["controller_returncode"] = 0
        write(path, value)
    elif damage == "seed42_failed":
        path = original / "seed42/state.json"
        value = runner.restart.read(path)
        value["status"] = "FAILED"
        write(path, value)
    elif damage == "wrong_completed_seeds":
        path = original / "state.json"
        value = runner.restart.read(path)
        value["completed_seeds"] = []
        write(path, value)
    elif damage == "pending_dir":
        (original / "seed43" / recovery.LATEST_REMAINING[0][1]).mkdir()
    elif damage == "budget_ready":
        path = original / f"seed43/native_budget-{recovery.LATEST_REMAINING[0][1]}.json"
        value = runner.restart.read(path)
        value["status"] = "READY"
        write(path, value)
    elif damage == "imported_raw":
        (base / "seed42/m2_lr1e7_eta025/new.jsonl").write_text("unbound")
    elif damage in {"old_v4", "old_v3"}:
        target = tmp_path / ("timeout-v4" if damage == "old_v4" else "failed-initialization")
        (target / "changed.log").write_text("changed")
    elif damage == "science":
        (tmp_path / "live/src/science.py").write_bytes(b"changed method")
    else:
        path = original / "recovery_contract.json"
        value = runner.restart.read(path)
        if damage == "source_changes":
            value["source_changes"] = {"forged": {"after_sha256": "wrong"}}
        else:
            value["original"] = str(original)
        write(path, value)
    with pytest.raises((RuntimeError, ValueError)):
        recovery.verify_bindings(base, root)


def test_latest_run_imports_seed42_without_requests_and_executes_three_seed43_cells(
    runnable, monkeypatch,
):
    base, calls, retired, post, stopped = runnable
    for seed, cell in recovery.LATEST_PREFIX[len(recovery.PREFIX):]:
        write(base / f"seed{seed}" / cell / "summary.json", dict(
            cell={"cell_id": cell}, status="COMPLETE", historical=True,
        ))
    write(base / "recovery_contract.json", {"continuation_kind": recovery.LATEST_KIND})
    events = []
    monkeypatch.setattr(recovery, "wait_native_budget",
                        lambda child, cell: events.append(("gate", child.name, cell)))
    monkeypatch.setattr(recovery, "_isolated_runtime_preflight",
                        lambda child, cell, p: events.append(("probe", child.name, cell)))
    before = {(s, c): (base / f"seed{s}" / c / "summary.json").read_bytes()
              for s, c in recovery.LATEST_PREFIX}
    recovery.run(base)
    assert calls == retired == post == list(recovery.LATEST_REMAINING)
    assert events == [(phase, "seed43", cell) for _, cell in recovery.LATEST_REMAINING
                      for phase in ("gate", "probe")]
    state = runner.restart.read(base / "state.json")
    assert (state["imported_trajectories"], state["new_trajectories"]) == (880, 456)
    seed42 = runner.restart.read(base / "seed42/state.json")
    seed43 = runner.restart.read(base / "seed43/state.json")
    assert seed42["status"] == "COMPLETE" and seed42["current_cell"] is None
    assert (seed42["imported_trajectories"], seed42["new_trajectories"]) == (668, 0)
    assert (seed43["imported_trajectories"], seed43["new_trajectories"]) == (212, 456)
    assert all((base / f"seed{s}" / c / "summary.json").read_bytes() == data
               for (s, c), data in before.items())


@pytest.mark.parametrize("fail_at", ["budget", "probe", "native"])
def test_latest_failure_stops_without_repeating_imports_or_following_cells(
    runnable, monkeypatch, fail_at,
):
    base, calls, *_ = runnable
    for seed, cell in recovery.LATEST_PREFIX[len(recovery.PREFIX):]:
        write(base / f"seed{seed}" / cell / "summary.json", dict(
            cell={"cell_id": cell}, status="COMPLETE", historical=True,
        ))
    write(base / "recovery_contract.json", {"continuation_kind": recovery.LATEST_KIND})
    monkeypatch.setattr(recovery, "_isolated_runtime_preflight", lambda *a: None)

    def fail(*args, **kwargs):
        raise RuntimeError("single explicit failed attempt")

    target = dict(budget=(recovery, "wait_native_budget"),
                  probe=(recovery, "_isolated_runtime_preflight"),
                  native=(runner, "run_cell"))[fail_at]
    monkeypatch.setattr(*target, fail)
    with pytest.raises(RuntimeError, match="single explicit"):
        recovery.run(base)
    assert not calls
    assert runner.restart.read(base / "seed42/state.json")["status"] == "COMPLETE"
    assert runner.restart.read(base / "seed43/state.json")["status"] == "FAILED"
    assert not any((base / "seed43" / c).exists() for _, c in recovery.LATEST_REMAINING)


@pytest.mark.parametrize("conflicting_seed43_head", [False, True])
def test_latest_audit_groups_heads_by_seed_and_replays_only_two_new_archives(
    latest_history, monkeypatch, conflicting_seed43_head,
):
    original, snapshot = latest_history
    summaries, heads, replayed = [], {}, []
    for seed, cell_id in recovery.LATEST_PREFIX:
        summary = runner.restart.read(original / f"seed{seed}" / cell_id / "summary.json")
        digest = f"head-seed{seed}"
        if conflicting_seed43_head and seed == 43 and cell_id != "sft0":
            digest += "-changed"
        summary.update(critic_fit={}, placement={}, gpu_handoff={},
                       critic_initial_head_sha256=digest)
        summaries.append(summary)
        canonical = recovery._canonical_cell(original, seed, cell_id)
        spec = next(s for s in runner.cells(comparison.PROFILES[seed]) if s.cell_id == cell_id)
        heads[canonical / "artifacts/critic_initialization/head-rank0.json"] = dict(
            seed=seed, execution_identity_sha256=summary["execution_identity_sha256"], arm=spec.arm,
            mode="cpu_seed_v1", head_sha256=digest, model_path=str(comparison.MODEL),
        )
    read = runner.restart.read
    monkeypatch.setattr(runner.restart, "read",
                        lambda path: heads[path] if path in heads else read(path))
    monkeypatch.setattr(recovery, "_latest_failure", lambda path: summaries)
    monkeypatch.setattr(recovery.readout.shared, "_cell", lambda cell, seed, cid, *a, **k: (
        [None] * (60 if cid == "sft0" else 152), [],
    ))
    monkeypatch.setattr(runner, "validate_extra_evidence", lambda *a, **k: {})
    monkeypatch.setattr(runner.tuning, "validate_fit_evidence", lambda *a, **k: {})
    monkeypatch.setattr(runner, "validate_placement", lambda *a, **k: {})
    monkeypatch.setattr(runner, "validate_handoff", lambda *a, **k: {})
    monkeypatch.setattr(recovery, "_archive_evidence",
                        lambda cell, summary: replayed.append((cell.parent.name, cell.name)))
    if conflicting_seed43_head:
        with pytest.raises(RuntimeError, match="shared critic head"):
            recovery.audit_timeout(original, snapshot=snapshot, kind=recovery.LATEST_KIND)
    else:
        result = recovery.audit_timeout(original, snapshot=snapshot, kind=recovery.LATEST_KIND)
        assert (result["imported_trajectories"], result["new_trajectories"]) == (880, 456)
        assert result["completed_new_trajectories_in_original"] == 364
        assert replayed == [("seed42", "m2_lr1e7_eta025"), ("seed43", "m2_lr1e7_eta025")]
        assert result["archive_replay_cells"] == [[42, "m2_lr1e7_eta025"], [43, "m2_lr1e7_eta025"]]


def test_latest_prepare_preserves_seven_copied_summaries_without_creating_pending_cells(
    latest_history, tmp_path, monkeypatch,
):
    original, snapshot = latest_history
    before = recovery._inventory(original)
    audit_base = tmp_path / "latest-audit"
    latest_binding(latest_history, audit_base)
    audit = runner.restart.read(audit_base / "recovery_contract.json")
    audit["status"] = "COMPLETE"
    monkeypatch.setattr(recovery, "audit_timeout", lambda *a, **k: audit)

    def prepare(target, **kwargs):
        target.mkdir()
        for seed in comparison.SEEDS:
            child, source = target / f"seed{seed}", original / f"seed{seed}"
            shutil.copytree(source / "prepared", child / "prepared")
            for name in ("model_identity.json", "settings.json", "selection.json"):
                shutil.copy2(source / name, child / name)

    monkeypatch.setattr(comparison, "prepare", prepare)
    monkeypatch.setattr(comparison, "seal", lambda base, continuation: recovery.verify_bindings(
        base, {"continuation": continuation},
    ))
    base = tmp_path / "prepared-latest"
    result = recovery.prepare_timeout(base, original=original, snapshot=snapshot,
                                      coordinated_reservation_directory="holder",
                                      kind=recovery.LATEST_KIND)
    assert (result["imported_trajectories"], result["new_trajectories"]) == (880, 456)
    assert all((base / f"seed{s}" / c / "summary.json").read_bytes() == (
        original / f"seed{s}" / c / "summary.json").read_bytes() for s, c in recovery.LATEST_PREFIX)
    assert all(not (base / f"seed{s}" / c).exists() for s, c in recovery.LATEST_REMAINING)
    assert recovery._inventory(original) == before
