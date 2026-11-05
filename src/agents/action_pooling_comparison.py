"""Fixed four-round, two-seed pooling ablation from the accepted recovery SFT.

Every arm is fresh. Historical R5 outcomes are descriptive inputs, never cells
in this comparison. The original multi-round disk and GPU limits still apply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.agents import learning_restart as restart
from src.agents import low_memory_pilot as runner
from src.agents import order_recovery_learning as recovery
from src.agents.screening_launcher import atomic_json

NAME = "shared_recovery_sft_action_pooling_comparison_v1"
PLAN = "docs/2026-10-12_M2_ACTION_POOLING_COMPARISON_PLAN.md"
DEFAULT = runner.REPO_ROOT / "runs/m2-action-pooling-comparison-20261012-v2"
STAGE1 = runner.REPO_ROOT / "runs/m2-order-recovery-acceptance-all-20261011-v2"
MODEL = runner.REPO_ROOT / "runs/m2-order-recovery-sft-20261011-v1/merged_model"
SFT_GATE = runner.REPO_ROOT / "runs/m2-order-recovery-sft-acceptance-20261011-v1/readout.json"
MERGE_RECEIPT_SHA256 = "faa98d2cd6d871d937505a9d52573658c25d68b328796df0fb5126760b8f9574"
SEEDS = (42, 43)
TRAIN_NUMBERS = (1, 65, 9, 73, 17, 149, 25, 157)
EVAL_NUMBERS = (210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 441)
ROOT_INPUTS = ("comparison_settings.json", "shared_sft.json", "stage1_completion.json",
               "source_identity.json", "plan.md")
RECLAIM_SECONDS = 300


def _profile(seed):
    rl = (
        runner.CellSpec("m1_lr1e7", "M1", 0.0, 1e-6, 4),
        runner.CellSpec("m1_lr2e7", "M1", 0.0, 1e-6, 4),
        runner.CellSpec("m2_plain_lr1e7_eta025", "M2-plain", 0.25, 1e-6, 4),
        runner.CellSpec("m2_lr1e7_eta025", "M2", 0.25, 1e-6, 4),
    )
    return runner.ActionPoolingExperimentProfile(
        name=f"shared_recovery_sft_action_pooling_seed{seed}_v1", plan=PLAN,
        train_numbers=TRAIN_NUMBERS, eval_numbers=EVAL_NUMBERS,
        evaluation_steps=(2, 4), rounds=4, include_sft=True,
        retire_completed_moments=True, archive_completed_actors=True,
        evaluation_incomplete_mode="retain_prefix_v1", seed=seed,
        actor_learning_rate=1e-7,
        cell_specs=(runner.CellSpec("sft0", "M1", 0.0, 1e-6, 0),)
        + (rl if seed == 42 else tuple(reversed(rl))),
        cell_actor_learning_rates=(("m1_lr2e7", 2e-7),),
    )


PROFILES = {seed: _profile(seed) for seed in SEEDS}


def settings():
    return dict(
        profile=NAME, plan=PLAN, seeds=list(SEEDS),
        profiles={str(seed): asdict(p) for seed, p in PROFILES.items()},
        planned_trajectories=1336, training_trajectories=256, evaluation_trajectories=1080,
        rl_cells=8, sft_baselines=2, rounds=4,
        training_updates_per_role_per_cell=4,
        training_episode_ids=PROFILES[42].train_ids,
        evaluation_episode_ids=PROFILES[42].eval_ids,
        evaluation_steps=[2, 4], evaluation_lanes={"greedy": [0], "sampled": [1, 2, 3, 4]},
        observation_profile=runner.RECOVERY_OBSERVATION_PROFILE,
        common_invalid_action_penalty=runner.PENALTY,
        critic_learning_rate=1e-6, m2_actor_learning_rate=1e-7, residual_eta=0.25,
        plain_definition="same_session_same_disagreement_gate_individual_Y_minus_group_mean_v1",
        pooled_definition="same_session_action_family_mean_Y_minus_group_mean_v1",
        common_actor_advantage="global_real_step_whiten_GAE_plus_eta_residual_v1",
        critic_residual=False, standard_grpo_claim=False,
        native_interaction_cap=256, native_session_recovery_cap=None,
        prompt_cap=8192, response_cap=4096, enable_thinking=False,
        selection_scope="original_train_development_only", untouched_holdout=False,
        formal_valid_opened=False, final_oos_opened=False,
        execution="serial_fresh_one_shot_v1", automatic_retry=False, resume_supported=False,
        automatic_actions=False, actor_retention="verified_lossless_tensor_xor_v1",
        minimum_free_disk_bytes=107 * 1024**3,
        disk_reclaim_wait_seconds=RECLAIM_SECONDS,
        disk_budget_profile="multi_round_three_generation_adam_v1",
        gpu_monitor_profile="aggregate_80GiB_after_verified_holder_release_v1",
        aggregate_gpu_stop_bytes=80 * 1024**3,
        holder_release_wait_seconds=1800,
        interpretation="paired_development_ablation_not_confirmatory_method_advantage",
    )


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def child_directory(base, seed):
    if type(seed) is not int or seed not in PROFILES:
        raise ValueError("pooling comparison seed must be 42 or 43")
    return base / f"seed{seed}"


def _stage1_proof():
    api = recovery.acceptance
    receipt = restart.read(STAGE1 / "completion_runtime_receipt.json")
    manifest = api.verify_completed(STAGE1, receipt.get("manifest_sha256"), check_models=False)
    audit = api.raw_audit_gate(STAGE1, manifest, completed_runtime=True)
    readout = restart.read(STAGE1 / "readout.json")
    exit_receipt = restart.read(STAGE1 / "service_exit_receipt.json")
    if (receipt.get("status") != "PASS" or receipt.get("acceptance") != "PASS"
            or receipt.get("controller_status") != "COMPLETE"
            or receipt.get("controller_returncode") != 0
            or receipt.get("numeric_exit_status_recorded") is not True
            or receipt.get("trajectories") != 1200 or receipt.get("requests") != 6579
            or receipt.get("holder_returned_76GiB_and_unclaimed") is not True
            or receipt.get("raw_audit_sha256") != audit.get("sha256")
            or receipt.get("readout_sha256") != restart.digest(STAGE1 / "readout.json")
            or receipt.get("service_exit_receipt_sha256")
            != restart.digest(STAGE1 / "service_exit_receipt.json")
            or audit.get("status") != "PASS" or audit.get("acceptance_pass") is not True
            or readout.get("status") != "OBSERVED" or readout.get("acceptance") != "PASS"
            or readout.get("behavioral_gate") != "PASS"
            or exit_receipt.get("controller_returncode") != 0
            or exit_receipt.get("successful_main_exit") is not True):
        raise RuntimeError("stage2 requires the retained actual complete stage1 acceptance")
    files = ("manifest.json", "readout.json", "raw_audit.json", "runtime_sources.zip",
             "completion_runtime_receipt.json", "service_exit_receipt.json")
    return dict(status="PASS", directory=str(STAGE1),
                common_sft_fingerprint=manifest["models"]["sft"]["fingerprint"],
                source_sha256={str(STAGE1 / p): restart.digest(STAGE1 / p) for p in files})


def prepare(base, *, coordinated_reservation_directory):
    if base.exists():
        raise RuntimeError("pooling comparison requires a fresh root directory")
    proof = _stage1_proof()
    lineage = recovery._validate_sft_inputs(MODEL, MODEL.parent / "lineage_receipt.json", SFT_GATE)
    if proof["common_sft_fingerprint"] != lineage["model_fingerprint"]:
        raise RuntimeError("pooling initialization differs from completed recovery acceptance")
    if restart.digest(MODEL / "merge_receipt.json") != MERGE_RECEIPT_SHA256:
        raise RuntimeError("pooling comparison requires the accepted new shared SFT")
    base.mkdir(parents=True, exist_ok=False)
    atomic_json(base / "comparison_settings.json", settings())
    atomic_json(base / "stage1_completion.json", proof)
    atomic_json(base / "shared_sft.json", dict(
        model_snapshot_path=str(MODEL), merge_receipt_sha256=MERGE_RECEIPT_SHA256,
        recovery_acceptance=lineage,
    ))
    token = uuid4().hex
    for seed, profile in PROFILES.items():
        runner.prepare(child_directory(base, seed), model_path=MODEL,
                       merge_receipt_sha256=MERGE_RECEIPT_SHA256, profile=profile,
                       coordinated_reservation_directory=coordinated_reservation_directory,
                       coordinator_token=token)
    refresh_status(base)


async def input_preflight(base):
    _check_settings(base)
    for seed, profile in PROFILES.items():
        await runner.input_preflight(child_directory(base, seed), profile=profile)


def _check_settings(base):
    if not _same(restart.read(base / "comparison_settings.json"), settings()):
        raise RuntimeError("pooling comparison settings differ from the fixed plan")


def seal(base, *, continuation=None):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("pooling comparison already sealed or started")
    _check_settings(base)
    for seed, profile in PROFILES.items():
        runner.seal(child_directory(base, seed), profile=profile)
    shutil.copy2(runner.REPO_ROOT / PLAN, base / "plan.md")
    sources = {}
    for relative, path in runner.source_files().items():
        target = base / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative] = restart.digest(path)
    atomic_json(base / "source_identity.json", sources)
    contract = dict(
        profile=NAME, sealed_at_SGT=restart.now(),
        inputs={p: restart.digest(base / p) for p in ROOT_INPUTS},
        child_contracts={str(s): restart.digest(child_directory(base, s) / "run_contract.json")
                         for s in SEEDS},
        planned_trajectories=1336, automatic_retry=False, resume_supported=False,
    )
    if continuation is not None:
        contract["continuation"] = continuation
    atomic_json(base / "run_contract.json", contract)
    check_contract(base)
    refresh_status(base)


def check_contract(base):
    _check_settings(base)
    contract = restart.read(base / "run_contract.json")
    if (contract.get("profile") != NAME or set(contract.get("inputs", {})) != set(ROOT_INPUTS)
            or set(contract.get("child_contracts", {})) != {str(s) for s in SEEDS}
            or contract.get("planned_trajectories") != 1336
            or contract.get("automatic_retry") is not False
            or contract.get("resume_supported") is not False):
        raise RuntimeError("pooling root identity or budget differs")
    for relative, expected in contract["inputs"].items():
        if restart.digest(base / relative) != expected:
            raise RuntimeError("sealed pooling input changed: " + relative)
    sources = restart.read(base / "source_identity.json")
    current = runner.source_files()
    if set(sources) != set(current):
        raise RuntimeError("pooling runtime source inventory changed")
    for relative, path in current.items():
        if (restart.digest(path) != sources[relative]
                or restart.digest(base / "sources" / relative) != sources[relative]):
            raise RuntimeError("pooling runtime source changed: " + relative)
    proof = restart.read(base / "stage1_completion.json")
    for filename, expected in proof["source_sha256"].items():
        if restart.digest(Path(filename)) != expected:
            raise RuntimeError("historical recovery acceptance changed: " + filename)
    for seed, profile in PROFILES.items():
        child = child_directory(base, seed)
        if child.is_symlink() or restart.digest(child / "run_contract.json") != (
                contract["child_contracts"][str(seed)]):
            raise RuntimeError("pooling sealed child changed")
        runner.check_contract(child, profile=profile)
    if "continuation" in contract:
        from src.agents.action_pooling_recovery import verify_bindings

        verify_bindings(base, contract)


def refresh_status(base):
    state = restart.read(base / "state.json") if (base / "state.json").exists() else {}
    rows = []
    for seed, profile in PROFILES.items():
        for spec in runner.cells(profile):
            cell = child_directory(base, seed) / spec.cell_id
            summary_path = cell / "summary.json"
            summary = restart.read(summary_path) if summary_path.exists() else {}
            count = 0
            for path in (cell / "artifacts/screening_evidence").glob("*.jsonl"):
                with path.open("rb") as stream:
                    count += sum(bool(line.strip()) for line in stream)
            rows.append(dict(seed=seed, cell_id=spec.cell_id, arm=spec.arm,
                             actor_lr=profile.actor_lr(spec.cell_id),
                             status=summary.get("status", "PENDING"),
                             landed_trajectories=count, planned=60 if spec.zero else 152))
    result = dict(profile=NAME, at_SGT=restart.now(), status=state.get("status", "PREPARED"),
                  current_seed=state.get("current_seed"), cells=rows, planned_trajectories=1336,
                  landed_trajectories=sum(r["landed_trajectories"] for r in rows),
                  automatic_retry=False, resume_supported=False)
    atomic_json(base / "live_status.json", result)
    lines = ["# M2 动作池化对照", "", f"更新：{result['at_SGT']}",
             f"执行状态：{result['status']}；已返回 {result['landed_trajectories']}/1336 条。",
             "", "| Seed | 臂 | 状态 | 返回/计划 |", "|---|---|---|---:|"]
    lines.extend(f"| {r['seed']} | {r['cell_id']} | {r['status']} | "
                 f"{r['landed_trajectories']}/{r['planned']} |" for r in rows)
    lines += ["", "固定两seed、四轮、新共同SFT及v2观察。Train开发消融，非正式Valid/OOS。",
              "未完成收益为空，不自动重试；方法判断见最终配对读数。", ""]
    (base / "LIVE_STATUS.md").write_text("\n".join(lines))
    return result


def _readout(base, *, allow_partial):
    from src.agents.action_pooling_readout import write_readout

    return write_readout(base, allow_partial=allow_partial)


def wait_archived_disk(seed_base, spec, summary):
    """Wait on actual df only after this cell's verified owned retirements."""
    archive = restart.read(seed_base / spec.cell_id / "actor_archive_receipt.json")
    moment = restart.read(seed_base / spec.cell_id / "moment_pruning_receipt.json")
    if (archive.get("status") != "COMPLETE" or moment.get("status") != "COMPLETE"
            or archive.get("execution_identity_sha256") != summary["execution_identity_sha256"]
            or moment.get("execution_identity_sha256") != summary["execution_identity_sha256"]):
        raise RuntimeError("disk waiting requires verified owned completed-cell retirement")
    required = runner.disk_requirement_bytes(PROFILES[42], 1)
    path = seed_base / spec.cell_id / "disk_reclaim_receipt.json"
    if path.exists():
        raise RuntimeError("disk reclaim wait cannot be retried")
    started, samples = time.monotonic(), []
    while True:
        free = shutil.disk_usage(seed_base).free
        elapsed = time.monotonic() - started
        samples.append(dict(elapsed_seconds=elapsed, actual_free_bytes=free))
        ready = free >= required
        timed_out = elapsed >= RECLAIM_SECONDS
        atomic_json(path, dict(status="READY" if ready else "TIMEOUT" if timed_out else "WAITING",
                               required_free_bytes=required, maximum_seconds=RECLAIM_SECONDS,
                               samples=samples, automatic_retry=False,
                               counts_deleted_bytes_as_free=False))
        if ready:
            return
        if timed_out:
            raise RuntimeError("actual multi-round disk capacity did not return within 300 seconds")
        time.sleep(min(5, RECLAIM_SECONDS - elapsed))


def _validate_partial_readout(base, current_seed, current_cell, result):
    """Permit only actual pending work; completed evidence errors stop the run."""
    if current_seed not in SEEDS:
        raise RuntimeError("partial readout has an unknown current seed")
    completed, expected_issues = set(), []
    for seed, profile in PROFILES.items():
        path = child_directory(base, seed) / "state.json"
        try:
            state = restart.read(path)
        except FileNotFoundError as error:
            if SEEDS.index(seed) <= SEEDS.index(current_seed):
                raise RuntimeError(
                    "partial readout is missing a started seed controller"
                ) from error
            expected_issues.append(dict(seed=seed, reason=str(error)))
            continue
        if SEEDS.index(seed) > SEEDS.index(current_seed):
            raise RuntimeError("partial readout found an unexpectedly started future seed")
        if state.get("status") not in {
            "RUNNING", "COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES",
        }:
            raise RuntimeError("partial readout cannot continue a failed/unknown seed controller")
        if state["status"] == "RUNNING":
            if seed != current_seed:
                raise RuntimeError("partial readout found a prior seed still RUNNING")
            expected_issues.append(dict(seed=seed, reason="seed controller is not COMPLETE"))
        by_id = {}
        for row in state.get("cells", []):
            cell_id = row.get("cell", {}).get("cell_id")
            if cell_id not in {spec.cell_id for spec in runner.cells(profile)} or cell_id in by_id:
                raise RuntimeError("partial readout has duplicate/unknown controller cells")
            by_id[cell_id] = row
        for spec in runner.cells(profile):
            row = by_id.get(spec.cell_id)
            status = row.get("status") if row is not None else None
            if status in {"COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES"}:
                completed.add((seed, spec.cell_id))
            elif row is None or status in {"PENDING", "RUNNING"}:
                if state["status"] != "RUNNING":
                    raise RuntimeError("completed seed controller still has unfinished cells")
                expected_issues.append(dict(
                    seed=seed, cell_id=spec.cell_id, reason="cell is not COMPLETE",
                ))
            else:
                raise RuntimeError("partial readout cannot ignore a failed/unknown cell")
    validated = [(row.get("seed"), row.get("cell_id")) for row in result.get("cells", [])]
    if (len(validated) != len(set(validated)) or set(validated) != completed
            or (current_seed, current_cell) not in completed):
        raise RuntimeError("partial readout did not validate every completed cell")
    expected = sorted(json.dumps(issue, sort_keys=True) for issue in expected_issues)
    actual = sorted(json.dumps(issue, sort_keys=True) for issue in result.get("issues", []))
    if actual != expected:
        raise RuntimeError("partial readout contains a completed-evidence or unexpected issue")


def after_cell(seed_base, spec, summary):
    if not spec.zero:
        from src.agents.parameter_grid_archive import archive_completed_cell

        summary["actor_archive"] = archive_completed_cell(seed_base, spec, summary)
        atomic_json(seed_base / spec.cell_id / "summary.json", summary)
        # The strict reader compares the persisted child summary to state.
        state = restart.read(seed_base / "state.json")
        if state["cells"][-1]["execution_identity_sha256"] != summary["execution_identity_sha256"]:
            raise RuntimeError("completed-cell identity changed before publishing archive")
        state["cells"][-1] = summary
        atomic_json(seed_base / "state.json", state)
    result = _readout(seed_base.parent, allow_partial=True)
    _validate_partial_readout(seed_base.parent, summary["seed"], spec.cell_id, result)
    if not spec.zero and not (seed_base.name == "seed43" and spec.cell_id == "m1_lr1e7"):
        wait_archived_disk(seed_base, spec, summary)


def _monitor(base, stop):
    while not stop.is_set():
        try:
            refresh_status(base)
        except Exception as error:
            atomic_json(base / "monitor_error.json", dict(at_SGT=restart.now(), error=str(error)))
        stop.wait(30)


def run(base):
    contract_path = base / "run_contract.json"
    if contract_path.exists() and "continuation" in restart.read(contract_path):
        raise RuntimeError("explicit continuation requires action_pooling_recovery.run")
    if (base / "state.json").exists():
        raise RuntimeError("pooling run requires new state; no resume or retry")
    check_contract(base)
    if shutil.disk_usage(base).free < settings()["minimum_free_disk_bytes"]:
        raise RuntimeError("four-round training requires at least 107 GiB actual free disk")
    for seed in SEEDS:
        if runner.memory_coordinator_config(child_directory(base, seed)) is None:
            raise RuntimeError("pooling training requires its sealed persistent holder contract")
    state = dict(profile=NAME, status="RUNNING", started_at_SGT=restart.now(),
                 current_seed=None, completed_seeds=[], planned_trajectories=1336,
                 automatic_retry=False, resume_supported=False)
    atomic_json(base / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(base, stop), daemon=True)
    monitor.start()
    try:
        for seed, profile in PROFILES.items():
            check_contract(base)
            state["current_seed"] = seed
            atomic_json(base / "state.json", state)
            runner.run(child_directory(base, seed), profile=profile, post_cell=after_cell)
            if restart.read(child_directory(base, seed) / "state.json").get("status") != "COMPLETE":
                raise RuntimeError("pooling seed did not finish its fixed budget")
            state["completed_seeds"].append(seed)
            atomic_json(base / "state.json", state)
        result = _readout(base, allow_partial=False)
        if result.get("execution_status") not in ("COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES"):
            raise RuntimeError("pooling fixed population did not validate")
        state.update(status=result["execution_status"], financial_status=result["financial_status"],
                     current_seed=None)
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        refresh_status(base)
        atomic_json(base / "service_completion.json", dict(
            profile=NAME, status=state["status"], completed_seeds=state["completed_seeds"],
            planned_trajectories=1336, automatic_retry=False, resume_supported=False,
            method_advantage_claimed=False,
            financial_status=state.get("financial_status", "PARTIAL"),
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--coordinated-reservation-directory", type=Path)
    args = parser.parse_args()
    base = args.directory.resolve()
    if args.command == "prepare":
        holder = args.coordinated_reservation_directory or runner.current_reservation_directory()
        prepare(base, coordinated_reservation_directory=holder)
    elif args.command == "preflight":
        asyncio.run(input_preflight(base))
    elif args.command == "seal":
        seal(base)
    elif args.command == "check":
        check_contract(base)
    else:
        run(base)
    print(json.dumps(dict(command=args.command, status="PASS", profile=NAME)))


if __name__ == "__main__":
    main()
