#!/usr/bin/env python3
"""Revalidate the seed43 prefix, preserve its failure, run only the unstarted M2."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import low_memory_pilot as runner  # noqa: E402
from src.agents.actor_export import validate_export_artifact  # noqa: E402
from src.agents.low_memory_comparison_seed43 import PROFILE  # noqa: E402

ORIGINAL = ROOT / "runs/m2-sft-four-round-comparison-seed43-20261005"
DEFAULT = ROOT / "runs/m2-sft-four-round-comparison-seed43-20261005-recovery"
AUDIT = ROOT / "runs/m2-seed43-revalidation-20261005"
ERROR = "RuntimeError: common penalty evidence target or complete-step count differs"
SELF = "scripts/diagnostics/m2_seed43_validation_recovery.py"


def verify_original_failure(original):
    state = runner.restart.read(original / "state.json")
    m1 = runner.restart.read(original / "m1/summary.json")
    if (state.get("status") != "FAILED" or state.get("seed") != 43
            or state.get("current_cell") != "m1" or state.get("error") != ERROR
            or m1.get("status") != "FAILED" or m1.get("error") != ERROR
            or m1.get("returncode") != 0 or m1.get("seed") != 43
            or (original / "m2_eta025").exists()):
        raise RuntimeError("only the known post-training seed43 validator failure is eligible")
    return state


def verify_original_contract(original):
    """Accept precisely the recorded seed-check repair, with no algorithm edits."""
    seal = runner.restart.read(original / "run_contract.json")
    if seal["profile"] != PROFILE.name or set(seal["inputs"]) != set(runner.INPUTS):
        raise RuntimeError("original seed43 input inventory differs")
    for name, digest in seal["inputs"].items():
        if runner.restart.digest(original / name) != digest:
            raise RuntimeError("original sealed input changed: " + name)
    sources = runner.restart.read(original / "source_identity.json")
    current = runner.source_files()
    if set(current) - set(sources) not in (set(), {SELF}) or set(sources) - set(current):
        raise RuntimeError("unreviewed source inventory changes")
    repairs = {}
    for name, digest in sources.items():
        frozen = original / "sources" / name
        if runner.restart.digest(frozen) != digest:
            raise RuntimeError("original frozen source changed: " + name)
        expected = frozen.read_text()
        if name == "src/agents/common_validity_penalty.py":
            expected = expected.replace(
                "def validate_common_penalty_evidence(cell, spec):",
                "def validate_common_penalty_evidence(cell, spec, *, expected_seed=42):",
                1,
            ).replace(
                '    """Confirm the saved training tensors implement the new shared target."""\n',
                '    """Confirm the saved training tensors implement the new shared target."""\n'
                '    if type(expected_seed) is not int or not 0 <= expected_seed < 2**32:\n'
                '        raise ValueError("expected penalty-evidence seed '
                'must be an integer in [0, 2**32)")\n',
                1,
            ).replace('or record.get("run_seed") != 42',
                      'or record.get("run_seed") != expected_seed', 1)
        elif name == "src/agents/low_memory_pilot.py":
            expected = expected.replace(
                "penalty = common.validate_common_penalty_evidence(cell, spec)",
                "penalty = common.validate_common_penalty_evidence("
                "cell, spec, expected_seed=profile.seed)",
                1,
            )
        if current[name].read_text() != expected:
            raise RuntimeError("source changes exceed the seed validator fix: " + name)
        after = runner.restart.digest(current[name])
        if after != digest:
            repairs[name] = {"before_sha256": digest, "after_sha256": after}
    if set(repairs) != {"src/agents/common_validity_penalty.py", "src/agents/low_memory_pilot.py"}:
        raise RuntimeError("the two required seed validator repairs are not present")
    for name, digest in runner.restart.read(original / "selection.json")["source_sha256"].items():
        if runner.restart.digest(Path(name)) != digest:
            raise RuntimeError("original selection source changed")
    manifest = runner.restart.read(original / "prepared/manifest.json")
    for row in [*manifest["datasets"].values(), manifest["calibration"]]:
        if runner.restart.digest(original / "prepared" / row["path"]) != row["sha256"]:
            raise RuntimeError("original prepared data changed")
    runner.read_settings(original, profile=PROFILE)
    return {"status": "PASSED", "repairs": repairs, "frozen_source_files": len(sources),
            "original_inputs": seal["inputs"]}


def revalidate_cell(original, spec):
    cell = original / spec.cell_id
    old = runner.restart.read(cell / "summary.json")
    config = runner.restart.read(cell / "resolved_config.json")
    cfg = runner.restart.read(cell / "settings.json")
    manifest = runner.restart.read(original / "prepared/manifest.json")
    condition = runner.cell_settings(runner.read_settings(original, profile=PROFILE),
                                     spec.cell_id, profile=PROFILE)
    if (cfg != condition or config["run"]["seed"] != 43 or old.get("returncode") != 0
            or old["cell"] != asdict(spec) or config["screening"]["arm"] != spec.arm
            or config["screening"]["residual_eta"] != spec.eta):
        raise RuntimeError("original completed cell conditions differ")
    identity = old["execution_identity_sha256"]
    steps = [0] if spec.zero else list(PROFILE.evaluation_steps)
    result = copy.deepcopy(old)
    result["evidence"] = runner.validate_cell_evidence(
        cell, stage="zero-shot" if spec.zero else "screening", expected_rounds=max(1, spec.rounds),
        arm=spec.arm, seed=43, train_episode_ids=PROFILE.train_ids,
        eval_episode_ids=runner.EVAL_IDS, execution_identity_sha256=identity,
        calibration_payload_sha256=manifest["calibration"]["payload_sha256"],
        expected_eval_steps=steps, expected_eta=spec.eta,
        artifact_profile=cfg["artifact_profile"], eval_rollout_n=5,
    )
    result["critic_fit"] = runner.tuning.validate_fit_evidence(
        cell, spec=spec, execution_identity=identity,
    )
    result.update(runner.validate_extra_evidence(cell, spec, identity, profile=PROFILE))
    result["placement"] = runner.validate_placement(cell, spec, identity)
    result["gpu_handoff"] = runner.validate_handoff(cell, identity)
    head = runner.restart.read(cell / "artifacts/critic_initialization/head-rank0.json")
    model = runner.restart.read(original / "model_identity.json")
    if (head["seed"] != 43 or head["execution_identity_sha256"] != identity
            or head["mode"] != "cpu_seed_v1" or head["arm"] != spec.arm
            or head["model_path"] != model["model_snapshot_path"]
            or head["phase"] != "critic_head_after_hf_load_before_fsdp"):
        raise RuntimeError("original actual critic initialization differs")
    result["critic_initial_head_sha256"] = head["head_sha256"]
    if not spec.zero:
        validate_export_artifact(cell / "checkpoints/global_step_4", identity_sha256=identity,
                                 verify_hashes=True)
    result["evaluation_by_step"] = {
        str(step): runner.evaluation_readout(cell, policy_step=step, profile=PROFILE)
        for step in steps
    }
    result["evaluation"] = result["evaluation_by_step"][str(steps[-1])]
    result.pop("error", None)
    result.update(
        status="COMPLETE", actual_rounds=spec.rounds, revalidated_at_SGT=runner.restart.now(),
        imported_from=str(cell),
        original_summary_sha256=runner.restart.digest(cell / "summary.json"),
        original_controller_status=old["status"], retrained=False,
    )
    return result


def audit_original(original, audit):
    verify_original_failure(original)
    contract = verify_original_contract(original)
    summaries = [revalidate_cell(original, spec) for spec in runner.cells(PROFILE)[:2]]
    if len({row["critic_initial_head_sha256"] for row in summaries}) != 1:
        raise RuntimeError("the original completed cells do not share the critic head")
    audit.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "PASSED", "at_SGT": runner.restart.now(), "seed": 43,
               "original": str(original), "contract": contract, "cells": summaries,
               "completed_trajectories": 92, "remaining_trajectories": 72,
               "failure_records": {name: runner.restart.digest(original / name)
                                   for name in ["state.json", "service_completion.json",
                                                "m1/summary.json", "FINAL_RESULTS.md"]}}
    runner.atomic_json(audit / "revalidation.json", receipt)
    return receipt


def copy_cell(source, destination):
    """Copy evidence; share immutable weight payloads without adding disk usage."""
    if any(path.is_symlink() for path in source.rglob("*")):
        raise RuntimeError("original cell contains redirected files")

    def copy_file(src, dst):
        if str(src).endswith(".safetensors"):
            os.link(src, dst)
            return dst
        return shutil.copy2(src, dst)

    shutil.copytree(
        source, destination, copy_function=copy_file,
        ignore=lambda path, names: ({"moment_scratch"}
                                   if Path(path).name == "placement" else set()),
    )


def prepare_remaining(original, base, audit, reservation_directory):
    verify_original_failure(original)
    verify_original_contract(original)
    receipt = runner.restart.read(audit / "revalidation.json")
    tests = runner.restart.read(audit / "implementation_validation.json")
    if (receipt["status"] != "PASSED" or receipt["original"] != str(original)
            or tests.get("status") != "PASSED"):
        raise RuntimeError("read-only revalidation and fix tests must pass")
    for name, digest in receipt["failure_records"].items():
        if runner.restart.digest(original / name) != digest:
            raise RuntimeError("original failure record changed")
    model = runner.restart.read(original / "model_identity.json")
    runner.prepare(base, model_path=Path(model["model_snapshot_path"]),
                   merge_receipt_sha256=model["merge_receipt_sha256"],
                   reservation_directory=reservation_directory, profile=PROFILE)
    for name in ["train.parquet", "dev.parquet", "reward_calibration.json"]:
        if (runner.restart.digest(base / "prepared" / name)
                != runner.restart.digest(original / "prepared" / name)):
            raise RuntimeError("recovery prepared data differs")
    shutil.copy2(original / "input_preflight.json", base / "input_preflight.json")
    shutil.copy2(audit / "revalidation.json", base / "revalidation.json")
    shutil.copy2(audit / "implementation_validation.json", base / "implementation_validation.json")
    for summary in receipt["cells"]:
        name = summary["cell"]["cell_id"]
        copy_cell(original / name, base / name)
        runner.atomic_json(base / name / "summary.json", summary)
    runner.seal(base, profile=PROFILE)
    spec = runner.cells(PROFILE)[1]
    source = original / "m1/artifacts/placement/moment_scratch"
    target = base / "m1/artifacts/placement/moment_scratch"
    if not source.is_dir() or source.is_symlink() or target.exists():
        raise RuntimeError("original owned temporary moment directory differs")
    movement = {"status": "PREPARED", "at_SGT": runner.restart.now(),
                "source": str(source), "destination": str(target),
                "identity": receipt["cells"][1]["execution_identity_sha256"],
                "reason": "retire fully revalidated completed M1; no training resume"}
    runner.atomic_json(base / "moment_relocation.json", movement)
    source.rename(target)
    movement.update(status="MOVED", at_SGT=runner.restart.now())
    runner.atomic_json(base / "moment_relocation.json", movement)
    summary = runner.restart.read(base / "m1/summary.json")
    summary["moment_pruning"] = runner.retire_completed_moments(
        base, spec, summary, profile=PROFILE,
    )
    runner.atomic_json(base / "m1/summary.json", summary)
    runner.atomic_json(base / "recovery_contract.json", {
        "status": "SEALED", "at_SGT": runner.restart.now(), "original": str(original),
        "only_new_cell": "m2_eta025", "new_trajectory_budget": 72,
        "complete_comparison_budget": 164, "no_retraining_of_prefix": True,
        "inputs": {name: runner.restart.digest(base / name) for name in [
            "revalidation.json", "implementation_validation.json", "sft0/summary.json",
            "m1/summary.json", "moment_relocation.json", "m1/moment_pruning_receipt.json"]},
    })
    print(json.dumps({"status": "PREPARED", "directory": str(base), "new_trajectories": 72}))


def run_remaining(base):
    if (base / "state.json").exists() or (base / "m2_eta025").exists():
        raise RuntimeError("one-shot continuation already attempted")
    runner.check_contract(base, profile=PROFILE)
    recovery = runner.restart.read(base / "recovery_contract.json")
    if (recovery["status"] != "SEALED" or recovery["only_new_cell"] != "m2_eta025"
            or recovery["new_trajectory_budget"] != 72):
        raise RuntimeError("recovery budget differs")
    for name, digest in recovery["inputs"].items():
        if runner.restart.digest(base / name) != digest:
            raise RuntimeError("recovery import changed: " + name)
    original = Path(recovery["original"])
    verify_original_failure(original)
    receipt = runner.restart.read(base / "revalidation.json")
    for name, digest in receipt["failure_records"].items():
        if runner.restart.digest(original / name) != digest:
            raise RuntimeError("original failure history changed")
    state = dict(status="RUNNING", seed=43, profile=PROFILE.name,
                 started_at_SGT=runner.restart.now(), current_cell="m2_eta025",
                 planned_trajectories=164, automatic_retry=False, resume_supported=False,
                 cells=[runner.restart.read(base / name / "summary.json")
                        for name in ["sft0", "m1"]],
                 recovery=recovery, evaluation_steps=[2, 4], train_episode_ids=PROFILE.train_ids,
                 evaluation_episode_ids=runner.EVAL_IDS)
    runner.atomic_json(base / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(
        target=runner.monitor_status, args=(base, stop, PROFILE), daemon=True,
    )
    monitor.start()
    try:
        if shutil.disk_usage(base).free < 107 * 1024**3:
            raise RuntimeError("insufficient disk for the remaining M2 cell")
        summary = runner.run_cell(base, "m2_eta025", profile=PROFILE)
        state["cells"].append(summary)
        runner.atomic_json(base / "state.json", state)
        summary["moment_pruning"] = runner.retire_completed_moments(
            base, runner.cells(PROFILE)[2], summary, profile=PROFILE,
        )
        runner.atomic_json(base / "m2_eta025/summary.json", summary)
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = runner.restart.now()
        runner.atomic_json(base / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        runner.stop_reservation(base)
        runner.refresh_status(base, profile=PROFILE)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnostics/m2_low_memory_comparison_readout.py"),
         "--directory", str(base), "--seed", "43"], cwd=ROOT, check=True,
    )
    runner.atomic_json(base / "service_completion.json", {
        "status": "COMPLETE", "at_SGT": runner.restart.now(), "seed": 43,
        "training_controller": "COMPLETE", "comparison_readout": "PASS",
        "original_failure_preserved": True, "new_trajectories": 72,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "prepare", "run"])
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--audit-directory", type=Path, default=AUDIT)
    parser.add_argument("--reservation-directory", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        audit_original(args.original.resolve(), args.audit_directory.resolve())
        print(json.dumps({"status": "PASSED", "completed_trajectories": 92}))
    elif args.command == "prepare":
        prepare_remaining(args.original.resolve(), args.directory.resolve(),
                          args.audit_directory.resolve(), args.reservation_directory)
    else:
        base = args.directory.resolve()
        try:
            run_remaining(base)
        except BaseException as error:
            runner.stop_reservation(base)
            runner.atomic_json(base / "service_completion.json", {
                "status": "FAILED", "at_SGT": runner.restart.now(), "seed": 43,
                "error": f"{type(error).__name__}: {error}", "original_failure_preserved": True,
            })
            raise


if __name__ == "__main__":
    main()
