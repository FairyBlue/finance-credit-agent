"""Fresh, two-seed one-round recovery sanity checks from one new shared SFT.

This fixed engineering acceptance is separate from the four-round R5 experiment.
Recovery SFT train/CPU merge runs through coldstart_sft before prepare; the merged
model and receipt are mandatory inputs. No existing actor is continued or patched.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from src.agents import learning_restart as restart
from src.agents import low_memory_pilot as runner
from src.agents.screening_launcher import atomic_json


def _acceptance_api():
    # The shared acceptance helpers import Agent-R1's observation/action types.
    # Keep this CLI usable before the worker-specific PYTHONPATH is constructed.
    agent_path = str(runner.REPO_ROOT / "third_party/agent-r1")
    if agent_path not in sys.path:
        sys.path.insert(0, agent_path)
    return importlib.import_module("src.agents.order_recovery_acceptance")


acceptance = _acceptance_api()
NAME = "shared_recovery_sft_fresh_one_round_stage1_sanity_v1"
PLAN = "docs/2026-10-11_M2_ORDER_RECOVERY_LEARNING_PLAN.md"
DEFAULT = runner.REPO_ROOT / "runs/m2-order-recovery-learning-stage1"
SEEDS = (42, 43)
TRAIN_NUMBERS = (1, 65, 9, 73, 17, 149, 25, 157)
EVAL_NUMBERS = (210, 231, 252, 273, 336, 399, 420, 441)
OLD_SHARED_SFT_RECEIPT = "cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db"
ROOT_INPUTS = ("learning_settings.json", "shared_sft.json", "source_identity.json", "plan.md")
RECLAIM_WAIT_SECONDS, RECLAIM_POLL_SECONDS = 300, 5
RECLAIM_PROFILE = "verified_previous_seed_moment_df_wait_v1"


def _profile(seed):
    rl = (
        runner.CellSpec("m1", "M1", 0.0, 1e-6, 1),
        runner.CellSpec("m2_eta025", "M2", 0.25, 1e-6, 1),
    )
    return runner.RecoveryExperimentProfile(
        name=f"shared_recovery_sft_seed{seed}_stage1_sanity_v1", plan=PLAN,
        train_numbers=TRAIN_NUMBERS, eval_numbers=EVAL_NUMBERS,
        evaluation_steps=(1,), rounds=1, include_sft=True,
        retire_completed_moments=True, seed=seed, actor_learning_rate=1e-7,
        cell_specs=(runner.CellSpec("sft0", "M1", 0.0, 1e-6, 0),)
        + (rl if seed == 42 else tuple(reversed(rl))),
        evaluation_incomplete_mode="retain_prefix_v1",
        acceptance_gate_sha256=acceptance.fingerprint(acceptance.ENDPOINT_GATE),
    )


PROFILES = {seed: _profile(seed) for seed in SEEDS}


def profile_for_seed(seed):
    if type(seed) is not int or seed not in PROFILES:
        raise ValueError("recovery acceptance seed must be 42 or 43")
    return PROFILES[seed]


def settings():
    return dict(
        profile=NAME, plan=PLAN, purpose="stage1_recovery_sanity",
        four_round_performance_evidence=False, seeds=list(SEEDS),
        profiles={str(seed): asdict(profile) for seed, profile in PROFILES.items()},
        prepared_training_episode_ids=PROFILES[42].train_ids,
        consumed_training_episode_ids=PROFILES[42].train_ids[:2],
        evaluation_episode_ids=PROFILES[42].eval_ids,
        evaluation_selection="six_known_failure_plus_two_normal_train_windows",
        untouched_holdout=False,
        planned_trajectories=sum(p.planned_trajectories for p in PROFILES.values()),
        training_trajectories=32, evaluation_trajectories=240,
        sft_baselines=2, rl_cells=4, training_updates_per_role_per_cell=1,
        actor_learning_rate=1e-7, critic_learning_rate=1e-6, m2_residual_eta=0.25,
        common_invalid_action_penalty=runner.PENALTY,
        observation_profile=runner.RECOVERY_OBSERVATION_PROFILE,
        acceptance_gate=acceptance.ENDPOINT_GATE,
        acceptance_gate_sha256=acceptance.fingerprint(acceptance.ENDPOINT_GATE),
        native_interaction_cap=256, prompt_cap=8192, response_cap=4096,
        native_session_recovery_cap=None,
        strict_recovery_acceptance="separate_order_recovery_episode_4_followups_v1",
        model_retention="four_complete_actor_bf16_hf_v1",
        disk_budget_profile=runner.ONE_ROUND_DISK_PROFILE,
        initial_free_disk_required_bytes=runner.disk_requirement_bytes(PROFILES[42], 4),
        execution="serial_fresh_one_shot_v1", automatic_retry=False,
        resume_supported=False, automatic_actions=False,
        seed_boundary_disk_wait=dict(
            profile=RECLAIM_PROFILE, maximum_seconds=RECLAIM_WAIT_SECONDS,
            poll_seconds=RECLAIM_POLL_SECONDS, requires_verified_owned_retirement=True,
            uses_actual_free_bytes=True, training_retry=False,
        ),
    )


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def child_directory(base, seed):
    profile_for_seed(seed)
    return base / f"seed{seed}"


def _validate_sft_inputs(model_path, lineage_receipt, gate_readout):
    """Validate the common initialization and its completed independently audited gate."""
    model_path, lineage_receipt, gate_readout = (
        Path(path).resolve() for path in (model_path, lineage_receipt, gate_readout)
    )
    with tempfile.TemporaryDirectory(prefix="recovery-sft-contract-") as temporary:
        spec = Path(temporary) / "models.json"
        atomic_json(spec, {"sft": {"model_path": str(model_path),
                                   "lineage_receipt": str(lineage_receipt)}})
        common = acceptance.model_contracts(spec, "sft")["sft"]
    prior = restart.read(gate_readout)
    manifest_path = gate_readout.parent / "manifest.json"
    manifest = acceptance.verify_completed(
        gate_readout.parent, prior.get("manifest_sha256"), check_models=False,
    )
    audit = acceptance.raw_audit_gate(gate_readout.parent, manifest, completed_runtime=True)
    if (prior.get("status") != "OBSERVED" or prior.get("acceptance") != "PASS"
            or prior.get("mode") != "sft" or manifest.get("mode") != "sft"
            or prior.get("endpoint_gate_sha256") != settings()["acceptance_gate_sha256"]
            or prior.get("manifest_sha256") != restart.digest(manifest_path)
            or audit.get("status") != "PASS" or audit.get("acceptance_pass") is not True
            or prior.get("raw_audit_sha256") != audit.get("sha256")
            or manifest["models"]["sft"]["fingerprint"] != common["fingerprint"]):
        raise RuntimeError(
            "new common SFT must pass the same independently audited acceptance gate"
        )
    return dict(
        model_fingerprint=common["fingerprint"], lineage_receipt=str(lineage_receipt),
        lineage_receipt_sha256=restart.digest(lineage_receipt),
        sft_gate_readout=str(gate_readout), sft_gate_readout_sha256=restart.digest(gate_readout),
        sft_gate_manifest_sha256=restart.digest(manifest_path),
        sft_gate_raw_audit_sha256=audit["sha256"],
    )


def prepare(base, *, model_path, merge_receipt_sha256, sft_gate_readout,
            sft_lineage_receipt=None, coordinated_reservation_directory=None):
    """Pin the same newly merged recovery SFT for all actor/ref/critic roles."""
    if base.exists():
        raise RuntimeError("recovery acceptance requires a new root directory")
    if merge_receipt_sha256 == OLD_SHARED_SFT_RECEIPT:
        raise RuntimeError("recovery acceptance requires a new shared recovery SFT")
    model = dict(source="local_export", name=str(model_path.resolve()),
                 merge_receipt_sha256=merge_receipt_sha256)
    name, revision, snapshot = runner._resolve_exact_model_snapshot({"model": model})
    budget = runner.verify_one_round_model_budget(snapshot)
    lineage_receipt = sft_lineage_receipt or snapshot.parent / "lineage_receipt.json"
    lineage = _validate_sft_inputs(snapshot, lineage_receipt, sft_gate_readout)
    base.mkdir(parents=True, exist_ok=False)
    atomic_json(base / "learning_settings.json", settings())
    receipt = restart.read(snapshot / "merge_receipt.json")
    atomic_json(base / "shared_sft.json", dict(
        model=name, model_revision=revision, model_snapshot_path=str(snapshot),
        merge_receipt_sha256=merge_receipt_sha256, disk_budget=budget,
        merge_file_inventory=receipt["files"],
        merge_source_model=receipt["model"], merge_source_adapter=receipt["adapter"],
        recovery_acceptance=lineage,
    ))
    from uuid import uuid4

    token = uuid4().hex if coordinated_reservation_directory is not None else None
    for seed, profile in PROFILES.items():
        runner.prepare(
            child_directory(base, seed), model_path=snapshot,
            merge_receipt_sha256=merge_receipt_sha256, profile=profile,
            coordinated_reservation_directory=coordinated_reservation_directory,
            coordinator_token=token,
        )
    refresh_status(base)


async def input_preflight(base):
    _check_settings(base)
    for seed, profile in PROFILES.items():
        await runner.input_preflight(child_directory(base, seed), profile=profile)


def _check_settings(base):
    if not _same(restart.read(base / "learning_settings.json"), settings()):
        raise RuntimeError("recovery acceptance settings differ from the fixed budget")


def seal(base):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("recovery acceptance already sealed or started")
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
    atomic_json(base / "run_contract.json", dict(
        profile=NAME, sealed_at_SGT=restart.now(),
        inputs={path: restart.digest(base / path) for path in ROOT_INPUTS},
        child_contracts={str(seed): restart.digest(
            child_directory(base, seed) / "run_contract.json"
        ) for seed in SEEDS},
        planned_trajectories=settings()["planned_trajectories"],
        automatic_retry=False, resume_supported=False,
    ))
    check_contract(base)
    refresh_status(base)


def check_contract(base):
    _check_settings(base)
    contract = restart.read(base / "run_contract.json")
    if (contract.get("profile") != NAME or set(contract.get("inputs", {})) != set(ROOT_INPUTS)
            or set(contract.get("child_contracts", {})) != {str(seed) for seed in SEEDS}
            or contract.get("planned_trajectories") != settings()["planned_trajectories"]
            or contract.get("automatic_retry") is not False
            or contract.get("resume_supported") is not False):
        raise RuntimeError("recovery acceptance contract identity or budget differs")
    for relative, digest in contract["inputs"].items():
        if restart.digest(base / relative) != digest:
            raise RuntimeError("sealed recovery acceptance input changed: " + relative)
    sources = restart.read(base / "source_identity.json")
    current = runner.source_files()
    if set(sources) != set(current):
        raise RuntimeError("recovery acceptance source inventory changed")
    for relative, path in current.items():
        if (restart.digest(path) != sources[relative]
                or restart.digest(base / "sources" / relative) != sources[relative]):
            raise RuntimeError("recovery acceptance source changed: " + relative)
    shared = restart.read(base / "shared_sft.json")
    if shared["merge_receipt_sha256"] == OLD_SHARED_SFT_RECEIPT:
        raise RuntimeError("old shared SFT cannot satisfy recovery acceptance")
    snapshot = Path(shared["model_snapshot_path"])
    receipt = restart.read(snapshot / "merge_receipt.json")
    if (restart.digest(snapshot / "merge_receipt.json") != shared["merge_receipt_sha256"]
            or not _same(receipt["files"], shared["merge_file_inventory"])
            or receipt["model"] != shared["merge_source_model"]
            or receipt["adapter"] != shared["merge_source_adapter"]):
        raise RuntimeError("shared SFT merge receipt or full weight inventory changed")
    lineage = shared["recovery_acceptance"]
    if not _same(_validate_sft_inputs(snapshot, lineage["lineage_receipt"],
                                    lineage["sft_gate_readout"]), lineage):
        raise RuntimeError("common SFT lineage or acceptance evidence changed")
    if not _same(runner.verify_one_round_model_budget(snapshot), shared["disk_budget"]):
        raise RuntimeError("shared recovery SFT disk budget changed")
    coordinator = None
    for seed, profile in PROFILES.items():
        child = child_directory(base, seed)
        if child.is_symlink():
            raise RuntimeError("recovery acceptance child directory must not be linked")
        if restart.digest(child / "run_contract.json") != contract["child_contracts"][str(seed)]:
            raise RuntimeError(f"sealed child contract changed: seed{seed}")
        runner.check_contract(child, profile=profile)
        actual = restart.read(child / "model_identity.json")
        if any(actual[key] != shared[key] for key in (
            "model_revision", "model_snapshot_path", "merge_receipt_sha256",
        )):
            raise RuntimeError("child actor/ref/critic shared SFT identity differs")
        config = runner.memory_coordinator_config(child)
        # Children must share the same holder and lease, but their local evidence
        # paths differ. None is allowed for CPU prepare; run requires the holder.
        identity = None if config is None else {
            key: value for key, value in config.items() if key != "evidence_directory"
        }
        if seed == SEEDS[0]:
            coordinator = identity
        elif identity != coordinator:
            raise RuntimeError("both seeds must use the same frozen GPU holder lease")


def _readout(base, *, allow_partial):
    rows, incomplete = [], 0
    for seed, profile in PROFILES.items():
        heads = []
        for spec in runner.cells(profile):
            path = child_directory(base, seed) / spec.cell_id / "summary.json"
            summary = restart.read(path) if path.exists() else {}
            if summary.get("status") != "COMPLETE":
                if not allow_partial:
                    raise RuntimeError("not all recovery acceptance cells completed")
                rows.append(dict(seed=seed, cell=spec.cell_id,
                                 status=summary.get("status", "PENDING")))
                continue
            if summary.get("actual_rounds") != spec.rounds:
                raise RuntimeError("recovery acceptance actual update budget differs")
            if not spec.zero:
                head = summary.get("critic_initial_head_sha256")
                if not isinstance(head, str) or len(head) != 64:
                    raise RuntimeError("fresh common critic initialization evidence missing")
                heads.append(head)
            failures = summary.get("incomplete_evaluations", 0)
            if type(failures) is not int or failures < 0:
                raise RuntimeError("invalid incomplete-evaluation count")
            incomplete += failures
            rows.append(dict(seed=seed, cell=spec.cell_id, status="COMPLETE",
                             incomplete_evaluations=failures,
                             evaluation_by_step=summary.get("evaluation_by_step", {})))
        if len(heads) == 2 and heads[0] != heads[1]:
            raise RuntimeError("M1/M2 fresh critic heads differ within the same seed")
    result = dict(profile=NAME, cells=rows, incomplete_evaluations=incomplete,
                  financial_status="INCOMPLETE" if incomplete else "COMPLETE",
                  four_round_performance_evidence=False,
                  strict_four_followup_acceptance_performed=False)
    atomic_json(base / "readout.json", result)
    return result



def export_model_spec(base):
    """Publish genuine actor lineage after all four fresh cells, without loading GPU."""
    check_contract(base)
    _readout(base, allow_partial=False)
    shared = restart.read(base / "shared_sft.json")
    common = shared["recovery_acceptance"]
    protocol_path = base / "learning_settings.json"
    protocol_sha = restart.digest(protocol_path)
    specification = {"sft": {"model_path": shared["model_snapshot_path"],
                              "lineage_receipt": common["lineage_receipt"]}}
    for seed, profile in PROFILES.items():
        child = child_directory(base, seed)
        for spec in runner.cells(profile):
            if spec.zero:
                continue
            cell = child / spec.cell_id
            summary = restart.read(cell / "summary.json")
            if summary.get("status") != "COMPLETE" or summary.get("actual_rounds") != 1:
                raise RuntimeError("actor lineage requires one completed fresh training round")
            step = cell / "checkpoints/global_step_1"
            from src.agents.actor_export import validate_export_artifact

            validate_export_artifact(
                step, identity_sha256=summary["execution_identity_sha256"], verify_hashes=True,
            )
            model = step / "actor/huggingface"
            key = f"seed{seed}_{spec.arm.lower()}"
            lineage_path = cell / "lineage_receipt.json"
            evidence_paths = [
                protocol_path, base / "run_contract.json", base / "shared_sft.json",
                Path(common["lineage_receipt"]), Path(common["sft_gate_readout"]),
                child / "run_contract.json", child / "settings.json",
                cell / "settings.json", cell / "resolved_config.json", cell / "command.json",
                cell / "summary.json", step / "actor_export_complete.json",
                step / "actor/actor_export_receipt.json",
                cell / "artifacts/critic_initialization/head-rank0.json",
            ]
            evidence_paths.extend(
                path for folder in ("actor_update", "critic_fit", "placement", "screening_evidence")
                for path in (cell / "artifacts" / folder).rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl"}
            )
            if any(path.is_symlink() or not path.is_file() for path in evidence_paths):
                raise RuntimeError("actor lineage requires genuine regular training evidence")
            evidence = {str(path.resolve()): restart.digest(path) for path in evidence_paths}
            files = acceptance.hf_inventory(model, excluded={lineage_path.resolve()})
            lineage = dict(
                status="COMPLETE", model_key=key,
                model_fingerprint=acceptance.fingerprint(files),
                action_protocol=acceptance.PROTOCOL,
                observation_profile=runner.RECOVERY_OBSERVATION_PROFILE,
                acceptance_gate_sha256=settings()["acceptance_gate_sha256"],
                training_updates=1, method=spec.arm, training_seed=seed,
                common_sft_fingerprint=common["model_fingerprint"],
                common_training_protocol_sha256=protocol_sha,
                actor_learning_rate=1e-7, critic_learning_rate=1e-6,
                m2_residual_eta=spec.eta, training_evidence_sha256=evidence,
            )
            # A failed or changed prior export remains inspectable; no overwrite.
            if lineage_path.exists():
                if not _same(restart.read(lineage_path), lineage):
                    raise RuntimeError("existing actor lineage differs from validated evidence")
            else:
                atomic_json(lineage_path, lineage)
            specification[key] = dict(model_path=str(model.resolve()),
                                      lineage_receipt=str(lineage_path.resolve()))
    destination = base / "model_spec_all.json"
    if destination.exists() and not _same(restart.read(destination), specification):
        raise RuntimeError("existing all-model specification differs")
    atomic_json(destination, specification)
    acceptance.model_contracts(destination, "all")
    return destination


def refresh_status(base):
    state = restart.read(base / "state.json") if (base / "state.json").exists() else {}
    atomic_json(base / "live_status.json", dict(
        profile=NAME, at_SGT=restart.now(), status=state.get("status", "PREPARED"),
        current_seed=state.get("current_seed"), completed_seeds=state.get("completed_seeds", []),
        planned_trajectories=settings()["planned_trajectories"],
        purpose="one_round_stage1_sanity_not_four_round_performance",
    ))


def _owned_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError("retirement evidence leaves its owned cell")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise RuntimeError("retirement evidence must not contain symbolic links")
    return path


def _verified_retirements(base, seed):
    """Verify only this batch's completed cells and retained ownership inventories."""
    profile = profile_for_seed(seed)
    child = _owned_path(base, f"seed{seed}")
    state_path = _owned_path(child, "state.json")
    state = restart.read(state_path)
    if (state.get("status") != "COMPLETE" or state.get("profile") != profile.name
            or state.get("seed") != seed):
        raise RuntimeError("disk waiting requires this batch's completed previous seed")
    specifications = list(runner.cells(profile))
    summaries = state.get("cells", [])
    if ([row.get("cell") for row in summaries] != [asdict(spec) for spec in specifications]):
        raise RuntimeError("previous seed's completed cell population differs")
    evidence, roots = {str(state_path): restart.digest(state_path)}, []
    for spec, recorded in zip(specifications, summaries, strict=True):
        if spec.zero:
            continue
        cell = _owned_path(child, spec.cell_id)
        summary_path = _owned_path(cell, "summary.json")
        summary = restart.read(summary_path)
        identity = summary.get("execution_identity_sha256")
        if (summary != recorded or summary.get("status") != "COMPLETE"
                or summary.get("returncode") != 0 or summary.get("seed") != seed
                or summary.get("actual_rounds") != spec.rounds
                or summary.get("cell") != asdict(spec)
                or not isinstance(identity, str) or len(identity) != 64
                or any(character not in "0123456789abcdef" for character in identity)):
            raise RuntimeError("disk waiting requires successful current-batch RL cells")
        receipt_path = _owned_path(cell, "moment_pruning_receipt.json")
        pruning = summary.get("moment_pruning", {})
        if (pruning.get("status") != "COMPLETE"
                or pruning.get("receipt") != str(receipt_path)):
            raise RuntimeError("retirement receipt is not owned by this batch's completed cell")
        receipt = restart.read(receipt_path)
        if (receipt.get("status") != "COMPLETE" or receipt.get("cell") != spec.cell_id
                or receipt.get("execution_identity_sha256") != identity
                or receipt.get("purpose") != "temporary moments; no resume supported"):
            raise RuntimeError("retirement receipt identity or completion differs")
        scratch = _owned_path(cell, "artifacts/placement/moment_scratch")
        roots.append(scratch)
        expected, roles = {}, set()
        kept = receipt.get("kept_manifests", [])
        if len(kept) != 2:
            raise RuntimeError("retirement requires both retained role manifests")
        for entry in kept:
            manifest_path = _owned_path(cell, entry["path"])
            parts = manifest_path.relative_to(scratch).parts
            if (len(parts) != 3 or not parts[0].isdecimal()
                    or parts[1] not in {"actor", "critic"} or parts[2] != "manifest.json"
                    or parts[1] in roles or restart.digest(manifest_path) != entry["sha256"]):
                raise RuntimeError("retired role manifest identity or hash differs")
            role = parts[1]
            roles.add(role)
            owner_path = _owned_path(cell, str(manifest_path.with_name("owner.json")
                                              .relative_to(cell)))
            manifest, owner = restart.read(manifest_path), restart.read(owner_path)
            generation = manifest.get("generation", "")
            if (manifest.get("schema_version") != 1
                    or manifest.get("identity") != identity or owner.get("identity") != identity
                    or manifest.get("role") != role or owner.get("role") != role
                    or entry.get("role") != role
                    or not manifest.get("owner")
                    or manifest["owner"] != owner.get("owner")
                    or manifest["owner"] != entry.get("owner")
                    or generation != entry.get("generation")
                    or not generation.startswith("generation-")
                    or not generation.removeprefix("generation-").isdecimal()
                    or (Path("/proc") / parts[0]).exists()):
                raise RuntimeError("retired moment ownership, generation or worker exit differs")
            generation_dir = _owned_path(cell, str((manifest_path.parent / generation)
                                                   .relative_to(cell)))
            if not generation_dir.is_dir():
                raise RuntimeError("retired moment generation is missing")
            role_bytes = 0
            for parameter in manifest["parameters"]:
                for chunks in parameter["moments"].values():
                    for chunk in chunks:
                        name = chunk["file"]
                        if Path(name).name != name or not name.endswith(".raw"):
                            raise RuntimeError("retired moment chunk leaves its generation")
                        path = generation_dir / name
                        relative = str(path.relative_to(cell))
                        if (relative in expected or path.exists() or path.is_symlink()
                                or type(chunk["bytes"]) is not int or chunk["bytes"] <= 0):
                            raise RuntimeError(
                                "retired moment payload is duplicated or still present"
                            )
                        expected[relative] = (chunk["bytes"], chunk["sha256"])
                        role_bytes += chunk["bytes"]
            if role_bytes != manifest["payload_bytes"]:
                raise RuntimeError("retired role payload total differs")
            evidence[str(manifest_path)] = restart.digest(manifest_path)
            evidence[str(owner_path)] = restart.digest(owner_path)
        files = receipt.get("files", [])
        actual = {}
        for row in files:
            path = _owned_path(cell, row["path"])
            if (row["path"] in actual or path.exists() or path.is_symlink()
                    or row.get("device") != cell.stat().st_dev
                    or type(row.get("inode")) is not int or row["inode"] <= 0):
                raise RuntimeError("retirement receipt contains foreign or undeleted payloads")
            actual[row["path"]] = (row["bytes"], row["sha256"])
        total = sum(size for size, _ in expected.values())
        if (actual != expected or not total
                or receipt.get("planned_bytes") != total or receipt.get("deleted_bytes") != total
                or receipt.get("deleted_files") != len(expected)
                or pruning.get("deleted_bytes") != total
                or pruning.get("deleted_files") != len(expected)
                or any(scratch.rglob("*.raw"))):
            raise RuntimeError("retirement inventory, deleted counts or raw absence differs")
        evidence[str(summary_path)] = restart.digest(summary_path)
        evidence[str(receipt_path)] = restart.digest(receipt_path)
    return evidence, roots


def wait_seed_disk_reclaim(base, previous_seed, next_seed, *, clock=None, sleep=None,
                           disk_usage=None):
    """Wait for real filesystem accounting after verified retirement, never retry RL."""
    if (previous_seed, next_seed) != SEEDS:
        raise RuntimeError("disk waiting is restricted to this batch's seed boundary")
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    disk_usage = disk_usage or shutil.disk_usage
    profile = profile_for_seed(next_seed)
    exports = sum(not spec.zero for spec in runner.cells(profile))
    required = runner.disk_requirement_bytes(profile, exports)
    path = base / f"seed{next_seed}_disk_reclaim_receipt.json"
    if path.exists():
        raise RuntimeError("disk wait receipt already exists; no retry or overwrite")
    started = clock()
    receipt = dict(profile=RECLAIM_PROFILE, status="CHECKING", previous_seed=previous_seed,
                   next_seed=next_seed, required_free_bytes=required,
                   maximum_seconds=RECLAIM_WAIT_SECONDS, poll_seconds=RECLAIM_POLL_SECONDS,
                   automatic_retry=False, uses_actual_free_bytes=True, samples=[])
    evidence, roots = {}, []
    try:
        while True:
            for filename, digest in evidence.items():
                if restart.digest(Path(filename)) != digest:
                    raise RuntimeError("verified retirement evidence changed while waiting")
            if any(any(root.rglob("*.raw")) for root in roots):
                raise RuntimeError("retired raw payload reappeared while waiting")
            elapsed, free = clock() - started, disk_usage(base).free
            if type(free) is not int or free < 0:
                raise RuntimeError("actual free disk telemetry is invalid")
            receipt["samples"].append(dict(elapsed_seconds=elapsed, free_bytes=free))
            if free >= required:
                receipt["status"] = "READY"
                return receipt
            if not evidence:
                evidence, roots = _verified_retirements(base, previous_seed)
                receipt["verified_retirement_evidence_sha256"] = evidence
            if elapsed >= RECLAIM_WAIT_SECONDS:
                receipt["status"] = "TIMEOUT"
                raise RuntimeError("actual free disk did not reach the unchanged next-seed gate")
            receipt["status"] = "WAITING"
            atomic_json(path, receipt)
            remaining = RECLAIM_WAIT_SECONDS - (clock() - started)
            if remaining > 0:
                sleep(min(RECLAIM_POLL_SECONDS, remaining))
    except BaseException as error:
        if receipt["status"] != "TIMEOUT":
            receipt["status"] = "FAILED"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(path, receipt)


def run(base):
    if (base / "state.json").exists():
        raise RuntimeError("new recovery acceptance state required; no retry or resume")
    check_contract(base)
    for seed in SEEDS:
        if runner.memory_coordinator_config(child_directory(base, seed)) is None:
            raise RuntimeError("recovery training requires the persistent GPU holder contract")
    required = settings()["initial_free_disk_required_bytes"]
    free = shutil.disk_usage(base).free
    if free < required:
        raise RuntimeError("insufficient disk for one-round moments, four actors and reserve")
    state = dict(profile=NAME, status="RUNNING", started_at_SGT=restart.now(),
                 current_seed=None, completed_seeds=[], free_disk_bytes=free,
                 required_free_disk_bytes=required, automatic_retry=False, resume_supported=False)
    atomic_json(base / "state.json", state)
    try:
        for seed, profile in PROFILES.items():
            check_contract(base)
            if state["completed_seeds"]:
                wait_seed_disk_reclaim(base, state["completed_seeds"][-1], seed)
            state["current_seed"] = seed
            atomic_json(base / "state.json", state)
            refresh_status(base)
            runner.run(child_directory(base, seed), profile=profile)
            if restart.read(child_directory(base, seed) / "state.json").get("status") != "COMPLETE":
                raise RuntimeError(f"seed{seed} did not finish its fixed budget")
            check_contract(base)
            state["completed_seeds"].append(seed)
            atomic_json(base / "state.json", state)
            _readout(base, allow_partial=True)
        readout = _readout(base, allow_partial=False)
        model_spec = export_model_spec(base)
        state["model_spec_all"] = str(model_spec.resolve())
        state.update(status="COMPLETE_WITH_EVALUATION_FAILURES" if readout["incomplete_evaluations"]
                     else "COMPLETE", current_seed=None,
                     financial_status=readout["financial_status"])
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)
        refresh_status(base)
        atomic_json(base / "service_completion.json", dict(
            profile=NAME, status=state["status"], completed_seeds=state["completed_seeds"],
            automatic_retry=False, resume_supported=False, four_round_performance_evidence=False,
            strict_four_followup_acceptance_performed=False,
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--merge-receipt-sha256")
    parser.add_argument("--sft-lineage-receipt", type=Path)
    parser.add_argument("--sft-gate-readout", type=Path)
    parser.add_argument("--coordinated-reservation-directory", type=Path)
    args = parser.parse_args()
    base = args.directory.resolve()
    if args.command == "prepare":
        if (args.model is None or args.merge_receipt_sha256 is None
                or args.sft_gate_readout is None):
            parser.error(
                "prepare requires new --model, --merge-receipt-sha256 and --sft-gate-readout"
            )
        holder = args.coordinated_reservation_directory or runner.current_reservation_directory()
        prepare(base, model_path=args.model, merge_receipt_sha256=args.merge_receipt_sha256,
                sft_lineage_receipt=args.sft_lineage_receipt,
                sft_gate_readout=args.sft_gate_readout,
                coordinated_reservation_directory=holder)
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
