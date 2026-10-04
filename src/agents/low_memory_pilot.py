"""Two-update full-parameter 4B engineering pilot for disk-backed chunked Adam.

Fresh M1 and M2 cells use the approved common SFT, reward and first two
coverage batches. This controller never resumes or retries a failed update.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from src.agents import common_validity_penalty as common
from src.agents import learning_restart as restart
from src.agents import train_coverage as coverage
from src.agents import training_tuning as tuning
from src.agents.launcher import (
    _finalize_model_paths,
    _resolve_exact_model_snapshot,
    _training_environment,
)
from src.agents.metadata import collect_run_hashes
from src.agents.screening_launcher import (
    REPO_ROOT,
    atomic_json,
    build_cell_command,
    build_cell_config,
    capture_process,
    read_jsonl,
    set_overrides,
    validate_cell_evidence,
    verify_gpu_runtime_and_model,
)

PROFILE = "common_sft_low_memory_pilot_v1"
PENALTY = 0.05
PLAN = "docs/2026-09-30_M2_LOW_MEMORY_PILOT_PLAN.md"
TRAIN_NUMBERS = [1, 65, 9, 73]
TRAIN_IDS = [f"formal-train-{i:04d}" for i in TRAIN_NUMBERS]
EVAL_IDS = restart.EVAL_IDS
EVAL_STEPS = [2]
DEFAULT = REPO_ROOT / "runs/m2-low-memory-pilot-20260930"
INPUTS = (
    "settings.json",
    "model_identity.json",
    "input_preflight.json",
    "prepared/manifest.json",
    "selection.json",
    "source_identity.json",
    "plan.md",
    "gpu_handoff_config.json",
)
CellSpec = tuning.CellSpec
MIN_FREE_GPU_MIB = 60 * 1024
WAIT_SECONDS = 12 * 3600


def settings(model):
    if model.get("source") != "local_export" or model.get("revision") is not None:
        raise ValueError("low-memory pilot comparison requires the shared pinned local SFT")
    result = restart.settings()
    result.update(
        artifact_profile="actor_bf16_hf_v1",
        action_protocol=tuning.PROTOCOL,
        critic_initialization="cpu_seed_v1",
        m2_residual_eta=0.0,
        common_validity_penalty=dict(
            profile="environment_rejection_v1",
            coefficient=PENALTY,
            applies_to="training_only",
            financial_terminal_score_unchanged=True,
        ),
        plan=PLAN,
        arms=["M1", "M2"],
        training_coverage=dict(
            profile=PROFILE,
            evaluation_steps=list(EVAL_STEPS),
            actor_learning_rate=2e-7,
            critic_learning_rate=1e-6,
            training_episode_ids=list(TRAIN_IDS),
            evaluation_episode_ids=list(EVAL_IDS),
            evaluation_lanes={"greedy": [0], "sampled": [1, 2, 3, 4]},
            training_trajectories_per_arm=16,
            planned_trajectories=72,
            automatic_retry=False,
            resume_supported=False,
            adaptive_gate=False,
        ),
    )
    result["models"]["screening"] = copy.deepcopy(model)
    result["training"].update(
        episodes=4,
        total_epochs=1,
        total_training_steps=2,
        learning_rate=2e-7,
        critic_learning_rate=1e-6,
        test_freq=2,
        save_freq=2,
        max_checkpoints=1,
        max_infrastructure_resumes=0,
    )
    result["evaluation"].update(
        rollout_n=5,
        do_sample=True,
        temperature=1.0,
        primary="same_worker_separate_sampled_and_greedy",
    )
    result["resources"].update(
        profile="chunked_adam_disk_v1",
        chunk_bytes=64 * 1024**2,
        gpu_reserve_bytes=2 * 1024**3,
        host_reserve_bytes=2 * 1024**3,
        disk_reserve_bytes=8 * 1024**3,
    )
    return result


def read_settings(base):
    value = restart.read(base / "settings.json")
    if not tuning._same(value, settings(value["models"]["screening"])):
        raise RuntimeError("low-memory pilot settings differ from the fixed plan")
    return value


def cells():
    return (
        CellSpec("m1", "M1", 0.0, 1e-6, 2),
        CellSpec("m2_eta025", "M2", 0.25, 1e-6, 2),
    )


def cell_settings(run_settings, cell_id):
    spec = next(spec for spec in cells() if spec.cell_id == cell_id)
    value = copy.deepcopy(run_settings)
    value["training_coverage"]["cell"] = asdict(spec)
    value["m2_residual_eta"] = spec.eta
    if spec.zero:
        value["training"].update(total_training_steps=1, test_freq=1, save_freq=1)
    return value


def select_episodes(formal, old):
    """Use exactly the first two old coverage batches, without outcome selection."""
    full, source = coverage.select_episodes(formal, old)
    selected = full[:4]
    if [row["episode_id"] for row in selected] != TRAIN_IDS:
        raise RuntimeError("the first four coverage windows differ from the fixed plan")
    selection = copy.deepcopy(source)
    selection.update(
        criterion="first four windows of the preserved date-only coverage selection",
        coverage_reference_profile=coverage.PROFILE,
        coverage_full_training_episode_ids=list(coverage.TRAIN_IDS),
        training_episode_ids=list(TRAIN_IDS),
        unique_training_sessions=len(
            {str(day)[:10] for row in selected for day in row["episode_dates"]}
        ),
        windows=source["windows"][:4],
    )
    return selected, selection


def prepare(base, *, model_path, merge_receipt_sha256, reservation_directory=None):
    import pandas as pd

    from src.agents.config import load_experiment_config
    from src.data.screening import _dataset_rows, _environment_common

    if base.exists():
        raise RuntimeError("low-memory pilot run requires a new directory")
    model = dict(
        source="local_export",
        name=str(model_path.resolve()),
        merge_receipt_sha256=merge_receipt_sha256,
    )
    name, identity, path = _resolve_exact_model_snapshot({"model": model})
    model["name"] = name
    original = REPO_ROOT / "artifacts/m2_screening"
    old = restart.read(original / "manifest.json")
    formal_path = Path(old["original_manifest"]["path"])
    if restart.digest(formal_path) != old["original_manifest"]["sha256"]:
        raise RuntimeError("reviewed formal manifest changed")
    calibration = original / old["calibration"]["path"]
    if restart.digest(calibration) != old["calibration"]["sha256"]:
        raise RuntimeError("existing calibration changed")
    formal = yaml.safe_load(formal_path.read_text())
    selected, selection = select_episodes(formal, old)
    config = load_experiment_config(REPO_ROOT / "configs/base.yaml")
    environment = _environment_common(
        config=config, source=old["source"], calibration=restart.read(calibration)
    )
    training = _dataset_rows(
        role="train",
        episodes=selected,
        indices=[n - 1 for n in TRAIN_NUMBERS],
        environment_common=environment,
    )
    for row in training:
        row["extra_info"]["screening_role"] = "low_memory_pilot_train"
    old_run = REPO_ROOT / "runs/m2-training-tuning-20260914"
    old_dev = old_run / "prepared/dev.parquet"
    old_manifest = restart.read(old_run / "prepared/manifest.json")
    if restart.digest(old_dev) != old_manifest["datasets"]["dev"]["sha256"]:
        raise RuntimeError("fixed four Train evaluation windows changed")
    base.mkdir(parents=True, exist_ok=False)
    handoff = {"profile": "disabled"}
    if reservation_directory is not None:
        from src.agents.gpu_handoff import config_for_holder

        handoff = config_for_holder(reservation_directory)
    atomic_json(base / "gpu_handoff_config.json", handoff)
    target = base / "prepared"
    target.mkdir()
    pd.DataFrame(training).to_parquet(target / "train.parquet", index=False)
    shutil.copy2(old_dev, target / "dev.parquet")
    shutil.copy2(calibration, target / "reward_calibration.json")
    datasets = {
        n: dict(
            path=n + ".parquet",
            episode_ids=ids,
            row_count=len(ids),
            sha256=restart.digest(target / (n + ".parquet")),
        )
        for n, ids in (("train", TRAIN_IDS), ("dev", EVAL_IDS))
    }
    manifest = dict(
        datasets=datasets,
        calibration=old["calibration"],
        source_manifest_sha256=restart.digest(formal_path),
        eval_is_reserved_train_subset=True,
        train_eval_dates_disjoint=True,
        created_at_SGT=restart.now(),
        selection=selection,
    )
    manifest["hashes"] = {
        "payload_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    }
    atomic_json(target / "manifest.json", manifest)
    selection.update(
        actor_learning_rate=2e-7,
        critic_learning_rate=1e-6,
        actor_lr_and_date_coverage_changed_together=False,
        same_as_coverage_first_two_rounds=True,
        common_invalid_action_penalty=PENALTY,
        critic_lr_basis=(
            "completed same-batch critic-fit comparison; unchanged from previous main run"
        ),
        source_sha256={
            str(p): restart.digest(p)
            for p in (
                formal_path,
                original / "manifest.json",
                calibration,
                old_dev,
                old_run / "selection.json",
            )
        },
    )
    atomic_json(base / "selection.json", selection)
    atomic_json(base / "settings.json", settings(model))
    atomic_json(
        base / "model_identity.json",
        dict(
            model=name,
            model_revision=identity,
            model_snapshot_path=str(path),
            model_source="local_export",
            merge_receipt_sha256=merge_receipt_sha256,
        ),
    )


async def input_preflight(base):
    import pandas as pd

    agent_path = str(REPO_ROOT / "third_party/agent-r1")
    if agent_path not in sys.path:
        sys.path.insert(0, agent_path)
    from agent_r1.env.base import Action
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    config = read_settings(base)
    _, identity, snapshot = _resolve_exact_model_snapshot({"model": config["models"]["screening"]})
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    records, session_sets = [], {}
    for dataset, expected in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        rows = pd.read_parquet(base / "prepared" / f"{dataset}.parquet").to_dict("records")
        if [r["extra_info"]["episode_id"] for r in rows] != expected or [
            r["extra_info"]["index"] for r in rows
        ] != list(range(len(expected))):
            raise RuntimeError("prepared episode or sample-index order changed")
        session_sets[dataset] = set()
        for row in rows:
            if row["extra_info"]["source_split"] != "train":
                raise RuntimeError("low-memory pilot trial may only use original Train rows")
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(
                observation_profile="history_features_v1", reward_mode="finance_trajectory"
            )
            if len(kwargs["episode_dates"]) != 21:
                raise RuntimeError("low-memory pilot trial requires twenty-one-session windows")
            session_sets[dataset].update(kwargs["episode_dates"])
            env = TradingEnv(**kwargs)
            obs = env.reset(raw_prompt=list(row["prompt"]))
            counts = []
            for session in range(21):
                messages = apply_screening_action_protocol(obs.messages, tuning.PROTOCOL)
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tools=env.tool_schemas,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                )
                counts.append(len(tokens))
                if len(tokens) > 8192:
                    raise RuntimeError("low-memory pilot prompt exceeds common budget")
                if session < 20:
                    obs, _, done, info = await env.step(
                        Action(text='<tool_call>{"name":"hold","arguments":{}}</tool_call>')
                    )
                    if info.get("invalid_action") or done != (session == 19):
                        raise RuntimeError("CPU complete-window replay failed")
            records.append(
                dict(
                    episode=row["extra_info"]["episode_id"],
                    complete=env.episode_complete,
                    max_prompt_tokens=max(counts),
                )
            )
    if session_sets["train"] & session_sets["dev"]:
        raise RuntimeError("training and evaluation sessions overlap")
    atomic_json(
        base / "input_preflight.json",
        dict(
            status="PASSED",
            at_SGT=restart.now(),
            windows=records,
            model_revision=identity,
            tokenizer_path=str(snapshot),
            action_protocol=tuning.PROTOCOL,
            enable_thinking=False,
        ),
    )


def source_files():
    return tuning._source_files()


def seal(base):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("low-memory pilot trial already sealed or started")
    read_settings(base)
    pre = restart.read(base / "input_preflight.json")
    if (
        pre.get("status") != "PASSED"
        or [r.get("episode") for r in pre.get("windows", [])] != TRAIN_IDS + EVAL_IDS
        or any(r.get("complete") is not True for r in pre["windows"])
    ):
        raise RuntimeError("all eight complete CPU windows must pass")
    shutil.copy2(REPO_ROOT / PLAN, base / "plan.md")
    sources = {}
    for relative, path in source_files().items():
        target = base / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative] = restart.digest(path)
    atomic_json(base / "source_identity.json", sources)
    atomic_json(
        base / "run_contract.json",
        dict(
            profile=PROFILE,
            sealed_at_SGT=restart.now(),
            inputs={p: restart.digest(base / p) for p in INPUTS},
        ),
    )
    check_contract(base)


def check_contract(base):
    seal = restart.read(base / "run_contract.json")
    if seal.get("profile") != PROFILE or set(seal.get("inputs", {})) != set(INPUTS):
        raise RuntimeError("low-memory pilot identity is incomplete")
    for p, digest in seal["inputs"].items():
        if restart.digest(base / p) != digest:
            raise RuntimeError("sealed low-memory pilot input changed: " + p)
    sources = restart.read(base / "source_identity.json")
    if set(sources) != set(source_files()):
        raise RuntimeError("low-memory pilot source inventory changed")
    for p, path in source_files().items():
        if restart.digest(path) != sources[p]:
            raise RuntimeError("low-memory pilot source changed: " + p)
    for p, digest in restart.read(base / "selection.json")["source_sha256"].items():
        if restart.digest(Path(p)) != digest:
            raise RuntimeError("low-memory pilot selection source changed: " + p)
    manifest = restart.read(base / "prepared/manifest.json")
    for name, ids in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        row = manifest["datasets"][name]
        if row["episode_ids"] != ids or row["row_count"] != len(ids):
            raise RuntimeError("low-memory pilot window budget changed")
    for row in [*manifest["datasets"].values(), manifest["calibration"]]:
        if restart.digest(base / "prepared" / row["path"]) != row["sha256"]:
            raise RuntimeError("low-memory pilot prepared data changed")
    cfg = read_settings(base)
    _, identity, path = _resolve_exact_model_snapshot({"model": cfg["models"]["screening"]})
    recorded = restart.read(base / "model_identity.json")
    pre = restart.read(base / "input_preflight.json")
    if (
        recorded["model_revision"] != identity
        or recorded["model_snapshot_path"] != str(path)
        or pre["model_revision"] != identity
        or pre["action_protocol"] != tuning.PROTOCOL
        or pre["enable_thinking"] is not False
    ):
        raise RuntimeError("shared SFT or prompt identity changed")


def make_config(base, manifest, cell_id):
    cfg = cell_settings(read_settings(base), cell_id)
    spec = cfg["training_coverage"]["cell"]
    result = build_cell_config(
        settings=cfg,
        manifest=manifest,
        prepared_dir=base / "prepared",
        arm=spec["arm"],
        stage="zero-shot" if spec["rounds"] == 0 else "screening",
        seed=42,
        cell_dir=base / cell_id,
    )
    result["training"].update(cfg["training"], critic_initialization="cpu_seed_v1")
    result["run"].update(name=f"low-memory-pilot-{cell_id}-s42", kind=PROFILE)
    result["screening"].update(
        coverage_cell_id=cell_id,
        invalid_action_penalty=PENALTY,
        common_validity_penalty=copy.deepcopy(cfg["common_validity_penalty"]),
    )
    if handoff_config(base) is not None:
        result["screening"]["gpu_handoff_config_sha256"] = restart.digest(
            base / "gpu_handoff_config.json"
        )
    return result


def handoff_config(base):
    path = base / "gpu_handoff_config.json"
    if not path.exists():
        # Old fixtures/config construction may omit the optional resource hook.
        # A real prepared run must include the file in its sealed INPUTS.
        return None
    value = restart.read(path)
    if value == {"profile": "disabled"}:
        return None
    from src.agents.gpu_handoff import load_config

    return load_config(path, restart.digest(path))


def build_command(*, config, run_settings, hashes, cell_dir):
    base_settings = settings(run_settings["models"]["screening"])
    spec = next(
        s for s in cells() if s.cell_id == run_settings["training_coverage"]["cell"]["cell_id"]
    )
    if not tuning._same(run_settings, cell_settings(base_settings, spec.cell_id)):
        raise RuntimeError("cell condition differs from fixed low-memory pilot settings")
    expected = {**run_settings["training"], "critic_initialization": "cpu_seed_v1"}
    if any(not tuning._same(config["training"].get(k), v) for k, v in expected.items()):
        raise RuntimeError("resolved training budget/LR differs")
    if (
        config["screening"]["arm"] != spec.arm
        or config["screening"]["residual_eta"] != spec.eta
        or config["model"] != run_settings["models"]["screening"]
        or config["run"]["seed"] != 42
        or config["screening"]["stage"] != ("zero-shot" if spec.zero else "screening")
        or config["screening"]["artifact_profile"] != "actor_bf16_hf_v1"
        or config["screening"]["coverage_cell_id"] != spec.cell_id
        or config["screening"].get("invalid_action_penalty") != PENALTY
        or config["screening"].get("common_validity_penalty")
        != run_settings["common_validity_penalty"]
    ):
        raise RuntimeError("resolved arm, eta or shared model differs")
    command = build_cell_command(
        config=config, settings=run_settings, hashes=hashes, cell_dir=cell_dir
    )
    prefix = "+ray_kwargs.ray_init.runtime_env.env_vars."
    command = set_overrides(
        command,
        {
            prefix + "FINANCE_COMMON_INVALID_PENALTY": json.dumps(str(PENALTY)),
            prefix + "FINANCE_SCREENING_CRITIC_INIT": "cpu_seed_v1",
            prefix + "FINANCE_SCREENING_CRITIC_INIT_EVIDENCE_DIR": str(
                cell_dir / "artifacts/critic_initialization"
            ),
            prefix + "FINANCE_CRITIC_FIT_PROBE_DIR": str(cell_dir / "artifacts/critic_fit"),
            prefix + "FINANCE_ACTOR_UPDATE_PROBE_DIR": str(cell_dir / "artifacts/actor_update"),
            prefix + "FINANCE_PAIRED_VALIDATION_PROFILE": "greedy1_sample4_v1",
            prefix + "FINANCE_PAIRED_VALIDATION_EVIDENCE_DIR": str(
                cell_dir / "artifacts/paired_validation"
            ),
            "critic.model.tokenizer_path": config["model"]["name"],
            "+actor_rollout_ref.ref.model.path": config["model"]["name"],
            "data.validation_shuffle": False,
            "actor_rollout_ref.rollout.val_kwargs.n": 5,
            "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
            "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        },
    )
    if handoff_config(cell_dir.parent) is not None:
        path = cell_dir.parent / "gpu_handoff_config.json"
        digest = restart.digest(path)
        if config["screening"].get("gpu_handoff_config_sha256") != digest:
            raise RuntimeError("handoff config differs from execution identity")
        command = set_overrides(command, {
            prefix + "FINANCE_GPU_HANDOFF_CONFIG": str(path.resolve()),
            prefix + "FINANCE_GPU_HANDOFF_CONFIG_SHA256": json.dumps(digest),
            prefix + "FINANCE_GPU_HANDOFF_EVIDENCE_DIR": str(
                (cell_dir / "artifacts/gpu_handoff").resolve()
            ),
        })
    return command


def validate_extra_evidence(cell, spec, identity):
    # Each implementation validates its real runtime schema and fixed budget.
    from src.agents.actor_update_probe import validate_run_evidence as validate_actor
    from src.agents.paired_validation import validate_run_evidence as validate_paired

    actor = validate_actor(
        cell / "artifacts/actor_update",
        expected_updates=spec.rounds,
        arm=spec.arm,
        execution_identity=identity,
    )
    paired = validate_paired(
        cell / "artifacts/paired_validation",
        expected_steps=[0] if spec.zero else EVAL_STEPS,
        expected_episode_ids=EVAL_IDS,
        arm=spec.arm,
        execution_identity=identity,
    )
    penalty = common.validate_common_penalty_evidence(cell, spec)
    return dict(actor_update=actor, paired_validation=paired, common_penalty_evidence=penalty)


def validate_placement(cell, spec, identity):
    rows = read_jsonl(cell / "artifacts/placement")
    if not rows or any(
        row.get("profile") != "chunked_adam_disk_v1"
        or row.get("identity_sha256") != identity
        or row.get("failed") is True
        or str(row.get("phase", "")).endswith(":error")
        for row in rows
    ):
        raise RuntimeError("chunked placement identity or boundary failure")
    events = [r["detail"] for r in rows if r.get("event") == "exchange"]
    if any(r.get("event") == "failed" for r in events):
        raise RuntimeError("chunked placement recorded a failed operation")
    result = {"status": "PASSED", "updates": {}, "exchanges": []}
    for role in ("critic", "actor"):
        done = [r for r in events if r.get("event") == "optimizer_complete"
                and r.get("role") == role]
        updates = [r["detail"] for r in events if r.get("event") == "optimizer_chunked"
                   and r.get("role") == role]
        if [r.get("role_step") for r in done] != list(range(1, spec.rounds + 1)) or (
            len(updates) != spec.rounds or any(
                r.get("event") != "chunked_adamw_step_complete"
                or r.get("chunks", 0) < 1
                or not 0 < r.get("maximum_temporary_bytes", 0) <= 64 * 1024**2
                or r.get("moment_transfer_bytes", 0) <= 0
                for r in updates
            )
        ):
            raise RuntimeError("chunked Adam did not complete exactly two real CUDA updates")
        result["updates"][role] = updates
    pending = None
    for row in events:
        if row.get("event") == "exchange_start":
            if pending is not None:
                raise RuntimeError("overlapping optimizer state exchanges")
            pending = row
        elif row.get("event") == "exchange_complete":
            if pending is None:
                raise RuntimeError("optimizer state exchange is missing its start")
            active = row["active_role"]
            inactive = "critic" if active == "actor" else "actor"
            if (row["moment_cpu_bytes_by_role"][inactive] != 0
                    or row["disk_status"][inactive] != "offloaded"
                    or row["disk_status"][active] != "resident"):
                raise RuntimeError("inactive optimizer moments remained in CPU memory")
            result["exchanges"].append(dict(
                active_role=active,
                elapsed_seconds=(row["time_ns"] - pending["time_ns"]) / 1e9,
                moment_cpu_bytes_by_role=row["moment_cpu_bytes_by_role"],
            ))
            pending = None
    if pending is not None or len(result["exchanges"]) != 2 * spec.rounds:
        raise RuntimeError("incomplete actor/critic disk exchanges")
    for field in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
        values = [r[field] for r in rows if field in r]
        if not values:
            raise RuntimeError("worker CUDA peak evidence missing")
        result[field] = max(values)
    result["peak_container_memory_bytes_sampled"] = max(
        int(r.get("memory.current", 0)) for r in rows
    )
    return result


def validate_handoff(cell, identity):
    config = handoff_config(cell.parent)
    if config is None:
        return {"status": "DISABLED"}
    paths = list((cell / "artifacts/gpu_handoff").glob("receipt-*.json"))
    if len(paths) != 1:
        raise RuntimeError("one shared-worker GPU handoff receipt is required")
    receipt = restart.read(paths[0])
    if (receipt.get("status") != "PASSED" or receipt.get("identity") != identity
            or receipt.get("holder_config") != config or receipt.get("first_role") != "critic"
            or [row.get("role") for row in receipt.get("role_checks", [])]
            != ["reference", "actor"]
            or receipt.get("worker_memory", {}).get("driver_free_bytes", 0)
            < MIN_FREE_GPU_MIB * 1024**2):
        raise RuntimeError("GPU handoff initialization evidence differs from the fixed profile")
    states = [receipt["after"], *[row["holder_status"] for row in receipt["role_checks"]]]
    if any(state.get("paused") is not True or any(
        state.get(key) != 0 for key in (
            "payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"
        )
    ) for state in states):
        raise RuntimeError("GPU holder was not inactive throughout model initialization")
    return dict(status="PASSED", receipt=str(paths[0]), worker_pid=receipt["worker_pid"],
                free_after_release_bytes=receipt["worker_memory"]["driver_free_bytes"])


def evaluation_readout(cell):
    """Descriptive only; the caller has already validated the complete fixed grid."""
    rows = read_jsonl(cell / "artifacts/screening_evidence")
    result = {}
    for lane, siblings in (("greedy", {0}), ("sampled", {1, 2, 3, 4})):
        selected = []
        for row in rows:
            identity = row.get("rollout_identity", row.get("identity"))
            if identity["validate"] and identity["rollout_n"] in siblings:
                selected.append(row)
        if len(selected) != len(EVAL_IDS) * len(siblings):
            raise RuntimeError("evaluation lane size differs")
        metrics = {}
        for field in ("raw_net_return", "transaction_cost", "terminal_score"):
            values = [r["metrics"][field] for r in selected]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                raise RuntimeError("nonfinite evaluation outcome")
            metrics["mean_" + field] = math.fsum(values) / len(values)
        steps = [step for r in selected for step in r["steps"]]
        result[lane] = dict(
            trajectories=len(selected), decisions=len(steps),
            invalid_decisions=sum(s.get("invalid_action") is True for s in steps),
            **metrics,
        )
    return result


def refresh_status(base):
    state = restart.read(base / "state.json") if (base / "state.json").exists() else {}
    queue = restart.read(base / "gpu_queue_state.json") if (
        base / "gpu_queue_state.json"
    ).exists() else {}
    status = state.get("status", "PREPARED")
    current = state.get("current_cell")
    if status == "RUNNING" and queue.get("cell") == current and (
        queue.get("status") == "WAITING_FOR_GPU"
    ):
        status = "WAITING_FOR_GPU"
    lines = ["# 4B 低显存两轮工程验收", "", f"当前进展（新加坡时间）：{restart.now()}",
             "", f"状态：**{status}**；当前臂：{current or '尚未开始'}。",
             "", "固定预算：M1、M2 各 16 条训练 + 20 条同 worker 评估，共 72 条。",
             "4B 全参数；共同 SFT v3；non-thinking；CPU 活跃 Adam moments + 磁盘非活跃 moments。",
             "模型导出不含优化器；本轮不支持断点续训或自动重跑。", ""]
    if queue:
        free = queue.get("free_MiB")
        lines += [f"GPU 准入读数：{free / 1024:.2f} GiB 空闲（门槛 60 GiB）。" if free
                  is not None else "GPU 空闲读数暂缺。", ""]
        if queue.get("holder_payload_bytes") is not None:
            lines += [f"占位实际持有 {queue['holder_payload_bytes'] / 1024**3:.2f} GiB；"
                      f"空闲加可释放占位 {queue['available_bytes'] / 1024**3:.2f} GiB。",
                      "只在 critic 首次加载前释放；训练期间及两臂之间不自动回补。", ""]
    for spec in cells():
        cell = base / spec.cell_id
        summary = restart.read(cell / "summary.json") if (
            cell / "summary.json"
        ).exists() else {}
        observed = 0
        for path in (cell / "artifacts/screening_evidence").glob("*.jsonl"):
            with path.open() as stream:
                observed += sum(bool(line.strip()) for line in stream)
        lines += [f"- {spec.cell_id}：{summary.get('status', 'PENDING')}；"
                  f"已落盘轨迹 {observed}/36。"]
        if summary.get("error"):
            lines += [f"  错误：{summary['error']}"]
        if summary.get("placement"):
            p = summary["placement"]
            lines += [f"  训练 worker 的 PyTorch 峰值分配："
                      f"{p['gpu_peak_allocated_bytes'] / 1024**3:.2f} GiB；"
                      f"整臂耗时 {summary['elapsed_seconds'] / 60:.1f} 分钟。"]
        if summary.get("evaluation"):
            for lane, row in summary["evaluation"].items():
                lines += [f"  {lane}：平均净收益 {100 * row['mean_raw_net_return']:.4f}%；"
                          f"非法请求 {row['invalid_decisions']}/{row['decisions']}。"]
    if state.get("error"):
        lines += ["", "停止原因：" + state["error"]]
    lines += ["", "本轮衡量资源路径能否跑通；两轮 Train 读数不能证明 M2 优于 M1。",
              "阶段显存包含本进程指标；nvidia-smi 的整卡读数还包含其他用户进程。", ""]
    path = base / "LIVE_STATUS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    if state.get("status") in {"COMPLETE", "FAILED"}:
        shutil.copy2(path, base / "FINAL_RESULTS.md")


def monitor_status(base, stop):
    while not stop.is_set():
        try:
            refresh_status(base)
        except Exception as error:
            atomic_json(base / "monitor_error.json", dict(at_SGT=restart.now(), error=str(error)))
        stop.wait(30)


def run_cell(base, cell_id):
    started = time.monotonic()
    check_contract(base)
    phase = read_settings(base)
    spec = next(s for s in cells() if s.cell_id == cell_id)
    cfg = cell_settings(phase, cell_id)
    manifest = restart.read(base / "prepared/manifest.json")
    cell = base / cell_id
    cell.mkdir(exist_ok=False)
    config = make_config(base, manifest, cell_id)
    agent = REPO_ROOT / config["framework"]["agent_r1_path"]
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent,
        data_version=config["data"]["version"],
        expected_agent_r1_commit=config["framework"]["agent_r1_commit"],
        require_clean_repo=False,
    )
    atomic_json(cell / "resolved_config.json", config)
    atomic_json(cell / "settings.json", cfg)
    command = build_command(config=config, run_settings=cfg, hashes=hashes, cell_dir=cell)
    runtime_path = base / "runtime_preflight.json"
    if not runtime_path.exists():
        atomic_json(runtime_path, verify_gpu_runtime_and_model(config))
    runtime = restart.read(runtime_path)
    recorded = restart.read(base / "model_identity.json")
    if any(runtime[k] != recorded[k] for k in ("model_revision", "model_snapshot_path")):
        raise RuntimeError("runtime preflight does not belong to shared SFT")
    command = _finalize_model_paths(
        command, arm_id=config["arm"]["id"], model_snapshot_path=runtime["model_snapshot_path"]
    )
    env = _training_environment(
        agent_path=agent,
        probe_root=cell / "artifacts/update_probe",
        c_compiler=runtime["c_compiler"],
    )
    env["PYTHONUNBUFFERED"] = "1"
    for key in list(env):
        if key.startswith(
            (
                "FINANCE_SCREENING_",
                "FINANCE_LEARNING_",
                "FINANCE_CRITIC_",
                "FINANCE_ACTOR_",
                "FINANCE_PAIRED_",
                "FINANCE_COMMON_",
                "FINANCE_CHUNKED_",
                "FINANCE_GPU_HANDOFF_",
            )
        ):
            env.pop(key)
    # Apply the exact same setting to the launcher and its Ray workers.  Never
    # inherit an unrelated experiment's penalty from the supervising shell.
    env["FINANCE_COMMON_INVALID_PENALTY"] = str(PENALTY)
    atomic_json(cell / "command.json", command)
    opts = dict(a.lstrip("+").split("=", 1) for a in command[3:])
    identity = opts["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"]
    evsteps = [0] if spec.zero else EVAL_STEPS
    summary = dict(
        status="RUNNING",
        cell=asdict(spec),
        started_at_SGT=restart.now(),
        requested_rounds=spec.rounds,
        model_revision=recorded["model_revision"],
        artifact_profile=cfg["artifact_profile"],
        optimizer_included=False,
        resume_supported=False,
        evaluation_steps=evsteps,
        execution_identity_sha256=identity,
        common_validity_penalty=copy.deepcopy(cfg["common_validity_penalty"]),
    )
    atomic_json(cell / "summary.json", summary)
    try:
        # Finish hashes, runtime/compiler checks and command construction while
        # the separate holder still owns its allocations. The worker performs
        # the actual release and a second admission immediately before loading.
        wait_gpu(base, cell_id)
        code = capture_process(command, environment=env, cell_dir=cell)
        summary["returncode"] = code
        if code:
            raise RuntimeError(f"{cell_id} trainer exited {code}; evidence retained")
        summary["evidence"] = validate_cell_evidence(
            cell,
            stage=config["screening"]["stage"],
            expected_rounds=max(1, spec.rounds),
            arm=spec.arm,
            seed=42,
            train_episode_ids=TRAIN_IDS,
            eval_episode_ids=EVAL_IDS,
            execution_identity_sha256=identity,
            calibration_payload_sha256=manifest["calibration"]["payload_sha256"],
            expected_eval_steps=evsteps,
            expected_eta=spec.eta,
            artifact_profile=cfg["artifact_profile"],
            eval_rollout_n=5,
        )
        summary["critic_fit"] = tuning.validate_fit_evidence(
            cell, spec=spec, execution_identity=identity
        )
        summary.update(validate_extra_evidence(cell, spec, identity))
        summary["placement"] = validate_placement(cell, spec, identity)
        summary["gpu_handoff"] = validate_handoff(cell, identity)
        summary["evaluation"] = evaluation_readout(cell)
        head = restart.read(cell / "artifacts/critic_initialization/head-rank0.json")
        if (
            head.get("seed") != 42
            or head.get("mode") != "cpu_seed_v1"
            or head.get("arm") != spec.arm
            or head.get("execution_identity_sha256") != identity
            or head.get("model_path") != str(Path(runtime["model_snapshot_path"]).resolve())
            or head.get("phase") != "critic_head_after_hf_load_before_fsdp"
        ):
            raise RuntimeError("actual initial critic head identity differs")
        summary["critic_initial_head_sha256"] = head["head_sha256"]
        if spec.arm == "M2":
            baseline = restart.read(base / "m1/summary.json")
            if (
                baseline["status"] != "COMPLETE"
                or head["head_sha256"] != baseline["critic_initial_head_sha256"]
            ):
                raise RuntimeError("initial critic head differs from shared SFT baseline")
        check_contract(base)
        summary.update(actual_rounds=spec.rounds, status="COMPLETE")
        return summary
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["finished_at_SGT"] = restart.now()
        atomic_json(cell / "summary.json", summary)


def gpu_free_mib():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    ).strip().splitlines()
    if len(output) != 1 or not output[0].strip().isdigit():
        raise RuntimeError("expected one GPU with an integer memory.free reading")
    return int(output[0])


def wait_gpu(base, cell_id, timeout=WAIT_SECONDS):
    handoff = handoff_config(base)
    started = time.monotonic()
    record = dict(status="WAITING_FOR_GPU", cell=cell_id, started_at_SGT=restart.now(),
                  minimum_free_MiB=MIN_FREE_GPU_MIB, maximum_wait_seconds=timeout,
                  stops_other_processes=False, automatic_retry=False)
    while True:
        record.update(at_SGT=restart.now(), waited_seconds=time.monotonic() - started)
        try:
            if handoff is None:
                free = gpu_free_mib()
                available_mib = free
            else:
                from src.agents.gpu_handoff import available_bytes

                resources = available_bytes(handoff)
                record.update(resources)
                free = resources["driver_free_bytes"] // 1024**2
                available_mib = resources["available_bytes"] // 1024**2
            record.update(free_MiB=free, last_telemetry_error=None)
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            free = -1
            available_mib = -1
            record.update(free_MiB=None, last_telemetry_error=str(error))
            if handoff is not None:
                # A dead/changed holder cannot become valid through polling.
                record["status"] = "HANDOFF_UNAVAILABLE"
                atomic_json(base / "gpu_queue_state.json", record)
                raise
        if time.monotonic() - started >= timeout:
            record["status"] = "WAIT_TIMEOUT"
            atomic_json(base / "gpu_queue_state.json", record)
            raise RuntimeError("60 GiB free GPU condition did not arrive within queue limit")
        if available_mib >= MIN_FREE_GPU_MIB:
            record["status"] = "ADMITTED"
            atomic_json(base / "gpu_queue_state.json", record)
            return
        atomic_json(base / "gpu_queue_state.json", record)
        time.sleep(min(45, max(0, timeout - (time.monotonic() - started))))


def stop_reservation(base):
    """Release this run's holder on exit; never manipulate training/foreign PIDs."""
    from src.agents.gpu_handoff import reservation_status
    from src.agents.gpu_reservation_service import request

    receipt = dict(at_SGT=restart.now())
    try:
        config = handoff_config(base)
        if config is None:
            return
        receipt["before"] = reservation_status(config, minimum_lease_seconds=0)
        receipt["after"] = request(config["directory"], {"op": "stop"})
        receipt["status"] = "STOP_REQUESTED"
    except Exception as error:
        receipt.update(status="STOP_UNCONFIRMED", error=f"{type(error).__name__}: {error}")
    atomic_json(base / "reservation_cleanup.json", receipt)


def run(base):
    if (base / "state.json").exists():
        raise RuntimeError("new state required; no implicit retry")
    check_contract(base)
    state = dict(
        status="RUNNING",
        started_at_SGT=restart.now(),
        cells=[],
        current_cell=None,
        profile=PROFILE,
        planned_trajectories=72,
        automatic_retry=False,
        resume_supported=False,
        train_episode_ids=TRAIN_IDS,
        evaluation_episode_ids=EVAL_IDS,
        evaluation_steps=EVAL_STEPS,
    )
    atomic_json(base / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_status, args=(base, stop), daemon=True)
    monitor.start()
    try:
        for index, spec in enumerate(cells()):
            exports = sum(not s.zero for s in cells()[index:])
            # Two retained role snapshots plus one replacement generation (~90 GiB),
            # remaining actor exports and reserve. Previous cell scratch is kept.
            required = (90 + 9 * exports + 8) * 1024**3
            free = shutil.disk_usage(base).free
            if free < required:
                raise RuntimeError(
                    "insufficient space for disk moment generations, exports and reserve"
                )
            state.update(current_cell=spec.cell_id, free_disk_bytes=free)
            atomic_json(base / "state.json", state)
            state["cells"].append(run_cell(base, spec.cell_id))
            atomic_json(base / "state.json", state)
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        stop_reservation(base)
        refresh_status(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--reservation-directory", type=Path)
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "runs/m2-coldstart-v3-20260914/merged_model"
    )
    parser.add_argument(
        "--merge-receipt-sha256",
        default="cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db",
    )
    args = parser.parse_args()
    base = args.directory.resolve()
    if args.command == "prepare":
        prepare(base, model_path=args.model, merge_receipt_sha256=args.merge_receipt_sha256,
                reservation_directory=args.reservation_directory)
    elif args.command == "preflight":
        asyncio.run(input_preflight(base))
    elif args.command == "seal":
        seal(base)
    elif args.command == "check":
        check_contract(base)
    else:
        run(base)
    print(json.dumps(dict(command=args.command, status="PASS", at_SGT=restart.now())))


if __name__ == "__main__":
    main()
