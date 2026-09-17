"""Fixed eight-update, wider-Train M1/M2 comparison from the common SFT."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from src.agents import learning_restart as restart
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
    set_overrides,
    validate_cell_evidence,
    verify_gpu_runtime_and_model,
)

PROFILE = "common_sft_train_coverage_v1"
PLAN = "docs/2026-09-15_M2_TRAIN_COVERAGE_PLAN.md"
TRAIN_NUMBERS = [1, 65, 9, 73, 17, 149, 25, 157, 33, 165, 41, 173, 49, 181, 57, 189]
TRAIN_IDS = [f"formal-train-{i:04d}" for i in TRAIN_NUMBERS]
EVAL_IDS = restart.EVAL_IDS
EVAL_STEPS = [2, 4, 6, 8]
DEFAULT = REPO_ROOT / "runs/m2-train-coverage-20260915"
READER = REPO_ROOT / "scripts/diagnostics/m2_train_coverage_readout.py"
INPUTS = (
    "settings.json",
    "model_identity.json",
    "input_preflight.json",
    "prepared/manifest.json",
    "selection.json",
    "source_identity.json",
    "plan.md",
)
CellSpec = tuning.CellSpec


def settings(model):
    if model.get("source") != "local_export" or model.get("revision") is not None:
        raise ValueError("coverage comparison requires the shared pinned local SFT")
    result = restart.settings()
    result.update(
        artifact_profile="actor_bf16_hf_v1",
        action_protocol=tuning.PROTOCOL,
        critic_initialization="cpu_seed_v1",
        m2_residual_eta=0.0,
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
            training_trajectories_per_arm=64,
            planned_trajectories=308,
            automatic_retry=False,
            resume_supported=False,
            adaptive_gate=False,
        ),
    )
    result["models"]["screening"] = copy.deepcopy(model)
    result["training"].update(
        episodes=16,
        total_epochs=1,
        total_training_steps=8,
        learning_rate=2e-7,
        critic_learning_rate=1e-6,
        test_freq=2,
        save_freq=8,
        max_checkpoints=1,
        max_infrastructure_resumes=0,
    )
    result["evaluation"].update(
        rollout_n=5,
        do_sample=True,
        temperature=1.0,
        primary="same_worker_separate_sampled_and_greedy",
    )
    return result


def read_settings(base):
    value = restart.read(base / "settings.json")
    if not tuning._same(value, settings(value["models"]["screening"])):
        raise RuntimeError("coverage settings differ from the fixed plan")
    return value


def cells():
    return (
        CellSpec("sft0", "M1", 0.0, 1e-6, 0),
        CellSpec("m1", "M1", 0.0, 1e-6, 8),
        CellSpec("m2_eta025", "M2", 0.25, 1e-6, 8),
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
    episodes = formal["splits"]["train"]
    if len(episodes) != 441 or any(
        e["episode_id"] != f"formal-train-{i + 1:04d}" for i, e in enumerate(episodes)
    ):
        raise RuntimeError("reviewed original Train episode grid changed")

    def days(episode):
        return {str(day)[:10] for day in episode["episode_dates"]}

    excluded = set().union(*(days(e) for e in old["selected_episodes"]["dev"]))
    excluded.update(set().union(*(days(episodes[int(e.rsplit("-", 1)[1]) - 1]) for e in EVAL_IDS)))
    eligible = [
        e
        for e in episodes
        if not days(e) & excluded
        and not any("2023-05-23" <= day <= "2023-06-13" for day in days(e))
    ]
    chosen = [eligible[i * (len(eligible) - 1) // 15] for i in range(16)]
    ordered = [chosen[i] for pair in zip(range(8), range(8, 16), strict=True) for i in pair]
    if [e["episode_id"] for e in ordered] != TRAIN_IDS or len(eligible) != 121:
        raise RuntimeError("date-only sixteen-window selection no longer matches the plan")
    if any(days(ordered[i]) & days(ordered[i + 1]) for i in range(0, 16, 2)):
        raise RuntimeError("each two-window training batch must have disjoint sessions")
    return ordered, dict(
        criterion=(
            "16 equally spaced indices across eligible chronological starts; then interleave halves"
        ),
        eligible_start_count=len(eligible),
        eligible_indices=[i * 8 for i in range(16)],
        unique_training_sessions=len(set().union(*(days(e) for e in ordered))),
        training_eval_dates_disjoint=True,
        old_screen_dev_dates_preserved=True,
        coldstart_fresh_dates_preserved=True,
        windows=[
            dict(episode_id=e["episode_id"], first_session=min(days(e)), last_session=max(days(e)))
            for e in ordered
        ],
    )


def prepare(base, *, model_path, merge_receipt_sha256):
    import pandas as pd

    from src.agents.config import load_experiment_config
    from src.data.screening import _dataset_rows, _environment_common

    if base.exists():
        raise RuntimeError("coverage run requires a new directory")
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
        row["extra_info"]["screening_role"] = "train_coverage_train"
    old_run = REPO_ROOT / "runs/m2-training-tuning-20260914"
    old_dev = old_run / "prepared/dev.parquet"
    old_manifest = restart.read(old_run / "prepared/manifest.json")
    if restart.digest(old_dev) != old_manifest["datasets"]["dev"]["sha256"]:
        raise RuntimeError("fixed four Train evaluation windows changed")
    base.mkdir(parents=True, exist_ok=False)
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
        actor_lr_and_date_coverage_changed_together=True,
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
                raise RuntimeError("coverage may only use original Train rows")
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(
                observation_profile="history_features_v1", reward_mode="finance_trajectory"
            )
            if len(kwargs["episode_dates"]) != 21:
                raise RuntimeError("coverage requires twenty-one-session windows")
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
                    raise RuntimeError("coverage prompt exceeds common budget")
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
    files = tuning._source_files()
    files[str(READER.relative_to(REPO_ROOT))] = READER
    monitor = REPO_ROOT / "scripts/diagnostics/m2_train_coverage_monitor.py"
    files[str(monitor.relative_to(REPO_ROOT))] = monitor
    helper = REPO_ROOT / "scripts/diagnostics/m2_training_tuning_readout.py"
    files[str(helper.relative_to(REPO_ROOT))] = helper
    return files


def seal(base):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("coverage already sealed or started")
    read_settings(base)
    pre = restart.read(base / "input_preflight.json")
    if (
        pre.get("status") != "PASSED"
        or [r.get("episode") for r in pre.get("windows", [])] != TRAIN_IDS + EVAL_IDS
        or any(r.get("complete") is not True for r in pre["windows"])
    ):
        raise RuntimeError("all twenty complete CPU windows must pass")
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
        raise RuntimeError("coverage identity is incomplete")
    for p, digest in seal["inputs"].items():
        if restart.digest(base / p) != digest:
            raise RuntimeError("sealed coverage input changed: " + p)
    sources = restart.read(base / "source_identity.json")
    if set(sources) != set(source_files()):
        raise RuntimeError("coverage source inventory changed")
    for p, path in source_files().items():
        if restart.digest(path) != sources[p]:
            raise RuntimeError("coverage source changed: " + p)
    for p, digest in restart.read(base / "selection.json")["source_sha256"].items():
        if restart.digest(Path(p)) != digest:
            raise RuntimeError("coverage selection source changed: " + p)
    manifest = restart.read(base / "prepared/manifest.json")
    for name, ids in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        row = manifest["datasets"][name]
        if row["episode_ids"] != ids or row["row_count"] != len(ids):
            raise RuntimeError("coverage window budget changed")
    for row in [*manifest["datasets"].values(), manifest["calibration"]]:
        if restart.digest(base / "prepared" / row["path"]) != row["sha256"]:
            raise RuntimeError("coverage prepared data changed")
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
    result["run"].update(name=f"train-coverage-{cell_id}-s42", kind=PROFILE)
    result["screening"].update(coverage_cell_id=cell_id)
    return result


def build_command(*, config, run_settings, hashes, cell_dir):
    base_settings = settings(run_settings["models"]["screening"])
    spec = next(
        s for s in cells() if s.cell_id == run_settings["training_coverage"]["cell"]["cell_id"]
    )
    if not tuning._same(run_settings, cell_settings(base_settings, spec.cell_id)):
        raise RuntimeError("cell condition differs from fixed coverage settings")
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
    ):
        raise RuntimeError("resolved arm, eta or shared model differs")
    command = build_cell_command(
        config=config, settings=run_settings, hashes=hashes, cell_dir=cell_dir
    )
    prefix = "+ray_kwargs.ray_init.runtime_env.env_vars."
    command = set_overrides(
        command,
        {
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
    return dict(actor_update=actor, paired_validation=paired)


def run_cell(base, cell_id):
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
            )
        ):
            env.pop(key)
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
    )
    atomic_json(cell / "summary.json", summary)
    try:
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
        if not spec.zero:
            baseline = restart.read(base / "sft0/summary.json")
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
        summary["finished_at_SGT"] = restart.now()
        atomic_json(cell / "summary.json", summary)


def wait_gpu(timeout=120):
    started = time.monotonic()
    while True:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        if int(q.strip()) < 1024:
            return
        if time.monotonic() - started >= timeout:
            raise RuntimeError("training needs the shared H100 free; no other process stopped")
        time.sleep(5)


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
        planned_trajectories=308,
        automatic_retry=False,
        resume_supported=False,
        train_episode_ids=TRAIN_IDS,
        evaluation_episode_ids=EVAL_IDS,
        evaluation_steps=EVAL_STEPS,
    )
    atomic_json(base / "state.json", state)
    try:
        for index, spec in enumerate(cells()):
            exports = sum(not s.zero for s in cells()[index:])
            required = (9 * exports + 8) * 1024**3
            free = shutil.disk_usage(base).free
            if free < required:
                raise RuntimeError(
                    "insufficient space for two final actors and fixed reserve; no cleanup"
                )
            wait_gpu()
            state.update(current_cell=spec.cell_id, free_disk_bytes=free)
            atomic_json(base / "state.json", state)
            state["cells"].append(run_cell(base, spec.cell_id))
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
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
        prepare(base, model_path=args.model, merge_receipt_sha256=args.merge_receipt_sha256)
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
