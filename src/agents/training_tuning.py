"""Fresh critic-LR diagnostics followed by a separately frozen common-SFT trial."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agents import learning_restart as restart
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

TRAIN_IDS, EVAL_IDS = restart.TRAIN_IDS, restart.EVAL_IDS
PROTOCOL = "single_tool_portfolio_v1"
PROFILE = "common_sft_training_tuning_v1"
DIAGNOSTIC_ARTIFACT = "critic_fit_diagnostic_v1"
COMPARISON_ARTIFACT = "actor_bf16_hf_v1"
CRITIC_LRS = (1e-5, 1e-6)
_SEALED_INPUTS = (
    "settings.json",
    "model_identity.json",
    "input_preflight.json",
    "prepared/manifest.json",
    "selection.json",
    "source_identity.json",
    "plan.md",
)


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    arm: str
    eta: float
    critic_lr: float
    rounds: int

    @property
    def zero(self):
        return self.rounds == 0


def _valid_lr(value):
    if type(value) not in (int, float) or value not in CRITIC_LRS:
        raise ValueError("critic LR must be one of the frozen diagnostic candidates 1e-5/1e-6")
    return float(value)


def _same(left, right):
    # JSON contracts distinguish booleans and numbers, unlike Python dict equality.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def settings(model, *, phase, critic_lr=None):
    """Construct the entire phase contract; CLI overrides never extend its budget."""
    if model.get("source") != "local_export" or model.get("revision") is not None:
        raise ValueError("training tuning requires a pinned common local SFT export")
    if phase not in {"diagnostic", "comparison"}:
        raise ValueError("unknown training tuning phase")
    if phase == "diagnostic":
        if critic_lr is not None:
            raise ValueError("diagnostic phase compares both fixed critic LRs")
        selected_lr, rounds, epochs = None, 2, 1
    else:
        selected_lr, rounds, epochs = _valid_lr(critic_lr), 8, 2
    value = restart.settings()
    value.update(
        artifact_profile=(DIAGNOSTIC_ARTIFACT if phase == "diagnostic" else COMPARISON_ARTIFACT),
        action_protocol=PROTOCOL,
        observation_profile="history_features_v1",
        critic_initialization="cpu_seed_v1",
        m2_residual_eta=0.0,
        plan="docs/2026-09-14_M2_TRAINING_TUNING_PLAN.md",
        training_tuning={
            "profile": PROFILE,
            "phase": phase,
            "selected_critic_learning_rate": selected_lr,
            "evaluation_steps": [2] if phase == "diagnostic" else [2, 4, 6, 8],
            "automatic_retry": False,
            "resume_supported": False,
            "adaptive_gate": False,
        },
    )
    value["models"]["screening"] = copy.deepcopy(model)
    value["training"].update(
        episodes=8,
        total_epochs=epochs,
        total_training_steps=rounds,
        learning_rate=1e-6,
        critic_learning_rate=selected_lr or CRITIC_LRS[0],
        test_freq=2,
        save_freq=rounds,
        max_checkpoints=1,
        max_infrastructure_resumes=0,
    )
    return value


def read_settings(base):
    value = restart.read(base / "settings.json")
    try:
        contract = value["training_tuning"]
        expected = settings(
            value["models"]["screening"],
            phase=contract["phase"],
            critic_lr=contract["selected_critic_learning_rate"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid training tuning settings contract") from error
    if not _same(value, expected):
        raise RuntimeError("settings differ from the frozen training tuning contract")
    return value


def cells(run_settings):
    contract = run_settings["training_tuning"]
    lr = contract["selected_critic_learning_rate"] or CRITIC_LRS[0]
    baseline = CellSpec("sft0", "M1", 0.0, lr, 0)
    if contract["phase"] == "diagnostic":
        return (
            baseline,
            CellSpec("critic_lr_1e5", "M1", 0.0, 1e-5, 2),
            CellSpec("critic_lr_1e6", "M1", 0.0, 1e-6, 2),
        )
    return (
        baseline,
        CellSpec("m1", "M1", 0.0, lr, 8),
        CellSpec("m2_eta1", "M2", 1.0, lr, 8),
        CellSpec("m2_eta025", "M2", 0.25, lr, 8),
    )


def _cell(run_settings, cell_id):
    matches = [spec for spec in cells(run_settings) if spec.cell_id == cell_id]
    if len(matches) != 1:
        raise ValueError("unregistered tuning cell")
    return matches[0]


def cell_settings(run_settings, cell_id):
    spec = _cell(run_settings, cell_id)
    value = copy.deepcopy(run_settings)
    value["training_tuning"]["cell"] = asdict(spec)
    value["m2_residual_eta"] = spec.eta
    value["training"].update(critic_learning_rate=spec.critic_lr)
    if spec.zero:
        value["training"].update(
            total_epochs=1,
            total_training_steps=1,
            test_freq=1,
            save_freq=1,
        )
    return value


def _selection(*, phase, critic_lr, diagnostic_directory, selection_reason, model_revision):
    if phase == "diagnostic":
        if diagnostic_directory is not None or selection_reason is not None:
            raise ValueError("diagnostic phase must not contain a prior selection")
        return {"phase": phase, "selected_critic_learning_rate": None}
    _valid_lr(critic_lr)
    if diagnostic_directory is None or not selection_reason or not selection_reason.strip():
        raise ValueError(
            "comparison requires completed diagnostics and an explicit selection reason"
        )
    diagnostic_directory = Path(diagnostic_directory).expanduser().resolve()
    previous = read_settings(diagnostic_directory)
    if previous["training_tuning"]["phase"] != "diagnostic":
        raise RuntimeError("selection source is not a diagnostic phase")
    identity = restart.read(diagnostic_directory / "model_identity.json")
    if identity["model_revision"] != model_revision:
        raise RuntimeError("selected diagnostics used a different SFT identity")
    state = restart.read(diagnostic_directory / "state.json")
    expected = cells(previous)
    if state.get("status") != "COMPLETE" or len(state.get("cells", [])) != len(expected):
        raise RuntimeError("both critic diagnostics and their SFT0 must complete before selection")
    sources = ["settings.json", "model_identity.json", "state.json", "run_contract.json"]
    for spec, recorded in zip(expected, state["cells"], strict=True):
        relative = f"{spec.cell_id}/summary.json"
        summary = restart.read(diagnostic_directory / relative)
        if (
            summary != recorded
            or summary.get("status") != "COMPLETE"
            or summary.get("cell") != asdict(spec)
            or summary.get("actual_rounds") != spec.rounds
            or summary.get("model_revision") != model_revision
        ):
            raise RuntimeError("diagnostic cell evidence does not match its frozen schedule")
        sources.append(relative)
        if not spec.zero:
            probe_files = sorted(
                (diagnostic_directory / spec.cell_id / "artifacts/critic_fit").glob("*.jsonl")
            )
            if not probe_files:
                raise RuntimeError("diagnostic selection requires the actual critic fit evidence")
            validate_fit_evidence(
                diagnostic_directory / spec.cell_id,
                spec=spec,
                execution_identity=summary["execution_identity_sha256"],
            )
            sources.extend(str(path.relative_to(diagnostic_directory)) for path in probe_files)
    return {
        "phase": phase,
        "selected_critic_learning_rate": float(critic_lr),
        "reason": selection_reason.strip(),
        "diagnostic_directory": str(diagnostic_directory),
        "diagnostic_model_revision": model_revision,
        "source_sha256": {name: restart.digest(diagnostic_directory / name) for name in sources},
        "selected_at_SGT": restart.now(),
    }


def prepare(
    base,
    *,
    model_path,
    merge_receipt_sha256,
    phase,
    critic_lr=None,
    diagnostic_directory=None,
    selection_reason=None,
):
    model = {
        "source": "local_export",
        "name": str(Path(model_path).expanduser().resolve()),
        "merge_receipt_sha256": merge_receipt_sha256,
    }
    name, identity, path = _resolve_exact_model_snapshot({"model": model})
    model["name"] = name
    run_settings = settings(model, phase=phase, critic_lr=critic_lr)
    selection = _selection(
        phase=phase,
        critic_lr=critic_lr,
        diagnostic_directory=diagnostic_directory,
        selection_reason=selection_reason,
        model_revision=identity,
    )
    restart.prepare(base)
    atomic_json(base / "settings.json", run_settings)
    atomic_json(base / "selection.json", selection)
    atomic_json(
        base / "model_identity.json",
        {
            "model": name,
            "model_revision": identity,
            "model_snapshot_path": str(path),
            "model_source": "local_export",
            "merge_receipt_sha256": merge_receipt_sha256,
        },
    )


async def input_preflight(base):
    """Replay all twelve complete windows under the shared real tokenizer on CPU."""
    import pandas as pd
    from agent_r1.env.base import Action
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    run_settings = read_settings(base)
    _, identity, snapshot = _resolve_exact_model_snapshot(
        {"model": run_settings["models"]["screening"]}
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    records = []
    for dataset, expected in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        rows = pd.read_parquet(base / "prepared" / f"{dataset}.parquet").to_dict("records")
        if [row["extra_info"]["episode_id"] for row in rows] != expected:
            raise RuntimeError("prepared episode order differs from the frozen Train split")
        if [row["extra_info"]["index"] for row in rows] != list(range(len(expected))):
            raise RuntimeError("prepared sample indices differ from the epoch-repeating grid")
        for row in rows:
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(
                observation_profile="history_features_v1", reward_mode="finance_trajectory"
            )
            if len(kwargs["episode_dates"]) != 21:
                raise RuntimeError("tuning requires the original 21-session windows")
            env = TradingEnv(**kwargs)
            obs = env.reset(raw_prompt=list(row["prompt"]))
            counts = []
            for session in range(21):
                messages = apply_screening_action_protocol(obs.messages, PROTOCOL)
                tokens = tokenizer.apply_chat_template(
                    messages,
                    tools=env.tool_schemas,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                )
                counts.append(len(tokens))
                if len(tokens) > 8192:
                    raise RuntimeError("common SFT tuning prompt exceeds 8192 tokens")
                if session < 20:
                    obs, _, done, info = await env.step(
                        Action(text='<tool_call>{"name":"hold","arguments":{}}</tool_call>')
                    )
                    if info.get("invalid_action") or done != (session == 19):
                        raise RuntimeError("CPU full-window replay failed")
            records.append(
                {
                    "episode": row["extra_info"]["episode_id"],
                    "max_prompt_tokens": max(counts),
                    "complete": env.episode_complete,
                }
            )
    atomic_json(
        base / "input_preflight.json",
        {
            "status": "PASSED",
            "at_SGT": restart.now(),
            "windows": records,
            "model_revision": identity,
            "tokenizer_path": str(snapshot),
            "action_protocol": PROTOCOL,
            "enable_thinking": False,
        },
    )


def _source_files():
    files = {p for folder in ("src", "runtime_hooks") for p in (REPO_ROOT / folder).rglob("*.py")}
    files.update(
        REPO_ROOT / name
        for name in (
            "configs/base.yaml",
            "configs/m2_screening_r5_generation_pilot.yaml",
        )
    )
    return {str(path.relative_to(REPO_ROOT)): path for path in files}


def _check_selection(base, run_settings):
    selected = restart.read(base / "selection.json")
    phase = run_settings["training_tuning"]["phase"]
    lr = run_settings["training_tuning"]["selected_critic_learning_rate"]
    if selected.get("phase") != phase or selected.get("selected_critic_learning_rate") != lr:
        raise RuntimeError("critic LR selection differs from phase settings")
    if phase == "comparison":
        root = Path(selected["diagnostic_directory"])
        for name, digest in selected["source_sha256"].items():
            if restart.digest(root / name) != digest:
                raise RuntimeError("diagnostic evidence changed after critic LR selection")


def seal(base):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("phase is already sealed or started; use a new run directory")
    run_settings = read_settings(base)
    _check_selection(base, run_settings)
    preflight = restart.read(base / "input_preflight.json")
    windows = preflight.get("windows", [])
    if (
        preflight.get("status") != "PASSED"
        or [row.get("episode") for row in windows] != TRAIN_IDS + EVAL_IDS
        or any(row.get("complete") is not True for row in windows)
    ):
        raise RuntimeError("all twelve complete CPU window preflights are required before sealing")
    restart.seal(base)
    shutil.copy2(REPO_ROOT / run_settings["plan"], base / "plan.md")
    atomic_json(
        base / "run_contract.json",
        {
            "profile": PROFILE,
            "sealed_at_SGT": restart.now(),
            "inputs": {name: restart.digest(base / name) for name in _SEALED_INPUTS},
        },
    )
    check_contract(base)


def check_contract(base):
    contract = restart.read(base / "run_contract.json")
    if contract.get("profile") != PROFILE or set(contract["inputs"]) != set(_SEALED_INPUTS):
        raise RuntimeError("incomplete tuning source/input identity")
    for name, expected in contract["inputs"].items():
        if restart.digest(base / name) != expected:
            raise RuntimeError(f"tuning input changed after sealing: {name}")
    sources = restart.read(base / "source_identity.json")
    if set(sources) != set(_source_files()):
        raise RuntimeError("source file set changed after tuning freeze")
    for name, path in _source_files().items():
        if restart.digest(path) != sources[name]:
            raise RuntimeError(f"source changed after tuning freeze: {name}")
    manifest = restart.read(base / "prepared/manifest.json")
    for name, expected_ids in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        item = manifest["datasets"][name]
        if item["episode_ids"] != expected_ids or item["row_count"] != len(expected_ids):
            raise RuntimeError("prepared data no longer matches the frozen Train grid")
    for item in [*manifest["datasets"].values(), manifest["calibration"]]:
        if restart.digest(base / "prepared" / item["path"]) != item["sha256"]:
            raise RuntimeError("prepared data changed after tuning freeze")
    run_settings = read_settings(base)
    _check_selection(base, run_settings)
    _, identity, path = _resolve_exact_model_snapshot(
        {"model": run_settings["models"]["screening"]}
    )
    recorded = restart.read(base / "model_identity.json")
    preflight = restart.read(base / "input_preflight.json")
    if (
        recorded["model_revision"] != identity
        or recorded["model_snapshot_path"] != str(path)
        or preflight["model_revision"] != identity
        or preflight["action_protocol"] != PROTOCOL
        or preflight["enable_thinking"] is not False
    ):
        raise RuntimeError("model or rendered prompt identity differs from the tuning contract")


def make_config(base, manifest, cell_id):
    run_settings = cell_settings(read_settings(base), cell_id)
    spec = run_settings["training_tuning"]["cell"]
    config = build_cell_config(
        settings=run_settings,
        manifest=manifest,
        prepared_dir=base / "prepared",
        arm=spec["arm"],
        stage="zero-shot" if spec["rounds"] == 0 else "screening",
        seed=42,
        cell_dir=base / cell_id,
    )
    phase = run_settings["training_tuning"]["phase"]
    config["run"].update(name=f"training-tuning-{phase}-{cell_id}-s42", kind=PROFILE)
    config["screening"].update(tuning_phase=phase, tuning_cell_id=cell_id)
    config["training"].update(run_settings["training"], critic_initialization="cpu_seed_v1")
    return config


def build_command(*, config, run_settings, hashes, cell_dir):
    """Reject drift before building the real trainer command or installing any gate."""
    contract = run_settings["training_tuning"]
    phase_settings = settings(
        run_settings["models"]["screening"],
        phase=contract["phase"],
        critic_lr=contract["selected_critic_learning_rate"],
    )
    spec = _cell(phase_settings, contract["cell"]["cell_id"])
    if not _same(run_settings, cell_settings(phase_settings, spec.cell_id)):
        raise ValueError("cell settings differ from the frozen tuning condition")
    expected_training = {**run_settings["training"], "critic_initialization": "cpu_seed_v1"}
    if any(
        not _same(config["training"].get(key), value) for key, value in expected_training.items()
    ):
        raise ValueError(
            "resolved training budget, LR, or initialization differs from tuning settings"
        )
    expected_screening = {
        "arm": spec.arm,
        "stage": "zero-shot" if spec.zero else "screening",
        "artifact_profile": run_settings["artifact_profile"],
        "residual_eta": spec.eta,
        "tuning_cell_id": spec.cell_id,
        "tuning_phase": contract["phase"],
    }
    if any(
        not _same(config["screening"].get(key), value) for key, value in expected_screening.items()
    ):
        raise ValueError("resolved cell arm, eta, or artifact differs from tuning settings")
    if config["model"] != run_settings["models"]["screening"] or config["run"]["seed"] != 42:
        raise ValueError("tuning cells must share the frozen SFT and seed")
    command = build_cell_command(
        config=config,
        settings=run_settings,
        hashes=hashes,
        cell_dir=cell_dir,
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
            "critic.model.tokenizer_path": config["model"]["name"],
            "+actor_rollout_ref.ref.model.path": config["model"]["name"],
            "data.validation_shuffle": False,
        },
    )
    if any("FINANCE_LEARNING_" in argument for argument in command):
        raise RuntimeError("tuning must not install the old adaptive gate")
    return command


def validate_fit_evidence(cell, *, spec, execution_identity):
    """Require a matching same-batch probe for every real critic update, including bad fits."""
    directory = cell / "artifacts/critic_fit"
    records = read_jsonl(directory)
    if spec.zero:
        if records:
            raise RuntimeError("SFT0 unexpectedly contains critic update fit evidence")
        return {"updates": 0, "policy_steps": [], "source_sha256": {}}
    if len(records) != spec.rounds or sorted(row.get("policy_step", -1) for row in records) != list(
        range(1, spec.rounds + 1)
    ):
        raise RuntimeError("missing or duplicate per-update critic fit evidence")
    records.sort(key=lambda row: row["policy_step"])
    diagnostics = {
        row["policy_step"]: row for row in read_jsonl(cell / "artifacts/sibling_diagnostics")
    }
    critic_updates = sorted(
        (row for row in read_jsonl(cell / "artifacts/update_probe") if row.get("role") == "critic"),
        key=lambda row: row.get("global_update_count", -1),
    )
    if len(critic_updates) != spec.rounds:
        raise RuntimeError("critic fit evidence lacks matching optimizer updates")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    for step, (row, update) in enumerate(zip(records, critic_updates, strict=True), start=1):
        expected = {
            "schema_version": 1,
            "phase": "critic_same_batch_pre_post",
            "status": "COMPLETE",
            "policy_step": step,
            "update_call_index": step,
            "arm": spec.arm,
            "execution_identity_sha256": execution_identity,
            "planned_mini_batch_count": 1,
            "ppo_epochs": 1,
            "extra_forward_passes": 1,
            "returns_recomputed": False,
        }
        diagnostic = diagnostics.get(step, {})
        if (
            any(row.get(key) != value for key, value in expected.items())
            or row.get("real_step_count", 0) <= 0
            or row.get("real_step_count") != diagnostic.get("real_step_count")
            or row.get("value_token_count") != row.get("real_step_count")
            or row.get("padding_row_count") != diagnostic.get("padding_row_count")
            or diagnostic.get("eta") != spec.eta
            or row.get("forward_meta_info")
            != {
                "micro_batch_size": 1,
                "max_token_len": 32768,
                "use_dynamic_bsz": False,
            }
            or row.get("dropout", {}).get("any_nonzero") is not False
            or row.get("dropout", {}).get("fit_comparison_uses_same_mode") is not True
            or row.get("mode_restored") is not True
            or not row.get("critic_learning_rates")
            or any(value != spec.critic_lr for value in row["critic_learning_rates"])
        ):
            raise RuntimeError("critic fit probe contract, condition, or row mask mismatch")
        gradients = row.get("optimizer_grad_norms_before_clip", [])
        if (
            len(gradients) != 1
            or not finite(gradients[0])
            or row.get("pid") != update.get("pid")
            or not finite(update.get("grad_norm"))
            or not math.isclose(gradients[0], update["grad_norm"], rel_tol=1e-7, abs_tol=1e-8)
        ):
            raise RuntimeError("critic fit gradient does not match its real optimizer update")
        for phase in ("pre", "post"):
            fit = row.get(phase, {})
            scalars = (
                "mse",
                "half_mse",
                "rmse",
                "mean_bias_value_minus_return",
                "half_squared_mean_bias",
                "error_std_population",
            )
            if any(not finite(fit.get(key)) for key in scalars):
                raise RuntimeError("critic fit is missing finite pre/post scalar statistics")
            for field in ("values", "returns"):
                if any(
                    not finite(fit.get(field, {}).get(key))
                    for key in (
                        "mean",
                        "std_population",
                        "min",
                        "max",
                    )
                ):
                    raise RuntimeError("critic fit is missing finite value/target statistics")
            if fit.get("explained_variance") is not None and not finite(fit["explained_variance"]):
                raise RuntimeError("critic fit explained variance must be finite when defined")
        if row["pre"]["returns"] != row["post"]["returns"]:
            raise RuntimeError("critic fit target changed between pre/post measurements")
        if not finite(row.get("extra_forward_seconds_synchronized")):
            raise RuntimeError("critic fit extra forward timing is missing or non-finite")
    return {
        "updates": len(records),
        "policy_steps": [row["policy_step"] for row in records],
        "source_sha256": {
            str(path.relative_to(cell)): restart.digest(path)
            for path in sorted(directory.glob("*.jsonl"))
        },
    }


def run_cell(base, cell_id):
    check_contract(base)
    phase_settings = read_settings(base)
    spec = _cell(phase_settings, cell_id)
    run_settings = cell_settings(phase_settings, cell_id)
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
    atomic_json(cell / "settings.json", run_settings)
    command = build_command(config=config, run_settings=run_settings, hashes=hashes, cell_dir=cell)
    runtime_path = base / "runtime_preflight.json"
    if not runtime_path.exists():
        atomic_json(runtime_path, verify_gpu_runtime_and_model(config))
    runtime = restart.read(runtime_path)
    recorded = restart.read(base / "model_identity.json")
    if any(runtime[key] != recorded[key] for key in ("model_revision", "model_snapshot_path")):
        raise RuntimeError("cached runtime preflight belongs to another SFT")
    command = _finalize_model_paths(
        command,
        arm_id=config["arm"]["id"],
        model_snapshot_path=runtime["model_snapshot_path"],
    )
    environment = _training_environment(
        agent_path=agent,
        probe_root=cell / "artifacts/update_probe",
        c_compiler=runtime["c_compiler"],
    )
    environment["PYTHONUNBUFFERED"] = "1"
    for key in list(environment):
        if key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_")):
            environment.pop(key)
    atomic_json(cell / "command.json", command)
    options = dict(argument.lstrip("+").split("=", 1) for argument in command[3:])
    identity = options["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"]
    eval_steps = [0] if spec.zero else phase_settings["training_tuning"]["evaluation_steps"]
    summary = {
        "status": "RUNNING",
        "cell": asdict(spec),
        "started_at_SGT": restart.now(),
        "requested_rounds": spec.rounds,
        "model_revision": recorded["model_revision"],
        "artifact_profile": run_settings["artifact_profile"],
        "optimizer_included": False,
        "resume_supported": False,
        "evaluation_steps": eval_steps,
        "execution_identity_sha256": identity,
    }
    atomic_json(cell / "summary.json", summary)
    try:
        code = capture_process(command, environment=environment, cell_dir=cell)
        summary["returncode"] = code
        if code:
            raise RuntimeError(f"{cell_id} trainer exited {code}; inspect trainer.log")
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
            expected_eval_steps=eval_steps,
            expected_eta=spec.eta,
            artifact_profile=run_settings["artifact_profile"],
        )
        summary["critic_fit"] = validate_fit_evidence(
            cell,
            spec=spec,
            execution_identity=identity,
        )
        head = restart.read(cell / "artifacts/critic_initialization/head-rank0.json")
        if (
            head.get("seed") != 42
            or head.get("mode") != "cpu_seed_v1"
            or head.get("arm") != spec.arm
            or head.get("execution_identity_sha256") != identity
            or head.get("model_path") != str(Path(runtime["model_snapshot_path"]).resolve())
            or head.get("phase") != "critic_head_after_hf_load_before_fsdp"
        ):
            raise RuntimeError("actual critic head initialization differs from tuning settings")
        summary["critic_initial_head_sha256"] = head["head_sha256"]
        if not spec.zero:
            baseline = restart.read(base / "sft0/summary.json")
            if baseline["status"] != "COMPLETE" or (
                head["head_sha256"] != baseline["critic_initial_head_sha256"]
            ):
                raise RuntimeError("initial critic head differs from the fresh SFT0")
        check_contract(base)
        summary.update(actual_rounds=spec.rounds, status="COMPLETE")
        return summary
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_at_SGT"] = restart.now()
        atomic_json(cell / "summary.json", summary)


def run(base):
    if (base / "state.json").exists():
        raise RuntimeError("tuning phase already has state; no implicit retry or resume")
    check_contract(base)
    run_settings = read_settings(base)
    schedule = cells(run_settings)
    state = {
        "status": "RUNNING",
        "started_at_SGT": restart.now(),
        "cells": [],
        "phase": run_settings["training_tuning"]["phase"],
        "schedule": [asdict(spec) for spec in schedule],
        "automatic_retry": False,
        "resume_supported": False,
        "adaptive_gate": False,
        "train_episode_ids": TRAIN_IDS,
        "evaluation_episode_ids": EVAL_IDS,
        "evaluation_steps": run_settings["training_tuning"]["evaluation_steps"],
    }
    try:
        for index, spec in enumerate(schedule):
            remaining_exports = (
                sum(not cell.zero for cell in schedule[index:])
                if run_settings["artifact_profile"] == COMPARISON_ARTIFACT
                else 0
            )
            free = shutil.disk_usage(base).free
            required = (9 * remaining_exports + 16) * 1024**3
            if free < required:
                raise RuntimeError("insufficient space for fixed tuning artifacts; no cleanup")
            state.update(current_cell=spec.cell_id, free_disk_bytes=free)
            atomic_json(base / "state.json", state)
            state["cells"].append(run_cell(base, spec.cell_id))
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["updated_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--phase", choices=("diagnostic", "comparison"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--merge-receipt-sha256")
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--diagnostic-directory", type=Path)
    parser.add_argument("--selection-reason")
    args = parser.parse_args()
    base = args.directory.expanduser().resolve()
    if args.command == "prepare":
        if args.model is None or args.merge_receipt_sha256 is None or args.phase is None:
            parser.error("prepare requires --model, --merge-receipt-sha256, and --phase")
        prepare(
            base,
            model_path=args.model,
            merge_receipt_sha256=args.merge_receipt_sha256,
            phase=args.phase,
            critic_lr=args.critic_learning_rate,
            diagnostic_directory=args.diagnostic_directory,
            selection_reason=args.selection_reason,
        )
    elif args.command == "preflight":
        asyncio.run(input_preflight(base))
    elif args.command == "seal":
        seal(base)
    else:
        run(base)


if __name__ == "__main__":
    main()
