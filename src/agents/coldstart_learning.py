"""Fresh fixed-two-round M1/M2 research runs from one pinned common SFT export."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
from pathlib import Path

from src.agents import learning_restart as restart
from src.agents.launcher import (
    _finalize_model_paths,
    _resolve_exact_model_snapshot,
    _training_environment,
)
from src.agents.metadata import collect_run_hashes
from src.agents.screening_checkpoint import RESEARCH_PROFILE
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

TRAIN_IDS, EVAL_IDS = restart.TRAIN_IDS, restart.EVAL_IDS
ROUNDS = 2
BASE = REPO_ROOT / "runs/m2-common-sft-fixed2-20260913"
PROTOCOL = "single_tool_portfolio_v1"
_SEALED_INPUTS = (
    "settings.json", "model_identity.json", "input_preflight.json", "prepared/manifest.json",
)


def settings(model):
    """Reuse the validated runtime budget, replacing only declared shared inputs."""
    if model.get("source") != "local_export" or model.get("revision") is not None:
        raise ValueError("common SFT learning requires an explicitly pinned local export")
    value = restart.settings()
    value.update(
        artifact_profile=RESEARCH_PROFILE,
        action_protocol=PROTOCOL,
        critic_initialization="cpu_seed_v1",
        observation_profile="history_features_v1",
        plan="docs/2026-09-14_M2_COMMON_SFT_SHORT_RL_PLAN.md",
    )
    value["models"]["screening"] = copy.deepcopy(model)
    value["training"].update(
        total_training_steps=ROUNDS,
        save_freq=ROUNDS,
        max_checkpoints=1,
        max_infrastructure_resumes=0,
    )
    return value


def read_settings(base):
    value = restart.read(base / "settings.json")
    if value != settings(value["models"]["screening"]):
        raise RuntimeError("settings differ from the fixed shared SFT research contract")
    return value


def prepare(base, *, model_path, merge_receipt_sha256):
    model = {
        "source": "local_export",
        "name": str(model_path),
        "merge_receipt_sha256": merge_receipt_sha256,
    }
    name, identity, path = _resolve_exact_model_snapshot({"model": model})
    model["name"] = name
    # Reuse only the existing Train splitter; no old checkpoint or optimizer is loaded.
    restart.prepare(base)
    atomic_json(base / "settings.json", settings(model))
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
    """Render the actual shared protocol/tokenizer over complete CPU hold replays."""
    import pandas as pd
    from agent_r1.env.base import Action
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    run_settings = read_settings(base)
    _, identity, snapshot = _resolve_exact_model_snapshot(
        {
            "model": run_settings["models"]["screening"],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    records = []
    for dataset, expected_ids in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        rows = pd.read_parquet(base / "prepared" / f"{dataset}.parquet").to_dict("records")
        if [row["extra_info"]["episode_id"] for row in rows] != expected_ids:
            raise RuntimeError("prepared episode order differs from the declared Train split")
        for row in rows:
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(
                observation_profile=run_settings["observation_profile"],
                reward_mode="finance_trajectory",
            )
            if len(kwargs["episode_dates"]) != 21:
                raise RuntimeError("fixed2 research requires the original 21-session windows")
            env = TradingEnv(**kwargs)
            obs = env.reset(raw_prompt=list(row["prompt"]))
            counts = []
            for session in range(21):
                messages = apply_screening_action_protocol(obs.messages, PROTOCOL)
                ids = tokenizer.apply_chat_template(
                    messages,
                    tools=env.tool_schemas,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                )
                counts.append(len(ids))
                if len(ids) > 8192:
                    raise RuntimeError("common SFT RL prompt exceeds 8192 tokens")
                if session < 20:
                    obs, _, done, info = await env.step(
                        Action(
                            text='<tool_call>{"name":"hold","arguments":{}}</tool_call>',
                        )
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


def seal(base):
    read_settings(base)
    if restart.read(base / "input_preflight.json").get("status") != "PASSED":
        raise RuntimeError("complete CPU input preflight is required before sealing")
    restart.seal(base)
    atomic_json(
        base / "run_contract.json",
        {
            "sealed_at_SGT": restart.now(),
            "inputs": {name: restart.digest(base / name) for name in _SEALED_INPUTS},
        },
    )


def check_contract(base):
    restart.check_sources(base)
    contract = restart.read(base / "run_contract.json")
    if set(contract["inputs"]) != set(_SEALED_INPUTS):
        raise RuntimeError("incomplete fixed2 input identity")
    for name, expected in contract["inputs"].items():
        if restart.digest(base / name) != expected:
            raise RuntimeError(f"fixed2 input changed after sealing: {name}")
    run_settings = read_settings(base)
    _, identity, path = _resolve_exact_model_snapshot(
        {
            "model": run_settings["models"]["screening"],
        }
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
        raise RuntimeError("model identity or rendered input preflight differs from the run")


def make_config(base, manifest, arm, *, zero=False):
    run_settings = read_settings(base)
    cell = base / ("base" if zero else arm.lower())
    cfg = build_cell_config(
        settings=run_settings,
        manifest=manifest,
        prepared_dir=base / "prepared",
        arm=arm,
        stage="zero-shot" if zero else "screening",
        seed=42,
        cell_dir=cell,
    )
    cfg["run"].update(name=f"common-sft-fixed2-{cell.name}-s42", kind="common_sft_fixed2")
    cfg["training"].update(
        episodes=8,
        total_epochs=1,
        total_training_steps=1 if zero else ROUNDS,
        critic_initialization="cpu_seed_v1",
        save_freq=1 if zero else ROUNDS,
    )
    return cfg


def build_command(*, config, run_settings, hashes, cell_dir):
    if (
        config["training"].get("critic_initialization") != "cpu_seed_v1"
        or run_settings.get("critic_initialization") != "cpu_seed_v1"
        or run_settings["resources"]["profile"] != "pageable_adam_v1"
        or run_settings.get("artifact_profile") != RESEARCH_PROFILE
    ):
        raise ValueError("fixed2 requires the shared initialization and research artifact profile")
    zero = config["screening"]["stage"] == "zero-shot"
    if config["training"]["total_training_steps"] != (1 if zero else ROUNDS):
        raise ValueError("fixed2 budget cannot adapt or extend")
    command = build_cell_command(
        config=config,
        settings=run_settings,
        hashes=hashes,
        cell_dir=cell_dir,
    )
    model_path = config["model"]["name"]
    ray_env_prefix = "+ray_kwargs.ray_init.runtime_env.env_vars."
    command = set_overrides(
        command,
        {
            ray_env_prefix + "FINANCE_SCREENING_CRITIC_INIT": "cpu_seed_v1",
            ray_env_prefix + "FINANCE_SCREENING_CRITIC_INIT_EVIDENCE_DIR": str(
                cell_dir / "artifacts/critic_initialization"
            ),
            "critic.model.tokenizer_path": model_path,
            "+actor_rollout_ref.ref.model.path": model_path,
        },
    )
    if any("FINANCE_LEARNING_" in argument for argument in command):
        raise RuntimeError("fixed2 must not install the old adaptive M2 gate")
    return command


def run_cell(base, arm, *, zero=False):
    check_contract(base)
    manifest = restart.read(base / "prepared/manifest.json")
    run_settings = read_settings(base)
    cell = base / ("base" if zero else arm.lower())
    cell.mkdir(exist_ok=False)
    cfg = make_config(base, manifest, arm, zero=zero)
    agent = REPO_ROOT / cfg["framework"]["agent_r1_path"]
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent,
        data_version=cfg["data"]["version"],
        expected_agent_r1_commit=cfg["framework"]["agent_r1_commit"],
        require_clean_repo=False,
    )
    atomic_json(cell / "resolved_config.json", cfg)
    atomic_json(cell / "settings.json", run_settings)
    command = build_command(
        config=cfg,
        run_settings=run_settings,
        hashes=hashes,
        cell_dir=cell,
    )
    runtime_path = base / "runtime_preflight.json"
    if not runtime_path.exists():
        atomic_json(runtime_path, verify_gpu_runtime_and_model(cfg))
    runtime = restart.read(runtime_path)
    recorded = restart.read(base / "model_identity.json")
    if any(runtime[key] != recorded[key] for key in ("model_revision", "model_snapshot_path")):
        raise RuntimeError("cached GPU runtime preflight belongs to another initial model")
    command = _finalize_model_paths(
        command,
        arm_id=cfg["arm"]["id"],
        model_snapshot_path=runtime["model_snapshot_path"],
    )
    environment = _training_environment(
        agent_path=agent,
        probe_root=cell / "artifacts/update_probe",
        c_compiler=runtime["c_compiler"],
    )
    environment["PYTHONUNBUFFERED"] = "1"
    for key in list(environment):
        if key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_")):
            environment.pop(key)
    atomic_json(cell / "command.json", command)
    options = dict(argument.lstrip("+").split("=", 1) for argument in command[3:])
    identity = options["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"]
    summary = {
        "status": "RUNNING",
        "arm": "SFT_BASELINE" if zero else arm,
        "started_at_SGT": restart.now(),
        "requested_rounds": 0 if zero else ROUNDS,
        "model_revision": recorded["model_revision"],
        "artifact_profile": RESEARCH_PROFILE,
        "optimizer_included": False,
        "resume_supported": False,
    }
    atomic_json(cell / "summary.json", summary)
    try:
        code = capture_process(command, environment=environment, cell_dir=cell)
        summary["returncode"] = code
        if code:
            raise RuntimeError(f"{cell.name} trainer exited {code}; inspect trainer.log")
        summary["evidence"] = validate_cell_evidence(
            cell,
            stage=cfg["screening"]["stage"],
            expected_rounds=1 if zero else ROUNDS,
            arm=arm,
            seed=42,
            train_episode_ids=TRAIN_IDS,
            eval_episode_ids=EVAL_IDS,
            execution_identity_sha256=identity,
            calibration_payload_sha256=manifest["calibration"]["payload_sha256"],
            expected_eval_count=4,
            artifact_profile=RESEARCH_PROFILE,
        )
        head = restart.read(cell / "artifacts/critic_initialization/head-rank0.json")
        if (
            head.get("seed") != 42
            or head.get("mode") != "cpu_seed_v1"
            or head.get("arm") != arm
            or head.get("execution_identity_sha256") != identity
            or head.get("model_path") != str(Path(runtime["model_snapshot_path"]).resolve())
            or head.get("phase") != "critic_head_after_hf_load_before_fsdp"
        ):
            raise RuntimeError("actual critic head initialization evidence differs from the run")
        summary["critic_initial_head_sha256"] = head["head_sha256"]
        if not zero:
            baseline_head = restart.read(base / "base/summary.json")["critic_initial_head_sha256"]
            if head["head_sha256"] != baseline_head:
                raise RuntimeError("initial critic head differs from the common SFT baseline")
        check_contract(base)
        summary["actual_rounds"] = 0 if zero else ROUNDS
        summary["status"] = "COMPLETE"
        return summary
    except BaseException as error:
        summary.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_at_SGT"] = restart.now()
        atomic_json(cell / "summary.json", summary)


def run(base):
    if (base / "state.json").exists():
        raise RuntimeError("fixed2 run already has state; no implicit retry or resume")
    check_contract(base)
    state = {
        "status": "RUNNING",
        "started_at_SGT": restart.now(),
        "cells": [],
        "training_updates_per_arm": ROUNDS,
        "adaptive_gate": False,
        "automatic_retry": False,
        "resume_supported": False,
        "effective_train_episode_ids": TRAIN_IDS[: ROUNDS * 2],
        "evaluation_episode_ids": EVAL_IDS,
    }
    try:
        for arm, zero, remaining in (("M1", True, 2), ("M1", False, 2), ("M2", False, 1)):
            # About31.4GiB/model+extra pair, with a separate16GiB working reserve.
            free = shutil.disk_usage(base).free
            if free < (32 * remaining + 16) * 1024**3:
                raise RuntimeError("insufficient space for retained research artifacts; no cleanup")
            state.update(current_cell="base" if zero else arm.lower(), free_disk_bytes=free)
            atomic_json(base / "state.json", state)
            state["cells"].append(run_cell(base, arm, zero=zero))
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
    parser.add_argument("--directory", type=Path, default=BASE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--merge-receipt-sha256")
    args = parser.parse_args()
    args.directory = args.directory.expanduser().resolve()
    if args.command == "prepare":
        if args.model is None or args.merge_receipt_sha256 is None:
            parser.error("prepare requires --model and --merge-receipt-sha256")
        prepare(
            args.directory, model_path=args.model, merge_receipt_sha256=args.merge_receipt_sha256
        )
    elif args.command == "preflight":
        asyncio.run(input_preflight(args.directory))
    elif args.command == "seal":
        seal(args.directory)
    else:
        run(args.directory)


if __name__ == "__main__":
    main()
