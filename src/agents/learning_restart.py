"""Bounded Train-only M2 learning restart using the existing training stack."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from src.agents.launcher import _finalize_model_paths, _training_environment
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

TRAIN_IDS = [f"formal-train-{i:04d}" for i in range(1, 30, 4)]
EVAL_IDS = [f"formal-train-{i:04d}" for i in range(113, 126, 4)]
BASE = REPO_ROOT / "runs/m2-learning-restart-20260913-critic-seeded-v1"


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings():
    value = yaml.safe_load(
        (REPO_ROOT / "configs/m2_screening_r5_generation_pilot.yaml").read_text()
    )
    value.update(
        arms=["M1", "M2"],
        seeds=[42],
        observation_profile="history_features_v1",
        capture_interactions=True,
        critic_initialization="cpu_seed_v1",
        plan="docs/2026-09-13_M2_LEARNING_RESTART_PLAN.md",
    )
    value["training"].update(
        episodes=8,
        total_epochs=1,
        max_prompt_length=8192,
        max_checkpoints=2,
        max_infrastructure_resumes=0,
    )
    return value


def make_config(base, manifest, arm, rounds, *, zero=False):
    cell = base / ("base" if zero else arm.lower())
    stage = "zero-shot" if zero else "screening"
    cfg = build_cell_config(
        settings=settings(),
        manifest=manifest,
        prepared_dir=base / "prepared",
        arm=arm,
        stage=stage,
        seed=42,
        cell_dir=cell,
    )
    cfg["run"].update(name=f"learning-restart-{cell.name}-s42", kind="learning_restart")
    cfg["training"].update(
        episodes=8,
        total_epochs=1,
        total_training_steps=rounds,
        critic_initialization=settings()["critic_initialization"],
        save_freq=2 if arm == "M2" and not zero else rounds,
    )
    return cfg


def prepare(base):
    import pandas as pd

    base.mkdir(exist_ok=False)
    target = base / "prepared"
    target.mkdir()
    original = REPO_ROOT / "artifacts/m2_screening"
    old = read(original / "manifest.json")
    for name in ("train",):
        assert digest(original / old["datasets"][name]["path"]) == old["datasets"][name]["sha256"]
    assert digest(original / old["calibration"]["path"]) == old["calibration"]["sha256"]
    frame = pd.read_parquet(original / "train.parquet")
    datasets = {}
    dates: dict[str, set[str]] = {}
    for name, ids in (("train", TRAIN_IDS), ("dev", EVAL_IDS)):
        rows = []
        dates[name] = set()
        for i, episode in enumerate(ids):
            matches = [
                r for r in frame.to_dict("records") if r["extra_info"]["episode_id"] == episode
            ]
            assert len(matches) == 1
            row = copy.deepcopy(matches[0])
            assert row["extra_info"]["source_split"] == "train"
            row["extra_info"].update(index=i, screening_role="learning_restart_" + name)
            env = json.loads(row["env_kwargs"])
            assert len(env["episode_dates"]) == 21
            dates[name].update(env["episode_dates"])
            rows.append(row)
        pd.DataFrame(rows).to_parquet(target / f"{name}.parquet", index=False)
        datasets[name] = {
            "path": f"{name}.parquet",
            "episode_ids": ids,
            "sha256": digest(target / f"{name}.parquet"),
            "row_count": len(ids),
        }
    assert not dates["train"] & dates["dev"]
    shutil.copy2(original / "reward_calibration.json", target / "reward_calibration.json")
    manifest = {
        "datasets": datasets,
        "calibration": old["calibration"],
        "source_manifest_sha256": digest(original / "manifest.json"),
        "source_train_sha256": digest(original / "train.parquet"),
        "eval_is_reserved_train_subset": True,
        "train_eval_dates_disjoint": True,
        "created_at_SGT": now(),
    }
    manifest["hashes"] = {
        "payload_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    }
    atomic_json(target / "manifest.json", manifest)
    atomic_json(base / "settings.json", settings())


async def input_preflight(base):
    """Exercise the real neutral observation through complete CPU hold replays."""
    import pandas as pd
    from agent_r1.env.base import Action
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import apply_screening_action_protocol
    from src.env.trading import TradingEnv

    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots"
    snapshot /= settings()["models"]["screening"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    records = []
    for dataset in ("train", "dev"):
        for row in pd.read_parquet(base / "prepared" / f"{dataset}.parquet").to_dict("records"):
            kwargs = json.loads(row["env_kwargs"])
            kwargs.update(
                observation_profile="history_features_v1", reward_mode="finance_trajectory"
            )
            env = TradingEnv(**kwargs)
            obs = env.reset(raw_prompt=list(row["prompt"]))
            counts = []
            for session in range(21):
                messages = apply_screening_action_protocol(obs.messages, "single_tool_v1")
                ids = tokenizer.apply_chat_template(
                    messages,
                    tools=env.tool_schemas,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                )
                counts.append(len(ids))
                assert len(ids) <= 8192, (row["extra_info"]["episode_id"], session, len(ids))
                if session < 20:
                    obs, _, done, info = await env.step(
                        Action(text='<tool_call>{"name":"hold","arguments":{}}</tool_call>')
                    )
                    assert not info.get("invalid_action")
                    assert done == (session == 19)
            records.append(
                {
                    "episode": row["extra_info"]["episode_id"],
                    "max_prompt_tokens": max(counts),
                    "complete": env.episode_complete,
                }
            )
    atomic_json(base / "input_preflight.json", {"status": "PASSED", "windows": records})


def seal(base):
    sources = sorted(
        {p for folder in ("src", "runtime_hooks") for p in (REPO_ROOT / folder).rglob("*.py")}
    )
    sources += [
        REPO_ROOT / "configs/base.yaml",
        REPO_ROOT / "configs/m2_screening_r5_generation_pilot.yaml",
    ]
    identities = {}
    for path in sources:
        relative = path.relative_to(REPO_ROOT)
        target = base / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        identities[str(relative)] = digest(path)
    atomic_json(base / "source_identity.json", identities)


def check_sources(base):
    for name, sha in read(base / "source_identity.json").items():
        assert digest(REPO_ROOT / name) == sha, f"source changed: {name}"
    m = read(base / "prepared/manifest.json")
    for item in list(m["datasets"].values()) + [m["calibration"]]:
        assert digest(base / "prepared" / item["path"]) == item["sha256"]



def build_restart_command(*, config, run_settings, hashes, cell_dir):
    """Build the actual fresh-run command, including shared critic initialization."""
    mode = run_settings.get("critic_initialization")
    if mode != "cpu_seed_v1":
        raise ValueError("learning restart requires critic_initialization=cpu_seed_v1")
    if config["training"].get("critic_initialization") != mode:
        raise ValueError("resolved critic initialization differs from restart settings")
    if run_settings.get("resources", {}).get("profile") != "pageable_adam_v1":
        raise ValueError("seeded critic initialization requires pageable_adam_v1 installation")
    command = build_cell_command(
        config=config, settings=run_settings, hashes=hashes, cell_dir=cell_dir,
    )
    overrides = {
        "+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_CRITIC_INIT": mode,
    }
    if config["screening"]["arm"] == "M2" and config["screening"]["stage"] != "zero-shot":
        overrides["+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_LEARNING_GATE_DIR"] = str(
            cell_dir / "gate"
        )
    return set_overrides(command, overrides)


def run_cell(base, arm, rounds, *, zero=False):
    check_sources(base)
    manifest = read(base / "prepared/manifest.json")
    cell = base / ("base" if zero else arm.lower())
    cell.mkdir(exist_ok=False)
    cfg = make_config(base, manifest, arm, rounds, zero=zero)
    stage = cfg["screening"]["stage"]
    agent = REPO_ROOT / cfg["framework"]["agent_r1_path"]
    hashes = collect_run_hashes(
        repo_root=REPO_ROOT,
        agent_r1_path=agent,
        data_version=cfg["data"]["version"],
        expected_agent_r1_commit=cfg["framework"]["agent_r1_commit"],
        require_clean_repo=False,
    )
    # This authorized research build records source bytes, including existing worktree edits.
    atomic_json(cell / "resolved_config.json", cfg)
    run_settings = settings()
    atomic_json(cell / "settings.json", run_settings)
    command = build_restart_command(
        config=cfg, run_settings=run_settings, hashes=hashes, cell_dir=cell,
    )
    runtime_path = base / "runtime_preflight.json"
    if not runtime_path.exists():
        atomic_json(runtime_path, verify_gpu_runtime_and_model(cfg))
    runtime = read(runtime_path)
    command = _finalize_model_paths(
        command, arm_id=cfg["arm"]["id"], model_snapshot_path=runtime["model_snapshot_path"]
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
    options = dict(arg.lstrip("+").split("=", 1) for arg in command[3:])
    identity = options["ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"]
    summary = {
        "status": "RUNNING",
        "arm": arm,
        "started_at_SGT": now(),
        "requested_rounds": rounds,
        "source_identity": str(base / "source_identity.json"),
    }
    atomic_json(cell / "summary.json", summary)
    try:
        code = capture_process(command, environment=environment, cell_dir=cell)
        summary["returncode"] = code
        if code:
            raise RuntimeError(f"{cell.name} trainer exited {code}; see trainer.log")
        actual = rounds
        if arm == "M2" and not zero:
            gate = read(cell / "gate/decision.json")
            actual = gate["target_rounds"]
        summary["actual_rounds"] = 0 if zero else actual
        summary["evidence"] = validate_cell_evidence(
            cell,
            stage=stage,
            expected_rounds=actual,
            arm=arm,
            seed=42,
            train_episode_ids=TRAIN_IDS,
            eval_episode_ids=EVAL_IDS,
            execution_identity_sha256=identity,
            calibration_payload_sha256=manifest["calibration"]["payload_sha256"],
            expected_eval_count=4,
        )
        check_sources(base)
        summary["status"] = "COMPLETE"
        return summary
    except BaseException as exc:
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["finished_at_SGT"] = now()
        atomic_json(cell / "summary.json", summary)


def run(base):
    state = {
        "status": "RUNNING",
        "started_at_SGT": now(),
        "cells": [],
        "automatic_retry": False,
        "training_updates_max_per_arm": 4,
    }
    try:
        for arm, zero in (("M1", True), ("M2", False), ("M1", False)):
            rounds = (
                4
                if arm == "M2"
                else (read(base / "m2/summary.json")["actual_rounds"] if not zero else 1)
            )
            state["current_cell"] = "base" if zero else arm.lower()
            atomic_json(base / "state.json", state)
            state["cells"].append(run_cell(base, arm, rounds, zero=zero))
        state["status"] = "COMPLETE"
    except BaseException as exc:
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["updated_at_SGT"] = now()
        atomic_json(base / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "seal", "run"))
    parser.add_argument("--directory", type=Path, default=BASE)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory)
    elif args.command == "preflight":
        asyncio.run(input_preflight(args.directory))
    elif args.command == "seal":
        seal(args.directory)
    else:
        run(args.directory)


if __name__ == "__main__":
    main()
