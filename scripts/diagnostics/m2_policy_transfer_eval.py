#!/usr/bin/env python3
"""Reload four fixed actors on the original twelve Train windows; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "runs/m2-training-tuning-20260914"
DEFAULT_DIRECTORY = ROOT / "runs/m2-policy-transfer-eval-20260914"
PLAN = ROOT / "docs/2026-09-14_M2_POLICY_TRANSFER_EVAL_PLAN.md"
PROFILE = "policy_transfer_eval_v1"
CELL_IDS = ("sft0", "m1", "m2_eta1", "m2_eta025")
TRAIN_IDS = tuple(f"formal-train-{i:04d}" for i in range(1, 30, 4))
EVAL_IDS = tuple(f"formal-train-{i:04d}" for i in range(113, 126, 4))
PROTOCOL = "single_tool_portfolio_v1"
SFT_RECEIPT_SHA = "cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db"


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_files():
    files = {
        str(path.relative_to(ROOT)): path
        for folder in ("src", "runtime_hooks")
        for path in (ROOT / folder).rglob("*.py")
    }
    files.update(
        {
            name: ROOT / name
            for name in (
                "configs/base.yaml",
                "configs/m2_screening_r5_generation_pilot.yaml",
                "scripts/diagnostics/m2_policy_transfer_eval.py",
                "scripts/diagnostics/m2_policy_transfer_readout.py",
            )
        }
    )
    return files


def check_sft(model, expected_receipt):
    """Recheck the pinned, previously validated export without rehashing large weights."""
    model = Path(model)
    receipt_path = model / "merge_receipt.json"
    if model.is_symlink() or receipt_path.is_symlink() or sha(receipt_path) != expected_receipt:
        raise RuntimeError("SFT merge receipt identity differs")
    receipt = read(receipt_path)
    files = receipt.get("files", {})
    if receipt.get("status") != "COMPLETE" or not files:
        raise RuntimeError("SFT export is not complete")
    if {p.name for p in model.iterdir()} != set(files) | {"merge_receipt.json"}:
        raise RuntimeError("SFT export inventory differs")
    for name, item in files.items():
        path = model / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise RuntimeError("SFT export has a redirected or invalid artifact")
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError("SFT export artifact size differs")
        if not name.endswith(".safetensors") and sha(path) != item["sha256"]:
            raise RuntimeError("SFT export small artifact digest differs")
    return {"receipt_sha256": expected_receipt, "files": files}


def model_sources(parent=PARENT):
    """Return exact prior identities, and validate all light-weight artifact evidence."""
    from src.agents.actor_export import validate_export_artifact

    parent = Path(parent)
    state = read(parent / "state.json")
    identity = read(parent / "model_identity.json")
    if state.get("status") != "COMPLETE" or len(state.get("cells", [])) != 4:
        raise RuntimeError("the fixed prior comparison is not complete")
    if identity.get("merge_receipt_sha256") != SFT_RECEIPT_SHA:
        raise RuntimeError("prior comparison did not use the fixed common SFT")
    sources = {}
    baseline_head = None
    for cell_id, recorded in zip(CELL_IDS, state["cells"], strict=True):
        summary = read(parent / cell_id / "summary.json")
        if (
            recorded != summary
            or summary.get("status") != "COMPLETE"
            or summary.get("cell", {}).get("cell_id") != cell_id
            or summary.get("actual_rounds") != (0 if cell_id == "sft0" else 8)
            or summary.get("model_revision") != identity["model_revision"]
        ):
            raise RuntimeError("prior cell summary differs from its completed schedule")
        head = summary["critic_initial_head_sha256"]
        if baseline_head is None:
            baseline_head = head
        if head != baseline_head:
            raise RuntimeError("prior comparison did not share the actual critic seed")
        item = {
            "cell_id": cell_id,
            "prior_summary_sha256": sha(parent / cell_id / "summary.json"),
            "prior_execution_identity_sha256": summary["execution_identity_sha256"],
            "trained_updates": summary["actual_rounds"],
        }
        if cell_id == "sft0":
            model = Path(identity["model_snapshot_path"])
            item.update(check_sft(model, SFT_RECEIPT_SHA))
            item["model_revision"] = identity["model_revision"]
        else:
            checkpoint = parent / cell_id / "checkpoints/global_step_8"
            marker = validate_export_artifact(
                checkpoint,
                identity_sha256=summary["execution_identity_sha256"],
                verify_hashes=False,
            )
            receipt_path = checkpoint / "actor/actor_export_receipt.json"
            receipt = read(receipt_path)
            if sha(receipt_path) != marker["actor_receipt_sha256"]:
                raise RuntimeError("prior actor export receipt digest differs")
            model = checkpoint / "actor/huggingface"
            item.update(
                receipt_sha256=sha(receipt_path),
                files=receipt["hf"]["files"],
                model_revision="actor-export-receipt-sha256:" + sha(receipt_path),
                completion_marker_sha256=sha(checkpoint / "actor_export_complete.json"),
            )
        item["model_path"] = str(model.resolve())
        sources[cell_id] = item
    return sources


def make_contexts(frames):
    full = []
    dates = {}
    for dataset, ids, group in (
        ("train", TRAIN_IDS, "training_windows"),
        ("dev", EVAL_IDS, "evaluation_windows"),
    ):
        rows = frames[dataset]
        if [row["extra_info"]["episode_id"] for row in rows] != list(ids):
            raise RuntimeError("prepared episode order differs from the original grid")
        dates[group] = set()
        for row in rows:
            extra = row["extra_info"]
            kwargs = json.loads(row["env_kwargs"])
            if extra["source_split"] != "train" or len(kwargs["episode_dates"]) != 21:
                raise RuntimeError("cross evaluation must use original 21-session Train windows")
            dates[group].update(kwargs["episode_dates"])
            kwargs.update(
                observation_profile="history_features_v1", reward_mode="finance_trajectory"
            )
            full.append(
                {
                    "id": extra["episode_id"],
                    "group": group,
                    "state_kind": "cash",
                    "env_kwargs": kwargs,
                    "raw_prompt": list(row["prompt"]),
                    "prefix": [],
                    "expected_start_session_index": 0,
                }
            )
    if dates["training_windows"] & dates["evaluation_windows"]:
        raise RuntimeError("training and evaluation window dates overlap")
    return {"teacher_labels_included": False, "short": [], "full": full}


def settings():
    return {
        "profile": PROFILE,
        "parent_directory": str(PARENT),
        "cell_ids": list(CELL_IDS),
        "models": 4,
        "training_episode_ids": list(TRAIN_IDS),
        "evaluation_episode_ids": list(EVAL_IDS),
        "contexts_per_model": 12,
        "greedy_per_context": 1,
        "sampled_per_context": 4,
        "trajectories_per_model": 60,
        "total_trajectories": 240,
        "training_updates": 0,
        "model_exports": 0,
        "gpu_release_timeout_seconds": 120,
        "automatic_retry": False,
        "resume_supported": False,
        "seed": 42,
        "request_limit": 128,
        "prompt_limit": 8192,
        "response_limit": 4096,
        "action_protocol": PROTOCOL,
        "enable_thinking": False,
        "seed_grid": "coldstart_eval.grid: step=1; sample_index=0..11; greedy+sample0..3",
        "prior_trainer_request_limit": 256,
        "model_validation": (
            "pinned receipts, inventory, sizes and small-file hashes; "
            "large weight content hashes are not recomputed"
        ),
    }


def prepare(base, *, plan=PLAN):
    import pandas as pd

    base = Path(base).resolve()
    if base.exists():
        raise RuntimeError("new evaluation directory required; no overwrite")
    if not Path(plan).is_file():
        raise RuntimeError("explicit evaluation plan is required before preparation")
    original_sources = read(PARENT / "source_identity.json")
    for relative, expected in original_sources.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("original runtime source changed: " + relative)
    manifest = read(PARENT / "prepared/manifest.json")
    pinned = [
        PARENT / name
        for name in (
            "state.json",
            "model_identity.json",
            "settings.json",
            "run_contract.json",
            "source_identity.json",
            "prepared/manifest.json",
        )
    ]
    frames = {}
    for dataset in ("train", "dev"):
        info = manifest["datasets"][dataset]
        path = PARENT / "prepared" / info["path"]
        if sha(path) != info["sha256"]:
            raise RuntimeError("original prepared parquet changed")
        frames[dataset] = pd.read_parquet(path).to_dict("records")
        pinned.append(path)
    calibration = PARENT / "prepared" / manifest["calibration"]["path"]
    if sha(calibration) != manifest["calibration"]["sha256"]:
        raise RuntimeError("original calibration changed")
    pinned.append(calibration)
    models = model_sources()
    contexts = make_contexts(frames)
    pinned += [PARENT / cell / "summary.json" for cell in CELL_IDS]
    base.mkdir(parents=True, exist_ok=False)
    save(base / "settings.json", settings())
    save(base / "contexts.json", contexts)
    save(base / "models.json", models)
    save(base / "parent_evidence.json", {str(path): sha(path) for path in pinned})
    shutil.copy2(plan, base / "plan.md")
    save(base / "source_identity.json", {name: sha(path) for name, path in source_files().items()})
    names = (
        "settings.json",
        "contexts.json",
        "models.json",
        "parent_evidence.json",
        "plan.md",
        "source_identity.json",
    )
    save(
        base / "run_contract.json",
        {
            "profile": PROFILE,
            "sealed_at_SGT": now(),
            "inputs": {name: sha(base / name) for name in names},
        },
    )
    check_contract(base)
    return {"status": "PREPARED", "directory": str(base), "planned_trajectories": 240}


def check_contract(base):
    contract = read(base / "run_contract.json")
    expected_inputs = {
        "settings.json",
        "contexts.json",
        "models.json",
        "parent_evidence.json",
        "plan.md",
        "source_identity.json",
    }
    if contract.get("profile") != PROFILE or set(contract.get("inputs", {})) != expected_inputs:
        raise RuntimeError("evaluation seal is incomplete")
    for name, digest in contract["inputs"].items():
        if sha(base / name) != digest:
            raise RuntimeError("sealed evaluation input changed: " + name)
    if read(base / "settings.json") != settings():
        raise RuntimeError("evaluation settings are not the fixed 240-trajectory contract")
    sources = read(base / "source_identity.json")
    if set(sources) != set(source_files()):
        raise RuntimeError("runtime source file set changed")
    for name, path in source_files().items():
        if sha(path) != sources[name]:
            raise RuntimeError("runtime source changed: " + name)
    for name, digest in read(base / "parent_evidence.json").items():
        if sha(name) != digest:
            raise RuntimeError("parent evaluation evidence changed: " + name)
    if model_sources() != read(base / "models.json"):
        raise RuntimeError("pinned actor export identity or inventory changed")


def cell_command(base, cell_id, model):
    if cell_id not in CELL_IDS:
        raise ValueError("unregistered evaluation cell")
    return [
        str(ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "src.agents.coldstart_eval",
        "--model",
        model["model_path"],
        "--contexts",
        str(base / "contexts.json"),
        "--output",
        str(base / cell_id),
        "--action-protocol",
        PROTOCOL,
        "--expected-short",
        "0",
        "--expected-full",
        "12",
    ]


def wait_gpu_idle():
    """Observe natural worker cleanup; never terminate another GPU process."""
    started = time.monotonic()
    deadline = started + 120
    observations = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("GPU did not naturally release below 1024 MiB within 120 seconds")
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=min(10, remaining),
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if not values or any(value < 0 for value in values):
            raise RuntimeError("GPU memory query returned no valid devices")
        observations.append({"at_SGT": now(), "memory_used_MiB": values})
        if all(value < 1024 for value in values):
            return {
                "status": "IDLE",
                "elapsed_seconds": time.monotonic() - started,
                "observations": observations,
            }
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def validate_result(path, contexts):
    value = read(path)
    if value.get("status") != "OBSERVED" or value.get("action_protocol") != PROTOCOL:
        raise RuntimeError("evaluation result has no completed observation")
    expected = {}
    for index, context in enumerate(contexts["full"]):
        for sibling in range(5):
            lane = "greedy" if sibling == 0 else f"sample{sibling - 1}"
            expected[f"full-{context['id']}-{lane}"] = (index, sibling, context)
    rows = value.get("grid", [])
    if len(rows) != 60 or Counter(r.get("id") for r in rows) != Counter(expected.keys()):
        raise RuntimeError("evaluation result is missing or duplicates planned trajectories")
    for row in rows:
        index, sibling, context = expected[row["id"]]
        if (
            row.get("status") != "COMPLETE"
            or row.get("failure")
            or row.get("context") != context
            or row.get("horizon") != "full"
            or row.get("sampling") != ("greedy" if sibling == 0 else "sample")
            or row.get("identity")
            != {
                "step": 1,
                "sample_index": index,
                "rollout_n": max(0, sibling - 1),
                "validate": sibling == 0,
            }
            or row.get("request_limit") != 128
            or not 1 <= row.get("requests", 0) <= 128
            or not 1 <= row.get("max_response_tokens", 0) < 4096
            or not 1 <= row.get("max_prompt_tokens", 0) <= 8192
            or row.get("prefix_fill_count") != 0
            or row.get("prefix_info") != []
            or row.get("takeover_state", {}).get("session_index") != 0
            or row.get("takeover_state", {}).get("positions") != {}
            or row.get("final_state", {}).get("session_index") != 20
            or row.get("terminal_metrics", {}).get("session_count") != 21
            or row.get("terminal_metrics", {}).get("calibration_sha256")
            != context["env_kwargs"]["screening_calibration_sha256"]
        ):
            raise RuntimeError(
                "trajectory is incomplete or differs from the full cash-state contract"
            )
        for number in (
            row.get("takeover_net_return"),
            row["terminal_metrics"].get("terminal_score"),
        ):
            if type(number) not in (int, float) or not math.isfinite(number):
                raise RuntimeError("completed trajectory lacks finite outcome evidence")
    return {
        "complete": 60,
        "planned": 60,
        "requests": sum(r["requests"] for r in rows),
        "result_sha256": sha(path),
    }


def run(base):
    base = Path(base).resolve()
    if (base / "state.json").exists():
        raise RuntimeError("evaluation already has state; no retry or resume")
    check_contract(base)
    state = {
        "status": "RUNNING",
        "started_at_SGT": now(),
        "planned_trajectories": 240,
        "cells": [],
        "automatic_retry": False,
    }
    with (base / "state.json").open("x") as stream:
        json.dump(state, stream, indent=2)
    contexts, models = read(base / "contexts.json"), read(base / "models.json")
    try:
        for folder in ("commands", "processes", "logs"):
            (base / folder).mkdir(exist_ok=False)
        for cell_id in CELL_IDS:
            check_contract(base)
            if (base / cell_id).exists():
                raise RuntimeError("evaluation child directory already exists")
            if shutil.disk_usage(base).free < 5 * 1024**3:
                raise RuntimeError("evaluation has less than the fixed 5 GiB free-space reserve")
            cell = {"cell_id": cell_id, "status": "RUNNING", "started_at_SGT": now()}
            state["cells"].append(cell)
            state["current_cell"] = cell_id
            save(base / "state.json", state)
            cell["gpu_before"] = wait_gpu_idle()
            save(base / "state.json", state)
            command = cell_command(base, cell_id, models[cell_id])
            save(base / "commands" / f"{cell_id}.json", command)
            environment = dict(os.environ)
            for key in list(environment):
                if key.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_")):
                    environment.pop(key)
            environment.update(
                PYTHONPATH=f"{ROOT}:{ROOT / 'third_party/agent-r1'}",
                PYTHONUNBUFFERED="1",
                TOKENIZERS_PARALLELISM="false",
            )
            with (base / "logs" / f"{cell_id}.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                save(
                    base / "processes" / f"{cell_id}.json",
                    {"pid": process.pid, "started_at_SGT": now(), "command": command},
                )
                cell["returncode"] = process.wait()
            if cell["returncode"] != 0:
                raise RuntimeError(f"{cell_id} evaluator exited {cell['returncode']}")
            cell["evidence"] = validate_result(base / cell_id / "result.json", contexts)
            if read(base / cell_id / "state.json") != read(base / cell_id / "result.json"):
                raise RuntimeError("final evaluator state/result disagree")
            check_contract(base)
            cell.update(status="COMPLETE", finished_at_SGT=now())
            save(base / "state.json", state)
        state.update(status="COMPLETE", completed_trajectories=240)
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if state["cells"] and state["cells"][-1]["status"] == "RUNNING":
            state["cells"][-1].update(status="FAILED", finished_at_SGT=now(), error=state["error"])
        raise
    finally:
        state["finished_at_SGT"] = now()
        save(base / "state.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args(argv)
    base = args.directory.resolve()
    sys.path.insert(0, str(ROOT))
    if args.command == "prepare":
        result = prepare(base, plan=args.plan)
    elif args.command == "check":
        check_contract(base)
        result = {"status": "PASS", "directory": str(base)}
    else:
        result = run(base)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
