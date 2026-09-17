#!/usr/bin/env python3
"""Practical Train portfolio comparison using the existing complete-trajectory executor."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/m2-train-practice-20260912"
PLAN = ROOT / "docs/2026-09-12_M2_TRAIN_PRACTICE_PLAN.md"
CONDITION = ROOT / "scripts/diagnostics/m2_train_practice_conditions.py"
spec = importlib.util.spec_from_file_location(
    "practice_executor", ROOT / "scripts/diagnostics/m2_decision_probe.py"
)
assert spec and spec.loader
E: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
H = E.H
CONDITIONS = tuple(
    f"{p}_{s}" for p in ("baseline", "guide", "examples") for s in ("cash", "invested")
)
MODEL = ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c")


def grid_plan():
    return [
        {
            "grid_index": wi * 24 + ci * 4 + ri,
            "model_key": "screening",
            "cohort_index": wi,
            "condition": c,
            "window_index": wi,
            "episode_id": H.EPISODES[wi],
            "rollout_id": ri,
            "start_session": 4,
            "request_limit": 64,
            "trajectory_id": f"{c}-w{wi}-r{ri}",
            "identity": {"step": 1, "sample_index": wi, "rollout_n": ri, "validate": False},
            "initial_status": "UNSTARTED",
        }
        for wi in range(2)
        for ci, c in enumerate(CONDITIONS)
        for ri in range(4)
    ]


def budget_contract():
    return {
        "trajectory_count": 48,
        "max_requests_total": 3072,
        "max_active_batch": 32,
        "request_limit_per_trajectory": 64,
        "prompt_cap": 8192,
        "response_cap": 4096,
        "context_cap": 12288,
        "model_advances_to_terminal": 16,
        "initialization": "four scripted real transitions, excluded from model actions",
        "training_updates": 0,
    }


def source_paths():
    return sorted(
        set(
            H.source_paths()
            + [
                Path(__file__),
                CONDITION,
                PLAN,
                ROOT / "scripts/diagnostics/m2_decision_probe.py",
                ROOT / "scripts/diagnostics/m2_decision_conditions.py",
                ROOT / "tests/test_m2_train_practice.py",
            ]
        )
    )


def validate_prefix(env: Any, receipt: dict[str, Any], plan: dict[str, Any]):
    assert env.practice_condition == plan["condition"] == receipt["condition"]
    trace = [asdict(t) for t in env.credit_step_trace]
    assert (
        receipt["credit_step_trace"] == trace
        and len(trace) == 4
        and not any(t["invalid_action"] for t in trace)
    )
    assert env.current_session_index == 4 and not env.episode_complete
    assert receipt["model_requests"] == 0 and receipt["is_model_request"] is False
    expected = (
        ["buy", "buy", "buy", "hold"] if plan["condition"].endswith("_invested") else ["hold"] * 4
    )
    assert [t["action"] for t in trace] == expected
    assert len(env.ledger.fills) == (3 if expected[0] == "buy" else 0)
    E.check_trace(trace, done=False, session=4, sessions=21)


def validate_manifest(m):
    assert m["diagnostic_id"] == "m2_train_practice_v1" and m["grid"] == grid_plan()
    assert m["budget"] == budget_contract() and list(m["models"]) == ["screening"]
    assert tuple(m["models"]["screening"]["initial_prompts"]) == CONDITIONS
    assert all(w["episode_id"].startswith("formal-train-") for w in m["windows"])


E.HERE = BASE
E.CONDITION_SCRIPT = CONDITION
E.PLAN_DOC = PLAN
E.grid_plan = grid_plan
E.budget_contract = budget_contract
E.validate_prefix = validate_prefix
E.validate_manifest = validate_manifest
E.source_paths = source_paths
E.MODEL_CONDITIONS = {"screening": CONDITIONS}
E.REQUEST_LIMITS = {"screening": 64}
E.START_SESSIONS = {"screening": 4}
H.MODELS = {"screening": MODEL}
E.LIMITATIONS = [
    "Practical Train iteration, not formal M2 performance evidence.",
    "Prompt guidance and in-context examples, not weight training.",
    "Scripted initial portfolio fills are excluded from model trade counts.",
    "Full episode score includes initialization; compare within the same starting portfolio.",
    "Existing environment, ledger and inference engine; no probability scoring.",
]


def prepare():
    os.environ.update(
        USE_TORCH="0", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    H.add_import_paths()
    from transformers import AutoTokenizer

    C = E.module_from_path("practice_conditions_prepare", CONDITION)
    old = H.read(ROOT / "runs/m2-transfer-diagnostic-20260912/trajectory-v1/manifest.json")
    model = copy.deepcopy(old["models"]["screening"])
    model["initial_prompts"] = {}
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
    rows = H.read_train_rows()
    receipts: dict[str, Any] = {}
    for c in CONDITIONS:
        model["initial_prompts"][c] = []
        receipts[c] = []
        for wi in range(2):
            env = C.create_env(rows, wi, c, 4)
            obs, rec = asyncio.run(C.reset_with_prefix(env))
            validate_prefix(env, rec, {"condition": c})
            prompt = C.render_prompt(tokenizer, obs, env.tool_schemas)
            assert prompt["prompt_token_count"] <= 8192
            model["initial_prompts"][c].append(prompt)
            receipts[c].append(rec)
    # Baseline cash input remains exactly the previously tested input.
    assert (
        model["initial_prompts"]["baseline_cash"]
        == old["models"]["screening"]["initial_prompts"]["features"]
    )
    m = {
        "status": "PREPARED",
        "schema_version": 1,
        "diagnostic_id": "m2_train_practice_v1",
        "at": H.now(),
        "grid": grid_plan(),
        "budget": budget_contract(),
        "models": {"screening": model},
        "source_identity": E.source_identity(),
        "source_file_sha256": {str(H.PILOT.relative_to(ROOT)): H.PILOT_SHA},
        "package_versions": H.package_versions(),
        "prefix_receipts": receipts,
        "windows": [
            {
                "episode_id": H.EPISODES[i],
                "raw_prompt": r["prompt"],
                "env_kwargs": json.loads(r["env_kwargs"]),
            }
            for i, r in enumerate(rows)
        ],
        "limitations": E.LIMITATIONS,
    }
    validate_manifest(m)
    BASE.mkdir(exist_ok=True)
    H.write_new(BASE / "manifest.json", m)
    H.write_new(
        BASE / "preparation.json",
        {
            "status": "PASSED",
            "initial_prompt_counts": {
                c: [p["prompt_token_count"] for p in v] for c, v in model["initial_prompts"].items()
            },
            "baseline_cash_exact": True,
            "new_model_requests": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        E.run(BASE / "manifest.json", args.expected_manifest_sha256, "screening")
