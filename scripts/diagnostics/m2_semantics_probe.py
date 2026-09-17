#!/usr/bin/env python3
"""Fixed 1.7B semantic-display diagnostic; isolated reuse of a frozen executor.

This adapter never edits the prior runner or its evidence. A fresh, unregistered
module instance owns every explicitly rebound global and its own helper module.
The saved executor binding makes this small specialization part of the new
manifest identity. No training, replacement sampling, or automatic GPU work.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "runs/m2-semantics-diagnostic-20260912/trajectory-v1"
PLAN_DOC = ROOT / "docs/2026-09-12_M2_SEMANTICS_DIAGNOSTIC_PLAN.md"
CONDITION_SCRIPT = ROOT / "scripts/diagnostics/m2_semantics_conditions.py"
EXECUTOR_SCRIPT = ROOT / "scripts/diagnostics/m2_decision_probe.py"
EXECUTOR_SHA = "fc7a42dd514cc2313a5c0ce82bb16d7381dbe702f3b8d3e2c62f5ed085640f80"
BASELINE_MANIFEST = ROOT / "runs/m2-decision-diagnostic-20260909/probe-v1/manifest.json"
BASELINE_SHA = "2f79114d1488f742f5b4aa27a464679b212e0deb72fa1622b5c46ebff0db8bdb"
CONDITIONS = ("memory_only", "availability", "time_semantics", "availability_time")
MODEL = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")


def isolated_executor() -> Any:
    if hashlib.sha256(EXECUTOR_SCRIPT.read_bytes()).hexdigest() != EXECUTOR_SHA:
        raise RuntimeError("Frozen decision executor changed")
    spec = importlib.util.spec_from_file_location("semantics_isolated_executor", EXECUTOR_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = isolated_executor()
H = E.H
LIMITATIONS = [
    "One fresh 1.7B base; no training, parameter updates, checkpoint or Dev policy evaluation.",
    "64 fixed identities: two original Train windows, four semantic conditions, eight siblings.",
    "A 64-request horizon is HORIZON_REACHED unless the environment actually terminates.",
    "No response-cap retries, new samples, automatic actions or return-based selection.",
    "Native terminal metrics are descriptive and do not construct training advantages/residuals.",
    "The frozen executor's feature_evidence field transports semantic_display evidence here; "
    "it contains no precomputed decision features.",
    "The memory_only initial prompt is an exact token/message/tool replay of the prior baseline; "
    "new conditions, batches and sampled continuations are a new diagnostic population.",
]


def grid_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in range(2):
        for condition in CONDITIONS:
            for sibling in range(8):
                rows.append(
                    {
                        "grid_index": len(rows),
                        "model_key": "wiring",
                        "cohort_index": window,
                        "condition": condition,
                        "window_index": window,
                        "episode_id": H.EPISODES[window],
                        "rollout_id": sibling,
                        "start_session": 0,
                        "request_limit": 64,
                        "trajectory_id": f"wiring-{condition}-w{window}-r{sibling}",
                        "identity": {
                            "step": 1,
                            "sample_index": window,
                            "rollout_n": sibling,
                            "validate": False,
                        },
                        "initial_status": "UNSTARTED",
                    }
                )
    return rows


def budget_contract() -> dict[str, Any]:
    return {
        "trajectory_count": 64,
        "model_trajectories": {"wiring": 64},
        "request_limits": {"wiring": 64},
        "start_sessions": {"wiring": 0},
        "siblings_per_window_condition": 8,
        "max_active_batch": 32,
        "max_requests_total": 4096,
        "max_response_tokens_total": 16777216,
        "prompt_cap": 8192,
        "response_cap": 4096,
        "context_cap": 12288,
        "terminal_sessions": 21,
        "terminal_total_advances": 20,
        "seed_env_step_counts": "model requests only; no scripted prefix transitions",
    }


def executor_binding() -> dict[str, Any]:
    return {
        "source": str(EXECUTOR_SCRIPT.relative_to(ROOT)),
        "sha256": EXECUTOR_SHA,
        "isolation": "fresh ModuleType via spec_from_file_location; not inserted into sys.modules",
        "reused_callables": [
            "execute_grid",
            "run",
            "initial_state",
            "request_seed",
            "check_trace",
            "validate_prefix",
            "engine_kwargs",
            "verify_files",
            "source_identity",
        ],
        "overrides": {
            "HERE": str(HERE.relative_to(ROOT)),
            "PLAN_DOC": str(PLAN_DOC.relative_to(ROOT)),
            "CONDITION_SCRIPT": str(CONDITION_SCRIPT.relative_to(ROOT)),
            "H.MODELS": {"wiring": list(MODEL)},
            "MODEL_CONDITIONS": {"wiring": list(CONDITIONS)},
            "REQUEST_LIMITS": {"wiring": 64},
            "START_SESSIONS": {"wiring": 0},
            "MAX_REQUESTS": 4096,
            "grid_plan": "m2_semantics_probe.grid_plan",
            "budget_contract": "m2_semantics_probe.budget_contract",
            "validate_manifest": "m2_semantics_probe.validate_manifest",
            "source_paths": "m2_semantics_probe.source_paths",
            "LIMITATIONS": LIMITATIONS,
        },
        "semantic_evidence": {
            "kind": "semantic_display",
            "transport_field": "feature_evidence",
            "meaning": "semantics diagnostic evidence, not precomputed features",
        },
    }


def source_paths() -> list[Path]:
    return sorted(
        set(
            [p for p in H.source_paths() if p != H.PLAN_DOC]
            + [
                Path(__file__).resolve(),
                EXECUTOR_SCRIPT,
                CONDITION_SCRIPT,
                PLAN_DOC,
                ROOT / "scripts/diagnostics/m2_decision_conditions.py",
            ]
        )
    )


def baseline() -> dict[str, Any]:
    H.ensure(H.digest(BASELINE_MANIFEST) == BASELINE_SHA, "Prior exact baseline manifest changed")
    return H.read(BASELINE_MANIFEST)


def verify_binding() -> None:
    H.ensure(H.digest(EXECUTOR_SCRIPT) == EXECUTOR_SHA, "Executor source drift")
    expected = {
        "HERE": HERE,
        "PLAN_DOC": PLAN_DOC,
        "CONDITION_SCRIPT": CONDITION_SCRIPT,
        "MODEL_CONDITIONS": {"wiring": CONDITIONS},
        "REQUEST_LIMITS": {"wiring": 64},
        "START_SESSIONS": {"wiring": 0},
        "MAX_REQUESTS": 4096,
        "grid_plan": grid_plan,
        "budget_contract": budget_contract,
        "validate_manifest": validate_manifest,
        "source_paths": source_paths,
        "LIMITATIONS": LIMITATIONS,
    }
    H.ensure(
        all(getattr(E, key) == value for key, value in expected.items()),
        "Isolated executor binding drift",
    )
    H.ensure(H.MODELS == {"wiring": MODEL} and E.H is H, "Helper model binding drift")


def validate_manifest(manifest: dict[str, Any]) -> None:
    verify_binding()
    H.ensure(
        manifest.get("status") == "PREPARED"
        and manifest.get("schema_version") == 1
        and manifest.get("diagnostic_id") == "m2_semantics_v1",
        "Wrong semantic manifest",
    )
    H.ensure(manifest.get("executor_binding") == executor_binding(), "Undeclared executor binding")
    H.ensure(
        manifest.get("grid") == grid_plan() and manifest.get("budget") == budget_contract(),
        "Registered condition/seed population or horizon changed",
    )
    H.ensure(
        list(manifest["models"]) == ["wiring"] and len(manifest["windows"]) == 2,
        "Wrong model/window population",
    )
    model = manifest["models"]["wiring"]
    H.ensure(
        (model["name"], model["revision"]) == MODEL
        and model["snapshot"] == str(H.snapshot("wiring"))
        and model["engine_kwargs"] == E.engine_kwargs(H.snapshot("wiring")),
        "Base/resource identity drift",
    )
    H.ensure(list(model["initial_prompts"]) == list(CONDITIONS), "Wrong initial condition order")
    old_prompts = baseline()["models"]["wiring"]["initial_prompts"]["memory_only"]
    H.ensure(
        model["initial_prompts"]["memory_only"] == old_prompts,
        "Memory-only baseline is not exact old input",
    )
    for condition, prompts in model["initial_prompts"].items():
        H.ensure(len(prompts) == 2, "Two initial windows required")
        for index, prompt in enumerate(prompts):
            ids, old = prompt["prompt_ids"], old_prompts[index]
            H.ensure(
                isinstance(ids, list)
                and 0 < len(ids) <= 8192
                and all(type(t) is int and t >= 0 for t in ids)
                and prompt["prompt_token_count"] == len(ids)
                and prompt["prompt_sha256"] == H.token_sha(ids)
                and prompt["template_kwargs"] == {"enable_thinking": False}
                and prompt["action_protocol"] == "single_tool_v1",
                "Prompt generation/budget identity drift",
            )
            H.ensure(
                prompt["tools"] == old["tools"] and prompt["messages"][:-1] == old["messages"][:-1],
                "Semantic condition altered shared tools or task/protocol prefix",
            )
            if condition != "memory_only":
                H.ensure(
                    ids != old["prompt_ids"], "Registered semantic display intervention missing"
                )


def bind_executor() -> None:
    # This object and its H helper were freshly imported above; no registered
    # production module or prior diagnostic instance is ever patched.
    E.HERE, E.PLAN_DOC, E.CONDITION_SCRIPT = HERE, PLAN_DOC, CONDITION_SCRIPT
    H.MODELS = {"wiring": MODEL}
    E.MODEL_CONDITIONS, E.REQUEST_LIMITS, E.START_SESSIONS = (
        {"wiring": CONDITIONS},
        {"wiring": 64},
        {"wiring": 0},
    )
    E.MAX_REQUESTS, E.LIMITATIONS = 4096, LIMITATIONS
    E.grid_plan, E.budget_contract = grid_plan, budget_contract
    E.validate_manifest, E.source_paths = validate_manifest, source_paths
    verify_binding()


bind_executor()


def prepare(manifest_path: Path) -> dict[str, Any]:
    H.ensure(
        manifest_path.resolve() == HERE / "manifest.json" and not manifest_path.exists(),
        "Use a new semantic manifest; never overwrite",
    )
    verify_binding()
    os.environ.update(
        USE_TORCH="0",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    H.add_import_paths()
    from transformers import AutoTokenizer

    conditions = E.module_from_path("semantics_conditions_prepare", CONDITION_SCRIPT)
    source = E.source_identity()
    prior = baseline()["models"]["wiring"]
    model = {
        key: copy.deepcopy(prior[key])
        for key in ("name", "revision", "snapshot", "metadata_sha256", "weight_sha256")
    }
    model["engine_kwargs"] = E.engine_kwargs(Path(model["snapshot"]))
    for name, expected in model["metadata_sha256"].items():
        H.ensure(
            H.digest(Path(model["snapshot"]) / name) == expected, "Tokenizer/model metadata changed"
        )
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
    rows = H.read_train_rows()
    model["initial_prompts"] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        model["initial_prompts"][condition], receipts[condition] = [], []
        for index in range(2):
            plan = next(
                p for p in grid_plan() if p["condition"] == condition and p["window_index"] == index
            )
            env = conditions.create_env(rows, index, condition, 0)
            obs, receipt = asyncio.run(conditions.reset_with_prefix(env))
            E.validate_prefix(env, receipt, plan)
            H.ensure(
                env.semantic_evidence == env.feature_evidence
                and env.semantic_evidence["kind"] == "semantic_display",
                "Semantic evidence transport mismatch",
            )
            model["initial_prompts"][condition].append(
                conditions.render_prompt(tokenizer, obs, env.tool_schemas)
            )
            receipts[condition].append(receipt)
    manifest = {
        "status": "PREPARED",
        "schema_version": 1,
        "diagnostic_id": "m2_semantics_v1",
        "at": H.now(),
        "executor_binding": executor_binding(),
        "grid": grid_plan(),
        "budget": budget_contract(),
        "source_identity": source,
        "package_versions": H.package_versions(),
        "source_file_sha256": {
            str(BASELINE_MANIFEST.relative_to(ROOT)): BASELINE_SHA,
            str(H.PILOT.relative_to(ROOT)): H.PILOT_SHA,
        },
        "models": {"wiring": model},
        "prefix_receipts": {"wiring": receipts},
        "windows": [
            {
                "episode_id": H.EPISODES[i],
                "raw_prompt": r["prompt"],
                "env_kwargs": json.loads(r["env_kwargs"]),
            }
            for i, r in enumerate(rows)
        ],
        "weight_inventory_source": (
            "SHA-bound prior base inventory; run rehashes actual weights before GPU"
        ),
        "limitations": LIMITATIONS,
    }
    validate_manifest(manifest)
    H.ensure(
        E.source_identity()["source_sha256"] == source["source_sha256"],
        "Source changed during prepare",
    )
    H.ensure(
        not any(x in sys.modules for x in ("torch", "ray", "vllm")),
        "CPU prepare imported GPU framework",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    H.write_new(manifest_path, manifest)
    saved = H.read(manifest_path)
    for prompts in saved["models"]["wiring"]["initial_prompts"].values():
        for prompt in prompts:
            ids = list(
                tokenizer.apply_chat_template(
                    prompt["messages"],
                    tools=prompt["tools"],
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )
            )
            H.ensure(ids == prompt["prompt_ids"], "Serialized messages/tools changed prompt tokens")
    H.write_new(
        HERE / "preparation.json",
        {
            "status": "PASSED",
            "at": H.now(),
            "manifest_sha256": H.digest(manifest_path),
            "script_sha256": H.digest(Path(__file__)),
            "executor_binding": executor_binding(),
            "new_model_requests": 0,
            "gpu_frameworks_imported": False,
            "budget": budget_contract(),
            "baseline_initial_tokens_equal": True,
            "prompt_counts": {
                c: [p["prompt_token_count"] for p in ps]
                for c, ps in model["initial_prompts"].items()
            },
        },
    )
    return saved


def verify(manifest_path: Path, expected_sha: str) -> dict[str, Any]:
    H.ensure(manifest_path.resolve() == HERE / "manifest.json", "Wrong semantic manifest path")
    H.ensure(H.digest(manifest_path) == expected_sha, "Manifest SHA mismatch")
    manifest = H.read(manifest_path)
    E.verify_files(manifest)
    return manifest


def run(manifest_path: Path, expected_sha: str) -> int:
    verify_binding()
    return int(E.run(manifest_path, expected_sha, "wiring"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run"):
        action = sub.add_parser(name)
        action.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
        if name != "prepare":
            action.add_argument("--expected-manifest-sha256", required=True)
        if name == "run":
            action.add_argument("--model", choices=["wiring"], default="wiring")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest)
        print(json.dumps({"status": "PREPARED", "manifest_sha256": H.digest(args.manifest)}))
        return 0
    if args.command == "verify":
        verify(args.manifest, args.expected_manifest_sha256)
        print(json.dumps({"status": "VERIFIED", "new_model_requests": 0}))
        return 0
    return run(args.manifest, args.expected_manifest_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
