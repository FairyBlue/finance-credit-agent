#!/usr/bin/env python3
"""Fixed 96-trajectory Train decision diagnostics, without training or retries.

4B compares raw/precomputed information after four recorded hold transitions.
1.7B compares four observation ablations from session zero. Horizons are bounded
continuations and never stand in for complete episodes. GPU work starts only
through run, after the CPU-frozen manifest and source checks.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "runs/m2-decision-diagnostic-20260909/probe-v1"
PLAN_DOC = ROOT / "docs/2026-09-09_M2_DECISION_DIAGNOSTIC_PLAN.md"
CONDITION_SCRIPT = ROOT / "scripts/diagnostics/m2_decision_conditions.py"
HOLD_SCRIPT = ROOT / "scripts/diagnostics/m2_hold_probe.py"
HOLD_SHA = "7d610d7ff5343fac95b023470d9af699615ca93572d8bf7b62950e36d1a7153a"
PRIOR_MANIFEST = ROOT / "runs/m2-hold-diagnostic-20260909/probe-v1/manifest.json"
PRIOR_SHA = "0b3ac1091f20182a0ae0ad74fad817b0f68394956bb5ceefbf6c24ea74916ab3"
MODEL_CONDITIONS = {
    "screening": ("raw", "features"),
    "wiring": ("raw", "history_only", "memory_only", "both"),
}
REQUEST_LIMITS = {"screening": 128, "wiring": 64}
START_SESSIONS = {"screening": 4, "wiring": 0}
PROMPT_CAP, RESPONSE_CAP, CONTEXT_CAP = 8192, 4096, 12288
MAX_BATCH, MAX_REQUESTS = 32, 8192
ADVANCE_ACTIONS = {"buy", "sell", "hold", "abstain"}
OBSERVED_STATUSES = {"COMPLETE", "HORIZON_REACHED", "INCOMPLETE"}


def module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if hashlib.sha256(HOLD_SCRIPT.read_bytes()).hexdigest() != HOLD_SHA:
    raise RuntimeError("Frozen inference helper changed")
H = module_from_path("decision_hold_helpers", HOLD_SCRIPT)
LIMITATIONS = [
    "Fresh bases; no training, actor/critic, gradients, checkpoint or Dev policy evaluation.",
    "4B has four scripted hold transitions before the first model request; "
    "they are not sampled experience.",
    "Precomputed features are a read-only information intervention, not model tool calls.",
    "128/64 requests are finite continuation horizons: HORIZON_REACHED is not COMPLETE.",
    "All 96 planned identities remain in the population; "
    "no retries, replacements or return selection.",
    "Eight diagnostic sibling identities use the production seed derivation; training remains n=4.",
    "Batch composition and new resources prevent claims of tokenwise replay "
    "of earlier diagnostics.",
    "Native terminal metrics are descriptive; no training advantages, residuals or main screening.",
]


def grid_plan() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for model in H.MODELS:
        for cohort in range(1 if model == "screening" else 2):
            windows = (0, 1) if model == "screening" else (cohort,)
            for condition in MODEL_CONDITIONS[model]:
                for window in windows:
                    for sibling in range(8):
                        grid.append(
                            {
                                "grid_index": len(grid),
                                "model_key": model,
                                "cohort_index": cohort,
                                "condition": condition,
                                "window_index": window,
                                "episode_id": H.EPISODES[window],
                                "rollout_id": sibling,
                                "start_session": START_SESSIONS[model],
                                "request_limit": REQUEST_LIMITS[model],
                                "trajectory_id": f"{model}-{condition}-w{window}-r{sibling}",
                                "identity": {
                                    "step": 1,
                                    "sample_index": window,
                                    "rollout_n": sibling,
                                    "validate": False,
                                },
                                "initial_status": "UNSTARTED",
                            }
                        )
    return grid


def budget_contract() -> dict[str, Any]:
    return {
        "trajectory_count": 96,
        "model_trajectories": {"screening": 32, "wiring": 64},
        "request_limits": REQUEST_LIMITS,
        "start_sessions": START_SESSIONS,
        "siblings_per_window_condition": 8,
        "max_active_batch": MAX_BATCH,
        "max_requests_total": MAX_REQUESTS,
        "max_response_tokens_total": 33554432,
        "prompt_cap": PROMPT_CAP,
        "response_cap": RESPONSE_CAP,
        "context_cap": CONTEXT_CAP,
        "terminal_sessions": 21,
        "terminal_total_advances": 20,
        "seed_env_step_counts": "model requests only, excluding scripted prefix holds",
    }


def engine_kwargs(model: Path) -> dict[str, Any]:
    result = H.engine_kwargs(model)
    result.update(
        max_model_len=CONTEXT_CAP,
        max_num_seqs=MAX_BATCH,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.50,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
    )
    return result


def request_seed(plan: dict[str, Any], model_request_index: int) -> int:
    H.ensure(
        type(model_request_index) is int and 0 <= model_request_index < plan["request_limit"],
        "Model request index exceeds registered horizon",
    )
    H.add_import_paths()
    from src.agents.rollout_seed import derive_request_seed

    return derive_request_seed(42, plan["identity"], model_request_index)


def source_paths() -> list[Path]:
    return sorted(
        set(
            [p for p in H.source_paths() if p != H.PLAN_DOC]
            + [Path(__file__).resolve(), CONDITION_SCRIPT, PLAN_DOC]
        )
    )


def source_identity() -> dict[str, Any]:
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "agent_r1_commit": subprocess.check_output(
            ["git", "-C", "third_party/agent-r1", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): H.digest(p) for p in source_paths()},
    }


def check_trace(trace: list[dict[str, Any]], *, done: bool, session: int, sessions: int) -> None:
    H.ensure(
        sessions == 21 and type(session) is int and 0 <= session <= 20, "Wrong session population"
    )
    previous = 0
    for index, step in enumerate(trace):
        H.ensure(
            type(step["step_index"]) is int
            and step["step_index"] == index
            and type(step["decision_session_index"]) is int
            and step["decision_session_index"] == previous,
            "Discontinuous causal trace",
        )
        H.ensure(
            type(step["done"]) is bool and type(step["invalid_action"]) is bool,
            "Nonboolean step flags",
        )
        advance = step["action"] in ADVANCE_ACTIONS and not step["invalid_action"]
        result = step["result_session_index"]
        H.ensure(
            type(result) is int and result == previous + int(advance) and result <= 20,
            "Action/session mismatch",
        )
        H.ensure(not step["done"] or index == len(trace) - 1, "Transition after terminal")
        previous = result
    H.ensure(type(done) is bool and previous == session, "Environment/trace disagreement")
    if done:
        H.ensure(
            bool(trace) and trace[-1]["done"] is True and session == 20,
            "COMPLETE requires twenty total advances",
        )
        H.ensure(
            [
                s["decision_session_index"]
                for s in trace
                if s["result_session_index"] > s["decision_session_index"]
            ]
            == list(range(20)),
            "Terminal advancing session sequence is incomplete",
        )
    else:
        H.ensure(session < 20 and not any(s["done"] for s in trace), "Unreported terminal")


def validate_prefix(env: Any, receipt: dict[str, Any], plan: dict[str, Any]) -> None:
    start = plan["start_session"]
    trace = [asdict(step) for step in env.credit_step_trace]
    H.ensure(
        env.decision_condition == plan["condition"] and env.diagnostic_start_session == start,
        "Condition/state construction drift",
    )
    H.ensure(
        receipt["is_model_request"] is False
        and receipt["model_requests"] == 0
        and receipt["start_session"] == receipt["prefix_action_count"] == start
        and receipt["credit_step_trace"] == trace
        and receipt["interaction_trace"] == env.interaction_trace
        and len(trace) == len(env.interaction_trace) == start,
        "Prefix must be separate recorded environment transitions, never model requests",
    )
    H.ensure(
        all(s["action"] == "hold" and not s["invalid_action"] for s in trace),
        "Prefix is not the registered four holds",
    )
    H.ensure(not env.ledger.positions and not env.ledger.fills, "Prefix changed portfolio holdings")
    check_trace(
        trace,
        done=env.episode_complete,
        session=env.current_session_index,
        sessions=env.episode_session_count,
    )
    H.ensure(
        not env.episode_complete and env.current_session_index == start, "Wrong model start session"
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    H.ensure(
        manifest.get("status") == "PREPARED"
        and manifest.get("schema_version") == 1
        and manifest.get("diagnostic_id") == "m2_decision_v1",
        "Wrong decision manifest",
    )
    H.ensure(
        manifest.get("grid") == grid_plan() and manifest.get("budget") == budget_contract(),
        "Registered population, horizon or seed identity changed",
    )
    H.ensure(
        list(manifest["models"]) == list(H.MODELS) and len(manifest["windows"]) == 2,
        "Wrong base/window population",
    )
    for key, model in manifest["models"].items():
        H.ensure(
            (model["name"], model["revision"]) == H.MODELS[key]
            and model["snapshot"] == str(H.snapshot(key))
            and model["engine_kwargs"] == engine_kwargs(H.snapshot(key)),
            "Base/engine identity changed",
        )
        H.ensure(
            list(model["initial_prompts"]) == list(MODEL_CONDITIONS[key]), "Wrong condition grid"
        )
        for condition, prompts in model["initial_prompts"].items():
            H.ensure(len(prompts) == 2, "Two windows required per condition")
            for index, prompt in enumerate(prompts):
                ids = prompt["prompt_ids"]
                H.ensure(
                    isinstance(ids, list)
                    and 0 < len(ids) <= PROMPT_CAP
                    and all(type(t) is int and t >= 0 for t in ids)
                    and prompt["prompt_token_count"] == len(ids)
                    and prompt["prompt_sha256"] == H.token_sha(ids)
                    and prompt["template_kwargs"] == {"enable_thinking": False}
                    and prompt["action_protocol"] == "single_tool_v1",
                    "Prompt contract changed",
                )
                baseline = model["initial_prompts"]["raw"][index]
                H.ensure(
                    prompt["tools"] == baseline["tools"]
                    and prompt["messages"][:-1] == baseline["messages"][:-1],
                    "Condition changed shared tools or prompt prefix",
                )
                if condition != "raw":
                    H.ensure(
                        ids != baseline["prompt_ids"], "Registered information intervention missing"
                    )


def verify_files(
    manifest: dict[str, Any], *, model_key: str | None = None, verify_weights: bool = False
) -> None:
    validate_manifest(manifest)
    sources = manifest["source_identity"]["source_sha256"]
    H.ensure(
        set(sources) == {str(p.relative_to(ROOT)) for p in source_paths()}, "Source inventory drift"
    )
    for name, expected in {**sources, **manifest["source_file_sha256"]}.items():
        H.ensure(H.digest(ROOT / name) == expected, f"Source/input drift: {name}")
    H.ensure(
        H.package_versions() == manifest["package_versions"], "Inference package version drift"
    )
    for key in [model_key] if model_key else H.MODELS:
        model = manifest["models"][key]
        files = {**model["metadata_sha256"], **(model["weight_sha256"] if verify_weights else {})}
        for name, expected in files.items():
            H.ensure(
                H.digest(Path(model["snapshot"]) / name) == expected,
                f"Model file drift: {key}/{name}",
            )


def prepare(manifest_path: Path) -> dict[str, Any]:
    H.ensure(
        manifest_path.resolve() == HERE / "manifest.json" and not manifest_path.exists(),
        "Use a new fixed decision manifest, never overwrite",
    )
    os.environ.update(
        USE_TORCH="0",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    H.add_import_paths()
    from transformers import AutoTokenizer

    conditions = module_from_path("decision_conditions_prepare", CONDITION_SCRIPT)
    source = source_identity()
    H.ensure(H.digest(PRIOR_MANIFEST) == PRIOR_SHA, "Prior base inventory changed")
    prior = H.read(PRIOR_MANIFEST)
    rows = H.read_train_rows()
    models: dict[str, Any] = {}
    prefix_receipts: dict[str, Any] = {}
    for key, old_model in prior["models"].items():
        model = {
            name: copy.deepcopy(old_model[name])
            for name in ("name", "revision", "snapshot", "metadata_sha256", "weight_sha256")
        }
        model["engine_kwargs"] = engine_kwargs(Path(model["snapshot"]))
        for name, expected in model["metadata_sha256"].items():
            H.ensure(
                H.digest(Path(model["snapshot"]) / name) == expected,
                "Base tokenizer/metadata changed",
            )
        tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
        model["initial_prompts"] = {}
        prefix_receipts[key] = {}
        for condition in MODEL_CONDITIONS[key]:
            model["initial_prompts"][condition] = []
            prefix_receipts[key][condition] = []
            for index in range(2):
                plan = next(
                    p
                    for p in grid_plan()
                    if p["model_key"] == key
                    and p["condition"] == condition
                    and p["window_index"] == index
                )
                env = conditions.create_env(rows, index, condition, plan["start_session"])
                obs, receipt = asyncio.run(conditions.reset_with_prefix(env))
                validate_prefix(env, receipt, plan)
                prompt = conditions.render_prompt(tokenizer, obs, env.tool_schemas)
                H.ensure(
                    len(prompt["prompt_ids"]) <= PROMPT_CAP, "Initial prompt exceeds fixed cap"
                )
                model["initial_prompts"][condition].append(prompt)
                prefix_receipts[key][condition].append(receipt)
        models[key] = model
    manifest = {
        "status": "PREPARED",
        "schema_version": 1,
        "diagnostic_id": "m2_decision_v1",
        "at": H.now(),
        "grid": grid_plan(),
        "budget": budget_contract(),
        "source_identity": source,
        "package_versions": H.package_versions(),
        "source_file_sha256": {
            str(PRIOR_MANIFEST.relative_to(ROOT)): PRIOR_SHA,
            str(H.PILOT.relative_to(ROOT)): H.PILOT_SHA,
        },
        "models": models,
        "prefix_receipts": prefix_receipts,
        "windows": [
            {
                "episode_id": H.EPISODES[i],
                "raw_prompt": row["prompt"],
                "env_kwargs": json.loads(row["env_kwargs"]),
            }
            for i, row in enumerate(rows)
        ],
        "weight_inventory_source": (
            "Prior SHA-bound inventory; run rehashes actual weights before GPU"
        ),
        "limitations": LIMITATIONS,
    }
    validate_manifest(manifest)
    H.ensure(
        source_identity()["source_sha256"] == source["source_sha256"],
        "Source changed during prepare",
    )
    H.ensure(
        not any(name in sys.modules for name in ("torch", "ray", "vllm")),
        "CPU preparation imported GPU framework",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    H.write_new(manifest_path, manifest)
    saved = H.read(manifest_path)
    for model in saved["models"].values():
        tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
        for prompts in model["initial_prompts"].values():
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
                H.ensure(ids == prompt["prompt_ids"], "Serialized prompt/schema ordering changed")
    H.write_new(
        HERE / "preparation.json",
        {
            "status": "PASSED",
            "at": H.now(),
            "manifest_sha256": H.digest(manifest_path),
            "script_sha256": H.digest(Path(__file__)),
            "new_model_requests": 0,
            "gpu_frameworks_imported": False,
            "budget": budget_contract(),
            "prompt_counts": {
                key: {
                    c: [p["prompt_token_count"] for p in prompts]
                    for c, prompts in model["initial_prompts"].items()
                }
                for key, model in models.items()
            },
        },
    )
    return saved


def initial_state(model_key: str, manifest_sha: str) -> dict[str, Any]:
    return {
        "status": "STARTING",
        "at": H.now(),
        "model_key": model_key,
        "pid": os.getpid(),
        "manifest_sha256": manifest_sha,
        "budget": budget_contract(),
        "grid": [
            {
                **p,
                "status": "UNSTARTED",
                "requests_submitted": 0,
                "responses_mapped": 0,
                "responses_decoded": 0,
                "actions_applied": 0,
                "response_tokens": 0,
                "steps": [],
                "invalid_count": 0,
                "prefix_steps_count": 0,
                "action_counts": {},
                "requested_tool_counts": {},
                "invalid_errors": {},
                "max_prompt_tokens": 0,
                "max_response_tokens": 0,
                "consecutive_repeated_tool_calls": 0,
                "nonadvancing_repeated_tool_calls": 0,
                "fill_count": 0,
                "ever_held_positions": False,
                "visible_close_count_min": None,
                "visible_close_count_max": None,
            }
            for p in grid_plan()
            if p["model_key"] == model_key
        ],
        "requests_submitted": 0,
        "responses_returned": 0,
        "responses_mapped": 0,
        "responses_decoded": 0,
        "actions_applied": 0,
        "prefix_actions_applied": 0,
        "returned_but_unmapped": 0,
        "evidence_files_sha256": {},
        "parameters_updated": False,
        "dev_policy_evaluations": 0,
        "limitations": LIMITATIONS,
    }


def remember(directory: Path, state: dict[str, Any], name: str, value: Any) -> None:
    H.write_new(directory / name, value)
    state["evidence_files_sha256"][name] = H.digest(directory / name)


def summaries(grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    counters = (
        "requests_submitted",
        "responses_mapped",
        "responses_decoded",
        "actions_applied",
        "prefix_steps_count",
        "invalid_count",
        "fill_count",
        "consecutive_repeated_tool_calls",
        "nonadvancing_repeated_tool_calls",
    )
    for row in grid:
        key = (row["condition"], row["window_index"])
        group = groups.setdefault(
            key,
            {
                "condition": key[0],
                "window_index": key[1],
                "trajectory_count": 0,
                "status_counts": {},
                "action_counts": {},
                "requested_tool_counts": {},
                "invalid_errors": {},
                "max_prompt_tokens": 0,
                "max_response_tokens": 0,
                "visible_close_count_min": None,
                "visible_close_count_max": None,
                "trajectories_with_positions": 0,
                **{name: 0 for name in counters},
            },
        )
        group["trajectory_count"] += 1
        status = row["status"]
        group["status_counts"][status] = group["status_counts"].get(status, 0) + 1
        for name in counters:
            group[name] += row[name]
        for name in ("action_counts", "requested_tool_counts", "invalid_errors"):
            for label, count in row[name].items():
                group[name][label] = group[name].get(label, 0) + count
        for name in ("max_prompt_tokens", "max_response_tokens"):
            group[name] = max(group[name], row[name])
        for name, choose in (("visible_close_count_min", min), ("visible_close_count_max", max)):
            if row[name] is not None:
                group[name] = row[name] if group[name] is None else choose(group[name], row[name])
        group["trajectories_with_positions"] += int(row["ever_held_positions"])
    return list(groups.values())


def persist_state(directory: Path, state: dict[str, Any]) -> None:
    state["summary_by_condition_window"] = summaries(state["grid"])
    H.set_state(directory, state)


def record_model_action(row: dict[str, Any], env: Any, text: str, before_session: int) -> None:
    """Summarize only actual policy actions, excluding every scripted prefix step."""
    step = env.credit_step_trace[-1]
    row["action_counts"][step.action] = row["action_counts"].get(step.action, 0) + 1
    info = env.interaction_trace[-1]["info"]
    if step.invalid_action:
        error = str(info.get("error", "invalid action without error detail"))
        row["invalid_errors"][error] = row["invalid_errors"].get(error, 0) + 1
    row["fill_count"] += len(info.get("fills", []))
    row["ever_held_positions"] |= bool(env.ledger.positions)
    try:
        _, calls = env.format_wrapper.parse_response(text)
        if len(calls) != 1:
            return
        call = calls[0]
        encoded = json.dumps(
            {"name": call.name, "arguments": call.arguments},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (ValueError, TypeError, KeyError):
        return
    row["requested_tool_counts"][call.name] = row["requested_tool_counts"].get(call.name, 0) + 1
    signature = hashlib.sha256(encoded.encode()).hexdigest()
    repeated = signature == row.get("last_tool_signature")
    row["consecutive_repeated_tool_calls"] += int(repeated)
    row["nonadvancing_repeated_tool_calls"] += int(
        repeated
        and before_session == row.get("last_tool_session")
        and env.current_session_index == before_session
    )
    row["last_tool_signature"], row["last_tool_session"] = signature, before_session


def set_incomplete(row: dict[str, Any], kind: str, detail: str) -> None:
    row.update(
        status="INCOMPLETE",
        complete=False,
        failure_kind=kind,
        failure_detail=detail,
        finished_at=H.now(),
    )


def finish_horizon(row: dict[str, Any]) -> None:
    row.update(
        status="HORIZON_REACHED",
        complete=False,
        finished_at=H.now(),
        horizon_reason="registered_model_request_budget_without_environment_terminal",
    )


def check_batch(outputs: list[Any], submissions: list[dict[str, Any]]) -> None:
    H.ensure(len(outputs) == len(submissions), "Missing/extra returned outputs; no replacements")
    for output, submission in zip(outputs, submissions, strict=True):
        H.ensure(output.request_id == str(submission["request_index"]), "Request ID/order mismatch")
        H.ensure(
            list(output.prompt_token_ids) == submission["prompt_ids"]
            and len(output.outputs) == 1
            and output.outputs[0].index == 0,
            "Prompt/sample mapping mismatch",
        )


async def execute_grid(
    llm: Any,
    sampling_class: Any,
    manifest: dict[str, Any],
    model_key: str,
    directory: Path,
    state: dict[str, Any],
    conditions: Any,
    make_env: Any,
    make_action: Any,
    check_sources: Any,
) -> dict[str, Any]:
    tokenizer = llm.get_tokenizer()
    try:
        for cohort in sorted({row["cohort_index"] for row in state["grid"]}):
            check_sources()
            members = [r for r in state["grid"] if r["cohort_index"] == cohort]
            by_index = {r["grid_index"]: r for r in members}
            contexts = {}
            for row in members:
                env = make_env(row["window_index"], row["condition"], row["start_session"])
                obs, receipt = await conditions.reset_with_prefix(env)
                validate_prefix(env, receipt, row)
                prompt = conditions.render_prompt(tokenizer, obs, env.tool_schemas)
                H.ensure(
                    prompt
                    == manifest["models"][model_key]["initial_prompts"][row["condition"]][
                        row["window_index"]
                    ],
                    "Initial condition prompt differs from frozen real prefix",
                )
                remember(
                    directory, state, f"trajectory-{row['grid_index']:03d}.prefix.json", receipt
                )
                row.update(
                    status="ACTIVE",
                    started_at=H.now(),
                    prefix_action_count=row["start_session"],
                    prefix_steps_count=row["start_session"],
                    current_session_index=env.current_session_index,
                    steps=[asdict(s) for s in env.credit_step_trace],
                )
                state["prefix_actions_applied"] += row["start_session"]
                contexts[row["grid_index"]] = {"env": env, "obs": obs}
            round_index = 0
            while any(row["status"] == "ACTIVE" for row in members):
                check_sources()
                submissions: list[dict[str, Any]] = []
                for row in members:
                    if row["status"] != "ACTIVE":
                        continue
                    if row["requests_submitted"] >= row["request_limit"]:
                        finish_horizon(row)
                        continue
                    context = contexts[row["grid_index"]]
                    prompt = conditions.render_prompt(
                        tokenizer, context["obs"], context["env"].tool_schemas
                    )
                    row["max_prompt_tokens"] = max(
                        row["max_prompt_tokens"], len(prompt["prompt_ids"])
                    )
                    pit_counts = context["env"].diagnostic_pit_counts
                    values = list(pit_counts.values())
                    H.ensure(
                        bool(values) and all(type(n) is int and n >= 0 for n in values),
                        "Invalid diagnostic visible-close record counts",
                    )
                    row["visible_close_count_min"] = (
                        min(values)
                        if row["visible_close_count_min"] is None
                        else min(row["visible_close_count_min"], min(values))
                    )
                    row["visible_close_count_max"] = (
                        max(values)
                        if row["visible_close_count_max"] is None
                        else max(row["visible_close_count_max"], max(values))
                    )
                    if len(prompt["prompt_ids"]) > PROMPT_CAP:
                        set_incomplete(
                            row, "PROMPT_CAP", "Full observation exceeded 8192 prompt tokens"
                        )
                        remember(
                            directory,
                            state,
                            f"trajectory-{row['grid_index']:03d}.prompt_failure.json",
                            {
                                "at": H.now(),
                                "trajectory_id": row["trajectory_id"],
                                "model_request_index": row["requests_submitted"],
                                **prompt,
                            },
                        )
                        continue
                    seed = request_seed(row, row["requests_submitted"])
                    submissions.append(
                        {
                            **prompt,
                            "request_index": state["requests_submitted"] + len(submissions),
                            "trajectory_request_index": row["requests_submitted"],
                            "grid_index": row["grid_index"],
                            "trajectory_id": row["trajectory_id"],
                            "condition": row["condition"],
                            "identity": row["identity"],
                            "episode_id": row["episode_id"],
                            "seed": seed,
                            "seed_env_step": row["requests_submitted"],
                            "environment_trace_index": len(context["env"].credit_step_trace),
                            "environment_session_index": context["env"].current_session_index,
                            "feature_evidence": context["env"].feature_evidence,
                            "diagnostic_pit_counts": pit_counts,
                        }
                    )
                persist_state(directory, state)
                if not submissions:
                    break
                H.ensure(
                    len(submissions) <= MAX_BATCH
                    and state["requests_submitted"] + len(submissions) <= 4096,
                    "Fixed per-model request budget exceeded",
                )
                params = [sampling_class(**H.sampling_kwargs(s)) for s in submissions]
                effective = [H.effective_params(llm, p) for p in params]
                prefix = f"cohort-{cohort}-round-{round_index:03d}"
                remember(
                    directory,
                    state,
                    prefix + ".submitted.json",
                    {
                        "at": H.now(),
                        "requests": [
                            {**s, "requested_sampling_params": H.sampling_kwargs(s), **e}
                            for s, e in zip(submissions, effective, strict=True)
                        ],
                    },
                )
                for submission in submissions:
                    by_index[submission["grid_index"]]["requests_submitted"] += 1
                state["requests_submitted"] += len(submissions)
                state["last_batch"] = {
                    "prefix": prefix,
                    "phase": "SUBMITTED",
                    "size": len(submissions),
                }
                persist_state(directory, state)
                outputs = llm.generate(
                    [{"prompt_token_ids": s["prompt_ids"]} for s in submissions],
                    params,
                    use_tqdm=False,
                )
                raw = [H.raw_output(output) for output in outputs]
                remember(directory, state, prefix + ".raw.json", {"at": H.now(), "outputs": raw})
                state["responses_returned"] += len(raw)
                state["returned_but_unmapped"] = (
                    state["responses_returned"] - state["responses_mapped"]
                )
                state["last_batch"]["phase"] = "RAW_OBSERVED"
                persist_state(directory, state)
                check_batch(outputs, submissions)
                for submission in submissions:
                    by_index[submission["grid_index"]]["responses_mapped"] += 1
                state["responses_mapped"] += len(submissions)
                state["returned_but_unmapped"] = (
                    state["responses_returned"] - state["responses_mapped"]
                )
                state["last_batch"]["phase"] = "MAPPED"
                check_sources()
                audited = []
                try:
                    for submission, returned, settings in zip(
                        submissions, raw, effective, strict=True
                    ):
                        response = returned["outputs"][0]
                        tokens = response["token_ids"]
                        H.ensure(
                            all(type(t) is int and t >= 0 for t in tokens),
                            "Malformed returned tokens",
                        )
                        response = {
                            **response,
                            "raw_text": tokenizer.decode(tokens, skip_special_tokens=False),
                            "action_text": tokenizer.decode(tokens, skip_special_tokens=True),
                        }
                        complete = H.response_complete(
                            tokens, response["raw_finish_reason"], H.EOS_IDS
                        )
                        row = by_index[submission["grid_index"]]
                        row["responses_decoded"] += 1
                        row["response_tokens"] += len(tokens)
                        row["max_response_tokens"] = max(row["max_response_tokens"], len(tokens))
                        state["responses_decoded"] += 1
                        if not complete:
                            # Persist a permanent terminal diagnostic state before any peer action.
                            set_incomplete(
                                row, "RESPONSE_INCOMPLETE", "Native EOS/4096-cap contract failed"
                            )
                        audited.append(
                            {
                                "request_index": submission["request_index"],
                                "grid_index": row["grid_index"],
                                "response": response,
                                "response_sha256": H.token_sha(tokens),
                                "response_complete": complete,
                                "effective_sampling": settings,
                            }
                        )
                except BaseException:
                    remember(
                        directory,
                        state,
                        prefix + ".audited_prefix.json",
                        {"rows": audited, "is_complete_batch": False, "returned_count": len(raw)},
                    )
                    raise
                remember(
                    directory, state, prefix + ".audited.json", {"at": H.now(), "rows": audited}
                )
                state["last_batch"]["phase"] = "DECODED"
                persist_state(directory, state)
                for submission, observed in zip(submissions, audited, strict=True):
                    row = by_index[submission["grid_index"]]
                    if not observed["response_complete"]:
                        H.ensure(
                            row["status"] == "INCOMPLETE", "Incomplete response was reactivated"
                        )
                        continue
                    H.ensure(
                        row["status"] == "ACTIVE",
                        "Attempt to apply output to a terminal trajectory",
                    )
                    context = contexts[row["grid_index"]]
                    env, response = context["env"], observed["response"]
                    before = len(env.credit_step_trace)
                    before_session = env.current_session_index
                    obs, _reward, done, _info = await env.step(
                        make_action(response["action_text"], response["token_ids"])
                    )
                    row["actions_applied"] += 1
                    state["actions_applied"] += 1
                    trace = [asdict(s) for s in env.credit_step_trace]
                    interactions = env.interaction_trace
                    row.update(
                        steps=trace,
                        current_session_index=env.current_session_index,
                        invalid_count=sum(s["invalid_action"] for s in trace),
                    )
                    record_model_action(row, env, response["action_text"], before_session)
                    remember(
                        directory,
                        state,
                        f"request-{submission['request_index']:06d}.applied.json",
                        {
                            "status": "ACTION_APPLIED",
                            "at": H.now(),
                            "request_index": submission["request_index"],
                            "grid_index": row["grid_index"],
                            "trajectory_request_index": submission["trajectory_request_index"],
                            "environment_trace_index": before,
                            "prefix_action_count": row["prefix_action_count"],
                            "before_session": before_session,
                            "result_session": env.current_session_index,
                            "done": done,
                            "new_trace": trace[before:],
                            "environment_interaction_trace": interactions[before:],
                            "environment_complete": env.episode_complete,
                        },
                    )
                    H.ensure(
                        len(trace) == before + 1
                        and len(interactions) == len(trace)
                        and len(trace) == row["prefix_action_count"] + row["actions_applied"]
                        and interactions[-1]["response"]["sha256"]
                        == hashlib.sha256(response["action_text"].encode()).hexdigest(),
                        "Response/prefix/step capture alignment failed",
                    )
                    H.ensure(
                        type(done) is bool and done is env.episode_complete,
                        "Environment terminal flag mismatch",
                    )
                    check_trace(
                        trace,
                        done=done,
                        session=env.current_session_index,
                        sessions=env.episode_session_count,
                    )
                    context["obs"] = obs
                    if done:
                        row.update(
                            status="COMPLETE",
                            complete=True,
                            finished_at=H.now(),
                            terminal_metrics=env.screening_terminal_metrics(),
                            fills=[asdict(fill) for fill in env.ledger.fills],
                        )
                    elif row["requests_submitted"] >= row["request_limit"]:
                        finish_horizon(row)
                    state["last_batch"]["phase"] = "APPLY_PREFIX"
                    state["last_batch"]["last_applied_request_index"] = submission["request_index"]
                state["last_batch"]["phase"] = "FINISHED"
                persist_state(directory, state)
                round_index += 1
                print(
                    json.dumps(
                        {
                            "cohort": cohort,
                            "round": round_index,
                            "requests": state["requests_submitted"],
                            "complete": sum(r["status"] == "COMPLETE" for r in state["grid"]),
                            "horizon": sum(r["status"] == "HORIZON_REACHED" for r in state["grid"]),
                            "incomplete": sum(r["status"] == "INCOMPLETE" for r in state["grid"]),
                        }
                    ),
                    flush=True,
                )
        H.ensure(
            all(row["status"] in OBSERVED_STATUSES for row in state["grid"]),
            "Unattempted fixed population remains",
        )
        H.ensure(
            state["requests_submitted"]
            == state["responses_returned"]
            == state["responses_mapped"]
            == state["responses_decoded"]
            <= 4096,
            "Successful observation lost returned requests",
        )
        state.update(
            status="OBSERVED",
            finished_at=H.now(),
            trajectory_count=len(state["grid"]),
            complete_count=sum(r["status"] == "COMPLETE" for r in state["grid"]),
            horizon_count=sum(r["status"] == "HORIZON_REACHED" for r in state["grid"]),
            incomplete_count=sum(r["status"] == "INCOMPLETE" for r in state["grid"]),
            is_training_gate=False,
            is_screening_pass=False,
        )
        persist_state(directory, state)
        H.write_new(directory / "result.json", state)
        return state
    except BaseException as error:
        for row in state["grid"]:
            if row["status"] == "ACTIVE":
                row.update(
                    status="ABORTED_RUNTIME",
                    failure_kind=type(error).__name__,
                    failure_detail=str(error),
                )
        state.update(
            status="STOPPED", finished_at=H.now(), error_type=type(error).__name__, error=str(error)
        )
        persist_state(directory, state)
        raise


def run(manifest_path: Path, expected_sha: str, model_key: str) -> int:
    H.ensure(manifest_path.resolve() == HERE / "manifest.json", "Wrong decision manifest path")
    directory = HERE / model_key
    H.ensure(
        not directory.exists() and not directory.is_symlink(), "Refusing existing model attempt"
    )
    directory.mkdir(parents=True, exist_ok=False)
    state = initial_state(model_key, expected_sha)
    persist_state(directory, state)

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Diagnostic interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        H.ensure(H.digest(manifest_path) == expected_sha, "Manifest SHA mismatch")
        manifest = H.read(manifest_path)
        verify_files(manifest, model_key=model_key, verify_weights=True)
        H.ensure(
            H.version("vllm") == "0.9.2" and H.version("torch") == "2.7.0+cu128",
            "Wrong inference runtime",
        )
        H.ensure(not any(k.startswith("FINANCE_") for k in os.environ), "Inherited training hooks")
        H.ensure(os.environ.get("USE_TORCH", "1") != "0", "CPU-only prepare environment inherited")
        for key in ("VLLM_USE_V1", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            H.ensure(os.environ.get(key, "1") == "1", f"Wrong {key}")
            os.environ[key] = "1"
        H.add_import_paths()
        from agent_r1.env.base import Action
        from transformers import AutoTokenizer

        conditions = module_from_path("decision_conditions_run", CONDITION_SCRIPT)
        rows = H.read_train_rows()
        model = manifest["models"][model_key]
        tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
        for condition in MODEL_CONDITIONS[model_key]:
            for index, row in enumerate(rows):
                H.ensure(
                    row["prompt"] == manifest["windows"][index]["raw_prompt"]
                    and json.loads(row["env_kwargs"]) == manifest["windows"][index]["env_kwargs"],
                    "Train row changed",
                )
                plan = next(
                    p
                    for p in grid_plan()
                    if p["model_key"] == model_key
                    and p["condition"] == condition
                    and p["window_index"] == index
                )
                env = conditions.create_env(rows, index, condition, plan["start_session"])
                obs, receipt = asyncio.run(conditions.reset_with_prefix(env))
                validate_prefix(env, receipt, plan)
                H.ensure(
                    conditions.render_prompt(tokenizer, obs, env.tool_schemas)
                    == model["initial_prompts"][condition][index],
                    "Live initial condition/prefix prompt changed",
                )
        remember(
            directory,
            state,
            "environment_preflight.json",
            {
                "status": "PASSED",
                "at": H.now(),
                "episodes": list(H.EPISODES),
                "conditions": list(MODEL_CONDITIONS[model_key]),
                "new_model_requests": 0,
            },
        )
        from vllm import LLM, SamplingParams

        llm = LLM(**model["engine_kwargs"])
        state.update(status="RUNNING", engine_ready_at=H.now())
        persist_state(directory, state)

        def check_sources() -> None:
            H.ensure(H.digest(manifest_path) == expected_sha, "Manifest drift during execution")
            verify_files(manifest, model_key=model_key)

        asyncio.run(
            execute_grid(
                llm,
                SamplingParams,
                manifest,
                model_key,
                directory,
                state,
                conditions,
                lambda index, condition, start: conditions.create_env(
                    rows, index, condition, start
                ),
                lambda text, ids: Action(text=text, token_ids=ids),
                check_sources,
            )
        )
        return 0
    except BaseException as error:
        state.update(
            status="STOPPED", finished_at=H.now(), error_type=type(error).__name__, error=str(error)
        )
        persist_state(directory, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify", "run"):
        action = sub.add_parser(name)
        action.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
        if name != "prepare":
            action.add_argument("--expected-manifest-sha256", required=True)
        if name == "run":
            action.add_argument("--model", choices=list(H.MODELS), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest)
        print(json.dumps({"status": "PREPARED", "manifest_sha256": H.digest(args.manifest)}))
        return 0
    if args.command == "verify":
        H.ensure(H.digest(args.manifest) == args.expected_manifest_sha256, "Manifest SHA mismatch")
        verify_files(H.read(args.manifest))
        print(json.dumps({"status": "VERIFIED", "new_model_requests": 0}))
        return 0
    return run(args.manifest, args.expected_manifest_sha256, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
