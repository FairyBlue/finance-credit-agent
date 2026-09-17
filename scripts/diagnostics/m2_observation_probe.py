#!/usr/bin/env python3
"""Bounded paired Train trajectories for legacy versus history/result-memory observations.

The fixed population is 32 trajectories, with no training, replacement, or Dev
policy evaluation. prepare/verify never import torch/Ray/vLLM. A generation or
prompt cap ends only that trajectory; infrastructure/identity failure stops the
model. The remaining fixed population is always retained in the evidence.
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
HERE = ROOT / "runs/m2-hold-diagnostic-20260909/observation-v1"
PLAN_DOC = ROOT / "docs/2026-09-09_M2_OBSERVATION_DIAGNOSTIC_PLAN.md"
HOLD_SCRIPT = ROOT / "scripts/diagnostics/m2_hold_probe.py"
HOLD_SHA = "7d610d7ff5343fac95b023470d9af699615ca93572d8bf7b62950e36d1a7153a"
PRIOR_MANIFEST = ROOT / "runs/m2-hold-diagnostic-20260909/probe-v1/manifest.json"
PRIOR_SHA = "0b3ac1091f20182a0ae0ad74fad817b0f68394956bb5ceefbf6c24ea74916ab3"
PROFILES = ("legacy", "history_memory_v1")
MAX_STEPS = 256
MAX_PROMPT = 8192
MAX_RESPONSE = 4096
MAX_CONTEXT = 12288
MAX_TRAJECTORIES = 32
MAX_REQUESTS = 8192
ADVANCE_ACTIONS = {"buy", "sell", "hold", "abstain"}

# Reuse the frozen standalone inference contract, never its historical CLI/gates.
if hashlib.sha256(HOLD_SCRIPT.read_bytes()).hexdigest() != HOLD_SHA:
    raise RuntimeError("Frozen hold diagnostic helper changed")
_spec = importlib.util.spec_from_file_location("observation_hold_helpers", HOLD_SCRIPT)
assert _spec is not None and _spec.loader is not None
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

LIMITATIONS = [
    "Fresh base policies, not R5 trained weights; no policy updates are performed.",
    "Both conditions use repaired schemas and single_tool_v1 without the C objective.",
    "This is a paired observation diagnostic on two Train windows, not a strategy comparison.",
    "Fixed seeds do not promise tokenwise replay when observations or active batch sizes differ.",
    "Incomplete trajectories remain negative results; no refill, retry, "
    "arm-wise removal or reward selection.",
    "Terminal scorer metrics are descriptive; no training advantages, residual calculation, "
    "return-based selection, Dev policy evaluation, formal Valid, Final OOS or main screening.",
    "Both conditions use a new 8192-prompt/12288-context resource contract; "
    "earlier training gates do not validate this configuration.",
]


def grid_plan() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model_key in H.MODELS:
        for profile in PROFILES:
            for window_index, episode_id in enumerate(H.EPISODES):
                for rollout_id in range(4):
                    result.append(
                        {
                            "grid_index": len(result),
                            "model_key": model_key,
                            "profile": profile,
                            "window_index": window_index,
                            "episode_id": episode_id,
                            "rollout_id": rollout_id,
                            "trajectory_id": f"{model_key}-{profile}-w{window_index}-r{rollout_id}",
                            "identity": {
                                "step": 1,
                                "sample_index": window_index,
                                "rollout_n": rollout_id,
                                "validate": False,
                            },
                            "initial_status": "UNSTARTED",
                        }
                    )
    return result


def budget_contract() -> dict[str, int]:
    return {
        "trajectories": 32,
        "trajectories_per_model": 16,
        "siblings_per_window": 4,
        "max_active_batch": 8,
        "max_requests_per_trajectory": MAX_STEPS,
        "max_requests_total": MAX_REQUESTS,
        "prompt_cap": MAX_PROMPT,
        "response_cap": MAX_RESPONSE,
        "context_cap": MAX_CONTEXT,
        "max_response_tokens_total": MAX_REQUESTS * MAX_RESPONSE,
        "terminal_sessions": 21,
        "terminal_advances": 20,
    }


def engine_kwargs(model: Path) -> dict[str, Any]:
    result = H.engine_kwargs(model)
    result["max_model_len"] = MAX_CONTEXT
    return result


def request_seed(planned: dict[str, Any], env_step: int) -> int:
    H.add_import_paths()
    from src.agents.rollout_seed import derive_request_seed

    H.ensure(
        type(env_step) is int and 0 <= env_step < MAX_STEPS,
        "Environment request index outside budget",
    )
    return derive_request_seed(42, planned["identity"], env_step)


def source_paths() -> list[Path]:
    return sorted(
        set([p for p in H.source_paths() if p != H.PLAN_DOC] + [PLAN_DOC, Path(__file__).resolve()])
    )


def source_identity() -> dict[str, Any]:
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "agent_r1_commit": subprocess.check_output(
            ["git", "-C", "third_party/agent-r1", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): H.digest(path) for path in source_paths()},
    }


def prior_evidence() -> tuple[dict[str, Any], dict[str, str]]:
    H.ensure(H.digest(PRIOR_MANIFEST) == PRIOR_SHA, "Prior frozen first-step manifest drift")
    prior = H.read(PRIOR_MANIFEST)
    inventory = {str(PRIOR_MANIFEST.relative_to(ROOT)): PRIOR_SHA}
    for model_key in H.MODELS:
        path = PRIOR_MANIFEST.parent / model_key / "result.json"
        result = H.read(path)
        H.ensure(
            result.get("status") == "OBSERVED"
            and result.get("model_key") == model_key
            and result.get("manifest_sha256") == PRIOR_SHA
            and result.get("request_count") == 24,
            "Both previous 24-request model diagnostics must be observed",
        )
        planned = [p for p in H.build_request_plan() if p["model_key"] == model_key]
        rows = result.get("requests", [])
        H.ensure(
            len(rows) == 24
            and all(
                all(row.get(k) == v for k, v in plan.items())
                for row, plan in zip(rows, planned, strict=True)
            ),
            "Previous fixed request identities changed",
        )
        inventory[str(path.relative_to(ROOT))] = H.digest(path)
    return prior, inventory


def render(tokenizer: Any, observation: Any, tools: Any) -> dict[str, Any]:
    H.add_import_paths()
    from src.agents.screening_prompt import apply_screening_action_protocol

    H.ensure(
        observation.messages is not None
        and observation.text is None
        and observation.token_ids is None,
        "Expected actual message observation",
    )
    original = copy.deepcopy(observation.messages)
    messages = apply_screening_action_protocol(observation.messages, "single_tool_v1")
    H.ensure(observation.messages == original, "Prompt helper mutated the environment observation")
    ids = list(
        tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
    )
    H.ensure(
        bool(ids) and all(type(token) is int and token >= 0 for token in ids),
        "Malformed prompt tokens",
    )
    return {
        "messages": messages,
        "tools": copy.deepcopy(tools),
        "template_kwargs": {"enable_thinking": False},
        "action_protocol": "single_tool_v1",
        "prompt_ids": ids,
        "prompt_sha256": H.token_sha(ids),
        "prompt_token_count": len(ids),
    }


def make_real_env(rows: list[dict[str, Any]], index: int, profile: str) -> Any:
    from src.env.trading import TradingEnv

    H.ensure(profile in PROFILES, "Unknown observation condition")
    env = TradingEnv(
        **{
            **json.loads(rows[index]["env_kwargs"]),
            "reward_mode": "outcome_only",
            "capture_interactions": True,
            "observation_profile": profile,
        }
    )
    H.ensure(env.observation_profile == profile, "Environment ignored observation profile")
    return env


def prepare(manifest_path: Path) -> dict[str, Any]:
    H.ensure(
        manifest_path.resolve() == HERE / "manifest.json" and not manifest_path.exists(),
        "Manifest must be the new fixed observation-v1 path",
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

    prior, provenance = prior_evidence()
    source = source_identity()
    rows = H.read_train_rows()
    windows = [
        {
            "episode_id": H.EPISODES[i],
            "source_index": 4 * i,
            "raw_prompt": row["prompt"],
            "env_kwargs": json.loads(row["env_kwargs"]),
        }
        for i, row in enumerate(rows)
    ]
    contexts: dict[str, list[tuple[Any, Any]]] = {}
    for profile in PROFILES:
        contexts[profile] = []
        for index, row in enumerate(rows):
            env = make_real_env(rows, index, profile)
            obs = env.reset(raw_prompt=copy.deepcopy(row["prompt"]))
            H.ensure(
                env.current_session_index == 0
                and not env.episode_complete
                and env.episode_session_count == 21
                and not env.credit_step_trace,
                "Wrong initial Train environment",
            )
            contexts[profile].append((env, obs))
    models = {}
    for key, old_model in prior["models"].items():
        H.ensure(
            (old_model["name"], old_model["revision"]) == H.MODELS[key]
            and old_model["snapshot"] == str(H.snapshot(key)),
            "Prior base model identity changed",
        )
        model = {
            k: copy.deepcopy(old_model[k])
            for k in (
                "name",
                "revision",
                "snapshot",
                "metadata_sha256",
                "weight_sha256",
                "engine_kwargs",
            )
        }
        model["engine_kwargs"] = engine_kwargs(Path(model["snapshot"]))
        for filename, expected in model["metadata_sha256"].items():
            H.ensure(
                H.digest(Path(model["snapshot"]) / filename) == expected, "Model metadata drift"
            )
        tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
        model["initial_prompts"] = {}
        for profile, pairs in contexts.items():
            prompts = [render(tokenizer, obs, env.tool_schemas) for env, obs in pairs]
            H.ensure(
                all(p["prompt_token_count"] <= MAX_PROMPT for p in prompts),
                "Initial prompt already exceeds fixed capacity",
            )
            if profile == "legacy":
                H.ensure(
                    all(
                        p["prompt_ids"] == old_model["prompts"]["B"][i]["prompt_ids"]
                        for i, p in enumerate(prompts)
                    ),
                    "Legacy profile differs from prior repaired-schema B",
                )
            model["initial_prompts"][profile] = prompts
        models[key] = model
    manifest = {
        "status": "PREPARED",
        "schema_version": 1,
        "at": H.now(),
        "diagnostic_id": "m2_observation_trajectory_v1",
        "grid": grid_plan(),
        "budget": budget_contract(),
        "source_identity": source,
        "package_versions": H.package_versions(),
        "windows": windows,
        "models": models,
        "source_file_sha256": {**provenance, str(H.PILOT.relative_to(ROOT)): H.PILOT_SHA},
        "weight_inventory_source": (
            "Prior SHA-bound base inventory; run rehashes actual model weights before GPU"
        ),
        "common_objective_text": None,
        "action_protocol": "single_tool_v1",
        "single_trajectory_failure": (
            "Record INCOMPLETE and continue remaining fixed population without refill"
        ),
        "runtime_failure": "STOP model; preserve UNSTARTED entries; no retry",
        "limitations": LIMITATIONS,
    }
    H.ensure(
        source_identity()["source_sha256"] == source["source_sha256"],
        "Source drift during preparation",
    )
    validate_manifest(manifest)
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
                H.ensure(
                    ids == prompt["prompt_ids"], "Serialized tool/message ordering changed tokens"
                )
    H.write_new(
        HERE / "preparation.json",
        {
            "status": "PASSED",
            "at": H.now(),
            "manifest_sha256": H.digest(manifest_path),
            "script_sha256": H.digest(Path(__file__)),
            "real_train_resets": 4,
            "new_model_requests": 0,
            "gpu_frameworks_imported": False,
            "budget": budget_contract(),
            "prompt_counts": {
                key: {
                    p: [r["prompt_token_count"] for r in prompts]
                    for p, prompts in model["initial_prompts"].items()
                }
                for key, model in models.items()
            },
        },
    )
    return saved


def validate_manifest(manifest: dict[str, Any]) -> None:
    H.ensure(
        manifest.get("status") == "PREPARED"
        and manifest.get("schema_version") == 1
        and manifest.get("diagnostic_id") == "m2_observation_trajectory_v1",
        "Wrong trajectory diagnostic manifest",
    )
    H.ensure(
        manifest.get("budget") == budget_contract() and manifest.get("grid") == grid_plan(),
        "Fixed grid or request budget changed",
    )
    H.ensure(
        manifest.get("common_objective_text") is None
        and manifest.get("action_protocol") == "single_tool_v1",
        "Common prompt protocol changed",
    )
    H.ensure(
        list(manifest["models"]) == list(H.MODELS) and len(manifest["windows"]) == 2,
        "Model/window population changed",
    )
    for key, model in manifest["models"].items():
        H.ensure(
            (model["name"], model["revision"]) == H.MODELS[key]
            and model["snapshot"] == str(H.snapshot(key))
            and model["engine_kwargs"] == engine_kwargs(H.snapshot(key)),
            "Engine/base contract changed",
        )
        H.ensure(
            list(model["initial_prompts"]) == list(PROFILES), "Observation condition/order changed"
        )
        for profile, prompts in model["initial_prompts"].items():
            H.ensure(len(prompts) == 2, "Two initial prompts required per profile")
            for index, prompt in enumerate(prompts):
                ids = prompt["prompt_ids"]
                H.ensure(
                    isinstance(ids, list)
                    and 0 < len(ids) <= MAX_PROMPT
                    and all(type(t) is int and t >= 0 for t in ids)
                    and prompt["prompt_token_count"] == len(ids)
                    and prompt["prompt_sha256"] == H.token_sha(ids)
                    and prompt["template_kwargs"] == {"enable_thinking": False},
                    "Initial prompt contract drift",
                )
                H.ensure(
                    prompt["tools"] == model["initial_prompts"]["legacy"][index]["tools"],
                    "Tools differ between observation conditions",
                )
                legacy = model["initial_prompts"]["legacy"][index]
                if profile == "history_memory_v1":
                    H.ensure(
                        prompt["prompt_ids"] != legacy["prompt_ids"]
                        and prompt["messages"][:-1] == legacy["messages"][:-1],
                        "New observation must change state, while preserving the common prompt",
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
    H.ensure(H.package_versions() == manifest["package_versions"], "Runtime package version drift")
    for key in [model_key] if model_key else H.MODELS:
        model = manifest["models"][key]
        inventory = {
            **model["metadata_sha256"],
            **(model["weight_sha256"] if verify_weights else {}),
        }
        for name, expected in inventory.items():
            H.ensure(
                H.digest(Path(model["snapshot"]) / name) == expected,
                f"Model file drift: {key}/{name}",
            )


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
            "Discontinuous decision trace",
        )
        result = step["result_session_index"]
        H.ensure(
            type(result) is int
            and type(step["done"]) is bool
            and type(step["invalid_action"]) is bool,
            "Noncanonical transition/status types",
        )
        advancing = step["action"] in ADVANCE_ACTIONS and not step["invalid_action"]
        H.ensure(
            result == previous + int(advancing) and result <= 20,
            "Action/session transition mismatch",
        )
        H.ensure(not step["done"] or index == len(trace) - 1, "Action after terminal")
        previous = result
    H.ensure(previous == session and type(done) is bool, "Environment/trace session mismatch")
    if done:
        H.ensure(
            bool(trace) and trace[-1]["done"] is True and session == 20,
            "Terminal requires all twenty advances",
        )
        H.ensure(
            [
                s["decision_session_index"]
                for s in trace
                if s["result_session_index"] > s["decision_session_index"]
            ]
            == list(range(20)),
            "Terminal advancing sessions are incomplete",
        )
    else:
        H.ensure(session < 20 and not any(s["done"] for s in trace), "Unreported terminal")


def initial_state(model_key: str, manifest_sha: str) -> dict[str, Any]:
    return {
        "status": "STARTING",
        "at": H.now(),
        "model_key": model_key,
        "manifest_sha256": manifest_sha,
        "pid": os.getpid(),
        "budget": budget_contract(),
        "grid": [
            {
                **row,
                "status": "UNSTARTED",
                "requests_submitted": 0,
                "responses_returned": 0,
                "steps": [],
                "invalid_count": 0,
                "current_session_index": 0,
                "response_tokens": 0,
            }
            for row in grid_plan()
            if row["model_key"] == model_key
        ],
        "evidence_files_sha256": {},
        "requests_submitted": 0,
        "responses_returned": 0,
        "actions_applied": 0,
        "parameters_updated": False,
        "dev_policy_evaluations": 0,
        "limitations": LIMITATIONS,
    }


def remember(directory: Path, state: dict[str, Any], name: str, value: Any) -> None:
    H.write_new(directory / name, value)
    state["evidence_files_sha256"][name] = H.digest(directory / name)


def incomplete(row: dict[str, Any], kind: str, detail: str) -> None:
    row.update(
        status="INCOMPLETE",
        complete=False,
        failure_kind=kind,
        failure_detail=detail,
        finished_at=H.now(),
    )


def check_batch(outputs: list[Any], submissions: list[dict[str, Any]]) -> None:
    H.ensure(len(outputs) == len(submissions), "Missing/extra outputs; no replacement allowed")
    for output, submission in zip(outputs, submissions, strict=True):
        H.ensure(
            output.request_id == str(submission["request_index"]), "vLLM request ID/order drift"
        )
        H.ensure(
            list(output.prompt_token_ids) == submission["prompt_ids"]
            and len(output.outputs) == 1
            and output.outputs[0].index == 0,
            "Prompt/sample mapping drift",
        )


async def execute_grid(
    llm: Any,
    sampling_class: Any,
    manifest: dict[str, Any],
    model_key: str,
    directory: Path,
    state: dict[str, Any],
    make_env: Any,
    make_action: Any,
    check_sources: Any,
) -> dict[str, Any]:
    tokenizer = llm.get_tokenizer()
    try:
        for profile in PROFILES:
            check_sources()
            members = [row for row in state["grid"] if row["profile"] == profile]
            contexts = {}
            for row in members:
                env = make_env(row["window_index"], profile)
                obs = env.reset(
                    raw_prompt=copy.deepcopy(manifest["windows"][row["window_index"]]["raw_prompt"])
                )
                H.ensure(
                    env.current_session_index == 0
                    and not env.episode_complete
                    and env.episode_session_count == 21
                    and not env.credit_step_trace,
                    "Unexpected initial environment",
                )
                prompt = render(tokenizer, obs, env.tool_schemas)
                H.ensure(
                    prompt
                    == manifest["models"][model_key]["initial_prompts"][profile][
                        row["window_index"]
                    ],
                    "Initial model prompt differs from frozen actual Train reset",
                )
                contexts[row["grid_index"]] = {"env": env, "obs": obs, "env_step": 0}
                row.update(status="ACTIVE", started_at=H.now())
            round_index = 0
            while any(row["status"] == "ACTIVE" for row in members):
                check_sources()
                submissions: list[dict[str, Any]] = []
                for row in members:
                    if row["status"] != "ACTIVE":
                        continue
                    context = contexts[row["grid_index"]]
                    if context["env_step"] >= MAX_STEPS:
                        incomplete(row, "STEP_CAP", "256 requests exhausted without terminal")
                        continue
                    prompt = render(tokenizer, context["obs"], context["env"].tool_schemas)
                    if len(prompt["prompt_ids"]) > MAX_PROMPT:
                        incomplete(
                            row,
                            "PROMPT_CAP",
                            "Full observation exceeded the registered 8192 prompt cap",
                        )
                        remember(
                            directory,
                            state,
                            f"trajectory-{row['grid_index']:02d}.prompt_failure.json",
                            {
                                "at": H.now(),
                                "trajectory_id": row["trajectory_id"],
                                "env_step": context["env_step"],
                                **prompt,
                            },
                        )
                        continue
                    seed = request_seed(row, context["env_step"])
                    submissions.append(
                        {
                            **prompt,
                            "request_index": state["requests_submitted"] + len(submissions),
                            "grid_index": row["grid_index"],
                            "trajectory_id": row["trajectory_id"],
                            "identity": row["identity"],
                            "episode_id": row["episode_id"],
                            "profile": profile,
                            "env_step": context["env_step"],
                            "seed": seed,
                        }
                    )
                H.set_state(directory, state)
                if not submissions:
                    break
                H.ensure(
                    len(submissions) <= 8
                    and state["requests_submitted"] + len(submissions) <= 16 * MAX_STEPS,
                    "Model request budget exceeded",
                )
                params = [
                    sampling_class(**H.sampling_kwargs(submission)) for submission in submissions
                ]
                effective = [H.effective_params(llm, param) for param in params]
                prefix = f"condition-{profile}-round-{round_index:03d}"
                remember(
                    directory,
                    state,
                    prefix + ".submitted.json",
                    {
                        "status": "SUBMITTED",
                        "at": H.now(),
                        "requests": [
                            {**s, **e} for s, e in zip(submissions, effective, strict=True)
                        ],
                    },
                )
                for submission in submissions:
                    row = next(r for r in members if r["grid_index"] == submission["grid_index"])
                    row["requests_submitted"] += 1
                state["requests_submitted"] += len(submissions)
                H.set_state(directory, state)
                outputs = llm.generate(
                    [{"prompt_token_ids": s["prompt_ids"]} for s in submissions],
                    params,
                    use_tqdm=False,
                )
                raw = [H.raw_output(output) for output in outputs]
                remember(directory, state, prefix + ".raw.json", {"at": H.now(), "outputs": raw})
                state["responses_returned"] += len(raw)
                check_batch(outputs, submissions)
                check_sources()
                audited = []
                for submission, returned, settings in zip(submissions, raw, effective, strict=True):
                    response = returned["outputs"][0]
                    tokens = response["token_ids"]
                    H.ensure(
                        all(type(t) is int and t >= 0 for t in tokens), "Malformed response tokens"
                    )
                    response = {
                        **response,
                        "raw_text": tokenizer.decode(tokens, skip_special_tokens=False),
                        "action_text": tokenizer.decode(tokens, skip_special_tokens=True),
                    }
                    complete = H.response_complete(tokens, response["raw_finish_reason"], H.EOS_IDS)
                    audited.append(
                        {
                            "request_index": submission["request_index"],
                            "grid_index": submission["grid_index"],
                            "response": response,
                            "response_sha256": H.token_sha(tokens),
                            "response_complete": complete,
                            "effective_sampling": settings,
                        }
                    )
                remember(
                    directory, state, prefix + ".audited.json", {"at": H.now(), "rows": audited}
                )
                for submission, observed in zip(submissions, audited, strict=True):
                    row = next(r for r in members if r["grid_index"] == submission["grid_index"])
                    context = contexts[row["grid_index"]]
                    row["responses_returned"] += 1
                    row["response_tokens"] += len(observed["response"]["token_ids"])
                    if not observed["response_complete"]:
                        incomplete(
                            row,
                            "RESPONSE_INCOMPLETE",
                            "Response failed unchanged EOS/4096-cap contract",
                        )
                        continue
                    env = context["env"]
                    before = len(env.credit_step_trace)
                    before_session = env.current_session_index
                    response = observed["response"]
                    obs, _reward, done, _info = await env.step(
                        make_action(response["action_text"], response["token_ids"])
                    )
                    state["actions_applied"] += 1
                    context["env_step"] += 1
                    trace = [asdict(step) for step in env.credit_step_trace]
                    interactions = env.interaction_trace
                    row.update(
                        steps=trace,
                        current_session_index=env.current_session_index,
                        invalid_count=sum(step["invalid_action"] for step in trace),
                    )
                    remember(
                        directory,
                        state,
                        f"request-{submission['request_index']:06d}.applied.json",
                        {
                            "status": "ACTION_APPLIED",
                            "at": H.now(),
                            "request_index": submission["request_index"],
                            "grid_index": row["grid_index"],
                            "env_step": submission["env_step"],
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
                        and interactions[-1]["response"]["sha256"]
                        == hashlib.sha256(response["action_text"].encode()).hexdigest(),
                        "Environment action/capture alignment failed",
                    )
                    H.ensure(
                        type(done) is bool and done is env.episode_complete,
                        "Environment completion flags differ",
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
                            fills=[asdict(fill) for fill in env.ledger.fills],
                            terminal_metrics=env.screening_terminal_metrics(),
                        )
                    elif context["env_step"] >= MAX_STEPS:
                        incomplete(row, "STEP_CAP", "256 requests exhausted without terminal")
                    H.set_state(directory, state)
                round_index += 1
                H.set_state(directory, state)
                print(
                    json.dumps(
                        {
                            "profile": profile,
                            "round": round_index,
                            "requests": state["requests_submitted"],
                            "complete": sum(r["status"] == "COMPLETE" for r in state["grid"]),
                            "incomplete": sum(r["status"] == "INCOMPLETE" for r in state["grid"]),
                        }
                    ),
                    flush=True,
                )
        H.ensure(
            all(row["status"] in {"COMPLETE", "INCOMPLETE"} for row in state["grid"]),
            "Fixed population still has unattempted trajectories",
        )
        state.update(
            status="OBSERVED",
            finished_at=H.now(),
            complete_count=sum(row["status"] == "COMPLETE" for row in state["grid"]),
            incomplete_count=sum(row["status"] == "INCOMPLETE" for row in state["grid"]),
            trajectory_count=16,
            is_training_gate=False,
            is_screening_pass=False,
        )
        H.set_state(directory, state)
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
        H.set_state(directory, state)
        raise


def run(manifest_path: Path, expected_sha: str, model_key: str) -> int:
    H.ensure(manifest_path.resolve() == HERE / "manifest.json", "Wrong manifest path")
    directory = HERE / model_key
    H.ensure(
        not directory.exists() and not directory.is_symlink(), "Refusing existing model attempt"
    )
    directory.mkdir(parents=True, exist_ok=False)
    state = initial_state(model_key, expected_sha)
    H.set_state(directory, state)

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
        H.ensure(os.environ.get("USE_TORCH", "1") != "0", "Cannot inherit CPU-only prepare process")
        for key in ("VLLM_USE_V1", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            H.ensure(os.environ.get(key, "1") == "1", f"Wrong {key}")
            os.environ[key] = "1"
        H.add_import_paths()
        from agent_r1.env.base import Action
        from transformers import AutoTokenizer

        rows = H.read_train_rows()
        model = manifest["models"][model_key]
        tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
        for profile in PROFILES:
            for index, row in enumerate(rows):
                H.ensure(
                    row["prompt"] == manifest["windows"][index]["raw_prompt"]
                    and json.loads(row["env_kwargs"]) == manifest["windows"][index]["env_kwargs"],
                    "Train row drift",
                )
                env = make_real_env(rows, index, profile)
                obs = env.reset(raw_prompt=copy.deepcopy(row["prompt"]))
                H.ensure(
                    render(tokenizer, obs, env.tool_schemas)
                    == model["initial_prompts"][profile][index],
                    "Live Train initial prompt differs",
                )
        remember(
            directory,
            state,
            "environment_preflight.json",
            {
                "status": "PASSED",
                "at": H.now(),
                "episodes": list(H.EPISODES),
                "profiles": list(PROFILES),
                "new_model_requests": 0,
            },
        )
        from vllm import LLM, SamplingParams

        llm = LLM(**model["engine_kwargs"])
        state.update(status="RUNNING", engine_ready_at=H.now())
        H.set_state(directory, state)

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
                lambda index, profile: make_real_env(rows, index, profile),
                lambda text, ids: Action(text=text, token_ids=ids),
                check_sources,
            )
        )
        return 0
    except BaseException as error:
        state.update(
            status="STOPPED", finished_at=H.now(), error_type=type(error).__name__, error=str(error)
        )
        H.set_state(directory, state)
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
