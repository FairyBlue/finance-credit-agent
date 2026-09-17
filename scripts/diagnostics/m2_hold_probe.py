#!/usr/bin/env python3
"""Frozen, Train-only first-action diagnostic: 24 requests per fresh base model.

prepare/verify do not import torch, Ray or vLLM. run performs no training and no
retry. Complete responses enter the real environment once; incomplete responses
remain negative observations. This is not a trajectory or policy-quality gate.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "runs/m2-hold-diagnostic-20260909/probe-v1"
PLAN_DOC = ROOT / "docs/2026-09-09_M2_HOLD_DIAGNOSTIC_PLAN.md"
PILOT = ROOT / "artifacts/m2_screening/pilot_train.parquet"
PILOT_SHA = "c6ef2977af3fad0b28cb5e38c4f58bb2f42d066b0d4228cbb40b6852a759a4b3"
OLD = ROOT / "runs/m2-screening-20260908-v10/diagnostics/single_tool_v1/trajectories_base"
OLD_RESULT_SHA = "b43a806df5f7c4dd8d1c15f571f80a6ffe922c5c0ef6c9ba847e785a918c9a83"
MECHANISM = ROOT / "runs/m2-hold-diagnostic-20260909/mechanism_cpu_evidence.json"
MECHANISM_SHA = "314fa4b9ed62bb6851414e391e89b3564fb8121ef6e566b6ebfc9bc047524a87"
V11 = ROOT / "runs/m2-screening-20260908-v11"
MODELS = {
    "screening": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "wiring": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}
CONDITIONS = ("A", "B", "C")
EPISODES = ("formal-train-0001", "formal-train-0005")
EOS_IDS = {151643, 151645}
OBJECTIVE = (
    "Your objective is to grow the portfolio's value after commission and slippage over "
    "the episode, while managing drawdown, turnover and concentration. Use the available "
    "point-in-time evidence to decide whether a trade is justified. Holding cash is "
    "allowed; there is no requirement to trade."
)
PACKAGES = ("torch", "vllm", "verl", "transformers", "tokenizers", "pyarrow", "pydantic")
LIMITATIONS = [
    "First actions only: no continuation after tool feedback or full-episode evaluation.",
    "Fresh base weights, not the R5 post-update policy; no actor checkpoint is loaded.",
    "Standalone inference and changed batches do not promise tokenwise R5 replay.",
    "A-to-B changes schema; B-to-C adds an objective under the repaired schema only.",
    "No return selection, residual estimation, training, Dev evaluation, refill or retry.",
    "CPU preparation replays frozen initial observations; run rechecks real PIT resets before GPU.",
]


class DiagnosticStop(RuntimeError):
    """An identity, resource or engine failure; never replaced by another request."""


def ensure(value: bool, message: str) -> None:
    if not value:
        raise DiagnosticStop(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def token_sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    ensure(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    # Key order affects the tokenizer's Jinja tojson rendering of tool schemas.
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=False, allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_state(directory: Path, value: dict[str, Any]) -> None:
    temporary = directory / f".state-{time.time_ns()}.json"
    write_new(temporary, value)
    temporary.replace(directory / "state.json")


def add_import_paths() -> None:
    for path in (ROOT, ROOT / "third_party/agent-r1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_request_plan() -> list[dict[str, Any]]:
    add_import_paths()
    from src.agents.rollout_seed import derive_request_seed

    result: list[dict[str, Any]] = []
    for model_key in MODELS:
        model_index = 0
        for condition in CONDITIONS:
            for window_index, episode_id in enumerate(EPISODES):
                for rollout_id in range(4):
                    identity = {
                        "step": 1,
                        "sample_index": window_index,
                        "rollout_n": rollout_id,
                        "validate": False,
                    }
                    result.append(
                        {
                            "request_index": len(result),
                            "model_request_index": model_index,
                            "model_key": model_key,
                            "condition": condition,
                            "window_index": window_index,
                            "episode_id": episode_id,
                            "rollout_id": rollout_id,
                            "identity": identity,
                            "env_step": 0,
                            "seed": derive_request_seed(42, identity, 0),
                        }
                    )
                    model_index += 1
    return result


def sampling_kwargs(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 4096,
        "ignore_eos": False,
        "seed": plan["seed"],
        "repetition_penalty": 1.0,
        "logprobs": None,
    }


def response_complete(
    token_ids: list[int], finish_reason: Any, effective_eos_ids: set[int] | list[int]
) -> bool:
    return bool(
        isinstance(token_ids, list)
        and 0 < len(token_ids) < 4096
        and all(type(token) is int and token >= 0 for token in token_ids)
        and finish_reason == "stop"
        and token_ids[-1] in effective_eos_ids
    )


def validate_batch_mapping(
    outputs: list[Any], plans: list[dict[str, Any]], prompt_ids_by_window: dict[int, list[int]]
) -> None:
    ensure(len(outputs) == len(plans), "Missing or extra output; no replacement requests")
    for output, plan in zip(outputs, plans, strict=True):
        ensure(output.request_id == str(plan["model_request_index"]), "Request ID/order mismatch")
        ensure(
            list(output.prompt_token_ids) == prompt_ids_by_window[plan["window_index"]],
            "Returned prompt differs from submitted prompt",
        )
        ensure(
            len(output.outputs) == 1 and output.outputs[0].index == 0,
            "Expected one output at sample index zero",
        )


def engine_kwargs(model: Path) -> dict[str, Any]:
    return {
        "model": str(model),
        "tokenizer": str(model),
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.20,
        "disable_custom_all_reduce": True,
        "enable_sleep_mode": False,
        "enforce_eager": False,
        "seed": 42,
        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]},
        "override_generation_config": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_new_tokens": 4096,
        },
    }


def source_paths() -> list[Path]:
    upstream = ROOT / "third_party/agent-r1/agent_r1"
    paths = [
        Path(__file__).resolve(),
        PLAN_DOC,
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((upstream / "env").rglob("*.py")),
        *sorted((upstream / "tool").rglob("*.py")),
        upstream / "agent_flow/agent_flow.py",
        upstream / "agent_flow/agent_env_loop.py",
    ]
    return sorted(set(paths))


def snapshot(model_key: str) -> Path:
    name, revision = MODELS[model_key]
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + name.replace("/", "--"))
        / "snapshots"
        / revision
    )


def package_versions() -> dict[str, str]:
    return {name: version(name) for name in PACKAGES}


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "dirty_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--").encode()).hexdigest(),
        "agent_r1_commit": git("-C", "third_party/agent-r1", "rev-parse", "HEAD"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
    }


def read_train_rows() -> list[dict[str, Any]]:
    ensure(digest(PILOT) == PILOT_SHA, "Pilot Train parquet changed")
    from pyarrow import parquet  # type: ignore[import-untyped]

    rows: list[dict[str, Any]] = parquet.read_table(PILOT).to_pylist()
    ensure(len(rows) == 2, "Exactly two pilot Train windows required")
    for index, row in enumerate(rows):
        extra = row["extra_info"]
        kwargs = json.loads(row["env_kwargs"])
        ensure(
            extra["episode_id"] == EPISODES[index]
            and extra["source_index"] == 4 * index
            and extra["index"] == index
            and extra["source_split"] == "train"
            and extra["screening_role"] == "pilot_train",
            "Wrong Train window identity",
        )
        ensure(
            kwargs["screening"] is True
            and kwargs["input_file_count"] == 9220
            and kwargs["data_start"] == "2023-01-03"
            and kwargs["data_end"] == "2024-10-31"
            and len(kwargs["episode_dates"]) == len(kwargs["episode_open_times"]) == 21,
            "Train-only PIT boundary changed",
        )
        ensure(isinstance(row["prompt"], list) and bool(row["prompt"]), "Missing raw prompt")
    return rows


def r5_prompt_matches(expected: list[dict[str, Any]]) -> dict[str, str]:
    evidence_files: dict[str, str] = {}
    for cell in ("pilot-m1-s42", "pilot-m2-s42"):
        found: set[int] = set()
        for path in sorted((V11 / cell / "artifacts/screening_evidence").glob("*.jsonl")):
            for line in path.read_text().splitlines():
                trajectory = json.loads(line)
                identity = trajectory.get("identity", {})
                if (
                    identity.get("step") != 1
                    or identity.get("validate") is not False
                    or identity.get("rollout_n") != 0
                ):
                    continue
                index = identity["sample_index"]
                ensure(index in {0, 1} and index not in found, "Ambiguous R5 initial evidence")
                request = trajectory["requests"][0]
                ensure(
                    request["prompt_sha256"] == expected[index]["prompt_sha256"]
                    and request["prompt_token_count"] == len(expected[index]["prompt_ids"]),
                    "Frozen initial state differs from actual R5 pilot prompt",
                )
                found.add(index)
                evidence_files[str(path.relative_to(ROOT))] = digest(path)
        ensure(found == {0, 1}, f"Missing both real R5 initial prompts in {cell}")
    return evidence_files


def prepare(manifest_path: Path) -> dict[str, Any]:
    ensure(manifest_path.resolve() == HERE / "manifest.json", "Use the fixed new probe-v1 manifest")
    ensure(not manifest_path.exists(), "Manifest already exists; do not overwrite")
    os.environ.update(
        USE_TORCH="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
    )
    add_import_paths()
    from agent_r1.env import ToolEnv
    from transformers import AutoTokenizer

    from src.agents.screening_prompt import SINGLE_TOOL_V1_RULES, apply_screening_action_protocol
    from src.env import trading as trading_module  # noqa: F401 - register actual tools

    ensure(OBJECTIVE in PLAN_DOC.read_text(), "Objective is not registered verbatim")
    source = source_identity()
    rows = read_train_rows()
    old_result = OLD / "result.json"
    submitted = OLD / "batch-00-round-00000.submitted.json"
    ensure(
        digest(old_result) == OLD_RESULT_SHA and digest(MECHANISM) == MECHANISM_SHA,
        "Old observation provenance changed",
    )
    ensure(
        digest(submitted) == read(old_result)["evidence_files_sha256"][submitted.name],
        "Old two-window submission changed",
    )
    old_requests = read(submitted)["requests"]
    selected = [old_requests[0], old_requests[4]]
    mechanism = read(MECHANISM)["actual_4B_first_prompt"]
    ensure(
        selected[0]["prompt_ids"] == mechanism["prompt_ids"]
        and selected[0]["tools"] == mechanism["tools"],
        "Wrong old R5 schema/prompt",
    )
    new_tools = ToolEnv(tools=list(trading_module.TradingEnv.TOOL_NAMES)).tool_schemas
    ensure(new_tools != selected[0]["tools"], "Conditions A/B must use different schemas")
    windows = []
    for index, (row, previous) in enumerate(zip(rows, selected, strict=True)):
        kwargs = json.loads(row["env_kwargs"])
        state = json.loads(previous["messages"][-1]["content"])
        expected_messages = apply_screening_action_protocol(
            [*copy.deepcopy(row["prompt"]), copy.deepcopy(previous["messages"][-1])],
            "single_tool_v1",
        )
        ensure(
            previous["episode_id"] == EPISODES[index]
            and expected_messages == previous["messages"]
            and state["current_date"] == kwargs["episode_dates"][0]
            and state["cash"] == kwargs["initial_cash"]
            and state["positions"] == {}
            and state["session_index"] == 0,
            "Frozen observation is not the pilot initial state",
        )
        ensure(previous["tools"] == selected[0]["tools"], "Old tool schema differs by window")
        windows.append(
            {
                "episode_id": EPISODES[index],
                "source_index": 4 * index,
                "raw_prompt": row["prompt"],
                "env_kwargs": kwargs,
                "initial_messages": expected_messages,
                "old_prompt_sha256": previous["prompt_sha256"],
            }
        )
    evidence = r5_prompt_matches(selected)
    models: dict[str, Any] = {}
    for model_key, (name, revision) in MODELS.items():
        model = snapshot(model_key)
        metadata_paths = sorted(
            [p for p in model.iterdir() if p.is_file() and p.suffix in {".json", ".txt"}]
        )
        metadata = {p.name: digest(p) for p in metadata_paths}
        weight_files = sorted(model.glob("*.safetensors"))
        ensure(bool(weight_files), "Base model weights are missing")
        # LFS blobs are content-addressed. Record actual bytes too, before any GPU work.
        weight_hashes = {p.name: digest(p) for p in weight_files}
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        prompts: dict[str, Any] = {}
        for condition in CONDITIONS:
            prompts[condition] = []
            for window in windows:
                messages = copy.deepcopy(window["initial_messages"])
                if condition == "C":
                    system = next(message for message in messages if message["role"] == "system")
                    ensure(
                        system["content"].endswith(SINGLE_TOOL_V1_RULES), "Missing common protocol"
                    )
                    system["content"] += "\n\n" + OBJECTIVE
                tools = copy.deepcopy(selected[0]["tools"] if condition == "A" else new_tools)
                ids = list(
                    tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                        enable_thinking=False,
                    )
                )
                ensure(0 < len(ids) <= 4096, "Prompt exceeds frozen budget")
                if condition == "A":
                    ensure(token_sha(ids) == window["old_prompt_sha256"], "Old prompt token drift")
                prompts[condition].append(
                    {
                        "messages": messages,
                        "tools": tools,
                        "template_kwargs": {"enable_thinking": False},
                        "prompt_ids": ids,
                        "prompt_sha256": token_sha(ids),
                        "prompt_token_count": len(ids),
                    }
                )
        models[model_key] = {
            "name": name,
            "revision": revision,
            "snapshot": str(model),
            "metadata_sha256": metadata,
            "weight_sha256": weight_hashes,
            "engine_kwargs": engine_kwargs(model),
            "prompts": prompts,
        }
    manifest = {
        "status": "PREPARED",
        "schema_version": 1,
        "created_at": now(),
        "diagnostic_id": "m2_hold_first_step_v1",
        "request_limit": 48,
        "source_identity": source,
        "package_versions": package_versions(),
        "plan_document": str(PLAN_DOC.relative_to(ROOT)),
        "objective": OBJECTIVE,
        "objective_sha256": hashlib.sha256(OBJECTIVE.encode()).hexdigest(),
        "source_file_sha256": {
            str(PILOT.relative_to(ROOT)): PILOT_SHA,
            str(old_result.relative_to(ROOT)): OLD_RESULT_SHA,
            str(submitted.relative_to(ROOT)): digest(submitted),
            str(MECHANISM.relative_to(ROOT)): MECHANISM_SHA,
            **evidence,
        },
        "windows": windows,
        "models": models,
        "request_plan": build_request_plan(),
        "limitations": LIMITATIONS,
    }
    ensure(
        source_identity()["source_sha256"] == source["source_sha256"],
        "Source changed during prepare",
    )
    ensure(
        not any(name in sys.modules for name in ("torch", "ray", "vllm")),
        "CPU preparation imported a GPU framework",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_new(manifest_path, manifest)
    # Re-read serialized key order and re-tokenize, guarding the previous sorted-JSON failure.
    saved = read(manifest_path)
    for model in saved["models"].values():
        tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
        verify_tokenized_prompts(tokenizer, model)
    validate_manifest(saved)
    write_new(
        HERE / "preparation.json",
        {
            "status": "PASSED",
            "at": now(),
            "manifest_sha256": digest(manifest_path),
            "script_sha256": digest(Path(__file__)),
            "request_count": 48,
            "model_requests": {k: 24 for k in MODELS},
            "gpu_frameworks_imported": False,
            "new_model_requests": 0,
            "prompt_counts": {
                k: {c: [p["prompt_token_count"] for p in m["prompts"][c]] for c in CONDITIONS}
                for k, m in models.items()
            },
        },
    )
    return saved


def verify_tokenized_prompts(tokenizer: Any, model: dict[str, Any]) -> None:
    for prompts in model["prompts"].values():
        for prompt in prompts:
            ids = list(
                tokenizer.apply_chat_template(
                    prompt["messages"],
                    tools=prompt["tools"],
                    add_generation_prompt=True,
                    tokenize=True,
                    **prompt["template_kwargs"],
                )
            )
            ensure(ids == prompt["prompt_ids"], "Tokenizer differs from frozen serialized prompt")


def validate_manifest(manifest: dict[str, Any]) -> None:
    ensure(
        manifest.get("status") == "PREPARED"
        and manifest.get("diagnostic_id") == "m2_hold_first_step_v1"
        and manifest.get("schema_version") == 1
        and manifest.get("request_limit") == 48,
        "Wrong diagnostic manifest",
    )
    ensure(manifest.get("request_plan") == build_request_plan(), "Request grid/seed drift")
    ensure(
        manifest.get("objective") == OBJECTIVE
        and manifest.get("objective_sha256") == hashlib.sha256(OBJECTIVE.encode()).hexdigest(),
        "Objective changed",
    )
    ensure(list(manifest["models"]) == list(MODELS), "Unexpected base models/order")
    for model_key, model in manifest["models"].items():
        ensure(
            (model["name"], model["revision"]) == MODELS[model_key]
            and model["snapshot"] == str(snapshot(model_key))
            and model["engine_kwargs"] == engine_kwargs(snapshot(model_key)),
            "Model/engine drift",
        )
        ensure(list(model["prompts"]) == list(CONDITIONS), "Condition grid drift")
        for condition, prompts in model["prompts"].items():
            ensure(len(prompts) == 2, "Each condition requires two Train windows")
            for index, prompt in enumerate(prompts):
                ids = prompt["prompt_ids"]
                ensure(
                    prompt["template_kwargs"] == {"enable_thinking": False}
                    and isinstance(ids, list)
                    and 0 < len(ids) <= 4096
                    and all(type(token) is int and token >= 0 for token in ids)
                    and prompt["prompt_token_count"] == len(ids)
                    and prompt["prompt_sha256"] == token_sha(ids),
                    "Prompt contract drift",
                )
                baseline = model["prompts"]["A"][index]
                repaired = model["prompts"]["B"][index]
                if condition == "B":
                    ensure(
                        prompt["messages"] == baseline["messages"]
                        and prompt["tools"] != baseline["tools"],
                        "A/B contrast changed",
                    )
                if condition == "C":
                    expected = copy.deepcopy(repaired["messages"])
                    next(m for m in expected if m["role"] == "system")["content"] += (
                        "\n\n" + OBJECTIVE
                    )
                    ensure(
                        prompt["messages"] == expected and prompt["tools"] == repaired["tools"],
                        "B/C contrast changed",
                    )


def verify_files(
    manifest: dict[str, Any], *, model_key: str | None = None, verify_weights: bool = False
) -> None:
    validate_manifest(manifest)
    expected_sources = manifest["source_identity"]["source_sha256"]
    ensure(
        set(expected_sources) == {str(p.relative_to(ROOT)) for p in source_paths()},
        "Runtime source inventory drift",
    )
    for name, expected in {**expected_sources, **manifest["source_file_sha256"]}.items():
        ensure(digest(ROOT / name) == expected, f"Frozen source/input drift: {name}")
    ensure(package_versions() == manifest["package_versions"], "Package version drift")
    for key in [model_key] if model_key else MODELS:
        model = manifest["models"][key]
        inventory = dict(model["metadata_sha256"])
        if verify_weights:
            inventory.update(model["weight_sha256"])
        for name, expected in inventory.items():
            ensure(
                digest(Path(model["snapshot"]) / name) == expected,
                f"Model file drift: {key}/{name}",
            )


def raw_output(output: Any) -> dict[str, Any]:
    responses = []
    for response in output.outputs:
        tokens = list(response.token_ids)
        responses.append(
            {
                "index": response.index,
                "token_ids": tokens,
                "raw_finish_reason": response.finish_reason,
                "raw_stop_reason": response.stop_reason,
                "vllm_text": response.text,
            }
        )
    return {
        "request_id": output.request_id,
        "prompt_token_ids": list(output.prompt_token_ids),
        "outputs": responses,
    }


def effective_params(llm: Any, sampling: Any) -> dict[str, Any]:
    processor = llm.llm_engine.processor
    eos = processor.input_preprocessor.get_eos_token_id(None)
    effective = sampling.clone()
    effective.update_from_generation_config(processor.generation_config_fields, eos)
    effective.update_from_tokenizer(processor.tokenizer.get_lora_tokenizer(None))
    ensure(
        eos == 151645
        and set(effective.all_stop_token_ids) == EOS_IDS
        and effective.ignore_eos is False,
        "Effective EOS contract drift",
    )
    fields = {key: getattr(effective, key) for key in sampling_kwargs({"seed": 42})}
    ensure(fields == sampling_kwargs({"seed": sampling.seed}), "Effective sampling drift")
    return {
        "sampling_params": fields,
        "effective_stop_token_ids": sorted(effective.all_stop_token_ids),
        "tokenizer_eos_token_id": eos,
        "effective_sampling_repr": repr(effective),
        "generation_config_fields": dict(processor.generation_config_fields),
    }


async def execute_batches(
    llm: Any,
    sampling_class: Any,
    manifest: dict[str, Any],
    model_key: str,
    directory: Path,
    make_env: Any,
    make_action: Any,
    check_sources: Any,
) -> list[dict[str, Any]]:
    """Three fixed batches; negative generations continue, infrastructure errors propagate."""
    tokenizer = llm.get_tokenizer()
    model = manifest["models"][model_key]
    verify_tokenized_prompts(tokenizer, model)
    result: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        check_sources()
        plans = [
            p
            for p in manifest["request_plan"]
            if p["model_key"] == model_key and p["condition"] == condition
        ]
        prompt_ids = {i: p["prompt_ids"] for i, p in enumerate(model["prompts"][condition])}
        params = [sampling_class(**sampling_kwargs(plan)) for plan in plans]
        effective = [effective_params(llm, param) for param in params]
        submitted = [
            {**plan, "prompt_ids": prompt_ids[plan["window_index"]], **record}
            for plan, record in zip(plans, effective, strict=True)
        ]
        write_new(
            directory / f"condition-{condition}.submitted.json",
            {"status": "SUBMITTED", "at": now(), "requests": submitted},
        )
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids[p["window_index"]]} for p in plans],
            params,
            use_tqdm=False,
        )
        # Save every returned response before any mapping or environment validation.
        raw = [raw_output(output) for output in outputs]
        write_new(directory / f"condition-{condition}.raw.json", {"at": now(), "outputs": raw})
        validate_batch_mapping(outputs, plans, prompt_ids)
        check_sources()
        rows = []
        for plan, output, response_record, params_record in zip(
            plans, outputs, raw, effective, strict=True
        ):
            response = response_record["outputs"][0]
            tokens = response["token_ids"]
            ensure(
                all(type(token) is int and token >= 0 for token in tokens),
                "Malformed output tokens",
            )
            # Raw tokens for the whole batch are durable even if decoding fails.
            response["raw_text"] = tokenizer.decode(tokens, skip_special_tokens=False)
            response["action_text"] = tokenizer.decode(tokens, skip_special_tokens=True)
            complete = response_complete(tokens, response["raw_finish_reason"], EOS_IDS)
            row = {
                **plan,
                **params_record,
                "vllm_request_id": output.request_id,
                "prompt_ids": prompt_ids[plan["window_index"]],
                "prompt_sha256": token_sha(prompt_ids[plan["window_index"]]),
                "response": response,
                "response_sha256": token_sha(tokens),
                "response_token_count": len(tokens),
                "complete": complete,
                "environment_applied": False,
                "environment_interaction_trace": [],
            }
            if complete:
                env = make_env(plan["window_index"])
                observation = env.reset(
                    raw_prompt=copy.deepcopy(
                        manifest["windows"][plan["window_index"]]["raw_prompt"]
                    )
                )
                from src.agents.screening_prompt import apply_screening_action_protocol

                ensure(
                    apply_screening_action_protocol(observation.messages, "single_tool_v1")
                    == manifest["windows"][plan["window_index"]]["initial_messages"],
                    "First-step environment state drift",
                )
                await env.step(make_action(response["action_text"], tokens))
                trace = env.interaction_trace
                ensure(
                    len(trace) == 1
                    and trace[0]["response"]["sha256"]
                    == hashlib.sha256(response["action_text"].encode()).hexdigest(),
                    "Environment did not capture exactly the generated response",
                )
                ensure(
                    len(env.credit_step_trace) == 1
                    and env.current_session_index in {0, 1}
                    and not env.episode_complete,
                    "First action exceeded one-step scope",
                )
                row.update(
                    environment_applied=True,
                    environment_interaction_trace=trace,
                    action=env.credit_step_trace[0].action,
                    invalid_action=env.credit_step_trace[0].invalid_action,
                    step_trace=asdict(env.credit_step_trace[0]),
                    fills=[asdict(fill) for fill in env.ledger.fills],
                )
            else:
                row["not_applied_reason"] = "RESPONSE_INCOMPLETE_UNDER_FROZEN_CAP"
            write_new(directory / f"request-{plan['model_request_index']:02d}.json", row)
            rows.append(row)
            result.append(row)
        write_new(
            directory / f"condition-{condition}.observed.json",
            {"status": "OBSERVED", "at": now(), "requests": rows},
        )
        print(
            json.dumps(
                {
                    "condition": condition,
                    "observed": len(rows),
                    "complete": sum(row["complete"] for row in rows),
                }
            ),
            flush=True,
        )
    return result


def run(manifest_path: Path, expected_sha: str, model_key: str) -> int:
    ensure(manifest_path.resolve() == HERE / "manifest.json", "Wrong diagnostic manifest path")
    directory = HERE / model_key
    ensure(not directory.exists() and not directory.is_symlink(), "Refusing existing model attempt")
    directory.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "STARTING",
        "model_key": model_key,
        "at": now(),
        "manifest_sha256": expected_sha,
        "pid": os.getpid(),
    }
    set_state(directory, state)
    try:
        ensure(digest(manifest_path) == expected_sha, "Manifest SHA mismatch")
        manifest = read(manifest_path)
        verify_files(manifest, model_key=model_key, verify_weights=True)
        ensure(
            version("vllm") == "0.9.2" and version("torch") == "2.7.0+cu128",
            "Unexpected pinned inference runtime",
        )
        ensure(
            not any(name.startswith("FINANCE_") for name in os.environ), "Training hooks inherited"
        )
        for key in ("VLLM_USE_V1", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            ensure(os.environ.get(key, "1") == "1", f"Wrong {key}")
            os.environ[key] = "1"
        ensure(
            os.environ.get("USE_TORCH", "1") != "0",
            "GPU run cannot inherit CPU prepare USE_TORCH=0",
        )
        add_import_paths()
        from agent_r1.env.base import Action

        from src.agents.screening_prompt import apply_screening_action_protocol
        from src.env.trading import TradingEnv

        rows = read_train_rows()

        def make_env(index: int) -> Any:
            kwargs = {
                **json.loads(rows[index]["env_kwargs"]),
                "reward_mode": "outcome_only",
                "capture_interactions": True,
            }
            return TradingEnv(**kwargs)

        # Actual immutable Train data and resets are verified before importing vLLM.
        for index, row in enumerate(rows):
            ensure(
                row["prompt"] == manifest["windows"][index]["raw_prompt"]
                and json.loads(row["env_kwargs"]) == manifest["windows"][index]["env_kwargs"],
                "Prepared window metadata drift",
            )
            env = make_env(index)
            obs = env.reset(raw_prompt=copy.deepcopy(row["prompt"]))
            ensure(
                apply_screening_action_protocol(obs.messages, "single_tool_v1")
                == manifest["windows"][index]["initial_messages"],
                "Actual PIT initial state differs",
            )
            ensure(
                env.tool_schemas == manifest["models"][model_key]["prompts"]["B"][index]["tools"],
                "Current environment tool schemas differ",
            )
        write_new(
            directory / "environment_preflight.json",
            {
                "status": "PASSED",
                "at": now(),
                "episodes": list(EPISODES),
                "source_split": "train",
                "gpu_imported": False,
            },
        )
        from vllm import LLM, SamplingParams

        llm = LLM(**manifest["models"][model_key]["engine_kwargs"])
        state.update(status="RUNNING", engine_ready_at=now())
        set_state(directory, state)

        def check_sources() -> None:
            ensure(digest(manifest_path) == expected_sha, "Manifest changed during run")
            verify_files(manifest, model_key=model_key)

        results = asyncio.run(
            execute_batches(
                llm,
                SamplingParams,
                manifest,
                model_key,
                directory,
                make_env,
                lambda text, ids: Action(text=text, token_ids=ids),
                check_sources,
            )
        )
        ensure(len(results) == 24, "Model did not observe exactly 24 requests")
        result = {
            "status": "OBSERVED",
            "model_key": model_key,
            "at": now(),
            "manifest_sha256": expected_sha,
            "source_identity": manifest["source_identity"],
            "request_count": len(results),
            "requests": results,
            "limitations": LIMITATIONS,
            "is_trajectory_gate": False,
            "is_training_gate": False,
        }
        write_new(directory / "result.json", result)
        state.update(
            status="OBSERVED", finished_at=now(), result_sha256=digest(directory / "result.json")
        )
        set_state(directory, state)
        return 0
    except BaseException as error:
        state.update(
            status="STOPPED",
            finished_at=now(),
            error_type=type(error).__name__,
            error=str(error),
            observed_request_files=len(list(directory.glob("request-*.json"))),
            submitted_batches=len(list(directory.glob("condition-*.submitted.json"))),
            returned_batches=len(list(directory.glob("condition-*.raw.json"))),
        )
        set_state(directory, state)
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
            action.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest)
        print(json.dumps({"status": "PREPARED", "manifest_sha256": digest(args.manifest)}))
        return 0
    if args.command == "verify":
        ensure(digest(args.manifest) == args.expected_manifest_sha256, "Manifest SHA mismatch")
        verify_files(read(args.manifest))
        print(json.dumps({"status": "VERIFIED", "new_model_requests": 0}))
        return 0
    return run(args.manifest, args.expected_manifest_sha256, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
