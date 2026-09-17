#!/usr/bin/env python3
"""Evaluate one frozen actor with two fixed non-thinking sampling profiles."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "single_tool_portfolio_v1"
PROMPT_CAP, RESPONSE_CAP, REQUEST_CAP = 8192, 4096, 128
TRAIN_IDS = tuple(f"formal-train-{i:04d}" for i in range(1, 30, 4))
EVAL_IDS = tuple(f"formal-train-{i:04d}" for i in range(113, 126, 4))
PROFILES = {
    "current": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0},
    "qwen_recommended": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=json_default) + "\n")
    temporary.replace(path)


def validate_contexts(contexts):
    full = contexts.get("full", [])
    if (
        contexts.get("teacher_labels_included") is not False
        or contexts.get("short") != []
        or [context.get("id") for context in full] != list(TRAIN_IDS + EVAL_IDS)
    ):
        raise ValueError("sampling comparison requires the original twelve Train contexts")
    for index, context in enumerate(full):
        kwargs = context.get("env_kwargs", {})
        if (
            context.get("prefix") != []
            or context.get("expected_start_session_index") != 0
            or context.get("state_kind") != "cash"
            or context.get("group") != ("training_windows" if index < 8 else "evaluation_windows")
            or len(kwargs.get("episode_dates", [])) != 21
            or kwargs.get("observation_profile") != "history_features_v1"
            or kwargs.get("reward_mode") != "finance_trajectory"
        ):
            raise ValueError("sampling contexts changed their full cash-state window contract")


def grid(contexts):
    validate_contexts(contexts)
    rows = []
    for index, context in enumerate(contexts["full"]):
        for sibling in range(4):
            for profile in PROFILES:
                rows.append(
                    {
                        "id": f"full-{context['id']}-{profile}-sample{sibling}",
                        "context_id": context["id"],
                        "context": context,
                        "horizon": "full",
                        "state_kind": "cash",
                        "sampling": "sample",
                        "profile": profile,
                        "identity": {
                            "step": 1,
                            "sample_index": index,
                            "rollout_n": sibling,
                            "validate": False,
                        },
                        "request_limit": REQUEST_CAP,
                        "status": "UNSTARTED",
                        "requests": 0,
                        "response_tokens": 0,
                        "actions": {},
                        "invalid": 0,
                        "model_fills": [],
                    }
                )
    return rows


def sampling_parameters(profile, identity, request_index):
    from src.agents.rollout_seed import derive_request_seed

    if profile not in PROFILES:
        raise ValueError("unregistered sampling profile")
    return {
        **PROFILES[profile],
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": RESPONSE_CAP,
        "seed": derive_request_seed(42, identity, request_index),
        "n": 1,
    }


def _native_json(value):
    if isinstance(value, Enum):
        return {
            "enum": type(value).__name__,
            "name": value.name,
            "value": _native_json(value.value),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _native_json(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_native_json(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_native_json(item) for item in value]
    raise RuntimeError(f"unexpected native sampling parameter object: {type(value).__name__}")


def native_parameters(sampling_class, requested):
    native = sampling_class(**requested)
    if any(getattr(native, key, object()) != value for key, value in requested.items()):
        raise RuntimeError("native SamplingParams altered a fixed sampling profile")
    fields = getattr(native, "__struct_fields__", tuple(requested))
    evidence = {key: _native_json(getattr(native, key)) for key in fields}
    return native, evidence, repr(native)


def finish_trajectory(row, env, visible, info):
    """Only an actual complete legal terminal transition gets a terminal outcome."""
    metrics = env.screening_terminal_metrics()
    context = row["context"]
    if (
        env.episode_complete is not True
        or env.current_session_index != 20
        or visible.get("session_index") != 20
        or info.get("invalid_action")
        or metrics.get("session_count") != 21
        or metrics.get("calibration_sha256")
        != context["env_kwargs"]["screening_calibration_sha256"]
    ):
        raise RuntimeError("terminal state differs from the fixed complete legal window")
    net_return = info["equity"] / row["takeover_equity"] - 1.0
    if not math.isfinite(net_return) or not math.isfinite(metrics["terminal_score"]):
        raise RuntimeError("complete trajectory has non-finite outcomes")
    row.update(
        status="COMPLETE",
        finished_at_SGT=now(),
        final_state=visible,
        terminal_metrics=metrics,
        takeover_net_return=net_return,
    )


def summary(rows):
    return {
        profile: {
            "planned": 48,
            "statuses": dict(Counter(row["status"] for row in rows if row["profile"] == profile)),
            "failures": dict(
                Counter(
                    row["failure"]
                    for row in rows
                    if row["profile"] == profile and row.get("failure")
                )
            ),
            "requests": sum(row["requests"] for row in rows if row["profile"] == profile),
            "invalid": sum(row["invalid"] for row in rows if row["profile"] == profile),
        }
        for profile in PROFILES
    }


async def evaluate(llm, tokenizer, sampling_class, contexts, output):
    from agent_r1.env.base import Action

    from src.agents.coldstart_data import parse_state, render_prompt
    from src.env.trading import TradingEnv

    rows = grid(contexts)
    state = {
        "status": "RUNNING",
        "started_at_SGT": now(),
        "grid": rows,
        "action_protocol": PROTOCOL,
        "enable_thinking": False,
        "sampling_profiles": PROFILES,
        "planned_trajectories": 96,
        "grid_order": "context_then_sibling_then_profile",
        "automatic_retry": False,
    }
    save(output / "state.json", state)
    try:
        with (output / "requests.jsonl").open("x") as evidence:
            for start in range(0, len(rows), 32):
                members, runtime = rows[start : start + 32], {}
                for row in members:
                    context = row["context"]
                    env = TradingEnv(**context["env_kwargs"])
                    obs = env.reset(raw_prompt=context["raw_prompt"])
                    visible = parse_state(obs)
                    if (
                        env.current_session_index != 0
                        or visible["session_index"] != 0
                        or visible["positions"] != {}
                        or env.ledger.fills
                        or visible["cash"] != context["env_kwargs"]["initial_cash"]
                        or visible["portfolio_equity"] != context["env_kwargs"]["initial_cash"]
                    ):
                        raise RuntimeError("sampling trajectory did not start from original cash")
                    row.update(
                        status="ACTIVE",
                        prefix_info=[],
                        prefix_fill_count=0,
                        takeover_equity=visible["portfolio_equity"],
                        takeover_state=visible,
                        max_model_position_weight=0,
                    )
                    runtime[row["id"]] = [env, obs]
                while any(row["status"] == "ACTIVE" for row in members):
                    submissions = []
                    for row in members:
                        if row["status"] != "ACTIVE":
                            continue
                        if row["requests"] >= REQUEST_CAP:
                            row.update(status="INCOMPLETE", failure="REQUEST_CAP")
                            continue
                        env, obs = runtime[row["id"]]
                        prompt = render_prompt(tokenizer, env, obs, action_protocol=PROTOCOL)
                        ids = tokenizer.encode(prompt, add_special_tokens=False)
                        row["max_prompt_tokens"] = max(row.get("max_prompt_tokens", 0), len(ids))
                        if len(ids) > PROMPT_CAP:
                            row.update(
                                status="INCOMPLETE",
                                failure="PROMPT_CAP",
                                blocked_prompt={
                                    "prompt": prompt,
                                    "prompt_token_ids": ids,
                                    "request_index": row["requests"],
                                },
                            )
                            continue
                        requested = sampling_parameters(
                            row["profile"], row["identity"], row["requests"]
                        )
                        native, native_evidence, native_repr = native_parameters(
                            sampling_class, requested
                        )
                        submissions.append(
                            (row, ids, prompt, requested, native, native_evidence, native_repr)
                        )
                    save(output / "state.json", state)
                    if not submissions:
                        break
                    outputs = llm.generate(
                        [{"prompt_token_ids": item[1]} for item in submissions],
                        [item[4] for item in submissions],
                        use_tqdm=False,
                    )
                    if len(outputs) != len(submissions):
                        raise RuntimeError("generation response count mismatch")
                    for submission, returned in zip(submissions, outputs, strict=True):
                        row, ids, prompt, params, _, native_evidence, native_repr = submission
                        if list(returned.prompt_token_ids) != ids or len(returned.outputs) != 1:
                            raise RuntimeError("generation response mapping mismatch")
                        response = returned.outputs[0]
                        tokens = list(response.token_ids)
                        text = tokenizer.decode(tokens, skip_special_tokens=True)
                        env, _ = runtime[row["id"]]
                        record = {
                            "at_SGT": now(),
                            "trajectory_id": row["id"],
                            "profile": row["profile"],
                            "request_index": row["requests"],
                            "prompt": prompt,
                            "prompt_token_ids": ids,
                            "sampling_params": params,
                            "native_sampling_params": native_evidence,
                            "native_sampling_params_stage": "constructed_before_engine_submission",
                            "native_sampling_params_repr": native_repr,
                            "response": text,
                            "response_token_ids": tokens,
                            "finish_reason": response.finish_reason,
                        }
                        row["requests"] += 1
                        row["response_tokens"] += len(tokens)
                        row["max_response_tokens"] = max(
                            row.get("max_response_tokens", 0), len(tokens)
                        )
                        if (
                            not tokens
                            or response.finish_reason != "stop"
                            or len(tokens) >= RESPONSE_CAP
                        ):
                            row.update(status="INCOMPLETE", failure="RESPONSE_INCOMPLETE")
                        else:
                            obs, reward, done, info = await env.step(
                                Action(text=text, token_ids=tokens)
                            )
                            runtime[row["id"]][1] = obs
                            trace = asdict(env.credit_step_trace[-1])
                            row["actions"][trace["action"]] = (
                                row["actions"].get(trace["action"], 0) + 1
                            )
                            row["invalid"] += int(trace["invalid_action"])
                            row["model_fills"] = [asdict(fill) for fill in env.ledger.fills]
                            visible = parse_state(obs)
                            row["max_model_position_weight"] = max(
                                row["max_model_position_weight"],
                                max(
                                    (p["weight"] for p in visible["positions"].values()), default=0
                                ),
                            )
                            record.update(info=info, trace=trace, reward=reward)
                            if done:
                                finish_trajectory(row, env, visible, info)
                        evidence.write(
                            json.dumps(record, allow_nan=False, default=json_default) + "\n"
                        )
                        evidence.flush()
                    save(output / "state.json", state)
                    print(
                        json.dumps(
                            {
                                "at_SGT": now(),
                                "requests": sum(row["requests"] for row in rows),
                                "statuses": dict(Counter(row["status"] for row in rows)),
                            }
                        ),
                        flush=True,
                    )
                for row in members:
                    env, _ = runtime[row["id"]]
                    row["steps_including_prefix"] = [asdict(step) for step in env.credit_step_trace]
                    row["interaction_trace_including_prefix"] = env.interaction_trace
        state.update(status="OBSERVED", finished_at_SGT=now(), summary=summary(rows))
        save(output / "result.json", state)
        return state
    except BaseException as error:
        state.update(
            status="FAILED", error=f"{type(error).__name__}: {error}", finished_at_SGT=now()
        )
        raise
    finally:
        save(output / "state.json", state)


def engine_settings(model):
    """The same standalone engine resources used by the prior cross-model check."""
    return dict(
        model=str(model),
        tokenizer=str(model),
        dtype="bfloat16",
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=12288,
        max_num_seqs=32,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.50,
        disable_custom_all_reduce=True,
        enable_sleep_mode=False,
        enforce_eager=False,
        seed=42,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
        override_generation_config={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_new_tokens": 4096,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
    contexts = json.loads(args.contexts.read_text())
    validate_contexts(contexts)
    args.output.mkdir(parents=True, exist_ok=False)
    from src.agents.launcher import _resolve_c_compiler

    compiler, compiler_version = _resolve_c_compiler(parent=dict(os.environ))
    os.environ.update(
        CC=compiler,
        VLLM_USE_V1="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = engine_settings(model)
    save(
        args.output / "config.json",
        {
            "engine": engine,
            "started_at_SGT": now(),
            "contexts_sha256": hashlib.sha256(args.contexts.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "action_protocol": PROTOCOL,
            "enable_thinking": False,
            "sampling_profiles": PROFILES,
            "expected_full": 12,
            "expected_short": 0,
            "trajectories": 96,
            "request_limit": REQUEST_CAP,
            "prompt_limit": PROMPT_CAP,
            "response_limit": RESPONSE_CAP,
            "compiler": {"CC": compiler, "version": compiler_version},
        },
    )
    llm = LLM(**engine)
    asyncio.run(evaluate(llm, tokenizer, SamplingParams, contexts, args.output))


if __name__ == "__main__":
    main()
