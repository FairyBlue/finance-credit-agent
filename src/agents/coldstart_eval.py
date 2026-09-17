"""Autonomous before/after cold-start evaluation on reserved Train contexts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agent_r1.env.base import Action

from src.agents.coldstart_data import parse_state, render_prompt
from src.agents.launcher import _resolve_c_compiler
from src.agents.rollout_seed import derive_request_seed
from src.agents.screening_prompt import validate_action_protocol
from src.env.trading import TradingEnv

PROMPT_CAP, RESPONSE_CAP = 8192, 4096


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=json_default))
    temporary.replace(path)


def prefix_expectations(context):
    prefix = context.get("prefix", [])
    if not isinstance(prefix, list) or any(
        not isinstance(text, str) or not text for text in prefix
    ):
        raise ValueError("evaluation prefix actions must be nonempty strings")
    expected = context.get("prefix_expected_invalid", [False] * len(prefix))
    if (
        not isinstance(expected, list)
        or len(expected) != len(prefix)
        or any(type(value) is not bool for value in expected)
    ):
        raise ValueError("evaluation prefix requires one boolean expectation per action")
    if "prefix_expected_errors" in context:
        errors = context["prefix_expected_errors"]
        if not isinstance(errors, list) or len(errors) != len(prefix) or any(
            (not isinstance(error, str) or not error) if invalid else error is not None
            for invalid, error in zip(expected, errors, strict=True)
        ):
            raise ValueError("evaluation prefix error expectations must match invalid actions")
    if "expected_start_session_index" in context:
        index = context["expected_start_session_index"]
        if type(index) is not int or index < 0:
            raise ValueError("evaluation start session index must be a nonnegative integer")
    return expected


def validate_contexts(contexts, expected_short=16, expected_full=4):
    if contexts.get("teacher_labels_included") is not False:
        raise ValueError("evaluation contexts must exclude teacher labels")
    for horizon, count in (("short", expected_short), ("full", expected_full)):
        population = contexts.get(horizon)
        if (
            type(count) is not int
            or count < 0
            or not isinstance(population, list)
            or len(population) != count
        ):
            raise ValueError(f"unexpected reserved {horizon} evaluation population")
        ids = set()
        for context in population:
            identity = context.get("id")
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError(f"invalid or duplicate {horizon} evaluation context id")
            ids.add(identity)
            prefix_expectations(context)


def grid(contexts):
    result: list[dict[str, Any]] = []
    for horizon in ("short", "full"):
        for context in contexts[horizon]:
            context_index = len(result) // 5
            for sibling in range(5):
                lane = "greedy" if sibling == 0 else "sample" + str(sibling - 1)
                result.append(
                    {
                        "id": f"{horizon}-{context['id']}-{lane}",
                        "horizon": horizon,
                        "context_id": context["id"],
                        "state_kind": context.get("state_kind", "cash"),
                        "sampling": "greedy" if sibling == 0 else "sample",
                        "identity": {
                            "step": 1,
                            "sample_index": context_index,
                            "rollout_n": max(0, sibling - 1),
                            "validate": sibling == 0,
                        },
                        "request_limit": 32 if horizon == "short" else 128,
                        "status": "UNSTARTED",
                        "requests": 0,
                        "response_tokens": 0,
                        "actions": {},
                        "invalid": 0,
                        "model_fills": [],
                        "context": context,
                    }
                )
    return result


def summary(rows):
    result = {}
    for horizon in ("short", "full"):
        for sampling in ("greedy", "sample"):
            selected = [r for r in rows if r["horizon"] == horizon and r["sampling"] == sampling]
            complete = [r for r in selected if r["status"] == "COMPLETE"]
            actions: Counter[str] = Counter()
            fills = []
            for row in selected:
                actions.update(row["actions"])
                fills.extend(row["model_fills"])

            def mean(values):
                return sum(values) / len(values) if values else None

            outcome_groups = defaultdict(list)
            for row in complete:
                outcome_groups[row["context_id"]].append(row["takeover_net_return"])
            result[horizon + "_" + sampling] = {
                "planned": len(selected),
                "complete": len(complete),
                "statuses": dict(Counter(r["status"] for r in selected)),
                "requests": sum(r["requests"] for r in selected),
                "actions": dict(actions),
                "invalid": sum(r["invalid"] for r in selected),
                "trajectories_with_model_fills": sum(bool(r["model_fills"]) for r in selected),
                "fill_sides": dict(Counter(f["side"] for f in fills)),
                "fill_symbols": dict(Counter(f["symbol"] for f in fills)),
                "model_transaction_cost": sum(f["commission"] + f["slippage_cost"] for f in fills),
                "mean_takeover_net_return_complete_only": mean(
                    [r["takeover_net_return"] for r in complete]
                ),
                "min_takeover_net_return": min(
                    (r["takeover_net_return"] for r in complete), default=None
                ),
                "max_takeover_net_return": max(
                    (r["takeover_net_return"] for r in complete), default=None
                ),
                "mean_terminal_score_complete_only": mean(
                    [r["terminal_metrics"]["terminal_score"] for r in complete]
                ),
                "max_position_weight": max(
                    (r.get("max_model_position_weight", 0) for r in selected), default=0
                ),
                "max_drawdown_complete_only": max(
                    (r["terminal_metrics"]["max_drawdown"] for r in complete), default=None
                ),
                "sibling_return_range": {key: max(v) - min(v) for key, v in outcome_groups.items()},
                "contexts_with_distinct_completed_outcomes": sum(
                    max(v) - min(v) > 1e-10 for v in outcome_groups.values()
                ),
            }
    return result


async def evaluate(
    llm, tokenizer, sampling_class, contexts, output, *, action_protocol="single_tool_v1"
):
    action_protocol = validate_action_protocol(action_protocol)
    rows = grid(contexts)
    state = {
        "status": "RUNNING", "started_at_SGT": now(), "grid": rows,
        "action_protocol": action_protocol,
    }
    save(output / "state.json", state)
    with (output / "requests.jsonl").open("x") as evidence:
        for start in range(0, len(rows), 32):
            members = rows[start : start + 32]
            runtime = {}
            for row in members:
                context = row["context"]
                env = TradingEnv(**context["env_kwargs"])
                obs = env.reset(raw_prompt=context["raw_prompt"])
                prefix_info = []
                expected_invalid = prefix_expectations(context)
                for prefix_index, (text, expected) in enumerate(zip(
                    context.get("prefix", []), expected_invalid, strict=True
                )):
                    obs, _, done, info = await env.step(Action(text=text))
                    if done or bool(info.get("invalid_action", False)) != expected:
                        raise RuntimeError("evaluation prefix differs from declared outcome")
                    if "prefix_expected_errors" in context and info.get("error") != (
                        context["prefix_expected_errors"][prefix_index]
                    ):
                        raise RuntimeError("evaluation prefix error differs from declared reason")
                    prefix_info.append(info)
                if "expected_start_session_index" in context and env.current_session_index != (
                    context["expected_start_session_index"]
                ):
                    raise RuntimeError("evaluation prefix differs from declared start session")
                visible = parse_state(obs)
                row.update(
                    status="ACTIVE",
                    prefix_info=prefix_info,
                    prefix_fill_count=len(env.ledger.fills),
                    takeover_equity=visible["portfolio_equity"],
                    takeover_state=visible,
                    max_model_position_weight=max(
                        (p["weight"] for p in visible["positions"].values()), default=0
                    ),
                )
                runtime[row["id"]] = [env, obs]
            while any(r["status"] == "ACTIVE" for r in members):
                submissions = []
                for row in members:
                    if row["status"] != "ACTIVE":
                        continue
                    if row["requests"] >= row["request_limit"]:
                        row.update(status="INCOMPLETE", failure="REQUEST_CAP")
                        continue
                    env, obs = runtime[row["id"]]
                    prompt = render_prompt(tokenizer, env, obs, action_protocol=action_protocol)
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    row["max_prompt_tokens"] = max(row.get("max_prompt_tokens", 0), len(ids))
                    if len(ids) > PROMPT_CAP:
                        row.update(status="INCOMPLETE", failure="PROMPT_CAP")
                        continue
                    seed = derive_request_seed(42, row["identity"], row["requests"])
                    params = dict(
                        temperature=0.0 if row["sampling"] == "greedy" else 1.0,
                        top_p=1.0,
                        top_k=-1,
                        repetition_penalty=1.0,
                        max_tokens=RESPONSE_CAP,
                        seed=seed,
                        n=1,
                    )
                    submissions.append((row, ids, prompt, params))
                save(output / "state.json", state)
                if not submissions:
                    break
                outputs = llm.generate(
                    [{"prompt_token_ids": s[1]} for s in submissions],
                    [sampling_class(**s[3]) for s in submissions],
                    use_tqdm=False,
                )
                if len(outputs) != len(submissions):
                    raise RuntimeError("generation response count mismatch")
                for (row, ids, prompt, params), returned in zip(submissions, outputs, strict=True):
                    if list(returned.prompt_token_ids) != ids or len(returned.outputs) != 1:
                        raise RuntimeError("generation response mapping mismatch")
                    response = returned.outputs[0]
                    tokens = list(response.token_ids)
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    env, _ = runtime[row["id"]]
                    record = {
                        "at_SGT": now(),
                        "trajectory_id": row["id"],
                        "request_index": row["requests"],
                        "prompt": prompt,
                        "prompt_token_ids": ids,
                        "sampling_params": params,
                        "response": text,
                        "response_token_ids": tokens,
                        "finish_reason": response.finish_reason,
                    }
                    row["requests"] += 1
                    row["response_tokens"] += len(tokens)
                    row["max_response_tokens"] = max(row.get("max_response_tokens", 0), len(tokens))
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
                        row["actions"][trace["action"]] = row["actions"].get(trace["action"], 0) + 1
                        row["invalid"] += int(trace["invalid_action"])
                        row["model_fills"] = [
                            asdict(f) for f in env.ledger.fills[row["prefix_fill_count"] :]
                        ]
                        if not trace["invalid_action"]:
                            visible = parse_state(obs)
                            row["max_model_position_weight"] = max(
                                row["max_model_position_weight"],
                                max(
                                    (p["weight"] for p in visible["positions"].values()), default=0
                                ),
                            )
                        record.update(info=info, trace=trace, reward=reward)
                        if done:
                            if not env.episode_complete:
                                raise RuntimeError("terminal flags disagree")
                            row.update(
                                status="COMPLETE",
                                finished_at_SGT=now(),
                                final_state=visible,
                                terminal_metrics=env.screening_terminal_metrics(),
                                takeover_net_return=info["equity"] / row["takeover_equity"] - 1.0,
                            )
                    evidence.write(json.dumps(record, allow_nan=False, default=json_default) + "\n")
                    evidence.flush()
                save(output / "state.json", state)
                print(
                    json.dumps(
                        {
                            "at_SGT": now(),
                            "requests": sum(r["requests"] for r in rows),
                            "statuses": dict(Counter(r["status"] for r in rows)),
                        }
                    ),
                    flush=True,
                )
            for row in members:
                env, _ = runtime[row["id"]]
                row["steps_including_prefix"] = [asdict(s) for s in env.credit_step_trace]
                row["interaction_trace_including_prefix"] = env.interaction_trace
    state.update(status="OBSERVED", finished_at_SGT=now(), summary=summary(rows))
    save(output / "state.json", state)
    save(output / "result.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-protocol", default="single_tool_v1")
    parser.add_argument("--expected-short", type=int, default=16)
    parser.add_argument("--expected-full", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contexts = json.loads(args.contexts.read_text())
    validate_contexts(contexts, args.expected_short, args.expected_full)
    action_protocol = validate_action_protocol(args.action_protocol)
    compiler, compiler_version = _resolve_c_compiler(parent=dict(os.environ))
    os.environ["CC"] = compiler
    for key, value in {
        "VLLM_USE_V1": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }.items():
        os.environ[key] = value
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine: dict[str, Any] = dict(
        model=str(args.model),
        tokenizer=str(args.model),
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
    save(
        args.output / "config.json",
        {
            "engine": engine,
            "started_at_SGT": now(),
            "contexts_sha256": hashlib.sha256(args.contexts.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "action_protocol": action_protocol,
            "expected_short": args.expected_short,
            "expected_full": args.expected_full,
            "compiler": {"CC": compiler, "version": compiler_version},
        },
    )
    llm = LLM(**engine)
    asyncio.run(
        evaluate(
            llm, tokenizer, SamplingParams, contexts, args.output,
            action_protocol=action_protocol,
        )
    )


if __name__ == "__main__":
    main()
