"""Native multi-tool mandate-repair rollouts with complete task-group budgets.

Only the environment's public initial messages, schema and tool results enter
the model prompt. The task's hidden verifier data stays inside the environment.
This worker starts a GPU model only when its CLI is explicitly invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from collections import Counter
from functools import partial
from pathlib import Path

from src.agents.mandate_repair_metrics import (
    local_difference,
    summarize_outcomes,
    trajectory_diagnostics,
)
from src.agents.same_prefix_online_collect import _helper, canonical, require, write

PROFILE = "mandate_repair_native_collect_v1"
MAX_CONTEXT, MAX_RESPONSE, MAX_CALLS, MAX_BATCH = 8192, 512, 8, 32
INFORMATION_TOOLS = ("search_docs", "read_document", "calculate")


def environment_factory(environment="mandate_repair", diagnostic_mode="standard"):
    if environment == "financial_evidence":
        from src.agents.financial_evidence_env import FinancialEvidenceEnv

        factory = partial(FinancialEvidenceEnv, diagnostic_mode=diagnostic_mode)
        factory.maximum_calls = FinancialEvidenceEnv.maximum_calls
        return factory
    if environment != "mandate_repair" or diagnostic_mode != "standard":
        raise ValueError("diagnostics require the financial_evidence environment")
    from src.agents.mandate_repair_env import MandateRepairEnv

    return MandateRepairEnv


def request_seed(seed: int, task_id: str, replicate_index: int, call_index: int) -> int:
    payload = canonical([PROFILE, seed, task_id, replicate_index, call_index])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def read_tasks(path: Path, limit: int | None = None) -> list[dict]:
    raw = path.read_text()
    if path.suffix == ".json":
        value = json.loads(raw)
        tasks = value["tasks"] if isinstance(value, dict) else value
    else:
        tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(isinstance(tasks, list) and tasks, "nonempty task list")
    require(
        all(isinstance(task, dict) and isinstance(task.get("task_id"), str) for task in tasks),
        "each task has a string task_id",
    )
    require(len({task["task_id"] for task in tasks}) == len(tasks), "unique task identities")
    if limit is not None:
        require(limit > 0, "positive tasks limit")
        tasks = tasks[:limit]
    return tasks


def _evaluation_fingerprint(adapter):
    directory = Path(adapter)
    config = json.loads((directory / "adapter_config.json").read_text())
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    weights = directory / "adapter_model.safetensors"
    return dict(
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(canonical(config).encode()).hexdigest(),
    )


def reuse_evaluation(out, provenance, tasks, started):
    """Reuse only an identical, genuinely sampled T0 evaluation in this run/split."""
    if not (
        out.parent.name == "eval" and out.name.endswith(("_dev", "_test"))
        and provenance["temperature"] == 0 and provenance["repetitions"] == 1
        and provenance["seed"] == 43 and provenance["decode_mode"] == "native"
        and provenance["engine_mode"] == "eager" and provenance["max_response_tokens"] == 512
        and provenance["adapter_path"] and provenance["aux_token_budget"] == 0
    ):
        return None
    provenance["evaluation_fingerprint"] = _evaluation_fingerprint(provenance["adapter_path"])
    suffix = "_test" if out.name.endswith("_test") else "_dev"
    keys = ("profile", "base_model_path", "data_path", "temperature", "repetitions", "seed",
            "decode_mode", "engine_mode", "max_context_tokens", "max_response_tokens",
            "max_tool_calls", "action_protocol")
    task_ids = [task["task_id"] for task in tasks]
    legacy_defaults = {"max_context_tokens": MAX_CONTEXT, "max_tool_calls": MAX_CALLS}
    legacy_observations = dict(
        mandate_repair="mandate_repair_known_quote_v1",
        financial_evidence="financial_evidence_conditional_mandate_v1",
    )
    for path in sorted(out.parent.glob(f"*{suffix}/collection_summary.json")):
        if path.parent == out:
            continue
        try:
            source = json.loads(path.read_text())
            if (source.get("evaluation_reused") or source.get("status") != "COMPLETE"
                    or any(
                        source.get(key, legacy_defaults.get(key)) != provenance[key]
                        for key in keys
                    )
                    or source.get("environment", "mandate_repair") != provenance.get(
                        "environment", "mandate_repair")
                    or source.get("diagnostic_mode", "standard") != provenance.get(
                        "diagnostic_mode", "standard")
                    or source.get("observation_profile", legacy_observations.get(
                        source.get("environment", "mandate_repair"))) != provenance.get(
                        "observation_profile", legacy_observations.get(
                            provenance.get("environment", "mandate_repair")))
                    or source.get("task_ids") != task_ids):
                continue
            receipt_time = (path.parent / "collection_started.json").stat().st_mtime_ns
            source_data_sha = source.get("data_sha256")
            if source_data_sha is None:
                data = Path(source["data_path"])
                if data.stat().st_mtime_ns > receipt_time:
                    continue
                source_data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
            if source_data_sha != provenance["data_sha256"]:
                continue
            fingerprint = source.get("evaluation_fingerprint")
            if fingerprint is None:
                adapter = Path(source["adapter_path"])
                files = [adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
                if any(file.stat().st_mtime_ns > receipt_time for file in files):
                    continue
                fingerprint = _evaluation_fingerprint(adapter)
            if fingerprint != provenance["evaluation_fingerprint"]:
                continue
            rows = [json.loads(line) for line in
                    (path.parent / "trajectory_results.jsonl").read_text().splitlines()]
            if len(rows) != len(tasks) or [row["task_id"] for row in rows] != task_ids:
                continue
        except (OSError, KeyError, ValueError, TypeError):
            continue
        source_cost = {key: source.get(key, 0) for key in (
            "model_requests", "prompt_tokens", "response_tokens", "total_tokens",
        )}
        for row in rows:
            row.update(evaluation_reused=True, evaluation_source=str(path.parent),
                       evaluation_requested_adapter_path=provenance["adapter_path"])
        with (out / "trajectory_results.jsonl").open("x") as stream:
            stream.writelines(canonical(row) + "\n" for row in rows)
        for name in ("requests.jsonl", "group_results.jsonl"):
            (out / name).touch(exist_ok=False)
        summary = dict(source)
        summary.update(provenance)
        for key in (
            "model_requests", "prompt_tokens", "response_tokens", "total_tokens",
            "parent_model_requests", "parent_tokens", "aux_model_requests", "aux_tokens",
            "aux_prompt_tokens", "aux_response_tokens", "complete_model_requests",
            "failed_model_requests", "incomplete_generations", "aux_continuations_started",
        ):
            summary[key] = 0
        summary.update(
            results=rows, evaluation_reused=True, evaluation_source=str(path.parent),
            source_sampled_cost=source_cost, source_sampled_tokens=source_cost["total_tokens"],
            source_requests_path=str(path.parent / "requests.jsonl"),
            newly_sampled_trajectories=0, elapsed_seconds=time.monotonic() - started,
        )
        for name in ("collection_summary.json", "summary.json", "collection_state.json"):
            write(out / name, summary)
        print(canonical(dict(evaluation_reused=True, evaluation_source=str(path.parent),
                             actual_model_requests=0, actual_total_tokens=0)), flush=True)
        return summary
    return None


def json_sampling_factory(module, schema: dict, temperature: float, max_response: int, seed: int):
    """Use the installed vLLM strict JSON API, including the older local build."""
    sampling_class = module.SamplingParams
    fields = set(inspect.signature(sampling_class).parameters)
    fields.update(getattr(sampling_class, "__struct_fields__", ()))
    if hasattr(module, "StructuredOutputsParams") and "structured_outputs" in fields:
        guide_class, keyword = module.StructuredOutputsParams, "structured_outputs"
    elif hasattr(module, "GuidedDecodingParams") and "guided_decoding" in fields:
        guide_class, keyword = module.GuidedDecodingParams, "guided_decoding"
    else:
        raise RuntimeError("installed vLLM has no supported strict JSON decoding API")

    def factory(rs):
        return sampling_class(
            temperature=temperature,
            max_tokens=max_response,
            seed=rs,
            logprobs=1,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            **{keyword: guide_class(json=schema)},
        )

    probe = factory(seed)
    require(getattr(probe, keyword).json == schema, "strict public JSON schema")
    return factory, dict(
        api=keyword,
        parameter_class=guide_class.__name__,
        module_path=module.__file__,
        schema=schema,
        logprobs_source="vllm_guided_json_reported",
        logprobs_must_match_training_distribution=True,
    )


def native_sampling_factory(module, temperature: float, max_response: int, seed: int):
    def factory(rs):
        return module.SamplingParams(
            temperature=temperature, max_tokens=max_response, seed=rs,
            top_p=1.0, top_k=-1, repetition_penalty=1.0, logprobs=1,
        )

    factory(seed)
    return factory, dict(
        api="native", module_path=module.__file__,
        logprobs_source="vllm_native_raw_reported",
        sampling_temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0,
        matches_hf_raw_distribution_at_temperature_one=True,
    )


def decoded_generation(result, prompt_ids: list[int], max_response: int) -> dict:
    record = dict(
        generation_status="GENERATION_ERROR",
        generation_failure_reason=None,
        response_text=None,
        response_token_ids=[],
        response_logprobs=[],
        response_logprobs_complete=False,
        returned_finished=getattr(result, "finished", None),
        finish_reason=None,
        stop_reason=None,
        action=None,
    )
    if list(getattr(result, "prompt_token_ids", [])) != prompt_ids:
        record["generation_failure_reason"] = "PROMPT_TOKEN_MISMATCH"
        return record
    outputs = getattr(result, "outputs", [])
    if len(outputs) != 1:
        record["generation_failure_reason"] = "EXPECTED_SINGLE_OUTPUT"
        return record
    answer = outputs[0]
    tokens = list(answer.token_ids)
    logprobs = []
    returned_logprobs = getattr(answer, "logprobs", None)
    for index, token in enumerate(tokens):
        value = None
        if returned_logprobs is not None and index < len(returned_logprobs):
            distribution = returned_logprobs[index]
            entry = distribution.get(token) if isinstance(distribution, dict) else None
            candidate = getattr(entry, "logprob", None)
            if isinstance(candidate, (int, float)) and math.isfinite(candidate):
                value = float(candidate)
        logprobs.append(value)
    record.update(
        response_text=answer.text,
        response_token_ids=tokens,
        response_logprobs=logprobs,
        response_logprobs_complete=bool(tokens) and all(v is not None for v in logprobs),
        finish_reason=answer.finish_reason,
        stop_reason=answer.stop_reason,
    )
    try:
        text = answer.text.strip()
        if text.startswith("<tool_call>") and text.endswith("</tool_call>"):
            text = text[len("<tool_call>"):-len("</tool_call>")]
        action = json.loads(text)
        require(
            isinstance(action, dict) and set(action) == {"name", "arguments"}
            and isinstance(action["name"], str) and isinstance(action["arguments"], dict),
            "one raw JSON tool call with name and arguments",
        )
        record["action"] = action
    except (ValueError, TypeError) as error:
        record["action_parse_error"] = f"INVALID_TOOL_JSON: {error}"
    if not result.finished or answer.finish_reason != "stop":
        record["generation_failure_reason"] = "UNFINISHED_OR_TOKEN_CAPPED"
    elif not tokens or len(tokens) > max_response:
        record["generation_failure_reason"] = "INVALID_RESPONSE_TOKEN_COUNT"
    else:
        record["generation_status"] = "COMPLETE"
    return record


def _safe_result(state: dict) -> dict:
    try:
        result = dict(state["env"].result())
    except Exception as error:
        result = dict(
            task_id=state["task_id"], status="ENVIRONMENT_ERROR", success=False,
            reward=None, metrics={}, trace=[],
            result_error=f"{type(error).__name__}: {error}",
        )
    result["environment_status"] = result.get("status")
    failure = state["failure"]
    result.update(
        task_id=state["task_id"],
        trajectory_key=state["trajectory_key"],
        replicate_index=state["replicate_index"],
        request_keys=state["request_keys"],
        call_count=state["env"].calls_used,
        collector_failure=failure,
        terminal_reward=result.get("reward") if failure is None else None,
        status="COMPLETE" if failure is None and state["env"].done else (
            failure["status"] if failure is not None else "INCOMPLETE"
        ),
        role="parent",
    )
    if failure is not None:
        result.update(success=False, reward=None)
    result.update(trajectory_diagnostics(state["env"], result))
    return result


def _stable_order(seed, value):
    return hashlib.sha256(canonical([PROFILE, seed, value]).encode()).digest()


def _auxiliary_rollouts(
    states, parents, env_class, llm, tokenizer, sampling_factory, logs, counts,
    *, seed, temperature, max_response, token_budget, aux_token_budget,
    model_path, adapter_path, lora_request, arm, local_credit_target="reward",
    max_context=MAX_CONTEXT, max_auxiliary_parents=None,
):
    """Intervene at one real information response without copying the parent tail."""
    max_calls = getattr(env_class, "maximum_calls", MAX_CALLS)
    parent_map = {row["trajectory_key"]: row for row in parents}
    selected, jobs, local_rows, unsupported = {}, [], [], []

    def append(name, value):
        logs[name].write(canonical(value) + "\n")
        logs[name].flush()

    for state in states:
        key = state["trajectory_key"]
        parent = parent_map[key]
        candidates = state["information_candidates"]
        if parent["terminal_reward"] is None or not candidates:
            unsupported.append(dict(
                trajectory_key=key, status="UNSUPPORTED",
                reason="PARENT_HAS_NO_TERMINAL_REWARD" if parent["terminal_reward"] is None
                else "NO_SUCCESSFUL_INFORMATION_STEP",
            ))
            continue
        anchor = min(candidates, key=lambda row: _stable_order(seed, row["request_key"]))
        selected[key] = dict(state=state, parent=parent, anchor=anchor, results=[])
    ordered_keys = sorted(selected, key=lambda value: _stable_order(seed, value))
    if max_auxiliary_parents is not None:
        for key in ordered_keys[max_auxiliary_parents:]:
            del selected[key]
            unsupported.append(dict(
                trajectory_key=key, status="UNSUPPORTED", reason="PREDEFINED_PARENT_LIMIT"
            ))
        ordered_keys = ordered_keys[:max_auxiliary_parents]
    for key in ordered_keys:
        for replicate in range(2):
            jobs.append((key, replicate))
    cursor = 0
    while cursor < len(jobs):
        available_total = (
            token_budget - counts["total_tokens"] if token_budget is not None else math.inf
        )
        available_aux = aux_token_budget - counts["aux_tokens"]
        planned, reserved = [], 0
        while cursor < len(jobs) and len(planned) < MAX_BATCH:
            key, replicate = jobs[cursor]
            entry = selected[key]
            anchor = entry["anchor"]
            remaining_calls = max_calls - (anchor["prefix_snapshot"]["calls_used"] + 1)
            worst = max(0, remaining_calls) * max_context
            if reserved + worst > min(available_total, available_aux):
                if planned:
                    break  # Complete the admitted batch, release its unused reserve, then retry.
                entry.setdefault("skipped_replicates", []).append(replicate)
                counts["aux_budget_skipped_continuations"] += 1
                cursor += 1
                continue
            planned.append((key, replicate))
            reserved += worst
            cursor += 1
        if not planned:
            continue
        branch_states = []
        for key, replicate in planned:
            entry = selected[key]
            anchor, original = entry["anchor"], entry["state"]
            env = env_class(original["task"])
            messages = json.loads(canonical(anchor["messages"]))
            failure = None
            try:
                require(env.initial_messages() == messages[:2], "same public branch initial prefix")
                history = anchor["prefix_snapshot"]["action_history"]
                for index, raw in enumerate(history):
                    replayed = env.step(raw)
                    require(
                        canonical(replayed) == messages[3 + index * 2]["content"],
                        "branch replay equals original tool response",
                    )
                require(env.calls_used == anchor["prefix_snapshot"]["calls_used"],
                        "branch replay call count")
                intervention = env.step(anchor["response_text"], intervention=True)
                require(intervention.get("information_unavailable") is True,
                        "information response actually intervened")
                messages.extend([
                    dict(role="assistant", content=anchor["response_text"]),
                    dict(role="tool", content=canonical(intervention)),
                ])
            except Exception as error:
                failure = dict(status="REPLAY_ERROR", reason=f"{type(error).__name__}: {error}")
            branch_key = f"branch/{key}/cf{replicate}"
            state = dict(
                task_id=original["task_id"], trajectory_key=branch_key,
                replicate_index=replicate, env=env, messages=messages,
                failure=failure, request_keys=[], parent_trajectory_key=key,
                parent_request_key=anchor["request_key"],
            )
            branch_states.append(state)
            append("branch_anchors", dict(
                trajectory_key=branch_key, parent_trajectory_key=key,
                parent_request_key=anchor["request_key"],
                continuation_replicate=replicate,
                intervention="information_response_unavailable",
                original_prefix_messages=anchor["messages"],
                original_information_action=anchor["response_text"],
                branch_initial_messages=messages, parent_tail_copied=False,
                continuation_policy="current_policy", arm=arm,
            ))
        counts["aux_continuations_started"] += len(branch_states)
        for _ in range(max_calls):
            pending = []
            for state in branch_states:
                env = state["env"]
                if state["failure"] is not None or env.done:
                    continue
                messages = json.loads(canonical(state["messages"]))
                ids = list(tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                ))
                call_index = env.calls_used
                request_key = f"{state['trajectory_key']}/call{call_index}"
                record = dict(
                    request_key=request_key, trajectory_key=state["trajectory_key"],
                    task_id=state["task_id"], replicate_index=state["replicate_index"],
                    parent_trajectory_key=state["parent_trajectory_key"],
                    parent_request_key=state["parent_request_key"], call_index=call_index,
                    role="auxiliary", arm=arm, temperature=temperature, decode_mode="native",
                    request_seed=int.from_bytes(_stable_order(seed, request_key)[:8], "big")
                    & ((1 << 63) - 1),
                    model_path=model_path, base_model_path=model_path, adapter_path=adapter_path,
                    messages=messages, prompt_token_ids=ids, prompt_tokens=len(ids),
                    prompt_text=tokenizer.decode(ids, skip_special_tokens=False),
                    continuation_policy="current_policy", parent_tail_copied=False,
                    logprobs_source="vllm_native_raw_reported", constrained_decoding=False,
                )
                if len(ids) + max_response > max_context:
                    state["failure"] = dict(status="PROMPT_LIMIT", reason="branch context limit")
                    record.update(
                        status="PROMPT_LIMIT", model_request_submitted=False,
                        response_token_ids=[], response_text=None, tool_result=None,
                    )
                    append("branch_requests", record)
                else:
                    pending.append((state, record))
            if not pending:
                break
            for start in range(0, len(pending), MAX_BATCH):
                chunk = pending[start:start + MAX_BATCH]
                kwargs = dict(use_tqdm=False)
                if lora_request is not None:
                    kwargs["lora_request"] = lora_request
                prompt_count = sum(record["prompt_tokens"] for _, record in chunk)
                for name in ("model_requests", "aux_model_requests"):
                    counts[name] += len(chunk)
                for name in ("prompt_tokens", "aux_prompt_tokens", "total_tokens", "aux_tokens"):
                    counts[name] += prompt_count
                try:
                    outputs = llm.generate(
                        [{"prompt_token_ids": r["prompt_token_ids"]} for _, r in chunk],
                        [sampling_factory(r["request_seed"]) for _, r in chunk], **kwargs,
                    )
                    require(len(outputs) == len(chunk), "one output per auxiliary request")
                    answers = [decoded_generation(result, record["prompt_token_ids"], max_response)
                               for (_, record), result in zip(chunk, outputs, strict=True)]
                except Exception as error:
                    answers = [dict(
                        generation_status="MODEL_ERROR", response_text=None,
                        response_token_ids=[], generation_failure_reason=str(error),
                    ) for _ in chunk]
                for (state, record), answer in zip(chunk, answers, strict=True):
                    record.update(answer, model_request_submitted=True, tool_result=None)
                    nresponse = len(record["response_token_ids"])
                    record["response_tokens"] = nresponse
                    for name in ("response_tokens", "aux_response_tokens", "total_tokens",
                                 "aux_tokens"):
                        counts[name] += nresponse
                    state["request_keys"].append(record["request_key"])
                    if record["response_text"] is None or record.get(
                        "generation_failure_reason"
                    ) in {"PROMPT_TOKEN_MISMATCH", "EXPECTED_SINGLE_OUTPUT"}:
                        state["failure"] = dict(
                            status=record["generation_status"],
                            reason=record.get("generation_failure_reason"),
                        )
                        record["status"] = record["generation_status"]
                    else:
                        try:
                            tool_result = state["env"].step(record["response_text"])
                            state["messages"].extend([
                                dict(role="assistant", content=record["response_text"]),
                                dict(role="tool", content=canonical(tool_result)),
                            ])
                            record.update(status="COMPLETE", tool_result=tool_result,
                                          tool_success=tool_result.get("ok"),
                                          tool_error=tool_result.get("error"))
                        except Exception as error:
                            state["failure"] = dict(status="ENVIRONMENT_ERROR", reason=str(error))
                            record["status"] = "ENVIRONMENT_ERROR"
                    append("branch_requests", record)
            require(counts["aux_tokens"] <= aux_token_budget, "hard auxiliary token allowance")
            require(token_budget is None or counts["total_tokens"] <= token_budget,
                    "hard total token allowance including branches")
        for state in branch_states:
            result = _safe_result(state)
            result.update(
                role="auxiliary", parent_trajectory_key=state["parent_trajectory_key"],
                parent_request_key=state["parent_request_key"], arm=arm,
                continuation_policy="current_policy", model_path=model_path,
                base_model_path=model_path, adapter_path=adapter_path,
                seed=seed, temperature=temperature,
                continuation_replicate=state["replicate_index"],
            )
            selected[state["parent_trajectory_key"]]["results"].append(result)
            append("branch_results", result)
    for key, entry in selected.items():
        results = entry["results"]
        complete_set = results and all(
            row["status"] == "COMPLETE" and row["terminal_reward"] is not None for row in results
        )
        # A single admitted continuation is valid at the budget tail. An
        # attempted continuation failure is never discarded to keep its peer.
        rewards = [row["terminal_reward"] for row in results] if complete_set else []
        if rewards:
            difference = local_difference(entry["parent"], results, local_credit_target)
            local = difference["unscaled_local_difference"]
            row = dict(
                trajectory_key=key, request_key=entry["anchor"]["request_key"],
                status="COMPLETE", continuation_policy="current_policy", arm=arm,
                **difference, local_credit=local if arm == "m3" else 0.0,
                local_credit_enabled=arm == "m3",
                planned_continuations=2, complete_continuations=len(rewards),
            )
            local_rows.append(row)
            append("local_credit", row)
        else:
            unsupported.append(dict(
                trajectory_key=key, request_key=entry["anchor"]["request_key"],
                status="UNSUPPORTED", reason=("AUXILIARY_BUDGET" if not results
                                               else "NO_COMPLETE_COUNTERFACTUAL"),
                attempted_continuations=len(results),
            ))
    for row in unsupported:
        append("unsupported", row)
    return local_rows, unsupported


def collect_groups(
    tasks: list[dict], env_class, llm, tokenizer, sampling_factory, out: Path,
    *, seed: int, temperature: float, repetitions: int, max_response: int,
    token_budget: int | None, max_groups: int | None, model_path: str,
    adapter_path: str | None = None, lora_request=None, decode_mode: str = "native",
    arm: str = "rloo", aux_token_budget: int = 0, local_credit_target: str = "reward",
    max_context: int = MAX_CONTEXT, max_auxiliary_parents: int | None = None,
    mc_selection: str = "none", mc_selection_seed: int = 42,
    mc_max_anchors: int = 32, mc_selection_budget: int = 0,
) -> dict:
    """Reserve a full worst-case task group before admitting any of its replicas."""
    if max_groups is not None:
        require(max_groups > 0, "positive max groups")
        tasks = tasks[:max_groups]
    max_calls = getattr(env_class, "maximum_calls", MAX_CALLS)
    worst_case_group_tokens = repetitions * max_calls * max_context
    group_batch = max(1, MAX_BATCH // repetitions)
    counts, tool_counts = Counter(), Counter()
    trajectories, group_results = [], []
    logs = {
        name: (out / name).open("x")
        for name in ("requests.jsonl", "trajectory_results.jsonl", "group_results.jsonl")
    }
    selector = None
    if mc_selection != "none":
        from src.agents.financial_mc_selection import PrefixSelector

        require(arm == "rloo", "MC selection does not use response interventions")
        selector = PrefixSelector(
            mc_selection, mc_selection_seed, mc_max_anchors, mc_selection_budget,
            max_context, max_calls,
        )
        logs["prefix_selections.jsonl"] = (out / "prefix_selections.jsonl").open("x")
    local_rows, unsupported = [], []
    if arm != "rloo":
        branch_dir = out / "branches"
        branch_dir.mkdir(exist_ok=False)
        paths = dict(
            branch_requests=branch_dir / "requests.jsonl",
            branch_results=branch_dir / "trajectory_results.jsonl",
            branch_anchors=branch_dir / "anchors.jsonl",
            unsupported=branch_dir / "unsupported.jsonl",
            local_credit=out / "local_credit.jsonl",
        )
        logs.update({name: path.open("x") for name, path in paths.items()})

    def append(name, record):
        logs[name].write(canonical(record) + "\n")
        logs[name].flush()

    def progress(phase):
        temporary = out / "collection_state.tmp"
        temporary.write_text(canonical(dict(
            status="RUNNING", phase=phase, task_groups_finished=len(group_results),
            planned_task_groups=len(tasks), trajectories_finished=len(trajectories),
            token_budget=token_budget, **counts,
        )) + "\n")
        temporary.replace(out / "collection_state.json")

    cursor = 0
    try:
        while cursor < len(tasks):
            capacity = group_batch
            if token_budget is not None:
                capacity = min(capacity, max(
                    0, (token_budget - counts["total_tokens"]) // worst_case_group_tokens
                ))
            if capacity == 0:
                for task in tasks[cursor:]:
                    group = dict(
                        task_id=task["task_id"], status="NOT_STARTED_BUDGET",
                        repetitions=repetitions, model_failure=False,
                        required_worst_case_tokens=worst_case_group_tokens,
                        remaining_budget_tokens=token_budget - counts["total_tokens"],
                    )
                    group_results.append(group)
                    append("group_results.jsonl", group)
                counts["not_started_budget_groups"] += len(tasks) - cursor
                break
            admitted = tasks[cursor : cursor + capacity]
            cursor += len(admitted)
            states = []
            for task in admitted:
                for replicate_index in range(repetitions):
                    env = env_class(task)
                    states.append(dict(
                        task_id=task["task_id"], replicate_index=replicate_index,
                        trajectory_key=f"{task['task_id']}/rep{replicate_index}",
                        env=env, messages=env.initial_messages(), request_keys=[],
                        action_history=[], failure=None, information_candidates=[], task=task,
                    ))
            counts["admitted_task_groups"] += len(admitted)
            progress("tool_rollout")
            for call_index in range(max_calls):
                pending = []
                for state in states:
                    env = state["env"]
                    if state["failure"] is not None or env.done:
                        continue
                    messages = json.loads(canonical(state["messages"]))
                    ids = list(tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    ))
                    key = f"{state['trajectory_key']}/call{call_index}"
                    record = dict(
                        request_key=key, task_id=state["task_id"],
                        trajectory_key=state["trajectory_key"],
                        replicate_index=state["replicate_index"], call_index=call_index,
                        request_seed=request_seed(seed, state["task_id"],
                                                  state["replicate_index"], call_index),
                        role="parent", model_path=model_path, base_model_path=model_path,
                        adapter_path=adapter_path, temperature=temperature,
                        messages=messages,
                        prompt_text=tokenizer.decode(ids, skip_special_tokens=False),
                        prompt_token_ids=ids, prompt_tokens=len(ids),
                        prefix_snapshot=dict(
                            task_id=state["task_id"], calls_used=env.calls_used,
                            messages=messages, action_history=list(state["action_history"]),
                        ),
                        max_response_tokens=max_response,
                        constrained_decoding=decode_mode == "json", decode_mode=decode_mode,
                        logprobs_source=("vllm_guided_json_reported" if decode_mode == "json"
                                         else "vllm_native_raw_reported"),
                    )
                    if len(ids) + max_response > max_context:
                        state["failure"] = dict(
                            status="PROMPT_LIMIT", reason="context budget exceeded",
                            request_key=key,
                        )
                        record.update(
                            status="PROMPT_LIMIT", generation_status="PROMPT_LIMIT",
                            model_request_submitted=False, response_token_ids=[],
                            response_text=None, response_logprobs=[], action=None, tool_result=None,
                        )
                        append("requests.jsonl", record)
                        continue
                    if selector is not None:
                        # Persist a choice using the exact visible prefix before
                        # the current action or its outcome has been generated.
                        selection = selector.consider(
                            key, state["trajectory_key"], messages, env.calls_used,
                        )
                        append("prefix_selections.jsonl", selection)
                        record["mc_prefix_selection"] = selection
                    pending.append((state, record))
                if not pending:
                    break
                for start in range(0, len(pending), MAX_BATCH):
                    chunk = pending[start : start + MAX_BATCH]
                    params = [sampling_factory(record["request_seed"]) for _, record in chunk]
                    kwargs = dict(use_tqdm=False)
                    if lora_request is not None:
                        kwargs["lora_request"] = lora_request
                    counts["model_requests"] += len(chunk)
                    counts["parent_model_requests"] += len(chunk)
                    counts["prompt_tokens"] += sum(record["prompt_tokens"] for _, record in chunk)
                    counts["total_tokens"] += sum(record["prompt_tokens"] for _, record in chunk)
                    counts["parent_tokens"] += sum(record["prompt_tokens"] for _, record in chunk)
                    try:
                        outputs = llm.generate(
                            [{"prompt_token_ids": r["prompt_token_ids"]} for _, r in chunk],
                            params, **kwargs,
                        )
                        require(len(outputs) == len(chunk), "one native output per model request")
                        answers = [decoded_generation(result, record["prompt_token_ids"],
                                                      max_response)
                                   for (_, record), result in zip(chunk, outputs, strict=True)]
                    except Exception as error:
                        answers = [dict(
                            generation_status="MODEL_ERROR",
                            generation_failure_reason=f"{type(error).__name__}: {error}",
                            response_text=None, response_token_ids=[], response_logprobs=[],
                            response_logprobs_complete=False, action=None,
                        ) for _ in chunk]
                    for (state, record), answer in zip(chunk, answers, strict=True):
                        record.update(answer, model_request_submitted=True)
                        record["response_tokens"] = len(record["response_token_ids"])
                        counts["response_tokens"] += record["response_tokens"]
                        counts["total_tokens"] += record["response_tokens"]
                        counts["parent_tokens"] += record["response_tokens"]
                        if record["generation_status"] != "COMPLETE":
                            counts["incomplete_generations"] += 1
                        state["request_keys"].append(record["request_key"])
                        record["tool_result"] = None
                        if record["response_text"] is None or record.get(
                            "generation_failure_reason"
                        ) in {"PROMPT_TOKEN_MISMATCH", "EXPECTED_SINGLE_OUTPUT"}:
                            counts["failed_model_requests"] += 1
                            state["failure"] = dict(
                                status=record["generation_status"],
                                reason=record["generation_failure_reason"],
                                request_key=record["request_key"],
                            )
                            record["status"] = record["generation_status"]
                            append("requests.jsonl", record)
                            continue
                        action = record["action"]
                        state["messages"].append(dict(
                            role="assistant", content=record["response_text"],
                        ))
                        state["action_history"].append(record["response_text"])
                        try:
                            # The environment counts malformed output as a real
                            # failed call and returns feedback; never repair it.
                            tool_result = state["env"].step(record["response_text"])
                            record["tool_result"] = tool_result
                            record["tool_success"] = tool_result.get("ok")
                            record["tool_error"] = tool_result.get("error")
                            if action and action["name"] in INFORMATION_TOOLS and tool_result.get(
                                "ok"
                            ) is True:
                                state["information_candidates"].append(record)
                            state["messages"].append(dict(
                                role="tool", content=canonical(tool_result),
                            ))
                            tool_counts[action["name"] if action else "invalid_output"] += 1
                            record["status"] = "COMPLETE"
                            counts["complete_model_requests"] += 1
                        except Exception as error:
                            state["failure"] = dict(
                                status="ENVIRONMENT_ERROR",
                                reason=f"{type(error).__name__}: {error}",
                                request_key=record["request_key"],
                            )
                            record["status"] = "ENVIRONMENT_ERROR"
                        append("requests.jsonl", record)
                require(
                    token_budget is None or counts["total_tokens"] <= token_budget,
                    "worst-case task-group token reservation holds",
                )
                progress("tool_rollout")
            for state in states:
                result = _safe_result(state)
                result.update(model_path=model_path, base_model_path=model_path,
                              adapter_path=adapter_path, seed=seed, temperature=temperature)
                trajectories.append(result)
                append("trajectory_results.jsonl", result)
            for task in admitted:
                rows = [r for r in trajectories if r["task_id"] == task["task_id"]]
                complete = len(rows) == repetitions and all(r["status"] == "COMPLETE" for r in rows)
                group = dict(
                    task_id=task["task_id"], status="COMPLETE" if complete else "HAS_FAILURES",
                    repetitions=repetitions, completed_replicates=sum(
                        r["status"] == "COMPLETE" for r in rows),
                    successes=sum(r.get("success") is True for r in rows),
                    trajectory_keys=[r["trajectory_key"] for r in rows],
                )
                group_results.append(group)
                append("group_results.jsonl", group)
            if arm != "rloo":
                auxiliary, skipped = _auxiliary_rollouts(
                    states, trajectories, env_class, llm, tokenizer, sampling_factory, logs, counts,
                    seed=seed, temperature=temperature, max_response=max_response,
                    token_budget=token_budget, aux_token_budget=aux_token_budget,
                    model_path=model_path, adapter_path=adapter_path, lora_request=lora_request,
                    arm=arm, local_credit_target=local_credit_target,
                    max_context=max_context, max_auxiliary_parents=max_auxiliary_parents,
                )
                local_rows.extend(auxiliary)
                unsupported.extend(skipped)
            print(f"Mandate groups {len(group_results)}/{len(tasks)}; "
                  f"requests={counts['model_requests']} tokens={counts['total_tokens']}",
                  flush=True)
            progress("task_group_complete")
        if arm != "rloo":
            write(out / "local_credit.json", {
                row["request_key"]: row["local_credit"] for row in local_rows
            })
        completed = [r for r in trajectories if r["status"] == "COMPLETE"]
        return dict(
            status="COMPLETE" if len(completed) == len(tasks) * repetitions else (
                "BUDGET_EXHAUSTED" if counts["not_started_budget_groups"]
                else "COMPLETE_WITH_FAILURES"
            ),
            **{name: counts[name] for name in (
                "model_requests", "prompt_tokens", "response_tokens", "total_tokens",
                "admitted_task_groups", "not_started_budget_groups", "complete_model_requests",
                "failed_model_requests",
                "incomplete_generations",
                "parent_model_requests", "parent_tokens", "aux_model_requests", "aux_tokens",
                "aux_prompt_tokens", "aux_response_tokens", "aux_continuations_started",
                "aux_budget_skipped_continuations",
            )},
            planned_task_groups=len(tasks), planned_trajectories=len(tasks) * repetitions,
            started_trajectories=len(trajectories), complete_trajectories=len(completed),
            failed_trajectories=len(trajectories) - len(completed),
            successes=sum(row.get("success") is True for row in completed),
            success_rate_all_planned=(
                sum(row.get("success") is True for row in completed) / (len(tasks) * repetitions)
                if len(completed) == len(tasks) * repetitions else None
            ),
            token_budget=token_budget, worst_case_group_tokens=worst_case_group_tokens,
            unused_budget_tokens=(token_budget - counts["total_tokens"] if token_budget else None),
            tool_counts=dict(tool_counts), task_ids=[task["task_id"] for task in tasks],
            budget_exhaustion_is_model_failure=False, failed_slots_imputed=False,
            results=trajectories, groups=group_results,
            arm=arm, aux_token_budget=aux_token_budget, local_credit_target=local_credit_target,
            outcome_diagnostics=summarize_outcomes(trajectories),
            local_credit_supported_parents=len(local_rows),
            local_credit_unsupported_parents=len(unsupported),
            local_credit_path=str(out / "local_credit.jsonl") if arm != "rloo" else None,
            mc_selection=mc_selection,
            mc_selected_anchors=selector.selected if selector else 0,
            mc_reserved_auxiliary_tokens=selector.reserved if selector else 0,
            mc_selections_path=str(out / "prefix_selections.jsonl") if selector else None,
        )
    finally:
        for stream in logs.values():
            stream.close()


def run(args) -> dict:
    environment = getattr(args, "environment", "mandate_repair")
    diagnostic_mode = getattr(args, "diagnostic_mode", "standard")
    env_class = environment_factory(environment, diagnostic_mode)
    max_calls = getattr(env_class, "maximum_calls", MAX_CALLS)
    max_context = getattr(args, "max_context_tokens", MAX_CONTEXT)
    out = args.output.resolve()
    base_model = (args.base_model or args.model).resolve()
    adapter = args.adapter.resolve() if args.adapter is not None else None
    require(base_model.is_dir(), "existing common BF16 base model")
    require(adapter is None or adapter.is_dir(), "existing adapter if requested")
    require(args.repetitions > 0 and args.repetitions <= MAX_BATCH, "one to32 replicas")
    require(math.isfinite(args.temperature) and args.temperature >= 0, "nonnegative temperature")
    require(1 <= args.max_response_tokens <= MAX_RESPONSE, "response budget up to512")
    require(args.token_budget is None or args.token_budget > 0, "positive optional token budget")
    require(args.aux_token_budget >= 0, "nonnegative auxiliary token allowance")
    require(
        getattr(args, "max_auxiliary_parents", None) is None
        or args.max_auxiliary_parents > 0, "positive optional auxiliary-parent limit",
    )
    require(diagnostic_mode == "standard" or args.arm == "rloo", "diagnostic-only rollouts")
    require(args.arm == "rloo" or args.decode_mode == "native", "main branches use native policy")
    tasks = read_tasks(args.data.resolve(), args.tasks_limit)
    if args.max_groups is not None:
        require(args.max_groups > 0, "positive maximum task groups")
        tasks = tasks[:args.max_groups]
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "collection_started.json").exists(), "fresh collector output required")
    started = time.monotonic()
    schema_env = env_class(tasks[0])
    schema = schema_env.choice_schema()
    adapter_sha256 = None
    if adapter is not None:
        from src.agents.mandate_repair_train import adapter_digest

        adapter_sha256 = adapter_digest(adapter)
    provenance = dict(
        profile=PROFILE, data_path=str(args.data.resolve()), model_path=str(base_model),
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        base_model_path=str(base_model), adapter_path=str(adapter) if adapter else None,
        adapter_sha256=adapter_sha256,
        seed=args.seed, temperature=args.temperature, repetitions=args.repetitions,
        max_context_tokens=max_context, max_response_tokens=args.max_response_tokens,
        max_tool_calls=max_calls, role="parent", environment=environment,
        diagnostic_mode=diagnostic_mode,
        observation_profile=schema_env.visible()["profile"],
        action_protocol="raw_json_or_single_tool_call_wrapper_v1", decode_mode=args.decode_mode,
        constrained_decoding=args.decode_mode == "json", main_protocol=args.decode_mode == "native",
        arm=args.arm, aux_token_budget=args.aux_token_budget,
        local_credit_target=args.local_credit_target,
        engine_mode=args.engine_mode,
        evaluation_reused=False,
        hidden_verifier_in_model_prompt=diagnostic_mode == "oracle_facts",
        mc_selection=getattr(args, "mc_selection", "none"),
        mc_selection_seed=getattr(args, "mc_selection_seed", 42),
        mc_max_anchors=getattr(args, "mc_max_anchors", 32),
        mc_selection_budget=getattr(args, "mc_selection_budget", 0),
        privileged_correct_facts_diagnostic=diagnostic_mode == "oracle_facts",
        formal_valid_access=False,
        formal_final_oos_access=False, optimizer_updates=0,
    )
    write(out / "collection_started.json", dict(
        **provenance, task_ids=[t["task_id"] for t in tasks],
    ))
    reused = reuse_evaluation(out, provenance, tasks, started)
    if reused is not None:
        return reused
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        USE_TORCH="1", USE_TF="0", USE_FLAX="0", VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    from vllm import LLM, sampling_params

    if args.decode_mode == "json":
        sampling_factory, decoding = json_sampling_factory(
            sampling_params, schema, args.temperature, args.max_response_tokens, args.seed,
        )
    else:
        sampling_factory, decoding = native_sampling_factory(
            sampling_params, args.temperature, args.max_response_tokens, args.seed,
        )
    write(out / "constrained_decoding.json", decoding)
    helper = _helper("mandate_repair_native_worker", "m2_continuation_gpu_worker.py")
    compiler, version = helper.resolve_compiler()
    os.environ["CC"] = compiler
    settings = helper.engine_settings(str(base_model))
    settings.update(max_model_len=max_context, seed=args.seed, gpu_memory_utilization=0.35)
    if args.engine_mode == "eager":
        settings.update(
            enforce_eager=True,
            compilation_config=dict(level=0, use_cudagraph=False),
        )
    settings["override_generation_config"].update(
        temperature=args.temperature, max_new_tokens=args.max_response_tokens,
    )
    lora_request = None
    if adapter is not None:
        settings.update(enable_lora=True, max_lora_rank=16)
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("mandate_current_actor", 1, str(adapter))
    write(out / "engine_settings.json", settings)
    write(out / "compiler.json", dict(CC=compiler, version=version))
    print("Loading native mandate-repair actor", str(base_model), str(adapter), flush=True)
    llm = LLM(**settings)
    summary = collect_groups(
        tasks, env_class, llm, llm.get_tokenizer(), sampling_factory, out,
        seed=args.seed, temperature=args.temperature, repetitions=args.repetitions,
        max_response=args.max_response_tokens, token_budget=args.token_budget,
        max_groups=None, model_path=str(base_model), adapter_path=str(adapter) if adapter else None,
        lora_request=lora_request, decode_mode=args.decode_mode,
        arm=args.arm, aux_token_budget=args.aux_token_budget,
        local_credit_target=args.local_credit_target,
        max_context=max_context,
        max_auxiliary_parents=getattr(args, "max_auxiliary_parents", None),
        mc_selection=getattr(args, "mc_selection", "none"),
        mc_selection_seed=getattr(args, "mc_selection_seed", 42),
        mc_max_anchors=getattr(args, "mc_max_anchors", 32),
        mc_selection_budget=getattr(args, "mc_selection_budget", 0),
    )
    summary.update(**provenance, constrained_decoding_api=decoding["api"],
                   elapsed_seconds=time.monotonic() - started)
    write(out / "collection_summary.json", summary)
    write(out / "summary.json", summary)
    public_summary = {k: v for k, v in summary.items() if k not in {"results", "groups"}}
    temporary = out / "collection_state.tmp"
    temporary.write_text(canonical(public_summary) + "\n")
    temporary.replace(out / "collection_state.json")
    print(canonical(public_summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model", type=Path)
    models.add_argument("--base-model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tasks-limit", type=int)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-response-tokens", type=int, default=MAX_RESPONSE)
    parser.add_argument(
        "--max-context-tokens", type=int, choices=(8192, 16384), default=MAX_CONTEXT,
    )
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--decode-mode", choices=("native", "json"), default="native")
    parser.add_argument("--arm", choices=("rloo", "sham", "m3"), default="rloo")
    parser.add_argument("--aux-token-budget", type=int, default=0)
    parser.add_argument("--local-credit-target", choices=("reward", "business"), default="reward")
    parser.add_argument("--engine-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument(
        "--environment", choices=("mandate_repair", "financial_evidence"),
        default="mandate_repair",
    )
    parser.add_argument(
        "--diagnostic-mode", choices=("standard", "oracle_facts", "fixed_retrieval"),
        default="standard",
    )
    parser.add_argument("--max-auxiliary-parents", type=int)
    parser.add_argument("--mc-selection", choices=("none", "uniform", "priority"), default="none")
    parser.add_argument("--mc-selection-seed", type=int, default=42)
    parser.add_argument("--mc-max-anchors", type=int, default=32)
    parser.add_argument("--mc-selection-budget", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
