"""Independent current-policy continuations from explicit pre-action selections.

This worker estimates a state-value baseline.  It never executes the recorded
parent action before branching and never reads the parent terminal outcome.
Selection is supplied by the parent collector, not inferred from its results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from src.agents.financial_credit_probe import _load_actor, write_progress
from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.mandate_repair_collect import _safe_result, decoded_generation, read_tasks
from src.agents.same_prefix_online_collect import canonical, require, write

PROFILE = "financial_pre_action_mc_state_value_v1"
ESTIMATOR = "independent_current_policy_state_value_mean_v1"
MAX_RESPONSE = 512


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stable_seed(seed, request_key, replicate, call_index):
    """A namespace distinct from parent sampling and allocation randomness."""
    raw = canonical([PROFILE, "auxiliary_state_value", seed, request_key, replicate, call_index])
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def select_anchors(tasks, parents, selections, maximum=32):
    """Honor an explicit pre-action manifest without inspecting parent actions."""
    task_map = {task["task_id"]: task for task in tasks}
    requests = {
        row["request_key"]: row for row in read_rows(Path(parents) / "requests.jsonl")
        if row.get("role", "parent") == "parent"
    }
    chosen, seen = [], set()
    for selection in read_rows(selections):
        if selection.get("selected", True) is not True:
            continue
        key = selection.get("parent_request_key", selection.get("request_key"))
        require(isinstance(key, str) and key in requests, "selected parent request exists")
        require(key not in seen, "one selection per parent request")
        seen.add(key)
        probability = selection.get("selection_probability")
        require(
            type(probability) in (int, float) and math.isfinite(probability)
            and 0 < probability <= 1,
            "finite positive pre-action selection probability",
        )
        require(selection.get("pre_action_selection") is True, "selection precedes parent action")
        anchor = requests[key]
        require(anchor["task_id"] in task_map, "selected request belongs to supplied train tasks")
        require(anchor.get("messages") and anchor.get("prefix_snapshot"), "real parent prefix")
        if "observed_calls_used" in selection:
            require(
                selection["observed_calls_used"] == anchor["prefix_snapshot"]["calls_used"],
                "selection call count matches the pre-action snapshot",
            )
        chosen.append(dict(task=task_map[anchor["task_id"]], anchor=anchor, selection=selection))
    return chosen[:maximum], len(chosen)


def restore_branch(entry, replicate):
    """Replay only earlier calls; the next action is resampled by the actor."""
    anchor, task = entry["anchor"], entry["task"]
    env = FinancialEvidenceEnv(task)
    messages = copy.deepcopy(anchor["messages"])
    require(env.initial_messages() == messages[:2], "same public initial messages")
    history = anchor["prefix_snapshot"]["action_history"]
    require(len(messages) == 2 + 2 * len(history), "messages end before the parent action")
    for index, raw in enumerate(history):
        require(messages[2 + 2 * index] == dict(role="assistant", content=raw),
                "recorded prefix assistant action matches snapshot")
        response = env.step(raw)
        require(
            messages[3 + 2 * index] == dict(role="tool", content=canonical(response)),
            "replayed tool response equals the actual prefix",
        )
    require(env.calls_used == anchor["prefix_snapshot"]["calls_used"], "same pre-action call count")
    require(not env.done and env.calls_used < env.maximum_calls, "a real remaining action exists")
    key = f"mc/{anchor['request_key']}/rep{replicate}"
    return dict(
        task_id=task["task_id"], trajectory_key=key, replicate_index=replicate,
        env=env, messages=messages, failure=None, request_keys=[],
        parent_request_key=anchor["request_key"],
        parent_trajectory_key=anchor["trajectory_key"],
    )


def _raw_response_tokens(output):
    return sum(len(getattr(item, "token_ids", ())) for item in getattr(output, "outputs", ()))


def run(args):
    require(args.repetitions == 2, "exactly two independent MC continuations")
    require(args.token_budget >= 0 and args.max_anchors > 0, "nonnegative finite worker budget")
    require("test" not in args.data.name.casefold(), "MC state baselines use Train only")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "started.json").exists(), "fresh MC output directory")
    tasks = read_tasks(args.data.resolve())
    provenance = json.loads((args.parents / "collection_started.json").read_text())
    from src.agents.mandate_repair_train import adapter_digest

    adapter_sha256 = adapter_digest(args.adapter.resolve())
    require(
        Path(provenance["adapter_path"]).resolve() == args.adapter.resolve()
        and provenance.get("adapter_sha256") == adapter_sha256,
        "parent prefixes and auxiliary continuations use the exact current actor",
    )
    require(provenance.get("environment") == "financial_evidence", "financial environment")
    require(provenance.get("temperature", 1) == 1, "current raw policy uses temperature one")
    require(provenance.get("decode_mode", "native") == "native", "raw native action policy")
    anchors, selected_count = select_anchors(
        tasks, args.parents, args.selections, args.max_anchors,
    )
    started = time.monotonic()
    write(out / "started.json", dict(
        profile=PROFILE, estimator=ESTIMATOR, data_path=str(args.data.resolve()),
        parent_collection=str(args.parents.resolve()),
        selections_path=str(args.selections.resolve()),
        adapter_path=str(args.adapter.resolve()), adapter_sha256=adapter_sha256,
        base_model=str(args.base_model.resolve()), token_budget=args.token_budget,
        repetitions=args.repetitions, seed=args.seed, maximum_anchors=args.max_anchors,
        parent_action_excluded=True, parent_return_excluded=True, aux_rng_independent=True,
        continuation_policy="current_policy", temperature=1.0,
        auxiliary_results_are_actor_training_examples=False,
        reward_semantics="original_environment_terminal_reward_including_common_prefix_cost",
        formal_valid_access=False, formal_final_oos_access=False, fresh_test_access=False,
    ))
    counts = Counter(dict.fromkeys((
        "model_requests", "prompt_tokens", "response_tokens", "total_tokens",
        "unobserved_response_tokens_upper_bound", "anchors_started", "completed_baselines",
        "incomplete_baselines", "not_started_budget_anchors", "failed_model_requests",
    ), 0))
    rows = []
    logs = {
        name: (out / f"{name}.jsonl").open("x")
        for name in ("anchors", "requests", "submissions", "trajectory_results",
                     "state_value_baselines", "mc_baselines", "unsupported")
    }

    def append(name, value):
        logs[name].write(canonical(value) + "\n")
        logs[name].flush()

    def budget_charge():
        return counts["total_tokens"] + counts["unobserved_response_tokens_upper_bound"]

    def worst_case(entry):
        calls = (FinancialEvidenceEnv.maximum_calls
                 - entry["anchor"]["prefix_snapshot"]["calls_used"])
        return calls * args.max_context_tokens * args.repetitions

    # A zero or unusable budget must not load a GPU model.
    loadable = any(0 < worst_case(entry) <= args.token_budget for entry in anchors)
    llm, tokenizer, sampling, lora = _load_actor(args, out) if loadable else (None,) * 4
    try:
        for entry in anchors:
            anchor, selection = entry["anchor"], entry["selection"]
            worst = worst_case(entry)
            if worst <= 0 or budget_charge() + worst > args.token_budget:
                counts["not_started_budget_anchors"] += 1
                append("unsupported", dict(
                    parent_request_key=anchor["request_key"], status="NOT_STARTED_BUDGET",
                    reason="FULL_TWO_CONTINUATION_RESERVATION", worst_case_tokens=worst,
                ))
                continue
            states = [restore_branch(entry, replicate) for replicate in range(args.repetitions)]
            counts["anchors_started"] += 1
            anchor_cost = counts["total_tokens"]
            anchor_requests = counts["model_requests"]
            for state in states:
                append("anchors", dict(
                    parent_request_key=anchor["request_key"],
                    trajectory_key=state["trajectory_key"],
                    parent_trajectory_key=anchor["trajectory_key"],
                    replicate_index=state["replicate_index"],
                    branch_initial_messages=state["messages"],
                    prefix_calls_used=state["env"].calls_used,
                    selection_probability=selection["selection_probability"],
                    selection_policy=selection.get("selection_policy"), pre_action_selection=True,
                    parent_action_excluded=True, parent_return_excluded=True,
                    continuation_policy="current_policy", aux_rng_independent=True,
                ))
            for _ in range(FinancialEvidenceEnv.maximum_calls):
                pending = []
                for state in states:
                    if state["failure"] is not None or state["env"].done:
                        continue
                    call_index = state["env"].calls_used
                    messages = copy.deepcopy(state["messages"])
                    ids = list(tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    ))
                    record = dict(
                        request_key=f"{state['trajectory_key']}/call{call_index}",
                        trajectory_key=state["trajectory_key"], task_id=state["task_id"],
                        parent_request_key=anchor["request_key"], role="auxiliary_state_value",
                        replicate_index=state["replicate_index"], call_index=call_index,
                        messages=messages, prompt_token_ids=ids, prompt_tokens=len(ids),
                        adapter_path=str(args.adapter.resolve()), adapter_sha256=adapter_sha256,
                        temperature=1.0, decode_mode="native", constrained_decoding=False,
                        request_seed=stable_seed(args.seed, anchor["request_key"],
                                                 state["replicate_index"], call_index),
                    )
                    if len(ids) + MAX_RESPONSE > args.max_context_tokens:
                        state["failure"] = dict(status="PROMPT_LIMIT", reason="MC context limit")
                        record.update(status="PROMPT_LIMIT", model_request_submitted=False,
                                      response_tokens=0, response_token_ids=[])
                        append("requests", record)
                    else:
                        pending.append((state, record))
                if not pending:
                    break
                counts["model_requests"] += len(pending)
                counts["prompt_tokens"] += sum(r["prompt_tokens"] for _, r in pending)
                counts["total_tokens"] += sum(r["prompt_tokens"] for _, r in pending)
                for _, record in pending:
                    append("submissions", dict(
                        request_key=record["request_key"], prompt_tokens=record["prompt_tokens"],
                        maximum_response_tokens=MAX_RESPONSE, request_seed=record["request_seed"],
                    ))
                write_progress(out / "progress.json", dict(
                    status="RUNNING", phase="MODEL_REQUEST_SUBMITTED", **counts,
                    budget_charge_tokens=budget_charge(),
                ))
                try:
                    kwargs = dict(use_tqdm=False)
                    if lora is not None:
                        kwargs["lora_request"] = lora
                    outputs = llm.generate(
                        [{"prompt_token_ids": record["prompt_token_ids"]} for _, record in pending],
                        [sampling(record["request_seed"]) for _, record in pending], **kwargs,
                    )
                except Exception as error:
                    outputs = None
                    model_error = f"{type(error).__name__}: {error}"
                else:
                    model_error = None
                # Count returned raw tokens even if validation or JSON parsing fails.
                raw_counts = [_raw_response_tokens(value) for value in outputs or ()]
                counts["response_tokens"] += sum(raw_counts)
                counts["total_tokens"] += sum(raw_counts)
                if outputs is None:
                    counts["unobserved_response_tokens_upper_bound"] += len(pending) * MAX_RESPONSE
                elif len(outputs) < len(pending):
                    counts["unobserved_response_tokens_upper_bound"] += (
                        len(pending) - len(outputs)
                    ) * MAX_RESPONSE
                for index, (state, record) in enumerate(pending):
                    if outputs is None or index >= len(outputs):
                        answer = dict(
                            generation_status="MODEL_ERROR", response_text=None,
                            response_token_ids=[], generation_failure_reason=model_error
                            or "MISSING_MODEL_OUTPUT",
                        )
                    else:
                        answer = decoded_generation(outputs[index], record["prompt_token_ids"],
                                                    MAX_RESPONSE)
                    record.update(answer, model_request_submitted=True)
                    record["response_tokens"] = raw_counts[index] if index < len(raw_counts) else 0
                    record["response_tokens_unobservable"] = (
                        outputs is None or index >= len(outputs)
                    )
                    if outputs is not None and index < len(outputs):
                        record["raw_returned_response_token_ids"] = [
                            list(getattr(item, "token_ids", ()))
                            for item in getattr(outputs[index], "outputs", ())
                        ]
                    state["request_keys"].append(record["request_key"])
                    if answer["response_text"] is None or answer.get(
                        "generation_failure_reason"
                    ) in {
                        "PROMPT_TOKEN_MISMATCH", "EXPECTED_SINGLE_OUTPUT",
                    }:
                        counts["failed_model_requests"] += 1
                        state["failure"] = dict(status=answer["generation_status"],
                                                reason=answer.get("generation_failure_reason"))
                    else:
                        try:
                            response = state["env"].step(answer["response_text"])
                            state["messages"].extend([
                                dict(role="assistant", content=answer["response_text"]),
                                dict(role="tool", content=canonical(response)),
                            ])
                            record["tool_result"] = response
                        except Exception as error:
                            state["failure"] = dict(status="ENVIRONMENT_ERROR", reason=str(error))
                    record["status"] = (
                        "COMPLETE" if state["failure"] is None else state["failure"]["status"]
                    )
                    append("requests", record)
                require(budget_charge() <= args.token_budget, "hard full-pair token reservation")
                write_progress(out / "progress.json", dict(
                    status="RUNNING", phase="CONTINUATIONS", **counts,
                    budget_charge_tokens=budget_charge(),
                ))
            outcomes = []
            for state in states:
                result = _safe_result(state)
                result.update(role="auxiliary_state_value",
                              parent_request_key=anchor["request_key"])
                append("trajectory_results", result)
                outcomes.append(result)
            returns = [row.get("terminal_reward") for row in outcomes]
            complete = all(
                row["status"] == "COMPLETE" and type(value) in (int, float) and math.isfinite(value)
                for row, value in zip(outcomes, returns, strict=True)
            )
            row = dict(
                record_type="state_value_baseline", estimator=ESTIMATOR,
                request_key=anchor["request_key"], parent_request_key=anchor["request_key"],
                trajectory_key=anchor["trajectory_key"], task_id=anchor["task_id"],
                status="COMPLETE" if complete else "INCOMPLETE", m=args.repetitions,
                aux_returns=returns,
                mc_baseline=math.fsum(returns) / args.repetitions if complete else None,
                aux_trajectory_keys=[state["trajectory_key"] for state in states],
                adapter_sha256=adapter_sha256, continuation_policy="current_policy",
                selection_probability=selection["selection_probability"],
                selection_policy=selection.get("selection_policy"), pre_action_selection=True,
                parent_action_excluded=True, parent_return_excluded=True, aux_rng_independent=True,
                reward_semantics="original_environment_terminal_reward_including_common_prefix_cost",
                prefix_calls_used=anchor["prefix_snapshot"]["calls_used"],
                actual_generation_tokens=counts["total_tokens"] - anchor_cost,
                model_requests=counts["model_requests"] - anchor_requests,
            )
            for name in ("state_value_baselines", "mc_baselines"):
                append(name, row)
            counts["completed_baselines" if complete else "incomplete_baselines"] += 1
            rows.append(row)
            print(canonical(dict(parent_request_key=anchor["request_key"], status=row["status"],
                                 mc_baseline=row["mc_baseline"], **counts)), flush=True)
        status = "COMPLETE"
        if counts["incomplete_baselines"]:
            status = "COMPLETE_WITH_FAILURES"
        elif counts["not_started_budget_anchors"]:
            status = "BUDGET_EXHAUSTED"
        summary = dict(
            status=status, profile=PROFILE, estimator=ESTIMATOR, **counts,
            planned_anchors=len(anchors), selected_requests=selected_count,
            budget_charge_tokens=budget_charge(), token_budget=args.token_budget,
            unused_budget_tokens=args.token_budget - budget_charge(),
            auxiliary_tokens=counts["total_tokens"],
            mc_baseline_path=str(out / "state_value_baselines.jsonl"),
            state_value_baseline_path=str(out / "state_value_baselines.jsonl"),
            actor_training_examples=0, elapsed_seconds=time.monotonic() - started,
            formal_valid_access=False, formal_final_oos_access=False, fresh_test_access=False,
        )
        write_progress(out / "summary.json", summary)
        write_progress(out / "progress.json", summary)
        print(canonical(summary), flush=True)
        return summary
    finally:
        for stream in logs.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "parents", "base-model", "adapter", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--selections", "--selection", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=2_000_000)
    parser.add_argument("--max-context-tokens", type=int, choices=(16384,), default=16384)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--max-anchors", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
