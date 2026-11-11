"""One independent post-action continuation for each genuine parent action.

The original parent return and this continuation form a two-sample action Q.
Groups are selected using only public pre-action histories. The recorded action
is executed unchanged before the new continuation; no auxiliary token trains
the actor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.agents.financial_credit_probe import _load_actor, write_progress
from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.financial_mc_probe import _raw_response_tokens, read_rows, restore_branch
from src.agents.financial_mc_selection import target_report_features
from src.agents.mandate_repair_collect import _safe_result, decoded_generation, read_tasks
from src.agents.same_prefix_online_collect import canonical, require, write

PROFILE = "financial_fixed_action_q_continuation_v1"
ESTIMATOR = "parent_plus_independent_post_action_return_mean_v1"
MAX_RESPONSE = 512


def stable_seed(seed, request_key, call_index):
    raw = canonical([PROFILE, "independent_post_action", seed, request_key, call_index])
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def history_digest(messages):
    return hashlib.sha256(canonical(messages).encode()).hexdigest()


def select_groups(tasks, requests, parents):
    """Select the first public target-read history, never the following action."""
    request_groups, parent_groups = defaultdict(list), defaultdict(list)
    for row in requests:
        require(row.get("role", "parent") == "parent", "only genuine parent requests")
        request_groups[row["trajectory_key"]].append(row)
    for row in parents:
        require(row.get("role", "parent") == "parent", "only genuine parent outcomes")
        parent_groups[row["task_id"]].append(row)
    groups = []
    for task in tasks:
        group = sorted(parent_groups[task["task_id"]], key=lambda row: row["replicate_index"])
        require(
            len(group) == 4 and {row["replicate_index"] for row in group} == set(range(4)),
            "complete K4 parent groups required",
        )
        entries, reason = [], None
        for parent in group:
            chosen = None
            for row in sorted(
                request_groups[parent["trajectory_key"]], key=lambda value: value["call_index"]
            ):
                features = target_report_features(
                    row["messages"],
                    row["call_index"],
                    FinancialEvidenceEnv.maximum_calls,
                )
                if features["target_report_reads"] > 0 and features["calculation_returns"] == 0:
                    chosen = row
                    break
            if chosen is None:
                reason = "NO_PUBLIC_TARGET_PREFIX"
            entries.append(dict(task=task, parent=parent, anchor=chosen))
        if not reason:
            histories = {
                (history_digest(e["anchor"]["messages"]), e["anchor"]["call_index"])
                for e in entries
            }
            if len(histories) != 1:
                reason = "NO_SHARED_HISTORY"
        groups.append(dict(task_id=task["task_id"], entries=entries, unsupported_reason=reason))
    return groups


def restore_post_action(entry):
    """Restore h, then execute the exact already-sampled a once, including bad JSON."""
    anchor = entry["anchor"]
    require(anchor.get("prefix_snapshot"), "recorded pre-action environment snapshot required")
    require(isinstance(anchor.get("response_text"), str), "real sampled parent response required")
    state = restore_branch(entry, 1)
    response = state["env"].step(anchor["response_text"])
    require(
        response == anchor["tool_result"], "fixed parent action reproduces original tool result"
    )
    state["messages"].extend(
        [
            dict(role="assistant", content=anchor["response_text"]),
            dict(role="tool", content=canonical(response)),
        ]
    )
    require(
        state["env"].calls_used == anchor["call_index"] + 1,
        "parent action consumes its original tool call exactly once",
    )
    state.update(
        trajectory_key=f"action-q/{anchor['request_key']}/aux1",
        parent_request_key=anchor["request_key"],
        parent_trajectory_key=entry["parent"]["trajectory_key"],
    )
    return state


def group_reservation(states, max_context):
    return sum(
        max(0, state["env"].maximum_calls - state["env"].calls_used) * max_context
        for state in states
        if not state["env"].done
    )


def action_q_record(entry, result, adapter_sha256, *, complete, reason=None):
    anchor, parent = entry["anchor"], entry["parent"]
    first = parent.get("terminal_reward")
    second = result.get("terminal_reward") if result is not None else None
    return dict(
        estimator=ESTIMATOR,
        m=2,
        task_id=parent["task_id"],
        trajectory_key=parent["trajectory_key"],
        parent_request_key=anchor["request_key"] if anchor is not None else None,
        status="COMPLETE" if complete else "UNSUPPORTED",
        unsupported_reason=reason,
        parent_return=first,
        auxiliary_return=second,
        q_mean=math.fsum([first, second]) / 2 if complete else None,
        auxiliary_trajectory_key=result["trajectory_key"] if result is not None else None,
        adapter_sha256=adapter_sha256,
        continuation_policy="current_policy",
        prefix_history_sha256=history_digest(anchor["messages"]) if anchor else None,
        prefix_calls_used=anchor["call_index"] if anchor else None,
        parent_action_executed_once=True,
        parent_action_resampled=False,
        parent_action_text=anchor["response_text"] if anchor else None,
        parent_action=anchor.get("action") if anchor else None,
        auxiliary_only=True,
        auxiliary_enters_actor=False,
        selection="first_public_target_report_return_before_any_calculation_return",
        selection_uses_parent_outcome=False,
        selection_uses_next_action=False,
        selection_computed_by_history_only_replay=True,
    )


def run(args):
    from src.agents.mandate_repair_train import adapter_digest

    require(args.token_budget >= 0, "nonnegative auxiliary token allowance")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "started.json").exists(), "fresh action Q worker output required")
    tasks = read_tasks(args.data.resolve())
    require(all(t["split"] == "train" for t in tasks), "action Q sampling accepts Train only")
    provenance = __import__("json").loads((args.parents / "collection_started.json").read_text())
    adapter_sha = adapter_digest(args.adapter.resolve())
    require(
        Path(provenance["adapter_path"]).resolve() == args.adapter.resolve()
        and provenance["adapter_sha256"] == adapter_sha,
        "parent and post-action continuation must use the exact sampled actor",
    )
    require(
        Path(provenance["base_model_path"]).resolve() == args.base_model.resolve(),
        "parent and auxiliary base model identity",
    )
    require(
        provenance["environment"] == "financial_evidence"
        and provenance["temperature"] == 1
        and provenance["decode_mode"] == "native",
        "same native temperature-one financial policy",
    )
    require(
        provenance["data_sha256"] == hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "same exact Train batch",
    )
    parents = read_rows(args.parents / "trajectory_results.jsonl")
    groups = select_groups(tasks, read_rows(args.parents / "requests.jsonl"), parents)
    # Restore and reserve only after the public-history choices have been fixed.
    for group in groups:
        group["states"] = []
        if group["unsupported_reason"]:
            continue
        if not all(
            e["parent"].get("status") == "COMPLETE"
            and type(e["parent"].get("terminal_reward")) in (int, float)
            and math.isfinite(e["parent"]["terminal_reward"])
            for e in group["entries"]
        ):
            group["unsupported_reason"] = "INCOMPLETE_PARENT_GROUP"
            continue
        try:
            group["states"] = [restore_post_action(e) for e in group["entries"]]
        except (ValueError, KeyError, TypeError) as error:
            group["unsupported_reason"] = f"POST_ACTION_RESTORE_FAILED:{error}"
        group["reservation"] = group_reservation(group["states"], args.max_context_tokens)
    started = time.monotonic()
    write(
        out / "started.json",
        dict(
            profile=PROFILE,
            estimator=ESTIMATOR,
            data_path=str(args.data.resolve()),
            parent_collection=str(args.parents.resolve()),
            base_model_path=str(args.base_model.resolve()),
            adapter_path=str(args.adapter.resolve()),
            adapter_sha256=adapter_sha,
            token_budget=args.token_budget,
            seed=args.seed,
            temperature=1.0,
            max_context_tokens=args.max_context_tokens,
            max_response_tokens=MAX_RESPONSE,
            m=2,
            new_continuations_per_parent=1,
            parent_return_is_first_sample=True,
            fixed_parent_action=True,
            auxiliary_enters_actor=False,
            formal_valid_access=False,
            formal_final_oos_access=False,
            fresh_test_access=False,
        ),
    )
    counts = Counter(
        dict.fromkeys(
            (
                "model_requests",
                "prompt_tokens",
                "response_tokens",
                "total_tokens",
                "unobserved_response_tokens_upper_bound",
                "completed_groups",
                "unsupported_groups",
                "completed_action_q_values",
                "not_started_budget_groups",
                "failed_model_requests",
            ),
            0,
        )
    )
    logs = {
        name: (out / f"{name}.jsonl").open("x")
        for name in (
            "requests",
            "submissions",
            "trajectory_results",
            "action_q_values",
            "selections",
        )
    }

    def append(name, value):
        logs[name].write(canonical(value) + "\n")
        logs[name].flush()

    def charge():
        return counts["total_tokens"] + counts["unobserved_response_tokens_upper_bound"]

    can_load = any(
        not g["unsupported_reason"] and 0 < g["reservation"] <= args.token_budget for g in groups
    )
    llm, tokenizer, sampling, lora = _load_actor(args, out) if can_load else (None,) * 4
    try:
        for group in groups:
            reason = group["unsupported_reason"]
            if not reason and charge() + group["reservation"] > args.token_budget:
                reason = "FULL_K4_POST_ACTION_RESERVATION"
                counts["not_started_budget_groups"] += 1
            append(
                "selections",
                dict(
                    task_id=group["task_id"],
                    parent_request_keys=[
                        e["anchor"]["request_key"] if e["anchor"] else None
                        for e in group["entries"]
                    ],
                    status="UNSUPPORTED" if reason else "ADMITTED",
                    unsupported_reason=reason,
                    selection_uses_parent_outcome=False,
                    selection_uses_next_action=False,
                    group_complete_before_value_replacement=True,
                ),
            )
            if reason:
                counts["unsupported_groups"] += 1
                for entry in group["entries"]:
                    append(
                        "action_q_values",
                        action_q_record(entry, None, adapter_sha, complete=False, reason=reason),
                    )
                continue
            states = group["states"]
            for _ in range(FinancialEvidenceEnv.maximum_calls):
                pending = []
                for state in states:
                    if state["failure"] is not None or state["env"].done:
                        continue
                    messages = copy.deepcopy(state["messages"])
                    ids = list(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                    call = state["env"].calls_used
                    record = dict(
                        request_key=f"{state['trajectory_key']}/call{call}",
                        trajectory_key=state["trajectory_key"],
                        task_id=state["task_id"],
                        parent_request_key=state["parent_request_key"],
                        parent_trajectory_key=state["parent_trajectory_key"],
                        role="auxiliary_action_q",
                        replicate_index=1,
                        call_index=call,
                        messages=messages,
                        prompt_token_ids=ids,
                        prompt_tokens=len(ids),
                        adapter_path=str(args.adapter.resolve()),
                        adapter_sha256=adapter_sha,
                        temperature=1.0,
                        decode_mode="native",
                        constrained_decoding=False,
                        request_seed=stable_seed(args.seed, state["parent_request_key"], call),
                    )
                    if len(ids) + MAX_RESPONSE > args.max_context_tokens:
                        state["failure"] = dict(
                            status="PROMPT_LIMIT", reason="action Q context limit"
                        )
                        record.update(
                            status="PROMPT_LIMIT",
                            model_request_submitted=False,
                            response_tokens=0,
                            response_token_ids=[],
                        )
                        append("requests", record)
                    else:
                        pending.append((state, record))
                if not pending:
                    break
                counts["model_requests"] += len(pending)
                counts["prompt_tokens"] += sum(r["prompt_tokens"] for _, r in pending)
                counts["total_tokens"] += sum(r["prompt_tokens"] for _, r in pending)
                for _, record in pending:
                    append(
                        "submissions",
                        dict(
                            request_key=record["request_key"],
                            prompt_tokens=record["prompt_tokens"],
                            maximum_response_tokens=MAX_RESPONSE,
                            request_seed=record["request_seed"],
                        ),
                    )
                write_progress(
                    out / "progress.json",
                    dict(status="RUNNING", **counts, budget_charge_tokens=charge()),
                )
                try:
                    kwargs = dict(use_tqdm=False)
                    if lora is not None:
                        kwargs["lora_request"] = lora
                    outputs = llm.generate(
                        [{"prompt_token_ids": r["prompt_token_ids"]} for _, r in pending],
                        [sampling(r["request_seed"]) for _, r in pending],
                        **kwargs,
                    )
                except Exception as error:
                    outputs, model_error = None, f"{type(error).__name__}: {error}"
                else:
                    model_error = None
                raw_counts = [_raw_response_tokens(v) for v in outputs or ()]
                counts["response_tokens"] += sum(raw_counts)
                counts["total_tokens"] += sum(raw_counts)
                counts["unobserved_response_tokens_upper_bound"] += (
                    len(pending) - min(len(outputs or ()), len(pending))
                ) * MAX_RESPONSE
                for index, (state, record) in enumerate(pending):
                    if outputs is None or index >= len(outputs):
                        answer = dict(
                            generation_status="MODEL_ERROR",
                            response_text=None,
                            response_token_ids=[],
                            generation_failure_reason=model_error or "MISSING_MODEL_OUTPUT",
                        )
                    else:
                        answer = decoded_generation(
                            outputs[index], record["prompt_token_ids"], MAX_RESPONSE
                        )
                    record.update(answer, model_request_submitted=True)
                    record["response_tokens"] = raw_counts[index] if index < len(raw_counts) else 0
                    record["response_tokens_unobservable"] = outputs is None or index >= len(
                        outputs
                    )
                    state["request_keys"].append(record["request_key"])
                    if answer["response_text"] is None or answer.get(
                        "generation_failure_reason"
                    ) in {
                        "PROMPT_TOKEN_MISMATCH",
                        "EXPECTED_SINGLE_OUTPUT",
                    }:
                        counts["failed_model_requests"] += 1
                        state["failure"] = dict(
                            status=answer["generation_status"],
                            reason=answer.get("generation_failure_reason"),
                        )
                    else:
                        try:
                            response = state["env"].step(answer["response_text"])
                            state["messages"].extend(
                                [
                                    dict(role="assistant", content=answer["response_text"]),
                                    dict(role="tool", content=canonical(response)),
                                ]
                            )
                            record["tool_result"] = response
                        except Exception as error:
                            state["failure"] = dict(status="ENVIRONMENT_ERROR", reason=str(error))
                    record["status"] = (
                        "COMPLETE" if state["failure"] is None else state["failure"]["status"]
                    )
                    append("requests", record)
                require(charge() <= args.token_budget, "hard full K4 continuation reservation")
                write_progress(
                    out / "progress.json",
                    dict(status="RUNNING", **counts, budget_charge_tokens=charge()),
                )
            outcomes = []
            for state in states:
                result = _safe_result(state)
                result.update(
                    role="auxiliary_action_q",
                    parent_request_key=state["parent_request_key"],
                    parent_trajectory_key=state["parent_trajectory_key"],
                )
                append("trajectory_results", result)
                outcomes.append(result)
            complete = all(
                r["status"] == "COMPLETE"
                and type(r.get("terminal_reward")) in (int, float)
                and math.isfinite(r["terminal_reward"])
                for r in outcomes
            )
            counts["completed_groups" if complete else "unsupported_groups"] += 1
            counts["completed_action_q_values"] += 4 if complete else 0
            for entry, result in zip(group["entries"], outcomes, strict=True):
                append(
                    "action_q_values",
                    action_q_record(
                        entry,
                        result,
                        adapter_sha,
                        complete=complete,
                        reason=None if complete else "INCOMPLETE_AUXILIARY_GROUP",
                    ),
                )
            print(
                canonical(
                    dict(
                        stage="ACTION_Q_GROUP",
                        task_id=group["task_id"],
                        complete=complete,
                        **counts,
                    )
                ),
                flush=True,
            )
        summary = dict(
            profile=PROFILE,
            estimator=ESTIMATOR,
            status="COMPLETE"
            if counts["unsupported_groups"] == 0
            else (
                "BUDGET_EXHAUSTED"
                if counts["not_started_budget_groups"]
                else "COMPLETE_WITH_UNSUPPORTED"
            ),
            **counts,
            planned_groups=len(groups),
            planned_action_q_values=4 * len(groups),
            budget_charge_tokens=charge(),
            token_budget=args.token_budget,
            unused_budget_tokens=args.token_budget - charge(),
            action_q_values_path=str(out / "action_q_values.jsonl"),
            auxiliary_tokens=counts["total_tokens"],
            actor_training_examples=0,
            elapsed_seconds=time.monotonic() - started,
            formal_valid_access=False,
            formal_final_oos_access=False,
            fresh_test_access=False,
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
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--max-context-tokens", type=int, choices=(16384,), default=16384)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
