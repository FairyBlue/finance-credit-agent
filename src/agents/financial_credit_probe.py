"""Paired, current-policy response interventions at recorded financial prefixes.

Only the selected response is changed. Later retries and equivalent documents
remain available. This estimates a response-intervention effect, not a known
true action advantage. Auxiliary continuations never become actor examples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.mandate_repair_collect import (
    _safe_result,
    decoded_generation,
    native_sampling_factory,
    read_tasks,
)
from src.agents.mandate_repair_metrics import business_score
from src.agents.same_prefix_online_collect import _helper, canonical, require, write

PROFILE = "financial_credit_paired_response_probe_v1"
ESTIMATOR = "paired_factual_mask_mean_v1"
CONDITIONS = ("factual", "masked", "placebo")


def write_progress(path, value):
    """Progress is replaced repeatedly; immutable artifacts use write-once."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical(value) + "\n")
    temporary.replace(path)


def stable_seed(seed, *parts):
    raw = canonical([PROFILE, seed, *parts]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def select_anchors(tasks, parents, seed=42, maximum=16):
    """First target-report read per parent; one seeded parent per base question.

    The selector uses the public condition and returned report title, never the
    parent outcome or financial gold value/program.
    """
    task_map = {task["task_id"]: task for task in tasks}
    selected, seen = {}, set()
    with (Path(parents) / "requests.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            task = task_map.get(row.get("task_id"))
            if task is None or row.get("role", "parent") != "parent":
                continue
            key = row["trajectory_key"]
            if key in seen:
                continue
            action, response = row.get("action") or {}, row.get("tool_result") or {}
            condition = task["condition"]
            wanted = f"{condition['company']} annual report {condition['report_year']}".casefold()
            if not (
                action.get("name") == "read_document"
                and response.get("ok") is True
                and isinstance(response.get("text"), str)
                and response.get("title", "").casefold().startswith(wanted + ",")
                and row.get("response_text")
                and row.get("prefix_snapshot")
            ):
                continue
            seen.add(key)
            base = task.get("base_scenario_id", task["task_id"].rsplit("-v", 1)[0])
            candidate = dict(task=task, anchor=row, base_scenario_id=base)
            previous = selected.get(base)
            if previous is None or stable_seed(seed, row["request_key"]) < stable_seed(
                seed, previous["anchor"]["request_key"]
            ):
                selected[base] = candidate
    ordered = sorted(selected.values(), key=lambda x: stable_seed(seed, x["base_scenario_id"]))
    return ordered[:maximum], len(ordered)


def restore_branch(entry, condition, replicate):
    anchor, task = entry["anchor"], entry["task"]
    env = FinancialEvidenceEnv(task)
    messages = copy.deepcopy(anchor["messages"])
    require(env.initial_messages() == messages[:2], "actual same-policy public initial prefix")
    for index, raw in enumerate(anchor["prefix_snapshot"]["action_history"]):
        response = env.step(raw)
        require(
            canonical(response) == messages[3 + index * 2]["content"],
            "recorded financial tool prefix replays exactly",
        )
    require(env.calls_used == anchor["prefix_snapshot"]["calls_used"], "same prefix tool count")
    response = env.step(anchor["response_text"], intervention=condition == "masked")
    placebo_fields = []
    if condition != "masked":
        require(canonical(response) == canonical(anchor["tool_result"]), "same factual response")
    if condition == "placebo":
        for field in ("available_at", "effective_at"):
            if field in response:
                placebo_fields.append(field)
                del response[field]
        if not placebo_fields:
            response["formatting_note"] = "Archive formatting metadata omitted."
            placebo_fields = ["added_formatting_note"]
        env.trace[-1]["response"] = copy.deepcopy(response)
    messages.extend(
        [
            dict(role="assistant", content=anchor["response_text"]),
            dict(role="tool", content=canonical(response)),
        ]
    )
    key = f"probe/{anchor['trajectory_key']}/{condition}/rep{replicate}"
    return dict(
        task_id=task["task_id"],
        trajectory_key=key,
        replicate_index=replicate,
        env=env,
        messages=messages,
        failure=None,
        request_keys=[],
        condition=condition,
        parent_request_key=anchor["request_key"],
        parent_trajectory_key=anchor["trajectory_key"],
        base_scenario_id=entry["base_scenario_id"],
        placebo_fields=placebo_fields,
    )


def gate_result(rows, minimum=8, required_reproducible=3, conditions=CONDITIONS):
    """Fixed engineering screen, not a significance test or universal claim."""
    usable = [row for row in rows if row["status"] == "COMPLETE"]
    full = [row for row in usable if row.get("placebo_target_values")]
    positives, reversals, noisy = [], [], []
    key_losses, placebo_losses = [], []
    for row in full:
        factual, masked, placebo = (
            row[name]
            for name in ("factual_target_values", "masked_target_values", "placebo_target_values")
        )
        losses = [a - b for a, b in zip(factual, masked, strict=True)]
        controls = [a - b for a, b in zip(factual, placebo, strict=True)]
        key_losses.extend(losses)
        placebo_losses.extend(controls)
        if all(loss > 0 and loss > control for loss, control in zip(losses, controls, strict=True)):
            positives.append(row["base_scenario_id"])
        if all(loss < 0 for loss in losses):
            reversals.append(row["base_scenario_id"])
        if min(losses) < max(losses):
            noisy.append(row["base_scenario_id"])
    mean_key = math.fsum(key_losses) / len(key_losses) if key_losses else None
    mean_placebo = math.fsum(placebo_losses) / len(placebo_losses) if placebo_losses else None
    if set(conditions) != set(CONDITIONS):
        status, reason = "INCONCLUSIVE", "TRAINING_LOCAL_ESTIMATOR_WITHOUT_PLACEBO_GATE"
    elif len({row["base_scenario_id"] for row in full}) != len(full):
        status, reason = "INCONCLUSIVE", "DUPLICATE_BASE_QUESTION_SUPPORT"
    elif len(full) < minimum:
        status, reason = "INCONCLUSIVE", "TOO_FEW_COMPLETE_INDEPENDENT_BASE_QUESTIONS"
    elif len(positives) >= required_reproducible and mean_key > abs(mean_placebo):
        status, reason = "PASS", "REPEATED_KEY_BUSINESS_LOSS_EXCEEDS_METADATA_PLACEBO"
    else:
        status, reason = (
            "FAIL",
            (
                "NO_DETECTABLE_LOCAL_BUSINESS_SIGNAL_AT_THIS_PROBE_SCALE"
                if all(value == 0 for value in key_losses + placebo_losses)
                else "INSUFFICIENT_REPRODUCIBILITY_OR_PLACEBO_SEPARATION"
            ),
        )
    return dict(
        status=status,
        reason=reason,
        estimand=ESTIMATOR,
        complete_anchors=len(usable),
        complete_gate_base_questions=len(full),
        minimum_complete_gate_base_questions=minimum,
        required_reproducible_base_questions=required_reproducible,
        reproducible_key_loss_base_questions=positives,
        repeated_reverse_effect_base_questions=reversals,
        varying_paired_key_loss_base_questions=noisy,
        mean_factual_minus_masked_business=mean_key,
        mean_factual_minus_placebo_business=mean_placebo,
        placebo_is_metadata_format_control_not_unrelated_tool_step=True,
        statistical_significance_claim=False,
        no_true_action_advantage_claim=True,
        later_rereading_and_equivalent_sources_unchanged=True,
    )


def _load_actor(args, out):
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        USE_TORCH="1",
        USE_TF="0",
        USE_FLAX="0",
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    from vllm import LLM, sampling_params
    from vllm.lora.request import LoRARequest

    helper = _helper("financial_credit_probe_worker", "m2_continuation_gpu_worker.py")
    compiler, version = helper.resolve_compiler()
    os.environ["CC"] = compiler
    settings = helper.engine_settings(str(args.base_model.resolve()))
    settings.update(
        max_model_len=args.max_context_tokens,
        seed=args.seed,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        compilation_config=dict(level=0, use_cudagraph=False),
        enable_lora=True,
        max_lora_rank=16,
    )
    settings["override_generation_config"].update(temperature=1.0, max_new_tokens=512)
    write(out / "engine_settings.json", settings)
    write(out / "compiler.json", dict(CC=compiler, version=version))
    factory, decoding = native_sampling_factory(sampling_params, 1.0, 512, args.seed)
    write(out / "decoding.json", decoding)
    print("Loading paired financial response probe", str(args.adapter), flush=True)
    llm = LLM(**settings)
    return (
        llm,
        llm.get_tokenizer(),
        factory,
        LoRARequest("financial_probe_actor", 1, str(args.adapter.resolve())),
    )


def run(args):
    require(args.repetitions == 2, "fixed two paired continuation replicates")
    require(len(set(args.conditions)) == len(args.conditions), "unique intervention conditions")
    require({"factual", "masked"} <= set(args.conditions), "paired factual and masked required")
    require(args.token_budget >= 0 and args.max_anchors > 0, "nonnegative finite probe budget")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "started.json").exists(), "fresh probe output directory")
    tasks = read_tasks(args.data.resolve())
    provenance = json.loads((args.parents / "collection_started.json").read_text())
    require(
        Path(provenance["adapter_path"]).resolve() == args.adapter.resolve(),
        "parent prefixes come from the current probe actor",
    )
    require(provenance.get("environment") == "financial_evidence", "same financial environment")
    from src.agents.mandate_repair_train import adapter_digest

    adapter_sha256 = adapter_digest(args.adapter.resolve())
    require(
        provenance.get("adapter_sha256") == adapter_sha256,
        "recorded prefixes and continuations share exact adapter weights",
    )
    parents = {
        row["trajectory_key"]: row
        for row in (
            json.loads(line)
            for line in (args.parents / "trajectory_results.jsonl").read_text().splitlines()
        )
    }
    anchors, eligible = select_anchors(tasks, args.parents, args.seed, args.max_anchors)
    started = time.monotonic()
    write(
        out / "started.json",
        dict(
            profile=PROFILE,
            estimator=ESTIMATOR,
            data_path=str(args.data.resolve()),
            parent_collection=str(args.parents.resolve()),
            base_model=str(args.base_model.resolve()),
            adapter_path=str(args.adapter.resolve()),
            adapter_sha256=adapter_sha256,
            seed=args.seed,
            temperature=1.0,
            repetitions=args.repetitions,
            conditions=args.conditions,
            maximum_anchors=args.max_anchors,
            token_budget=args.token_budget,
            selector="first_public_target_report_read_no_parent_outcome_selection",
            eligible_distinct_base_questions=eligible,
            intervention_scope="one_actual_response_only_later_retries_unchanged",
            auxiliary_results_are_actor_training_examples=False,
            formal_valid_access=False,
            formal_final_oos_access=False,
        ),
    )
    counts = Counter(
        dict.fromkeys(
            (
                "model_requests",
                "prompt_tokens",
                "response_tokens",
                "total_tokens",
                "anchors_started",
                "anchors_complete",
                "not_started_budget_anchors",
            ),
            0,
        )
    )
    local_rows = []
    admissible = any(
        (10 - (entry["anchor"]["prefix_snapshot"]["calls_used"] + 1))
        * args.max_context_tokens
        * args.repetitions
        * len(args.conditions)
        <= args.token_budget
        for entry in anchors
    )
    llm, tokenizer, sampling, lora = _load_actor(args, out) if admissible else (None,) * 4
    logs = {
        name: (out / f"{name}.jsonl").open("x")
        for name in ("anchors", "requests", "trajectory_results", "local_credit", "unsupported")
    }

    def append(name, value):
        logs[name].write(canonical(value) + "\n")
        logs[name].flush()

    try:
        for entry in anchors:
            anchor = entry["anchor"]
            remaining_calls = 10 - (anchor["prefix_snapshot"]["calls_used"] + 1)
            worst = (
                remaining_calls * args.max_context_tokens * args.repetitions * len(args.conditions)
            )
            if counts["total_tokens"] + worst > args.token_budget:
                counts["not_started_budget_anchors"] += 1
                append(
                    "unsupported",
                    dict(
                        base_scenario_id=entry["base_scenario_id"],
                        reason="FULL_PAIRED_ANCHOR_BUDGET_RESERVATION",
                        worst_case_tokens=worst,
                    ),
                )
                continue
            states = []
            for replicate in range(args.repetitions):
                for condition in args.conditions:
                    state = restore_branch(entry, condition, replicate)
                    states.append(state)
                    append(
                        "anchors",
                        dict(
                            trajectory_key=state["trajectory_key"],
                            condition=condition,
                            replicate_index=replicate,
                            parent_request_key=anchor["request_key"],
                            base_scenario_id=entry["base_scenario_id"],
                            original_prefix_messages=anchor["messages"],
                            original_information_action=anchor["response_text"],
                            original_tool_result=anchor["tool_result"],
                            branch_initial_messages=state["messages"],
                            placebo_fields=state["placebo_fields"],
                            parent_tail_copied=False,
                            later_rereading_and_equivalent_sources_unchanged=True,
                        ),
                    )
            counts["anchors_started"] += 1
            for _ in range(10):
                pending = []
                for state in states:
                    if state["failure"] is not None or state["env"].done:
                        continue
                    env = state["env"]
                    messages = copy.deepcopy(state["messages"])
                    ids = list(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                    record = dict(
                        request_key=f"{state['trajectory_key']}/call{env.calls_used}",
                        trajectory_key=state["trajectory_key"],
                        task_id=state["task_id"],
                        parent_request_key=state["parent_request_key"],
                        condition=state["condition"],
                        replicate_index=state["replicate_index"],
                        role="auxiliary_probe",
                        call_index=env.calls_used,
                        messages=messages,
                        prompt_token_ids=ids,
                        prompt_tokens=len(ids),
                        temperature=1.0,
                        decode_mode="native",
                        constrained_decoding=False,
                        adapter_path=str(args.adapter.resolve()),
                        request_seed=stable_seed(
                            args.seed,
                            anchor["request_key"],
                            state["replicate_index"],
                            env.calls_used,
                        ),
                    )
                    if len(ids) + 512 > args.max_context_tokens:
                        state["failure"] = dict(status="PROMPT_LIMIT", reason="probe context limit")
                        record.update(status="PROMPT_LIMIT", model_request_submitted=False)
                        append("requests", record)
                    else:
                        pending.append((state, record))
                if not pending:
                    break
                counts["model_requests"] += len(pending)
                prompt_tokens = sum(record["prompt_tokens"] for _, record in pending)
                counts["prompt_tokens"] += prompt_tokens
                counts["total_tokens"] += prompt_tokens
                try:
                    outputs = llm.generate(
                        [{"prompt_token_ids": record["prompt_token_ids"]} for _, record in pending],
                        [sampling(record["request_seed"]) for _, record in pending],
                        lora_request=lora,
                        use_tqdm=False,
                    )
                    require(len(outputs) == len(pending), "one actual output per request")
                    answers = [
                        decoded_generation(output, record["prompt_token_ids"], 512)
                        for output, (_, record) in zip(outputs, pending, strict=True)
                    ]
                except Exception as error:
                    answers = [
                        dict(
                            generation_status="MODEL_ERROR",
                            response_text=None,
                            response_token_ids=[],
                            generation_failure_reason=str(error),
                        )
                        for _ in pending
                    ]
                for (state, record), answer in zip(pending, answers, strict=True):
                    record.update(answer, model_request_submitted=True)
                    record["response_tokens"] = len(answer["response_token_ids"])
                    counts["response_tokens"] += record["response_tokens"]
                    counts["total_tokens"] += record["response_tokens"]
                    state["request_keys"].append(record["request_key"])
                    if answer["response_text"] is None or answer.get(
                        "generation_failure_reason"
                    ) in {
                        "PROMPT_TOKEN_MISMATCH",
                        "EXPECTED_SINGLE_OUTPUT",
                    }:
                        state["failure"] = dict(
                            status=answer["generation_status"],
                            reason=answer.get("generation_failure_reason"),
                        )
                    else:
                        response = state["env"].step(answer["response_text"])
                        state["messages"].extend(
                            [
                                dict(role="assistant", content=answer["response_text"]),
                                dict(role="tool", content=canonical(response)),
                            ]
                        )
                        record["tool_result"] = response
                    record["status"] = (
                        "COMPLETE" if state["failure"] is None else state["failure"]["status"]
                    )
                    append("requests", record)
                require(counts["total_tokens"] <= args.token_budget, "hard probe generation cap")
                write_progress(
                    out / "progress.json",
                    dict(status="RUNNING", **counts, current_base=entry["base_scenario_id"]),
                )
            outcomes = {condition: [] for condition in args.conditions}
            for state in states:
                result = _safe_result(state)
                result.update(
                    role="auxiliary_probe",
                    condition=state["condition"],
                    parent_request_key=anchor["request_key"],
                    base_scenario_id=entry["base_scenario_id"],
                    paired_continuation_replicate=state["replicate_index"],
                )
                append("trajectory_results", result)
                outcomes[state["condition"]].append(result)
            complete = all(
                business_score(row) is not None for rows in outcomes.values() for row in rows
            )
            row = dict(
                trajectory_key=anchor["trajectory_key"],
                request_key=anchor["request_key"],
                base_scenario_id=entry["base_scenario_id"],
                task_id=entry["task"]["task_id"],
                status="COMPLETE" if complete else "INCOMPLETE_PAIRED_CONTINUATION",
                credit_estimator=ESTIMATOR,
                local_credit_target="business",
                continuation_policy="current_policy",
                arm=args.local_credit_arm,
                adapter_sha256=adapter_sha256,
                parent_target_value=business_score(parents[anchor["trajectory_key"]]),
                parent_reward=parents[anchor["trajectory_key"]].get("terminal_reward"),
                planned_continuations=args.repetitions * len(args.conditions),
                complete_continuations=sum(
                    business_score(r) is not None for values in outcomes.values() for r in values
                ),
            )
            for condition in args.conditions:
                row[f"{condition}_target_values"] = [
                    business_score(value) for value in outcomes[condition]
                ]
                row[f"{condition}_rewards"] = [
                    value.get("terminal_reward") for value in outcomes[condition]
                ]
            row["counterfactual_target_values"] = row["masked_target_values"]
            row["counterfactual_rewards"] = row["masked_rewards"]
            if complete:
                diff = (
                    math.fsum(row["factual_target_values"]) / args.repetitions
                    - math.fsum(row["masked_target_values"]) / args.repetitions
                )
                row.update(
                    unscaled_local_difference=diff,
                    local_credit=diff if args.local_credit_arm != "sham" else 0.0,
                    local_credit_enabled=args.local_credit_arm != "sham",
                )
                append("local_credit", row)
                counts["anchors_complete"] += 1
            else:
                append("unsupported", row)
            local_rows.append(row)
            print(
                canonical(
                    dict(
                        anchor=entry["base_scenario_id"],
                        status=row["status"],
                        local_difference=row.get("unscaled_local_difference"),
                        **counts,
                    )
                ),
                flush=True,
            )
        gate = gate_result(local_rows, args.minimum_gate_anchors, conditions=args.conditions)
        summary = dict(
            status="BUDGET_EXHAUSTED" if counts["not_started_budget_anchors"] else "COMPLETE",
            profile=PROFILE,
            credit_estimator=ESTIMATOR,
            **counts,
            planned_anchors=len(anchors),
            eligible_distinct_base_questions=eligible,
            conditions=args.conditions,
            repetitions=args.repetitions,
            unused_budget_tokens=args.token_budget - counts["total_tokens"],
            token_budget=args.token_budget,
            auxiliary_tokens=counts["total_tokens"],
            local_credit_path=str(out / "local_credit.jsonl"),
            gate_status=gate["status"],
            elapsed_seconds=time.monotonic() - started,
            actor_training_examples=0,
            formal_valid_access=False,
            formal_final_oos_access=False,
        )
        write(out / "gate.json", gate)
        write(out / "summary.json", summary)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=2_000_000)
    parser.add_argument("--max-context-tokens", type=int, default=16384, choices=(16384,))
    parser.add_argument("--max-anchors", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--minimum-gate-anchors", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--local-credit-arm", choices=("sham", "m3", "shuffled"), default="m3")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
