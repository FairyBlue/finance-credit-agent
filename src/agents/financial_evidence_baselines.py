"""Gold-selected Train demonstrations and explicitly privileged CPU references."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from src.agents.financial_evidence_data import DEFAULT, canonical, write_json
from src.agents.financial_evidence_env import FinancialEvidenceEnv
from src.agents.mandate_repair_baselines import require_frozen_models

PROFILE = "financial_evidence_teacher_and_privileged_reference_v1"
EXPECTED = dict(train=800, development=100, test=100)


def load_tasks(root, split, *, allow_frozen_test=False):
    if split not in EXPECTED:
        raise ValueError("unknown financial task partition")
    if split == "test":
        if not allow_frozen_test:
            raise PermissionError("test reference requires explicit frozen-test entry point")
        require_frozen_models(root)
    tasks = [
        json.loads(line)
        for line in (Path(root) / "data" / f"{split}.jsonl").read_text().splitlines()
    ]
    if len(tasks) != EXPECTED[split] or any(t["split"] != split for t in tasks):
        raise ValueError("fixed financial task count/split changed")
    return tasks


def teacher_actions(env, task, perform):
    """Gold selects Train steps; prefixes contain only actual public observations."""
    condition, gold = task["condition"], task["gold"]
    perform("search_docs", {"query": f"{condition['company']} {condition['report_year']} mandate"})
    mandatory = gold["required_citations"]
    sufficient = [group[0] for group in gold["evidence_groups"]]
    citations = [*mandatory, *sufficient]
    for identifier in citations:
        response = perform("read_document", {"doc_id": identifier})
        if not response.get("ok"):
            raise RuntimeError("teacher evidence access failed")
    calculated = perform("calculate", {"expression": gold["expression"]})
    if not calculated.get("ok"):
        raise RuntimeError("teacher arithmetic failed")
    plan = perform("solve", {"constraints": copy.deepcopy(gold["constraints"])})
    if not plan.get("ok"):
        raise RuntimeError("teacher explicit solve failed")
    perform("submit", dict(
        plan_id=plan["plan_id"], citations=citations,
        metric=dict(
            value=calculated["value"], period=gold["metric_period"],
            unit=gold["metric_unit"], triggered=gold["triggered"],
        ),
    ))
    if not env.done or not env.success or env.calls_used > 10:
        raise RuntimeError(f"teacher failed {task['task_id']}: {env.result()}")


def build_teacher(output):
    root = Path(output).resolve()
    target = root / "teacher"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists():
        return json.loads((target / "summary.json").read_text())
    counts, adaptation_counts, types = Counter(), Counter(), Counter()
    with (
        (target / "train_requests.jsonl").open("x") as requests,
        (target / "adapter_requests.jsonl").open("x") as adaptation,
        (target / "train_trajectories.jsonl").open("x") as trajectories,
    ):
        for task_index, task in enumerate(load_tasks(root, "train")):
            env = FinancialEvidenceEnv(task)
            messages = env.initial_messages()

            def perform(name, arguments, *, task=task, env=env, messages=messages,
                        task_index=task_index):
                completion = canonical(dict(name=name, arguments=arguments))
                record = dict(
                    id=f"{task['task_id']}/call{env.calls_used + 1}",
                    task_id=task["task_id"], messages=copy.deepcopy(messages),
                    completion=completion,
                    metadata=dict(
                        source_split="train", profile=PROFILE,
                        teacher_used_gold_to_choose_steps=True,
                        prefix_contains_only_actual_tool_observations=True,
                        call_index=env.calls_used + 1, tool_name=name,
                    ),
                )
                requests.write(canonical(record) + "\n")
                if task_index < 64:
                    adaptation.write(canonical(record) + "\n")
                    adaptation_counts["requests"] += 1
                messages.append(dict(role="assistant", content=completion))
                response = env.step(completion)
                messages.append(dict(role="tool", content=canonical(response)))
                counts["requests"] += 1
                return response

            teacher_actions(env, task, perform)
            trajectories.write(canonical(env.result()) + "\n")
            counts["tasks"] += 1
            counts["successes"] += int(env.success)
            types[task["task_type"]] += 1
            if task_index < 64:
                adaptation_counts["tasks"] += 1
                adaptation_counts["successes"] += int(env.success)
            if counts["tasks"] % 100 == 0:
                requests.flush()
                adaptation.flush()
                trajectories.flush()
                print(canonical(dict(stage="TEACHER", counts=dict(counts))), flush=True)
    summary = dict(
        status="COMPLETE", profile=PROFILE, source_split="train",
        counts=dict(counts), adaptation_counts=dict(adaptation_counts),
        task_type_counts=dict(types), all_tasks_successful=counts["successes"] == 800,
        requests_path=str(target / "train_requests.jsonl"),
        adapter_requests_path=str(target / "adapter_requests.jsonl"),
        trajectories_path=str(target / "train_trajectories.jsonl"),
        adaptation_task_selection="first64 fixed Train tasks, 32 complete source-report pairs",
        adaptation_examples_are_subset_of_full_sft_teacher=True,
        full_sft_is_independent_learning_arm_not_rl_initialization=True,
        maximum_tool_calls=10, actual_calls_per_teacher=6,
        labels="one raw tool JSON completion plus EOS per actual production prefix",
        financial_gold_programs_never_in_observations=True,
        reserved_test_opened=False, formal_valid_access=False, formal_final_oos_access=False,
    )
    write_json(target / "summary.json", summary)
    return summary


def oracle_rollout(task):
    env = FinancialEvidenceEnv(task)

    def perform(name, arguments):
        response = env.step(dict(name=name, arguments=arguments))
        if not response.get("ok"):
            raise RuntimeError(f"oracle tool failure: {response}")
        return response

    teacher_actions(env, task, perform)
    result = env.result()
    result["policy"] = "privileged_full_information_oracle"
    result["privileged_gold_selection"] = True
    return result


def aggregate(rows):
    successes = [r for r in rows if r.get("success")]
    correct_feasible = [
        r for r in successes if (r.get("verification") or {}).get("business_feasible")
    ]
    return dict(
        tasks=len(rows), successes=len(successes), success_rate=len(successes) / len(rows),
        business_correct=sum(
            (r.get("verification") or {}).get("business_correct") is True for r in rows
        ),
        portfolio_correct=sum(
            (r.get("verification") or {}).get("portfolio_correct") is True for r in rows
        ),
        metric_correct=sum(
            (r.get("verification") or {}).get("metric_correct") is True for r in rows
        ),
        period_correct=sum(
            (r.get("verification") or {}).get("period_correct") is True for r in rows
        ),
        unit_correct=sum(
            (r.get("verification") or {}).get("unit_correct") is True for r in rows
        ),
        condition_correct=sum(
            (r.get("verification") or {}).get("condition_correct") is True for r in rows
        ),
        evidence_valid=sum(
            (r.get("verification") or {}).get("evidence_valid") is True for r in rows
        ),
        mean_tool_calls=sum(r["calls_used"] for r in rows) / len(rows),
        successful_feasible_tasks=len(correct_feasible),
        mean_cost_successful_feasible=sum(
            r["metrics"]["transaction_cost"] for r in correct_feasible
        ) / len(correct_feasible) if correct_feasible else None,
        task_types={
            kind: dict(tasks=sum(r["task_type"] == kind for r in rows),
                       successes=sum(r["task_type"] == kind and r["success"] for r in rows))
            for kind in sorted({r["task_type"] for r in rows})
        },
    )


def build_baselines(output, *, split="development"):
    root = Path(output).resolve()
    label = "dev" if split == "development" else "test"
    target = root / "baselines"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{label}_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    tasks = load_tasks(root, split, allow_frozen_test=split == "test")
    results = [oracle_rollout(task) for task in tasks]
    trajectory_path = target / f"{label}_trajectories.jsonl"
    trajectory_path.write_text("".join(canonical(r) + "\n" for r in results))
    summary = dict(
        status="COMPLETE", profile=PROFILE, source_split=split,
        policies=dict(privileged_full_information_oracle=aggregate(results)),
        privileged_oracle_marked=True,
        oracle_has_program_and_answer_not_a_fair_retrieval_method=True,
        fair_retrieval_and_correct_fact_agent_diagnostics_run_on_gpu_separately=True,
        reserved_test_opened=split == "test",
        trajectory_path=str(trajectory_path), formal_valid_access=False,
        formal_final_oos_access=False,
    )
    if split == "test":
        summary["frozen_models_receipt"] = str(require_frozen_models(root))
    write_json(path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--part", choices=("all", "teacher", "baselines", "frozen-test-baselines"),
                        default="all")
    args = parser.parse_args()
    if args.part in ("all", "teacher"):
        value = build_teacher(args.output)
        print(canonical(dict(stage="TEACHER_READY", counts=value["counts"])), flush=True)
    if args.part in ("all", "baselines", "frozen-test-baselines"):
        value = build_baselines(
            args.output, split="test" if args.part == "frozen-test-baselines" else "development"
        )
        print(canonical(dict(stage="BASELINES_READY", source_split=value["source_split"],
                             policies=value["policies"])), flush=True)


if __name__ == "__main__":
    main()
