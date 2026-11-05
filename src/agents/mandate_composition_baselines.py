"""Composition-task teachers and strong same-tools public-document baselines.

Teacher gold selects Train labels only; public parsing accepts returned text,
never a task/gold object. Test evaluation requires completed frozen endpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.agents.mandate_repair_baselines import aggregate, privileged_oracle, require_frozen_models
from src.agents.mandate_repair_env import MandateRepairEnv, canonical

PROFILE = "mandate_composition_teachers_public_baselines_v2"
EXPECTED = dict(train=2000, sft_train=800, rl_train=1200, development=200, test=200)


def load_tasks(root, split, *, allow_frozen_test=False):
    if split not in EXPECTED:
        raise ValueError("unknown composition task partition")
    if split == "test":
        if not allow_frozen_test:
            raise PermissionError(
                "reserved test requires explicit frozen-test-baselines entry point"
            )
        require_frozen_models(root)
    tasks = [json.loads(line) for line in (Path(root) / "data" / f"{split}.jsonl").open()]
    if len(tasks) != EXPECTED[split]:
        raise ValueError(f"fixed task count mismatch for {split}")
    return tasks


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def teacher_actions(env, task, perform):
    perform("search_docs", {"query": "current mandate restriction classification allocation"})
    for identifier in task["gold"]["required_citations"]:
        result = perform("read_document", {"doc_id": identifier})
        if not result.get("ok"):
            raise RuntimeError("teacher failed to retrieve evidence")
    constraints = copy.deepcopy(task["gold"]["constraints"])
    if not constraints["sector_max"]:
        constraints["sectors"] = {}
    perform("calculate", {"expression": f"{constraints['cash_min']} - cash"})
    result = perform("solve", {"constraints": constraints})
    if not result.get("ok"):
        raise RuntimeError("teacher explicit solve failed")
    perform(
        "submit", {"plan_id": result["plan_id"], "citations": task["gold"]["required_citations"]}
    )
    if not env.done or not env.success or env.calls_used > 8:
        raise RuntimeError(f"teacher task failed: {task['task_id']}")


def build_teacher(root):
    root = Path(root).resolve()
    target = root / "teacher"
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    counts, types, cap_counts = Counter(), Counter(), Counter()
    with (
        (target / "train_requests.jsonl").open("x") as requests,
        (target / "train_trajectories.jsonl").open("x") as trajectories,
    ):
        for task in load_tasks(root, "sft_train"):
            env = MandateRepairEnv(task)
            messages = env.initial_messages()

            def perform(name, arguments, *, task=task, env=env, messages=messages):
                completion = canonical(dict(name=name, arguments=arguments))
                record = dict(
                    id=f"{task['task_id']}/call{env.calls_used + 1}",
                    task_id=task["task_id"],
                    messages=copy.deepcopy(messages),
                    completion=completion,
                    metadata=dict(
                        source_split="train",
                        source_portion="sft_train",
                        profile=PROFILE,
                        teacher_used_gold_to_choose_steps=True,
                        prefix_contains_only_actual_tool_observations=True,
                        call_index=env.calls_used + 1,
                        tool_name=name,
                    ),
                )
                requests.write(canonical(record) + "\n")
                messages.append(dict(role="assistant", content=completion))
                result = env.step(completion)
                messages.append(dict(role="tool", content=canonical(result)))
                counts["requests"] += 1
                return result

            teacher_actions(env, task, perform)
            trajectories.write(canonical(env.result()) + "\n")
            counts["tasks"] += 1
            counts["successes"] += int(env.success)
            types[task["task_type"]] += 1
            cap_counts[len(task["gold"]["constraints"]["sector_max"])] += 1
            if counts["tasks"] % 100 == 0:
                requests.flush()
                trajectories.flush()
                print(canonical(dict(stage="TEACHER", counts=dict(counts))), flush=True)
    summary = dict(
        status="COMPLETE",
        profile=PROFILE,
        source_split="train",
        source_portion="sft_train",
        counts=dict(counts),
        task_type_counts=dict(types),
        sector_cap_cardinalities=dict(cap_counts),
        all_tasks_successful=counts["tasks"] == counts["successes"],
        requests_path=str(target / "train_requests.jsonl"),
        trajectories_path=str(target / "train_trajectories.jsonl"),
        context_roles=["system", "user", "assistant", "tool"],
        gold_selection=(
            "Train-only teacher labels; actual tool observations are the only prefix information"
        ),
        rl_task_groups_not_used_for_teacher=True,
        maximum_tool_calls=8,
        reserved_test_opened=False,
        formal_valid_access=False,
        formal_final_oos_access=False,
    )
    write_json(summary_path, summary)
    print(canonical(dict(stage="TEACHER_READY", **summary)), flush=True)
    return summary


def _first(text, patterns):
    for pattern in patterns:
        result = re.search(pattern, text)
        if result:
            return result.group(1)
    raise ValueError("required public numeric clause not found")


def parse_public_documents(mandate, restrictions, classification=None, allocation=None):
    """Parse public returned natural-language text without access to task gold."""
    text = mandate["text"]
    cash = _first(
        text,
        (
            r"Cash available must be at least USD ([\d,.]+)\.",
            r"Retain at least USD ([\d,.]+) in cash\.",
            r"The closing cash balance must be no less than USD ([\d,.]+)\.",
            r"Cash floor after fees: USD ([\d,.]+)\.",
            r"End-of-task cash floor: USD ([\d,.]+)\.",
            r"Available closing cash must reach USD ([\d,.]+) or more\.",
            r"Minimum final cash required: USD ([\d,.]+)\.",
            r"Set aside a final cash balance of at least USD ([\d,.]+)\.",
        ),
    )
    exposure = _first(
        text,
        (
            r"Total stock exposure must not exceed ([\d.]+) percent",
            r"Aggregate equity exposure must be at most ([\d.]+) percent",
            r"Total equity exposure ceiling: ([\d.]+) percent",
            r"Keep combined stock positions within ([\d.]+) percent",
            r"Final aggregate stock exposure is limited to ([\d.]+) percent",
            r"The stock allocation ceiling is ([\d.]+) percent",
            r"At completion all equities together may occupy no more than ([\d.]+) percent",
            r"Total post-cost equity allocation: maximum ([\d.]+) percent",
        ),
    )

    def caps(patterns, value_text):
        values = {}
        for pattern in patterns:
            values.update(
                {name: float(value) / 100 for name, value in re.findall(pattern, value_text)}
            )
        return values

    symbol_caps = caps(
        (
            r"The holding in ([A-Z]+) is capped at ([\d.]+) percent",
            r"Limit ([A-Z]+) to ([\d.]+) percent",
            r"Single-name ceiling for ([A-Z]+): ([\d.]+) percent",
        ),
        text,
    )
    sector_text = text + ("\n" + allocation["text"] if allocation else "")
    sector_caps = caps(
        (
            r"Combined ([A-Za-z]+) holdings may not exceed ([\d.]+) percent",
            r"Cap combined ([A-Za-z]+) exposure at ([\d.]+) percent",
            r"Aggregate industry ceiling for ([A-Za-z]+): ([\d.]+) percent",
        ),
        sector_text,
    )
    sectors = {}
    if sector_caps:
        if classification is None:
            raise ValueError("public industry caps require observed classifications")
        sectors = dict(
            re.findall(
                r"([A-Z]+) belongs to the ([A-Za-z]+) industry bucket\.", classification["text"]
            )
        )
    restriction_text = restrictions["text"]
    sell = set(re.findall(r"Do not sell any shares of ([A-Z]+)\.", restriction_text))
    sell.update(re.findall(r"No sales of ([A-Z]+) are permitted\.", restriction_text))
    sell.update(re.findall(r"([A-Z]+) may not be sold\.", restriction_text))
    buy = set(re.findall(r"Do not buy any shares of ([A-Z]+)\.", restriction_text))
    buy.update(re.findall(r"No purchases of ([A-Z]+) are permitted\.", restriction_text))
    buy.update(re.findall(r"([A-Z]+) may not be bought\.", restriction_text))
    return dict(
        cash_min=float(cash.replace(",", "")),
        exposure_max=float(exposure) / 100,
        symbol_max=symbol_caps,
        sector_max=sector_caps,
        sectors=sectors,
        cannot_sell=sorted(sell),
        cannot_buy=sorted(buy),
    )


def public_rule_rollout(task, *, missing_key_document=False):
    env = MandateRepairEnv(task)

    def perform(name, arguments, **kwargs):
        response = env.step(dict(name=name, arguments=arguments), **kwargs)
        if not response.get("ok"):
            raise ValueError(response.get("error", "public tool failed"))
        return response

    try:
        visible = env.visible()
        as_of = datetime.fromisoformat(visible["as_of"])
        # Full catalog is public in the same initial observation used by actor.
        catalog = [
            d
            for d in visible["document_catalog"]
            if datetime.fromisoformat(d["effective_at"]) <= as_of
        ]
        perform("search_docs", {"query": "current mandate restriction classification allocation"})
        mandates = sorted(
            [d for d in catalog if "mandate" in d["title"].lower()],
            key=lambda d: (d["effective_at"], d["doc_id"]),
            reverse=True,
        )
        chosen = mandates[0]
        mandate = perform(
            "read_document", {"doc_id": chosen["doc_id"]}, intervention=missing_key_document
        )
        if missing_key_document:
            chosen = mandates[1]
            mandate = perform("read_document", {"doc_id": chosen["doc_id"]})
        citations = [chosen["doc_id"]]
        restricted = next(d for d in catalog if "restriction" in d["title"].lower())
        restrictions = perform("read_document", {"doc_id": restricted["doc_id"]})
        citations.append(restricted["doc_id"])
        allocation = None
        if "current allocation schedule" in mandate["text"]:
            item = next(d for d in catalog if "allocation schedule" in d["title"].lower())
            allocation = perform("read_document", {"doc_id": item["doc_id"]})
            citations.append(item["doc_id"])
        sector_text = mandate["text"] + (allocation["text"] if allocation else "")
        classification = None
        if any(
            needle in sector_text
            for needle in ("Combined ", "Cap combined ", "Aggregate industry ceiling")
        ):
            item = next(d for d in catalog if "classification" in d["title"].lower())
            classification = perform("read_document", {"doc_id": item["doc_id"]})
            citations.append(item["doc_id"])
        constraints = parse_public_documents(mandate, restrictions, classification, allocation)
        # Optional arithmetic is unnecessary for four-document cases; both
        # variants stay inside the actor's unchanged eight-tool-call budget.
        if env.calls_used <= 4:
            perform("calculate", {"expression": f"{constraints['cash_min']} - cash"})
        plan = perform("solve", {"constraints": constraints})
        perform("submit", {"plan_id": plan["plan_id"], "citations": citations})
    except (ValueError, KeyError, IndexError, StopIteration, RuntimeError) as error:
        env.done = True
        env.failure_reason = f"RULE_BASELINE_ERROR:{type(error).__name__}:{error}"
    result = env.result()
    result.update(
        policy="no_key_document" if missing_key_document else "public_rule_retrieval",
        policy_used_gold=False,
        information_access=(
            "same visible catalog/text and same eight-call tools; current mandate masked"
        )
        if missing_key_document
        else "same visible catalog/text and same eight-call tools",
    )
    return result


def build_baselines(root, *, split="development", allow_frozen_test=False):
    root = Path(root).resolve()
    if split == "test":
        if not allow_frozen_test:
            raise PermissionError(
                "reserved test requires explicit frozen-test-baselines entry point"
            )
        frozen = require_frozen_models(root)
    elif split == "development":
        frozen = None
    else:
        raise ValueError("baselines evaluate development or explicitly frozen test")
    target = root / "baselines"
    target.mkdir(parents=True, exist_ok=True)
    prefix = "test" if split == "test" else "dev"
    summary_path = target / f"{prefix}_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    groups = {
        p: []
        for p in ("privileged_full_information_oracle", "public_rule_retrieval", "no_key_document")
    }
    with (target / f"{prefix}_trajectories.jsonl").open("x") as stream:
        for task in load_tasks(root, split, allow_frozen_test=allow_frozen_test):
            results = (
                privileged_oracle(task),
                public_rule_rollout(task),
                public_rule_rollout(task, missing_key_document=True),
            )
            for result in results:
                groups[result["policy"]].append(result)
                stream.write(canonical(result) + "\n")
    summary = dict(
        status="COMPLETE",
        profile=PROFILE,
        source_split=split,
        policies={p: aggregate(rows) for p, rows in groups.items()},
        fair_policy_uses_hidden_gold=False,
        privileged_oracle_marked=True,
        reserved_test_opened=split == "test",
        frozen_models_receipt=str(frozen) if frozen else None,
        formal_valid_access=False,
        formal_final_oos_access=False,
        trajectory_path=str(target / f"{prefix}_trajectories.jsonl"),
    )
    write_json(summary_path, summary)
    print(canonical(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "teacher", "baselines"), default="all")
    parser.add_argument("--part", choices=("frozen-test-baselines",))
    args = parser.parse_args()
    if args.part:
        build_baselines(args.output, split="test", allow_frozen_test=True)
        return
    if args.only in ("all", "teacher"):
        build_teacher(args.output)
    if args.only in ("all", "baselines"):
        build_baselines(args.output)


if __name__ == "__main__":
    main()
