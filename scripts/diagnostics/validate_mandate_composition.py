#!/usr/bin/env python3
"""Small CPU wiring checks on Train/Dev only, parallel to real model work."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.agents.mandate_repair_env import MandateRepairEnv, canonical


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "data/manifest.json").read_text())
    checks, details = {}, {}
    expected = dict(train=2000, sft_train=800, rl_train=1200, development=200, test=200)
    checks["manifest_fixed_task_counts"] = manifest.get("counts") == expected
    groups = {k: set(v) for k, v in manifest["base_scenario_groups"].items()}
    checks["base_groups_disjoint_across_splits"] = not (
        groups["train"] & groups["development"]
        or groups["train"] & groups["test"]
        or groups["development"] & groups["test"]
    )
    checks["sft_and_rl_base_groups_disjoint_cover_train"] = (
        not groups["sft_train"] & groups["rl_train"]
        and groups["sft_train"] | groups["rl_train"] == groups["train"]
    )
    data = {
        split: rows(root / "data" / f"{split}.jsonl")
        for split in ("sft_train", "rl_train", "development")
    }
    checks["train_dev_actual_task_counts"] = all(
        len(values) == expected[split] for split, values in data.items()
    )
    checks["train_all_industry_cap_cardinalities_present"] = all(
        {0, 1, 2, 3}.issubset({len(t["gold"]["constraints"]["sector_max"]) for t in values})
        for split, values in data.items()
        if split in {"sft_train", "rl_train"}
    )
    checks["no_duplicate_train_dev_task_ids"] = len(
        {t["task_id"] for values in data.values() for t in values}
    ) == sum(map(len, data.values()))
    bad_public, bad_catalog = [], []
    sample_teachers = {}
    for split, values in data.items():
        for task in values:
            env = MandateRepairEnv(task)
            visible = env.visible()
            public_text = " ".join(message["content"] for message in env.initial_messages())
            if any(
                marker in public_text
                for marker in (
                    '"gold"',
                    '"oracle_plan"',
                    '"base_scenario_id"',
                    '"task_type"',
                    '"training_portion"',
                    '"pair_id"',
                    '"superseded"',
                    '"kind"',
                )
            ):
                bad_public.append(task["task_id"])
            if not all(
                set(d) == {"doc_id", "title", "available_at", "effective_at"}
                for d in visible["document_catalog"]
            ):
                bad_catalog.append(task["task_id"])
            if split == "sft_train":
                sample_teachers.setdefault(len(task["gold"]["required_citations"]), task)
    checks["public_initial_observations_hide_gold_and_group_metadata"] = not bad_public
    checks["public_catalog_hides_document_kind_and_superseded_flag"] = not bad_catalog
    details["public_observation_checked_tasks"] = sum(map(len, data.values()))
    details["public_observation_failures"] = bad_public[:10]
    details["catalog_metadata_failures"] = bad_catalog[:10]
    for split in ("sft_train", "rl_train"):
        industry_counts = Counter(
            sector for t in data[split] for sector in t["gold"]["constraints"]["sector_max"]
        )
        checks[f"{split}_all_five_industries_present"] = len(industry_counts) == 5
        details[f"{split}_industry_counts"] = dict(industry_counts)
    details["business_sensitive_pairs_train_dev"] = {
        split: manifest["business_sensitive_pair_counts"][split]
        for split in ("train", "development")
    }
    checks["train_dev_include_business_sensitive_pairs"] = all(
        count > 0 for count in details["business_sensitive_pairs_train_dev"].values()
    )
    # Every label is an actual one-tool response; inspect the complete teacher
    # file once and replay only one route at each required-document cardinality.
    teacher_file = root / "teacher/train_requests.jsonl"
    teacher_summary = root / "teacher/summary.json"
    if teacher_summary.exists():
        count, failures = 0, []
        replay = {t["task_id"]: [] for t in sample_teachers.values()}
        with teacher_file.open() as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                text = " ".join(message["content"] for message in row["messages"])
                if '"gold"' in text or '"oracle_plan"' in text:
                    failures.append(row["id"])
                if row["task_id"] in replay:
                    replay[row["task_id"]].append(row)
        checks["all_teacher_prefixes_hide_gold"] = not failures
        details["teacher_requests_checked"] = count
        replays = []
        for task in sample_teachers.values():
            env, messages = MandateRepairEnv(task), MandateRepairEnv(task).initial_messages()
            prefix_equal = True
            for row in replay[task["task_id"]]:
                prefix_equal &= row["messages"] == messages
                response = env.step(row["completion"])
                messages.extend(
                    [
                        dict(role="assistant", content=row["completion"]),
                        dict(role="tool", content=canonical(response)),
                    ]
                )
            replays.append(
                dict(
                    task_id=task["task_id"],
                    prefix_equal=prefix_equal,
                    success=env.success,
                    calls=env.calls_used,
                    required_documents=len(task["gold"]["required_citations"]),
                )
            )
        checks["teacher_sample_prefixes_equal_real_tool_replay"] = all(
            r["prefix_equal"] and r["success"] and r["calls"] <= 8 for r in replays
        )
        details["teacher_route_replays"] = replays
    else:
        details["teacher_check_status"] = "PENDING_TEACHER_COMPLETE"
    result = dict(
        status="PASSED" if all(checks.values()) else "FAILED",
        checks=checks,
        details=details,
        reserved_test_tasks_or_results_opened=False,
        test_manifest_group_metadata_only=True,
        cpu_only=True,
        actual_test_evaluation_performed=False,
    )
    (root / "composition_validation_checks.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(dict(status=result["status"], passed=sum(checks.values()), total=len(checks))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if validate(args.output)["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
