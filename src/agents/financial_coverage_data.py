"""CPU-only fixed Train preparation for the lean V3 coverage experiment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.agents.financial_learning_data import (
    paired_questions,
    question_key,
    read_rows,
    report_key,
    sha,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
CREDIT = ROOT / "runs/m3-financial-credit-20261030-v1"
PARENT = ROOT / "runs/agentic-rl-learning-20261030-v2"
DEFAULT = ROOT / "runs/agentic-rl-coverage-20261105-v3"
SEED = 20261105
ROLLOUT_SCOPES = (
    PARENT,
    ROOT / "runs/agentic-rl-mc-20261030-v1",
    CREDIT / "probe/parents",
)


def select_reports(tasks, seen_reports, holdout, *, seed=SEED, report_count=24):
    """Select by metadata only; outcomes and gold never affect the ordering."""
    pairs = paired_questions(tasks)
    by_report = {}
    for pair in pairs.values():
        report = report_key(pair[0])
        if report in by_report:
            raise ValueError("coverage pool requires one base question per report")
        by_report[report] = pair
    hold_reports = {report_key(task) for task in holdout}
    hold_sources = {question_key(task) for task in holdout}
    order = sorted(set(by_report) - set(seen_reports))
    candidate_count = len(order)
    random.Random(seed).shuffle(order)
    selected, excluded = [], []
    for report in order:
        pair = by_report[report]
        documents = {
            document.get("source_id")
            for task in pair
            for document in task.get("documents", [])
        }
        if report in hold_reports or documents & hold_sources:
            excluded.append(report)
            continue
        if len(selected) < report_count:
            selected.append(report)
    if len(selected) != report_count:
        raise ValueError(f"need {report_count} eligible reports, found {len(selected)}")
    return dict(
        tasks=[task for report in selected for task in by_report[report]],
        reports=selected,
        candidate_count=candidate_count,
        excluded_direct_holdout_overlap=excluded,
    )


def prepare(output=DEFAULT):
    output = Path(output).resolve()
    pool_path = CREDIT / "data/train.jsonl"
    holdout_path = PARENT / "data/holdout.jsonl"
    original_train = ROOT / "runs/m3-financial-evidence-20261030-v2/data/train.jsonl"
    tasks, holdout = read_rows(pool_path), read_rows(holdout_path)
    paired_questions(holdout)
    if len(tasks) != 256 or len(holdout) != 32:
        raise ValueError("fixed 256-task Train pool and 32-task old holdout required")
    task_index = {task["task_id"]: task for task in tasks}
    seen, rollout_files = set(), []
    for scope in ROLLOUT_SCOPES:
        for path in sorted(scope.rglob("trajectory_results.jsonl")):
            rollout_files.append(path)
            for record in read_rows(path):
                if record.get("task_id") in task_index:
                    seen.add(report_key(task_index[record["task_id"]]))
    chosen = select_reports(tasks, seen, holdout)
    if chosen["candidate_count"] != 66:
        raise ValueError("recent-rollout inventory differs from the planned 66 reports")
    data = output / "data"
    panel = data / "train_panel.jsonl"
    write_rows(panel, chosen["tasks"])
    batches = []
    for index in range(6):
        batch = chosen["tasks"][index * 8 : (index + 1) * 8]
        path = data / f"batch_round{index:03d}.jsonl"
        write_rows(path, batch)
        batches.append(dict(
            round=index,
            path=str(path),
            sha256=sha(path),
            tasks=len(batch),
            reports=chosen["reports"][index * 4 : (index + 1) * 4],
            parent_trajectories=len(batch) * 4,
        ))
    hold_reports = {report_key(task) for task in holdout}
    available_reports = {
        "/".join((document.get("source_id") or "").split("/")[:2])
        for task in chosen["tasks"]
        for document in task.get("documents", [])
    }
    old_reports = {report_key(task) for task in read_rows(original_train)}
    manifest = dict(
        profile="financial_coverage_lean_v3",
        status="FROZEN",
        seed=SEED,
        selection="sorted report IDs shuffled once; no correctness or gold selection",
        train_panel_path=str(panel),
        holdout_path=str(holdout_path),
        selected_reports=chosen["reports"],
        selected_sources=[question_key(task) for task in chosen["tasks"][::2]],
        batches=batches,
        max_batches=6,
        train_panel_tasks=48,
        train_panel_reports=24,
        repetitions=4,
        planned_parent_trajectories=192,
        holdout_tasks=32,
        holdout_reports=len(hold_reports),
        evaluation_repetitions=2,
        evaluation_seed=145,
        planned_evaluation_trajectories=64,
        source_sha256={str(path): sha(path) for path in (
            pool_path, holdout_path, original_train, *rollout_files
        )},
        counts=dict(
            pool_tasks=len(tasks),
            pool_reports=len({report_key(task) for task in tasks}),
            reports_with_recent_rollout=len(seen),
            candidate_reports_without_recent_rollout=chosen["candidate_count"],
            excluded_direct_holdout_overlap=chosen["excluded_direct_holdout_overlap"],
        ),
        exposure=dict(
            selected_reports_seen_as_initial_sft_target=sorted(
                set(chosen["reports"]) & old_reports
            ),
            holdout_target_reports_in_training_documents=sorted(hold_reports & available_reports),
            target_report_overlap=False,
            holdout_target_source_id_in_training_documents=False,
            not_a_claim_of_initial_sft_or_pretraining_unseen=True,
            rollout_scope_paths=[str(path) for path in ROLLOUT_SCOPES],
        ),
        reserved_test_read=False,
        historical_market_oos_read=False,
    )
    write_json(data / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = prepare(args.output)
    print(f"Prepared {result['train_panel_reports']} reports, 6 batches, 192 parents")


if __name__ == "__main__":
    main()
