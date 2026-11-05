"""Ensure lean coverage preparation keeps paired tasks and excludes direct leakage."""

import pytest

from src.agents.financial_coverage_data import select_reports


def task_pair(report, documents=()):
    return [
        dict(
            split="train", task_id=f"{report}-{variant}", variant=variant,
            source={"report_id": report, "finqa_id": report + "/question"},
            documents=[{"source_id": source} for source in documents],
        )
        for variant in (0, 1)
    ]


def test_selection_is_metadata_only_and_keeps_threshold_pair_together():
    tasks = [task for i in range(8) for task in task_pair(f"r{i}")]
    result = select_reports(tasks, {"r0"}, task_pair("hold"), report_count=4)
    changed = [{**task, "gold": {"answer": 123}, "success": True} for task in tasks]
    repeated = select_reports(changed, {"r0"}, task_pair("hold"), report_count=4)
    assert repeated["reports"] == result["reports"]
    assert "r0" not in result["reports"]
    assert len(result["tasks"]) == 8
    for index in range(0, 8, 2):
        a, b = result["tasks"][index:index + 2]
        assert a["source"]["report_id"] == b["source"]["report_id"]
        assert (a["variant"], b["variant"]) == (0, 1)


def test_direct_holdout_source_in_distractor_is_skipped_before_selection():
    tasks = task_pair("bad", ["hold/question"]) + task_pair("good") + task_pair("hold")
    result = select_reports(tasks, set(), task_pair("hold"), report_count=1)
    assert result["reports"] == ["good"]
    assert set(result["excluded_direct_holdout_overlap"]) == {"bad", "hold"}
    with pytest.raises(ValueError, match="eligible reports"):
        select_reports(tasks, set(), task_pair("hold"), report_count=2)
