"""The diagnostic panel must retain pairs and isolate report-level holdout."""

from src.agents.financial_learning_data import paired_questions, select_panel


def task(source, report, variant, task_id):
    return dict(
        split="train",
        task_id=task_id,
        variant=variant,
        source=dict(finqa_id=source, report_id=report),
    )


def test_selection_uses_source_metadata_keeps_pairs_and_disjoint_reports():
    tasks = [task("question-A", "report-X", variant, f"opaque-A-{variant}") for variant in (0, 1)]
    tasks += [task("question-B", "report-X", variant, f"opaque-B-{variant}") for variant in (0, 1)]
    tasks += [task("question-C", "report-Y", variant, f"opaque-C-{variant}") for variant in (0, 1)]
    tasks += [task("question-D", "report-Z", variant, f"opaque-D-{variant}") for variant in (0, 1)]
    probe = [dict(task_id="opaque-B-1", success=index == 0) for index in range(4)]
    result = select_panel(tasks, probe, panel_size=1, holdout_size=3)
    assert result["selected_sources"] == ["question-B"]
    assert {row["task_id"] for row in result["panel"]} == {"opaque-B-0", "opaque-B-1"}
    assert set(result["holdout_reports"]) == {"report-Y", "report-Z"}
    assert len(result["holdout"]) == 4
    assert all(row["source"]["report_id"] != "report-X" for row in result["holdout"])
    assert result == select_panel(tasks, probe, panel_size=1, holdout_size=3)


def test_insufficient_mixed_questions_get_one_fixed_random_fill():
    tasks = [
        task(f"Q{index}", f"R{index}", variant, f"opaque-{index}-{variant}")
        for index in range(4)
        for variant in (0, 1)
    ]
    result = select_panel(tasks, [], panel_size=2, holdout_size=2)
    assert result["mixed_selected_sources"] == []
    assert len(result["random_fill_sources"]) == 2
    assert len(result["panel"]) == len(result["holdout"]) == 4
    assert set(result["selected_reports"]).isdisjoint(result["holdout_reports"])


def test_bad_threshold_pair_is_rejected_before_freezing():
    import pytest

    with pytest.raises(ValueError, match="two threshold variants"):
        paired_questions([task("Q", "R", 0, "opaque")])
    with pytest.raises(ValueError, match="Train tasks"):
        paired_questions([dict(task("Q", "R", 0, "opaque"), split="test")])
