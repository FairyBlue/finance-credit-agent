"""Recover missing fixed Dev outcomes without resampling or double billing."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "financial_recovery_test", DIRECTORY / "resume_financial_learning.py"
)
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


@pytest.fixture
def sources():
    tasks = [{"task_id": f"task-{i:03d}"} for i in range(100)]
    outcomes = [
        dict(task_id=task["task_id"], replicate_index=0, status="COMPLETE", success=i % 2 == 0)
        for i, task in enumerate(tasks)
    ]
    identity = {key: "fixed-identity" for key in RECOVERY.IDENTITY_FIELDS}
    original = dict(identity, planned_trajectories=100, complete_trajectories=95, total_tokens=950)
    additional = dict(identity, planned_trajectories=5, complete_trajectories=5, total_tokens=50)
    return tasks, original, outcomes[:95], additional, outcomes[95:]


def test_merge_is_exact_disjoint_union_and_keeps_original_records(sources):
    tasks, old_summary, old, new_summary, new = sources
    before = json.dumps(old)
    merged, results = RECOVERY.merge_results(tasks, old_summary, old, new_summary, new)
    assert len(results) == 100
    assert len({item["task_id"] for item in results}) == 100
    assert [item["task_id"] for item in results] == [task["task_id"] for task in tasks]
    assert json.dumps(old) == before
    assert merged["successes"] == 50
    assert merged["total_tokens"] == 1000
    assert merged["new_recovery_generation_tokens"] == 50
    assert merged["generation_billing_source"] is False


def test_merge_rejects_resampling_a_completed_task(sources):
    tasks, old_summary, old, new_summary, new = sources
    new[0] = dict(old[0])
    with pytest.raises(RuntimeError, match="previously unstarted"):
        RECOVERY.merge_results(tasks, old_summary, old, new_summary, new)


def test_merge_rejects_changed_checkpoint_or_sampling_identity(sources):
    tasks, old_summary, old, new_summary, new = sources
    new_summary["adapter_sha256"] = "different-weights"
    with pytest.raises(RuntimeError, match="adapter_sha256"):
        RECOVERY.merge_results(tasks, old_summary, old, new_summary, new)


def test_missing_ids_reject_unknown_original_outcomes(sources):
    tasks, _, old, _, _ = sources
    old[0] = dict(old[0], task_id="outside-original-dev")
    with pytest.raises(RuntimeError, match="strict subset"):
        RECOVERY.missing_task_ids(tasks, old)


def test_derived_view_does_not_charge_original_or_recovery_tokens_twice(tmp_path, sources):
    tasks, old_summary, old, new_summary, new = sources
    merged, results = RECOVERY.merge_results(tasks, old_summary, old, new_summary, new)
    for directory, amount in (("original", 950), (RECOVERY.RECOVERY_NAME, 50)):
        path = tmp_path / "eval" / directory
        path.mkdir(parents=True)
        (path / "requests.jsonl").write_text(
            json.dumps(dict(prompt_tokens=amount, response_tokens=0)) + "\n"
        )
    controller = RECOVERY.Controller.__new__(RECOVERY.Controller)
    controller.output = tmp_path
    controller.state = {"stages": []}
    before = controller.costs()
    view = tmp_path / "eval_views/continued_sft_dev"
    view.mkdir(parents=True)
    (view / "summary.json").write_text(json.dumps(merged))
    (view / "trajectory_results.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in results)
    )
    assert controller.costs() == before
    assert controller.costs()["evaluation_generation_tokens"] == 1000
    assert RECOVERY.generation_tokens(view) == 0


def test_final_step3_review_prefers_complete_view_and_skips_raw_recovery_subset(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "financial_step3_view_test", DIRECTORY / "run_financial_learning_step3.py"
    )
    step3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step3)
    summary = dict(
        status="BUDGET_EXHAUSTED",
        planned_trajectories=100,
        complete_trajectories=95,
        successes=78,
        total_tokens=950,
        outcome_diagnostics=dict(metric_correct=78),
    )
    original = tmp_path / "eval/continued_sft_dev/summary.json"
    original.parent.mkdir(parents=True)
    original.write_text(json.dumps(summary))
    retained = original.read_bytes()
    view = tmp_path / "eval_views/continued_sft_dev/summary.json"
    view.parent.mkdir(parents=True)
    view.write_text(
        json.dumps(
            dict(
                summary,
                status="COMPLETE",
                complete_trajectories=100,
                successes=82,
                outcome_diagnostics=dict(metric_correct=82),
            )
        )
    )
    raw = tmp_path / "eval" / RECOVERY.RECOVERY_NAME / "summary.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(dict(summary, planned_trajectories=5, complete_trajectories=5, successes=4))
    )
    data = tmp_path / "data/train_panel.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"task_id":"train-only"}\n')
    (tmp_path / "state.json").write_text(json.dumps(dict(status="RUNNING", stages=[])))
    controller = step3.Controller.__new__(step3.Controller)
    controller.root = tmp_path
    controller.output = tmp_path / "step3"
    controller.output.mkdir()
    controller.path = controller.output / "state.json"
    controller.state = dict(stages=[], arms={}, gate={})
    controller.costs = lambda: dict(total_new_generation_tokens=1000)
    controller.report = lambda: None
    controller.finish("UNRESOLVED_KEY_DECISION_SIGNAL")
    review = (tmp_path / "RESEARCH_REVIEW.md").read_text()
    assert "| 第二步/continued_sft_dev | 82/100 | 82 |" in review
    assert RECOVERY.RECOVERY_NAME not in review
    assert original.read_bytes() == retained


def test_missing_optional_fingerprint_is_recomputed_from_the_actual_artifacts(tmp_path, sources):
    from src.agents.mandate_repair_collect import _evaluation_fingerprint

    tasks, old_summary, old, new_summary, new = sources
    adapter = tmp_path / "fixed_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"target_modules":["q_proj","v_proj"]}')
    (adapter / "adapter_model.safetensors").write_bytes(b"same exact frozen weights")
    old_summary["adapter_path"] = str(adapter)
    new_summary["adapter_path"] = str(adapter)
    old_summary["evaluation_fingerprint"] = _evaluation_fingerprint(adapter)
    new_summary.pop("evaluation_fingerprint")
    before = json.dumps(new_summary)
    merged, _ = RECOVERY.merge_results(tasks, old_summary, old, new_summary, new)
    inferred = merged["recovered_identity_fields"]["evaluation_fingerprint"]
    assert inferred["value"] == old_summary["evaluation_fingerprint"]
    assert inferred["raw_summary_modified"] is False
    assert json.dumps(new_summary) == before
    (adapter / "adapter_model.safetensors").write_bytes(b"changed weights must be rejected")
    with pytest.raises(RuntimeError, match="evaluation_fingerprint"):
        RECOVERY.merge_results(tasks, old_summary, old, new_summary, new)
