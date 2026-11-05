"""Check fixed holdout denominators and the shared experiment resource boundary."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "financial_transfer_test", DIRECTORY / "run_financial_transfer_probe.py"
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_report_clustering_keeps_threshold_versions_and_replicas_together():
    tasks = [
        {"task_id": task, "source": {"report_id": report}}
        for report in ("a", "b")
        for task in (report + "v0", report + "v1")
    ]
    reference = [
        {
            "task_id": task["task_id"],
            "replicate_index": replica,
            "status": "COMPLETE",
            "success": False,
        }
        for task in tasks
        for replica in range(2)
    ]
    candidate = [{**row, "success": row["task_id"].startswith("a")} for row in reference]
    result = PROBE.paired_report_difference(tasks, reference, candidate)
    assert result["paired_trajectories"] == 8
    assert result["report_clusters"] == 2
    assert result["success_difference_high_rl_minus_sft"] == 0.5
    assert result["report_differences"] == {"a": 1.0, "b": 0.0}
    with pytest.raises(ValueError, match="denominator"):
        PROBE.paired_report_difference(tasks, reference, candidate[:-1])
    with pytest.raises(ValueError, match="denominator"):
        PROBE.paired_report_difference(tasks, reference, candidate + candidate[:1])


def test_new_probe_must_fit_inside_the_original_combined_cap():
    assert PROBE.combined_allowance(37_000_000, 2_000_000) == 5_000_000
    with pytest.raises(PROBE.BudgetStop, match="reserve both"):
        PROBE.combined_allowance(40_000_000, 0)
    with pytest.raises(PROBE.BudgetStop, match="exceeded"):
        PROBE.combined_allowance(37_000_000, PROBE.PROBE_CAP + 1)


def test_failed_sampling_attempts_remain_in_the_combined_bill(tmp_path):
    active = tmp_path / "eval/sft/sample"
    failed = tmp_path / "eval/sft/failed_attempts/sample_previous"
    for path, prompt, response in ((active, 3, 2), (failed, 7, 1)):
        path.mkdir(parents=True)
        (path / "requests.jsonl").write_text(
            json.dumps({"prompt_tokens": prompt, "response_tokens": response}) + "\n"
        )
    controller = PROBE.Controller.__new__(PROBE.Controller)
    controller.output = tmp_path
    controller.parent_costs = {"total_new_generation_tokens": 37_000_000, "gpu_stage_seconds": 100}
    controller.state = {"stages": [{"seconds": 20, "gpu_worker": True}]}
    costs = controller.costs()
    assert costs["generation_tokens_by_model"]["sft"] == 13
    assert costs["diagnostic_generation_tokens"] == 13
    assert costs["total_new_generation_tokens"] == 37_000_013
    assert costs["gpu_stage_seconds"] == 120
