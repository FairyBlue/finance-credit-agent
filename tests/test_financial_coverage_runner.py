"""Guard the lean coverage run's resource and fixed-denominator boundaries."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))
SPEC = importlib.util.spec_from_file_location(
    "financial_coverage_runner_for_test", DIAGNOSTICS / "run_financial_coverage.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def controller(tmp_path):
    instance = RUNNER.Controller.__new__(RUNNER.Controller)
    instance.output = tmp_path
    instance.path = tmp_path / "state.json"
    instance.state = {"stages": [], "arms": {}, "recovery_allocation": {}}
    instance.report = lambda: None
    return instance


def write_request(directory, prompt, response):
    directory.mkdir(parents=True)
    (directory / "requests.jsonl").write_text(
        json.dumps({"prompt_tokens": prompt, "response_tokens": response}) + "\n"
    )


def test_failed_sampling_requests_are_charged_to_the_training_cap(controller):
    write_request(controller.output / "training/wide/round_000/collect", 3, 2)
    failed = controller.output / "training/wide/round_000/failed_attempts/collect_old"
    write_request(failed, 7, 1)
    costs = controller.costs()
    assert costs["training_generation_tokens"] == 13
    assert costs["total_new_generation_tokens"] == 13
    assert costs["reused_reference_new_tokens"] == 0
    (failed / "requests.jsonl").write_text(
        json.dumps({"prompt_tokens": RUNNER.TRAIN_CAP, "response_tokens": 1}) + "\n"
    )
    with pytest.raises(RUNNER.BudgetStop, match="including failed requests"):
        controller.costs()


def test_partial_fixed_parent_batch_never_enters_actor_training(controller):
    data = controller.output / "data/batch_round000.jsonl"
    data.parent.mkdir()
    data.write_text("".join(json.dumps({"task_id": f"t{i}"}) + "\n" for i in range(8)))
    controller.collection = lambda *a, **k: (
        {"status": "RESOURCE_LIMIT", "planned_trajectories": 32, "complete_trajectories": 28},
        False,
    )
    controller.stage = lambda *a, **k: pytest.fail("incomplete parent batch must not train")
    with pytest.raises(RUNNER.BudgetStop, match="not trained"):
        controller.train_wide()
    assert not (controller.output / "training/wide/round_000/train").exists()
    assert not controller.state["arms"]


def test_endpoint_rejects_missing_duplicate_or_incomplete_samples():
    tasks = [{"task_id": f"t{i}"} for i in range(32)]
    outcomes = [
        {"task_id": task["task_id"], "replicate_index": replica, "status": "COMPLETE"}
        for task in tasks
        for replica in range(2)
    ]
    assert len(RUNNER.indexed_outcomes(tasks, outcomes)) == 64
    with pytest.raises(ValueError, match="denominator"):
        RUNNER.indexed_outcomes(tasks, outcomes[:-1])
    with pytest.raises(ValueError, match="denominator"):
        RUNNER.indexed_outcomes(tasks, outcomes + outcomes[:1])
    incomplete = [{**row, "status": "FAILED"} if index == 0 else row
                  for index, row in enumerate(outcomes)]
    with pytest.raises(ValueError, match="must be complete"):
        RUNNER.indexed_outcomes(tasks, incomplete)


def test_two_hour_limit_stops_before_any_worker_is_launched(controller, monkeypatch):
    controller.state["stages"] = [{"seconds": 7200, "gpu_worker": True}]
    monkeypatch.setattr(
        RUNNER.subprocess, "Popen", lambda *a, **k: pytest.fail("2-hour cap must stop worker")
    )
    with pytest.raises(RUNNER.BudgetStop, match="2-hour"):
        controller.stage("forbidden_after_cap", RUNNER.COLLECTOR, [])
    assert len(controller.state["stages"]) == 1
