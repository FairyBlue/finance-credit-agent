"""Resource boundaries and direct-training control flow for the action-Q run."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "financial_action_q_runner_test", DIRECTORY / "run_financial_action_q.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def controller(tmp_path):
    instance = RUNNER.Controller.__new__(RUNNER.Controller)
    instance.output = tmp_path
    instance.path = tmp_path / "state.json"
    instance.state = dict(stages=[], arms={})
    instance.report = lambda: None
    instance.contract = dict(
        reused_first_collection=str(tmp_path / "old_first"), initial_adapter_sha256="source"
    )
    return instance


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_failed_auxiliary_requests_are_charged_to_training_budget(controller):
    failed = controller.output / "training/action_q/round_000/failed_attempts/action_q_old"
    failed.mkdir(parents=True)
    (failed / "requests.jsonl").write_text(
        json.dumps(dict(prompt_tokens=RUNNER.TRAIN_CAP, response_tokens=1)) + "\n"
    )
    with pytest.raises(RUNNER.BudgetStop, match="including failed requests"):
        controller.costs()


def test_partial_parent_batch_cannot_launch_q_or_train(controller):
    directory = controller.output / "training/action_q"
    write_json(
        directory / "progress.json",
        dict(rounds=[dict(train="previous")], final_adapter="actor", completed_optimizer_steps=4),
    )
    controller.collection = lambda *a, **kw: ({"complete_trajectories": 28}, False)
    controller.stage = lambda *a, **kw: pytest.fail("partial parents must not train")
    with pytest.raises(RUNNER.BudgetStop, match="incomplete parent batch"):
        controller.train_wide()


def test_local_stage_limit_admits_q_modules_and_rejects_other_workers(controller):
    controller.state["stages"] = [dict(gpu_worker=True, seconds=RUNNER.GPU_SECONDS)]
    for module in (RUNNER.COLLECTOR, RUNNER.Q_AUX, RUNNER.Q_TRAINER):
        with pytest.raises(RUNNER.BudgetStop, match="2-hour"):
            controller.stage("exhausted", module, [])
    with pytest.raises(RuntimeError, match="admitted"):
        controller.stage("forbidden", "src.agents.mandate_repair_train", [])


def test_no_variance_or_q_support_gate_blocks_the_fixed_training(controller, monkeypatch):
    monkeypatch.setattr(
        RUNNER,
        "verify_reusable_first_batch",
        lambda *a: dict(status="COMPLETE", complete_trajectories=32, successes=16, total_tokens=10),
    )
    controller.collection = lambda *a, **kw: (
        dict(status="COMPLETE", complete_trajectories=32, successes=16, total_tokens=10),
        True,
    )
    stages = []

    def stage(name, module, args, gpu_worker=True):
        stages.append(module)
        output = Path(args[args.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if module == RUNNER.Q_AUX:
            (output / "action_q_values.jsonl").write_text("")
            write_json(output / "summary.json", dict(status="BUDGET_EXHAUSTED"))
        elif module == RUNNER.Q_TRAINER:
            write_json(
                output / "summary.json",
                dict(
                    recipe={"action_advantage_estimator": RUNNER.ESTIMATOR},
                    optimizer_updates=4,
                    completed_optimizer_steps=4 * stages.count(module),
                    replaced_target_requests=0,
                    complete_q_task_groups=0,
                ),
            )
        else:
            pytest.fail("no separate pilot or probe stage permitted")

    controller.stage = stage
    actor = controller.train_wide()
    assert actor == controller.output / "training/action_q/round_005/train/adapter"
    assert stages.count(RUNNER.Q_AUX) == 6
    assert stages.count(RUNNER.Q_TRAINER) == 6
    assert len(controller.state["arms"]["wide"]["rounds"]) == 6
