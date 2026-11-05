"""Guard resource accounting, fixed denominators and real resume in the V2 runner."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIAGNOSTICS = Path(__file__).resolve().parents[1] / "scripts/diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))
SPEC = importlib.util.spec_from_file_location(
    "financial_learning_runner_for_test", DIAGNOSTICS / "run_financial_learning.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def controller(tmp_path):
    instance = RUNNER.Controller.__new__(RUNNER.Controller)
    instance.output = tmp_path
    instance.path = tmp_path / "state.json"
    instance.state = {"stages": [], "arms": {}}
    instance.report = lambda: None
    return instance


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_partial_training_resumes_exact_saved_state(controller):
    destination = controller.output / "round_000/train"
    destination.mkdir(parents=True)
    (destination / "training_state.pt").write_bytes(b"real FP32 and Adam state")
    (destination / "optimizer_updates.jsonl").write_text('{"optimizer_step": 2}\n')
    run, checkpoint = controller.fresh_training(destination)
    assert run is True
    assert checkpoint.read_bytes() == b"real FP32 and Adam state"
    assert not destination.exists()
    assert (checkpoint.parent / "optimizer_updates.jsonl").exists()
    assert controller.state["training_resumes"][0]["retains_fp32_and_adam"] is True


def test_complete_training_is_never_restarted(controller):
    destination = controller.output / "train"
    write_json(destination / "summary.json", {"status": "COMPLETE"})
    assert controller.fresh_training(destination) == (False, None)


def test_failed_requests_are_charged_to_the_arm_allowance(controller):
    root = controller.output / "training/rloo_lr2e6/failed_attempts/collect_failure"
    root.mkdir(parents=True)
    (root / "requests.jsonl").write_text(
        json.dumps({"prompt_tokens": RUNNER.ARM_CAP, "response_tokens": 1}) + "\n"
    )
    with pytest.raises(RuntimeError, match="rloo_lr2e6 generation"):
        controller.costs()


def test_source_drift_rejected_after_first_worker_use(controller, monkeypatch):
    source_root = controller.output / "source"
    source = source_root / "example/worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("ORIGINAL = True\n")
    monkeypatch.setattr(RUNNER, "ROOT", source_root)
    controller.source_receipt("example.worker")
    source.write_text("ORIGINAL = False\n")
    with pytest.raises(RuntimeError, match="source changed after first use"):
        controller.source_receipt("example.worker")
    assert (controller.output / "frozen_sources/example/worker.py").read_text() == (
        "ORIGINAL = True\n"
    )


def test_completed_collection_reused_without_new_allowance(controller):
    data = controller.output / "data/panel.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"task_id": "a"}\n{"task_id": "b"}\n')
    destination = controller.output / "baseline"
    write_json(
        destination / "summary.json",
        {"planned_trajectories": 8, "complete_trajectories": 8},
    )
    controller.stage = lambda *a, **k: pytest.fail("cached collection must not launch GPU")
    _, complete = controller.collection(
        "cached",
        data,
        Path("actor"),
        destination,
        repetitions=4,
        temperature=1,
        allowance=0,
    )
    assert complete


def test_partial_fixed_batch_never_enters_actor_training(controller):
    batch = controller.output / "data/batch_round000.jsonl"
    batch.parent.mkdir(parents=True)
    batch.write_text("".join(json.dumps({"task_id": f"t{i}"}) + "\n" for i in range(8)))
    controller.collection = lambda *a, **k: (
        {"planned_trajectories": 32, "complete_trajectories": 28},
        False,
    )
    controller.stage = lambda *a, **k: pytest.fail("partial parent batch must not train")
    result = controller.train_arm("rloo_lr2e6", 2e-6)
    assert result == RUNNER.INITIAL
    assert controller.state["arms"]["rloo_lr2e6"]["stop_reason"] == (
        "INCOMPLETE_FIXED_BATCH_NOT_TRAINED"
    )
    assert controller.state["arms"]["rloo_lr2e6"]["rounds"] == []


def test_reserved_test_cannot_be_opened_by_evaluate(controller):
    with pytest.raises(RuntimeError, match="reserved Test"):
        controller.evaluate("rloo", Path("actor"), "test")


def test_parent_sampling_streams_are_stable_and_distinct_across_arms_and_rounds():
    keys = [(arm, index) for arm in RUNNER.ARMS for index in range(6)]
    seeds = [RUNNER.parent_collection_seed(arm, index) for arm, index in keys]
    assert len(set(seeds)) == len(keys)
    assert all(0 <= seed < 2**31 - 1 for seed in seeds)
    assert seeds == [RUNNER.parent_collection_seed(arm, index) for arm, index in keys]
    assert 142 not in seeds  # Independent baseline is not reused as a training random stream.
