"""Exercise controller boundaries with the evaluator's own fake-runtime output."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


controller = load(
    "sampling_run_controller_test", ROOT / "scripts/diagnostics/m2_sampling_profile_run.py"
)
runtime_tests = load(
    "sampling_eval_runtime_fixtures", ROOT / "tests/test_m2_sampling_profile_eval.py"
)


@pytest.fixture(scope="module")
def native_results(tmp_path_factory):
    """Use evaluate(), rather than inventing dictionaries matching the validator."""
    outputs = {}
    for failure in (None, "REQUEST_CAP"):
        path = tmp_path_factory.mktemp("native_sampling_" + str(failure))
        with pytest.MonkeyPatch.context() as monkeypatch:
            runtime_tests.test_runtime_keeps_all_96_lanes_and_never_fabricates_incomplete_outcomes(
                path, monkeypatch, failure
            )
        outputs[failure] = path
    return outputs


@pytest.mark.parametrize("failure", [None, "REQUEST_CAP"])
def test_controller_accepts_real_evaluator_grid_and_retains_request_cap(native_results, failure):
    observed = controller.validate_result(native_results[failure], runtime_tests.contexts())
    assert observed["observed"] == 96
    assert observed["complete"] == (96 if failure is None else 95)
    assert observed["incomplete"] == int(failure is not None)
    assert observed["requests"] == (1920 if failure is None else 2028)
    assert controller.CELL_IDS == ("sft0", "m1", "m2_eta1", "m2_eta025")
    assert controller.settings()["planned_trajectories"] == 4 * observed["observed"] == 384
    assert controller.settings()["profiles"] == runtime_tests.entry.PROFILES
    result = json.loads((native_results[failure] / "result.json").read_text())
    if failure:
        assert not any(
            key in result["grid"][0]
            for key in ("takeover_net_return", "terminal_metrics", "final_state")
        )


@pytest.mark.parametrize(
    "damage",
    ["native_seed", "requested_seed", "raw_profile", "row_profile", "missing_lane", "top_profile"],
)
def test_controller_rejects_changed_profile_seed_or_population(native_results, tmp_path, damage):
    directory = tmp_path / "damaged"
    shutil.copytree(native_results[None], directory)
    if damage in ("native_seed", "requested_seed", "raw_profile"):
        path = directory / "requests.jsonl"
        lines = path.read_text().splitlines()
        first = json.loads(lines[0])
        if damage == "native_seed":
            first["native_sampling_params"]["seed"] += 1
        elif damage == "requested_seed":
            first["sampling_params"]["seed"] += 1
        else:
            first["profile"] = "qwen_recommended"
        lines[0] = json.dumps(first)
        path.write_text("\n".join(lines) + "\n")
    else:
        result = json.loads((directory / "result.json").read_text())
        if damage == "row_profile":
            result["grid"][0]["profile"] = "qwen_recommended"
        elif damage == "missing_lane":
            result["grid"].pop()
        else:
            result["sampling_profiles"]["current"]["temperature"] = 0.0
        for name in ("result.json", "state.json"):
            (directory / name).write_text(json.dumps(result))
    with pytest.raises(RuntimeError):
        controller.validate_result(directory, runtime_tests.contexts())


@pytest.mark.parametrize("damage", ["invented_outcome", "terminal_session"])
def test_request_cap_cannot_become_a_terminal_outcome(native_results, tmp_path, damage):
    directory = tmp_path / "damaged_cap"
    shutil.copytree(native_results["REQUEST_CAP"], directory)
    result = json.loads((directory / "result.json").read_text())
    row = result["grid"][0]
    if damage == "invented_outcome":
        row["takeover_net_return"] = 0.0
    else:
        row["steps_including_prefix"][-1]["result_session_index"] = 20
        path = directory / "requests.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if record["trajectory_id"] == row["id"] and record["request_index"] == 127:
                record["trace"]["result_session_index"] = 20
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    for name in ("result.json", "state.json"):
        (directory / name).write_text(json.dumps(result))
    with pytest.raises(RuntimeError):
        controller.validate_result(directory, runtime_tests.contexts())


@pytest.mark.parametrize("final_refresh_succeeds", [True, False])
def test_monitor_refreshes_after_service_finishes_and_preserves_incomplete(
    tmp_path, monkeypatch, final_refresh_succeeds
):
    state_path = tmp_path / "state.json"
    state_path.write_text('{"status":"RUNNING"}')
    refresh_calls = []

    def refresh(base):
        refresh_calls.append(base)
        finished = len(refresh_calls) == 2 and final_refresh_succeeds
        execution = "ALL_CELLS_OBSERVED_WITH_INCOMPLETE" if finished else "RUNNING"
        readout = {
            "status": "INCOMPLETE",
            "execution_status": execution,
            "terminal_observed": 384,
            "complete": 383,
            "incomplete": 1,
            "problems": [],
        }
        (base / "readout.json").write_text(json.dumps(readout))
        if finished:
            (base / "FINAL_RESULTS.md").write_text("383 complete; 1 REQUEST_CAP retained")
        return SimpleNamespace(returncode=0, stdout="readout", stderr="")

    def service_ends_between_readout_and_status(*args, **kwargs):
        state_path.write_text('{"status":"ALL_CELLS_OBSERVED_WITH_INCOMPLETE"}')
        return SimpleNamespace(
            returncode=0,
            stdout="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
            stderr="",
        )

    monkeypatch.setattr(controller, "refresh", refresh)
    monkeypatch.setattr(controller.subprocess, "run", service_ends_between_readout_and_status)
    assert controller.monitor(tmp_path, "mock.service") == (0 if final_refresh_succeeds else 2)
    assert len(refresh_calls) == 2
    assert json.loads((tmp_path / "readout.json").read_text())["status"] == "INCOMPLETE"
    monitor = json.loads((tmp_path / "monitor_state.json").read_text())
    assert monitor["status"] == (
        "ALL_CELLS_OBSERVED_WITH_INCOMPLETE" if final_refresh_succeeds else "STOPPED_EARLY"
    )


def test_gpu_capacity_accepts_occupied_gpu_with_sufficient_free_memory(monkeypatch):
    calls = []
    readings = iter(("95830, 15600, 80230, 95\n", "95830, 34390, 61440, 80\n"))

    def query_only(command, **kwargs):
        assert command == [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=next(readings), stderr="")

    def no_wait(seconds):
        pytest.fail("sufficient free GPU memory must not wait for another job to stop")

    monkeypatch.setattr(controller.subprocess, "run", query_only)
    monkeypatch.setattr(controller.time, "sleep", no_wait)
    monkeypatch.setattr(controller, "now", lambda: "2026-09-15T20:18:00+08:00")
    first = controller.wait_gpu_capacity()
    assert first == {
        "total_mib": 95830,
        "used_mib": 15600,
        "free_mib": 80230,
        "utilization_percent": 95,
        "minimum_free_mib": 61440,
        "checked_at_SGT": "2026-09-15T20:18:00+08:00",
    }
    assert controller.wait_gpu_capacity()["free_mib"] == 61440
    assert len(calls) == 2


def test_gpu_capacity_waits_then_times_out_without_stopping_another_job(monkeypatch):
    elapsed, sleeps, queries = [0.0], [], []

    def query_only(command, **kwargs):
        assert command[0] == "nvidia-smi" and len(command) == 3
        assert command[1].startswith("--query-gpu=")
        queries.append(command)
        return SimpleNamespace(returncode=0, stdout="95830, 34391, 61439, 80\n", stderr="")

    def advance(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(controller.subprocess, "run", query_only)
    monkeypatch.setattr(controller.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(controller.time, "sleep", advance)
    with pytest.raises(RuntimeError, match="insufficient GPU capacity; no other job was stopped"):
        controller.wait_gpu_capacity(timeout=10)
    assert elapsed[0] == 10
    assert sleeps == [5, 5] and len(queries) == 3
