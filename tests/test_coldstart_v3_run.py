"""CPU checks for bounded collection, reused baseline identity and no silent retry."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "coldstart_v3_controller",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_coldstart_v3_run.py",
)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    old = repo / "runs/v2"
    run = repo / "runs/v3"
    run.mkdir(parents=True)
    source = repo / "src/evaluation.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen old evaluation\n")
    model = old / "merged_model"
    initial = {"path": str(model), "merge_receipt_sha256": "receipt-pin", "files": {}}
    monkeypatch.setattr(controller, "ROOT", repo)
    monkeypatch.setattr(controller, "MODEL", model)
    write(old / "controller.json", {"status": "COMPLETE"})
    write(old / "source_manifest.json", {"src/evaluation.py": controller.sha(source)})
    write(old / "exported_model_identity.json", initial)
    inputs = {}
    for suite in ("regression", "fresh"):
        phase = old / f"after_{suite}"
        write(phase / "result.json", {"status": "OBSERVED", "suite": suite})
        (phase / "requests.jsonl").write_text('{"response":"exact original bytes"}\n')
        write(phase / "config.json", {
            "engine": {"model": str(model)}, "action_protocol": controller.PROTOCOL,
        })
        for name in ("result.json", "requests.jsonl"):
            inputs[f"after_{suite}/{name}"] = controller.sha(phase / name)
    write(old / "readout.json", {"status": "COMPLETE", "inputs_sha256": inputs})
    return repo, old, run, source, initial


def test_fixed_phases_have_one_collection_no_new_before_and_two_epochs(tmp_path):
    phases = controller.phases(tmp_path)
    assert [item["name"] for item in phases] == [
        "prepare", "collection", "dataset", "sft", "merged_model",
        "after_regression", "after_fresh", "readout",
    ]
    collect = phases[1]["argv"]
    assert collect[collect.index("--model") + 1] == str(controller.MODEL)
    assert collect[collect.index("--expected-short") + 1] == "0"
    assert collect[collect.index("--expected-full") + 1] == "8"
    assert collect[collect.index("--action-protocol") + 1] == controller.PROTOCOL
    train = phases[3]["argv"]
    for key, expected in (
        ("--model", str(controller.MODEL)), ("--epochs", "2"), ("--lr", "0.00005"),
        ("--micro-batch", "2"), ("--gradient-accumulation", "8"), ("--seed", "42"),
    ):
        assert train[train.index(key) + 1] == expected
    assert not any("learning_restart" in part for item in phases for part in item["argv"])
    for item in phases[5:7]:
        argv = item["argv"]
        assert argv[argv.index("--model") + 1] == str(tmp_path / "merged_model")


def test_prepare_phase_arguments_are_accepted_by_real_cli(tmp_path, monkeypatch):
    from src.agents import coldstart_onpolicy_collect as collect

    observed = []

    async def fake_prepare(output):
        observed.append(output)

    monkeypatch.setattr(collect, "prepare", fake_prepare)
    argv = controller.phases(tmp_path)[0]["argv"]
    assert argv[:2] == ["-m", "src.agents.coldstart_onpolicy_collect"]
    monkeypatch.setattr("sys.argv", ["coldstart_onpolicy_collect", *argv[2:]])
    collect.main()
    assert observed == [tmp_path / "collection/prepared"]


def test_reuse_copies_exact_bytes_and_records_no_regeneration(baseline):
    _, old, run, _, initial = baseline
    result = controller.reuse_baseline(run, initial, old=old)
    assert result["regenerated"] is False
    assert result["model_identity"] == initial
    assert result["source_artifacts_sha256"] == {
        name: controller.sha(old / name) for name in (
            "readout.json", "controller.json", "source_manifest.json",
            "exported_model_identity.json",
        )
    }
    assert len(result["files"]) == 6
    for relative, evidence in result["files"].items():
        assert (run / relative).read_bytes() == Path(evidence["source"]).read_bytes()
        assert controller.sha(run / relative) == evidence["sha256"]
    assert controller.read(run / "baseline_reuse.json") == result


def test_reuse_rejects_changed_source_before_copying(baseline):
    _, old, run, source, initial = baseline
    source.write_text("# changed evaluation\n")
    with pytest.raises(RuntimeError, match="cannot reuse baseline after source change"):
        controller.reuse_baseline(run, initial, old=old)
    assert not (run / "before_regression").exists()


def test_reuse_rejects_different_evaluated_model_identity(baseline):
    _, old, run, _, initial = baseline
    altered = copy.deepcopy(initial)
    altered["merge_receipt_sha256"] = "different-export"
    with pytest.raises(RuntimeError, match="differs from the v2 evaluated export"):
        controller.reuse_baseline(run, altered, old=old)


@pytest.mark.parametrize("name", ["result.json", "requests.jsonl"])
def test_reuse_rejects_changed_original_result_or_requests(baseline, name):
    _, old, run, _, initial = baseline
    path = old / "after_regression" / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(RuntimeError, match="v2 evaluation evidence changed"):
        controller.reuse_baseline(run, initial, old=old)


def test_reuse_rejects_changed_protocol(baseline):
    _, old, run, _, initial = baseline
    path = old / "after_fresh/config.json"
    config = controller.read(path)
    config["action_protocol"] = "single_tool_v1"
    write(path, config)
    with pytest.raises(RuntimeError, match="model/protocol mismatch"):
        controller.reuse_baseline(run, initial, old=old)


def test_frozen_check_rejects_copied_evidence_tampering(baseline):
    _, old, run, _, initial = baseline
    evidence = controller.reuse_baseline(run, initial, old=old)
    target = run / "before_fresh/requests.jsonl"
    target.write_text(target.read_text() + " ")
    with pytest.raises(RuntimeError, match="baseline copy changed"):
        controller.check_frozen(run, {}, evidence)


def test_frozen_check_rejects_source_and_preparation_tampering(baseline):
    repo, _, run, source, _ = baseline
    sources = {str(source.relative_to(repo)): controller.sha(source)}
    prepared_path = run / "collection/prepared/collection_contexts.json"
    write(prepared_path, {"original": "eight frozen source windows"})
    prepared = {str(prepared_path.relative_to(run)): controller.sha(prepared_path)}
    controller.check_frozen(run, sources, {"files": {}}, prepared)
    prepared_path.write_text("changed contexts")
    with pytest.raises(RuntimeError, match="collection preparation changed"):
        controller.check_frozen(run, sources, {"files": {}}, prepared)
    source.write_text("changed source")
    with pytest.raises(RuntimeError, match="live source changed"):
        controller.check_frozen(run, sources, {"files": {}})


@pytest.mark.parametrize(
    "row_count,accepted", [(740, False), (741, True), (1380, True), (1381, False)],
)
def test_aggregated_data_budget_requires_new_data_and_respects_cap(
    tmp_path, monkeypatch, row_count, accepted,
):
    manifest = tmp_path / "dataset/manifest.json"
    write(manifest, {"row_count": row_count})
    monkeypatch.setattr(controller, "verify_dataset", lambda run: {"row_count": row_count})
    args = (tmp_path, {}, {"files": {}}, {}, controller.sha(manifest))
    if accepted:
        controller.check_frozen(*args)
    else:
        with pytest.raises(RuntimeError, match="fixed row budget"):
            controller.check_frozen(*args)


def test_changed_dataset_manifest_is_rejected_before_training(tmp_path, monkeypatch):
    manifest = tmp_path / "dataset/manifest.json"
    write(manifest, {"row_count": 1000})
    frozen = controller.sha(manifest)
    write(manifest, {"row_count": 1001})
    monkeypatch.setattr(controller, "verify_dataset", lambda run: pytest.fail("too late"))
    with pytest.raises(RuntimeError, match="dataset manifest changed"):
        controller.check_frozen(tmp_path, {}, {"files": {}}, dataset=frozen)


@pytest.mark.parametrize("existing", ["controller.json", "sources", "collection", "sft"])
def test_existing_run_identity_cannot_restart_or_resample(tmp_path, monkeypatch, existing):
    marker = tmp_path / existing
    if marker.suffix:
        marker.write_text('{"status":"FAILED"}')
    else:
        marker.mkdir()
    monkeypatch.setattr("sys.argv", ["controller", "--directory", str(tmp_path)])
    monkeypatch.setattr(controller, "verify_old_runtime", lambda: pytest.fail("must stop first"))
    with pytest.raises(RuntimeError, match="run identity already used"):
        controller.main()


def test_collection_failure_is_recorded_once_and_training_never_starts(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["controller", "--directory", str(tmp_path)])
    monkeypatch.setattr(controller, "verify_old_runtime", lambda: None)
    monkeypatch.setattr(controller, "model_identity", lambda model: {"path": str(model)})
    monkeypatch.setattr(controller, "seal_sources", lambda run: {})
    monkeypatch.setattr(controller, "reuse_baseline", lambda run, initial: {"files": {}})
    monkeypatch.setattr(controller, "_resolve_c_compiler", lambda parent: ("/cpu/cc", "fake"))
    monkeypatch.setattr(
        controller.shutil, "disk_usage", lambda run: SimpleNamespace(free=50 * 1024**3),
    )
    launched = []

    def execute(run, item, state, env):
        launched.append(item["name"])
        item["status"] = "RUNNING"
        state["phases"].append(item)
        if item["name"] == "collection":
            raise RuntimeError("deliberate collection failure")
        assert item["name"] == "prepare"
        write(run / "collection/prepared/manifest.json", {"status": "COMPLETE"})
        item["status"] = "COMPLETE"

    monkeypatch.setattr(controller, "execute_phase", execute)
    with pytest.raises(RuntimeError, match="deliberate collection failure"):
        controller.main()
    assert launched == ["prepare", "collection"]
    state = controller.read(tmp_path / "controller.json")
    assert state["status"] == "FAILED"
    assert state["phases"][-1]["status"] == "FAILED"
    assert state["automatic_retry"] is False
    assert state["automatic_rl_launch"] is False
    assert state["baseline_regenerated"] is False
    with pytest.raises(RuntimeError, match="run identity already used"):
        controller.main()
    assert launched == ["prepare", "collection"]
