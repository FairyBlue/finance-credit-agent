"""Bounded controller lifecycle and source/data-seal regressions, without GPU."""

from types import SimpleNamespace

import pytest

from src.agents import common_validity_penalty as entry


def test_run_stops_after_first_failed_cell_and_preserves_budget(tmp_path, monkeypatch):
    calls, waits = [], []
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(entry, "wait_gpu", lambda: waits.append(True))
    monkeypatch.setattr(
        entry.shutil, "disk_usage", lambda base: SimpleNamespace(free=100 * 1024**3)
    )

    def run_cell(base, cell):
        calls.append(cell)
        if cell == "m1":
            raise RuntimeError("fixture trainer exited 7")
        return {"status": "COMPLETE", "cell": cell}

    monkeypatch.setattr(entry, "run_cell", run_cell)
    with pytest.raises(RuntimeError, match="trainer exited 7"):
        entry.run(tmp_path)
    state = entry.restart.read(tmp_path / "state.json")
    assert calls == ["sft0", "m1"] and len(waits) == 2
    assert state["status"] == "FAILED" and state["current_cell"] == "m1"
    assert len(state["cells"]) == 1
    assert state["planned_trajectories"] == 20 + 2 * (32 + 40)
    assert state["automatic_retry"] is state["resume_supported"] is False
    with pytest.raises(RuntimeError, match="no implicit retry"):
        entry.run(tmp_path)
    assert calls == ["sft0", "m1"]


def test_insufficient_disk_stops_before_gpu_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(entry.shutil, "disk_usage", lambda base: SimpleNamespace(free=25 * 1024**3))
    monkeypatch.setattr(entry, "wait_gpu", lambda: pytest.fail("must not query GPU"))
    monkeypatch.setattr(entry, "run_cell", lambda *args: pytest.fail("must not train"))
    sentinel = tmp_path / "existing-evidence.txt"
    sentinel.write_text("preserve")
    with pytest.raises(RuntimeError, match="insufficient space"):
        entry.run(tmp_path)
    assert sentinel.read_text() == "preserve"
    state = entry.restart.read(tmp_path / "state.json")
    assert state["status"] == "FAILED" and state["cells"] == []


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    base = tmp_path / "new-run"
    (base / "prepared").mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    monkeypatch.setattr(entry, "source_files", lambda: {"src/fixture.py": source})
    snapshot = tmp_path / "sft"
    model = {"source": "local_export", "name": str(snapshot), "merge_receipt_sha256": "a" * 64}
    monkeypatch.setattr(
        entry,
        "_resolve_exact_model_snapshot",
        lambda config: (str(snapshot), "fixture-model-revision", snapshot),
    )
    entry.atomic_json(base / "settings.json", entry.settings(model))
    entry.atomic_json(
        base / "model_identity.json",
        {"model_revision": "fixture-model-revision", "model_snapshot_path": str(snapshot)},
    )
    entry.atomic_json(
        base / "input_preflight.json",
        {
            "model_revision": "fixture-model-revision",
            "action_protocol": entry.tuning.PROTOCOL,
            "enable_thinking": False,
        },
    )
    entry.atomic_json(
        base / "source_identity.json", {"src/fixture.py": entry.restart.digest(source)}
    )
    entry.atomic_json(
        base / "selection.json", {"source_sha256": {str(source): entry.restart.digest(source)}}
    )
    manifest = {"datasets": {}}
    for name, ids in (("train", entry.TRAIN_IDS), ("dev", entry.EVAL_IDS)):
        path = base / "prepared" / f"{name}.parquet"
        path.write_bytes(b"fixed dataset fixture")
        manifest["datasets"][name] = {
            "path": path.name,
            "sha256": entry.restart.digest(path),
            "episode_ids": ids,
            "row_count": len(ids),
        }
    calibration = base / "prepared/reward_calibration.json"
    calibration.write_text("{}\n")
    manifest["calibration"] = {
        "path": calibration.name,
        "sha256": entry.restart.digest(calibration),
    }
    entry.atomic_json(base / "prepared/manifest.json", manifest)
    (base / "plan.md").write_text("frozen fixture plan")
    entry.atomic_json(
        base / "run_contract.json",
        {
            "profile": entry.PROFILE,
            "inputs": {name: entry.restart.digest(base / name) for name in entry.INPUTS},
        },
    )
    return base, source


@pytest.mark.parametrize("changed", ["source", "dataset", "input"])
def test_contract_rejects_source_data_or_input_drift_before_run(sealed, monkeypatch, changed):
    base, source = sealed
    entry.check_contract(base)
    target = (
        source
        if changed == "source"
        else (base / "prepared/train.parquet" if changed == "dataset" else base / "settings.json")
    )
    target.write_text("changed")
    monkeypatch.setattr(entry, "wait_gpu", lambda: pytest.fail("must not query GPU"))
    with pytest.raises(RuntimeError, match="changed"):
        entry.run(base)
    assert not (base / "state.json").exists()
