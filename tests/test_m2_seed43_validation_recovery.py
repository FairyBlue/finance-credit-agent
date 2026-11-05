"""No rerolls after a validator-only failure; immutable imports and fresh M2."""

import json
import runpy
from pathlib import Path

import pytest

helpers = runpy.run_path(str(Path(__file__).resolve().parents[1]
                            / "scripts/diagnostics/m2_seed43_validation_recovery.py"))


def failure(tmp_path):
    (tmp_path / "m1").mkdir()
    (tmp_path / "state.json").write_text(json.dumps({
        "status": "FAILED", "seed": 43, "current_cell": "m1", "error": helpers["ERROR"],
    }))
    (tmp_path / "m1/summary.json").write_text(json.dumps({
        "status": "FAILED", "seed": 43, "error": helpers["ERROR"], "returncode": 0,
    }))
    return tmp_path


def test_known_completed_trainer_failure_is_eligible_without_changing_original(tmp_path):
    original = failure(tmp_path)
    before = (original / "state.json").read_bytes()
    assert helpers["verify_original_failure"](original)["status"] == "FAILED"
    assert (original / "state.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["trainer_failure", "other_error", "other_seed", "started_m2"])
def test_no_other_failure_or_already_started_m2_is_eligible(tmp_path, mutation):
    original = failure(tmp_path)
    path = original / "m1/summary.json"
    value = json.loads(path.read_text())
    if mutation == "trainer_failure":
        value["returncode"] = 1
    elif mutation == "other_error":
        value["error"] = "CUDA OOM"
    elif mutation == "other_seed":
        value["seed"] = 42
    else:
        (original / "m2_eta025").mkdir()
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="eligible"):
        helpers["verify_original_failure"](original)


def test_copy_preserves_evidence_and_shares_only_immutable_model_weights(tmp_path):
    source = tmp_path / "original"
    (source / "artifacts/placement/moment_scratch/123/actor").mkdir(parents=True)
    (source / "artifacts/placement/moment_scratch/123/actor/unused.raw").write_bytes(b"moments")
    (source / "artifacts/evidence.jsonl").write_bytes(b'{"seed":43}\n')
    (source / "checkpoints").mkdir()
    weights = source / "checkpoints/model.safetensors"
    weights.write_bytes(b"immutable-example")
    destination = tmp_path / "import"
    helpers["copy_cell"](source, destination)
    assert (destination / "artifacts/evidence.jsonl").read_bytes() == b'{"seed":43}\n'
    assert ((destination / "artifacts/evidence.jsonl").stat().st_ino
            != (source / "artifacts/evidence.jsonl").stat().st_ino)
    assert not (destination / "artifacts/placement/moment_scratch").exists()
    assert (destination / "checkpoints/model.safetensors").stat().st_ino == weights.stat().st_ino
    retained = source / "artifacts/placement/moment_scratch/123/actor/unused.raw"
    assert retained.read_bytes() == b"moments"


def test_copy_rejects_redirected_original_artifacts(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    (source / "redirected.json").symlink_to(tmp_path / "other.json")
    with pytest.raises(RuntimeError, match="redirected"):
        helpers["copy_cell"](source, tmp_path / "import")


@pytest.mark.parametrize("already", ["state.json", "m2_eta025"])
def test_continuation_cannot_repeat_a_started_attempt(tmp_path, already):
    if already.endswith("json"):
        (tmp_path / already).write_text("{}")
    else:
        (tmp_path / already).mkdir()
    with pytest.raises(RuntimeError, match="one-shot"):
        helpers["run_remaining"](tmp_path)
