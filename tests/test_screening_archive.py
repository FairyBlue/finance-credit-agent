"""Archive safety checks use synthetic temporary checkpoints, never run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agents import screening_archive as archive


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cell = tmp_path / "cell"
    cell.mkdir()
    _write(cell / "summary.json", {"status": "PASSED", "stage": "screening"})
    _write(
        cell / "resolved_config.json",
        {
            "screening": {"stage": "screening"},
            "tracking": {"log_root": str(cell)},
            "training": {"total_training_steps": 16},
        },
    )
    _write(cell / "command.json", [archive._IDENTITY_KEY + "a" * 64])
    _write(cell / "process.json", {"pid": 2147483647})
    final = cell / "checkpoints/global_step_16"
    sizes = {}
    for role in ("actor", "critic"):
        for kind in ("model", "optim", "extra_state"):
            relative = f"{role}/{kind}_world_size_1_rank_0.pt"
            path = final / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-checkpoint-state")
            sizes[relative] = path.stat().st_size
    (final / "data.pt").write_bytes(b"dataloader-state")
    sizes["data.pt"] = (final / "data.pt").stat().st_size
    _write(
        final / "screening_complete.json",
        {
            "schema_version": 1,
            "identity_sha256": "a" * 64,
            "world_size": 1,
            "global_step": 16,
            "files": sizes,
        },
    )
    _write(final / "actor/fsdp_config.json", {"world_size": 1})
    _write(cell / "artifacts/screening_evidence/trajectory.json", {"complete": True})
    monkeypatch.setattr(archive, "_resource_check", lambda *args: {"checked": 1})
    monkeypatch.setattr(archive, "_run_merger", _fake_merger)
    monkeypatch.setattr(
        archive,
        "_verify_model_loading",
        lambda *args: {
            "architecture_load_verified": True,
            "tokenizer_load_verified": True,
        },
    )
    return cell


def _fake_merger(actor: Path, destination: Path, cell: Path) -> list[str]:
    _write(destination / "config.json", {"torch_dtype": "bfloat16", "model_type": "qwen3"})
    _write(destination / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    _write(destination / "tokenizer.json", {"version": "1.0"})
    header = json.dumps(
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            }
        }
    ).encode()
    (destination / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"1234"
    )
    return ["fake-merger", str(actor), str(destination)]


def _states(cell: Path) -> list[Path]:
    return sorted((cell / "checkpoints").rglob("*.pt"))


def test_export_is_verified_before_only_resume_states_are_pruned(completed_cell):
    result = archive.archive_completed_cell(completed_cell, "screening")
    assert result["status"] == "ARCHIVED"
    assert result["hf_validation"]["dtype"] == "bfloat16"
    assert result["hf_validation"]["architecture_load_verified"]
    assert len(result["removed_files"]) == 7
    assert result["reclaimed_bytes"] > 0
    assert _states(completed_cell) == []
    assert (completed_cell / "actor_hf/model.safetensors").is_file()
    assert (completed_cell / "artifacts/screening_evidence/trajectory.json").is_file()
    assert (completed_cell / "checkpoints/global_step_16/screening_complete.json").is_file()
    assert archive.archive_completed_cell(completed_cell, "screening") == result


@pytest.mark.parametrize("failure", ["merger", "tokenizer", "weights", "dtype", "loading", "disk"])
def test_any_export_failure_retains_every_checkpoint_state(completed_cell, monkeypatch, failure):
    original = _states(completed_cell)
    if failure == "merger":

        def broken_merger(*args):
            raise RuntimeError("merger failed")

        monkeypatch.setattr(archive, "_run_merger", broken_merger)
    elif failure == "disk":

        def disk_failure(*args):
            raise RuntimeError("insufficient disk space")

        monkeypatch.setattr(archive, "_resource_check", disk_failure)
    elif failure == "loading":

        def loading_failure(*args):
            raise RuntimeError("missing model keys")

        monkeypatch.setattr(archive, "_verify_model_loading", loading_failure)
    else:

        def damaged(actor, destination, cell):
            command = _fake_merger(actor, destination, cell)
            if failure == "tokenizer":
                (destination / "tokenizer.json").unlink()
            elif failure == "weights":
                path = destination / "model.safetensors"
                path.write_bytes(path.read_bytes()[:-1])
            else:
                _write(destination / "config.json", {"torch_dtype": "float32"})
            return command

        monkeypatch.setattr(archive, "_run_merger", damaged)
    with pytest.raises(RuntimeError):
        archive.archive_completed_cell(completed_cell, "screening")
    assert _states(completed_cell) == original
    assert not (completed_cell / "archive.json").exists()


@pytest.mark.parametrize("stage", ["wiring", "pilot"])
def test_smoke_and_pilot_drop_weights_without_export_but_preserve_evidence(
    completed_cell,
    monkeypatch,
    stage,
):
    summary = archive._json(completed_cell / "summary.json")
    summary["stage"] = stage
    _write(completed_cell / "summary.json", summary)
    config = archive._json(completed_cell / "resolved_config.json")
    config["screening"]["stage"] = stage
    _write(completed_cell / "resolved_config.json", config)

    def unexpected_merger(*args):
        raise AssertionError("wiring/pilot must not export inference weights")

    monkeypatch.setattr(archive, "_run_merger", unexpected_merger)
    result = archive.archive_completed_cell(completed_cell, stage)
    assert result["actor_hf"] is None and len(result["removed_files"]) == 7
    assert (completed_cell / "summary.json").is_file()


def test_non_passed_cell_is_never_archived(completed_cell):
    _write(completed_cell / "summary.json", {"status": "FAILED", "stage": "screening"})
    with pytest.raises(RuntimeError, match="PASSED"):
        archive.archive_completed_cell(completed_cell, "screening")
    assert len(_states(completed_cell)) == 7


def test_missing_complete_marker_prevents_export(completed_cell):
    (completed_cell / "checkpoints/global_step_16/screening_complete.json").unlink()
    with pytest.raises(RuntimeError, match="marker"):
        archive.archive_completed_cell(completed_cell, "screening")
    assert len(_states(completed_cell)) == 7


def test_checkpoint_symlink_cannot_target_another_directory(completed_cell, tmp_path):
    outside = tmp_path / "external.pt"
    outside.write_bytes(b"keep-me")
    (completed_cell / "checkpoints/global_step_16/actor/redirect.pt").symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        archive.archive_completed_cell(completed_cell, "screening")
    assert outside.read_bytes() == b"keep-me" and len(_states(completed_cell)) == 8


def test_live_process_prevents_archive(completed_cell, monkeypatch):
    monkeypatch.setattr(archive.os, "kill", lambda *args: None)
    with pytest.raises(RuntimeError, match="still alive"):
        archive.archive_completed_cell(completed_cell, "screening")
    assert len(_states(completed_cell)) == 7


def test_unknown_files_in_checkpoint_tree_are_not_deleted(completed_cell):
    note = completed_cell / "checkpoints/global_step_16/manual-note.txt"
    note.write_text("keep this evidence")
    archive.archive_completed_cell(completed_cell, "screening")
    assert note.read_text() == "keep this evidence"


def test_resource_check_refuses_insufficient_disk(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(archive.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    monkeypatch.setattr(archive, "_host_available_bytes", lambda: 100 * archive._GIB)
    with pytest.raises(RuntimeError, match="disk headroom"):
        archive._resource_check(tmp_path, 16 * archive._GIB)


def test_resource_check_accounts_for_duplicate_cpu_model_allocation(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(
        archive.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * archive._GIB)
    )
    monkeypatch.setattr(archive, "_host_available_bytes", lambda: 20 * archive._GIB)
    with pytest.raises(RuntimeError, match="CPU memory"):
        archive._resource_check(tmp_path, 16 * archive._GIB)
