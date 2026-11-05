"""Tiny real XOR archives exercise the explicit historical ownership migration."""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from src.agents import actor_export
from src.agents import parameter_grid_archive as archive

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/archive_stage2_historical_actors.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_historical_storage", SCRIPT)
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_model(path, value):
    path.mkdir(parents=True)
    weight_map = {}
    for index in range(2):
        name, tensor = f"part{index}.safetensors", f"tensor{index}"
        payload = bytes([value + index]) * 64
        header = json.dumps({tensor: dict(dtype="BF16", shape=[32], data_offsets=[0, 64])})
        encoded = header.encode()
        (path / name).write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
        weight_map[tensor] = name
    write_json(
        path / "model.safetensors.index.json",
        dict(metadata=dict(total_size=128), weight_map=weight_map),
    )
    write_json(path / "config.json", dict(torch_dtype="bfloat16"))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (path / name).write_text("fixture tokenizer")


def export_receipt(checkpoint, identity, step):
    hf = checkpoint / "actor/huggingface"
    receipt = dict(
        schema_version=1,
        artifact_profile="actor_bf16_hf_v1",
        identity_sha256=identity,
        global_step=step,
        world_size=1,
        training_dtype="float32",
        export_dtype="bfloat16",
        actor_included=True,
        critic_included=False,
        optimizer_included=False,
        resume_supported=False,
        lossy_fp32_to_bf16=True,
        source_copy_readback_verified=True,
        training_storage_version_preserved=True,
        hf=actor_export._hf_evidence(hf, verify_hashes=True),
    )
    write_json(hf.parent / actor_export.ACTOR_RECEIPT, receipt)
    write_json(
        checkpoint / actor_export.EXPORT_COMPLETION_MARKER,
        actor_export.export_completion_document(
            checkpoint, profile="actor_bf16_hf_v1", identity=identity, step=step
        ),
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(storage, "REPO", repo)
    directory = repo / "runs/m2-action-pooling-storage-20261012-v1"
    monkeypatch.setattr(storage, "DEFAULT", directory)
    monkeypatch.setattr(storage, "RESERVE", 0)
    monkeypatch.setattr(storage, "source_identity", lambda: {"frozen_test_runtime": "a" * 64})
    monkeypatch.setattr(
        storage, "_validate_evidence", lambda *args, **kwargs: dict(status="PASS", fixture=True)
    )
    baseline = repo / storage.BASELINE
    write_model(baseline, 2)
    write_json(baseline / "merge_receipt.json", dict(profile="fixture"))
    monkeypatch.setattr(
        storage, "BASELINE_RECEIPT", storage.digest(baseline / "merge_receipt.json")
    )
    originals = {}
    for index, (name, rel, step, aliases) in enumerate(storage.CATALOG):
        cell = repo / rel
        identity = f"{index + 1:064x}"
        cp = cell / "checkpoints" / f"global_step_{step}"
        hf = cp / "actor/huggingface"
        write_model(hf, 5 + index)
        export_receipt(cp, identity, step)
        cfg = dict(
            models=dict(
                screening=dict(
                    source="local_export",
                    name=str(baseline),
                    merge_receipt_sha256=storage.BASELINE_RECEIPT,
                )
            )
        )
        write_json(cell / "settings.json", cfg)
        write_json(cell.parent / "state.json", dict(status="COMPLETE"))
        summary = dict(
            status="COMPLETE",
            returncode=0,
            actual_rounds=step,
            cell=dict(cell_id=cell.name, rounds=step),
            execution_identity_sha256=identity,
        )
        for role in ("actor", "critic"):
            write_json(
                cell / f"artifacts/placement/moment_scratch/999999999/{role}/owner.json",
                dict(identity=identity, role=role, owner=f"fixture_{role}"),
            )
        for alias_rel in aliases:
            alias = repo / alias_rel
            alias_cp = alias / "checkpoints" / f"global_step_{step}"
            alias_hf = alias_cp / "actor/huggingface"
            alias_hf.mkdir(parents=True)
            for source in hf.iterdir():
                if source.suffix == ".safetensors":
                    os.link(source, alias_hf / source.name)
                else:
                    shutil.copyfile(source, alias_hf / source.name)
            shutil.copyfile(
                hf.parent / actor_export.ACTOR_RECEIPT, alias_hf.parent / actor_export.ACTOR_RECEIPT
            )
            shutil.copyfile(
                cp / actor_export.EXPORT_COMPLETION_MARKER,
                alias_cp / actor_export.EXPORT_COMPLETION_MARKER,
            )
            write_json(
                alias / "summary.json",
                dict(
                    status="FAILED",
                    returncode=0,
                    execution_identity_sha256=identity,
                    error="preserved original audit failure",
                ),
            )
            write_json(alias / "settings.json", cfg)
            write_json(alias.parent / "state.json", dict(status="FAILED"))
            summary.update(
                retrained=False,
                imported_from=str(alias),
                original_summary_sha256=storage.digest(alias / "summary.json"),
            )
        write_json(cell / "summary.json", summary)
        originals[name] = {p.name: p.read_bytes() for p in hf.iterdir()}
    return repo, directory, originals


def raw_paths(repo):
    return sorted(
        p
        for _, rel, step, aliases in storage.CATALOG
        for source in (rel, *aliases)
        for p in (repo / source / "checkpoints" / f"global_step_{step}" / "actor/huggingface").glob(
            "*.safetensors"
        )
    )


def test_check_is_readonly_and_hardlink_population_is_not_double_counted(fixture):
    repo, directory, _ = fixture
    before = {p: p.read_bytes() for p in raw_paths(repo)}
    result = storage.inspect()
    assert len(result["groups"]) == 6
    assert result["unique_shards"] == 12 and result["visible_shard_paths"] == 14
    assert result["unique_bytes"] == sum(p.stat().st_size for p in before) - sum(
        f["bytes"] for f in result["groups"][2]["files"]
    )
    assert not directory.exists()
    assert {p: p.read_bytes() for p in before} == before


def test_pack_proves_exact_restore_without_retiring_and_retire_keeps_old_evidence(fixture):
    repo, directory, originals = fixture
    preserved = {p: p.read_bytes() for p in repo.rglob("*.json")}
    proof = storage.pack(directory)
    assert proof["status"] == "PACK_VERIFIED" and proof["source_shards_deleted"] == 0
    assert len(raw_paths(repo)) == 14
    receipt = storage.retire(directory)
    assert receipt["status"] == "COMPLETE"
    assert len(receipt["deleted_paths"]) == 14
    assert receipt["deleted_unique_shards"] == 12 and not raw_paths(repo)
    assert all(p.read_bytes() == before for p, before in preserved.items())
    for name, _, _, _ in storage.CATALOG:
        restored = directory / "restored_for_test" / name
        archive.restore_model(directory / "archives" / name, restored)
        assert {p.name: p.read_bytes() for p in restored.iterdir()} == originals[name]
    with pytest.raises(RuntimeError, match="already attempted"):
        storage.retire(directory)


@pytest.mark.parametrize("damage", ["extra_link", "missing_alias", "alive", "failed", "symlink"])
def test_ineligible_ownership_never_creates_storage_or_deletes_sources(fixture, damage):
    repo, directory, _ = fixture
    paths = raw_paths(repo)
    if damage == "extra_link":
        os.link(paths[0], repo / "foreign_link")
    elif damage == "missing_alias":
        alias = repo / storage.CATALOG[2][3][0]
        next((alias / "checkpoints/global_step_4/actor/huggingface").glob("*.safetensors")).unlink()
    elif damage == "alive":
        cell = repo / storage.CATALOG[0][1]
        worker = cell / "artifacts/placement/moment_scratch/999999999"
        worker.rename(worker.with_name(str(os.getpid())))
    elif damage == "failed":
        cell = repo / storage.CATALOG[0][1]
        summary = storage.read(cell / "summary.json")
        summary["status"] = "FAILED"
        write_json(cell / "summary.json", summary)
    else:
        source = paths[0]
        moved = source.with_name("hidden_source")
        source.rename(moved)
        source.symlink_to(moved)
    before = {p: p.read_bytes() for p in raw_paths(repo)}
    with pytest.raises((RuntimeError, FileNotFoundError)):
        storage.pack(directory)
    assert {p: p.read_bytes() for p in before} == before and not directory.exists()


@pytest.mark.parametrize("damage", ["archive", "source", "summary", "runtime"])
def test_all_groups_are_reverified_before_first_unlink(fixture, monkeypatch, damage):
    repo, directory, _ = fixture
    storage.pack(directory)
    if damage == "archive":
        packed = next((directory / "archives" / storage.CATALOG[-1][0]).glob("*.xor.zlib"))
        packed.write_bytes(packed.read_bytes()[:-1])
    elif damage == "source":
        path = raw_paths(repo)[-1]
        path.write_bytes(path.read_bytes()[:-1] + b"x")
    elif damage == "summary":
        path = repo / storage.CATALOG[-1][1] / "summary.json"
        data = storage.read(path)
        data["annotation"] = "changed after pack"
        write_json(path, data)
    else:
        monkeypatch.setattr(storage, "source_identity", lambda: {"mutated": "b" * 64})
    before = {p: p.read_bytes() for p in raw_paths(repo)}
    with pytest.raises((RuntimeError, ValueError)):
        storage.retire(directory)
    assert {p: p.read_bytes() for p in before} == before
    assert not (directory / "retirement_receipt.json").exists()


def test_fixed_output_and_fresh_pack_directory_are_enforced(fixture):
    _, directory, _ = fixture
    with pytest.raises(RuntimeError, match="fixed new"):
        storage.pack(directory.with_name("other"))
    storage.pack(directory)
    with pytest.raises(RuntimeError, match="fresh storage"):
        storage.pack(directory)
