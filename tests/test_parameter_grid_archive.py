"""CPU fixtures verify exact storage round trips and fail-closed retirement."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from src.agents import actor_export
from src.agents import parameter_grid_archive as archive
from src.agents.training_tuning import CellSpec


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_shard(path, tensors):
    header, payload = {}, b""
    for name, data in tensors.items():
        header[name] = dict(
            dtype="BF16",
            shape=[len(data) // 2],
            data_offsets=[len(payload), len(payload) + len(data)],
        )
        payload += data
    encoded = json.dumps(header).encode() + b" " * 7
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def write_model(directory, shards):
    directory.mkdir(parents=True)
    mapping, total = {}, 0
    for filename, tensors in shards.items():
        write_shard(directory / filename, tensors)
        mapping.update({name: filename for name in tensors})
        total += sum(len(data) for data in tensors.values())
    write_json(
        directory / "model.safetensors.index.json",
        dict(metadata=dict(total_size=total), weight_map=mapping),
    )
    write_json(directory / "config.json", dict(torch_dtype="bfloat16"))
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (directory / filename).write_text("fixture metadata\n")


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "BLOCK_BYTES", 16)
    baseline, target = tmp_path / "baseline", tmp_path / "target"
    a, b = bytes(range(64)), bytes(reversed(range(64)))
    write_model(baseline, {"base.safetensors": {"tensor.b": b, "tensor.a": a}})
    write_json(baseline / "merge_receipt.json", dict(profile="fixture"))
    changed = bytearray(a)
    changed[3] ^= 1
    changed[53] ^= 5
    write_model(
        target,
        {
            "target-b.safetensors": {"tensor.b": b},
            "target-a.safetensors": {"tensor.a": bytes(changed)},
        },
    )
    return baseline, target, tmp_path / "packed"


def test_differently_sharded_models_restore_exact_original_bytes(models, tmp_path):
    baseline, target, packed = models
    original = {p.name: p.read_bytes() for p in target.iterdir()}
    baseline_before = {p.name: p.read_bytes() for p in baseline.iterdir()}
    result = archive.pack_model(target, baseline, packed)
    assert result["status"] == "PACK_VERIFIED"
    restored = tmp_path / "restored"
    assert archive.restore_model(packed, restored)["status"] == "COMPLETE"
    assert {p.name: p.read_bytes() for p in restored.iterdir()} == original
    assert {p.name: p.read_bytes() for p in target.iterdir()} == original
    assert {p.name: p.read_bytes() for p in baseline.iterdir()} == baseline_before


@pytest.mark.parametrize("damage", ["packed", "baseline", "metadata", "path"])
def test_corruption_is_detected_before_raw_exports_are_touched(models, damage):
    baseline, target, packed = models
    original = {p.name: p.read_bytes() for p in target.iterdir()}
    archive.pack_model(target, baseline, packed)
    if damage == "packed":
        path = next(packed.glob("*.xor.zlib"))
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "baseline":
        path = baseline / "base.safetensors"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif damage == "metadata":
        (packed / "metadata/config.json").write_text("changed")
    else:
        manifest = json.loads((packed / "manifest.json").read_text())
        next(iter(manifest["shards"].values()))["packed_file"] = "../outside"
        write_json(packed / "manifest.json", manifest)
    with pytest.raises(RuntimeError):
        archive.validate_archive(packed)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == original


def test_symlinks_and_reused_directories_are_rejected(models, tmp_path):
    baseline, target, packed = models
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinks"):
        archive.pack_model(alias, baseline, packed)
    archive.pack_model(target, baseline, packed)
    with pytest.raises(RuntimeError, match="new directory"):
        archive.pack_model(target, baseline, packed)
    with pytest.raises(RuntimeError, match="new directory"):
        archive.restore_model(packed, target)


def test_tensor_shape_or_dtype_mismatch_is_rejected(models):
    baseline, target, packed = models
    write_shard(target / "target-b.safetensors", {"tensor.b": bytes(66)})
    with pytest.raises(RuntimeError, match="tensor contracts differ|total size"):
        archive.pack_model(target, baseline, packed)


def completed_cell(tmp_path, models):
    baseline, target, _ = models
    seed = tmp_path / "seed"
    spec = CellSpec("m1_lr1e7", "M1", 0.0, 1e-6, 4)
    cell = seed / spec.cell_id
    checkpoint = cell / "checkpoints/global_step_4"
    hf = checkpoint / "actor/huggingface"
    hf.parent.mkdir(parents=True)
    target.rename(hf)
    identity = "a" * 64
    original_hf = actor_export._hf_evidence(hf, verify_hashes=True)
    receipt = dict(
        schema_version=1,
        artifact_profile="actor_bf16_hf_v1",
        identity_sha256=identity,
        global_step=4,
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
        hf=original_hf,
    )
    write_json(hf.parent / actor_export.ACTOR_RECEIPT, receipt)
    marker = actor_export.export_completion_document(
        checkpoint,
        profile="actor_bf16_hf_v1",
        identity=identity,
        step=4,
    )
    write_json(checkpoint / actor_export.EXPORT_COMPLETION_MARKER, marker)
    write_json(
        cell / "moment_pruning_receipt.json",
        dict(
            status="COMPLETE",
            execution_identity_sha256=identity,
        ),
    )
    for role in ("actor", "critic"):
        write_json(
            cell / f"artifacts/placement/moment_scratch/999999999/{role}/owner.json",
            dict(identity=identity, role=role),
        )
    summary = dict(
        status="COMPLETE",
        returncode=0,
        actual_rounds=4,
        cell=asdict(spec),
        execution_identity_sha256=identity,
        moment_pruning=dict(status="COMPLETE"),
    )
    write_json(cell / "summary.json", summary)
    settings = dict(
        training_coverage=dict(profile="common_sft_parameter_grid_seed42_v1"),
        models=dict(
            screening=dict(
                source="local_export",
                name=str(baseline),
                merge_receipt_sha256=actor_export._sha(baseline / "merge_receipt.json"),
            )
        ),
    )
    write_json(seed / "settings.json", settings)
    write_json(
        seed / "run_contract.json",
        dict(
            profile="common_sft_parameter_grid_seed42_v1",
            inputs={"settings.json": actor_export._sha(seed / "settings.json")},
        ),
    )
    return seed, spec, summary, cell, hf, original_hf


def test_completed_new_cell_retires_only_raw_shards_and_restores(models, tmp_path):
    seed, spec, summary, cell, hf, original_hf = completed_cell(tmp_path, models)
    result = archive.archive_completed_cell(seed, spec, summary)
    assert result["status"] == "COMPLETE" and result["restore_required"] is True
    assert not list(hf.glob("*.safetensors"))
    assert (hf / "model.safetensors.index.json").exists()
    assert (hf.parent / actor_export.ACTOR_RECEIPT).exists()
    assert (hf.parent.parent / actor_export.EXPORT_COMPLETION_MARKER).exists()
    restored = tmp_path / "reconstructed"
    archive.restore_cell_archive(cell, restored)
    assert actor_export._hf_evidence(restored, verify_hashes=True) == original_hf


@pytest.mark.parametrize(
    "damage", ["failed", "old_profile", "alive_worker", "unretired", "linked_raw", "pack_failure"]
)
def test_ineligible_or_failed_archive_preserves_all_raw_shards(
    models, tmp_path, monkeypatch, damage
):
    seed, spec, summary, cell, hf, _ = completed_cell(tmp_path, models)
    original = {p.name: p.read_bytes() for p in hf.glob("*.safetensors")}
    if damage == "failed":
        summary["status"] = "FAILED"
        write_json(cell / "summary.json", summary)
    elif damage == "old_profile":
        write_json(seed / "run_contract.json", dict(profile="common_sft_low_memory_comparison_v1"))
    elif damage == "alive_worker":
        source = cell / "artifacts/placement/moment_scratch/999999999"
        source.rename(source.with_name(str(os.getpid())))
    elif damage == "unretired":
        summary["moment_pruning"]["status"] = "FAILED"
        write_json(cell / "summary.json", summary)
    elif damage == "linked_raw":
        os.link(next(hf.glob("*.safetensors")), tmp_path / "another-link")
    else:

        def fail(*args, **kwargs):
            raise RuntimeError("injected pack verification failure")

        monkeypatch.setattr(archive, "validate_archive", fail)
    with pytest.raises(RuntimeError):
        archive.archive_completed_cell(seed, spec, summary)
    assert {p.name: p.read_bytes() for p in hf.glob("*.safetensors")} == original


def test_zero_shot_has_no_new_export_to_archive(tmp_path):
    spec = CellSpec("sft0", "M1", 0.0, 1e-6, 0)
    assert archive.archive_completed_cell(tmp_path / "absent", spec, {})["status"] == "SKIPPED"


def test_interrupted_raw_retirement_keeps_verified_recoverable_archive(
    models, tmp_path, monkeypatch
):
    seed, spec, summary, cell, hf, original_hf = completed_cell(tmp_path, models)
    original_unlink = Path.unlink
    calls = 0

    def interrupt_second_shard(path, *args, **kwargs):
        nonlocal calls
        if path.parent == hf and path.suffix == ".safetensors":
            calls += 1
            if calls == 2:
                raise OSError("injected filesystem retirement failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_second_shard)
    with pytest.raises(OSError, match="injected filesystem"):
        archive.archive_completed_cell(seed, spec, summary)
    receipt = json.loads((cell / "actor_archive_receipt.json").read_text())
    assert receipt["status"] == "FAILED" and len(receipt["deleted_files"]) == 1
    assert len(list(hf.glob("*.safetensors"))) == 1
    restored = tmp_path / "restored-after-interruption"
    archive.restore_cell_archive(cell, restored)
    assert actor_export._hf_evidence(restored, verify_hashes=True) == original_hf


def reshape_shard(path, shapes):
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + size])
    for name, shape in shapes.items():
        header[name]["shape"] = shape
    encoded = json.dumps(header).encode() + b" " * 3
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data[8 + size :])


@pytest.fixture
def tied_models(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "BLOCK_BYTES", 16)
    baseline, target, packed = (
        tmp_path / "tied-base",
        tmp_path / "tied-target",
        tmp_path / "tied-pack",
    )
    embedding = bytes(range(64))
    write_model(
        baseline,
        {
            "base.safetensors": {
                "model.embed_tokens.weight": embedding,
                "model.norm.weight": bytes(8),
            }
        },
    )
    write_model(
        target,
        {
            "embedding.safetensors": {"model.embed_tokens.weight": embedding},
            "head.safetensors": {"lm_head.weight": bytes(reversed(range(64)))},
            "norm.safetensors": {"model.norm.weight": bytes(8)},
        },
    )
    reshape_shard(baseline / "base.safetensors", {"model.embed_tokens.weight": [4, 8]})
    reshape_shard(target / "embedding.safetensors", {"model.embed_tokens.weight": [4, 8]})
    reshape_shard(target / "head.safetensors", {"lm_head.weight": [4, 8]})
    for directory in (baseline, target):
        write_json(
            directory / "config.json",
            dict(
                torch_dtype="bfloat16",
                tie_word_embeddings=True,
                vocab_size=4,
                hidden_size=8,
            ),
        )
    write_json(baseline / "merge_receipt.json", dict(profile="fixture"))
    return baseline, target, packed


def test_hf_omitted_tied_head_preserves_independent_exported_tensor(tied_models, tmp_path):
    baseline, target, packed = tied_models
    original = {p.name: p.read_bytes() for p in target.iterdir()}
    archive.pack_model(target, baseline, packed)
    manifest = json.loads((packed / "manifest.json").read_text())
    assert manifest["tensor_aliases"] == {"lm_head.weight": "model.embed_tokens.weight"}
    restored = tmp_path / "tied-restored"
    archive.restore_model(packed, restored)
    assert {p.name: p.read_bytes() for p in restored.iterdir()} == original
    assert (restored / "head.safetensors").read_bytes() != (
        restored / "embedding.safetensors"
    ).read_bytes()


@pytest.mark.parametrize(
    "damage", ["baseline_untied", "target_untied", "dimensions", "head_shape", "unknown_tensor"]
)
def test_tied_head_alias_requires_exact_metadata_and_dimensions(tied_models, damage):
    baseline, target, packed = tied_models
    if damage in {"baseline_untied", "target_untied", "dimensions"}:
        directory = baseline if damage == "baseline_untied" else target
        config = json.loads((directory / "config.json").read_text())
        if damage == "dimensions":
            config["hidden_size"] = 16
        else:
            config["tie_word_embeddings"] = False
        write_json(directory / "config.json", config)
    elif damage == "head_shape":
        reshape_shard(target / "head.safetensors", {"lm_head.weight": [8, 4]})
    else:
        path = target / "head.safetensors"
        write_shard(path, {"unknown.weight": bytes(64)})
        index = json.loads((target / "model.safetensors.index.json").read_text())
        index["weight_map"]["unknown.weight"] = index["weight_map"].pop("lm_head.weight")
        write_json(target / "model.safetensors.index.json", index)
    with pytest.raises(RuntimeError, match="verified tied output head"):
        archive.pack_model(target, baseline, packed)
    assert not packed.exists()
    assert len(list(target.glob("*.safetensors"))) == 3


def test_restoring_tied_archive_checks_baseline_config_hash(tied_models, tmp_path):
    baseline, target, packed = tied_models
    archive.pack_model(target, baseline, packed)
    config = json.loads((baseline / "config.json").read_text())
    config["tie_word_embeddings"] = False
    write_json(baseline / "config.json", config)
    with pytest.raises(RuntimeError, match="SFT metadata changed"):
        archive.restore_model(packed, tmp_path / "refused-restore")
