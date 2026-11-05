"""Lossless BF16 actor archives for the new parameter grid only.

Each tensor is XORed against the immutable shared SFT and compressed in bounded
blocks. Packing and restoring are CPU-only. Raw exports are retired only after
all packed shards reconstruct their exact original file hashes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import struct
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.agents import actor_export

PROFILE = "lossless_tensor_xor_v1"
PROFILES = {
    "common_sft_parameter_grid_seed42_v1",
    "common_sft_parameter_grid_seed43_v1",
    "common_sft_parameter_grid_seed42_v2",
    "common_sft_parameter_grid_seed43_v2",
    "shared_recovery_sft_action_pooling_seed42_v1",
    "shared_recovery_sft_action_pooling_seed43_v1",
}
BLOCK_BYTES = 4 * 1024**2
_MAGIC = b"FCAXOR1\n"


def _safe_path(path):
    path = Path(path).absolute()
    if any(part == ".." for part in path.parts) or any(
        parent.is_symlink() for parent in (path, *path.parents)
    ):
        raise RuntimeError("archive paths must not contain symlinks or parent traversal")
    return path


def _file(directory, filename):
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in {"", ".", ".."}
    ):
        raise RuntimeError("unsafe archive filename")
    return _safe_path(directory / filename)


def _read(path):
    try:
        value = json.loads(_safe_path(path).read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("invalid archive JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("archive JSON must be an object")
    return value


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    actor_export._fsync_directory(path.parent)


def _header_layout(header_bytes, file_bytes):
    if len(header_bytes) < 8:
        raise RuntimeError("truncated safetensors header")
    length = int.from_bytes(header_bytes[:8], "little")
    if not 0 < length <= 8 * 1024**2 or len(header_bytes) != length + 8:
        raise RuntimeError("invalid safetensors header length")
    header = json.loads(header_bytes[8:])
    entries = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = item.get("shape"), item.get("data_offsets")
        if (
            item.get("dtype") != "BF16"
            or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or offsets[1] - offsets[0] != 2 * math.prod(shape)
        ):
            raise RuntimeError("archive requires valid matching BF16 tensors")
        entries.append(dict(name=name, shape=shape, start=offsets[0], end=offsets[1]))
    entries.sort(key=lambda row: row["start"])
    cursor = 0
    for row in entries:
        if row["start"] != cursor or row["end"] < cursor:
            raise RuntimeError("safetensors offsets overlap or contain gaps")
        cursor = row["end"]
    if not entries or cursor + len(header_bytes) != file_bytes:
        raise RuntimeError("safetensors file size differs from its payload")
    return entries


def _layout(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < length <= 8 * 1024**2:
            raise RuntimeError("invalid safetensors header length")
        header = prefix + stream.read(length)
    return header, _header_layout(header, path.stat().st_size)


def _model_layout(directory):
    index = _read(directory / "model.safetensors.index.json")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("missing model weight map")
    tensors, shards = {}, {}
    for filename in sorted(set(mapping.values())):
        path = _file(directory, filename)
        if not filename.endswith(".safetensors") or not path.is_file():
            raise RuntimeError("invalid model shard")
        header, entries = _layout(path)
        shards[filename] = dict(header=header, entries=entries)
        for entry in entries:
            name = entry["name"]
            if name in tensors or mapping.get(name) != filename:
                raise RuntimeError("model tensor coverage differs from its index")
            tensors[name] = {**entry, "filename": filename, "header_bytes": len(header)}
    if set(tensors) != set(mapping):
        raise RuntimeError("model index contains absent tensors")
    actual = {path.name for path in directory.glob("*.safetensors")}
    if actual != set(shards):
        raise RuntimeError("model contains unindexed safetensors shards")
    return tensors, shards


def _xor(left, right):
    if len(left) != len(right):
        raise RuntimeError("baseline tensor read is truncated")
    return np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    ).tobytes()


def _aligned_baseline_tensors(baseline_tensors, target_tensors, baseline_config, target_config):
    """Allow only HF's explicitly configured omission of its tied output head."""
    extra = set(target_tensors) - set(baseline_tensors)
    if set(baseline_tensors) - set(target_tensors):
        raise RuntimeError("target and shared SFT tensor contracts differ")
    aligned, aliases = dict(baseline_tensors), {}
    if extra:
        head, embedding = "lm_head.weight", "model.embed_tokens.weight"
        dimensions = [baseline_config.get("vocab_size"), baseline_config.get("hidden_size")]
        if (
            extra != {head}
            or baseline_config.get("tie_word_embeddings") is not True
            or target_config.get("tie_word_embeddings") is not True
            or any(type(n) is not int or n <= 0 for n in dimensions)
            or [target_config.get("vocab_size"), target_config.get("hidden_size")] != dimensions
            or embedding not in baseline_tensors
            or embedding not in target_tensors
            or baseline_tensors[embedding]["shape"] != dimensions
            or target_tensors[embedding]["shape"] != dimensions
            or target_tensors[head]["shape"] != dimensions
        ):
            raise RuntimeError("missing target tensor is not a verified tied output head")
        aligned[head] = baseline_tensors[embedding]
        aliases[head] = embedding
    if any(row["shape"] != aligned[name]["shape"] for name, row in target_tensors.items()):
        raise RuntimeError("target and shared SFT tensor contracts differ")
    return aligned, aliases


def pack_model(model_dir, baseline_dir, archive_dir):
    """Create and verify a new archive; never remove or write either model."""
    model_dir, baseline_dir, archive_dir = map(_safe_path, (model_dir, baseline_dir, archive_dir))
    if archive_dir.exists():
        raise RuntimeError("archive requires a new directory; no implicit retry")
    if (
        model_dir == baseline_dir
        or archive_dir.is_relative_to(model_dir)
        or archive_dir.is_relative_to(baseline_dir)
    ):
        raise RuntimeError("archive must be separate from model and shared SFT")
    original_hf = actor_export._hf_evidence(model_dir, verify_hashes=True)
    target_tensors, target_shards = _model_layout(model_dir)
    baseline_tensors, baseline_shards = _model_layout(baseline_dir)
    baseline_tensors, aliases = _aligned_baseline_tensors(
        baseline_tensors,
        target_tensors,
        _read(baseline_dir / "config.json"),
        _read(model_dir / "config.json"),
    )
    baseline = dict(
        directory=str(baseline_dir),
        index_sha256=actor_export._sha(baseline_dir / "model.safetensors.index.json"),
        config_sha256=actor_export._sha(baseline_dir / "config.json"),
        merge_receipt_sha256=actor_export._sha(_file(baseline_dir, "merge_receipt.json")),
        shards={
            name: dict(
                bytes=(baseline_dir / name).stat().st_size,
                sha256=actor_export._sha(baseline_dir / name),
            )
            for name in baseline_shards
        },
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    metadata = archive_dir / "metadata"
    metadata.mkdir()
    for name in original_hf["files"]:
        if not name.endswith(".safetensors"):
            source, target = _file(model_dir, name), _file(metadata, name)
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
    manifest = dict(
        schema_version=1,
        profile=PROFILE,
        block_bytes=BLOCK_BYTES,
        baseline=baseline,
        tensor_aliases=aliases,
        original_hf=original_hf,
        shards={},
    )
    handles = {name: (baseline_dir / name).open("rb") for name in baseline_shards}
    try:
        for name, shard in target_shards.items():
            path, packed = model_dir / name, archive_dir / (name + ".xor.zlib")
            original_hash = hashlib.sha256(shard["header"])
            stat = path.stat()
            with path.open("rb") as source, packed.open("xb") as output:
                output.write(_MAGIC)
                for entry in shard["entries"]:
                    base = baseline_tensors[entry["name"]]
                    source.seek(len(shard["header"]) + entry["start"])
                    handle = handles[base["filename"]]
                    handle.seek(base["header_bytes"] + base["start"])
                    remaining = entry["end"] - entry["start"]
                    while remaining:
                        size = min(BLOCK_BYTES, remaining)
                        block = source.read(size)
                        if len(block) != size:
                            raise RuntimeError("target tensor read is truncated")
                        original_hash.update(block)
                        compressed = zlib.compress(_xor(block, handle.read(size)), level=1)
                        output.write(struct.pack("<II", size, len(compressed)))
                        output.write(compressed)
                        remaining -= size
                output.flush()
                os.fsync(output.fileno())
            if original_hash.hexdigest() != original_hf["sha256"][name]:
                raise RuntimeError("source model changed during packing")
            manifest["shards"][name] = dict(
                header_base64=base64.b64encode(shard["header"]).decode(),
                original_bytes=stat.st_size,
                original_sha256=original_hash.hexdigest(),
                original_device=stat.st_dev,
                original_inode=stat.st_ino,
                packed_file=packed.name,
                packed_bytes=packed.stat().st_size,
                packed_sha256=actor_export._sha(packed),
            )
    finally:
        for handle in handles.values():
            handle.close()
    actor_export._fsync_directory(metadata)
    _write(archive_dir / "manifest.json", manifest)
    validate_archive(archive_dir)
    _write(
        archive_dir / "verification.json",
        dict(
            status="PACK_VERIFIED",
            profile=PROFILE,
            manifest_sha256=actor_export._sha(archive_dir / "manifest.json"),
            exact_original_shard_hashes_verified=True,
        ),
    )
    return dict(
        status="PACK_VERIFIED",
        profile=PROFILE,
        archive=str(archive_dir),
        original_bytes=sum(r["original_bytes"] for r in manifest["shards"].values()),
        packed_bytes=sum(r["packed_bytes"] for r in manifest["shards"].values()),
    )


def _validated_manifest(archive_dir):
    manifest = _read(archive_dir / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("profile") != PROFILE
        or manifest.get("block_bytes") != BLOCK_BYTES
    ):
        raise RuntimeError("invalid actor archive profile")
    baseline = manifest["baseline"]
    directory = _safe_path(baseline["directory"])
    for filename, digest in (
        ("model.safetensors.index.json", baseline["index_sha256"]),
        ("config.json", baseline["config_sha256"]),
        ("merge_receipt.json", baseline["merge_receipt_sha256"]),
    ):
        if actor_export._sha(_file(directory, filename)) != digest:
            raise RuntimeError("shared SFT metadata changed after packing")
    tensors, shards = _model_layout(directory)
    if set(shards) != set(baseline["shards"]):
        raise RuntimeError("shared SFT shard inventory changed")
    for name, row in baseline["shards"].items():
        path = _file(directory, name)
        if path.stat().st_size != row["bytes"] or actor_export._sha(path) != row["sha256"]:
            raise RuntimeError("shared SFT shard changed after packing")
    tensors, aliases = _aligned_baseline_tensors(
        tensors,
        manifest["original_hf"]["tensors"],
        _read(directory / "config.json"),
        _read(archive_dir / "metadata/config.json"),
    )
    if aliases != manifest.get("tensor_aliases"):
        raise RuntimeError("archive tied-tensor mapping differs from verified HF metadata")
    return manifest, directory, tensors


def _replay(archive_dir, name, row, tensors, handles, destination=None):
    _file(archive_dir, name)
    packed = _file(archive_dir, row["packed_file"])
    if (
        packed.stat().st_size != row["packed_bytes"]
        or actor_export._sha(packed) != row["packed_sha256"]
    ):
        raise RuntimeError("packed actor shard changed or is truncated")
    header = base64.b64decode(row["header_base64"], validate=True)
    entries = _header_layout(header, row["original_bytes"])
    digest = hashlib.sha256(header)
    if destination is not None:
        destination.write(header)
    with packed.open("rb") as source:
        if source.read(len(_MAGIC)) != _MAGIC:
            raise RuntimeError("invalid actor XOR stream")
        for entry in entries:
            base = tensors.get(entry["name"])
            if base is None or entry["shape"] != base["shape"]:
                raise RuntimeError("packed tensor differs from shared SFT")
            handle = handles[base["filename"]]
            handle.seek(base["header_bytes"] + base["start"])
            remaining = entry["end"] - entry["start"]
            while remaining:
                frame = source.read(8)
                if len(frame) != 8:
                    raise RuntimeError("truncated actor XOR frame")
                size, compressed_size = struct.unpack("<II", frame)
                if (
                    size != min(BLOCK_BYTES, remaining)
                    or not 0 < compressed_size <= BLOCK_BYTES + 65536
                ):
                    raise RuntimeError("invalid bounded actor XOR frame")
                compressed = source.read(compressed_size)
                decoder = zlib.decompressobj()
                delta = decoder.decompress(compressed, size + 1)
                if (
                    len(compressed) != compressed_size
                    or len(delta) != size
                    or not decoder.eof
                    or decoder.unused_data
                    or decoder.unconsumed_tail
                ):
                    raise RuntimeError("invalid or truncated compressed actor block")
                block = _xor(delta, handle.read(size))
                digest.update(block)
                if destination is not None:
                    destination.write(block)
                remaining -= size
        if source.read(1):
            raise RuntimeError("unexpected bytes after packed actor tensors")
    if digest.hexdigest() != row["original_sha256"]:
        raise RuntimeError("reconstructed actor bytes differ from original export")


def _replay_all(archive_dir, manifest, baseline_dir, tensors, output_dir=None):
    handles = {name: (baseline_dir / name).open("rb") for name in manifest["baseline"]["shards"]}
    try:
        for name, row in manifest["shards"].items():
            if output_dir is None:
                _replay(archive_dir, name, row, tensors, handles)
            else:
                with _file(output_dir, name).open("xb") as destination:
                    _replay(archive_dir, name, row, tensors, handles, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
    finally:
        for handle in handles.values():
            handle.close()


def validate_archive(archive_dir):
    """Replay every shard to its original SHA without creating full raw files."""
    archive_dir = _safe_path(archive_dir)
    manifest, baseline, tensors = _validated_manifest(archive_dir)
    original = manifest["original_hf"]
    if set(manifest["shards"]) != {
        name for name in original["files"] if name.endswith(".safetensors")
    }:
        raise RuntimeError("archive shard inventory differs from original export")
    for name, row in manifest["shards"].items():
        if (
            row["original_bytes"] != original["files"][name]
            or row["original_sha256"] != original["sha256"][name]
        ):
            raise RuntimeError("archive original shard identity differs")
    for name in original["files"]:
        if not name.endswith(".safetensors"):
            path = _file(archive_dir / "metadata", name)
            if (
                path.stat().st_size != original["files"][name]
                or actor_export._sha(path) != original["sha256"][name]
            ):
                raise RuntimeError("archived HF metadata changed")
    _replay_all(archive_dir, manifest, baseline, tensors)
    return dict(
        status="PACK_VERIFIED",
        profile=PROFILE,
        manifest_sha256=actor_export._sha(archive_dir / "manifest.json"),
    )


def restore_model(archive_dir, output_dir):
    """Rebuild the exact HF files in a new directory for normal model loading."""
    archive_dir, output_dir = map(_safe_path, (archive_dir, output_dir))
    if output_dir.exists():
        raise RuntimeError("restore requires a new directory; no overwrite or retry")
    manifest, baseline, tensors = _validated_manifest(archive_dir)
    if output_dir.is_relative_to(archive_dir) or output_dir.is_relative_to(baseline):
        raise RuntimeError("restore output must be separate from archive and shared SFT")
    output_dir.mkdir(parents=True, exist_ok=False)
    _replay_all(archive_dir, manifest, baseline, tensors, output_dir)
    for name in manifest["original_hf"]["files"]:
        if not name.endswith(".safetensors"):
            shutil.copyfile(_file(archive_dir / "metadata", name), _file(output_dir, name))
            with (output_dir / name).open("rb") as stream:
                os.fsync(stream.fileno())
    actor_export._fsync_directory(output_dir)
    if actor_export._hf_evidence(output_dir, verify_hashes=True) != manifest["original_hf"]:
        raise RuntimeError("restored HF model differs from original export")
    return dict(
        status="COMPLETE",
        profile=PROFILE,
        model_directory=str(output_dir),
        original_shard_hashes_verified=True,
    )


def _worker_stopped(cell, identity):
    owners = list((cell / "artifacts/placement/moment_scratch").glob("*/*/owner.json"))
    if len(owners) != 2 or {p.parent.name for p in owners} != {"actor", "critic"}:
        raise RuntimeError("archive requires both retired optimizer ownership records")
    for path in owners:
        _safe_path(path)
        owner = _read(path)
        pid = path.parent.parent.name
        if (
            not pid.isdecimal()
            or owner.get("identity") != identity
            or owner.get("role") != path.parent.name
            or (Path("/proc") / pid).exists()
        ):
            raise RuntimeError("actor archive worker is alive or ownership differs")


def archive_completed_cell(seed_base, spec, summary):
    """Retire only a fully verified new-grid export, preserving a lossless archive."""
    if spec.zero:
        return {"status": "SKIPPED", "reason": "SFT baseline has no new actor export"}
    seed_base = _safe_path(seed_base)
    cell = _file(seed_base, spec.cell_id)
    settings, contract = _read(seed_base / "settings.json"), _read(seed_base / "run_contract.json")
    if (
        contract.get("profile") not in PROFILES
        or settings.get("training_coverage", {}).get("profile") != contract["profile"]
        or contract.get("inputs", {}).get("settings.json")
        != actor_export._sha(seed_base / "settings.json")
        or summary.get("status") != "COMPLETE"
        or summary.get("returncode") != 0
        or summary.get("actual_rounds") != spec.rounds
        or summary.get("cell") != asdict(spec)
        or _read(cell / "summary.json") != summary
        or summary.get("moment_pruning", {}).get("status") != "COMPLETE"
    ):
        raise RuntimeError("only fully completed new-grid cells may archive their actor")
    identity = summary["execution_identity_sha256"]
    pruning = _read(cell / "moment_pruning_receipt.json")
    if pruning.get("status") != "COMPLETE" or pruning.get("execution_identity_sha256") != identity:
        raise RuntimeError("actor archive requires successful current-cell moment retirement")
    _worker_stopped(cell, identity)
    checkpoint = cell / "checkpoints" / f"global_step_{spec.rounds}"
    actor_export.validate_export_artifact(checkpoint, identity_sha256=identity, verify_hashes=True)
    model = settings["models"]["screening"]
    baseline = _safe_path(model["name"])
    if model.get("source") != "local_export" or actor_export._sha(
        baseline / "merge_receipt.json"
    ) != model.get("merge_receipt_sha256"):
        raise RuntimeError("actor archive shared SFT is not pinned")
    source = checkpoint / "actor/huggingface"
    archive = cell / "actor_archive"
    receipt_path = cell / "actor_archive_receipt.json"
    if receipt_path.exists():
        raise RuntimeError("actor archive retirement already attempted; no retry")
    result = pack_model(source, baseline, archive)
    manifest = _read(archive / "manifest.json")
    # Every source is rechecked before any unlink; a packing/verification failure
    # leaves all original raw actor shards in place.
    actor_export.validate_export_artifact(checkpoint, identity_sha256=identity, verify_hashes=True)
    _worker_stopped(cell, identity)
    for name, row in manifest["shards"].items():
        path = _file(source, name)
        stat = path.stat()
        if (
            stat.st_nlink != 1
            or stat.st_dev != row["original_device"]
            or stat.st_ino != row["original_inode"]
            or stat.st_size != row["original_bytes"]
        ):
            raise RuntimeError("actor source ownership changed before retirement")
    receipt = {
        **result,
        "status": "PACK_VERIFIED",
        "execution_identity_sha256": identity,
        "cell": spec.cell_id,
        "manifest_sha256": actor_export._sha(archive / "manifest.json"),
        "deleted_files": [],
        "deleted_bytes": 0,
        "restore_required": True,
    }
    _write(receipt_path, receipt)
    try:
        for name, row in manifest["shards"].items():
            path = _file(source, name)
            stat = path.stat()
            if (
                stat.st_nlink != 1
                or (stat.st_dev, stat.st_ino, stat.st_size)
                != (row["original_device"], row["original_inode"], row["original_bytes"])
                or actor_export._sha(path) != row["original_sha256"]
            ):
                raise RuntimeError("actor source changed after pack verification")
            path.unlink()
            receipt["deleted_files"].append(name)
            receipt["deleted_bytes"] += row["original_bytes"]
            _write(receipt_path, receipt)
        actor_export._fsync_directory(source)
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _write(receipt_path, receipt)
    return dict(
        status="COMPLETE",
        profile=PROFILE,
        receipt=str(receipt_path),
        archive=str(archive),
        original_bytes=result["original_bytes"],
        packed_bytes=result["packed_bytes"],
        deleted_bytes=receipt["deleted_bytes"],
        restore_required=True,
    )


def restore_cell_archive(cell, output_dir):
    cell = _safe_path(cell)
    receipt = _read(cell / "actor_archive_receipt.json")
    archive = cell / "actor_archive"
    if (
        receipt.get("status") not in {"COMPLETE", "PACK_VERIFIED", "FAILED"}
        or receipt.get("profile") != PROFILE
        or receipt.get("manifest_sha256") != actor_export._sha(archive / "manifest.json")
    ):
        raise RuntimeError("cell actor archive receipt differs")
    return restore_model(archive, output_dir)
