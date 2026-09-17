"""Resolve a completed, externally pinned local HF export without loading models."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECEIPT = "merge_receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED = frozenset({
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
})


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(data: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"local export {name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"local export {name} must be an object")
    return value


def resolve_local_export(model: Mapping[str, Any]) -> tuple[str, str, Path]:
    """Validate coldstart_sft's flat export inventory and return its receipt identity.

    The receipt SHA is supplied by the caller, not learned from the directory.
    No network, tensor deserialization, CUDA, or write operations are performed.
    The export must remain immutable after resolution, as for an HF snapshot.
    """
    if model.get("source") != "local_export":
        raise RuntimeError("local export resolution requires model.source=local_export")
    if model.get("revision") is not None:
        raise RuntimeError(
            "local export must omit model.revision; use model.merge_receipt_sha256"
        )
    expected = model.get("merge_receipt_sha256")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise RuntimeError("model.merge_receipt_sha256 must be 64 lowercase hex characters")
    name = model.get("name") or model.get("train")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("local export requires a nonempty model.name or model.train")
    root = Path(name).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError("local export model path must be an existing absolute directory")
    root = root.resolve()
    receipt_path = root / RECEIPT
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError("local export merge_receipt.json is missing or not a regular file")
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != expected:
        raise RuntimeError("local export merge receipt SHA256 mismatch")
    receipt = _object(receipt_bytes, RECEIPT)
    if receipt.get("status") != "COMPLETE":
        raise RuntimeError("local export merge receipt is not COMPLETE")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("local export receipt requires a nonempty files inventory")
    for relative, evidence in files.items():
        if (
            not isinstance(relative, str) or relative in {"", ".", "..", RECEIPT}
            or Path(relative).name != relative or "/" in relative or "\\" in relative
        ):
            raise RuntimeError("local export receipt must contain only flat artifact filenames")
        if (
            not isinstance(evidence, dict) or set(evidence) != {"bytes", "sha256"}
            or type(evidence["bytes"]) is not int or evidence["bytes"] <= 0
            or not isinstance(evidence["sha256"], str)
            or _SHA256.fullmatch(evidence["sha256"]) is None
        ):
            raise RuntimeError(f"local export inventory entry is invalid: {relative}")
    actual = {path.name for path in root.iterdir() if path.name != RECEIPT}
    if actual != set(files):
        raise RuntimeError("local export directory differs from complete receipt inventory")
    if not _REQUIRED.issubset(files):
        raise RuntimeError("local export receipt is missing model config or tokenizer files")
    for relative, evidence in files.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"local export artifact is not a regular local file: {relative}")
        if path.stat().st_size != evidence["bytes"]:
            raise RuntimeError(f"local export artifact byte count mismatch: {relative}")
        if _digest(path) != evidence["sha256"]:
            raise RuntimeError(f"local export artifact SHA256 mismatch: {relative}")
    # Parse the small config/index files only after their bytes are verified.
    for relative in _REQUIRED:
        _object((root / relative).read_bytes(), relative)
    tokenizer = _object((root / "tokenizer_config.json").read_bytes(), "tokenizer_config.json")
    if "chat_template.jinja" not in files and not tokenizer.get("chat_template"):
        raise RuntimeError("local export has no recorded chat template")
    weights = {name for name in files if name.endswith(".safetensors")}
    index_name = "model.safetensors.index.json"
    if index_name in files:
        index = _object((root / index_name).read_bytes(), index_name)
        mapping = index.get("weight_map")
        if (
            not isinstance(mapping, dict) or not mapping
            or any(not isinstance(key, str) or not key for key in mapping)
            or any(not isinstance(value, str) for value in mapping.values())
            or set(mapping.values()) != weights or "model.safetensors" in weights
        ):
            raise RuntimeError("local export shard index does not cover the recorded weights")
    elif weights != {"model.safetensors"}:
        raise RuntimeError("local export requires one model.safetensors or a complete shard index")
    # Pin both provenance and all recorded artifact hashes through the external receipt SHA.
    return str(root), "local-export-sha256:" + expected, root
