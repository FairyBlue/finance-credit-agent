"""CPU-only export identity tests, runnable in staging or after applying the patch."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/agents/launcher.py").is_file())
CANDIDATE = Path(__file__).with_name("local_export.py")
if CANDIDATE.is_file():
    spec = importlib.util.spec_from_file_location("local_export_candidate", CANDIDATE)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
else:
    from src.agents import local_export as helper


def pin(root, *, status="COMPLETE", mutate=None):
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in root.iterdir() if p.name != helper.RECEIPT
    }
    receipt = {"status": status, "files": files}
    if mutate:
        mutate(receipt)
    payload = json.dumps(receipt, sort_keys=True).encode()
    (root / helper.RECEIPT).write_bytes(payload)
    return {
        "source": "local_export", "name": str(root),
        "merge_receipt_sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.fixture
def export(tmp_path):
    root = tmp_path / "merged_model"
    root.mkdir()
    contents = {
        "config.json": {"model_type": "qwen3"},
        "generation_config.json": {"eos_token_id": 2},
        "tokenizer.json": {"version": "1.0"},
        "tokenizer_config.json": {"chat_template": "{{ messages }}"},
    }
    for name, value in contents.items():
        (root / name).write_text(json.dumps(value))
    # Resolver checks identity/inventory; tensor loading remains the model loader's job.
    (root / "model.safetensors").write_bytes(b"fixed-model-fixture")
    return root, pin(root)


def test_single_file_export_returns_absolute_path_and_external_identity(export):
    root, config = export
    assert helper.resolve_local_export(config) == (
        str(root), "local-export-sha256:" + config["merge_receipt_sha256"], root,
    )


def test_sharded_export_and_external_template_are_supported(export):
    root, _ = export
    (root / "model.safetensors").rename(root / "model-00001-of-00002.safetensors")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "weight1": "model-00001-of-00002.safetensors",
            "weight2": "model-00002-of-00002.safetensors",
        },
    }))
    (root / "tokenizer_config.json").write_text("{}")
    (root / "chat_template.jinja").write_text("{{ messages }}")
    assert helper.resolve_local_export(pin(root))[2] == root


@pytest.mark.parametrize("field,value", [
    ("source", "hf"), ("revision", "a" * 40), ("merge_receipt_sha256", None),
    ("merge_receipt_sha256", "A" * 64), ("merge_receipt_sha256", "a" * 63),
    ("name", "relative/export"),
])
def test_explicit_local_source_identity_and_absolute_path_required(export, field, value):
    _, config = export
    config[field] = value
    with pytest.raises(RuntimeError):
        helper.resolve_local_export(config)


def test_receipt_cannot_supply_its_own_identity(export):
    root, original = export
    (root / "model.safetensors").write_bytes(b"changed-model-fixtur")
    pin(root)
    with pytest.raises(RuntimeError, match="receipt SHA256 mismatch"):
        helper.resolve_local_export(original)


@pytest.mark.parametrize("change", ["same_size", "truncate", "delete", "extra"])
def test_missing_modified_or_unrecorded_artifact_rejected(export, change):
    root, config = export
    target = root / "model.safetensors"
    if change == "same_size":
        target.write_bytes(b"X" * target.stat().st_size)
    elif change == "truncate":
        target.write_bytes(b"X")
    elif change == "delete":
        target.unlink()
    else:
        (root / "pytorch_model.bin").write_bytes(b"unrecorded alternative weights")
    with pytest.raises(RuntimeError, match="inventory|SHA256|byte count"):
        helper.resolve_local_export(config)


@pytest.mark.parametrize("change", [
    "status", "missing_config", "missing_template", "shard", "unsafe", "bool_size",
])
def test_internally_incomplete_receipt_is_not_a_loadable_export(export, change):
    root, _ = export
    mutate = None
    status = "COMPLETE"
    if change == "status":
        status = "FAILED"
    elif change == "missing_config":
        (root / "tokenizer_config.json").unlink()
    elif change == "missing_template":
        (root / "tokenizer_config.json").write_text("{}")
    elif change == "shard":
        (root / "model.safetensors").rename(root / "model-00001-of-00002.safetensors")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "model-00002-of-00002.safetensors"},
        }))
    elif change == "unsafe":
        def mutate(receipt):
            receipt["files"].update({"../outside": {"bytes": 1, "sha256": "a" * 64}})
    else:
        def mutate(receipt):
            receipt["files"]["config.json"].update(bytes=True)
    with pytest.raises(RuntimeError):
        helper.resolve_local_export(pin(root, status=status, mutate=mutate))


def test_symlink_artifact_is_not_a_self_contained_export(export, tmp_path):
    root, _ = export
    target = root / "model.safetensors"
    outside = tmp_path / "outside.safetensors"
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(RuntimeError, match="regular local file"):
        helper.resolve_local_export(pin(root))


def test_staged_launcher_branch_and_original_hf_path_both_resolve(export, tmp_path, monkeypatch):
    from src.agents import launcher

    source = inspect.getsource(launcher._resolve_exact_model_snapshot)
    # Exercise the exact six-line integration without touching the frozen file.
    anchor = '    """Resolve only the configured immutable revision from local files."""\n'
    branch = (
        '\n    model = _required(config, "model")\n'
        '    if model.get("source") == "local_export":\n'
        '        from .local_export import resolve_local_export\n'
        '\n'
        '        return resolve_local_export(model)\n'
    )
    if "from .local_export import resolve_local_export" not in source:
        assert source.count(anchor) == 1
        source = source.replace(anchor, anchor + branch)
    monkeypatch.setitem(sys.modules, "src.agents.local_export", helper)
    namespace = {
        "Path": Path, "Any": object, "os": os, "_required": launcher._required,
        "__package__": "src.agents",
    }
    exec(compile(source, str(ROOT / "src/agents/launcher.py"), "exec"), namespace)
    resolve = namespace["_resolve_exact_model_snapshot"]
    root, config = export
    assert resolve({"model": config}) == helper.resolve_local_export(config)
    revision = "a" * 40
    snapshot = tmp_path / "models--Qwen--Qwen3-4B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    hf = {"model": {"name": "Qwen/Qwen3-4B", "revision": revision}}
    assert resolve(hf, cache_root=tmp_path) == ("Qwen/Qwen3-4B", revision, snapshot)
    hf["model"]["revision"] = "b" * 40
    with pytest.raises(RuntimeError, match="exact model revision is not cached"):
        resolve(hf, cache_root=tmp_path)
    hf["model"]["name"] = str(root)
    with pytest.raises(RuntimeError, match="does not match configured exact revision"):
        resolve(hf, cache_root=tmp_path)
