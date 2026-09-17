"""Bounded, fresh-only actor inference exports for the single-rank FSDP runtime.

Training parameters remain FP32. Only detached CPU copies are rounded to BF16.
No critic, optimizer, scheduler, dataloader or resume state is saved. Imports do
not initialize torch or a device; the production installer is explicitly gated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any

ACTOR_EXPORT_PROFILE = "actor_bf16_hf_v1"
DIAGNOSTIC_PROFILE = "critic_fit_diagnostic_v1"
EXPORT_PROFILES = frozenset({ACTOR_EXPORT_PROFILE, DIAGNOSTIC_PROFILE})
ACTOR_SAVE_CONTENT = "finance_actor_bf16_hf"
ARTIFACT_PROFILE_ENV = "FINANCE_SCREENING_ARTIFACT_PROFILE"
EXPORT_COMPLETION_MARKER = "actor_export_complete.json"
ACTOR_RECEIPT = "actor_export_receipt.json"
_GIB = 1024**3
_BLOCK = 8 * 1024**2
_CONVERSION_ELEMENTS = 8 * 1024**2


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    # New run paths only: partial writes remain inspectable and are never reused.
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inventory(directory: Path) -> dict[str, int]:
    result = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("actor export refuses symlinks")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = path.stat().st_size
    return result


def _safetensor_evidence(path: Path, *, verify_hashes: bool) -> tuple[dict[str, Any], str]:
    """Inspect headers and optionally hash payloads in one bounded sequential read."""
    size = path.stat().st_size
    whole = hashlib.sha256()
    with path.open("rb") as handle:
        prefix = handle.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < length <= min(size - 8, 8 * 1024**2):
            raise RuntimeError("invalid actor safetensors header length")
        encoded = handle.read(length)
        header = json.loads(encoded)
        whole.update(prefix)
        whole.update(encoded)
        entries = []
        for name, value in header.items():
            if name == "__metadata__":
                continue
            shape, offsets = value.get("shape"), value.get("data_offsets")
            if (
                value.get("dtype") != "BF16"
                or not isinstance(shape, list)
                or any(type(x) is not int or x < 0 for x in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or offsets[1] - offsets[0] != 2 * math.prod(shape)
            ):
                raise RuntimeError("invalid BF16 actor tensor header")
            entries.append((offsets[0], offsets[1], name, shape))
        cursor = 0
        result = {}
        for start, end, name, shape in sorted(entries):
            if start != cursor or end < start:
                raise RuntimeError("actor payload offsets overlap or contain gaps")
            item: dict[str, Any] = {"shape": shape, "dtype": "BF16"}
            if verify_hashes:
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    block = handle.read(min(_BLOCK, remaining))
                    if not block:
                        raise RuntimeError("truncated actor tensor payload")
                    digest.update(block)
                    whole.update(block)
                    remaining -= len(block)
                item["sha256"] = digest.hexdigest()
            result[name] = item
            cursor = end
        if not result or cursor + length + 8 != size:
            raise RuntimeError("actor safetensors payload size mismatch")
    return result, whole.hexdigest() if verify_hashes else ""


def _hf_evidence(directory: Path, *, verify_hashes: bool) -> dict[str, Any]:
    if directory.is_symlink():
        raise RuntimeError("actor HF directory must not be redirected")
    files = _inventory(directory)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if files.get(name, 0) <= 0:
            raise RuntimeError(f"missing actor HF metadata: {name}")
    config = _json(directory / "config.json")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise RuntimeError("actor inference configuration must declare bfloat16")
    index = _json(directory / "model.safetensors.index.json")
    actual, tensors, hashes = {}, {}, {}
    payload_bytes = 0
    for filename in sorted(files):
        if filename.endswith(".safetensors"):
            shard, digest = _safetensor_evidence(
                directory / filename, verify_hashes=verify_hashes
            )
            for name, info in shard.items():
                if name in tensors:
                    raise RuntimeError("duplicate actor tensor in exported shards")
                tensors[name], actual[name] = info, filename
                payload_bytes += 2 * math.prod(info["shape"])
            if verify_hashes:
                hashes[filename] = digest
        else:
            hashes[filename] = _sha(directory / filename)
    if not tensors or index.get("weight_map") != actual:
        raise RuntimeError("actor weight index differs from shard tensor coverage")
    if index.get("metadata", {}).get("total_size") != payload_bytes:
        raise RuntimeError("actor weight index total size mismatch")
    return {"files": files, "sha256": hashes, "tensors": tensors}


def _cpu_tensor(source: Any, admit: Callable[[int], None]) -> Any:
    """Copy/cast in small chunks; never create a full GPU/CPU FP32 duplicate."""
    import torch

    if source.layout != torch.strided or not source.is_contiguous():
        raise RuntimeError("actor export requires contiguous dense source tensors")
    if source.dtype != torch.float32:
        raise RuntimeError("actor export source must be FP32 training parameters")
    size = source.numel() * 2
    admit(size + _CONVERSION_ELEMENTS * 6)
    result = torch.empty(source.shape, dtype=torch.bfloat16, device="cpu", pin_memory=False)
    target, flat = result.view(-1), source.detach().view(-1)
    for start in range(0, source.numel(), _CONVERSION_ELEMENTS):
        stop = min(start + _CONVERSION_ELEMENTS, source.numel())
        converted = flat[start:stop].to(device="cpu", dtype=torch.bfloat16, non_blocking=False)
        if not bool(torch.isfinite(converted).all()):
            raise RuntimeError("actor export contains a nonfinite BF16 value")
        target[start:stop].copy_(converted)
        del converted
    return result


def _tensor_sha(tensor: Any) -> str:
    import torch

    # NumPy has no BF16 dtype; this byte view does not copy the tensor.
    view = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), _BLOCK):
        digest.update(view[start:start + _BLOCK])
    return digest.hexdigest()


def _write_weight_shards(
    tensors: Mapping[str, Any], directory: Path, *,
    admit: Callable[[int], None], max_shard_bytes: int = _GIB,
) -> dict[str, Any]:
    """Write complete tensors in bounded shards, with per-tensor cast receipts."""
    from safetensors.torch import save_file

    if type(max_shard_bytes) is not int or not 0 < max_shard_bytes <= _GIB:
        raise RuntimeError("actor export shards must be at most one GiB")
    plans: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name, tensor in sorted(tensors.items()):
        size = tensor.numel() * 2
        if size <= 0 or size > max_shard_bytes:
            raise RuntimeError("actor tensor cannot fit the configured shard bound")
        if current and current_size + size > max_shard_bytes:
            plans.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        plans.append(current)
    if not plans:
        raise RuntimeError("actor export has no tensors")
    manifest, mapping = {}, {}
    total = 0
    for number, names in enumerate(plans, 1):
        filename = f"model-{number:05d}-of-{len(plans):05d}.safetensors"
        shard = {}
        for name in names:
            tensor = _cpu_tensor(tensors[name], admit)
            manifest[name] = {
                "shape": list(tensor.shape), "dtype": "BF16", "sha256": _tensor_sha(tensor),
            }
            shard[name], mapping[name] = tensor, filename
            total += tensor.numel() * 2
        del tensor
        # safetensors may materialize one payload-sized bytes copy while writing.
        admit(sum(t.numel() * 2 for t in shard.values()))
        path = directory / filename
        if path.exists():
            raise RuntimeError("actor export refuses to overwrite an existing shard")
        save_file(shard, str(path), metadata={"format": "pt"})
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        del shard
    _write_json(directory / "model.safetensors.index.json", {
        "metadata": {"total_size": total}, "weight_map": mapping,
    })
    return manifest


def _clean_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")


def _expected_shapes(config: Any) -> dict[str, list[int]]:
    import torch
    from accelerate import init_empty_weights  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM

    # A meta architecture supplies the HF contract without allocating model weights.
    with torch.random.fork_rng(devices=[]), init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(copy.deepcopy(config), torch_dtype=torch.bfloat16)
    shapes = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    del model
    return shapes


def _export_manager(manager: Any, local_path: str, global_step: int) -> None:
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from src.agents.screening_placement import _REGISTRY, admit_host_transfer

    registry = _REGISTRY
    if registry is None or not registry.busy:
        raise RuntimeError("actor export requires the active guarded placement save boundary")
    worker = registry.workers.get("actor")
    if (
        worker is None or worker.checkpoint_manager is not manager
        or worker.actor_module_fsdp is not manager.model
        or manager.rank != 0 or manager.world_size != 1
        or not isinstance(manager.model, FSDP)
        or list(manager.checkpoint_load_contents)
    ):
        raise RuntimeError("actor export manager does not match the registered fresh actor")
    registry.check_identity()
    model = manager.model
    registry.backend.validate_training(model)
    config = model._fsdp_wrapped_module.config
    if config.model_type != "qwen3" or config.architectures != ["Qwen3ForCausalLM"]:
        raise RuntimeError("actor export currently supports the registered Qwen3 causal model")
    expected = _expected_shapes(config)
    identity = os.environ.get("FINANCE_SCREENING_IDENTITY_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise RuntimeError("actor export requires its frozen execution identity")
    directory = Path(local_path)
    if not directory.is_absolute() or directory.name != "actor" or directory.is_symlink():
        raise RuntimeError("actor export requires a new absolute actor directory")
    if directory.parent.name != f"global_step_{global_step}" or global_step <= 0:
        raise RuntimeError("actor export path does not match the updated policy step")
    directory.mkdir(parents=True, exist_ok=False)
    hf = directory / "huggingface"
    hf.mkdir()
    before = [(h.flat_param.data_ptr(), h.flat_param._version) for h in model._all_handles]

    def admit(size: int) -> None:
        admit_host_transfer(Path("/sys/fs/cgroup"), size, 2 * _GIB)

    # NO_SHARD cannot summon with offload_to_cpu=True. writeback=False alone does
    # not prevent in-place mutation in NO_SHARD: all operations below only read.
    with torch.no_grad(), FSDP.summon_full_params(
        model, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=False,
    ):
        tensors = {}
        for name, tensor in model.named_parameters(remove_duplicate=False):
            name = _clean_name(name)
            if name in tensors or name not in expected:
                raise RuntimeError("actor parameter names do not match the HF architecture")
            tensors[name] = tensor
        for name, tensor in model.named_buffers(remove_duplicate=False):
            name = _clean_name(name)
            # HF state_dict excludes nonpersistent rotary buffers.
            if name in expected:
                if name in tensors:
                    raise RuntimeError("duplicate actor parameter/buffer name")
                tensors[name] = tensor
        if {name: list(t.shape) for name, t in tensors.items()} != expected:
            raise RuntimeError("actor export does not cover every HF tensor and shape")
        manifest = _write_weight_shards(tensors, hf, admit=admit)
        del tensors
    after = [(h.flat_param.data_ptr(), h.flat_param._version) for h in model._all_handles]
    registry.check_identity()
    registry.backend.validate_training(model)
    if after != before:
        raise RuntimeError("actor training storage/version changed during inference export")
    export_config = copy.deepcopy(config)
    export_config.torch_dtype = torch.bfloat16
    export_config.save_pretrained(hf)
    tokenizer = manager.processing_class
    if tokenizer is None:
        raise RuntimeError("actor export requires its actual training tokenizer")
    tokenizer.save_pretrained(hf)
    generation = getattr(model._fsdp_wrapped_module, "generation_config", None)
    if generation is not None:
        copy.deepcopy(generation).save_pretrained(hf)
    for path in hf.iterdir():
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(hf)
    evidence = _hf_evidence(hf, verify_hashes=True)
    if evidence["tensors"] != manifest:
        raise RuntimeError("actor export readback differs from detached BF16 source copies")
    _write_json(directory / ACTOR_RECEIPT, {
        "schema_version": 1, "artifact_profile": ACTOR_EXPORT_PROFILE,
        "identity_sha256": identity, "global_step": global_step, "world_size": 1,
        "training_dtype": "float32", "export_dtype": "bfloat16",
        "actor_included": True, "critic_included": False,
        "optimizer_included": False, "resume_supported": False,
        "lossy_fp32_to_bf16": True, "source_copy_readback_verified": True,
        "training_storage_version_preserved": True, "max_shard_bytes": _GIB,
        "hf": evidence,
    })
    _fsync_directory(directory)


def validate_actor_receipt(
    directory: Path, *, identity_sha256: str, global_step: int, verify_hashes: bool = True,
) -> dict[str, Any]:
    if directory.is_symlink() or not (directory / ACTOR_RECEIPT).is_file():
        raise RuntimeError("actor export receipt is missing or redirected")
    receipt = _json(directory / ACTOR_RECEIPT)
    required = {
        "schema_version": 1, "artifact_profile": ACTOR_EXPORT_PROFILE,
        "identity_sha256": identity_sha256, "global_step": global_step, "world_size": 1,
        "training_dtype": "float32", "export_dtype": "bfloat16",
        "actor_included": True, "critic_included": False,
        "optimizer_included": False, "resume_supported": False,
        "lossy_fp32_to_bf16": True, "source_copy_readback_verified": True,
        "training_storage_version_preserved": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("actor export receipt contract or identity mismatch")
    observed = _hf_evidence(directory / "huggingface", verify_hashes=verify_hashes)
    expected = copy.deepcopy(receipt.get("hf", {}))
    if not verify_hashes:
        expected["sha256"] = {
            k: v for k, v in expected.get("sha256", {}).items() if not k.endswith(".safetensors")
        }
        for value in expected.get("tensors", {}).values():
            value.pop("sha256", None)
    if observed != expected:
        raise RuntimeError("actor export contents changed after verified saving")
    allowed = {ACTOR_RECEIPT} | {"huggingface/" + k for k in observed["files"]}
    if set(_inventory(directory)) != allowed:
        raise RuntimeError("actor export contains unexpected files")
    return receipt


def export_completion_document(
    directory: Path, *, profile: str, identity: str, step: int,
) -> dict[str, Any]:
    document = {
        "schema_version": 1, "artifact_profile": profile,
        "identity_sha256": identity, "global_step": step, "world_size": 1,
        "actor_included": profile == ACTOR_EXPORT_PROFILE, "critic_included": False,
        "optimizer_included": False, "resume_supported": False,
    }
    if profile == ACTOR_EXPORT_PROFILE:
        validate_actor_receipt(
            directory / "actor", identity_sha256=identity, global_step=step, verify_hashes=False,
        )
        document["actor_receipt_sha256"] = _sha(directory / "actor" / ACTOR_RECEIPT)
    elif profile == DIAGNOSTIC_PROFILE:
        document["diagnostic_only"] = True
        document["model_included"] = False
        document["evidence_scope"] = "run metrics/probes/evaluation validated separately"
    else:
        raise RuntimeError("unknown actor/diagnostic export profile")
    return document


def validate_export_artifact(
    directory: str | Path, *, identity_sha256: str, world_size: int = 1,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    directory = Path(directory)
    marker = directory / EXPORT_COMPLETION_MARKER
    if directory.is_symlink() or not marker.is_file() or marker.is_symlink():
        raise RuntimeError("actor/diagnostic export completion marker is missing or redirected")
    step_match = re.fullmatch(r"global_step_([0-9]+)", directory.name)
    if step_match is None or world_size != 1:
        raise RuntimeError("actor/diagnostic export requires a single-rank policy step")
    step = int(step_match[1])
    document = _json(marker)
    profile = document.get("artifact_profile")
    if profile not in EXPORT_PROFILES:
        raise RuntimeError("invalid actor/diagnostic export profile")
    expected = export_completion_document(
        directory, profile=profile, identity=identity_sha256, step=step,
    )
    if document != expected:
        raise RuntimeError("actor/diagnostic export marker identity or contract mismatch")
    if profile == ACTOR_EXPORT_PROFILE:
        validate_actor_receipt(
            directory / "actor", identity_sha256=identity_sha256,
            global_step=step, verify_hashes=verify_hashes,
        )
        allowed = {EXPORT_COMPLETION_MARKER} | {
            "actor/" + name for name in _inventory(directory / "actor")
        }
    else:
        if step != 2:
            raise RuntimeError("critic diagnostic receipt requires exactly two updates")
        allowed = {EXPORT_COMPLETION_MARKER}
    if set(_inventory(directory)) != allowed:
        raise RuntimeError("inference/diagnostic artifact contains unrequested training state")
    return document


validate_actor_export_artifact = validate_export_artifact


def install_worker_export_hook() -> bool:
    """Retain native Ray/placement worker methods; replace only the opted-in saver."""
    if os.environ.get(ARTIFACT_PROFILE_ENV) != ACTOR_EXPORT_PROFILE:
        return False
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    original: Any = FSDPCheckpointManager.save_checkpoint
    if getattr(original, "_finance_actor_export", False):
        return False

    @wraps(original)
    def save_checkpoint(
        manager: Any, local_path: str, hdfs_path: str | None = None,
        global_step: int = 0, max_ckpt_to_keep: int | None = None,
    ) -> Any:
        if ACTOR_SAVE_CONTENT not in manager.checkpoint_save_contents:
            return original(manager, local_path, hdfs_path, global_step, max_ckpt_to_keep)
        if (
            list(manager.checkpoint_save_contents) != [ACTOR_SAVE_CONTENT]
            or hdfs_path is not None or max_ckpt_to_keep is not None
            or os.environ.get(ARTIFACT_PROFILE_ENV) != ACTOR_EXPORT_PROFILE
        ):
            raise RuntimeError("actor export forbids mixed state, remote storage and pruning")
        return _export_manager(manager, local_path, global_step)

    save_checkpoint._finance_actor_export = True  # type: ignore[attr-defined]
    FSDPCheckpointManager.save_checkpoint = save_checkpoint  # type: ignore[method-assign,assignment]
    return True
