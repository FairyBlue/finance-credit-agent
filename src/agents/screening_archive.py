"""Preserve successful screening actors before removing cell-local resume states.

The pinned verl FSDP merger converts tensors to BF16 on CPU. Its rank-zero
inspection retains an extra FP32 copy, so conversion must follow trainer exit,
with sufficient host memory and disk space. This archive preserves inference
weights and all run evidence; it intentionally cannot resume FP32 training.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .screening_checkpoint import validate_checkpoint

_GIB = 1024**3
_STAGES = frozenset({"wiring", "pilot", "screening", "zero-shot"})
_STATE_PATH = re.compile(
    r"global_step_[0-9]+/(?:data\.pt|(?:actor|critic)/"
    r"(?:model|optim|extra_state)_world_size_[0-9]+_rank_[0-9]+\.pt)"
)
_IDENTITY_KEY = "+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256="


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _inventory(directory: Path) -> dict[str, int]:
    """Never follow symlinks when inventorying or pruning a checkpoint subtree."""
    result = {}
    for current, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            if path.is_symlink():
                raise RuntimeError(f"archive refuses symlink in checkpoint/artifact tree: {path}")
        for name in files:
            path = Path(current) / name
            result[path.relative_to(directory).as_posix()] = path.stat().st_size
    return result


def _run_identity(cell: Path) -> str:
    command = json.loads((cell / "command.json").read_text(encoding="utf-8"))
    if not isinstance(command, list):
        raise RuntimeError("archive needs the recorded resolved command")
    identities = [
        arg[len(_IDENTITY_KEY) :]
        for arg in command
        if isinstance(arg, str) and arg.startswith(_IDENTITY_KEY)
    ]
    if len(identities) != 1 or re.fullmatch(r"[0-9a-f]{64}", identities[0]) is None:
        raise RuntimeError("archive cannot identify the checkpoint execution contract")
    return identities[0]


def _assert_trainer_stopped(cell: Path) -> None:
    process = _json(cell / "process.json")
    pid = process.get("pid")
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("archive requires the recorded trainer process identity")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise RuntimeError("cannot prove trainer process has exited") from error
    raise RuntimeError("trainer process is still alive; do not merge concurrently with training")


def _host_available_bytes() -> int:
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    if available is None:
        raise RuntimeError("cannot determine CPU memory available for model export")
    root = Path("/sys/fs/cgroup")
    if (root / "memory.current").is_file():
        current = int((root / "memory.current").read_text())
        stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        reclaimable = int(stats.get("inactive_file", 0))
        for filename in ("memory.max", "memory.high"):
            path = root / filename
            if path.is_file() and path.read_text().strip() != "max":
                headroom = int(path.read_text()) - current + reclaimable
                available = min(available, max(0, headroom))
    return available


def _resource_check(cell: Path, actor_bytes: int) -> dict[str, int]:
    # BF16 output is about half the FP32 shard. Reserve a full shard's worth of
    # disk space and allow for the merger's duplicate rank-zero CPU allocation.
    required_disk = actor_bytes + 2 * _GIB
    required_host = 3 * actor_bytes + 4 * _GIB
    disk_free = shutil.disk_usage(cell).free
    host_free = _host_available_bytes()
    if disk_free < required_disk:
        raise RuntimeError(
            f"insufficient archive disk headroom: need {required_disk} bytes, have {disk_free}; "
            "checkpoint states were retained"
        )
    if host_free < required_host:
        raise RuntimeError(
            f"insufficient CPU memory for pinned merger: need approximately {required_host} bytes, "
            f"have {host_free}; checkpoint states were retained"
        )
    return {
        "actor_shard_bytes": actor_bytes,
        "required_free_disk_bytes": required_disk,
        "disk_free_bytes": disk_free,
        "estimated_host_bytes": required_host,
        "host_available_bytes": host_free,
    }


def _cpu_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FINANCE_")
    }
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return environment


def _run_merger(actor: Path, destination: Path, cell: Path) -> list[str]:
    if version("verl") != "0.7.0":
        raise RuntimeError("screening archive requires pinned verl==0.7.0")
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor),
        "--target_dir",
        str(destination),
        "--use_cpu_initialization",
    ]
    with (cell / "archive-merger.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            env=_cpu_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "actor HF export failed; checkpoint states were retained; see archive-merger.log"
        )
    return command


def _safetensor_shapes(path: Path) -> dict[str, list[int]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        if not 0 < header_length <= min(size - 8, 100 * 1024 * 1024):
            raise RuntimeError("invalid exported safetensors header")
        header = json.loads(handle.read(header_length))
    tensors = {}
    intervals = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if (
            entry.get("dtype") != "BF16"
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise RuntimeError("exported weights must be valid BF16 safetensors")
        start, end = offsets
        if start < 0 or end - start != 2 * math.prod(shape):
            raise RuntimeError("exported tensor byte count does not match its shape")
        tensors[name] = shape
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise RuntimeError("exported tensor payload has gaps or overlapping offsets")
        cursor = end
    if not tensors or cursor + header_length + 8 != size:
        raise RuntimeError("exported safetensors payload is incomplete")
    return tensors


_VERIFY_PROGRAM = """
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
model, info = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, trust_remote_code=False, device_map='meta',
    torch_dtype=torch.bfloat16, output_loading_info=True,
)
for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
    if info.get(key):
        raise RuntimeError('incomplete exported model: ' + key + '=' + str(info[key]))
if not tokenizer.encode('hold'):
    raise RuntimeError('exported tokenizer cannot encode an action')
print(json.dumps({'architecture_load_verified': True, 'tokenizer_load_verified': True,
                  'parameter_count': sum(p.numel() for p in model.parameters())}))
"""


def _verify_model_loading(directory: Path, cell: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", _VERIFY_PROGRAM, str(directory)],
        env=_cpu_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    (cell / "archive-validation.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "exported actor/tokenizer cannot load completely; checkpoint states retained"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _verify_hf(directory: Path, cell: Path) -> dict[str, Any]:
    inventory = _inventory(directory)
    config = _json(directory / "config.json")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise RuntimeError("exported HF config must declare BF16 inference")
    for name in ("tokenizer_config.json", "tokenizer.json"):
        if inventory.get(name, 0) <= 0:
            raise RuntimeError(f"exported tokenizer asset missing: {name}")
        _json(directory / name)
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("exported actor has no safetensors weights")
    index_path = directory / "model.safetensors.index.json"
    mapping = _json(index_path).get("weight_map") if index_path.exists() else None
    if mapping is None and (len(shards) != 1 or shards[0].name != "model.safetensors"):
        raise RuntimeError("sharded actor export requires a complete HF weight index")
    actual = {}
    for shard in shards:
        for key in _safetensor_shapes(shard):
            if key in actual:
                raise RuntimeError("duplicate tensor in exported model shards")
            actual[key] = shard.name
    if mapping is not None and mapping != actual:
        raise RuntimeError("exported weight index does not exactly match shard tensors")
    loading = _verify_model_loading(directory, cell)
    hashes = {}
    for relative in sorted(inventory):
        digest = sha256()
        with (directory / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return {
        "dtype": "bfloat16",
        "tensor_count": len(actual),
        "files": inventory,
        "sha256": hashes,
        **loading,
    }


def archive_completed_cell(cell_dir: str | Path, stage: str) -> dict[str, Any]:
    """Archive a PASSED cell; export failure never removes its checkpoint states."""
    supplied = Path(cell_dir).expanduser().absolute()
    cell = supplied.resolve()
    if supplied != cell or not cell.is_dir() or stage not in _STAGES:
        raise RuntimeError("archive requires a real cell directory and a registered stage")
    summary = _json(cell / "summary.json")
    config = _json(cell / "resolved_config.json")
    if (
        summary.get("status") != "PASSED"
        or summary.get("stage") != stage
        or config.get("screening", {}).get("stage") != stage
        or Path(config.get("tracking", {}).get("log_root", "")).resolve() != cell
    ):
        raise RuntimeError("archive is restricted to its own PASSED screening cell")
    _assert_trainer_stopped(cell)
    identity = _run_identity(cell)
    receipt_path = cell / "archive.json"
    if receipt_path.exists():
        previous = _json(receipt_path)
        if previous.get("status") == "ARCHIVED" and previous.get("identity_sha256") == identity:
            return previous
        raise RuntimeError("archive has an unfinished receipt; inspect it before proceeding")
    checkpoints = cell / "checkpoints"
    if checkpoints.is_symlink() or checkpoints.resolve() != cell / "checkpoints":
        raise RuntimeError("archive refuses a redirected checkpoint directory")
    if stage == "zero-shot":
        if checkpoints.exists() and _inventory(checkpoints):
            raise RuntimeError("zero-shot unexpectedly has checkpoint files")
        result = {
            "status": "ARCHIVED",
            "stage": stage,
            "identity_sha256": identity,
            "removed_files": [],
            "reclaimed_bytes": 0,
            "actor_hf": None,
        }
        _write_json(receipt_path, result)
        return result
    step = config["training"]["total_training_steps"]
    if type(step) is not int or step <= 0:
        raise RuntimeError("archive requires a complete final training step")
    inventory = _inventory(checkpoints)
    final = checkpoints / f"global_step_{step}"
    checkpoint = validate_checkpoint(final, identity_sha256=identity)
    result = {
        "status": "PREPARED",
        "stage": stage,
        "identity_sha256": identity,
        "source_checkpoint": checkpoint,
        "checkpoint_inventory": inventory,
        "actor_hf": None,
        "removed_files": [],
        "reclaimed_bytes": 0,
    }
    if stage == "screening":
        actor = final / "actor"
        fsdp = _json(actor / "fsdp_config.json")
        if fsdp.get("world_size") != 1:
            raise RuntimeError("screening archive currently supports the single-GPU FSDP layout")
        actor_bytes = sum(
            size for name, size in checkpoint["files"].items() if name.startswith("actor/model_")
        )
        result["resource_precheck"] = _resource_check(cell, actor_bytes)
        target = cell / "actor_hf"
        if target.exists() or target.is_symlink():
            raise RuntimeError("archive refuses to overwrite an existing actor_hf directory")
        staging = cell / f"actor_hf.incomplete-{uuid4().hex}"
        staging.mkdir()
        result["merger_command"] = _run_merger(actor, staging, cell)
        result["hf_validation"] = _verify_hf(staging, cell)
        staging.rename(target)
        result["actor_hf"] = str(target)
    # Persist provenance outside the checkpoint subtree before removing anything.
    _write_json(receipt_path, result)
    validate_checkpoint(final, identity_sha256=identity)
    if _json(cell / "summary.json").get("status") != "PASSED":
        raise RuntimeError("cell status changed before archive cleanup; checkpoint states retained")
    removed = []
    for relative in sorted(inventory):
        if _STATE_PATH.fullmatch(relative):
            path = checkpoints / relative
            if path.is_symlink() or path.resolve() != checkpoints / relative:
                raise RuntimeError("checkpoint path changed before archive cleanup")
            path.unlink()
            removed.append(relative)
    result.update(
        status="ARCHIVED",
        removed_files=removed,
        reclaimed_bytes=sum(inventory[name] for name in removed),
    )
    _write_json(receipt_path, result)
    return result
