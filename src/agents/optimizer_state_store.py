"""Fresh-run scratch storage for an inactive Adam optimizer's CPU moments.

This is deliberately not a restart/checkpoint format.  The same optimizer,
parameter objects, state dictionaries and scalar steps must survive a round trip.
Any failure poisons the store: callers must stop the run, never retry or restore
a partially completed transition automatically.  Model parameters are not saved.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

_MOMENTS = ("exp_avg", "exp_avg_sq")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_file_cache(fd: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    payload = _json_bytes(value)
    with temporary.open("xb", buffering=0) as output:
        if output.write(payload) != len(payload):
            raise OSError("short manifest write")
        os.fsync(output.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)
    if path.read_bytes() != payload:
        raise OSError("manifest readback differs from written bytes")


def _scalar_step(state: dict[str, Any]) -> dict[str, Any] | None:
    if "step" not in state:
        return None
    step = state["step"]
    # Importing this module remains possible in CPU-only, torch-free tooling.
    import torch

    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError("Adam step must be scalar")
        value = step.item()
        metadata = {
            "kind": "tensor", "dtype": str(step.dtype),
            "shape": list(step.shape), "device": str(step.device), "value": value,
        }
    elif isinstance(step, (int, float)) and not isinstance(step, bool):
        value = step
        metadata = {"kind": type(step).__name__, "value": value}
    else:
        raise ValueError("unsupported Adam scalar step")
    if not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError("Adam step must be a nonnegative finite integer")
    return metadata


class MomentStore:
    """Store only FP32 Adam moments, in bounded, checksummed raw files.

    ``admit_host(nbytes)`` runs before allocating each restored CPU moment and
    should raise if the shared host budget cannot admit it. ``emit(event_dict)``
    optionally records lifecycle events. A fresh, absolute ``directory`` is
    owned exclusively by this object; existing directories are never adopted.
    """

    def __init__(
        self,
        directory: str | Path,
        identity: Any,
        role: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
        disk_reserve_bytes: int = 2 * 1024**3,
        admit_host: Any = None,
        emit: Any = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            raise ValueError("moment scratch directory must be absolute")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", role):
            raise ValueError("invalid optimizer role")
        if isinstance(chunk_bytes, bool) or chunk_bytes < 4 or chunk_bytes % 4:
            raise ValueError("chunk_bytes must be a positive multiple of four")
        if disk_reserve_bytes < 0:
            raise ValueError("disk reserve must be nonnegative")
        self.identity = json.loads(_json_bytes(identity))
        self.role = role
        self.chunk_bytes = chunk_bytes
        self.disk_reserve_bytes = disk_reserve_bytes
        self.admit_host = admit_host
        self.emit = emit
        self.status = "resident"
        self.manifest_path = self.directory / "manifest.json"
        self._owner = uuid.uuid4().hex
        self._optimizer: Any = None
        self._parameters: list[Any] = []
        self._parameter_specs: list[tuple[tuple[int, ...], str]] = []
        self._states: list[Any] = []
        self._steps: list[Any] = []
        self._group_lengths: list[int] = []
        self._manifest: dict[str, Any] | None = None
        self._generation = 0
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        _atomic_json(self.directory / "owner.json", {
            "owner": self._owner, "identity": self.identity, "role": self.role,
            "purpose": "fresh-run Adam moment scratch; not a restart checkpoint",
        })

    def _event(self, event: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit({"event": event, "role": self.role, **fields})

    def _fail(self, error: Exception) -> None:
        self.status = "failed"
        # Preserve the original error, including a full disk or unavailable mount.
        with contextlib.suppress(Exception):
            _atomic_json(self.directory / "failure.json", {
                "role": self.role, "generation": self._generation,
                "error_type": type(error).__name__, "error": str(error),
                "instruction": "stop; no automatic retry or partial-state recovery",
            })

    def _boundary(self, optimizer: Any, *, bind: bool = False) -> list[Any]:
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        group_lengths = [len(group["params"]) for group in optimizer.param_groups]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("duplicate optimizer parameters")
        if any(p.grad is not None for p in parameters):
            raise ValueError("all parameter gradients must be absent at a store boundary")
        if bind and self._optimizer is None:
            self._optimizer = optimizer
            self._parameters = list(parameters)
            self._parameter_specs = [(tuple(p.shape), str(p.dtype)) for p in parameters]
            self._group_lengths = group_lengths
        if optimizer is not self._optimizer or group_lengths != self._group_lengths:
            raise ValueError("optimizer identity or parameter groups changed")
        if len(parameters) != len(self._parameters) or any(
            left is not right for left, right in zip(parameters, self._parameters, strict=True)
        ):
            raise ValueError("optimizer parameter identity/order changed")
        if [(tuple(p.shape), str(p.dtype)) for p in parameters] != self._parameter_specs:
            raise ValueError("optimizer parameter shape/dtype changed")
        return parameters

    def _write_chunk(self, path: Path, data: memoryview) -> dict[str, Any]:
        digest = hashlib.sha256(data).hexdigest()
        with path.open("xb+", buffering=0) as output:
            offset = 0
            while offset < len(data):
                written = output.write(data[offset:])
                if not written:
                    raise OSError("short Adam moment write")
                offset += written
            os.fsync(output.fileno())
            output.seek(0)
            check = hashlib.sha256()
            buffer = bytearray(min(self.chunk_bytes, len(data)))
            count = 0
            while size := output.readinto(buffer):
                check.update(memoryview(buffer)[:size])
                count += size
            if count != len(data) or check.hexdigest() != digest:
                raise OSError("Adam moment write verification failed")
            _drop_file_cache(output.fileno())
        return {"file": path.name, "bytes": len(data), "sha256": digest}

    def _remove_superseded(self, manifest: dict[str, Any]) -> None:
        directory = self.directory / manifest["generation"]
        names = {
            chunk["file"] for entry in manifest["parameters"]
            for chunks in entry["moments"].values() for chunk in chunks
        }
        # Do not remove anything unexpectedly introduced into our scratch directory.
        if {path.name for path in directory.iterdir()} != names:
            raise ValueError("foreign files in superseded moment generation")
        for name in sorted(names):
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("unexpected file type in superseded moment generation")
        for name in sorted(names):
            (directory / name).unlink()
        directory.rmdir()
        _sync_directory(self.directory)

    def offload(self, optimizer: Any) -> Path:
        """Durably verify all moments before removing any resident moment keys."""
        if self.status != "resident":
            raise RuntimeError(f"offload requires resident state, got {self.status}")
        try:
            import torch

            parameters = self._boundary(optimizer, bind=True)
            entries = []
            states = []
            steps = []
            payload_bytes = 0
            for index, parameter in enumerate(parameters):
                if parameter.dtype != torch.float32:
                    raise ValueError("FP32 optimizer parameters are required")
                state = optimizer.state.get(parameter)
                states.append(state)
                steps.append(None if state is None else state.get("step"))
                mapping = {} if state is None else state
                present = [key in mapping for key in _MOMENTS]
                if any(present) != all(present):
                    raise ValueError("incomplete Adam moment pair")
                if "max_exp_avg_sq" in mapping:
                    raise ValueError("AMSGrad is not supported by this moment store")
                step = _scalar_step(mapping)
                if all(present) and step is None:
                    raise ValueError("initialized Adam moments require a scalar step")
                if not any(present) and step is not None and step["value"] != 0:
                    raise ValueError("nonzero Adam step without moments")
                for key in _MOMENTS if all(present) else ():
                    tensor = mapping[key]
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                        or tensor.shape != parameter.shape or not tensor.is_contiguous()
                    ):
                        raise ValueError(
                            "moments must be contiguous CPU FP32 parameter-shaped tensors"
                        )
                    payload_bytes += tensor.numel() * tensor.element_size()
                entries.append({
                    "index": index, "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype), "step": step,
                    "state_present": state is not None, "moments": {},
                })
            if shutil.disk_usage(self.directory).free < payload_bytes + self.disk_reserve_bytes:
                raise OSError("insufficient disk space for moment snapshot plus reserve")
            self._generation += 1
            generation = f"generation-{self._generation:06d}"
            destination = self.directory / generation
            destination.mkdir(mode=0o700)
            for entry, state in zip(entries, states, strict=True):
                for key in _MOMENTS if state is not None and "exp_avg" in state else ():
                    data = memoryview(state[key].detach().numpy()).cast("B")
                    chunks = []
                    for chunk_index, offset in enumerate(range(0, len(data), self.chunk_bytes)):
                        filename = f"p{entry['index']:06d}-{key}-{chunk_index:06d}.raw"
                        chunks.append(self._write_chunk(
                            destination / filename, data[offset:offset + self.chunk_bytes],
                        ))
                    entry["moments"][key] = chunks
                    del data
            _sync_directory(destination)
            manifest = {
                "schema_version": 1, "owner": self._owner, "identity": self.identity,
                "role": self.role, "generation": generation, "payload_bytes": payload_bytes,
                "chunk_bytes": self.chunk_bytes, "group_lengths": self._group_lengths,
                "parameters": entries,
            }
            _atomic_json(self.manifest_path, manifest)
            old_manifest = self._manifest
            self._manifest = manifest
            self._states = states
            self._steps = steps
            # The manifest and every payload file are now durable and verified.
            for state in states:
                if state is not None:
                    for key in _MOMENTS:
                        state.pop(key, None)
            self.status = "offloaded"
            if old_manifest is not None:
                self._remove_superseded(old_manifest)
            self._event("moments_offloaded", payload_bytes=payload_bytes, generation=generation)
            return self.manifest_path
        except Exception as error:
            self._fail(error)
            raise

    def _read_chunk(self, path: Path, chunk: dict[str, Any], target: memoryview) -> None:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != chunk["bytes"]:
            raise ValueError("moment chunk missing, unsafe or wrong size")
        with path.open("rb", buffering=0) as source:
            offset = 0
            while offset < len(target):
                size = source.readinto(target[offset:])
                if not size:
                    raise OSError("short Adam moment read")
                offset += size
            if source.read(1) or hashlib.sha256(target).hexdigest() != chunk["sha256"]:
                raise ValueError("moment chunk checksum/length mismatch")
            _drop_file_cache(source.fileno())

    def restore(self, optimizer: Any) -> None:
        """Restore one moment allocation at a time; retain scalar step objects."""
        if self.status != "offloaded":
            raise RuntimeError(f"restore requires offloaded state, got {self.status}")
        try:
            import torch

            parameters = self._boundary(optimizer)
            manifest = json.loads(self.manifest_path.read_bytes())
            if manifest != self._manifest:
                raise ValueError("moment manifest identity or content changed")
            # Validate every parameter and step before allocating/restoring any moment.
            for parameter, entry, state, step_object in zip(
                parameters, manifest["parameters"], self._states, self._steps, strict=True,
            ):
                if (
                    list(parameter.shape) != entry["shape"]
                    or str(parameter.dtype) != entry["dtype"]
                ):
                    raise ValueError("optimizer parameter shape/dtype changed")
                current = optimizer.state.get(parameter)
                if current is not state or (current is not None) != entry["state_present"]:
                    raise ValueError("optimizer state dictionary identity changed")
                mapping = {} if current is None else current
                if any(key in mapping for key in _MOMENTS):
                    raise ValueError("moments already resident during restore")
                if _scalar_step(mapping) != entry["step"] or mapping.get("step") is not step_object:
                    raise ValueError("Adam scalar step changed while offloaded")
            for parameter, entry, state in zip(
                parameters, manifest["parameters"], self._states, strict=True,
            ):
                for key, chunks in entry["moments"].items():
                    nbytes = parameter.numel() * 4
                    if self.admit_host is not None:
                        self.admit_host(nbytes)
                    tensor = torch.empty(tuple(entry["shape"]), dtype=torch.float32, device="cpu")
                    target = memoryview(tensor.numpy()).cast("B")
                    offset = 0
                    for chunk in chunks:
                        end = offset + chunk["bytes"]
                        if end > nbytes or chunk["bytes"] > self.chunk_bytes:
                            raise ValueError("invalid moment chunk bounds")
                        self._read_chunk(
                            self.directory / manifest["generation"] / chunk["file"],
                            chunk, target[offset:end],
                        )
                        offset = end
                    if offset != nbytes:
                        raise ValueError("moment payload does not match parameter shape")
                    del target
                    state[key] = tensor
                    del tensor
            self.status = "resident"
            self._event("moments_restored", payload_bytes=manifest["payload_bytes"])
        except Exception as error:
            self._fail(error)
            raise
