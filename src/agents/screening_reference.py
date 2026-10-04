"""Opt-in pageable reference placement for the fixed single-GPU screening runtime.

Parameter placement changes only; model values, role and inference calls are preserved.
The dependency guards are required because this adapter uses pinned FSDP internals.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

PROFILE_ENV = "FINANCE_SCREENING_RESOURCE_PROFILE"
PROFILE = "manual_ref"
PROFILES = frozenset({"manual_ref", "pageable_adam_v1", "chunked_adam_disk_v1"})
IDENTITY_ENV = "FINANCE_SCREENING_IDENTITY_SHA256"
EVIDENCE_ENV = "FINANCE_SCREENING_PLACEMENT_DIR"
MARKER = "_finance_manual_reference_profile"
STATE = "_finance_manual_reference_state"
VERSIONS = {"torch": "2.7.0+cu128", "verl": "0.7.0"}
SOURCE_HASHES = {
    "verl/workers/fsdp_workers.py": (
        "54e0cce9ca9b4e8d2430a4064798e3a47"
        "a0bd345d5102c078597395bdbe8a5ed"
    ),
    "verl/utils/fsdp_utils.py": "57069140830aad0b43c9576df666d28b494683b821f019cfd592598f759444ce",
    "torch/distributed/fsdp/_flat_param.py": (
        "3f9a73e6e4084582a7145a51f029d5efd"
        "cff09a2eb081bf54bea56d28fbc2a6d"
    ),
    "torch/distributed/fsdp/_runtime_utils.py": (
        "3f68e9eef23398ebddca71702b420d70e"
        "56c6475dfa22e03b1b91a35458e8a75"
    ),
}
BUILDER_PARAMETERS = (
    "self",
    "model_path",
    "fsdp_config",
    "optim_config",
    "override_model_config",
    "use_remove_padding",
    "use_fused_kernels",
    "enable_gradient_checkpointing",
    "trust_remote_code",
    "use_liger",
    "role",
    "enable_activation_offload",
    "use_tiled_mlp",
    "tiled_mlp_shards",
)
logger = logging.getLogger(__name__)


def verify_installed_sources() -> dict[str, str]:
    """Read distribution metadata and files; do not import torch or probe devices."""
    for package, expected in VERSIONS.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"manual_ref requires {package}=={expected}; found {actual}")
    observed = {}
    for relative, expected in SOURCE_HASHES.items():
        package = relative.split("/")[0]
        path = Path(str(metadata.distribution(package).locate_file(relative)))
        observed[relative] = sha256(path.read_bytes()).hexdigest()
        if observed[relative] != expected:
            raise RuntimeError(f"manual_ref source mismatch: {relative}")
    return observed


def _identity() -> str:
    value = os.environ.get(IDENTITY_ENV, "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("manual_ref requires a frozen lowercase SHA256 screening identity")
    return value


def _get(config: Any, name: str, default: Any = None) -> Any:
    getter = getattr(config, "get", None)
    return getter(name, default) if callable(getter) else getattr(config, name, default)


@dataclass
class ReferenceState:
    model: Any
    identity: str
    forward_count: int = 0
    busy: bool = False
    failed: bool = False


class EventLog:
    """Best-effort metadata only; telemetry cannot replace a training exception."""

    def __init__(self, directory: Path, identity: str):
        self.directory = directory
        self.identity = identity
        self.profile = os.environ.get(PROFILE_ENV, PROFILE)

    def __call__(self, phase: str, event: str, **details: Any) -> None:
        try:
            record = {
                "schema_version": 1,
                "profile": self.profile,
                "identity_sha256": self.identity,
                "pid": os.getpid(),
                "time": datetime.now(timezone.utc).isoformat(),
                "monotonic_ns": time.monotonic_ns(),
                "phase": phase,
                "event": event,
                **details,
            }
            for filename in ("memory.current", "memory.events"):
                try:
                    record[filename] = (Path("/sys/fs/cgroup") / filename).read_text().strip()
                except OSError as error:
                    record[filename + "_error"] = type(error).__name__
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"manual-ref-{os.getpid()}.jsonl").open("a") as output:
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        except Exception:
            logger.warning("manual_ref diagnostic write failed", exc_info=True)


class TorchBackend:
    """Actual FSDP1 mechanics, deliberately never constructed by CPU fixtures."""

    def __init__(self):
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        self.torch, self.FSDP, self.lazy_init = torch, FSDP, _lazy_init

    def initialize(self, model: Any) -> None:
        if not isinstance(model, self.FSDP):
            raise RuntimeError("manual_ref requires real FSDP1")
        if model.world_size != 1 or model.sharding_strategy.name != "NO_SHARD":
            raise RuntimeError("manual_ref supports only world_size=1/NO_SHARD")
        if model.compute_device is None or model.compute_device.type != "cuda":
            raise RuntimeError("manual_ref production backend requires CUDA")
        self.lazy_init(model, model)  # Must run while the newly built reference is GPU.
        self.validate(model)

    def validate(self, model: Any) -> None:
        if model._is_root is not True or model.training_state.name != "IDLE":
            raise RuntimeError("manual_ref requires an idle, initialized FSDP root")
        if not model._all_handles:
            raise RuntimeError("manual_ref found no parameter handles")
        for handle in model._all_handles:
            flat = handle.flat_param
            if handle._offload_params or handle._use_orig_params:
                raise RuntimeError("manual_ref requires no CPUOffload and use_orig_params=False")
            if handle._training_state.name != "IDLE":
                raise RuntimeError("manual_ref parameter handle is busy")
            if getattr(flat, "_cpu_grad", None) is not None or flat.grad is not None:
                raise RuntimeError("manual_ref must not allocate or retain reference gradients")
            if flat.dtype != self.torch.bfloat16 or flat._local_shard.dtype != flat.dtype:
                raise RuntimeError("manual_ref reference must remain BF16")
            if (
                flat.data_ptr() != flat._local_shard.data_ptr()
                or flat.size() != flat._local_shard.size()
            ):
                raise RuntimeError("manual_ref flat parameter/local shard aliases diverged")
            if flat.device.type == "cpu" and flat.is_pinned():
                raise RuntimeError("manual_ref CPU reference must be pageable")
            for name, module, _ in flat._param_infos:
                view = getattr(module, name)
                if (
                    view.device != flat.device
                    or view.dtype != flat.dtype
                    or view.untyped_storage().data_ptr() != flat.untyped_storage().data_ptr()
                ):
                    raise RuntimeError("manual_ref module weight view retains a different storage")
            for name, module, _, primary_name, primary_module, _ in flat._shared_param_infos:
                if getattr(module, name) is not getattr(primary_module, primary_name):
                    raise RuntimeError("manual_ref shared weight alias diverged")

    def move(self, model: Any, destination: str) -> None:
        if destination not in ("cpu", "compute"):
            raise ValueError("invalid manual_ref destination")
        self.validate(model)
        device = self.torch.device("cpu") if destination == "cpu" else model.compute_device
        self.torch.cuda.synchronize(model.compute_device)
        with self.torch.no_grad():
            for handle in model._all_handles:
                flat = handle.flat_param
                handle.flat_param_to(device, non_blocking=False)
                flat._local_shard = flat.data
                # flat_param_to refreshes views only for use_orig_params=True.
                # Our forward-only NO_SHARD reference must also release the old
                # module tensor views, or they keep the previous device storage alive.
                handle._use_unsharded_views(as_params=False)
                if flat.device != device:
                    raise RuntimeError("manual_ref transfer did not reach its destination")
                if destination == "cpu" and flat.is_pinned():
                    raise RuntimeError("manual_ref unexpectedly produced pinned memory")
        self.torch.cuda.synchronize(model.compute_device)
        self.validate(model)
        self.torch.cuda.empty_cache()

    def no_grad(self) -> Any:
        return self.torch.no_grad()

    def snapshot(self, model: Any) -> dict[str, Any]:
        self.validate(model)
        cpu = gpu = cpu_grad = 0
        for handle in model._all_handles:
            flat = handle.flat_param
            size = flat.numel() * flat.element_size()
            cpu += size if flat.device.type == "cpu" else 0
            gpu += size if flat.device.type == "cuda" else 0
            grad = getattr(flat, "_cpu_grad", None)
            cpu_grad += grad.numel() * grad.element_size() if grad is not None else 0
        return {
            "reference_cpu_bytes": cpu,
            "reference_gpu_bytes": gpu,
            "reference_cpu_grad_bytes": cpu_grad,
            "handle_count": len(model._all_handles),
        }


def patch_worker_class(
    worker_class: type[Any],
    *,
    backend: Any,
    emit: Callable[..., None],
    identity: str,
) -> bool:
    """Private injection seam for CPU tests; production calls only after file/version guards."""
    profile = os.environ.get(PROFILE_ENV)
    if profile not in PROFILES or _identity() != identity:
        raise RuntimeError("manual_ref requires an authorized profile and matching identity")
    marker = {"identity": identity, "profile": profile}
    if worker_class.__dict__.get(MARKER):
        if worker_class.__dict__[MARKER] != marker:
            raise RuntimeError("manual_ref already installed for a different identity/profile")
        return False
    original_builder = worker_class._build_model_optimizer
    original_forward = worker_class.compute_ref_log_prob
    signature = inspect.signature(original_builder)
    if tuple(signature.parameters) != BUILDER_PARAMETERS:
        raise RuntimeError("manual_ref builder signature changed")
    if tuple(inspect.signature(original_forward).parameters) != ("self", "data"):
        raise RuntimeError("manual_ref forward signature changed")

    def check_profile() -> None:
        if os.environ.get(PROFILE_ENV) != profile or _identity() != identity:
            raise RuntimeError("manual_ref profile/identity changed after installation")

    def event(phase: str, action: str, **details: Any) -> None:
        try:
            emit(phase, action, **details)
        except Exception:
            logger.warning("manual_ref event callback failed", exc_info=True)

    def transfer(state: ReferenceState, destination: str, phase: str) -> None:
        event(phase, "enter", destination=destination, forward_count=state.forward_count)
        backend.move(state.model, destination)
        event(
            phase,
            "success",
            destination=destination,
            forward_count=state.forward_count,
            **backend.snapshot(state.model),
        )

    @wraps(original_builder)
    def build(worker: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(worker, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["role"] != "ref":
            return original_builder(worker, *args, **kwargs)
        check_profile()
        config = bound.arguments["fsdp_config"]
        if (
            not worker._is_ref
            or worker._is_lora
            or _get(_get(worker.config, "actor"), "strategy") != "fsdp"
            or _get(config, "model_dtype") not in ("bf16", "bfloat16")
            or _get(config, "use_orig_params", False)
            or bound.arguments["optim_config"] is not None
            or bound.arguments["use_tiled_mlp"]
            or bound.arguments["enable_gradient_checkpointing"]
            or bound.arguments["enable_activation_offload"]
        ):
            raise RuntimeError("manual_ref unsupported reference builder configuration")
        if hasattr(worker, STATE):
            raise RuntimeError("manual_ref reference already built for this worker")
        event("reference_init", "enter")
        state = None
        try:
            bound.arguments["role"] = "actor"
            result = original_builder(*bound.args, **bound.kwargs)
            if len(result) != 4 or result[1] is not None or result[2] is not None:
                raise RuntimeError(
                    "manual_ref builder unexpectedly constructed an optimizer/scheduler"
                )
            state = ReferenceState(result[0], identity)
            setattr(worker, STATE, state)
            event("reference_lazy_init", "enter")
            backend.initialize(state.model)
            event("reference_lazy_init", "success", **backend.snapshot(state.model))
            transfer(state, "cpu", "reference_init_offload")
            event("reference_init", "success", **backend.snapshot(state.model))
            return result
        except BaseException as error:
            if state is not None:
                state.failed = True
            event("reference_init", "error", error_type=type(error).__name__)
            raise

    @wraps(original_forward)
    def forward(worker: Any, data: Any) -> Any:
        check_profile()
        state = getattr(worker, STATE, None)
        if (
            not isinstance(state, ReferenceState)
            or state.identity != identity
            or state.model is not worker.ref_module_fsdp
            or state.failed
            or state.busy
        ):
            raise RuntimeError(
                "manual_ref missing, mismatched, failed or reentrant reference state"
            )
        state.busy = True
        state.forward_count += 1
        event("reference_forward", "enter", forward_count=state.forward_count)
        try:
            transfer(state, "compute", "reference_load")
            event("reference_compute", "enter", forward_count=state.forward_count)
            with backend.no_grad():
                result = original_forward(worker, data)
            event("reference_compute", "success", forward_count=state.forward_count)
        except BaseException as error:
            state.failed = True
            event(
                "reference_forward",
                "error",
                error_type=type(error).__name__,
                forward_count=state.forward_count,
            )
            try:
                transfer(state, "cpu", "reference_error_offload")
            except BaseException as cleanup_error:
                event("reference_error_offload", "error", error_type=type(cleanup_error).__name__)
            raise  # Preserve the original exception even if cleanup also fails.
        else:
            try:
                transfer(state, "cpu", "reference_offload")
            except BaseException as error:
                state.failed = True
                event(
                    "reference_forward",
                    "error",
                    error_type=type(error).__name__,
                    forward_count=state.forward_count,
                )
                raise
            event("reference_forward", "success", forward_count=state.forward_count)
            return result
        finally:
            state.busy = False

    worker_class._build_model_optimizer = build
    worker_class.compute_ref_log_prob = forward
    setattr(worker_class, MARKER, marker)
    return True


def install_from_environment() -> bool:
    """Opt-in screening integration entrypoint. Inactive profile performs no heavy imports."""
    if os.environ.get(PROFILE_ENV) not in PROFILES:
        return False
    identity = _identity()
    path = os.environ.get(EVIDENCE_ENV)
    if not path or not Path(path).is_absolute():
        raise RuntimeError("manual_ref requires an absolute reference evidence directory")
    sources = verify_installed_sources()
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    emit = EventLog(Path(path), identity)
    installed = patch_worker_class(
        ActorRolloutRefWorker,
        backend=TorchBackend(),
        emit=emit,
        identity=identity,
    )
    if installed:
        emit("reference_adapter", "installed", source_sha256=sources, versions=VERSIONS)
    return installed
