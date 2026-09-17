"""Guarded pageable Adam placement for the fixed single-GPU screening runtime.

No torch import/device activity occurs at module import. The installer is the
production seam; patch_worker_classes is a CPU-test injection seam only.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from functools import wraps
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from src.agents.screening_exchange import ExchangeCoordinator, ExchangeError, Operations, TensorInfo

PROFILE = "pageable_adam_v1"
PROFILE_ENV = "FINANCE_SCREENING_RESOURCE_PROFILE"
IDENTITY_ENV = "FINANCE_SCREENING_IDENTITY_SHA256"
DIRECTORY_ENV = "FINANCE_SCREENING_PLACEMENT_DIR"
MARKER = "_finance_pageable_adam_profile"
VERSIONS = {"torch": "2.7.0+cu128", "verl": "0.7.0"}
SOURCE_HASHES = {
    "verl/workers/fsdp_workers.py": (
        "54e0cce9ca9b4e8d2430a4064798e3a47a0bd345d5102c078597395bdbe8a5ed"
    ),
    "verl/workers/actor/dp_actor.py": (
        "2f708de798943838c9724a127506ac6a9ec41532fc39b417056fb9bd10e57504"
    ),
    "verl/workers/critic/dp_critic.py": (
        "ab19baa918f1a6dcdc33b8b54cdca867c1f4df6e356172149935fa2db2a085ad"
    ),
    "torch/optim/adam.py": ("78a0aa4a6b182fd586e40f964409cf3a9e2b935a1f8dcb62f31d1a4ca4c2e59a"),
    "torch/optim/adamw.py": ("3112d8799ef9ab608b855a5708087d5576971463923c41d4a6a227006871f551"),
    "torch/distributed/fsdp/_flat_param.py": (
        "3f9a73e6e4084582a7145a51f029d5efdcff09a2eb081bf54bea56d28fbc2a6d"
    ),
    "torch/distributed/fsdp/_runtime_utils.py": (
        "3f68e9eef23398ebddca71702b420d70e56c6475dfa22e03b1b91a35458e8a75"
    ),
}
AGENT_SOURCE_HASHES = {
    "workers/fsdp_workers.py": ("e710ad4a756a7b47ba9d3d2adb3f0f651c682621d563b56d09e5cbfc37c3c9d2"),
    "workers/actor/dp_actor.py": (
        "9c5359ae1ab87a153676bdaeb5a86b8beb9ce9d87b25350f8d4bd0438556cc0c"
    ),
    "workers/critic/dp_critic.py": (
        "b5241c5933efe6276799cba14a1727f474edc61f98b16f2179c3db334ba682c9"
    ),
}
_REGISTRY: Any = None


def verify_installed_sources() -> dict[str, str]:
    for package, expected in VERSIONS.items():
        if metadata.version(package) != expected:
            raise ExchangeError(f"unsupported {package} version")
    observed = {}
    for relative, expected in SOURCE_HASHES.items():
        path = Path(str(metadata.distribution(relative.split("/")[0]).locate_file(relative)))
        observed[relative] = sha256(path.read_bytes()).hexdigest()
        if observed[relative] != expected:
            raise ExchangeError(f"source changed: {relative}")
    agent_root = Path(__file__).resolve().parents[2] / "third_party/agent-r1/agent_r1"
    for relative, expected in AGENT_SOURCE_HASHES.items():
        key = "agent_r1/" + relative
        observed[key] = sha256((agent_root / relative).read_bytes()).hexdigest()
        if observed[key] != expected:
            raise ExchangeError(f"source changed: {key}")
    return observed


def _identity() -> str:
    value = os.environ.get(IDENTITY_ENV, "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ExchangeError("a frozen lowercase SHA256 screening identity is required")
    return value


def _get(config: Any, key: str, default: Any = None) -> Any:
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def admit_host_transfer(cgroup_root: Path, size: int, reserve: int) -> None:
    """Check actual charged memory against the hard cgroup capacity, without cache credit.

    memory.high is a reclaim/throttling threshold and remains in phase telemetry.
    It is not a second hard allocation ceiling. Keep a reserve below memory.max.
    """
    if size < 0 or reserve < 0:
        raise ExchangeError("transfer size and reserve must be nonnegative")
    current = int((cgroup_root / "memory.current").read_text())
    maximum = (cgroup_root / "memory.max").read_text().strip()
    if maximum == "max":
        raise ExchangeError("a finite cgroup hard memory limit is required")
    limit = int(maximum)
    if current < 0 or limit <= 0:
        raise ExchangeError("invalid cgroup memory accounting")
    if limit - current < size + reserve:
        raise MemoryError(
            "CPU destination cannot admit next moment plus host reserve: "
            f"current={current}, hard_limit={limit}, transfer={size}, reserve={reserve}"
        )


class Evidence:
    def __init__(self, directory: Path, identity: str):
        self.directory, self.identity = directory, identity

    def __call__(self, event: str, **fields: Any) -> None:
        try:
            row = {
                "schema_version": 1,
                "profile": PROFILE,
                "identity_sha256": self.identity,
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                "event": event,
                **fields,
            }
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"adam-placement-{os.getpid()}.jsonl").open("a") as output:
                output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        except Exception:
            # Telemetry never replaces an original training/transfer exception.
            pass


class TorchBackend:
    """Real backend, constructed only by the explicitly enabled installer."""

    def __init__(self, *, gpu_reserve_bytes: int, host_reserve_bytes: int):
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        if min(gpu_reserve_bytes, host_reserve_bytes) < 0:
            raise ExchangeError("reserves must be nonnegative")
        self.torch, self.FSDP, self.lazy_init = torch, FSDP, _lazy_init
        self.gpu_reserve_bytes, self.host_reserve_bytes = gpu_reserve_bytes, host_reserve_bytes
        self.gpu_device = "cuda:0"

    def describe(self, tensor: Any) -> TensorInfo:
        if not isinstance(tensor, self.torch.Tensor) or tensor.layout != self.torch.strided:
            raise ExchangeError("dense torch tensors are required")
        return TensorInfo(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            device=str(tensor.device),
            storage_id=tensor.untyped_storage()._cdata,
            storage_bytes=tensor.untyped_storage().nbytes(),
            payload_bytes=tensor.numel() * tensor.element_size(),
            contiguous=tensor.is_contiguous(),
            storage_offset=int(tensor.storage_offset()),
            pinned=tensor.is_pinned(),
            is_view=tensor._base is not None,
        )

    def move(self, tensor: Any, destination: str, *, non_blocking: bool) -> Any:
        if non_blocking or self.torch.is_grad_enabled():
            raise ExchangeError("only synchronous no_grad transfers are supported")
        if destination == "cpu":
            # Explicit pageable allocation; no pinned allocator/default assumptions.
            result = self.torch.empty_like(tensor, device="cpu", pin_memory=False)
            result.copy_(tensor, non_blocking=False)
            return result
        return tensor.to(device=destination, dtype=tensor.dtype, non_blocking=False)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.gpu_device)
        self.torch.cuda.empty_cache()
        host_empty_cache = getattr(self.torch._C, "_host_emptyCache", None)
        if callable(host_empty_cache):
            host_empty_cache()

    def admit(self, destination: str, size: int) -> None:
        # Previous source references are gone by this boundary. Release unused
        # allocator cache before comparing driver free bytes to the next copy.
        self.synchronize()
        if destination == "cpu":
            admit_host_transfer(Path("/sys/fs/cgroup"), size, self.host_reserve_bytes)
        else:
            free, _ = self.torch.cuda.mem_get_info(self.gpu_device)
            if free < size + self.gpu_reserve_bytes:
                raise MemoryError("GPU destination cannot admit next moment plus GPU reserve")

    def operations(self, boundary_check: Any) -> Operations:
        return Operations(
            describe=self.describe,
            scalar_value=lambda tensor: tensor.item(),
            move=self.move,
            synchronize=self.synchronize,
            no_grad=self.torch.no_grad,
            boundary_check=boundary_check,
            admit_transfer=self.admit,
        )

    def validate_worker(self, worker: Any, role: str) -> None:
        if worker.world_size != 1 or worker.rank != 0:
            raise ExchangeError("worker must be colocated rank 0 of world_size 1")
        if worker._is_offload_param is not False or worker._is_offload_optimizer is not False:
            raise ExchangeError("native parameter and optimizer transfers must both be disabled")
        if getattr(worker, "_is_lora", False) or getattr(worker, "use_orig_params", False):
            raise ExchangeError("LoRA/use_orig_params is unsupported")
        model_config = _get(worker.config, "model")
        fsdp_config = (
            _get(_get(worker.config, "actor"), "fsdp_config")
            if role == "actor"
            else _get(model_config, "fsdp_config")
        )
        if (
            _get(fsdp_config, "param_offload") is not False
            or _get(fsdp_config, "optimizer_offload") is not False
        ):
            raise ExchangeError("config must disable both native offloads before construction")
        if _get(model_config, "enable_activation_offload", False):
            raise ExchangeError("activation offload is outside this profile")
        if _get(_get(model_config, "tiled_mlp", {}), "enabled", False):
            raise ExchangeError("tiled model is outside this profile")

    def register_training(self, model: Any, optimizer: Any) -> None:
        if not isinstance(model, self.FSDP) or type(optimizer) is not self.torch.optim.AdamW:
            raise ExchangeError("exact torch AdamW and FSDP1 required")
        self.lazy_init(model, model)
        self.validate_training(model)
        if optimizer.state:
            raise ExchangeError("fresh-only adapter requires initially empty Adam states")

    def validate_training(self, model: Any) -> None:
        if (
            not isinstance(model, self.FSDP)
            or model.world_size != 1
            or model.sharding_strategy.name != "NO_SHARD"
            or str(model.compute_device) != self.gpu_device
            or model._is_root is not True
            or model.training_state.name != "IDLE"
        ):
            raise ExchangeError("training FSDP1/NO_SHARD/idle/device contract invalid")
        if not model._all_handles:
            raise ExchangeError("training FSDP has no handles")
        for handle in model._all_handles:
            flat = handle.flat_param
            if (
                handle._offload_params
                or handle._use_orig_params
                or handle._training_state.name != "IDLE"
            ):
                raise ExchangeError("training handle offload/view/state contract invalid")
            if (
                flat.grad is not None
                or getattr(flat, "_saved_grad_shard", None) is not None
                or getattr(flat, "_cpu_grad", None) is not None
            ):
                raise ExchangeError("unexpected live training gradients at full-update boundary")
            if (
                flat.dtype != self.torch.float32
                or str(flat.device) != self.gpu_device
                or not flat.requires_grad
            ):
                raise ExchangeError("full FP32 training parameters must stay on GPU")
            if (
                flat.data_ptr() != flat._local_shard.data_ptr()
                or flat.size() != flat._local_shard.size()
            ):
                raise ExchangeError("training flat/local-shard pointer invariant changed")

    def validate_reference(self, worker: Any) -> None:
        state = getattr(worker, "_finance_manual_reference_state", None)
        if (
            state is None
            or state.model is not worker.ref_module_fsdp
            or state.identity != _identity()
            or state.failed
            or state.busy
        ):
            raise ExchangeError(
                "reference adapter identity/model is missing, changed, failed or busy"
            )
        model = worker.ref_module_fsdp
        if not isinstance(model, self.FSDP) or model.world_size != 1:
            raise ExchangeError("reference must be registered FSDP1 world_size 1")
        if model.training_state.name != "IDLE" or not model._all_handles:
            raise ExchangeError("reference must be initialized and idle")
        for handle in model._all_handles:
            flat = handle.flat_param
            if (
                handle._offload_params
                or flat.device.type != "cpu"
                or flat.is_pinned()
                or flat.dtype != self.torch.bfloat16
                or flat.grad is not None
                or getattr(flat, "_cpu_grad", None) is not None
            ):
                raise ExchangeError(
                    "fixed reference must remain pageable BF16 CPU without gradients"
                )

    def metrics(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "gpu_reserve_bytes": self.gpu_reserve_bytes,
            "host_reserve_bytes": self.host_reserve_bytes,
        }
        free, total = self.torch.cuda.mem_get_info(self.gpu_device)
        row.update(
            gpu_driver_free_bytes=free,
            gpu_driver_total_bytes=total,
            gpu_allocated_bytes=self.torch.cuda.memory_allocated(self.gpu_device),
            gpu_reserved_bytes=self.torch.cuda.memory_reserved(self.gpu_device),
        )
        for filename in (
            "memory.current",
            "memory.high",
            "memory.max",
            "memory.events",
            "memory.stat",
        ):
            row[filename] = (Path("/sys/fs/cgroup") / filename).read_text().strip()
        return row


class WorkerRegistry:
    def __init__(self, *, identity: str, backend: Any, emit: Any):
        self.identity, self.backend, self.emit = identity, backend, emit
        self.pid = os.getpid()
        self.workers: dict[str, Any] = {}
        self.reference: Any = None
        self.busy = False
        self.cursor = 0
        self.coordinator = ExchangeCoordinator(
            identity=identity,
            operations=backend.operations(self.boundary_check),
            gpu_device=backend.gpu_device,
        )

    def flush(self, phase: str) -> None:
        try:
            for row in self.coordinator.events[self.cursor :]:
                self.emit("exchange", phase=phase, detail=row)
            self.cursor = len(self.coordinator.events)
            self.emit(
                "boundary",
                phase=phase,
                active_role=self.coordinator.active_role,
                failed=self.coordinator.failed,
                **self.backend.metrics(),
            )
        except Exception:
            pass

    def check_identity(self) -> None:
        if (
            os.environ.get(PROFILE_ENV) != PROFILE
            or _identity() != self.identity
            or os.getpid() != self.pid
            or self.coordinator.failed
        ):
            raise ExchangeError("placement profile/PID/identity changed or coordinator failed")

    def boundary_check(self) -> None:
        self.check_identity()
        if self.reference is None:
            raise ExchangeError("manual pageable reference must register before compute")
        self.backend.validate_reference(self.reference)
        for role, worker in self.workers.items():
            self.backend.validate_worker(worker, role)
            model = worker.actor_module_fsdp if role == "actor" else worker.critic_module
            optimizer = worker.actor_optimizer if role == "actor" else worker.critic_optimizer
            registered_model, registered_optimizer = self.coordinator._roles[role]
            if model is not registered_model or optimizer is not registered_optimizer:
                raise ExchangeError("registered training module/optimizer identity changed")
            self.backend.validate_training(model)

    def register(self, worker: Any, role: str | None) -> None:
        self.check_identity()
        if role is not None:
            self.backend.validate_worker(worker, role)
            model = worker.actor_module_fsdp if role == "actor" else worker.critic_module
            optimizer = worker.actor_optimizer if role == "actor" else worker.critic_optimizer
            self.backend.register_training(model, optimizer)
            self.coordinator.register(role, model, optimizer, identity=self.identity)
            if role in self.workers and self.workers[role] is not worker:
                raise ExchangeError("distinct worker already registered for role")
            self.workers[role] = worker
        if getattr(worker, "_is_ref", False):
            self.backend.validate_reference(worker)
            if self.reference is not None and self.reference is not worker:
                raise ExchangeError("distinct reference worker already registered")
            self.reference = worker
        self.flush("registered:" + (role or "reference"))

    def begin(self, phase: str, *, activate: str | None = None) -> None:
        self.check_identity()
        if self.busy:
            raise ExchangeError("another full worker operation is in progress")
        self.busy = True
        if activate is not None:
            self.coordinator.activate(activate, identity=self.identity)
        else:
            self.coordinator.assert_ready("actor", identity=self.identity)
        self.flush(phase + ":enter")

    def finish(self, phase: str, role: str) -> None:
        self.coordinator.assert_ready(role, identity=self.identity)
        self.flush(phase + ":success")
        self.busy = False

    def fail(self, error: BaseException, phase: str) -> None:
        self.coordinator._fail(error)
        self.busy = False
        self.flush(phase + ":error")


def patch_worker_classes(
    actor_class: type[Any], critic_class: type[Any], registry: WorkerRegistry
) -> bool:
    """CPU-injectable seam. Installation validates source/version before calling."""
    # Opt-in only; activate in both arms of a fresh run, never mid-run.
    initialization_mode = os.environ.get("FINANCE_SCREENING_CRITIC_INIT")
    if initialization_mode:
        from src.agents.critic_initialization import patch_critic_initialization

        patch_critic_initialization(critic_class, mode=initialization_mode)
    expected = {
        "init_model": ("self",),
        "update_actor": ("self", "data"),
        "update_critic": ("self", "data"),
        "compute_log_prob": ("self", "data"),
        "compute_ref_log_prob": ("self", "data"),
        "compute_values": ("self", "data"),
        "rollout_mode": ("self",),
        "save_checkpoint": ("self", "local_path", "hdfs_path", "global_step", "max_ckpt_to_keep"),
        "load_checkpoint": ("self", "local_path", "hdfs_path", "del_local_after_load"),
    }
    mappings = (
        (
            actor_class,
            "actor",
            (
                "init_model",
                "update_actor",
                "save_checkpoint",
                "load_checkpoint",
                "compute_log_prob",
                "compute_ref_log_prob",
                "rollout_mode",
            ),
        ),
        (
            critic_class,
            "critic",
            ("init_model", "update_critic", "save_checkpoint", "load_checkpoint", "compute_values"),
        ),
    )
    if actor_class.__dict__.get("_finance_manual_reference_profile") != {
        "profile": PROFILE,
        "identity": registry.identity,
    }:
        raise ExchangeError("matching reference adapter must install before placement wrappers")
    markers = [klass.__dict__.get(MARKER) for klass, _, _ in mappings]
    identity_marker = (registry.identity, registry.pid)
    if any(markers):
        if markers == [identity_marker, identity_marker]:
            return False
        raise ExchangeError("placement class installation conflicts with existing identity")
    originals = {}
    for klass, _, names in mappings:
        for name in names:
            original = getattr(klass, name)
            if tuple(inspect.signature(original).parameters) != expected[name]:
                raise ExchangeError(f"upstream signature changed: {klass.__name__}.{name}")
            if inspect.iscoroutinefunction(original) != (name == "rollout_mode"):
                raise ExchangeError(f"upstream coroutine behavior changed: {name}")
            originals[klass, name] = original

    def wrap(original: Any, role: str, name: str) -> Any:
        if name == "rollout_mode":

            @wraps(original)
            async def rollout(worker: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    registry.begin(name)
                    result = await original(worker, *args, **kwargs)
                    registry.finish(name, "actor")
                    return result
                except BaseException as error:
                    registry.fail(error, name)
                    raise

            return rollout

        @wraps(original)
        def wrapped(worker: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                if name == "load_checkpoint":
                    raise ExchangeError(
                        "pageable_adam_v1 is fresh-only; checkpoint loading is unsupported"
                    )
                if name == "init_model":
                    registry.check_identity()
                    registered_role = role if role == "critic" or worker._is_actor else None
                    if registered_role in registry.workers or registry.reference is worker:
                        raise ExchangeError("worker init_model already completed")
                    if registered_role is not None:
                        registry.backend.validate_worker(worker, registered_role)
                    result = original(worker, *args, **kwargs)
                    registry.register(worker, registered_role)
                    return result
                target = role if name in ("update_actor", "update_critic") else None
                if role == "actor" and name != "compute_ref_log_prob" and not worker._is_actor:
                    raise ExchangeError("a non-actor worker cannot use an actor operation")
                registry.begin(name, activate=target)
                result = original(worker, *args, **kwargs)
                registry.finish(name, target or "actor")
                return result
            except BaseException as error:
                registry.fail(error, name)
                raise

        return wrapped

    for klass, role, names in mappings:
        for name in names:
            setattr(klass, name, wrap(originals[klass, name], role, name))
        setattr(klass, MARKER, identity_marker)
    return True


def install_from_environment(
    *,
    gpu_reserve_bytes: int = 2 * 1024**3,
    host_reserve_bytes: int = 2 * 1024**3,
) -> bool:
    """Install only in explicitly configured screening Ray workers."""
    global _REGISTRY
    if os.environ.get(PROFILE_ENV) != PROFILE:
        return False
    identity = _identity()
    directory = os.environ.get(DIRECTORY_ENV, "")
    if not directory or not Path(directory).is_absolute():
        raise ExchangeError("placement diagnostics require an explicit absolute directory")
    sources = verify_installed_sources()
    if _REGISTRY is not None:
        _REGISTRY.check_identity()
        return False
    import verl.workers.fsdp_workers as workers

    worker_file = Path(
        str(metadata.distribution("verl").locate_file("verl/workers/fsdp_workers.py"))
    )
    for klass, names in (
        (
            workers.ActorRolloutRefWorker,
            ("init_model", "update_actor", "save_checkpoint", "load_checkpoint"),
        ),
        (
            workers.CriticWorker,
            ("init_model", "update_critic", "save_checkpoint", "load_checkpoint"),
        ),
    ):
        for name in names:
            original = inspect.unwrap(getattr(klass, name))
            if Path(original.__code__.co_filename).resolve() != worker_file.resolve():
                raise ExchangeError(f"unexpected in-memory worker implementation: {name}")
    emit = Evidence(Path(directory), identity)
    backend = TorchBackend(
        gpu_reserve_bytes=gpu_reserve_bytes, host_reserve_bytes=host_reserve_bytes
    )
    registry = WorkerRegistry(identity=identity, backend=backend, emit=emit)
    installed = patch_worker_classes(workers.ActorRolloutRefWorker, workers.CriticWorker, registry)
    _REGISTRY = registry
    emit(
        "installed",
        sources=sources,
        gpu_reserve_bytes=gpu_reserve_bytes,
        host_reserve_bytes=host_reserve_bytes,
    )
    return installed
