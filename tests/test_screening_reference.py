"""Reference adapter invariants with CPU tensors; CUDA initialization is forbidden."""

from __future__ import annotations

import inspect
import json
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.agents import screening_reference as adapter

IDENTITY = "a" * 64


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU prototype tests may not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setenv(adapter.PROFILE_ENV, adapter.PROFILE)
    monkeypatch.setenv(adapter.IDENTITY_ENV, IDENTITY)


class Handle:
    def __init__(self, offload: bool):
        self.flat_param = torch.nn.Parameter(torch.tensor([1, 2, 3], dtype=torch.bfloat16))
        self.flat_param._local_shard = self.flat_param.data
        self._offload_params = offload
        self._use_orig_params = False
        self._training_state = SimpleNamespace(name="IDLE")
        self.moves: list[tuple[str, bool]] = []
        self.module = SimpleNamespace()
        self.flat_param._param_infos = [("weight", self.module, "primary")]
        self.flat_param._shared_param_infos = [
            ("tied", self.module, "shared", "weight", self.module, "primary"),
        ]
        self._use_unsharded_views(as_params=False)

    def flat_param_to(self, device: torch.device, *, non_blocking: bool) -> None:
        assert device.type == "cpu"
        self.moves.append((str(device), non_blocking))
        # Force an actual new CPU storage so alias refresh is tested even on CPU.
        self.flat_param.data = self.flat_param.to(device, non_blocking=non_blocking, copy=True)

    def _use_unsharded_views(self, *, as_params: bool) -> None:
        assert as_params is False
        self.module.weight = self.flat_param[:]
        self.module.tied = self.module.weight


class Model:
    def __init__(self, role: str):
        self._all_handles = [Handle(offload=role != "actor"), Handle(offload=role != "actor")]
        self._is_root = False
        self.training_state = SimpleNamespace(name="IDLE")
        self.compute_device = torch.device("cpu")
        self.placement = "compute"


class CpuBackend(adapter.TorchBackend):
    """Exercise the real transfer/validation functions with CPU tensors and fake CUDA calls."""

    def __init__(self):
        self.sync_count = self.empty_count = 0
        self.fail_destinations: list[str] = []
        self.calls: list[str] = []
        self.torch = SimpleNamespace(
            bfloat16=torch.bfloat16,
            device=torch.device,
            no_grad=torch.no_grad,
            cuda=SimpleNamespace(synchronize=self.sync, empty_cache=self.empty),
        )

    def sync(self, device: torch.device) -> None:
        assert device.type == "cpu"
        self.sync_count += 1

    def empty(self) -> None:
        self.empty_count += 1

    def initialize(self, model: Model) -> None:
        self.calls.append("initialize")
        model._is_root = True
        self.validate(model)

    def move(self, model: Model, destination: str) -> None:
        self.calls.append(destination)
        if self.fail_destinations and self.fail_destinations[0] == destination:
            self.fail_destinations.pop(0)
            raise RuntimeError("injected transfer failure: " + destination)
        super().move(model, destination)
        model.placement = destination


def setup_worker(monkeypatch: pytest.MonkeyPatch):
    backend = CpuBackend()
    events = []

    class Worker:
        def __init__(self):
            self._is_ref = True
            self._is_lora = False
            self.config = SimpleNamespace(actor={"strategy": "fsdp"})
            self.builder_calls = []
            self.forward_calls = 0
            self.fail_forward = False
            self.ref_module_fsdp = None

        def _build_model_optimizer(
            self,
            model_path,
            fsdp_config,
            optim_config,
            override_model_config,
            use_remove_padding=False,
            use_fused_kernels=False,
            enable_gradient_checkpointing=False,
            trust_remote_code=False,
            use_liger=False,
            role="actor",
            enable_activation_offload=False,
            use_tiled_mlp=False,
            tiled_mlp_shards=4,
        ):
            self.builder_calls.append((model_path, fsdp_config, optim_config, role))
            optimizer = object() if optim_config is not None and role == "actor" else None
            return Model(role), optimizer, None, {"name": model_path}

        def compute_ref_log_prob(self, data):
            self.forward_calls += 1
            assert self.ref_module_fsdp.placement == "compute"
            assert not torch.is_grad_enabled()
            if self.fail_forward:
                raise ValueError("original forward failure")
            return data + self.ref_module_fsdp._all_handles[0].module.weight.sum()

    def emit(phase, event, **details):
        events.append({"phase": phase, "event": event, **details})

    Worker.compute_ref_log_prob._test_dispatch = {"mode": "actor"}
    assert adapter.patch_worker_class(Worker, backend=backend, emit=emit, identity=IDENTITY)
    return Worker, backend, events


def build_reference(worker, **overrides):
    kwargs = dict(
        model_path="fixed-reference",
        fsdp_config={"model_dtype": "bf16"},
        optim_config=None,
        override_model_config={},
        role="ref",
    )
    kwargs.update(overrides)
    result = worker._build_model_optimizer(**kwargs)
    worker.ref_module_fsdp = result[0]
    return result


def test_reference_round_trips_keep_values_aliases_and_events(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    result = build_reference(worker)
    assert result[1] is result[2] is None
    assert worker.builder_calls == [("fixed-reference", {"model_dtype": "bf16"}, None, "actor")]
    assert worker._is_ref and worker.config.actor["strategy"] == "fsdp"
    assert worker.ref_module_fsdp.placement == "cpu"
    for _ in range(3):
        output = worker.compute_ref_log_prob(torch.tensor(4.0))
        assert output.item() == 10
        assert not output.requires_grad
        assert worker.ref_module_fsdp.placement == "cpu"
        for handle in worker.ref_module_fsdp._all_handles:
            flat = handle.flat_param
            assert torch.equal(flat, torch.tensor([1, 2, 3], dtype=torch.bfloat16))
            assert flat.data_ptr() == flat._local_shard.data_ptr()
            assert handle.module.weight.untyped_storage().data_ptr() == flat.data_ptr()
            assert handle.module.tied is handle.module.weight
            assert not flat.is_pinned() and flat.grad is None
            assert not hasattr(flat, "_cpu_grad")
            assert all(non_blocking is False for _, non_blocking in handle.moves)
    assert worker.forward_calls == 3
    assert backend.calls == ["initialize", "cpu"] + ["compute", "cpu"] * 3
    assert backend.sync_count == 14 and backend.empty_count == 7
    assert torch.is_grad_enabled()
    assert any(
        event["phase"] == "reference_init" and event["event"] == "success" for event in events
    )
    assert (
        sum(
            event["phase"] == "reference_forward" and event["event"] == "success"
            for event in events
        )
        == 3
    )
    assert all(event.get("reference_cpu_grad_bytes", 0) == 0 for event in events)


def test_actor_builder_passes_through_without_reference_state(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    optimizer_config = {"lr": 1e-6}
    result = build_reference(worker, role="actor", optim_config=optimizer_config)
    assert result[1] is not None
    assert worker.builder_calls[0][2] is optimizer_config
    assert not hasattr(worker, adapter.STATE)
    assert backend.calls == [] and events == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("optim_config", {"lr": 1e-6}),
        ("fsdp_config", {"model_dtype": "fp32"}),
        ("fsdp_config", {"model_dtype": "bf16", "use_orig_params": True}),
        ("enable_gradient_checkpointing", True),
        ("enable_activation_offload", True),
        ("use_tiled_mlp", True),
    ],
)
def test_invalid_reference_configuration_fails_before_build(monkeypatch, field, value):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    with pytest.raises(RuntimeError, match="unsupported"):
        build_reference(worker, **{field: value})
    assert worker.builder_calls == [] and backend.calls == []


def test_original_forward_error_survives_cleanup_error_and_poisons_state(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    worker.fail_forward = True
    backend.fail_destinations = ["cpu"]
    with pytest.raises(ValueError, match="original forward failure"):
        worker.compute_ref_log_prob(torch.tensor(0.0))
    state = getattr(worker, adapter.STATE)
    assert state.failed and not state.busy
    assert any(
        event["phase"] == "reference_error_offload" and event["event"] == "error"
        for event in events
    )
    with pytest.raises(RuntimeError, match="failed"):
        worker.compute_ref_log_prob(torch.tensor(0.0))
    assert worker.forward_calls == 1


def test_load_failure_does_not_compute_and_attempts_cpu_cleanup(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    backend.fail_destinations = ["compute"]
    with pytest.raises(RuntimeError, match="injected transfer failure"):
        worker.compute_ref_log_prob(torch.tensor(0.0))
    assert worker.forward_calls == 0
    assert backend.calls[-2:] == ["compute", "cpu"]
    assert worker.ref_module_fsdp.placement == "cpu"


def test_successful_compute_but_failed_offload_is_a_failed_reference(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    backend.fail_destinations = ["cpu"]
    with pytest.raises(RuntimeError, match="transfer failure"):
        worker.compute_ref_log_prob(torch.tensor(0.0))
    assert getattr(worker, adapter.STATE).failed
    assert not any(
        event["phase"] == "reference_forward" and event["event"] == "success" for event in events
    )


@pytest.mark.parametrize("fault", ["cpu_grad", "grad", "alias", "view", "tied", "offload", "busy"])
def test_model_invariants_fail_before_forward(monkeypatch, fault):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    handle = worker.ref_module_fsdp._all_handles[0]
    if fault == "cpu_grad":
        handle.flat_param._cpu_grad = torch.zeros_like(handle.flat_param)
    elif fault == "grad":
        handle.flat_param.grad = torch.zeros_like(handle.flat_param)
    elif fault == "alias":
        handle.flat_param._local_shard = handle.flat_param.detach().clone()
    elif fault == "view":
        handle.module.weight = handle.module.weight.detach().clone()
    elif fault == "tied":
        handle.module.tied = handle.module.weight.detach().clone()
    elif fault == "offload":
        handle._offload_params = True
    else:
        getattr(worker, adapter.STATE).busy = True
    with pytest.raises(RuntimeError):
        worker.compute_ref_log_prob(torch.tensor(0.0))
    assert worker.forward_calls == 0


def test_identity_change_and_duplicate_install_are_rejected(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    assert not adapter.patch_worker_class(
        Worker, backend=backend, emit=lambda *a, **k: None, identity=IDENTITY
    )
    monkeypatch.setenv(adapter.IDENTITY_ENV, "b" * 64)
    with pytest.raises(RuntimeError, match="different identity"):
        adapter.patch_worker_class(
            Worker, backend=backend, emit=lambda *a, **k: None, identity="b" * 64
        )
    monkeypatch.setenv(adapter.IDENTITY_ENV, IDENTITY)
    worker = Worker()
    build_reference(worker)
    monkeypatch.setenv(adapter.IDENTITY_ENV, "b" * 64)
    with pytest.raises(RuntimeError, match="identity changed"):
        worker.compute_ref_log_prob(torch.tensor(0.0))
    assert worker.forward_calls == 0


def test_unset_profile_is_noop_before_dependency_loading(monkeypatch):
    monkeypatch.delenv(adapter.PROFILE_ENV)
    monkeypatch.setattr(adapter, "verify_installed_sources", lambda: pytest.fail("guard called"))
    assert adapter.install_from_environment() is False


def test_bad_source_is_rejected_before_runtime_import(tmp_path, monkeypatch):
    for name in adapter.SOURCE_HASHES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mismatching pinned source")
    monkeypatch.setattr(adapter.metadata, "version", lambda name: adapter.VERSIONS[name])
    monkeypatch.setattr(
        adapter.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda name: tmp_path / name),
    )
    with pytest.raises(RuntimeError, match="source mismatch"):
        adapter.verify_installed_sources()


def test_wrong_torch_version_is_rejected(monkeypatch):
    monkeypatch.setattr(adapter.metadata, "version", lambda name: "2.8.0")
    with pytest.raises(RuntimeError, match="requires torch"):
        adapter.verify_installed_sources()


def test_actual_installed_source_versions_match_guard():
    assert adapter.verify_installed_sources() == adapter.SOURCE_HASHES


def test_async_subclass_inherits_wrappers_and_dispatch_metadata(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)

    class AsyncWorker(Worker):
        async def wake_up(self):
            return True

    assert AsyncWorker.compute_ref_log_prob is Worker.compute_ref_log_prob
    assert AsyncWorker.compute_ref_log_prob._test_dispatch == {"mode": "actor"}
    assert inspect.iscoroutinefunction(AsyncWorker.wake_up)
    assert not inspect.iscoroutinefunction(AsyncWorker.compute_ref_log_prob)
    worker = AsyncWorker()
    build_reference(worker)
    assert worker.compute_ref_log_prob(torch.tensor(4.0)).item() == 10


def test_moving_reference_releases_old_module_views(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    handle = worker.ref_module_fsdp._all_handles[0]
    old_view = weakref.ref(handle.module.weight)
    old_storage_address = handle.module.weight.data_ptr()
    backend.move(worker.ref_module_fsdp, "compute")
    assert old_view() is None
    assert handle.module.weight.data_ptr() != old_storage_address
    assert handle.module.tied is handle.module.weight


@pytest.mark.parametrize("profile", sorted(adapter.PROFILES))
def test_supported_profiles_keep_reference_contract(monkeypatch, profile):
    monkeypatch.setenv(adapter.PROFILE_ENV, profile)
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    assert worker.compute_ref_log_prob(torch.tensor(4.0)).item() == 10


def test_profile_change_is_rejected_even_with_same_identity(monkeypatch):
    Worker, backend, events = setup_worker(monkeypatch)
    worker = Worker()
    build_reference(worker)
    monkeypatch.setenv(adapter.PROFILE_ENV, "pageable_adam_v1")
    with pytest.raises(RuntimeError, match="profile/identity changed"):
        worker.compute_ref_log_prob(torch.tensor(4.0))
    assert worker.forward_calls == 0


def test_event_log_is_metadata_and_io_failure_is_nonfatal(tmp_path):
    log = adapter.EventLog(tmp_path, IDENTITY)
    log("reference_init", "success", reference_cpu_grad_bytes=0)
    records = [
        json.loads(line)
        for path in tmp_path.glob("manual-ref-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["identity_sha256"] == IDENTITY
    assert records[0]["reference_cpu_grad_bytes"] == 0
    blocked = tmp_path / "ordinary-file"
    blocked.write_text("not a directory")
    adapter.EventLog(blocked, IDENTITY)("reference_init", "enter")
