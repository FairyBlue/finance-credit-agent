"""Real CPU AdamW integration through phase gating and disk moment exchange."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.agents.chunked_placement import (  # noqa: E402
    PROFILE,
    ChunkedWorkerRegistry,
    DiskChunkCoordinator,
)
from src.agents.screening_exchange import ExchangeError  # noqa: E402
from src.agents.screening_placement import IDENTITY_ENV, PROFILE_ENV  # noqa: E402

IDENTITY = "c" * 64


class Backend:
    gpu_device = "cuda:0"

    def __init__(self):
        self.admissions = []
        self.synchronizations = 0
        self.deny_host = False

    def operations(self, boundary_check):
        # The inherited constructor creates a coordinator which is then replaced;
        # none of the old exchange Operations are used by the new registry.
        return None

    def admit(self, device, size):
        self.admissions.append((device, size))
        if self.deny_host and device == "cpu":
            raise MemoryError("fixture host admission denied")

    def synchronize(self):
        self.synchronizations += 1

    def validate_worker(self, worker, role):
        assert role in {"actor", "critic"}

    def register_training(self, model, optimizer):
        self.validate_training(model)
        assert not optimizer.state

    def validate_training(self, model):
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ExchangeError("fixture live gradient at full update boundary")

    def validate_reference(self, worker):
        if worker.reference_failed:
            raise ExchangeError("fixture reference failed")

    def metrics(self):
        return {"cpu_test_backend": True}


def model_and_optimizer(offset=0):
    model = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.linspace(-0.3, 0.4, parameter.numel()).reshape(parameter.shape)
                            + offset / 10 + index / 20)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": parameters[:2], "lr": 0.01, "weight_decay": 0.13},
         {"params": parameters[2:], "lr": 0.003, "betas": (0.7, 0.98), "maximize": True}],
        foreach=False, fused=False, weight_decay=0.09,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    return model, optimizer, scheduler


def registry_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, PROFILE)
    monkeypatch.setenv(IDENTITY_ENV, IDENTITY)
    monkeypatch.setenv("FINANCE_SCREENING_ARTIFACT_PROFILE", "actor_bf16_hf_v1")
    backend = Backend()
    records = []
    registry = ChunkedWorkerRegistry(
        identity=IDENTITY, backend=backend,
        emit=lambda event, **fields: records.append((event, fields)),
        directory=tmp_path, chunk_bytes=64, disk_reserve_bytes=0,
    )
    models = {}
    reference = SimpleNamespace(_is_ref=True, reference_failed=False)
    registry.register(reference, None)
    for offset, role in enumerate(("actor", "critic")):
        model, optimizer, scheduler = model_and_optimizer(offset)
        if role == "actor":
            worker = SimpleNamespace(
                actor_module_fsdp=model, actor_optimizer=optimizer, _is_ref=False,
            )
        else:
            worker = SimpleNamespace(critic_module=model, critic_optimizer=optimizer)
        registry.register(worker, role)
        models[role] = model, optimizer, scheduler
    return registry, backend, models, records


def accumulate_and_clip(model, round_index):
    for microbatch in range(2):
        inputs = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 20
        inputs = inputs + microbatch / 10 + round_index / 20
        target = torch.tensor([[0.1, 0.5], [0.2, -0.1], [-0.3, 0.4]])
        loss = (model(inputs) - target).square().mean() / 2
        loss.backward()
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.37, foreach=False)


def assert_matches_stock(actual, expected):
    model, optimizer, scheduler = actual
    expected_model, expected_optimizer, expected_scheduler = expected
    assert scheduler.state_dict() == expected_scheduler.state_dict()
    assert [g["lr"] for g in optimizer.param_groups] == [
        g["lr"] for g in expected_optimizer.param_groups
    ]
    for parameter, expected_parameter in zip(
        model.parameters(), expected_model.parameters(), strict=True,
    ):
        torch.testing.assert_close(parameter, expected_parameter, rtol=0, atol=0)
        state = optimizer.state.get(parameter, {})
        expected_state = expected_optimizer.state.get(expected_parameter, {})
        assert set(state) == set(expected_state)
        for key, value in state.items():
            assert value.device.type == "cpu" and not value.is_pinned()
            torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)


def test_two_alternating_rounds_match_stock_and_preserve_schedulers(
    tmp_path, monkeypatch, recwarn,
):
    registry, backend, candidates, records = registry_fixture(tmp_path, monkeypatch)
    references = {role: model_and_optimizer(offset)
                  for offset, role in enumerate(("actor", "critic"))}
    identities = {role: id(values[1]) for role, values in candidates.items()}
    for round_index in range(2):
        for role in ("critic", "actor"):
            phase = "update_" + role
            registry.begin(phase, activate=role)
            assert registry.current_phase == phase
            actual = candidates[role]
            expected = references[role]
            torch.testing.assert_close(
                accumulate_and_clip(actual[0], round_index),
                accumulate_and_clip(expected[0], round_index), rtol=0, atol=0,
            )
            expected[1].step()
            expected[2].step()
            actual[1].step()
            actual[2].step()
            assert_matches_stock(actual, expected)
            assert id(actual[1]) == identities[role] and type(actual[1]) is torch.optim.AdamW
            expected[1].zero_grad(set_to_none=True)
            actual[1].zero_grad(set_to_none=True)
            registry.finish(phase, role)
            assert not registry.busy and registry.current_phase is None
            inactive = "actor" if role == "critic" else "critic"
            inactive_optimizer = candidates[inactive][1]
            assert all(not ({"exp_avg", "exp_avg_sq"} & state.keys())
                       for state in inactive_optimizer.state.values())
            assert registry.coordinator.stores[inactive].status == "offloaded"
            manifest = json.loads(registry.coordinator.stores[inactive].manifest_path.read_text())
            assert manifest["role"] == inactive and manifest["identity"] == IDENTITY
    assert registry.coordinator.step_counts == {"actor": 2, "critic": 2}
    assert not registry.coordinator.failed
    assert records and backend.admissions and backend.synchronizations >= 4
    assert all(device == "cpu" for device, _ in backend.admissions)
    assert not any("optimizer.step" in str(warning.message) for warning in recwarn)
    # Each incoming restore must occur after outgoing moments were dropped and
    # the coordinator ran its reclamation stage within that same transition.
    transition = []
    restores = 0
    for event in registry.coordinator.events:
        if event["event"] == "exchange_start":
            transition = []
        transition.append(event)
        if event["event"] == "disk_state" and event["detail"]["event"] == "moments_restored":
            restores += 1
            assert any(row["event"] == "disk_state"
                       and row["detail"]["event"] == "moments_offloaded" for row in transition)
            assert any("reclaim" in row["event"] for row in transition)
    assert restores == 3


@pytest.mark.parametrize("phase", [None, "compute_log_prob", "save_checkpoint"])
def test_optimizer_step_outside_matching_worker_update_is_rejected(tmp_path, monkeypatch, phase):
    registry, _, models, _ = registry_fixture(tmp_path, monkeypatch)
    if phase is not None:
        registry.begin(phase)
    model, optimizer, _ = models["actor"]
    before = copy.deepcopy(model.state_dict())
    accumulate_and_clip(model, 0)
    with pytest.raises(ExchangeError):
        optimizer.step()
    assert registry.coordinator.failed and registry.coordinator.step_counts["actor"] == 0
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    with pytest.raises(ExchangeError):
        registry.begin("update_actor", activate="actor")


def test_wrong_role_optimizer_does_not_mutate_any_state(tmp_path, monkeypatch):
    registry, _, models, _ = registry_fixture(tmp_path, monkeypatch)
    registry.begin("update_critic", activate="critic")
    model, optimizer, _ = models["actor"]
    accumulate_and_clip(model, 0)
    with pytest.raises(ExchangeError):
        optimizer.step()
    assert registry.coordinator.failed and not optimizer.state
    assert registry.coordinator.step_counts == {"actor": 0, "critic": 0}


def test_changed_identity_during_update_is_rejected(tmp_path, monkeypatch):
    registry, _, models, _ = registry_fixture(tmp_path, monkeypatch)
    registry.begin("update_actor", activate="actor")
    model, optimizer, _ = models["actor"]
    accumulate_and_clip(model, 0)
    monkeypatch.setenv(IDENTITY_ENV, "d" * 64)
    with pytest.raises(ExchangeError):
        optimizer.step()
    assert registry.coordinator.failed and not optimizer.state


def test_wrong_finish_phase_fails_and_blocks_further_compute(tmp_path, monkeypatch):
    registry, _, _, _ = registry_fixture(tmp_path, monkeypatch)
    registry.begin("update_critic", activate="critic")
    with pytest.raises(ExchangeError):
        registry.finish("update_actor", "critic")
    assert registry.coordinator.failed
    with pytest.raises(ExchangeError):
        registry.begin("compute_values")


def test_host_admission_failure_is_permanent(tmp_path, monkeypatch):
    registry, backend, models, _ = registry_fixture(tmp_path, monkeypatch)
    registry.begin("update_actor", activate="actor")
    model, optimizer, _ = models["actor"]
    accumulate_and_clip(model, 0)
    backend.deny_host = True
    with pytest.raises(MemoryError, match="admission denied"):
        optimizer.step()
    registry.fail(MemoryError("fixture"), "update_actor")
    assert registry.coordinator.failed and not registry.busy and registry.current_phase is None
    with pytest.raises(ExchangeError):
        registry.begin("update_actor", activate="actor")


def test_disk_restore_failure_cannot_be_retried(tmp_path, monkeypatch):
    registry, _, models, _ = registry_fixture(tmp_path, monkeypatch)
    registry.begin("update_critic", activate="critic")
    model, optimizer, _ = models["critic"]
    accumulate_and_clip(model, 0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    registry.finish("update_critic", "critic")
    registry.begin("update_actor", activate="actor")
    registry.finish("update_actor", "actor")
    store = registry.coordinator.stores["critic"]
    manifest = json.loads(store.manifest_path.read_text())
    first = manifest["parameters"][0]["moments"]["exp_avg"][0]
    path = store.directory / manifest["generation"] / first["file"]
    path.write_bytes(b"x" * first["bytes"])
    with pytest.raises(ValueError, match="checksum"):
        registry.begin("update_critic", activate="critic")
    assert registry.coordinator.failed and store.status == "failed"
    with pytest.raises(ExchangeError):
        registry.begin("update_actor", activate="actor")


def test_boundary_rejects_live_gradients(tmp_path, monkeypatch):
    registry, _, models, _ = registry_fixture(tmp_path, monkeypatch)
    registry.begin("update_actor", activate="actor")
    accumulate_and_clip(models["actor"][0], 0)
    with pytest.raises(ExchangeError, match="live gradient"):
        registry.finish("update_actor", "actor")
    assert registry.coordinator.failed


def test_native_optimizer_checkpoint_is_not_claimed_supported(tmp_path, monkeypatch):
    registry, _, _, _ = registry_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("FINANCE_SCREENING_ARTIFACT_PROFILE")
    with pytest.raises(ExchangeError, match="actor inference exports"):
        registry.begin("save_checkpoint")
    assert registry.coordinator.failed and registry.current_phase is None


def test_direct_coordinator_register_rejects_initialized_optimizer(tmp_path):
    model, optimizer, _ = model_and_optimizer()
    accumulate_and_clip(model, 0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    coordinator = DiskChunkCoordinator(
        identity=IDENTITY, backend=Backend(), boundary_check=lambda: None,
        allow_step=lambda role: False,
        directory=tmp_path, chunk_bytes=64, disk_reserve_bytes=0,
    )
    with pytest.raises(ExchangeError, match="fresh"):
        coordinator.register("actor", model, optimizer, identity=IDENTITY)
