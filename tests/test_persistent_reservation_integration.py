"""CPU integration checks for persistent reservation and unchanged RL execution."""

from __future__ import annotations

import asyncio
import copy
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_gpu_memory_coordinator import config as _coordinator_fixture
from test_low_memory_pilot import Clock, command_for
from test_low_memory_pilot import prepared as _prepared_fixture
from test_screening_placement import Backend, worker_classes

from src.agents import gpu_memory_coordinator as memory
from src.agents import gpu_reservation_service as holder_service
from src.agents import low_memory_pilot as runner
from src.agents import parameter_grid as grid
from src.agents.metadata import RunHashes
from src.agents.screening_placement import (
    IDENTITY_ENV,
    PROFILE,
    PROFILE_ENV,
    WorkerRegistry,
    patch_worker_classes,
)

GIB = memory.GIB
IDENTITY = "a" * 64


@pytest.fixture
def coordinator_config(tmp_path):
    return _coordinator_fixture.__wrapped__(tmp_path)


@pytest.fixture
def prepared(tmp_path):
    return _prepared_fixture.__wrapped__(tmp_path)


@pytest.mark.parametrize("cell_id", ["m1", "m2_eta025"])
def test_coordinator_binding_reaches_real_ray_without_changing_training(
    prepared, coordinator_config, cell_id,
):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    base, _, _ = prepared
    _, original_command = command_for(prepared, cell_id)
    path = base / "gpu_coordinator_config.json"
    digest = memory.freeze_config(path, coordinator_config)
    config, command = command_for(prepared, cell_id)
    assert config["screening"]["gpu_coordinator_config_sha256"] == digest
    assert "gpu_coordinator_config.json" in runner.input_files(base)
    with hydra.initialize_config_dir(
        config_dir=str(runner.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        before = hydra.compose(config_name="agent_ppo_trainer", overrides=original_command[3:])
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    for group in ("data", "actor_rollout_ref", "critic", "algorithm", "trainer"):
        assert OmegaConf.to_container(actual[group], resolve=True) == OmegaConf.to_container(
            before[group], resolve=True,
        )
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert str(path) == env.FINANCE_GPU_COORDINATOR_CONFIG
    assert digest == env.FINANCE_GPU_COORDINATOR_CONFIG_SHA256
    assert "FINANCE_GPU_HANDOFF_CONFIG" not in env
    assert actual.actor_rollout_ref.actor.optim.lr == 2e-7
    assert actual.critic.optim.lr == 1e-6
    assert actual.trainer.total_training_steps == 2
    assert actual.data.apply_chat_template_kwargs.enable_thinking is False
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == (0 if cell_id == "m1" else 0.25)


def test_modified_coordinator_cannot_reuse_a_frozen_command(prepared, coordinator_config):
    base, settings, _ = prepared
    path = base / "gpu_coordinator_config.json"
    digest = memory.freeze_config(path, coordinator_config)
    config, _ = command_for(prepared, "m1")
    changed = runner.restart.read(path)
    changed["wait_timeout_seconds"] += 1
    runner.atomic_json(path, changed)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        memory.load_config(path, digest)
    with pytest.raises(RuntimeError, match="coordinator differs from execution identity"):
        runner.build_command(
            config=config, run_settings=runner.cell_settings(settings, "m1"),
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / "m1",
        )


def _queue_fixture(tmp_path, monkeypatch, coordinator_config, sample):
    clock = Clock()
    commands = []
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(runner, "handoff_config", lambda base: None)
    monkeypatch.setattr(runner, "memory_coordinator_config", lambda base: coordinator_config)

    def claim(config):
        assert config == coordinator_config
        commands.append("claim")
        return {}

    monkeypatch.setattr(memory, "controller_claim", claim)
    monkeypatch.setattr(memory, "admission_snapshot", sample)
    return SimpleNamespace(clock=clock, commands=commands, directory=tmp_path)


def _queue_sample(config, free_gib, held_gib):
    return dict(
        gpu_uuid=config["gpu_uuid"], driver_free_bytes=free_gib * GIB,
        driver_total_bytes=96 * GIB, holder_payload_bytes=held_gib * GIB,
        available_bytes=(free_gib + held_gib) * GIB,
        required_peak_bytes=76 * GIB,
        holder_status={"target_bytes": 76 * GIB, "payload_bytes": held_gib * GIB},
    )


@pytest.mark.parametrize("enough_eventually", [True, False])
def test_production_queue_counts_actual_held_bytes_not_nominal_target(
    tmp_path, monkeypatch, coordinator_config, enough_eventually,
):
    readings = []

    def sample(config):
        if enough_eventually and readings:
            result = _queue_sample(config, 6, 70)
        else:
            result = _queue_sample(config, 10, 0)
        readings.append(result)
        return result

    fixture = _queue_fixture(tmp_path, monkeypatch, coordinator_config, sample)
    if enough_eventually:
        runner.wait_gpu(tmp_path, "m1", timeout=100)
        state = runner.restart.read(tmp_path / "gpu_queue_state.json")
        assert state["status"] == "ADMITTED"
        assert state["minimum_free_MiB"] == 76 * 1024
        assert state["free_MiB"] == 6 * 1024
        assert state["holder_payload_bytes"] == 70 * GIB
        assert state["available_bytes"] == 76 * GIB
        assert fixture.clock.sleeps == [45]
        assert fixture.commands == ["claim"]
    else:
        with pytest.raises(RuntimeError, match="76 GiB GPU capacity"):
            runner.wait_gpu(tmp_path, "m1", timeout=50)
        state = runner.restart.read(tmp_path / "gpu_queue_state.json")
        assert state["status"] == "WAIT_TIMEOUT"
        assert fixture.commands == []
        assert fixture.clock.sleeps == [45, 5]
    assert readings[0]["available_bytes"] == 10 * GIB
    # Queue preparation cannot release the actual holder before worker admission.
    assert not (tmp_path / "m1").exists()


def test_dead_persistent_holder_stops_queue_without_launch_or_retry(
    tmp_path, monkeypatch, coordinator_config,
):
    def sample(config):
        raise RuntimeError("persistent holder is not live and healthy")

    fixture = _queue_fixture(tmp_path, monkeypatch, coordinator_config, sample)
    with pytest.raises(RuntimeError, match="not live and healthy"):
        runner.wait_gpu(tmp_path, "m1", timeout=100)
    state = runner.restart.read(tmp_path / "gpu_queue_state.json")
    assert state["status"] == "HANDOFF_UNAVAILABLE"
    assert fixture.commands == [] and fixture.clock.sleeps == []
    assert not (tmp_path / "m1").exists()


def test_grid_optional_coordination_shares_one_fresh_lease_and_preserves_default(
    tmp_path, monkeypatch,
):
    calls = []

    def prepare(base, **kwargs):
        calls.append((base, kwargs))
        base.mkdir()

    monkeypatch.setattr(grid.runner, "prepare", prepare)
    monkeypatch.setattr(grid, "refresh_status", lambda base: None)
    grid.prepare(tmp_path / "default", model_path=Path("/model"))
    assert len(calls) == 2
    assert all("coordinated_reservation_directory" not in row for _, row in calls)
    assert all("coordinator_token" not in row for _, row in calls)
    assert all(row["reservation_directory"] is None for _, row in calls)
    tokens = []
    for name in ("coordinated-one", "coordinated-two"):
        calls.clear()
        grid.prepare(
            tmp_path / name, model_path=Path("/model"),
            coordinated_reservation_directory=tmp_path / "persistent-holder",
        )
        assert [row["profile"].seed for _, row in calls] == [42, 43]
        leases = [row["coordinator_token"] for _, row in calls]
        assert leases[0] == leases[1] and len(leases[0]) == 32
        assert all(row["coordinated_reservation_directory"] == tmp_path / "persistent-holder"
                   and row["reservation_directory"] is None for _, row in calls)
        tokens.append(leases[0])
    assert tokens[0] != tokens[1]


def test_absent_current_holder_pointer_keeps_existing_cli_default(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    assert runner.current_reservation_directory() is None


@pytest.mark.parametrize("field", ["owner_sha256", "holder_pid", "holder_start_time_ticks"])
def test_current_holder_pointer_rejects_replaced_identity(
    tmp_path, monkeypatch, coordinator_config, field,
):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    pointer = {
        key: coordinator_config[key] for key in (
            "profile", "directory", "owner_sha256", "holder_pid", "holder_start_time_ticks",
        )
    }
    path = tmp_path / "runs/GPU_RESERVATION_CURRENT.json"
    path.parent.mkdir()
    monkeypatch.setattr(holder_service, "process_identity", lambda pid: {
        "pid": pid, "start_time_ticks": coordinator_config["holder_start_time_ticks"],
    })
    runner.atomic_json(path, pointer)
    assert runner.current_reservation_directory() == Path(coordinator_config["directory"])
    changed = copy.deepcopy(pointer)
    changed[field] = "f" * 64 if field == "owner_sha256" else changed[field] + 1
    runner.atomic_json(path, changed)
    with pytest.raises(RuntimeError, match="holder identity changed"):
        runner.current_reservation_directory()


@pytest.mark.parametrize("entrypoint", [runner, grid])
def test_fresh_cli_prepare_automatically_propagates_current_holder(
    tmp_path, monkeypatch, entrypoint,
):
    directory = tmp_path / "holder"
    calls = []
    monkeypatch.setattr(runner, "current_reservation_directory", lambda: directory)
    monkeypatch.setattr(entrypoint, "prepare", lambda base, **kwargs: calls.append((base, kwargs)))
    monkeypatch.setattr(sys, "argv", ["fixture", "prepare", "--directory", str(tmp_path / "new")])
    entrypoint.main()
    assert len(calls) == 1
    assert calls[0][0] == tmp_path / "new"
    assert calls[0][1]["coordinated_reservation_directory"] == directory
    if entrypoint is runner:
        assert calls[0][1]["reservation_directory"] is None


def _placement_fixture(monkeypatch, denied_phase=None):
    monkeypatch.setenv(PROFILE_ENV, PROFILE)
    monkeypatch.setenv(IDENTITY_ENV, IDENTITY)
    monkeypatch.delenv("FINANCE_SCREENING_CRITIC_INIT", raising=False)
    monkeypatch.delenv("FINANCE_GPU_HANDOFF_CONFIG", raising=False)
    events = []
    denial = MemoryError("fixture GPU phase admission denied")

    class Coordinator:
        def before_phase(self, phase):
            events.append(("before", phase))
            if phase == denied_phase:
                raise denial

        def after_phase(self, phase):
            events.append(("after", phase))

        def fail(self, error, phase):
            events.append(("failed", phase, error))

    monkeypatch.setattr(memory, "coordinator_from_environment", lambda identity: Coordinator())
    actor_class, critic_class = worker_classes()
    for worker_class, role in ((actor_class, "actor"), (critic_class, "critic")):
        original = worker_class.init_model

        def instrument_init(original, role):
            @wraps(original)
            def init(worker):
                actual_role = "reference" if role == "actor" and not worker._is_actor else role
                events.append(("original", "init_" + actual_role))
                return original(worker)

            return init

        worker_class.init_model = instrument_init(original, role)
    original_rollout = actor_class.rollout_mode

    @wraps(original_rollout)
    async def rollout(worker):
        events.append(("original", "rollout_mode"))
        return await original_rollout(worker)

    actor_class.rollout_mode = rollout
    registry = WorkerRegistry(identity=IDENTITY, backend=Backend(), emit=lambda *a, **kw: None)
    patch_worker_classes(actor_class, critic_class, registry)
    return SimpleNamespace(
        actor=actor_class(), critic=critic_class(), reference=actor_class(actor=False, ref=True),
        registry=registry, events=events, denial=denial,
    )


def test_actual_placement_checks_capacity_before_each_init_and_engine_wake(monkeypatch):
    fixture = _placement_fixture(monkeypatch)
    for worker in (fixture.critic, fixture.reference, fixture.actor):
        assert worker.init_model() is worker.result
    assert asyncio.run(fixture.actor.rollout_mode()) is fixture.actor.result
    assert fixture.events == [
        (action, phase)
        for phase in ("init_critic", "init_reference", "init_actor", "rollout_mode")
        for action in ("before", "original", "after")
    ]
    assert not fixture.registry.coordinator.failed


@pytest.mark.parametrize("denied_phase", ["init_critic", "rollout_mode"])
def test_placement_admission_failure_does_not_call_original_or_retry(monkeypatch, denied_phase):
    fixture = _placement_fixture(monkeypatch, denied_phase)
    if denied_phase == "rollout_mode":
        for worker in (fixture.critic, fixture.reference, fixture.actor):
            worker.init_model()
        def invoke():
            return asyncio.run(fixture.actor.rollout_mode())

        worker = fixture.actor
    else:
        invoke = fixture.critic.init_model
        worker = fixture.critic
    previous_calls = list(worker.calls)
    with pytest.raises(MemoryError) as caught:
        invoke()
    assert caught.value is fixture.denial
    assert worker.calls == previous_calls
    assert ("original", denied_phase) not in fixture.events
    assert ("after", denied_phase) not in fixture.events
    assert fixture.events[-1] == ("failed", denied_phase, fixture.denial)
    assert fixture.registry.coordinator.failed
