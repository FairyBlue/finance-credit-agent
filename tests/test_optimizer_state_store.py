"""CPU checks for durable, bounded Adam moment swaps (not checkpoint recovery)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.agents.optimizer_state_store import MomentStore  # noqa: E402


def _optimizer(*, initialized=True):
    parameters = [torch.nn.Parameter(torch.arange(12.0).reshape(3, 4)),
                  torch.nn.Parameter(torch.tensor([2.0, -3.0]))]
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    if initialized:
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer, parameters


def _store(tmp_path, **kwargs):
    return MomentStore(tmp_path / "moments", {"run": "test-run"}, "actor",
                       chunk_bytes=16, disk_reserve_bytes=0, **kwargs)


def _payload_paths(store):
    return sorted(store.directory.glob("generation-*/*.raw"))


def test_round_trip_and_repeated_update_preserve_parameters_steps_and_state_identity(tmp_path):
    optimizer, parameters = _optimizer()
    admitted = []
    events = []
    store = _store(tmp_path, admit_host=admitted.append, emit=events.append)
    state_objects = [optimizer.state[parameter] for parameter in parameters]
    step_objects = [state["step"] for state in state_objects]
    parameter_values = [parameter.detach().clone() for parameter in parameters]
    for generation in (1, 2, 3):
        expected = [{key: state[key].clone() for key in ("exp_avg", "exp_avg_sq")}
                    for state in state_objects]
        step_values = [step.item() for step in step_objects]
        manifest_path = store.offload(optimizer)
        assert store.status == "offloaded"
        assert manifest_path.is_file()
        assert len(list(store.directory.glob("generation-*"))) == 1
        assert all(path.stat().st_size <= 16 for path in _payload_paths(store))
        manifest = json.loads(manifest_path.read_bytes())
        assert manifest["generation"] == f"generation-{generation:06d}"
        assert manifest["payload_bytes"] == sum(parameter.numel() * 8 for parameter in parameters)
        assert all(set(state) == {"step"} for state in state_objects)
        assert [step.item() for step in step_objects] == step_values
        store.restore(optimizer)
        assert store.status == "resident"
        for index, parameter in enumerate(parameters):
            state = optimizer.state[parameter]
            assert state is state_objects[index]
            assert state["step"] is step_objects[index]
            assert state["step"].item() == step_values[index]
            for key in expected[index]:
                torch.testing.assert_close(state[key], expected[index][key], rtol=0, atol=0)
            torch.testing.assert_close(parameter, parameter_values[index], rtol=0, atol=0)
        if generation < 3:
            for parameter in parameters:
                parameter.grad = torch.full_like(parameter, generation + 0.5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            parameter_values = [parameter.detach().clone() for parameter in parameters]
    assert admitted == [48, 48, 8, 8] * 3
    assert [event["event"] for event in events] == [
        "moments_offloaded", "moments_restored",
    ] * 3


def test_empty_optimizer_survives_offload_restore_and_initializes_later(tmp_path):
    optimizer, parameters = _optimizer(initialized=False)
    # Preserve even an explicitly existing empty dictionary.
    empty_state = optimizer.state[parameters[1]]
    store = _store(tmp_path)
    store.offload(optimizer)
    assert not _payload_paths(store)
    store.restore(optimizer)
    assert parameters[0] not in optimizer.state
    assert optimizer.state[parameters[1]] is empty_state
    assert empty_state == {}
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    store.offload(optimizer)
    store.restore(optimizer)
    assert all(optimizer.state[parameter]["step"].item() == 1 for parameter in parameters)


@pytest.mark.parametrize("damage", ["corruption", "truncation", "missing"])
def test_damaged_payload_fails_closed_preserves_evidence(tmp_path, damage):
    optimizer, _ = _optimizer()
    store = _store(tmp_path)
    store.offload(optimizer)
    path = _payload_paths(store)[0]
    if damage == "missing":
        path.unlink()
    elif damage == "truncation":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        payload = bytearray(path.read_bytes())
        payload[0] ^= 0xFF
        path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum|wrong size"):
        store.restore(optimizer)
    assert store.status == "failed"
    assert store.manifest_path.exists()
    assert (store.directory / "failure.json").exists()
    with pytest.raises(RuntimeError, match="failed"):
        store.restore(optimizer)


def test_late_write_failure_keeps_all_resident_state(tmp_path, monkeypatch):
    optimizer, parameters = _optimizer()
    store = _store(tmp_path)
    original = store._write_chunk
    call_count = 0

    def fail_late(path, data):
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            raise OSError("simulated disk write failure")
        return original(path, data)

    monkeypatch.setattr(store, "_write_chunk", fail_late)
    with pytest.raises(OSError, match="simulated"):
        store.offload(optimizer)
    assert store.status == "failed"
    assert not store.manifest_path.exists()
    assert len(_payload_paths(store)) == 3
    assert all("exp_avg" in optimizer.state[p] and "exp_avg_sq" in optimizer.state[p]
               for p in parameters)


def test_late_restore_failure_never_auto_rolls_back_or_retries(tmp_path, monkeypatch):
    optimizer, parameters = _optimizer()
    store = _store(tmp_path)
    store.offload(optimizer)
    original = store._read_chunk

    def fail_second_moment(path, chunk, target):
        if "exp_avg_sq" in path.name:
            raise OSError("late read failure")
        return original(path, chunk, target)

    monkeypatch.setattr(store, "_read_chunk", fail_second_moment)
    with pytest.raises(OSError, match="late read"):
        store.restore(optimizer)
    assert store.status == "failed"
    assert "exp_avg" in optimizer.state[parameters[0]]
    assert "exp_avg_sq" not in optimizer.state[parameters[0]]
    assert optimizer.state[parameters[0]]["step"].item() == 1
    assert "exp_avg" not in optimizer.state[parameters[1]]
    assert _payload_paths(store)
    with pytest.raises(RuntimeError, match="failed"):
        store.restore(optimizer)


@pytest.mark.parametrize("change", ["order", "shape", "step", "step_object", "state", "identity"])
def test_restore_rejects_changed_optimizer_contract_before_allocating(tmp_path, change):
    optimizer, parameters = _optimizer()
    allocations = []
    store = _store(tmp_path, admit_host=allocations.append)
    store.offload(optimizer)
    if change == "order":
        optimizer.param_groups[0]["params"].reverse()
    elif change == "shape":
        parameters[0].data = parameters[0].detach().reshape(2, 6)
    elif change == "step":
        optimizer.state[parameters[0]]["step"].add_(1)
    elif change == "step_object":
        optimizer.state[parameters[0]]["step"] = optimizer.state[parameters[0]]["step"].clone()
    elif change == "state":
        optimizer.state[parameters[0]] = dict(optimizer.state[parameters[0]])
    else:
        optimizer = torch.optim.AdamW(parameters)
    with pytest.raises(ValueError, match="changed"):
        store.restore(optimizer)
    assert not allocations
    assert store.status == "failed"


def test_host_admission_failure_allocates_no_next_moment(tmp_path):
    optimizer, parameters = _optimizer()
    requests = []

    def admit(nbytes):
        requests.append(nbytes)
        if len(requests) == 2:
            raise MemoryError("host reserve")

    store = _store(tmp_path, admit_host=admit)
    store.offload(optimizer)
    with pytest.raises(MemoryError, match="host reserve"):
        store.restore(optimizer)
    assert store.status == "failed"
    assert "exp_avg" in optimizer.state[parameters[0]]
    assert "exp_avg_sq" not in optimizer.state[parameters[0]]


@pytest.mark.parametrize("phase", ["offload", "restore"])
def test_boundary_requires_absent_gradients(tmp_path, phase):
    optimizer, parameters = _optimizer()
    store = _store(tmp_path)
    if phase == "restore":
        store.offload(optimizer)
    parameters[0].grad = torch.zeros_like(parameters[0])
    with pytest.raises(ValueError, match="gradients must be absent"):
        getattr(store, phase)(optimizer)
    assert store.status == "failed"


def test_disk_reserve_rejects_before_writing_payload(tmp_path, monkeypatch):
    optimizer, _ = _optimizer()
    store = _store(tmp_path)
    monkeypatch.setattr("src.agents.optimizer_state_store.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="insufficient disk"):
        store.offload(optimizer)
    assert not _payload_paths(store)
    assert store.status == "failed"


def test_changed_manifest_identity_rejected(tmp_path):
    optimizer, _ = _optimizer()
    store = _store(tmp_path)
    store.offload(optimizer)
    manifest = json.loads(store.manifest_path.read_bytes())
    manifest["identity"]["run"] = "wrong-run"
    store.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest identity"):
        store.restore(optimizer)


def test_foreign_files_in_old_generation_are_never_deleted(tmp_path):
    optimizer, _ = _optimizer()
    store = _store(tmp_path)
    store.offload(optimizer)
    foreign = store.directory / "generation-000001" / "foreign.txt"
    foreign.write_text("do not remove")
    store.restore(optimizer)
    with pytest.raises(ValueError, match="foreign files"):
        store.offload(optimizer)
    assert foreign.read_text() == "do not remove"
    assert (store.directory / "generation-000002").is_dir()
    assert json.loads(store.manifest_path.read_bytes())["generation"] == "generation-000002"
    assert store.status == "failed"


def test_store_requires_fresh_absolute_owned_directory(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        MomentStore(Path("relative"), "run", "actor")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        MomentStore(existing, "run", "actor")


def test_partial_or_non_cpu_fp32_moments_rejected(tmp_path):
    optimizer, parameters = _optimizer()
    del optimizer.state[parameters[0]]["exp_avg_sq"]
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="incomplete Adam moment pair"):
        store.offload(optimizer)


def test_amsgrad_is_explicitly_rejected(tmp_path):
    optimizer, parameters = _optimizer()
    optimizer.state[parameters[0]]["max_exp_avg_sq"] = torch.ones_like(parameters[0])
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="AMSGrad"):
        store.offload(optimizer)
