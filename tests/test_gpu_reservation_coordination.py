"""CPU proofs for cgroup-bound handoff, idle recapture and retry persistence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agents import gpu_reservation_service as service
from src.agents.gpu_reservation import GIB, Allocation, Reservation


class Backend:
    def __init__(self):
        self.allocated = 0

    def snapshot(self):
        return dict(free_bytes=100 * GIB - self.allocated, total_bytes=100 * GIB,
                    allocator_allocated_bytes=self.allocated,
                    allocator_reserved_bytes=self.allocated)

    def allocate(self, nbytes):
        self.allocated += nbytes
        return Allocation(nbytes, object())

    def release(self, blocks):
        self.allocated -= sum(block.nbytes for block in blocks)
        for block in blocks:
            block.payload = None
        blocks.clear()

    @staticmethod
    def is_oom(error):
        return False


@pytest.fixture
def broker():
    backend = Backend()
    reservation = Reservation(backend, target_bytes=4 * GIB, maximum_bytes=80 * GIB,
                              persistent=True)
    owner = dict(pid=100, start_time_ticks=123, cgroup_path="/user.slice/train.service",
                 process_group_id=100, session_id=100)
    worker = dict(pid=101, start_time_ticks=124, cgroup_path="/user.slice/train.service",
                  process_group_id=101, session_id=101)
    processes = {100: owner, 101: worker}
    telemetry = {"populated": True, "error": None}

    def identity(pid):
        if pid not in processes:
            raise ProcessLookupError(pid)
        return dict(processes[pid])

    def populated(path):
        assert path == "/user.slice/train.service"
        if telemetry["error"]:
            raise telemetry["error"]
        return telemetry["populated"]

    holder = service.CoordinatedReservation(reservation, identity_reader=identity,
                                            populated_reader=populated)
    for _ in range(4):
        holder.tick()
    return holder, backend, owner, worker, processes, telemetry


def command(holder, op, **fields):
    return service.handle_request(holder, dict(op=op, **fields))


def test_claim_pause_release_refill_and_idle_recapture(broker):
    holder, backend, owner, worker, processes, telemetry = broker
    state = command(holder, "claim", lease_token="run", owner=owner)
    assert state["paused"] and state["payload_bytes"] == 4 * GIB
    command(holder, "attach", lease_token="run", workers=[worker])
    command(holder, "pause", lease_token="run")
    assert holder.tick()["payload_bytes"] == 4 * GIB
    command(holder, "release", lease_token="run", bytes=4 * GIB)
    for _ in range(8):
        assert holder.tick()["payload_bytes"] == 0
    command(holder, "refill", lease_token="run", target_bytes=2 * GIB)
    holder.tick()
    assert holder.tick()["payload_bytes"] == 2 * GIB
    processes.clear()
    telemetry["populated"] = False
    state = holder.tick()
    assert state["claim"] is None and state["payload_bytes"] == 3 * GIB
    assert state["last_claim_end"]["reason"] == "cgroup_empty"
    assert holder.tick()["payload_bytes"] == 4 * GIB
    assert backend.allocated == 4 * GIB


def test_owner_death_does_not_refill_until_reparented_worker_group_is_empty(broker):
    holder, _, owner, worker, processes, telemetry = broker
    command(holder, "claim", lease_token="run", owner=owner, workers=[worker])
    command(holder, "release", lease_token="run", bytes=4 * GIB)
    del processes[owner["pid"]]
    # Worker uses a different process group; cgroup covers it after reparenting.
    for _ in range(10):
        state = holder.tick()
        assert state["claim"] and state["payload_bytes"] == 0
    with pytest.raises(RuntimeError, match="still populated"):
        command(holder, "unclaim", lease_token="run")
    del processes[worker["pid"]]
    telemetry["populated"] = False
    assert command(holder, "unclaim", lease_token="run")["claim"] is None
    assert holder.tick()["payload_bytes"] == GIB


def test_unknown_cgroup_telemetry_fails_paused_and_cannot_refill(broker):
    holder, _, owner, _, _, telemetry = broker
    command(holder, "claim", lease_token="run", owner=owner)
    command(holder, "release", lease_token="run", bytes=4 * GIB)
    telemetry["error"] = PermissionError("cannot read scope")
    state = holder.tick()
    assert state["paused"] and state["payload_bytes"] == 0
    assert state["claim"]["cgroup_populated"] is None
    assert "PermissionError" in state["claim"]["watchdog_error"]
    with pytest.raises(RuntimeError, match="verified live"):
        command(holder, "refill", lease_token="run", target_bytes=GIB)
    telemetry["error"] = None
    assert holder.tick()["claim"]["cgroup_populated"] is True
    assert holder.tick()["payload_bytes"] == 0  # never automatic claimed refill


def test_process_moved_out_of_empty_scope_prevents_idle_growth(broker):
    holder, _, owner, _, processes, telemetry = broker
    command(holder, "claim", lease_token="run", owner=owner)
    command(holder, "release", lease_token="run", bytes=4 * GIB)
    processes[100]["cgroup_path"] = "/other.service"
    telemetry["populated"] = False
    state = holder.tick()
    assert state["claim"] and state["paused"] and state["payload_bytes"] == 0
    assert "outside an empty cgroup" in state["claim"]["watchdog_error"]


@pytest.mark.parametrize("op,fields", [
    ("pause", {}), ("release", {"bytes": GIB}),
    ("refill", {"target_bytes": GIB}), ("unclaim", {}),
    ("attach", {"workers": []}),
])
def test_claimed_mutations_require_matching_lease(broker, op, fields):
    holder, _, owner, _, _, _ = broker
    command(holder, "claim", lease_token="run", owner=owner)
    with pytest.raises(RuntimeError, match="lease_token"):
        command(holder, op, lease_token="another-run", **fields)
    assert holder.status()["payload_bytes"] == 4 * GIB


def test_legacy_target_cannot_restart_growth_during_live_claim(broker):
    holder, _, owner, _, _, _ = broker
    command(holder, "claim", lease_token="run", owner=owner)
    with pytest.raises(RuntimeError, match="claim-bound"):
        command(holder, "target", bytes=GIB, resume=True)
    assert holder.status()["paused"]


def test_claim_rejects_pid_reuse_wrong_cgroup_and_non_dedicated_scope(broker):
    holder, _, owner, _, processes, _ = broker
    with pytest.raises(RuntimeError, match="does not match"):
        command(holder, "claim", lease_token="run", owner={**owner, "start_time_ticks": 1})
    with pytest.raises(RuntimeError, match="does not match"):
        command(holder, "claim", lease_token="run", owner={**owner, "cgroup_path": "/x"})
    processes[100]["cgroup_path"] = "/user@1001.service/session.scope"
    with pytest.raises(ValueError, match="dedicated"):
        command(holder, "claim", lease_token="run", owner=processes[100])
    assert holder.status()["claim"] is None


def test_attach_rejects_other_training_cgroup_and_is_deduplicated(broker):
    holder, _, owner, worker, processes, _ = broker
    command(holder, "claim", lease_token="run", owner=owner)
    command(holder, "attach", lease_token="run", workers=[worker, worker])
    assert len(holder.status()["claim"]["workers"]) == 1
    processes[101]["cgroup_path"] = "/other.service"
    with pytest.raises(RuntimeError, match="outside"):
        command(holder, "attach", lease_token="run", workers=[processes[101]])


def test_emergency_stop_releases_claimed_holder_without_touching_training(broker):
    holder, backend, owner, worker, processes, _ = broker
    command(holder, "claim", lease_token="run", owner=owner, workers=[worker])
    assert command(holder, "stop")["closed"]
    assert backend.allocated == 0 and len(processes) == 2


def test_finite_default_service_rejects_claim_commands():
    holder = Reservation(Backend(), target_bytes=GIB)
    with pytest.raises(ValueError, match="explicit persistent"):
        command(holder, "claim", lease_token="run", owner={})


def test_disk_retry_ledger_has_no_lifetime_mutation_quota(tmp_path):
    path = tmp_path / "responses.sqlite3"
    ledger = service.MutationLedger(path)
    for index in range(10050):
        request = dict(op="release", bytes=index + 1, request_id=str(index))
        ledger.remember(str(index), request, dict(ok=True, state={"released": index + 1}))
    assert not hasattr(ledger, "replies")
    ledger.close()
    ledger = service.MutationLedger(path)
    old = dict(op="release", bytes=1, request_id="0")
    assert ledger.lookup("0", old)["state"]["released"] == 1
    with pytest.raises(ValueError, match="different command"):
        ledger.lookup("0", {**old, "bytes": 2})
    assert ledger.lookup("new", old) is None
    ledger.close()


def test_process_identity_and_kernel_scope_reader(tmp_path, monkeypatch):
    identity = service.process_identity(os.getpid())
    assert identity["pid"] == os.getpid() and identity["start_time_ticks"] > 0
    assert identity["cgroup_path"].startswith("/")
    monkeypatch.setattr(service, "CGROUP_ROOT", tmp_path)
    scope = tmp_path / "test.service"
    scope.mkdir()
    events = scope / "cgroup.events"
    events.write_text("populated 1\nfrozen 0\n")
    assert service.cgroup_populated("/test.service")
    events.write_text("populated 0\nfrozen 0\n")
    assert not service.cgroup_populated("/test.service")
    events.write_text("frozen 0\n")
    with pytest.raises(RuntimeError, match="unavailable"):
        service.cgroup_populated("/test.service")
    events.unlink()
    assert not service.cgroup_populated("/test.service")
    with pytest.raises(ValueError):
        service.cgroup_populated("/../other.service")
    with pytest.raises(ValueError):
        service.process_identity(True)
    assert Path(identity["cgroup_path"]).is_absolute()
