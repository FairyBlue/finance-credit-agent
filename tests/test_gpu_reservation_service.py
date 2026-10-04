"""Exercise the real leased socket service in separate CPU-only processes."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from src.agents import gpu_reservation_service as service

REPO = Path(__file__).resolve().parents[1]

# The actual service loop, signals, file lock and Unix socket run in a child.
# Only allocator operations are replaced: byte counts stand in for GPU tensors.
CHILD = r"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from src.agents import gpu_reservation_service as service
from src.agents.gpu_reservation import Allocation

config = json.loads(sys.argv[1])
root = Path(config['root'])
directory = Path(config['directory'])
service.ROOT = root
service.__file__ = str(root / 'src/agents/gpu_reservation_service.py')

class FakeBackend:
    def __init__(self, device=0):
        assert device == 0
        self.allocated = 0
        self.allocations = 0
        time.sleep(config.get('initialization_delay', 0))

    def record(self, kind, amount):
        with (directory / 'allocator.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(kind=kind, bytes=amount,
                                         allocated=self.allocated)) + '\n')

    def snapshot(self):
        return dict(free_bytes=256-self.allocated, total_bytes=256,
                    allocator_allocated_bytes=self.allocated,
                    allocator_reserved_bytes=self.allocated)

    def allocate(self, nbytes):
        self.allocations += 1
        if self.allocations == config.get('fail_allocation_number'):
            raise RuntimeError('injected allocator failure')
        self.allocated += nbytes
        self.record('allocate', nbytes)
        return Allocation(nbytes, object())

    def release(self, blocks):
        amount = sum(block.nbytes for block in blocks)
        for block in blocks:
            block.payload = None
        blocks.clear()
        self.allocated -= amount
        self.record('release', amount)
        marker = directory / 'fail-next-release'
        if amount and marker.exists():
            marker.unlink()
            raise RuntimeError('injected release failure')

    def is_oom(self, error):
        return False

args = SimpleNamespace(directory=directory, device=0,
                       target_gib=config.get('target', 24), max_gib=64,
                       block_gib=8, reserve_gib=4,
                       ttl_hours=config.get('ttl_seconds', 30)/3600,
                       poll_seconds=0.1)
result = service.serve(args, backend_factory=FakeBackend)
assert 'torch' not in sys.modules, 'CPU protocol tests must not import torch'
raise SystemExit(result)
"""


class Daemon:
    def __init__(self, process, directory, log):
        self.process, self.directory, self.log = process, directory, log

    @property
    def socket_path(self):
        return service.control_socket(self.directory)

    def connect(self):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(self.socket_path))
        return client

    def raw(self, data, *, half_close=False):
        with self.connect() as client:
            client.sendall(data)
            if half_close:
                client.shutdown(socket.SHUT_WR)
            return read_reply(client)

    def command(self, op, **fields):
        return service.request(self.directory, {"op": op, **fields}, timeout=5)

    def wait_state(self, predicate, *, timeout=5):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            assert self.process.poll() is None, self.log.read_text()
            last = self.command("status")
            if predicate(last):
                return last
            time.sleep(0.025)
        pytest.fail(f"service state did not match: {last}")

    def finish(self, expected=0, *, timeout=5):
        assert self.process.wait(timeout=timeout) == expected, self.log.read_text()
        assert not self.socket_path.exists()
        return json.loads((self.directory / "status.json").read_text())


def read_reply(client):
    data = bytearray()
    while b"\n" not in data:
        part = client.recv(8192)
        assert part, "service disconnected before replying"
        data.extend(part)
        assert len(data) < 1024 * 1024
    return json.loads(data.split(b"\n", 1)[0])


@pytest.fixture
def daemons():
    # Keep AF_UNIX paths well below Linux's 108-byte limit and isolate the
    # project-global flock from any real keeper launched by a user.
    root = Path(tempfile.mkdtemp(prefix="gpu-service-test-"))
    sources = root / "src/agents"
    sources.mkdir(parents=True)
    for name in ("gpu_reservation.py", "gpu_reservation_service.py"):
        shutil.copyfile(REPO / "src/agents" / name, sources / name)
    children = []

    def start(*, ready=True, **options):
        directory = root / f"run-{len(children)}"
        log = root / f"child-{len(children)}.log"
        config = dict(root=str(root), directory=str(directory), **options)
        with log.open("w") as stream:
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", CHILD, json.dumps(config)],
                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
            )
        daemon = Daemon(process, directory, log)
        children.append(daemon)
        if ready:
            deadline = time.monotonic() + 5
            while not daemon.socket_path.exists():
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.01)
            daemon.command("status")
        return daemon

    try:
        yield start
    finally:
        for daemon in children:
            if daemon.process.poll() is None:
                daemon.process.terminate()
                try:
                    daemon.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    daemon.process.kill()
                    daemon.process.wait(timeout=5)
        shutil.rmtree(root)


def test_pause_release_and_explicit_resume_control_growth(daemons):
    daemon = daemons()
    state = daemon.wait_state(lambda row: row["payload_bytes"] == 24)
    assert state["status"] == "HOLDING"
    assert state["allocation_only"] and not state["training_connected"]
    assert Path(state["directory"]) == daemon.directory
    assert os.stat(daemon.directory).st_mode & 0o777 == 0o700
    assert os.stat(daemon.socket_path).st_mode & 0o777 == 0o600
    assert state["at_SGT"].endswith("+08:00")

    assert daemon.command("pause")["paused"]
    state = daemon.command("release", bytes=9)
    assert state["released_payload_bytes"] == 16  # Whole blocks, explicitly disclosed.
    assert state["payload_bytes"] == 8 and state["paused"]
    time.sleep(0.4)
    assert daemon.command("status")["payload_bytes"] == 8
    state = daemon.command("target", bytes=32, resume=False)
    assert state["paused"] and state["target_bytes"] == 32
    time.sleep(0.4)
    assert daemon.command("status")["payload_bytes"] == 8
    daemon.command("target", bytes=32, resume=True)
    daemon.wait_state(lambda row: row["payload_bytes"] == 32)
    state = daemon.command("stop")
    assert state["closed"] and state["payload_bytes"] == 0
    final = daemon.finish()
    assert final["status"] == "CLOSED"
    assert final["allocator_allocated_bytes"] == 0


def test_release_request_id_survives_status_traffic_and_conflicts(daemons):
    daemon = daemons()
    daemon.wait_state(lambda row: row["payload_bytes"] == 24)
    first = daemon.command("release", bytes=8, request_id="release-once")
    for _ in range(140):
        assert daemon.command("status")["payload_bytes"] == 16
    replay = daemon.command("release", bytes=8, request_id="release-once")
    assert replay == first
    assert daemon.command("status")["payload_bytes"] == 16
    with pytest.raises(RuntimeError, match="reused for a different command"):
        daemon.command("release", bytes=16, request_id="release-once")
    rows = [json.loads(line) for line in (daemon.directory / "commands.jsonl")
            .read_text().splitlines()]
    assert sum(row["command"]["request_id"] == "release-once" for row in rows) == 1


def test_disconnect_after_release_allows_safe_readback_and_retry(daemons):
    daemon = daemons()
    daemon.wait_state(lambda row: row["payload_bytes"] == 24)
    payload = {"op": "release", "bytes": 8, "request_id": "lost-response"}
    with daemon.connect() as client:
        client.sendall(json.dumps(payload).encode() + b"\n")
        # Close without reading: the daemon must retain the mutation identity.
    state = daemon.wait_state(lambda row: row["payload_bytes"] == 16)
    assert state["paused"]
    replay = service.request(daemon.directory, payload)
    assert replay["released_payload_bytes"] == 8
    assert daemon.command("status")["payload_bytes"] == 16


@pytest.mark.parametrize("payload", [
    b"{broken\n",
    b"[]\n",
    b"null\n",
    b'{"op":"status"}\n',
    b'{"op":"unknown","request_id":"bad"}\n',
    b'{"op":"release","bytes":true,"request_id":"bad"}\n',
    b'{"op":"release","bytes":-1,"request_id":"bad"}\n',
    b'{"op":"target","bytes":65,"resume":true,"request_id":"bad"}\n',
    b'{"op":"target","bytes":4,"resume":"yes","request_id":"bad"}\n',
    b'{"op":"status","extra":1,"request_id":"bad"}\n',
    b'{"op":"status","request_id":[]}\n',
    b'{"op":[],"request_id":"bad"}\n',
])
def test_bad_commands_leave_service_usable_and_state_unchanged(daemons, payload):
    daemon = daemons(target=8)
    assert daemon.command("pause")["payload_bytes"] == 8
    reply = daemon.raw(payload)
    assert not reply["ok"] and reply["error"]
    state = daemon.command("status")
    assert state["payload_bytes"] == 8 and state["paused"]
    assert state["target_bytes"] == 8


def test_oversized_and_unterminated_requests_are_rejected(daemons):
    daemon = daemons(target=0)
    reply = daemon.raw(b"x" * (service.MAX_MESSAGE + 1))
    assert not reply["ok"] and "too large" in reply["error"]
    reply = daemon.raw(b'{"op":"status"}', half_close=True)
    assert not reply["ok"] and "newline" in reply["error"]
    assert daemon.command("status")["payload_bytes"] == 0


def test_receive_deadline_bounds_slow_partial_clients(daemons):
    daemon = daemons(target=0)
    with daemon.connect() as client:
        start = time.monotonic()
        client.sendall(b"{")
        time.sleep(1.7)
        client.sendall(b'"op"')
        reply = read_reply(client)
        elapsed = time.monotonic() - start
    assert not reply["ok"] and "Timeout" in reply["error"]
    assert elapsed < 4.2, "per-recv idle timeout lets a slow client extend the lease"
    assert daemon.command("status")["payload_bytes"] == 0


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_closes_and_releases_allocations(daemons, sig):
    daemon = daemons(target=8)
    assert daemon.command("status")["payload_bytes"] == 8
    daemon.process.send_signal(sig)
    final = daemon.finish()
    assert final["status"] == "CLOSED" and final["payload_bytes"] == 0
    assert final["allocator_reserved_bytes"] == 0


def test_lease_expiry_is_terminal_and_releases_memory(daemons):
    daemon = daemons(target=8, ttl_seconds=0.6)
    state = daemon.command("target", bytes=16, resume=True)
    assert state["remaining_seconds"] <= 0.6
    final = daemon.finish()
    assert final["status"] == "EXPIRED" and final["closed"]
    assert final["payload_bytes"] == 0 and final["allocator_reserved_bytes"] == 0


def test_backend_initialization_does_not_extend_lease(daemons):
    daemon = daemons(ready=False, ttl_seconds=0.1, initialization_delay=0.3)
    assert daemon.process.wait(timeout=5) != 0
    failure = json.loads((daemon.directory / "failure.json").read_text())
    assert "lease expired during backend initialization" in failure["error"]
    assert not daemon.socket_path.exists()
    assert not (daemon.directory / "allocator.jsonl").exists()


def test_project_lock_blocks_second_holder_and_is_released_on_exit(daemons):
    first = daemons(target=0)
    second = daemons(ready=False, target=0)
    assert second.process.wait(timeout=5) != 0
    assert "another project GPU holder is running" in second.log.read_text()
    assert not second.socket_path.exists()
    first.command("stop")
    first.finish()
    third = daemons(target=8)
    assert third.command("status")["payload_bytes"] == 8


def test_allocator_failure_exits_nonzero_and_preserves_failure(daemons):
    daemon = daemons(ready=False, fail_allocation_number=2)
    final = daemon.finish(expected=1)
    assert final["status"] == "FAILED" and final["closed"]
    assert final["payload_bytes"] == 0 and final["allocator_allocated_bytes"] == 0
    assert "injected allocator failure" in final["error"]
    failure = json.loads((daemon.directory / "failure.json").read_text())
    assert "injected allocator failure" in failure["error"]


def test_control_release_failure_is_not_reported_as_successful_exit(daemons):
    daemon = daemons(target=8)
    (daemon.directory / "fail-next-release").touch()
    with pytest.raises(RuntimeError, match="injected release failure"):
        daemon.command("release", bytes=8)
    final = daemon.finish(expected=1)
    assert final["status"] == "FAILED" and final["payload_bytes"] == 0
    assert "injected release failure" in final["error"]


def test_client_helper_uses_same_id_for_uncertain_retry(daemons):
    daemon = daemons(target=8)
    request_id = uuid.uuid4().hex
    payload = {"op": "release", "bytes": 8, "request_id": request_id}
    first = service.request(daemon.directory, payload)
    second = service.request(daemon.directory, payload)
    assert first == second
    assert payload["request_id"] == request_id
    assert first["payload_bytes"] == 0 and first["paused"]
