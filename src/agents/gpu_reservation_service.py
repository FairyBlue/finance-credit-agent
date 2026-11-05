"""Bounded GPU memory holder with a local control socket and an expiring lease.

This is best-effort occupancy, not a cross-process allocation guarantee. The
training controller must pause/release it before attempting model allocations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import sqlite3
import sys
import time
import uuid
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.agents.gpu_reservation import Reservation, TorchBackend

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3
MAX_MESSAGE = 4096
TERMINAL = {"CLOSED", "EXPIRED", "FAILED"}
CGROUP_ROOT = Path("/sys/fs/cgroup")


def process_identity(pid):
    """Bind a same-user process to its Linux start tick and cgroup, not PID alone."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("pid must be a positive integer")
    path = Path(f"/proc/{pid}")
    if path.stat().st_uid != os.getuid():
        raise RuntimeError("claim process must belong to this user")
    stat = (path / "stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    if fields[0] in {"Z", "X"}:
        raise RuntimeError("claim process is no longer running")
    cgroups = (path / "cgroup").read_text().splitlines()
    unified = [line[3:] for line in cgroups if line.startswith("0::")]
    if len(unified) != 1:
        raise RuntimeError("claim requires unified cgroup v2")
    # Re-read the tick after reading the cgroup to detect PID reuse mid-read.
    second = (path / "stat").read_text()
    if second[second.rfind(")") + 2:].split()[19] != fields[19]:
        raise RuntimeError("process identity changed while reading")
    return dict(pid=pid, start_time_ticks=int(fields[19]), cgroup_path=unified[0],
                process_group_id=int(fields[2]), session_id=int(fields[3]))


def cgroup_populated(cgroup_path):
    """Read the kernel's full-group liveness, including reparented Ray workers."""
    if not isinstance(cgroup_path, str) or not cgroup_path.startswith("/"):
        raise ValueError("invalid cgroup path")
    components = Path(cgroup_path).parts[1:]
    if not components or any(part in {".", ".."} for part in components):
        raise ValueError("invalid cgroup path")
    path = CGROUP_ROOT.joinpath(*components) / "cgroup.events"
    try:
        rows = dict(line.split() for line in path.read_text().splitlines())
    except FileNotFoundError:
        # Kernel cgroups cannot be removed while populated. Only callers which
        # verified this scope existed when claiming may use absence as death.
        return False
    value = rows.get("populated")
    if value not in {"0", "1"}:
        raise RuntimeError("cgroup populated telemetry is unavailable")
    return value == "1"


class CoordinatedReservation:
    """An explicit persistent idle holder with one cgroup-bound training claim.

    A claim pauses growth before model allocation. Only its lease can release or
    refill. Controller death alone never starts idle growth: the kernel must
    report the entire dedicated cgroup empty. Unreadable telemetry fails paused.
    """

    def __init__(self, holder, *, identity_reader=process_identity,
                 populated_reader=cgroup_populated):
        self.holder = holder
        self.idle_target_bytes = holder.target_bytes
        self.identity_reader = identity_reader
        self.populated_reader = populated_reader
        self.claim_state = None
        self.last_claim_end = None

    def _identity(self, supplied):
        if not isinstance(supplied, dict):
            raise ValueError("process identity must be an object")
        allowed = {"pid", "start_time_ticks", "cgroup_path", "process_group_id", "session_id"}
        if set(supplied) - allowed:
            raise ValueError("unexpected process identity fields")
        if type(supplied.get("start_time_ticks")) is not int or supplied["start_time_ticks"] <= 0:
            raise ValueError("start_time_ticks must be a positive integer")
        actual = self.identity_reader(supplied.get("pid"))
        if any(actual.get(key) != supplied.get(key)
               for key in ("pid", "start_time_ticks", "cgroup_path")):
            raise RuntimeError("claim process identity does not match")
        return actual

    def _require_claim(self, token):
        if self.claim_state is None or token != self.claim_state["lease_token"]:
            raise RuntimeError("matching live claim lease_token is required")

    def _watch(self):
        if self.claim_state is None:
            return
        try:
            live = self.populated_reader(self.claim_state["owner"]["cgroup_path"])
            if type(live) is not bool:
                raise RuntimeError("invalid cgroup liveness")
            if not live:
                for member in [self.claim_state["owner"], *self.claim_state["workers"]]:
                    try:
                        actual = self.identity_reader(member["pid"])
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    except RuntimeError as error:
                        if "no longer running" in str(error):
                            continue
                        raise
                    if actual["start_time_ticks"] == member["start_time_ticks"]:
                        raise RuntimeError("claimed process remains live outside an empty cgroup")
            self.claim_state["cgroup_populated"] = live
            self.claim_state["watchdog_error"] = None
        except (OSError, RuntimeError, ValueError) as error:
            self.claim_state["cgroup_populated"] = None
            self.claim_state["watchdog_error"] = f"{type(error).__name__}: {error}"
            self.holder.pause()
            return
        if not live:
            self.last_claim_end = dict(lease_token=self.claim_state["lease_token"],
                                       reason="cgroup_empty", at_SGT=now())
            self.claim_state = None
            self.holder.set_target(self.idle_target_bytes, resume=True)

    def status(self):
        return {**self.holder.status(), "coordination_profile": "persistent_claim_v1",
                "idle_target_bytes": self.idle_target_bytes,
                "training_connected": self.claim_state is not None,
                "claim": self.claim_state, "last_claim_end": self.last_claim_end}

    def tick(self):
        self._watch()
        self.holder.tick()
        return self.status()

    def claim(self, token, owner, workers=()):
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise ValueError("lease_token must contain 1 to 128 characters")
        if self.claim_state is not None:
            self._require_claim(token)
            if self.claim_state["owner"] != self._identity(owner):
                raise RuntimeError("claim lease already belongs to a different owner")
            return self.status()
        actual = self._identity(owner)
        components = Path(actual["cgroup_path"]).parts
        if actual["pid"] == os.getpid() or not components[-1].endswith(".service"):
            raise ValueError("claim owner must run in a dedicated systemd service cgroup")
        # A missing/empty initial scope cannot establish a live lease.
        if self.populated_reader(actual["cgroup_path"]) is not True:
            raise RuntimeError("claim cgroup must exist and be populated")
        verified = self._workers(actual, workers)
        self.holder.pause()
        self.claim_state = dict(lease_token=token, owner=actual, workers=verified,
                               claimed_at_SGT=now(), cgroup_populated=True, watchdog_error=None)
        return self.status()

    def _workers(self, owner, workers):
        if not isinstance(workers, (list, tuple)) or len(workers) > 256:
            raise ValueError("workers must be a list of at most 256 identities")
        result = []
        for supplied in workers:
            actual = self._identity(supplied)
            scope = owner["cgroup_path"].rstrip("/")
            if actual["cgroup_path"] != scope and not actual["cgroup_path"].startswith(scope + "/"):
                raise RuntimeError("worker is outside the claimed training cgroup")
            if actual not in result:
                result.append(actual)
        return result

    def attach(self, token, workers):
        self._require_claim(token)
        verified = self._workers(self.claim_state["owner"], workers)
        combined = self.claim_state["workers"] + verified
        if len(combined) > 256:
            raise ValueError("claim worker identity limit reached")
        for worker in verified:
            if worker not in self.claim_state["workers"]:
                self.claim_state["workers"].append(worker)
        return self.status()

    def release(self, amount, *, token=None):
        if self.claim_state is not None:
            self._require_claim(token)
        result = self.holder.release(amount)
        return {**self.status(), "released_payload_bytes": result["released_payload_bytes"]}

    def refill(self, token, target):
        self._require_claim(token)
        self._watch()
        self._require_claim(token)
        if self.claim_state["cgroup_populated"] is not True:
            raise RuntimeError("refill requires verified live training cgroup")
        self.holder.set_target(target, resume=True)
        return self.status()

    def unclaim(self, token):
        self._require_claim(token)
        self._watch()
        if self.claim_state is not None:
            raise RuntimeError("training cgroup still populated or liveness is uncertain")
        return self.status()

    def pause(self, *, token=None):
        if self.claim_state is not None:
            self._require_claim(token)
        self.holder.pause()
        return self.status()

    def set_target(self, amount, *, resume=False):
        if self.claim_state is not None:
            raise RuntimeError("use claim-bound refill while training owns the holder")
        self.holder.set_target(amount, resume=resume)
        self.idle_target_bytes = amount
        return self.status()

    def close(self):
        self.holder.close()
        return self.status()


class MutationLedger:
    """Exact retry responses on disk; constant Python memory for durable service."""

    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS replies (request_id TEXT PRIMARY KEY, "
            "command TEXT NOT NULL, reply TEXT NOT NULL)")

    def lookup(self, key, command):
        row = self.connection.execute(
            "SELECT command, reply FROM replies WHERE request_id = ?", (key,)).fetchone()
        if row is None:
            return None
        if json.loads(row[0]) != command:
            raise ValueError("request_id reused for a different command")
        return json.loads(row[1])

    def remember(self, key, command, reply):
        self.connection.execute(
            "INSERT INTO replies VALUES (?, ?, ?)",
            (key, json.dumps(command, allow_nan=False), json.dumps(reply, allow_nan=False)))
        self.connection.commit()

    def close(self):
        self.connection.close()


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def gib_bytes(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("GiB must be finite and nonnegative")
    return int(number * GIB)


def control_socket(directory):
    """Keep AF_UNIX paths below Linux's limit even for long experiment paths."""
    runtime = Path(f"/run/user/{os.getuid()}")
    if not runtime.is_dir():
        runtime = Path(f"/tmp/fca-gpu-holder-{os.getuid()}")
        runtime.mkdir(mode=0o700, exist_ok=True)
    if runtime.is_symlink() or runtime.stat().st_uid != os.getuid():
        raise RuntimeError("unsafe local socket directory")
    key = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:24]
    return runtime / f"fca-gpu-{key}.sock"


def handle_request(holder, request):
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    op = request.get("op")
    if op in {"claim", "attach", "refill", "unclaim"}:
        if not isinstance(holder, CoordinatedReservation):
            raise ValueError("claim commands require the explicit persistent profile")
        fields = {"claim": {"owner", "workers"}, "attach": {"workers"},
                  "refill": {"target_bytes"}, "unclaim": set()}[op]
        if set(request) - {"op", "request_id", "lease_token"} - fields:
            raise ValueError("unexpected claim command fields")
        token = request.get("lease_token")
        if op == "claim":
            return holder.claim(token, request.get("owner"), request.get("workers", []))
        if op == "attach":
            return holder.attach(token, request.get("workers"))
        if op == "refill":
            return holder.refill(token, request.get("target_bytes"))
        return holder.unclaim(token)
    if op in {"status", "pause", "stop"}:
        allowed = {"op", "request_id"}
        if op == "pause" and isinstance(holder, CoordinatedReservation):
            allowed.add("lease_token")
        if set(request) - allowed:
            raise ValueError("unexpected command fields")
        if op == "pause" and isinstance(holder, CoordinatedReservation):
            return holder.pause(token=request.get("lease_token"))
        return {"status": holder.status, "pause": holder.pause, "stop": holder.close}[op]()
    if op == "release":
        allowed = {"op", "request_id", "bytes"}
        if isinstance(holder, CoordinatedReservation):
            allowed.add("lease_token")
        if set(request) - allowed:
            raise ValueError("unexpected release fields")
        amount = request.get("bytes")
        if type(amount) is not int or amount <= 0:
            raise ValueError("release bytes must be a positive integer")
        if isinstance(holder, CoordinatedReservation):
            return holder.release(amount, token=request.get("lease_token"))
        return holder.release(amount)
    if op == "target":
        if set(request) - {"op", "request_id", "bytes", "resume"}:
            raise ValueError("unexpected target fields")
        amount, resume = request.get("bytes"), request.get("resume", False)
        if type(amount) is not int or amount < 0 or type(resume) is not bool:
            raise ValueError("target bytes or resume flag invalid")
        return holder.set_target(amount, resume=resume)
    raise ValueError("unknown command")


def receive(connection):
    deadline = time.monotonic() + 3
    data = bytearray()
    while b"\n" not in data and len(data) <= MAX_MESSAGE:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("control request exceeded three seconds")
        connection.settimeout(remaining)
        part = connection.recv(min(1024, MAX_MESSAGE + 1 - len(data)))
        if not part:
            raise ValueError("missing newline terminated command")
        data.extend(part)
    if len(data) > MAX_MESSAGE:
        raise ValueError("command too large")
    return json.loads(data.split(b"\n", 1)[0])


def request(directory, payload, timeout=15):
    payload = {**payload, "request_id": payload.get("request_id", uuid.uuid4().hex)}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(control_socket(directory)))
        client.sendall(json.dumps(payload, allow_nan=False).encode() + b"\n")
        parts = bytearray()
        while b"\n" not in parts:
            data = client.recv(65536)
            if not data:
                raise RuntimeError("control connection closed before response; inspect status")
            parts.extend(data)
            if len(parts) > 1024 * 1024:
                raise RuntimeError("oversized control response")
    response = json.loads(parts.split(b"\n", 1)[0])
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "command failed"))
    return response["state"]


def write_status(directory, metadata, state):
    record = {**metadata, **state, "at_SGT": now()}
    atomic_json(directory / "status.json", record)
    # Field aliases are intentionally absent: expose the engine's actual schema.
    payload = state.get("held_bytes", state.get("payload_bytes", 0))
    lines = ["# GPU 显存占位试验", "", f"当前进展（新加坡时间）：{record['at_SGT']}", "",
             f"状态：**{state.get('status', 'UNKNOWN')}**；进程 PID：{os.getpid()}。",
             f"持有张量：{payload / GIB:.2f} GiB；"
             f"目标：{state.get('target_bytes', 0) / GIB:.2f} GiB。",
             ("有效期：显式持久占位，直到 stop 或服务终止。" if metadata.get("persistent")
              else f"最晚退出：{metadata['expires_at_SGT']}。"), "",
             "仅申请并持有空张量，不运行推理或训练；增长有总量和显存余量限制。",
             "释放后其他进程也可申请；实际训练分配前仍须检查可用显存。", "",
             "控制命令：status、pause、release --gib N、target --gib N --resume、stop。",
             "release 会暂停自动增长；恢复增长需要显式 target --resume。", ""]
    if metadata.get("persistent"):
        claim = state.get("claim")
        lines += ["持久协调命令：claim、attach、refill、unclaim。", "",
                  f"训练租约：{'已认领' if claim else '空闲占位'}；"
                  "只有完整训练 cgroup 退出后才自动恢复空闲增长。", ""]
        if claim:
            lines += [f"训练 cgroup：`{claim['owner']['cgroup_path']}`；"
                      f"仍有进程：{claim['cgroup_populated']}。", ""]
    if state.get("error"):
        lines += ["错误：" + str(state["error"]), ""]
    (directory / "LIVE_STATUS.md").write_text("\n".join(lines))
    return record


def serve(args, *, backend_factory=TorchBackend, identity_reader=process_identity,
          populated_reader=cgroup_populated):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise RuntimeError("run directory must be owned by this user")
    os.chmod(directory, 0o700)
    if (directory / "owner.json").exists():
        raise RuntimeError("holder directory already used; choose a new directory")
    lock_path = ROOT / "runs/.gpu-reservation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(lock_fd)
        raise RuntimeError("another project GPU holder is running") from None
    holder = None
    server = None
    stopped = False
    socket_path = control_socket(directory)
    lease_started = time.monotonic()
    persistent = getattr(args, "persistent", False)
    if type(persistent) is not bool:
        raise ValueError("persistent must be explicitly boolean")
    expires = (None if persistent else
               datetime.now(ZoneInfo("Asia/Singapore")) + timedelta(hours=args.ttl_hours))
    metadata = dict(
        schema_version=1, pid=os.getpid(), started_at_SGT=now(),
        expires_at_SGT=None if expires is None else expires.isoformat(timespec="seconds"),
        directory=str(directory), cuda_device=args.device,
        control_socket=str(socket_path),
        allocation_only=True, training_connected=False, automatic_restart=False,
        persistent=persistent,
        context_memory_included_in_allocator_limit=False,
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT / "src/agents/gpu_reservation.py")},
    )
    atomic_json(directory / "owner.json", metadata)

    def signal_stop(_number, _frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, signal_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    ledger = MutationLedger(directory / "mutation_replies.sqlite3")
    last_write = last_tick = 0.0
    try:
        backend = backend_factory(device=args.device)
        remaining = args.ttl_hours * 3600 - (time.monotonic() - lease_started)
        if not persistent and remaining <= 0:
            raise RuntimeError("lease expired during backend initialization")
        holder = Reservation(
            backend, target_bytes=args.target_gib,
            maximum_bytes=args.max_gib, block_bytes=args.block_gib,
            reserve_bytes=args.reserve_gib,
            ttl_seconds=args.ttl_hours * 3600 if persistent else remaining,
            persistent=persistent,
        )
        if persistent:
            holder = CoordinatedReservation(holder, identity_reader=identity_reader,
                                            populated_reader=populated_reader)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(4)
        server.settimeout(0.25)
        while not stopped:
            current = time.monotonic()
            if current - last_tick >= args.poll_seconds:
                state = holder.tick()
                last_tick = current
                if state.get("closed") or state.get("status") in TERMINAL:
                    break
            if current - last_write >= 5:
                write_status(directory, metadata, holder.status())
                last_write = current
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    command = receive(connection)
                    key = command.get("request_id") if isinstance(command, dict) else None
                    if not isinstance(key, str) or not 1 <= len(key) <= 128:
                        raise ValueError("request_id is required")
                    reply = ledger.lookup(key, command)
                    if reply is None:
                        result = handle_request(holder, command)
                        reply = dict(ok=True, state=write_status(directory, metadata, result))
                        # Disk journal keeps exact mutation retries without an
                        # ever-growing Python dict or a finite lifetime quota.
                        if command.get("op") != "status":
                            ledger.remember(key, command, reply)
                        with (directory / "commands.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(at_SGT=now(), command=command,
                                                         status=result.get("status"))) + "\n")
                    stopped = bool(reply.get("state", {}).get("closed")) or (
                        reply.get("state", {}).get("status") in TERMINAL
                    )
                except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                    reply = dict(ok=False, error=f"{type(error).__name__}: {error}")
                # The request may have committed even if its client disconnected.
                with suppress(BrokenPipeError, ConnectionResetError):
                    connection.sendall(json.dumps(reply, allow_nan=False).encode() + b"\n")
        return 1 if holder.status().get("status") == "FAILED" else 0
    except BaseException as error:
        atomic_json(directory / "failure.json", dict(at_SGT=now(), error=repr(error)))
        raise
    finally:
        try:
            if holder is not None:
                try:
                    state = holder.close()
                except BaseException as error:
                    atomic_json(directory / "failure.json", dict(at_SGT=now(), error=repr(error)))
                    write_status(directory, metadata, holder.status())
                    raise
                write_status(directory, metadata, state)
        finally:
            if server is not None:
                server.close()
            if socket_path.exists():
                socket_path.unlink()
            os.close(lock_fd)
            ledger.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("serve", "status", "pause", "release", "target", "stop",
                            "claim", "attach", "refill", "unclaim")
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gib", type=gib_bytes)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-id", help="reuse on an uncertain control-command retry")
    parser.add_argument("--lease-token", help="training claim identity")
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--worker-pid", type=int, action="append", default=[])
    parser.add_argument("--persistent", action="store_true",
                        help="explicit indefinite, cgroup-coordinated idle occupancy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target-gib", type=gib_bytes, default=30 * GIB)
    parser.add_argument("--max-gib", type=gib_bytes, default=59 * GIB)
    parser.add_argument("--block-gib", type=gib_bytes, default=GIB)
    parser.add_argument("--reserve-gib", type=gib_bytes, default=2 * GIB)
    parser.add_argument("--ttl-hours", type=float, default=6)
    parser.add_argument("--poll-seconds", type=float, default=1)
    args = parser.parse_args(argv)
    if args.command == "serve":
        if not math.isfinite(args.ttl_hours) or not 0 < args.ttl_hours <= 24:
            parser.error("ttl-hours must be in (0, 24]")
        if not math.isfinite(args.poll_seconds) or not 0.1 <= args.poll_seconds <= 60:
            parser.error("poll-seconds must be in [0.1, 60]")
        cap = 80 if args.persistent else 60
        if not 0 < args.max_gib <= cap * GIB:
            parser.error(f"maximum allocator occupancy must be in (0, {cap}] GiB")
        if args.persistent and args.reserve_gib < 2 * GIB:
            parser.error("persistent occupancy requires at least 2 GiB free headroom")
        if not 0 <= args.target_gib <= args.max_gib or not 0 < args.block_gib <= args.max_gib:
            parser.error("target and block size must fit the maximum")
        return serve(args)
    payload = dict(op=args.command)
    if args.request_id is not None:
        payload["request_id"] = args.request_id
    if args.lease_token is not None:
        payload["lease_token"] = args.lease_token
    if args.command in {"release", "target", "refill"}:
        if args.gib is None:
            parser.error("release and target require --gib")
        payload["target_bytes" if args.command == "refill" else "bytes"] = args.gib
    if args.command == "target":
        payload["resume"] = args.resume
    if args.command in {"claim", "attach", "refill", "unclaim"} and args.lease_token is None:
        parser.error("claim commands require --lease-token")
    if args.command == "claim":
        if args.owner_pid is None:
            parser.error("claim requires --owner-pid")
        payload["owner"] = process_identity(args.owner_pid)
        payload["workers"] = [process_identity(pid) for pid in args.worker_pid]
    if args.command == "attach":
        payload["workers"] = [process_identity(pid) for pid in args.worker_pid]
    result = request(args.directory, payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
