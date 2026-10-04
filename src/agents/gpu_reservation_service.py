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
    if op in {"status", "pause", "stop"}:
        if set(request) - {"op", "request_id"}:
            raise ValueError("unexpected command fields")
        return {"status": holder.status, "pause": holder.pause, "stop": holder.close}[op]()
    if op == "release":
        if set(request) - {"op", "request_id", "bytes"}:
            raise ValueError("unexpected release fields")
        amount = request.get("bytes")
        if type(amount) is not int or amount <= 0:
            raise ValueError("release bytes must be a positive integer")
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
             f"最晚退出：{metadata['expires_at_SGT']}。", "",
             "仅申请并持有空张量，不运行推理或训练；增长有总量和显存余量限制。",
             "释放后其他进程也可申请，无法保证训练接手。当前尚未接入 4B 训练。", "",
             "控制命令：status、pause、release --gib N、target --gib N --resume、stop。",
             "release 会暂停自动增长；恢复增长需要显式 target --resume。", ""]
    if state.get("error"):
        lines += ["错误：" + str(state["error"]), ""]
    (directory / "LIVE_STATUS.md").write_text("\n".join(lines))
    return record


def serve(args, *, backend_factory=TorchBackend):
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
    expires = datetime.now(ZoneInfo("Asia/Singapore")) + timedelta(hours=args.ttl_hours)
    metadata = dict(
        schema_version=1, pid=os.getpid(), started_at_SGT=now(),
        expires_at_SGT=expires.isoformat(timespec="seconds"),
        directory=str(directory), cuda_device=args.device,
        control_socket=str(socket_path),
        allocation_only=True, training_connected=False, automatic_restart=False,
        context_memory_included_in_allocator_limit=False,
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT / "src/agents/gpu_reservation.py")},
    )
    atomic_json(directory / "owner.json", metadata)

    def signal_stop(_number, _frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, signal_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    replies = {}
    last_write = last_tick = 0.0
    try:
        backend = backend_factory(device=args.device)
        remaining = args.ttl_hours * 3600 - (time.monotonic() - lease_started)
        if remaining <= 0:
            raise RuntimeError("lease expired during backend initialization")
        holder = Reservation(
            backend, target_bytes=args.target_gib,
            maximum_bytes=args.max_gib, block_bytes=args.block_gib,
            reserve_bytes=args.reserve_gib, ttl_seconds=remaining,
        )
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
                    if key in replies:
                        old_command, reply = replies[key]
                        if old_command != command:
                            raise ValueError("request_id reused for a different command")
                    else:
                        if command.get("op") != "status" and len(replies) >= 10000:
                            raise RuntimeError("mutation command limit reached; inspect status")
                        result = handle_request(holder, command)
                        reply = dict(ok=True, state=write_status(directory, metadata, result))
                        # Keep every mutation id for this lease: a delayed retry
                        # must never free a second block after a lost response.
                        if command.get("op") != "status":
                            replies[key] = (command, reply)
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
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("serve", "status", "pause", "release", "target", "stop")
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gib", type=gib_bytes)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-id", help="reuse on an uncertain control-command retry")
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
        if not 0 < args.max_gib <= 60 * GIB:
            parser.error("maximum allocator occupancy must be in (0, 60] GiB")
        if not 0 <= args.target_gib <= args.max_gib or not 0 < args.block_gib <= args.max_gib:
            parser.error("target and block size must fit the maximum")
        return serve(args)
    payload = dict(op=args.command)
    if args.request_id is not None:
        payload["request_id"] = args.request_id
    if args.command in {"release", "target"}:
        if args.gib is None:
            parser.error("release and target require --gib")
        payload["bytes"] = args.gib
    if args.command == "target":
        payload["resume"] = args.resume
    result = request(args.directory, payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
