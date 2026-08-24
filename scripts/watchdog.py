#!/usr/bin/env python3
"""Kill a collapsed training process so the serial queue can continue."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REWARD_PATTERN = re.compile(
    r"(?:reward(?:_score)?|train/reward)[\s\"']*[:=][\s\"']*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
RESPONSE_MARKERS = ("response_text", "generated_text", "model_output")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _record(events_file: Path, *, pid: int, reason: str) -> None:
    events_file.parent.mkdir(parents=True, exist_ok=True)
    with events_file.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "pid": pid,
                    "reason": reason,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _terminate(pid: int, *, grace_seconds: float = 10.0) -> None:
    if not _alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.25)
    if _alive(pid):
        os.kill(pid, signal.SIGKILL)


def monitor(
    *,
    pid: int,
    log_file: Path,
    events_file: Path,
    poll_seconds: float,
    zero_reward_limit: int,
    repeat_limit: int,
) -> int:
    if pid <= 1:
        raise ValueError("refusing to monitor a system-level PID")
    offset = 0
    zero_rewards = 0
    recent_outputs: deque[str] = deque(maxlen=repeat_limit)
    while _alive(pid):
        if log_file.exists():
            with log_file.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                lines = stream.readlines()
                offset = stream.tell()
            for line in lines:
                for match in REWARD_PATTERN.finditer(line):
                    reward = float(match.group(1))
                    zero_rewards = zero_rewards + 1 if abs(reward) <= 1e-12 else 0
                    if zero_rewards >= zero_reward_limit:
                        reason = f"reward remained zero for {zero_rewards} logged updates"
                        _record(events_file, pid=pid, reason=reason)
                        _terminate(pid)
                        return 2
                lowered = line.casefold()
                if any(marker in lowered for marker in RESPONSE_MARKERS):
                    normalized = " ".join(line.split())
                    recent_outputs.append(normalized)
                    if len(recent_outputs) == repeat_limit and len(set(recent_outputs)) == 1:
                        reason = f"identical model output repeated {repeat_limit} times"
                        _record(events_file, pid=pid, reason=reason)
                        _terminate(pid)
                        return 3
        time.sleep(poll_seconds)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--events-file", type=Path, default=Path("logs/watchdog_events.jsonl"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--zero-reward-limit", type=int, default=8)
    parser.add_argument("--repeat-limit", type=int, default=8)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.zero_reward_limit < 2 or args.repeat_limit < 2:
        parser.error("poll seconds must be positive and limits must be at least 2")
    return monitor(
        pid=args.pid,
        log_file=args.log_file,
        events_file=args.events_file,
        poll_seconds=args.poll_seconds,
        zero_reward_limit=args.zero_reward_limit,
        repeat_limit=args.repeat_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
