"""The serial queue must fail closed before it can launch an experiment."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts/run_queue.sh"


def _queue_environment(tmp_path: Path, **overrides: str) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "uv-invocations.log"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_UV_LOG"
if [[ "$*" == *"src.agents.run"* && "$*" == *"--preflight-only"* ]]; then
  if [[ -n "${FAKE_PREFLIGHT_FAIL_CONFIG:-}" && "$*" == *"$FAKE_PREFLIGHT_FAIL_CONFIG"* ]]; then
    exit 41
  fi
elif [[ "$*" == *"src.agents.run"* ]]; then
  if [[ -n "${FAKE_TRAIN_FAIL_CONFIG:-}" && "$*" == *"$FAKE_TRAIN_FAIL_CONFIG"* ]]; then
    exit 42
  fi
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_UV_LOG": str(invocation_log),
            "FINANCE_AGENT_LOG_ROOT": str(tmp_path / "queue-logs"),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            **overrides,
        }
    )
    return environment, invocation_log


def _run_queue(tmp_path: Path, *arguments: str, **overrides: str) -> tuple[int, list[str]]:
    environment, invocation_log = _queue_environment(tmp_path, **overrides)
    result = subprocess.run(
        [str(QUEUE), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    invocations = (
        invocation_log.read_text(encoding="utf-8").splitlines()
        if invocation_log.exists()
        else []
    )
    return result.returncode, invocations


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("configs/arm_a1.yaml",),
        ("--formal", "configs/smoke_1p7b.yaml"),
    ],
)
def test_queue_rejects_implicit_or_mixed_authorization_without_starting_uv(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    returncode, invocations = _run_queue(tmp_path, *arguments)

    assert returncode != 0
    assert invocations == []


def test_queue_allows_only_the_explicit_smoke_and_preflights_it_first(tmp_path: Path) -> None:
    returncode, invocations = _run_queue(tmp_path, "configs/smoke_1p7b.yaml")

    assert returncode == 0
    agent_runs = [line for line in invocations if "src.agents.run" in line]
    assert len(agent_runs) == 2
    assert "--preflight-only" in agent_runs[0]
    assert "--preflight-only" not in agent_runs[1]
    assert all(" --no-sync " in f" {line} " for line in invocations)


def test_formal_queue_preflight_failure_starts_no_training(tmp_path: Path) -> None:
    returncode, invocations = _run_queue(
        tmp_path,
        "--formal",
        "configs/arm_a1.yaml",
        "configs/arm_a2.yaml",
        FAKE_PREFLIGHT_FAIL_CONFIG="arm_a2.yaml",
    )

    assert returncode == 41
    agent_runs = [line for line in invocations if "src.agents.run" in line]
    assert len(agent_runs) == 2
    assert all("--preflight-only" in line for line in agent_runs)


def test_formal_queue_training_failure_stops_later_configs(tmp_path: Path) -> None:
    returncode, invocations = _run_queue(
        tmp_path,
        "--formal",
        "configs/arm_a1.yaml",
        "configs/arm_a2.yaml",
        FAKE_TRAIN_FAIL_CONFIG="arm_a1.yaml",
    )

    assert returncode == 42
    agent_runs = [line for line in invocations if "src.agents.run" in line]
    preflights = [line for line in agent_runs if "--preflight-only" in line]
    training = [line for line in agent_runs if "--preflight-only" not in line]
    assert len(preflights) == 2
    assert len(training) == 1
    assert "arm_a1.yaml" in training[0]
