"""Reproducibility hashes injected into every upstream/W&B run config."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or f"git command failed in {repo}")
    return process.stdout.strip()


@dataclass(frozen=True, slots=True)
class RunHashes:
    agent_r1_commit: str
    repo_commit: str
    data_version: str
    repo_dirty: bool

    def as_config(self) -> dict[str, str | bool]:
        return asdict(self)


def collect_run_hashes(
    *,
    repo_root: Path,
    agent_r1_path: Path,
    data_version: str,
    expected_agent_r1_commit: str,
    require_clean_repo: bool,
) -> RunHashes:
    try:
        repo_commit = _git(repo_root, "rev-parse", "HEAD")
    except RuntimeError:
        repo_commit = "UNCOMMITTED"
    agent_commit = _git(agent_r1_path, "rev-parse", "HEAD")
    if agent_commit != expected_agent_r1_commit:
        raise RuntimeError(
            f"Agent-R1 checkout mismatch: expected {expected_agent_r1_commit}, got {agent_commit}"
        )
    dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=normal"))
    if require_clean_repo and (dirty or repo_commit == "UNCOMMITTED"):
        raise RuntimeError(
            "training requires a committed, clean repository; commit the scaffold first"
        )
    if not data_version.strip() or "TODO" in data_version.upper():
        raise RuntimeError("data.version must be an immutable, non-TODO identifier")
    return RunHashes(
        agent_r1_commit=agent_commit,
        repo_commit=repo_commit,
        data_version=data_version,
        repo_dirty=dirty,
    )
