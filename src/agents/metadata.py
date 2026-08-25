"""Reproducibility hashes injected into every upstream run config and local log."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


def composite_data_version(*, history_manifest_sha256: str, stockbench_manifest_sha256: str) -> str:
    """Name the exact ordered pair of train/valid and final-OOS snapshots."""

    for name, value in (
        ("history", history_manifest_sha256),
        ("stockbench", stockbench_manifest_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} manifest SHA-256 must be 64 lowercase hex characters")
    payload = (
        f"history\t{history_manifest_sha256}\nstockbench\t{stockbench_manifest_sha256}\n"
    ).encode()
    return f"composite-raw-manifest-sha256:{sha256(payload).hexdigest()}"


def validate_composite_data_version(
    *,
    configured: str,
    history_manifest_sha256: str,
    stockbench_manifest_sha256: str,
) -> str:
    """Reject stale run metadata after either frozen source manifest changes."""

    expected = composite_data_version(
        history_manifest_sha256=history_manifest_sha256,
        stockbench_manifest_sha256=stockbench_manifest_sha256,
    )
    if configured != expected:
        raise RuntimeError(
            f"data.version does not match its source manifests: expected {expected}, "
            f"got {configured}"
        )
    return configured


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
