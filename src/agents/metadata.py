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
    # Git uses leading status bytes (notably the clean submodule marker). Keep
    # them while normalizing only the line ending written by the CLI.
    return process.stdout.rstrip("\r\n")


@dataclass(frozen=True, slots=True)
class RunHashes:
    agent_r1_commit: str
    repo_commit: str
    data_version: str
    repo_dirty: bool

    def as_config(self) -> dict[str, str | bool]:
        return asdict(self)


def observe_run_hashes(
    *,
    repo_root: Path,
    agent_r1_path: Path,
    data_version: str,
) -> RunHashes:
    """Capture best-effort identities before validation so failures remain recordable."""

    try:
        repo_commit = _git(repo_root, "rev-parse", "HEAD")
    except RuntimeError:
        repo_commit = "UNCOMMITTED"
    try:
        agent_commit = _git(agent_r1_path, "rev-parse", "HEAD")
    except RuntimeError:
        agent_commit = "UNCOMMITTED"
    try:
        dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=normal"))
    except RuntimeError:
        dirty = True
    return RunHashes(
        agent_r1_commit=agent_commit,
        repo_commit=repo_commit,
        data_version=data_version,
        repo_dirty=dirty,
    )


def collect_run_hashes(
    *,
    repo_root: Path,
    agent_r1_path: Path,
    data_version: str,
    expected_agent_r1_commit: str,
    require_clean_repo: bool,
) -> RunHashes:
    hashes = observe_run_hashes(
        repo_root=repo_root,
        agent_r1_path=agent_r1_path,
        data_version=data_version,
    )
    if hashes.agent_r1_commit != expected_agent_r1_commit:
        raise RuntimeError(
            "Agent-R1 checkout mismatch: "
            f"expected {expected_agent_r1_commit}, got {hashes.agent_r1_commit}"
        )
    if require_clean_repo and (hashes.repo_dirty or hashes.repo_commit == "UNCOMMITTED"):
        raise RuntimeError(
            "training requires a committed, clean repository; commit the scaffold first"
        )
    if not data_version.strip() or "TODO" in data_version.upper():
        raise RuntimeError("data.version must be an immutable, non-TODO identifier")
    return hashes


_LOCAL_ARTIFACT_ROOTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "checkpoints",
    "data",
    "graft",
    "htmlcov",
    "logs",
    "outputs",
    "runs",
    "wandb",
}
_SUBMODULE_ARTIFACT_ROOTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "checkpoints",
    "dataset",
    "logs",
    "outputs",
    "wandb",
}
_INERT_EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "requires.txt",
    "top_level.txt",
}


def _nul_paths(value: str) -> set[str]:
    return {path for path in value.split("\0") if path}


def _is_allowed_artifact(path: str, *, roots: set[str]) -> bool:
    candidate = Path(path)
    if not candidate.parts:
        return False
    if candidate.parts[0] in roots or "__pycache__" in candidate.parts:
        return True
    # Editable/build installs may leave setuptools metadata next to the source
    # tree.  These files are inert for Python import; keep executable plugin
    # metadata (for example entry_points.txt) and arbitrary files fail-closed.
    if candidate.name in _INERT_EGG_INFO_FILES and any(
        part.endswith(".egg-info") for part in candidate.parts[:-1]
    ):
        return True
    return candidate.name == ".DS_Store" or (
        len(candidate.parts) == 1 and candidate.name.startswith(".coverage")
    ) or (
        len(candidate.parts) == 1
        and (candidate.name == ".env" or candidate.name.startswith(".env."))
    )


def _untracked_or_ignored_paths(repo: Path, *, excluded_roots: set[str]) -> set[str]:
    candidates = _nul_paths(
        _git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", ".")
    )
    exclusions = tuple(
        f":(exclude,glob){root}/**" for root in sorted(excluded_roots)
    ) + (
        ":(exclude,glob)__pycache__/**",
        ":(exclude,glob)**/__pycache__/**",
    )
    candidates.update(
        _nul_paths(
            _git(
                repo,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
                "--",
                ".",
                *exclusions,
            )
        )
    )
    return candidates


def verify_validation_identity(
    *,
    repo_root: Path,
    agent_r1_path: Path,
    expected_repo_commit: str,
    expected_agent_r1_commit: str,
    allowed_untracked_paths: tuple[Path, ...] = (),
    required_tracked_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Bind a validation to an external commit and reject local execution overlays."""

    if len(expected_repo_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_repo_commit
    ):
        raise RuntimeError("--expected-commit must be a full lowercase 40-character commit")
    actual_repo_commit = _git(repo_root, "rev-parse", "HEAD")
    if actual_repo_commit != expected_repo_commit:
        raise RuntimeError(
            f"validation commit mismatch: expected {expected_repo_commit}, got {actual_repo_commit}"
        )
    tracked_status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    )
    if tracked_status:
        raise RuntimeError("validation requires a clean tracked working tree")

    tracked_config_sources: list[str] = []
    for path in required_tracked_paths:
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"validation config source is outside the repository: {path}"
            ) from error
        relative_text = relative.as_posix()
        _git(repo_root, "ls-files", "--error-unmatch", "--", relative_text)
        tracked_config_sources.append(relative_text)

    actual_agent_commit = _git(agent_r1_path, "rev-parse", "HEAD")
    if actual_agent_commit != expected_agent_r1_commit:
        raise RuntimeError(
            f"Agent-R1 checkout mismatch: expected {expected_agent_r1_commit}, "
            f"got {actual_agent_commit}"
        )
    relative_agent = agent_r1_path.resolve().relative_to(repo_root.resolve())
    submodule_status = _git(repo_root, "submodule", "status", "--", str(relative_agent))
    expected_status = f" {expected_agent_r1_commit}"
    if not submodule_status.startswith(expected_status):
        raise RuntimeError("Agent-R1 submodule is not at the clean pinned commit")
    agent_tracked_status = _git(
        agent_r1_path,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if agent_tracked_status:
        raise RuntimeError("Agent-R1 submodule working tree has tracked modifications")

    candidates = _untracked_or_ignored_paths(
        repo_root,
        excluded_roots=_LOCAL_ARTIFACT_ROOTS,
    )
    allowed = {
        path.resolve().relative_to(repo_root.resolve()).as_posix()
        for path in allowed_untracked_paths
    }
    disallowed = sorted(
        path
        for path in candidates
        if path not in allowed and not _is_allowed_artifact(path, roots=_LOCAL_ARTIFACT_ROOTS)
    )
    if disallowed:
        raise RuntimeError(
            "validation found untracked or ignored repository overlays: "
            + ", ".join(disallowed)
        )

    agent_candidates = _untracked_or_ignored_paths(
        agent_r1_path,
        excluded_roots=_SUBMODULE_ARTIFACT_ROOTS,
    )
    agent_disallowed = sorted(
        path
        for path in agent_candidates
        if not _is_allowed_artifact(path, roots=_SUBMODULE_ARTIFACT_ROOTS)
    )
    if agent_disallowed:
        raise RuntimeError(
            "validation found untracked or ignored Agent-R1 overlays: "
            + ", ".join(agent_disallowed)
        )
    return {
        "expected_repo_commit": expected_repo_commit,
        "repo_commit": actual_repo_commit,
        "tracked_working_tree": "clean",
        "agent_r1_commit": actual_agent_commit,
        "submodule_status": submodule_status,
        "tracked_config_sources": tracked_config_sources,
        "experiment_overlays": [],
    }
