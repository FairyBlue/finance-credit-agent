"""Durable local run records for reproducible experiments and failed preflights."""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

import yaml

from .metadata import RunHashes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_value(repo: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _gpu_inventory() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": type(error).__name__}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": completed.stderr.strip() or f"exit_{completed.returncode}",
        }
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"available": bool(devices), "devices": devices}


@dataclass(slots=True)
class LocalRunRecorder:
    """Own one append-only local run directory and its terminal summary."""

    path: Path
    events: TextIO
    metrics: TextIO
    started_at_utc: str
    started_monotonic: float
    last_completed_stage: str

    @classmethod
    def create(
        cls,
        *,
        run_root: str | Path,
        config: dict[str, Any],
        command: list[str],
        hashes: RunHashes,
        run_id: str | None = None,
        repo_root: str | Path | None = None,
    ) -> LocalRunRecorder:
        root = Path(run_root).expanduser().resolve()
        repo = Path(repo_root or Path.cwd()).expanduser().resolve()
        name = str(config.get("run", {}).get("name", "run")).replace("/", "-")
        identifier = run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-{name}-{hashes.repo_commit[:8]}"
        )
        path = root / identifier
        path.mkdir(parents=True, exist_ok=False)
        (path / "checkpoints").mkdir()
        (path / "artifacts").mkdir()
        (path / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        config_path = path / "resolved_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        config_sha256 = sha256(config_path.read_bytes()).hexdigest()
        diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
            check=False,
            capture_output=True,
        ).stdout
        _write_json(
            path / "git_state.json",
            {
                **hashes.as_config(),
                "branch": _git_value(repo, "branch", "--show-current"),
                "head": _git_value(repo, "rev-parse", "HEAD"),
                "origin_main": _git_value(repo, "rev-parse", "origin/main"),
                "diff_sha256": sha256(diff).hexdigest(),
                "working_tree": "dirty" if hashes.repo_dirty else "clean",
                "recorded_at_utc": _utc_now(),
            },
        )
        _write_json(
            path / "environment.json",
            {
                "captured_at_utc": _utc_now(),
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "packages": {
                    name: _package_version(name) for name in ("torch", "verl", "vllm", "flash-attn")
                },
                "gpu": _gpu_inventory(),
            },
        )
        data = config.get("data", {})
        split_source = data.get("split_source", {}) if isinstance(data, dict) else {}
        active_sources = (
            sorted({str(value) for value in split_source.values()})
            if isinstance(split_source, dict)
            else []
        )
        sources = data.get("sources", {}) if isinstance(data, dict) else {}
        data_document = {
            "data_version": data.get("version") if isinstance(data, dict) else None,
            "episode_manifest": (data.get("episode_manifest") if isinstance(data, dict) else None),
            "processed": data.get("processed") if isinstance(data, dict) else None,
            "split_source": split_source,
            "active_sources": {
                source: sources.get(source)
                for source in active_sources
                if isinstance(sources, dict)
            },
        }
        data_path = path / "data_manifest.json"
        _write_json(data_path, data_document)
        data_manifest_sha256 = sha256(data_path.read_bytes()).hexdigest()
        events = (path / "events.log").open("a", encoding="utf-8")
        metrics = (path / "metrics.jsonl").open("a", encoding="utf-8")
        started = _utc_now()
        recorder = cls(
            path=path,
            events=events,
            metrics=metrics,
            started_at_utc=started,
            started_monotonic=time.monotonic(),
            last_completed_stage="record_created",
        )
        recorder.append_event("run_started", run_id=identifier)
        model = config.get("model", {})
        training = config.get("training", {})
        _write_json(
            path / "summary.json",
            {
                "status": "RUNNING",
                "started_at_utc": started,
                "run_id": identifier,
                "resolved_config_sha256": config_sha256,
                "data_manifest_sha256": data_manifest_sha256,
                "model_name": (
                    model.get("name") or model.get("train") if isinstance(model, dict) else None
                ),
                "model_revision": (model.get("revision") if isinstance(model, dict) else None),
                "seed": config.get("run", {}).get("seed"),
                "episodes": (training.get("episodes") if isinstance(training, dict) else None),
                "rollout_n": (training.get("rollout_n") if isinstance(training, dict) else None),
            },
        )
        return recorder

    def append_event(self, event: str, **fields: object) -> None:
        self.events.write(
            json.dumps(
                {"at_utc": _utc_now(), "event": event, **fields},
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        self.events.flush()
        if event != "subprocess_output":
            self.last_completed_stage = event

    def record_command(self, command: list[str]) -> None:
        """Replace the provisional command after preflight resolves immutable inputs."""

        rendered = shlex.join(command) + "\n"
        (self.path / "command.txt").write_text(rendered, encoding="utf-8")
        self.append_event(
            "command_resolved",
            command_sha256=sha256(rendered.encode()).hexdigest(),
        )

    def append_metric(self, step: int, **metrics: float | int | bool) -> None:
        self.metrics.write(
            json.dumps({"at_utc": _utc_now(), "step": step, **metrics}, sort_keys=True) + "\n"
        )
        self.metrics.flush()

    def finish(
        self,
        *,
        returncode: int | None,
        error: BaseException | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        status = "PASSED" if returncode == 0 and error is None else "FAILED"
        optimizer = extra.get("optimizer_update_evidence") if extra else None
        optimizer = optimizer if isinstance(optimizer, dict) else {}
        summary_path = self.path / "summary.json"
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        summary: dict[str, object] = {
            **existing,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _utc_now(),
            "returncode": returncode,
            "exit_code": returncode if returncode is not None else 1,
            "wall_clock_seconds": time.monotonic() - self.started_monotonic,
            "last_completed_stage": self.last_completed_stage,
            "global_step": int(optimizer.get("global_update_count", 0)),
            "optimizer_step_count": int(optimizer.get("optimizer_step_count", 0)),
            "backward_count": int(optimizer.get("backward_count", 0)),
            "peak_gpu_memory_bytes": int(optimizer.get("peak_gpu_memory_bytes", 0)),
        }
        if error is not None:
            summary["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        if extra:
            summary.update(extra)
        self.append_event("run_finished", status=status, returncode=returncode)
        _write_json(summary_path, summary)
        self.events.close()
        self.metrics.close()
