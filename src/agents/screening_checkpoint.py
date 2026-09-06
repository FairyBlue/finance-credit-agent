"""Opt-in completeness markers for the pinned screening trainer checkpoints.

This adds no retry or sampling policy. It rejects incomplete resume inputs before
verl can silently start its dataloader again, and marks only successful saves of
both complete optimizer roles. File sizes detect missing/truncated artifacts;
they are not content hashes or a claim of bitwise numerical reproducibility.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

SCREENING_IDENTITY_ENV = "FINANCE_SCREENING_IDENTITY_SHA256"
COMPLETION_MARKER = "screening_complete.json"
_CONTENTS = frozenset({"model", "optimizer", "extra"})


@dataclass(frozen=True, slots=True)
class _FinalCheckpoint:
    """A save made by this trainer immediately before its final evaluation."""

    directory: Path
    identity: str
    step: int
    world_size: int
    actor_worker: Any
    critic_worker: Any
    document: dict[str, Any]
    result: Any


def _identity() -> str:
    value = os.environ.get(SCREENING_IDENTITY_ENV, "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError("screening checkpoint requires FINANCE_SCREENING_IDENTITY_SHA256")
    return value


def _configuration(trainer: Any) -> int:
    if not trainer.use_critic:
        raise RuntimeError("screening checkpoint requires both actor and critic")
    config = trainer.config
    for role, checkpoint in (
        ("actor", config.actor_rollout_ref.actor.checkpoint),
        ("critic", config.critic.checkpoint),
    ):
        if checkpoint.get("async_save", False) is not False:
            raise RuntimeError("screening checkpoints forbid async_save")
        for direction in ("save_contents", "load_contents"):
            if not _CONTENTS.issubset(set(checkpoint.get(direction, []))):
                raise RuntimeError(
                    f"screening {role} {direction} must include model/optimizer/extra"
                )
    if config.trainer.get("default_hdfs_dir") is not None:
        raise RuntimeError("screening checkpoints currently require local storage")
    dimensions = (config.trainer.n_gpus_per_node, config.trainer.nnodes)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise RuntimeError("screening checkpoint world size must be positive")
    return dimensions[0] * dimensions[1]


def _required_files(world_size: int) -> set[str]:
    return {"data.pt"} | {
        f"{role}/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for role in ("actor", "critic")
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }


def _file_sizes(directory: Path, world_size: int) -> dict[str, int]:
    sizes = {}
    for relative in sorted(_required_files(world_size)):
        path = directory / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"incomplete screening checkpoint: missing or empty {relative}")
        sizes[relative] = path.stat().st_size
    return sizes


def _checkpoint_step(directory: Path) -> int:
    match = re.fullmatch(r"global_step_([0-9]+)", directory.name)
    if match is None:
        raise RuntimeError("screening resume path must name a complete global_step_N directory")
    return int(match[1])


def validate_checkpoint(
    directory: str | Path,
    *,
    identity_sha256: str,
    world_size: int = 1,
) -> dict[str, Any]:
    """Check the marker and all required files without loading model-sized tensors."""
    directory = Path(directory).expanduser().resolve()
    marker = directory / COMPLETION_MARKER
    if not marker.is_file():
        raise RuntimeError("screening checkpoint completion marker is missing")
    try:
        document = json.loads(marker.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise RuntimeError("invalid screening checkpoint completion marker") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError("invalid screening checkpoint completion marker schema")
    if document.get("identity_sha256") != identity_sha256:
        raise RuntimeError("screening checkpoint execution identity mismatch")
    if document.get("global_step") != _checkpoint_step(directory):
        raise RuntimeError("screening checkpoint marker global_step mismatch")
    if document.get("world_size") != world_size:
        raise RuntimeError("screening checkpoint world size mismatch")
    sizes = _file_sizes(directory, world_size)
    if document.get("files") != sizes:
        raise RuntimeError("screening checkpoint files changed after completion")
    return document


def _write_marker(directory: Path, document: dict[str, Any]) -> None:
    temporary = directory / f".{COMPLETION_MARKER}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(directory / COMPLETION_MARKER)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def install_checkpoint_hooks(trainer_class: type[Any]) -> bool:
    """Wrap only the screening Agent-R1 subclass, leaving verl's base unchanged."""
    if trainer_class.__dict__.get("_finance_screening_checkpoint_hooks", False):
        return False
    original_save = trainer_class._save_checkpoint
    original_load = trainer_class._load_checkpoint
    original_validate = trainer_class._validate

    @wraps(original_save)
    def save_checkpoint(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        identity = _identity()
        world_size = _configuration(trainer)
        step = trainer.global_steps
        if type(step) is not int or step < 0:
            raise RuntimeError("invalid screening checkpoint global step")
        directory = (
            Path(trainer.config.trainer.default_local_dir).expanduser().resolve()
            / f"global_step_{step}"
        )
        saved = getattr(trainer, "_finance_screening_final_checkpoint", None)
        if (
            isinstance(saved, _FinalCheckpoint)
            and (saved.directory, saved.identity, saved.step, saved.world_size)
            == (directory, identity, step, world_size)
            and saved.actor_worker is trainer.actor_rollout_wg
            and saved.critic_worker is trainer.critic_wg
        ):
            # The pinned fit performs no update between final evaluation and
            # its subsequent save. Keep the pre-evaluation RNG/data boundary.
            document = validate_checkpoint(
                directory, identity_sha256=identity, world_size=world_size
            )
            if document != saved.document:
                raise RuntimeError("screening final checkpoint marker changed after saving")
            return saved.result
        trainer._finance_screening_final_checkpoint = None
        # A failed rewrite of this step must not retain an earlier success mark.
        (directory / COMPLETION_MARKER).unlink(missing_ok=True)
        result = original_save(trainer, *args, **kwargs)
        sizes = _file_sizes(directory, world_size)
        _write_marker(
            directory,
            {
                "schema_version": 1,
                "identity_sha256": identity,
                "global_step": step,
                "world_size": world_size,
                "files": sizes,
            },
        )
        return result

    @wraps(original_load)
    def load_checkpoint(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        trainer._finance_screening_final_checkpoint = None
        identity = _identity()
        world_size = _configuration(trainer)
        mode = trainer.config.trainer.get("resume_mode", "disable")
        if mode == "disable":
            return original_load(trainer, *args, **kwargs)
        if mode != "resume_path":
            raise RuntimeError(
                "screening resume requires an explicit resume_path; auto is forbidden"
            )
        path = trainer.config.trainer.get("resume_from_path")
        if not isinstance(path, str) or not path:
            raise RuntimeError("screening resume_path is missing")
        validate_checkpoint(path, identity_sha256=identity, world_size=world_size)
        return original_load(trainer, *args, **kwargs)

    @wraps(original_validate)
    def validate(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        step = trainer.global_steps
        if (
            type(step) is int
            and step > 0
            and step == trainer.total_training_steps
            and not trainer.config.trainer.get("val_only", False)
        ):
            # Upstream evaluates before saving. Preserve both updated model
            # roles before a strict final rollout can fail or be interrupted.
            # This is a diagnostic/recovery artifact, not proof of exact resume.
            identity = _identity()
            world_size = _configuration(trainer)
            directory = (
                Path(trainer.config.trainer.default_local_dir).expanduser().resolve()
                / f"global_step_{step}"
            )
            result = trainer._save_checkpoint()
            document = validate_checkpoint(
                directory, identity_sha256=identity, world_size=world_size
            )
            trainer._finance_screening_final_checkpoint = _FinalCheckpoint(
                directory=directory,
                identity=identity,
                step=step,
                world_size=world_size,
                actor_worker=trainer.actor_rollout_wg,
                critic_worker=trainer.critic_wg,
                document=document,
                result=result,
            )
        return original_validate(trainer, *args, **kwargs)

    trainer_class._save_checkpoint = save_checkpoint
    trainer_class._load_checkpoint = load_checkpoint
    trainer_class._validate = validate
    trainer_class._finance_screening_checkpoint_hooks = True
    return True
