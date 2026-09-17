"""Opt-in completeness markers for the pinned screening trainer checkpoints.

This adds no retry or sampling policy. It rejects incomplete resume inputs before
verl can silently start its dataloader again, and marks only successful saves of
both complete optimizer roles by default. A separately labelled, fresh-only
research profile saves both model roles and extra state without optimizers.
Actor-only BF16 inference exports and two-update critic diagnostic receipts have
separate explicit profiles; neither can restore training or its critic state.
File sizes detect missing/truncated artifacts;
they are not content hashes or a claim of bitwise numerical reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

from .actor_export import (
    ACTOR_EXPORT_PROFILE,
    ACTOR_SAVE_CONTENT,
    DIAGNOSTIC_PROFILE,
    EXPORT_COMPLETION_MARKER,
    EXPORT_PROFILES,
    export_completion_document,
    validate_export_artifact,
)

SCREENING_IDENTITY_ENV = "FINANCE_SCREENING_IDENTITY_SHA256"
COMPLETION_MARKER = "screening_complete.json"
RESEARCH_PROFILE = "research_model_extra_v1"
RESEARCH_COMPLETION_MARKER = "research_artifact_complete.json"
_CONTENTS = frozenset({"model", "optimizer", "extra"})
_RESEARCH_CONTENTS = frozenset({"model", "extra"})


@dataclass(frozen=True, slots=True)
class _FinalCheckpoint:
    """A save made by this trainer immediately before its final evaluation."""

    directory: Path
    identity: str
    step: int
    world_size: int
    profile: str
    actor_worker: Any
    critic_worker: Any
    document: dict[str, Any]
    result: Any


def _identity() -> str:
    value = os.environ.get(SCREENING_IDENTITY_ENV, "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError("screening checkpoint requires FINANCE_SCREENING_IDENTITY_SHA256")
    return value


def _profile(trainer: Any) -> str:
    value = trainer.config.trainer.get("finance_artifact_profile", "full")
    if value not in ("full", RESEARCH_PROFILE, *EXPORT_PROFILES):
        raise RuntimeError("unknown finance_artifact_profile")
    return value


def _configuration(trainer: Any) -> int:
    if not trainer.use_critic:
        raise RuntimeError("screening checkpoint requires both actor and critic")
    config = trainer.config
    profile = _profile(trainer)
    if profile in {RESEARCH_PROFILE, *EXPORT_PROFILES} and (
        config.trainer.get("resume_mode", "disable") != "disable"
        or config.trainer.get("resume_from_path") is not None
    ):
        raise RuntimeError("research artifacts are fresh-only; resume is unsupported")
    if profile == DIAGNOSTIC_PROFILE and (
        not config.trainer.get("val_only", False) and trainer.total_training_steps != 2
    ):
        raise RuntimeError("critic diagnostic profile requires exactly two training updates")
    for role, checkpoint in (
        ("actor", config.actor_rollout_ref.actor.checkpoint),
        ("critic", config.critic.checkpoint),
    ):
        if checkpoint.get("async_save", False) is not False:
            raise RuntimeError("screening checkpoints forbid async_save")
        for direction in ("save_contents", "load_contents"):
            contents = checkpoint.get(direction, [])
            if profile in EXPORT_PROFILES:
                expected = (
                    [ACTOR_SAVE_CONTENT]
                    if profile == ACTOR_EXPORT_PROFILE and role == "actor"
                    and direction == "save_contents" else []
                )
                if list(contents) != expected:
                    raise RuntimeError(
                        f"inference/diagnostic {role} {direction} must be {expected}"
                    )
            elif profile == RESEARCH_PROFILE:
                if set(contents) != _RESEARCH_CONTENTS or len(contents) != 2:
                    raise RuntimeError(
                        f"research {role} {direction} must be exactly model/extra without optimizer"
                    )
            elif not _CONTENTS.issubset(set(contents)):
                raise RuntimeError(
                    f"screening {role} {direction} must include model/optimizer/extra"
                )
    if config.trainer.get("default_hdfs_dir") is not None:
        raise RuntimeError("screening checkpoints currently require local storage")
    dimensions = (config.trainer.n_gpus_per_node, config.trainer.nnodes)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise RuntimeError("screening checkpoint world size must be positive")
    world_size = dimensions[0] * dimensions[1]
    if profile in EXPORT_PROFILES and world_size != 1:
        raise RuntimeError("actor/diagnostic exports require world size one")
    return world_size


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


def _research_required_files(world_size: int) -> set[str]:
    return {"data.pt"} | {
        f"{role}/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for role in ("actor", "critic")
        for kind in ("model", "extra_state")
        for rank in range(world_size)
    } | {
        f"{role}/{name}"
        for role in ("actor", "critic")
        for name in (
            "fsdp_config.json", "huggingface/config.json", "huggingface/tokenizer.json",
            "huggingface/tokenizer_config.json", "huggingface/chat_template.jinja",
        )
    }


def _reject_full_artifacts(directory: Path) -> None:
    if (directory / COMPLETION_MARKER).exists() or any(
        directory.glob("*/optim_world_size_*_rank_*.pt")
    ):
        raise RuntimeError("research artifact directory must not contain full checkpoint artifacts")


def _metadata_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _research_evidence(directory: Path, world_size: int) -> tuple[dict[str, int], dict[str, str]]:
    _reject_full_artifacts(directory)
    required = _research_required_files(world_size)
    metadata = {
        path.relative_to(directory).as_posix()
        for role in ("actor", "critic")
        for path in (directory / role / "huggingface").rglob("*")
        if path.is_file()
    } | {f"{role}/fsdp_config.json" for role in ("actor", "critic")}
    sizes = {}
    for relative in sorted(required | metadata):
        path = directory / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"incomplete research artifact: missing or empty {relative}")
        sizes[relative] = path.stat().st_size
    for role in ("actor", "critic"):
        try:
            fsdp = json.loads((directory / role / "fsdp_config.json").read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError("invalid research FSDP metadata") from error
        if (
            not isinstance(fsdp, dict) or fsdp.get("world_size") != world_size
            or type(fsdp.get("FSDP_version")) is not int or fsdp["FSDP_version"] not in (1, 2)
        ):
            raise RuntimeError("research FSDP metadata world size/version mismatch")
    # Only small HF/FSDP metadata is hashed; model-sized tensor integrity remains
    # the same missing/truncated-file check as the default full profile.
    hashes = {
        relative: _metadata_hash(directory / relative)
        for relative in sorted(metadata)
    }
    return sizes, hashes


def validate_research_artifact(
    directory: str | Path, *, identity_sha256: str, world_size: int = 1,
) -> dict[str, Any]:
    """Validate inference/research evidence, never grant optimizer resume support."""
    if type(world_size) is not int or world_size < 1:
        raise RuntimeError("research artifact world size must be positive")
    directory = Path(directory).expanduser().resolve()
    path = directory / RESEARCH_COMPLETION_MARKER
    if not path.is_file():
        raise RuntimeError("research artifact completion marker is missing")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("invalid research artifact completion marker") from error
    if (
        not isinstance(document, dict) or document.get("schema_version") != 1
        or document.get("artifact_profile") != RESEARCH_PROFILE
        or document.get("optimizer_included") is not False
        or document.get("resume_supported") is not False
    ):
        raise RuntimeError("invalid research artifact completion marker schema")
    if document.get("identity_sha256") != identity_sha256:
        raise RuntimeError("research artifact execution identity mismatch")
    if document.get("global_step") != _checkpoint_step(directory):
        raise RuntimeError("research artifact marker global_step mismatch")
    if document.get("world_size") != world_size:
        raise RuntimeError("research artifact world size mismatch")
    sizes, hashes = _research_evidence(directory, world_size)
    if document.get("files") != sizes or document.get("metadata_sha256") != hashes:
        raise RuntimeError("research artifact files changed after completion")
    return document


def _validator(profile: str):
    if profile in EXPORT_PROFILES:
        # The worker already read back every BF16 byte before issuing its receipt.
        # Repeated driver checks use headers and metadata; post-run validation
        # uses validate_export_artifact's default full sequential hash readback.
        def validate(directory: Path, *, identity_sha256: str, world_size: int):
            return validate_export_artifact(
                directory, identity_sha256=identity_sha256,
                world_size=world_size, verify_hashes=False,
            )

        return validate
    return validate_research_artifact if profile == RESEARCH_PROFILE else validate_checkpoint


def _write_marker(
    directory: Path, document: dict[str, Any], *, marker_name: str = COMPLETION_MARKER,
) -> None:
    temporary = directory / f".{marker_name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(directory / marker_name)
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
        profile = _profile(trainer)
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
            and (saved.directory, saved.identity, saved.step, saved.world_size, saved.profile)
            == (directory, identity, step, world_size, profile)
            and saved.actor_worker is trainer.actor_rollout_wg
            and saved.critic_worker is trainer.critic_wg
        ):
            # The pinned fit performs no update between final evaluation and
            # its subsequent save. Keep the pre-evaluation RNG/data boundary.
            document = _validator(profile)(
                directory, identity_sha256=identity, world_size=world_size
            )
            if document != saved.document:
                raise RuntimeError("screening final checkpoint marker changed after saving")
            return saved.result
        trainer._finance_screening_final_checkpoint = None
        if profile in EXPORT_PROFILES:
            if step != trainer.total_training_steps or step <= 0:
                raise RuntimeError("actor/diagnostic artifacts are final-update only")
            # Never prune or rewrite earlier artifacts, including a failed save.
            if directory.exists():
                raise RuntimeError("actor/diagnostic export refuses an existing step directory")
            directory.mkdir(parents=True)
            if profile == ACTOR_EXPORT_PROFILE:
                result = trainer.actor_rollout_wg.save_checkpoint(
                    str(directory / "actor"), None, step, max_ckpt_to_keep=None,
                )
            else:
                result = None
            document = export_completion_document(
                directory, profile=profile, identity=identity, step=step,
            )
            _write_marker(directory, document, marker_name=EXPORT_COMPLETION_MARKER)
            _validator(profile)(directory, identity_sha256=identity, world_size=world_size)
            return result
        # A failed rewrite of this step must not retain an earlier success mark.
        marker_name = (
            RESEARCH_COMPLETION_MARKER if profile == RESEARCH_PROFILE else COMPLETION_MARKER
        )
        (directory / marker_name).unlink(missing_ok=True)
        if profile == RESEARCH_PROFILE:
            _reject_full_artifacts(directory)
        result = original_save(trainer, *args, **kwargs)
        if profile == RESEARCH_PROFILE:
            sizes, hashes = _research_evidence(directory, world_size)
            _write_marker(directory, {
                "schema_version": 1, "artifact_profile": profile,
                "identity_sha256": identity, "global_step": step, "world_size": world_size,
                "optimizer_included": False, "resume_supported": False,
                "files": sizes, "metadata_sha256": hashes,
            }, marker_name=marker_name)
            return result
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
        profile = _profile(trainer)
        if profile in {RESEARCH_PROFILE, *EXPORT_PROFILES}:
            _configuration(trainer)
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
            document = _validator(profile)(
                directory, identity_sha256=identity, world_size=world_size
            )
            trainer._finance_screening_final_checkpoint = _FinalCheckpoint(
                directory=directory,
                identity=identity,
                step=step,
                world_size=world_size,
                profile=profile,
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
