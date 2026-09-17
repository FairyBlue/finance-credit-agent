"""Opt-in, same-process M2 decision after two complete optimizer rounds.

Install after the existing screening checkpoint/resource hooks. A successful
step-2 save is the boundary: keep training to four rounds when the first two
batches contain effective M2 residual, otherwise evaluate that saved policy and
return from fit. No optimizer restore, extra generation, or reward changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from .screening_checkpoint import validate_checkpoint

ENABLE_ENV = "FINANCE_LEARNING_RESTART_GATE"
ENABLE_VALUE = "stop2_if_no_signal_v1"
DIRECTORY_ENV = "FINANCE_LEARNING_GATE_DIR"
_MARKER = "_finance_learning_restart_gate_settings"


class _StopAfterGate(BaseException):
    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer


@dataclass(frozen=True)
class _Settings:
    diagnostics: Path
    directory: Path
    seed: int
    identity: str


def _settings() -> _Settings | None:
    enabled = os.environ.get(
        ENABLE_ENV, ENABLE_VALUE if os.environ.get(DIRECTORY_ENV) else None
    )
    if enabled is None:
        return None
    if enabled != ENABLE_VALUE or os.environ.get("FINANCE_SCREENING_ARM") != "M2":
        raise RuntimeError("learning restart gate requires its explicit M2-only setting")
    diagnostics = Path(os.environ.get("FINANCE_SCREENING_DIAGNOSTICS_DIR", ""))
    directory = Path(
        os.environ.get(DIRECTORY_ENV, str(diagnostics.parent / "learning_restart_gate"))
    )
    if not diagnostics.is_absolute() or not directory.is_absolute():
        raise RuntimeError("learning restart gate directories must be absolute")
    seed_text = os.environ.get("FINANCE_SCREENING_RUN_SEED", "")
    identity = os.environ.get("FINANCE_SCREENING_IDENTITY_SHA256", "")
    if not seed_text.isdecimal() or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise RuntimeError("learning restart gate requires the run seed and execution identity")
    return _Settings(diagnostics, directory, int(seed_text), identity)


def _write(directory: Path, name: str, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive files make a repeated run or interrupted decision visible.
    with (directory / name).open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _evidence(settings: _Settings) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    sources = []
    for path in sorted(settings.diagnostics.glob("advantage-*.jsonl")):
        content = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
        for line in content.decode("utf-8").splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError("M2 gate advantage evidence must be an object")
            records.append(row)
    if len(records) != 2 or sorted(row.get("policy_step", -1) for row in records) != [1, 2]:
        raise RuntimeError("M2 gate requires exactly policy steps 1 and 2")
    records.sort(key=lambda row: row["policy_step"])
    for row in records:
        if (
            row.get("schema_version") != 1
            or row.get("phase") != "advantage_pre_update"
            or row.get("arm") != "M2"
            or row.get("run_seed") != settings.seed
            or row.get("trajectory_count") != 8
            or row.get("episode_group_count") != 2
            or row.get("advancing_row_count") != 160
            or row.get("whitening_scope") != "logical_rollout_batch_real_step_rows"
        ):
            raise RuntimeError("M2 gate evidence does not match two complete n4 batches")
        scope = row.get("advancing_steps")
        if not isinstance(scope, dict) or scope.get("row_count") != 160:
            raise RuntimeError("M2 gate is missing advancing residual diagnostics")
        count = scope.get("nonzero_S_count")
        exact_count = scope.get("exact_nonzero_S_count")
        mean = scope.get("mean_abs_S")
        if (
            type(count) is not int
            or type(exact_count) is not int
            or not 0 <= count <= exact_count <= 160
            or isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(mean)
            or mean < 0
            or (count > 0 and mean == 0)
        ):
            raise RuntimeError("M2 gate residual diagnostics are invalid")
    return records, sources


def patch_trainer(trainer_class: type[Any]) -> bool:
    """Enable with DIRECTORY_ENV or ENABLE_ENV/ENABLE_VALUE, and arm M2.

    Receipts go to DIRECTORY_ENV, defaulting to a sibling of the diagnostics
    directory named learning_restart_gate. decision.json selects completed or
    planned rounds via target_rounds=2/4; the outer launcher validates actual
    optimizer, trajectory and evaluation evidence independently.
    """
    settings = _settings()
    if settings is None:
        return False
    existing = trainer_class.__dict__.get(_MARKER)
    if existing is not None:
        if existing != settings:
            raise RuntimeError("cannot change the installed M2 gate settings")
        return False
    if not trainer_class.__dict__.get("_finance_screening_checkpoint_hooks", False):
        raise RuntimeError("install the screening checkpoint hooks before the M2 gate")
    original_fit = trainer_class.fit
    original_save = trainer_class._save_checkpoint

    @wraps(original_save)
    def save_checkpoint(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_save(trainer, *args, **kwargs)
        if trainer.global_steps != 2 or getattr(trainer, "_finance_restart_gate_decided", False):
            return result
        checkpoint = Path(trainer.config.trainer.default_local_dir) / "global_step_2"
        marker = validate_checkpoint(checkpoint, identity_sha256=settings.identity)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "arm": "M2",
            "run_seed": settings.seed,
            "identity_sha256": settings.identity,
            "pid": os.getpid(),
            "checkpoint_step": 2,
            "checkpoint_path": str(checkpoint),
            "post_update_checkpoint_saved": True,
            "checkpoint_marker": marker,
            "updated_at_sgt": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        }
        _write(settings.directory, "checkpoint_step2.json", receipt)
        records, sources = _evidence(settings)
        nonzero = sum(row["advancing_steps"]["nonzero_S_count"] for row in records)
        receipt.update(
            diagnostics=records,
            diagnostic_sources=sources,
            effective_nonzero_S_count=nonzero,
            target_rounds=4 if nonzero else 2,
        )
        if nonzero:
            receipt.update(decision="CONTINUE_TO_4", final_evaluation_completed=False)
            _write(settings.directory, "decision.json", receipt)
            trainer._finance_restart_gate_decided = True
            return result

        # global_steps=2, total_training_steps=4: the existing final-evaluation
        # wrapper does not save again. The successful checkpoint above precedes
        # this evaluation, using the still-resident actor and critic workers.
        trainer._validate()
        finalize = getattr(trainer.actor_rollout_wg, "async_calls_finalize_fn_exec", None)
        if callable(finalize):
            finalize(blocking=True)
        receipt.update(
            decision="STOP_AFTER_2_NO_EFFECTIVE_M2_RESIDUAL",
            final_evaluation_completed=True,
            async_calls_finalized=callable(finalize),
        )
        _write(settings.directory, "decision.json", receipt)
        trainer._finance_restart_gate_decided = True
        raise _StopAfterGate(trainer)

    @wraps(original_fit)
    def fit(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        config = trainer.config.trainer
        if (
            trainer.total_training_steps != 4
            or config.get("save_freq") != 2
            or config.get("test_freq") != 4
            or config.get("resume_mode", "disable") != "disable"
            or config.get("val_only", False)
        ):
            raise RuntimeError("M2 gate requires fresh four-round training, save=2 and test=4")
        if (settings.directory / "checkpoint_step2.json").exists():
            raise RuntimeError("M2 gate output already exists; no implicit retry or resume")
        try:
            return original_fit(trainer, *args, **kwargs)
        except _StopAfterGate as stopped:
            if stopped.trainer is not trainer:
                raise
            return None

    trainer_class._save_checkpoint = save_checkpoint
    trainer_class.fit = fit
    setattr(trainer_class, _MARKER, settings)
    return True
