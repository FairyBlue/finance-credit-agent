"""Tests for reusable A3 trajectory step-credit evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.credit_evidence import (
    STEP_CREDIT_EVIDENCE_DIR_ENV,
    STEP_CREDIT_EVIDENCE_SCHEME,
    record_step_credit_evidence,
    summarize_step_credit_evidence,
)
from src.agents.rollout_seed import RolloutIdentity
from src.rewards import STEP_CREDIT_OUTPUT_SEMANTICS


def _record(
    *,
    validate: bool,
    sample_index: int,
    advantage: float,
    credits: list[float],
) -> None:
    assert record_step_credit_evidence(
        identity=RolloutIdentity(
            step=1,
            sample_index=sample_index,
            rollout_n=0,
            validate=validate,
        ),
        base_seed=42,
        mechanism="m1_uniform",
        output_semantics=STEP_CREDIT_OUTPUT_SEMANTICS,
        trajectory_advantage=advantage,
        credits=credits,
        actions=["retrieve", "hold"],
    )


def test_step_credit_evidence_summarizes_training_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STEP_CREDIT_EVIDENCE_DIR_ENV, str(tmp_path))
    _record(validate=False, sample_index=0, advantage=0.9, credits=[0.45, 0.45])
    _record(validate=True, sample_index=1, advantage=-0.4, credits=[-0.2, -0.2])

    evidence = summarize_step_credit_evidence(
        tmp_path,
        expected_mechanism="m1_uniform",
        expected_output_semantics=STEP_CREDIT_OUTPUT_SEMANTICS,
        expected_base_seed=42,
    )

    assert evidence["scheme"] == STEP_CREDIT_EVIDENCE_SCHEME
    assert evidence["trajectory_count"] == 2
    assert evidence["training_trajectory_count"] == 1
    assert evidence["validation_trajectory_count"] == 1
    assert evidence["training_nonzero_trajectory_advantage_count"] == 1
    assert evidence["validation_nonzero_trajectory_advantage_count"] == 1
    assert evidence["completed_trajectory_count"] == 2
    assert evidence["conserved_trajectory_count"] == 2
    assert evidence["maximum_absolute_conservation_error"] == 0.0
    assert evidence["trajectory_advantage_minimum"] == -0.4
    assert evidence["trajectory_advantage_maximum"] == 0.9
    assert evidence["total_step_count"] == 4
    assert len(str(evidence["trajectory_identity_sha256"])) == 64
    assert len(str(evidence["canonical_sha256"])) == 64


def test_step_credit_evidence_rejects_duplicate_trajectory_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STEP_CREDIT_EVIDENCE_DIR_ENV, str(tmp_path))
    _record(validate=False, sample_index=0, advantage=0.2, credits=[0.1, 0.1])
    _record(validate=False, sample_index=0, advantage=0.2, credits=[0.1, 0.1])

    with pytest.raises(RuntimeError, match="duplicate step-credit trajectory identity"):
        summarize_step_credit_evidence(
            tmp_path,
            expected_mechanism="m1_uniform",
            expected_output_semantics=STEP_CREDIT_OUTPUT_SEMANTICS,
            expected_base_seed=42,
        )


def test_m1_step_credit_evidence_rejects_nonuniform_conserved_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STEP_CREDIT_EVIDENCE_DIR_ENV, str(tmp_path))
    _record(validate=False, sample_index=0, advantage=0.9, credits=[0.4, 0.5])

    with pytest.raises(RuntimeError, match="M1 evidence is not uniform"):
        summarize_step_credit_evidence(
            tmp_path,
            expected_mechanism="m1_uniform",
            expected_output_semantics=STEP_CREDIT_OUTPUT_SEMANTICS,
            expected_base_seed=42,
        )


def test_step_credit_evidence_rejects_nonfinite_or_nonconserving_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STEP_CREDIT_EVIDENCE_DIR_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="must be finite"):
        _record(validate=False, sample_index=0, advantage=float("nan"), credits=[0.0, 0.0])
    with pytest.raises(RuntimeError, match="does not conserve"):
        _record(validate=False, sample_index=0, advantage=0.9, credits=[0.4, 0.4])


def test_step_credit_recording_is_optional_outside_launcher_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(STEP_CREDIT_EVIDENCE_DIR_ENV, raising=False)
    assert not record_step_credit_evidence(
        identity=RolloutIdentity(step=0, sample_index=0, rollout_n=0, validate=False),
        base_seed=42,
        mechanism="m1_uniform",
        output_semantics=STEP_CREDIT_OUTPUT_SEMANTICS,
        trajectory_advantage=0.0,
        credits=[0.0],
        actions=["hold"],
    )
