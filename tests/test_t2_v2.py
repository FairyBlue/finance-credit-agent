"""Test-first contracts for the history-only pooled T2 v2 probe."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.agents.config import load_experiment_config
from src.data import MarketObservation, StockBenchAdapter
from src.evaluation.leakage_probe_v2 import (
    FROZEN_T2_V2_FEATURE_NAMES,
    BalancedAccuracyEvidence,
    PooledDirectionSplit,
    RidgeDirectionModel,
    T2V2Dataset,
    T2V2ProbeResult,
    balanced_accuracy,
    build_history_t2_v2_dataset,
    deterministic_weak_control_flip_mask,
    fit_ridge_direction_model,
    moving_block_session_draws,
    run_t2_v2_probe,
    session_clustered_balanced_accuracy,
)
from src.evaluation.t2_v2 import (
    ConfiguredT2V2Run,
    t2_v2_artifact_bytes,
    t2_v2_protocol_sha256,
    validate_t2_v2_config,
    verify_t2_v2_artifact,
)


def _pooled_split(*, start: str) -> PooledDirectionSplit:
    sessions = np.asarray(
        [
            np.datetime64(start),
            np.datetime64(start),
            np.datetime64(start) + 1,
            np.datetime64(start) + 1,
        ]
    )
    return PooledDirectionSplit(
        sessions=sessions,
        symbols=np.asarray(["AAA", "BBB", "AAA", "BBB"]),
        features=np.arange(20, dtype=np.float64).reshape(4, 5),
        shifted_directions=np.asarray([0, 1, 1, 0]),
    )


def _evidence(*, detected: bool) -> BalancedAccuracyEvidence:
    return BalancedAccuracyEvidence(
        point_estimate=0.75 if detected else 0.5,
        lower_bound=0.500001 if detected else 0.5,
        sample_count=40,
        session_count=20,
    )


def test_pooled_dataset_keeps_train_and_valid_separate() -> None:
    dataset = T2V2Dataset(
        train=_pooled_split(start="2024-01-02"),
        valid=_pooled_split(start="2024-02-01"),
    )

    assert dataset.feature_names == FROZEN_T2_V2_FEATURE_NAMES
    assert dataset.train.features.shape == (4, 5)
    assert dataset.valid.features.shape == (4, 5)
    assert not dataset.train.features.flags.writeable


def test_pooled_split_rejects_duplicate_stock_session_rows() -> None:
    with pytest.raises(ValueError, match="stock/session pair must be unique"):
        PooledDirectionSplit(
            sessions=np.asarray(["2024-01-02", "2024-01-02"], dtype="datetime64[D]"),
            symbols=np.asarray(["AAA", "AAA"]),
            features=np.ones((2, 5)),
            shifted_directions=np.asarray([0, 1]),
        )


def test_pooled_dataset_rejects_overlapping_train_valid_sessions() -> None:
    split = _pooled_split(start="2024-01-02")
    with pytest.raises(ValueError, match="must not overlap"):
        T2V2Dataset(train=split, valid=split)


def test_balanced_accuracy_weights_direction_classes_equally() -> None:
    truth = np.asarray([0] * 8 + [1] * 2)
    predictions = np.asarray([0] * 8 + [0, 1])

    assert float(np.mean(truth == predictions)) == 0.9
    assert balanced_accuracy(truth, predictions) == 0.75


def test_balanced_accuracy_fails_closed_without_both_classes() -> None:
    with pytest.raises(ValueError, match="requires both direction classes"):
        balanced_accuracy(np.zeros(10, dtype=int), np.zeros(10, dtype=int))


def test_weak_control_flip_mask_is_keyed_not_order_dependent() -> None:
    sessions = np.asarray(
        ["2024-11-01", "2024-11-01", "2024-11-04", "2024-11-04"],
        dtype="datetime64[D]",
    )
    symbols = np.asarray(["AAA", "BBB", "AAA", "BBB"])
    original = deterministic_weak_control_flip_mask(sessions, symbols)
    permutation = np.asarray([3, 1, 0, 2])
    permuted = deterministic_weak_control_flip_mask(
        sessions[permutation], symbols[permutation]
    )

    restored = np.empty_like(permuted)
    restored[permutation] = permuted
    assert np.array_equal(original, restored)


def test_weak_control_flip_mask_has_frozen_golden_vector() -> None:
    sessions = np.asarray(
        ["2024-11-01", "2024-11-01", "2024-11-04", "2024-11-04"],
        dtype="datetime64[D]",
    )
    symbols = np.asarray(["AAA", "BBB", "AAA", "BBB"])

    assert deterministic_weak_control_flip_mask(sessions, symbols).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_moving_block_draws_are_contiguous_and_deterministic() -> None:
    sessions = np.datetime64("2024-01-01") + np.arange(12)
    first = moving_block_session_draws(
        sessions, block_size_sessions=5, bootstrap_samples=4
    )
    second = moving_block_session_draws(
        sessions, block_size_sessions=5, bootstrap_samples=4
    )

    assert first.shape == (4, 12)
    assert np.array_equal(first, second)
    for draw in first:
        assert np.all(np.diff(draw[:5]) == 1)
        assert np.all(np.diff(draw[5:10]) == 1)
        assert np.all(np.diff(draw[10:]) == 1)


def test_clustered_bootstrap_is_invariant_to_stock_row_order() -> None:
    sessions = np.repeat(np.datetime64("2024-01-01") + np.arange(10), 2)
    truth = np.tile(np.asarray([0, 1]), 10)
    predictions = truth.copy()
    predictions[-4:] = 1 - predictions[-4:]
    first = session_clustered_balanced_accuracy(
        sessions=sessions,
        y_true=truth,
        y_pred=predictions,
        block_size_sessions=2,
        bootstrap_samples=128,
    )
    permutation = np.random.default_rng(91).permutation(len(sessions))
    second = session_clustered_balanced_accuracy(
        sessions=sessions[permutation],
        y_true=truth[permutation],
        y_pred=predictions[permutation],
        block_size_sessions=2,
        bootstrap_samples=128,
    )

    assert first == second
    assert first.point_estimate == 0.8
    assert first.sample_count == 20
    assert first.session_count == 10


@pytest.mark.parametrize(
    ("real_detected", "strong_detected", "weak_detected"),
    list(product((False, True), repeat=3)),
)
def test_t2_v2_pass_requires_exact_three_condition_verdict(
    real_detected: bool,
    strong_detected: bool,
    weak_detected: bool,
) -> None:
    result = T2V2ProbeResult(
        real_pipeline=_evidence(detected=real_detected),
        strong_positive_control=_evidence(detected=strong_detected),
        weak_positive_control=_evidence(detected=weak_detected),
    )

    assert result.passed is (
        strong_detected and weak_detected and not real_detected
    )


def test_detection_lower_bound_is_strictly_above_random_baseline() -> None:
    assert not _evidence(detected=False).detected
    assert _evidence(detected=True).detected


def _history_fixture() -> tuple[
    StockBenchAdapter,
    list[datetime],
    np.ndarray,
    np.ndarray,
]:
    session_days = np.busday_offset(
        np.datetime64("2024-01-02"), np.arange(31), roll="forward"
    )
    session_closes = [
        datetime.fromisoformat(f"{day}T21:00:00+00:00") for day in session_days
    ]
    returns = np.column_stack(
        (
            np.linspace(-0.015, 0.025, 30),
            np.asarray([0.012 if index % 3 else -0.018 for index in range(30)]),
        )
    )
    closes = np.vstack(
        (
            np.asarray([[100.0, 80.0]]),
            np.asarray([[100.0, 80.0]]) * np.cumprod(1.0 + returns, axis=0),
        )
    )
    observations: list[MarketObservation] = []
    for session_index, (session_close, session_day) in enumerate(
        zip(session_closes, session_days, strict=True)
    ):
        for symbol_index, symbol in enumerate(("AAA", "BBB")):
            observations.append(
                MarketObservation(
                    observation_id=f"market:{symbol}:{session_day}",
                    kind="market",
                    symbol=symbol,
                    event_at=session_close,
                    available_at=session_close,
                    payload={
                        "date": str(session_day),
                        "close": float(closes[session_index, symbol_index]),
                    },
                )
            )
    return (
        StockBenchAdapter(observations, data_version="history-fixture-v1"),
        session_closes,
        session_days,
        closes,
    )


def test_history_builder_uses_frozen_features_and_shifted_label_inside_each_split() -> None:
    adapter, session_closes, session_days, closes = _history_fixture()

    dataset = build_history_t2_v2_dataset(
        history_adapter=adapter,
        session_closes=session_closes,
        symbols=("AAA", "BBB"),
        train_start=str(session_days[0]),
        train_end=str(session_days[25]),
        valid_start=str(session_days[26]),
        valid_end=str(session_days[30]),
    )

    returns = closes[1:] / closes[:-1] - 1.0
    feature_session_index = 20
    expected_aaa = np.asarray(
        [
            returns[feature_session_index - 1, 0],
            returns[feature_session_index - 20 : feature_session_index, 0].mean(),
            returns[feature_session_index - 20 : feature_session_index, 0].std(ddof=0),
            returns[feature_session_index - 1].std(ddof=0),
            (returns[feature_session_index - 1] > 0.0).mean(),
        ]
    )
    assert dataset.feature_names == FROZEN_T2_V2_FEATURE_NAMES
    assert dataset.train.features[0] == pytest.approx(expected_aaa)
    assert dataset.train.shifted_directions[0] == int(
        returns[feature_session_index + 1, 0] >= 0.0
    )
    assert dataset.train.sessions.tolist() == np.repeat(session_days[20:24], 2).tolist()
    assert dataset.valid.sessions.tolist() == np.repeat(session_days[26:29], 2).tolist()
    assert dataset.train.symbols.tolist() == ["AAA", "BBB"] * 4
    assert dataset.valid.symbols.tolist() == ["AAA", "BBB"] * 3


def test_ridge_direction_model_freezes_train_only_standardization() -> None:
    train_features = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    )
    train_labels = np.asarray([0, 0, 1, 1])
    model = fit_ridge_direction_model(train_features, train_labels)

    assert isinstance(model, RidgeDirectionModel)
    assert model.feature_means == pytest.approx([2.5, 25.0])
    assert model.feature_scales == pytest.approx(
        train_features.std(axis=0, ddof=0)
    )
    assert not model.feature_means.flags.writeable
    assert model.predict(np.asarray([[1_000_000.0, -1_000_000.0]])).shape == (1,)
    assert model.feature_means == pytest.approx([2.5, 25.0])


def _control_power_dataset() -> T2V2Dataset:
    symbols = np.asarray([f"S{index:02d}" for index in range(20)])

    def split(*, start: str, session_count: int) -> PooledDirectionSplit:
        unique_sessions = np.datetime64(start) + np.arange(session_count)
        sessions = np.repeat(unique_sessions, len(symbols))
        row_symbols = np.tile(symbols, session_count)
        directions = np.tile(np.arange(len(symbols)) % 2, session_count)
        return PooledDirectionSplit(
            sessions=sessions,
            symbols=row_symbols,
            features=np.zeros((len(sessions), len(FROZEN_T2_V2_FEATURE_NAMES))),
            shifted_directions=directions,
        )

    return T2V2Dataset(
        train=split(start="2023-01-03", session_count=120),
        valid=split(start="2024-11-01", session_count=100),
    )


def test_t2_v2_runner_detects_both_controls_but_not_the_real_pipeline() -> None:
    dataset = _control_power_dataset()

    result = run_t2_v2_probe(dataset, bootstrap_samples=256)

    assert result.real_pipeline.point_estimate == 0.5
    assert not result.real_pipeline.detected
    assert result.strong_positive_control.point_estimate == 1.0
    assert result.strong_positive_control.detected
    assert 0.55 <= result.weak_positive_control.point_estimate <= 0.65
    assert result.weak_positive_control.detected
    assert result.passed
    assert result == run_t2_v2_probe(dataset, bootstrap_samples=256)
    assert dataset.train.features.shape[1] == len(FROZEN_T2_V2_FEATURE_NAMES)
    assert dataset.valid.features.shape[1] == len(FROZEN_T2_V2_FEATURE_NAMES)


def test_configured_t2_v2_accepts_only_the_frozen_history_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")

    settings = validate_t2_v2_config(config)

    assert settings.history_source_name == "history"
    assert settings.train_start.isoformat() == "2023-01-03"
    assert settings.valid_end.isoformat() == "2025-02-28"
    assert settings.block_size_sessions == 5
    assert settings.bootstrap_samples == 2_000
    assert settings.seed == 1729


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("data", "split_source", "valid"), "stockbench"),
        (
            ("evaluation", "shifted_label_probe_v2", "source_policy", "stockbench_oos_access"),
            "allowed",
        ),
        (("evaluation", "shifted_label_probe_v2", "metric"), "accuracy"),
        (
            ("evaluation", "shifted_label_probe_v2", "bootstrap", "block_size_sessions"),
            6,
        ),
    ],
)
def test_configured_t2_v2_rejects_protocol_drift_before_data_access(
    path: tuple[str, ...], invalid_value: object
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    with pytest.raises(ValueError, match="T2 v2 configuration drift"):
        validate_t2_v2_config(config)


def test_t2_v2_cli_is_importable_outside_the_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(root / "scripts/run_t2_v2_probe.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--output" in completed.stdout


def _persisted_run(config: dict[str, Any]) -> ConfiguredT2V2Run:
    return ConfiguredT2V2Run(
        result=T2V2ProbeResult(
            real_pipeline=BalancedAccuracyEvidence(
                point_estimate=0.499,
                lower_bound=0.494,
                sample_count=1_560,
                session_count=78,
            ),
            strong_positive_control=BalancedAccuracyEvidence(
                point_estimate=1.0,
                lower_bound=1.0,
                sample_count=1_560,
                session_count=78,
            ),
            weak_positive_control=BalancedAccuracyEvidence(
                point_estimate=0.595,
                lower_bound=0.575,
                sample_count=1_560,
                session_count=78,
            ),
        ),
        protocol_sha256=t2_v2_protocol_sha256(config),
        history_manifest_sha256=(
            "2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90"
        ),
        history_file_count=10_820,
        train_sample_count=8_780,
        train_session_count=439,
        valid_sample_count=1_560,
        valid_session_count=78,
    )


def _bind_artifact(
    config: dict[str, Any], *, repo_root: Path, raw_bytes: bytes
) -> Path:
    artifact = repo_root / "t2_v2_result.json"
    artifact.write_bytes(raw_bytes)
    probe = config["evaluation"]["shifted_label_probe_v2"]
    probe["implementation_status"] = "passed"
    probe["result_artifact"] = artifact.name
    probe["result_sha256"] = sha256(raw_bytes).hexdigest()
    return artifact


def test_t2_v2_artifact_verifier_accepts_bound_three_condition_evidence(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")
    raw_bytes = t2_v2_artifact_bytes(_persisted_run(config))
    _bind_artifact(config, repo_root=tmp_path, raw_bytes=raw_bytes)

    evidence = verify_t2_v2_artifact(config, repo_root=tmp_path)

    assert evidence == {
        "artifact": str((tmp_path / "t2_v2_result.json").resolve()),
        "artifact_sha256": sha256(raw_bytes).hexdigest(),
        "history_manifest_sha256": (
            "2612e707446b1c8bcd73ac6583b84558d3c4eddbaa9fc20adf7023c3c8a76c90"
        ),
        "passed": True,
        "protocol_sha256": t2_v2_protocol_sha256(config),
        "stockbench_oos_accessed": False,
    }


def test_t2_v2_artifact_verifier_rejects_raw_byte_tampering(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")
    raw_bytes = t2_v2_artifact_bytes(_persisted_run(config))
    artifact = _bind_artifact(config, repo_root=tmp_path, raw_bytes=raw_bytes)
    artifact.write_bytes(raw_bytes.replace(b'"passed": true', b'"passed":false'))

    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        verify_t2_v2_artifact(config, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "key", "invalid_value", "message"),
    [
        ("protocol", "sha256", "0" * 64, "protocol SHA-256 mismatch"),
        (
            "protocol",
            "positive_control_features_persisted",
            0,
            "protocol metadata mismatch",
        ),
        ("dataset", "stockbench_oos_accessed", True, "StockBench OOS"),
        ("real_pipeline", "lower_bound", 0.6, "three-condition verdict"),
        ("weak_positive_control", "lower_bound", 0.5, "three-condition verdict"),
    ],
)
def test_t2_v2_artifact_verifier_recomputes_semantics_after_hash_binding(
    tmp_path: Path,
    section: str,
    key: str,
    invalid_value: object,
    message: str,
) -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs/base.yaml")
    document = cast(dict[str, Any], _persisted_run(config).to_document())
    if section in {"real_pipeline", "weak_positive_control"}:
        document["result"][section][key] = invalid_value
    else:
        document[section][key] = invalid_value
    raw_bytes = (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    _bind_artifact(config, repo_root=tmp_path, raw_bytes=raw_bytes)

    with pytest.raises(RuntimeError, match=message):
        verify_t2_v2_artifact(config, repo_root=tmp_path)
