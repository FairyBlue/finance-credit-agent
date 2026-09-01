"""History-only orchestration for the preregistered T2 v2 leakage probe."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, time
from hashlib import sha256
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np

from src.data import AvailabilityPolicy, StockBenchAdapter, verify_raw_daily_manifest

from .leakage_probe_v2 import (
    DEFAULT_T2_V2_ALPHA,
    DEFAULT_T2_V2_BASELINE,
    DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
    DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
    DEFAULT_T2_V2_SEED,
    FROZEN_T2_V2_FEATURE_NAMES,
    RIDGE_PENALTY,
    T2_V2_ROLLING_WINDOW_SESSIONS,
    WEAK_CONTROL_FLIP_PROBABILITY,
    BalancedAccuracyEvidence,
    T2V2ProbeResult,
    build_history_t2_v2_dataset,
    run_t2_v2_probe,
)

FROZEN_T2_V2_SYMBOLS = (
    "GS",
    "MSFT",
    "HD",
    "V",
    "SHW",
    "CAT",
    "MCD",
    "UNH",
    "AXP",
    "AMGN",
    "TRV",
    "CRM",
    "JPM",
    "IBM",
    "HON",
    "BA",
    "AMZN",
    "AAPL",
    "PG",
    "JNJ",
)


def _required(document: dict[str, Any], *path: str) -> Any:
    current: Any = document
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing configuration key: {'.'.join(path)}")
        current = current[key]
    return current


def _expect(document: dict[str, Any], path: tuple[str, ...], expected: object) -> None:
    try:
        actual = _required(document, *path)
    except ValueError as error:
        raise ValueError(f"T2 v2 configuration drift: {error}") from error
    if actual != expected:
        rendered = ".".join(path)
        raise ValueError(
            f"T2 v2 configuration drift: {rendered} must be {expected!r}, got {actual!r}"
        )


@dataclass(frozen=True, slots=True)
class T2V2ProtocolSettings:
    """Validated runtime values copied from the frozen protocol configuration."""

    history_source_name: str
    symbols: tuple[str, ...]
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    block_size_sessions: int
    bootstrap_samples: int
    seed: int
    alpha: float
    baseline: float


def validate_t2_v2_config(config: dict[str, Any]) -> T2V2ProtocolSettings:
    """Reject configuration drift before any history or OOS path is opened."""

    exact_values: tuple[tuple[tuple[str, ...], object], ...] = (
        (("data", "split_source", "train"), "history"),
        (("data", "split_source", "valid"), "history"),
        (("data", "windows", "train", "start"), "2023-01-03"),
        (("data", "windows", "train", "end"), "2024-10-31"),
        (("data", "windows", "valid", "start"), "2024-11-01"),
        (("data", "windows", "valid", "end"), "2025-02-28"),
        (("data", "sources", "history", "input_window", "start"), "2023-01-03"),
        (("data", "sources", "history", "input_window", "end"), "2025-02-28"),
        (("data", "universe", "symbols"), list(FROZEN_T2_V2_SYMBOLS)),
        (("data", "availability", "timezone"), "America/New_York"),
        (("data", "availability", "exchange_calendar"), "XNYS"),
        (("data", "availability", "market_close_time"), "16:00:00"),
        (
            ("evaluation", "shifted_label_probe_v2", "source_policy"),
            {
                "allowed_source": "history",
                "fit_split": "train",
                "decision_split": "valid",
                "stockbench_oos_access": "forbidden",
            },
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "sample_policy"),
            {
                "unit": "stock_session",
                "pooling": "pooled_across_stocks",
                "bootstrap_cluster": "session",
            },
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "features"),
            {
                "frozen": True,
                "availability": "causal_as_of_session_close",
                "names": list(FROZEN_T2_V2_FEATURE_NAMES),
            },
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "label"),
            {
                "kind": "per_stock_direction",
                "aligned_horizon_sessions": 1,
                "shifted_horizon_sessions": 2,
                "shift_sessions": 1,
            },
        ),
        (("evaluation", "shifted_label_probe_v2", "metric"), "balanced_accuracy"),
        (
            ("evaluation", "shifted_label_probe_v2", "random_baseline"),
            DEFAULT_T2_V2_BASELINE,
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "bootstrap"),
            {
                "method": "session_clustered_moving_block",
                "block_size_sessions": DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
                "samples": DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
                "seed": DEFAULT_T2_V2_SEED,
                "alpha": DEFAULT_T2_V2_ALPHA,
            },
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "detection_rule"),
            "one_sided_lower_bound_above_random_baseline",
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "positive_controls"),
            {
                "persistence": "ephemeral_not_persisted",
                "strong": {
                    "feature": "exact_future_direction",
                    "flip_probability": 0.0,
                },
                "weak": {
                    "feature": "future_direction_with_independent_deterministic_flips",
                    "flip_probability": WEAK_CONTROL_FLIP_PROBABILITY,
                    "approximate_balanced_accuracy": 0.60,
                    "flip_generator": {
                        "kind": "sha256_threshold_by_session_symbol_v1",
                        "seed": DEFAULT_T2_V2_SEED,
                        "input_format": "seed<TAB>YYYY-MM-DD<TAB>SYMBOL<LF>",
                    },
                },
            },
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "pass_criteria"),
            {
                "strong_positive_control_detected": True,
                "weak_positive_control_detected": True,
                "real_pipeline_significantly_detected": False,
            },
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "weak_control_miss_policy"),
            "record_probe_power_boundary",
        ),
        (
            (
                "evaluation",
                "shifted_label_probe_v2",
                "real_pipeline_non_detection_semantics",
            ),
            "no_significant_evidence_not_proof_of_absence",
        ),
        (
            ("evaluation", "shifted_label_probe_v2", "require_before_main_training"),
            True,
        ),
    )
    for path, expected in exact_values:
        _expect(config, path, expected)
    status = _required(config, "evaluation", "shifted_label_probe_v2", "implementation_status")
    if status not in {
        "preregistered_pending_implementation",
        "implemented_pending_run",
        "passed",
    }:
        raise ValueError(
            "T2 v2 configuration drift: unsupported implementation_status "
            f"{status!r}"
        )
    return T2V2ProtocolSettings(
        history_source_name="history",
        symbols=FROZEN_T2_V2_SYMBOLS,
        train_start=date(2023, 1, 3),
        train_end=date(2024, 10, 31),
        valid_start=date(2024, 11, 1),
        valid_end=date(2025, 2, 28),
        block_size_sessions=DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
        bootstrap_samples=DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
        seed=DEFAULT_T2_V2_SEED,
        alpha=DEFAULT_T2_V2_ALPHA,
        baseline=DEFAULT_T2_V2_BASELINE,
    )


def _t2_v2_protocol_payload(config: dict[str, Any]) -> dict[str, object]:
    """Return the immutable scientific/runtime fields bound to result evidence."""

    settings = validate_t2_v2_config(config)
    source = _required(config, "data", "sources", settings.history_source_name)
    probe = _required(config, "evaluation", "shifted_label_probe_v2")
    if not isinstance(source, dict) or not isinstance(probe, dict):
        raise ValueError("T2 v2 source and probe configuration must be mappings")
    return {
        "schema_version": "t2_v2_protocol_v1",
        "source": {
            "name": settings.history_source_name,
            "version": _required(source, "version"),
            "input_manifest_sha256": _required(source, "input_manifest_sha256"),
            "files": int(_required(source, "files")),
            "sessions_per_symbol": int(_required(source, "sessions_per_symbol")),
            "input_window": _required(source, "input_window"),
        },
        "windows": {
            "train": _required(config, "data", "windows", "train"),
            "valid": _required(config, "data", "windows", "valid"),
        },
        "symbols": list(settings.symbols),
        "availability": {
            "timezone": _required(config, "data", "availability", "timezone"),
            "exchange_calendar": _required(
                config, "data", "availability", "exchange_calendar"
            ),
            "market_close_time": _required(
                config, "data", "availability", "market_close_time"
            ),
        },
        "probe": {
            key: probe[key]
            for key in (
                "source_policy",
                "sample_policy",
                "features",
                "label",
                "metric",
                "random_baseline",
                "bootstrap",
                "detection_rule",
                "positive_controls",
                "pass_criteria",
                "weak_control_miss_policy",
                "real_pipeline_non_detection_semantics",
                "require_before_main_training",
            )
        },
        "implementation": {
            "model": "ridge_linear_direction_v1",
            "ridge_penalty": RIDGE_PENALTY,
            "standardization": "train_only_population_std",
            "rolling_window_sessions": T2_V2_ROLLING_WINDOW_SESSIONS,
            "rolling_std_ddof": 0,
            "positive_control_injection": "append_ephemeral_feature",
            "zero_return_direction": "non_negative_is_up",
        },
    }


def t2_v2_protocol_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        _t2_v2_protocol_payload(config),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class ConfiguredT2V2Run:
    """Aggregate, persistence-safe evidence from one configured T2 v2 run."""

    result: T2V2ProbeResult
    protocol_sha256: str
    history_manifest_sha256: str
    history_file_count: int
    train_sample_count: int
    train_session_count: int
    valid_sample_count: int
    valid_session_count: int

    @property
    def passed(self) -> bool:
        return self.result.passed

    def to_document(self) -> dict[str, object]:
        return {
            "schema_version": "t2_v2_result_v2",
            "status": "passed" if self.passed else "failed",
            "passed": self.passed,
            "history_manifest_sha256": self.history_manifest_sha256,
            "history_file_count": self.history_file_count,
            "dataset": {
                "source": "history",
                "stockbench_oos_accessed": False,
                "train_sample_count": self.train_sample_count,
                "train_session_count": self.train_session_count,
                "valid_sample_count": self.valid_sample_count,
                "valid_session_count": self.valid_session_count,
            },
            "protocol": {
                "sha256": self.protocol_sha256,
                "metric": "balanced_accuracy",
                "baseline": DEFAULT_T2_V2_BASELINE,
                "block_size_sessions": DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
                "bootstrap_samples": DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
                "seed": DEFAULT_T2_V2_SEED,
                "alpha": DEFAULT_T2_V2_ALPHA,
                "positive_control_features_persisted": False,
            },
            "result": {
                "real_pipeline": asdict(self.result.real_pipeline),
                "strong_positive_control": asdict(
                    self.result.strong_positive_control
                ),
                "weak_positive_control": asdict(self.result.weak_positive_control),
            },
            "interpretation": {
                "weak_control_miss": "probe_power_boundary",
                "real_pipeline_non_detection": (
                    "no_significant_evidence_not_proof_of_absence"
                ),
            },
        }


def t2_v2_artifact_bytes(configured_run: ConfiguredT2V2Run) -> bytes:
    """Serialize aggregate evidence without persisting positive-control features."""

    return (
        json.dumps(
            configured_run.to_document(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"T2 v2 artifact {name} must be a mapping")
    return value


def _require_exact_keys(
    document: dict[str, Any], *, expected: set[str], name: str
) -> None:
    if set(document) != expected:
        raise RuntimeError(
            f"T2 v2 artifact {name} keys mismatch: "
            f"expected {sorted(expected)}, got {sorted(document)}"
        )


def _artifact_evidence(
    document: object,
    *,
    name: str,
    valid_sample_count: int,
    valid_session_count: int,
) -> BalancedAccuracyEvidence:
    evidence = _artifact_mapping(document, name=name)
    expected_keys = {
        "point_estimate",
        "lower_bound",
        "sample_count",
        "session_count",
        "baseline",
        "block_size_sessions",
        "bootstrap_samples",
        "alpha",
        "seed",
    }
    _require_exact_keys(evidence, expected=expected_keys, name=name)
    for key in ("point_estimate", "lower_bound", "baseline", "alpha"):
        value = evidence[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"T2 v2 artifact {name}.{key} must be numeric")
        if not math.isfinite(float(value)):
            raise RuntimeError(f"T2 v2 artifact {name}.{key} must be finite")
    for key in (
        "sample_count",
        "session_count",
        "block_size_sessions",
        "bootstrap_samples",
        "seed",
    ):
        if isinstance(evidence[key], bool) or not isinstance(evidence[key], int):
            raise RuntimeError(f"T2 v2 artifact {name}.{key} must be an integer")
    if not 0.0 <= float(evidence["point_estimate"]) <= 1.0:
        raise RuntimeError(f"T2 v2 artifact {name}.point_estimate must be in [0, 1]")
    if not 0.0 <= float(evidence["lower_bound"]) <= 1.0:
        raise RuntimeError(f"T2 v2 artifact {name}.lower_bound must be in [0, 1]")
    exact_metadata = {
        "sample_count": valid_sample_count,
        "session_count": valid_session_count,
        "baseline": DEFAULT_T2_V2_BASELINE,
        "block_size_sessions": DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
        "bootstrap_samples": DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
        "alpha": DEFAULT_T2_V2_ALPHA,
        "seed": DEFAULT_T2_V2_SEED,
    }
    for key, expected in exact_metadata.items():
        if evidence[key] != expected:
            raise RuntimeError(
                f"T2 v2 artifact {name}.{key} must be {expected!r}, "
                f"got {evidence[key]!r}"
            )
    return BalancedAccuracyEvidence(**evidence)


def verify_t2_v2_artifact(
    config: dict[str, Any], *, repo_root: str | Path
) -> dict[str, object]:
    """Recompute the frozen verdict from a hash-bound aggregate result artifact."""

    settings = validate_t2_v2_config(config)
    probe = _required(config, "evaluation", "shifted_label_probe_v2")
    if not isinstance(probe, dict) or probe.get("implementation_status") != "passed":
        raise RuntimeError("T2 v2 implementation_status must be passed")
    root = Path(repo_root).expanduser().resolve()
    artifact = _resolved_path(root, _required(probe, "result_artifact"))
    if not artifact.is_relative_to(root):
        raise RuntimeError("T2 v2 result artifact must be inside the repository")
    if not artifact.is_file():
        raise RuntimeError(f"T2 v2 result artifact is missing: {artifact}")
    raw_bytes = artifact.read_bytes()
    actual_sha256 = sha256(raw_bytes).hexdigest()
    expected_sha256 = str(_required(probe, "result_sha256"))
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "T2 v2 artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        loaded = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("T2 v2 artifact is not valid UTF-8 JSON") from error
    document = _artifact_mapping(loaded, name="root")
    _require_exact_keys(
        document,
        expected={
            "schema_version",
            "status",
            "passed",
            "history_manifest_sha256",
            "history_file_count",
            "dataset",
            "protocol",
            "result",
            "interpretation",
        },
        name="root",
    )
    if document["schema_version"] != "t2_v2_result_v2":
        raise RuntimeError("T2 v2 artifact schema_version mismatch")
    expected_protocol_sha256 = t2_v2_protocol_sha256(config)
    protocol = _artifact_mapping(document["protocol"], name="protocol")
    _require_exact_keys(
        protocol,
        expected={
            "sha256",
            "metric",
            "baseline",
            "block_size_sessions",
            "bootstrap_samples",
            "seed",
            "alpha",
            "positive_control_features_persisted",
        },
        name="protocol",
    )
    if protocol["sha256"] != expected_protocol_sha256:
        raise RuntimeError("T2 v2 artifact protocol SHA-256 mismatch")
    expected_protocol_metadata = {
        "metric": "balanced_accuracy",
        "baseline": DEFAULT_T2_V2_BASELINE,
        "block_size_sessions": DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
        "bootstrap_samples": DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
        "seed": DEFAULT_T2_V2_SEED,
        "alpha": DEFAULT_T2_V2_ALPHA,
        "positive_control_features_persisted": False,
    }
    if protocol["positive_control_features_persisted"] is not False:
        raise RuntimeError("T2 v2 artifact protocol metadata mismatch")
    if any(
        protocol[key] != value
        for key, value in expected_protocol_metadata.items()
        if key != "positive_control_features_persisted"
    ):
        raise RuntimeError("T2 v2 artifact protocol metadata mismatch")

    source = _required(config, "data", "sources", settings.history_source_name)
    if not isinstance(source, dict):
        raise RuntimeError("T2 v2 history source must be a mapping")
    if document["history_manifest_sha256"] != source.get("input_manifest_sha256"):
        raise RuntimeError("T2 v2 artifact history manifest SHA-256 mismatch")
    if document["history_file_count"] != int(_required(source, "files")):
        raise RuntimeError("T2 v2 artifact history file count mismatch")

    calendar = xcals.get_calendar(
        str(_required(config, "data", "availability", "exchange_calendar"))
    )
    train_session_total = len(
        calendar.sessions_in_range(
            settings.train_start.isoformat(), settings.train_end.isoformat()
        )
    )
    valid_session_total = len(
        calendar.sessions_in_range(
            settings.valid_start.isoformat(), settings.valid_end.isoformat()
        )
    )
    expected_train_sessions = train_session_total - T2_V2_ROLLING_WINDOW_SESSIONS - 2
    expected_valid_sessions = valid_session_total - 2
    expected_train_sample_count = expected_train_sessions * len(settings.symbols)
    expected_valid_sample_count = expected_valid_sessions * len(settings.symbols)
    expected_dataset = {
        "source": "history",
        "stockbench_oos_accessed": False,
        "train_sample_count": expected_train_sample_count,
        "train_session_count": expected_train_sessions,
        "valid_sample_count": expected_valid_sample_count,
        "valid_session_count": expected_valid_sessions,
    }
    dataset = _artifact_mapping(document["dataset"], name="dataset")
    _require_exact_keys(dataset, expected=set(expected_dataset), name="dataset")
    if dataset.get("stockbench_oos_accessed") is not False:
        raise RuntimeError("T2 v2 artifact reports StockBench OOS access")
    if dataset != expected_dataset:
        raise RuntimeError("T2 v2 artifact dataset identity mismatch")

    result_document = _artifact_mapping(document["result"], name="result")
    _require_exact_keys(
        result_document,
        expected={
            "real_pipeline",
            "strong_positive_control",
            "weak_positive_control",
        },
        name="result",
    )
    result = T2V2ProbeResult(
        real_pipeline=_artifact_evidence(
            result_document["real_pipeline"],
            name="real_pipeline",
            valid_sample_count=expected_valid_sample_count,
            valid_session_count=expected_valid_sessions,
        ),
        strong_positive_control=_artifact_evidence(
            result_document["strong_positive_control"],
            name="strong_positive_control",
            valid_sample_count=expected_valid_sample_count,
            valid_session_count=expected_valid_sessions,
        ),
        weak_positive_control=_artifact_evidence(
            result_document["weak_positive_control"],
            name="weak_positive_control",
            valid_sample_count=expected_valid_sample_count,
            valid_session_count=expected_valid_sessions,
        ),
    )
    if document["passed"] is not True or document["status"] != "passed" or not result.passed:
        raise RuntimeError("T2 v2 artifact fails the frozen three-condition verdict")
    interpretation = _artifact_mapping(document["interpretation"], name="interpretation")
    expected_interpretation = {
        "weak_control_miss": "probe_power_boundary",
        "real_pipeline_non_detection": "no_significant_evidence_not_proof_of_absence",
    }
    if interpretation != expected_interpretation:
        raise RuntimeError("T2 v2 artifact interpretation metadata mismatch")
    return {
        "artifact": str(artifact),
        "artifact_sha256": actual_sha256,
        "history_manifest_sha256": str(document["history_manifest_sha256"]),
        "passed": True,
        "protocol_sha256": expected_protocol_sha256,
        "stockbench_oos_accessed": False,
    }


def _resolved_path(repo_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def run_configured_t2_v2_probe(
    config: dict[str, Any], *, repo_root: str | Path
) -> ConfiguredT2V2Run:
    """Verify frozen history bytes, build the dataset, and run T2 v2."""

    settings = validate_t2_v2_config(config)
    root = Path(repo_root).expanduser().resolve()
    source = _required(config, "data", "sources", settings.history_source_name)
    if not isinstance(source, dict):
        raise ValueError("history source configuration must be a mapping")
    manifest_root = _resolved_path(root, _required(source, "manifest_root"))
    storage_root = _resolved_path(root, _required(source, "storage_root"))
    availability = _required(config, "data", "availability")
    if not isinstance(availability, dict):
        raise ValueError("data.availability must be a mapping")
    calendar_name = str(_required(availability, "exchange_calendar"))
    audit = verify_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage_root,
        symbols=settings.symbols,
        start=settings.train_start,
        end=settings.valid_end,
        expected_sha256=str(_required(source, "input_manifest_sha256")),
        expected_file_count=int(_required(source, "files")),
        expected_sessions_per_symbol=int(_required(source, "sessions_per_symbol")),
        exchange_calendar=calendar_name,
    )
    expected_raw_sha256_by_path: dict[str, str] = {}
    for line in audit.manifest_bytes.decode("utf-8").splitlines():
        relative, digest = line.split("\t")
        expected_raw_sha256_by_path[str((manifest_root / relative).resolve())] = digest
    policy = AvailabilityPolicy(
        str(_required(availability, "timezone")),
        time.fromisoformat(str(_required(availability, "market_close_time"))),
        calendar_name,
    )
    adapter = StockBenchAdapter.from_storage(
        storage_root,
        symbols=settings.symbols,
        data_version=str(_required(source, "version")),
        availability_policy=policy,
        # The raw-byte manifest already binds every exact Parquet byte. Logical
        # sidecars are redundant here and would add 10,820 small-file reads.
        verify_market_checksums=False,
        expected_raw_sha256_by_path=expected_raw_sha256_by_path,
        start_date=settings.train_start,
        end_date=settings.valid_end,
    )
    calendar = xcals.get_calendar(calendar_name)
    session_closes = [
        calendar.session_close(session).to_pydatetime()
        for session in calendar.sessions_in_range(
            settings.train_start.isoformat(), settings.valid_end.isoformat()
        )
    ]
    dataset = build_history_t2_v2_dataset(
        history_adapter=adapter,
        session_closes=session_closes,
        symbols=settings.symbols,
        train_start=settings.train_start,
        train_end=settings.train_end,
        valid_start=settings.valid_start,
        valid_end=settings.valid_end,
    )
    result = run_t2_v2_probe(
        dataset,
        block_size_sessions=settings.block_size_sessions,
        bootstrap_samples=settings.bootstrap_samples,
        seed=settings.seed,
        alpha=settings.alpha,
        baseline=settings.baseline,
    )
    return ConfiguredT2V2Run(
        result=result,
        protocol_sha256=t2_v2_protocol_sha256(config),
        history_manifest_sha256=audit.manifest_sha256,
        history_file_count=audit.file_count,
        train_sample_count=len(dataset.train.features),
        train_session_count=len(np.unique(dataset.train.sessions)),
        valid_sample_count=len(dataset.valid.features),
        valid_session_count=len(np.unique(dataset.valid.sessions)),
    )
