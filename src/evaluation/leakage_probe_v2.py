"""Test-first statistical core for the history-only T2 v2 leakage probe.

This module deliberately contains no storage adapter, CLI, artifact writer, or
launcher integration. It freezes the reusable contracts that those later
layers must call: pooled stock/session data, balanced accuracy, deterministic
weak-control flips, session-clustered moving-block bootstrap, and the final
three-condition verdict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from fractions import Fraction
from hashlib import sha256
from numbers import Integral, Real

import numpy as np

from src.data import StockBenchAdapter

DEFAULT_T2_V2_SEED = 1729
DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS = 5
DEFAULT_T2_V2_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_T2_V2_ALPHA = 0.05
DEFAULT_T2_V2_BASELINE = 0.5
WEAK_CONTROL_FLIP_PROBABILITY = 0.40
RIDGE_PENALTY = 1e-6
T2_V2_ROLLING_WINDOW_SESSIONS = 20

FROZEN_T2_V2_FEATURE_NAMES = (
    "stock_return_1d",
    "stock_return_mean_20d",
    "stock_return_volatility_20d",
    "cross_sectional_dispersion_1d",
    "cross_sectional_positive_fraction_1d",
)


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = values.copy()
    copied.setflags(write=False)
    return copied


def _as_sessions(values: Sequence[object] | np.ndarray, *, name: str) -> np.ndarray:
    sessions = np.asarray(values, dtype="datetime64[D]")
    if sessions.ndim != 1 or not len(sessions):
        raise ValueError(f"{name} must be a non-empty one-dimensional session array")
    if np.any(np.isnat(sessions)):
        raise ValueError(f"{name} must not contain NaT")
    return sessions


def _as_symbols(values: Sequence[str] | np.ndarray, *, name: str) -> np.ndarray:
    symbols = np.asarray(values, dtype=str)
    if symbols.ndim != 1 or not len(symbols):
        raise ValueError(f"{name} must be a non-empty one-dimensional symbol array")
    for symbol in symbols:
        if (
            not symbol
            or not symbol.isascii()
            or symbol != symbol.strip()
            or symbol != symbol.upper()
        ):
            raise ValueError(f"{name} must contain canonical uppercase ASCII symbols")
    return symbols


def _as_binary_vector(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a non-empty one-dimensional binary array")
    if not np.all(np.isin(vector, (0, 1))):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return vector.astype(np.int8, copy=False)


def _as_feature_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("features must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix


@dataclass(frozen=True, slots=True)
class PooledDirectionSplit:
    """One already-separated Train or Valid stock-by-session sample pool."""

    sessions: np.ndarray
    symbols: np.ndarray
    features: np.ndarray
    shifted_directions: np.ndarray

    def __post_init__(self) -> None:
        sessions = _as_sessions(self.sessions, name="sessions")
        symbols = _as_symbols(self.symbols, name="symbols")
        features = _as_feature_matrix(self.features)
        directions = _as_binary_vector(
            self.shifted_directions, name="shifted_directions"
        )
        row_count = len(sessions)
        if len(symbols) != row_count or len(features) != row_count or len(directions) != row_count:
            raise ValueError("sessions, symbols, features, and shifted_directions must align")
        keys = {
            (np.datetime_as_string(session, unit="D"), symbol)
            for session, symbol in zip(sessions, symbols, strict=True)
        }
        if len(keys) != row_count:
            raise ValueError("each stock/session pair must be unique within a split")
        object.__setattr__(self, "sessions", _readonly_copy(sessions))
        object.__setattr__(self, "symbols", _readonly_copy(symbols))
        object.__setattr__(self, "features", _readonly_copy(features))
        object.__setattr__(self, "shifted_directions", _readonly_copy(directions))


@dataclass(frozen=True, slots=True)
class T2V2Dataset:
    """History-only Train/Valid pools kept separate by construction."""

    train: PooledDirectionSplit
    valid: PooledDirectionSplit
    feature_names: tuple[str, ...] = FROZEN_T2_V2_FEATURE_NAMES

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != FROZEN_T2_V2_FEATURE_NAMES:
            raise ValueError("T2 v2 feature names must match the frozen protocol")
        expected_width = len(FROZEN_T2_V2_FEATURE_NAMES)
        if self.train.features.shape[1] != expected_width:
            raise ValueError("train feature width does not match the frozen protocol")
        if self.valid.features.shape[1] != expected_width:
            raise ValueError("valid feature width does not match the frozen protocol")
        if np.intersect1d(self.train.sessions, self.valid.sessions).size:
            raise ValueError("Train and Valid sessions must not overlap")


def _as_boundary_date(value: date | str, *, name: str) -> np.datetime64:
    try:
        boundary = np.datetime64(value, "D")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an ISO date") from error
    if np.isnat(boundary):
        raise ValueError(f"{name} must not be NaT")
    return boundary


def _normalize_session_closes(
    values: Sequence[datetime],
) -> tuple[tuple[datetime, ...], np.ndarray]:
    if not values:
        raise ValueError("session_closes must not be empty")
    normalized: list[datetime] = []
    for value in values:
        if not isinstance(value, datetime):
            raise ValueError("session_closes must contain datetimes")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("session_closes must be timezone-aware")
        normalized.append(value.astimezone(timezone.utc))
    if any(
        current >= following
        for current, following in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("session_closes must be strictly increasing")
    sessions = np.asarray(
        [value.date().isoformat() for value in normalized], dtype="datetime64[D]"
    )
    if len(np.unique(sessions)) != len(sessions):
        raise ValueError("session_closes must contain at most one close per session")
    return tuple(normalized), sessions


def _causal_close_matrix(
    *,
    history_adapter: StockBenchAdapter,
    session_closes: tuple[datetime, ...],
    sessions: np.ndarray,
    symbols: np.ndarray,
) -> np.ndarray:
    rows: list[list[float]] = []
    symbol_set = set(symbols)
    for session_close, session in zip(session_closes, sessions, strict=True):
        rendered_session = np.datetime_as_string(session, unit="D")
        closes: dict[str, float] = {}
        for observation in history_adapter.get_observations(session_close):
            if (
                observation.kind != "market"
                or observation.symbol not in symbol_set
                or observation.payload.get("date") != rendered_session
            ):
                continue
            if observation.symbol in closes:
                raise ValueError(
                    f"duplicate close for {observation.symbol} on {rendered_session}"
                )
            close = float(observation.payload.get("close", float("nan")))
            if not math.isfinite(close) or close <= 0.0:
                raise ValueError(
                    f"non-positive or non-finite close for "
                    f"{observation.symbol} on {rendered_session}"
                )
            closes[observation.symbol] = close
        missing = [symbol for symbol in symbols if symbol not in closes]
        if missing:
            raise ValueError(f"missing closes on {rendered_session} for: {missing}")
        rows.append([closes[symbol] for symbol in symbols])
    return np.asarray(rows, dtype=np.float64)


def build_history_t2_v2_dataset(
    *,
    history_adapter: StockBenchAdapter,
    session_closes: Sequence[datetime],
    symbols: Sequence[str] | np.ndarray,
    train_start: date | str,
    train_end: date | str,
    valid_start: date | str,
    valid_end: date | str,
) -> T2V2Dataset:
    """Build the frozen pooled stock/session probe from causal history snapshots.

    A feature row at close ``t`` uses returns ending no later than ``t``. Its
    shifted target is the same stock's direction from ``t+1`` to ``t+2``.
    Rows whose target would cross the Train or Valid boundary are excluded.
    """

    normalized_closes, normalized_sessions = _normalize_session_closes(session_closes)
    normalized_symbols = _as_symbols(symbols, name="symbols")
    if len(np.unique(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("symbols must be unique")
    boundaries = {
        "train_start": _as_boundary_date(train_start, name="train_start"),
        "train_end": _as_boundary_date(train_end, name="train_end"),
        "valid_start": _as_boundary_date(valid_start, name="valid_start"),
        "valid_end": _as_boundary_date(valid_end, name="valid_end"),
    }
    if not (
        boundaries["train_start"]
        <= boundaries["train_end"]
        < boundaries["valid_start"]
        <= boundaries["valid_end"]
    ):
        raise ValueError("Train and Valid date boundaries must be ordered and disjoint")
    for name, boundary in boundaries.items():
        if not np.any(normalized_sessions == boundary):
            raise ValueError(f"{name} is absent from session_closes")

    closes = _causal_close_matrix(
        history_adapter=history_adapter,
        session_closes=normalized_closes,
        sessions=normalized_sessions,
        symbols=normalized_symbols,
    )
    stock_returns = closes[1:] / closes[:-1] - 1.0
    if not np.all(np.isfinite(stock_returns)):
        raise ValueError("history close returns must be finite")

    def build_split(*, start: np.datetime64, end: np.datetime64) -> PooledDirectionSplit:
        split_positions = np.flatnonzero(
            (normalized_sessions >= start) & (normalized_sessions <= end)
        )
        if not len(split_positions):
            raise ValueError("T2 v2 split contains no sessions")
        first_position = int(split_positions[0])
        last_position = int(split_positions[-1])
        first_feature_position = max(
            first_position, T2_V2_ROLLING_WINDOW_SESSIONS
        )
        # ``range`` stops before last_position - 1, so t+2 remains in split.
        feature_positions = range(first_feature_position, last_position - 1)
        row_sessions: list[np.datetime64] = []
        row_symbols: list[str] = []
        feature_rows: list[list[float]] = []
        shifted_directions: list[int] = []
        for feature_position in feature_positions:
            current_returns = stock_returns[feature_position - 1]
            rolling_returns = stock_returns[
                feature_position
                - T2_V2_ROLLING_WINDOW_SESSIONS : feature_position
            ]
            shifted_returns = stock_returns[feature_position + 1]
            dispersion = float(current_returns.std(ddof=0))
            positive_fraction = float((current_returns > 0.0).mean())
            for symbol_position, symbol in enumerate(normalized_symbols):
                row_sessions.append(normalized_sessions[feature_position])
                row_symbols.append(symbol)
                stock_window = rolling_returns[:, symbol_position]
                feature_rows.append(
                    [
                        float(current_returns[symbol_position]),
                        float(stock_window.mean()),
                        float(stock_window.std(ddof=0)),
                        dispersion,
                        positive_fraction,
                    ]
                )
                shifted_directions.append(
                    int(shifted_returns[symbol_position] >= 0.0)
                )
        if not feature_rows:
            raise ValueError(
                "T2 v2 split lacks enough sessions for 20-session features "
                "and an in-split shifted target"
            )
        return PooledDirectionSplit(
            sessions=np.asarray(row_sessions, dtype="datetime64[D]"),
            symbols=np.asarray(row_symbols),
            features=np.asarray(feature_rows, dtype=np.float64),
            shifted_directions=np.asarray(shifted_directions, dtype=np.int8),
        )

    return T2V2Dataset(
        train=build_split(
            start=boundaries["train_start"], end=boundaries["train_end"]
        ),
        valid=build_split(
            start=boundaries["valid_start"], end=boundaries["valid_end"]
        ),
    )


def balanced_accuracy(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> float:
    """Return the mean recall of the two frozen direction classes."""

    truth = _as_binary_vector(y_true, name="y_true")
    predictions = _as_binary_vector(y_pred, name="y_pred")
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have equal length")
    if not np.any(truth == 0) or not np.any(truth == 1):
        raise ValueError("balanced accuracy requires both direction classes")
    negative_recall = float(np.mean(predictions[truth == 0] == 0))
    positive_recall = float(np.mean(predictions[truth == 1] == 1))
    return (negative_recall + positive_recall) / 2.0


@dataclass(frozen=True, slots=True)
class RidgeDirectionModel:
    """A ridge direction probe carrying immutable Train-only scaling state."""

    feature_means: np.ndarray
    feature_scales: np.ndarray
    coefficients: np.ndarray
    ridge_penalty: float = RIDGE_PENALTY

    def __post_init__(self) -> None:
        means = np.asarray(self.feature_means, dtype=np.float64)
        scales = np.asarray(self.feature_scales, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if means.ndim != 1 or not len(means):
            raise ValueError("feature_means must be a non-empty vector")
        if scales.shape != means.shape or np.any(scales <= 0.0):
            raise ValueError("feature_scales must be positive and align with feature_means")
        if coefficients.shape != (len(means) + 1,):
            raise ValueError("coefficients must contain one intercept plus each feature weight")
        if not (
            np.all(np.isfinite(means))
            and np.all(np.isfinite(scales))
            and np.all(np.isfinite(coefficients))
        ):
            raise ValueError("ridge model state must be finite")
        if not math.isfinite(self.ridge_penalty) or self.ridge_penalty <= 0.0:
            raise ValueError("ridge_penalty must be finite and positive")
        object.__setattr__(self, "feature_means", _readonly_copy(means))
        object.__setattr__(self, "feature_scales", _readonly_copy(scales))
        object.__setattr__(self, "coefficients", _readonly_copy(coefficients))

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        matrix = _as_feature_matrix(features)
        if matrix.shape[1] != len(self.feature_means):
            raise ValueError("prediction feature width does not match the fitted model")
        standardized = (matrix - self.feature_means) / self.feature_scales
        design = np.column_stack((np.ones(len(standardized)), standardized))
        return (design @ self.coefficients >= 0.0).astype(np.int8)


def fit_ridge_direction_model(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    ridge_penalty: float = RIDGE_PENALTY,
) -> RidgeDirectionModel:
    """Fit the frozen linear direction probe and its scaler on Train only."""

    matrix = _as_feature_matrix(features)
    directions = _as_binary_vector(labels, name="labels")
    if len(matrix) != len(directions):
        raise ValueError("features and labels must have equal length")
    if not np.any(directions == 0) or not np.any(directions == 1):
        raise ValueError("ridge direction fitting requires both direction classes")
    if (
        isinstance(ridge_penalty, bool)
        or not isinstance(ridge_penalty, Real)
        or not math.isfinite(float(ridge_penalty))
        or ridge_penalty <= 0.0
    ):
        raise ValueError("ridge_penalty must be finite and positive")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    standardized = (matrix - means) / scales
    design = np.column_stack((np.ones(len(standardized)), standardized))
    signed_directions = directions.astype(np.float64) * 2.0 - 1.0
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_penalty)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ signed_directions
    )
    return RidgeDirectionModel(
        feature_means=means,
        feature_scales=scales,
        coefficients=coefficients,
        ridge_penalty=float(ridge_penalty),
    )


def deterministic_weak_control_flip_mask(
    sessions: Sequence[object] | np.ndarray,
    symbols: Sequence[str] | np.ndarray,
    *,
    probability: float = WEAK_CONTROL_FLIP_PROBABILITY,
    seed: int = DEFAULT_T2_V2_SEED,
) -> np.ndarray:
    """Apply the preregistered SHA-256 threshold independently to each row key."""

    normalized_sessions = _as_sessions(sessions, name="sessions")
    normalized_symbols = _as_symbols(symbols, name="symbols")
    if len(normalized_sessions) != len(normalized_symbols):
        raise ValueError("sessions and symbols must have equal length")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise ValueError("probability must be numeric")
    fraction = Fraction(str(float(probability)))
    if not 0 <= fraction <= 1:
        raise ValueError("probability must lie in [0, 1]")
    threshold = fraction.numerator * (1 << 256) // fraction.denominator
    mask = []
    for session, symbol in zip(normalized_sessions, normalized_symbols, strict=True):
        rendered_session = np.datetime_as_string(session, unit="D")
        key = f"{int(seed)}\t{rendered_session}\t{symbol}\n".encode()
        mask.append(int.from_bytes(sha256(key).digest(), "big") < threshold)
    return np.asarray(mask, dtype=np.bool_)


def moving_block_session_draws(
    sessions: Sequence[object] | np.ndarray,
    *,
    block_size_sessions: int = DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
    bootstrap_samples: int = DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_T2_V2_SEED,
) -> np.ndarray:
    """Draw ordered session positions in contiguous, non-circular moving blocks."""

    ordered_sessions = _as_sessions(sessions, name="sessions")
    if not np.all(ordered_sessions[1:] > ordered_sessions[:-1]):
        raise ValueError("bootstrap sessions must be unique and strictly increasing")
    for value, name in (
        (block_size_sessions, "block_size_sessions"),
        (bootstrap_samples, "bootstrap_samples"),
        (seed, "seed"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if block_size_sessions < 1 or block_size_sessions > len(ordered_sessions):
        raise ValueError("block_size_sessions must fit inside the session window")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    blocks_per_sample = math.ceil(len(ordered_sessions) / int(block_size_sessions))
    generator = np.random.default_rng(int(seed))
    starts = generator.integers(
        0,
        len(ordered_sessions) - int(block_size_sessions) + 1,
        size=(int(bootstrap_samples), blocks_per_sample),
    )
    offsets = np.arange(int(block_size_sessions))
    return (starts[..., None] + offsets).reshape(int(bootstrap_samples), -1)[
        :, : len(ordered_sessions)
    ]


@dataclass(frozen=True, slots=True)
class BalancedAccuracyEvidence:
    """Point estimate and preregistered one-sided bootstrap lower bound."""

    point_estimate: float
    lower_bound: float
    sample_count: int
    session_count: int
    baseline: float = DEFAULT_T2_V2_BASELINE
    block_size_sessions: int = DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS
    bootstrap_samples: int = DEFAULT_T2_V2_BOOTSTRAP_SAMPLES
    alpha: float = DEFAULT_T2_V2_ALPHA
    seed: int = DEFAULT_T2_V2_SEED

    @property
    def detected(self) -> bool:
        return self.lower_bound > self.baseline


def session_clustered_balanced_accuracy(
    *,
    sessions: Sequence[object] | np.ndarray,
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    block_size_sessions: int = DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
    bootstrap_samples: int = DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_T2_V2_SEED,
    alpha: float = DEFAULT_T2_V2_ALPHA,
    baseline: float = DEFAULT_T2_V2_BASELINE,
) -> BalancedAccuracyEvidence:
    """Bootstrap balanced accuracy by resampling whole stock/session clusters."""

    row_sessions = _as_sessions(sessions, name="sessions")
    truth = _as_binary_vector(y_true, name="y_true")
    predictions = _as_binary_vector(y_pred, name="y_pred")
    if len(row_sessions) != len(truth) or len(truth) != len(predictions):
        raise ValueError("sessions, y_true, and y_pred must have equal length")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, Real)
        or not 0.0 <= baseline <= 1.0
    ):
        raise ValueError("baseline must lie in [0, 1]")

    unique_sessions, inverse = np.unique(row_sessions, return_inverse=True)
    draws = moving_block_session_draws(
        unique_sessions,
        block_size_sessions=block_size_sessions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    session_count = len(unique_sessions)
    true_negative = np.bincount(
        inverse, weights=((truth == 0) & (predictions == 0)), minlength=session_count
    )
    false_positive = np.bincount(
        inverse, weights=((truth == 0) & (predictions == 1)), minlength=session_count
    )
    true_positive = np.bincount(
        inverse, weights=((truth == 1) & (predictions == 1)), minlength=session_count
    )
    false_negative = np.bincount(
        inverse, weights=((truth == 1) & (predictions == 0)), minlength=session_count
    )
    negative_count = (true_negative + false_positive)[draws].sum(axis=1)
    positive_count = (true_positive + false_negative)[draws].sum(axis=1)
    if np.any(negative_count == 0) or np.any(positive_count == 0):
        raise ValueError("every bootstrap sample must contain both direction classes")
    estimates = (
        true_negative[draws].sum(axis=1) / negative_count
        + true_positive[draws].sum(axis=1) / positive_count
    ) / 2.0
    lower_bound = float(np.quantile(estimates, float(alpha), method="linear"))
    return BalancedAccuracyEvidence(
        point_estimate=balanced_accuracy(truth, predictions),
        lower_bound=lower_bound,
        sample_count=len(truth),
        session_count=session_count,
        baseline=float(baseline),
        block_size_sessions=int(block_size_sessions),
        bootstrap_samples=int(bootstrap_samples),
        alpha=float(alpha),
        seed=int(seed),
    )


@dataclass(frozen=True, slots=True)
class T2V2ProbeResult:
    """The three preregistered detection arms and their joint verdict."""

    real_pipeline: BalancedAccuracyEvidence
    strong_positive_control: BalancedAccuracyEvidence
    weak_positive_control: BalancedAccuracyEvidence

    @property
    def passed(self) -> bool:
        return (
            self.strong_positive_control.detected
            and self.weak_positive_control.detected
            and not self.real_pipeline.detected
        )


def _fit_predict_arm(
    *,
    dataset: T2V2Dataset,
    train_control: np.ndarray | None = None,
    valid_control: np.ndarray | None = None,
) -> np.ndarray:
    train_features = dataset.train.features
    valid_features = dataset.valid.features
    if (train_control is None) != (valid_control is None):
        raise ValueError("Train and Valid controls must be supplied together")
    if train_control is not None and valid_control is not None:
        if len(train_control) != len(train_features) or len(valid_control) != len(
            valid_features
        ):
            raise ValueError("control features must align with their split")
        train_features = np.column_stack((train_features, train_control))
        valid_features = np.column_stack((valid_features, valid_control))
    model = fit_ridge_direction_model(
        train_features, dataset.train.shifted_directions
    )
    return model.predict(valid_features)


def run_t2_v2_probe(
    dataset: T2V2Dataset,
    *,
    block_size_sessions: int = DEFAULT_T2_V2_BLOCK_SIZE_SESSIONS,
    bootstrap_samples: int = DEFAULT_T2_V2_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_T2_V2_SEED,
    alpha: float = DEFAULT_T2_V2_ALPHA,
    baseline: float = DEFAULT_T2_V2_BASELINE,
) -> T2V2ProbeResult:
    """Run the real pipeline and both ephemeral preregistered positive controls."""

    real_predictions = _fit_predict_arm(dataset=dataset)
    strong_predictions = _fit_predict_arm(
        dataset=dataset,
        train_control=dataset.train.shifted_directions,
        valid_control=dataset.valid.shifted_directions,
    )
    train_flip_mask = deterministic_weak_control_flip_mask(
        dataset.train.sessions, dataset.train.symbols, seed=seed
    )
    valid_flip_mask = deterministic_weak_control_flip_mask(
        dataset.valid.sessions, dataset.valid.symbols, seed=seed
    )
    weak_predictions = _fit_predict_arm(
        dataset=dataset,
        train_control=np.bitwise_xor(
            dataset.train.shifted_directions, train_flip_mask.astype(np.int8)
        ),
        valid_control=np.bitwise_xor(
            dataset.valid.shifted_directions, valid_flip_mask.astype(np.int8)
        ),
    )

    def evidence(predictions: np.ndarray) -> BalancedAccuracyEvidence:
        return session_clustered_balanced_accuracy(
            sessions=dataset.valid.sessions,
            y_true=dataset.valid.shifted_directions,
            y_pred=predictions,
            block_size_sessions=block_size_sessions,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            alpha=alpha,
            baseline=baseline,
        )

    return T2V2ProbeResult(
        real_pipeline=evidence(real_predictions),
        strong_positive_control=evidence(strong_predictions),
        weak_positive_control=evidence(weak_predictions),
    )
