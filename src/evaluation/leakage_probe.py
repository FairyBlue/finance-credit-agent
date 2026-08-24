"""Deterministic out-of-sample probe for the T2 label-shift leakage check."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral

import numpy as np

from src.data import StockBenchAdapter

DEFAULT_BOOTSTRAP_SEED = 1729


@dataclass(frozen=True, slots=True)
class DirectionProbeDataset:
    """Causal equal-weight portfolio features with frozen split boundaries."""

    sessions: np.ndarray
    features: np.ndarray
    next_session_directions: np.ndarray
    train_end: int
    test_start: int
    feature_names: tuple[str, ...]


def _close_matrix(
    *,
    adapter: StockBenchAdapter,
    session_closes: Sequence[datetime],
    symbols: tuple[str, ...],
) -> np.ndarray:
    rows: list[list[float]] = []
    for session_close in session_closes:
        expected_date = session_close.date().isoformat()
        prices: dict[str, float] = {}
        for observation in adapter.get_observations(session_close):
            if (
                observation.kind == "market"
                and observation.symbol in symbols
                and observation.payload.get("date") == expected_date
                and "close" in observation.payload
            ):
                if observation.symbol in prices:
                    raise ValueError(
                        f"duplicate close for {observation.symbol} on {expected_date}"
                    )
                prices[observation.symbol] = float(observation.payload["close"])
        missing = [symbol for symbol in symbols if symbol not in prices]
        if missing:
            raise ValueError(f"missing closes on {expected_date} for: {missing}")
        row = [prices[symbol] for symbol in symbols]
        if any(not math.isfinite(value) or value <= 0.0 for value in row):
            raise ValueError(f"non-positive or non-finite close on {expected_date}")
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def build_equal_weight_direction_probe_dataset(
    *,
    history_adapter: StockBenchAdapter,
    oos_adapter: StockBenchAdapter,
    history_session_closes: Sequence[datetime],
    oos_session_closes: Sequence[datetime],
    symbols: Sequence[str],
    train_end: date,
    test_start: date,
    rolling_window: int = 20,
) -> DirectionProbeDataset:
    """Build T2 features through the sole ``get_observations(as_of)`` read API.

    Features at session ``t`` use only close returns available by ``t``. The
    target is the direction of the equal-weight cross-sectional return from
    ``t`` to the next XNYS session. T2 moves that target one further session
    ahead. The validation gap remains isolated between split boundaries.
    """

    normalized_symbols = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
        raise ValueError("symbols must contain at least one non-empty ticker")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least two sessions")
    history_closes = tuple(history_session_closes)
    oos_closes = tuple(oos_session_closes)
    all_sessions = history_closes + oos_closes
    if len(history_closes) < rolling_window + 2 or len(oos_closes) < 3:
        raise ValueError("probe needs sufficient history and at least three OOS sessions")
    if any(
        current >= following
        for current, following in zip(all_sessions, all_sessions[1:], strict=False)
    ):
        raise ValueError("probe session closes must be strictly increasing")

    closes = np.vstack(
        (
            _close_matrix(
                adapter=history_adapter,
                session_closes=history_closes,
                symbols=normalized_symbols,
            ),
            _close_matrix(
                adapter=oos_adapter,
                session_closes=oos_closes,
                symbols=normalized_symbols,
            ),
        )
    )
    stock_returns = closes[1:] / closes[:-1] - 1.0
    portfolio_returns = stock_returns.mean(axis=1)
    return_sessions = all_sessions[1:]
    feature_rows: list[list[float]] = []
    feature_sessions: list[str] = []
    directions: list[int] = []
    for index in range(rolling_window - 1, len(portfolio_returns) - 1):
        window = portfolio_returns[index - rolling_window + 1 : index + 1]
        cross_section = stock_returns[index]
        feature_rows.append(
            [
                float(portfolio_returns[index]),
                float(window.mean()),
                float(window.std(ddof=0)),
                float(cross_section.std(ddof=0)),
                float((cross_section > 0.0).mean()),
            ]
        )
        feature_sessions.append(return_sessions[index].date().isoformat())
        directions.append(int(portfolio_returns[index + 1] >= 0.0))

    probe_sessions = np.asarray(feature_sessions, dtype="datetime64[D]")
    train_boundary = int(np.searchsorted(probe_sessions, np.datetime64(train_end), side="right"))
    test_boundary = int(np.searchsorted(probe_sessions, np.datetime64(test_start), side="left"))
    if not 2 < train_boundary < test_boundary < len(probe_sessions) - 1:
        raise ValueError("frozen dates do not define usable train, validation, and OOS probe sets")
    return DirectionProbeDataset(
        sessions=probe_sessions,
        features=np.asarray(feature_rows, dtype=np.float64),
        next_session_directions=np.asarray(directions, dtype=np.int8),
        train_end=train_boundary,
        test_start=test_boundary,
        feature_names=(
            "portfolio_return_1d",
            f"portfolio_return_mean_{rolling_window}d",
            f"portfolio_return_volatility_{rolling_window}d",
            "cross_sectional_dispersion_1d",
            "cross_sectional_positive_fraction_1d",
        ),
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile moving-block bootstrap interval for a scalar difference."""

    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class TemporalLeakageProbeResult:
    """Observable T2 outcomes on the common out-of-sample session set."""

    aligned_accuracy: float
    shifted_accuracy: float
    aligned_minus_chance_interval: BootstrapInterval
    shifted_minus_chance_interval: BootstrapInterval
    aligned_minus_shifted_interval: BootstrapInterval
    test_samples: int
    block_size: int
    alpha: float
    seed: int

    @property
    def aligned_significantly_above_chance(self) -> bool:
        return self.aligned_minus_chance_interval.lower > 0.0

    @property
    def shifted_consistent_with_chance(self) -> bool:
        return self.shifted_minus_chance_interval.contains(0.0)

    @property
    def aligned_significantly_above_shifted(self) -> bool:
        return self.aligned_minus_shifted_interval.lower > 0.0


def _as_feature_matrix(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty one- or two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix


def _fit_direction_probe(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    signed_labels = labels * 2.0 - 1.0
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.eye(design.shape[1], dtype=np.float64) * 1e-6
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ signed_labels)


def _predict_direction(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(features)), features))
    return (design @ coefficients >= 0.0).astype(np.int8)


def _moving_block_indices(
    *, sample_count: int, block_size: int, bootstrap_samples: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    blocks_per_sample = math.ceil(sample_count / block_size)
    starts = generator.integers(
        0,
        sample_count - block_size + 1,
        size=(bootstrap_samples, blocks_per_sample),
    )
    offsets = np.arange(block_size)
    return (starts[..., None] + offsets).reshape(bootstrap_samples, -1)[:, :sample_count]


def _bootstrap_interval(
    values: np.ndarray, indices: np.ndarray, *, alpha: float
) -> BootstrapInterval:
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    return BootstrapInterval(lower=float(lower), upper=float(upper))


def run_temporal_leakage_probe(
    *,
    sessions: Sequence[object] | np.ndarray,
    features: Sequence[Sequence[float]] | np.ndarray,
    next_session_directions: Sequence[int] | np.ndarray,
    train_end: int,
    test_start: int,
    block_size: int = 5,
    alpha: float = 0.05,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = 2_000,
) -> TemporalLeakageProbeResult:
    """Compare aligned and one-session-lagged probes on a chronological OOS split.

    ``train_end`` is the exclusive training boundary and ``test_start`` is the
    inclusive test boundary; the intervening sessions are a reserved validation
    window. Both probes use identical feature rows and model code. Label shifting
    is performed separately inside train and test windows so it cannot cross a
    temporal boundary.
    """

    matrix = _as_feature_matrix(features)
    labels = np.asarray(next_session_directions)
    ordered_sessions = np.asarray(sessions)
    sample_count = len(matrix)
    if labels.ndim != 1 or ordered_sessions.ndim != 1:
        raise ValueError("sessions and next_session_directions must be one-dimensional")
    if len(labels) != sample_count or len(ordered_sessions) != sample_count:
        raise ValueError("sessions, features, and next_session_directions must have equal length")
    if sample_count < 4:
        raise ValueError("at least four chronological sessions are required")
    if not np.all(ordered_sessions[1:] > ordered_sessions[:-1]):
        raise ValueError("sessions must be strictly increasing")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("next_session_directions must contain only binary 0/1 labels")
    labels = labels.astype(np.int8, copy=False)
    if isinstance(train_end, bool) or not isinstance(train_end, Integral):
        raise ValueError("train_end must be an integer boundary")
    if isinstance(test_start, bool) or not isinstance(test_start, Integral):
        raise ValueError("test_start must be an integer boundary")
    if not 2 < train_end < test_start < sample_count - 1:
        raise ValueError("boundaries must define non-empty train, validation, and test windows")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    # "Shift one session later" means X_t is paired with y_{t+2}, not the
    # already-known return ending at t. Keep common rows within each partition.
    train_features = matrix[: train_end - 2]
    test_features = matrix[test_start:-1]
    if block_size > len(test_features):
        raise ValueError("block_size cannot exceed the common OOS sample count")

    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    standardized_train = (train_features - means) / scales
    standardized_test = (test_features - means) / scales

    aligned_coefficients = _fit_direction_probe(standardized_train, labels[: train_end - 2])
    shifted_coefficients = _fit_direction_probe(standardized_train, labels[1 : train_end - 1])
    aligned_predictions = _predict_direction(standardized_test, aligned_coefficients)
    shifted_predictions = _predict_direction(standardized_test, shifted_coefficients)
    aligned_hits = (aligned_predictions == labels[test_start:-1]).astype(np.float64)
    shifted_hits = (shifted_predictions == labels[test_start + 1 :]).astype(np.float64)

    indices = _moving_block_indices(
        sample_count=len(test_features),
        block_size=int(block_size),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return TemporalLeakageProbeResult(
        aligned_accuracy=float(aligned_hits.mean()),
        shifted_accuracy=float(shifted_hits.mean()),
        aligned_minus_chance_interval=_bootstrap_interval(aligned_hits - 0.5, indices, alpha=alpha),
        shifted_minus_chance_interval=_bootstrap_interval(shifted_hits - 0.5, indices, alpha=alpha),
        aligned_minus_shifted_interval=_bootstrap_interval(
            aligned_hits - shifted_hits, indices, alpha=alpha
        ),
        test_samples=len(test_features),
        block_size=int(block_size),
        alpha=float(alpha),
        seed=int(seed),
    )
