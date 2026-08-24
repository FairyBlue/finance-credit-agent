"""Pure transforms over already point-in-time-gated observation snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

import numpy as np

from .stockbench import ObservationSnapshot, StockBenchAdapter, _require_aware


class FullSampleFitError(ValueError):
    """Raised when a statistic fit window reaches the full dataset boundary."""


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Closed, timezone-aware window used to pre-register temporal boundaries."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _require_aware(self.start, name="window.start"))
        object.__setattr__(self, "end", _require_aware(self.end, name="window.end"))
        if self.start >= self.end:
            raise ValueError("window.start must be earlier than window.end")


@dataclass(frozen=True, slots=True)
class TrainWindowStandardizer:
    """Statistics fitted through the adapter's gated API and a strict train window."""

    means: Mapping[str, float]
    scales: Mapping[str, float]
    train_window: TimeWindow

    @classmethod
    def fit(
        cls,
        adapter: StockBenchAdapter,
        *,
        feature_fields: Sequence[str],
        train_window: TimeWindow,
        dataset_window: TimeWindow,
    ) -> TrainWindowStandardizer:
        """Fit only inside an explicit train window that ends before the dataset does."""

        if not feature_fields:
            raise ValueError("at least one feature field is required")
        if train_window.start < dataset_window.start or train_window.end > dataset_window.end:
            raise ValueError("train_window must be contained by dataset_window")
        if train_window.start == dataset_window.start and train_window.end == dataset_window.end:
            raise FullSampleFitError(
                "full-sample fitting is forbidden; provide a strict train window"
            )
        if train_window.end >= dataset_window.end:
            raise FullSampleFitError("train_window must end before validation/test data")

        snapshot = adapter.get_observations(train_window.end)
        rows = [
            observation
            for observation in snapshot
            if observation.kind == "market" and observation.available_at >= train_window.start
        ]
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for field in feature_fields:
            values = np.asarray(
                [float(row.payload[field]) for row in rows if field in row.payload],
                dtype=np.float64,
            )
            if values.size == 0:
                raise ValueError(f"no train observations contain feature {field!r}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"feature {field!r} contains non-finite train values")
            means[field] = float(values.mean())
            scale = float(values.std(ddof=0))
            scales[field] = scale if scale > 0.0 else 1.0

        # TODO(human): line-review the standardization window before any experiment.
        return cls(
            means=MappingProxyType(means),
            scales=MappingProxyType(scales),
            train_window=train_window,
        )

    def transform(self, snapshot: ObservationSnapshot) -> np.ndarray:
        """Apply train-only statistics to an already gated observation snapshot."""

        fields = tuple(self.means)
        rows: list[list[float]] = []
        for observation in snapshot:
            if observation.kind != "market":
                continue
            missing = [field for field in fields if field not in observation.payload]
            if missing:
                raise ValueError(
                    f"observation {observation.observation_id!r} lacks features: {missing}"
                )
            rows.append(
                [
                    (float(observation.payload[field]) - self.means[field])
                    / self.scales[field]
                    for field in fields
                ]
            )
        values = np.asarray(rows, dtype=np.float64)
        if values.size and not np.all(np.isfinite(values)):
            raise ValueError("transformed features contain non-finite values")
        return values.reshape((-1, len(fields)))


def numeric_feature_bytes(snapshot: ObservationSnapshot, *, fields: Sequence[str]) -> bytes:
    """Materialize market fields as deterministic little-endian float64 bytes."""

    if not fields:
        raise ValueError("at least one feature field is required")
    values: list[list[float]] = []
    for observation in snapshot:
        if observation.kind != "market":
            continue
        values.append([float(observation.payload.get(field, np.nan)) for field in fields])
    return np.asarray(values, dtype="<f8").tobytes(order="C")
