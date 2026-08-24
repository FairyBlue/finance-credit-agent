"""Pre-registered temporal leakage checks from EVAL_PROTOCOL.md section 6."""

import json
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    AvailabilityPolicy,
    FullSampleFitError,
    MarketObservation,
    StockBenchAdapter,
    TimeWindow,
    TrainWindowStandardizer,
    numeric_feature_bytes,
)
from src.data.stockbench import ObservationKind, dataframe_content_sha256
from src.evaluation import (
    build_equal_weight_direction_probe_dataset,
    run_temporal_leakage_probe,
)

UTC = timezone.utc


def _observation(
    observation_id: str,
    *,
    kind: ObservationKind,
    available_at: datetime,
    close: float | None = None,
    text: str | None = None,
) -> MarketObservation:
    payload: dict[str, float | str] = {}
    if close is not None:
        payload["close"] = close
    if text is not None:
        payload["text"] = text
    return MarketObservation(
        observation_id=observation_id,
        kind=kind,
        symbol="AAA",
        event_at=available_at,
        available_at=available_at,
        payload=payload,
    )


def test_t1_future_price_corruption_cannot_change_train_snapshot_bits() -> None:
    as_of = datetime(2025, 1, 2, 23, 59, tzinfo=UTC)
    train_price = _observation(
        "price-train", kind="market", available_at=datetime(2025, 1, 2, tzinfo=UTC), close=101.25
    )
    original_future = _observation(
        "price-test", kind="market", available_at=datetime(2025, 2, 2, tzinfo=UTC), close=105.0
    )
    corrupted_future = _observation(
        "price-test", kind="market", available_at=datetime(2025, 2, 2, tzinfo=UTC), close=-999999.0
    )

    original = StockBenchAdapter([train_price, original_future], data_version="fixture-v1")
    corrupted = StockBenchAdapter(
        [train_price, corrupted_future], data_version="fixture-v1-corrupt"
    )

    original_snapshot = original.get_observations(as_of)
    corrupted_snapshot = corrupted.get_observations(as_of)
    assert original_snapshot.canonical_bytes() == corrupted_snapshot.canonical_bytes()
    assert numeric_feature_bytes(original_snapshot, fields=("close",)) == numeric_feature_bytes(
        corrupted_snapshot, fields=("close",)
    )


def test_t3_standardizer_fits_only_the_explicit_train_window() -> None:
    train_start = datetime(2025, 1, 1, tzinfo=UTC)
    train_end = datetime(2025, 1, 2, 23, 59, tzinfo=UTC)
    test_end = datetime(2025, 2, 28, tzinfo=UTC)
    adapter = StockBenchAdapter(
        [
            _observation("p1", kind="market", available_at=train_start, close=10.0),
            _observation(
                "p2", kind="market", available_at=datetime(2025, 1, 2, tzinfo=UTC), close=20.0
            ),
            _observation(
                "future-outlier",
                kind="market",
                available_at=datetime(2025, 2, 1, tzinfo=UTC),
                close=1000000.0,
            ),
        ],
        data_version="fixture-v1",
    )

    scaler = TrainWindowStandardizer.fit(
        adapter,
        feature_fields=("close",),
        train_window=TimeWindow(start=train_start, end=train_end),
        dataset_window=TimeWindow(start=train_start, end=test_end),
    )

    assert scaler.means == {"close": 15.0}
    assert scaler.scales == {"close": 5.0}
    transformed = scaler.transform(adapter.get_observations(test_end))
    assert transformed[:2, 0].tolist() == pytest.approx([-1.0, 1.0])


def test_t3_full_sample_fit_is_rejected() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 28, tzinfo=UTC)
    adapter = StockBenchAdapter(
        [_observation("p1", kind="market", available_at=start, close=10.0)],
        data_version="fixture-v1",
    )
    full_window = TimeWindow(start=start, end=end)

    with pytest.raises(FullSampleFitError, match="full-sample fitting is forbidden"):
        TrainWindowStandardizer.fit(
            adapter,
            feature_fields=("close",),
            train_window=full_window,
            dataset_window=full_window,
        )


def test_t4_deleting_future_fundamental_cannot_change_historical_retrieval() -> None:
    as_of = datetime(2025, 1, 10, 23, 59, tzinfo=UTC)
    historical_fundamental = _observation(
        "fundamental-history",
        kind="fundamental",
        available_at=datetime(2025, 1, 10, 12, 0, tzinfo=UTC),
        text="Known at the query time",
    )
    future_fundamental = _observation(
        "fundamental-future",
        kind="fundamental",
        available_at=datetime(2025, 1, 11, 12, 0, tzinfo=UTC),
        text="Not known at the query time",
    )

    with_future = StockBenchAdapter(
        [historical_fundamental, future_fundamental], data_version="fixture-with-future"
    )
    without_future = StockBenchAdapter([historical_fundamental], data_version="fixture-no-future")

    assert (
        with_future.get_observations(as_of).canonical_bytes()
        == without_future.get_observations(as_of).canonical_bytes()
    )


def test_stockbench_v1_ignores_news_and_assigns_calendar_close_availability(
    tmp_path: Path,
) -> None:
    market_dir = tmp_path / "parquet" / "AAA" / "day"
    news_dir = tmp_path / "cache" / "news_by_day" / "AAA"
    market_dir.mkdir(parents=True)
    news_dir.mkdir(parents=True)
    parquet_path = market_dir / "2025-01-02.parquet"
    market_frame = pd.DataFrame(
        {
            "date": [date(2025, 1, 2)],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000.0],
            "vwap": [100.5],
        }
    )
    market_frame.to_parquet(parquet_path, index=False)
    parquet_path.with_suffix(".parquet.sha256").write_text(
        dataframe_content_sha256(
            market_frame,
            sort_by=[column for column in market_frame.columns if column != "date"],
        ),
        encoding="utf-8",
    )
    publication = datetime(2025, 1, 2, 15, 0, tzinfo=UTC)
    (news_dir / "2025-01-02.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": 7,
                        "datetime": int(publication.timestamp()),
                        "published_utc": "2025-01-02 15:00:00",
                        "title": "Visible before close",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = StockBenchAdapter.from_storage(
        tmp_path,
        symbols=["AAA"],
        data_version="fixture-tree-sha",
        availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
    )

    before_close = adapter.get_observations(datetime(2025, 1, 2, 20, 0, tzinfo=UTC))
    at_close = adapter.get_observations(datetime(2025, 1, 2, 21, 0, tzinfo=UTC))
    assert list(before_close) == []
    assert [item.kind for item in at_close] == ["market"]


def test_t2_shifted_labels_significantly_reduce_oos_performance() -> None:
    """A one-session label shift must destroy a real OOS directional signal."""

    rng = np.random.default_rng(20250824)
    sample_count = 360
    directions = rng.integers(0, 2, size=sample_count)
    signed_direction = directions * 2.0 - 1.0
    features = np.column_stack(
        (
            signed_direction + rng.normal(0.0, 0.08, size=sample_count),
            rng.normal(0.0, 1.0, size=sample_count),
        )
    )
    sessions = np.datetime64("2024-01-01") + np.arange(sample_count)

    result = run_temporal_leakage_probe(
        sessions=sessions,
        features=features,
        next_session_directions=directions,
        train_end=180,
        test_start=240,
    )

    assert result.aligned_accuracy > 0.95
    assert 0.4 <= result.shifted_accuracy <= 0.6
    assert result.aligned_significantly_above_chance
    assert result.aligned_significantly_above_shifted
    assert result.shifted_consistent_with_chance
    assert result == run_temporal_leakage_probe(
        sessions=sessions,
        features=features,
        next_session_directions=directions,
        train_end=180,
        test_start=240,
    )


def test_t2_probe_rejects_misaligned_input_lengths() -> None:
    with pytest.raises(ValueError, match="must have equal length"):
        run_temporal_leakage_probe(
            sessions=np.arange(12),
            features=np.ones((12, 1)),
            next_session_directions=np.zeros(11, dtype=int),
            train_end=6,
            test_start=9,
        )


@pytest.mark.parametrize(
    ("train_end", "test_start"),
    [(2, 8), (6, 6), (8, 7), (6, 11)],
)
def test_t2_probe_rejects_invalid_temporal_boundaries(train_end: int, test_start: int) -> None:
    with pytest.raises(ValueError, match="boundaries must define"):
        run_temporal_leakage_probe(
            sessions=np.arange(12),
            features=np.arange(12, dtype=float),
            next_session_directions=np.arange(12) % 2,
            train_end=train_end,
            test_start=test_start,
        )


def test_t2_probe_rejects_non_chronological_sessions() -> None:
    sessions = np.arange(12)
    sessions[5], sessions[6] = sessions[6], sessions[5]

    with pytest.raises(ValueError, match="sessions must be strictly increasing"):
        run_temporal_leakage_probe(
            sessions=sessions,
            features=np.arange(12, dtype=float),
            next_session_directions=np.arange(12) % 2,
            train_end=6,
            test_start=9,
        )


def test_t2_shift_never_uses_a_label_at_or_beyond_the_train_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both fits use common X rows whose targets remain strictly in train."""

    import src.evaluation.leakage_probe as probe_module

    captured_labels: list[np.ndarray] = []

    def capture_fit(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        captured_labels.append(labels.copy())
        return np.zeros(features.shape[1] + 1)

    monkeypatch.setattr(probe_module, "_fit_direction_probe", capture_fit)
    labels = np.asarray([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    probe_module.run_temporal_leakage_probe(
        sessions=np.arange(12),
        features=np.arange(12, dtype=float),
        next_session_directions=labels,
        train_end=8,
        test_start=9,
        block_size=1,
        bootstrap_samples=10,
    )

    assert np.array_equal(captured_labels[0], labels[:6])
    assert np.array_equal(captured_labels[1], labels[1:7])


def test_t2_dataset_uses_only_asof_closes_and_frozen_temporal_boundaries() -> None:
    sessions = [datetime(2025, 1, 1 + offset, 21, 0, tzinfo=UTC) for offset in range(30)]
    bars = []
    close = 100.0
    for index, session in enumerate(sessions):
        close *= 1.01 if index % 3 else 0.995
        bars.append(
            MarketObservation(
                observation_id=f"bar-{index}",
                kind="market",
                symbol="AAA",
                event_at=session,
                available_at=session,
                payload={"date": session.date().isoformat(), "close": close},
            )
        )
    history = StockBenchAdapter(bars[:20], data_version="history-fixture")
    oos = StockBenchAdapter(bars[20:], data_version="oos-fixture")

    dataset = build_equal_weight_direction_probe_dataset(
        history_adapter=history,
        oos_adapter=oos,
        history_session_closes=sessions[:20],
        oos_session_closes=sessions[20:],
        symbols=["AAA"],
        train_end=sessions[11].date(),
        test_start=sessions[20].date(),
        rolling_window=5,
    )

    assert dataset.feature_names == (
        "portfolio_return_1d",
        "portfolio_return_mean_5d",
        "portfolio_return_volatility_5d",
        "cross_sectional_dispersion_1d",
        "cross_sectional_positive_fraction_1d",
    )
    assert np.all(np.isfinite(dataset.features))
    assert dataset.train_end > 2
    assert dataset.test_start > dataset.train_end
    assert dataset.sessions[dataset.test_start] == np.datetime64("2025-01-21")
