"""Pre-registered temporal leakage checks from EVAL_PROTOCOL.md section 6."""

import json
from datetime import date, datetime, time, timezone
from hashlib import sha256
from pathlib import Path

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

UTC = timezone.utc


def _observation(
    observation_id: str,
    *,
    kind: str,
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


def test_t4_deleting_future_news_cannot_change_historical_retrieval() -> None:
    as_of = datetime(2025, 1, 10, 23, 59, tzinfo=UTC)
    historical_news = _observation(
        "news-history",
        kind="news",
        available_at=datetime(2025, 1, 10, 12, 0, tzinfo=UTC),
        text="Known at the query time",
    )
    future_news = _observation(
        "news-future",
        kind="news",
        available_at=datetime(2025, 1, 11, 12, 0, tzinfo=UTC),
        text="Not known at the query time",
    )

    with_future = StockBenchAdapter(
        [historical_news, future_news], data_version="fixture-with-future"
    )
    without_future = StockBenchAdapter([historical_news], data_version="fixture-no-future")

    assert with_future.get_observations(as_of).canonical_bytes() == without_future.get_observations(
        as_of
    ).canonical_bytes()


def test_stockbench_offline_layout_assigns_news_epoch_and_after_close_availability(
    tmp_path: Path,
) -> None:
    market_dir = tmp_path / "parquet" / "AAA" / "day"
    news_dir = tmp_path / "cache" / "news_by_day" / "AAA"
    market_dir.mkdir(parents=True)
    news_dir.mkdir(parents=True)
    parquet_path = market_dir / "2025-01-02.parquet"
    pd.DataFrame(
        {
            "date": [date(2025, 1, 2)],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1000.0],
            "vwap": [100.5],
        }
    ).to_parquet(parquet_path, index=False)
    parquet_path.with_suffix(".parquet.sha256").write_text(
        sha256(parquet_path.read_bytes()).hexdigest(), encoding="utf-8"
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
    assert [item.kind for item in before_close] == ["news"]
    assert [item.kind for item in at_close] == ["news", "market"]
    assert before_close[0].available_at == publication


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="T2 needs the frozen training pipeline, OOS metric, and significance threshold",
)
def test_t2_shifted_labels_significantly_reduce_oos_performance() -> None:
    """T2 skeleton: shifting labels by one period must significantly hurt OOS performance."""
    pytest.fail("TODO(human): freeze the OOS metric and significance threshold before enabling T2")
