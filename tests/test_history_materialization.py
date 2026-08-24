"""Historical train/valid data is frozen before it reaches the environment."""

from datetime import date, datetime, time, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import AvailabilityPolicy, MarketObservation, StockBenchAdapter
from src.data.history import (
    HistoryWindow,
    StooqDailyBarProvider,
    YFinanceDailyBarProvider,
    materialize_daily_history,
)
from src.data.stockbench import dataframe_content_sha256


class _FixtureProvider:
    name = "fixture"
    version = "fixture-v1"

    def fetch(self, symbol: str, *, start: date, end: date) -> pd.DataFrame:
        assert symbol == "AAA"
        assert (start, end) == (date(2025, 1, 2), date(2025, 1, 3))
        return pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        )


class _MissingSessionProvider(_FixtureProvider):
    def fetch(self, symbol: str, *, start: date, end: date) -> pd.DataFrame:
        return super().fetch(symbol, start=start, end=end).iloc[:1]


class _SecondSymbolFailsProvider(_FixtureProvider):
    def fetch(self, symbol: str, *, start: date, end: date) -> pd.DataFrame:
        if symbol == "BBB":
            raise RuntimeError("provider failed")
        return super().fetch(symbol, start=start, end=end)


def test_history_snapshot_is_complete_checksummed_and_asof_gated(tmp_path: Path) -> None:
    manifest = materialize_daily_history(
        provider=_FixtureProvider(),
        symbols=["AAA"],
        window=HistoryWindow(start=date(2025, 1, 2), end=date(2025, 1, 3)),
        output_root=tmp_path,
    )

    parquet_files = sorted(tmp_path.glob("parquet/AAA/day/*.parquet"))
    manifest_bytes = b"".join(
        f"{path.relative_to(tmp_path).as_posix()}\t{sha256(path.read_bytes()).hexdigest()}\n".encode()
        for path in parquet_files
    )
    assert len(parquet_files) == 2
    assert manifest.file_count == 2
    assert manifest.sessions_per_symbol == {"AAA": 2}
    assert manifest.input_manifest_sha256 == sha256(manifest_bytes).hexdigest()

    adapter = StockBenchAdapter.from_storage(
        tmp_path,
        symbols=["AAA"],
        data_version=f"fixture:{manifest.input_manifest_sha256}",
        availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
    )
    before_first_close = adapter.get_observations(datetime(2025, 1, 2, 20, 59, tzinfo=timezone.utc))
    at_first_close = adapter.get_observations(datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc))
    assert len(before_first_close) == 0
    assert [item.payload["close"] for item in at_first_close] == [101.0]


def test_stockbench_sidecar_verifies_logical_dataframe_content(tmp_path: Path) -> None:
    day_root = tmp_path / "parquet" / "AAPL" / "day"
    day_root.mkdir(parents=True)
    parquet_path = day_root / "2025-03-03.parquet"
    pd.DataFrame(
        {
            "date": [date(2025, 3, 3)],
            "open": [241.79],
            "high": [244.0272],
            "low": [236.112],
            "close": [238.03],
            "volume": [47183985.0],
            "vwap": [239.9029],
        }
    ).to_parquet(parquet_path, index=False)
    canonical = (
        b'[{"close":238.03,"date":"2025-03-03","high":244.0272,'
        b'"low":236.112,"open":241.79,"volume":47183985.0,"vwap":239.9029}]'
    )
    logical_hash = sha256(canonical).hexdigest()
    assert logical_hash != sha256(parquet_path.read_bytes()).hexdigest()
    parquet_path.with_suffix(".parquet.sha256").write_text(logical_hash, encoding="utf-8")

    adapter = StockBenchAdapter.from_storage(
        tmp_path,
        symbols=["AAPL"],
        data_version="stockbench-fixture-tree",
        availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
    )

    snapshot = adapter.get_observations(datetime(2025, 3, 3, 21, 0, tzinfo=timezone.utc))
    assert [item.payload["close"] for item in snapshot] == [238.03]


def test_history_snapshot_rejects_a_missing_xnys_session(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AAA history is missing XNYS sessions: 2025-01-03"):
        materialize_daily_history(
            provider=_MissingSessionProvider(),
            symbols=["AAA"],
            window=HistoryWindow(start=date(2025, 1, 2), end=date(2025, 1, 3)),
            output_root=tmp_path,
        )


def test_market_available_at_uses_the_xnys_early_close() -> None:
    policy = AvailabilityPolicy("America/New_York", time(16, 0))

    assert policy.market_available_at(date(2025, 7, 3)) == datetime(
        2025, 7, 3, 17, 0, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError, match="2025-07-04 is not an XNYS session"):
        policy.market_available_at(date(2025, 7, 4))


def test_availability_policy_rejects_calendar_timezone_or_regular_close_mismatch() -> None:
    with pytest.raises(ValueError, match="XNYS timezone is America/New_York"):
        AvailabilityPolicy("Asia/Tokyo", time(16, 0))
    with pytest.raises(ValueError, match="XNYS regular close is 16:00:00"):
        AvailabilityPolicy("America/New_York", time(15, 59))


def test_yfinance_provider_freezes_unadjusted_daily_prices_and_inclusive_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def download(symbol: str, **kwargs: object) -> pd.DataFrame:
        captured.update({"symbol": symbol, **kwargs})
        return pd.DataFrame({"Close": [101.0]}, index=pd.to_datetime(["2025-01-03"]))

    fake = SimpleNamespace(__version__="fixture-yf", download=download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake)

    provider = YFinanceDailyBarProvider(corporate_actions_through=date(2026, 8, 24))
    frame = provider.fetch("AAPL", start=date(2025, 1, 2), end=date(2025, 1, 3))

    assert provider.version == "fixture-yf"
    assert frame.iloc[0]["Close"] == 101.0
    assert captured == {
        "symbol": "AAPL",
        "start": "2025-01-02",
        "end": "2026-08-25",
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "prepost": False,
        "repair": False,
        "keepna": False,
        "progress": False,
        "threads": False,
        "multi_level_index": False,
    }


def test_yfinance_provider_reverses_splits_after_the_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def download(symbol: str, **kwargs: object) -> pd.DataFrame:
        assert symbol == "AAPL"
        return pd.DataFrame(
            {
                "Open": [24.0, 25.0, 102.0],
                "High": [26.0, 27.0, 104.0],
                "Low": [23.0, 24.0, 100.0],
                "Close": [25.0, 26.0, 103.0],
                "Volume": [400.0, 440.0, 120.0],
                "Dividends": [0.0, 0.0, 0.0],
                "Stock Splits": [0.0, 0.0, 4.0],
            },
            index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03"]),
        )

    fake = SimpleNamespace(__version__="fixture-yf", download=download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake)

    provider = YFinanceDailyBarProvider(corporate_actions_through=date(2025, 2, 3))
    frame = provider.fetch("AAPL", start=date(2025, 1, 2), end=date(2025, 1, 3))

    assert frame["Close"].tolist() == [100.0, 104.0]
    assert frame["Volume"].tolist() == [100.0, 110.0]


def test_stooq_provider_uses_the_frozen_us_daily_csv_query() -> None:
    captured: list[str] = []

    def get_bytes(url: str) -> bytes:
        captured.append(url)
        return (
            b"Date,Open,High,Low,Close,Volume\n"
            b"2025-01-03,101,103,100,102,1100\n"
            b"2025-01-02,100,102,99,101,1000\n"
        )

    provider = StooqDailyBarProvider(http_get=get_bytes)
    frame = provider.fetch("AAPL", start=date(2025, 1, 2), end=date(2025, 1, 3))

    assert captured == ["https://stooq.com/q/d/l/?s=aapl.us&d1=20250102&d2=20250103&i=d"]
    assert frame.index.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-03"]
    assert frame["Close"].tolist() == [101, 102]


def test_technical_return_is_available_only_after_its_session_close(tmp_path: Path) -> None:
    manifest = materialize_daily_history(
        provider=_FixtureProvider(),
        symbols=["AAA"],
        window=HistoryWindow(start=date(2025, 1, 2), end=date(2025, 1, 3)),
        output_root=tmp_path,
    )
    adapter = StockBenchAdapter.from_storage(
        tmp_path,
        symbols=["AAA"],
        data_version=f"fixture:{manifest.input_manifest_sha256}",
        availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
    )

    first_close = adapter.get_observations(datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc))
    second_close = adapter.get_observations(datetime(2025, 1, 3, 21, 0, tzinfo=timezone.utc))

    assert "technical_return_1d" not in first_close[-1].payload
    assert second_close[-1].payload["technical_return_1d"] == pytest.approx(102.0 / 101.0 - 1.0)


def test_adapter_ignores_source_technical_columns_and_derives_returns_causally(
    tmp_path: Path,
) -> None:
    day_root = tmp_path / "parquet" / "AAA" / "day"
    day_root.mkdir(parents=True)
    for trading_date, close in ((date(2025, 1, 2), 101.0), (date(2025, 1, 3), 102.0)):
        frame = pd.DataFrame(
            [
                {
                    "date": trading_date,
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 2.0,
                    "close": close,
                    "volume": 1_000.0,
                    "technical_return_1d": 999.0,
                    "technical_future_return": -999.0,
                }
            ]
        )
        path = day_root / f"{trading_date.isoformat()}.parquet"
        frame.to_parquet(path, index=False)
        path.with_suffix(".parquet.sha256").write_text(
            dataframe_content_sha256(
                frame,
                sort_by=[column for column in frame.columns if column != "date"],
            ),
            encoding="utf-8",
        )

    adapter = StockBenchAdapter.from_storage(
        tmp_path,
        symbols=["AAA"],
        data_version="fixture-causal-technical",
        availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
    )

    first = adapter.get_observations(datetime(2025, 1, 2, 21, 0, tzinfo=timezone.utc))[-1]
    second = adapter.get_observations(datetime(2025, 1, 3, 21, 0, tzinfo=timezone.utc))[-1]
    assert "technical_return_1d" not in first.payload
    assert "technical_future_return" not in first.payload
    assert second.payload["technical_return_1d"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert "technical_future_return" not in second.payload


def test_v1_rejects_news_observations() -> None:
    with pytest.raises(ValueError, match="unsupported observation kind: news"):
        MarketObservation(
            observation_id="news:future-phase",
            kind="news",  # type: ignore[arg-type]
            symbol="AAPL",
            event_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            available_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            payload={"title": "Phase 2 only"},
        )


def test_history_manifest_records_a_timezone_aware_retrieval_time(tmp_path: Path) -> None:
    manifest = materialize_daily_history(
        provider=_FixtureProvider(),
        symbols=["AAA"],
        window=HistoryWindow(start=date(2025, 1, 2), end=date(2025, 1, 3)),
        output_root=tmp_path,
    )

    retrieved_at = datetime.fromisoformat(manifest.retrieved_at_utc)
    assert retrieved_at.tzinfo == timezone.utc
    assert manifest.provider_settings == {}


def test_history_materialization_refuses_to_mix_with_an_existing_snapshot(
    tmp_path: Path,
) -> None:
    (tmp_path / "stale-file").write_text("old snapshot", encoding="utf-8")

    with pytest.raises(ValueError, match="history output_root must be empty"):
        materialize_daily_history(
            provider=_FixtureProvider(),
            symbols=["AAA"],
            window=HistoryWindow(start=date(2025, 1, 2), end=date(2025, 1, 3)),
            output_root=tmp_path,
        )


def test_failed_history_download_does_not_leave_a_partial_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "snapshot"

    with pytest.raises(RuntimeError, match="provider failed"):
        materialize_daily_history(
            provider=_SecondSymbolFailsProvider(),
            symbols=["AAA", "BBB"],
            window=HistoryWindow(start=date(2025, 1, 2), end=date(2025, 1, 3)),
            output_root=output,
        )

    assert not output.exists()


def test_storage_ingestion_reads_only_the_frozen_input_window(tmp_path: Path) -> None:
    day_root = tmp_path / "parquet" / "AAPL" / "day"
    day_root.mkdir(parents=True)
    for session, close in (
        (date(2025, 2, 28), 99.0),
        (date(2025, 3, 3), 100.0),
        (date(2025, 7, 1), 101.0),
    ):
        frame = pd.DataFrame(
            {
                "date": [session],
                "open": [close],
                "high": [close],
                "low": [close],
                "close": [close],
                "volume": [1000.0],
            }
        )
        path = day_root / f"{session.isoformat()}.parquet"
        frame.to_parquet(path, index=False)
        path.with_suffix(".parquet.sha256").write_text(
            dataframe_content_sha256(
                frame,
                sort_by=[column for column in frame.columns if column != "date"],
            ),
            encoding="utf-8",
        )

    adapter = StockBenchAdapter.from_storage(
        tmp_path,
        symbols=["AAPL"],
        data_version="stockbench-oos-window-fixture",
        availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
        start_date=date(2025, 3, 3),
        end_date=date(2025, 6, 30),
    )

    snapshot = adapter.get_observations(datetime(2025, 7, 2, tzinfo=timezone.utc))
    assert [item.payload["date"] for item in snapshot] == ["2025-03-03"]


def test_storage_ingestion_rejects_a_partition_with_a_mismatched_internal_date(
    tmp_path: Path,
) -> None:
    day_root = tmp_path / "parquet" / "AAPL" / "day"
    day_root.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "date": [date(2025, 3, 4)],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1000.0],
        }
    )
    path = day_root / "2025-03-03.parquet"
    frame.to_parquet(path, index=False)
    path.with_suffix(".parquet.sha256").write_text(
        dataframe_content_sha256(
            frame,
            sort_by=[column for column in frame.columns if column != "date"],
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partition date mismatch"):
        StockBenchAdapter.from_storage(
            tmp_path,
            symbols=["AAPL"],
            data_version="mismatched-date-fixture",
            availability_policy=AvailabilityPolicy("America/New_York", time(16, 0)),
            start_date=date(2025, 3, 3),
            end_date=date(2025, 6, 30),
        )
