"""Materialize reproducible daily-bar snapshots for the train/valid windows."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol
from urllib.request import Request, urlopen

import exchange_calendars as xcals
import pandas as pd

from .stockbench import dataframe_content_sha256


@dataclass(frozen=True)
class HistoryWindow:
    """Inclusive source window; split membership is configured separately."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("history window start must not follow end")


class DailyBarProvider(Protocol):
    """Network boundary implemented by yfinance or Stooq providers."""

    name: str
    version: str

    def fetch(self, symbol: str, *, start: date, end: date) -> pd.DataFrame:
        """Return one inclusive daily OHLCV frame indexed by session date."""


class YFinanceDailyBarProvider:
    """Pinned yfinance daily-bar boundary; network output is frozen immediately."""

    name = "yfinance"

    def __init__(self, *, corporate_actions_through: date) -> None:
        try:
            import yfinance
        except ImportError as error:
            raise RuntimeError(
                "install the frozen extras with `uv sync --frozen --extra gpu --extra data` "
                "on H100, or use a separate CPU-only data environment"
            ) from error
        self._client = yfinance
        self.version = str(yfinance.__version__)
        self._corporate_actions_through = corporate_actions_through
        self._download_settings = {
            "actions": True,
            "auto_adjust": False,
            "interval": "1d",
            "keepna": False,
            "multi_level_index": False,
            "prepost": False,
            "progress": False,
            "repair": False,
            "threads": False,
        }
        self.settings = {
            **self._download_settings,
            "corporate_actions_through": corporate_actions_through.isoformat(),
            "split_deadjustment": "reverse_all_later_split_ratios",
            "volume_rounding": "nearest_share",
        }

    def fetch(self, symbol: str, *, start: date, end: date) -> pd.DataFrame:
        """Download daily OHLCV and reconstruct contemporaneous nominal prices."""

        if self._corporate_actions_through < end:
            raise ValueError("corporate_actions_through must not precede the requested window")

        frame = self._client.download(
            symbol,
            start=start.isoformat(),
            end=(self._corporate_actions_through + timedelta(days=1)).isoformat(),
            **self._download_settings,
        )
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(f"yfinance returned no DataFrame for {symbol}")
        adjusted = frame.copy()
        if "Stock Splits" in adjusted.columns:
            ratios = pd.to_numeric(adjusted["Stock Splits"], errors="coerce").fillna(0.0)
            future_factors: list[float] = [1.0] * len(adjusted)
            running_factor = 1.0
            for index in range(len(adjusted) - 1, -1, -1):
                future_factors[index] = running_factor
                ratio = float(ratios.iloc[index])
                if ratio != 0.0:
                    if not math.isfinite(ratio) or ratio <= 0.0:
                        raise ValueError(f"invalid yfinance split ratio for {symbol}: {ratio}")
                    running_factor *= ratio
            factors = pd.Series(future_factors, index=adjusted.index)
            for column in ("Open", "High", "Low", "Close"):
                if column in adjusted.columns:
                    adjusted[column] = pd.to_numeric(adjusted[column]) * factors
            if "Volume" in adjusted.columns:
                adjusted["Volume"] = (pd.to_numeric(adjusted["Volume"]) / factors).round()

        index = pd.DatetimeIndex(adjusted.index)
        if index.tz is not None:
            index = index.tz_convert(None)
        session_dates = index.normalize()
        mask = (session_dates >= pd.Timestamp(start)) & (session_dates <= pd.Timestamp(end))
        return adjusted.loc[mask]


def _download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "finance-credit-agent/0.1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
        return response.read()


class StooqDailyBarProvider:
    """Dependency-free fallback for Stooq's public US daily CSV endpoint."""

    name = "stooq"
    version = "daily-csv-v1"

    def __init__(self, *, http_get: Callable[[str], bytes] | None = None) -> None:
        self._http_get = http_get or _download_bytes
        self.settings = {
            "endpoint": "https://stooq.com/q/d/l/",
            "interval": "d",
            "price_basis": "provider-defined-requires-human-audit",
        }

    def fetch(self, symbol: str, *, start: date, end: date) -> pd.DataFrame:
        ticker = symbol.strip().lower()
        if not ticker:
            raise ValueError("Stooq symbol must not be empty")
        url = f"https://stooq.com/q/d/l/?s={ticker}.us&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
        frame = pd.read_csv(BytesIO(self._http_get(url)), parse_dates=["Date"], index_col="Date")
        if frame.empty:
            raise RuntimeError(f"Stooq returned no daily bars for {symbol}")
        return frame.sort_index()


@dataclass(frozen=True)
class HistoryManifest:
    """Immutable identity of the normalized files consumed by the environment."""

    provider: str
    provider_version: str
    provider_settings: dict[str, str | bool]
    retrieved_at_utc: str
    window_start: str
    window_end: str
    symbols: tuple[str, ...]
    sessions_per_symbol: dict[str, int]
    file_count: int
    input_manifest_sha256: str


def _normalize_frame(frame: pd.DataFrame, *, symbol: str, window: HistoryWindow) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol} history must use a DatetimeIndex")
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in frame]
    required = ("open", "high", "low", "close", "volume")
    missing = set(required).difference(normalized.columns)
    if missing:
        raise ValueError(f"{symbol} history is missing columns: {sorted(missing)}")
    normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None).normalize()
    if normalized.index.has_duplicates:
        raise ValueError(f"{symbol} history contains duplicate sessions")
    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)
    normalized = normalized.loc[(normalized.index >= start) & (normalized.index <= end), required]
    if normalized.empty:
        raise ValueError(f"{symbol} history contains no sessions in the frozen window")
    if normalized.isna().any().any():
        raise ValueError(f"{symbol} history contains missing OHLCV values")
    return normalized.sort_index()


def _materialize_daily_history_in_place(
    *,
    provider: DailyBarProvider,
    symbols: Sequence[str],
    window: HistoryWindow,
    output_root: str | Path,
) -> HistoryManifest:
    """Freeze provider output into the checksum-verified daily storage layout."""

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "history output_root must be empty; use a clean path for every immutable snapshot"
        )
    normalized_symbols = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
        raise ValueError("symbols must contain at least one non-empty ticker")
    calendar = xcals.get_calendar("XNYS")
    expected_sessions = {
        timestamp.date()
        for timestamp in calendar.sessions_in_range(
            window.start.isoformat(), window.end.isoformat()
        )
    }

    parquet_files: list[Path] = []
    sessions_per_symbol: dict[str, int] = {}
    for symbol in normalized_symbols:
        frame = _normalize_frame(
            provider.fetch(symbol, start=window.start, end=window.end),
            symbol=symbol,
            window=window,
        )
        actual_sessions = {timestamp.date() for timestamp in frame.index}
        missing_sessions = sorted(expected_sessions.difference(actual_sessions))
        if missing_sessions:
            rendered = ", ".join(session.isoformat() for session in missing_sessions)
            raise ValueError(f"{symbol} history is missing XNYS sessions: {rendered}")
        unexpected_sessions = sorted(actual_sessions.difference(expected_sessions))
        if unexpected_sessions:
            rendered = ", ".join(session.isoformat() for session in unexpected_sessions)
            raise ValueError(f"{symbol} history contains non-XNYS sessions: {rendered}")
        sessions_per_symbol[symbol] = len(frame)
        day_root = root / "parquet" / symbol / "day"
        day_root.mkdir(parents=True, exist_ok=True)
        for timestamp, row in frame.iterrows():
            trading_date = timestamp.date()
            output = day_root / f"{trading_date.isoformat()}.parquet"
            document = pd.DataFrame(
                [{"date": trading_date, **{column: float(row[column]) for column in frame.columns}}]
            )
            document.to_parquet(output, index=False)
            logical_hash = dataframe_content_sha256(
                document,
                sort_by=[column for column in document.columns if column != "date"],
            )
            output.with_suffix(".parquet.sha256").write_text(logical_hash, encoding="utf-8")
            parquet_files.append(output)

    manifest_bytes = b"".join(
        (f"{path.relative_to(root).as_posix()}\t{sha256(path.read_bytes()).hexdigest()}\n").encode()
        for path in sorted(parquet_files)
    )
    manifest = HistoryManifest(
        provider=provider.name,
        provider_version=provider.version,
        provider_settings=dict(getattr(provider, "settings", {})),
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        window_start=window.start.isoformat(),
        window_end=window.end.isoformat(),
        symbols=normalized_symbols,
        sessions_per_symbol=sessions_per_symbol,
        file_count=len(parquet_files),
        input_manifest_sha256=sha256(manifest_bytes).hexdigest(),
    )
    (root / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_daily_history(
    *,
    provider: DailyBarProvider,
    symbols: Sequence[str],
    window: HistoryWindow,
    output_root: str | Path,
) -> HistoryManifest:
    """Atomically freeze a provider snapshot without exposing partial downloads."""

    final_root = Path(output_root).expanduser().resolve()
    if final_root.exists() and (not final_root.is_dir() or any(final_root.iterdir())):
        raise ValueError(
            "history output_root must be empty; use a clean path for every immutable snapshot"
        )
    final_root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{final_root.name}-", dir=final_root.parent) as temporary:
        staging_root = Path(temporary)
        manifest = _materialize_daily_history_in_place(
            provider=provider,
            symbols=symbols,
            window=window,
            output_root=staging_root,
        )
        if final_root.exists():
            final_root.rmdir()
        staging_root.replace(final_root)
    return manifest
