"""StockBench adaptation with an unavoidable point-in-time gate."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, overload
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
ObservationKind: TypeAlias = Literal["market", "fundamental", "technical"]


def _require_aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _freeze_json(value: Any) -> JsonValue:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"payload value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def dataframe_content_sha256(frame: Any, *, sort_by: Sequence[str] = ()) -> str:
    """Reproduce StockBench's logical DataFrame sidecar checksum exactly."""

    columns = list(frame.columns)
    canonical_frame = frame.sort_values(list(sort_by)) if sort_by else frame

    def json_default(value: object) -> str:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    records = canonical_frame[columns].to_dict(orient="records")
    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AvailabilityPolicy:
    """Explicit policy for mapping source timestamps to point-in-time availability.

    StockBench bars carry dates only.  Requiring this object keeps the market
    timezone and decision boundary visible at every real-data ingestion call.
    """

    market_timezone: str
    market_close_time: time
    exchange_calendar: str = "XNYS"

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        if self.market_close_time.tzinfo is not None:
            raise ValueError("market_close_time must be naive; market_timezone supplies the zone")
        calendar = xcals.get_calendar(self.exchange_calendar)
        if str(calendar.tz) != self.market_timezone:
            raise ValueError(
                f"{self.exchange_calendar} timezone is {calendar.tz}, not {self.market_timezone}"
            )
        regular_close = calendar.close_times[-1][1]
        if regular_close != self.market_close_time:
            raise ValueError(
                f"{self.exchange_calendar} regular close is {regular_close.isoformat()}, "
                f"not {self.market_close_time.isoformat()}"
            )

    def market_available_at(self, trading_date: date) -> datetime:
        """Return the actual exchange close, including holidays and early closes."""

        # TODO(human): line-review exchange calendar, early closes, timezone, and bar boundary.
        calendar = xcals.get_calendar(self.exchange_calendar)
        session = trading_date.isoformat()
        if not calendar.is_session(session):
            raise ValueError(f"{session} is not an {self.exchange_calendar} session")
        close = calendar.session_close(session).to_pydatetime()
        return _require_aware(close, name="market close").astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """One immutable point-in-time datum exposed to an agent.

    `available_at` is deliberately distinct from `event_at`: it is the earliest
    instant at which a replaying agent may see the datum.
    """

    observation_id: str
    kind: ObservationKind
    symbol: str
    event_at: datetime
    available_at: datetime
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must not be empty")
        if self.kind not in {"market", "fundamental", "technical"}:
            raise ValueError(f"unsupported observation kind: {self.kind}")
        object.__setattr__(self, "event_at", _require_aware(self.event_at, name="event_at"))
        object.__setattr__(
            self, "available_at", _require_aware(self.available_at, name="available_at")
        )
        # TODO(human): line-review StockBench field mapping and available_at assignment.
        copied_payload = deepcopy(dict(self.payload))
        json.dumps(copied_payload, allow_nan=False)
        object.__setattr__(self, "payload", _freeze_json(copied_payload))

    def canonical_record(self) -> dict[str, object]:
        """Return a deterministic serialization record for leakage assertions."""

        return {
            "available_at": self.available_at.isoformat(),
            "event_at": self.event_at.isoformat(),
            "kind": self.kind,
            "observation_id": self.observation_id,
            "payload": _thaw_json(self.payload),
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class ObservationSnapshot(Sequence[MarketObservation]):
    """Immutable result of one point-in-time read."""

    as_of: datetime
    items: tuple[MarketObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _require_aware(self.as_of, name="as_of"))
        if any(item.available_at > self.as_of for item in self.items):
            raise ValueError("snapshot contains an observation from the future")

    @overload
    def __getitem__(self, index: int) -> MarketObservation: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[MarketObservation, ...]: ...

    def __getitem__(self, index: int | slice) -> MarketObservation | tuple[MarketObservation, ...]:
        return self.items[index]

    def __iter__(self) -> Iterator[MarketObservation]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def canonical_bytes(self) -> bytes:
        records = [item.canonical_record() for item in self.items]
        return json.dumps(
            records,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class StockBenchAdapter:
    """In-memory point-in-time index over normalized StockBench observations.

    Construction is an ingestion boundary.  The only public method that reads
    observation content is :meth:`get_observations`, which always requires an
    explicit `as_of` timestamp and enforces `available_at <= as_of` internally.
    """

    def __init__(
        self,
        observations: Sequence[MarketObservation],
        *,
        data_version: str,
    ) -> None:
        if not data_version.strip():
            raise ValueError("data_version must be an immutable version identifier")
        ids = [item.observation_id for item in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observation_id values must be unique")
        self._observations = tuple(
            sorted(observations, key=lambda item: (item.available_at, item.observation_id))
        )
        self._data_version = data_version

    @classmethod
    def from_storage(
        cls,
        storage_root: str | Path,
        *,
        symbols: Sequence[str],
        data_version: str,
        availability_policy: AvailabilityPolicy,
        verify_market_checksums: bool = True,
        expected_raw_sha256_by_path: Mapping[str, str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> StockBenchAdapter:
        """Ingest StockBench's immutable offline market storage layout.

        V1 reads only `parquet/{symbol}/day`; news is a Phase 2 protocol item.
        In particular, this method never invokes StockBench's network-backed
        `DataHub`, and it deliberately excludes indicators/fundamentals pending a
        human point-in-time audit.
        """

        root = Path(storage_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"StockBench storage root does not exist: {root}")
        normalized_symbols = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
        if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
            raise ValueError("symbols must contain at least one non-empty ticker")
        if (start_date is None) != (end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start_date must not follow end_date")

        observations: list[MarketObservation] = []
        for symbol in normalized_symbols:
            market_rows = cls._load_market_files(
                root,
                symbol=symbol,
                policy=availability_policy,
                verify_checksums=verify_market_checksums,
                expected_raw_sha256_by_path=expected_raw_sha256_by_path,
                start_date=start_date,
                end_date=end_date,
            )
            previous_close: float | None = None
            for observation in market_rows:
                payload = dict(observation.payload)
                close = float(payload["close"])
                if previous_close is not None:
                    payload["technical_return_1d"] = close / previous_close - 1.0
                previous_close = close
                observations.append(
                    MarketObservation(
                        observation_id=observation.observation_id,
                        kind=observation.kind,
                        symbol=observation.symbol,
                        event_at=observation.event_at,
                        available_at=observation.available_at,
                        payload=payload,
                    )
                )
        return cls(observations, data_version=data_version)

    @staticmethod
    def _load_market_files(
        root: Path,
        *,
        symbol: str,
        policy: AvailabilityPolicy,
        verify_checksums: bool,
        expected_raw_sha256_by_path: Mapping[str, str] | None,
        start_date: date | None,
        end_date: date | None,
    ) -> list[MarketObservation]:
        import pandas as pd

        day_dir = root / "parquet" / symbol / "day"
        if not day_dir.is_dir():
            raise FileNotFoundError(f"missing StockBench day bars for {symbol}: {day_dir}")
        output: list[MarketObservation] = []
        required = {"date", "open", "high", "low", "close", "volume"}
        for parquet_path in sorted(day_dir.glob("*.parquet")):
            path_date = date.fromisoformat(parquet_path.stem)
            if (
                start_date is not None
                and end_date is not None
                and not start_date <= path_date <= end_date
            ):
                continue
            if expected_raw_sha256_by_path is None:
                frame = pd.read_parquet(parquet_path)
            else:
                resolved_path = str(parquet_path.resolve())
                expected_raw_sha256 = expected_raw_sha256_by_path.get(resolved_path)
                if expected_raw_sha256 is None:
                    raise RuntimeError(
                        f"daily bar is absent from the frozen raw manifest: {parquet_path}"
                    )
                raw_bytes = parquet_path.read_bytes()
                actual_raw_sha256 = sha256(raw_bytes).hexdigest()
                if actual_raw_sha256 != expected_raw_sha256:
                    raise RuntimeError(
                        "raw daily-bar SHA-256 mismatch: "
                        f"expected {expected_raw_sha256}, got {actual_raw_sha256}: "
                        f"{parquet_path}"
                    )
                # Parse exactly the bytes just verified, rather than reopening a
                # path that could have changed between verification and ingestion.
                frame = pd.read_parquet(BytesIO(raw_bytes))
            if verify_checksums:
                checksum_path = parquet_path.with_suffix(parquet_path.suffix + ".sha256")
                if not checksum_path.is_file():
                    raise FileNotFoundError(f"missing StockBench checksum: {checksum_path}")
                expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
                actual = dataframe_content_sha256(
                    frame,
                    sort_by=[column for column in frame.columns if column != "date"],
                )
                if expected != actual:
                    raise ValueError(f"StockBench checksum mismatch: {parquet_path}")
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")
            for row_number, row in frame.iterrows():
                trading_date = pd.Timestamp(row["date"]).date()
                if trading_date != path_date:
                    raise ValueError(
                        f"partition date mismatch in {parquet_path}: "
                        f"path={path_date.isoformat()} row={trading_date.isoformat()}"
                    )
                available_at = policy.market_available_at(trading_date)
                payload = {field: float(row[field]) for field in required if field != "date"}
                if "vwap" in frame.columns:
                    payload["vwap"] = float(row["vwap"])
                # Source technical_* columns are never trusted. V1 derives its
                # indicators only after chronological OHLCV ingestion.
                payload["date"] = trading_date.isoformat()
                output.append(
                    MarketObservation(
                        observation_id=f"market:{symbol}:{trading_date.isoformat()}:{row_number}",
                        kind="market",
                        symbol=symbol,
                        event_at=available_at,
                        available_at=available_at,
                        payload=payload,
                    )
                )
        return output

    def get_observations(self, as_of: datetime) -> ObservationSnapshot:
        """Return exactly the observations available at `as_of`, never later ones."""

        gated_as_of = _require_aware(as_of, name="as_of")
        # This predicate is the non-bypassable temporal boundary. TODO(human): line-review it.
        visible = tuple(item for item in self._observations if item.available_at <= gated_as_of)
        return ObservationSnapshot(as_of=gated_as_of, items=visible)
