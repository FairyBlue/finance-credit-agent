"""StockBench adaptation with an unavoidable point-in-time gate."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, overload
from zoneinfo import ZoneInfo

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
ObservationKind: TypeAlias = Literal["market", "news", "fundamental"]


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


@dataclass(frozen=True, slots=True)
class AvailabilityPolicy:
    """Explicit policy for mapping source timestamps to point-in-time availability.

    StockBench bars carry dates only.  Requiring this object keeps the market
    timezone and decision boundary visible at every real-data ingestion call.
    """

    market_timezone: str
    market_close_time: time
    news_delay: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        if self.market_close_time.tzinfo is not None:
            raise ValueError("market_close_time must be naive; market_timezone supplies the zone")
        if self.news_delay < timedelta(0):
            raise ValueError("news_delay cannot be negative")

    def market_available_at(self, trading_date: date) -> datetime:
        # TODO(human): line-review exchange calendar, early closes, timezone, and bar boundary.
        local_close = datetime.combine(
            trading_date,
            self.market_close_time,
            tzinfo=ZoneInfo(self.market_timezone),
        )
        return local_close.astimezone(timezone.utc)

    def news_available_at(self, epoch_seconds: int | float) -> datetime:
        # TODO(human): line-review whether publication latency needs a conservative delay.
        return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc) + self.news_delay


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
        if self.kind not in {"market", "news", "fundamental"}:
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
    ) -> StockBenchAdapter:
        """Ingest StockBench's immutable offline market/news storage layout.

        Only `parquet/{symbol}/day` and `cache/news_by_day/{symbol}` are read.
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

        observations: list[MarketObservation] = []
        for symbol in normalized_symbols:
            observations.extend(
                cls._load_market_files(
                    root,
                    symbol=symbol,
                    policy=availability_policy,
                    verify_checksums=verify_market_checksums,
                )
            )
            observations.extend(
                cls._load_news_files(root, symbol=symbol, policy=availability_policy)
            )
        return cls(observations, data_version=data_version)

    @staticmethod
    def _load_market_files(
        root: Path,
        *,
        symbol: str,
        policy: AvailabilityPolicy,
        verify_checksums: bool,
    ) -> list[MarketObservation]:
        import pandas as pd

        day_dir = root / "parquet" / symbol / "day"
        if not day_dir.is_dir():
            raise FileNotFoundError(f"missing StockBench day bars for {symbol}: {day_dir}")
        output: list[MarketObservation] = []
        required = {"date", "open", "high", "low", "close", "volume", "vwap"}
        for parquet_path in sorted(day_dir.glob("*.parquet")):
            if verify_checksums:
                checksum_path = parquet_path.with_suffix(parquet_path.suffix + ".sha256")
                if not checksum_path.is_file():
                    raise FileNotFoundError(f"missing StockBench checksum: {checksum_path}")
                expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
                actual = sha256(parquet_path.read_bytes()).hexdigest()
                if expected != actual:
                    raise ValueError(f"StockBench checksum mismatch: {parquet_path}")
            frame = pd.read_parquet(parquet_path)
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{parquet_path} is missing columns: {sorted(missing)}")
            for row_number, row in frame.iterrows():
                trading_date = pd.Timestamp(row["date"]).date()
                available_at = policy.market_available_at(trading_date)
                output.append(
                    MarketObservation(
                        observation_id=f"market:{symbol}:{trading_date.isoformat()}:{row_number}",
                        kind="market",
                        symbol=symbol,
                        event_at=available_at,
                        available_at=available_at,
                        payload={field: float(row[field]) for field in required if field != "date"}
                        | {"date": trading_date.isoformat()},
                    )
                )
        return output

    @staticmethod
    def _load_news_files(
        root: Path,
        *,
        symbol: str,
        policy: AvailabilityPolicy,
    ) -> list[MarketObservation]:
        news_dir = root / "cache" / "news_by_day" / symbol
        if not news_dir.exists():
            return []
        output: list[MarketObservation] = []
        for json_path in sorted(news_dir.glob("*.json")):
            document = json.loads(json_path.read_text(encoding="utf-8"))
            items = document.get("items")
            if not isinstance(items, list):
                raise ValueError(f"{json_path} must contain an items list")
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not isinstance(item.get("datetime"), (int, float)):
                    raise ValueError(f"{json_path} item {index} lacks numeric datetime epoch")
                event_at = datetime.fromtimestamp(float(item["datetime"]), tz=timezone.utc)
                available_at = policy.news_available_at(item["datetime"])
                source_id = str(item.get("id", index))
                output.append(
                    MarketObservation(
                        observation_id=f"news:{symbol}:{source_id}",
                        kind="news",
                        symbol=symbol,
                        event_at=event_at,
                        available_at=available_at,
                        payload=item,
                    )
                )
        return output

    def get_observations(self, as_of: datetime) -> ObservationSnapshot:
        """Return exactly the observations available at `as_of`, never later ones."""

        gated_as_of = _require_aware(as_of, name="as_of")
        # This predicate is the non-bypassable temporal boundary. TODO(human): line-review it.
        visible = tuple(item for item in self._observations if item.available_at <= gated_as_of)
        return ObservationSnapshot(as_of=gated_as_of, items=visible)
