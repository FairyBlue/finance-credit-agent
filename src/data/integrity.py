"""Runtime audits that bind configured data identities to on-disk bytes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path

import exchange_calendars as xcals

from .stockbench import StockBenchAdapter


@dataclass(frozen=True, slots=True)
class RawManifestAudit:
    """Canonical raw-byte manifest and its aggregate identity."""

    manifest_sha256: str
    manifest_bytes: bytes
    file_count: int
    sessions_per_symbol: dict[str, int]


@dataclass(frozen=True, slots=True)
class OverlapReconciliationAudit:
    """Relative cross-vendor differences on one pre-OOS overlap session."""

    as_of: datetime
    maximum_relative_difference_bps: dict[str, float]
    per_symbol_relative_difference_bps: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class SessionCalendarAudit:
    """Canonical exchange-session list used to resolve observation/return counts."""

    session_count: int
    canonical_bytes: bytes
    sha256: str
    first_session: date
    last_session: date


def _normalized_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not normalized or any(not symbol for symbol in normalized):
        raise ValueError("symbols must contain at least one non-empty ticker")
    return normalized


def audit_exchange_sessions(
    *, start: date, end: date, exchange_calendar: str = "XNYS"
) -> SessionCalendarAudit:
    """Hash ascending ``YYYY-MM-DD<LF>`` session dates, including the final LF."""

    if start > end:
        raise ValueError("session window start must not follow end")
    calendar = xcals.get_calendar(exchange_calendar)
    sessions = tuple(
        timestamp.date()
        for timestamp in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    )
    if not sessions:
        raise ValueError("session window contains no exchange sessions")
    canonical = "".join(f"{session.isoformat()}\n" for session in sessions).encode()
    return SessionCalendarAudit(
        session_count=len(sessions),
        canonical_bytes=canonical,
        sha256=sha256(canonical).hexdigest(),
        first_session=sessions[0],
        last_session=sessions[-1],
    )


def audit_raw_daily_manifest(
    *,
    manifest_root: str | Path,
    storage_root: str | Path,
    symbols: Sequence[str],
    start: date,
    end: date,
    exchange_calendar: str = "XNYS",
) -> RawManifestAudit:
    """Hash every expected session file using the frozen manifest byte format.

    Paths are relative to ``manifest_root`` and sorted by their UTF-8 POSIX
    representation. Each line is ``path<TAB>lowercase-raw-sha256<LF>``; the
    final line also ends in LF.
    """

    if start > end:
        raise ValueError("manifest start must not follow end")
    root = Path(manifest_root).expanduser().resolve()
    storage = Path(storage_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"manifest root does not exist: {root}")
    if not storage.is_dir():
        raise FileNotFoundError(f"storage root does not exist: {storage}")
    if not storage.is_relative_to(root):
        raise ValueError("storage_root must be inside manifest_root")

    normalized_symbols = _normalized_symbols(symbols)
    calendar = xcals.get_calendar(exchange_calendar)
    sessions = tuple(
        timestamp.date()
        for timestamp in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    )
    if not sessions:
        raise ValueError("manifest window contains no exchange sessions")

    files: list[Path] = []
    for symbol in normalized_symbols:
        day_root = storage / "parquet" / symbol / "day"
        if not day_root.is_dir():
            raise FileNotFoundError(f"missing daily-bar directory for {symbol}: {day_root}")
        expected_names = {f"{session.isoformat()}.parquet" for session in sessions}
        for filename in sorted(expected_names):
            path = day_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing daily bar for {symbol} on {path.stem}: {path}")
            files.append(path)
        for path in day_root.glob("*.parquet"):
            try:
                partition_date = date.fromisoformat(path.stem)
            except ValueError as error:
                raise ValueError(f"non-date daily-bar partition: {path}") from error
            if start <= partition_date <= end and path.name not in expected_names:
                raise ValueError(f"non-session daily bar inside manifest window: {path}")

    relative_files = sorted(
        ((path.relative_to(root).as_posix(), path) for path in files),
        key=lambda item: item[0].encode("utf-8"),
    )
    manifest_bytes = b"".join(
        f"{relative}\t{sha256(path.read_bytes()).hexdigest()}\n".encode()
        for relative, path in relative_files
    )
    return RawManifestAudit(
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
        manifest_bytes=manifest_bytes,
        file_count=len(relative_files),
        sessions_per_symbol={symbol: len(sessions) for symbol in normalized_symbols},
    )


def verify_raw_daily_manifest(
    *,
    manifest_root: str | Path,
    storage_root: str | Path,
    symbols: Sequence[str],
    start: date,
    end: date,
    expected_sha256: str,
    expected_file_count: int,
    expected_sessions_per_symbol: int,
    exchange_calendar: str = "XNYS",
) -> RawManifestAudit:
    """Recompute a frozen source manifest and reject any identity drift."""

    audit = audit_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage_root,
        symbols=symbols,
        start=start,
        end=end,
        exchange_calendar=exchange_calendar,
    )
    if audit.manifest_sha256 != expected_sha256:
        raise RuntimeError(
            "raw manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {audit.manifest_sha256}"
        )
    if audit.file_count != expected_file_count:
        raise RuntimeError(
            f"raw manifest file count mismatch: expected {expected_file_count}, "
            f"got {audit.file_count}"
        )
    unexpected_sessions = {
        symbol: count
        for symbol, count in audit.sessions_per_symbol.items()
        if count != expected_sessions_per_symbol
    }
    if unexpected_sessions:
        raise RuntimeError(
            "raw manifest sessions-per-symbol mismatch: "
            f"expected {expected_sessions_per_symbol}, got {unexpected_sessions}"
        )
    return audit


def verify_overlap_reconciliation(
    *,
    left: StockBenchAdapter,
    right: StockBenchAdapter,
    as_of: datetime,
    symbols: Sequence[str],
    tolerance_bps: Mapping[str, float],
) -> OverlapReconciliationAudit:
    """Fail if two frozen vendors disagree beyond pre-registered tolerances."""

    normalized_symbols = _normalized_symbols(symbols)
    fields = tuple(tolerance_bps)
    if not fields:
        raise ValueError("overlap reconciliation needs at least one field")
    for field, tolerance in tolerance_bps.items():
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise ValueError(f"overlap tolerance for {field} must be finite and non-negative")
    expected_date = as_of.date().isoformat()

    def values(adapter: StockBenchAdapter) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for observation in adapter.get_observations(as_of):
            if (
                observation.kind != "market"
                or observation.symbol not in normalized_symbols
                or observation.payload.get("date") != expected_date
            ):
                continue
            if observation.symbol in output:
                raise RuntimeError(
                    f"duplicate overlap observation for {observation.symbol} on {expected_date}"
                )
            missing = [field for field in fields if field not in observation.payload]
            if missing:
                raise RuntimeError(
                    f"overlap observation for {observation.symbol} is missing: {missing}"
                )
            output[observation.symbol] = {
                field: float(observation.payload[field]) for field in fields
            }
        missing_symbols = [symbol for symbol in normalized_symbols if symbol not in output]
        if missing_symbols:
            raise RuntimeError(
                f"missing overlap observations on {expected_date} for: {missing_symbols}"
            )
        return output

    left_values = values(left)
    right_values = values(right)
    per_symbol: dict[str, dict[str, float]] = {}
    maxima = {field: 0.0 for field in fields}
    for symbol in normalized_symbols:
        per_symbol[symbol] = {}
        for field in fields:
            left_value = left_values[symbol][field]
            right_value = right_values[symbol][field]
            if (
                not math.isfinite(left_value)
                or not math.isfinite(right_value)
                or left_value <= 0.0
                or right_value <= 0.0
            ):
                raise RuntimeError(f"invalid overlap {field} value for {symbol}")
            difference = abs(left_value / right_value - 1.0) * 10_000.0
            per_symbol[symbol][field] = difference
            maxima[field] = max(maxima[field], difference)
            if difference > float(tolerance_bps[field]):
                raise RuntimeError(
                    f"cross-source overlap {field} mismatch for {symbol}: "
                    f"{difference:.6f} bps exceeds {float(tolerance_bps[field]):.6f} bps"
                )
    return OverlapReconciliationAudit(
        as_of=as_of,
        maximum_relative_difference_bps=maxima,
        per_symbol_relative_difference_bps=per_symbol,
    )
