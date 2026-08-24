"""Raw daily-bar manifests bind run metadata to immutable bytes."""

from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from src.data import MarketObservation, StockBenchAdapter
from src.data.integrity import (
    audit_exchange_sessions,
    audit_raw_daily_manifest,
    verify_overlap_reconciliation,
    verify_raw_daily_manifest,
)


def test_xnys_session_list_has_a_canonical_count_and_checksum() -> None:
    audit = audit_exchange_sessions(start=date(2025, 1, 2), end=date(2025, 1, 3))

    assert audit.session_count == 2
    assert audit.canonical_bytes == b"2025-01-02\n2025-01-03\n"
    assert audit.sha256 == sha256(audit.canonical_bytes).hexdigest()


def _fixture_storage(tmp_path: Path) -> tuple[Path, Path, bytes]:
    manifest_root = tmp_path / "snapshot"
    storage_root = manifest_root / "storage"
    day_root = storage_root / "parquet" / "AAA" / "day"
    day_root.mkdir(parents=True)
    files = []
    for trading_date, payload in (("2025-01-02", b"first"), ("2025-01-03", b"second")):
        path = day_root / f"{trading_date}.parquet"
        path.write_bytes(payload)
        files.append(path)
    manifest_bytes = b"".join(
        (
            f"{path.relative_to(manifest_root).as_posix()}\t"
            f"{sha256(path.read_bytes()).hexdigest()}\n"
        ).encode()
        for path in sorted(files)
    )
    return manifest_root, storage_root, manifest_bytes


def test_raw_manifest_uses_sorted_posix_paths_raw_sha_and_terminal_lf(tmp_path: Path) -> None:
    manifest_root, storage_root, manifest_bytes = _fixture_storage(tmp_path)

    audit = audit_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage_root,
        symbols=["AAA"],
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
    )

    assert audit.file_count == 2
    assert audit.sessions_per_symbol == {"AAA": 2}
    assert audit.manifest_sha256 == sha256(manifest_bytes).hexdigest()
    assert audit.manifest_bytes == manifest_bytes


def test_raw_manifest_verification_detects_byte_tampering(tmp_path: Path) -> None:
    manifest_root, storage_root, _ = _fixture_storage(tmp_path)
    original = audit_raw_daily_manifest(
        manifest_root=manifest_root,
        storage_root=storage_root,
        symbols=["AAA"],
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
    )
    (storage_root / "parquet/AAA/day/2025-01-03.parquet").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="raw manifest SHA-256 mismatch"):
        verify_raw_daily_manifest(
            manifest_root=manifest_root,
            storage_root=storage_root,
            symbols=["AAA"],
            start=date(2025, 1, 2),
            end=date(2025, 1, 3),
            expected_sha256=original.manifest_sha256,
            expected_file_count=2,
            expected_sessions_per_symbol=2,
        )


def test_raw_manifest_rejects_a_missing_exchange_session(tmp_path: Path) -> None:
    manifest_root, storage_root, _ = _fixture_storage(tmp_path)
    (storage_root / "parquet/AAA/day/2025-01-03.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="missing daily bar.*2025-01-03"):
        audit_raw_daily_manifest(
            manifest_root=manifest_root,
            storage_root=storage_root,
            symbols=["AAA"],
            start=date(2025, 1, 2),
            end=date(2025, 1, 3),
        )


def _overlap_adapter(*, close: float) -> StockBenchAdapter:
    available_at = datetime(2025, 2, 28, 21, 0, tzinfo=timezone.utc)
    return StockBenchAdapter(
        [
            MarketObservation(
                observation_id=f"bar-{close}",
                kind="market",
                symbol="AAA",
                event_at=available_at,
                available_at=available_at,
                payload={
                    "date": "2025-02-28",
                    "open": 99.0,
                    "close": close,
                    "volume": 1000.0,
                },
            )
        ],
        data_version=f"fixture-{close}",
    )


def test_overlap_reconciliation_detects_cross_source_price_basis_drift() -> None:
    as_of = datetime(2025, 2, 28, 21, 0, tzinfo=timezone.utc)
    audit = verify_overlap_reconciliation(
        left=_overlap_adapter(close=100.0),
        right=_overlap_adapter(close=100.00001),
        as_of=as_of,
        symbols=["AAA"],
        tolerance_bps={"open": 0.01, "close": 0.01, "volume": 0.01},
    )
    assert audit.maximum_relative_difference_bps["close"] < 0.01

    with pytest.raises(RuntimeError, match="cross-source overlap close mismatch"):
        verify_overlap_reconciliation(
            left=_overlap_adapter(close=100.0),
            right=_overlap_adapter(close=99.0),
            as_of=as_of,
            symbols=["AAA"],
            tolerance_bps={"open": 0.01, "close": 0.01, "volume": 0.01},
        )
