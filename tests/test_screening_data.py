"""Screening selection, holdout isolation, and exact parquet row identity."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import yaml
from test_screening_reward import calibration_inputs

from src.agents.config import load_experiment_config
from src.data import screening
from src.data.integrity import RawManifestAudit
from src.rewards.screening import canonical_sha256, screening_payload_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_selection_counts_chronology_and_drift() -> None:
    config = load_experiment_config(ROOT / "configs/base.yaml")
    manifest = yaml.safe_load((ROOT / "configs/episode_manifest.reviewed.yaml").read_text())
    selected = screening.select_screening_episodes(manifest=manifest, config=config)
    train_days = {day for row in selected["train"] for day in row["episode_dates"]}
    dev_days = {day for row in selected["dev"] for day in row["episode_dates"]}
    assert len(selected["train"]) == 32 and len(selected["dev"]) == 12
    assert len(train_days) == 145 and len(dev_days) == 252
    assert not train_days.intersection(dev_days)
    assert max(train_days) < min(dev_days)
    manifest["splits"]["train"][4]["episode_dates"][0] = "2025-03-03T21:00:00+00:00"
    with pytest.raises(ValueError, match="sessions have drifted"):
        screening.select_screening_episodes(manifest=manifest, config=config)


def test_calibration_loader_rejects_dev_before_opening_price_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly its selected episode IDs"):
        screening.load_selected_calibration_markets(
            episodes=[{"episode_id": "formal-train-0210"}],
            source={"manifest_root": str(tmp_path), "storage_root": str(tmp_path)},
            raw_manifest_bytes=b"",
            symbols=[],
            exchange_calendar="XNYS",
        )


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    markets, benchmarks = calibration_inputs()
    original = yaml.safe_load((ROOT / "configs/episode_manifest.reviewed.yaml").read_text())
    for index in screening.SCREEN_TRAIN_INDICES:
        row = original["splits"]["train"][index]
        row["benchmark_closes"] = list(benchmarks[row["episode_id"]])
    for index in screening.SCREEN_DEV_INDICES:
        original["splits"]["train"][index]["benchmark_closes"] = list(
            next(iter(benchmarks.values()))
        )
    original_path = tmp_path / "reviewed.yaml"
    original_path.write_text(yaml.safe_dump(original))
    manifest_bytes = b"synthetic/train-only\t" + b"a" * 64 + b"\n"
    digest = sha256(manifest_bytes).hexdigest()
    source = {
        "manifest_root": str(tmp_path),
        "storage_root": str(tmp_path),
        "data_version": "synthetic-train",
        "data_start": "2023-01-03",
        "data_end": "2024-10-31",
        "input_manifest_sha256": digest,
        "input_file_count": 9220,
        "input_sessions_per_symbol": 461,
    }
    audit = RawManifestAudit(digest, manifest_bytes, 9220, {f"S{i:02d}": 461 for i in range(20)})
    monkeypatch.setattr(screening, "_audit_train_source", lambda *args: (source, audit))
    monkeypatch.setattr(screening, "load_selected_calibration_markets", lambda **kwargs: markets)
    output = tmp_path / "prepared"
    screening.prepare_screening_datasets(
        config_path=ROOT / "configs/base.yaml",
        manifest_path=original_path,
        output_dir=output,
        repo_root=ROOT,
    )
    return output


def test_prepared_rows_bind_calibration_and_pilot_indices(prepared: Path) -> None:
    manifest = screening.validate_prepared_screening(prepared, ROOT)
    assert {role: item["row_count"] for role, item in manifest["datasets"].items()} == {
        "train": 32,
        "dev": 12,
        "pilot_train": 2,
        "pilot_eval": 2,
    }
    for role in ("pilot_train", "pilot_eval"):
        rows = pq.read_table(prepared / f"{role}.parquet").to_pylist()
        assert [row["extra_info"]["index"] for row in rows] == [0, 1]
        assert [row["extra_info"]["source_index"] for row in rows] == [0, 4]
        environment = json.loads(rows[0]["env_kwargs"])
        assert environment["data_end"] == "2024-10-31"
        assert (
            environment["screening_calibration_sha256"] == manifest["calibration"]["payload_sha256"]
        )
        assert "finance_epsilon" not in environment


def test_rehashed_extra_env_kwargs_are_still_rejected(prepared: Path) -> None:
    path = prepared / "train.parquet"
    rows = pq.read_table(path).to_pylist()
    environment = json.loads(rows[0]["env_kwargs"])
    environment["unreviewed_parameter"] = 123
    rows[0]["env_kwargs"] = json.dumps(environment, sort_keys=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["datasets"]["train"]["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest["datasets"]["train"]["rows_sha256"] = canonical_sha256(rows)
    manifest["hashes"]["payload_sha256"] = screening_payload_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row/env_kwargs contract"):
        screening.validate_prepared_screening(prepared, ROOT)


def test_cpu_baselines_share_the_final_scorer_and_benchmark(
    prepared: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markets, _ = calibration_inputs()
    market = next(iter(markets.values()))
    calls = []

    def load_dev(**kwargs: object) -> dict:
        calls.append(kwargs["role"])
        return {f"formal-train-{index + 1:04d}": market for index in screening.SCREEN_DEV_INDICES}

    monkeypatch.setattr(screening, "_load_selected_markets", load_dev)
    results = screening.evaluate_screening_baselines(prepared, ROOT)
    assert calls == ["dev"]
    assert len(results) == 24
    assert len({row["episode_id"] for row in results}) == 12
    for row in results:
        if row["baseline"] == "cash":
            assert row["raw_net_return"] == row["risk_penalty"] == 0.0
        else:
            assert row["raw_net_return"] == pytest.approx(row["benchmark_return"])
            assert row["terminal_score"] <= 0.0


def test_selection_is_unaffected_by_unread_formal_holdout_contents() -> None:
    config = load_experiment_config(ROOT / "configs/base.yaml")
    original = yaml.safe_load((ROOT / "configs/episode_manifest.reviewed.yaml").read_text())
    expected = screening.select_screening_episodes(manifest=original, config=config)
    modified = deepcopy(original)
    modified["splits"]["valid"] = "must never be inspected"
    modified["splits"]["test"] = "must never be inspected"
    assert screening.select_screening_episodes(manifest=modified, config=config) == expected
