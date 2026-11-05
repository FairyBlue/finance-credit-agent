"""Real-ledger current API labels, strict boundary semantics and split isolation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agents import order_recovery_probe as probe
from src.data import MarketObservation, StockBenchAdapter
from src.env.order_feedback import ORDER_DIRECTION_TOLERANCE

if os.environ.get("ORDER_RECOVERY_SFT_MODULE"):
    spec = importlib.util.spec_from_file_location(
        "staged_order_recovery_sft_data", os.environ["ORDER_RECOVERY_SFT_MODULE"],
    )
    assert spec is not None and spec.loader is not None
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
else:
    from src.agents import order_recovery_sft_data as data


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tools"] and kwargs["tokenize"] is False
        return json.dumps(messages, sort_keys=True)


def fixture_row() -> dict:
    days = (1, 2, 3, 6, 7, 8, 9, 10)
    dates = [datetime(2023, 2, day, 21, tzinfo=timezone.utc) for day in days]
    symbols = [f"STOCK{index:02d}" for index in range(20)]
    bars = [MarketObservation(
        observation_id=f"sft-{symbol}-{day}", kind="market", symbol=symbol,
        event_at=stamp, available_at=stamp,
        payload={"date": stamp.date().isoformat(), "open": 100 + index * 0.11,
                 "close": 100 + index * 0.15},
    ) for index, (day, stamp) in enumerate(zip(days, dates, strict=True)) for symbol in symbols]
    kwargs = {
        "adapter": StockBenchAdapter(bars, data_version="recovery-sft-fixture"),
        "episode_dates": dates,
        "episode_open_times": [stamp.replace(hour=14, minute=30) for stamp in dates],
        "symbols": symbols, "initial_cash": 1000.0,
        "commission_bps": 10.0, "slippage_bps": 5.0,
        "drawdown_coefficient": 0.0, "turnover_coefficient": 0.0,
        "benchmark_closes": [100 + i * 0.1 for i in range(len(dates))],
        "screening": True, "finance_scorer_id": "benchmark_scale_v2",
        "finance_volatility_floor": 0.001, "screening_calibration_sha256": "d" * 64,
        "risk_lambda": 0.5, "turnover_limit": 1.0, "position_limit": 0.9,
    }
    return {"env_kwargs": kwargs, "prompt": [{"role": "user", "content": "Manage portfolio."}],
            "extra_info": {"episode_id": "fixture-Train", "source_split": "train"}}


def test_recovery_labels_are_diverse_and_come_from_real_cost_bearing_prefixes() -> None:
    rows = asyncio.run(data.recovery_records([fixture_row()], Tokenizer()))
    assert len(rows) == 24
    assert Counter(json.loads(row["completion"][11:-12])["name"] for row in rows) == {
        "buy": 10, "sell": 10, "hold": 4,
    }
    assert len({row["metadata"]["primary_symbol"] for row in rows}) == 20
    assert {row["metadata"]["state_spec"]["history"] for row in rows} == {
        "single", "repeat4", "ABAB",
    }
    for row in rows:
        meta = row["metadata"]
        assert meta["decision_session_index"] == 2
        assert len(meta["ledger_before_label"]["fills"]) == 2
        assert all(fill["commission"] > 0 and fill["slippage_cost"] > 0
                   for fill in meta["ledger_before_label"]["fills"])
        assert meta["future_prices_used_to_choose_label"] is False
        call = json.loads(row["completion"][11:-12])
        if call["name"] == "hold":
            assert meta["expected_direction"] == "no_trade"
        else:
            assert call["arguments"]["target_weight"] == meta["requested_target"]
            assert call["name"] == data.direction_for_target(
                meta["current_weight"], meta["requested_target"],
            )
        assert "API execution exercise" not in row["prompt"]


def test_boundary_rule_uses_validator_thresholds_and_nearest_representable_values() -> None:
    weight = 0.37
    lower, upper = weight - ORDER_DIRECTION_TOLERANCE, weight + ORDER_DIRECTION_TOLERANCE
    assert data.direction_for_target(weight, lower) == "no_trade"
    assert data.direction_for_target(weight, upper) == "no_trade"
    assert data.direction_for_target(weight, math.nextafter(lower, 0.0)) == "sell"
    assert data.direction_for_target(weight, math.nextafter(upper, 1.0)) == "buy"
    with pytest.raises(ValueError, match="finite fractions"):
        data.direction_for_target(weight, math.nan)


def test_current_label_validation_does_not_advance_or_read_future() -> None:
    async def exercise():
        env, _, semantics = await data.prepared_state(
            fixture_row(), 0, 0, data.recovery_specs()[0],
        )
        index, stamp = env.current_session_index, env.current_date
        original = env._adapter.get_observations
        reads = []

        def checked(timestamp, **kwargs):
            assert timestamp <= stamp
            reads.append(timestamp)
            return original(timestamp, **kwargs)

        env._adapter.get_observations = checked
        data.validate_label_without_future(env, data.label_for(semantics), semantics)
        assert env.current_session_index == index and reads

    asyncio.run(exercise())


def test_new_acceptance_cases_replay_old_and_v2_without_teacher_submission() -> None:
    bank = asyncio.run(data.acceptance_bank([fixture_row()], Tokenizer()))
    assert len(bank["cases"]) == 8
    assert bank["budget"]["requests_per_trajectory"] == 4
    for case in bank["cases"]:
        assert case["training_example"] is case["teacher_call_is_submitted"] is False
        assert case["source_identity"]["step"] == 0
        assert case["raw_state"]["session_index"] == 2
        assert case["raw_state"]["order_rejection"]["version"] == "order_rejection_v2"
        env, _ = asyncio.run(probe.replay_case(case, data.PROFILE))
        assert probe.ledger_state(env) == case["takeover_ledger"]


def test_overlap_or_non_train_source_is_rejected() -> None:
    row = fixture_row()
    with pytest.raises(ValueError, match="overlap"):
        data.guard_source(row, {"2023-02-03"})
    row["extra_info"]["source_split"] = "valid"
    with pytest.raises(ValueError, match="Only original Train"):
        data.guard_source(row, set())


def test_sft_artifact_writer_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "dataset_manifest.json"
    data.write_json(path, {"status": "COMPLETE"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        data.write_json(path, {})
    assert path.read_bytes() == before


def test_replay_preserves_original_bytes_with_fixed_quotas_and_rejects_dev_dates(monkeypatch):
    row = fixture_row()
    monkeypatch.setattr(data, "REPLAY_QUOTAS", {("api", "buy"): 2, ("api", "abstain"): 1})
    records = []
    for index, action in enumerate(("buy", "buy", "buy", "abstain")):
        records.append({
            "id": f"replay-{index}", "kind": "api",
            "prompt": '{"current_date":"2023-02-03T21:00:00+00:00"}' + str(index),
            "completion": '<tool_call>' + json.dumps({"name": action, "arguments": {}})
                          + '</tool_call>',
            "metadata": {"source_episode": "fixture-Train", "example_family": "clean",
                         "decision_date": "2023-02-03T21:00:00+00:00",
                         "prefix_expected_invalid": [False]},
        })
    lines = [json.dumps(record, indent=None) + "\n" for record in records]
    selected, provenance = data.select_replay(lines, {"fixture-Train": row}, set())
    assert len(selected) == len(provenance) == 3
    assert all(line in lines for line in selected)
    assert data.select_replay(list(reversed(lines)), {"fixture-Train": row}, set())[0] == selected
    with pytest.raises(ValueError, match="overlap"):
        data.select_replay(lines, {"fixture-Train": row}, {"2023-02-03"})
