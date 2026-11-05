"""Meaningful cost-bearing executions and strict cross-profile accounting checks."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest
from agent_r1.env.base import Action

from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import tool_call
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv

_PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_order_recovery_accounting.py"
_SPEC = importlib.util.spec_from_file_location("m2_order_recovery_accounting", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def case_fixture(tmp_path: Path) -> dict:
    days = (3, 4, 5, 6, 9, 10, 11, 12)
    dates = [datetime(2023, 1, day, 21, tzinfo=timezone.utc) for day in days]
    bars = [
        MarketObservation(
            observation_id=f"audit-{symbol}-{day}", kind="market", symbol=symbol,
            event_at=stamp, available_at=stamp,
            payload={"date": stamp.date().isoformat(), "open": 100 + index * 0.5,
                     "close": 100 + index * 0.6},
        )
        for index, (day, stamp) in enumerate(zip(days, dates, strict=True))
        for symbol in ("AAA", "BBB")
    ]
    kwargs = {
        "adapter": StockBenchAdapter(bars, data_version="accounting-audit-fixture"),
        "episode_dates": dates, "episode_open_times": [stamp.replace(hour=14, minute=30)
                                                        for stamp in dates],
        "symbols": ["AAA", "BBB"], "initial_cash": 1000.0,
        "commission_bps": 10.0, "slippage_bps": 5.0,
        "drawdown_coefficient": 0.0, "turnover_coefficient": 0.0,
        "benchmark_closes": [100 + i * 0.3 for i in range(len(dates))],
        "observation_profile": probe.OLD, "reward_mode": "finance_trajectory",
        "screening": True, "finance_scorer_id": "benchmark_scale_v2",
        "finance_volatility_floor": 0.001, "screening_calibration_sha256": "d" * 64,
        "risk_lambda": 0.5, "turnover_limit": 1.0, "position_limit": 0.9,
        "capture_interactions": True,
    }
    env = TradingEnv(**kwargs)
    prompt = [{"role": "user", "content": "Manage the portfolio."}]
    env.reset(raw_prompt=prompt)
    for text in (tool_call("buy", {"symbol": "AAA", "target_weight": 0.4}),
                 tool_call("buy", {"symbol": "AAA", "target_weight": 0.3})):
        asyncio.run(env.step(Action(text=text)))
    record = {
        "episode_id": "accounting-fixture", "base_seed": 42,
        "identity": {"rollout_n": 0, "validate": True, "step": 4, "sample_index": 0},
        "interactions": env.interaction_trace,
        "steps": json.loads(probe.canonical([asdict(step) for step in env.credit_step_trace])),
    }
    case = probe.source_case(
        record, {"env_kwargs": kwargs, "prompt": prompt, "extra_info": {"source_split": "train"}},
        kind="rejection", source=tmp_path / "fixture.jsonl", line_number=1, line_sha256="fixture",
    )
    case["takeover_ledger"] = probe.ledger_state(env)
    return case


def test_cost_bearing_buy_sell_and_both_explicit_recoveries_are_exact(tmp_path: Path) -> None:
    result = asyncio.run(audit.audit_cases([case_fixture(tmp_path)]))
    counts = result["counts"]
    assert counts["complete_profile_episodes"] == 6
    assert counts["exact_terminal_finance_comparisons"] == 4
    assert counts["scripted_rejections"] == 2
    assert counts["rejection_no_mutation_checks"] == 6
    assert counts["explicit_buy_fills"] == counts["explicit_sell_fills"] == 6
    assert counts["nonzero_commission_slippage_fill_checks"] == 12
    for record in result["records"]:
        metrics = record["terminal_finance"]
        assert metrics["commission"] > 0 and metrics["slippage_cost"] > 0
        assert metrics["transaction_cost"] == pytest.approx(
            metrics["commission"] + metrics["slippage_cost"], rel=1e-14,
        )
        assert metrics["session_count"] == 8
    recovery = result["records"][1]["steps"]
    assert [json.loads(step["submitted_action"].removeprefix("<tool_call>")
                       .removesuffix("</tool_call>"))["name"] for step in recovery[:4]] == [
        "buy", "sell", "sell", "buy",
    ]


def test_even_tiny_profile_cost_drift_fails_exact_comparison(tmp_path: Path) -> None:
    class DriftingEnv(TradingEnv):
        async def step(self, action):
            obs, reward, done, info = await super().step(action)
            if (self.observation_profile == probe.ENDPOINT_OBSERVATION
                    and not info.get("invalid_action") and info.get("cost_fraction", 0) > 0
                    and action.text and '"BBB"' in action.text):
                info = {**info, "cost_fraction": info["cost_fraction"] + 1e-16}
            return obs, reward, done, info

    with pytest.raises(ValueError, match="Accounting drift:.*:step"):
        asyncio.run(audit.audit_cases([case_fixture(tmp_path)], env_class=DriftingEnv))


def test_rejection_cannot_change_cash(tmp_path: Path) -> None:
    class MutatingEnv(TradingEnv):
        async def step(self, action):
            obs, reward, done, info = await super().step(action)
            if (self.observation_profile == probe.ENDPOINT_OBSERVATION
                    and info.get("invalid_action") and action.text
                    and '"target_weight":0.0' in action.text.replace(" ", "")):
                self.ledger._cash -= 0.01
            return obs, reward, done, info

    with pytest.raises(ValueError, match="rejected action mutated economics"):
        asyncio.run(audit.audit_cases([case_fixture(tmp_path)], env_class=MutatingEnv))


def test_production_rejects_injected_adapter_fixture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real Train-only"):
        audit.validate_storage_contract([case_fixture(tmp_path)])


def test_evidence_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    audit.write_artifact(path, {"status": "PASSED"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.write_artifact(path, {"status": "different"})
    with pytest.raises(FileExistsError):
        audit.run_audit(tmp_path / "does-not-exist", path)
    assert path.read_bytes() == original
