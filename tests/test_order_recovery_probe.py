"""Real-environment replay and bounded recovery without invented terminal returns."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agent_r1.env.base import Action

from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import parse_state, tool_call
from src.agents.rollout_seed import derive_request_seed
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv


def real_case(tmp_path: Path, *, rejection: bool = True) -> dict[str, Any]:
    days = (2, 3, 6, 7, 8, 10)
    dates = [datetime(2025, 1, day, 21, tzinfo=timezone.utc) for day in days]
    adapter = StockBenchAdapter(
        [
            MarketObservation(
                observation_id=f"bar-{day}",
                kind="market",
                symbol="AAA",
                event_at=stamp,
                available_at=stamp,
                payload={"date": f"2025-01-{day:02d}", "open": 100.0, "close": 100.0},
            )
            for day, stamp in zip(days, dates, strict=True)
        ],
        data_version="order-recovery-fixture",
    )
    kwargs = {
        "adapter": adapter,
        "episode_dates": dates,
        "episode_open_times": [datetime(2025, 1, day, 14, 30, tzinfo=timezone.utc) for day in days],
        "symbols": ["AAA"],
        "initial_cash": 1000.0,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "drawdown_coefficient": 0.0,
        "turnover_coefficient": 0.0,
        "benchmark_closes": [100.0] * len(days),
        "observation_profile": probe.OLD,
        "reward_mode": "finance_trajectory",
        "finance_epsilon": 0.001,
        "risk_lambda": 1.0,
        "turnover_limit": 1.0,
        "position_limit": 0.9,
        "capture_interactions": True,
    }
    env = TradingEnv(**kwargs)
    raw_prompt = [{"role": "user", "content": "Manage the portfolio."}]
    env.reset(raw_prompt=raw_prompt)
    prefix = [tool_call("buy", {"symbol": "AAA", "target_weight": 0.5})]
    if rejection:
        prefix.append(tool_call("buy", {"symbol": "AAA", "target_weight": 0.4}))
    for text in prefix:
        asyncio.run(env.step(Action(text=text)))
    record = {
        "episode_id": "fixture-rejection" if rejection else "fixture-normal",
        "base_seed": 42,
        "identity": {"step": 4, "sample_index": 9, "rollout_n": 0, "validate": True},
        "interactions": env.interaction_trace,
        "steps": [json.loads(probe.canonical(asdict(step))) for step in env.credit_step_trace],
    }
    case = probe.source_case(
        record,
        {"env_kwargs": kwargs, "prompt": raw_prompt, "extra_info": {"source_split": "train"}},
        kind="rejection" if rejection else "normal",
        source=tmp_path / "source.jsonl",
        line_number=1,
        line_sha256="fixture",
    )
    case["takeover_ledger"] = probe.ledger_state(env)
    return case


def test_real_prefix_matches_history_and_new_profile_preserves_economics(tmp_path: Path) -> None:
    case = real_case(tmp_path)
    old, old_obs = asyncio.run(probe.replay_case(case, probe.OLD))
    new, new_obs = asyncio.run(probe.replay_case(case, probe.NEW))
    assert old.current_session_index == new.current_session_index == 1
    assert probe.ledger_state(old) == probe.ledger_state(new) == case["takeover_ledger"]
    old_state, new_state = parse_state(old_obs), parse_state(new_obs)
    assert "order_rejection" not in old_state
    assert new_state["order_rejection"]["session_advanced"] is False
    assert new_state["order_rejection"]["trade_executed"] is False
    assert old_state["decision_features"] == new_state["decision_features"]
    assert old_state["compute_history"] == new_state["compute_history"]


def test_replay_rejects_old_observation_drift(tmp_path: Path) -> None:
    case = real_case(tmp_path)
    document = case["prefix_interactions"][-1]["observation"]
    document["text"] += " "
    document["char_count"] = len(document["text"])
    document["sha256"] = hashlib.sha256(document["text"].encode()).hexdigest()
    with pytest.raises(ValueError, match="historical observation drift"):
        asyncio.run(probe.replay_case(case))


def test_replay_rejects_invalid_error_or_accounting_drift(tmp_path: Path) -> None:
    case = real_case(tmp_path)
    case["prefix_interactions"][-1]["info"]["error"] = "different error"
    with pytest.raises(ValueError, match="invalid/error/accounting drift"):
        asyncio.run(probe.replay_case(case))


def test_truncated_historical_response_is_not_a_prefix(tmp_path: Path) -> None:
    case = real_case(tmp_path)
    case["prefix_interactions"][0]["response"]["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        asyncio.run(probe.replay_case(case))


def test_normal_prefix_has_no_rejection_feedback(tmp_path: Path) -> None:
    case = real_case(tmp_path, rejection=False)
    env, obs = asyncio.run(probe.replay_case(case, probe.NEW))
    assert "order_rejection" not in parse_state(obs)
    assert env.current_session_index == 1


class Tokenizer:
    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs["enable_thinking"] is False
        return probe.canonical(messages)

    def encode(self, prompt: str, **kwargs: Any) -> list[int]:
        return [10, len(prompt)]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        if tokens and tokens[0] == 88:
            return tool_call("buy", {"symbol": "AAA", "target_weight": 0.4})
        return tool_call("hold")


class Engine:
    def __init__(self, mode: str = "hold") -> None:
        self.mode, self.calls, self.requests = mode, 0, 0

    def generate(self, prompts: Any, params: Any, **kwargs: Any) -> list[Any]:
        self.calls += 1
        outputs = []
        for prompt, _ in zip(prompts, params, strict=True):
            tokens = [88 if self.mode == "reject" else 10, 151645]
            if self.mode == "truncate":
                tokens = [10] * probe.RESPONSE_CAP
            returned_ids = list(prompt["prompt_token_ids"])
            if self.mode == "mapping":
                returned_ids = [99]
            outputs.append(
                SimpleNamespace(
                    request_id=str(self.requests),
                    prompt_token_ids=returned_ids,
                    outputs=[
                        SimpleNamespace(
                            token_ids=tokens,
                            finish_reason="length" if self.mode == "truncate" else "stop",
                            stop_reason=None,
                        )
                    ],
                )
            )
            self.requests += 1
        return outputs


def run_case(
    tmp_path: Path, *, mode: str = "hold", rejection: bool = True
) -> tuple[Engine, dict[str, Any], Path]:
    case = real_case(tmp_path, rejection=rejection)
    manifest = {
        "cases": [case],
        "models": {
            "sft": {
                "identities": [{"id": "seed42_sft", "base_seed": 42}],
            }
        },
    }
    output = tmp_path / "worker"
    output.mkdir()
    engine = Engine(mode)
    result = asyncio.run(
        probe.execute(
            engine,
            Tokenizer(),
            SimpleNamespace,
            manifest,
            "sft",
            output,
        )
    )
    return engine, result, output


def test_legal_recovery_stops_after_one_request_without_terminal_return(tmp_path: Path) -> None:
    engine, result, output = run_case(tmp_path)
    assert result["status"] == "OBSERVED"
    assert result["summary"]["statuses"] == {"RECOVERED": 10}
    assert engine.requests == 10
    for row in result["grid"]:
        assert row["requests"] == 1
        assert row["advanced_to_session"] == 2
        assert len(row["continuation_trace"]) == 1
        assert row["continuation_trace"][0]["done"] is False
    serialized = (output / "result.json").read_text()
    assert '"terminal_metrics"' not in serialized
    assert '"net_return"' not in serialized
    assert '"COMPLETE"' not in serialized


def test_normal_legal_continuation_is_advanced_not_recovered(tmp_path: Path) -> None:
    _, result, _ = run_case(tmp_path, rejection=False)
    assert result["summary"]["statuses"] == {"ADVANCED": 10}


def test_repeated_rejection_stops_at_four_without_replacement(tmp_path: Path) -> None:
    engine, result, output = run_case(tmp_path, mode="reject")
    assert engine.requests == 40
    assert result["summary"]["statuses"] == {"INCOMPLETE": 10}
    for row in result["grid"]:
        assert row["requests"] == row["invalid_requests"] == 4
        assert row["failure"] == "REQUEST_CAP"
        assert row["final_ledger"]["session_index"] == row["takeover_session"] == 1
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert len(records) == 40
    assert all(record["environment_applied"] for record in records)
    assert all(record["ledger_before"] == record["ledger_after"] for record in records)


def test_response_truncation_is_preserved_and_not_applied_to_environment(tmp_path: Path) -> None:
    engine, result, output = run_case(tmp_path, mode="truncate")
    assert engine.requests == 10
    assert result["summary"]["statuses"] == {"INCOMPLETE": 10}
    for row in result["grid"]:
        assert row["failure"] == "RESPONSE_INCOMPLETE"
        assert row["continuation_trace"] == []
        assert row["final_ledger"]["session_index"] == 1
    records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert all(record["environment_applied"] is False for record in records)
    assert all(len(record["response_token_ids"]) == 4096 for record in records)


def test_bad_mapping_saves_raw_batch_before_failing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping differs"):
        run_case(tmp_path, mode="mapping")
    raw = json.loads((tmp_path / "worker/raw.jsonl").read_text().splitlines()[0])
    assert len(raw["outputs"]) == 8
    assert probe.read(tmp_path / "worker/state.json")["status"] == "FAILED"


def test_request_seed_uses_registered_case_and_real_prefix_request_count(tmp_path: Path) -> None:
    case = real_case(tmp_path)
    manifest = {
        "cases": [case],
        "models": {
            "sft": {
                "identities": [{"id": "seed43_sft", "base_seed": 43}],
            }
        },
    }
    rows = probe.grid(manifest, "sft")
    original = copy.deepcopy(rows[0]["rollout_identity"])
    params = probe.sampling_parameters(rows[0], len(case["prefix"]))
    assert params["seed"] == derive_request_seed(43, original, 2)
    assert rows[0]["sampling"] == "greedy"
    assert rows[1]["sampling"] == "sampled"
    assert rows[1]["rollout_identity"]["rollout_n"] == 1
    assert rows[0]["rollout_identity"]["step"] == case["source_identity"]["step"]
    rows[0]["requests"] = 1
    assert probe.sampling_parameters(rows[0], 2)["seed"] == derive_request_seed(43, original, 3)


def test_budget_is_local_recovery_only() -> None:
    budget = probe.budget()
    assert budget["planned_trajectories"] == 8 * 6 * 2 * 5 == 480
    assert budget["maximum_requests"] == 1920
    assert budget["training_updates"] == 0
    assert budget["full_episode_evaluation"] is False
    assert probe.engine_settings(Path("fixture"))["gpu_memory_utilization"] == 0.35


def test_endpoint_protocol_pairs_feedback_versions_with_same_seeds(tmp_path: Path) -> None:
    case = real_case(tmp_path)
    manifest = {
        "budget": probe.budget(probe.ENDPOINT_PROFILE),
        "cases": [case],
        "models": {"sft": {"identities": [{"id": "seed42_sft", "base_seed": 42}]}},
    }
    rows = probe.grid(manifest, "sft")
    assert len(rows) == 10
    assert {row["profile"] for row in rows} == {probe.NEW, probe.ENDPOINT_OBSERVATION}
    for left, right in zip(rows[:5], rows[5:], strict=True):
        assert left["rollout_identity"] == right["rollout_identity"]
        assert probe.sampling_parameters(left, 2) == probe.sampling_parameters(right, 2)
    assert probe.budget(probe.ENDPOINT_PROFILE)["maximum_requests"] == 1920


def test_readout_retains_all_480_slots_before_any_worker_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "cases": [
            {
                "id": f"case{i}",
                "kind": "rejection" if i < 6 else "normal",
                "source_identity": {"step": 4, "sample_index": i, "rollout_n": 0, "validate": True},
            }
            for i in range(8)
        ],
        "models": {},
        "limitations": ["No financial returns"],
    }
    for key in probe.MODEL_KEYS:
        seeds = (42, 43) if key == "sft" else (int(key[4:6]),)
        manifest["models"][key] = {
            "identities": [
                {"id": f"seed{seed}_sft" if key == "sft" else key, "base_seed": seed}
                for seed in seeds
            ],
        }
    probe.save(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(probe, "verify", lambda *args: manifest)
    result = probe.readout(tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert result["summary"]["planned"] == 480
    assert result["summary"]["statuses"] == {"UNSTARTED": 480}
