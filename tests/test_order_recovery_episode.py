"""Full-window recovery budgets and accounting with real CPU trading environments."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.agents import order_recovery_episode as episode
from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import tool_call
from src.agents.rollout_seed import derive_request_seed
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv

HOLD = tool_call("hold")
BUY = tool_call("buy", {"symbol": "AAA", "target_weight": 0.5})
REJECT = tool_call("buy", {"symbol": "AAA", "target_weight": 0.2})
SELL = tool_call("sell", {"symbol": "AAA", "target_weight": 0.2})
RETRIEVE = tool_call("retrieve", {"query": "AAA"})


def case_fixture() -> dict[str, Any]:
    days = (2, 3, 4, 5)
    dates = [datetime(2024, 1, day, 21, tzinfo=timezone.utc) for day in days]
    adapter = StockBenchAdapter(
        [
            MarketObservation(
                observation_id=f"bar-{day}",
                kind="market",
                symbol="AAA",
                event_at=stamp,
                available_at=stamp,
                payload={"date": f"2024-01-{day:02d}", "open": 100.0, "close": 100.0},
            )
            for day, stamp in zip(days, dates, strict=True)
        ],
        data_version="order-recovery-fullreset-fixture",
    )
    return {
        "id": "fixture-rejection",
        "kind": "rejection",
        "source_identity": {"step": 4, "sample_index": 9, "rollout_n": 0, "validate": True},
        "raw_prompt": [{"role": "user", "content": "Manage the portfolio."}],
        # Deliberately retained on this test-only input. reset_case must never execute it.
        "prefix": [BUY, REJECT],
        "env_kwargs": {
            "adapter": adapter,
            "episode_dates": dates,
            "episode_open_times": [
                datetime(2024, 1, day, 14, 30, tzinfo=timezone.utc) for day in days
            ],
            "symbols": ["AAA"],
            "initial_cash": 1000.0,
            "commission_bps": 10.0,
            "slippage_bps": 20.0,
            "drawdown_coefficient": 0.0,
            "turnover_coefficient": 0.0,
            "benchmark_closes": [100.0] * len(days),
            "observation_profile": episode.OBSERVATION_PROFILE,
            "reward_mode": "finance_trajectory",
            "finance_epsilon": 0.001,
            "screening": True,
            "finance_scorer_id": "benchmark_scale_v2",
            "finance_volatility_floor": 0.001,
            "screening_calibration_sha256": "f" * 64,
            "risk_lambda": 1.0,
            "turnover_limit": 1.0,
            "position_limit": 0.9,
            "capture_interactions": True,
        },
    }


def models() -> dict[str, Any]:
    result = {}
    for key in episode.MODEL_KEYS:
        seeds = (42, 43) if key == "sft" else (int(key[4:6]),)
        result[key] = {
            "identities": [
                {"id": f"seed{seed}_sft" if key == "sft" else key, "base_seed": seed}
                for seed in seeds
            ]
        }
    return result


class Tokenizer:
    def __init__(self, script: list[str], *, oversized_prompt: bool = False) -> None:
        self.script = script
        self.oversized_prompt = oversized_prompt
        self.prompts: list[str] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs["enable_thinking"] is False
        return probe.canonical(messages)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        self.prompts.append(text)
        return [len(self.prompts)] * (episode.PROMPT_CAP + 1 if self.oversized_prompt else 1)

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return self.script[min(tokens[0], len(self.script) - 1)] if tokens else ""


class Engine:
    def __init__(self, *, mode: str = "normal") -> None:
        self.mode, self.calls, self.requests = mode, 0, 0

    def generate(self, prompts: Any, params: Any, **kwargs: Any) -> list[Any]:
        outputs = []
        for prompt, _ in zip(prompts, params, strict=True):
            tokens = [self.calls, 151645]
            if self.mode == "truncate":
                tokens = [0] * episode.RESPONSE_CAP
            if self.mode == "empty":
                tokens = []
            outputs.append(
                SimpleNamespace(
                    request_id=str(self.requests),
                    prompt_token_ids=(
                        [9999] if self.mode == "mapping" else prompt["prompt_token_ids"]
                    ),
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
        self.calls += 1
        return outputs


def run_case(
    tmp_path: Path,
    script: list[str],
    *,
    mode: str = "normal",
    oversized_prompt: bool = False,
) -> tuple[dict[str, Any], Engine, Tokenizer, Path]:
    manifest = {
        "cases": [case_fixture()],
        "models": {"sft": {"identities": [{"id": "seed42_sft", "base_seed": 42}]}},
    }
    output = tmp_path / "worker"
    output.mkdir()
    engine, tokenizer = Engine(mode=mode), Tokenizer(script, oversized_prompt=oversized_prompt)
    result = asyncio.run(
        episode.execute(
            engine,
            tokenizer,
            SimpleNamespace,
            manifest,
            "sft",
            output,
        )
    )
    return result, engine, tokenizer, output


def records(output: Path, name: str = "requests.jsonl") -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / name).read_text().splitlines()]


def test_fullreset_runs_entire_window_without_replaying_historical_orders(tmp_path: Path) -> None:
    result, engine, _, output = run_case(tmp_path, [HOLD])
    assert engine.requests == 5 * 3
    assert result["summary"]["performance_status"] == "COMPLETE"
    for row in result["grid"]:
        assert row["status"] == "COMPLETE"
        assert row["initial_ledger"] == {
            "cash": 1000.0,
            "positions": {},
            "fills": [],
            "session_index": 0,
        }
        assert row["initial_state"]["session_index"] == 0
        assert row["requests"] == len(row["episode_trace"]) == 3
        assert row["final_ledger"]["session_index"] == 3
        assert row["terminal_metrics"]["transaction_cost"] == 0
        assert row["terminal_metrics"]["step_count"] == 3
        assert row["full_window_net_return"] == 0
    request = records(output)[0]
    assert request["seed_env_step"] == 0
    assert request["sampling_params"]["seed"] == derive_request_seed(
        42,
        request["rollout_identity"],
        0,
    )
    assert records(output, "raw.jsonl") and records(output, "submissions.jsonl")


def test_recovery_continues_to_terminal_and_detects_later_new_loop(tmp_path: Path) -> None:
    result, engine, _, output = run_case(tmp_path, [BUY, REJECT, HOLD, REJECT, SELL])
    assert engine.requests == 25
    for row in result["grid"]:
        assert row["status"] == "COMPLETE"
        assert row["invalid_requests"] == 2
        assert [event["session_index"] for event in row["recovery_events"]] == [1, 2]
        assert [event["followup_requests"] for event in row["recovery_events"]] == [1, 1]
        assert all(event["status"] == "RECOVERED" for event in row["recovery_events"])
        assert row["recovery_events"][1]["advance_action"] == "sell"
        fills = row["final_ledger"]["fills"]
        assert len(fills) == 2
        cost = math.fsum(fill["commission"] + fill["slippage_cost"] for fill in fills)
        assert cost > 0
        assert row["terminal_metrics"]["transaction_cost"] == pytest.approx(cost)
        assert row["full_window_net_return"] == pytest.approx(-cost / 1000)
    assert result["summary"]["groups"][0]["trajectories_with_later_recovery_events"] > 0
    applied = records(output)
    rejected = [row for row in applied if row["feedback"].get("invalid_action")]
    assert all(row["ledger_before"] == row["ledger_after"] for row in rejected)
    assert all(
        row["reward"] == 0.0
        and row["trace"]["result_session_index"] == row["trace"]["decision_session_index"]
        for row in rejected
    )
    feedback = json.loads(
        rejected[0]["observation_messages"][-1]["content"]
        .removeprefix("<tool_response>")
        .removesuffix("</tool_response>")
        .strip()
    )
    assert feedback["order_rejection"]["version"] == "order_rejection_v2"


def test_first_invalid_plus_only_four_followups_never_injects_hold(tmp_path: Path) -> None:
    result, engine, _, output = run_case(tmp_path, [BUY, REJECT])
    assert engine.requests == 30
    assert result["summary"]["performance_status"] == "INCOMPLETE"
    for row in result["grid"]:
        assert row["status"] == "INCOMPLETE" and row["failure"] == "RECOVERY_CAP"
        assert row["requests"] == 6 and row["invalid_requests"] == 5
        assert row["recovery_events"][0]["followup_requests"] == 4
        assert row["final_ledger"]["session_index"] == 1
        assert "terminal_metrics" not in row and "full_window_net_return" not in row
        assert [step["action"] for step in row["episode_trace"]] == ["buy"] + ["invalid"] * 5
    assert all(
        group["mean_full_window_net_return"] is None and group["transaction_cost"] is None
        for group in result["summary"]["groups"]
    )
    rejected = [row for row in records(output) if row["feedback"].get("invalid_action")]
    assert all(row["ledger_before"] == row["ledger_after"] for row in rejected)


def test_fourth_followup_can_explicitly_recover_and_finish(tmp_path: Path) -> None:
    result, _, _, _ = run_case(tmp_path, [BUY, REJECT, REJECT, REJECT, REJECT, HOLD, HOLD])
    for row in result["grid"]:
        assert row["status"] == "COMPLETE"
        assert row["requests"] == 7
        assert row["recovery_events"][0]["followup_requests"] == 4
        assert row["recovery_events"][0]["status"] == "RECOVERED"


def test_valid_nonadvancing_tools_consume_recovery_budget(tmp_path: Path) -> None:
    result, engine, _, _ = run_case(tmp_path, [BUY, REJECT, RETRIEVE])
    assert engine.requests == 30
    for row in result["grid"]:
        assert row["failure"] == "RECOVERY_CAP"
        assert row["invalid_requests"] == 1
        assert row["recovery_events"][0]["followup_requests"] == 4
        assert row["final_ledger"]["session_index"] == 1


def test_format_invalid_also_opens_four_followup_recovery_budget(tmp_path: Path) -> None:
    result, engine, _, _ = run_case(tmp_path, ["No tool call"])
    assert engine.requests == 25
    for row in result["grid"]:
        assert row["failure"] == "RECOVERY_CAP"
        assert row["requests"] == row["invalid_requests"] == 5
        assert row["recovery_events"][0]["trigger_request_index"] == 0
        assert row["recovery_events"][0]["followup_requests"] == 4
        assert row["recovery_events"][0]["direction_rejection"] is False
        assert row["final_ledger"] == row["initial_ledger"]


def test_original_256_total_cap_also_bounds_legal_tool_loops(tmp_path: Path) -> None:
    result, engine, _, _ = run_case(tmp_path, [RETRIEVE])
    assert engine.requests == 5 * 256
    for row in result["grid"]:
        assert row["failure"] == "REQUEST_CAP" and row["requests"] == 256
        assert row["invalid_requests"] == 0 and row["recovery_events"] == []
        assert row["final_ledger"]["session_index"] == 0
        assert len(row["episode_trace"]) == 256


@pytest.mark.parametrize("mode", ["truncate", "empty"])
def test_incomplete_response_is_preserved_and_never_applied(tmp_path: Path, mode: str) -> None:
    result, engine, _, output = run_case(tmp_path, [HOLD], mode=mode)
    assert engine.requests == 5
    for row in result["grid"]:
        assert row["failure"] == "RESPONSE_INCOMPLETE"
        assert row["episode_trace"] == []
        assert row["final_ledger"]["session_index"] == 0
        assert "terminal_metrics" not in row
    assert all(record["environment_applied"] is False for record in records(output))


def test_prompt_overflow_keeps_prompt_without_submitting(tmp_path: Path) -> None:
    result, engine, _, output = run_case(tmp_path, [HOLD], oversized_prompt=True)
    assert engine.requests == 0
    assert all(row["failure"] == "PROMPT_CAP" and row["requests"] == 0 for row in result["grid"])
    assert all(
        len(row["prompt_token_ids"]) == 8193 and row["submitted"] is False
        for row in records(output)
    )


def test_bad_mapping_persists_submission_and_raw_before_failing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping differs"):
        run_case(tmp_path, [HOLD], mode="mapping")
    output = tmp_path / "worker"
    assert len(records(output, "raw.jsonl")[0]["outputs"]) == 5
    assert len(records(output, "submissions.jsonl")[0]["requests"]) == 5
    state = probe.read(output / "state.json")
    assert state["status"] == "FAILED"
    assert all(row["requests"] == 1 for row in state["grid"])


def test_budget_and_paired_seeds_keep_240_slots_and_one_engine_for_sft() -> None:
    cases = [{**case_fixture(), "id": name} for name in probe.FAILURE_IDS + probe.NORMAL_IDS]
    manifest = {"cases": cases, "models": models()}
    rows = [row for key in episode.MODEL_KEYS for row in episode.grid(manifest, key)]
    assert len(rows) == len({row["id"] for row in rows}) == 240
    assert sum(row["sampling"] == "greedy" for row in rows) == 48
    assert sum(row["sampling"] == "sampled" for row in rows) == 192
    assert episode.budget()["maximum_requests"] == 61440
    assert episode.budget()["recovery_followup_requests"] == 4
    selected = [
        row
        for row in rows
        if row["base_seed"] == 43
        and row["case_id"] == cases[0]["id"]
        and row["rollout_identity"]["rollout_n"] == 3
    ]
    params = [probe.sampling_parameters(row, 0) for row in selected]
    assert len(params) == 3 and all(value == params[0] for value in params)
    assert all(row["rollout_identity"]["validate"] is True for row in rows)


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fixture = case_fixture()
    adapter = fixture["env_kwargs"].pop("adapter")

    def factory(**kwargs: Any) -> TradingEnv:
        return TradingEnv(adapter=adapter, **kwargs)

    monkeypatch.setattr(episode, "TradingEnv", factory)
    cases = []
    for index, name in enumerate(probe.FAILURE_IDS + probe.NORMAL_IDS):
        case = copy.deepcopy(fixture)
        case.update(
            id=name,
            kind="rejection" if index < 6 else "normal",
            source={"path": "historical.jsonl", "line_number": index + 1},
        )
        cases.append(case)
    source = {
        "cases": cases,
        "models": models(),
        "evidence_sha256": {},
        "runtime_sha256": {
            str(Path(episode.__file__).resolve()): probe.sha(Path(episode.__file__))
        },
    }
    selection = tmp_path / "selection"
    selection.mkdir()
    probe.save(selection / "manifest.json", source)
    monkeypatch.setattr(probe, "verify", lambda *args: source)
    destination = tmp_path / "full"
    episode.prepare(destination, selection)
    return destination


def test_prepare_freezes_fullreset_without_prefix_or_mutable_short_state(prepared: Path) -> None:
    manifest = episode.verify(prepared)
    assert manifest["budget"]["planned_trajectories"] == 240
    assert all(
        "prefix" not in case and case["initial_ledger"]["session_index"] == 0
        for case in manifest["cases"]
    )
    assert not any(Path(name).name == "state.json" for name in manifest["evidence_sha256"])
    selection = Path(manifest["selection_manifest"])
    probe.save(selection.parent / "state.json", {"status": "RUNNING"})
    episode.verify(prepared)
    selection.write_text(selection.read_text() + " ")
    with pytest.raises(ValueError, match="Frozen full-reset source changed"):
        episode.verify(prepared)


@pytest.mark.parametrize("damage", ["digest", "budget", "prefix", "seed", "profile"])
def test_verify_rejects_manifest_and_contract_drift(prepared: Path, damage: str) -> None:
    manifest = probe.read(prepared / "manifest.json")
    if damage == "digest":
        with pytest.raises(ValueError, match="digest differs"):
            episode.verify(prepared, "0" * 64)
        return
    if damage == "budget":
        manifest["budget"]["requests_per_trajectory"] = 257
    elif damage == "prefix":
        manifest["cases"][0]["prefix"] = [HOLD]
    elif damage == "seed":
        manifest["models"]["seed42_m1"]["identities"][0]["base_seed"] = 43
    else:
        manifest["observation_profile"] = probe.NEW
    probe.save(prepared / "manifest.json", manifest)
    with pytest.raises(ValueError):
        episode.verify(prepared)


def test_readout_keeps_all_unstarted_slots_and_never_calls_them_complete(prepared: Path) -> None:
    result = episode.readout(prepared)
    assert result["status"] == "INCOMPLETE"
    assert result["summary"]["planned"] == 240
    assert result["summary"]["statuses"] == {"UNSTARTED": 240}
    assert result["summary"]["performance_status"] == "INCOMPLETE"


@pytest.fixture
def observed_worker(prepared: Path) -> Path:
    output = prepared / "workers/sft"
    output.mkdir(parents=True)
    asyncio.run(
        episode.execute(
            Engine(),
            Tokenizer([HOLD]),
            SimpleNamespace,
            episode.verify(prepared),
            "sft",
            output,
        )
    )
    return prepared


def test_observed_worker_never_hides_other_unstarted_actors(observed_worker: Path) -> None:
    result = episode.readout(observed_worker)
    assert result["status"] == "INCOMPLETE"
    assert result["workers"]["sft"] == "OBSERVED"
    assert result["summary"]["planned"] == 240
    assert result["summary"]["statuses"] == {"COMPLETE": 80, "UNSTARTED": 160}
    assert result["summary"]["performance_status"] == "INCOMPLETE"


@pytest.mark.parametrize("damage", ["identity", "request_cap", "recovery_cap", "terminal_failure"])
def test_readout_rejects_observed_identity_budget_and_terminal_drift(
    observed_worker: Path,
    damage: str,
) -> None:
    path = observed_worker / "workers/sft/result.json"
    result = probe.read(path)
    row = result["grid"][0]
    if damage == "identity":
        row["base_seed"] = 43
    elif damage == "request_cap":
        row["requests"] = 257
    elif damage == "recovery_cap":
        row["recovery_events"] = [{"status": "RECOVERED", "followup_requests": 5}]
    else:
        row.update(status="INCOMPLETE", failure="RECOVERY_CAP")
    probe.save(path, result)
    with pytest.raises(ValueError):
        episode.readout(observed_worker)
