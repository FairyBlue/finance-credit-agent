#!/usr/bin/env python3
"""CPU accounting/reward diagnostic on six fixed policies and two pilot Train windows.

No model requests, policy training, reward changes, Dev/OOS access or return-based
selection. The independent cash reconstruction uses raw-open trade values plus
separate costs, rather than the environment's NAV/turnover arrays.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "runs/m2-decision-diagnostic-20260909/reward-v1"
PILOT = ROOT / "artifacts/m2_screening/pilot_train.parquet"
PILOT_SHA = "c6ef2977af3fad0b28cb5e38c4f58bb2f42d066b0d4228cbb40b6852a759a4b3"
CALIBRATION = ROOT / "artifacts/m2_screening/reward_calibration.json"
CALIBRATION_SHA = "9538c46673ee25cbc984331e1bc91d484fa6527efae7071910bf92d90ec3a1bd"
REFERENCE = ROOT / "runs/m2-hold-diagnostic-20260909/observation-v1/manifest.json"
REFERENCE_SHA = "67197f289d423300be588fb354fd6b84338be6c10507f7aaacd413c1deb27a98"
PLAN = ROOT / "docs/2026-09-09_M2_DECISION_DIAGNOSTIC_PLAN.md"
EPISODES = ("formal-train-0001", "formal-train-0005")
STRATEGIES = (
    "cash",
    "aapl_buy_hold_10pct",
    "aapl_buy_hold_30pct",
    "first5_staggered_10pct",
    "aapl_roundtrip_10pct",
    "aapl_buy_hold_60pct",
)
RECONCILIATION_FIELDS = (
    "terminal_score",
    "raw_net_return",
    "benchmark_return",
    "benchmark_excess",
    "agent_volatility",
    "benchmark_volatility",
    "score_denominator",
    "max_drawdown",
    "turnover",
    "max_daily_turnover",
    "maximum_position_weight",
    "risk_penalty",
    "transaction_cost",
    "commission",
    "slippage_cost",
)


def ensure(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def cpu_import_paths() -> None:
    os.environ.update(
        USE_TORCH="0",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    for path in (ROOT, ROOT / "third_party/agent-r1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def strategy_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "diagnostic_id": "m2_reward_probe_v1",
        "episodes": list(EPISODES),
        "source_indices": [0, 4],
        "ordered_strategy_ids": list(STRATEGIES),
        "strategy_definitions": {
            "cash": "hold on all twenty decisions",
            "aapl_buy_hold_10pct": "decision 0: buy AAPL target_weight=0.1; then hold",
            "aapl_buy_hold_30pct": "decision 0: buy AAPL target_weight=0.3; then hold",
            "first5_staggered_10pct": (
                "decisions 0..4: buy symbols[decision_index] target_weight=0.1 in the original "
                "frozen universe order, one new stock per day; decisions 5..19 hold"
            ),
            "aapl_roundtrip_10pct": (
                "decision 0: buy AAPL target_weight=0.1; decision 1: sell AAPL target_weight=0; "
                "decisions 2..19 hold; both orders execute at their respective next opens"
            ),
            "aapl_buy_hold_60pct": (
                "decision 0: buy AAPL target_weight=0.6; then hold; "
                "intentional soft position-risk exposure"
            ),
        },
        "trajectories": 12,
        "sessions_per_trajectory": 21,
        "advances_per_trajectory": 20,
        "total_environment_steps": 240,
        "one_tool_per_decision": True,
        "model_requests": 0,
        "observation_profile": "legacy",
        "reward_mode": "finance_trajectory",
        "capture_interactions": True,
        "scorer": "benchmark_scale_v2",
        "reward_or_cost_or_risk_changes": False,
        "selection_rule": (
            "none; publish every fixed strategy, including losses and underperformance"
        ),
        "soft_limit_note": (
            "Actual close-marked weights, including drift above a 0.3 target, determine risk."
        ),
    }


def policy_actions(strategy: str, symbols: Sequence[str]) -> list[dict[str, Any]]:
    ensure(strategy in STRATEGIES, "Unknown fixed strategy")
    ensure(
        len(symbols) >= 5 and len(set(symbols)) == len(symbols) and "AAPL" in symbols,
        "Fixed policies require AAPL and at least five unique ordered symbols",
    )
    actions: list[dict[str, Any]] = [{"name": "hold", "arguments": {}} for _ in range(20)]
    if strategy == "first5_staggered_10pct":
        for index, symbol in enumerate(symbols[:5]):
            actions[index] = {"name": "buy", "arguments": {"symbol": symbol, "target_weight": 0.1}}
    elif strategy != "cash":
        weight = {
            "aapl_buy_hold_10pct": 0.1,
            "aapl_buy_hold_30pct": 0.3,
            "aapl_roundtrip_10pct": 0.1,
            "aapl_buy_hold_60pct": 0.6,
        }[strategy]
        actions[0] = {"name": "buy", "arguments": {"symbol": "AAPL", "target_weight": weight}}
        if strategy == "aapl_roundtrip_10pct":
            actions[1] = {"name": "sell", "arguments": {"symbol": "AAPL", "target_weight": 0.0}}
    return actions


def source_identity() -> dict[str, Any]:
    sources = [Path(__file__).resolve()]
    for directory in (
        ROOT / "src/env",
        ROOT / "src/data",
        ROOT / "src/rewards",
        ROOT / "third_party/agent-r1/agent_r1/env",
    ):
        sources.extend(directory.rglob("*.py"))
    return {
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "agent_r1_commit": subprocess.check_output(
            ["git", "-C", "third_party/agent-r1", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "python_version": sys.version,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "pyarrow", "pydantic", "exchange-calendars")
        },
        "files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))},
    }


def load_fixed_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cpu_import_paths()
    from pyarrow import parquet  # type: ignore[import-untyped]

    from src.rewards.screening import load_screening_calibration

    for path, expected in (
        (PILOT, PILOT_SHA),
        (CALIBRATION, CALIBRATION_SHA),
        (REFERENCE, REFERENCE_SHA),
    ):
        ensure(digest(path) == expected, f"Frozen reward diagnostic input changed: {path}")
    reference = json.loads(REFERENCE.read_text())
    calibration = load_screening_calibration(CALIBRATION)
    rows = parquet.read_table(PILOT).to_pylist()
    ensure(len(rows) == 2, "Only the two fixed pilot Train windows may be evaluated")
    windows = []
    for index, row in enumerate(rows):
        extra, kwargs = row["extra_info"], json.loads(row["env_kwargs"])
        ensure(
            extra["episode_id"] == EPISODES[index]
            and extra["source_index"] == 4 * index
            and extra["index"] == index
            and extra["source_split"] == "train"
            and extra["screening_role"] == "pilot_train",
            "Pilot identity drift",
        )
        ensure(
            kwargs == reference["windows"][index]["env_kwargs"]
            and row["prompt"] == reference["windows"][index]["raw_prompt"],
            "Frozen Train row drift",
        )
        ensure(
            kwargs["screening"] is True
            and kwargs["data_start"] == "2023-01-03"
            and kwargs["data_end"] == "2024-10-31"
            and kwargs["input_file_count"] == 9220,
            "Reward diagnostic escaped Train-only data identity",
        )
        ensure(
            len(kwargs["episode_dates"]) == len(kwargs["episode_open_times"]) == 21,
            "Expected twenty-one sessions",
        )
        reward = calibration["reward"]
        ensure(
            kwargs["screening_calibration_sha256"] == calibration["hashes"]["payload_sha256"]
            and kwargs["risk_lambda"] == reward["risk_lambda"]
            and kwargs["finance_volatility_floor"] == reward["volatility_floor"]
            and kwargs["turnover_limit"] == reward["turnover_limit"]
            and kwargs["position_limit"] == reward["position_limit"],
            "Frozen scorer parameters drift",
        )
        windows.append(
            {
                "episode_id": EPISODES[index],
                "source_index": 4 * index,
                "raw_prompt": row["prompt"],
                "env_kwargs": kwargs,
            }
        )
    return windows, calibration


def fixed_grid(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ensure(
        tuple(window["episode_id"] for window in windows) == EPISODES,
        "Fixed window identity/count drift",
    )
    return [
        {
            "grid_index": i * len(STRATEGIES) + j,
            "episode_id": window["episode_id"],
            "window_index": i,
            "strategy_id": strategy,
            "actions": policy_actions(strategy, window["env_kwargs"]["symbols"]),
        }
        for i, window in enumerate(windows)
        for j, strategy in enumerate(STRATEGIES)
    ]


def prepare(output: Path) -> dict[str, Any]:
    ensure(
        not (output / "manifest.json").exists() and not (output / "result.json").exists(),
        "Refusing to overwrite a registered or executed reward probe",
    )
    windows, calibration = load_fixed_inputs()
    registry = strategy_registry()
    source = source_identity()
    grid = fixed_grid(windows)
    manifest = {
        "status": "PREREGISTERED",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "registry": registry,
        "registry_payload_sha256": canonical_sha(registry),
        "plan_sha256": digest(PLAN),
        "grid": grid,
        "windows": windows,
        "source_identity": source,
        "input_files_sha256": {
            str(p.relative_to(ROOT)): sha
            for p, sha in (
                (PILOT, PILOT_SHA),
                (CALIBRATION, CALIBRATION_SHA),
                (REFERENCE, REFERENCE_SHA),
            )
        },
        "calibration_payload_sha256": calibration["hashes"]["payload_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "strategy_registry.json", registry)
    write_new(output / "manifest.json", manifest)
    return manifest


def close_enough(actual: float, expected: float, field: str) -> None:
    ensure(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
        f"Independent reconciliation failed: {field}: {actual!r} != {expected!r}",
    )


def parse_stamp(value: datetime | str) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def market_marks(env: Any, kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    marks = []
    for close_time, open_time in zip(
        kwargs["episode_dates"], kwargs["episode_open_times"], strict=True
    ):
        stamp = (
            close_time if isinstance(close_time, datetime) else datetime.fromisoformat(close_time)
        )
        prices = {}
        for record in env._adapter.get_observations(stamp):
            if (
                record.kind != "market"
                or record.symbol not in kwargs["symbols"]
                or record.payload.get("date") != stamp.date().isoformat()
            ):
                continue
            ensure(
                record.available_at <= stamp and record.symbol not in prices,
                "Ambiguous or not-yet-visible current-session marks",
            )
            prices[record.symbol] = {key: float(record.payload[key]) for key in ("open", "close")}
            ensure(
                all(math.isfinite(value) and value > 0 for value in prices[record.symbol].values()),
                "Invalid raw market price",
            )
        ensure(set(prices) == set(kwargs["symbols"]), "Incomplete fixed-universe marks")
        marks.append(
            {
                "close_time": stamp.isoformat(),
                "open_time": open_time.isoformat()
                if isinstance(open_time, datetime)
                else open_time,
                "prices": prices,
            }
        )
    return marks


def recompute_from_tape(
    *,
    kwargs: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    marks: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct positions/cash from raw-open trade values and separate costs."""
    ensure(len(actions) == len(steps) == 20 and len(marks) == 21, "Incomplete accounting tape")
    initial = float(kwargs["initial_cash"])
    cash = initial
    positions: dict[str, float] = {}
    equities, turnovers, position_weights = [initial], [], []
    commission_total = slippage_total = 0.0
    cash_without_costs = initial
    daily = []
    for index, (action, step) in enumerate(zip(actions, steps, strict=True)):
        before, after = marks[index], marks[index + 1]
        info = step["info"]
        ensure(
            step["decision_index"] == index
            and step["result_index"] == index + 1
            and step["done"] is (index == 19)
            and info["invalid_action"] is False,
            "Decision/advance/terminal alignment changed",
        )
        fills = info["fills"]
        ensure(
            len(fills) == (0 if action["name"] == "hold" else 1), "Fixed action/fill count mismatch"
        )
        prior_equity = cash + math.fsum(
            positions[s] * before["prices"][s]["close"] for s in positions
        )
        open_equity = cash + math.fsum(positions[s] * after["prices"][s]["open"] for s in positions)
        daily_notional = 0.0
        for fill in fills:
            symbol = fill["symbol"]
            ensure(
                symbol == action["arguments"]["symbol"] and fill["side"] == action["name"],
                "Fill is not the registered stock/direction",
            )
            ensure(
                parse_stamp(fill["decision_at"]) == parse_stamp(before["close_time"])
                and parse_stamp(fill["executed_at"]) == parse_stamp(after["open_time"])
                and parse_stamp(fill["executed_at"]) > parse_stamp(fill["decision_at"]),
                "Fill was not at the next open",
            )
            raw, shares = float(after["prices"][symbol]["open"]), float(fill["shares"])
            ensure(math.isfinite(shares) and shares > 0, "Invalid filled shares")
            direction = 1 if fill["side"] == "buy" else -1
            close_enough(fill["raw_open_price"], raw, "raw opening price")
            close_enough(
                fill["target_weight"], action["arguments"]["target_weight"], "target weight"
            )
            desired = open_equity * action["arguments"]["target_weight"] / raw
            close_enough(
                positions.get(symbol, 0.0) + direction * shares, desired, "registered target shares"
            )
            execution = raw * (1 + direction * kwargs["slippage_bps"] / 10_000)
            notional = shares * execution
            commission = notional * kwargs["commission_bps"] / 10_000
            slippage = shares * abs(execution - raw)
            for field, expected in (
                ("execution_price", execution),
                ("notional", notional),
                ("commission", commission),
                ("slippage_cost", slippage),
            ):
                close_enough(fill[field], expected, field)
            # Raw-price cash flow plus explicit costs is independent of the
            # production ledger's execution-price cash update.
            cash_without_costs -= direction * shares * raw
            cash -= direction * shares * raw + commission + slippage
            positions[symbol] = positions.get(symbol, 0.0) + direction * shares
            ensure(cash >= -1e-8 and positions[symbol] >= -1e-10, "Shorting/leverage appeared")
            commission_total += commission
            slippage_total += slippage
            daily_notional += notional
        equity = cash + math.fsum(positions[s] * after["prices"][s]["close"] for s in positions)
        gross_same_fills = cash_without_costs + math.fsum(
            positions[s] * after["prices"][s]["close"] for s in positions
        )
        close_enough(
            gross_same_fills - equity,
            commission_total + slippage_total,
            "cumulative cost cash drag",
        )
        close_enough(equity, info["equity"], "end-of-day NAV")
        close_enough(equity / prior_equity - 1, info["portfolio_return_net"], "daily net return")
        turnover = daily_notional / prior_equity
        close_enough(turnover, info["turnover"], "daily bilateral turnover")
        max_position = max(
            (positions[s] * after["prices"][s]["close"] / equity for s in positions), default=0.0
        )
        equities.append(equity)
        turnovers.append(turnover)
        position_weights.append(max_position)
        daily.append(
            {
                "session_index": index + 1,
                "cash": cash,
                "positions": dict(positions),
                "net_equity": equity,
                "equity_without_costs_same_filled_shares": gross_same_fills,
                "turnover": turnover,
                "max_position_weight": max_position,
            }
        )
    benchmark = [float(value) for value in kwargs["benchmark_closes"]]
    ensure(
        len(benchmark) == 21 and all(value > 0 and math.isfinite(value) for value in benchmark),
        "Invalid frozen benchmark levels",
    )
    daily_returns = [b / a - 1 for a, b in zip(equities[:-1], equities[1:], strict=True)]
    benchmark_returns = [b / a - 1 for a, b in zip(benchmark[:-1], benchmark[1:], strict=True)]
    drawdown = max(1 - value / max(equities[: i + 1]) for i, value in enumerate(equities))
    maximum_turnover, maximum_position = max(turnovers), max(position_weights)
    turnover_excess = max(0.0, maximum_turnover - kwargs["turnover_limit"])
    position_excess = max(0.0, maximum_position - kwargs["position_limit"])
    risk = max(drawdown, turnover_excess, position_excess)
    net_return, benchmark_return = equities[-1] / initial - 1, benchmark[-1] / benchmark[0] - 1
    volatility = statistics.pstdev(benchmark_returns)
    denominator = max(volatility, kwargs["finance_volatility_floor"])
    core = (net_return - benchmark_return) / denominator
    score = core - kwargs["risk_lambda"] * risk
    return {
        "terminal_score": score,
        "raw_net_return": net_return,
        "benchmark_return": benchmark_return,
        "benchmark_excess": net_return - benchmark_return,
        "agent_volatility": statistics.pstdev(daily_returns),
        "benchmark_volatility": volatility,
        "score_denominator": denominator,
        "max_drawdown": drawdown,
        "turnover": math.fsum(turnovers),
        "max_daily_turnover": maximum_turnover,
        "maximum_position_weight": maximum_position,
        "risk_penalty": risk,
        "turnover_excess": turnover_excess,
        "position_excess": position_excess,
        "transaction_cost": commission_total + slippage_total,
        "commission": commission_total,
        "slippage_cost": slippage_total,
        "finance_core": core,
        "weighted_risk_deduction": kwargs["risk_lambda"] * risk,
        "net_return_without_costs_same_filled_shares": (
            cash_without_costs
            + math.fsum(positions[s] * marks[-1]["prices"][s]["close"] for s in positions)
        )
        / initial
        - 1,
        "daily_reconstruction": daily,
    }


class RewardProbeFailure(ValueError):
    """Preserve observed accounting evidence if a fixed trajectory fails validation."""

    def __init__(self, error: Exception, evidence: dict[str, Any]) -> None:
        super().__init__(str(error))
        self.evidence = evidence
        self.original_error_type = type(error).__name__


def execute_strategy(window: Mapping[str, Any], planned: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {**planned, "steps": [], "marks": [], "model_requests": 0}
    try:
        return _execute_strategy(window, planned, evidence)
    except Exception as error:
        evidence["observed_advances"] = max(
            (step["result_index"] for step in evidence["steps"]), default=0
        )
        raise RewardProbeFailure(error, evidence) from error


def _execute_strategy(
    window: Mapping[str, Any], planned: Mapping[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    cpu_import_paths()
    from agent_r1.env.base import Action

    from src.env.trading import TradingEnv

    kwargs = dict(window["env_kwargs"])
    kwargs.update(
        reward_mode="finance_trajectory", observation_profile="legacy", capture_interactions=True
    )
    actions = policy_actions(planned["strategy_id"], kwargs["symbols"])
    ensure(actions == planned["actions"], "Registered fixed action sequence changed")
    env = TradingEnv(**kwargs)
    env.reset(raw_prompt=copy.deepcopy(window["raw_prompt"]))
    marks = market_marks(env, kwargs)
    evidence["marks"] = marks
    steps = evidence["steps"]
    for index, action in enumerate(actions):
        observation, reward, done, info = asyncio.run(
            env.step(
                Action(
                    text="<tool_call>" + json.dumps(action) + "</tool_call>",
                )
            )
        )
        steps.append(
            {
                "decision_index": index,
                "result_index": env.current_session_index,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "observation": observation.messages,
            }
        )
        ensure(
            not info["invalid_action"]
            and env.current_session_index == index + 1
            and done is (index == 19),
            "Fixed policy failed to make twenty legal advances",
        )
        if index < 19:
            ensure(reward == 0.0, "Nonterminal finance_trajectory reward leaked")
    ensure(
        env.episode_complete and env.episode_session_count == 21,
        "Incomplete fixed-policy trajectory",
    )
    actual = env.screening_terminal_metrics()
    evidence.update(
        environment_metrics=actual,
        credit_step_trace=[asdict(s) for s in env.credit_step_trace],
        fills=[asdict(fill) for fill in env.ledger.fills],
    )
    independent = recompute_from_tape(kwargs=kwargs, actions=actions, marks=marks, steps=steps)
    checks = {}
    for field in RECONCILIATION_FIELDS:
        value = actual[field]
        close_enough(float(value), independent[field], field)
        checks[field] = {
            "environment": value,
            "independent": independent[field],
            "passed": True,
        }
    ensure(
        actual["invalid_count"] == 0
        and actual["step_count"] == 20
        and actual["session_count"] == 21,
        "Terminal counts differ from fixed contract",
    )
    close_enough(
        steps[-1]["reward"], independent["terminal_score"], "actual emitted terminal reward"
    )
    return {
        **planned,
        "status": "VERIFIED",
        "steps": steps,
        "marks": marks,
        "credit_step_trace": [asdict(s) for s in env.credit_step_trace],
        "fills": [asdict(fill) for fill in env.ledger.fills],
        "environment_metrics": actual,
        "independent_metrics": independent,
        "reconciliation": checks,
        "model_requests": 0,
        "observed_advances": env.current_session_index,
    }


def mathematical_properties() -> dict[str, Any]:
    """Controlled scorer properties complement, rather than replace, ledger replay."""
    cpu_import_paths()
    from src.rewards.finance import RiskComponents
    from src.rewards.screening import score_benchmark_scaled

    common = {
        "benchmark_return": 0.02,
        "benchmark_volatility": 0.01,
        "volatility_floor": 0.001,
        "risk_lambda": 2.0,
    }
    risk = RiskComponents(0.03, 0.0, 0.0)
    lower = score_benchmark_scaled(episode_return=0.01, risk=risk, **common).advantage
    higher = score_benchmark_scaled(episode_return=0.02, risk=risk, **common).advantage
    costly = score_benchmark_scaled(episode_return=0.009, risk=risk, **common).advantage
    violation = score_benchmark_scaled(
        episode_return=0.01, risk=RiskComponents(0.03, 0, 0.3), **common
    ).advantage
    ensure(higher > lower > costly and violation < lower, "Controlled reward monotonicity failed")
    return {
        "status": "PASSED",
        "higher_return_increases_score_other_inputs_fixed": [lower, higher],
        "extra_cost_reduces_net_return_and_score_other_inputs_fixed": [lower, costly],
        "dominant_positive_position_excess_decreases_score": [lower, violation],
        "scope": (
            "Controlled finite examples, not additional market trajectories or exhaustive proof"
        ),
    }


def run(output: Path, expected_manifest_sha: str) -> dict[str, Any]:
    ensure(not (output / "result.json").exists(), "Refusing to replace an executed diagnostic")
    ensure(
        digest(output / "manifest.json") == expected_manifest_sha, "Reward manifest SHA mismatch"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    ensure(
        manifest["registry"] == strategy_registry()
        and manifest["registry_payload_sha256"] == canonical_sha(strategy_registry())
        and json.loads((output / "strategy_registry.json").read_text()) == strategy_registry(),
        "Registry changed",
    )
    ensure(
        source_identity()["files_sha256"] == manifest["source_identity"]["files_sha256"],
        "Source changed after registration",
    )
    ensure(digest(PLAN) == manifest["plan_sha256"], "Plan changed after registration")
    windows, _ = load_fixed_inputs()
    ensure(windows == manifest["windows"], "Source windows changed")
    ensure(manifest["grid"] == fixed_grid(windows), "Fixed grid/action identity drift")
    rows, evidence = [], {}
    for planned in manifest["grid"]:
        try:
            record = execute_strategy(windows[planned["window_index"]], planned)
        except Exception as error:
            record = {
                **getattr(error, "evidence", planned),
                "status": "FAILED",
                "error_type": getattr(error, "original_error_type", type(error).__name__),
                "error": str(error),
            }
        name = f"trajectory-{planned['grid_index']:02d}.json"
        write_new(output / name, record)
        evidence[name] = digest(output / name)
        rows.append(
            {
                key: value
                for key, value in record.items()
                if key
                not in {"steps", "marks", "credit_step_trace", "fills", "actions", "reconciliation"}
            }
        )
    ensure(len(rows) == 12, "A fixed strategy/window disappeared")
    properties = mathematical_properties()
    source_unchanged = (
        source_identity()["files_sha256"] == manifest["source_identity"]["files_sha256"]
        and digest(PLAN) == manifest["plan_sha256"]
    )
    result = {
        "status": "PASSED"
        if source_unchanged and all(r["status"] == "VERIFIED" for r in rows)
        else "FAILED",
        "status_meaning": (
            "Accounting/scorer reconciliation only; no policy or M2 effectiveness claim"
        ),
        "manifest_sha256": expected_manifest_sha,
        "registry_payload_sha256": manifest["registry_payload_sha256"],
        "source_unchanged": source_unchanged,
        "grid": rows,
        "evidence_files_sha256": evidence,
        "mathematical_properties": properties,
        "model_requests": 0,
        "trajectories": 12,
        "total_advances": sum(r.get("observed_advances", 0) for r in rows),
        "verified_advances": sum(20 for r in rows if r["status"] == "VERIFIED"),
        "gpu_frameworks_imported": [k for k in ("torch", "ray", "vllm") if k in sys.modules],
        "limitations": [
            "Deterministic policies are diagnostic controls, "
            "not model decisions or learned strategies.",
            "Every fixed result is retained; no ranking/selection or new reward calibration.",
            "Benchmark levels are the frozen external equal-weight benchmark; "
            "no replacement benchmark is fitted.",
            "The cost-free comparison holds executed shares fixed; "
            "it is not a separately rerun zero-cost policy.",
            "The position limit is a soft risk term; actual marked weights "
            "may exceed a nominal 0.3 target.",
        ],
    }
    ensure(not result["gpu_frameworks_imported"], "CPU reward probe imported a GPU framework")
    write_new(output / "result.json", result)
    lines = [
        "# 固定 Train 策略的奖励核验",
        "",
        f"核验状态：{result['status']}。",
        "固定 6 策略 × 2 窗口；无模型请求，不按收益筛选。收益与Y均保留正负值。",
        "",
        "| 窗口 | 策略 | 净收益 | 基准收益 | 总费用 | 最大回撤 | 持仓超限 | Y |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "VERIFIED":
            lines.append(f"| {row['episode_id']} | {row['strategy_id']} | FAILED | | | | | |")
            continue
        m = row["independent_metrics"]
        lines.append(
            f"| {row['episode_id']} | {row['strategy_id']} | {m['raw_net_return']:.6%} | "
            f"{m['benchmark_return']:.6%} | {m['transaction_cost']:.6f} | "
            f"{m['max_drawdown']:.6%} | {m['position_excess']:.6f} | {m['terminal_score']:.8f} |"
        )
    lines += [
        "",
        "每条轨迹实际推进20次，独立按成交/原始开盘与收盘重建cash/NAV、费用、风险和冻结分母；"
        "逐值核对环境原始指标与最后实际奖励。",
        "",
        "0.6目标用于验证软持仓风险扣分，不是推荐策略。静态数学性质检查仅补充真实账本核验。",
    ]
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("registry", "prepare", "run"))
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "registry":
        print(json.dumps(strategy_registry(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "prepare":
        prepare(args.output)
        print(
            json.dumps(
                {
                    "status": "PREREGISTERED",
                    "manifest_sha256": digest(args.output / "manifest.json"),
                }
            )
        )
        return 0
    ensure(bool(args.expected_manifest_sha256), "run requires the exact preregistered manifest SHA")
    result = run(args.output, args.expected_manifest_sha256)
    print(
        json.dumps(
            {
                "status": result["status"],
                "trajectories": result["trajectories"],
                "model_requests": 0,
            }
        )
    )
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
