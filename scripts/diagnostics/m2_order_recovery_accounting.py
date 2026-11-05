#!/usr/bin/env python3
"""CPU audit: identical submitted actions must retain identical trading economics.

This checks the frozen eight-case recovery bank, not policy quality. Each case
starts from its historical prefix, then runs two explicit scripts to terminal.
Normal storage-backed TradingEnv construction performs the 9,220-file integrity
preflight; this command supplies neither an adapter nor a preflight bypass.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_r1.env.base import Action  # noqa: E402

from src.agents import order_recovery_probe as probe  # noqa: E402
from src.agents.coldstart_data import tool_call  # noqa: E402
from src.env.trading import TradingEnv  # noqa: E402

PROFILES = (probe.OLD, probe.NEW, probe.ENDPOINT_OBSERVATION)
SCENARIOS = ("legal_actions", "rejection_then_explicit_action")
SOURCE_FILES = (
    "scripts/diagnostics/m2_order_recovery_accounting.py",
    "tests/test_m2_order_recovery_accounting.py",
    "src/agents/order_recovery_probe.py",
    "src/agents/coldstart_data.py",
    "src/env/trading.py",
    "src/env/accounting.py",
    "src/env/order_feedback.py",
    "src/env/observation_memory.py",
    "src/env/observation_features.py",
    "src/data/integrity.py",
    "src/data/stockbench.py",
    "src/rewards/screening.py",
)


def normalized(value: Any) -> Any:
    return json.loads(probe.canonical(value))


def require_equal(left: Any, right: Any, context: str) -> None:
    """Compare exact JSON values; no tolerance hides a profile-specific drift."""
    if normalized(left) != normalized(right):
        raise ValueError(f"Accounting drift: {context}")


def economic_state(env: TradingEnv) -> dict[str, Any]:
    return normalized({
        "ledger": {
            **probe.ledger_state(env),
            "pending_orders": [asdict(order) for order in env.ledger.pending_orders],
            "execution_skips": [asdict(order) for order in env.ledger.last_execution_skips],
        },
        "current_date": env.current_date,
        "equity_curve": env._equity_curve,
        "net_returns": env._net_returns,
        "turnovers": env._turnovers,
        "high_water_mark": env._high_water_mark,
        "maximum_position_weight": env._maximum_position_weight,
        "done": env._done,
    })


def accounting_result(env: TradingEnv, reward: float, done: bool,
                      info: dict[str, Any]) -> dict[str, Any]:
    return normalized({
        "economics": economic_state(env),
        "reward": reward,
        "done": done,
        "info": info,
        "credit_trace": [asdict(step) for step in env.credit_step_trace],
        "invalid_calls": env._invalid_calls,
        "tool_calls": env._tool_calls,
        "abstentions": env._abstentions,
    })


def check_costs(env: TradingEnv, before: dict[str, Any], info: dict[str, Any],
                context: str) -> list[Any]:
    previous_fills = len(before["ledger"]["fills"])
    fills = list(env.ledger.fills[previous_fills:])
    if info.get("invalid_action"):
        require_equal(economic_state(env), before, context + " rejected action mutated economics")
        return fills
    equity_before = before["equity_curve"][-1]
    cost = sum(fill.total_cost for fill in fills)
    if not math.isclose(info["cost_fraction"], cost / equity_before,
                        rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"Cost fraction does not debit fill costs: {context}")
    expected_cash = before["ledger"]["cash"]
    expected_positions = dict(before["ledger"]["positions"])
    for fill in fills:
        expected_commission = fill.notional * env.ledger.policy.commission_rate
        expected_slippage = fill.shares * abs(fill.execution_price - fill.raw_open_price)
        if not math.isclose(fill.commission, expected_commission, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Commission formula differs: {context}")
        if not math.isclose(fill.slippage_cost, expected_slippage, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Slippage formula differs: {context}")
        if min(fill.commission, fill.slippage_cost, fill.total_cost) <= 0:
            raise ValueError(f"Audit requires an actual fill with nonzero costs: {context}")
        if not math.isclose(fill.notional, fill.shares * fill.execution_price,
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Execution notional differs: {context}")
        sign = 1 if fill.side == "buy" else -1
        expected_cash -= sign * fill.notional + fill.commission
        expected_positions[fill.symbol] = (
            expected_positions.get(fill.symbol, 0.0) + sign * fill.shares
        )
    if not math.isclose(env.ledger.cash, expected_cash, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(f"Cash did not debit executed notional and commission: {context}")
    for symbol in set(expected_positions) | set(env.ledger.positions):
        if not math.isclose(env.ledger.positions.get(symbol, 0.0),
                            expected_positions.get(symbol, 0.0), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Position did not reflect executed shares: {context}")
    return fills


def validate_storage_contract(cases: list[dict[str, Any]]) -> dict[str, Any]:
    contracts = []
    for case in cases:
        kwargs = case["env_kwargs"]
        if ("adapter" in kwargs or kwargs.get("input_file_count") != 9220
                or kwargs.get("input_sessions_per_symbol") != 461
                or len(set(kwargs.get("symbols", ()))) != 20
                or kwargs.get("data_start") != "2023-01-03"
                or kwargs.get("data_end") != "2024-10-31"
                or kwargs.get("screening") is not True
                or kwargs.get("reward_mode") != "finance_trajectory"
                or kwargs.get("commission_bps", 0) <= 0
                or kwargs.get("slippage_bps", 0) <= 0):
            raise ValueError("Audit requires the real Train-only 20 x 461 storage contract")
        contracts.append({key: kwargs[key] for key in (
            "manifest_root", "storage_root", "input_manifest_sha256", "input_file_count",
            "input_sessions_per_symbol", "data_start", "data_end", "data_version",
        )})
    if not contracts or any(contract != contracts[0] for contract in contracts):
        raise ValueError("Case bank must share one frozen raw storage contract")
    return {**contracts[0], "verified_by": "TradingEnv._cached_storage_adapter",
            "adapter_injected": False, "preflight_bypassed": False}


def scenario_script(env: TradingEnv, scenario: str) -> list[dict[str, Any]]:
    if scenario == "legal_actions":
        unheld = [symbol for symbol in env.symbols
                  if env.ledger.positions.get(symbol, 0.0) == 0.0]
        if not unheld:
            raise ValueError("Legal script needs an unheld symbol for a fixed buy target")
        symbol = unheld[0]
        return [
            {"name": "buy", "arguments": {"symbol": symbol, "target_weight": 0.2}},
            {"name": "sell", "arguments": {"symbol": symbol, "target_weight": 0.0}},
            {"name": "hold", "arguments": {}},
            {"name": "abstain", "arguments": {"reason": "fixed accounting audit"}},
        ]
    held = [symbol for symbol in env.symbols if env.ledger.positions.get(symbol, 0.0) > 0]
    if not held:
        raise ValueError("Rejection script needs an existing position")
    symbol = held[0]
    return [
        {"name": "buy", "arguments": {"symbol": symbol, "target_weight": 0.0},
         "error": "buy target_weight must be above the current weight"},
        {"name": "sell", "arguments": {"symbol": symbol, "target_weight": 0.0}},
        {"name": "sell", "arguments": {"symbol": symbol, "target_weight": 0.2},
         "error": "sell target_weight must be below the current weight"},
        {"name": "buy", "arguments": {"symbol": symbol, "target_weight": 0.2}},
    ]


async def audit_cases(cases: list[dict[str, Any]], *,
                      env_class: Any = TradingEnv) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    records = []
    for case in cases:
        for scenario in SCENARIOS:
            environments = {}
            for profile in PROFILES:
                env, _ = await probe.replay_case(case, profile, env_class=env_class)
                environments[profile] = env
                counts["historical_prefix_replays"] += 1
                counts["historical_prefix_requests"] += len(case["prefix"])
            baseline = environments[PROFILES[0]]
            initial = economic_state(baseline)
            for profile, env in environments.items():
                require_equal(economic_state(env), initial,
                              f"{case['id']}:{scenario}:{profile}:prefix")
                counts["prefix_economic_comparisons"] += profile != PROFILES[0]
            script = scenario_script(baseline, scenario)
            if len(baseline._dates) - 1 - baseline.current_session_index < 4:
                raise ValueError("Accounting bank needs four legal continuation sessions")
            steps = []
            index = 0
            while not baseline._done:
                call = script[index] if index < len(script) else {"name": "hold", "arguments": {}}
                text = tool_call(call["name"], call["arguments"])
                results = {}
                for profile, env in environments.items():
                    before = economic_state(env)
                    _, reward, done, info = await env.step(Action(text=text))
                    label = f"{case['id']}:{scenario}:{index}:{profile}"
                    expected_error = call.get("error")
                    if bool(info.get("invalid_action")) != bool(expected_error):
                        raise ValueError(f"Unexpected action validity: {label}: {info}")
                    if expected_error:
                        if info.get("error") != expected_error or reward != 0.0 or done:
                            raise ValueError(f"Direction rejection contract differs: {label}")
                        counts["rejection_no_mutation_checks"] += 1
                    fills = check_costs(env, before, info, label)
                    counts["cost_fraction_checks"] += not bool(expected_error)
                    counts["nonzero_commission_slippage_fill_checks"] += len(fills)
                    if call["name"] in {"buy", "sell"} and not expected_error:
                        if not fills or any(fill.side != call["name"] for fill in fills):
                            raise ValueError(f"Explicit legal {call['name']} did not fill: {label}")
                        counts[f"explicit_{call['name']}_fills"] += len(fills)
                    if call["name"] in {"hold", "abstain"} and fills:
                        raise ValueError(f"No-trade action generated a fill: {label}")
                    results[profile] = accounting_result(env, reward, done, info)
                    counts["continuation_environment_steps"] += 1
                reference = results[PROFILES[0]]
                for profile in PROFILES[1:]:
                    require_equal(results[profile], reference,
                                  f"{case['id']}:{scenario}:{index}:{profile}:step")
                    counts["exact_step_comparisons"] += 1
                steps.append({"index": index, "submitted_action": text,
                              "result_sha256": hashlib.sha256(
                                  probe.canonical(reference).encode()).hexdigest(),
                              "result": reference})
                counts["paired_action_steps"] += 1
                counts[f"submitted_{call['name']}"] += 1
                counts["scripted_rejections"] += bool(call.get("error"))
                index += 1
            terminals = {profile: env.screening_terminal_metrics()
                         for profile, env in environments.items()}
            for profile in PROFILES[1:]:
                require_equal(terminals[profile], terminals[PROFILES[0]],
                              f"{case['id']}:{scenario}:{profile}:terminal finance")
                counts["exact_terminal_finance_comparisons"] += 1
            counts["complete_profile_episodes"] += len(PROFILES)
            records.append({"case_id": case["id"], "case_kind": case["kind"],
                            "scenario": scenario, "initial_economics": initial,
                            "steps": steps, "terminal_finance": terminals[PROFILES[0]]})
    return {"counts": dict(sorted(counts.items())), "records": records}


def write_artifact(output: Path, artifact: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(artifact, indent=2, allow_nan=False) + "\n")


def run_audit(recovery_directory: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite accounting evidence: {output}")
    manifest_path = recovery_directory.resolve() / "manifest.json"
    manifest = probe.read(manifest_path)
    cases = manifest["cases"]
    if [case["id"] for case in cases] != list(probe.FAILURE_IDS + probe.NORMAL_IDS):
        raise ValueError("Accounting audit requires the original fixed eight-case bank")
    storage = validate_storage_contract(cases)
    source_hashes = {str((ROOT / name).resolve()): probe.sha(ROOT / name)
                     for name in SOURCE_FILES}
    manifest_sha256 = probe.sha(manifest_path)
    result = asyncio.run(audit_cases(cases))
    for path, digest in source_hashes.items():
        if probe.sha(Path(path)) != digest:
            raise ValueError(f"Audit source changed during execution: {path}")
    if probe.sha(manifest_path) != manifest_sha256:
        raise ValueError("Recovery case bank changed during execution")
    artifact = {
        "status": "PASSED",
        "audit": "m2_order_recovery_same_action_accounting_v1",
        "profiles": list(PROFILES), "scenarios": list(SCENARIOS),
        "comparison": "exact JSON value equality; no numeric tolerance between profiles",
        "recovery_manifest": str(manifest_path), "recovery_manifest_sha256": manifest_sha256,
        "source_sha256": source_hashes,
        "historical_source_identity": [case["source"] for case in cases],
        "historical_source_files_sha256_from_manifest": manifest["evidence_sha256"],
        "historical_sources_rehashed": False,
        "raw_storage_preflight": storage,
        "gpu_used": False, "model_inference": False,
        "limitations": [
            "Same submitted action contract only; model behavior may differ between prompts.",
            "Fixed eight selected Train states and scripted continuations "
            "are not independent trials.",
            "Terminal finance belongs to these audit scripts, not to any learned policy.",
            "Historical source identities are bound by the manifest; "
            "source JSONL files are not reread.",
        ],
        **result,
    }
    write_artifact(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_audit(args.recovery_directory, args.output)
    print(json.dumps({"status": artifact["status"], "output": str(args.output.resolve()),
                      "counts": artifact["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
