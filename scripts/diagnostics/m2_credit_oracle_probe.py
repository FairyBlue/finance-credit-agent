#!/usr/bin/env python3
"""Sealed CPU-only credit oracle experiment, separate from production training."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.env import ExecutionPolicy, PortfolioLedger  # noqa: E402
from src.rewards.sibling_credit import SiblingStep, compute_sibling_credit  # noqa: E402

PLAN = ROOT / "docs/2026-10-14_M2_CREDIT_ORACLE_PROBE_PLAN.md"
DEFAULT_OUT = ROOT / "runs/m2-credit-oracle-probe-20261014-v1"
METHODS = ("plain", "family_pool", "exact_same_gate", "exact_order_gate")
PATTERNS = {
    "homogeneous": ("buy_A", "buy_A", "hold", "hold"),
    "heterogeneous": ("buy_A", "buy_B", "hold", "hold"),
    "singleton": ("buy_A", "buy_B", "sell_A", "sell_B"),
    "unanimous": ("buy_A", "buy_A", "buy_B", "buy_B"),
    "state_mismatch": ("buy_A", "buy_B", "hold", "hold"),
}
STATE_TARGETS = ((0.2, 0.2), (0.1, 0.1), (0.3, 0.1), (0.1, 0.3))
CONFIG = {
    "schema_version": 1,
    "profile": "cpu_local_order_credit_oracle_v1",
    "seeds": [1729, 1730],
    "group_sizes": [4, 8, 16],
    "groups_per_condition_seed": 4096,
    "market_sigma": [0.0, 0.03, 0.10],
    "future_sigma_pp": [0.0, 0.25, 1.0],
    "common_shift_pp": [0.0, 2.0, 8.0],
    "initial_cash": 1_000_000.0,
    "commission_bps": 10.0,
    "slippage_bps": 5.0,
    "open_prices": {"A": 100.0, "B": 100.0},
    "base_close_prices": {"A": 104.0, "B": 97.0},
    "state_targets": STATE_TARGETS,
    "patterns": PATTERNS,
    "methods": METHODS,
    "rank_tolerance": 1e-10,
    "scientific_claim": "Conditional synthetic credit diagnosis only, no financial superiority",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PLAN,
        ROOT / "src/env/accounting.py",
        ROOT / "src/env/__init__.py",
        ROOT / "src/rewards/sibling_credit.py",
    )


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "contract.json",
        {
            "status": "SEALED_BEFORE_SIMULATION",
            "sealed_at_UTC": datetime.now(timezone.utc).isoformat(),
            "config": CONFIG,
            "source_sha256": {str(path): sha(path) for path in source_files()},
            "python": sys.version,
            "numpy": np.__version__,
            "production_changes": False,
            "formal_valid_oos_access": False,
        },
    )


def stamp(day: int, hour: int) -> datetime:
    return datetime(2023, 1, day, hour, tzinfo=timezone.utc)


def make_state(targets: tuple[float, float]) -> PortfolioLedger:
    ledger = PortfolioLedger(
        initial_cash=CONFIG["initial_cash"],
        policy=ExecutionPolicy(
            commission_bps=CONFIG["commission_bps"],
            slippage_bps=CONFIG["slippage_bps"],
        ),
    )
    for symbol, target, day in zip(("A", "B"), targets, (3, 4), strict=True):
        ledger.queue_target_order(symbol, target_weight=target, decision_at=stamp(day, 20))
        assert len(ledger.execute_next_open(stamp(day + 1, 14), CONFIG["open_prices"])) == 1
    return ledger


def local_oracles() -> list[dict]:
    result = []
    for state_id, targets in enumerate(STATE_TARGETS):
        state = make_state(targets)
        equity = state.equity(CONFIG["open_prices"])
        pre_exposure = (
            sum(state.positions[s] * CONFIG["open_prices"][s] for s in ("A", "B")) / equity
        )
        hold_equity = state.equity(CONFIG["base_close_prices"])
        for action in ("buy_A", "buy_B", "sell_A", "sell_B", "hold"):
            branch = copy.deepcopy(state)
            fills = ()
            if action != "hold":
                side, symbol = action.split("_")
                branch.queue_target_order(
                    symbol,
                    target_weight=0.4 if side == "buy" else 0.0,
                    direction=side,
                    decision_at=stamp(5, 20),
                )
                fills = branch.execute_next_open(stamp(6, 14), CONFIG["open_prices"])
                assert len(fills) == 1 and fills[0].side == side
            assert not branch.pending_orders and branch.cash >= 0
            post_exposure = (
                sum(branch.positions.get(s, 0) * CONFIG["open_prices"][s] for s in ("A", "B"))
                / equity
            )
            q = 100 * (branch.equity(CONFIG["base_close_prices"]) - hold_equity) / equity
            # Reconstruct the conditional one-period effect independently of equity().
            expected_q = (
                100
                * (
                    branch.cash
                    - state.cash
                    + sum(
                        (branch.positions.get(s, 0) - state.positions.get(s, 0))
                        * CONFIG["base_close_prices"][s]
                        for s in ("A", "B")
                    )
                )
                / equity
            )
            assert math.isclose(q, expected_q, abs_tol=1e-12)
            for market_shift in (-0.10, 0.03):
                closes = {
                    s: p + CONFIG["open_prices"][s] * market_shift
                    for s, p in CONFIG["base_close_prices"].items()
                }
                realized = 100 * (branch.equity(closes) - state.equity(closes)) / equity
                assert math.isclose(
                    realized, q + 100 * (post_exposure - pre_exposure) * market_shift, abs_tol=1e-12
                )
            result.append(
                {
                    "state_id": state_id,
                    "state_targets": targets,
                    "action": action,
                    "family": action.split("_")[0],
                    "state_equity": equity,
                    "state_cash": state.cash,
                    "state_positions": dict(state.positions),
                    "pre_exposure": pre_exposure,
                    "post_exposure": post_exposure,
                    "hold_payoff_pp": 100 * (hold_equity / equity - 1),
                    "q_pp": q,
                    "order_cost": sum(f.total_cost for f in fills),
                }
            )
    return result


def center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=-1, keepdims=True)


def pool(values: np.ndarray, labels: list[str]) -> np.ndarray:
    output = np.empty_like(values)
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        output[:, indices] = values[:, indices].mean(axis=1, keepdims=True)
    return center(output)


def predictions(
    y: np.ndarray, actions: list[str], states: list[int], inherited: np.ndarray
) -> dict[str, np.ndarray]:
    families = [a.split("_")[0] for a in actions]
    diverse = len(set(families)) >= 2
    zero = np.zeros_like(y)
    result = {
        "plain": center(y) if diverse else zero,
        "family_pool": pool(y, families) if diverse else zero,
        "exact_same_gate": pool(y, actions) if diverse else zero,
        "exact_order_gate": pool(y, actions) if len(set(actions)) >= 2 else zero,
    }
    if len(set(states)) > 1:
        # Uses unavailable hold counterfactual information: diagnostic reference only.
        labels = [f"{s}/{a}" for s, a in zip(states, actions, strict=True)]
        result["oracle_baseline_state_exact"] = pool(y - inherited, labels)
    return result


def metric(predicted: np.ndarray, target: np.ndarray) -> tuple[dict, np.ndarray]:
    tolerance = CONFIG["rank_tolerance"]
    error = np.mean((predicted - target) ** 2, axis=1)
    i, j = np.triu_indices(predicted.shape[1], 1)
    true_diff = target[:, i] - target[:, j]
    pred_diff = predicted[:, i] - predicted[:, j]
    valid = np.abs(true_diff) > tolerance
    ties = np.abs(pred_diff) <= tolerance
    strict = (true_diff * pred_diff > 0) & ~ties
    denom = valid.sum(axis=1)
    assert np.all(denom > 0), "scenario unexpectedly has no distinguishable oracle pairs"
    ranked = ((strict & valid).sum(axis=1) + 0.5 * (ties & valid).sum(axis=1)) / denom
    sign_valid = np.abs(target) > tolerance
    sign_denom = sign_valid.sum(axis=1)
    assert np.all(sign_denom > 0)
    neutral = np.abs(predicted) <= tolerance
    correct = ((target * predicted > 0) & ~neutral & sign_valid).sum(axis=1) / sign_denom
    return {
        "mse_pp_squared": float(error.mean()),
        "mse_mc_standard_error": float(error.std(ddof=1) / math.sqrt(len(error))),
        "pair_rank_accuracy_ties_half": float(ranked.mean()),
        "pair_prediction_tie_rate": float(((ties & valid).sum(axis=1) / denom).mean()),
        "sign_correct_rate": float(correct.mean()),
        "sign_neutral_rate": float(((neutral & sign_valid).sum(axis=1) / sign_denom).mean()),
        "oracle_pair_count": int(valid.sum()),
        "oracle_nonzero_member_count": int(sign_valid.sum()),
        "oracle_mean_square_pp_squared": float(np.mean(target**2)),
    }, error


def check_production_formula(oracles: list[dict]) -> int:
    lookup = {(r["state_id"], r["action"]): r for r in oracles}
    checked = 0
    for scenario, pattern in PATTERNS.items():
        state_ids = [1, 0, 2, 3] if scenario == "state_mismatch" else [0] * 4
        y = np.array(
            [
                [
                    lookup[s, a]["hold_payoff_pp"] + lookup[s, a]["q_pp"] + 0.1 * i
                    for i, (s, a) in enumerate(zip(state_ids, pattern, strict=True))
                ]
            ]
        )
        wanted = predictions(y, list(pattern), state_ids, np.zeros_like(y))
        fixture = []
        for rollout, action in enumerate(pattern):
            for day in range(20):
                fixture.append(
                    SiblingStep(
                        trajectory_uid=f"toy-{rollout}",
                        group_uid="algebra-fixture",
                        episode_id="toy-fixture-not-a-real-episode",
                        policy_step=1,
                        rollout_id=rollout,
                        rollout_n=4,
                        step_index=day,
                        step_count=20,
                        decision_session_index=day,
                        result_session_index=day + 1,
                        action=action.split("_")[0],
                        invalid_action=False,
                        done=day == 19,
                        episode_complete=True,
                        terminal_score=float(y[0, rollout]),
                    )
                )
        actual = compute_sibling_credit(fixture)
        assert np.allclose(np.array(actual.plain)[::20], wanted["plain"][0], atol=1e-12, rtol=0)
        assert np.allclose(
            np.array(actual.pooled)[::20], wanted["family_pool"][0], atol=1e-12, rtol=0
        )
        checked += 1
    return checked


def plot_results(out: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, scenario in zip(axes.flat, PATTERNS, strict=False):
        for method in METHODS:
            xs, ys = [], []
            for sigma in CONFIG["future_sigma_pp"]:
                selected = [
                    r
                    for r in rows
                    if r["scenario"] == scenario
                    and r["n"] == 4
                    and r["market_sigma"] == 0
                    and r["target"] == "expected"
                    and r["future_sigma_pp"] == sigma
                    and r["method"] == method
                ]
                xs.append(sigma)
                ys.append(sum(r["mse_pp_squared"] for r in selected) / len(selected))
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Independent future disturbance std (pp)")
        ax.set_ylabel("Oracle MSE (pp squared)")
        ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center")
    figure.suptitle("CPU synthetic credit oracle: n=4, market shock=0, two simulation seeds")
    figure.savefig(out / "oracle_mse.png", dpi=180)
    figure.savefig(out / "oracle_mse.svg")
    plt.close(figure)


def run(out: Path) -> None:
    contract = json.loads((out / "contract.json").read_text())
    assert contract["config"] == json.loads(json.dumps(CONFIG))
    for path, value in contract["source_sha256"].items():
        assert sha(Path(path)) == value, f"sealed source changed: {path}"
    assert not any(name in sys.modules for name in ("torch", "vllm", "transformers"))
    write_new(
        out / "start_receipt.json",
        {
            "started_at_UTC": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": sha(out / "contract.json"),
            "cuda_modules_loaded": False,
        },
    )
    oracles = local_oracles()
    write_new(out / "ledger_oracles.json", oracles)
    lookup = {(r["state_id"], r["action"]): r for r in oracles}
    formula_checks = check_production_formula(oracles)
    rows, analytic_checks = [], []
    shift_max = 0.0
    total_groups = 0
    total_members = 0
    for scenario_id, (scenario, pattern) in enumerate(PATTERNS.items()):
        for n in CONFIG["group_sizes"]:
            actions = list(pattern) * (n // 4)
            states = ([1, 0, 2, 3] if scenario == "state_mismatch" else [0] * 4) * (n // 4)
            entries = [lookup[s, a] for s, a in zip(states, actions, strict=True)]
            q = np.array([r["q_pp"] for r in entries])
            hold = np.array([r["hold_payoff_pp"] for r in entries])
            pre = np.array([r["pre_exposure"] for r in entries])
            post = np.array([r["post_exposure"] for r in entries])
            families = [a.split("_")[0] for a in actions]
            for market_id, market_sigma in enumerate(CONFIG["market_sigma"]):
                for future_id, future_sigma in enumerate(CONFIG["future_sigma_pp"]):
                    for seed in CONFIG["seeds"]:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, n, scenario_id, market_id, future_id])
                        )
                        groups = CONFIG["groups_per_condition_seed"]
                        z = rng.normal(0, market_sigma, (groups, 1))
                        epsilon = rng.normal(0, future_sigma, (groups, n))
                        inherited = hold + 100 * pre * z
                        y = hold + q + 100 * post * z + epsilon
                        pred = predictions(y, actions, states, inherited)
                        expected = np.broadcast_to(center(q), y.shape)
                        realized = center(q + 100 * (post - pre) * z)
                        for c in CONFIG["common_shift_pp"]:
                            shifted = predictions(y + c, actions, states, inherited)
                            shift_max = max(
                                shift_max,
                                *(float(np.max(np.abs(pred[m] - shifted[m]))) for m in pred),
                            )
                        plain_energy = float(np.mean(pred["plain"] ** 2))
                        for target_name, target in (("expected", expected), ("realized", realized)):
                            _, reference_error = metric(pred["plain"], target)
                            for method, value in pred.items():
                                values, error = metric(value, target)
                                difference = error - reference_error
                                labels = families if method == "family_pool" else actions
                                if method == "oracle_baseline_state_exact":
                                    labels = [
                                        f"{s}/{a}" for s, a in zip(states, actions, strict=True)
                                    ]
                                histogram = Counter(labels)
                                active = (
                                    len(set(families)) >= 2
                                    if method in ("plain", "family_pool", "exact_same_gate")
                                    else len(set(actions)) >= 2
                                )
                                peer = sum(histogram[label] >= 2 for label in labels) / n
                                item = {
                                    "scenario": scenario,
                                    "n": n,
                                    "market_sigma": market_sigma,
                                    "future_sigma_pp": future_sigma,
                                    "seed": seed,
                                    "target": target_name,
                                    "method": method,
                                    "groups": groups,
                                    "members": groups * n,
                                    **values,
                                    "paired_mse_difference_vs_plain": float(difference.mean()),
                                    "paired_difference_mc_standard_error": float(
                                        difference.std(ddof=1) / math.sqrt(groups)
                                    ),
                                    "gate_active": active,
                                    "same_label_peer_member_fraction": peer
                                    if method != "plain"
                                    else 0.0,
                                    "loo_peer_coverage_fraction": peer
                                    if active and method != "plain"
                                    else 0.0,
                                    "signal_energy_vs_plain": float(np.mean(value**2))
                                    / plain_energy
                                    if plain_energy > 1e-20
                                    else None,
                                    "oracle_information_used": method
                                    == "oracle_baseline_state_exact",
                                }
                                rows.append(item)
                                if (
                                    scenario == "homogeneous"
                                    and market_sigma == 0
                                    and target_name == "expected"
                                    and method in ("plain", "family_pool")
                                ):
                                    theoretical = (
                                        future_sigma**2 * ((n - 1) if method == "plain" else 1) / n
                                    )
                                    assert (
                                        abs(values["mse_pp_squared"] - theoretical)
                                        <= 6 * values["mse_mc_standard_error"] + 1e-12
                                    )
                                    analytic_checks.append(
                                        {
                                            "n": n,
                                            "future_sigma_pp": future_sigma,
                                            "seed": seed,
                                            "method": method,
                                            "observed": values["mse_pp_squared"],
                                            "analytic": theoretical,
                                            "status": "PASS",
                                        }
                                    )
                        total_groups += groups
                        total_members += groups * n
    assert shift_max < 1e-10
    for scenario in ("heterogeneous", "singleton"):
        selected = [
            r
            for r in rows
            if r["scenario"] == scenario
            and r["n"] == 4
            and r["market_sigma"] == 0
            and r["future_sigma_pp"] == 0
            and r["target"] == "expected"
        ]
        assert all(r["mse_pp_squared"] < 1e-20 for r in selected if r["method"] == "plain")
        assert all(r["mse_pp_squared"] > 0 for r in selected if r["method"] == "family_pool")
    assert total_groups == 1_105_920 and total_members == 10_321_920
    with (out / "metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_new(
        out / "engineering_checks.json",
        {
            "status": "PASS",
            "production_n4_algebra_fixtures": formula_checks,
            "fixture_limit": "20-session metadata is algebra-only, not a replayed ledger episode",
            "ledger_oracle_linear_market_checks": 40,
            "common_shift_max_error": shift_max,
            "homogeneous_independent_noise_analytic_checks": analytic_checks,
            "zero_noise_heterogeneous_bias_observed": True,
            "groups": total_groups,
            "members": total_members,
            "no_sample_exclusions": True,
            "loo_metrics": "coverage only, no LOO error estimate",
        },
    )
    plot_results(out, rows)
    assert not any(name in sys.modules for name in ("torch", "vllm", "transformers"))
    for path, value in contract["source_sha256"].items():
        assert sha(Path(path)) == value
    write_new(
        out / "completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
            "groups": total_groups,
            "members": total_members,
            "metric_rows": len(rows),
            "files_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
            "formal_financial_claim": False,
            "production_changes": False,
            "cuda_modules_loaded": False,
            "model_requests": 0,
            "optimizer_updates": 0,
        },
    )
    print(
        f"COMPLETE: {total_groups} synthetic groups, {total_members} members, "
        f"{len(rows)} metric rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
        print("SEALED", args.output / "contract.json")
    else:
        try:
            run(args.output)
        except Exception as error:
            receipt = args.output / "failure.json"
            if args.output.is_dir() and not receipt.exists():
                write_new(
                    receipt,
                    {
                        "status": "FAILED",
                        "failed_at_UTC": datetime.now(timezone.utc).isoformat(),
                        "error": repr(error),
                        "automatic_retry": False,
                    },
                )
            raise


if __name__ == "__main__":
    main()
