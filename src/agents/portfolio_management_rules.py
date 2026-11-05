"""Run fixed portfolio actions on Train, with independent shared-ledger controls."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(prepared, output):
    from src.agents.broad_cpu_baselines import replay
    from src.agents.portfolio_management import PortfolioManagement, load_train_markets

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    markets = load_train_markets(prepared, output / "input")
    records, checks = [], []
    with (output / "trajectory_results.jsonl").open("x") as stream:
        for eid, market in markets.items():
            for strategy in ("hold", "momentum", "reversal", "defensive", "cash"):
                env = PortfolioManagement(market, initialize_equal_weight=strategy != "cash")
                while not env.done:
                    env.step("hold" if strategy == "cash" else strategy)
                result = dict(env.result(), strategy=strategy)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                records.append(result)
                assert result["status"] == "COMPLETE", (eid, strategy)
                assert len(result["decisions"]) == 4, (eid, strategy)
                assert len(result["equity_curve"]) == 21, (eid, strategy)
                if strategy in {"hold", "cash"}:
                    reference = replay(
                        market.kwargs, market.opens, market.closes,
                        equal_weight=strategy == "hold",
                    )
                    difference = float(np.max(np.abs(
                        np.asarray(result["equity_curve"]) - reference["equity_curve"]
                    )))
                    cost_error = abs(
                        result["metrics"]["transaction_cost"] - reference["transaction_cost"]
                    )
                    assert np.allclose(
                        result["equity_curve"], reference["equity_curve"], rtol=1e-10, atol=1e-7
                    ), (eid, strategy, difference)
                    assert cost_error < 1e-6, (eid, strategy, cost_error)
                    if strategy == "hold":
                        assert abs(result["metrics"]["benchmark_excess"]) < 1e-10
                    checks.append(dict(
                        episode_id=eid, strategy=strategy, status="PASS",
                        maximum_daily_equity_error=difference, transaction_cost_error=cost_error,
                    ))
            print(f"FINISHED_RULES {eid} 5/5", flush=True)
    write(output / "checks.json", dict(
        status="PASS", shared_ledger_reference_checks=checks,
        complete_trajectories=len(records), expected_trajectories=len(markets) * 5,
        decisions_per_episode=4, daily_curve_points=21,
    ))
    summary = dict(
        status="COMPLETE", source_split="train", episode_ids=list(markets),
        model_requests=0, optimizer_updates=0, trajectories=len(records),
        elapsed_seconds=time.monotonic() - started,
        formal_valid_access=False, formal_final_oos_access=False,
        strategies={},
    )
    for strategy in ("hold", "momentum", "reversal", "defensive", "cash"):
        rows = [r for r in records if r["strategy"] == strategy]
        summary["strategies"][strategy] = dict(
            complete_trajectories=len(rows),
            **{key: float(np.mean([r["metrics"][key] for r in rows])) for key in (
                "raw_net_return", "benchmark_excess", "terminal_financial_Y",
                "transaction_cost", "max_drawdown", "turnover",
            )},
        )
    write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.output), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
