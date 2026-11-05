"""Evaluate cash and reviewed equal-weight buy-and-hold on the fixed Valid grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agents.broad_prepare import ROOT, load_config
from src.agents.same_prefix_online_collect import _install_adapter


def replay(kwargs, opens, closes, *, equal_weight):
    from src.env import ExecutionPolicy, PortfolioLedger
    from src.rewards import RiskComponents
    from src.rewards.screening import score_benchmark_scaled

    symbols = kwargs["symbols"]
    ledger = PortfolioLedger(
        initial_cash=float(kwargs["initial_cash"]),
        policy=ExecutionPolicy(float(kwargs["commission_bps"]), float(kwargs["slippage_bps"])),
    )
    close_times = [datetime.fromisoformat(s) for s in kwargs["episode_dates"]]
    open_times = [datetime.fromisoformat(s) for s in kwargs["episode_open_times"]]
    equities = [ledger.equity(closes[0])]
    turnovers, drawdowns, positions = [], [], []
    peak = equities[0]
    for decision in range(20):
        if equal_weight and decision == 0:
            for symbol in symbols:
                ledger.queue_target_order(
                    symbol, target_weight=1 / len(symbols), decision_at=close_times[decision]
                )
        fills = ledger.execute_next_open(open_times[decision + 1], opens[decision + 1])
        equity = ledger.equity(closes[decision + 1])
        turnovers.append(math.fsum(fill.notional for fill in fills) / equities[-1])
        peak = max(peak, equity)
        drawdowns.append(max(0.0, (peak - equity) / peak))
        positions.append(
            max(
                (
                    shares * closes[decision + 1][symbol] / equity
                    for symbol, shares in ledger.positions.items()
                ),
                default=0,
            )
        )
        equities.append(equity)
    benchmark = np.asarray(kwargs["benchmark_closes"], dtype=float)
    if equal_weight and not np.allclose(equities, benchmark, rtol=1e-9, atol=0):
        raise ValueError("shared-ledger equal-weight replay differs from reviewed benchmark")
    net_return = equities[-1] / equities[0] - 1
    benchmark_return = float(benchmark[-1] / benchmark[0] - 1)
    benchmark_volatility = float(np.std(np.diff(benchmark) / benchmark[:-1], ddof=0))
    risk = RiskComponents(
        max_drawdown=max(drawdowns),
        turnover_excess=max(0.0, max(turnovers) - kwargs["turnover_limit"]),
        position_excess=max(0.0, max(positions) - kwargs["position_limit"]),
    )
    score = score_benchmark_scaled(
        episode_return=net_return,
        benchmark_return=benchmark_return,
        benchmark_volatility=benchmark_volatility,
        volatility_floor=kwargs["finance_volatility_floor"],
        risk_lambda=kwargs["risk_lambda"],
        risk=risk,
    )
    return dict(
        status="COMPLETE",
        raw_net_return=net_return,
        terminal_financial_Y=score.advantage,
        benchmark_return=benchmark_return,
        benchmark_excess=net_return - benchmark_return,
        benchmark_volatility=benchmark_volatility,
        max_drawdown=max(drawdowns),
        turnover=math.fsum(turnovers),
        max_daily_turnover=max(turnovers),
        maximum_position_weight=max(positions),
        transaction_cost=math.fsum(fill.total_cost for fill in ledger.fills),
        commission=math.fsum(fill.commission for fill in ledger.fills),
        slippage_cost=math.fsum(fill.slippage_cost for fill in ledger.fills),
        fill_count=len(ledger.fills),
        invalid_count=0,
        model_requests=0,
        terminal_session_index=20,
        session_count=21,
        equity_curve=equities,
        scorer_id=kwargs["finance_scorer_id"],
        calibration_sha256=kwargs["screening_calibration_sha256"],
    )


def run(args):
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
    from src.agents.broad_valid_env import make_valid_environment

    config = load_config(args.experiment_config)
    prepared_path, output = args.prepared.resolve(), args.output.resolve()
    if prepared_path != Path(config["prepared_valid_path"]).resolve():
        raise ValueError("CPU baselines are restricted to the configured history Valid input")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("CPU baseline result already exists")
    started = time.monotonic()
    source = pq.read_table(prepared_path).to_pylist()
    if [r["extra_info"]["episode_id"] for r in source] != config["valid_episode_ids"]:
        raise ValueError("fixed60 Valid episode identities required")
    prepared = {}
    for row in source:
        if row["extra_info"]["source_split"] != "valid":
            raise ValueError("Valid-only CPU baseline source")
        kwargs = json.loads(row["env_kwargs"])
        kwargs.update(capture_interactions=True, reward_mode="finance_trajectory")
        prepared[row["extra_info"]["episode_id"]] = dict(kwargs=kwargs, prompt=row["prompt"])
    os.environ["FCA_ONLINE_ADAPTER_CACHE"] = str(prepared_path.parent / "valid_adapter.pkl")
    _install_adapter(prepared, output)
    rows, price_cache = [], {}
    for eid, data in prepared.items():
        env = make_valid_environment(
            data["kwargs"], experiment_config=args.experiment_config, episode_id=eid
        )
        env.reset(raw_prompt=data["prompt"])
        kwargs = data["kwargs"]
        opens, closes = [], []
        for day in kwargs["episode_dates"]:
            if str(day)[:10] >= "2025-03-03":
                raise ValueError("Final OOS is excluded")
            if day not in price_cache:
                now = datetime.fromisoformat(day)
                observations = env._adapter.get_observations(now)
                current = {
                    o.symbol: o.payload
                    for o in observations
                    if o.kind == "market"
                    and o.payload.get("date") == str(day)[:10]
                    and o.symbol in kwargs["symbols"]
                }
                if set(current) != set(kwargs["symbols"]):
                    raise ValueError("exact current-session20-symbol baseline market required")
                price_cache[day] = (
                    {s: float(current[s]["open"]) for s in kwargs["symbols"]},
                    {s: float(current[s]["close"]) for s in kwargs["symbols"]},
                )
            opening, closing = price_cache[day]
            opens.append(opening)
            closes.append(closing)
        for name, equal in (("cash", False), ("buy_and_hold", True)):
            rows.append(
                dict(
                    episode_id=eid,
                    baseline=name,
                    **replay(kwargs, opens, closes, equal_weight=equal),
                )
            )
    summary = dict(
        status="COMPLETE",
        source_split="valid",
        formal_valid_access=True,
        formal_final_oos_access=False,
        prepared_path=str(prepared_path),
        episode_ids=config["valid_episode_ids"],
        planned_trajectories=120,
        complete_trajectories=len(rows),
        model_requests=0,
        prompt_tokens=0,
        response_tokens=0,
        deterministic_replicas_per_window=1,
        duplicate_model_replicas_not_counted=True,
        execution="shared PortfolioLedger, commission/slippage and after-close next-open fills",
        equal_weight_schedule="twenty equal target orders on first decision, then hold",
        reviewed_benchmark_replay_verified=True,
        calibration_refit=False,
        elapsed_seconds=time.monotonic() - started,
        results=rows,
    )
    with (output / "results.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    columns = [k for k in rows[0] if k != "equity_curve"]
    with (output / "results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for baseline in ("cash", "buy_and_hold"):
        subset = [row for row in rows if row["baseline"] == baseline]
        metrics[baseline] = {
            key: math.fsum(row[key] for row in subset) / 60
            for key in (
                "raw_net_return",
                "terminal_financial_Y",
                "transaction_cost",
                "benchmark_excess",
                "turnover",
                "max_drawdown",
            )
        }
    summary_light = {k: v for k, v in summary.items() if k != "results"}
    summary_light["baselines"] = metrics
    with (output / "summary.json").open("x") as stream:
        json.dump(summary_light, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary_light, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
