#!/usr/bin/env python3
"""Reconstruct observed coverage-evaluation PnL from recorded fills and frozen Train bars.

CPU only. No policy/env replay, market downloads, action counterfactuals or training.
The cash/quantity ledger is reconstructed from actual fills. Historical open/close
bars provide full precision; each used bar is checked against the recorded market
observation and each recorded fill open. All four fixed windows stay in every mean.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/m2-train-coverage-20260916-v2"
DEFAULT_OUT = ROOT / "runs/m2-train-coverage-diagnosis-20260917/behavior"
MODELS = ("sft0", "m1", "m2_eta025")
EPISODES = tuple(f"formal-train-{n:04d}" for n in (113, 117, 121, 125))
CHECKPOINTS = (2, 4, 6, 8)
MONEY_TOLERANCE = 2e-6
MEAN_FIELDS = (
    "net_return_pct",
    "gross_market_return_pct",
    "commission_pct",
    "slippage_pct",
    "cost_pct",
    "gross_market_pnl_USD",
    "commission_USD",
    "slippage_USD",
    "cost_USD",
    "overnight_pnl_USD",
    "intraday_pnl_USD",
    "net_pnl_USD",
    "Y",
    "Y_gross_market",
    "Y_cost",
    "Y_benchmark",
    "Y_risk",
    "risk_penalty",
    "max_drawdown",
    "max_daily_turnover",
    "maximum_position_weight",
    "turnover_excess",
    "position_excess",
    "avg_close_cash_pct",
    "final_cash_pct",
    "fills",
    "buys",
    "sells",
    "holds",
    "invalid",
    "requests",
    "skipped_orders",
    "avg_stock_count",
)


def mean(values):
    return math.fsum(values) / len(values)


def near(a, b, tolerance=MONEY_TOLERANCE, label=""):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > tolerance:
        raise ValueError(f"reconstruction mismatch {label}: {a} vs {b}")
    return abs(a - b)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def csv_text(rows):
    if not rows:
        return ""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
        )
    return out.getvalue()


class Sources:
    def __init__(self, run):
        self.run = run
        self.files = {}
        self.prices = {}

    def add(self, path):
        path = path.resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if key not in self.files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    digest.update(block)
            self.files[key] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        return self.files[key]["sha256"]

    def json(self, path):
        self.add(path)
        return json.loads(path.read_text())

    def bar(self, storage, symbol, day):
        key = (str(storage), symbol, day)
        if key not in self.prices:
            path = storage / "parquet" / symbol / "day" / f"{day}.parquet"
            self.add(path)
            rows = pd.read_parquet(path).to_dict("records")
            if len(rows) != 1 or str(rows[0]["date"]) != day:
                raise ValueError(f"unexpected market bar {path}")
            self.prices[key] = {field: float(rows[0][field]) for field in ("open", "close")}
        return self.prices[key]


def visible_state(interaction):
    observation = interaction["observation"]
    if observation["truncated"] or interaction["response"]["truncated"]:
        raise ValueError("truncated interaction cannot reconstruct exact state")
    messages = json.loads(observation["text"])
    for message in reversed(messages):
        content = message.get("content", "").strip()
        if content.startswith("<tool_response>"):
            return json.loads(
                content.removeprefix("<tool_response>").removesuffix("</tool_response>")
            )
    raise ValueError("no exact portfolio observation")


def load_episodes(run, sources):
    sources.add(run / "prepared/dev.parquet")
    manifest = sources.json(run / "prepared/manifest.json")
    if sources.add(run / "prepared/dev.parquet") != manifest["datasets"]["dev"]["sha256"]:
        raise ValueError("prepared fixed evaluation data hash differs")
    result = {}
    for row in pd.read_parquet(run / "prepared/dev.parquet").to_dict("records"):
        if row["extra_info"]["source_split"] != "train":
            raise ValueError("only original Train evaluation data may be opened")
        env = json.loads(row["env_kwargs"])
        dates = [d[:10] for d in env["episode_dates"]]
        if len(dates) != 21 or min(dates) < env["data_start"] or max(dates) > env["data_end"]:
            raise ValueError("unexpected date grid or non-Train date")
        result[row["extra_info"]["episode_id"]] = env
    if tuple(result) != EPISODES:
        raise ValueError("fixed four evaluation windows changed")
    return result


def analyze_trajectory(raw, model, source, env, sources, calibration):
    identity = raw["identity"]
    if not raw["complete"] or raw["status"] != "COMPLETE" or not identity["validate"]:
        raise ValueError("analysis requires complete validation trajectories")
    metrics = raw["metrics"]
    if metrics["scorer_id"] != "benchmark_scale_v2" or (
        metrics["calibration_sha256"] != env["screening_calibration_sha256"]
    ):
        raise ValueError("scorer identity differs")
    dates = [d[:10] for d in env["episode_dates"]]
    storage = Path(env["storage_root"])
    initial = env["initial_cash"]
    step = identity["step"]
    sibling = identity["rollout_n"]
    lane = "greedy" if sibling == 0 else "sampled"
    episode = raw["episode_id"]
    uid = f"{model}-step{step}-{episode}-{lane}{sibling}"
    meta = {
        "trajectory_id": uid,
        "model": model,
        "policy_step": step,
        "episode_id": episode,
        "sampling": lane,
        "rollout_n": sibling,
    }
    # Retain all actually traded symbols, even if a trajectory ends flat in one.
    symbols = sorted({f["symbol"] for x in raw["interactions"] for f in x["info"].get("fills", [])})
    cash, quantities, previous_index = initial, defaultdict(float), 0
    equities = [initial]
    daily, fills_out = [], []
    counters = Counter()
    residuals = Counter()
    daily_turnovers, cash_ratios, stock_counts, max_weights = [], [], [], []
    invalid_errors = Counter()
    actions = []
    for request_index, interaction in enumerate(raw["interactions"]):
        state = visible_state(interaction)
        info = interaction["info"]
        decision, result = (
            interaction["decision_session_index"],
            interaction["result_session_index"],
        )
        if decision != previous_index or state["session_index"] != result:
            raise ValueError("request/state session alignment changed")
        action = json.loads(
            interaction["response"]["text"]
            .strip()
            .removeprefix("<tool_call>")
            .removesuffix("</tool_call>")
        )
        action_info = {
            "request_index": request_index,
            "decision_session_index": decision,
            "result_session_index": result,
            "tool": action["name"],
            "arguments": action.get("arguments", {}),
            "invalid": bool(info.get("invalid_action")),
        }
        actions.append(action_info)
        counters["requests"] += 1
        if info.get("invalid_action"):
            counters["invalid"] += 1
            invalid_errors[str(info.get("error", "unspecified"))] += 1
        if result == decision:
            if info.get("fills"):
                raise ValueError("nonadvancing request unexpectedly filled")
            near(state["cash"], cash, label="nonadvancing cash")
            continue
        if result != decision + 1:
            raise ValueError("session did not advance exactly once")
        counters[action["name"]] += 1
        counters["skipped_orders"] += len(info.get("skipped_orders", []))
        prior = dict(quantities)
        previous_equity = equities[-1]
        costs = defaultdict(lambda: [0.0, 0.0])
        for fill in info.get("fills", []):
            symbol = fill["symbol"]
            price = sources.bar(storage, symbol, dates[result])
            near(fill["raw_open_price"], price["open"], tolerance=1e-10, label="fill raw open")
            sign = 1 if fill["side"] == "buy" else -1
            delta = sign * fill["shares"]
            cash -= delta * fill["execution_price"] + fill["commission"]
            quantities[symbol] += delta
            if quantities[symbol] < -1e-10:
                raise ValueError("reconstructed short position")
            quantities[symbol] = max(0.0, quantities[symbol])
            near(abs(delta) * fill["execution_price"], fill["notional"], label="fill notional")
            near(
                abs(delta) * abs(fill["execution_price"] - fill["raw_open_price"]),
                fill["slippage_cost"],
                label="fill slippage",
            )
            costs[symbol][0] += fill["commission"]
            costs[symbol][1] += fill["slippage_cost"]
            counters["fills"] += 1
            counters[f"fills_{fill['side']}"] += 1
            fills_out.append(
                {
                    **meta,
                    "request_index": request_index,
                    "session_index": result,
                    "date": dates[result],
                    **fill,
                }
            )
        close_equity = cash + math.fsum(
            quantities[s] * sources.bar(storage, s, dates[result])["close"] for s in symbols
        )
        residuals["equity_USD"] = max(
            residuals["equity_USD"], near(close_equity, info["equity"], label="daily equity")
        )
        residuals["observation_cash_USD"] = max(
            residuals["observation_cash_USD"], near(cash, state["cash"], label="observation cash")
        )
        residuals["observation_equity_USD"] = max(
            residuals["observation_equity_USD"],
            near(close_equity, state["portfolio_equity"], label="observation equity"),
        )
        if state["current_date"][:10] != dates[result]:
            raise ValueError("observed date differs from frozen episode")
        end_weights = []
        for symbol in symbols:
            q0, q1 = prior.get(symbol, 0.0), quantities[symbol]
            prev = sources.bar(storage, symbol, dates[decision])
            now = sources.bar(storage, symbol, dates[result])
            for field in ("open", "close"):
                near(
                    now[field],
                    state["market"][symbol][field],
                    tolerance=5.1e-9,
                    label="rounded observation bar",
                )
            observed_q = state["positions"].get(symbol, {}).get("shares", 0.0)
            residuals["observation_shares"] = max(
                residuals["observation_shares"],
                near(q1, observed_q, tolerance=5.1e-9, label="observed shares"),
            )
            overnight = q0 * (now["open"] - prev["close"])
            intraday = q1 * (now["close"] - now["open"])
            commission, slippage = costs[symbol]
            weight = q1 * now["close"] / close_equity
            end_weights.append(weight)
            daily.append(
                {
                    **meta,
                    "session_index": result,
                    "date": dates[result],
                    "symbol": symbol,
                    "request_index": request_index,
                    "action": action["name"],
                    "action_symbol": action.get("arguments", {}).get("symbol"),
                    "shares_before_open": q0,
                    "shares_after_open": q1,
                    "previous_close": prev["close"],
                    "open": now["open"],
                    "close": now["close"],
                    "overnight_return_pct": (now["open"] / prev["close"] - 1) * 100,
                    "intraday_return_pct": (now["close"] / now["open"] - 1) * 100,
                    "close_to_close_return_pct": (now["close"] / prev["close"] - 1) * 100,
                    "prior_close_weight_pct": q0 * prev["close"] / previous_equity * 100,
                    "end_close_weight_pct": weight * 100,
                    "overnight_pnl_USD": overnight,
                    "intraday_pnl_USD": intraday,
                    "gross_market_pnl_USD": overnight + intraday,
                    "commission_USD": commission,
                    "slippage_USD": slippage,
                    "cost_USD": commission + slippage,
                    "net_pnl_USD": overnight + intraday - commission - slippage,
                    "end_cash_pct": cash / close_equity * 100,
                    "net_contribution_pp": (overnight + intraday - commission - slippage)
                    / initial
                    * 100,
                }
            )
        today = daily[-len(symbols) :] if symbols else []
        residuals["daily_identity_USD"] = max(
            residuals["daily_identity_USD"],
            near(
                math.fsum(x["net_pnl_USD"] for x in today),
                close_equity - previous_equity,
                label="daily PnL identity",
            ),
        )
        daily_turnovers.append(
            math.fsum(f["notional"] for f in info.get("fills", [])) / previous_equity
        )
        cash_ratios.append(cash / close_equity * 100)
        stock_counts.append(sum(q > 1e-10 for q in quantities.values()))
        max_weights.append(max(end_weights, default=0.0))
        equities.append(close_equity)
        previous_index = result
    if previous_index != 20 or len(equities) != 21 or not raw["steps"][-1]["done"]:
        raise ValueError("not one complete twenty-advance trajectory")
    totals = {
        field: math.fsum(d[field] for d in daily)
        for field in (
            "gross_market_pnl_USD",
            "overnight_pnl_USD",
            "intraday_pnl_USD",
            "commission_USD",
            "slippage_USD",
            "cost_USD",
            "net_pnl_USD",
        )
    }
    net_return = equities[-1] / initial - 1
    near(totals["net_pnl_USD"], equities[-1] - initial, label="terminal PnL identity")
    near(net_return, metrics["raw_net_return"], tolerance=1e-12, label="terminal return")
    for field, metric in (
        ("cost_USD", "transaction_cost"),
        ("commission_USD", "commission"),
        ("slippage_USD", "slippage_cost"),
    ):
        near(totals[field], metrics[metric], label=field)
    peak = initial
    max_dd = 0.0
    for equity in equities:
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
    max_turnover, max_weight = max(daily_turnovers), max(max_weights)
    turnover_excess = max(0.0, max_turnover - calibration["turnover_limit"])
    position_excess = max(0.0, max_weight - calibration["position_limit"])
    risk = max(max_dd, turnover_excess, position_excess)
    benchmark = env["benchmark_closes"]
    bench_return = benchmark[-1] / benchmark[0] - 1
    denominator = max(
        statistics.pstdev(b / a - 1 for a, b in zip(benchmark[:-1], benchmark[1:], strict=True)),
        calibration["volatility_floor"],
    )
    y_parts = {
        "Y_gross_market": totals["gross_market_pnl_USD"] / initial / denominator,
        "Y_cost": -totals["cost_USD"] / initial / denominator,
        "Y_benchmark": -bench_return / denominator,
        "Y_risk": -calibration["risk_lambda"] * risk,
    }
    Y = math.fsum(y_parts.values())
    for name, observed, expected in (
        ("drawdown", max_dd, metrics["max_drawdown"]),
        ("max turnover", max_turnover, metrics["max_daily_turnover"]),
        ("max weight", max_weight, metrics["maximum_position_weight"]),
        ("risk", risk, metrics["risk_penalty"]),
        ("denominator", denominator, metrics["score_denominator"]),
        ("benchmark", bench_return, metrics["benchmark_return"]),
        ("Y", Y, raw["terminal_score"]),
    ):
        residuals[name] = near(observed, expected, tolerance=1e-10, label=name)
    if counters["invalid"] != metrics["invalid_count"]:
        raise ValueError("invalid receipts differ from metrics")
    summary = {
        **meta,
        "source": source,
        "initial_cash_USD": initial,
        "net_return_pct": net_return * 100,
        "gross_market_return_pct": totals["gross_market_pnl_USD"] / initial * 100,
        "commission_pct": totals["commission_USD"] / initial * 100,
        "slippage_pct": totals["slippage_USD"] / initial * 100,
        "cost_pct": totals["cost_USD"] / initial * 100,
        **totals,
        "Y": Y,
        **y_parts,
        "risk_penalty": risk,
        "max_drawdown": max_dd,
        "max_daily_turnover": max_turnover,
        "maximum_position_weight": max_weight,
        "turnover_excess": turnover_excess,
        "position_excess": position_excess,
        "benchmark_return_pct": bench_return * 100,
        "score_denominator": denominator,
        "avg_close_cash_pct": mean(cash_ratios),
        "final_cash_pct": cash_ratios[-1],
        "avg_stock_count": mean(stock_counts),
        "fills": counters["fills"],
        "buys": counters["fills_buy"],
        "sells": counters["fills_sell"],
        "holds": counters["hold"],
        "invalid": counters["invalid"],
        "requests": counters["requests"],
        "skipped_orders": counters["skipped_orders"],
        "symbols": symbols,
        "invalid_errors": dict(invalid_errors),
        "reconciliation_absolute_errors": dict(residuals),
    }
    return summary, daily, fills_out, {**meta, "requests": actions}


def summarize(trajectories, daily, episodes):
    by_window = defaultdict(list)
    for row in trajectories:
        by_window[(row["model"], row["policy_step"], row["sampling"], row["episode_id"])].append(
            row
        )
    windows, groups = [], []
    for (model, step, lane, episode), rows in sorted(by_window.items()):
        expected = [0] if lane == "greedy" else [1, 2, 3, 4]
        if sorted(r["rollout_n"] for r in rows) != expected:
            raise ValueError("missing or duplicate sibling; fixed mean unavailable")
        windows.append(
            {
                "model": model,
                "policy_step": step,
                "sampling": lane,
                "episode_id": episode,
                "trajectory_count": len(rows),
                **{key: mean([r[key] for r in rows]) for key in MEAN_FIELDS},
                "invalid_requests": sum(r["invalid"] for r in rows),
                "observed_requests": sum(r["requests"] for r in rows),
                "members": [r["trajectory_id"] for r in rows],
            }
        )
    by_group = defaultdict(list)
    for row in windows:
        by_group[(row["model"], row["policy_step"], row["sampling"])].append(row)
    for (model, step, lane), rows in sorted(by_group.items()):
        if sorted(r["episode_id"] for r in rows) != list(EPISODES):
            raise ValueError("four-window group incomplete")
        groups.append(
            {
                "model": model,
                "policy_step": step,
                "sampling": lane,
                "window_count": 4,
                "trajectory_count": sum(r["trajectory_count"] for r in rows),
                **{key: mean([r[key] for r in rows]) for key in MEAN_FIELDS},
                "invalid_requests": sum(r["invalid_requests"] for r in rows),
                "observed_requests": sum(r["observed_requests"] for r in rows),
            }
        )
    symbols = sorted({d["symbol"] for d in daily})
    daily_groups = defaultdict(list)
    for d in daily:
        daily_groups[
            (
                d["model"],
                d["policy_step"],
                d["sampling"],
                d["episode_id"],
                d["session_index"],
                d["symbol"],
            )
        ].append(d)
    numeric = (
        "shares_before_open",
        "shares_after_open",
        "prior_close_weight_pct",
        "end_close_weight_pct",
        "overnight_pnl_USD",
        "intraday_pnl_USD",
        "gross_market_pnl_USD",
        "commission_USD",
        "slippage_USD",
        "cost_USD",
        "net_pnl_USD",
        "net_contribution_pp",
    )
    window_days = []
    for (model, step, lane, episode), members in sorted(by_window.items()):
        for session in range(1, 21):
            for symbol in symbols:
                rows = daily_groups[(model, step, lane, episode, session, symbol)]
                if len(rows) > len(members):
                    raise ValueError("duplicate daily symbol entry")
                # Untraded symbols have exactly zero quantity, PnL and transaction costs;
                # they remain in the fixed sibling denominator, not dropped as missing.
                window_days.append(
                    {
                        "model": model,
                        "policy_step": step,
                        "sampling": lane,
                        "episode_id": episode,
                        "session_index": session,
                        "date": episodes[episode]["episode_dates"][session][:10],
                        "symbol": symbol,
                        **{key: math.fsum(r[key] for r in rows) / len(members) for key in numeric},
                    }
                )
    return windows, groups, window_days


def contrasts(windows, groups, window_days):
    indexed = {(r["model"], r["policy_step"], r["sampling"]): r for r in groups}
    w_index = {(r["model"], r["policy_step"], r["sampling"], r["episode_id"]): r for r in windows}
    d_index = {
        (
            r["model"],
            r["policy_step"],
            r["sampling"],
            r["episode_id"],
            r["session_index"],
            r["symbol"],
        ): r
        for r in window_days
    }
    symbols = sorted({r["symbol"] for r in window_days})
    results, window_results, day_results, stock_results = [], [], [], []
    specs = [
        (f"{model}_step{start}_to8", model, 8, model, start)
        for model in ("m1", "m2_eta025")
        for start in (2, 4)
    ] + [("m2_minus_m1_step8", "m2_eta025", 8, "m1", 8)]
    for name, target_model, target_step, reference_model, reference_step in specs:
        for lane in ("sampled", "greedy"):
            cid = name + "_" + lane
            target, reference = (
                (target_model, target_step, lane),
                (reference_model, reference_step, lane),
            )
            diff = {key: indexed[target][key] - indexed[reference][key] for key in MEAN_FIELDS}
            prefix = {
                "contrast_id": cid,
                "sampling": lane,
                "target_model": target_model,
                "target_step": target_step,
                "reference_model": reference_model,
                "reference_step": reference_step,
            }
            own_windows, own_days = [], []
            for episode in EPISODES:
                a, b = w_index[(*target, episode)], w_index[(*reference, episode)]
                w = {
                    **prefix,
                    "episode_id": episode,
                    **{f"delta_{k}": a[k] - b[k] for k in MEAN_FIELDS},
                    "contribution_to_group_net_return_pp": (
                        a["net_return_pct"] - b["net_return_pct"]
                    )
                    / 4,
                    "contribution_to_group_Y": (a["Y"] - b["Y"]) / 4,
                }
                own_windows.append(w)
                window_results.append(w)
                for session in range(1, 21):
                    for symbol in symbols:
                        aa, bb = (
                            d_index[(*target, episode, session, symbol)],
                            d_index[(*reference, episode, session, symbol)],
                        )
                        keys = (
                            "shares_before_open",
                            "shares_after_open",
                            "prior_close_weight_pct",
                            "end_close_weight_pct",
                            "overnight_pnl_USD",
                            "intraday_pnl_USD",
                            "gross_market_pnl_USD",
                            "commission_USD",
                            "slippage_USD",
                            "cost_USD",
                            "net_pnl_USD",
                            "net_contribution_pp",
                        )
                        d = {
                            **prefix,
                            "episode_id": episode,
                            "session_index": session,
                            "date": aa["date"],
                            "symbol": symbol,
                            **{f"delta_{key}": aa[key] - bb[key] for key in keys},
                            "contribution_to_group_net_return_pp": (
                                aa["net_contribution_pp"] - bb["net_contribution_pp"]
                            )
                            / 4,
                            "target_prior_close_weight_pct": aa["prior_close_weight_pct"],
                            "reference_prior_close_weight_pct": bb["prior_close_weight_pct"],
                            "target_end_close_weight_pct": aa["end_close_weight_pct"],
                            "reference_end_close_weight_pct": bb["end_close_weight_pct"],
                        }
                        own_days.append(d)
                        day_results.append(d)
            own_stocks = []
            for symbol in symbols:
                rows = [d for d in own_days if d["symbol"] == symbol]
                s = {
                    **prefix,
                    "symbol": symbol,
                    **{
                        key: math.fsum(d[key] for d in rows) / 4
                        for key in (
                            "delta_overnight_pnl_USD",
                            "delta_intraday_pnl_USD",
                            "delta_gross_market_pnl_USD",
                            "delta_commission_USD",
                            "delta_slippage_USD",
                            "delta_cost_USD",
                            "delta_net_pnl_USD",
                        )
                    },
                    "contribution_to_group_net_return_pp": math.fsum(
                        d["contribution_to_group_net_return_pp"] for d in rows
                    ),
                    "delta_avg_close_weight_pp": mean(
                        [d["delta_end_close_weight_pct"] for d in rows]
                    ),
                }
                own_stocks.append(s)
                stock_results.append(s)
            near(
                math.fsum(d["contribution_to_group_net_return_pp"] for d in own_days),
                diff["net_return_pct"],
                tolerance=1e-10,
                label="contrast day-stock sum",
            )
            near(
                diff["gross_market_return_pct"] - diff["cost_pct"],
                diff["net_return_pct"],
                tolerance=1e-10,
                label="contrast gross minus cost",
            )
            near(
                math.fsum(
                    diff[key] for key in ("Y_gross_market", "Y_cost", "Y_benchmark", "Y_risk")
                ),
                diff["Y"],
                tolerance=1e-10,
                label="contrast Y parts",
            )
            results.append(
                {
                    **prefix,
                    "delta": diff,
                    "windows": own_windows,
                    "stocks": own_stocks,
                    "largest_negative_date_stock_contributions": sorted(
                        own_days, key=lambda d: d["contribution_to_group_net_return_pp"]
                    )[:8],
                    "largest_positive_date_stock_contributions": sorted(
                        own_days,
                        key=lambda d: d["contribution_to_group_net_return_pp"],
                        reverse=True,
                    )[:8],
                    "interpretation": (
                        "Difference of observed outcomes/exposures on the same dates; checkpoint "
                        "sampling states and subsequent actions may differ. Arithmetic allocation "
                        "is not a causal training-effect estimate."
                    ),
                }
            )
    return results, window_results, day_results, stock_results


def focus_0125(trajectories, actions, daily, comparison_days, fills):
    focus = [
        r
        for r in trajectories
        if r["episode_id"] == "formal-train-0125" and r["sampling"] == "sampled"
    ]
    first_actions = [
        {key: row[key] for key in ("model", "policy_step", "rollout_n")}
        | {"first_request": row["requests"][0]}
        for row in actions
        if row["episode_id"] == "formal-train-0125"
        and row["sampling"] == "sampled"
        and row["model"] == "m2_eta025"
    ]
    final = {(r["model"], r["rollout_n"]): r for r in focus if r["policy_step"] == 8}
    final_actions = {
        (r["model"], r["rollout_n"]): r["requests"]
        for r in actions
        if r["episode_id"] == "formal-train-0125"
        and r["sampling"] == "sampled"
        and r["policy_step"] == 8
    }
    siblings = []
    for sibling in (1, 2, 3, 4):
        a, b = final[("m2_eta025", sibling)], final[("m1", sibling)]
        different = None
        for left, right in zip(
            final_actions[("m2_eta025", sibling)], final_actions[("m1", sibling)], strict=False
        ):
            if any(
                left[k] != right[k]
                for k in (
                    "tool",
                    "arguments",
                    "invalid",
                    "decision_session_index",
                    "result_session_index",
                )
            ):
                different = {"m2_request": left, "m1_request": right}
                break
        delta = a["net_return_pct"] - b["net_return_pct"]
        siblings.append(
            {
                "rollout_n": sibling,
                "M2_net_return_pct": a["net_return_pct"],
                "M1_net_return_pct": b["net_return_pct"],
                "delta_net_return_pp": delta,
                "contribution_to_0125_mean_pp": delta / 4,
                "contribution_to_four_window_mean_pp": delta / 16,
                "first_observed_request_difference": different,
            }
        )
    prices = {
        (r["date"], r["symbol"]): {
            k: r[k]
            for k in (
                "open",
                "close",
                "previous_close",
                "overnight_return_pct",
                "intraday_return_pct",
                "close_to_close_return_pct",
            )
        }
        for r in daily
    }
    cases = []
    for contrast in ("m2_eta025_step2_to8_sampled", "m2_minus_m1_step8_sampled"):
        matching = [
            r
            for r in comparison_days
            if r["contrast_id"] == contrast and r["episode_id"] == "formal-train-0125"
        ]
        for row in sorted(matching, key=lambda r: r["delta_net_pnl_USD"])[:3]:
            cases.append({**row, "actual_market_prices": prices[(row["date"], row["symbol"])]})
    return {
        "trajectory_results": [
            {
                k: r[k]
                for k in (
                    "trajectory_id",
                    "model",
                    "policy_step",
                    "rollout_n",
                    "net_return_pct",
                    "gross_market_return_pct",
                    "cost_pct",
                    "avg_close_cash_pct",
                    "invalid",
                )
            }
            for r in focus
        ],
        "M2_first_requests": first_actions,
        "final_M2_minus_M1_same_sibling_outcomes": siblings,
        "largest_negative_0125_date_stock_cases": cases,
        "rare_UNH_fills_all_evaluations": [r for r in fills if r["symbol"] == "UNH"],
        "scope": "Observed sibling outcomes; no identical future states or causal claim.",
    }


def markdown(result):
    lines = [
        "# Train coverage：实际持仓、费用与 sampled 后期回落",
        "",
        f"生成时间：{result['created_at_SGT']} SGT（UTC+8）",
        "",
        "本分析只重建已完成的180条验收轨迹；全部为Train内固定四窗，保留每个窗口和sampled的四个siblings。",
        "逐股逐日核对：前日持股×（今开−昨收）＋开盘后持股×（今收−今开）−佣金−滑点＝净损益。",
        "完整精度行情取自原Train OHLC文件，并与轨迹中8位小数行情、实际成交开盘价交叉核对。",
        "",
        "| sampled 比较 | 净收益Δpp | 持仓毛损益Δpp | 费用变化pp | 现金变化pp | Y变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in result["contrasts"]:
        if c["sampling"] != "sampled":
            continue
        d = c["delta"]
        lines.append(
            f"| {c['contrast_id']} | {d['net_return_pct']:+.6f} "
            f"| {d['gross_market_return_pct']:+.6f} | {d['cost_pct']:+.6f} "
            f"| {d['avg_close_cash_pct']:+.3f} | {d['Y']:+.6f} |"
        )
    lines += [
        "",
        "费用变化为正表示花费增加；净收益变化＝行情持仓损益变化−费用变化。现金是20次推进后的日终现金/当时权益均值。",
        "",
    ]
    for c in result["contrasts"]:
        if c["sampling"] != "sampled":
            continue
        lines += [
            f"## {c['contrast_id']}",
            "",
            "按股票归属的净收益变化贡献（四窗平均，百分点）：",
            "",
        ]
        lines += [
            f"- {s['symbol']}：{s['contribution_to_group_net_return_pp']:+.6f}；"
            f"平均日终仓位变化 {s['delta_avg_close_weight_pp']:+.3f} 个百分点。"
            for s in c["stocks"]
        ]
        lines += ["", "各窗净收益变化（百分点；四窗平均为上表结果）：", ""]
        lines += [
            f"- {w['episode_id']}：{w['delta_net_return_pct']:+.6f}；"
            f"行情持仓损益 {w['delta_gross_market_return_pct']:+.6f}，"
            f"费用变化 {w['delta_cost_pct']:+.6f}。"
            for w in c["windows"]
        ]
        lines += ["", "对四窗均值贡献最负的日期/股票（完整列表另存CSV，并未剔除正贡献）：", ""]
        lines += [
            f"- {d['episode_id']} / {d['date']} / {d['symbol']}："
            f"{d['contribution_to_group_net_return_pp']:+.6f} pp；"
            f"该日持仓毛损益变化 {d['delta_gross_market_pnl_USD']:+.2f} USD。"
            for d in c["largest_negative_date_stock_contributions"][:4]
        ]
        yp = c["delta"]
        lines += [
            "",
            f"Y变化分解：持仓行情项 {yp['Y_gross_market']:+.6f}；"
            f"成本项 {yp['Y_cost']:+.6f}；benchmark项 {yp['Y_benchmark']:+.6f}；"
            f"风险项 {yp['Y_risk']:+.6f}。",
            "",
        ]
    focus = result["focus_0125"]
    lines += [
        "## 0125 的实际轨迹分支",
        "",
        "第8轮同编号siblings的实际结果：",
        "",
        "| sibling | M1净收益% | M2净收益% | M2−M1 pp |",
        "|---|---:|---:|---:|",
    ]
    for row in focus["final_M2_minus_M1_same_sibling_outcomes"]:
        lines.append(
            f"| {row['rollout_n']} | {row['M1_net_return_pct']:.6f} "
            f"| {row['M2_net_return_pct']:.6f} | {row['delta_net_return_pp']:+.6f} |"
        )
    lines += ["", "M2首个请求（同一0125起始状态；跨checkpoint的seed身份不同）：", ""]
    for step in (2, 4, 8):
        rows = sorted(
            [r for r in focus["M2_first_requests"] if r["policy_step"] == step],
            key=lambda r: r["rollout_n"],
        )
        summary = "; ".join(
            f"{r['rollout_n']}: {r['first_request']['tool']} "
            f"{r['first_request']['arguments'].get('symbol', '')}"
            for r in rows
        )
        lines.append(f"- 第{step}轮：{summary}。")
    lines += ["", "0125贡献最负的实际日期/股票，关联当日行情和平均日终持仓：", ""]
    for row in focus["largest_negative_0125_date_stock_cases"]:
        p = row["actual_market_prices"]
        lines.append(
            f"- {row['contrast_id']}，{row['date']} {row['symbol']}："
            f"日内涨跌 {p['intraday_return_pct']:+.3f}%，"
            f"日终仓位 {row['reference_end_close_weight_pct']:.3f}%→"
            f"{row['target_end_close_weight_pct']:.3f}%，"
            f"该股票日毛损益变化 {row['delta_gross_market_pnl_USD']:+.2f} USD。"
        )
    lines += [
        "",
        "这些是实际成交形成的路径差异，不能把不同checkpoint的siblings视为相同随机试验。",
        f"全部验收只出现 {len(focus['rare_UNH_fills_all_evaluations'])} 次UNH成交，"
        "明细保留在analysis.json；所有股票仍纳入完整收益。",
        "",
    ]
    lines += ["## 解释边界", ""] + [f"- {v}" for v in result["limits"]]
    lines += [
        "",
        "文件：analysis.json 保存汇总、来源哈希和对账残差；"
        "trajectories.csv、daily_stock_pnl.csv、fills.csv、window_means.csv、group_means.csv、"
        "contrast_windows.csv、contrast_date_stock.csv、contrast_stocks.csv "
        "和 actions.json 保留明细。",
        "",
    ]
    return "\n".join(lines)


def build(run, output):
    sources = Sources(run)
    state = sources.json(run / "state.json")
    if state["status"] != "COMPLETE":
        raise ValueError("analysis is fixed to the complete experiment")
    settings = sources.json(run / "settings.json")
    readout = sources.json(run / "readout.json")
    calibration_doc = sources.json(run / "prepared/reward_calibration.json")
    calibration = calibration_doc["reward"]
    episodes = load_episodes(run, sources)
    for p in (
        "src/env/accounting.py",
        "src/env/trading.py",
        "src/rewards/screening.py",
        "src/rewards/finance.py",
    ):
        sources.add(run / "sources" / p)
    sources.add(Path(__file__))
    trajectories, daily, fills, actions = [], [], [], []
    raw_counts = Counter()
    for model in MODELS:
        seen = set()
        for path in sorted((run / model / "artifacts/screening_evidence").glob("*.jsonl")):
            if sources.add(path) != readout["input_sha256"][str(path.relative_to(run))]:
                raise ValueError("trajectory source differs from completed readout")
            with path.open() as stream:
                for lineno, line in enumerate(stream, 1):
                    raw = json.loads(line)
                    raw_counts[model] += 1
                    identity = raw["identity"]
                    if not identity["validate"]:
                        continue
                    key = tuple(
                        identity[k] for k in ("step", "sample_index", "rollout_n", "validate")
                    )
                    if key in seen:
                        raise ValueError("duplicate evaluation identity")
                    seen.add(key)
                    if raw["episode_id"] != EPISODES[identity["sample_index"]] or identity[
                        "step"
                    ] not in ((0,) if model == "sft0" else CHECKPOINTS):
                        raise ValueError("evaluation outside frozen grid")
                    source = str(path.relative_to(run)) + f":{lineno}"
                    result = analyze_trajectory(
                        raw, model, source, episodes[raw["episode_id"]], sources, calibration
                    )
                    trajectories.append(result[0])
                    daily.extend(result[1])
                    fills.extend(result[2])
                    actions.append(result[3])
        if len(seen) != (20 if model == "sft0" else 80):
            raise ValueError("incomplete model evaluation population")
    windows, groups, window_days = summarize(trajectories, daily, episodes)
    comparison, comparison_windows, comparison_days, comparison_stocks = contrasts(
        windows, groups, window_days
    )
    for group in groups:
        reference = next(
            v["population"]
            for v in readout["cells"][group["model"]]["evaluations"]
            if v["policy_step"] == group["policy_step"] and v["sampling"] == group["sampling"]
        )
        for key, target in (
            ("net_return_pct", "net_return_pct"),
            ("Y", "Y"),
            ("cost_USD", "terminal_cost"),
        ):
            near(
                group[key],
                reference["fixed_window_mean"][target],
                tolerance=1e-9,
                label="previous readout",
            )
    errors = {
        key: max(t["reconciliation_absolute_errors"].get(key, 0.0) for t in trajectories)
        for key in {k for t in trajectories for k in t["reconciliation_absolute_errors"]}
    }
    now = datetime.now(timezone.utc)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "run_directory": str(run),
        "created_at_UTC": now.isoformat(),
        "created_at_SGT": now.astimezone(ZoneInfo("Asia/Singapore")).isoformat(),
        "source_files": sources.files,
        "raw_records_seen": dict(raw_counts),
        "evaluation_trajectories_analyzed": len(trajectories),
        "evaluation_groups": len(groups),
        "stock_day_records": len(daily),
        "fills_recorded": len(fills),
        "maximum_absolute_reconciliation_errors": errors,
        "calibration": calibration,
        "settings_training_coverage": settings["training_coverage"],
        "method": {
            "portfolio_start": "original all-cash 1,000,000 USD; no warm holdings",
            "gross_market_pnl": (
                "q_previous_close*(open_today-close_previous)"
                "+q_after_actual_open_fills*(close_today-open_today)"
            ),
            "net_pnl": "sum gross_market_pnl - actual commission - actual slippage",
            "terminal_Y": (
                "gross_return/denominator - cost_fraction/denominator "
                "- benchmark_return/denominator "
                "- risk_lambda*max(drawdown, turnover_excess, position_excess)"
            ),
            "denominator": (
                "max(population std of fixed benchmark daily returns, original volatility floor)"
            ),
            "averaging": "Four fixed siblings per window; four windows equally. Greedy separate.",
            "stock_day_contributions": (
                "sum to each window net return; window differences divided by four "
                "sum to group net-return difference"
            ),
            "untraded_symbol": (
                "zero position/PnL/cost, retained in fixed denominator; "
                "never treat missing trajectories as zero"
            ),
            "precision": (
                "frozen Train OHLC matched to recorded rounded observations and fill opens; "
                "no env or model replay"
            ),
        },
        "groups": groups,
        "windows": windows,
        "contrasts": comparison,
        "focus_0125": focus_0125(trajectories, actions, daily, comparison_days, fills),
        "limits": [
            "全部日期属于Train；四个滚动窗口有重叠，重复出现的市场日期不是独立样本。",
            "跨checkpoint的policy_step/采样种子不同；观测到的差异混合策略变化和随机轨迹差异。",
            "同窗口同日的持仓差异可以精确分摊实际损益差额；这不是训练更新的因果效应或替代策略收益。",
            "现金无计息，报告其实际占比；没有把未买入股票的上涨自动算成可实现利润。",
            "Y按每窗各自benchmark波动率缩放；四窗平均净收益变化的符号不必等于平均Y变化。",
            "最大回撤/换手/仓位风险项是非加性max；仅单独重算并列示整体风险项，未强行逐股分摊。",
            "未剔除0125或任何轨迹，也未按收益选择checkpoint/提示/交易；日期股贡献排序只是描述实际结果。",
            "学习率与训练日期覆盖同时变化，不能单独归因于其中一个；本分析也不决定M2是否最终优于M1/A1。",
        ],
    }
    artifacts = {
        "trajectories.csv": trajectories,
        "daily_stock_pnl.csv": daily,
        "fills.csv": fills,
        "window_means.csv": windows,
        "group_means.csv": groups,
        "window_date_stock_means.csv": window_days,
        "contrast_windows.csv": comparison_windows,
        "contrast_date_stock.csv": comparison_days,
        "contrast_stocks.csv": comparison_stocks,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in artifacts.items():
        atomic(output / name, csv_text(rows))
    atomic(
        output / "actions.json",
        json.dumps(actions, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic(
        output / "analysis.json",
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic(output / "REPORT.md", markdown(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = build(args.run.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "trajectories": result["evaluation_trajectories_analyzed"],
                "groups": result["evaluation_groups"],
                "reconciliation": result["maximum_absolute_reconciliation_errors"],
                "report": str(args.output / "REPORT.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
