"""CPU-only fixed-window readout for the 240-trajectory policy transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
CELLS = ("sft0", "m1", "m2_eta1", "m2_eta025")
GROUPS = {
    "training_windows": tuple(f"formal-train-{i:04d}" for i in (1, 5, 9, 13, 17, 21, 25, 29)),
    "evaluation_windows": tuple(f"formal-train-{i:04d}" for i in (113, 117, 121, 125)),
}
MODES = {"greedy": ("greedy",), "sample": tuple(f"sample{i}" for i in range(4))}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else None


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def scalar(row):
    metrics, final = row["terminal_metrics"], row["final_state"]
    equity = final["portfolio_equity"]
    if not finite(equity) or equity <= 0:
        raise ValueError("nonpositive final equity")
    values = {
        "net_return_pct": row["takeover_net_return"] * 100,
        "Y": metrics["terminal_score"],
        "cost_USD": metrics["transaction_cost"],
        "final_cash_pct": final["cash"] / equity * 100,
        "benchmark_return_pct": metrics["benchmark_return"] * 100,
    }
    if not all(finite(x) for x in values.values()):
        raise ValueError("nonfinite outcome")
    if not math.isclose(row["takeover_net_return"], metrics["raw_net_return"], abs_tol=1e-10):
        raise ValueError("cash-start takeover and complete-episode returns disagree")
    fills = row["model_fills"]
    return {
        **values,
        "requests": row["requests"],
        "invalid": row["invalid"],
        "holds": sum(
            s["action"] == "hold" and not s["invalid_action"] for s in row["steps_including_prefix"]
        ),
        "fills": len(fills),
        "fill_symbols": dict(Counter(f["symbol"] for f in fills)),
        "final_position_count": len(final["positions"]),
    }


def cell_readout(base, cell_id, problems):
    directory = base / cell_id
    result = read(directory / "result.json") or read(directory / "state.json") or {}
    raw_rows = result.get("grid", [])
    expected = {
        f"full-{episode}-{lane}": (group, episode, mode)
        for group, episodes in GROUPS.items()
        for episode in episodes
        for mode, lanes in MODES.items()
        for lane in lanes
    }
    rows = {}
    for row in raw_rows:
        identity = row.get("id")
        if identity not in expected or identity in rows:
            problems.append(f"{cell_id}: unexpected or duplicate trajectory {identity}")
            continue
        group, episode, mode = expected[identity]
        if (
            row.get("context_id") != episode
            or row.get("sampling") != mode
            or row.get("context", {}).get("group") != group
        ):
            problems.append(f"{cell_id}: inconsistent context/group/mode for {identity}")
            continue
        rows[identity] = row
    groups, windows = {}, {}
    for group, episodes in GROUPS.items():
        groups[group], windows[group] = {}, {}
        for mode, lanes in MODES.items():
            points, observed, complete = {}, [], []
            for episode in episodes:
                members = [rows.get(f"full-{episode}-{lane}") for lane in lanes]
                observed.extend(row for row in members if row)
                valid = []
                for row in members:
                    if row and row.get("status") == "COMPLETE":
                        try:
                            valid.append(scalar(row))
                        except (KeyError, TypeError, ValueError) as error:
                            problems.append(f"{cell_id}/{row['id']}: {error}")
                complete.extend(valid)
                point = {"complete": len(valid), "planned": len(lanes)}
                for metric in (
                    "net_return_pct",
                    "Y",
                    "cost_USD",
                    "final_cash_pct",
                    "benchmark_return_pct",
                    "final_position_count",
                ):
                    point[metric] = (
                        mean([v[metric] for v in valid]) if len(valid) == len(lanes) else None
                    )
                points[episode] = point
            all_complete = len(complete) == len(episodes) * len(lanes)
            aggregate = {
                "planned": len(episodes) * len(lanes),
                "complete": len(complete),
                "observed": len(observed),
                "status_counts": dict(Counter(x["status"] for x in observed)),
                "requests_observed": sum(x.get("requests", 0) for x in observed),
                "invalid_observed": sum(x.get("invalid", 0) for x in observed),
                "complete_fills": sum(x["fills"] for x in complete),
                "complete_holds": sum(x["holds"] for x in complete),
                "fill_symbols_complete": dict(
                    sum((Counter(x["fill_symbols"]) for x in complete), Counter())
                ),
            }
            for metric in (
                "net_return_pct",
                "Y",
                "cost_USD",
                "final_cash_pct",
                "benchmark_return_pct",
                "final_position_count",
            ):
                aggregate[metric] = (
                    mean([p[metric] for p in points.values()]) if all_complete else None
                )
            groups[group][mode], windows[group][mode] = aggregate, points
    return {
        "worker_status": result.get("status", "NOT_STARTED"),
        "groups": groups,
        "windows": windows,
    }


def build(base):
    state = read(base / "state.json") or {}
    problems = []
    cells = {cell: cell_readout(base, cell, problems) for cell in CELLS}
    paired = []
    for cell in CELLS[1:]:
        for group, episodes in GROUPS.items():
            for mode in MODES:
                a, b = cells[cell]["windows"][group][mode], cells["sft0"]["windows"][group][mode]
                pairs = [
                    {
                        "episode": episode,
                        **{
                            metric + "_delta": a[episode][metric] - b[episode][metric]
                            if finite(a[episode][metric]) and finite(b[episode][metric])
                            else None
                            for metric in ("net_return_pct", "Y", "cost_USD", "final_cash_pct")
                        },
                    }
                    for episode in episodes
                ]
                full = all(finite(p["net_return_pct_delta"]) for p in pairs)
                paired.append(
                    {
                        "cell": cell,
                        "group": group,
                        "sampling": mode,
                        "pairs": pairs,
                        "mean_return_delta_pp": mean([p["net_return_pct_delta"] for p in pairs])
                        if full
                        else None,
                        "mean_Y_delta": mean([p["Y_delta"] for p in pairs]) if full else None,
                    }
                )
    prior = read(ROOT / "runs/m2-training-tuning-preparation-20260914/comparison_summary.json")
    old = {
        x["cell_id"]: x
        for x in (prior or {}).get("evaluation_summary", [])
        if x["policy_step"] in (0, 8)
    }
    reload_checks = []
    for cell in CELLS:
        new = cells[cell]["groups"]["evaluation_windows"]["greedy"]["net_return_pct"]
        original = old.get(cell, {}).get("mean_net_return_pct")
        reload_checks.append(
            {
                "cell": cell,
                "old_worker_net_return_pct": original,
                "new_standalone_net_return_pct": new,
                "delta_pp": new - original if finite(new) and finite(original) else None,
            }
        )
    complete = sum(
        g["complete"]
        for cell in cells.values()
        for modes in cell["groups"].values()
        for g in modes.values()
    )
    return {
        "status": "COMPLETE"
        if state.get("status") == "COMPLETE" and complete == 240 and not problems
        else "INCOMPLETE",
        "execution_status": state.get("status", "NOT_STARTED"),
        "at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "complete": complete,
        "planned": 240,
        "cells": cells,
        "paired_vs_sft": paired,
        "reload_checks": reload_checks,
        "problems": problems,
        "readout_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def number(value):
    return f"{value:.4f}" if finite(value) else "—"


def markdown(data):
    completed = data["status"] == "COMPLETE"
    lines = [
        "# 模型训练后日期与解码方式交叉评估",
        "",
        f"当前进展：**{data['at_SGT']} SGT（UTC+8）**。",
        f"状态：{'已完成' if completed else data['execution_status']}；"
        f"完整轨迹 {data['complete']}/240。",
        "",
        "固定窗口先对同窗四条 sampled 等权平均，再对全部窗口等权平均；greedy每窗一条。缺失不补零。",
        "",
        "| 模型 | 日期组 | 模式 | 完整轨迹 | 净收益% | Y | 成本USD/窗 | "
        "非法/请求 | 成交/hold | 末期现金% |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, item in data["cells"].items():
        for group, modes in item["groups"].items():
            for mode, x in modes.items():
                lines.append(
                    f"| {cell} | {group} | {mode} | {x['complete']}/{x['planned']} | "
                    f"{number(x['net_return_pct'])} | {number(x['Y'])} | {number(x['cost_USD'])} | "
                    f"{x['invalid_observed']}/{x['requests_observed']} | "
                    f"{x['complete_fills']}/{x['complete_holds']} | "
                    f"{number(x['final_cash_pct'])} |"
                )
    lines += [
        "",
        "| 模型 − SFT | 训练日期 greedy Δpp | 训练日期 sampled Δpp | "
        "验收日期 greedy Δpp | 验收日期 sampled Δpp |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELLS[1:]:
        points = [x for x in data["paired_vs_sft"] if x["cell"] == cell]
        lines.append(
            "| "
            + cell
            + " | "
            + " | ".join(number(x["mean_return_delta_pp"]) for x in points)
            + " |"
        )
    lines += [
        "",
        "## 与旧训练 worker 的 greedy 读数分开核对",
        "",
        "| 模型 | 旧worker净收益% | 本批重载净收益% | 差值pp |",
        "|---|---:|---:|---:|",
    ]
    for x in data["reload_checks"]:
        lines.append(
            f"| {x['cell']} | {number(x['old_worker_net_return_pct'])} | "
            f"{number(x['new_standalone_net_return_pct'])} | {number(x['delta_pp'])} |"
        )
    lines += [
        "",
        "本批均为已有Train日期，单seed；新批次跨模型固定请求种子，不能当作旧训练相同样本。",
        "所有模型使用共同standalone引擎与128请求上限；旧trainer为256。旧worker与新重载路径差异单列，不用于冒充训练效果。",
        "逐窗结果、完整分母及来源见同目录 readout.json。本次仅评估现存模型，没有参数更新。",
        "",
    ]
    if data["problems"]:
        lines += ["读数问题："] + ["- " + problem for problem in data["problems"]]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = build(args.directory)
    if args.write:
        atomic(
            args.directory / "readout.json",
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        atomic(args.directory / "LIVE_STATUS.md", markdown(data))
        if data["status"] == "COMPLETE":
            atomic(args.directory / "FINAL_RESULTS.md", markdown(data))
    print(
        json.dumps(
            {k: data[k] for k in ("status", "execution_status", "complete", "planned", "problems")}
        )
    )


if __name__ == "__main__":
    main()
