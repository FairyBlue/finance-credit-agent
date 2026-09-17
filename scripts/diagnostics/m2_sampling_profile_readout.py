"""CPU-only fixed-denominator readout for the 384-trajectory sampling comparison.

Read native per-model state/result grids; do not import model/GPU libraries or
modify the evaluator. Terminal returns need all four siblings in every fixed
window. Cap failures remain observations with no invented terminal outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("sft0", "m1", "m2_eta1", "m2_eta025")
PROFILES = ("current", "qwen_recommended")
GROUPS = {
    "training_windows": tuple(f"formal-train-{i:04d}" for i in (1, 5, 9, 13, 17, 21, 25, 29)),
    "evaluation_windows": tuple(f"formal-train-{i:04d}" for i in (113, 117, 121, 125)),
}
EPISODES = tuple(episode for episodes in GROUPS.values() for episode in episodes)
TERMINAL_STATES = {"COMPLETE", "INCOMPLETE"}
ROW_STATES = TERMINAL_STATES | {"UNSTARTED", "ACTIVE", "FAILED", "ERROR"}
FINAL_CONTROLLER_STATES = {
    "COMPLETE",
    "ALL_CELLS_OBSERVED",
    "ALL_CELLS_OBSERVED_WITH_INCOMPLETE",
}
OUTCOMES = (
    "net_return_pct",
    "Y",
    "cost_USD",
    "final_cash_pct",
    "final_position_count",
)
BEHAVIOR_OUTCOMES = (
    "invalid_requests_per_trajectory",
    "invalid_request_rate_pct",
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def atomic(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load(path, problems, sources):
    if not path.exists():
        return None
    try:
        payload = path.read_bytes()
        sources[str(path)] = hashlib.sha256(payload).hexdigest()
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as error:
        problems.append(f"{path}: {error}")
        return None


def expected_grid():
    return {
        f"full-{episode}-{profile}-sample{sibling}": (group, episode, profile, sibling)
        for group, episodes in GROUPS.items()
        for episode in episodes
        for profile in PROFILES
        for sibling in range(4)
    }


def scalar(row):
    metrics, final = row["terminal_metrics"], row["final_state"]
    equity = final["portfolio_equity"]
    if not finite(equity) or equity <= 0:
        raise ValueError("nonpositive or nonfinite final equity")
    raw_return = row["takeover_net_return"]
    if not finite(raw_return) or not finite(metrics["raw_net_return"]):
        raise ValueError("nonfinite return")
    if not math.isclose(raw_return, metrics["raw_net_return"], rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("cash-start and terminal returns disagree")
    result = {
        "net_return_pct": raw_return * 100,
        "Y": metrics["terminal_score"],
        "cost_USD": metrics["transaction_cost"],
        "final_cash_pct": final["cash"] / equity * 100,
        "final_position_count": len(final["positions"]),
    }
    if not all(finite(value) for value in result.values()):
        raise ValueError("nonfinite terminal outcome")
    return result


def behavior(row):
    requests, invalid = row["requests"], row["invalid"]
    if type(requests) is not int or type(invalid) is not int or not 0 <= invalid <= requests:
        raise ValueError("invalid request counters")
    actions = row["actions"]
    if not isinstance(actions, dict) or any(type(v) is not int or v < 0 for v in actions.values()):
        raise ValueError("invalid action counters")
    fills = row["model_fills"]
    symbols = Counter(fill["symbol"] for fill in fills)
    sides = Counter(fill["side"] for fill in fills)
    costs = [fill["commission"] + fill["slippage_cost"] for fill in fills]
    if not all(finite(value) for value in costs):
        raise ValueError("nonfinite observed fill cost")
    return {
        "requests": requests,
        "invalid": invalid,
        "holds": actions.get("hold", 0),
        "buys": actions.get("buy", 0),
        "sells": actions.get("sell", 0),
        "actions": actions,
        "fills": len(fills),
        "fill_symbols": symbols,
        "fill_sides": sides,
        "fill_cost_USD": math.fsum(costs),
    }


def group_counts(members, planned):
    present = [m for m in members if m is not None]
    observed = [m for m in present if m["status"] != "UNSTARTED"]
    terminal = [m for m in present if m["status"] in TERMINAL_STATES]
    complete = [m for m in present if m["outcome"] is not None]
    behaviors = [m["behavior"] for m in observed if m["behavior"] is not None]
    symbols = sum((b["fill_symbols"] for b in behaviors), Counter())
    counts = {
        "planned": planned,
        "listed": len(present),
        "observed": len(observed),
        "terminal_observed": len(terminal),
        "complete": len(complete),
        "incomplete": sum(m["status"] == "INCOMPLETE" for m in present),
        "missing": planned - len(present),
        "status_counts": dict(Counter(m["status"] for m in present)),
        "failure_counts": dict(Counter(m["failure"] for m in present if m["failure"])),
        "requests_observed": sum(b["requests"] for b in behaviors),
        "invalid_observed": sum(b["invalid"] for b in behaviors),
        "holds_observed": sum(b["holds"] for b in behaviors),
        "buys_observed": sum(b["buys"] for b in behaviors),
        "sells_observed": sum(b["sells"] for b in behaviors),
        "fills_observed": sum(b["fills"] for b in behaviors),
        "actions_observed": dict(sum((Counter(b["actions"]) for b in behaviors), Counter())),
        "fill_sides_observed": dict(sum((b["fill_sides"] for b in behaviors), Counter())),
        "fill_symbols_observed": dict(symbols),
        "unique_filled_symbols_observed": len(symbols),
        "fill_cost_USD_observed": math.fsum(b["fill_cost_USD"] for b in behaviors),
        "behavior_evidence_complete": len(behaviors) == len(observed),
    }
    total_requests = counts["requests_observed"]
    counts["invalid_per_observed_request_pct"] = (
        counts["invalid_observed"] / total_requests * 100 if total_requests else None
    )
    return counts


def cell_readout(base, model, problems, sources):
    folder = base / model
    result_path = folder / "result.json"
    path = result_path if result_path.exists() else folder / "state.json"
    value = load(path, problems, sources) or {}
    raw_rows = value.get("grid", [])
    if not isinstance(raw_rows, list):
        problems.append(f"{model}: grid is not a list")
        raw_rows = []
    expected, rows, seen = expected_grid(), {}, set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            problems.append(f"{model}: trajectory is not an object")
            continue
        identity = raw.get("id")
        if not isinstance(identity, str) or identity not in expected or identity in seen:
            problems.append(f"{model}: unexpected or duplicate trajectory {identity}")
            continue
        seen.add(identity)
        group, episode, profile, sibling = expected[identity]
        if (
            raw.get("context_id") != episode
            or raw.get("profile") != profile
            or raw.get("sampling") != "sample"
            or raw.get("horizon") != "full"
            or raw.get("state_kind") != "cash"
            or raw.get("context", {}).get("group") != group
            or raw.get("identity")
            != {
                "step": 1,
                "sample_index": EPISODES.index(episode),
                "rollout_n": sibling,
                "validate": False,
            }
        ):
            problems.append(f"{model}: profile/context/sampling identity mismatch for {identity}")
            continue
        status = raw.get("status")
        if status not in ROW_STATES:
            problems.append(f"{model}/{identity}: unknown status {status}")
            continue
        item = {"status": status, "failure": raw.get("failure"), "outcome": None, "behavior": None}
        try:
            item["behavior"] = behavior(raw)
            if status == "COMPLETE":
                if raw.get("failure"):
                    raise ValueError("COMPLETE trajectory also reports failure")
                item["outcome"] = scalar(raw)
            elif status == "INCOMPLETE":
                if not isinstance(raw.get("failure"), str) or not raw["failure"]:
                    raise ValueError("INCOMPLETE trajectory lacks failure reason")
                if any(
                    k in raw for k in ("terminal_metrics", "takeover_net_return", "final_state")
                ):
                    raise ValueError("INCOMPLETE trajectory has terminal outcome fields")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            problems.append(f"{model}/{identity}: {error}")
            item["outcome"] = None
        rows[identity] = item
    groups, windows = {}, {}
    for group, episodes in GROUPS.items():
        groups[group], windows[group] = {}, {}
        for profile in PROFILES:
            points, all_members = {}, []
            for episode in episodes:
                members = [rows.get(f"full-{episode}-{profile}-sample{i}") for i in range(4)]
                point = group_counts(members, 4)
                for metric in OUTCOMES:
                    point[metric] = (
                        mean([m["outcome"][metric] for m in members])
                        if point["complete"] == 4
                        else None
                    )
                behavior_ready = (
                    point["terminal_observed"] == 4 and point["behavior_evidence_complete"]
                )
                point["invalid_requests_per_trajectory"] = (
                    point["invalid_observed"] / 4 if behavior_ready else None
                )
                point["invalid_request_rate_pct"] = (
                    point["invalid_per_observed_request_pct"] if behavior_ready else None
                )
                points[episode] = point
                all_members.extend(members)
            aggregate = group_counts(all_members, len(episodes) * 4)
            for metric in OUTCOMES:
                aggregate[metric] = (
                    mean([p[metric] for p in points.values()])
                    if aggregate["complete"] == aggregate["planned"]
                    else None
                )
            for metric in BEHAVIOR_OUTCOMES:
                values = [p[metric] for p in points.values()]
                aggregate[metric] = mean(values) if all(finite(v) for v in values) else None
            groups[group][profile], windows[group][profile] = aggregate, points
    return {
        "worker_status": value.get("status", "NOT_STARTED"),
        "source": str(path) if path.exists() else None,
        "groups": groups,
        "windows": windows,
    }


def paired(a, b, episodes, metrics):
    pairs = []
    for episode in episodes:
        point = {"episode_id": episode}
        for metric in metrics:
            av, bv = a[episode][metric], b[episode][metric]
            point[metric + "_delta"] = av - bv if finite(av) and finite(bv) else None
        pairs.append(point)
    result = {"planned_window_pairs": len(episodes), "pairs": pairs}
    for metric in metrics:
        values = [p[metric + "_delta"] for p in pairs]
        result["usable_" + metric + "_window_pairs"] = sum(finite(value) for value in values)
        result["mean_" + metric + "_delta"] = (
            mean(values) if all(finite(v) for v in values) else None
        )
    returns = [p["net_return_pct_delta"] for p in pairs]
    result["positive_return_windows"] = (
        sum(v > 0 for v in returns) if all(finite(v) for v in returns) else None
    )
    return result


def build(base):
    base = Path(base).resolve()
    problems, sources = [], {}
    state = load(base / "state.json", problems, sources) or {}
    cells = {model: cell_readout(base, model, problems, sources) for model in MODELS}
    profile_comparisons, rl_comparisons = [], []
    for model in MODELS:
        for group, episodes in GROUPS.items():
            points = cells[model]["windows"][group]
            profile_comparisons.append(
                {
                    "model": model,
                    "group": group,
                    "target": "qwen_recommended",
                    "reference": "current",
                    **paired(
                        points["qwen_recommended"],
                        points["current"],
                        episodes,
                        (*OUTCOMES, *BEHAVIOR_OUTCOMES),
                    ),
                }
            )
    for model in MODELS[1:]:
        for group, episodes in GROUPS.items():
            for profile in PROFILES:
                rl_comparisons.append(
                    {
                        "model": model,
                        "reference": "sft0",
                        "group": group,
                        "profile": profile,
                        **paired(
                            cells[model]["windows"][group][profile],
                            cells["sft0"]["windows"][group][profile],
                            episodes,
                            OUTCOMES,
                        ),
                    }
                )
    groups = [
        v for c in cells.values() for profiles in c["groups"].values() for v in profiles.values()
    ]
    totals = {
        key: sum(g[key] for g in groups)
        for key in ("observed", "terminal_observed", "complete", "incomplete")
    }
    all_observed = totals["terminal_observed"] == 384 and not problems
    controller_final = state.get("status") in FINAL_CONTROLLER_STATES
    return {
        "status": "COMPLETE"
        if totals["complete"] == 384 and not problems and controller_final
        else "INCOMPLETE",
        "execution_status": state.get("status", "NOT_STARTED"),
        "at_SGT": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "planned": 384,
        **totals,
        "all_cells_observed": all_observed,
        "final_ready": all_observed and controller_final,
        "observation_status": (
            "ALL_CELLS_OBSERVED"
            if totals["complete"] == 384
            else "ALL_CELLS_OBSERVED_WITH_INCOMPLETE"
        )
        if all_observed
        else "OBSERVATIONS_PENDING",
        "cells": cells,
        "profile_comparisons": profile_comparisons,
        "rl_vs_sft": rl_comparisons,
        "problems": problems,
        "source_sha256": sources,
        "readout_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definitions": {
            "terminal_means": (
                "four siblings equally weighted inside each fixed window; all eight/four "
                "windows required; missing/nonfinite/incomplete is null"
            ),
            "observed_behavior": (
                "request/action/fill counters include partial and capped trajectories; "
                "observed fill cost is not a complete-episode cost mean"
            ),
            "invalid_request_rate_pct": (
                "for paired comparisons, mean of each fully observed window's invalid/request "
                "ratio; the separate invalid_per_observed_request_pct pools all observed requests"
            ),
            "behavior_comparisons": (
                "require four terminal observations per window, including native incomplete; "
                "never pretend a capped trajectory has terminal return"
            ),
            "scope": (
                "one seed, existing overlapping Train windows, same-batch profile comparison; "
                "no historical worker baseline mixed in"
            ),
        },
    }


def number(value):
    return f"{value:.4f}" if finite(value) else "—"


def markdown(data):
    lines = [
        "# 当前与Qwen推荐采样参数：同批交叉评估",
        "",
        f"当前进展：**{data['at_SGT']} SGT（UTC+8）**。",
        f"执行状态：{data['execution_status']}；已终止观察 {data['terminal_observed']}/384，"
        f"完整 {data['complete']}，未完成 {data['incomplete']}。",
        "",
        "每模型每配置：训练日期8窗×4条、验收日期4窗×4条。先同窗等权，再固定窗口等权；任何缺失或cap不补零、不缩小收益分母。",
        "",
        "| 模型 | 日期组 | 配置 | 完整/计划 | 净收益% | Y | 成本USD/窗 | "
        "末期现金% | 非法/请求 | buy/sell/hold | 成交/股票数 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, cell in data["cells"].items():
        for group, profiles in cell["groups"].items():
            for profile, row in profiles.items():
                lines.append(
                    f"| {model} | {group} | {profile} | {row['complete']}/{row['planned']} | "
                    f"{number(row['net_return_pct'])} | {number(row['Y'])} | "
                    f"{number(row['cost_USD'])} | {number(row['final_cash_pct'])} | "
                    f"{row['invalid_observed']}/{row['requests_observed']} | "
                    f"{row['buys_observed']}/{row['sells_observed']}/{row['holds_observed']} | "
                    f"{row['fills_observed']}/{row['unique_filled_symbols_observed']} |"
                )
    lines += [
        "",
        "行为计数包括未完成轨迹；交易次数或股票数增加本身不等于收益改善。",
        "",
        "| 推荐配置−current | 日期组 | 净收益Δpp | Y差 | 成本差USD | "
        "非法次数差/轨迹 | 窗口非法率差pp | 收益改善窗 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in data["profile_comparisons"]:
        wins = p["positive_return_windows"]
        win_count = f"{wins}/{p['planned_window_pairs']}" if wins is not None else "—"
        lines.append(
            f"| {p['model']} | {p['group']} | {number(p['mean_net_return_pct_delta'])} | "
            f"{number(p['mean_Y_delta'])} | {number(p['mean_cost_USD_delta'])} | "
            f"{number(p['mean_invalid_requests_per_trajectory_delta'])} | "
            f"{number(p['mean_invalid_request_rate_pct_delta'])} | {win_count} |"
        )
    lines += [
        "",
        "非法比较需要固定窗口的四条终止观察齐全，可包含cap；收益比较要求四条全部完整。窗口非法率先逐窗计算再等权，与所有请求合并的非法率口径不同。",
        "",
        "| RL−同配置SFT | 日期组 | 配置 | 净收益Δpp | Y差 | 收益改善窗 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in data["rl_vs_sft"]:
        wins = p["positive_return_windows"]
        win_count = f"{wins}/{p['planned_window_pairs']}" if wins is not None else "—"
        lines.append(
            f"| {p['model']} | {p['group']} | {p['profile']} | "
            f"{number(p['mean_net_return_pct_delta'])} | "
            f"{number(p['mean_Y_delta'])} | {win_count} |"
        )
    failures = [
        (m, g, p, r["failure_counts"])
        for m, c in data["cells"].items()
        for g, ps in c["groups"].items()
        for p, r in ps.items()
        if r["failure_counts"]
    ]
    if failures:
        lines += ["", "已记录的失败类型："]
        lines.extend(
            f"- {m} / {g} / {p}: {json.dumps(counts, ensure_ascii=False)}"
            for m, g, p, counts in failures
        )
    lines += [
        "",
        "INCOMPLETE表示轨迹或证据尚未全部完整，与服务执行状态分开。所有窗口仍属于Train，单seed、滚动窗口重叠；本批未训练新模型，不混入旧worker收益。",
        "",
        "逐窗点、固定分母、来源SHA256见readout.json；平面表见summary.csv。",
    ]
    if data["problems"]:
        lines += ["", "证据问题：", *["- " + problem for problem in data["problems"]]]
    return "\n".join(lines) + "\n"


def summary_csv(data):
    fields = (
        "model",
        "group",
        "profile",
        "planned",
        "observed",
        "terminal_observed",
        "complete",
        "incomplete",
        *OUTCOMES,
        "requests_observed",
        "invalid_observed",
        "invalid_per_observed_request_pct",
        "buys_observed",
        "sells_observed",
        "holds_observed",
        "fills_observed",
        "unique_filled_symbols_observed",
        "fill_cost_USD_observed",
        "failure_counts",
        "fill_symbols_observed",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for model, cell in data["cells"].items():
        for group, profiles in cell["groups"].items():
            for profile, row in profiles.items():
                writer.writerow(
                    {
                        **row,
                        "model": model,
                        "group": group,
                        "profile": profile,
                        "failure_counts": json.dumps(row["failure_counts"], sort_keys=True),
                        "fill_symbols_observed": json.dumps(
                            row["fill_symbols_observed"], sort_keys=True
                        ),
                    }
                )
    return stream.getvalue()


def write(base, data):
    base = Path(base)
    atomic(
        base / "readout.json",
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic(base / "LIVE_STATUS.md", markdown(data))
    atomic(base / "summary.csv", summary_csv(data))
    if data["final_ready"]:
        atomic(base / "FINAL_RESULTS.md", markdown(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = build(args.directory)
    if args.write:
        write(args.directory, data)
    print(
        json.dumps(
            {
                key: data[key]
                for key in (
                    "status",
                    "execution_status",
                    "terminal_observed",
                    "complete",
                    "incomplete",
                    "planned",
                    "all_cells_observed",
                    "final_ready",
                    "problems",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
