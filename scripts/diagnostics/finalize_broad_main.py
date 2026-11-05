#!/usr/bin/env python3
"""Write the completed experiment's local report even if the chat disconnects."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "SFT": "共同SFT",
    "old_RLOO": "原16步RLOO",
    "broad_RLOO": "32窗RLOO终点",
    "menu_control": "共同菜单控制",
    "learned_ranker": "学习订单评分器",
    "cash": "全现金（CPU）",
    "buy_and_hold": "等权持有（CPU）",
}
DECISIONS = {
    "RANKER_WORTH_FOLLOWUP_WITH_BEHAVIOR_REVIEW": (
        "评分器有值得继续检查的Valid增量，需结合分日期表现和行为成本判断。"
    ),
    "KEEP_NATIVE_RLOO_STOP_CURRENT_RANKER": "保留原生RLOO主线，结束本次评分器变体。",
    "KEEP_SFT_DEMO_NO_RL_INCREMENT": "保留SFT demo，本轮没有建立RL的额外优势。",
    "NO_COMPLETE_LEARNING_ADVANTAGE_OVER_SIMPLE_BASELINES": (
        "未建立完整学习方案对简单基准的优势，下一阶段应转向任务或策略设计。"
    ),
    "INCOMPLETE_NO_CONCLUSION": "记录未完成或失败，不能给出完整固定分母的性能结论。",
}


def formatted(value, scale=1):
    return "NA" if value is None else f"{value * scale:.5f}"


def finalize(directory, report=None):
    root = Path(directory).resolve()
    report = report or json.loads((root / "summary/readout.json").read_text())
    execution = json.loads((root / "results.json").read_text())
    if not report["final_artifacts_allowed"]:
        raise RuntimeError("finalization requires a terminal experiment")
    day = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")
    decision = report["decision"]
    statement = DECISIONS.get(decision, decision)
    table = ["| 方案 | Valid平均净收益 | 金融Y | 完整轨迹 |", "|---|---:|---:|---:|"]
    for name, label in LABELS.items():
        if name in report["valid_arms"]:
            row = report["valid_arms"][name]["overall"]
            count, total = row["complete_parents"], 120
        else:
            row = report["cpu_baselines"][name]
            count, total = row["observed_trajectories"], 60
        table.append(
            f"| {label} | {formatted(row['mean_net_return'], 100)}% | "
            f"{formatted(row['mean_Y'])} | {count}/{total} |"
        )
    table_text = "\n".join(table).replace("NA%", "NA")
    budget = report["sampling_cost"]["totals"]
    free = shutil.disk_usage(ROOT).free / 2**30
    summary = (
        f"**{day}：扩大版三步已执行结束，状态`{report['execution_status']}`。** "
        f"新增RLOO更新{execution['added_updates']}次、累计{execution['cumulative_updates']}次；"
        f"Valid预定600槽，记录{report['observed_valid_parent_slots']}槽、"
        f"{report['complete_valid_parent_slots']}条到原终点。**{statement}**\n\n"
        f"{table_text}\n\n"
        f"固定分母不完整的整组收益保持NA，不以成功子集替代。Valid的60个窗口重叠，"
        f"这是一轮开发比较，不是60个独立市场样本，也不是Final OOS。"
        f"最终读数出口`{decision}`，看到Valid后没有追加训练、换窗口或调reward。\n\n"
        f"实际新增生成{budget['model_requests']}请求、{budget['prompt_tokens']}输入token、"
        f"{budget['response_tokens']}输出token；读取原采样与重复训练使用另列。"
        f"最终模型、失败、轨迹和完整读数保留；完成时磁盘可用约{free:.1f}GiB。"
        f"后台GPU阶段已结束，资源归还记录见state.json。\n\n"
    )
    relative = root.relative_to(ROOT)
    links = (
        f"[完整读数](../{relative}/summary/readout.json)、"
        f"[终点表](../{relative}/summary/summary.csv)、"
        f"[Valid比较图](../{relative}/summary/valid_comparison.png)、"
        f"[Train曲线](../{relative}/summary/train_curve.png)、"
        f"[最终报告](../{relative}/FINAL_REPORT.md)。\n\n"
    )
    final_path = root / "FINAL_REPORT.md"
    if not final_path.exists():
        final_path.write_text(
            "# 扩大版主实验完成报告\n\n"
            + summary
            + "完整数据见本目录 `summary/`、`results.json` 和 `completion.json`。\n"
        )
    current = ROOT / "docs/CURRENT_STATUS.md"
    text = current.read_text()
    marker = "**2026-10-28：扩大版三步已获授权，后台连续执行中。**"
    if marker in text:
        start = text.index(marker)
        end = text.index("\n\n", start) + 2
        text = text[:start] + summary + links + text[end:]
    elif f"../{relative}/FINAL_REPORT.md" not in text:
        title = "# 当前状态与下一步\n\n"
        text = text.replace(title, title + summary + links, 1)
    text = text.replace(
        "**扩大版三步计划（2026-10-28，执行中）：",
        "**扩大版三步计划（2026-10-28，已执行结束）：",
        1,
    )
    current.write_text(text)
    history = ROOT / "docs/history/CREDIT_ORACLE_DIAGNOSTICS.md"
    heading = f"## {day} 扩大版RLOO与订单评分器：完整Valid主实验结束"
    old = history.read_text()
    if heading not in old:
        history.write_text(
            old
            + "\n\n"
            + heading
            + "\n\n"
            + summary
            + f"证据：[完整读数](../../{relative}/summary/readout.json)、"
            f"[最终报告](../../{relative}/FINAL_REPORT.md)。\n"
        )
    return dict(status="FINALIZED", decision=decision, report=str(final_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
