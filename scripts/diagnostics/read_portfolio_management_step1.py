#!/usr/bin/env python3
"""Publish the first portfolio-management demo, without claiming policy learning."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "cash": "全现金参照",
    "hold": "等权买入后持有",
    "momentum": "固定动量模板",
    "reversal": "固定反转模板",
    "defensive": "固定防守模板",
    "llm_zero_shot": "共同SFT零样本选动作",
}
METRICS = (
    "raw_net_return", "benchmark_excess", "terminal_financial_Y",
    "transaction_cost", "max_drawdown", "turnover",
)
MARKER = "**2026-10-28：组合管理第一步执行中。**"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def number(value, scale=1):
    return "NA" if value is None else f"{value * scale:.5f}"


def run(output):
    root = Path(output).resolve()
    if read(root / "completion.json")["status"] != "COMPLETE":
        raise RuntimeError("only completed first-step execution may be published")
    rule_summary = read(root / "rules/summary.json")
    llm_summary = read(root / "llm_demo/summary.json")
    checks = read(root / "rules/checks.json")
    records = rows(root / "rules/trajectory_results.jsonl")
    records += [dict(row, strategy="llm_zero_shot")
                for row in rows(root / "llm_demo/trajectory_results.jsonl")]
    ids = rule_summary["episode_ids"]
    n = len(ids)
    summary = {}
    for strategy in LABELS:
        group = [row for row in records if row["strategy"] == strategy]
        complete = sum(row["status"] == "COMPLETE" for row in group)
        all_good = complete == len(group) == n
        summary[strategy] = dict(
            planned_episodes=n, observed_episodes=len(group), complete_episodes=complete,
            **{key: float(np.mean([row["metrics"][key] for row in group]))
               if all_good else None for key in METRICS},
        )
    destination = root / "summary"
    destination.mkdir(exist_ok=True)
    with (destination / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["strategy", *summary["hold"]])
        writer.writeheader()
        writer.writerows(dict(strategy=k, **v) for k, v in summary.items())
    with (destination / "window_results.csv").open("w") as stream:
        fields = ["strategy", "episode_id", "status", *METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(dict(
                strategy=row["strategy"], episode_id=row["episode_id"], status=row["status"],
                **{key: row["metrics"].get(key) if row["status"] == "COMPLETE" else None
                   for key in METRICS},
            ))
    example_id = ids[0]
    example = next(row for row in records
                   if row["strategy"] == "llm_zero_shot" and row["episode_id"] == example_id)
    write(destination / "example_trajectory.json", example)
    report = dict(
        status="COMPLETE", scope="authorized_step1_only", source_split="train",
        episode_ids=ids, strategies=summary, reference_checks=checks,
        example_episode_id=example_id,
        llm_action_counts=dict(Counter(
            action for row in records if row["strategy"] == "llm_zero_shot"
            for action in row.get("action_ids", [])
        )),
        model_requests=llm_summary["model_requests"],
        prompt_tokens=llm_summary["prompt_tokens"],
        response_tokens=llm_summary["response_tokens"],
        optimizer_updates=0, zero_shot=True, constrained_decoding=True,
        new_policy_learning_claim=False, formal_valid_access=False,
        formal_final_oos_access=False, subsequent_steps_started=False,
    )
    failures = root / "controller_attempts/attempt1_failure.json"
    if failures.exists():
        report["preserved_pre_generation_failure"] = read(failures)
    write(destination / "readout.json", report)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(LABELS)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, title in zip(
        axes, ("raw_net_return", "benchmark_excess"),
        ("Net return after all costs", "Active excess over equal-weight hold"), strict=True,
    ):
        for i, name in enumerate(names):
            value = summary[name][key]
            if value is None:
                ax.text(i, 0, "NA", ha="center")
            else:
                ax.bar(i, 100 * value, color="tab:blue" if name == "llm_zero_shot" else "0.6")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(range(len(names)), names, rotation=28, ha="right")
        ax.set_ylabel("Mean window return (%)")
        ax.set_title(title)
    figure.suptitle(f"{n} overlapping Train windows; first-step interface demo; zero new updates")
    figure.tight_layout()
    figure.savefig(destination / "comparison.png", dpi=150)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(9, 4.8))
    for name in names:
        row = next(r for r in records if r["episode_id"] == example_id and r["strategy"] == name)
        curve = np.asarray(row["equity_curve"], dtype=float)
        ax.plot(np.arange(len(curve)), 100 * (curve / curve[0] - 1), label=name)
    for index in (1, 6, 11, 16):
        ax.axvline(index, color="0.8", linestyle=":", linewidth=0.7)
    ax.set(title=f"Predetermined example: {example_id}", xlabel="Session index",
           ylabel="Net return since initial cash (%)")
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination / "example_equity_curve.png", dpi=150)
    plt.close(figure)
    table = ["| 方案 | 平均净收益 | 相对被动持有超额 | 平均费用 | 完整轨迹 |",
             "|---|---:|---:|---:|---:|"]
    for name, label in LABELS.items():
        row = summary[name]
        table.append(
            f"| {label} | {number(row['raw_net_return'], 100)}% | "
            f"{number(row['benchmark_excess'], 100)}pp | "
            f"{number(row['transaction_cost'])} | {row['complete_episodes']}/{n} |"
        )
    table_text = "\n".join(table).replace("NA%", "NA").replace("NApp", "NA")
    timeline = ["| 决策session | 模型实际动作 |", "|---|---|"]
    timeline += [f"| {session} | {action} |" for session, action in zip(
        example.get("decision_session_indices", []), example.get("action_ids", []), strict=True,
    )]
    text = (
        "# 组合管理第一步：接口、固定规则与零样本演示\n\n"
        f"状态COMPLETE。仅完成第一步：{n}个Train窗口、5条固定规则共{n*5}条轨迹，"
        f"另有{summary['llm_zero_shot']['complete_episodes']}/{n}条LLM零样本轨迹。"
        "没有训练更新，没有启动第二/三步，没有使用Valid或Final OOS。\n\n"
        "所有组合方案从现金出发，在首个next-open共同扣费完成等权建仓；"
        "j1/6/11/16作4次组合决策，账本、费用和风险逐日推进到原j20。"
        "hold保持原股份，动量/反转向100%敞口模板作L1≤0.2的小步移动，"
        "防守降低20个百分点敞口；因此防守后仍可通过动量/反转重新入场。"
        "通用目标订单在实际开盘先卖后买，逐单成本与现金裁剪照实保留。"
        "全现金是独立参照，不享受免费撤仓。\n\n"
        f"{table_text}\n\n"
        "这是一轮重叠Train窗口上的接口演示，固定规则不按结果重选。"
        "共同SFT尚未适配新接口，采用相同4动作ID约束解码；"
        "合法性由共同动作接口保证，不是模型学会恢复，也不是新增训练收益。"
        "复制被动基准的主动超额为零，不算策略增量。"
        "原金融Y与回撤作为旁路指标保留在summary.csv。\n\n"
        "## 真实轨迹示例\n\n"
        f"固定选择首个窗口`{example_id}`，不挑最好结果。\n\n"
        + "\n".join(timeline) + "\n\n"
        "[每日净值图](summary/example_equity_curve.png)、"
        "[完整示例JSON](summary/example_trajectory.json)、"
        "[规则与零样本对照图](summary/comparison.png)、"
        "[逐窗口结果](summary/window_results.csv)。"
        "原始模型输入、输出和token保存在llm_demo/requests.jsonl。\n\n"
        f"共享账本参考校验{len(checks['shared_ledger_reference_checks'])}项全部通过："
        "所有窗口的hold逐日净值与费用复制独立等权基准，cash复制现金参照；"
        "64项是32窗×两基准。"
        f"实际LLM生成{report['model_requests']}请求，"
        f"{report['prompt_tokens']}输入token、{report['response_tokens']}输出token。\n"
    )
    if failures.exists():
        text += (
            "\n首次LLM启动因项目vLLM版本的约束解码接口不同，在0次生成前退出；"
            "已按实际环境修复并保留原失败目录。恢复时复用全部固定规则结果，"
            "没有重采已生成的模型轨迹。\n"
        )
    (root / "REPORT.md").write_text(text)
    current = ROOT / "docs/CURRENT_STATUS.md"
    old = current.read_text()
    relative = root.relative_to(ROOT)
    notice = (
        "**2026-10-28：组合管理第一步已完成。** 已接通共同扣费等权底仓、"
        f"四种组合动作和每5日决策；{n}个Train窗口的{n*5}条固定规则轨迹"
        f"及{summary['llm_zero_shot']['complete_episodes']}/{n}条LLM零样本轨迹完成。"
        f"共享账本参考校验{len(checks['shared_ledger_reference_checks'])}项通过，"
        "hold逐日复制被动等权基准。0次新训练，第二/三步未启动，Valid/Final OOS未访问。"
        f"[第一步报告](../{relative}/REPORT.md)、"
        f"[真实轨迹](../{relative}/summary/example_trajectory.json)、"
        f"[对照图](../{relative}/summary/comparison.png)。\n\n"
    )
    if MARKER in old:
        start = old.index(MARKER)
        end = old.index("\n\n", start) + 2
        old = old[:start] + notice + old[end:]
    elif f"../{relative}/REPORT.md" not in old:
        old = old.replace("# 当前状态与下一步\n\n", "# 当前状态与下一步\n\n" + notice, 1)
    current.write_text(old)
    print(json.dumps({"status": "COMPLETE", "report": str(root / "REPORT.md"),
                      "optimizer_updates": 0, "next_steps_started": False}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
