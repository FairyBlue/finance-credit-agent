#!/usr/bin/env python3
"""Read recorded MC training progress; never open reserved task data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_financial_mc import ARMS, generation_tokens


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def readout(root):
    state = read(root / "state.json") or {}
    frozen = read(root / "frozen_models.json") or {}
    test_open = frozen.get("status") == "COMPLETE"
    terminal = []
    models = ["sft", *(arm + "_seed42" for arm in ARMS)]
    models += sorted(key for key in state.get("arms", {}) if key.endswith("seed43"))
    for split in ("dev", "test"):
        for model in models:
            if split == "test" and not test_open:
                item = None
                status = "NOT_OPEN_BEFORE_CONFIRMATION_AND_FREEZE"
            elif split == "test" and model not in frozen.get("models", {}):
                item = None
                status = "NOT_A_PREDECLARED_TEST_ENDPOINT"
            else:
                item = read(root / "eval" / f"{model}_{split}" / "summary.json")
                status = (item or {}).get("status", "PENDING")
                if item is None and state.get("status") in ("COMPLETE", "COMPLETE_LIMITED"):
                    status = "NOT_RUN_AT_RECORDED_EXIT"
            diagnostics = (item or {}).get("outcome_diagnostics", {})
            terminal.append(
                dict(
                    model=model,
                    split=split,
                    status=status,
                    successes=(item or {}).get("successes"),
                    planned=(item or {}).get("planned_trajectories"),
                    complete=(item or {}).get("complete_trajectories"),
                    metric_correct=diagnostics.get("metric_correct"),
                    condition_correct=diagnostics.get("condition_correct"),
                    portfolio_correct=diagnostics.get("portfolio_correct"),
                    tokens=(item or {}).get("total_tokens", 0),
                    evaluation_reused=(item or {}).get("evaluation_reused", False),
                )
            )
    wiring_summary = read(root / "wiring/mc/summary.json") or {}
    wiring = {
        key: value
        for key, value in wiring_summary.items()
        if key not in {"results", "baselines", "anchors", "requests"}
    }
    wiring.update(read(root / "wiring/receipt.json") or {})
    external_sampler = read(root / "wiring/sampler_runtime/state.json") or {}
    stages = state.get("stages", []) + external_sampler.get("stages", [])
    costs = dict(
        wiring_generation_tokens=generation_tokens(root / "wiring"),
        training_generation_tokens=generation_tokens(root / "training"),
        evaluation_generation_tokens=generation_tokens(root / "eval"),
        recorded_gpu_stage_seconds=sum(
            stage.get("seconds", 0) for stage in stages if stage.get("gpu_worker")
        ),
        includes_failed_generation=True,
        token_cost_basis="actual_logged_io_or_conservative_unknown_response_upper_bound",
    )
    costs["total_new_generation_tokens"] = sum(
        costs[key]
        for key in (
            "wiring_generation_tokens",
            "training_generation_tokens",
            "evaluation_generation_tokens",
        )
    )
    return dict(
        status=state.get("status"),
        current_step=state.get("current_step"),
        current_stage=state.get("current_stage"),
        wiring=wiring,
        step_statuses={
            key: state.get(key) for key in ("step1_status", "step2_status", "step3_status")
        },
        arms=state.get("arms", {}),
        terminal_table=terminal,
        costs=costs,
        dev_gate=read(root / "dev_gate.json"),
        confirmation=read(root / "confirmation.json"),
        decision=read(root / "decision.json"),
        test_endpoints_frozen=test_open,
        reserved_task_data_read=False,
    )


def write_report(root, result):
    directory = root / "summary"
    directory.mkdir(exist_ok=True)
    (directory / "readout.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# 有限预算MC状态baseline与分配：实际读数",
        "",
        f"状态：`{result['status']}`；阶段：`{result['current_stage']}`。",
        "",
        "所有训练臂从同一完整SFT开始；MC只替换所选动作前的状态baseline。",
        "全部真实父轨迹请求保留更新，辅助轨迹不作为额外SFT标签。",
        "这是MC适配和分配原型，不是完整VinePPO复现或显式动作Q估计。",
        "",
        "## 第一步接线",
        "",
        f"完整独立MC baseline：{result['wiring'].get('completed_baselines', 'PENDING')}。",
        "恢复前/后Adam有效步数："
        f"{result['wiring'].get('optimizer_steps_before_resume', 'PENDING')} / "
        f"{result['wiring'].get('optimizer_steps_after_resume', 'PENDING')}。",
        "接线父数据复用旧128条强SFT轨迹，新增父生成0；原3308883 token作为历史暴露单列。",
        "其选点清单是仅依据历史动作前缀回放生成，并非当时采样前已记录；",
        "第二步的新父采样才在模型生成当前动作之前记录选点。",
        "",
        "## 固定终点",
        "",
        "| 模型 | 区域 | 状态 | 成功/计划 | 完成 | 指标/触发/执行 | 新增生成token |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for item in result["terminal_table"]:
        report.append(
            f"| {item['model']} | {item['split']} | {item['status']} | "
            f"{item['successes']}/{item['planned']} | {item['complete']} | "
            f"{item['metric_correct']}/{item['condition_correct']}/{item['portfolio_correct']} | "
            f"{item['tokens']} |"
        )
    report += [
        "",
        "## 训练及生成账目",
        "",
        "| 臂 | 状态 | 父token | 辅助token | 总token | 有效更新 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, arm in result["arms"].items():
        report.append(
            f"| {key} | {arm.get('status')} | {arm.get('parent_tokens', 0)} | "
            f"{arm.get('auxiliary_tokens', 0)} | {arm.get('cumulative_tokens', 0)} | "
            f"{arm.get('completed_optimizer_steps', 0)} |"
        )
    costs = result["costs"]
    report += [
        "",
        f"新增接线/训练/评估token：{costs['wiring_generation_tokens']} / "
        f"{costs['training_generation_tokens']} / {costs['evaluation_generation_tokens']}；"
        f"总计{costs['total_new_generation_tokens']}，失败请求保留计费。",
        f"记录的GPU阶段墙钟：{costs['recorded_gpu_stage_seconds']:.1f}秒，含加载，非纯忙碌时间。",
        "相同预算上限不代表实际等token或等FLOPs；优势估计更稳定不自动代表学习更好。",
        "",
        f"seed42工程出口：`{(result['dev_gate'] or {}).get('status', 'PENDING')}`；"
        f"独立seed确认：`{(result['confirmation'] or {}).get('status', 'NOT_STARTED')}`。",
        f"最终决定：`{(result['decision'] or {}).get('decision', 'PENDING')}`。",
        "",
        "[实际状态](state.json) · [完整读数](summary/readout.json) · [决定](decision.json)",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(report))
    if result["decision"]:
        conclusion = [
            "# 本轮学习结果与长期目标回顾",
            "",
            f"出口：`{result['decision']['decision']}`。",
            "",
            "- RQ1：本轮使用独立当前策略续写的MC状态baseline，不把信息遮蔽效应当动作advantage。",
            "- RQ2：均匀MC与候选MC只改变额外续写的分配；具体实际成功率、支持和成本见REPORT。",
            "- RQ3：以完整Dev终点和条件性独立seed/新Test判断；未运行的部分不能算方法失败或成功。",
            "",
            "本轮有限结果不判定RL/credit的普遍上限；公式、局部误差或SFT提升不能替代独立学习增量。",
            "长期Brief是否调整应依据本轮实际出口回顾，不追溯改写旧实验。",
            "",
            "[完整结果](REPORT.md) · [决定](decision.json)",
            "",
        ]
        (root / "CONCLUSION.md").write_text("\n".join(conclusion))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readout(args.output)
    write_report(args.output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "current_step", "current_stage", "costs")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
