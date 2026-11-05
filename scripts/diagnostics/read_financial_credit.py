#!/usr/bin/env python3
"""Small readout of the bounded strong-SFT credit mechanism experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def compact(value):
    if value is None:
        return None
    return {
        key: item
        for key, item in value.items()
        if key not in {"results", "task_ids", "groups", "anchors"}
    }


def evaluations(root, state):
    frozen = read(root / "frozen_models.json")
    test_allowed = bool(frozen and frozen.get("status") == "COMPLETE")
    signal_closed = state.get("status") == "COMPLETE" and state.get("signal_gate", {}).get(
        "status"
    ) in ("FAIL", "INCONCLUSIVE")
    result = []
    for split in ("dev", "test"):
        for model in ("sft", "rloo_seed42", "sham_seed42", "m3_seed42", "shuffled_seed42"):
            path = root / "eval" / f"{model}_{split}"
            if split == "test" and not test_allowed:
                value = None
                status = "NOT_OPEN_AFTER_SIGNAL_GATE" if signal_closed else "NOT_OPEN_BEFORE_FREEZE"
            else:
                value = read(path / "summary.json")
                arm_status = state.get("arms", {}).get(model, {}).get("status")
                status = (value or {}).get("status") or (
                    "SKIPPED" if arm_status == "SKIPPED" else "PENDING"
                )
                if not value and state.get("status") == "COMPLETE" and arm_status == "SKIPPED":
                    status = "SKIPPED"
                elif not value and signal_closed:
                    status = "NOT_RUN_AFTER_SIGNAL_GATE"
            diag = (value or {}).get("outcome_diagnostics", {})
            result.append(
                dict(
                    model=model,
                    split=split,
                    status=status,
                    successful_N=(value or {}).get("successes"),
                    planned_N=(value or {}).get("planned_trajectories"),
                    metric_correct_N=diag.get("metric_correct"),
                    condition_correct_N=diag.get("condition_correct"),
                    portfolio_correct_N=diag.get("portfolio_correct"),
                    total_tokens=(value or {}).get("total_tokens", 0),
                    evaluation_reused=(value or {}).get("evaluation_reused", False),
                    evaluation_source=(value or {}).get("evaluation_source"),
                )
            )
    return result, test_allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    state = read(root / "state.json") or {}
    probe = read(root / "probe/parents/summary.json")
    paired = read(root / "probe/paired/summary.json")
    gate = read(root / "probe/paired/gate.json")
    terminal, test_allowed = evaluations(root, state)
    previous = root.parent / "m3-financial-evidence-20261030-v2/eval"
    old_dev = read(previous / "sft_dev/summary.json")
    old_test = read(previous / "sft_test/summary.json")
    historical = dict(
        source_run=str(previous.parent),
        dev_successes=(old_dev or {}).get("successes"),
        prior_test_successes=(old_test or {}).get("successes"),
        historical_only=True,
        new_test_result=False,
    )
    arms = state.get("arms", {})
    retired_probe_tokens = state.get("retired_paired_generation_tokens", 0)
    probe_tokens = (
        sum((item or {}).get("total_tokens", 0) for item in (probe, paired)) + retired_probe_tokens
    )
    new_arm_tokens = sum(item.get("cumulative_tokens", 0) for item in arms.values())
    value = dict(
        status=state.get("status"),
        current_stage=state.get("current_stage"),
        initial_adapter=state.get("initial_adapter"),
        parent_probe=compact(probe),
        paired_probe=compact(paired),
        gate=gate,
        training_arms=arms,
        terminal_table=terminal,
        test_readout_authorized=test_allowed,
        historical_sft_reference=historical,
        decision=read(root / "decision.json"),
        costs=dict(
            shared_probe_generation_tokens=probe_tokens,
            retired_paired_generation_tokens=retired_probe_tokens,
            new_arm_generation_tokens=new_arm_tokens,
            training_and_probe_generation_tokens=probe_tokens + new_arm_tokens,
            evaluation_generation_tokens=sum(item["total_tokens"] for item in terminal),
            recorded_gpu_stage_seconds=sum(
                item.get("seconds", 0) for item in state.get("stages", []) if item.get("gpu_worker")
            ),
        ),
    )
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    (summary / "readout.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# 强SFT后的局部credit验证",
        "",
        f"状态：`{value['status']}`；阶段：`{value['current_stage'] or value['status']}`。",
        "",
        "所有候选从同一个完整教师SFT出发，任务、工具、终局reward不变。",
        "父轨迹采样与信息干预探测本身不构成step-credit学习收益。",
        "",
        "## 局部信号探测",
        "",
        f"父轨迹生成token：{(probe or {}).get('total_tokens', 'PENDING')}。",
        f"配对续写生成token：{(paired or {}).get('total_tokens', 'PENDING')}。",
        f"退役失败配对生成token（仍计同一200万上限）：{retired_probe_tokens}。",
        f"信号出口：`{(gate or {}).get('status', 'PENDING')}`。",
        "",
        "局部信号通过后才进入ordinary RL / sham / 真实credit / 打乱credit对照；",
        "失败或预算下无法判断时停止，不为非零信号反复筛题或修改干预。",
        "",
        "## 固定终点",
        "",
        "| 模型 | 区域 | 状态 | 完整成功 | 指标/触发/执行正确 | 新增生成token |",
        "|---|---|---|---:|---|---:|",
    ]
    for item in terminal:
        report.append(
            f"| {item['model']} | {item['split']} | {item['status']} | "
            f"{item['successful_N']} / {item['planned_N']} | "
            f"{item['metric_correct_N']}/{item['condition_correct_N']}/"
            f"{item['portfolio_correct_N']} | {item['total_tokens']} |"
        )
    report += [
        "",
        f"前一轮SFT历史参考：开发{historical['dev_successes']}/100、"
        f"旧Test{historical['prior_test_successes']}/100；不是本轮新保留测试结果。",
        "",
        "## 训练与决定",
        "",
        "| 臂 | 状态 | 实际生成token | 有效更新 |",
        "|---|---|---:|---:|",
    ]
    for arm in ("rloo", "sham", "m3", "shuffled"):
        item = value["training_arms"].get(arm + "_seed42", {})
        report.append(
            f"| {arm} | {item.get('status', 'PENDING')} | "
            f"{item.get('cumulative_tokens', 'NA')} | "
            f"{item.get('completed_optimizer_steps', 'NA')} |"
        )
    report += [
        "",
        f"最终决定：`{(value['decision'] or {}).get('decision', 'PENDING')}`。",
        "",
        "所有父/辅助输入输出记账；额外过程真值不进入模型提示。实际消耗不同不称严格等算力。",
        "若通过gate，四臂首批复用同一强SFT的32×4父轨迹；共享生成只记一次全局成本，",
        "各臂数据暴露单列，不将复用说成免费计算或独立复验。每臂新采样上限275万，",
        "父probe上限500万、配对probe上限200万，含四臂的新训练生成总上限1800万。",
        f"实际共享probe＋新臂生成：{probe_tokens + new_arm_tokens}；"
        f"新增评估生成（另列）：{value['costs']['evaluation_generation_tokens']}。",
        "SFT Dev若精确复用上一轮同权重同任务确定性评估，新增成本0、来源明确。",
        "新Test仅在所有终点与训练选择冻结后开放；旧Test保持历史结果。",
        "",
        "[状态](state.json) · [完整读数](summary/readout.json) · [决定](decision.json)",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(report))
    print(
        json.dumps(
            dict(
                status=value["status"],
                stage=value["current_stage"],
                gate=(gate or {}).get("status"),
            )
        )
    )


if __name__ == "__main__":
    main()
