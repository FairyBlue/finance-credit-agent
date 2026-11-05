#!/usr/bin/env python3
"""Compact readout of recorded financial-evidence endpoints and actual costs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

DEFAULT = Path("runs/m3-financial-evidence-20261030-v1")
ARMS = [f"{arm}_seed{seed}" for seed in (42, 43) for arm in ("rloo", "sham", "m3")]
MODELS = ["original", "oracle_facts", "fixed_retrieval", "common", "sft", *ARMS]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def csv_write(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row)) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: "NA" if v is None else v for key, v in row.items()} for row in records
        )


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def average(values):
    values = [value for value in values if finite(value)]
    return math.fsum(values) / len(values) if values else None


def field(row, name):
    if name in row:
        return row[name]
    verification = row.get("verification") or {}
    if name in verification:
        return verification[name]
    for parent in ("financial_verification", "financial", "diagnostics"):
        nested = row.get(parent) or verification.get(parent) or {}
        if name in nested:
            return nested[name]
    return None


def success(row):
    return row.get("status") == "COMPLETE" and row.get("success") is True


def business(row):
    if row.get("status") != "COMPLETE":
        return False
    explicit = field(row, "business_correct")
    if explicit is not None:
        return explicit is True
    return False  # Do not equate a merely feasible conservative portfolio with correctness.


def aggregate(records, planned):
    correct = [row for row in records if success(row)]
    counts = Counter(row.get("status", "UNKNOWN") for row in records)
    measures = {}
    for name in (
        "fact_correct", "period_correct", "unit_correct", "metric_correct",
        "condition_correct", "portfolio_correct", "evidence_valid",
    ):
        known = [field(row, name) for row in records if field(row, name) is not None]
        measures[f"{name}_N"] = sum(value is True for value in known) if known else None
        measures[f"{name}_observed_N"] = len(known)
    return dict(
        planned_N=planned, observed_N=len(records), completed_N=counts["COMPLETE"],
        successful_N=len(correct), success_rate=len(correct) / planned,
        business_correct_N=sum(business(row) for row in records),
        correct_tool_calls_mean=average([row.get("calls_used") for row in correct]),
        correct_tool_calls_N=len(correct),
        correct_transaction_cost_mean=average([
            (row.get("metrics") or {}).get("transaction_cost") for row in correct
        ]),
        **measures,
        failure_categories=dict(Counter(row.get("failure_category", "unknown") for row in records)),
        status_counts=dict(counts),
    )


def missing_status(model, split, state, test_open):
    if model.endswith("seed43") and (state.get("seed43_decision") or {}).get("status") == "SKIPPED":
        return "SKIPPED"
    if model.startswith(("sham_", "m3_")) and state.get("step3_mode", "").startswith(
        "FROZEN_BASELINE_ONLY"
    ):
        return "SKIPPED"
    if split == "test" and not test_open:
        return "NOT_OPEN_BEFORE_FREEZE"
    return "MISSING_FINAL_RESULT" if state.get("status") == "COMPLETE" else "PENDING"


def compact_progress(root):
    arms = {}
    for path in sorted((root / "training").glob("*/progress.json")):
        data = read(path)
        value = {key: data.get(key) for key in (
            "arm", "seed", "status", "cumulative_tokens", "parent_tokens", "auxiliary_tokens",
            "token_budget", "auxiliary_token_cap", "completed_optimizer_steps",
            "budget_stop_reason", "information_probe", "final_adapter",
            "actor_task_groups", "rollout_groups_per_round", "zero_advantage_check_groups",
        )}
        value["unused_budget_tokens"] = data.get("token_budget", BUDGET) - data["cumulative_tokens"]
        value["rounds"] = [{key: row.get(key) for key in (
            "round", "cumulative_tokens", "tokens", "auxiliary_tokens", "optimizer_updates",
            "completed_optimizer_steps", "parent_successes", "parent_started_trajectories",
                "admitted_task_groups",
        )} for row in data.get("rounds", [])]
        arms[path.parent.name] = value
    return arms


BUDGET = 3_000_000


def comparison_rows(evaluations):
    output = []
    for split in ("dev", "test"):
        present = {
            key: value for (key, partition), value in evaluations.items() if partition == split
        }
        for model, reference in (
            ("common", "original"), ("sft", "common"), ("rloo_seed42", "common"),
            ("rloo_seed43", "common"), ("m3_seed42", "rloo_seed42"),
            ("m3_seed42", "sham_seed42"), ("m3_seed43", "rloo_seed43"),
            ("m3_seed43", "sham_seed43"),
        ):
            if model not in present or reference not in present:
                continue
            a = {row["task_id"]: row for row in present[reference]}
            b = {row["task_id"]: row for row in present[model]}
            if set(a) != set(b):
                continue
            common = [task for task in a if success(a[task]) and success(b[task])]
            before = average([a[task].get("calls_used") for task in common])
            after = average([b[task].get("calls_used") for task in common])
            output.append(dict(
                model=model, reference=reference, split=split, paired_tasks=len(a),
                wins=sum(success(b[t]) and not success(a[t]) for t in a),
                losses=sum(success(a[t]) and not success(b[t]) for t in a),
                common_success_N=len(common), common_success_calls_before=before,
                common_success_calls_after=after,
                common_success_call_reduction=1 - after / before if before else None,
                strict_equal_actual_compute_claim=False,
            ))
    return output


def retired_costs(state):
    inherited = state.get("inherited_common_adapter")
    if not inherited:
        return None
    prior_root = Path(inherited).parent.parent
    supervised = {}
    for name in ("common_adaptation", "teacher_sft"):
        summary = read(prior_root / name / "summary.json")
        if summary:
            supervised[name] = {key: summary.get(key) for key in (
                "status", "completed_optimizer_steps", "cursor", "total_prompt_tokens",
                "total_target_tokens", "elapsed_training_seconds",
            )}
        else:
            steps = rows(prior_root / name / "steps.jsonl")
            if steps:
                supervised[name] = dict(
                    status="RETIRED_PARTIAL", recorded_completed_optimizer_steps=steps[-1]["step"],
                    recorded_completed_examples=steps[-1].get("cursor"),
                    recorded_completed_label_token_uses=sum(
                        step.get("supervised_tokens", 0) for step in steps
                    ),
                    prompt_token_uses=None,
                    unfinished_step_work_not_reconstructed=True,
                )
    evaluation = sum(
        read(path).get("total_tokens", 0)
        for path in (prior_root / "eval").glob("*_dev/summary.json")
    )
    probe = read(prior_root / "probe/summary.json", {})
    return dict(
        source_run=str(prior_root), source_observation_had_id_shortcut=True,
        rl_generation_tokens_charged=state.get("prior_rl_generation_tokens", 0),
        recorded_development_evaluation_generation_tokens=evaluation,
        recorded_initial_probe_generation_tokens=probe.get("total_tokens", 0),
        supervised=supervised, old_rl_and_partial_teacher_are_not_new_endpoints=True,
        excluded_from_new_effect_tables=True,
    )


def run(root):
    root = root.resolve()
    target = root / "summary"
    target.mkdir(parents=True, exist_ok=True)
    state = read(root / "state.json", {})
    frozen = read(root / "frozen_models.json", {})
    test_open = frozen.get("status") == "COMPLETE"
    table, per_type, actual_evaluations = [], [], {}
    evaluation_tokens = 0
    for split in ("dev", "test"):
        for model in MODELS:
            path = root / "eval" / f"{model}_{split}"
            if split == "test" and not test_open:
                table.append(dict(
                    model=model, split=split, status=missing_status(model, split, state, test_open),
                    planned_N=100, successful_N=None,
                ))
                continue
            summary = read(path / "summary.json")
            if summary is None:
                table.append(dict(
                    model=model, split=split, status=missing_status(model, split, state, test_open),
                    planned_N=100, successful_N=None,
                ))
                continue
            records = rows(path / "trajectory_results.jsonl")
            ids = [row["task_id"] for row in records]
            if len(ids) != len(set(ids)) or summary["planned_trajectories"] != 100:
                raise ValueError("duplicate or changed 100-task endpoint denominator")
            stats = aggregate(records, 100)
            row = dict(
                model=model, split=split, status=summary["status"], **stats,
                diagnostic_mode=summary.get("diagnostic_mode", "standard"),
                privileged_diagnostic=model == "oracle_facts",
                same_model_reference=summary.get("same_model_reference"),
                total_tokens=summary.get("total_tokens", 0),
                evaluation_reused=summary.get("evaluation_reused", False),
                evaluation_source=summary.get("evaluation_source"),
            )
            evaluation_tokens += row["total_tokens"]
            table.append(row)
            actual_evaluations[(model, split)] = records
            groups = {}
            for record in records:
                groups.setdefault(record.get("task_type", "unknown"), []).append(record)
            for kind, values in groups.items():
                per_type.append(dict(model=model, split=split, task_type=kind,
                                     **aggregate(values, len(values))))
    arms = compact_progress(root)
    probe = read(root / "probe/summary.json", {})
    supervised = {}
    for key in ("common_adaptation", "teacher_sft"):
        summary = read(root / key / "summary.json")
        if summary:
            supervised[key] = {field: summary.get(field) for field in (
                "status", "examples", "completed_optimizer_steps", "total_prompt_tokens",
                "total_target_tokens", "elapsed_training_seconds", "initial_adapter",
            )}
    cpu = {}
    for split in ("dev", "test"):
        if split == "test" and not test_open:
            continue
        summary = read(root / "baselines" / f"{split}_summary.json")
        if summary:
            cpu[split] = {key: value for key, value in summary.items()
                          if key not in {"results", "trajectories", "tasks"}}
            for policy, values in summary.get("policies", {}).items():
                reference = dict(
                    model=policy, split=split, status="COMPLETE",
                    planned_N=values.get("tasks"), observed_N=values.get("tasks"),
                    completed_N=values.get("tasks"), successful_N=values.get("successes"),
                    success_rate=values.get("success_rate"),
                    business_correct_N=values.get("business_correct"),
                    privileged_diagnostic=True, total_tokens=0,
                    correct_tool_calls_mean=values.get("mean_tool_calls"),
                    correct_transaction_cost_mean=values.get("mean_cost_successful_feasible"),
                )
                for name in (
                    "metric_correct", "period_correct", "unit_correct", "condition_correct",
                    "portfolio_correct", "evidence_valid",
                ):
                    reference[f"{name}_N"] = values.get(name)
                table.append(reference)
    comparisons = comparison_rows(actual_evaluations)
    costs = dict(
        rl_generation_tokens=sum(row.get("cumulative_tokens", 0) for row in arms.values()),
        prior_rl_generation_tokens=state.get("prior_rl_generation_tokens", 0),
        evaluation_generation_tokens=evaluation_tokens,
        initial_probe_generation_tokens=probe.get("total_tokens", 0),
        sft_supervised_prompt_token_uses=sum(
            row.get("total_prompt_tokens") or 0 for row in supervised.values()
        ),
        sft_supervised_target_token_uses=sum(
            row.get("total_target_tokens") or 0 for row in supervised.values()
        ),
        recorded_gpu_stage_seconds=sum(
            stage.get("seconds", 0) for stage in state.get("stages", []) if stage.get("gpu_worker")
        ),
    )
    costs["absolute_rl_generation_tokens"] = (
        costs["rl_generation_tokens"] + costs["prior_rl_generation_tokens"]
    )
    retired = retired_costs(state)
    stage_display = state.get("current_stage") or (
        "COMPLETE" if state.get("status") == "COMPLETE" else "PENDING"
    )
    value = dict(
        status=state.get("status", "PENDING"), current_stage=state.get("current_stage"),
        current_stage_display=stage_display,
        frozen_models=frozen, test_readout_authorized=test_open, terminal_table=table,
        per_type=per_type, paired_success=comparisons, training_arms=arms,
        supervised=supervised, cpu_baselines=cpu, costs=costs,
        seed43_decision=state.get("seed43_decision"),
        information_probe=state.get("information_probe"),
        initial_probe_adapter=state.get("initial_probe_adapter"),
        execution_order_adjustment=read(root / "execution_order_adjustment.json"),
        inherited_common=read(root / "inherited_common.json"),
        inherited_common_adapter=state.get("inherited_common_adapter"),
        arm_token_budget=state.get("arm_token_budget", BUDGET),
        retired_prior_run_costs=retired,
        dataset_profile=read(root / "data/manifest.json", {}).get("profile"),
        decision=read(root / "decision.json"),
        raw_trajectories_embedded=False, strict_equal_actual_compute_claim=False,
    )
    write(target / "readout.json", value)
    csv_write(target / "table.csv", table)
    csv_write(target / "per_type.csv", per_type)
    csv_write(target / "paired_success.csv", comparisons)
    text = [
        "# 财报证据与条件式委托：执行结果", "",
        f"状态：`{value['status']}`；当前阶段：`{stage_display}`。",
        "original保留原模型未适配接口的迁移基线；common仅作一次固定64任务协议适配。",
        "若common标为INHERITED，它继承旧运行的64个Train监督checkpoint，新运行没有"
        "重新进行这64题训练；旧RL与旧部分完整SFT均不作为新候选。",
        "完整教师SFT是独立强监督基线，不是三臂RL初始化。oracle_facts与fixed_retrieval"
        "均使用同一common适配策略；前者显式提供正确事实。",
        "所有数据均来自已记录真实模型轨迹，缺失结果不是零分。", "",
        "## 固定终点", "",
        "| 模型 | 区域 | 状态 | 完整成功/100 | 业务正确 | 指标/期间/单位/触发正确 | 正确任务调用 |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in table:
        counts = "/".join(str(row.get(f"{key}_N", "NA")) for key in (
            "metric_correct", "period_correct", "unit_correct", "condition_correct"
        ))
        calls = row.get("correct_tool_calls_mean")
        text.append(
            f"| {row['model']} | {row['split']} | {row['status']} | "
            f"{row.get('successful_N', 'NA')} | {row.get('business_correct_N', 'NA')} | "
            f"{counts} | {f'{calls:.3f}' if finite(calls) else 'NA'} |"
        )
    text += ["", "## 训练与实际成本", "",
             "| 臂 | 状态 | 实际token | 父/辅助token | 有效更新 |",
             "|---|---|---:|---:|---:|"]
    for key in ARMS:
        arm = arms.get(key)
        if arm:
            text.append(
                f"| {key} | {arm['status']} | {arm['cumulative_tokens']} | "
                f"{arm['parent_tokens']}/{arm['auxiliary_tokens']} | "
                f"{arm['completed_optimizer_steps']} |"
            )
        else:
            status = missing_status(key, "dev", state, test_open)
            text.append(f"| {key} | {status} | NA | NA | NA |")
    text += [
        "", f"RL实际输入＋输出：{costs['rl_generation_tokens']}；起点采样："
        f"{costs['initial_probe_generation_tokens']}；评估新增生成："
        f"{costs['evaluation_generation_tokens']}。SFT监督输入/标签："
        f"{costs['sft_supervised_prompt_token_uses']}/{costs['sft_supervised_target_token_uses']}。",
        f"旧运行已消耗RL生成：{costs['prior_rl_generation_tokens']}；新旧累计RL："
        f"{costs['absolute_rl_generation_tokens']}。每个新臂统一上限："
        f"{value['arm_token_budget']}。1800万是含旧消耗的绝对RL上限，不是项目总成本。"
        "旧标签捷径运行的结果不混入本轮效果表；早停不同实际消耗不称严格等预算或等FLOPs。",
        "确定性相同权重评估若复用，成本为新增0、来源明确；不是新的独立复验。", "",
        "继承checkpoint时，旧标签捷径运行的RL、评估和已完成监督开销在readout的"
        "retired_prior_run_costs单列；部分SFT只报告有记录的已完成步数/标签token，"
        "不伪造未记录的输入开销，也不当成新端点。", "",
        "## 推进与出口", "",
        f"第三步模式：`{state.get('step3_mode', 'PENDING')}`；第二seed："
        f"`{(state.get('seed43_decision') or {}).get('status', 'PENDING')}`，原因："
        f"`{(state.get('seed43_decision') or {}).get('trigger', 'PENDING')}`。",
        f"决定：`{(value.get('decision') or {}).get('decision', 'PENDING')}`。", "",
        "业务正确包含财报事实/期间/单位、计算、条件触发和最终方案/合法不可行证明。"
        "不能将错误财报判断但保守组合可行算作正确信息决策。引用单列。",
        "未运行的M3不构成M3负面对照；辅助分支的业务差异不是算法优势证据。",
        "原始财报来自公开FinQA，委托规则与账户是模拟；不是真实PIT市场收益或部署证据。", "",
        "[完整读数](summary/readout.json) · [固定分母表](summary/table.csv) · "
        "[配对结果](summary/paired_success.csv) · [执行状态](state.json)",
    ]
    (root / "REPORT.md").write_text("\n".join(text) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    value = run(args.output)
    print(json.dumps(dict(status=value["status"], test_authorized=value["test_readout_authorized"],
                          rl_tokens=value["costs"]["rl_generation_tokens"])))


if __name__ == "__main__":
    main()
