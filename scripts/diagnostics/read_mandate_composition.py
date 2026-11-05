#!/usr/bin/env python3
"""Summarize recorded composition experiments; never launch evaluation or open tests early."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

CURVE = re.compile(r"^(.*)_curve(\d+)_dev$")
DEFAULT = Path("runs/m3-mandate-composition-20261029-v1")
ARM_NAMES = [f"{arm}_seed{seed}" for seed in (42, 43) for arm in ("rloo", "sham", "m3")]
MODELS = ["original", "sft", *ARM_NAMES]
TOKEN_FIELDS = ("model_requests", "prompt_tokens", "response_tokens", "total_tokens")


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def csv_write(path, records):
    fields = list(dict.fromkeys(k for row in records for k in row)) or ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: "NA" if v is None else v for k, v in row.items()} for row in records)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def fmt(value, multiplier=1):
    return "NA" if not finite(value) else f"{value * multiplier:.3f}"


def cell(value):
    return "NA" if value is None else str(value)


def business(row):
    if row.get("status") != "COMPLETE":
        return False
    if isinstance(row.get("business_correct"), bool):
        return row["business_correct"]
    v = row.get("verification") or {}
    return v.get("business_feasible") is True or v.get("correct_infeasible") is True


def success(row):
    return row.get("status") == "COMPLETE" and row.get("success") is True


def extraction(row):
    value = row.get("constraint_extraction") or {}
    if value.get("available") is True:
        return value.get("complete") is True
    return None


def aggregate(records):
    n = len(records)
    correct = [row for row in records if success(row)]
    costs = [r.get("metrics", {}).get("transaction_cost") for r in correct]
    costs = [v for v in costs if finite(v)]
    calls = [r.get("calls_used", r.get("call_count")) for r in correct]
    calls = [v for v in calls if finite(v)]
    known_extraction = [extraction(r) for r in records if extraction(r) is not None]
    statuses = Counter(r.get("status", "UNKNOWN") for r in records)
    evidence = sum(
        (
            r.get("evidence_valid") is True
            or (r.get("verification") or {}).get("evidence_valid") is True
        )
        and r.get("status") == "COMPLETE"
        for r in records
    )
    return dict(
        planned_N=n,
        observed_N=n - statuses["MISSING"],
        completed_N=statuses["COMPLETE"],
        successful_N=len(correct),
        success_rate_all_planned=len(correct) / n if n else None,
        business_correct_N=sum(business(r) for r in records),
        business_correct_rate=sum(business(r) for r in records) / n if n else None,
        evidence_valid_N=evidence,
        evidence_valid_rate=evidence / n if n else None,
        extraction_observed_N=len(known_extraction),
        extraction_complete_N=sum(known_extraction) if known_extraction else None,
        extraction_complete_rate=sum(known_extraction) / n if known_extraction and n else None,
        incomplete_extraction_but_business_correct_N=sum(
            business(r) and extraction(r) is False for r in records
        )
        if known_extraction
        else None,
        incomplete_extraction_but_task_success_N=sum(
            success(r) and extraction(r) is False for r in records
        )
        if known_extraction
        else None,
        correct_infeasible_N=sum(
            (r.get("verification") or {}).get("correct_infeasible") is True
            for r in records
            if r.get("status") == "COMPLETE"
        ),
        correct_transaction_cost_mean=mean(costs),
        correct_transaction_cost_N=len(costs),
        correct_tool_calls_mean=mean(calls),
        correct_tool_calls_N=len(calls),
        model_or_system_failure_N=sum(
            r.get("status") not in {"COMPLETE", "MISSING"} for r in records
        ),
        missing_N=statuses["MISSING"],
        status_counts=dict(statuses),
    )


def compact_generation(summary):
    return {
        k: summary.get(k)
        for k in (
            *TOKEN_FIELDS,
            "elapsed_seconds",
            "failed_model_requests",
            "incomplete_generations",
            "evaluation_reused",
            "evaluation_source",
            "source_sampled_tokens",
        )
    }


def task_metadata(root, split, cache):
    if split in cache:
        return cache[split]
    name = "development" if split == "dev" else "test"
    # Call only after the freeze gate has authorized test access.
    path = root / "data" / f"{name}.jsonl"
    cache[split] = {
        r["task_id"]: {
            k: r.get(k)
            for k in ("task_type", "base_scenario_id", "pair_id", "pair_variant", "variant")
        }
        for r in rows(path)
    }
    return cache[split]


def normalize(summary, records, metadata):
    ids = summary.get("task_ids") or list(metadata)
    repetitions = summary.get("repetitions", 1)
    expected = [(tid, rep) for tid in ids for rep in range(repetitions)]
    found = {(r["task_id"], r.get("replicate_index", 0)): r for r in records}
    if len(found) != len(records) or not set(found).issubset(set(expected)):
        raise ValueError("duplicate or undeclared evaluation trajectory slots")
    if metadata and set(ids) != set(metadata):
        raise ValueError("evaluation did not declare the full frozen evaluation task set")
    normalized = []
    for tid, rep in expected:
        row = dict(found.get((tid, rep), dict(task_id=tid, replicate_index=rep, status="MISSING")))
        row["task_type"] = (
            row.get("task_type") or metadata.get(tid, {}).get("task_type") or "UNKNOWN"
        )
        normalized.append(row)
    return normalized


def progress_readout(root):
    output = {}
    for path in sorted((root / "training").glob("*/progress.json")):
        value = read(path, {})
        compact = {
            k: value.get(k)
            for k in (
                "arm",
                "seed",
                "status",
                "token_budget",
                "auxiliary_token_cap",
                "cumulative_tokens",
                "parent_tokens",
                "auxiliary_tokens",
                "completed_optimizer_steps",
                "finished_at",
                "final_adapter",
                "stop_reason",
                "early_stop_reason",
            )
        }
        compact["auxiliary_tokens"] = value.get("auxiliary_tokens", value.get("aux_tokens", 0))
        compact["rounds"] = [
            {
                k: r.get(k)
                for k in (
                    "round",
                    "cumulative_tokens",
                    "tokens",
                    "auxiliary_tokens",
                    "optimizer_updates",
                    "completed_optimizer_steps",
                    "collect",
                    "train",
                    "parent_successes",
                    "parent_started_trajectories",
                    "auxiliary_allowance",
                )
            }
            for r in value.get("rounds", [])
        ]
        budget, actual = value.get("token_budget"), value.get("cumulative_tokens")
        compact["unused_token_budget"] = (
            budget - actual if finite(budget) and finite(actual) else None
        )
        compact["progress_path"] = str(path)
        output[path.parent.name] = compact
    return output


def placeholder_status(model, split, state, may_read_test):
    if model.endswith("seed43") and (state.get("seed43_decision") or {}).get("status") == "SKIPPED":
        return "NOT_RUN"
    if model in ARM_NAMES and state.get("step3_mode") == "FROZEN_SFT_BASELINE_ONLY":
        return "NOT_RUN"
    if split == "test" and not may_read_test:
        return "NOT_OPEN_BEFORE_FREEZE"
    return "PENDING"


def paired_rows(evaluations):
    output = []
    by_key = {(e["model"], e["split"]): e for e in evaluations if e["curve_round"] is None}
    for (model, split), after in by_key.items():
        if after["evaluation_status"] != "COMPLETE" or model == "original":
            continue
        references = ["original"] if model == "sft" else ["sft"]
        if model.startswith("m3_seed"):
            references.extend(model.replace("m3_", f"{arm}_", 1) for arm in ("rloo", "sham"))
        for reference in references:
            before = by_key.get((reference, split))
            if not before or before["evaluation_status"] != "COMPLETE":
                continue
            a = {(r["task_id"], r.get("replicate_index", 0)): r for r in before["records"]}
            b = {(r["task_id"], r.get("replicate_index", 0)): r for r in after["records"]}
            if set(a) != set(b):
                raise ValueError("paired task slots differ")
            wins = sum(success(b[k]) and not success(a[k]) for k in a)
            losses = sum(success(a[k]) and not success(b[k]) for k in a)
            common = [k for k in a if success(a[k]) and success(b[k])]
            calls_a = [a[k].get("calls_used", a[k].get("call_count")) for k in common]
            calls_b = [b[k].get("calls_used", b[k].get("call_count")) for k in common]
            output.append(
                dict(
                    split=split,
                    candidate=model,
                    reference=reference,
                    paired_N=len(a),
                    success_wins=wins,
                    success_losses=losses,
                    ties=len(a) - wins - losses,
                    success_difference_pp=100 * (wins - losses) / len(a) if a else None,
                    common_success_N=len(common),
                    common_success_reference_calls=mean(calls_a)
                    if all(map(finite, calls_a))
                    else None,
                    common_success_candidate_calls=mean(calls_b)
                    if all(map(finite, calls_b))
                    else None,
                )
            )
    return output


def declared_pairs(root, may_read_test):
    """Read only the generator's Train/Dev prefix until test endpoints freeze.

    The frozen generator emits one row per base scenario in partition order:
    Train, development, then test. We stop before requesting the first test row,
    rather than loading the whole file and filtering test rows afterwards.
    """
    manifest = read(root / "data/manifest.json", {})
    groups = manifest.get("base_scenario_groups", {})
    path = root / "data/pairing.jsonl"
    if not path.exists() or not groups:
        return {}
    counts = {split: len(groups.get(split, [])) for split in ("train", "development", "test")}
    limit = counts["train"] + counts["development"] + (counts["test"] if may_read_test else 0)
    permitted = {"train", "development", "test"} if may_read_test else {"train", "development"}
    output = {"dev": [], **({"test": []} if may_read_test else {})}
    observed = Counter()
    with path.open() as stream:
        for _ in range(limit):
            line = stream.readline()
            if not line:
                raise ValueError("declared pairing prefix incomplete")
            row = json.loads(line)
            split = row["split"]
            if split not in permitted:
                raise ValueError("pairing file violates frozen generator partition order")
            observed[split] += 1
            if split != "train":
                output["dev" if split == "development" else "test"].append(row)
    for split in permitted:
        if observed[split] != counts[split]:
            raise ValueError("declared pairing counts differ from manifest")
    return output


def paired_business_sensitivity(root, evaluations, may_read_test):
    pairs = declared_pairs(root, may_read_test)
    output = []
    for e in evaluations:
        if e["curve_round"] is not None or e["evaluation_status"] != "COMPLETE":
            continue
        if e["split"] not in pairs:
            continue
        indexed = {(r["task_id"], r.get("replicate_index", 0)): r for r in e["records"]}
        sensitive = [p for p in pairs[e["split"]] if p.get("business_sensitive") is True]
        counts = Counter()
        selected = []
        for pair in sensitive:
            if len(pair["task_ids"]) != 2:
                raise ValueError("business-sensitive pair must have exactly two versions")
            values = [indexed.get((tid, 0), {"status": "MISSING"}) for tid in pair["task_ids"]]
            selected.extend(values)
            passed = sum(success(r) for r in values)
            counts[("neither", "one", "both")[passed]] += 1
            counts["both_business_correct"] += all(business(r) for r in values)
            counts["incomplete_or_system_failure"] += any(
                r.get("status") != "COMPLETE" for r in values
            )
        extracted = [extraction(r) for r in selected if extraction(r) is not None]
        n = len(selected)
        output.append(
            dict(
                model=e["model"],
                split=e["split"],
                reference_total_pairs=len(pairs[e["split"]]),
                reference_business_sensitive_pairs=len(sensitive),
                both_task_successful_pairs=counts["both"],
                only_one_task_successful_pairs=counts["one"],
                neither_task_successful_pairs=counts["neither"],
                both_business_correct_pairs=counts["both_business_correct"],
                incomplete_or_system_failure_pairs=counts["incomplete_or_system_failure"],
                planned_sensitive_trajectories=n,
                extraction_available_N=len(extracted),
                extraction_complete_N=sum(extracted) if extracted else None,
                extraction_complete_rate_all_sensitive_trajectories=(
                    sum(extracted) / n if extracted and n else None
                ),
                incomplete_extraction_but_business_correct_N=sum(
                    extraction(r) is False and business(r) for r in selected
                )
                if extracted
                else None,
                reference_unique_trade_vector_match_required=False,
            )
        )
    return output


def cpu_baselines(root, may_read_test):
    summaries, table, by_type = {}, [], []
    for split in ("dev", "test"):
        if split == "test" and not may_read_test:
            continue
        summary = read(root / "baselines" / f"{split}_summary.json")
        if not summary:
            continue
        summaries[split] = {k: v for k, v in summary.items() if k not in {"results", "policies"}}
        for policy, value in summary.get("policies", {}).items():
            n, passed = value.get("tasks"), value.get("successes")
            if not finite(n) or not finite(passed):
                continue
            feasible, infeasible = value.get("business_feasible"), value.get("correct_infeasible")
            business_n = feasible + infeasible if finite(feasible) and finite(infeasible) else None
            table.append(
                dict(
                    model=policy,
                    model_kind="CPU privileged" if policy.startswith("privileged") else "CPU rule",
                    split=split,
                    evaluation_status=summary.get("status", "COMPLETE"),
                    planned_N=n,
                    observed_N=n,
                    completed_N=n,
                    successful_N=passed,
                    success_rate_all_planned=passed / n,
                    business_correct_N=business_n,
                    evidence_valid_N=value.get("evidence_valid"),
                    extraction_observed_N=value.get("extraction_observed_N"),
                    extraction_complete_N=value.get("extraction_complete_N"),
                    extraction_complete_rate=value.get("extraction_complete_rate"),
                    incomplete_extraction_but_task_success_N=value.get(
                        "incomplete_extraction_but_task_success_N"
                    ),
                    correct_transaction_cost_mean=value.get("mean_cost_successful_feasible"),
                    correct_transaction_cost_N=value.get("successful_feasible_tasks"),
                    correct_tool_calls_mean=value.get("mean_tool_calls") if passed == n else None,
                    correct_tool_calls_N=passed if passed == n else None,
                    model_or_system_failure_N=0,
                    missing_N=0,
                    total_tokens=0,
                )
            )
            by_type.extend(
                dict(
                    model=policy,
                    split=split,
                    task_type=k,
                    planned_N=v["tasks"],
                    successful_N=v["successes"],
                    success_rate_all_planned=v["successes"] / v["tasks"],
                )
                for k, v in value.get("task_types", {}).items()
            )
    return summaries, table, by_type


def report_markdown(report):
    state = report["controller_state"]
    lines = [
        "# 组合约束泛化与有效信息选择：执行与结果",
        "",
        f"状态：`{state.get('status', 'PENDING')}`；"
        f"当前阶段：`{state.get('current_stage', 'PENDING')}`。",
        "首版共同SFT作为original，本轮新教师SFT作为sft。所有结果来自真实已记录轨迹；未跑项不是零分。",
        "",
        (
            "所有候选已冻结，本表可包含冻结后运行的新保留测试。"
            if report["test_readout_authorized"]
            else "冻结尚未完成；本报告未打开test任务或test结果。"
        ),
        "",
        "## 固定终点结果",
        "",
        "| 模型 | 区域 | 状态 | 成功/预定N | 成功率% | 业务正确/证据有效 | "
        "条款完整/可核验N | 漏项仍成功 | 正确工具调用（N） | 正确成交费用（N） |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["terminal_table"]:
        complete = r.get("evaluation_status") == "COMPLETE"
        successes = (
            f"{r.get('successful_N')}/{r.get('planned_N')}" if complete else r["evaluation_status"]
        )
        lines.append(
            f"| {r['model']} | {r['split']} | {r['evaluation_status']} | {successes} | "
            f"{fmt(r.get('success_rate_all_planned'), 100)} | "
            f"{cell(r.get('business_correct_N'))}/{cell(r.get('evidence_valid_N'))} | "
            f"{cell(r.get('extraction_complete_N'))}/{cell(r.get('extraction_observed_N'))} | "
            f"{cell(r.get('incomplete_extraction_but_task_success_N'))} | "
            f"{fmt(r.get('correct_tool_calls_mean'))}（{cell(r.get('correct_tool_calls_N'))}） | "
            f"{fmt(r.get('correct_transaction_cost_mean'))}"
            f"（{cell(r.get('correct_transaction_cost_N'))}） |"
        )
    lines.extend(
        [
            "",
            "条款完整度缺少真实提交约束时标NA；完整任务成功仍要求业务与证据共同通过。"
            "费用只对正确且有实际成交指标的任务取均值，正确不可行声明不擅自记成零成本成交。",
        ]
    )
    if report["paired_business_sensitivity"]:
        lines.extend(
            [
                "",
                "## 业务敏感配对的实际表现",
                "",
                "只在离线参考确认单条内容变化确实改变业务的配对上统计。"
                "两版本各自按原独立验证器判定，不要求交易向量复制唯一oracle。",
                "",
                "| 模型 | 区域 | 敏感配对N | 全对/仅一对/都错 | 业务两题均正确 | "
                "条款完整/可核验/预定轨迹 | 漏项仍业务正确 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for p in report["paired_business_sensitivity"]:
            lines.append(
                f"| {p['model']} | {p['split']} | {p['reference_business_sensitive_pairs']} | "
                f"{p['both_task_successful_pairs']}/{p['only_one_task_successful_pairs']}/"
                f"{p['neither_task_successful_pairs']} | {p['both_business_correct_pairs']} | "
                f"{cell(p['extraction_complete_N'])}/{p['extraction_available_N']}/"
                f"{p['planned_sensitive_trajectories']} | "
                f"{cell(p['incomplete_extraction_but_business_correct_N'])} |"
            )
    lines.extend(
        [
            "",
            "## 实际RL预算与有效更新",
            "",
            "| 训练臂 | 状态 | 实际总token | 父/辅助token | 上限/尾余 | 有效更新 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in ARM_NAMES:
        p = report["training_arms"].get(model)
        if not p:
            status = placeholder_status(model, "dev", state, report["test_readout_authorized"])
            lines.append(f"| {model} | {status} | NA | NA | NA | NA |")
        else:
            lines.append(
                f"| {model} | {p.get('status')} | {p.get('cumulative_tokens')} | "
                f"{p.get('parent_tokens')}/{p.get('auxiliary_tokens')} | "
                f"{p.get('token_budget')}/{p.get('unused_token_budget')} | "
                f"{p.get('completed_optimizer_steps')} |"
            )
    costs = report["costs"]
    lines.extend(
        [
            "",
            f"已运行训练臂RL实际输入＋输出生成token：**{costs['rl_generation_tokens']}**。"
            f"评估新增生成token：{costs['evaluation_generation_tokens']}；起点探测生成token："
            f"{cell(costs['initial_probe_generation_tokens'])}。SFT监督使用单列，不能把RL预算称为项目总成本。",
            "预算上限最多每臂每seed300万、六臂1800万；完整回合准入造成的实际尾余分别报告。"
            "相同上限不等于实际等token或等FLOPs；所有辅助续写均计入其臂预算，辅助结果不作为教师。",
            "",
            "## 推进与出口",
            "",
            f"第三步模式：`{state.get('step3_mode', 'PENDING')}`。第二seed决策："
            f"`{(state.get('seed43_decision') or {}).get('status', 'PENDING')}`，触发："
            f"`{(state.get('seed43_decision') or {}).get('trigger', 'PENDING')}`。",
        ]
    )
    decision = report.get("decision")
    if decision:
        lines.extend(
            [
                "",
                f"最终决定：`{decision.get('decision', decision.get('status', 'PENDING'))}`。",
                decision.get("headline_zh", ""),
            ]
        )
    if state.get("status") == "COMPLETE" and state.get("step3_mode") == "FROZEN_SFT_BASELINE_ONLY":
        lines.extend(
            [
                "",
                "本轮普通RLOO首批无有效优势，0次更新，按计划结束；sham、M3与第二seed未训练。"
                "因此本轮无法比较局部credit机制，不是M3的负面训练结果。"
                "零步RLOO的确定性评估复用同权重新SFT轨迹，新增生成token为0，"
                "既不重复记采样成本，也不计作新的独立复验或RL学习收益。",
            ]
        )
    if report.get("conclusion_available"):
        lines.extend(["", "[简短执行结论](CONCLUSION.md)"])
    if report.get("examples_status") == "COMPLETE":
        lines.extend(
            [
                "",
                "[真实同题失败→成功示例](examples/README.md) · "
                "[固定端点错误归类](summary/endpoint_error_analysis.json)",
            ]
        )
    lines.extend(
        [
            "",
            "本轮保持模拟委托、真实历史已知报价和实际账本核验；不是未来收益或真实部署证明。"
            "局部业务credit与纯引用变化分开，不将删文档的全部损失称作金融决策贡献。"
            "配对版本、改写与跨seed重复不增加独立基础情景数。未以测试选择模型，"
            "未触及旧市场Final OOS，也未据此宣称算法首创。",
            "",
            "[固定分母明细](summary/table.csv) · [分类结果](summary/per_type.csv) · "
            "[配对成功与共同成功成本](summary/paired_success.csv) · "
            "[业务敏感配对结果](summary/paired_business_sensitivity.csv) · "
            "[实际累计token曲线](summary/development_token_curve.csv)",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output):
    root = Path(output).resolve()
    state = read(root / "state.json", {"status": "PENDING"})
    frozen = read(root / "frozen_models.json")
    may_read_test = frozen is not None and frozen.get("status") == "COMPLETE"
    progress = progress_readout(root)
    metadata, evaluations = {}, []
    for folder in sorted((root / "eval").glob("*")):
        split = "test" if folder.name.endswith("_test") else "dev"
        if split == "test" and not may_read_test:
            continue
        summary = read(folder / "summary.json")
        if not summary:
            continue
        curve = CURVE.match(folder.name)
        model = curve[1] if curve else folder.name.rsplit("_", 1)[0]
        idx = int(curve[2]) if curve else None
        path = folder / "trajectory_results.jsonl"
        records = rows(path) if path.exists() else summary.get("results", [])
        normalized = normalize(summary, records, task_metadata(root, split, metadata))
        metrics = aggregate(normalized)
        status = summary.get("status", "PENDING")
        if status != "COMPLETE":
            metrics["success_rate_all_planned"] = None
        rounds = progress.get(model, {}).get("rounds", [])
        matched = next((r for r in rounds if r.get("round") == idx), {})
        cumulative = (
            matched.get("cumulative_tokens")
            if idx is not None
            else progress.get(model, {}).get("cumulative_tokens")
        )
        evaluations.append(
            dict(
                model=model,
                split=split,
                curve_round=idx,
                evaluation_status=status,
                aggregate=metrics,
                generation=compact_generation(summary),
                records=normalized,
                cumulative_training_tokens=cumulative,
                summary_path=str(folder / "summary.json"),
            )
        )
    table, per_type, curves = [], [], []
    for e in evaluations:
        metrics = {k: v for k, v in e["aggregate"].items() if k != "status_counts"}
        row = dict(
            model=e["model"],
            model_kind="LLM",
            split=e["split"],
            evaluation_status=e["evaluation_status"],
            **metrics,
            **e["generation"],
            cumulative_training_tokens=e["cumulative_training_tokens"],
        )
        if e["curve_round"] is not None:
            curves.append(dict(round=e["curve_round"], **row))
        else:
            table.append(row)
            for kind in sorted({r["task_type"] for r in e["records"]}):
                value = aggregate([r for r in e["records"] if r["task_type"] == kind])
                per_type.append(
                    dict(
                        model=e["model"],
                        split=e["split"],
                        task_type=kind,
                        **{k: v for k, v in value.items() if k != "status_counts"},
                    )
                )
    present = {(r["model"], r["split"]) for r in table}
    for model in MODELS:
        for split in ("dev", "test"):
            if (model, split) not in present:
                table.append(
                    dict(
                        model=model,
                        model_kind="LLM",
                        split=split,
                        evaluation_status=placeholder_status(model, split, state, may_read_test),
                        planned_N=200,
                        observed_N=None,
                        successful_N=None,
                        success_rate_all_planned=None,
                    )
                )
    baseline_summary, baseline_table, baseline_types = cpu_baselines(root, may_read_test)
    table.extend(baseline_table)
    per_type.extend(baseline_types)
    table.sort(
        key=lambda r: (
            r["split"],
            MODELS.index(r["model"]) if r["model"] in MODELS else len(MODELS),
            r["model"],
        )
    )
    probe = read(root / "probe/summary.json", {})
    sft = read(root / "common_sft/summary.json", {})
    settings = read(root / "common_sft/settings.json", {})
    report = dict(
        status=state.get("status"),
        controller_state={k: v for k, v in state.items() if k not in {"stages", "arms"}},
        recorded_stage_count=len(state.get("stages", [])),
        frozen_models=frozen,
        test_readout_authorized=may_read_test,
        terminal_table=table,
        per_type=per_type,
        development_token_curve=curves,
        paired_success=paired_rows(evaluations),
        paired_business_sensitivity=paired_business_sensitivity(root, evaluations, may_read_test),
        training_arms=progress,
        cpu_baselines=baseline_summary,
        teacher_summary=read(root / "teacher/summary.json"),
        dataset_manifest=read(root / "data/manifest.json"),
        composition_validation=read(root / "composition_validation_checks.json"),
        examples_status="COMPLETE" if (root / "examples/comparison.json").exists() else "PENDING",
        conclusion_available=(root / "CONCLUSION.md").exists(),
        common_sft={
            k: sft.get(k)
            for k in ("status", "completed_optimizer_steps", "cursor", "elapsed_seconds")
        },
        costs=dict(
            rl_generation_tokens=sum(p.get("cumulative_tokens") or 0 for p in progress.values()),
            evaluation_generation_tokens=sum(
                e["generation"].get("total_tokens") or 0 for e in evaluations
            ),
            initial_probe_generation_tokens=probe.get("total_tokens"),
            sft_supervised_prompt_token_uses=settings.get("total_prompt_tokens"),
            sft_supervised_target_token_uses=settings.get("total_target_tokens"),
            recorded_gpu_stage_seconds=math.fsum(
                s["seconds"]
                for s in state.get("stages", [])
                if finite(s.get("seconds")) and s.get("gpu_worker", True)
            ),
        ),
        seed43_decision=state.get("seed43_decision"),
        decision=read(root / "decision.json"),
        raw_trajectories_embedded=False,
        model_or_test_selection_performed_by_readout=False,
        finite_trial_comparison_not_algorithm_novelty=True,
    )
    destination = root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / "readout.json", report)
    csv_write(destination / "table.csv", table)
    csv_write(destination / "per_type.csv", per_type)
    csv_write(destination / "paired_success.csv", report["paired_success"])
    csv_write(
        destination / "paired_business_sensitivity.csv", report["paired_business_sensitivity"]
    )
    csv_write(destination / "development_token_curve.csv", curves)
    (root / "REPORT.md").write_text(report_markdown(report))
    print(
        json.dumps(
            dict(
                status=report["status"],
                terminal_rows=len(table),
                test_authorized=may_read_test,
                rl_tokens=report["costs"]["rl_generation_tokens"],
            )
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
