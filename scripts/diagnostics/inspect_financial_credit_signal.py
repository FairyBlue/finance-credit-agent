#!/usr/bin/env python3
"""Summarize recorded Train response probes without sampling or environment replay."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.mandate_repair_metrics import business_score  # noqa: E402


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check(row, name):
    return row.get(name, (row.get("verification") or {}).get(name)) is True


def inspect(output):
    output = Path(output).resolve()
    probe = output / "probe/paired"
    summary = json.loads((probe / "summary.json").read_text())
    started = json.loads((probe / "started.json").read_text())
    gate = json.loads((probe / "gate.json").read_text())
    tasks = {row["task_id"]: row for row in read_rows(started["data_path"])}
    anchors = {row["trajectory_key"]: row for row in read_rows(probe / "anchors.jsonl")}
    results = read_rows(probe / "trajectory_results.jsonl")
    local = read_rows(probe / "local_credit.jsonl")
    per_condition = {}
    recovery = []
    for condition in started["conditions"]:
        selected = [row for row in results if row["condition"] == condition]
        valid = [row for row in selected if business_score(row) is not None]
        per_condition[condition] = dict(
            trajectories=len(selected),
            business_evaluable_trajectories=len(valid),
            status_counts=dict(Counter(row["status"] for row in selected)),
            business_successes=sum(business_score(row) == 1 for row in valid),
            full_successes=sum(row.get("success") is True for row in selected),
            metric_correct=sum(check(row, "metric_correct") for row in selected),
            condition_correct=sum(check(row, "condition_correct") for row in selected),
            portfolio_correct=sum(check(row, "portfolio_correct") for row in selected),
            evidence_valid=sum(check(row, "evidence_valid") for row in selected),
            call_counts=dict(Counter(str(row["call_count"]) for row in selected)),
            mean_calls_used=sum(row["call_count"] for row in selected) / len(selected)
            if selected
            else None,
            business_and_citation_reported_separately=True,
        )
    for row in results:
        if row["condition"] != "masked":
            continue
        anchor = anchors[row["trajectory_key"]]
        original = anchor["original_tool_result"]
        original_id = original["doc_id"]
        task = tasks[row["task_id"]]
        documents = {doc["doc_id"]: doc for doc in task["documents"]}
        original_source_id = documents[original_id].get("source_id")
        equivalent = {
            doc["doc_id"]
            for doc in task["documents"]
            if (
                doc.get("source_id") == original_source_id
                if original_source_id is not None
                else doc.get("text") == original["text"]
            )
        }
        anchor_call = (len(anchor["original_prefix_messages"]) - 2) // 2 + 1
        reads = [
            item
            for item in row.get("trace", [])
            if item["call_index"] > anchor_call
            and (item.get("action") or {}).get("name") == "read_document"
            and item.get("response", {}).get("ok") is True
        ]
        original_reads = [item for item in reads if item["response"].get("doc_id") == original_id]
        duplicate_reads = [
            item for item in reads if item["response"].get("doc_id") in equivalent - {original_id}
        ]
        observed = bool(original_reads or duplicate_reads)
        success = business_score(row) == 1
        recovery.append(
            dict(
                trajectory_key=row["trajectory_key"],
                base_scenario_id=row["base_scenario_id"],
                parent_request_key=row["parent_request_key"],
                anchor_call_index=anchor_call,
                original_document_id=original_id,
                original_document_kind=documents[original_id].get("kind"),
                original_financial_source_id=original_source_id,
                equivalent_document_ids=sorted(equivalent),
                reread_financial_source_ids=[
                    documents[item["response"]["doc_id"]].get("source_id")
                    for item in original_reads + duplicate_reads
                ],
                original_report_rereads=len(original_reads),
                equivalent_copy_reads=len(duplicate_reads),
                report_reacquired=observed,
                business_correct=success,
                full_success=row.get("success") is True,
                calls_used=row["call_count"],
                within_original_tool_budget=row["call_count"] <= 10,
                recovered_business_after_report_reacquisition=observed and success,
            )
        )
    analysis = dict(
        profile="recorded_financial_credit_signal_behavior_v1",
        scope="fresh Train probe only; no Test, no new generation or environment replay",
        source_probe=str(probe),
        gate=gate,
        actual_generation_tokens=summary["total_tokens"],
        per_condition=per_condition,
        masked_recovery=dict(
            trajectories=len(recovery),
            original_report_reread_trajectories=sum(
                r["original_report_rereads"] > 0 for r in recovery
            ),
            equivalent_copy_read_trajectories=sum(r["equivalent_copy_reads"] > 0 for r in recovery),
            report_reacquired_trajectories=sum(r["report_reacquired"] for r in recovery),
            business_recovered_after_reacquisition=sum(
                r["recovered_business_after_report_reacquisition"] for r in recovery
            ),
            all_within_original_tool_budget=all(r["within_original_tool_budget"] for r in recovery),
            rows=recovery,
        ),
        local_effects=dict(
            complete_supports=len(local),
            paired_difference_counts=dict(
                Counter(str(row["unscaled_local_difference"]) for row in local)
            ),
            nonzero_supports=sum(row["unscaled_local_difference"] != 0 for row in local),
            reproducible_key_loss_base_questions=len(gate["reproducible_key_loss_base_questions"]),
            required_reproducible_base_questions=gate["required_reproducible_base_questions"],
            failed_gate_does_not_mean_all_differences_zero=gate["status"] == "FAIL"
            and any(row["unscaled_local_difference"] != 0 for row in local),
        ),
        intervention_is_single_response_only=True,
        placebo_is_metadata_control_not_unrelated_tool_step=True,
        global_credit_uselessness_claim=False,
        document_identity_note="original_report_rereads means the exact requested doc_id; "
        "a different archive doc_id with the same FinQA source_id is an equivalent-source reread",
    )

    # Illustrations are selected from the complete fixed probe, never resampled.
    choices = []
    positive = [
        row
        for row in local
        if row["base_scenario_id"] in gate["reproducible_key_loss_base_questions"]
    ][:1]
    negative = [row for row in local if row["unscaled_local_difference"] < 0]
    noisy = [
        row
        for row in local
        if row["base_scenario_id"] in gate["varying_paired_key_loss_base_questions"]
    ]
    restored = [row for row in recovery if row["recovered_business_after_report_reacquisition"]]
    for candidates, field in (
        (positive, "base_scenario_id"),
        (negative, "base_scenario_id"),
        (noisy, "base_scenario_id"),
        (restored, "base_scenario_id"),
        (local, "base_scenario_id"),
    ):
        for row in candidates:
            if row[field] not in choices:
                choices.append(row[field])
            if len(choices) == 2:
                break
        if len(choices) == 2:
            break
    example_rows = [row for row in results if row["base_scenario_id"] in choices]
    trajectory_keys = {row["trajectory_key"] for row in example_rows}
    requests = []
    request_costs = {condition: Counter() for condition in started["conditions"]}
    with (probe / "requests.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("model_request_submitted"):
                counts = request_costs[row["condition"]]
                counts["model_requests"] += 1
                counts["prompt_tokens"] += row.get("prompt_tokens", 0)
                counts["response_tokens"] += row.get("response_tokens", 0)
                counts["generation_tokens"] += row.get("prompt_tokens", 0) + row.get(
                    "response_tokens", 0
                )
            if row["trajectory_key"] in trajectory_keys:
                requests.append(row)
    for condition, costs in request_costs.items():
        per_condition[condition]["actual_generation_cost"] = dict(costs)
    analysis["recorded_request_generation_tokens"] = sum(
        counts["generation_tokens"] for counts in request_costs.values()
    )
    analysis["cost_scope"] = (
        "completed resumed paired attempt only; earlier failed attempt is separate"
    )
    write_json(output / "summary/signal_behavior_analysis.json", analysis)
    examples = output / "examples"
    examples.mkdir(exist_ok=True)
    lines = [
        "# 强SFT的真实同前缀信息干预与恢复",
        "",
        "仅整理本批fresh Train的已记录轨迹；没有重新采样、额外环境重放或读取Test。"
        "每个问题从首次读到指定财报前的同一真实前缀分叉，三条件各两次配对续写。"
        "masked只令这一次返回不可用；之后仍可重读原文或等价副本，工具上限仍为10次。"
        "placebo仅去掉返回中的可用/生效日期元数据，保留正文；它是格式控制，不是无关工具步骤。",
        "",
        f"本批gate：`{gate['status']} / {gate['reason']}`。"
        f"完整局部支持{len(local)}个，非零均值差{analysis['local_effects']['nonzero_supports']}个。"
        f"两次都呈现超出placebo的关键业务损失共有"
        f"{len(gate['reproducible_key_loss_base_questions'])}个基础问题，"
        f"预设要求{gate['required_reproducible_base_questions']}个。"
        "未过门槛不等于差值全零；这个有限探测也不推论所有step credit无用。",
        "",
        "| 条件 | 完整轨迹 | 业务正确 | 完整成功 | 指标正确 | 触发正确 | 执行正确 | 引用有效 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, counts in per_condition.items():
        lines.append(
            f"| {condition} | {counts['business_evaluable_trajectories']} | "
            f"{counts['business_successes']} | {counts['full_successes']} | "
            f"{counts['metric_correct']} | {counts['condition_correct']} | "
            f"{counts['portfolio_correct']} | {counts['evidence_valid']} |"
        )
    lines.extend(
        [
            "",
            "| 条件 | 终局调用均值（同24条分母） | 实际生成请求 | 输入＋输出token |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition, counts in per_condition.items():
        cost = counts["actual_generation_cost"]
        lines.append(
            f"| {condition} | {counts['mean_calls_used']:.3f} | "
            f"{cost.get('model_requests', 0)} | {cost.get('generation_tokens', 0):,} |"
        )
    recovered = analysis["masked_recovery"]
    lines.extend(
        [
            "",
            f"masked的{len(recovery)}条续写中，{recovered['report_reacquired_trajectories']}条"
            f"实际再次取得原文/等价副本，其中{recovered['business_recovered_after_reacquisition']}条"
            "完成正确业务。业务与引用分开，不能把仅缺引用的失败解释成业务credit。",
            "这里“原文重读0”指没有再次读取同一个doc_id；等价资料按实际FinQA source_id核对，"
            "不是说没有重读同源财报内容。每条恢复记录保留实际doc_id与source_id。",
            "",
        ]
    )
    for base in choices:
        selected = [row for row in example_rows if row["base_scenario_id"] == base]
        name = f"{base}_recorded_probe.json"
        write_json(
            examples / name,
            dict(
                base_scenario_id=base,
                recorded_results=selected,
                recorded_anchors=[anchors[row["trajectory_key"]] for row in selected],
                recorded_requests=[
                    r
                    for r in requests
                    if r["trajectory_key"] in {row["trajectory_key"] for row in selected}
                ],
                new_generation=False,
                new_environment_replay=False,
            ),
        )
        lines.extend(
            [
                f"## {base}",
                "",
                f"[已记录原始前缀、输出与工具返回]({name})。",
                "",
                "| 配对重复 | 条件 | 业务正确 | 提交指标 | 提交触发 | 调用数 | 引用有效 |",
                "|---|---|---|---|---|---:|---|",
            ]
        )
        for row in selected:
            metric = row.get("submitted_metric") or {}
            lines.append(
                f"| {row['paired_continuation_replicate']} | {row['condition']} | "
                f"{business_score(row)} | {metric.get('value')} | "
                f"{metric.get('triggered')} | {row['call_count']} | "
                f"{check(row, 'evidence_valid')} |"
            )
        supports = next(row for row in local if row["base_scenario_id"] == base)
        if base in gate["reproducible_key_loss_base_questions"]:
            lines.append(
                "\n该题两次factual与placebo均完成正确业务，两次masked均失败；"
                "失败包含指标错误，不能只归为引用损失。"
            )
        elif supports["unscaled_local_difference"] < 0:
            lines.append(
                "\n该题两次factual均失败，masked一次失败、一次终局业务正确；"
                "局部差值因此反向，但没有重复成立。它说明下游采样与恢复会改变局部符号，"
                "不能据此断言财报信息本来有害，也不证明每个中间计算步骤都正确。"
            )
        selected_recovery = [r for r in recovery if r["base_scenario_id"] == base]
        for row in selected_recovery:
            lines.append(
                f"\nmasked `{row['trajectory_key']}`：原文重读"
                f"{row['original_report_rereads']}次、等价副本读取{row['equivalent_copy_reads']}次；"
                f"业务正确={row['business_correct']}，原10次预算内={row['within_original_tool_budget']}。"
            )
        lines.append("")
    lines.append(
        "[完整行为汇总](../summary/signal_behavior_analysis.json) · "
        "[信号门槛](../probe/paired/gate.json) · [固定分母读数](../REPORT.md)"
    )
    (examples / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                gate=gate["status"],
                per_condition=per_condition,
                masked_recovery={key: value for key, value in recovered.items() if key != "rows"},
            ),
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    inspect(parser.parse_args().output)


if __name__ == "__main__":
    main()
