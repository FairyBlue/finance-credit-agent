#!/usr/bin/env python3
"""Compact fixed-Dev financial endpoint diagnosis and a few raw-output replays."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from src.agents.financial_evidence_env import PROFILE, FinancialEvidenceEnv
from src.agents.mandate_repair_metrics import trajectory_diagnostics
from src.agents.same_prefix_online_collect import canonical


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def classify(records, tasks):
    categories, failures, failed_checks = Counter(), [], Counter()
    keys = (
        "metric_correct", "period_correct", "unit_correct", "condition_correct",
        "fact_correct", "portfolio_correct", "business_correct", "evidence_valid",
    )
    per_task = []
    for row in records:
        verified = row.get("verification") or {}
        if row.get("collector_failure"):
            category = "collector_or_system_failure"
        elif not row.get("submitted"):
            category = "no_submission"
        elif not verified.get("metric_correct"):
            category = "incorrect_financial_metric"
        elif not verified.get("period_correct"):
            category = "incorrect_measurement_period"
        elif not verified.get("unit_correct"):
            category = "incorrect_metric_unit"
        elif not verified.get("condition_correct"):
            category = "incorrect_conditional_trigger"
        elif not verified.get("portfolio_correct"):
            category = "incorrect_executed_portfolio_or_proof"
        elif not verified.get("evidence_valid"):
            category = "evidence_only_failure"
        else:
            category = "success"
        categories[category] += 1
        if not row.get("success"):
            failures.append(row["task_id"])
            failed_checks.update(key for key in keys if not verified.get(key))
        per_task.append(dict(
            task_id=row["task_id"], source_report=tasks[row["task_id"]]["source"]["report_id"],
            success=row.get("success") is True, primary_category=category,
            submitted_metric=row.get("submitted_metric"),
            checks={key: verified.get(key) for key in keys},
            calls=row.get("calls_used"),
        ))
    return dict(
        planned_tasks=100, observed_tasks=len(records),
        successes=sum(row.get("success") is True for row in records),
        submitted=sum(row.get("submitted") is True for row in records),
        correct_checks={
            key: sum((row.get("verification") or {}).get(key) is True for row in records)
            for key in keys
        },
        primary_failure_categories=dict(categories),
        failed_checks_overlap=dict(failed_checks), failed_task_ids=failures,
        primary_categories_prioritize_metric_then_period_unit_trigger_execution=True,
        no_missing_failure_slots_imputed=True,
        per_task=per_task,
    )


def request_records(path, keys):
    found = {}
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("request_key") in keys:
                found[value["request_key"]] = value
    if set(found) != keys:
        raise ValueError("missing recorded requests; no replacement sampling allowed")
    return found


def local_signal_counts(root):
    """Describe recorded auxiliary support, not a new counterfactual experiment."""
    result = {}
    for arm in ("sham_seed42", "m3_seed42"):
        directory = root / "training" / arm
        if not directory.exists():
            continue
        complete, parent_requests, parent_trajectories, by_round = [], 0, 0, []
        for path in sorted(directory.glob("round_*/collect/local_credit.jsonl")):
            records = rows(path)
            supported = [row for row in records if row.get("status") == "COMPLETE"]
            nonzero = [row for row in supported if math.isfinite(
                float(row.get("unscaled_local_difference", 0))
            ) and abs(float(row.get("unscaled_local_difference", 0))) > 1e-12]
            complete.extend(supported)
            summary_path = path.parent / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                parent_requests += summary.get("parent_model_requests", 0)
                parent_trajectories += summary.get("started_trajectories", 0)
            by_round.append(dict(
                round=path.parent.parent.name,
                complete_supported=len(supported), nonzero_unscaled_difference=len(nonzero),
            ))
        differences = [float(row.get("unscaled_local_difference", 0)) for row in complete]
        nonzero_count = sum(math.isfinite(value) and abs(value) > 1e-12 for value in differences)
        result[arm] = dict(
            complete_supported=len(complete), nonzero_unscaled_difference=nonzero_count,
            positive_unscaled_difference=sum(value > 1e-12 for value in differences),
            negative_unscaled_difference=sum(value < -1e-12 for value in differences),
            recorded_parent_model_requests=parent_requests,
            recorded_parent_trajectories=parent_trajectories,
            nonzero_local_rows_per_parent_request=(
                nonzero_count / parent_requests if parent_requests else None
            ),
            local_credit_enabled=arm.startswith("m3"),
            counts_are_recorded_support_exposures_not_independent_effect_estimates=True,
            rounds=by_round,
        )
    return result


def replay(task, recorded, requests):
    env = FinancialEvidenceEnv(task)
    messages = env.initial_messages()
    initial = json.loads(json.dumps(messages))
    if env.visible()["profile"] != PROFILE:
        raise ValueError("current opaque-ID observation required")
    if len(recorded["trace"]) != len(recorded["request_keys"]):
        raise ValueError("standard example must have a complete recorded request trace")
    checks, actual_requests = [], []
    for key, event in zip(recorded["request_keys"], recorded["trace"], strict=True):
        request = requests[key]
        if request["messages"] != messages:
            raise ValueError(f"recorded prefix differs in {key}")
        raw = request["response_text"]
        response = env.step(raw)
        if canonical(response) != canonical(event["response"]):
            raise ValueError(f"recorded tool result differs in {key}")
        checks.append(dict(request_key=key, same_prefix=True, same_tool_response=True))
        actual_requests.append({k: request.get(k) for k in (
            "request_key", "messages", "response_text", "model_path", "adapter_path",
            "prompt_tokens", "response_tokens", "request_seed", "generation_status",
        )})
        messages.extend([
            dict(role="assistant", content=raw),
            dict(role="tool", content=canonical(response)),
        ])
    actual = env.result()
    actual.update(status=recorded["status"], collector_failure=recorded.get("collector_failure"))
    diagnostic = trajectory_diagnostics(env, actual)
    if actual["success"] != recorded["success"]:
        raise ValueError("final replay success differs")
    for key in ("business_correct", "metric_correct", "period_correct", "unit_correct",
                "condition_correct", "portfolio_correct", "evidence_valid"):
        if bool(diagnostic.get(key)) != bool(recorded.get(key)):
            raise ValueError(f"final replay check differs: {key}")
    return dict(
        private_log_task_id=task["task_id"], public_task_id=env.visible()["task_id"],
        observation_profile=PROFILE, source_split="development",
        actual_initial_messages=initial, recorded_trajectory=recorded,
        actual_model_requests=actual_requests, replay_checks=checks,
        replayed_success=actual["success"], replayed_terminal_checks=diagnostic,
        no_model_generation=True, no_test_access=True,
    )


def inspect(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "data/manifest.json").read_text())
    if manifest.get("observation_profile") != PROFILE:
        raise ValueError("only the corrected v2 opaque-ID run is supported")
    tasks = {t["task_id"]: t for t in rows(root / "data/development.jsonl")}
    if len(tasks) != 100:
        raise ValueError("fixed 100-task Dev denominator required")
    records, analysis = {}, {}
    # Explicit policy names avoid reading reserved Test or transient curve slots.
    for model in ("original", "common", "fixed_retrieval", "oracle_facts", "rloo_seed42",
                  "sft", "sham_seed42", "m3_seed42", "rloo_seed43", "sham_seed43", "m3_seed43"):
        folder = root / "eval" / f"{model}_dev"
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        values = rows(folder / "trajectory_results.jsonl")
        if summary.get("status") != "COMPLETE" or len(values) != 100:
            continue
        if {row["task_id"] for row in values} != set(tasks):
            raise ValueError("endpoint does not cover the original fixed Dev tasks")
        records[model] = {row["task_id"]: row for row in values}
        analysis[model] = dict(
            classify(values, tasks), diagnostic_mode=summary.get("diagnostic_mode", "standard"),
            same_model_reference="common" if model in ("fixed_retrieval", "oracle_facts") else None,
        )
    summary_dir = root / "summary"
    summary_dir.mkdir(exist_ok=True)
    artifact = dict(
        status="COMPLETE", observation_profile=PROFILE, source_split="development",
        fixed_denominator=100, policies=analysis,
        no_test_access=True, no_extra_sampling=True,
        old_v1_results_not_combined=True,
        diagnosis_does_not_change_primary_success_or_scoring=True,
        auxiliary_local_signal_counts=local_signal_counts(root),
    )
    if "m3_seed42" in records:
        incremental = {}
        for control in ("common", "rloo_seed42", "sham_seed42", "sft"):
            if control not in records:
                continue
            comparisons = {}
            for field in ("success", "metric_correct", "condition_correct", "portfolio_correct"):
                improved, regressed = [], []
                for task_id in tasks:
                    before = records[control][task_id].get(field) is True
                    after = records["m3_seed42"][task_id].get(field) is True
                    if after and not before:
                        improved.append(task_id)
                    elif before and not after:
                        regressed.append(task_id)
                comparisons[field] = dict(
                    fixed_N=100, improved=len(improved), regressed=len(regressed),
                    net_change=len(improved) - len(regressed),
                    improved_task_ids=improved, regressed_task_ids=regressed,
                    affected_source_reports=len({
                        tasks[key]["source"]["report_id"] for key in improved + regressed
                    }),
                )
            incremental[control] = comparisons
        artifact["m3_seed42_paired_incremental_checks"] = incremental
        artifact["paired_tasks_are_not_independent_source_reports"] = True
    write(summary_dir / "endpoint_error_analysis.json", artifact)
    if not {"common", "sft"}.issubset(records):
        print(json.dumps(dict(status="PENDING_SFT_DEV", policies=list(records))))
        return artifact
    before, after = records["common"], records["sft"]
    improved = [key for key in tasks if not before[key]["success"] and after[key]["success"]]
    improved.sort(key=lambda key: (
        before[key].get("metric_correct") is True,
        before[key].get("condition_correct") is True,
        tasks[key]["source"]["report_id"], key,
    ))
    selected, reports = [], set()
    for key in improved:
        source = tasks[key]["source"]["report_id"]
        if source not in reports:
            selected.append((key, "common_failure_to_sft_success"))
            reports.add(source)
        if len(selected) == 2:
            break
    failed = [key for key in tasks if not after[key]["success"]]
    failed.sort(key=lambda key: (
        after[key].get("metric_correct") is True,
        after[key].get("submitted") is not True,
        tasks[key]["source"]["report_id"] in reports, key,
    ))
    if failed:
        selected.append((failed[0], "sft_remaining_failure"))
    examples = root / "examples"
    examples.mkdir(exist_ok=True)
    models = [key for key in ("common", "rloo_seed42", "sft", "sham_seed42", "m3_seed42")
              if key in records]
    request_maps = {}
    for model in models:
        source = root / "eval" / f"{model}_dev"
        summary = json.loads((source / "summary.json").read_text())
        if summary.get("evaluation_reused"):
            source = Path(summary["evaluation_source"])
        if source.resolve().parent != (root / "eval").resolve() or not source.name.endswith("_dev"):
            raise ValueError("example source must be Dev from this same corrected run")
        keys = {k for tid, _ in selected for k in records[model][tid]["request_keys"]}
        request_maps[model] = request_records(source / "requests.jsonl", keys)
    comparisons = []
    lines = [
        "# 财报条件委托：真实同题轨迹与剩余失败", "",
        "仅使用v2固定100题开发集，不重采、不读取Test、不混入v1旧观察结果。"
        "下方private ID只用于日志查找；模型初始观察是同基础情景稳定的opaque ID，"
        "不含v0/v1触发标签。示例是说明性选择，整体效果看完整固定分母。", "",
    ]
    for tid, label in selected:
        snapshots, initials = {}, []
        for model in models:
            snapshot = replay(tasks[tid], records[model][tid], request_maps[model])
            filename = f"{tid}_{model}.json"
            write(examples / filename, snapshot)
            initials.append(snapshot["actual_initial_messages"])
            row = records[model][tid]
            snapshots[model] = dict(
                path=filename, success=row["success"], metric=row.get("submitted_metric"),
                calls=row["calls_used"], financial_checks={
                    key: row.get(key) for key in (
                        "metric_correct", "condition_correct", "portfolio_correct", "evidence_valid"
                    )
                },
                calculate_calls=[dict(
                    expression=(event["action"].get("arguments") or {}).get("expression"),
                    result=event.get("response", {}).get("value"),
                    error=event.get("response", {}).get("error"),
                ) for event in row.get("trace", [])
                    if (event.get("action") or {}).get("name") == "calculate"],
            )
        if any(initial != initials[0] for initial in initials):
            raise ValueError("same-task model examples have different initial observations")
        task = tasks[tid]
        comparisons.append(dict(
            task_id=tid, type=label, same_initial_observation=True,
            question=task["condition"]["question"], gold_metric=task["gold"]["metric_value"],
            gold_triggered=task["gold"]["triggered"], policies=snapshots,
        ))
        title = "共同起点错误→教师SFT成功" if label.startswith("common_") else "SFT仍失败"
        old_checks = snapshots["common"]["financial_checks"]
        new_checks = snapshots["sft"]["financial_checks"]
        if label.startswith("common_"):
            corrected = []
            for key, description in (
                ("metric_correct", "财报取数/指标计算"),
                ("condition_correct", "条件触发判断"),
                ("portfolio_correct", "最终执行约束"),
                ("evidence_valid", "充分证据引用"),
            ):
                if not old_checks[key] and new_checks[key]:
                    corrected.append(description)
            explanation = "该同题实际改正环节：" + "、".join(corrected) + "。"
        elif not new_checks["metric_correct"]:
            explanation = "SFT的财报取值/指标计算仍错；"
            explanation += (
                "即使最终组合可行，严格任务也不算成功。"
                if new_checks["portfolio_correct"] else "最终执行也未通过全部业务核验。"
            )
        elif not new_checks["condition_correct"]:
            explanation = "SFT指标已正确，但条件触发判断仍错。"
        else:
            explanation = "SFT仍在执行约束或证据完整性环节失败，见原始核验字段。"
        links = "；".join(
            f"[{model}]({value['path']})：{'成功' if value['success'] else '失败'}"
            f"，指标`{(value['metric'] or {}).get('value')}`"
            for model, value in snapshots.items()
        )
        lines.extend([
            f"- **{tid}：{title}**。{links}。", "",
            f"  原问题：{task['condition']['question']}；"
            f"可执行目标值`{task['gold']['metric_value']}`，"
            f"条件触发`{task['gold']['triggered']}`。", "",
            f"  {explanation}这是同题轨迹定位，不将单例叙述当成平均因果效应。", "",
        ])
        for model in ("common", "sft"):
            expressions = snapshots[model]["calculate_calls"]
            rendered = "；".join(
                f"`{call['expression']}` → `{call['result']}`"
                + (f"（错误：{call['error']}）" if call["error"] else "")
                for call in expressions
            ) or "未完成实际calculate调用"
            lines.extend([f"  {model}实际计算：{rendered}。", ""])
    if not improved:
        lines.extend(["没有共同起点错误而SFT改正的同题，未制造改善示例。", ""])
    lines.extend([
        "JSON保留原始模型输出、实际messages和工具响应；每个请求前缀与终局已重放核对。",
        "", "[完整开发错误归类](../summary/endpoint_error_analysis.json) · "
        "[固定分母结果](../REPORT.md)",
    ])
    write(examples / "comparison.json", dict(
        status="COMPLETE", source_split="development", examples=comparisons,
        no_test_access=True, no_extra_sampling=True,
    ))
    (examples / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="COMPLETE", policies=list(analysis),
                          selected_examples=selected, no_test_access=True)))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    inspect(parser.parse_args().output)


if __name__ == "__main__":
    main()
