#!/usr/bin/env python3
"""Inspect frozen Dev endpoints and replay a few recorded examples without resampling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.agents.mandate_repair_env import MandateRepairEnv, canonical
from src.agents.mandate_repair_metrics import trajectory_diagnostics


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def classification(records, tasks):
    categories, fields, violations, errors, kinds = (Counter() for _ in range(5))
    failed, lucky, exact, submitted = [], [], [], []
    for row in records:
        key = row["task_id"]
        kinds[tasks[key]["task_type"]] += 1
        categories[row.get("failure_category", "unavailable")] += 1
        extracted = row.get("constraint_extraction") or {}
        if row.get("submitted"):
            submitted.append(key)
        if extracted.get("complete") is True:
            exact.append(key)
        if row.get("incomplete_extraction_but_business_correct"):
            lucky.append(key)
        if not row.get("success"):
            failed.append(key)
            fields.update(extracted.get("mismatched_fields", []))
            verified = row.get("verification") or {}
            violations.update(verified.get("violations", []))
            errors.update(verified.get("evidence_errors", []))
    return dict(
        planned_tasks=len(tasks),
        observed_tasks=len(records),
        successes=sum(r["success"] for r in records),
        business_correct=sum(r.get("business_correct") is True for r in records),
        evidence_valid=sum(r.get("evidence_valid") is True for r in records),
        submitted=len(submitted),
        complete_extraction=len(exact),
        incomplete_extraction_but_business_correct=len(lucky),
        incomplete_extraction_but_complete_task_success=sum(
            r.get("success") is True
            and (r.get("constraint_extraction") or {}).get("complete") is False
            for r in records
        ),
        failure_categories=dict(categories),
        failed_task_ids=failed,
        failed_mismatched_fields=dict(fields.most_common()),
        failed_business_violations=dict(violations.most_common()),
        failed_evidence_errors=dict(errors.most_common()),
        per_type={
            kind: dict(
                tasks=count,
                successes=sum(
                    r.get("success") is True
                    for r in records
                    if tasks[r["task_id"]]["task_type"] == kind
                ),
                business_correct=sum(
                    r.get("business_correct") is True
                    for r in records
                    if tasks[r["task_id"]]["task_type"] == kind
                ),
            )
            for kind, count in kinds.items()
        },
    )


def request_examples(path, keys):
    selected = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("request_key") in keys:
                selected[row["request_key"]] = row
    if set(selected) != keys:
        raise ValueError("recorded example request missing; no replacement generation allowed")
    return selected


def replay_example(task, recorded, requests):
    env = MandateRepairEnv(task)
    messages = env.initial_messages()
    checks = []
    request_records = []
    if len(recorded["trace"]) != len(recorded["request_keys"]):
        raise ValueError("example has incomplete recorded action trace")
    for key, event in zip(recorded["request_keys"], recorded["trace"], strict=True):
        request = requests[key]
        if request["messages"] != messages:
            raise ValueError("recorded example messages do not match real same-task prefix")
        raw = request["response_text"]
        actual = env.step(raw)
        if canonical(actual) != canonical(event["response"]):
            raise ValueError("example replay did not reproduce recorded tool response")
        checks.append(dict(request_key=key, same_recorded_prefix=True, same_tool_response=True))
        request_records.append(
            {
                k: request.get(k)
                for k in (
                    "request_key",
                    "model_path",
                    "adapter_path",
                    "messages",
                    "response_text",
                    "prompt_tokens",
                    "response_tokens",
                    "request_seed",
                    "generation_status",
                )
            }
        )
        messages.extend(
            [
                dict(role="assistant", content=raw),
                # Collector tool messages preserve its ASCII JSON serialization;
                # initial messages come directly from the environment in Unicode.
                dict(role="tool", content=json.dumps(actual, sort_keys=True, allow_nan=False)),
            ]
        )
    actual = env.result()
    if actual["success"] != recorded["success"]:
        raise ValueError("example replay changed final success")
    actual.update(status=recorded["status"], collector_failure=recorded.get("collector_failure"))
    diagnostic = trajectory_diagnostics(env, actual)
    if diagnostic["business_correct"] != recorded.get("business_correct"):
        raise ValueError("example replay changed business correctness")
    return dict(
        task_id=task["task_id"],
        source_split="development",
        actual_initial_messages=env.initial_messages(),
        recorded_trajectory=recorded,
        actual_model_requests=request_records,
        replay_checks=checks,
        replayed_success=env.success,
        replayed_business_correct=diagnostic["business_correct"],
        no_model_generation_performed=True,
        hidden_gold_in_initial_messages=False,
    )


def inspect(root):
    root = Path(root).resolve()
    tasks = {t["task_id"]: t for t in rows(root / "data/development.jsonl")}
    if len(tasks) != 200:
        raise ValueError("fixed 200-task Dev required")
    records, analysis = {}, {}
    for folder in sorted((root / "eval").glob("*_dev")):
        if "_curve" in folder.name or not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("status") != "COMPLETE":
            continue
        values = rows(folder / "trajectory_results.jsonl")
        if len(values) != 200 or {r["task_id"] for r in values} != set(tasks):
            raise ValueError("full fixed Dev endpoints required")
        model = folder.name.removesuffix("_dev")
        records[model] = {r["task_id"]: r for r in values}
        analysis[model] = classification(values, tasks)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    artifact = dict(
        status="COMPLETE",
        source_split="development",
        policies=analysis,
        no_reserved_test_access=True,
        no_extra_sampling=True,
        diagnostic_does_not_replace_primary_task_success=True,
    )
    write(output / "endpoint_error_analysis.json", artifact)
    if not {"original", "sft"}.issubset(records):
        print(json.dumps(dict(status="PENDING_NEW_SFT_DEV", completed_policies=list(records))))
        return artifact
    before, after = records["original"], records["sft"]
    improvements = [tid for tid in tasks if not before[tid]["success"] and after[tid]["success"]]

    def priority(tid):
        constraints = tasks[tid]["gold"]["constraints"]
        return (
            not before[tid].get("business_correct") and after[tid].get("business_correct"),
            len(constraints["sector_max"]) >= 2,
            len(set(constraints["sector_max"]) - {"Technology"}) > 0,
            (after[tid].get("constraint_extraction") or {}).get("complete") is True,
            len(constraints["sector_max"]),
        )

    ranked = sorted(improvements, key=lambda tid: (tuple(-int(v) for v in priority(tid)), tid))
    selected, bases = [], set()
    for tid in ranked:
        base = tasks[tid]["base_scenario_id"]
        if base in bases:
            continue
        selected.append(tid)
        bases.add(base)
        if len(selected) == 3:
            break
    examples = root / "examples"
    examples.mkdir(exist_ok=True)
    if not selected:
        (examples / "README.md").write_text(
            "# 真实开发轨迹示例\n\n本轮没有原SFT失败而新SFT成功的同题配对，未制造改善示例。\n"
            "全部结果见[固定终点分析](../summary/endpoint_error_analysis.json)。\n"
        )
        return artifact
    request_maps = {
        model: request_examples(
            root / "eval" / f"{model}_dev/requests.jsonl",
            {key for tid in selected for key in records[model][tid]["request_keys"]},
        )
        for model in ("original", "sft")
    }
    comparisons = []
    lines = [
        "# 真实同题轨迹：首版SFT失败到新SFT成功",
        "",
        "来自本轮固定200题开发集，未重采、未改工具上限、未读取保留测试。"
        "优先展示多行业上限的业务错误改正，各示例使用不同基础情景。"
        "这是说明性选择；整体效果以完整200题固定分母读数为准，不从三个案例推断平均优势。",
        "",
        "两模型的每条实际请求均与同一任务初始观察及真实工具回放一致，"
        "JSON保存原始模型输出、工具响应和实际messages，不向提示加入gold。",
        "",
    ]
    for index, tid in enumerate(selected, 1):
        snapshots = {}
        for model in ("original", "sft"):
            snapshot = replay_example(tasks[tid], records[model][tid], request_maps[model])
            filename = f"{index:02d}_{tid}_{model}.json"
            write(examples / filename, snapshot)
            snapshots[model] = dict(
                path=filename,
                success=records[model][tid]["success"],
                business_correct=records[model][tid].get("business_correct"),
                calls=records[model][tid]["calls_used"],
                extraction=records[model][tid].get("constraint_extraction"),
            )
        old_path, new_path = snapshots["original"]["path"], snapshots["sft"]["path"]
        constraints = tasks[tid]["gold"]["constraints"]
        caps_text = "、".join(
            f"{name}≤{100 * cap:g}%" for name, cap in constraints["sector_max"].items()
        )
        comparisons.append(
            dict(
                task_id=tid,
                same_actual_initial_context=True,
                industry_caps=constraints["sector_max"],
                **snapshots,
            )
        )
        lines.extend(
            [
                f"- **{tid}**：{caps_text}。"
                f"[首版SFT失败]({old_path}) → [新SFT成功]({new_path})；"
                f"实际工具调用{snapshots['original']['calls']}→{snapshots['sft']['calls']}。"
                f"原字段差异：`{snapshots['original']['extraction']['mismatched_fields']}`；"
                "新模型完整提取有效条款。",
                "",
            ]
        )
    write(
        examples / "comparison.json",
        dict(
            status="COMPLETE",
            comparisons=comparisons,
            selected_for_explanation_not_average_effect=True,
            replayed_all_tool_responses=True,
        ),
    )
    lines.append(
        "[完整端点错误归类](../summary/endpoint_error_analysis.json) · "
        "[固定分母结果](../REPORT.md)\n"
    )
    (examples / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                status="COMPLETE",
                selected_improvement_examples=selected,
                source_split="development",
                no_test_access=True,
            )
        )
    )
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    inspect(parser.parse_args().output)


if __name__ == "__main__":
    main()
