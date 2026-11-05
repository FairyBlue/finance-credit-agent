#!/usr/bin/env python3
"""Read recorded mandate-repair results without selecting models or opening new tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

CURVE = re.compile(r"^(.*)_curve(\d+)_dev$")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return math.fsum(values) / len(values) if values else None


def number(value, scale=1):
    return "NA" if value is None else f"{value * scale:.4f}"


def write_csv(path, records, fields=None):
    fields = fields or (list(records[0]) if records else ["status"])
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: "NA" if row.get(key) is None else row.get(key) for key in fields}
            for row in records
        )


def markdown(parts):
    """Tables need adjacent lines; paragraphs need a blank line between them."""
    blocks, table = [], []
    for part in parts:
        if part.startswith("|"):
            table.append(part.rstrip())
        else:
            if table:
                blocks.append("\n".join(table))
                table = []
            blocks.append(part.strip())
    if table:
        blocks.append("\n".join(table))
    return "\n\n".join(blocks) + "\n"


def task_map(root, split, cache):
    if split not in cache:
        manifest = read(root / "data/manifest.json")
        path = manifest["paths"][split]
        cache[split] = {row["task_id"]: row["task_type"] for row in rows(path)}
    return cache[split]


def normalized_results(summary, records, types):
    ids = summary["task_ids"]
    replicas = summary.get("repetitions", 1)
    mapping = {(row["task_id"], row.get("replicate_index", 0)): row for row in records}
    if len(mapping) != len(records) or not set(row[0] for row in mapping).issubset(ids):
        raise ValueError("duplicate or undeclared evaluation trajectory slots")
    expected = {(task_id, replica) for task_id in ids for replica in range(replicas)}
    if not set(mapping).issubset(expected):
        raise ValueError("undeclared evaluation replica")
    normalized = []
    for task_id in ids:
        for replica in range(replicas):
            row = dict(
                mapping.get(
                    (task_id, replica),
                    dict(
                        task_id=task_id,
                        replicate_index=replica,
                        status="MISSING",
                        success=False,
                    ),
                )
            )
            row["task_type"] = types[task_id]
            row["counted_success"] = row.get("status") == "COMPLETE" and row.get("success") is True
            normalized.append(row)
    return normalized


def aggregate(records):
    successes = [row for row in records if row["counted_success"]]
    correct_costs = [
        row["metrics"]["transaction_cost"]
        for row in successes
        if finite(row.get("metrics", {}).get("transaction_cost"))
    ]
    correct_calls = [row.get("calls_used", row.get("call_count")) for row in successes]
    correct_calls = [value for value in correct_calls if finite(value)]
    calls = [row.get("calls_used", row.get("call_count")) for row in records]
    costs = [row.get("metrics", {}).get("transaction_cost") for row in records]
    rewards = [row.get("terminal_reward", row.get("reward")) for row in records]
    status_counts = Counter(row.get("status", "UNKNOWN") for row in records)
    return dict(
        planned_N=len(records),
        observed_N=sum(row["status"] != "MISSING" for row in records),
        completed_N=status_counts["COMPLETE"],
        successful_N=len(successes),
        success_rate_all_planned=len(successes) / len(records) if records else None,
        failure_or_missing_N=len(records) - len(successes),
        model_or_system_failure_N=sum(
            row["status"] not in {"COMPLETE", "MISSING"} for row in records
        ),
        missing_N=status_counts["MISSING"],
        unsuccessful_completed_N=sum(
            row["status"] == "COMPLETE" and not row["counted_success"] for row in records
        ),
        correct_transaction_cost_mean=mean(correct_costs),
        correct_transaction_cost_N=len(correct_costs),
        correct_tool_calls_mean=mean(correct_calls),
        correct_tool_calls_N=len(correct_calls),
        correct_infeasible_N=sum(
            (row.get("verification") or {}).get("correct_infeasible") is True for row in successes
        ),
        business_feasible_N=sum(
            (row.get("verification") or {}).get("business_feasible") is True
            for row in records
            if row["status"] == "COMPLETE"
        ),
        evidence_valid_N=sum(
            (row.get("verification") or {}).get("evidence_valid") is True
            for row in records
            if row["status"] == "COMPLETE"
        ),
        actual_tool_calls=sum(value for value in calls if finite(value)),
        tool_calls_observed_N=sum(finite(value) for value in calls),
        transaction_cost_observed_sum=math.fsum(value for value in costs if finite(value)),
        transaction_cost_observed_N=sum(finite(value) for value in costs),
        reward_mean_all_planned=(
            mean(rewards) if all(finite(value) for value in rewards) else None
        ),
        status_counts=dict(status_counts),
    )


def progress_readout(root):
    progress = {}
    for path in sorted((root / "training").glob("*/progress.json")):
        source = read(path)
        value = dict(source)
        value["auxiliary_tokens"] = source.get("auxiliary_tokens", source.get("aux_tokens", 0))
        value["progress_path"] = str(path)
        budget, actual = source.get("token_budget"), source.get("cumulative_tokens")
        value["unused_token_budget"] = (
            budget - actual if finite(budget) and finite(actual) else None
        )
        value["actual_auxiliary_fraction"] = (
            value["auxiliary_tokens"] / actual if finite(actual) and actual > 0 else None
        )
        progress[path.parent.name] = value
    return progress


def cpu_baselines(root, may_read_test):
    sources, table, per_type = {}, [], []
    statuses = dict(
        development="NOT_RUN", test="NOT_RUN" if may_read_test else "NOT_OPEN_BEFORE_FREEZE"
    )
    for suffix, split in (("dev", "development"), ("test", "test")):
        if split == "test" and not may_read_test:
            continue
        path = root / "baselines" / f"{suffix}_summary.json"
        if not path.exists():
            continue
        summary = read(path)
        sources[split] = summary
        statuses[split] = summary["status"]
        for policy, value in summary["policies"].items():
            n, correct = value["tasks"], value["successes"]
            table.append(
                dict(
                    model=policy,
                    model_kind="CPU privileged" if policy.startswith("privileged_") else "CPU rule",
                    split=split,
                    evaluation_status=summary["status"],
                    planned_N=n,
                    observed_N=n,
                    completed_N=n,
                    successful_N=correct,
                    success_rate_all_planned=correct / n,
                    failure_or_missing_N=n - correct,
                    model_or_system_failure_N=0,
                    missing_N=0,
                    unsuccessful_completed_N=n - correct,
                    correct_transaction_cost_mean=value["mean_cost_successful_feasible"],
                    correct_transaction_cost_N=value["successful_feasible_tasks"],
                    correct_tool_calls_mean=value["mean_tool_calls"] if correct == n else None,
                    correct_tool_calls_N=correct if correct == n else 0,
                    correct_infeasible_N=value["correct_infeasible"],
                    business_feasible_N=value["business_feasible"],
                    evidence_valid_N=value["evidence_valid"],
                    actual_tool_calls=value["mean_tool_calls"] * n,
                    tool_calls_observed_N=n,
                    transaction_cost_observed_sum=None,
                    transaction_cost_observed_N=None,
                    reward_mean_all_planned=None,
                    model_requests=0,
                    prompt_tokens=0,
                    response_tokens=0,
                    total_tokens=0,
                    complete_model_requests=0,
                    failed_model_requests=0,
                    incomplete_generations=0,
                    elapsed_seconds=None,
                    cumulative_training_tokens=0,
                )
            )
            per_type += [
                dict(
                    model=policy,
                    split=split,
                    task_type=kind,
                    planned_N=group["tasks"],
                    successful_N=group["successes"],
                    success_rate_all_planned=group["successes"] / group["tasks"],
                )
                for kind, group in value["task_types"].items()
            ]
    return sources, table, per_type, statuses


def cumulative_tokens(progress, model, round_index):
    group = progress.get(model, {})
    if round_index is None:
        return group.get("cumulative_tokens")
    match = [row for row in group.get("rounds", []) if row["round"] == round_index]
    if len(match) > 1:
        raise ValueError("duplicate recorded training round")
    return match[0].get("cumulative_tokens") if match else None


def sft_attempt_costs(summary, settings, protocol):
    execution = protocol.get("execution", {})
    complete = summary is not None and summary.get("status") == "COMPLETE"
    logical = dict(
        optimizer_updates=summary.get("completed_optimizer_steps") if complete else None,
        example_uses=summary.get("cursor") if complete else None,
        prompt_token_uses=settings.get("total_prompt_tokens") if complete else None,
        target_token_uses=settings.get("total_target_tokens") if complete else None,
    )
    discarded = dict(
        optimizer_updates=execution.get("discarded_post_checkpoint_updates", 0),
        example_uses=execution.get("discarded_teacher_example_uses", 0),
        prompt_token_uses=execution.get("discarded_sft_prompt_token_uses", 0),
        target_token_uses=execution.get("discarded_sft_target_token_uses", 0),
    )
    return dict(
        logical_one_epoch=logical,
        discarded_attempt_uses=discarded,
        actual_attempt_uses={
            key: value + discarded[key] if value is not None else None
            for key, value in logical.items()
        },
        extra_uses_change_teacher_data=False,
    )


def paired_success(evaluations):
    paired = []
    by_key = {
        (row["model"], row["split"]): row for row in evaluations if row["curve_round"] is None
    }
    for (model, split), candidate in by_key.items():
        if model == "original":
            continue
        references = ["sft"] if model != "sft" else ["original"]
        if model.startswith("m3_seed"):
            references += [model.replace("m3_", "sham_", 1), model.replace("m3_", "rloo_", 1)]
        for reference in references:
            before = by_key.get((reference, split))
            if before is None:
                continue
            a = {(r["task_id"], r.get("replicate_index", 0)): r for r in before["slots"]}
            b = {(r["task_id"], r.get("replicate_index", 0)): r for r in candidate["slots"]}
            if set(a) != set(b):
                raise ValueError("paired evaluation task slots differ")
            wins = sum(b[key]["counted_success"] and not a[key]["counted_success"] for key in a)
            losses = sum(a[key]["counted_success"] and not b[key]["counted_success"] for key in a)
            paired.append(
                dict(
                    split=split,
                    candidate=model,
                    reference=reference,
                    planned_pairs=len(a),
                    success_wins=wins,
                    success_losses=losses,
                    ties=len(a) - wins - losses,
                    success_rate_difference_pp=100 * (wins - losses) / len(a),
                )
            )
    return paired


def plots(destination, evaluations, comparison_table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terminal = comparison_table
    splits = [
        split for split in ("development", "test") if any(row["split"] == split for row in terminal)
    ]
    if splits:
        figure, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5), squeeze=False)
        for ax, split in zip(axes[0], splits, strict=True):
            group = [row for row in terminal if row["split"] == split]
            for index, row in enumerate(group):
                ax.bar(
                    index,
                    100 * row["success_rate_all_planned"],
                    color=("tab:orange" if row["model"].startswith("m3") else "tab:blue"),
                )
            ax.set_xticks(
                range(len(group)), [row["model"] for row in group], rotation=40, ha="right"
            )
            ax.set(title=split, ylabel="Success / all predetermined slots (%)", ylim=(0, 105))
        figure.tight_layout()
        figure.savefig(destination / "success_comparison.png", dpi=160)
        plt.close(figure)
    curves = [
        row
        for row in evaluations
        if row["curve_round"] is not None and row["cumulative_training_tokens"] is not None
    ]
    if curves:
        figure, ax = plt.subplots(figsize=(10, 5))
        for model in sorted({row["model"] for row in curves}):
            group = sorted(
                (row for row in curves if row["model"] == model),
                key=lambda row: row["cumulative_training_tokens"],
            )
            ax.scatter(
                [row["cumulative_training_tokens"] / 1e6 for row in group],
                [100 * row["aggregate"]["success_rate_all_planned"] for row in group],
                label=model,
            )
        ax.set(
            title="Recorded development checkpoints; no token interpolation",
            xlabel="Actual cumulative parent + auxiliary generation tokens (millions)",
            ylabel="Success / all predetermined slots (%)",
            ylim=(0, 105),
        )
        ax.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(destination / "development_token_curve.png", dpi=160)
        plt.close(figure)


def run(output):
    root = Path(output).resolve()
    state = read(root / "state.json") if (root / "state.json").exists() else {"status": "UNKNOWN"}
    frozen_path = root / "frozen_models.json"
    frozen = read(frozen_path) if frozen_path.exists() else None
    may_read_test = frozen is not None and frozen.get("status") == "COMPLETE"
    progress = progress_readout(root)
    evaluations, task_types, skipped_test_paths = [], {}, []
    for folder in sorted((root / "eval").glob("*")):
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            continue
        split = "test" if folder.name.endswith("_test") else "development"
        if split == "test" and not may_read_test:
            skipped_test_paths.append(str(folder))
            continue
        curve = CURVE.match(folder.name)
        model = curve[1] if curve else folder.name.rsplit("_", 1)[0]
        round_index = int(curve[2]) if curve else None
        summary = read(summary_path)
        result_path = folder / "trajectory_results.jsonl"
        if not result_path.exists():
            result_path = folder / "trajectories.jsonl"
        records = rows(result_path) if result_path.exists() else summary.get("results", [])
        normalized = normalized_results(summary, records, task_map(root, split, task_types))
        group = aggregate(normalized)
        costs = {
            key: summary.get(key)
            for key in (
                "model_requests",
                "prompt_tokens",
                "response_tokens",
                "total_tokens",
                "complete_model_requests",
                "failed_model_requests",
                "incomplete_generations",
                "elapsed_seconds",
                "tool_counts",
            )
        }
        evaluations.append(
            dict(
                name=folder.name,
                model=model,
                split=split,
                curve_round=round_index,
                summary_path=str(summary_path),
                evaluation_status=summary.get("status"),
                reuse={
                    key: summary.get(key)
                    for key in (
                        "evaluation_reused",
                        "evaluation_source",
                        "source_sampled_tokens",
                        "source_sampled_cost",
                        "evaluation_fingerprint",
                    )
                },
                cumulative_training_tokens=cumulative_tokens(progress, model, round_index),
                aggregate=group,
                generation=costs,
                per_type={
                    kind: aggregate([row for row in normalized if row["task_type"] == kind])
                    for kind in sorted({row["task_type"] for row in normalized})
                },
                slots=normalized,
            )
        )
    terminal = [row for row in evaluations if row["curve_round"] is None]
    table = [
        dict(
            model=row["model"],
            model_kind="LLM",
            split=row["split"],
            evaluation_status=row["evaluation_status"],
            evaluation_reused=bool(row["reuse"]["evaluation_reused"]),
            evaluation_source=row["reuse"]["evaluation_source"],
            source_sampled_tokens=row["reuse"]["source_sampled_tokens"],
            evaluation_fingerprint=row["reuse"]["evaluation_fingerprint"],
            **{key: value for key, value in row["aggregate"].items() if key != "status_counts"},
            **{key: value for key, value in row["generation"].items() if key != "tool_counts"},
            cumulative_training_tokens=row["cumulative_training_tokens"],
        )
        for row in terminal
    ]
    per_type = [
        dict(
            model=row["model"],
            split=row["split"],
            task_type=kind,
            **{key: value for key, value in metrics.items() if key != "status_counts"},
        )
        for row in terminal
        for kind, metrics in row["per_type"].items()
    ]
    baselines, cpu_table, cpu_types, cpu_statuses = cpu_baselines(root, may_read_test)
    table.extend(cpu_table)
    per_type.extend(cpu_types)
    curve_rows = [
        dict(
            model=row["model"],
            round=row["curve_round"],
            evaluation_reused=bool(row["reuse"]["evaluation_reused"]),
            evaluation_source=row["reuse"]["evaluation_source"],
            source_sampled_tokens=row["reuse"]["source_sampled_tokens"],
            evaluation_fingerprint=row["reuse"]["evaluation_fingerprint"],
            actual_eval_prompt_tokens=row["generation"]["prompt_tokens"],
            actual_eval_response_tokens=row["generation"]["response_tokens"],
            actual_eval_total_tokens=row["generation"]["total_tokens"],
            cumulative_training_tokens=row["cumulative_training_tokens"],
            **{key: value for key, value in row["aggregate"].items() if key != "status_counts"},
        )
        for row in evaluations
        if row["curve_round"] is not None
    ]
    stages = state.get("stages", [])
    # The controller launches only GPU model workers; CPU teacher/baseline builds
    # happen outside its stage list. Older live-controller entries lack the flag.
    model_stages = [row for row in stages if row.get("gpu_worker", True) is True]
    sft_path = root / "common_sft/summary.json"
    settings_path = root / "common_sft/settings.json"
    report = dict(
        status=state.get("status"),
        controller_state=state,
        frozen_models=frozen,
        test_readout_authorized_by_complete_freeze=may_read_test,
        skipped_test_directories_before_freeze=skipped_test_paths,
        evaluations=evaluations,
        terminal_table=table,
        training_arms=progress,
        cpu_baselines=baselines,
        cpu_baseline_statuses=cpu_statuses,
        teacher_summary=(
            read(root / "teacher/summary.json")
            if (root / "teacher/summary.json").exists()
            else None
        ),
        sft_error_analysis=(
            read(root / "baselines/sft_error_analysis.json")
            if (root / "baselines/sft_error_analysis.json").exists()
            else None
        ),
        information_credit_diagnostics_path=(
            str(root / "summary/information_credit_diagnostics.json")
            if (root / "summary/information_credit_diagnostics.json").exists()
            else None
        ),
        protocol=read(root / "protocol.json") if (root / "protocol.json").exists() else {},
        validation_checks=(
            read(root / "validation_checks.json")
            if (root / "validation_checks.json").exists()
            else None
        ),
        next_open_auxiliary=(
            read(root / "baselines/dev_next_open_summary.json")
            if (root / "baselines/dev_next_open_summary.json").exists()
            else None
        ),
        evaluation_verification_replays={
            folder.name: read(folder / "verification_replay.json")
            for folder in (root / "eval").glob("*")
            if (folder / "verification_replay.json").exists()
            and (not folder.name.endswith("_test") or may_read_test)
        },
        paired_success=paired_success(evaluations),
        common_sft_summary=read(sft_path) if sft_path.exists() else None,
        common_sft_settings=read(settings_path) if settings_path.exists() else None,
        total_eval_generation={
            key: sum(row["generation"].get(key) or 0 for row in evaluations)
            for key in ("model_requests", "prompt_tokens", "response_tokens", "total_tokens")
        },
        controller_recorded_stage_wall_seconds=math.fsum(
            row["seconds"] for row in stages if finite(row.get("seconds"))
        ),
        recorded_gpu_model_stage_wall_seconds=(
            math.fsum(row["seconds"] for row in model_stages if finite(row.get("seconds")))
            if model_stages
            else state.get("gpu_elapsed_seconds")
        ),
        unflagged_controller_stages_are_gpu_model_workers=True,
        checkpoint_selection_performed=False,
        reused_evaluations_are_new_independent_experiments=False,
        task_or_window_selection_performed=False,
        success_fixed_denominator_includes_generation_system_and_missing_failures=True,
        successful_cost_means_always_include_their_N=True,
        token_checkpoint_interpolation_performed=False,
        formal_market_valid_access=False,
        formal_market_final_oos_access=False,
    )
    report["sft_attempt_costs"] = sft_attempt_costs(
        report["common_sft_summary"], report["common_sft_settings"] or {}, report["protocol"]
    )
    decision_path = root / "decision.json"
    report["final_decision"] = read(decision_path) if decision_path.exists() else None
    destination = root / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / "readout.json", report)
    write_csv(destination / "table.csv", table)
    write_csv(destination / "per_type.csv", per_type)
    write_csv(destination / "development_token_curve.csv", curve_rows)
    write_csv(destination / "paired_success.csv", report["paired_success"])
    if state.get("status") == "COMPLETE":
        plots(destination, evaluations, table)
    text = [
        "# 多工具投资约束修复：执行与结果\n",
        f"控制器状态：`{state.get('status')}`。本报告只读取已记录结果，没有选择checkpoint、任务或窗口。"
        f"冻结清单：{'COMPLETE' if may_read_test else '尚未完成'}；"
        + (
            "下表包括冻结后实际运行的test。"
            if may_read_test
            else "本报告只读development，未打开test任务或结果。"
        ),
        "任务使用Train历史已知报价、模拟账户与约束文档，回答多工具约束修复是否正确，"
        "不把它写成未来交易收益。主采样原生生成；解析、工具、生成及系统失败保留。"
        "成功率以事先规定任务槽为固定分母，错误/缺失计失败；运行中读数应与完成状态一起看。\n",
        "development与test各100个任务分别来自50个基础情景、每个情景2个改写；"
        "两个训练seed也复评同一批任务。改写与跨seed重复不增加独立任务数量。"
        "[原模型错误分析](baselines/original_error_analysis.json)、"
        "[真实交互示例](examples/README.md)。",
        "## 原模型、共同SFT及各臂终点\n",
        "| 模型 | 区域 | 成功/预定N | 成功率% | 业务可行/证据有效 | 正确费用均值（N） | "
        "正确工具调用均值（N） | 系统/生成失败 | 缺失 | 评估输入/输出token | 复用来源 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if report["final_decision"] and report["final_decision"].get("headline_zh"):
        text.insert(1, f"**{report['final_decision']['headline_zh']}** [简要结论](CONCLUSION.md)。")
    for row in table:
        reuse_source = Path(row["evaluation_source"]).name if row.get("evaluation_reused") else "—"
        text.append(
            f"| {row['model']} | {row['split']} | {row['successful_N']}/{row['planned_N']} | "
            f"{number(row['success_rate_all_planned'], 100)} | "
            f"{row['business_feasible_N']}/{row['evidence_valid_N']} | "
            f"{number(row['correct_transaction_cost_mean'])}"
            f"（{row['correct_transaction_cost_N']}） | "
            f"{number(row['correct_tool_calls_mean'])}（{row['correct_tool_calls_N']}） | "
            f"{row['model_or_system_failure_N']} | {row['missing_N']} | "
            f"{row['prompt_tokens']}/{row['response_tokens']} | {reuse_source} |"
        )
    text += [
        "温度0下，同一权重、语义配置、数据及native/eager/seed配置的确定性评估可以复用；"
        "复用保持原100任务分母，新增请求与token按0计，原采样成本及来源单列。"
        "这不是新的重复实验，也不能把相同策略的缓存评分写成RL训练增益。",
        "\n费用只在正确且实际有成本指标的任务上取均值，并明确N；"
        "正确不可行声明不被擅自当作一笔零成本成交。"
        "系统失败与业务错误分别计数，仍同时占据固定成功率分母。",
        "CPU privileged_full_information_oracle使用隐藏完整信息，仅作上限参照；"
        "public_rule_retrieval通过公开文档与真实工具取信息，单独列为可运行规则基准。"
        "业务可行与证据有效分列，任务总成功要求两者（或正确不可行声明）同时满足。",
    ]
    if may_read_test and cpu_statuses["test"] == "NOT_RUN":
        text.append(
            "冻结已完成；test的CPU oracle、公开规则及no-key基线尚未运行，"
            "结果未填零，也没有由读数脚本触发执行。"
        )
    elif may_read_test and "test" in baselines:
        text.append(
            "冻结后的test CPU强基线已并入同表；"
            "[CPU汇总](baselines/test_summary.json)、"
            "[逐任务真实轨迹](baselines/test_trajectories.jsonl)。"
        )
    for split, baseline in baselines.items():
        no_key = baseline["policies"].get("no_key_document")
        if no_key:
            text.append(
                f"{split}的no_key_document总成功{no_key['successes']}/{no_key['tasks']}，"
                f"其中业务可行{no_key['business_feasible']}/{no_key['tasks']}、"
                f"证据有效{no_key['evidence_valid']}/{no_key['tasks']}；总成功为零不代表业务可行数为零。"
            )
    analysis = report["sft_error_analysis"]
    if analysis:
        mixed = analysis["mixed_semantics"]
        text += [
            "## 共同SFT成功的含义\n",
            f"{analysis['successes']}/{analysis['planned_tasks']}表示最终业务可行与证据通过"
            "（包含正确不可行声明），不等于所有条款字段都抽取正确。"
            f"mixed类{mixed['tasks']}项中仅{mixed['complete_constraints_exact']}项字段全对，"
            f"另有{mixed['successful_but_constraints_mismatched']}个最终成功任务仍有字段差异。"
            f"本批{analysis['failures']}个失败都已读到正确证据，却在显式约束中把Financials写成Technology；"
            "求解器按传入的错误问题正常工作，问题不在求解器计算。"
            "[SFT逐项错误分析](baselines/sft_error_analysis.json)、"
            "[更新后的真实交互示例](examples/README.md)。",
        ]
    text += [
        "## 每类固定分母成功数\n",
        "| 模型 | 区域 | 类别 | 成功/预定N |",
        "|---|---|---|---:|",
    ]
    text += [
        f"| {row['model']} | {row['split']} | {row['task_type']} | "
        f"{row['successful_N']}/{row['planned_N']} |"
        for row in per_type
    ]
    text += [
        "\n## 真实预算与更新\n",
        "| 训练臂 | 预算上限 | 实际总token | 父/辅助token | 未使用尾余 | 有效更新 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, value in progress.items():
        updates = sum(row.get("optimizer_updates", 0) for row in value.get("rounds", []))
        text.append(
            f"| {model} | {value.get('token_budget')} | {value.get('cumulative_tokens')} | "
            f"{value.get('parent_tokens')}/{value['auxiliary_tokens']} | "
            f"{value['unused_token_budget']} | {updates} |"
        )
    zero_update_arms = [
        name
        for name, value in progress.items()
        if value.get("rounds") and value.get("completed_optimizer_steps") == 0
    ]
    if zero_update_arms:
        text.append(
            "截至已记录轮次，"
            + "、".join(zero_update_arms)
            + "累计0次有效RL更新；零优势批次保持原样跳过，复用终点评分不构成新增学习证据。"
        )
    sft_settings = report["common_sft_settings"] or {}
    text.append(
        "\n相同的是各臂生成token预算上限，实际消耗及尾余分别报告，不写成严格等算力。"
        "输入与输出token均计入；M3/sham辅助续写在预算内单列，辅助token不训练actor。"
        "开发曲线只画实际记录的累计token点，不插值制造相同token checkpoint。"
        f"共同SFT监督输入{sft_settings.get('total_prompt_tokens', 'NA')}token，"
        f"标签{sft_settings.get('total_target_tokens', 'NA')}token，和RL生成预算分列。"
    )
    attempts = report["sft_attempt_costs"]
    if attempts["logical_one_epoch"]["optimizer_updates"] is not None:
        logical, discarded, actual = (
            attempts["logical_one_epoch"],
            attempts["discarded_attempt_uses"],
            attempts["actual_attempt_uses"],
        )
        text.append(
            f"共同SFT逻辑训练仍为一轮{logical['optimizer_updates']}次更新、"
            f"{logical['example_uses']}条样本使用。一次执行优化重启废弃"
            f"{discarded['optimizer_updates']}次更新、{discarded['example_uses']}条样本使用，"
            f"额外消耗{discarded['prompt_token_uses']}输入、{discarded['target_token_uses']}标签token。"
            f"实际各attempt合计{actual['optimizer_updates']}次更新计算、"
            f"{actual['example_uses']}条样本使用、{actual['prompt_token_uses']}输入与"
            f"{actual['target_token_uses']}标签token，和逻辑一轮分列；没有重写数据。"
            "[执行记录](protocol.json)。"
        )
    text.append(
        "采样器后续默认eager以减少每次启动的编译开销；初始original/SFT开发评估及"
        "RLOO seed42首批的graph执行记录保留。执行模式变化不改变BF16/native分布或生成token预算，"
        "各次engine_settings均保留；实际GPU阶段wall time包含不同执行开销，"
        "不能直接当作严格的方法算力比较。"
    )
    if report["information_credit_diagnostics_path"]:
        text += [
            "## 信息credit的解释边界\n",
            "Train信息干预诊断按arm、seed、轮次和当时策略区分丢失成功的原因。"
            "已有快照中，证据完整性占相当部分，另有业务或证明错误及未提交；"
            "因此局部credit同时包含引用与流程完成价值，不能全部称为金融决策价值。"
            "完整分组计数以[信息干预诊断](summary/information_credit_diagnostics.json)为准；"
            "这项机制解释不替换主成功指标，也不改变训练协议。",
        ]
    next_open = report["next_open_auxiliary"]
    if next_open:
        text += [
            "## 执行时点变化的副读数\n",
            "主判仍是在固定已知报价下验证委托与证据，主表口径不变。"
            f"CPU将原报价下{next_open['feasible_plans_at_original_quote']}个可行计划"
            f"放到下一个真实开盘执行，实际执行{next_open['actually_executed']}个，"
            f"其中{next_open['original_mandate_feasible_at_next_open']}个仍满足原委托；"
            f"另有{next_open['scheduled_mandate_version_changed']}例跨时点委托版本变化，单独记录。"
            "这些数字只展示行情和生效时点改变带来的执行边界，"
            "不能将其比例当作主任务成功率，也没有替换主判报价或向模型提供未来报价。"
            "[副读数](baselines/dev_next_open_summary.json)、"
            "[逐任务记录](baselines/dev_next_open_diagnostic.jsonl)。",
        ]
    validation = report["validation_checks"]
    if validation:
        text.append(
            f"必要接线检查状态{validation['status']}，"
            f"{sum(value is True for value in validation['checks'].values())}/"
            f"{len(validation['checks'])}项通过。不可行声明须有实际约束证书；"
            "已有原模型输出用真实原动作CPU重放验证，原始生成与修正前记录保留，"
            "没有重新生成模型答案或修改教师数据。"
            "[检查记录](validation_checks.json)。"
        )
    text.append(
        f"控制器阶段wall time合计{number(report['controller_recorded_stage_wall_seconds'])}秒；"
        "已完成GPU模型阶段记录wall time"
        f"{number(report['recorded_gpu_model_stage_wall_seconds'])}秒。"
        "模型阶段时间包含其加载、采样或训练阶段开销，不冒充GPU核函数计时。\n"
        "[完整JSON](summary/readout.json)、[终点表](summary/table.csv)、"
        "[每类表](summary/per_type.csv)、[配对成功变化](summary/paired_success.csv)、"
        "[实际token曲线表](summary/development_token_curve.csv)。"
    )
    (root / "REPORT.md").write_text(markdown(text))
    print(
        json.dumps(
            dict(
                status=report["status"],
                terminal_evaluations=len(terminal),
                curve_checkpoints=len(curve_rows),
                test_read=may_read_test,
            )
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
