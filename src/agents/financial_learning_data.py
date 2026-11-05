"""Frozen Train panels and read-only signal diagnostics for learning cycle v2.

Only explicitly named historical Train files are read. Reserved Test data is
never discovered or loaded. Task truth stays in the environment dataset and is
never copied into policy messages by this preparation tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "runs/agentic-rl-mc-20261030-v1"
CREDIT = ROOT / "runs/m3-financial-credit-20261030-v1"
FINANCIAL = ROOT / "runs/m3-financial-evidence-20261030-v2"
DEFAULT = ROOT / "runs/agentic-rl-learning-20261030-v2"
SEED = 42


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_frozen(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"refusing to replace frozen preparation: {path}")
    if not path.exists():
        path.write_text(text)


def write_rows(path, rows):
    write_frozen(path, "".join(canonical(row) + "\n" for row in rows))


def write_json(path, value):
    write_frozen(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def question_key(task):
    return task["source"]["finqa_id"]


def report_key(task):
    return task["source"]["report_id"]


def paired_questions(tasks):
    groups = defaultdict(list)
    for task in tasks:
        if task["split"] != "train":
            raise ValueError("preparation accepts Train tasks only")
        groups[question_key(task)].append(task)
    for source, pair in groups.items():
        if len(pair) != 2 or {task["variant"] for task in pair} != {0, 1}:
            raise ValueError(f"two threshold variants required: {source}")
        if len({report_key(task) for task in pair}) != 1:
            raise ValueError("threshold pair must share one report")
        pair.sort(key=lambda task: task["variant"])
    return dict(groups)


def select_panel(tasks, probe, seed=SEED, panel_size=8, holdout_size=16):
    pairs = paired_questions(tasks)
    task_index = {task["task_id"]: task for task in tasks}
    probe_groups = defaultdict(list)
    for row in probe:
        if row["task_id"] not in task_index:
            raise ValueError("probe must come from the declared Train source")
        probe_groups[row["task_id"]].append(row)
    mixed_tasks = sorted(
        task_id
        for task_id, group in probe_groups.items()
        if len(group) == 4 and len({row.get("success") is True for row in group}) > 1
    )
    mixed_sources = sorted({question_key(task_index[task_id]) for task_id in mixed_tasks})
    rng = random.Random(seed)
    rng.shuffle(mixed_sources)
    sources, reports = [], set()
    for source in mixed_sources:
        report = report_key(pairs[source][0])
        if report not in reports and len(sources) < panel_size:
            sources.append(source)
            reports.add(report)
    mixed_selected = list(sources)
    remaining = sorted(set(pairs) - set(sources))
    rng.shuffle(remaining)
    for source in remaining:
        report = report_key(pairs[source][0])
        if report not in reports and len(sources) < panel_size:
            sources.append(source)
            reports.add(report)
    hold_sources, held_reports = [], set()
    hold_pool = sorted(set(pairs) - set(sources))
    random.Random(seed + 1).shuffle(hold_pool)
    for source in hold_pool:
        report = report_key(pairs[source][0])
        if report not in reports | held_reports and len(hold_sources) < holdout_size:
            hold_sources.append(source)
            held_reports.add(report)
    return dict(
        panel=[task for source in sources for task in pairs[source]],
        holdout=[task for source in hold_sources for task in pairs[source]],
        selected_sources=sources,
        selected_reports=sorted(reports),
        mixed_selected_sources=mixed_selected,
        prior_mixed_task_ids=mixed_tasks,
        random_fill_sources=[source for source in sources if source not in mixed_selected],
        holdout_sources=hold_sources,
        holdout_reports=sorted(held_reports),
    )


def prepare(output=DEFAULT):
    output = Path(output).resolve()
    tasks_path = CREDIT / "data/train.jsonl"
    probe_path = CREDIT / "probe/parents/trajectory_results.jsonl"
    old_train_path = FINANCIAL / "data/train.jsonl"
    teacher_path = FINANCIAL / "teacher/train_requests.jsonl"
    tasks, probe = read_rows(tasks_path), read_rows(probe_path)
    chosen = select_panel(tasks, probe)
    data = output / "data"
    write_rows(data / "train_panel.jsonl", chosen["panel"])
    write_rows(data / "holdout.jsonl", chosen["holdout"])
    pairs = paired_questions(chosen["panel"])
    order = chosen["selected_sources"]
    if len(order) < 4:
        raise ValueError("at least four paired base questions needed for the planned batch")
    batches = []
    for index in range(6):
        start = (index * 4) % len(order)
        sources = [order[(start + offset) % len(order)] for offset in range(4)]
        batch = [task for source in sources for task in pairs[source]]
        path = data / f"batch_round{index:03d}.jsonl"
        write_rows(path, batch)
        batches.append(
            dict(
                round=index,
                path=str(path),
                sha256=sha(path),
                base_questions=sources,
                tasks=len(batch),
                parent_trajectories=4 * len(batch),
            )
        )
    original_train = read_rows(old_train_path)
    original_index = {task["task_id"]: task for task in original_train}
    hold_reports = set(chosen["holdout_reports"])
    teacher, excluded = [], 0
    for record in read_rows(teacher_path):
        if record["task_id"] not in original_index:
            raise ValueError("teacher task must map to historical Train dataset")
        if report_key(original_index[record["task_id"]]) in hold_reports:
            excluded += 1
        else:
            teacher.append(record)
    write_rows(data / "teacher_filtered.jsonl", teacher)
    write_rows(data / "midpoint_panel.jsonl", chosen["panel"][:4])
    exposure = Counter(row["task_id"] for row in probe)
    exposure_sources = defaultdict(set)
    for row in probe:
        exposure_sources[row["task_id"]].add("strong_sft_probe128")
    for path in sorted((OLD / "training").glob("*/round_*/collect/trajectory_results.jsonl")):
        arm = path.parents[2].name
        for row in read_rows(path):
            exposure[row["task_id"]] += 1
            exposure_sources[row["task_id"]].add(arm)
    write_json(
        data / "exposure_ledger.json",
        dict(
            scope=(
                "known strong-SFT probe and completed MC-cycle Train collections; "
                "not complete pretraining history"
            ),
            previous_task_trajectory_counts=dict(sorted(exposure.items())),
            previous_task_sources={
                key: sorted(value) for key, value in sorted(exposure_sources.items())
            },
            panel_prior_sample_counts={
                task["task_id"]: exposure[task["task_id"]] for task in chosen["panel"]
            },
            holdout_prior_sample_counts={
                task["task_id"]: exposure[task["task_id"]] for task in chosen["holdout"]
            },
            not_previously_sampled_in_listed_runs=sum(
                exposure[task["task_id"]] == 0 for task in tasks
            ),
            not_a_claim_of_no_prior_report_or_pretraining_exposure=True,
        ),
    )
    manifest = dict(
        profile="financial_learning_signal_panel_v2",
        status="FROZEN",
        seed=SEED,
        selection=(
            "prior Train mixed-success base questions then one fixed random fill; no rescreening"
        ),
        train_panel_path=str(data / "train_panel.jsonl"),
        holdout_path=str(data / "holdout.jsonl"),
        teacher_filtered_path=str(data / "teacher_filtered.jsonl"),
        midpoint_panel_path=str(data / "midpoint_panel.jsonl"),
        panel_tasks=len(chosen["panel"]),
        panel_base_questions=len(chosen["selected_sources"]),
        panel_reports=chosen["selected_reports"],
        panel_source_ids=chosen["selected_sources"],
        selected_prior_mixed_source_ids=chosen["mixed_selected_sources"],
        random_fill_source_ids=chosen["random_fill_sources"],
        prior_mixed_task_ids=chosen["prior_mixed_task_ids"],
        holdout_tasks=len(chosen["holdout"]),
        holdout_reports=chosen["holdout_reports"],
        holdout_source_ids=chosen["holdout_sources"],
        pairs_kept_together=True,
        report_overlap_panel_holdout=False,
        teacher_records=len(teacher),
        teacher_excluded_holdout_records=excluded,
        teacher_tasks=len({record["task_id"] for record in teacher}),
        teacher_used_privileged_gold_to_choose_steps=True,
        teacher_generation_new_tokens=0,
        teacher_previous_generation_cost_not_zero_claim=True,
        holdout_excluded_from_this_cycle_rl_and_continued_sft=True,
        holdout_not_unseen_by_initial_sft_claim=True,
        old_sft_train_reports=len({report_key(task) for task in original_train}),
        holdout_reports_seen_by_initial_sft=sorted(
            hold_reports & {report_key(task) for task in original_train}
        ),
        new_test_opened=False,
        historical_market_oos_opened=False,
        baseline_repetitions=4,
        baseline_planned_trajectories=4 * len(chosen["panel"]),
        batches=batches,
        inputs={
            str(path): sha(path) for path in (tasks_path, probe_path, old_train_path, teacher_path)
        },
        output_sha256={path.name: sha(path) for path in sorted(data.glob("*.jsonl"))},
    )
    write_json(data / "manifest.json", manifest)
    return manifest


def action_stage(row):
    action = row.get("action") or {}
    name = action.get("name", "invalid_or_missing")
    if name == "read_document":
        title = str((row.get("tool_result") or {}).get("title", "")).lower()
        return "read_financial_report" if "annual report" in title else "read_other_evidence"
    return dict(
        search_docs="search",
        calculate="metric_calculation",
        solve="constraint_execution",
        submit="submission",
    ).get(name, name)


def normalized(values):
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    return [(value - mean) / std if std > 1e-12 else 0.0 for value in values]


def inspect_round(path):
    summary = json.loads((path / "train/summary.json").read_text())
    parents = read_rows(path / "collect/trajectory_results.jsonl")
    requests = read_rows(path / "collect/requests.jsonl")
    groups = defaultdict(list)
    for parent in parents:
        groups[parent["task_id"]].append(parent)
    parent_map = {row["trajectory_key"]: row for row in parents}
    raw_rloo = {}
    diversity = []
    for task_id, group in groups.items():
        if len(group) != 4:
            raise ValueError(f"expected actual four-parent group: {path}/{task_id}")
        total = math.fsum(row["terminal_reward"] for row in group)
        for row in group:
            raw_rloo[row["trajectory_key"]] = (
                row["terminal_reward"] - (total - row["terminal_reward"]) / 3
            )
        rewards = [row["terminal_reward"] for row in group]
        diversity.append(
            dict(
                task_id=task_id,
                rewards=rewards,
                mixed_reward=max(rewards) - min(rewards) > 1e-12,
                mixed_success=len({row.get("success") is True for row in group}) > 1,
                mixed_business=len({row.get("business_correct") is True for row in group}) > 1,
                mixed_metric=len(
                    {(row.get("verification") or {}).get("metric_correct") for row in group}
                )
                > 1,
                calls=[row["calls_used"] for row in group],
                transaction_costs=[
                    (row.get("metrics") or {}).get("transaction_cost") for row in group
                ],
            )
        )
    mc_path = path / "mc/state_value_baselines.jsonl"
    mc = read_rows(mc_path) if mc_path.exists() else []
    completed = {row["request_key"]: row for row in mc if row.get("status") == "COMPLETE"}
    usable = [
        row for row in requests if row.get("prompt_token_ids") and row.get("response_token_ids")
    ]
    rloo_values = [raw_rloo[row["trajectory_key"]] for row in usable]
    actual_values = [
        parent_map[row["trajectory_key"]]["terminal_reward"]
        - completed[row["request_key"]]["mc_baseline"]
        if row["request_key"] in completed
        else raw_rloo[row["trajectory_key"]]
        for row in usable
    ]
    actual_normalized = normalized(actual_values)
    rloo_normalized = normalized(rloo_values)
    executed = Counter(summary.get("batch_executed_trajectory_keys", []))
    stages = defaultdict(Counter)
    impacts = []
    for row, raw, adv, rloo_adv in zip(
        usable, actual_values, actual_normalized, rloo_normalized, strict=True
    ):
        bucket = stages[action_stage(row)]
        bucket["parent_requests"] += 1
        bucket["raw_nonzero_advantage_requests"] += abs(raw) > 1e-12
        bucket["normalized_nonzero_advantage_requests"] += abs(adv) > 1e-12
        uses = executed[row["trajectory_key"]] if adv != 0 else 0
        bucket["actual_backprop_request_uses"] += uses
        bucket["actual_backprop_response_tokens"] += uses * len(row["response_token_ids"])
        bucket["mc_replacements"] += row["request_key"] in completed
        if row["request_key"] in completed:
            local = completed[row["request_key"]]
            impacts.append(
                dict(
                    request_key=row["request_key"],
                    stage=action_stage(row),
                    rloo_raw_advantage=raw_rloo[row["trajectory_key"]],
                    mc_raw_advantage=raw,
                    rloo_normalized_advantage=rloo_adv,
                    mc_normalized_advantage=adv,
                    raw_sign_changed=(raw > 0) - (raw < 0)
                    != (raw_rloo[row["trajectory_key"]] > 0)
                    - (raw_rloo[row["trajectory_key"]] < 0),
                    normalized_sign_changed=(adv > 0) - (adv < 0)
                    != (rloo_adv > 0) - (rloo_adv < 0),
                    aux_returns=local["aux_returns"],
                    auxiliary_returns_differ=max(local["aux_returns"]) - min(local["aux_returns"])
                    > 1e-12,
                    mc_baseline=local["mc_baseline"],
                )
            )
    updates_path = path / "train/optimizer_updates.jsonl"
    updates = read_rows(updates_path) if updates_path.exists() else []
    return dict(
        round=path.name,
        train_status=summary["status"],
        tasks=len(groups),
        parent_trajectories=len(parents),
        parent_requests=len(usable),
        successes=sum(row.get("success") is True for row in parents),
        business_correct=sum(row.get("business_correct") is True for row in parents),
        task_diversity=diversity,
        mixed_reward_groups=sum(row["mixed_reward"] for row in diversity),
        mixed_success_groups=sum(row["mixed_success"] for row in diversity),
        mixed_metric_groups=sum(row["mixed_metric"] for row in diversity),
        advantages_std=summary["advantages_std"],
        optimizer_updates=summary["optimizer_updates"],
        trainable_hash_changed=summary["trainable_sha256_before"]
        != summary["trainable_sha256_after"],
        backprop_request_uses=summary.get("backprop_request_uses", 0),
        backprop_response_token_uses=summary.get("backprop_response_token_uses", 0),
        training_trajectory_uses=summary.get("training_trajectory_uses", 0),
        stages={key: dict(value) for key, value in sorted(stages.items())},
        mc_records=len(mc),
        mc_complete=len(completed),
        mc_impact=impacts,
        selected_anchor_decisions=sum(
            (row.get("mc_prefix_selection") or {}).get("selected") is True for row in requests
        ),
        selection_block_reasons=dict(
            Counter(
                (row.get("mc_prefix_selection") or {}).get("block_reason")
                for row in requests
                if (row.get("mc_prefix_selection") or {}).get("block_reason")
            )
        ),
        optimizer_records=updates,
        actual_new_generation=0,
        reserved_test_read=False,
    )


def inspect(output=DEFAULT):
    output = Path(output).resolve()
    arms = {}
    for arm in ("rloo_seed42", "mc_uniform_seed42", "mc_priority_seed42"):
        rounds = [inspect_round(path) for path in sorted((OLD / "training" / arm).glob("round_*"))]
        stages = defaultdict(Counter)
        for row in rounds:
            for stage, counts in row["stages"].items():
                stages[stage].update(counts)
        total_keys = (
            "tasks",
            "parent_trajectories",
            "parent_requests",
            "optimizer_updates",
            "backprop_request_uses",
            "backprop_response_token_uses",
            "training_trajectory_uses",
            "mixed_reward_groups",
            "mixed_success_groups",
            "mixed_metric_groups",
            "mc_complete",
            "selected_anchor_decisions",
        )
        totals = {key: sum(row[key] for row in rounds) for key in total_keys}
        totals.update(
            rounds=len(rounds),
            zero_advantage_rounds=[row["round"] for row in rounds if row["advantages_std"] == 0],
            mc_aux_return_disagreement_pairs=sum(
                impact["auxiliary_returns_differ"] for row in rounds for impact in row["mc_impact"]
            ),
            mc_raw_sign_changes=sum(
                impact["raw_sign_changed"] for row in rounds for impact in row["mc_impact"]
            ),
            mc_normalized_sign_changes=sum(
                impact["normalized_sign_changed"] for row in rounds for impact in row["mc_impact"]
            ),
            mc_coverage=sum(row["mc_complete"] for row in rounds)
            / sum(row["parent_requests"] for row in rounds),
        )
        arms[arm] = dict(
            totals=totals,
            stages={key: dict(value) for key, value in sorted(stages.items())},
            rounds=rounds,
        )
    result = dict(
        profile="financial_learning_existing_signal_inspection_v2",
        source_run=str(OLD),
        status="COMPLETE",
        new_generation_tokens=0,
        gpu_used=False,
        reserved_test_read=False,
        stages_are_diagnostic_actual_parent_actions_not_selection_inputs=True,
        normalization=(
            "recomputed whole usable-request batch population mean/std, matching historical trainer"
        ),
        mean_ratio_not_kl=True,
        post_update_per_request_probabilities_available=False,
        gradient_norm_is_optimizer_minibatch_not_isolated_action_gradient=True,
        arms=arms,
    )
    directory = output / "diagnostics"
    write_json(directory / "signal_readout.json", result)
    report = [
        "# 第一步：已有记录的实际训练信号",
        "",
        "状态：COMPLETE。仅CPU读取历史Train记录，新增生成0；未读取Test或市场OOS。",
        "",
        "| 臂 | 任务实例/父迹 | 有效更新 | 零方差轮数 | 反传响应token | MC替换/请求 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, item in arms.items():
        t = item["totals"]
        report.append(
            f"| {arm} | {t['tasks']}/{t['parent_trajectories']} | {t['optimizer_updates']} | "
            f"{len(t['zero_advantage_rounds'])}/{t['rounds']} | "
            f"{t['backprop_response_token_uses']} | "
            f"{t['mc_complete']}/{t['parent_requests']} |"
        )
    report += [
        "",
        "## 动作阶段的实际反传",
        "",
        "| 臂 | 阶段 | 请求 | 原始非零优势 | 白化后非零优势 | 实际反传请求 | MC替换 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, item in arms.items():
        for stage, s in item["stages"].items():
            report.append(
                f"| {arm} | {stage} | {s['parent_requests']} | "
                f"{s['raw_nonzero_advantage_requests']} | "
                f"{s['normalized_nonzero_advantage_requests']} | "
                f"{s['actual_backprop_request_uses']} | {s['mc_replacements']} |"
            )
    report += [
        "",
        "## 可确认与仍缺证据",
        "",
        "- 有效更新确实发生，存在非零梯度和改变的权重；不能称未训练。"
        "数百万输入＋输出采样token不等于实际反传量。",
        "- 两MC臂均有5轮全批优势方差为零，无有效更新。"
        "MC覆盖、回报差异和相对RLOO优势变化见JSON逐点记录。",
        "- 阶段请求获得非零优势不等于其取数选择得到正确因果信用；"
        "白化还可能让原始零优势请求变成非零，须区分两者。",
        "- 梯度范数来自混合请求的optimizer minibatch，不能归因于单个计算动作。"
        "平均ratio不是KL；旧日志没有逐请求更新后概率。",
        "- 已知终点未修好任何SFT失败题，但尚不能区分支持稀疏、"
        "有效学习量、关键动作更新不足和估计目标价值。",
        "- 第二步用独立重采的混合题正控、多题minibatch和两档学习率验证响应；"
        "不把Train正控当总体泛化。",
        "",
        "[逐批与逐点JSON](signal_readout.json) · [冻结任务与教师过滤](../data/manifest.json)",
        "",
    ]
    write_frozen(directory / "SIGNAL_REPORT.md", "\n".join(report))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.output)
    print(
        canonical(
            dict(
                stage="data_ready",
                output=str(args.output),
                panel_tasks=manifest["panel_tasks"],
                holdout_tasks=manifest["holdout_tasks"],
                teacher_records=manifest["teacher_records"],
            )
        ),
        flush=True,
    )
    if not args.prepare_only:
        result = inspect(args.output)
        print(
            canonical(
                dict(
                    stage="signal_complete",
                    totals={key: value["totals"] for key, value in result["arms"].items()},
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
