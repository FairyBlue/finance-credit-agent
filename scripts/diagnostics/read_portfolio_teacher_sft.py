#!/usr/bin/env python3
"""Read the authorized portfolio teacher/SFT step; keep the two Train subsets separate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SUBSETS = ("learning", "development")
RULES = ("hold", "momentum", "reversal", "defensive", "cash")
LABELS = {
    "hold": "被动等权持有",
    "momentum": "固定动量",
    "reversal": "固定反转",
    "defensive": "固定防守",
    "cash": "全现金参照",
    "common_sft": "原共同SFT",
    "teacher_sft": "教师SFT终点",
}
METRICS = (
    "raw_net_return",
    "benchmark_excess",
    "terminal_financial_Y",
    "transaction_cost",
    "max_drawdown",
    "turnover",
)
MARKER = "**2026-10-28：组合管理第二步执行中。**"
DONE_MARKER = "**2026-10-28：组合管理第二步已完成。**"


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def number(value, scale=1):
    return "NA" if value is None else f"{value * scale:.5f}"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def write_csv(path, records, fieldnames=None):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(records[0]))
        writer.writeheader()
        writer.writerows(
            {key: "NA" if value is None else value for key, value in row.items()} for row in records
        )


def ensure_baselines(root, manifest):
    """CPU-only shared ledger replays; may run while the GPU is training."""
    from src.agents.portfolio_management_rules import run as run_rules

    for subset in SUBSETS:
        destination = root / "baselines" / subset
        summary_path = destination / "summary.json"
        if not summary_path.exists():
            run_rules(Path(manifest["prepared_paths"][f"{subset}_eval"]), destination)
        summary = read(summary_path)
        expected = manifest[f"evaluation_{subset}_episode_ids"]
        if summary["status"] != "COMPLETE" or set(summary["episode_ids"]) != set(expected):
            raise RuntimeError(f"CPU baseline subset does not match the declared {subset} IDs")


def planned_rows(records, episode_ids, strategy, subset):
    """A failed or absent slot stays present and never contributes a terminal value."""
    mapping = {row["episode_id"]: row for row in records}
    if len(mapping) != len(records) or not set(mapping).issubset(episode_ids):
        raise RuntimeError(f"duplicate or undeclared episodes: {subset}/{strategy}")
    normalized = []
    for episode_id in episode_ids:
        source = mapping.get(episode_id, dict(episode_id=episode_id, status="MISSING"))
        normalized.append(dict(source, subset=subset, strategy=strategy))
    return normalized


def aggregate(records):
    complete = sum(row["status"] == "COMPLETE" for row in records)
    all_complete = complete == len(records)
    return dict(
        planned_episodes=len(records),
        observed_episodes=sum(row["status"] != "MISSING" for row in records),
        complete_episodes=complete,
        failed_or_missing_episodes=len(records) - complete,
        all_planned_trajectories_complete=all_complete,
        **{
            metric: (
                math.fsum(row["metrics"][metric] for row in records) / len(records)
                if all_complete
                and all(finite(row.get("metrics", {}).get(metric)) for row in records)
                else None
            )
            for metric in METRICS
        },
    )


def concentration(paired):
    deltas = [row["delta_candidate_minus_reference"]["raw_net_return"] for row in paired["rows"]]
    if any(value is None for value in deltas):
        return dict(
            status="NA_INCOMPLETE_PAIRS",
            largest_positive_episode_id=None,
            largest_positive_delta_pp=None,
            largest_positive_mean_contribution_pp=None,
            largest_positive_share_of_all_positive=None,
            mean_delta_without_largest_positive_pp=None,
            top_positive_windows=[],
        )
    positive = sorted(
        (
            (delta, row["episode_id"])
            for delta, row in zip(deltas, paired["rows"], strict=True)
            if delta > 0
        ),
        reverse=True,
    )
    largest, episode_id = positive[0] if positive else (None, None)
    positive_sum = math.fsum(value for value in deltas if value > 0)
    return dict(
        status="COMPLETE",
        largest_positive_episode_id=episode_id,
        largest_positive_delta_pp=100 * largest if largest is not None else None,
        largest_positive_mean_contribution_pp=(
            100 * largest / len(deltas) if largest is not None else None
        ),
        largest_positive_share_of_all_positive=(
            largest / positive_sum if largest is not None else None
        ),
        mean_delta_without_largest_positive_pp=(
            100
            * math.fsum(
                value
                for value, row in zip(deltas, paired["rows"], strict=True)
                if row["episode_id"] != episode_id
            )
            / (len(deltas) - 1)
            if largest is not None
            else 100 * math.fsum(deltas) / len(deltas)
        ),
        top_positive_windows=[
            dict(
                episode_id=eid, delta_pp=100 * delta, mean_contribution_pp=100 * delta / len(deltas)
            )
            for delta, eid in positive[:5]
        ],
        descriptive_only=True,
    )


def outcome(summary, paired):
    development = summary["development"]
    delta = paired["development"]["mean_net_return_delta_pp"]
    excess = development["teacher_sft"]["benchmark_excess"]
    if delta is None or excess is None:
        return "开发区存在失败或缺失轨迹，固定32分母收益与配对结论保持NA。"
    if delta > 0 and excess > 0:
        return "教师SFT在本轮开发区同时高于原共同SFT与被动等权持有；这是探索性开发读数。"
    if delta > 0:
        text = "教师SFT在本轮开发区均值略高于原共同SFT，但没有超过被动等权持有。"
        without = paired["development"]["concentration"][
            "mean_delta_without_largest_positive_pp"
        ]
        if without is not None and without < 0:
            text += "去掉最大正贡献窗后差值转负，尚未建立稳固的开发收益增量。"
        return text
    if delta < 0:
        return "教师SFT在本轮开发区低于原共同SFT，本轮监督初始化未建立开发收益增量。"
    return "教师SFT与原共同SFT在本轮开发区平均净收益相同。"


def plots(destination, summary, paired, steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    strategies = list(LABELS)
    for ax, subset in zip(axes, SUBSETS, strict=True):
        for index, strategy in enumerate(strategies):
            value = summary[subset][strategy]["benchmark_excess"]
            if value is None:
                ax.text(index, 0, "NA", ha="center")
            else:
                ax.bar(
                    index,
                    100 * value,
                    color=(
                        "tab:orange"
                        if strategy == "teacher_sft"
                        else "tab:blue"
                        if strategy == "common_sft"
                        else "0.65"
                    ),
                )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(
            title=f"{subset}: fixed 32-window denominator",
            ylabel="Mean excess over costed equal-weight hold (pp)",
        )
        ax.set_xticks(range(len(strategies)), strategies, rotation=35, ha="right")
    figure.suptitle("Portfolio teacher SFT: separate Train learning/development subsets")
    figure.tight_layout()
    figure.savefig(destination / "comparison.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, subset in zip(axes, SUBSETS, strict=True):
        group = paired[subset]
        for index, row in enumerate(group["rows"]):
            value = row["delta_candidate_minus_reference"]["raw_net_return"]
            if value is None:
                ax.text(index, 0, "NA", ha="center", fontsize=6)
            else:
                ax.bar(index, 100 * value, color="tab:blue" if value >= 0 else "tab:red")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(
            range(len(group["rows"])),
            [row["episode_id"].rsplit("-", 1)[-1] for row in group["rows"]],
            rotation=65,
            fontsize=7,
        )
        ax.set(title=f"{subset}: teacher SFT minus common SFT", ylabel="Net-return delta (pp)")
    figure.tight_layout()
    figure.savefig(destination / "paired_deltas.png", dpi=160)
    plt.close(figure)

    if steps:
        figure, ax = plt.subplots(figsize=(10, 4))
        ax.plot([row["step"] for row in steps], [row["loss"] for row in steps], linewidth=0.8)
        ax.set(
            title="One fixed SFT epoch; losses are on different minibatches",
            xlabel="Optimizer step",
            ylabel="Action ID + EOS cross entropy",
        )
        figure.tight_layout()
        figure.savefig(destination / "training_loss.png", dpi=160)
        plt.close(figure)


def publish_docs(root, report, conclusion):
    relative = root.relative_to(ROOT)
    groups, paired = report["subsets"], report["teacher_minus_common_sft"]
    details = []
    for subset, label in (("learning", "学习区"), ("development", "开发区")):
        common, teacher = groups[subset]["common_sft"], groups[subset]["teacher_sft"]
        delta = paired[subset]["mean_net_return_delta_pp"]
        details.append(
            f"{label}净收益{number(common['raw_net_return'], 100)}%→"
            f"{number(teacher['raw_net_return'], 100)}%"
            f"（{number(delta)}pp，{teacher['complete_episodes']}/32完整）"
        )
    notice = (
        DONE_MARKER
        + f" {report['training']['optimizer_steps']}次固定LoRA更新与两区预定评估已结束；"
        + "；".join(details).replace("NA%", "NA").replace("NApp", "NA")
        + "。"
        + conclusion
        + "开发区只隔离本轮新增监督，基础模型此前已接触旧Train，不称首次日期泛化。"
        "第三步RL未启动，本轮Valid/Final OOS未访问。"
        f"[第二步报告](../{relative}/REPORT.md)、"
        f"[完整读数](../{relative}/summary/readout.json)、"
        f"[比较图](../{relative}/summary/comparison.png)。\n\n"
    )
    current = ROOT / "docs/CURRENT_STATUS.md"
    content = current.read_text()
    marker = MARKER if MARKER in content else DONE_MARKER
    if marker in content:
        start = content.index(marker)
        end = content.index("\n\n", start) + 2
        content = content[:start] + notice + content[end:]
    else:
        content = content.replace("# 当前状态与下一步\n\n", "# 当前状态与下一步\n\n" + notice, 1)
    content = content.replace("第一步完成、第二步执行中，第三步未启动", "前两步完成，第三步未启动")
    current.write_text(content)
    history = ROOT / "docs/history/CREDIT_ORACLE_DIAGNOSTICS.md"
    heading = "## 2026-10-28 组合管理第二步：确定性教师与共同SFT"
    if heading not in history.read_text():
        with history.open("a") as stream:
            stream.write(
                f"\n\n{heading}\n\n"
                f"354个学习窗的3540状态、14160条候选执行后被动续持至原终点的确定性分支，"
                f"用于一轮LoRA动作ID监督，共{report['training']['optimizer_steps']}次更新。"
                "45个开发窗与学习终点隔22个交易日，42个交界窗口排除；"
                "两区各32预定评估窗口独立报告，失败保持NA。"
                + "；".join(details).replace("NA%", "NA").replace("NApp", "NA")
                + "。"
                + conclusion
                + "教师值不是当前策略Q，未来结果只作标签；"
                "基础模型有旧Train暴露，开发隔离仅针对新增监督。"
                "第三步RL未启动，本轮未访问Valid/Final OOS。"
                f"[完整报告](../../{relative}/REPORT.md)、"
                f"[配对与贡献读数](../../{relative}/summary/readout.json)。\n"
            )


def run(output, baselines_only=False):
    root = Path(output).resolve()
    manifest = read(root / "data/split_manifest.json")
    for subset in SUBSETS:
        ids = manifest[f"evaluation_{subset}_episode_ids"]
        if (
            len(ids) != len(set(ids))
            or len(ids) != 32
            or not all(eid.startswith("formal-train-") for eid in ids)
        ):
            raise RuntimeError("readout requires 32 declared Train windows per subset")
    if baselines_only:
        ensure_baselines(root, manifest)
        print(json.dumps(dict(status="BASELINES_COMPLETE", trajectories=320)), flush=True)
        return None
    completion = read(root / "completion.json")
    if completion["status"] != "COMPLETE":
        raise RuntimeError("only completion.status=COMPLETE may publish the second-step report")
    ensure_baselines(root, manifest)
    ready = read(root / "data/dataset_ready.json")
    training = read(root / "training/summary.json")
    settings = read(root / "training/settings.json")
    steps = rows(root / "training/steps.jsonl")
    if ready["status"] != "COMPLETE" or training["status"] != "COMPLETE":
        raise RuntimeError("teacher data and SFT training must be complete before publication")
    from src.agents.portfolio_teacher_eval import paired_readout

    records, summary, paired, evaluations, baseline_checks = [], {}, {}, {}, {}
    for subset in SUBSETS:
        ids = manifest[f"evaluation_{subset}_episode_ids"]
        rules = rows(root / "baselines" / subset / "trajectory_results.jsonl")
        baseline_checks[subset] = read(root / "baselines" / subset / "checks.json")
        groups = {
            strategy: planned_rows(
                [row for row in rules if row["strategy"] == strategy], ids, strategy, subset
            )
            for strategy in RULES
        }
        evaluations[subset] = {}
        for strategy in ("common_sft", "teacher_sft"):
            folder = root / "eval" / strategy / subset
            evaluation = read(folder / "summary.json")
            if evaluation["episode_ids"] != ids or evaluation["evaluation_subset"] != subset:
                raise RuntimeError(f"evaluation identity differs from {subset} manifest")
            groups[strategy] = planned_rows(
                rows(folder / "trajectory_results.jsonl"), ids, strategy, subset
            )
            evaluations[subset][strategy] = {
                key: evaluation[key]
                for key in (
                    "status",
                    "model_path",
                    "model_requests",
                    "prompt_tokens",
                    "response_tokens",
                    "complete_requests",
                    "failed_requests",
                    "constrained_decoding",
                    "constrained_decoding_api",
                    "seed",
                    "zero_shot",
                )
            }
        summary[subset] = {strategy: aggregate(group) for strategy, group in groups.items()}
        comparison = paired_readout(groups["common_sft"], groups["teacher_sft"], ids)
        comparison["concentration"] = concentration(comparison)
        paired[subset] = comparison
        records.extend(row for group in groups.values() for row in group)
    destination = root / "summary"
    destination.mkdir(exist_ok=True)
    summary_rows = [
        dict(subset=subset, strategy=strategy, **metrics)
        for subset, groups in summary.items()
        for strategy, metrics in groups.items()
    ]
    write_csv(destination / "summary.csv", summary_rows)
    write_csv(
        destination / "window_results.csv",
        [
            dict(
                subset=row["subset"],
                strategy=row["strategy"],
                episode_id=row["episode_id"],
                status=row["status"],
                **{
                    metric: (
                        row.get("metrics", {}).get(metric) if row["status"] == "COMPLETE" else None
                    )
                    for metric in METRICS
                },
                action_ids="|".join(row.get("action_ids", [])),
            )
            for row in records
        ],
    )
    write_csv(
        destination / "paired_results.csv",
        [
            dict(
                subset=subset,
                episode_id=row["episode_id"],
                reference_status=row["reference_status"],
                candidate_status=row["candidate_status"],
                complete_pair=row["complete_pair"],
                **{
                    f"delta_{metric}": value
                    for metric, value in row["delta_candidate_minus_reference"].items()
                },
            )
            for subset, comparison in paired.items()
            for row in comparison["rows"]
        ],
    )
    generated = {
        key: sum(
            value[strategy][key]
            for value in evaluations.values()
            for strategy in ("common_sft", "teacher_sft")
        )
        for key in ("model_requests", "prompt_tokens", "response_tokens")
    }
    report = dict(
        status=(
            "COMPLETE"
            if all(row["failed_or_missing_episodes"] == 0 for row in summary_rows)
            else "COMPLETE_WITH_MODEL_FAILURES"
        ),
        authorized_step=2,
        source_split="train",
        evaluation_temperature=0.0,
        subsets=summary,
        teacher_minus_common_sft=paired,
        evaluation_episode_ids={
            subset: manifest[f"evaluation_{subset}_episode_ids"] for subset in SUBSETS
        },
        split_counts=manifest["counts"],
        learning_end=manifest["learning_end"],
        development_start=manifest["development_start"],
        unused_gap_sessions=manifest["unused_gap_session_count"],
        teacher_data=ready,
        training=training,
        training_settings=settings,
        actual_training_log_steps=len(steps),
        training_step_sequence=[row["step"] for row in steps],
        merge_receipt=read(root / "current_actor/merge_receipt.json"),
        evaluations=evaluations,
        actual_new_generation=generated,
        action_counts={
            subset: {
                strategy: dict(
                    Counter(
                        action
                        for row in records
                        if row["subset"] == subset and row["strategy"] == strategy
                        for action in row.get("action_ids", [])
                    )
                )
                for strategy in ("common_sft", "teacher_sft")
            }
            for subset in SUBSETS
        },
        shared_ledger_reference_checks=baseline_checks,
        teacher_current_policy_q=False,
        future_teacher_labels_in_model_prompt=False,
        development_isolation="new teacher supervision only; source model has prior Train exposure",
        first_date_generalization_claim=False,
        independent_market_sample_claim=False,
        failed_slots_imputed=False,
        survivor_aggregate_used=False,
        formal_valid_access=False,
        formal_final_oos_access=False,
        rl_optimizer_updates=0,
        step3_started=False,
    )
    conclusion = outcome(summary, paired)
    report["conclusion"] = conclusion
    write(destination / "readout.json", report)
    plots(destination, summary, paired, steps)
    text = [
        "# 组合管理第二步：确定性教师与共同SFT\n",
        f"状态{report['status']}。{conclusion}",
        "仅执行第二步教师数据、固定一轮SFT与冻结终点评估；第三步RL未启动。"
        "本轮未读取Valid或Final OOS。\n",
        "354个学习窗口、3540个真实状态、14160条确定性候选分支："
        "执行一个候选动作后被动续持到原窗口j20，"
        "用扣费后相对同状态hold的增量产生动作标签。它不是当前策略Q估计。"
        "未来结果只用于学习标签，模型输入仍是当时可见信息。",
        f"固定LoRA r16/alpha32、LR5e-5、microbatch1、累计16条后更新，"
        f"一轮实际完成{training['optimizer_steps']}次更新；监督动作ID与EOS，"
        "完整提示参与注意力。冻结最终模型，不按评估挑轮次。\n",
        "## 两区终点读数\n",
        "学习区354窗，开发区45窗，交界排除42窗，间隔22个交易日；"
        "各自预定32个评估窗口。基础共同模型此前已接触旧Train，"
        "这次日期隔离只针对新增教师监督，不是首次日期泛化证据。"
        "窗口重叠，不当作32个独立市场样本；学习区与开发区分别报告，不与旧接口拼均值。",
        "所有组合方案共同扣费等权建仓，随后j1/6/11/16选择四动作ID，"
        "原金融Y仅为诊断，主读数是扣费净收益及相对被动等权持有超额。"
        "全现金为从现金出发的独立参照。动作合法性由统一接口与约束解码保证。\n",
    ]
    for subset, label in (("learning", "学习区"), ("development", "开发区")):
        text.extend(
            [
                f"### {label}（固定32分母）\n",
                "| 方案 | 净收益% | 主动超额pp | 金融Y | 总费用均值 | 最大回撤%均值 | 完整轨迹 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy, metrics in summary[subset].items():
            text.append(
                f"| {LABELS[strategy]} | {number(metrics['raw_net_return'], 100)} | "
                f"{number(metrics['benchmark_excess'], 100)} | "
                f"{number(metrics['terminal_financial_Y'])} | "
                f"{number(metrics['transaction_cost'])} | "
                f"{number(metrics['max_drawdown'], 100)} | {metrics['complete_episodes']}/32 |"
            )
        comparison, focused = paired[subset], paired[subset]["concentration"]
        text.append(
            "\n教师减原共同SFT：净收益"
            f"{number(comparison['mean_net_return_delta_pp'])}pp，"
            f"{number(comparison['net_return_window_wins']).replace('.00000', '')}胜/"
            f"{number(comparison['net_return_window_losses']).replace('.00000', '')}负/"
            f"{number(comparison['net_return_window_ties']).replace('.00000', '')}平，"
            f"{comparison['complete_pairs']}/32完整配对。"
            f"最大正贡献窗口`{focused['largest_positive_episode_id'] or 'NA'}`，"
            f"对总体均值贡献{number(focused['largest_positive_mean_contribution_pp'])}pp；"
            "去掉该窗口后平均差"
            f"{number(focused['mean_delta_without_largest_positive_pp'])}pp。"
            "贡献集中度仅作描述，不用于换窗口或选模型。\n"
        )
    text.append("## 动作与成本\n")
    for subset, label in (("learning", "学习区"), ("development", "开发区")):
        counts = report["action_counts"][subset]["teacher_sft"]
        total = sum(counts.values())
        leading = max(counts, key=counts.get) if counts else None
        before, after = summary[subset]["common_sft"], summary[subset]["teacher_sft"]
        text.append(
            f"{label}教师模型动作分布：`{counts}`。"
            + (
                f"最常见动作`{leading}`占{counts[leading]}/{total}"
                f"（{100 * counts[leading] / total:.2f}%）。"
                if leading else ""
            )
            + f"每轨迹费用{number(before['transaction_cost'])}→"
            f"{number(after['transaction_cost'])}。"
        )
    text.append(
        "新模型明显偏向反转动作，不能把相对旧模型的均值小增益解释为已经学会灵活调仓。"
        "开发区固定反转与被动等权的结果也保留在同表，未以动作类别变化替代经济收益判据。"
    )
    text.extend(
        [
            "## 实际开销与保留范围\n",
            f"两模型各预定64条父轨迹槽，共实际生成{generated['model_requests']}请求，"
            f"{generated['prompt_tokens']}输入token、{generated['response_tokens']}输出token。"
            "教师14160条确定性分支与320条固定规则轨迹均为CPU共享账本回放，"
            "不是LLM采样请求；监督训练token与更新数另见training/summary.json。",
            "失败或缺失槽保留原样，整组终局均值和完整配对均值在固定分母下记NA，"
            "不使用成功子集均值。本轮不追加标签、训练轮数、窗口或RL步骤。\n",
            "[比较图](summary/comparison.png)、[逐窗配对图](summary/paired_deltas.png)、"
            "[训练损失](summary/training_loss.png)、[分区汇总CSV](summary/summary.csv)、"
            "[逐窗CSV](summary/window_results.csv)、[配对CSV](summary/paired_results.csv)、"
            "[完整JSON](summary/readout.json)。最终模型位于current_actor，"
            "原始输入输出与轨迹位于eval目录；共同初始化不覆盖原共同模型。",
        ]
    )
    rendered = "\n\n".join(text).replace("|\n\n|", "|\n|")
    (root / "REPORT.md").write_text(rendered + "\n")
    publish_docs(root, report, conclusion)
    print(
        json.dumps(
            dict(
                status=report["status"],
                report=str(root / "REPORT.md"),
                sft_optimizer_updates=training["optimizer_steps"],
                rl_optimizer_updates=0,
                step3_started=False,
            )
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.baselines_only)


if __name__ == "__main__":
    main()
