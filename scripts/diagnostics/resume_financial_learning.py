#!/usr/bin/env python3
"""Finish only the five unstarted Dev evaluations in the authorized V2 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_financial_learning import (
    ARM_CAP,
    ARMS,
    DEFAULT,
    DEV,
    TOTAL_CAP,
    BudgetStop,
    digest,
    metrics,
)
from run_financial_learning import (
    Controller as LearningController,
)
from run_financial_mc import generation_tokens
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, now, read, save

RECOVERY_NAME = "continued_sft_dev_recovery20261105"
BASELINE_CAP = 2_800_000
EVALUATION_CAP = 16_200_000
IDENTITY_FIELDS = (
    "adapter_path",
    "adapter_sha256",
    "base_model_path",
    "seed",
    "temperature",
    "repetitions",
    "environment",
    "diagnostic_mode",
    "observation_profile",
    "decode_mode",
    "max_context_tokens",
    "max_response_tokens",
    "max_tool_calls",
    "engine_mode",
    "action_protocol",
    "evaluation_fingerprint",
)


def original_directory(root):
    return Path(root) / "eval/continued_sft_dev"


def recovery_directory(root):
    return Path(root) / "eval" / RECOVERY_NAME


def contract_path(root):
    return Path(root) / "recovery20261105/contract.json"


def missing_task_ids(tasks, completed):
    """Reject unknown, repeated or partial original outcomes before sampling anything."""
    expected = [item["task_id"] for item in tasks]
    actual = [item["task_id"] for item in completed]
    if len(expected) != 100 or len(set(expected)) != 100:
        raise RuntimeError("original fixed Dev must contain100unique tasks")
    if len(actual) != 95 or len(set(actual)) != 95:
        raise RuntimeError("recovery requires95unique original outcomes")
    if not set(actual) < set(expected):
        raise RuntimeError("original outcomes must be a strict subset of the fixed Dev")
    if any(
        item.get("status") != "COMPLETE" or item.get("replicate_index") != 0 for item in completed
    ):
        raise RuntimeError("only complete single-replica original outcomes can be retained")
    return [task for task in expected if task not in set(actual)]


def validate_identity(original, additional):
    inferred = {}
    for key in IDENTITY_FIELDS:
        actual = additional.get(key)
        if key == "evaluation_fingerprint" and actual is None:
            # The legacy collector records this field only for directories ending
            # in _dev/_test. Recompute the exact same artifact fingerprint for
            # the recovery directory; never weaken the identity comparison.
            from src.agents.mandate_repair_collect import _evaluation_fingerprint

            actual = _evaluation_fingerprint(Path(additional["adapter_path"]))
            inferred[key] = dict(
                value=actual,
                source="recomputed_from_recorded_adapter_weights_and_config",
                adapter_path=additional["adapter_path"],
                raw_summary_modified=False,
            )
        if key not in original or actual != original[key]:
            raise RuntimeError(f"recovery changed frozen evaluation identity:{key}")
    return inferred


def prepare(root):
    root = Path(root).resolve()
    source = original_directory(root)
    summary = read(source / "summary.json")
    if summary.get("planned_trajectories") != 100 or summary.get("complete_trajectories") != 95:
        raise RuntimeError("expected retained95of100 continuedSFTDev evaluation")
    if (
        summary.get("seed") != 43
        or summary.get("temperature") != 0
        or summary.get("repetitions") != 1
    ):
        raise RuntimeError("expected original native greedy seed43 evaluation")
    if summary.get("data_sha256") != digest(DEV) or Path(summary["data_path"]).resolve() != DEV:
        raise RuntimeError("original Dev identity changed")
    adapter = root / "training/continued_sft/adapter"
    from src.agents.mandate_repair_train import adapter_digest

    if Path(summary["adapter_path"]).resolve() != adapter:
        raise RuntimeError("continuedSFT endpoint path changed")
    if summary.get("adapter_sha256") != adapter_digest(adapter):
        raise RuntimeError("continuedSFT frozen weights changed")
    train = read(adapter.parent / "summary.json")
    if train.get("exported_adapter_sha256") != summary["adapter_sha256"]:
        raise RuntimeError("continuedSFT endpoint is not its completed training export")
    tasks = rows(DEV)
    original = rows(source / "trajectory_results.jsonl")
    missing = missing_task_ids(tasks, original)
    directory = root / "recovery20261105"
    directory.mkdir(exist_ok=True)
    missing_path = directory / "missing_dev5.jsonl"
    # Retain the original task bytes and order rather than rebuilding task objects.
    raw = "".join(
        line + "\n"
        for line in DEV.read_text().splitlines()
        if json.loads(line)["task_id"] in set(missing)
    )
    if missing_path.exists() and missing_path.read_text() != raw:
        raise RuntimeError("prepared missing-task list changed")
    if not missing_path.exists():
        missing_path.write_text(raw)
    original_sources = {
        str(source / name): digest(source / name)
        for name in (
            "summary.json",
            "trajectory_results.jsonl",
            "requests.jsonl",
            "collection_started.json",
        )
    }
    evaluation_spent = generation_tokens(root / "eval")
    baseline_spent = generation_tokens(root / "baseline")
    if baseline_spent > BASELINE_CAP or evaluation_spent >= EVALUATION_CAP:
        raise RuntimeError("reallocated allowances cannot cover existing consumption")
    contract = dict(
        profile="financial_learning_v2_complete_missing_dev5_20261105",
        original_sources=original_sources,
        original_completed=95,
        fixed_total=100,
        missing_task_ids=missing,
        missing_data=str(missing_path),
        missing_data_sha256=digest(missing_path),
        original_identity={key: summary[key] for key in IDENTITY_FIELDS},
        original_dev_data=str(DEV),
        original_dev_sha256=digest(DEV),
        recovery_output=str(recovery_directory(root)),
        merged_view=str(root / "eval_views/continued_sft_dev"),
        allocation=dict(
            original_baseline=3_000_000,
            original_step2_evaluation=16_000_000,
            baseline=BASELINE_CAP,
            step2_evaluation=EVALUATION_CAP,
            transfer_tokens=200_000,
            total=TOTAL_CAP,
            stage3_training=8_000_000,
            stage3_evaluation=5_000_000,
            changed_before_recovery_generation=True,
        ),
        purpose="complete original fixed denominator; no training, reselection, or Test access",
    )
    path = contract_path(root)
    if path.exists():
        previous = read(path)
        if previous["contract"] != contract:
            raise RuntimeError("frozen recovery contract changed")
    else:
        save(path, dict(frozen_at=now(), contract=contract))
    return read(path)["contract"]


def merge_results(tasks, original_summary, original, additional_summary, additional):
    """Construct100 real outcomes; no inferred success or duplicated trajectories."""
    expected = [task["task_id"] for task in tasks]
    missing = missing_task_ids(tasks, original)
    extra_ids = [item["task_id"] for item in additional]
    if len(extra_ids) != 5 or len(set(extra_ids)) != 5 or set(extra_ids) != set(missing):
        raise RuntimeError("recovery must contain exactly the five previously unstarted tasks")
    if any(
        item.get("status") != "COMPLETE" or item.get("replicate_index") != 0 for item in additional
    ):
        raise RuntimeError("all five recovery outcomes must be complete")
    if (
        additional_summary.get("complete_trajectories") != 5
        or additional_summary.get("planned_trajectories") != 5
    ):
        raise RuntimeError("incomplete recovery denominator")
    inferred_identity = validate_identity(original_summary, additional_summary)
    indexed = {item["task_id"]: item for item in original + additional}
    if len(indexed) != 100 or set(indexed) != set(expected):
        raise RuntimeError("merged evaluation does not match original100task universe")
    result = [indexed[task] for task in expected]
    from src.agents.mandate_repair_metrics import summarize_outcomes

    summary = {
        key: value for key, value in original_summary.items() if key not in {"results", "groups"}
    }
    count_fields = (
        "model_requests",
        "prompt_tokens",
        "response_tokens",
        "total_tokens",
        "admitted_task_groups",
        "complete_model_requests",
        "failed_model_requests",
        "incomplete_generations",
        "parent_model_requests",
        "parent_tokens",
        "aux_model_requests",
        "aux_tokens",
        "aux_prompt_tokens",
        "aux_response_tokens",
        "aux_continuations_started",
        "aux_budget_skipped_continuations",
        "elapsed_seconds",
    )
    for key in count_fields:
        summary[key] = original_summary.get(key, 0) + additional_summary.get(key, 0)
    summary.update(
        status="COMPLETE",
        planned_task_groups=100,
        planned_trajectories=100,
        started_trajectories=100,
        complete_trajectories=100,
        failed_trajectories=0,
        not_started_budget_groups=0,
        successes=sum(row.get("success") is True for row in result),
        success_rate_all_planned=sum(row.get("success") is True for row in result) / 100,
        task_ids=expected,
        outcome_diagnostics=summarize_outcomes(result),
        tool_counts=dict(
            Counter(original_summary.get("tool_counts", {}))
            + Counter(additional_summary.get("tool_counts", {}))
        ),
        derived_merged_evaluation=True,
        generation_billing_source=False,
        original_completed_trajectories=95,
        recovered_trajectories=5,
        new_recovery_generation_tokens=additional_summary["total_tokens"],
        retained_original_generation_tokens=original_summary["total_tokens"],
        token_budget=None,
        unused_budget_tokens=None,
        recovered_identity_fields=inferred_identity,
    )
    return summary, result


class Controller(LearningController):
    def __init__(self, output):
        self.recovery_contract = prepare(output)
        super().__init__(Path(output), resume=True)
        self.state["generation_caps"].update(baseline=BASELINE_CAP, step2_evaluation=EVALUATION_CAP)
        self.persist(
            recovery_contract=str(contract_path(self.output)),
            recovery_contract_sha256=digest(contract_path(self.output)),
            effective_allocation=self.recovery_contract["allocation"],
        )

    def costs(self):
        baseline = generation_tokens(self.output / "baseline")
        evaluation = generation_tokens(self.output / "eval")
        training = {key: generation_tokens(self.output / "training" / key) for key in ARMS}
        stage3 = generation_tokens(self.output / "step3")
        total = baseline + evaluation + sum(training.values()) + stage3
        for used, cap in (
            (baseline, BASELINE_CAP),
            (evaluation, EVALUATION_CAP),
            (total, TOTAL_CAP),
            *((amount, ARM_CAP) for amount in training.values()),
        ):
            if used > cap:
                raise RuntimeError("effective recovery/global allowance exceeded")
        return dict(
            baseline_generation_tokens=baseline,
            rl_training_generation_tokens=training,
            evaluation_generation_tokens=evaluation,
            stage3_generation_tokens=stage3,
            total_new_generation_tokens=total,
            gpu_stage_seconds=sum(
                item.get("seconds", 0)
                for item in self.state.get("stages", [])
                if item.get("gpu_worker")
            ),
            includes_failed_requests=True,
            supervised_tokens_accounted_separately=True,
            derived_views_charged=False,
        )

    def eval_allowance(self):
        return min(
            EVALUATION_CAP - generation_tokens(self.output / "eval"),
            TOTAL_CAP - self.costs()["total_new_generation_tokens"],
        )

    def ready(self):
        super().ready()
        self.validate_original_sources()
        source = self.output / "frozen_sources" / f"recovery_controller_attempt{self.attempt}.py"
        source.write_bytes(Path(__file__).read_bytes())
        self.persist(
            current_controller_source=str(source), current_controller_source_sha256=digest(source)
        )

    def validate_original_sources(self):
        for path, expected in self.recovery_contract["original_sources"].items():
            if digest(path) != expected:
                raise RuntimeError(f"retained original evaluation changed:{path}")

    def baseline(self):
        if read(self.output / "baseline/sft_panel/summary.json").get("complete_trajectories") != 64:
            raise RuntimeError("recovery cannot rerun the independent baseline")

    def claim_gpu(self):
        completed = recovery_directory(self.output) / "summary.json"
        if completed.exists() and read(completed).get("complete_trajectories") == 5:
            self.persist(recovery_compute_mode="CPU_MERGE_COMPLETED_FIVE; NO_GPU_WORKER")
            return
        super().claim_gpu()

    def train_arm(self, key, lr):
        progress = read(self.output / "training" / key / "progress.json")
        if progress.get("status") != "COMPLETE" or len(progress.get("rounds", [])) != 6:
            raise RuntimeError("recovery cannot restart unfinished RL training")
        return Path(progress["final_adapter"])

    def continued_sft(self):
        path = self.output / "training/continued_sft/summary.json"
        if read(path).get("status") not in {"COMPLETE", "COMPLETE_TIME_LIMIT"}:
            raise RuntimeError("recovery cannot restart continuedSFT training")
        return path.parent / "adapter"

    def stage(self, name, module, arguments, gpu_worker=True):
        if name != "recover_continued_sft_dev_missing" or module != COLLECTOR:
            raise RuntimeError("recovery only permits sampling the five missing Dev tasks")
        return super().stage(name, module, arguments, gpu_worker)

    def evaluate(self, key, adapter, split, *, repetitions=1, temperature=0, data=None):
        if key != "continued_sft" or split != "dev":
            directory = self.output / "eval" / f"{key}_{split}"
            if not (directory / "summary.json").exists():
                raise RuntimeError("recovery cannot add an unplanned endpoint")
            if read(directory / "summary.json").get("status") != "COMPLETE":
                raise RuntimeError("another original endpoint is incomplete")
            return read(directory / "summary.json")
        if Path(adapter).resolve() != Path(
            self.recovery_contract["original_identity"]["adapter_path"]
        ):
            raise RuntimeError("recovery endpoint changed")
        self.validate_original_sources()
        destination = recovery_directory(self.output)
        summary, complete = self.collection(
            "recover_continued_sft_dev_missing",
            Path(self.recovery_contract["missing_data"]),
            adapter,
            destination,
            repetitions=1,
            temperature=0,
            allowance=self.eval_allowance(),
            seed=43,
        )
        if not complete:
            raise BudgetStop("five missing Dev tasks did not complete within reallocated allowance")
        source = original_directory(self.output)
        combined, results = merge_results(
            rows(DEV),
            read(source / "summary.json"),
            rows(source / "trajectory_results.jsonl"),
            summary,
            rows(destination / "trajectory_results.jsonl"),
        )
        view = self.output / "eval_views/continued_sft_dev"
        view.mkdir(parents=True, exist_ok=True)
        combined.update(
            source_evaluation_directories=[str(source), str(destination)],
            source_summary_sha256={
                str(path): digest(path / "summary.json") for path in (source, destination)
            },
            recovery_contract_sha256=digest(contract_path(self.output)),
        )
        save(view / "summary.json", combined)
        (view / "trajectory_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in results)
        )
        self.persist(continued_sft_dev_merged_view=str(view), costs=self.costs())
        self.report()
        return combined

    def report(self):
        super().report()
        view = self.output / "eval_views/continued_sft_dev"
        if not (view / "summary.json").exists():
            return
        path = self.output / "summary/readout.json"
        result = read(path)
        result["evaluations"]["continued_sft_dev"] = metrics(view)
        result["evaluations"].pop(RECOVERY_NAME, None)
        result["continued_sft_dev_merged_view"] = str(view)
        result["recovery_raw_sources"] = [
            str(original_directory(self.output)),
            str(recovery_directory(self.output)),
        ]
        save(path, result)
        lines = [
            "# 有效信号与关键决策学习：V2实际读数",
            "",
            f"状态：`{self.state.get('status')}`；阶段：`{self.state.get('current_stage')}`。",
            "",
            "继续SFT Dev已用原95条与仅补采的5条构成固定100题；原始记录均保留。",
            "合并视图只用于统计，不产生额外计费；所有新Test与旧市场Final OOS均未访问。",
            "",
            "| 模型/区域 | 成功/计划 | 完成 | 指标/触发/执行 | 记录生成token |",
            "|---|---:|---:|---|---:|",
        ]
        for key, value in sorted(result["evaluations"].items()):
            lines.append(
                f"| {key} | {value['successes']}/{value['planned']} | {value['complete']} | "
                f"{value['metric_correct']}/{value['condition_correct']}/"
                f"{value['portfolio_correct']} | {value['tokens']} |"
            )
        cost = self.costs()
        lines += [
            "",
            f"全轮新增生成：{cost['total_new_generation_tokens']} token，"
            "含失败请求；合并视图未重复计费。",
            f"累计GPU工作进程阶段墙钟：{cost['gpu_stage_seconds']:.1f}秒（含加载）。",
            "",
            "[原95条](eval/continued_sft_dev/summary.json) · "
            "[补采5条](eval/continued_sft_dev_recovery20261105/summary.json) · "
            "[完整100题视图](eval_views/continued_sft_dev/summary.json)",
            "[恢复合同](recovery20261105/contract.json) · [实际状态](state.json)",
            "",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    contract = prepare(args.output)
    if args.prepare_only:
        print(
            json.dumps(
                dict(
                    contract=str(contract_path(args.output)),
                    missing_tasks=contract["missing_task_ids"],
                    allocation=contract["allocation"],
                ),
                ensure_ascii=False,
            )
        )
        return
    raise SystemExit(Controller(args.output).run())


if __name__ == "__main__":
    main()
