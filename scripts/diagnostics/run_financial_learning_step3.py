#!/usr/bin/env python3
"""One prespecified conditional V2 diagnostic after the complete Stage2 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from run_financial_learning import (
    DEFAULT,
    GPU_CAP_SECONDS,
    GROUP_RESERVATION,
    INITIAL,
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
from run_mandate_repair import COLLECTOR, MODEL, ROOT, TRAINER, gpu, now, read, save

WRAPPER = "src.agents.financial_learning_collect"
SAMPLER = "src.agents.financial_mc_probe"
ARMS = ("rloo", "mc_target")
ARM_CAP = 4_000_000
EVAL_CAP = 5_000_000
MEAN_DELTA_EPSILON = 1e-12
PROTOCOL_VERSION = "financial_learning_v2_step3_gate_20261030_v1"


def protocol():
    """Rules are frozen before Stage3 outcomes; no requirement to win on Dev."""
    return dict(
        version=PROTOCOL_VERSION,
        branches=["A_FIXED_TARGET_MC", "C_UNRESOLVED_SIGNAL"],
        automatic_task_rewrite=False,
        gate=dict(
            independent_panel_mixed_business_task_required=True,
            genuine_calculation_request_required=True,
            nonzero_actor_advantage_required=True,
            same_round_changed_trainable_weights_required=True,
            same_round_actual_finite_positive_gradient_required=True,
            same_recorded_response_old_new_finite_required=True,
            minimum_absolute_mean_logprob_delta_exclusive=MEAN_DELTA_EPSILON,
            advantage_alignment_required=False,
            dev_or_holdout_improvement_required=False,
        ),
        learning_rate_selection="Stage2 fixed report-holdout metric; ties use2e-6",
        arms=list(ARMS),
        initial_adapter=str(INITIAL),
        optimizer_seed=43,
        parent_batch_size=32,
        task_groups_per_batch=8,
        parent_repetitions=4,
        optimizer_minibatch_size=8,
        maximum_batches_per_arm=4,
        maximum_generation_tokens_per_arm=ARM_CAP,
        parent_quota=8,
        auxiliary_repetitions=2,
        auxiliary_enters_actor=False,
        selection="pre-rollout parent quota; first public target report before calculation return",
        evaluation=dict(
            cap=EVAL_CAP,
            primary="full task success",
            mandatory_panel="all16fixedTrainpaneltasks x4native trajectories perarm",
            optional_report_holdout="all32fixedreport-holdouttasks x1greedy perarm",
            selection_before_stage3_sampling=True,
            mean_cost_source="Stage2 independent SFT native panel total / complete trajectories",
            forecast="ceil(1.15 * trajectory_count * measured_mean) + last_group_reservation",
            last_group_reservation=GROUP_RESERVATION,
            choose_holdout_if_full192_trajectory_forecast_fits=True,
            otherwise_full128_panel_only=True,
            dev_evaluation=False,
            reserved_test_evaluation=False,
        ),
        global_generation_cap=TOTAL_CAP,
        global_gpu_stage_seconds_cap=GPU_CAP_SECONDS,
        method_claim=False,
        independent_replication_claim=False,
    )


def freeze_protocol(root):
    path = Path(root) / "step3_protocol.json"
    current = protocol()
    if path.exists():
        previous = read(path)
        if previous.get("rules") != current:
            raise RuntimeError("Stage3 prespecified protocol changed")
    else:
        save(path, dict(frozen_at=now(), rules=current, outcome_used_for_rule_selection=False))
    return path


def stream_seed(arm, round_index, purpose):
    raw = f"{PROTOCOL_VERSION}/optimizer43/{arm}/round{round_index:03d}/{purpose}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") % (2**31 - 1)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def evidence_gate(root):
    """Use genuine training requests and matching update receipts, never held-out correctness."""
    root = Path(root)
    baseline_path = root / "baseline/sft_panel/trajectory_results.jsonl"
    mixed = defaultdict(list)
    if baseline_path.exists():
        for result in rows(baseline_path):
            value = result.get("financial_business_correct", result.get("business_correct"))
            if type(value) is bool:
                mixed[result["task_id"]].append(value)
    mixed_tasks = sorted(
        task for task, values in mixed.items() if len(values) == 4 and len(set(values)) > 1
    )
    supporting, rejected, summaries = [], [], []
    for path in sorted((root / "training").glob("rloo_*/round_*/train/summary.json")):
        summary = read(path)
        directory = path.parent
        updates_path = directory / "optimizer_updates.jsonl"
        updates = rows(updates_path) if updates_path.exists() else []
        real_updates = [
            r
            for r in updates
            if r.get("status") == "UPDATED"
            and finite(r.get("gradient_norm_before_clip"))
            and r["gradient_norm_before_clip"] > 0
        ]
        updated_trajectories = {
            key for item in real_updates for key in item.get("trajectories", [])
        }
        changed = summary.get("trainable_sha256_before") is not None and summary.get(
            "trainable_sha256_before"
        ) != summary.get("trainable_sha256_after")
        requests_path = directory.parent / "collect/requests.jsonl"
        requests = (
            {r["request_key"]: r for r in rows(requests_path)} if requests_path.exists() else {}
        )
        diagnostic_path = directory / "request_probability_deltas.jsonl"
        diagnostics = rows(diagnostic_path) if diagnostic_path.exists() else []
        summaries.append(
            dict(
                round=str(directory.parent),
                actual_updates=len(real_updates),
                changed_weights=changed,
                probability_diagnostics=len(diagnostics),
            )
        )
        for item in diagnostics:
            request = requests.get(item.get("request_key"), {})
            action = request.get("action") or {}
            genuine = (
                request.get("role", "parent") == "parent"
                and action.get("name") == "calculate"
                and request.get("trajectory_key") == item.get("trajectory_key")
                and item.get("trajectory_key") in updated_trajectories
            )
            usable = (
                genuine
                and changed
                and finite(item.get("actor_advantage"))
                and item["actor_advantage"] != 0
            )
            measured = all(
                finite(item.get(key))
                for key in (
                    "old_mean_response_logprob",
                    "new_mean_response_logprob",
                    "mean_response_logprob_delta",
                )
            )
            responsive = measured and abs(item["mean_response_logprob_delta"]) > MEAN_DELTA_EPSILON
            record = dict(
                round=str(directory.parent),
                request_key=item.get("request_key"),
                genuine_updated_calculation=genuine,
                changed_weights=changed,
                actor_advantage=item.get("actor_advantage"),
                mean_logprob_delta=item.get("mean_response_logprob_delta"),
                aligned=bool(
                    measured
                    and finite(item.get("actor_advantage"))
                    and item["actor_advantage"] * item["mean_response_logprob_delta"] > 0
                ),
            )
            if usable and responsive:
                supporting.append(record)
            elif item.get("phase") == "calculation":
                rejected.append(record)
    passed = bool(mixed_tasks and supporting)
    return dict(
        status="PASS" if passed else "UNRESOLVED_SIGNAL",
        branch="A_FIXED_TARGET_MC" if passed else "C_UNRESOLVED_SIGNAL",
        evaluated_at=now(),
        mixed_business_task_ids=mixed_tasks,
        supporting_calculation_requests=supporting,
        rejected_calculation_diagnostics=rejected,
        inspected_training_rounds=summaries,
        dev_or_holdout_score_used=False,
        advantage_alignment_required=False,
        finite_zero_probability_change_sufficient=False,
        conclusion="local update responsiveness only; not evidence of learning generalization",
    )


def evaluation_contract(root):
    baseline = read(Path(root) / "baseline/sft_panel/summary.json")
    complete = baseline.get("complete_trajectories", 0)
    if complete != 64:
        raise RuntimeError("full independent64trajectory baseline required for fixed eval forecast")
    mean = baseline["total_tokens"] / complete
    full = math.ceil(1.15 * 192 * mean) + GROUP_RESERVATION
    panel = math.ceil(1.15 * 128 * mean) + GROUP_RESERVATION
    if full <= EVAL_CAP:
        scope = "PANEL_K4_AND_REPORT_HOLDOUT"
    elif panel <= EVAL_CAP:
        scope = "PANEL_K4_ONLY"
    else:
        scope = "INSUFFICIENT_FIXED_ENDPOINT_ALLOWANCE"
    return dict(
        status="FROZEN",
        frozen_at=now(),
        scope=scope,
        measured_native_mean_tokens=mean,
        panel_plus_holdout_forecast_tokens=full,
        panel_only_forecast_tokens=panel,
        evaluation_cap=EVAL_CAP,
        primary_metric="full_task_success",
        panel_tasks=16,
        panel_repetitions=4,
        holdout_tasks=32,
        holdout_repetitions=1,
        dev_evaluated=False,
        reserved_test_evaluated=False,
        selected_before_any_stage3_model_generation=True,
        panel_is_training_positive_control=True,
        holdout_already_used_to_select_lr=True,
    )


def paired_panel_comparison(root, output):
    """Cluster both thresholds and all replicas; this is a selected Train panel."""
    tasks = {task["task_id"]: task for task in rows(Path(root) / "data/train_panel.jsonl")}
    paths = {
        arm: Path(output) / "eval" / f"{arm}_train_panel/trajectory_results.jsonl" for arm in ARMS
    }
    if not all(path.exists() for path in paths.values()):
        return None
    indexed = {
        arm: {(item["task_id"], item["replicate_index"]): item for item in rows(path)}
        for arm, path in paths.items()
    }
    expected = {(task, replica) for task in tasks for replica in range(4)}
    if any(set(values) != expected for values in indexed.values()):
        return dict(status="INCOMPLETE_FIXED_PANEL", planned_pairs=len(expected))
    clusters = defaultdict(list)
    for key in sorted(expected):
        task_id, _ = key
        source = tasks[task_id]["source"]
        report = source["report_id"]
        plain, candidate = indexed["rloo"][key], indexed["mc_target"][key]
        clusters[report].append(float(candidate["success"]) - float(plain["success"]))
    means = [sum(values) / len(values) for values in clusters.values()]
    rng = random.Random(4303)
    draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(2000))
    return dict(
        status="COMPLETE",
        paired_trajectories=len(expected),
        report_clusters=len(clusters),
        full_success_difference_mc_minus_rloo=sum(means) / len(means),
        descriptive_report_bootstrap_95_interval=[draws[49], draws[1949]],
        report_differences={key: sum(value) / len(value) for key, value in clusters.items()},
        all_thresholds_and_replicas_kept_in_cluster=True,
        training_positive_control_not_independent_confirmation=True,
    )


class Controller(LearningController):
    def __init__(self, root, resume=False):
        self.root = Path(root).resolve()
        super().__init__(self.root / "step3", resume)
        self.state.update(
            experiment="financial_learning_v2_conditional_step3",
            authorized_steps=[3],
            step2_status="COMPLETE_IN_PARENT_RUN",
            step3_status="RUNNING",
            current_step=3,
            generation_caps=dict(per_arm=ARM_CAP, evaluation=EVAL_CAP, global_total=TOTAL_CAP),
            source_root=str(self.root),
            protocol_sha256=digest(freeze_protocol(self.root)),
        )
        self.persist()

    def costs(self):
        parent_state = read(self.root / "state.json") if (self.root / "state.json").exists() else {}
        parent_seconds = sum(
            s.get("seconds", 0) for s in parent_state.get("stages", []) if s.get("gpu_worker")
        )
        own_seconds = sum(
            s.get("seconds", 0) for s in self.state.get("stages", []) if s.get("gpu_worker")
        )
        stage2_baseline = generation_tokens(self.root / "baseline")
        stage2_training = sum(
            generation_tokens(self.root / "training" / arm) for arm in ("rloo_lr2e6", "rloo_lr1e5")
        )
        stage2_eval = generation_tokens(self.root / "eval")
        per_arm = {arm: generation_tokens(self.output / "training" / arm) for arm in ARMS}
        evaluation = generation_tokens(self.output / "eval")
        total = stage2_baseline + stage2_training + stage2_eval + sum(per_arm.values()) + evaluation
        if (
            evaluation > EVAL_CAP
            or any(value > ARM_CAP for value in per_arm.values())
            or total > TOTAL_CAP
        ):
            raise RuntimeError("fixed Stage3 or global generation allowance exceeded")
        return dict(
            stage2_generation_tokens=stage2_baseline + stage2_training + stage2_eval,
            stage3_training_generation_tokens=per_arm,
            stage3_evaluation_generation_tokens=evaluation,
            total_new_generation_tokens=total,
            step3_gpu_stage_seconds=own_seconds,
            parent_gpu_stage_seconds=parent_seconds,
            gpu_stage_seconds=parent_seconds + own_seconds,
            includes_failed_requests=True,
            supervision_accounted_separately=True,
        )

    def source_receipt(self, module):
        super().source_receipt(module)
        if module in {WRAPPER, SAMPLER}:
            for dependency in (
                COLLECTOR,
                "src.agents.financial_mc_selection",
                "src.agents.financial_credit_probe",
            ):
                super().source_receipt(dependency)

    def root_update(self, **values):
        state = read(self.root / "state.json")
        state.update(**values, updated_at=now(), step3_state=str(self.path))
        save(self.root / "state.json", state)

    def report(self):
        costs = self.costs()
        evaluations = {}
        for path in sorted((self.output / "eval").glob("*/summary.json")):
            evaluations[path.parent.name] = metrics(path.parent)
        result = dict(
            status=self.state.get("status"),
            current_stage=self.state.get("current_stage"),
            arms=self.state.get("arms", {}),
            evaluations=evaluations,
            costs=costs,
            gate=self.state.get("gate"),
            decision=self.state.get("decision"),
            reserved_test_read=False,
        )
        (self.output / "summary").mkdir(exist_ok=True)
        save(self.output / "summary/readout.json", result)
        lines = [
            "# V2第三步：固定目标阶段MC条件性对照",
            "",
            f"状态：`{result['status']}`；阶段：`{result['current_stage']}`。",
            "",
            "本轮是Train正控上的有限学习诊断；不宣称独立方法复现或测试集泛化。",
            "| 模型/区域 | 成功/计划 | 指标正确 | 新生成token |",
            "|---|---:|---:|---:|",
        ]
        for key, value in evaluations.items():
            lines.append(
                f"| {key} | {value['successes']}/{value['planned']} | "
                f"{value['metric_correct']} | {value['tokens']} |"
            )
        lines += [
            "",
            "| 臂 | 完成批次 | 有效更新 | MC pair | 实际生成token |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, value in self.state.get("arms", {}).items():
            count = sum(r.get("completed_baselines", 0) for r in value.get("rounds", []))
            lines.append(
                f"| {arm} | {len(value.get('rounds', []))} | "
                f"{value.get('completed_optimizer_steps', 0)} | {count} | "
                f"{value.get('cumulative_tokens', 0)} |"
            )
        lines += [
            "",
            f"全轮新增生成：{costs['total_new_generation_tokens']} token；"
            f"累计GPU阶段{costs['gpu_stage_seconds']:.1f}秒（含加载）。",
            "普通RLOO和MC的辅助/父采样、有效更新不同，预算上限相同不等于等FLOPs。",
            "新Test与旧市场Final OOS均未访问。",
            "",
            "[状态](state.json) · [固定规则](../step3_protocol.json) · "
            "[完整读数](summary/readout.json)",
            "",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines))
        # Parent Stage2 has stopped before this method is used; preserve its original tables.
        root_report = self.root / "REPORT.md"
        previous = root_report.read_text() if root_report.exists() else ""
        start = "<!-- V2_STEP3_STATUS -->"
        stop = "<!-- /V2_STEP3_STATUS -->"
        if start in previous and stop in previous:
            previous = previous.split(stop, 1)[1].lstrip("\n")
        header = (
            f"{start}\n第三步状态：`{result['status']}`；"
            f"[第三步实际报告](step3/REPORT.md)。\n{stop}\n\n"
        )
        root_report.write_text(header + previous)
        root_summary = self.root / "summary/readout.json"
        if root_summary.exists():
            summary = read(root_summary)
            summary["step3"] = dict(
                status=result["status"],
                state=str(self.path),
                report=str(self.output / "REPORT.md"),
                decision=self.state.get("decision"),
                costs=costs,
            )
            save(root_summary, summary)
        parent = read(self.root / "state.json")
        if parent.get("status") == "RUNNING" and self.state.get("status") == "RUNNING":
            self.root_update(
                current_stage=f"STEP3_{self.state.get('current_stage')}",
                step3_status="RUNNING",
                costs=costs,
            )

    def ready(self):
        step2 = read(self.root / "step2_decision.json")
        if step2.get("status") != "COMPLETE":
            raise RuntimeError("Stage2 fixed endpoints must complete before Stage3")
        manifest = self.root / "data/manifest.json"
        if read(manifest).get("status") not in {"FROZEN", "COMPLETE"}:
            raise RuntimeError("frozen data required")
        for name, expected in read(manifest).get("output_sha256", {}).items():
            if digest(self.root / "data" / name) != expected:
                raise RuntimeError(f"frozen input drift:{name}")
        self.state["learning_rate"] = step2["selected_learning_rate"]
        if self.state["learning_rate"] not in {2e-6, 1e-5}:
            raise RuntimeError("unplanned Stage2-selected learning rate")
        source = self.output / "frozen_sources" / f"controller_attempt{self.attempt}.py"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(Path(__file__).read_bytes())
        self.persist(
            parent_step2_decision_sha256=digest(self.root / "step2_decision.json"),
            controller_sha256=digest(Path(__file__)),
            controller_source=str(source),
        )

    def fixed_batch_data(self, index):
        return self.root / "data" / f"batch_round{index:03d}.jsonl"

    def train_arm(self, arm):
        directory = self.output / "training" / arm
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "progress.json"
        progress = (
            read(path)
            if path.exists()
            else dict(
                status="RUNNING",
                rounds=[],
                final_adapter=str(INITIAL),
                completed_optimizer_steps=0,
                cumulative_tokens=0,
                learning_rate=self.state["learning_rate"],
                optimizer_seed=43,
            )
        )
        if progress["status"] in {"COMPLETE", "COMPLETE_LIMITED"}:
            self.state["arms"][arm] = progress
            return Path(progress["final_adapter"])
        actor = Path(progress["final_adapter"])
        parent_floor = self.state["training_contract"]["parent_batch_allowance_floor"]
        for index in range(len(progress["rounds"]), 4):
            rd = directory / f"round_{index:03d}"
            rd.mkdir(exist_ok=True)
            data = self.fixed_batch_data(index)
            if len(rows(data)) != 8:
                raise RuntimeError("Stage3 batch must retain all8taskgroups")
            remaining = ARM_CAP - generation_tokens(directory)
            if remaining < parent_floor:
                progress["stop_reason"] = "NO_FULL_FIXED_PARENT_BATCH_FORECAST_ALLOWANCE"
                break
            parent_seed = stream_seed(arm, index, "parent")
            quota_seed = stream_seed(arm, index, "quota")
            auxiliary_seed = stream_seed(arm, index, "auxiliary")
            reserve = (
                min(8 * 2 * 10 * 16384, max(0, remaining - parent_floor))
                if arm == "mc_target"
                else 0
            )
            collection = rd / "collect"
            if self.fresh(collection):
                if arm == "mc_target":
                    self.stage(
                        f"{arm}_round{index:03d}_collect",
                        WRAPPER,
                        [
                            "--data",
                            data,
                            "--base-model",
                            MODEL,
                            "--adapter",
                            actor,
                            "--output",
                            collection,
                            "--seed",
                            parent_seed,
                            "--token-budget",
                            remaining - reserve,
                            "--mc-selection-budget",
                            reserve,
                            "--mc-max-anchors",
                            8,
                            "--mc-selection-seed",
                            quota_seed,
                        ],
                    )
                else:
                    self.collection(
                        f"{arm}_round{index:03d}_collect",
                        data,
                        actor,
                        collection,
                        repetitions=4,
                        temperature=1,
                        allowance=remaining,
                        seed=parent_seed,
                    )
            sampled = read(collection / "summary.json")
            if sampled.get("complete_trajectories") != 32:
                progress["stop_reason"] = "PARTIAL_FIXED_PARENT_BATCH_NOT_TRAINED"
                progress["partial_collection"] = str(collection)
                break
            baseline = None
            measured = {}
            if arm == "mc_target":
                mc = rd / "mc"
                if self.fresh(mc):
                    self.stage(
                        f"{arm}_round{index:03d}_mc",
                        SAMPLER,
                        [
                            "--data",
                            data,
                            "--parents",
                            collection,
                            "--selections",
                            collection / "prefix_selections.jsonl",
                            "--base-model",
                            MODEL,
                            "--adapter",
                            actor,
                            "--output",
                            mc,
                            "--seed",
                            auxiliary_seed,
                            "--token-budget",
                            min(reserve, ARM_CAP - generation_tokens(directory)),
                            "--repetitions",
                            2,
                            "--max-anchors",
                            8,
                        ],
                    )
                measured = read(mc / "summary.json")
                baseline = mc / "state_value_baselines.jsonl"
                if index == 0:
                    self.same_parent_diagnostic(
                        collection, baseline, rd / "same_parent_diagnostic.json"
                    )
            trained_path = rd / "train"
            need_train, partial_resume = self.fresh_training(trained_path)
            if need_train:
                args = [
                    "rl",
                    "--data",
                    collection,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    trained_path,
                    "--seed",
                    43,
                    "--collection-seed",
                    parent_seed,
                    "--arm",
                    "mc_uniform" if arm == "mc_target" else "rloo",
                    "--learning-rate",
                    self.state["learning_rate"],
                    "--trajectory-minibatch-size",
                    8,
                    "--mixed-task-minibatches",
                ]
                if baseline:
                    args += ["--mc-baselines", baseline]
                if index in {0, 3}:
                    args += ["--diagnostic-request-limit", 16]
                if partial_resume:
                    args += ["--resume-state", partial_resume]
                elif progress["rounds"]:
                    args += [
                        "--resume-state",
                        Path(progress["rounds"][-1]["train"]) / "training_state.pt",
                    ]
                self.stage(f"{arm}_round{index:03d}_train", TRAINER, args)
            trained = read(trained_path / "summary.json")
            actor = trained_path / "adapter"
            progress["rounds"].append(
                dict(
                    round=index,
                    collect=str(collection),
                    train=str(trained_path),
                    mc=str(rd / "mc") if baseline else None,
                    parent_seed=parent_seed,
                    quota_seed=quota_seed,
                    auxiliary_seed=auxiliary_seed,
                    parent_tokens=sampled.get("total_tokens", 0),
                    auxiliary_tokens=measured.get("total_tokens", 0),
                    completed_baselines=measured.get("completed_baselines", 0),
                    parent_requests=sampled.get("complete_model_requests", 0),
                    optimizer_updates=trained.get("optimizer_updates", 0),
                    completed_optimizer_steps=trained.get("completed_optimizer_steps", 0),
                    actual_mc_replaced_requests=trained.get("mc_baseline_requests", 0),
                )
            )
            progress.update(
                final_adapter=str(actor),
                cumulative_tokens=generation_tokens(directory),
                completed_optimizer_steps=trained.get("completed_optimizer_steps", 0),
            )
            save(path, progress)
            self.state["arms"][arm] = progress
            self.persist(costs=self.costs())
            self.report()
        progress.update(
            status="COMPLETE" if len(progress["rounds"]) == 4 else "COMPLETE_LIMITED",
            final_adapter=str(actor),
            cumulative_tokens=generation_tokens(directory),
            finished_at=now(),
            unused_tokens=ARM_CAP - generation_tokens(directory),
        )
        save(path, progress)
        self.state["arms"][arm] = progress
        self.persist()
        return actor

    def same_parent_diagnostic(self, collection, baseline, path):
        from src.agents.mandate_repair_train import build_batch

        plain, plain_summary = build_batch(collection, arm="rloo")
        mc, mc_summary = build_batch(collection, arm="mc_uniform", mc_baselines=baseline)
        left = {r["request_key"]: r for group in plain.values() for r in group["requests"]}
        right = {r["request_key"]: r for group in mc.values() for r in group["requests"]}
        differences = []
        for key, a in left.items():
            b = right[key]
            if b.get("mc_baseline_used"):
                differences.append(
                    dict(
                        request_key=key,
                        plain_raw=a["raw_combined_advantage"],
                        mc_raw=b["raw_combined_advantage"],
                        plain_actor=a["advantage"],
                        mc_actor=b["advantage"],
                        action=(a.get("action") or {}).get("name"),
                    )
                )
        save(
            path,
            dict(
                same_parent_diagnostic=True,
                actual_equal_budget_learning_comparison=False,
                plain_summary=plain_summary,
                mc_summary=mc_summary,
                matched=differences,
                changed_raw=sum(item["plain_raw"] != item["mc_raw"] for item in differences),
            ),
        )

    def evaluate_endpoint(self, arm, actor):
        scope = self.state["evaluation_contract"]["scope"]
        for split, repetitions, temperature in (("train_panel", 4, 1), ("holdout", 1, 0)):
            if split == "holdout" and scope != "PANEL_K4_AND_REPORT_HOLDOUT":
                continue
            destination = self.output / "eval" / f"{arm}_{split}"
            remaining = EVAL_CAP - generation_tokens(self.output / "eval")
            _, complete = self.collection(
                f"eval_{arm}_{split}",
                self.root / "data" / f"{split}.jsonl",
                actor,
                destination,
                repetitions=repetitions,
                temperature=temperature,
                allowance=remaining,
                seed=143,
            )
            self.report()
            if not complete:
                raise BudgetStop("Stage3 fixed endpoint denominator incomplete")

    def finish(self, reason):
        comparison = paired_panel_comparison(self.root, self.output)
        if comparison is not None:
            save(self.output / "paired_panel_comparison.json", comparison)
        decision = dict(
            status="COMPLETE",
            decision=reason,
            finished_at=now(),
            gate=self.state.get("gate"),
            evaluation_contract=self.state.get("evaluation_contract"),
            costs=self.costs(),
            credit_method_claim_supported=False,
            fresh_test_opened=False,
            independent_method_confirmation=False,
            paired_panel_comparison=comparison,
        )
        save(self.output / "decision.json", decision)
        self.persist(
            status="COMPLETE", step3_status="COMPLETE", current_stage=None, decision=decision
        )
        self.root_update(
            status="COMPLETE",
            step3_status="COMPLETE",
            current_step=3,
            current_stage=None,
            finished_at=now(),
            step3_decision=str(self.output / "decision.json"),
            costs=self.costs(),
        )
        review = [
            "# V2三步结果与长期目标回顾",
            "",
            f"第三步出口：`{reason}`。",
            "",
            "## 实际固定终点",
            "",
            "| 阶段/模型/区域 | 完整成功/计划 | 指标正确 |",
            "|---|---:|---:|",
        ]
        for phase, directory in (("第二步", self.root), ("第三步", self.output)):
            for path in sorted((directory / "eval").glob("*/summary.json")):
                if "midpoint" in path.parent.name or "_recovery" in path.parent.name:
                    continue
                merged_view = directory / "eval_views" / path.parent.name
                value = metrics(
                    merged_view if (merged_view / "summary.json").exists() else path.parent
                )
                review.append(
                    f"| {phase}/{path.parent.name} | {value['successes']}/{value['planned']} | "
                    f"{value['metric_correct']} |"
                )
        supporting_count = len(
            (self.state.get("gate") or {}).get("supporting_calculation_requests", [])
        )
        review += [
            "",
            f"第三步判据支持的计算请求：{supporting_count}。",
            f"本轮总新增生成：{self.costs()['total_new_generation_tokens']} token；"
            "含全部失败请求。",
            "",
            "- RQ1：检查真实计算请求的优势、实际更新和同响应概率变化；这些诊断不等于任务成功收益。",
            "- RQ2：仅在A分支比较固定目标阶段MC与普通RLOO；C分支未启动MC，不作方法否定。",
            "- RQ3：固定Train面板是训练正控，报告留出曾用于选学习率；"
            "本轮不能声称独立泛化或方法成立。",
            "",
            "下一轮应依据完整学习曲线与失败行为决定扩大、改估计器或改任务，不自动把有限负结果写成论文。",
            "[第二步结果](REPORT.md) · [第三步报告](step3/REPORT.md) · "
            "[第三步决定](step3/decision.json)",
            "",
        ]
        (self.root / "RESEARCH_REVIEW.md").write_text("\n".join(review))
        (self.root / "CONCLUSION.md").write_text("\n".join(review))
        (self.output / "CONCLUSION.md").write_text(
            f"# 第三步结论\n\n出口：`{reason}`。\n\n"
            "完整固定终点、成本和证据边界见[本轮回顾](../RESEARCH_REVIEW.md)。\n"
            "本轮未使用新Test，也不构成独立方法确认。\n"
        )
        self.report()

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal{signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.ready()
            self.root_update(
                status="RUNNING",
                step3_status="RUNNING",
                current_step=3,
                current_stage="STEP3_EVIDENCE_GATE",
            )
            gate = evidence_gate(self.root)
            save(self.output / "gate.json", gate)
            self.persist(gate=gate)
            if gate["status"] != "PASS":
                self.finish("UNRESOLVED_KEY_DECISION_SIGNAL")
                return 0
            contract_path = self.output / "evaluation_contract.json"
            contract = (
                read(contract_path) if contract_path.exists() else evaluation_contract(self.root)
            )
            if not contract_path.exists():
                save(contract_path, contract)
            self.persist(evaluation_contract=contract)
            if contract["scope"] == "INSUFFICIENT_FIXED_ENDPOINT_ALLOWANCE":
                raise BudgetStop("fixed Stage3 evaluation does not fit before training")
            previous = [
                read(p).get("total_tokens", 0)
                for p in (self.root / "training").glob("rloo_*/round_*/collect/summary.json")
                if read(p).get("complete_trajectories") == 32
            ]
            training_contract_path = self.output / "training_contract.json"
            training_contract = (
                read(training_contract_path)
                if training_contract_path.exists()
                else dict(
                    frozen_at=now(),
                    parent_batch_allowance_floor=max(previous, default=1_000_000)
                    + GROUP_RESERVATION,
                    formula="max Stage2 full32parent batch cost plus one K4worstgroup reserve",
                    arm_cap=ARM_CAP,
                    maximum_batches=4,
                    maximum_parent_quota=8,
                )
            )
            if not training_contract_path.exists():
                save(training_contract_path, training_contract)
            self.persist(training_contract=training_contract)
            self.claim_gpu()
            endpoints = {}
            for arm in ARMS:
                endpoints[arm] = self.train_arm(arm)
                self.evaluate_endpoint(arm, endpoints[arm])
            support = sum(
                r.get("completed_baselines", 0) for r in self.state["arms"]["mc_target"]["rounds"]
            )
            self.finish("LOCAL_MC_COMPARISON_RECORDED" if support else "NO_COMPLETE_MC_SUPPORT")
        except BudgetStop as error:
            self.persist(
                status="RESOURCE_LIMIT",
                step3_status="INCOMPLETE",
                error=str(error),
                finished_at=now(),
            )
            self.root_update(
                status="COMPLETE_LIMITED",
                step3_status="INCOMPLETE",
                current_stage=None,
                step3_error=str(error),
                costs=self.costs(),
            )
        except BaseException as error:
            code = 1
            self.persist(
                status="FAILED", step3_status="FAILED", error=f"{type(error).__name__}:{error}"
            )
            self.root_update(
                status="FAILED", step3_status="FAILED", step3_error=str(error), costs=self.costs()
            )
        finally:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.state["holder_restore"] = "REFILLING_76_GIB"
                except Exception as error:
                    self.state["holder_restore_error"] = str(error)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            self.persist()
            save(self.output / "completion.json", self.state)
            parent = read(self.root / "state.json")
            parent.update(
                step3_state=str(self.path),
                holder_restore=self.state.get("holder_restore", parent.get("holder_restore")),
                costs=self.costs(),
                updated_at=now(),
            )
            save(self.root / "state.json", parent)
            save(self.root / "completion.json", parent)
            self.report()
        return code


def wait_stage2(root):
    root = Path(root)
    while True:
        decision = root / "step2_decision.json"
        state = read(root / "state.json") if (root / "state.json").exists() else {}
        completion = root / "completion.json"
        released = read(completion) if completion.exists() else {}
        if (
            decision.exists()
            and read(decision).get("status") == "COMPLETE"
            and state.get("status") == "STEP2_COMPLETE"
            and released.get("status") == "STEP2_COMPLETE"
            and released.get("attempt") == state.get("attempt")
        ):
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
            # A completed JSON receipt can precede the old controller's cgroup exit.
            # The holder clears its claim only after that cgroup is actually empty.
            try:
                holder_state = gpu.request(str(holder), {"op": "status"}, timeout=15)
                if "claim" in holder_state and holder_state["claim"] is None:
                    return
            except (OSError, RuntimeError):
                pass
        if state.get("status") in {"FAILED", "RESOURCE_LIMIT", "COMPLETE_LIMITED"}:
            raise RuntimeError(f"Stage2 did not complete:{state.get('status')}")
        if state.get("status") == "COMPLETE" and state.get("step3_status") == "COMPLETE":
            return
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-stage2", action="store_true")
    parser.add_argument("--prepare-protocol", action="store_true")
    args = parser.parse_args()
    path = freeze_protocol(args.output)
    if args.prepare_protocol:
        print(path)
        return
    if args.wait_stage2:
        wait_stage2(args.output)
    if (args.output / "step3/decision.json").exists():
        print("Stage3 decision already complete; no new work")
        return
    raise SystemExit(Controller(args.output, args.resume).run())


if __name__ == "__main__":
    main()
