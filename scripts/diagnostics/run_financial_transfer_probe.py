#!/usr/bin/env python3
"""Bounded native-sampling diagnostic on the already-used report holdout."""

from __future__ import annotations

import argparse
import random
import signal
import subprocess
from collections import defaultdict
from pathlib import Path

from run_financial_evidence import ENVIRONMENT
from run_financial_learning import (
    GPU_CAP_SECONDS,
    INITIAL,
    TOTAL_CAP,
    BudgetStop,
    digest,
    metrics,
)
from run_financial_learning import Controller as LearningController
from run_financial_mc import generation_tokens
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, MODEL, ROOT, gpu, now, read, save

PARENT = ROOT / "runs/agentic-rl-learning-20261030-v2"
DEFAULT = ROOT / "runs/agentic-rl-transfer-probe-20261105-v1"
ARM_CAP = 2_200_000
PROBE_CAP = 2 * ARM_CAP
SEED = 145
MODELS = {
    "sft": INITIAL,
    "rloo_high": PARENT / "training/rloo_lr1e5/round_005/train/adapter",
}


def paired_report_difference(tasks, reference, candidate):
    expected = {(task["task_id"], replica) for task in tasks for replica in range(2)}
    task_map = {task["task_id"]: task for task in tasks}
    indexed = []
    for outcomes in (reference, candidate):
        current = {(row["task_id"], row["replicate_index"]): row for row in outcomes}
        if len(current) != len(outcomes) or set(current) != expected:
            raise ValueError("full unique two-replica task denominator required")
        if any(row.get("status") != "COMPLETE" for row in outcomes):
            raise ValueError("incomplete trajectories cannot support the comparison")
        indexed.append(current)
    grouped = defaultdict(list)
    for key in sorted(expected):
        report = task_map[key[0]]["source"]["report_id"]
        grouped[report].append(
            float(indexed[1][key]["success"]) - float(indexed[0][key]["success"])
        )
    means = [sum(values) / len(values) for values in grouped.values()]
    rng = random.Random(14503)
    draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(2000))
    return dict(
        paired_trajectories=len(expected),
        report_clusters=len(means),
        success_difference_high_rl_minus_sft=sum(means) / len(means),
        descriptive_report_bootstrap_95_interval=[draws[49], draws[1949]],
        report_differences={key: sum(value) / len(value) for key, value in grouped.items()},
        independently_held_out_confirmation=False,
        threshold_versions_and_replicas_clustered=True,
    )


def prepare(output):
    from src.agents.mandate_repair_train import adapter_digest

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = PARENT / "data/holdout.jsonl"
    tasks = rows(data)
    reports = {task["source"]["report_id"] for task in tasks}
    if len(tasks) != 32 or len(reports) != 16 or any(task["split"] != "train" for task in tasks):
        raise RuntimeError("fixed previously used 32-task/16-report holdout required")
    contract = dict(
        profile="financial_heldout_native_probability_probe_v1",
        data=str(data),
        data_sha256=digest(data),
        tasks=32,
        report_clusters=16,
        repetitions=2,
        seed=SEED,
        temperature=1,
        max_context_tokens=16384,
        max_response_tokens=512,
        environment="financial_evidence",
        decode_mode="native",
        maximum_calls=10,
        models={
            key: dict(path=str(path), adapter_sha256=adapter_digest(path))
            for key, path in MODELS.items()
        },
        parent_run=str(PARENT),
        parent_must_complete_before_gpu=True,
        generation_cap_per_model=ARM_CAP,
        diagnostic_generation_cap=PROBE_CAP,
        combined_generation_cap=TOTAL_CAP,
        combined_gpu_stage_seconds_cap=GPU_CAP_SECONDS,
        no_actor_training=True,
        reserved_test_read=False,
        historical_market_oos_read=False,
        already_used_development_holdout=True,
        initial_sft_report_exposure_not_excluded=True,
        protocol="docs/2026-11-05_HELDOUT_SAMPLING_PROBE.md",
    )
    path = output / "contract.json"
    if path.exists():
        previous = read(path)
        if previous["rules"] != contract:
            raise RuntimeError("frozen diagnostic contract changed")
    else:
        save(path, dict(frozen_at=now(), rules=contract))
    return contract


def combined_allowance(parent_tokens, own_tokens):
    if parent_tokens + PROBE_CAP > TOTAL_CAP:
        raise BudgetStop("remaining original44M allowance cannot reserve both diagnostic arms")
    if own_tokens > PROBE_CAP or parent_tokens + own_tokens > TOTAL_CAP:
        raise BudgetStop("combined generation allowance exceeded")
    return TOTAL_CAP - parent_tokens - own_tokens


class Controller(LearningController):
    def __init__(self, output, resume=False):
        self.contract = prepare(output)
        parent = read(PARENT / "state.json")
        if parent.get("status") != "COMPLETE" or parent.get("step3_status") != "COMPLETE":
            raise RuntimeError("V2 must complete before launching this diagnostic")
        self.parent_costs = read(PARENT / "step3/decision.json")["costs"]
        combined_allowance(self.parent_costs["total_new_generation_tokens"], 0)
        CompositionController.__init__(self, Path(output), resume)
        self.persist(
            experiment="heldout_native_sampling_probability_probe_v1",
            authorized_steps=[1],
            contract_sha256=digest(self.output / "contract.json"),
            frozen_parent_completion_sha256=digest(PARENT / "completion.json"),
            frozen_parent_costs=self.parent_costs,
            formal_valid_access=False,
            formal_final_oos_access=False,
            reserved_test_access=False,
        )
        source = self.output / "frozen_sources" / f"probe_controller_attempt{self.attempt}.py"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(Path(__file__).read_bytes())
        self.persist(controller_source=str(source), controller_source_sha256=digest(source))

    def costs(self):
        per_model = {key: generation_tokens(self.output / "eval" / key) for key in MODELS}
        own = sum(per_model.values())
        combined_allowance(self.parent_costs["total_new_generation_tokens"], own)
        if any(value > ARM_CAP for value in per_model.values()):
            raise BudgetStop("individual diagnostic arm allowance exceeded")
        seconds = sum(
            s.get("seconds", 0) for s in self.state.get("stages", []) if s.get("gpu_worker")
        )
        return dict(
            generation_tokens_by_model=per_model,
            diagnostic_generation_tokens=own,
            parent_generation_tokens=self.parent_costs["total_new_generation_tokens"],
            total_new_generation_tokens=self.parent_costs["total_new_generation_tokens"] + own,
            gpu_stage_seconds=self.parent_costs["gpu_stage_seconds"] + seconds,
            diagnostic_gpu_stage_seconds=seconds,
            raw_failures_included=True,
        )

    def stage(self, name, module, arguments, gpu_worker=True):
        if module != COLLECTOR:
            raise RuntimeError("this diagnostic only admits the sampling worker")
        return super().stage(name, module, arguments, gpu_worker)

    def report(self):
        evaluations = {}
        for key in MODELS:
            directory = self.output / "eval" / key / "sample"
            if (directory / "summary.json").exists():
                evaluations[key] = metrics(directory)
        result = dict(
            status=self.state.get("status"),
            stage=self.state.get("current_stage"),
            evaluations=evaluations,
            costs=self.costs(),
            comparison=self.state.get("comparison"),
            reserved_test_read=False,
        )
        save(self.output / "readout.json", result)
        lines = [
            "# 原留出题的当前策略采样频率诊断",
            "",
            f"状态：`{result['status']}`；阶段：`{result['stage']}`。",
            "",
            "固定旧报告留出16报告/32任务，每任务2条native轨迹；开发期诊断，非独立确认。",
            "",
            "| 模型 | 完整成功/计划 | 完成 | 指标正确 | 实际生成token |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, value in evaluations.items():
            lines.append(
                f"| {key} | {value['successes']}/{value['planned']} | "
                f"{value['complete']} | {value['metric_correct']} | {value['tokens']} |"
            )
        lines += [
            "",
            f"V2及本诊断合计新增生成：{result['costs']['total_new_generation_tokens']} token。",
            "结果只描述已使用留出题的采样频率；旧贪心22/32另列，不混分母。",
            "",
            "[冻结合同](contract.json) · [实际状态](state.json) · [完整读数](readout.json)",
            "",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines))

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.claim_gpu()
            for key, adapter in MODELS.items():
                destination = self.output / "eval" / key / "sample"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if self.fresh(destination):
                    self.stage(
                        f"sample_{key}",
                        COLLECTOR,
                        [
                            "--data",
                            self.contract["data"],
                            "--base-model",
                            MODEL,
                            "--adapter",
                            adapter,
                            "--output",
                            destination,
                            "--seed",
                            SEED,
                            "--temperature",
                            1,
                            "--repetitions",
                            2,
                            "--token-budget",
                            ARM_CAP - generation_tokens(self.output / "eval" / key),
                            "--engine-mode",
                            "eager",
                            *ENVIRONMENT,
                        ],
                    )
                summary = read(destination / "summary.json")
                if (
                    summary.get("complete_trajectories") != 64
                    or summary.get("status") != "COMPLETE"
                ):
                    raise BudgetStop(f"{key} fixed64trajectory denominator did not complete")
            comparison = paired_report_difference(
                rows(Path(self.contract["data"])),
                rows(self.output / "eval/sft/sample/trajectory_results.jsonl"),
                rows(self.output / "eval/rloo_high/sample/trajectory_results.jsonl"),
            )
            self.persist(
                status="COMPLETE",
                current_stage=None,
                comparison=comparison,
                finished_at=now(),
                costs=self.costs(),
            )
            save(
                self.output / "decision.json",
                dict(
                    status="COMPLETE",
                    comparison=comparison,
                    costs=self.costs(),
                    method_claim=False,
                    generalization_confirmation=False,
                ),
            )
        except BudgetStop as error:
            self.persist(status="RESOURCE_LIMIT", error=str(error), finished_at=now())
        except BaseException as error:
            code = 1
            self.persist(
                status="FAILED", error=f"{type(error).__name__}: {error}", finished_at=now()
            )
        finally:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=30)
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
            self.report()
            save(self.output / "completion.json", self.state)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output)
        print(args.output / "contract.json")
        return
    raise SystemExit(Controller(args.output, args.resume).run())


if __name__ == "__main__":
    main()
