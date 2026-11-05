#!/usr/bin/env python3
"""One bounded coverage screen: six RLOO batches and one reused holdout panel."""

from __future__ import annotations

import argparse
import hashlib
import signal
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

from run_financial_learning import INITIAL, BudgetStop, digest, metrics
from run_financial_learning import Controller as LearningController
from run_financial_mc import generation_tokens
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, MODEL, ROOT, TRAINER, gpu, now, read, save

DEFAULT = ROOT / "runs/agentic-rl-coverage-20261105-v3"
REFERENCE = ROOT / "runs/agentic-rl-transfer-probe-20261105-v1"
PLAN = ROOT / "docs/2026-11-05_AGENTIC_RL_THREE_STEP_PLAN_V3.md"
TRAIN_CAP, EVAL_CAP, RECOVERY_CAP = 6_000_000, 2_200_000, 800_000
TOTAL_CAP, GPU_SECONDS = 9_000_000, 7200


def sampling_seed(index):
    raw = f"financial-coverage-v3/native-parent/optimizer42/round{index:03d}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") % (2**31 - 1)


def indexed_outcomes(tasks, outcomes):
    expected = {(t["task_id"], i) for t in tasks for i in range(2)}
    indexed = {(r["task_id"], r["replicate_index"]): r for r in outcomes}
    if len(outcomes) != len(indexed) or set(indexed) != expected:
        raise ValueError("full unique fixed64trajectory denominator required")
    if any(r.get("status") != "COMPLETE" for r in outcomes):
        raise ValueError("all fixed evaluation trajectories must be complete")
    return indexed


def compare(tasks, candidate, references):
    candidate = indexed_outcomes(tasks, candidate)
    task_map = {t["task_id"]: t for t in tasks}
    comparisons = {}
    for name, outcomes in references.items():
        reference = indexed_outcomes(tasks, outcomes)
        reports = defaultdict(lambda: dict(wide_successes=0, reference_successes=0, samples=0))
        changed = []
        for key in sorted(candidate):
            new, old = candidate[key], reference[key]
            report = task_map[key[0]]["source"]["report_id"]
            report_row = reports[report]
            report_row["samples"] += 1
            report_row["wide_successes"] += int(new["success"])
            report_row["reference_successes"] += int(old["success"])
            if new["success"] != old["success"]:
                changed.append(
                    dict(
                        task_id=key[0],
                        replicate_index=key[1],
                        wide_success=new["success"],
                        reference_success=old["success"],
                    )
                )
        difference = sum(r["wide_successes"] - r["reference_successes"] for r in reports.values())
        comparisons[name] = dict(
            wide_minus_reference_successes=difference,
            wide_minus_reference_success_rate=difference / len(candidate),
            reports=dict(reports),
            matched_sample_label_changes=changed,
            sample_label_changes_are_not_causal_fixes=True,
        )
    return dict(
        planned_trajectories=len(candidate),
        report_clusters=len(reports),
        comparisons=comparisons,
        direction_screen_only=True,
        coverage_is_not_the_only_difference=True,
    )


def prepare(output):
    from src.agents.financial_coverage_data import prepare as prepare_data
    from src.agents.mandate_repair_train import adapter_digest

    output = Path(output).resolve()
    prepare_data(output)
    manifest = read(output / "data/manifest.json")
    old = read(REFERENCE / "contract.json")["rules"]
    if digest(Path(manifest["holdout_path"])) != old["data_sha256"]:
        raise RuntimeError("reused holdout identity changed")
    if adapter_digest(INITIAL) != old["models"]["sft"]["adapter_sha256"]:
        raise RuntimeError("common SFT identity changed")
    references = {}
    identity_fields = (
        "data_sha256",
        "seed",
        "temperature",
        "max_context_tokens",
        "max_response_tokens",
        "environment",
        "decode_mode",
    )
    for name in ("sft", "rloo_high"):
        directory = REFERENCE / "eval" / name / "sample"
        summary = read(directory / "summary.json")
        if summary.get("status") != "COMPLETE" or summary["complete_trajectories"] != 64:
            raise RuntimeError("reused reference must contain the full64trajectory panel")
        if any(summary.get(k) != old[k] for k in identity_fields):
            raise RuntimeError(f"reused {name} evaluation configuration changed")
        if summary["adapter_sha256"] != old["models"][name]["adapter_sha256"]:
            raise RuntimeError(f"reused {name} adapter identity changed")
        indexed_outcomes(
            rows(manifest["holdout_path"]), rows(directory / "trajectory_results.jsonl")
        )
        references[name] = dict(
            path=str(directory),
            successes=summary["successes"],
            metric_correct=summary["outcome_diagnostics"]["metric_correct"],
            outcomes_sha256=digest(directory / "trajectory_results.jsonl"),
        )
    rules = dict(
        profile="financial_coverage_lean_v3",
        data_manifest_sha256=digest(output / "data/manifest.json"),
        initial_adapter=str(INITIAL),
        initial_adapter_sha256=adapter_digest(INITIAL),
        references=references,
        evaluation={k: old[k] for k in identity_fields},
        parent_repetitions=4,
        evaluation_repetitions=2,
        batches=6,
        parent_trajectories=192,
        learning_rate=1e-5,
        optimizer_seed=42,
        minibatch_size=8,
        generation_caps=dict(
            training=TRAIN_CAP, evaluation=EVAL_CAP, recovery=RECOVERY_CAP, total=TOTAL_CAP
        ),
        gpu_stage_seconds_cap=GPU_SECONDS,
        reserved_test_read=False,
        historical_market_oos_read=False,
        no_intermediate_evaluation=True,
    )
    contract = output / "contract.json"
    if contract.exists() and read(contract)["rules"] != rules:
        raise RuntimeError("frozen coverage contract changed")
    if not contract.exists():
        save(contract, dict(frozen_at=now(), rules=rules))
    frozen = output / "frozen_sources"
    frozen.mkdir(exist_ok=True)
    if not (frozen / "plan.md").exists():
        (frozen / "plan.md").write_bytes(PLAN.read_bytes())
    return rules


class Controller(LearningController):
    def __init__(self, output, resume=False):
        self.contract = prepare(output)
        CompositionController.__init__(self, Path(output), resume)
        self.persist(
            experiment="financial_coverage_lean_v3",
            reserved_test_access=False,
            contract_sha256=digest(self.output / "contract.json"),
            recovery_allocation=self.state.get("recovery_allocation", {}),
        )
        source = self.output / "frozen_sources" / f"coverage_controller_attempt{self.attempt}.py"
        source.write_bytes(Path(__file__).read_bytes())
        self.persist(controller_source=str(source), controller_source_sha256=digest(source))

    def allowance(self, bucket):
        allocation = self.state.get("recovery_allocation", {})
        if any(v < 0 for v in allocation.values()) or sum(allocation.values()) > RECOVERY_CAP:
            raise BudgetStop("recovery allowance exceeded")
        return {"training": TRAIN_CAP, "evaluation": EVAL_CAP}[bucket] + allocation.get(bucket, 0)

    def costs(self):
        training = generation_tokens(self.output / "training")
        evaluation = generation_tokens(self.output / "eval")
        seconds = sum(
            s.get("seconds", 0) for s in self.state.get("stages", []) if s.get("gpu_worker")
        )
        if training > self.allowance("training") or evaluation > self.allowance("evaluation"):
            raise BudgetStop("generation bucket allowance exceeded, including failed requests")
        if training + evaluation > TOTAL_CAP:
            raise BudgetStop("9M generation allowance exceeded")
        return dict(
            training_generation_tokens=training,
            evaluation_generation_tokens=evaluation,
            total_new_generation_tokens=training + evaluation,
            gpu_stage_seconds=seconds,
            reused_reference_new_tokens=0,
            failed_requests_included=True,
        )

    def stage(self, name, module, arguments, gpu_worker=True):
        if module not in {COLLECTOR, TRAINER}:
            raise RuntimeError("only the existing collector and trainer are admitted")
        remaining = GPU_SECONDS - self.costs()["gpu_stage_seconds"]
        if remaining <= 0:
            raise BudgetStop("2-hour cumulative GPU-stage allowance exhausted")
        self.source_receipt(module)
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{name}_attempt{self.attempt}.log"
        entry = dict(name=name, status="RUNNING", log=str(log), started_at=now(), gpu_worker=True)
        self.state["stages"].append(entry)
        self.persist(current_stage=name)
        started = time.monotonic()
        try:
            with log.open("x") as stream:
                self.child = subprocess.Popen(
                    [str(ROOT / ".venv/bin/python"), "-u", "-m", module, *map(str, arguments)],
                    cwd=ROOT,
                    env=self.environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                gpu.request(
                    self.config["directory"],
                    dict(
                        op="attach",
                        lease_token=self.config["lease_token"],
                        workers=[gpu.process_identity(self.child.pid)],
                        request_id=uuid.uuid4().hex,
                    ),
                    timeout=60,
                )
                try:
                    code = self.child.wait(
                        timeout=max(0.1, remaining - (time.monotonic() - started))
                    )
                except subprocess.TimeoutExpired as error:
                    raise BudgetStop(
                        "2-hour cumulative GPU-stage wall-clock limit reached"
                    ) from error
                if code:
                    raise RuntimeError(f"{name} exited {code}; see {log}")
            entry["status"] = "COMPLETE"
        except BaseException as error:
            entry.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            self.child = None
            entry.update(seconds=time.monotonic() - started, finished_at=now())
            self.persist()
            self.report()

    def train_wide(self):
        directory = self.output / "training/wide"
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = (
            read(progress_path)
            if progress_path.exists()
            else dict(rounds=[], final_adapter=str(INITIAL), completed_optimizer_steps=0)
        )
        actor = Path(progress["final_adapter"])
        for index in range(len(progress["rounds"]), 6):
            rd = directory / f"round_{index:03d}"
            rd.mkdir(exist_ok=True)
            data = self.output / "data" / f"batch_round{index:03d}.jsonl"
            if len(rows(data)) != 8:
                raise RuntimeError("fixed eight-task parent batch required")
            seed = sampling_seed(index)
            collected, complete = self.collection(
                f"wide_round{index:03d}_collect",
                data,
                actor,
                rd / "collect",
                repetitions=4,
                temperature=1,
                allowance=self.allowance("training") - generation_tokens(directory),
                seed=seed,
            )
            if not complete or collected.get("status") != "COMPLETE":
                raise BudgetStop("incomplete fixed parent batch retained, not trained")
            destination = rd / "train"
            needed, partial = self.fresh_training(destination)
            if needed:
                args = [
                    "rl",
                    "--data",
                    rd / "collect",
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    destination,
                    "--seed",
                    42,
                    "--collection-seed",
                    seed,
                    "--arm",
                    "rloo",
                    "--learning-rate",
                    1e-5,
                    "--trajectory-minibatch-size",
                    8,
                    "--mixed-task-minibatches",
                ]
                previous = partial or (
                    Path(progress["rounds"][-1]["train"]) / "training_state.pt"
                    if progress["rounds"]
                    else None
                )
                if previous:
                    args += ["--resume-state", previous]
                self.stage(f"wide_round{index:03d}_train", TRAINER, args)
            trained = read(destination / "summary.json")
            actor = destination / "adapter"
            progress["rounds"].append(
                dict(
                    round=index,
                    collect=str(rd / "collect"),
                    train=str(destination),
                    sampling_seed=seed,
                    parent_trajectories=collected["complete_trajectories"],
                    successes=collected["successes"],
                    tokens=collected["total_tokens"],
                    optimizer_updates=trained["optimizer_updates"],
                    effective_advantage_requests=trained.get("effective_advantage_requests"),
                    completed_optimizer_steps=trained["completed_optimizer_steps"],
                )
            )
            progress.update(
                final_adapter=str(actor),
                completed_optimizer_steps=trained["completed_optimizer_steps"],
            )
            save(progress_path, progress)
            self.state["arms"]["wide"] = progress
            self.persist(costs=self.costs())
            self.report()
        return actor

    def report(self):
        result = dict(
            status=self.state.get("status"),
            current_stage=self.state.get("current_stage"),
            training=self.state.get("arms", {}).get("wide", {}),
            costs=self.costs(),
            comparison=self.state.get("comparison"),
            references=self.contract["references"],
        )
        evaluation = self.output / "eval/wide/sample"
        if (evaluation / "summary.json").exists():
            result["wide_evaluation"] = metrics(evaluation)
        save(self.output / "readout.json", result)
        lines = [
            "# 扩覆盖快速试验：实际进度与结果",
            "",
            f"状态：**{result['status']}**；当前阶段：{result['current_stage']}。",
            "",
            f"训练已完成{len(result['training'].get('rounds', []))}/6批；"
            "计划192父轨迹，只训练宽覆盖一臂。",
            "",
            "| 模型 | 原留出native完整成功 /64 | 指标正确 |",
            "|---|---:|---:|",
        ]
        for name, ref in self.contract["references"].items():
            lines.append(
                f"| {name}（复用，新增成本0） | {ref['successes']} | {ref['metric_correct']} |"
            )
        if "wide_evaluation" in result:
            value = result["wide_evaluation"]
            lines.append(
                f"| 宽覆盖 | {value['successes']}（完成{value['complete']}/64） | "
                f"{value['metric_correct']} |"
            )
        lines += [
            "",
            f"实际新增输入＋输出：{result['costs']['total_new_generation_tokens']:,} token；"
            f"GPU阶段{result['costs']['gpu_stage_seconds']:.1f}秒。",
            "上限900万token/2小时；旧对照不重采，未访问新Test或旧市场OOS。",
            "",
            "本轮仅作方向筛查；历史窄臂题目选择与新宽臂不同，不单独归因于覆盖。",
            "",
            "[完整读数](readout.json) · [实际状态](state.json) · "
            "[数据清单](data/manifest.json) · [冻结合同](contract.json)",
            "",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines))

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.claim_gpu()
            actor = self.train_wide()
            evaluation = self.output / "eval/wide/sample"
            evaluation.parent.mkdir(parents=True, exist_ok=True)
            manifest = read(self.output / "data/manifest.json")
            _, complete = self.collection(
                "eval_wide_holdout",
                Path(manifest["holdout_path"]),
                actor,
                evaluation,
                repetitions=2,
                temperature=1,
                allowance=self.allowance("evaluation") - generation_tokens(self.output / "eval"),
                seed=145,
            )
            if not complete:
                raise BudgetStop("fixed64trajectory endpoint evaluation incomplete")
            references = {
                k: rows(Path(v["path"]) / "trajectory_results.jsonl")
                for k, v in self.contract["references"].items()
            }
            comparison = compare(
                rows(manifest["holdout_path"]),
                rows(evaluation / "trajectory_results.jsonl"),
                references,
            )
            self.persist(
                status="COMPLETE",
                current_stage=None,
                finished_at=now(),
                comparison=comparison,
                costs=self.costs(),
            )
            save(
                self.output / "decision.json",
                dict(
                    status="COMPLETE",
                    comparison=comparison,
                    costs=self.costs(),
                    next_action="REVIEW_DIRECTION_FROM_FIXED_ENDPOINT",
                    additional_sampling_automatic=False,
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
