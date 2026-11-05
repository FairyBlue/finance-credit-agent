#!/usr/bin/env python3
"""Run the approved V2 finite learning diagnostic without opening reserved Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from run_financial_evidence import ENVIRONMENT
from run_financial_mc import generation_tokens
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, MODEL, ROOT, TRAINER, gpu, now, read, save

DEFAULT = ROOT / "runs/agentic-rl-learning-20261030-v2"
INITIAL = ROOT / "runs/m3-financial-evidence-20261030-v2/teacher_sft/adapter"
DEV = ROOT / "runs/m3-financial-credit-20261030-v1/data/development.jsonl"
PLAN = ROOT / "docs/2026-10-30_AGENTIC_RL_THREE_STEP_PLAN_V2.md"
ARMS = {"rloo_lr2e6": 2e-6, "rloo_lr1e5": 1e-5}
BASELINE_CAP = 3_000_000
ARM_CAP = 6_000_000
EVAL_CAP = 16_000_000
TOTAL_CAP = 44_000_000
GPU_CAP_SECONDS = 12 * 3600
GROUP_RESERVATION = 4 * 10 * 16384


class BudgetStop(RuntimeError):
    """A fixed resource allowance cannot admit a requested complete stage."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_collection_seed(arm, round_index):
    """Independent, reproducible RNG namespaces without Python's randomized hash()."""
    namespace = f"agentic-learning-v2/native-parent/base42/{arm}/round{round_index:03d}"
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big") % (2**31 - 1)


def metrics(directory):
    summary = read(Path(directory) / "summary.json")
    diagnostics = summary.get("outcome_diagnostics", {})
    return dict(
        successes=summary.get("successes"),
        complete=summary.get("complete_trajectories"),
        planned=summary.get("planned_trajectories"),
        metric_correct=diagnostics.get("metric_correct"),
        portfolio_correct=diagnostics.get("portfolio_correct"),
        condition_correct=diagnostics.get("condition_correct"),
        tokens=summary.get("total_tokens", 0),
        evaluation_reused=summary.get("evaluation_reused", False),
    )


class Controller(CompositionController):
    def __init__(self, output, resume=False, *, baseline_only=False):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "state.json"
        self.child, self.config, self.claimed = None, None, False
        self.attempt = 1
        if self.path.exists():
            previous = read(self.path)
            resumable = {
                "FAILED",
                "BASELINE_COMPLETE",
                "INTERRUPTED",
                "RESOURCE_LIMIT",
                "STEP2_COMPLETE",
            }
            if not resume or previous.get("status") not in resumable:
                raise RuntimeError("explicit resume requires a stopped/failed controller")
            self.attempt = previous.get("attempt", 0) + 1
            archive = self.output / "controller_attempts"
            archive.mkdir(exist_ok=True)
            save(archive / f"attempt{self.attempt - 1}.json", previous)
            self.state = previous
            self.state.pop("error", None)
        else:
            self.state = dict(
                stages=[],
                arms={},
                evaluations={},
                started_at=now(),
                experiment="agentic_rl_effective_learning_diagnostic_v2",
                formal_valid_access=False,
                formal_final_oos_access=False,
                reserved_test_access=False,
                initial_adapter=str(INITIAL),
                plan=str(PLAN),
                plan_sha256=digest(PLAN),
                generation_caps=dict(
                    baseline=BASELINE_CAP,
                    training_per_rl_arm=ARM_CAP,
                    step2_evaluation=EVAL_CAP,
                    total=TOTAL_CAP,
                ),
                gpu_stage_seconds_cap=GPU_CAP_SECONDS,
                step1_status="SEE_SIGNAL_REPORT",
                step2_status="RUNNING",
                step3_status="PENDING",
            )
        self.baseline_only = baseline_only
        self.state.update(status="RUNNING", attempt=self.attempt, baseline_only=baseline_only)
        self.environment = dict(os.environ)
        self.environment.update(
            PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "third_party/agent-r1")]),
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
        )
        self.persist()

    def costs(self):
        baseline = generation_tokens(self.output / "baseline")
        evaluation = generation_tokens(self.output / "eval")
        training = {key: generation_tokens(self.output / "training" / key) for key in ARMS}
        stage3 = generation_tokens(self.output / "step3")
        total = baseline + evaluation + sum(training.values()) + stage3
        seconds = sum(
            s.get("seconds", 0) for s in self.state.get("stages", []) if s.get("gpu_worker")
        )
        for value, limit, label in (
            (baseline, BASELINE_CAP, "baseline"),
            (evaluation, EVAL_CAP, "step2 evaluation"),
            (total, TOTAL_CAP, "total"),
            *((v, ARM_CAP, k) for k, v in training.items()),
        ):
            if value > limit:
                raise RuntimeError(f"actual {label} generation {value} exceeded {limit}")
        return dict(
            baseline_generation_tokens=baseline,
            rl_training_generation_tokens=training,
            evaluation_generation_tokens=evaluation,
            stage3_generation_tokens=stage3,
            total_new_generation_tokens=total,
            gpu_stage_seconds=seconds,
            includes_failed_requests=True,
            supervised_tokens_accounted_separately=True,
        )

    def persist(self, **values):
        self.state.update(**values, updated_at=now())
        save(self.path, self.state)

    def source_receipt(self, module):
        relatives = [module.replace(".", "/") + ".py"]
        if module == COLLECTOR:
            relatives += [
                "src/agents/financial_evidence_env.py",
                "src/agents/mandate_repair_env.py",
                "src/agents/mandate_repair_metrics.py",
                "src/agents/mandate_repair_verifier.py",
            ]
        receipts = self.state.setdefault("source_receipts", {})
        for relative in relatives:
            source = ROOT / relative
            target = self.output / "frozen_sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(source):
                raise RuntimeError(
                    f"source changed after first use: {relative}; explicit new protocol required"
                )
            if not target.exists():
                target.write_bytes(source.read_bytes())
            receipts.setdefault(relative, dict(sha256=digest(source), frozen_at=now()))
        # These are direct source identities, not an assertion that dirty main is a formal pin.
        self.persist()

    def stage(self, name, module, arguments, gpu_worker=True):
        self.costs()
        used = self.costs()["gpu_stage_seconds"]
        remaining = GPU_CAP_SECONDS - used
        if gpu_worker and remaining <= 0:
            raise BudgetStop("12-hour cumulative GPU-stage allowance exhausted")
        self.source_receipt(module)
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{name}_attempt{self.attempt}.log"
        entry = dict(
            name=name, status="RUNNING", log=str(log), started_at=now(), gpu_worker=gpu_worker
        )
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
                if gpu_worker:
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
                    code = self.child.wait(timeout=max(1, remaining) if gpu_worker else 3600)
                except subprocess.TimeoutExpired as error:
                    self.child.terminate()
                    try:
                        self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.child.kill()
                        self.child.wait()
                    raise BudgetStop("cumulative GPU-stage wall-clock limit reached") from error
            if code:
                raise RuntimeError(f"{name} exited {code}; see {log}")
            entry["status"] = "COMPLETE"
        except BaseException as error:
            entry.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            # Attach/receipt errors can occur after Popen but before wait; retain cleanup ownership.
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
            entry.update(seconds=time.monotonic() - started, finished_at=now())
            self.child = None
            self.persist()
            self.report()

    def report(self):
        costs = self.costs()
        evaluations = {}
        for path in sorted((self.output / "eval").glob("*/summary.json")):
            evaluations[path.parent.name] = metrics(path.parent)
        baseline = self.output / "baseline/sft_panel/summary.json"
        result = dict(
            status=self.state.get("status"),
            current_stage=self.state.get("current_stage"),
            baseline=metrics(baseline.parent) if baseline.exists() else None,
            arms=self.state.get("arms", {}),
            evaluations=evaluations,
            costs=costs,
            step2_decision=self.state.get("step2_decision"),
            reserved_test_read=False,
        )
        (self.output / "summary").mkdir(exist_ok=True)
        save(self.output / "summary/readout.json", result)
        lines = [
            "# 有效信号与关键决策学习：V2实际读数",
            "",
            f"状态：`{result['status']}`；当前阶段：`{result['current_stage']}`。",
            "",
            "本轮从同一完整强SFT出发。训练面板为有意选择的Train正控，留出按报告分组；旧Dev仅作方向读数。",
            "新Test与旧市场Final OOS均未访问。未完成的分母不作为完整比较。",
            "",
            "| 模型/区域 | 成功/计划 | 完成 | 指标/触发/执行 | 新生成token |",
            "|---|---:|---:|---|---:|",
        ]
        if result["baseline"]:
            evaluations = {"sft_panel_independent_native": result["baseline"], **evaluations}
        for key, value in evaluations.items():
            lines.append(
                f"| {key} | {value['successes']}/{value['planned']} | {value['complete']} | "
                f"{value['metric_correct']}/{value['condition_correct']}/"
                f"{value['portfolio_correct']} | {value['tokens']} |"
            )
        lines += [
            "",
            "| 学习臂 | 状态 | 完成批次 | 有效更新 | 实际生成token |",
            "|---|---|---:|---:|---:|",
        ]
        for key, value in result["arms"].items():
            lines.append(
                f"| {key} | {value.get('status')} | {len(value.get('rounds', []))} | "
                f"{value.get('completed_optimizer_steps', 0)} | "
                f"{value.get('cumulative_tokens', 0)} |"
            )
        lines += [
            "",
            f"新增生成合计：{costs['total_new_generation_tokens']} token，含失败请求。",
            f"累计GPU阶段墙钟：{costs['gpu_stage_seconds']:.1f}秒，含加载；非纯忙碌时间。",
            "继续SFT监督token单列，成本匹配按预定阶段墙钟，不宣称严格等FLOPs。",
            "",
            "[实际状态](state.json) · [完整读数](summary/readout.json) · "
            "[数据分组](data/manifest.json)",
            "",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines))

    def ready(self):
        manifest = self.output / "data/manifest.json"
        deadline = time.monotonic() + 3600
        while not manifest.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("data manifest was not prepared within one hour")
            self.persist(current_stage="WAITING_FROZEN_DATA")
            time.sleep(2)
        if read(manifest).get("status") not in {"COMPLETE", "FROZEN"}:
            raise RuntimeError("data manifest incomplete")
        if self.state.get("data_manifest_sha256") not in {None, digest(manifest)}:
            raise RuntimeError("frozen data manifest changed between controller attempts")
        for name, expected in read(manifest).get("output_sha256", {}).items():
            if digest(self.output / "data" / name) != expected:
                raise RuntimeError(f"frozen data artifact changed: {name}")
        signal_report = self.output / "diagnostics/SIGNAL_REPORT.md"
        if signal_report.exists():
            self.state.update(step1_status="COMPLETE", step1_report=str(signal_report))
        for name in ("train_panel.jsonl", "holdout.jsonl", "teacher_filtered.jsonl"):
            if not (self.output / "data" / name).exists():
                raise RuntimeError(f"data artifact missing: {name}")
        frozen_plan = self.output / "frozen_sources/plan.md"
        frozen_plan.parent.mkdir(exist_ok=True)
        if not frozen_plan.exists():
            frozen_plan.write_bytes(PLAN.read_bytes())
        if not (self.output / "frozen_sources/controller.py").exists():
            (self.output / "frozen_sources/controller.py").write_bytes(Path(__file__).read_bytes())
        controller_source = self.output / "frozen_sources" / f"controller_attempt{self.attempt}.py"
        controller_source.write_bytes(Path(__file__).read_bytes())
        helper_receipts = {}
        for name in (
            "run_mandate_repair.py",
            "run_mandate_composition.py",
            "run_financial_evidence.py",
            "run_financial_mc.py",
        ):
            helper = ROOT / "scripts/diagnostics" / name
            frozen = self.output / "frozen_sources" / f"attempt{self.attempt}" / name
            frozen.parent.mkdir(exist_ok=True)
            frozen.write_bytes(helper.read_bytes())
            helper_receipts[name] = dict(sha256=digest(helper), frozen=str(frozen))
        self.persist(
            current_controller_helper_receipts=helper_receipts,
            data_manifest_sha256=digest(manifest),
            current_controller_source_sha256=digest(Path(__file__)),
            current_controller_source=str(controller_source),
        )

    def collection(
        self, name, data, adapter, destination, *, repetitions, temperature, allowance, seed=42
    ):
        existing = (destination / "summary.json").exists()
        required_one_group = repetitions * 10 * 16384
        if not existing and allowance < required_one_group:
            raise BudgetStop(f"no complete task-group allowance for {name}")
        # The collector admits complete K4 groups incrementally; never launch a model per task.
        if self.fresh(destination):
            self.stage(
                name,
                COLLECTOR,
                [
                    "--data",
                    data,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    adapter,
                    "--output",
                    destination,
                    "--seed",
                    seed,
                    "--temperature",
                    temperature,
                    "--repetitions",
                    repetitions,
                    "--token-budget",
                    allowance,
                    "--engine-mode",
                    "eager",
                    *ENVIRONMENT,
                ],
            )
        summary = read(destination / "summary.json")
        expected = len(rows(data)) * repetitions
        if summary.get("planned_trajectories") != expected:
            raise RuntimeError(f"{name} changed fixed denominator")
        return summary, summary.get("complete_trajectories") == expected

    def baseline(self):
        self.persist(current_step=2, baseline_status="RUNNING")
        destination = self.output / "baseline/sft_panel"
        summary, complete = self.collection(
            "sft_independent_panel",
            self.output / "data/train_panel.jsonl",
            INITIAL,
            destination,
            repetitions=4,
            temperature=1,
            allowance=BASELINE_CAP - generation_tokens(self.output / "baseline"),
            seed=142,
        )
        self.persist(
            baseline_status="COMPLETE" if complete else "INCOMPLETE",
            baseline_metrics=metrics(destination),
        )
        if not complete:
            raise BudgetStop("independent panel did not complete its fixed denominator")
        return summary

    def eval_allowance(self):
        return min(
            EVAL_CAP - generation_tokens(self.output / "eval"),
            TOTAL_CAP - self.costs()["total_new_generation_tokens"],
        )

    def reuse_dev(self, destination):
        from src.agents.mandate_repair_train import adapter_digest

        source = ROOT / "runs/m3-financial-evidence-20261030-v2/eval/sft_dev"
        original = read(source / "summary.json")
        expected = dict(
            adapter_path=str(INITIAL),
            environment="financial_evidence",
            temperature=0.0,
            seed=43,
            repetitions=1,
            decode_mode="native",
            diagnostic_mode="standard",
            data_sha256=digest(DEV),
            max_context_tokens=16384,
        )
        if original.get("status") != "COMPLETE" or any(
            original.get(k) != v for k, v in expected.items()
        ):
            return False
        if original.get("adapter_sha256") != adapter_digest(INITIAL):
            return False
        destination.mkdir(parents=True, exist_ok=True)
        copied = {k: v for k, v in original.items() if k not in {"results", "groups"}}
        copied.update(
            total_tokens=0,
            model_requests=0,
            evaluation_reused=True,
            evaluation_source=str(source),
            source_sampled_tokens=original["total_tokens"],
        )
        save(destination / "summary.json", copied)
        shutil.copyfile(
            source / "trajectory_results.jsonl", destination / "trajectory_results.jsonl"
        )
        return True

    def evaluate(self, key, adapter, split, *, repetitions=1, temperature=0, data=None):
        if split == "test":
            raise RuntimeError("reserved Test is not part of V2")
        destination = self.output / "eval" / f"{key}_{split}"
        if key == "sft" and split == "dev" and not (destination / "summary.json").exists():
            self.reuse_dev(destination)
        if data is None:
            data = DEV if split == "dev" else self.output / "data" / f"{split}.jsonl"
        summary, complete = self.collection(
            f"eval_{key}_{split}",
            data,
            adapter,
            destination,
            repetitions=repetitions,
            temperature=temperature,
            allowance=max(1, self.eval_allowance()),
            seed=43,
        )
        summary.update(policy=key, split=split)
        save(destination / "summary.json", summary)
        self.report()
        if not complete:
            raise BudgetStop(f"{key}/{split} incomplete within fixed evaluation allowance")
        return summary

    def fresh_training(self, destination):
        if (destination / "summary.json").exists():
            return False, None
        checkpoint = destination / "training_state.pt"
        if checkpoint.exists():
            archive = destination.parent / "failed_attempts"
            archive.mkdir(exist_ok=True)
            retained = archive / f"{destination.name}_resume_{uuid.uuid4().hex[:8]}"
            destination.rename(retained)
            resume = retained / "training_state.pt"
            self.state.setdefault("training_resumes", []).append(
                dict(
                    destination=str(destination),
                    checkpoint=str(resume),
                    resumed_at=now(),
                    retains_fp32_and_adam=True,
                )
            )
            self.persist()
            return True, resume
        return self.fresh(destination), None

    def batch_data(self, index):
        result = self.output / "data" / f"batch_round{index:03d}.jsonl"
        if not result.exists():
            raise RuntimeError(f"frozen batch not ready: {result}")
        return result

    def train_arm(self, key, lr):
        directory = self.output / "training" / key
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = (
            read(progress_path)
            if progress_path.exists()
            else dict(
                status="RUNNING",
                learning_rate=lr,
                rounds=[],
                final_adapter=str(INITIAL),
                completed_optimizer_steps=0,
                cumulative_tokens=0,
                seed=42,
            )
        )
        if progress["status"] in {"COMPLETE", "COMPLETE_LIMITED"}:
            self.state["arms"][key] = progress
            return Path(progress["final_adapter"])
        actor = Path(progress["final_adapter"])
        for index in range(len(progress["rounds"]), 6):
            rd = directory / f"round_{index:03d}"
            rd.mkdir(exist_ok=True)
            selected = self.batch_data(index)
            task_count = len(rows(selected))
            if task_count != 8:
                raise RuntimeError("frozen V2 batch requires eight concrete tasks")
            remaining = ARM_CAP - generation_tokens(directory)
            if remaining < GROUP_RESERVATION:
                progress["stop_reason"] = "INSUFFICIENT_ALLOWANCE_FOR_FIXED_BATCH"
                break
            collection = rd / "collect"
            sampling_seed = parent_collection_seed(key, index)
            sampled, complete = self.collection(
                f"{key}_round{index:03d}_collect",
                selected,
                actor,
                collection,
                repetitions=4,
                temperature=1,
                allowance=remaining,
                seed=sampling_seed,
            )
            if not complete:
                progress["stop_reason"] = "INCOMPLETE_FIXED_BATCH_NOT_TRAINED"
                progress["partial_collection"] = str(collection)
                break
            trained_path = rd / "train"
            need_train, partial_resume = self.fresh_training(trained_path)
            if need_train:
                arguments = [
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
                    42,
                    "--collection-seed",
                    sampling_seed,
                    "--arm",
                    "rloo",
                    "--learning-rate",
                    lr,
                    "--trajectory-minibatch-size",
                    task_count,
                    "--mixed-task-minibatches",
                ]
                if index in {0, 5}:
                    arguments += ["--diagnostic-request-limit", 16]
                if partial_resume:
                    arguments += ["--resume-state", partial_resume]
                elif progress["rounds"]:
                    arguments += [
                        "--resume-state",
                        Path(progress["rounds"][-1]["train"]) / "training_state.pt",
                    ]
                self.stage(f"{key}_round{index:03d}_train", TRAINER, arguments)
            trained = read(trained_path / "summary.json")
            actor = trained_path / "adapter"
            progress["rounds"].append(
                dict(
                    round=index,
                    sampling_seed=sampling_seed,
                    sampling_namespace=f"agentic-learning-v2/native-parent/base42/{key}/round{index:03d}",
                    optimizer_seed=42,
                    collect=str(collection),
                    train=str(trained_path),
                    optimizer_updates=trained.get("optimizer_updates", 0),
                    completed_optimizer_steps=trained["completed_optimizer_steps"],
                    parent_trajectories=sampled["complete_trajectories"],
                    successes=sampled["successes"],
                    tokens=sampled["total_tokens"],
                    advantages_std=trained.get("advantages_std"),
                    effective_advantage_requests=trained.get("effective_advantage_requests"),
                    trainable_sha256_before=trained.get("trainable_sha256_before"),
                    trainable_sha256_after=trained.get("trainable_sha256_after"),
                )
            )
            progress.update(
                final_adapter=str(actor),
                completed_optimizer_steps=trained["completed_optimizer_steps"],
                cumulative_tokens=generation_tokens(directory),
            )
            save(progress_path, progress)
            self.state["arms"][key] = progress
            self.persist(costs=self.costs())
            self.report()
            if index == 2:
                self.midpoint(key, actor)
        progress.update(
            status="COMPLETE" if len(progress["rounds"]) == 6 else "COMPLETE_LIMITED",
            final_adapter=str(actor),
            cumulative_tokens=generation_tokens(directory),
            completed_at=now(),
            unused_tokens=ARM_CAP - generation_tokens(directory),
        )
        save(progress_path, progress)
        self.state["arms"][key] = progress
        self.persist(costs=self.costs())
        self.report()
        return actor

    def midpoint(self, key, actor):
        # Predeclare a small panel before any checkpoint result; reserve endpoints first.
        destination = self.output / "data/midpoint_panel.jsonl"
        if not destination.exists():
            panel = rows(self.output / "data/train_panel.jsonl")
            destination.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in panel[:2])
            )
        baseline = read(self.output / "baseline/sft_panel/summary.json")
        mean = baseline["total_tokens"] / baseline["complete_trajectories"]
        # Sum only endpoint suites not yet completed; reused initialSFT Dev is not an endpoint.
        mandatory = 0
        for endpoint in (*ARMS, "continued_sft"):
            for split, count in (
                ("train_panel", len(rows(self.output / "data/train_panel.jsonl")) * 4),
                ("holdout", len(rows(self.output / "data/holdout.jsonl"))),
                ("dev", 100),
            ):
                result = self.output / "eval" / f"{endpoint}_{split}" / "summary.json"
                if not result.exists() or read(result).get("complete_trajectories") != count:
                    mandatory += count
        estimated = mandatory * mean * 1.05 + len(rows(destination)) * 4 * mean
        if self.eval_allowance() < estimated:
            self.state.setdefault("midpoint_decisions", {})[key] = dict(
                status="SKIPPED_TO_RESERVE_COMPLETE_ENDPOINTS",
                forecast_tokens=estimated,
                remaining_tokens=self.eval_allowance(),
                decided_at=now(),
            )
            self.persist()
            return
        self.evaluate(
            key,
            "" if actor is None else actor,
            "midpoint_panel",
            repetitions=4,
            temperature=1,
            data=destination,
        )

    def endpoint(self, key, actor):
        self.evaluate(key, actor, "train_panel", repetitions=4, temperature=1)
        self.evaluate(key, actor, "holdout")
        self.evaluate(key, actor, "dev")

    def continued_sft(self):
        directory = self.output / "training/continued_sft"
        if (directory / "summary.json").exists():
            return directory / "adapter"
        times = {}
        for key in ARMS:
            times[key] = sum(
                s.get("seconds", 0)
                for s in self.state["stages"]
                if s.get("gpu_worker") and s["name"].startswith(key + "_round")
            )
        cap = min(times.values())
        if cap <= 0:
            raise RuntimeError(
                "cannot assign continuedSFT time budget without actual RL chain time"
            )
        contract_path = self.output / "continued_sft_contract.json"
        if contract_path.exists():
            contract = read(contract_path)
            if contract["teacher_data_sha256"] != digest(
                self.output / "data/teacher_filtered.jsonl"
            ):
                raise RuntimeError("continuedSFT teacher data changed after time contract freeze")
            cap = contract["maximum_training_seconds"]
        else:
            save(
                contract_path,
                dict(
                    frozen_at=now(),
                    rl_chain_gpu_seconds=times,
                    maximum_training_seconds=cap,
                    maximum_epochs=1,
                    teacher_data=str(self.output / "data/teacher_filtered.jsonl"),
                    teacher_data_sha256=digest(self.output / "data/teacher_filtered.jsonl"),
                    excludes_evaluation=True,
                    includes_rl_sampling_loading_oldlogprobs_updates=True,
                ),
            )
        need_train, partial_resume = self.fresh_training(directory)
        if need_train:
            arguments = [
                "sft",
                "--data",
                self.output / "data/teacher_filtered.jsonl",
                "--model",
                MODEL,
                "--adapter",
                INITIAL,
                "--output",
                directory,
                "--seed",
                43,
                "--learning-rate",
                5e-5,
                "--gradient-accumulation",
                16,
                "--max-training-seconds",
                cap,
            ]
            if partial_resume:
                arguments += ["--resume-state", partial_resume]
            self.stage("continued_sft", TRAINER, arguments)
        summary = read(directory / "summary.json")
        if summary.get("status") not in {"COMPLETE", "COMPLETE_TIME_LIMIT"}:
            raise RuntimeError("continuedSFT did not produce a complete update endpoint")
        self.persist(
            continued_sft=dict(
                status=summary["status"],
                maximum_training_seconds=cap,
                actual=summary,
                endpoint=str(directory / "adapter"),
            )
        )
        return directory / "adapter"

    def step2(self):
        self.evaluate("sft", INITIAL, "holdout")
        self.evaluate("sft", INITIAL, "dev")
        endpoints = {}
        for key, lr in ARMS.items():
            endpoints[key] = self.train_arm(key, lr)
            self.endpoint(key, endpoints[key])
        endpoints["continued_sft"] = self.continued_sft()
        self.endpoint("continued_sft", endpoints["continued_sft"])
        scores = {key: metrics(self.output / "eval" / f"{key}_holdout") for key in ARMS}
        selected = max(ARMS, key=lambda k: (scores[k]["metric_correct"], k == "rloo_lr2e6"))
        decision = dict(
            status="COMPLETE",
            completed_at=now(),
            selected_learning_rate=ARMS[selected],
            selected_arm=selected,
            selection_basis="report_holdout_metric_correct_tie_lower_lr",
            holdout_now_development_evidence=True,
            holdout=scores,
            stage3_status="PENDING_EVIDENCE_REVIEW",
            reserved_test_opened=False,
            endpoints={k: str(v) for k, v in endpoints.items()},
            costs=self.costs(),
        )
        save(self.output / "step2_decision.json", decision)
        self.persist(
            step2_status="COMPLETE", step2_decision=decision, step3_status="PENDING_EVIDENCE_REVIEW"
        )
        self.report()

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.ready()
            self.claim_gpu()
            self.baseline()
            if self.baseline_only:
                self.persist(status="BASELINE_COMPLETE", current_stage=None, finished_at=now())
            elif (self.output / "step2_decision.json").exists():
                self.persist(status="STEP2_COMPLETE", current_stage=None, finished_at=now())
            else:
                self.step2()
                self.persist(status="STEP2_COMPLETE", current_stage=None, finished_at=now())
        except BudgetStop as error:
            self.persist(
                status="RESOURCE_LIMIT", error=str(error), current_stage=None, finished_at=now()
            )
        except BaseException as error:
            code = 1
            self.persist(
                status="FAILED", error=f"{type(error).__name__}: {error}", finished_at=now()
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
                    self.state.update(holder_restore="WATCHDOG_PENDING", restore_error=str(error))
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            self.persist()
            save(self.output / "completion.json", self.state)
            self.report()
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--readout-only", action="store_true")
    args = parser.parse_args()
    if args.readout_only:
        controller = Controller.__new__(Controller)
        controller.output = args.output.resolve()
        controller.state = read(controller.output / "state.json")
        controller.report()
        print(json.dumps(controller.costs(), ensure_ascii=False))
        return
    raise SystemExit(Controller(args.output, args.resume, baseline_only=args.baseline_only).run())


if __name__ == "__main__":
    main()
