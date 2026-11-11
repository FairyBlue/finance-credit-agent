#!/usr/bin/env python3
"""Train with action-conditioned double-continuation advantages from real rollouts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import run_financial_coverage as coverage
from run_financial_learning import INITIAL, BudgetStop, digest
from run_financial_mc import generation_tokens
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, ROOT, gpu, now, read, save

DEFAULT = ROOT / "runs/agentic-rl-action-q-20261111-v6"
PREVIOUS = ROOT / "runs/agentic-rl-coverage-20261105-v3"
REFERENCE = ROOT / "runs/agentic-rl-transfer-probe-20261105-v1"
PLAN = ROOT / "docs/2026-11-11_AGENTIC_RL_ACTION_Q_TRAINING.md"
BASE_TRAINER = "src.agents.mandate_repair_train"
Q_TRAINER = "src.agents.financial_action_q_train"
Q_AUX = "src.agents.financial_action_q_collect"
TRAIN_CAP, EVAL_CAP, RECOVERY_CAP = 9_000_000, 2_200_000, 800_000
TOTAL_CAP, GPU_SECONDS = 12_000_000, 7200
AUX_BATCH_CAP = 1_200_000
ESTIMATOR = "parent_plus_independent_post_action_return_mean_v1"


def auxiliary_seed(index):
    import hashlib

    raw = f"financial-action-q-v6/independent-post-action/round{index:03d}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") % (2**31 - 1)


def frozen_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"frozen artifact changed: {path}")
    if not path.exists():
        path.write_bytes(payload)


def verify_reusable_first_batch(directory, initial_adapter_sha256):
    started = read(directory / "collection_started.json")
    summary = read(directory / "summary.json")
    if (
        started["adapter_sha256"] != initial_adapter_sha256
        or Path(started["adapter_path"]).resolve() != INITIAL.resolve()
        or summary.get("status") != "COMPLETE"
        or summary["complete_trajectories"] != 32
        or summary["seed"] != coverage.sampling_seed(0)
    ):
        raise RuntimeError("reused first batch must come from the exact initial SFT policy")
    return summary


def prepare(output):
    from src.agents.mandate_repair_train import adapter_digest

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if read(PREVIOUS / "state.json")["status"] != "COMPLETE":
        raise RuntimeError("V3 must have completed before this single-factor screen")
    previous = read(PREVIOUS / "data/manifest.json")
    data = output / "data"
    for name in ["train_panel.jsonl", *(f"batch_round{i:03d}.jsonl" for i in range(6))]:
        frozen_bytes(data / name, (PREVIOUS / "data" / name).read_bytes())
    initial_sha = adapter_digest(INITIAL)
    reused = PREVIOUS / "training/wide/round_000/collect"
    first = verify_reusable_first_batch(reused, initial_sha)
    if first["data_sha256"] != digest(data / "batch_round000.jsonl"):
        raise RuntimeError("reused first batch data differ from the fixed V3 first batch")
    manifest = dict(
        profile="financial_action_q_training_v6",
        status="FROZEN",
        train_panel_path=str(data / "train_panel.jsonl"),
        holdout_path=previous["holdout_path"],
        selected_reports=previous["selected_reports"],
        parent_trajectories=192,
        new_parent_trajectories=160,
        reused_parent_trajectories=32,
        reused_first_collection=str(reused),
        source_manifest_sha256=digest(PREVIOUS / "data/manifest.json"),
        output_sha256={p.name: digest(p) for p in sorted(data.glob("*.jsonl"))},
    )
    frozen_bytes(data / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    old = read(REFERENCE / "contract.json")["rules"]
    if digest(Path(manifest["holdout_path"])) != old["data_sha256"]:
        raise RuntimeError("reference evaluation data identity changed")
    if initial_sha != old["models"]["sft"]["adapter_sha256"]:
        raise RuntimeError("common SFT identity changed")
    identity_fields = (
        "data_sha256",
        "seed",
        "temperature",
        "max_context_tokens",
        "max_response_tokens",
        "environment",
        "decode_mode",
    )
    directories = {
        "sft": REFERENCE / "eval/sft/sample",
        "narrow_rloo": REFERENCE / "eval/rloo_high/sample",
        "all_steps_wide": PREVIOUS / "eval/wide/sample",
        "v5_feedback": ROOT / "runs/agentic-rl-invalid-feedback-20261110-v5/eval/wide/sample",
    }
    references = {}
    tasks = rows(manifest["holdout_path"])
    for name, directory in directories.items():
        summary = read(directory / "summary.json")
        if any(summary.get(k) != old[k] for k in identity_fields):
            raise RuntimeError(f"reference evaluation identity changed: {name}")
        coverage.indexed_outcomes(tasks, rows(directory / "trajectory_results.jsonl"))
        references[name] = dict(
            path=str(directory),
            successes=summary["successes"],
            metric_correct=summary["outcome_diagnostics"]["metric_correct"],
            invalid_output_requests=summary.get("tool_counts", {}).get("invalid_output", 0),
            no_submission=summary["outcome_diagnostics"]
            .get("failure_categories", {})
            .get("no_submission", 0),
            outcomes_sha256=digest(directory / "trajectory_results.jsonl"),
        )
    rules = dict(
        profile="financial_action_q_training_v6",
        actor_update_scope="calculate_and_parse_failures",
        action_advantage_estimator=ESTIMATOR,
        auxiliary_generation_cap_per_batch=AUX_BATCH_CAP,
        future_parent_generation_forecast_per_batch=900_000,
        unsupported_q_groups_use_original_rloo=True,
        auxiliary_actor_examples=False,
        variance_gate=False,
        initial_adapter=str(INITIAL),
        initial_adapter_sha256=initial_sha,
        data_manifest_sha256=digest(data / "manifest.json"),
        references=references,
        evaluation={k: old[k] for k in identity_fields},
        batches=6,
        parent_trajectories=192,
        new_parent_trajectories=160,
        reused_first_collection=str(reused),
        reused_parent_tokens=first["total_tokens"],
        reused_requests_sha256=digest(reused / "requests.jsonl"),
        reused_outcomes_sha256=digest(reused / "trajectory_results.jsonl"),
        learning_rate=1e-5,
        optimizer_seed=42,
        original_request_denominators=True,
        generation_caps=dict(
            training=TRAIN_CAP, evaluation=EVAL_CAP, recovery=RECOVERY_CAP, total=TOTAL_CAP
        ),
        gpu_stage_seconds_cap=7200,
        reserved_test_read=False,
        no_intermediate_evaluation=True,
        additional_sampling_automatic=False,
    )
    frozen_bytes(output / "contract.json", (json.dumps(rules, indent=2) + "\n").encode())
    frozen = output / "frozen_sources"
    frozen.mkdir(exist_ok=True)
    if not (frozen / "plan.md").exists():
        (frozen / "plan.md").write_bytes(PLAN.read_bytes())
    return rules


class Controller(coverage.Controller):
    def __init__(self, output, resume=False):
        self.contract = prepare(output)
        CompositionController.__init__(self, Path(output), resume)
        # This controller process substitutes the worker entrypoint only.
        # No historical trainer/controller file or prior run is modified.
        self.persist(
            experiment="financial_action_q_training_v6",
            reserved_test_access=False,
            contract_sha256=digest(self.output / "contract.json"),
            recovery_allocation=self.state.get("recovery_allocation", {}),
        )
        source = self.output / "frozen_sources" / f"action_q_controller_attempt{self.attempt}.py"
        source.write_bytes(Path(__file__).read_bytes())
        self.source_receipt(BASE_TRAINER)
        self.source_receipt(Q_TRAINER)
        self.source_receipt(Q_AUX)
        for module in (
            "src.agents.financial_mc_probe",
            "src.agents.financial_credit_probe",
            "src.agents.financial_evidence_env",
            "src.agents.same_prefix_online_collect",
        ):
            self.source_receipt(module)
        for name in (
            "run_financial_coverage.py",
            "run_financial_learning.py",
            "run_mandate_composition.py",
            "run_mandate_repair.py",
        ):
            frozen_bytes(
                self.output / "frozen_sources/helpers" / name,
                (ROOT / "scripts/diagnostics" / name).read_bytes(),
            )
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
            raise BudgetStop("12M generation allowance exceeded")
        return dict(
            training_generation_tokens=training,
            evaluation_generation_tokens=evaluation,
            total_new_generation_tokens=training + evaluation,
            gpu_stage_seconds=seconds,
            reused_reference_new_tokens=0,
            failed_requests_included=True,
        )

    def stage(self, name, module, arguments, gpu_worker=True):
        if module not in {COLLECTOR, Q_AUX, Q_TRAINER}:
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
        directory = self.output / "training/action_q"
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = (
            read(progress_path)
            if progress_path.exists()
            else dict(
                rounds=[],
                final_adapter=str(INITIAL),
                completed_optimizer_steps=0,
                actor_update_scope="calculate_and_parse_failures",
                action_advantage_estimator=ESTIMATOR,
            )
        )
        actor = Path(progress["final_adapter"])
        for index in range(len(progress["rounds"]), 6):
            rd = directory / f"round_{index:03d}"
            rd.mkdir(exist_ok=True)
            seed = coverage.sampling_seed(index)
            selected = self.output / "data" / f"batch_round{index:03d}.jsonl"
            if index == 0:
                collection = Path(self.contract["reused_first_collection"])
                sampled = verify_reusable_first_batch(
                    collection, self.contract["initial_adapter_sha256"]
                )
            else:
                collection = rd / "collect"
                sampled, complete = self.collection(
                    f"action_q_round{index:03d}_collect",
                    selected,
                    actor,
                    collection,
                    repetitions=4,
                    temperature=1,
                    allowance=self.allowance("training")
                    - generation_tokens(self.output / "training"),
                    seed=seed,
                )
                if not complete or sampled.get("status") != "COMPLETE":
                    raise BudgetStop("incomplete parent batch retained, not trained")
            auxiliary = rd / "action_q"
            if self.fresh(auxiliary):
                remaining = self.allowance("training") - generation_tokens(self.output / "training")
                # Preserve a measured allowance for future complete parent batches.
                parent_reserve = (5 - index) * 900_000
                aux_allowance = max(0, min(AUX_BATCH_CAP, remaining - parent_reserve))
                self.stage(
                    f"action_q_round{index:03d}_aux",
                    Q_AUX,
                    [
                        "--data",
                        selected,
                        "--parents",
                        collection,
                        "--base-model",
                        coverage.MODEL,
                        "--adapter",
                        actor,
                        "--output",
                        auxiliary,
                        "--seed",
                        auxiliary_seed(index),
                        "--token-budget",
                        aux_allowance,
                        "--max-context-tokens",
                        16384,
                    ],
                )
            if not (auxiliary / "action_q_values.jsonl").exists():
                raise RuntimeError(
                    "Q worker did not write its complete or unsupported group manifest"
                )
            destination = rd / "train"
            needed, partial = self.fresh_training(destination)
            if needed:
                args = [
                    "rl",
                    "--data",
                    collection,
                    "--base-model",
                    coverage.MODEL,
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
                    "--action-q-values",
                    auxiliary / "action_q_values.jsonl",
                ]
                previous = partial or (
                    Path(progress["rounds"][-1]["train"]) / "training_state.pt"
                    if progress["rounds"]
                    else None
                )
                if previous:
                    args += ["--resume-state", previous]
                self.stage(f"action_q_round{index:03d}_train", Q_TRAINER, args)
            trained = read(destination / "summary.json")
            if trained["recipe"].get("action_advantage_estimator") != ESTIMATOR:
                raise RuntimeError("wrong Q advantage training recipe")
            actor = destination / "adapter"
            progress["rounds"].append(
                dict(
                    round=index,
                    collect=str(collection),
                    train=str(destination),
                    action_q=str(auxiliary),
                    sampling_seed=seed,
                    parent_trajectories=sampled["complete_trajectories"],
                    reused_collection=index == 0,
                    successes=sampled["successes"],
                    tokens=sampled["total_tokens"],
                    new_parent_tokens=0 if index == 0 else sampled["total_tokens"],
                    auxiliary_tokens=generation_tokens(auxiliary),
                    optimizer_updates=trained["optimizer_updates"],
                    effective_advantage_requests=trained.get("effective_advantage_requests"),
                    replaced_target_requests=trained.get("replaced_target_requests", 0),
                    complete_q_task_groups=trained.get("complete_q_task_groups", 0),
                    changed_target_raw_advantages=trained.get("changed_target_raw_advantages", 0),
                    target_raw_advantage_l1_change=trained.get("target_raw_advantage_l1_change", 0),
                    unsupported_q_task_ids=trained.get("unsupported_q_task_ids", []),
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
        super().report()
        path = self.output / "REPORT.md"
        text = path.read_text().replace(
            "# 扩覆盖快速试验：实际进度与结果", "# V6动作Q双续写：真实训练进度与结果"
        )
        text = text.replace("只训练宽覆盖一臂", "动作Q优势用于计算及解析失败响应更新")
        text = text.replace("| 宽覆盖 |", "| 动作Q训练 |")
        text = text.replace("上限900万token/2小时", "上限1200万token/2小时")
        text = text.replace(
            "本轮仅作方向筛查；历史窄臂题目选择与新宽臂不同，不单独归因于覆盖。",
            "同一实际动作的父后继与独立后继均值估Q；目标请求换动作Q优势，其余原始RLOO不变。"
            "首批32父轨迹复用同一SFT记录、其余新采；不重复计费。",
        )
        format_rows = {
            name: {key: value[key] for key in ("invalid_output_requests", "no_submission")}
            for name, value in self.contract["references"].items()
        }
        summary_path = self.output / "eval/wide/sample/summary.json"
        if summary_path.exists():
            summary = read(summary_path)
            format_rows["v6_action_q"] = dict(
                invalid_output_requests=summary.get("tool_counts", {}).get("invalid_output", 0),
                no_submission=summary["outcome_diagnostics"]
                .get("failure_categories", {})
                .get("no_submission", 0),
            )
        save(self.output / "format_readout.json", format_rows)
        text += "\n| 模型 | 无效输出请求 | 未提交 /64 |\n|---|---:|---:|\n"
        for name, value in format_rows.items():
            text += f"| {name} | {value['invalid_output_requests']} | {value['no_submission']} |\n"
        text += "\n| 批次 | 更新 | Q任务组 | 替换请求 | 原始优势实变 | 辅助token |\n"
        text += "|---|---:|---:|---:|---:|---:|\n"
        for row in self.state.get("arms", {}).get("wide", {}).get("rounds", []):
            text += (
                f"| {row['round'] + 1} | {row['optimizer_updates']} | "
                f"{row.get('complete_q_task_groups', 0)} | "
                f"{row.get('replaced_target_requests', 0)} | "
                f"{row.get('changed_target_raw_advantages', 0)} | {row['auxiliary_tokens']} |\n"
            )
        text += "\n辅助续写不进入actor；预算不支持的完整任务组保留普通RLOO，不设方差准入门槛。\n"
        path.write_text(text)


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
