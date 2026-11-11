#!/usr/bin/env python3
"""Bounded calculation-response actor update screen, with existing controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_financial_coverage as coverage
from run_financial_learning import INITIAL, BudgetStop, digest
from run_financial_mc import generation_tokens
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import ROOT, read, save

DEFAULT = ROOT / "runs/agentic-rl-step-focus-20261110-v4"
PREVIOUS = ROOT / "runs/agentic-rl-coverage-20261105-v3"
REFERENCE = ROOT / "runs/agentic-rl-transfer-probe-20261105-v1"
PLAN = ROOT / "docs/2026-11-10_AGENTIC_RL_STEP_FOCUS_V4.md"
BASE_TRAINER = "src.agents.mandate_repair_train"
FOCUS_TRAINER = "src.agents.financial_step_focus_train"


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
        profile="financial_calculation_focus_v4",
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
            outcomes_sha256=digest(directory / "trajectory_results.jsonl"),
        )
    rules = dict(
        profile="financial_calculation_focus_v4",
        actor_update_scope="calculate_only",
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
            training=6_000_000, evaluation=2_200_000, recovery=800_000, total=9_000_000
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
        coverage.TRAINER = FOCUS_TRAINER
        self.persist(
            experiment="financial_calculation_focus_v4",
            reserved_test_access=False,
            contract_sha256=digest(self.output / "contract.json"),
            recovery_allocation=self.state.get("recovery_allocation", {}),
        )
        source = self.output / "frozen_sources" / f"focus_controller_attempt{self.attempt}.py"
        source.write_bytes(Path(__file__).read_bytes())
        self.source_receipt(BASE_TRAINER)
        self.source_receipt(FOCUS_TRAINER)
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

    def train_wide(self):
        directory = self.output / "training/calculate_only"
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = (
            read(progress_path)
            if progress_path.exists()
            else dict(
                rounds=[],
                final_adapter=str(INITIAL),
                completed_optimizer_steps=0,
                actor_update_scope="calculate_only",
            )
        )
        actor = Path(progress["final_adapter"])
        for index in range(len(progress["rounds"]), 6):
            rd = directory / f"round_{index:03d}"
            rd.mkdir(exist_ok=True)
            seed = coverage.sampling_seed(index)
            if index == 0:
                collection = Path(self.contract["reused_first_collection"])
                sampled = verify_reusable_first_batch(
                    collection, self.contract["initial_adapter_sha256"]
                )
            else:
                collection = rd / "collect"
                sampled, complete = self.collection(
                    f"focus_round{index:03d}_collect",
                    self.output / "data" / f"batch_round{index:03d}.jsonl",
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
                ]
                previous = partial or (
                    Path(progress["rounds"][-1]["train"]) / "training_state.pt"
                    if progress["rounds"]
                    else None
                )
                if previous:
                    args += ["--resume-state", previous]
                self.stage(f"focus_round{index:03d}_train", FOCUS_TRAINER, args)
            trained = read(destination / "summary.json")
            if trained["recipe"].get("actor_update_scope") != "calculate_only":
                raise RuntimeError("wrong actor update scope")
            actor = destination / "adapter"
            progress["rounds"].append(
                dict(
                    round=index,
                    collect=str(collection),
                    train=str(destination),
                    sampling_seed=seed,
                    parent_trajectories=sampled["complete_trajectories"],
                    reused_collection=index == 0,
                    successes=sampled["successes"],
                    tokens=sampled["total_tokens"],
                    new_tokens=0 if index == 0 else sampled["total_tokens"],
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
        super().report()
        path = self.output / "REPORT.md"
        text = path.read_text().replace(
            "# 扩覆盖快速试验：实际进度与结果", "# V4仅计算响应更新：实际进度与结果"
        )
        text = text.replace("只训练宽覆盖一臂", "仅计算响应参与actor更新")
        text = text.replace("| 宽覆盖 |", "| 仅计算响应更新 |")
        text = text.replace(
            "本轮仅作方向筛查；历史窄臂题目选择与新宽臂不同，不单独归因于覆盖。",
            "本轮只改变actor更新位置，保留原优势和原分母；不是无偏整体策略梯度。"
            "首批32父轨迹复用同一SFT记录、其余新采；不重复计费。",
        )
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
