#!/usr/bin/env python3
"""Persistent three-step controller for compositional mandate learning."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import subprocess
import time
import uuid
from pathlib import Path

from run_mandate_repair import (
    AUX_BUDGET,
    BUDGET,
    COLLECTOR,
    GROUP_RESERVATION,
    MODEL,
    ROOT,
    TRAINER,
    gpu,
    now,
    read,
    save,
)
from run_mandate_repair import (
    Controller as RepairController,
)

DEFAULT = ROOT / "runs/m3-mandate-composition-20261029-v1"
INITIAL_ADAPTER = ROOT / "runs/m3-mandate-repair-20261029-v1/common_sft/adapter"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def second_seed_trigger(evaluations):
    """Allocate a confirmation seed using frozen seed42 endpoints on Dev only."""
    indexed = {
        key: {r["task_id"]: r for r in evaluations[key]}
        for key in ("rloo_seed42", "sham_seed42", "m3_seed42")
    }
    task_ids = set(indexed["m3_seed42"])
    if len(task_ids) != 200 or any(set(values) != task_ids for values in indexed.values()):
        raise RuntimeError("all three complete fixed Dev denominators required")
    successful = {
        key: {task for task, row in values.items() if row.get("success") is True}
        for key, values in indexed.items()
    }
    common = set.intersection(*successful.values())
    rates = {key: len(value) / 200 for key, value in successful.items()}
    calls = {
        key: math.fsum(values[task]["calls_used"] for task in common) / len(common)
        if common else None
        for key, values in indexed.items()
    }
    comparisons = {}
    for control in ("rloo_seed42", "sham_seed42"):
        comparisons[control] = dict(
            success_gain=rates["m3_seed42"] - rates[control],
            common_success_call_reduction=(1 - calls["m3_seed42"] / calls[control])
            if common and calls[control] else None,
        )
    success_signal = all(c["success_gain"] >= 0.05 - 1e-12 for c in comparisons.values())
    cost_signal = bool(common) and all(
        c["success_gain"] >= -1e-12
        and c["common_success_call_reduction"] >= 0.20 - 1e-12
        for c in comparisons.values()
    )
    return dict(
        status="TRIGGERED" if success_signal or cost_signal else "SKIPPED",
        trigger="SUCCESS_GAIN_GE_5PP" if success_signal else (
            "COMMON_SUCCESS_CALL_REDUCTION_GE_20PCT" if cost_signal else "NO_THRESHOLD_GAIN"
        ),
        comparisons=comparisons,
        common_success_tasks=len(common),
        success_rates=rates,
        common_success_mean_calls=calls,
        uses_reserved_test=False,
        statistical_significance_claim=False,
    )


class Controller(RepairController):
    def report(self):
        script = ROOT / "scripts/diagnostics/read_mandate_composition.py"
        if not script.exists():
            return
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(script), "--output", str(self.output)],
            cwd=ROOT, env=self.environment, capture_output=True, text=True,
        )
        (self.output / "readout.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            self.persist(readout_error=result.stderr[-3000:])

    def evaluate(self, key, adapter, split="dev", metadata=None):
        if split == "test":
            frozen = self.output / "frozen_models.json"
            if not frozen.exists() or read(frozen).get("status") != "COMPLETE":
                raise RuntimeError("test opens only after all endpoints and choices freeze")
        destination = self.output / "eval" / f"{key}_{split}"
        data = self.output / "data" / ("development.jsonl" if split == "dev" else "test.jsonl")
        if self.fresh(destination):
            self.stage(f"eval_{key}_{split}", COLLECTOR, [
                "--data", data, "--base-model", MODEL, "--adapter", adapter,
                "--output", destination, "--seed", 43, "--temperature", 0,
                "--repetitions", 1, "--engine-mode", "eager",
            ])
        summary = read(destination / "summary.json")
        if summary["planned_trajectories"] != 200:
            raise RuntimeError("predefined 200-task evaluation denominator changed")
        summary.update(split=split, policy=key, **(metadata or {}))
        save(destination / "summary.json", summary)
        return summary

    def initial_probe(self):
        destination = self.output / "probe"
        selected = self.output / "probe_tasks.jsonl"
        if not selected.exists():
            tasks = rows(self.output / "data/rl_train.jsonl")
            random.Random(42).shuffle(tasks)
            selected.write_text("".join(json.dumps(t) + "\n" for t in tasks[:16]))
        if self.fresh(destination):
            self.stage("initial_training_probe", COLLECTOR, [
                "--data", selected, "--base-model", MODEL, "--adapter", INITIAL_ADAPTER,
                "--output", destination, "--seed", 42, "--temperature", 1,
                "--repetitions", 4, "--engine-mode", "eager",
            ])
        self.persist(initial_probe_complete=True, initial_probe_not_actor_training_data=True)

    def train_arm(self, arm, seed, common):
        key = f"{arm}_seed{seed}"
        directory = self.output / "training" / key
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = read(progress_path) if progress_path.exists() else dict(
            arm=arm, seed=seed, status="RUNNING", token_budget=BUDGET,
            auxiliary_token_cap=AUX_BUDGET if arm != "rloo" else 0,
            cumulative_tokens=0, parent_tokens=0, auxiliary_tokens=0, rounds=[],
            final_adapter=str(common), completed_optimizer_steps=0,
            local_credit_target="business" if arm != "rloo" else None,
        )
        actor = Path(progress["final_adapter"])
        if progress["status"] == "COMPLETE":
            return actor
        tasks = rows(self.output / "data/rl_train.jsonl")
        random.Random(seed).shuffle(tasks)
        round_index = len(progress["rounds"])
        while BUDGET - progress["cumulative_tokens"] >= GROUP_RESERVATION:
            rd = directory / f"round_{round_index:03d}"
            rd.mkdir(parents=True, exist_ok=True)
            selected = rd / "selected.jsonl"
            if not selected.exists():
                offset = round_index * 16
                selected.write_text("".join(
                    json.dumps(tasks[(offset + i) % len(tasks)]) + "\n" for i in range(16)
                ))
            collection = rd / "collect"
            remaining = BUDGET - progress["cumulative_tokens"]
            # Release at most a quarter of auxiliary allowance per 750k token
            # phase, carrying unused allowance forward to updated policies.
            phase = min(3, progress["cumulative_tokens"] // 750_000)
            auxiliary_allowance = max(
                0, min(AUX_BUDGET, (phase + 1) * 250_000) - progress["auxiliary_tokens"]
            ) if arm != "rloo" else 0
            if self.fresh(collection):
                args = [
                    "--data", selected, "--base-model", MODEL, "--adapter", actor,
                    "--output", collection, "--seed", seed, "--temperature", 1,
                    "--repetitions", 4, "--token-budget", remaining,
                    "--max-groups", 16, "--engine-mode", "eager",
                ]
                if arm != "rloo":
                    args += ["--arm", arm, "--aux-token-budget", auxiliary_allowance,
                             "--local-credit-target", "business"]
                self.stage(f"{key}_round{round_index:03d}_collect", COLLECTOR, args)
            collected = read(collection / "summary.json")
            spent = collected["total_tokens"]
            if spent <= 0:
                progress["budget_stop_reason"] = "NO_FULL_TASK_GROUP_ADMITTED"
                break
            destination = rd / "train"
            next_actor = destination / "adapter"
            if self.fresh(destination):
                args = [
                    "rl", "--data", collection, "--base-model", MODEL, "--adapter", actor,
                    "--output", destination, "--seed", seed, "--arm", arm,
                ]
                if progress["rounds"]:
                    args += [
                        "--resume-state",
                        Path(progress["rounds"][-1]["train"]) / "training_state.pt",
                    ]
                if arm != "rloo":
                    args += ["--local-credit", collection / "local_credit.jsonl",
                             "--local-credit-target", "business"]
                self.stage(f"{key}_round{round_index:03d}_train", TRAINER, args)
            trained = read(destination / "summary.json")
            aux = collected.get("auxiliary_tokens", collected.get("aux_tokens", 0))
            progress["cumulative_tokens"] += spent
            progress["auxiliary_tokens"] += aux
            progress["parent_tokens"] += spent - aux
            progress["completed_optimizer_steps"] = trained["completed_optimizer_steps"]
            progress["final_adapter"] = str(next_actor)
            progress["rounds"].append(dict(
                round=round_index, collect=str(collection), train=str(destination),
                cumulative_tokens=progress["cumulative_tokens"], tokens=spent,
                auxiliary_tokens=aux, auxiliary_allowance=auxiliary_allowance,
                optimizer_updates=trained["optimizer_updates"],
                completed_optimizer_steps=trained["completed_optimizer_steps"],
                parent_successes=collected.get("successes"),
                parent_started_trajectories=collected.get("started_trajectories"),
            ))
            if progress["cumulative_tokens"] > BUDGET or progress["auxiliary_tokens"] > AUX_BUDGET:
                raise RuntimeError("actual model token cap exceeded")
            save(progress_path, progress)
            self.state["arms"][key] = progress
            self.persist()
            actor = next_actor
            self.report()
            # The authorized first-batch zero-gradient exit precedes further
            # ordinary or auxiliary sampling, never a search for a lucky seed.
            if arm == "rloo" and round_index == 0 and trained["optimizer_updates"] == 0:
                progress["budget_stop_reason"] = "FIRST_BATCH_NO_EFFECTIVE_ADVANTAGE"
                break
            round_index += 1
        progress.update(
            status="COMPLETE", finished_at=now(), final_adapter=str(actor),
            unused_budget_tokens=BUDGET - progress["cumulative_tokens"],
            endpoints_not_strictly_equal_actual_tokens=True,
        )
        save(progress_path, progress)
        self.state["arms"][key] = progress
        self.persist()
        return actor

    def claim_gpu(self):
        holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder, lease_token=f"mandate-composition-{uuid.uuid4().hex}",
            evidence_directory=self.output / f"gpu_attempt{self.attempt}",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(self.output / f"gpu_config_attempt{self.attempt}.json", self.config)
        status = gpu.controller_claim(self.config)
        self.claimed = True
        gpu.request(str(holder), dict(
            op="release", lease_token=self.config["lease_token"], bytes=status["maximum_bytes"],
            request_id=uuid.uuid4().hex,
        ), timeout=60)
        released = gpu.reservation_status(self.config)
        if not released["paused"] or released["payload_bytes"]:
            raise RuntimeError("holder did not release for real worker")

    def wait_ready(self, path):
        deadline = time.monotonic() + 3600
        while not path.exists():
            self.persist(current_stage=f"WAITING_{path.parent.name.upper()}_{path.stem.upper()}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"required CPU artifact not ready: {path}")
            time.sleep(2)
        value = read(path)
        if value.get("status") != "COMPLETE":
            raise RuntimeError(f"required CPU artifact incomplete: {path}")

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.persist(current_step=1, plan="docs/2026-10-29_M3_COMPOSITION_PLAN.md",
                         initial_adapter=str(INITIAL_ADAPTER))
            self.wait_ready(self.output / "data/manifest.json")
            self.claim_gpu()
            self.evaluate("original", INITIAL_ADAPTER)
            self.initial_probe()
            self.report()
            self.wait_ready(self.output / "teacher/summary.json")
            self.persist(current_step=2, step1_status="COMPLETE")
            common = self.output / "common_sft"
            sft_resume = None
            if (common / "training_state.pt").exists() and not (common / "summary.json").exists():
                retained = (
                    self.output / "failed_attempts" / f"common_sft_resume_{uuid.uuid4().hex[:8]}"
                )
                retained.parent.mkdir(exist_ok=True)
                common.rename(retained)
                sft_resume = retained / "training_state.pt"
            if self.fresh(common):
                args = [
                    "sft", "--data", self.output / "teacher/train_requests.jsonl",
                    "--model", MODEL, "--adapter", INITIAL_ADAPTER,
                    "--output", common, "--seed", 43,
                ]
                if sft_resume:
                    args += ["--resume-state", sft_resume]
                self.stage("common_sft", TRAINER, args)
            common_actor = common / "adapter"
            endpoints = {"original": INITIAL_ADAPTER, "sft": common_actor}
            self.evaluate("sft", common_actor)
            endpoints["rloo_seed42"] = self.train_arm("rloo", 42, common_actor)
            self.evaluate("rloo_seed42", endpoints["rloo_seed42"])
            self.persist(current_step=3, step2_status="COMPLETE")
            self.report()
            signal_present = self.state["arms"]["rloo_seed42"]["completed_optimizer_steps"] > 0
            if signal_present:
                self.persist(step3_mode="THREE_ARM_COMPARISON")
                for arm in ("sham", "m3"):
                    key = f"{arm}_seed42"
                    endpoints[key] = self.train_arm(arm, 42, common_actor)
                    self.evaluate(key, endpoints[key])
                    self.report()
                allocation = second_seed_trigger({
                    key: rows(self.output / "eval" / f"{key}_dev/trajectory_results.jsonl")
                    for key in ("rloo_seed42", "sham_seed42", "m3_seed42")
                })
                self.persist(seed43_decision=allocation)
                if allocation["status"] == "TRIGGERED":
                    for arm in ("rloo", "sham", "m3"):
                        key = f"{arm}_seed43"
                        endpoints[key] = self.train_arm(arm, 43, common_actor)
                        self.evaluate(key, endpoints[key])
                        self.report()
            else:
                self.persist(
                    step3_mode="FROZEN_SFT_BASELINE_ONLY",
                    seed43_decision=dict(status="SKIPPED", trigger="NO_RL_SIGNAL"),
                )
            from src.agents.mandate_repair_train import adapter_digest

            if not (self.output / "frozen_models.json").exists():
                save(self.output / "frozen_models.json", dict(
                    status="COMPLETE", frozen_at=now(),
                    models={key: dict(
                        base_model=str(MODEL), adapter=str(path),
                        adapter_sha256=adapter_digest(path),
                    ) for key, path in endpoints.items()},
                    no_endpoint_selected_by_dev_score=True,
                    seed43_allocation=self.state["seed43_decision"],
                    reserved_test_used_for_training_or_allocation=False,
                ))
            # CPU frozen-test references run alongside actual GPU endpoints.
            baseline_log = self.output / "logs/frozen_test_baselines.log"
            with baseline_log.open("w") as stream:
                baseline = subprocess.Popen([
                    str(ROOT / ".venv/bin/python"), "-m",
                    "src.agents.mandate_composition_baselines", "--output", str(self.output),
                    "--part", "frozen-test-baselines",
                ], cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT)
                for key, adapter in endpoints.items():
                    self.evaluate(key, adapter, split="test")
                    self.report()
                if baseline.wait(timeout=1800):
                    raise RuntimeError(f"frozen-test baseline failed: {baseline_log}")
            total = sum(p["cumulative_tokens"] for p in self.state["arms"].values())
            if total > 18_000_000:
                raise RuntimeError("18M total RL generation cap exceeded")
            save(self.output / "decision.json", dict(
                status="COMPLETE", decision="REVIEW_FROZEN_ENDPOINT_RESULTS",
                step3_mode=self.state["step3_mode"],
                seed43_decision=self.state["seed43_decision"],
                actual_rl_generation_tokens=total,
                training_stopped=True, endpoints=list(endpoints), finished_at=now(),
            ))
            self.persist(status="COMPLETE", step3_status="COMPLETE", current_stage=None,
                         finished_at=now())
        except BaseException as error:
            code = 1
            self.persist(status="FAILED", error=f"{type(error).__name__}: {error}")
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
    args = parser.parse_args()
    raise SystemExit(Controller(args.output, args.resume).run())


if __name__ == "__main__":
    main()
