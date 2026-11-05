#!/usr/bin/env python3
"""Run the approved financial-evidence experiment in a persistent GPU service."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import subprocess
import uuid
from pathlib import Path

from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import (
    BUDGET,
    COLLECTOR,
    MODEL,
    ROOT,
    TRAINER,
    gpu,
    now,
    read,
    save,
)

DEFAULT = ROOT / "runs/m3-financial-evidence-20261030-v1"
INITIAL_ADAPTER = ROOT / "runs/m3-mandate-composition-20261029-v1/common_sft/adapter"
GROUP_RESERVATION = 4 * 10 * 16384
ENVIRONMENT = ["--environment", "financial_evidence", "--max-context-tokens", 16384]


def seed_allocation(evaluations):
    indexed = {
        key: {row["task_id"]: row for row in evaluations[key]}
        for key in ("rloo_seed42", "sham_seed42", "m3_seed42")
    }
    ids = set(indexed["m3_seed42"])
    if len(ids) != 100 or any(set(value) != ids for value in indexed.values()):
        raise RuntimeError("all three fixed 100-task Dev endpoints required")
    successes = {
        key: {task for task, row in values.items() if row.get("success") is True}
        for key, values in indexed.items()
    }
    common = set.intersection(*successes.values())
    rates = {key: len(values) / 100 for key, values in successes.items()}
    calls = {
        key: math.fsum(value[task]["calls_used"] for task in common) / len(common)
        if common else None
        for key, value in indexed.items()
    }
    comparisons = {
        control: dict(
            success_gain=rates["m3_seed42"] - rates[control],
            common_success_call_reduction=1 - calls["m3_seed42"] / calls[control]
            if common and calls[control] else None,
        )
        for control in ("rloo_seed42", "sham_seed42")
    }
    quality = all(row["success_gain"] >= 0.05 - 1e-12 for row in comparisons.values())
    cost = bool(common) and all(
        row["success_gain"] >= -1e-12
        and row["common_success_call_reduction"] >= 0.20 - 1e-12
        for row in comparisons.values()
    )
    return dict(
        status="TRIGGERED" if quality or cost else "SKIPPED",
        trigger="SUCCESS_GAIN_GE_5PP" if quality else (
            "COMMON_SUCCESS_CALL_REDUCTION_GE_20PCT" if cost else "NO_THRESHOLD_GAIN"
        ),
        comparisons=comparisons,
        common_success_tasks=len(common),
        success_rates=rates,
        common_success_mean_calls=calls,
        uses_reserved_test=False,
        statistical_significance_claim=False,
    )


class Controller(CompositionController):
    def __init__(
        self, output, resume, *, inherited_common=None, arm_budget=BUDGET, prior_rl_tokens=0
    ):
        if arm_budget < GROUP_RESERVATION or prior_rl_tokens < 0:
            raise ValueError("positive usable arm allowance and nonnegative prior RL cost required")
        if 6 * arm_budget + prior_rl_tokens > 18_000_000:
            raise ValueError("six new arm ceilings plus prior RL cost exceed absolute 18M cap")
        super().__init__(output, resume)
        self.inherited_common = inherited_common.resolve() if inherited_common else None
        self.arm_budget = arm_budget
        self.auxiliary_cap = arm_budget // 3
        self.prior_rl_tokens = prior_rl_tokens
        for name, value in (
            ("arm_token_budget", arm_budget), ("prior_rl_generation_tokens", prior_rl_tokens),
            ("inherited_common_adapter", str(self.inherited_common) if inherited_common else None),
        ):
            if name in self.state and self.state[name] != value:
                raise RuntimeError(f"resume cannot change {name}")
            self.state[name] = value
        self.persist()

    def check_absolute_rl_budget(self):
        actual = sum(value["cumulative_tokens"] for value in self.state["arms"].values())
        if actual + self.prior_rl_tokens > 18_000_000:
            raise RuntimeError("new plus prior RL tokens exceeded absolute 18M cap")

    def report(self):
        script = ROOT / "scripts/diagnostics/read_financial_evidence.py"
        if not script.exists():
            return
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(script), "--output", str(self.output)],
            cwd=ROOT, env=self.environment, capture_output=True, text=True,
        )
        (self.output / "readout.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            self.persist(readout_error=result.stderr[-3000:])

    def evaluate(self, key, adapter, split="dev", diagnostic="standard", metadata=None):
        if split == "test":
            receipt = self.output / "frozen_models.json"
            if not receipt.exists() or read(receipt).get("status") != "COMPLETE":
                raise RuntimeError("reserved test opens only after all endpoints freeze")
        destination = self.output / "eval" / f"{key}_{split}"
        data = self.output / "data" / ("development.jsonl" if split == "dev" else "test.jsonl")
        if self.fresh(destination):
            self.stage(f"eval_{key}_{split}", COLLECTOR, [
                "--data", data, "--base-model", MODEL, "--adapter", adapter,
                "--output", destination, "--seed", 43, "--temperature", 0,
                "--repetitions", 1, "--engine-mode", "eager",
                *ENVIRONMENT, "--diagnostic-mode", diagnostic,
            ])
        summary = read(destination / "summary.json")
        if summary["planned_trajectories"] != 100:
            raise RuntimeError("100-task fixed evaluation denominator changed")
        summary.update(split=split, policy=key, diagnostic_mode=diagnostic, **(metadata or {}))
        save(destination / "summary.json", summary)
        return summary

    def initial_probe(self, adapter):
        destination = self.output / "probe"
        started = destination / "collection_started.json"
        if started.exists():
            previous = read(started)
            if previous.get("adapter_path") != str(adapter.resolve()):
                retained = self.output / "failed_attempts" / f"raw_probe_{uuid.uuid4().hex[:8]}"
                retained.parent.mkdir(exist_ok=True)
                destination.rename(retained)
                self.persist(preserved_raw_probe=str(retained))
        selected = self.output / "probe_tasks.jsonl"
        if not selected.exists():
            tasks = rows(self.output / "data/train.jsonl")
            random.Random(42).shuffle(tasks)
            selected.write_text("".join(json.dumps(t) + "\n" for t in tasks[:16]))
        if self.fresh(destination):
            self.stage("initial_training_probe", COLLECTOR, [
                "--data", selected, "--base-model", MODEL, "--adapter", adapter,
                "--output", destination, "--seed", 42, "--temperature", 1,
                "--repetitions", 4, "--engine-mode", "eager", *ENVIRONMENT,
            ])
        self.persist(
            initial_probe_complete=True, initial_probe_not_actor_training_data=True,
            initial_probe_adapter=str(adapter.resolve()),
        )

    def sft(self, key, demonstrations, initial):
        destination = self.output / key
        resume = None
        if (destination / "training_state.pt").exists() and not (
            destination / "summary.json"
        ).exists():
            retained = self.output / "failed_attempts" / f"{key}_resume_{uuid.uuid4().hex[:8]}"
            retained.parent.mkdir(exist_ok=True)
            destination.rename(retained)
            resume = retained / "training_state.pt"
        if self.fresh(destination):
            args = [
                "sft", "--data", demonstrations, "--model", MODEL,
                "--adapter", initial, "--output", destination, "--seed", 43,
                "--max-length", 16384, "--auto-request-checkpointing",
            ]
            if resume:
                args += ["--resume-state", resume]
            self.stage(key, TRAINER, args)
        return destination / "adapter"

    def _new_progress(self, arm, seed, common):
        return dict(
            arm=arm, seed=seed, status="RUNNING", token_budget=self.arm_budget,
            auxiliary_token_cap=self.auxiliary_cap if arm != "rloo" else 0,
            cumulative_tokens=0, parent_tokens=0, auxiliary_tokens=0, rounds=[],
            final_adapter=str(common), completed_optimizer_steps=0,
            local_credit_target="business" if arm != "rloo" else None,
            environment="financial_evidence", maximum_tool_calls=10,
            rollout_groups_per_round=4, zero_advantage_check_groups=16,
            actor_task_groups=0,
        )

    def train_arm(self, arm, seed, common):
        key = f"{arm}_seed{seed}"
        directory = self.output / "training" / key
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        progress = read(progress_path) if progress_path.exists() else self._new_progress(
            arm, seed, common
        )
        actor = Path(progress["final_adapter"])
        if progress["status"] == "COMPLETE":
            return actor
        if progress["status"] == "SKIPPED_AFTER_PROBE":
            raise RuntimeError("cannot silently train an arm already closed after its probe")
        tasks = rows(self.output / "data/train.jsonl")
        random.Random(seed).shuffle(tasks)
        round_index = len(progress["rounds"])
        while self.arm_budget - progress["cumulative_tokens"] >= GROUP_RESERVATION:
            rd = directory / f"round_{round_index:03d}"
            rd.mkdir(parents=True, exist_ok=True)
            selected = rd / "selected.jsonl"
            if not selected.exists():
                offset = round_index * 4
                selected.write_text("".join(
                    json.dumps(tasks[(offset + i) % len(tasks)]) + "\n" for i in range(4)
                ))
            collection = rd / "collect"
            remaining = self.arm_budget - progress["cumulative_tokens"]
            phase = min(3, progress["cumulative_tokens"] // (self.arm_budget // 4))
            allowance = max(
                0, ((phase + 1) * self.auxiliary_cap // 4) - progress["auxiliary_tokens"]
            ) if arm != "rloo" else 0
            if self.fresh(collection):
                args = [
                    "--data", selected, "--base-model", MODEL, "--adapter", actor,
                    "--output", collection, "--seed", seed, "--temperature", 1,
                    "--repetitions", 4, "--token-budget", remaining,
                    "--max-groups", 4, "--engine-mode", "eager", *ENVIRONMENT,
                ]
                if arm != "rloo":
                    args += ["--arm", arm, "--aux-token-budget", allowance,
                             "--local-credit-target", "business"]
                self.stage(f"{key}_round{round_index:03d}_collect", COLLECTOR, args)
            collected = read(collection / "summary.json")
            spent = collected["total_tokens"]
            if spent <= 0:
                progress["budget_stop_reason"] = "NO_FULL_TASK_GROUP_ADMITTED"
                break
            destination = rd / "train"
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
            progress["actor_task_groups"] += collected["admitted_task_groups"]
            progress["completed_optimizer_steps"] = trained["completed_optimizer_steps"]
            actor = destination / "adapter"
            progress["final_adapter"] = str(actor)
            progress["rounds"].append(dict(
                round=round_index, collect=str(collection), train=str(destination),
                cumulative_tokens=progress["cumulative_tokens"], tokens=spent,
                auxiliary_tokens=aux, auxiliary_allowance=allowance,
                optimizer_updates=trained["optimizer_updates"],
                completed_optimizer_steps=trained["completed_optimizer_steps"],
                parent_successes=collected.get("successes"),
                parent_started_trajectories=collected.get("started_trajectories"),
                admitted_task_groups=collected["admitted_task_groups"],
            ))
            if (
                progress["cumulative_tokens"] > self.arm_budget
                or progress["auxiliary_tokens"] > self.auxiliary_cap
            ):
                raise RuntimeError("actual generation token cap exceeded")
            save(progress_path, progress)
            self.state["arms"][key] = progress
            self.check_absolute_rl_budget()
            self.persist()
            self.report()
            if (
                arm == "rloo" and progress["actor_task_groups"] >= 16
                and progress["completed_optimizer_steps"] == 0
            ):
                progress["budget_stop_reason"] = "FIRST_BATCH_NO_EFFECTIVE_ADVANTAGE"
                break
            round_index += 1
        progress.update(
            status="COMPLETE", finished_at=now(), final_adapter=str(actor),
            unused_budget_tokens=self.arm_budget - progress["cumulative_tokens"],
            endpoints_not_strictly_equal_actual_tokens=True,
        )
        save(progress_path, progress)
        self.state["arms"][key] = progress
        self.persist()
        return actor

    def information_probe(self, common):
        """Charge a bounded real intervention probe to sham, never teacher/RLOO."""
        directory = self.output / "training/sham_seed42"
        directory.mkdir(parents=True, exist_ok=True)
        progress_path = directory / "progress.json"
        if progress_path.exists():
            progress = read(progress_path)
            if progress.get("information_probe"):
                return progress["information_probe"]["business_difference_parents"] > 0
        else:
            progress = self._new_progress("sham", 42, common)
        destination = directory / "information_probe"
        selected = self.output / "probe_tasks.jsonl"
        if self.fresh(destination):
            self.stage("bounded_business_information_probe", COLLECTOR, [
                "--data", selected, "--base-model", MODEL, "--adapter", common,
                "--output", destination, "--seed", 42, "--temperature", 1,
                "--repetitions", 1, "--max-groups", 16,
                "--token-budget", self.arm_budget // 4,
                "--arm", "sham", "--aux-token-budget", self.auxiliary_cap // 4,
                "--local-credit-target", "business", "--engine-mode", "eager", *ENVIRONMENT,
                "--max-auxiliary-parents", 16,
            ])
        summary = read(destination / "summary.json")
        local = rows(destination / "local_credit.jsonl")
        differences = sum(
            row.get("status") == "COMPLETE"
            and abs(row.get("unscaled_local_difference", 0)) > 1e-12
            for row in local
        )
        aux = summary.get("auxiliary_tokens", summary.get("aux_tokens", 0))
        progress["cumulative_tokens"] += summary["total_tokens"]
        progress["parent_tokens"] += summary["total_tokens"] - aux
        progress["auxiliary_tokens"] += aux
        progress["information_probe"] = dict(
            collect=str(destination), tokens=summary["total_tokens"], auxiliary_tokens=aux,
            supported_parents=len(local), business_difference_parents=differences,
            maximum_parent_trajectories=16, actor_training_data=False,
        )
        if not differences:
            progress.update(status="SKIPPED_AFTER_PROBE", finished_at=now())
        save(progress_path, progress)
        self.state["arms"]["sham_seed42"] = progress
        self.check_absolute_rl_budget()
        self.persist(information_probe=progress["information_probe"])
        return differences > 0

    def claim_gpu(self):
        holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder, lease_token=f"financial-evidence-{uuid.uuid4().hex}",
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

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        cpu_baseline = None
        try:
            self.persist(
                current_step=1, plan="docs/2026-10-29_M3_FINANCIAL_EVIDENCE_PLAN.md",
                initial_adapter=str(INITIAL_ADAPTER), environment="financial_evidence",
                maximum_tool_calls=10, full_teacher_sft_is_rl_initialization=False,
            )
            self.wait_ready(self.output / "data/manifest.json")
            self.claim_gpu()
            self.evaluate("original", INITIAL_ADAPTER)
            self.wait_ready(self.output / "teacher/summary.json")
            self.persist(
                raw_migration_baseline_complete=True,
                protocol_adaptation_before_diagnostics=True,
            )
            if self.inherited_common is not None:
                from src.agents.mandate_repair_train import adapter_digest

                if not self.inherited_common.is_dir():
                    raise RuntimeError("inherited common adapter is missing")
                common = self.inherited_common
                save(self.output / "inherited_common.json", dict(
                    status="INHERITED", adapter=str(common), adapter_sha256=adapter_digest(common),
                    source_training_summary=str(common.parent / "summary.json"),
                    new_run_protocol_sft_updates=0,
                    previous_train_labels_were_visible_to_common=True,
                    all_new_rl_arms_share_exact_common=True,
                    old_rl_and_partial_teacher_models_used=False,
                ))
                self.persist(common_initialization="INHERITED_OLD_64_TRAIN_ADAPTATION")
            else:
                common = self.sft(
                    "common_adaptation", self.output / "teacher/adapter_requests.jsonl",
                    INITIAL_ADAPTER,
                )
            endpoints = {"original": INITIAL_ADAPTER, "common": common}
            self.evaluate("common", common)
            self.initial_probe(common)
            self.evaluate(
                "oracle_facts", common, diagnostic="oracle_facts",
                metadata=dict(same_model_reference="common"),
            )
            self.evaluate(
                "fixed_retrieval", common, diagnostic="fixed_retrieval",
                metadata=dict(same_model_reference="common"),
            )
            self.report()
            self.persist(current_step=2, step1_status="COMPLETE")
            endpoints["rloo_seed42"] = self.train_arm("rloo", 42, common)
            self.evaluate("rloo_seed42", endpoints["rloo_seed42"])
            endpoints["sft"] = self.sft(
                "teacher_sft", self.output / "teacher/train_requests.jsonl", common
            )
            self.evaluate("sft", endpoints["sft"])
            self.persist(current_step=3, step2_status="COMPLETE")
            self.report()
            rl_signal = self.state["arms"]["rloo_seed42"]["completed_optimizer_steps"] > 0
            information_signal = True if rl_signal else self.information_probe(common)
            if information_signal:
                self.persist(step3_mode="THREE_ARM_COMPARISON")
                for arm in ("sham", "m3"):
                    key = f"{arm}_seed42"
                    endpoints[key] = self.train_arm(arm, 42, common)
                    self.evaluate(key, endpoints[key])
                    self.report()
                allocation = seed_allocation({
                    key: rows(self.output / "eval" / f"{key}_dev/trajectory_results.jsonl")
                    for key in ("rloo_seed42", "sham_seed42", "m3_seed42")
                })
                self.persist(seed43_decision=allocation)
                if allocation["status"] == "TRIGGERED":
                    for arm in ("rloo", "sham", "m3"):
                        key = f"{arm}_seed43"
                        endpoints[key] = self.train_arm(arm, 43, common)
                        self.evaluate(key, endpoints[key])
                        self.report()
            else:
                self.persist(
                    step3_mode="FROZEN_BASELINE_ONLY_NO_BUSINESS_SIGNAL",
                    seed43_decision=dict(
                        status="SKIPPED", trigger="NO_BUSINESS_INFORMATION_SIGNAL"
                    ),
                )
            from src.agents.mandate_repair_train import adapter_digest

            frozen = self.output / "frozen_models.json"
            if not frozen.exists():
                save(frozen, dict(
                    status="COMPLETE", frozen_at=now(),
                    models={key: dict(
                        base_model=str(MODEL), adapter=str(path),
                        adapter_sha256=adapter_digest(path), diagnostic_mode="standard",
                    ) for key, path in endpoints.items()},
                    diagnostics={
                        key: dict(
                            role=role, same_model_reference="common", adapter=str(common),
                            adapter_sha256=adapter_digest(common),
                        )
                        for key, role in (
                            ("oracle_facts", "privileged"),
                            ("fixed_retrieval", "same_tools"),
                        )
                    },
                    no_endpoint_selected_by_dev_score=True,
                    seed43_allocation=self.state["seed43_decision"],
                    reserved_test_used_for_training_or_allocation=False,
                ))
            log = self.output / "logs/frozen_test_baselines.log"
            with log.open("w") as stream:
                cpu_baseline = subprocess.Popen([
                    str(ROOT / ".venv/bin/python"), "-m", "src.agents.financial_evidence_baselines",
                    "--output", str(self.output), "--part", "frozen-test-baselines",
                ], cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT)
                for key, adapter in endpoints.items():
                    self.evaluate(key, adapter, split="test")
                    self.report()
                # Diagnostics use the exact common policy with the adapted protocol.
                self.evaluate(
                    "oracle_facts", common, split="test", diagnostic="oracle_facts",
                    metadata=dict(same_model_reference="common"),
                )
                self.evaluate(
                    "fixed_retrieval", common, split="test", diagnostic="fixed_retrieval",
                    metadata=dict(same_model_reference="common"),
                )
                if cpu_baseline.wait(timeout=1800):
                    raise RuntimeError(f"frozen-test CPU baseline failed: {log}")
                cpu_baseline = None
            total = sum(value["cumulative_tokens"] for value in self.state["arms"].values())
            if total + self.prior_rl_tokens > 18_000_000:
                raise RuntimeError("18M actual RL generation cap exceeded")
            save(self.output / "decision.json", dict(
                status="COMPLETE", decision="REVIEW_FROZEN_ENDPOINT_RESULTS",
                step3_mode=self.state["step3_mode"], seed43_decision=self.state["seed43_decision"],
                actual_rl_generation_tokens=total, endpoints=list(endpoints),
                prior_rl_generation_tokens=self.prior_rl_tokens,
                absolute_rl_generation_tokens=total + self.prior_rl_tokens,
                arm_token_budget=self.arm_budget,
                full_teacher_sft_is_rl_initialization=False,
                strict_equal_actual_compute_claim=False, training_stopped=True,
                finished_at=now(),
            ))
            self.persist(
                status="COMPLETE", step3_status="COMPLETE", current_stage=None, finished_at=now()
            )
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
            if cpu_baseline is not None and cpu_baseline.poll() is None:
                cpu_baseline.terminate()
                cpu_baseline.wait(timeout=20)
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
    parser.add_argument("--inherited-common", type=Path)
    parser.add_argument("--arm-budget", type=int, default=BUDGET)
    parser.add_argument("--prior-rl-tokens", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(Controller(
        args.output, args.resume, inherited_common=args.inherited_common,
        arm_budget=args.arm_budget, prior_rl_tokens=args.prior_rl_tokens,
    ).run())


if __name__ == "__main__":
    main()
