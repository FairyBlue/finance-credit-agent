#!/usr/bin/env python3
"""Persistent, budgeted MC-state-baseline Agentic RL experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import subprocess
from pathlib import Path

from run_financial_evidence import ENVIRONMENT, GROUP_RESERVATION
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, MODEL, ROOT, TRAINER, gpu, now, read, save

DEFAULT = ROOT / "runs/agentic-rl-mc-20261030-v1"
SOURCE = ROOT / "runs/m3-financial-credit-20261030-v1"
INITIAL = ROOT / "runs/m3-financial-evidence-20261030-v2/teacher_sft/adapter"
SAMPLER = "src.agents.financial_mc_probe"
ARMS = ("rloo", "mc_uniform", "mc_priority")
WIRING_CAP = 2_000_000
ARM_CAP = 3_000_000
AUX_CAP = ARM_CAP // 3
EVAL_CAP = 32_000_000
TOTAL_CAP = 49_000_000
PAIR_RESERVATION = 2 * 10 * 16384


class BoundedEvaluationStop(RuntimeError):
    """An evaluation was not completed inside the unchanged total allowance."""


def generation_tokens(directory, kind=None):
    """Count completed and failed requests, without trusting lagging progress."""
    if not directory.exists():
        return 0
    directories = {path.parent for path in directory.rglob("requests.jsonl")}
    total = 0
    for leaf in directories:
        if kind is not None:
            auxiliary = any(
                part == "mc" or part.startswith("mc_") for part in leaf.relative_to(directory).parts
            )
            if auxiliary != (kind == "auxiliary"):
                continue
        raw = 0
        with (leaf / "requests.jsonl").open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        break  # An external readout may race the final append.
                    raise
                if item.get("model_request_submitted") is False:
                    continue
                raw += int(item.get("prompt_tokens", 0) or 0)
                raw += int(item.get("response_tokens", 0) or 0)
        summary = read(leaf / "summary.json") if (leaf / "summary.json").exists() else {}
        progress = read(leaf / "progress.json") if (leaf / "progress.json").exists() else {}
        total += max(
            raw,
            summary.get("total_tokens", 0),
            progress.get("total_tokens", 0),
            summary.get("budget_charge_tokens", 0),
            progress.get("budget_charge_tokens", 0),
        )
    return total


def dev_gate(values):
    """The prespecified engineering gate, not a statistical significance test."""
    for key in ("sft", *[arm + "_seed42" for arm in ARMS]):
        item = values[key]
        if item.get("planned_trajectories") != 100 or item.get("complete_trajectories") != 100:
            raise ValueError("gate requires all fixed 100-task Dev trajectories")
    candidate = values["mc_priority_seed42"]["successes"]
    controls = {arm: values[arm + "_seed42"]["successes"] for arm in ARMS[:2]}
    # A tie selects ordinary RLOO deterministically, not an observed-cost tie-break.
    strongest = max(ARMS[:2], key=lambda arm: (controls[arm], arm == "rloo"))
    passed = candidate >= values["sft"]["successes"] and all(
        candidate >= score + 5 for score in controls.values()
    )
    return dict(
        status="PASS" if passed else "NO_CLEAR_LEARNING_GAIN",
        candidate_successes=candidate,
        sft_successes=values["sft"]["successes"],
        control_successes=controls,
        strongest_control=strongest,
        required_success_margin=5,
        statistical_significance_claim=False,
        uses_reserved_test=False,
    )


class Controller(CompositionController):
    def __init__(self, output, resume=False, *, step1_only=False):
        super().__init__(output, resume)
        self.step1_only = step1_only
        self.persist(
            experiment="financial_mc_state_baseline_allocation_v1",
            initial_adapter=str(INITIAL),
            data_source=str(SOURCE / "data"),
            parent_data_reuse_source=str(SOURCE / "probe/parents"),
            wiring_token_cap=WIRING_CAP,
            arm_token_cap=ARM_CAP,
            auxiliary_token_cap=AUX_CAP,
            evaluation_token_cap=EVAL_CAP,
            total_new_generation_token_cap=TOTAL_CAP,
            candidate="prefix_only_fixed_priority_with_uniform_mix",
            full_vineppo_reproduction=False,
            explicit_action_q_estimator=False,
            step1_only=step1_only,
        )

    def data(self, name):
        local = self.output / "data" / name
        return local if local.exists() else SOURCE / "data" / name

    def costs(self):
        wiring = generation_tokens(self.output / "wiring")
        training = generation_tokens(self.output / "training")
        evaluation = generation_tokens(self.output / "eval")
        if wiring > WIRING_CAP:
            raise RuntimeError("wiring actual generation exceeded 2M")
        if evaluation > EVAL_CAP:
            raise RuntimeError("evaluation actual generation exceeded 32M")
        if wiring + training + evaluation > TOTAL_CAP:
            raise RuntimeError("all actual generation exceeded 49M")
        return dict(
            wiring_generation_tokens=wiring,
            training_generation_tokens=training,
            evaluation_generation_tokens=evaluation,
            total_new_generation_tokens=wiring + training + evaluation,
            cost_includes_failed_request_logs=True,
            reused_parent_exposure_tokens=3_308_883,
            reused_parent_new_generation_tokens=0,
        )

    def report(self):
        script = ROOT / "scripts/diagnostics/read_financial_mc.py"
        if not script.exists():
            return
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(script), "--output", str(self.output)],
            cwd=ROOT,
            env=self.environment,
            capture_output=True,
            text=True,
        )
        (self.output / "readout.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            self.persist(readout_error=result.stderr[-2000:])

    def checkpoint_progress(self, key, path, progress):
        save(path, progress)
        self.state["arms"][key] = progress
        self.persist(costs=self.costs())
        self.report()

    def mc(self, name, data, parents, selections, actor, output, allowance, *, max_anchors, seed):
        newly_run = self.fresh(output)
        if newly_run:
            # fresh() preserves partial attempts, and those raw tokens still count.
            owner = output.parent
            spent = generation_tokens(owner)
            owner_cap = WIRING_CAP if owner.name == "wiring" else None
            if owner_cap is not None:
                allowance = min(allowance, max(0, owner_cap - spent))
            self.stage(
                name,
                SAMPLER,
                [
                    "--data",
                    data,
                    "--parents",
                    parents,
                    "--selections",
                    selections,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    output,
                    "--seed",
                    seed,
                    "--token-budget",
                    max(0, allowance),
                    "--repetitions",
                    2,
                    "--max-anchors",
                    max_anchors,
                ],
            )
        summary = read(output / "summary.json")
        if (
            newly_run
            and max(summary.get("total_tokens", 0), summary.get("budget_charge_tokens", 0))
            > allowance
        ):
            raise RuntimeError("MC stage exceeded remaining generation allowance")
        return output / "state_value_baselines.jsonl", summary

    def train(
        self, name, collection, baseline, actor, output, arm, seed, previous=None, updates=None
    ):
        if self.fresh(output):
            args = [
                "rl",
                "--data",
                collection,
                "--base-model",
                MODEL,
                "--adapter",
                actor,
                "--output",
                output,
                "--seed",
                seed,
                "--arm",
                arm,
            ]
            if arm != "rloo":
                args += ["--mc-baselines", baseline]
            if previous:
                args += ["--resume-state", previous]
            if updates is not None:
                args += ["--updates", updates]
            self.stage(name, TRAINER, args)
        return read(output / "summary.json")

    def wiring(self):
        self.persist(current_step=1, step1_status="RUNNING")
        directory = self.output / "wiring"
        directory.mkdir(exist_ok=True)
        manifest = directory / "prefix_selections.jsonl"
        if not manifest.exists():
            raise RuntimeError(f"prefix-only wiring selection manifest not ready: {manifest}")
        parents = SOURCE / "probe/parents"
        identity = read(parents / "summary.json")
        if identity.get("adapter_path") != str(INITIAL) or identity.get("temperature") != 1:
            raise RuntimeError("wiring parents do not match the fixed current-policy SFT")
        baseline, measured = self.mc(
            "wiring_independent_mc",
            self.data("probe32.jsonl"),
            parents,
            manifest,
            INITIAL,
            directory / "mc",
            WIRING_CAP - generation_tokens(directory),
            max_anchors=32,
            seed=42,
        )
        if measured.get("completed_baselines", 0) <= 0:
            raise RuntimeError("no complete independent MC pair; cannot claim training integration")
        first = self.train(
            "wiring_actor_update",
            parents,
            baseline,
            INITIAL,
            directory / "train",
            "mc_priority",
            42,
            updates=4,
        )
        if first.get("completed_optimizer_steps", 0) <= 0:
            raise RuntimeError("wiring batch produced no effective actor update")
        restored = self.train(
            "wiring_saved_state_resume",
            parents,
            baseline,
            INITIAL,
            directory / "train_resumed",
            "mc_priority",
            42,
            previous=directory / "train/training_state.pt",
            updates=8,
        )
        before, after = first["completed_optimizer_steps"], restored["completed_optimizer_steps"]
        if after < before:
            raise RuntimeError("restored optimizer count regressed")
        receipt = dict(
            status="COMPLETE",
            completed_at=now(),
            completed_baselines=measured["completed_baselines"],
            optimizer_steps_before_resume=before,
            optimizer_steps_after_resume=after,
            resumed_optimizer_updates=restored.get("optimizer_updates_this_invocation", 0),
            state_resume_completed=True,
            step2_initialization=str(INITIAL),
            wiring_endpoint_used_for_step2=False,
            reused_parent_sampling_tokens=identity["total_tokens"],
            new_generation_tokens=generation_tokens(directory),
            full_parent_batch_retained=True,
        )
        save(directory / "receipt.json", receipt)
        self.persist(step1_status="COMPLETE", wiring=receipt, costs=self.costs())
        self.report()

    def train_arm(self, arm, seed):
        key = f"{arm}_seed{seed}"
        directory = self.output / "training" / key
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "progress.json"
        progress = (
            read(path)
            if path.exists()
            else dict(
                arm=arm,
                seed=seed,
                status="RUNNING",
                rounds=[],
                token_budget=ARM_CAP,
                auxiliary_token_cap=AUX_CAP if arm != "rloo" else 0,
                final_adapter=str(INITIAL),
                completed_optimizer_steps=0,
                cumulative_tokens=0,
                parent_tokens=0,
                auxiliary_tokens=0,
                step1_endpoint_used=False,
            )
        )
        if progress["status"] == "COMPLETE":
            return Path(progress["final_adapter"])
        tasks = rows(self.data("train.jsonl"))
        random.Random(seed).shuffle(tasks)
        actor = Path(progress["final_adapter"])
        index = len(progress["rounds"])
        while ARM_CAP - generation_tokens(directory) >= GROUP_RESERVATION:
            rd = directory / f"round_{index:03d}"
            rd.mkdir(exist_ok=True)
            selected = rd / "selected.jsonl"
            if not selected.exists():
                selected.write_text(
                    "".join(
                        json.dumps(tasks[(index * 4 + i) % len(tasks)], ensure_ascii=False) + "\n"
                        for i in range(4)
                    )
                )
            remaining = ARM_CAP - generation_tokens(directory)
            aux_spent = generation_tokens(directory, kind="auxiliary")
            # Reserve complete pairs now, rather than starving early rounds.
            allowance = (
                min(
                    max(0, AUX_CAP - aux_spent),
                    max(0, remaining - GROUP_RESERVATION),
                    2 * PAIR_RESERVATION,
                )
                if arm != "rloo"
                else 0
            )
            collection = rd / "collect"
            if self.fresh(collection):
                remaining = ARM_CAP - generation_tokens(directory)
                allowance = min(allowance, max(0, remaining - GROUP_RESERVATION))
                arguments = [
                    "--data",
                    selected,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    collection,
                    "--seed",
                    seed,
                    "--temperature",
                    1,
                    "--repetitions",
                    4,
                    "--token-budget",
                    remaining - allowance,
                    "--max-groups",
                    4,
                    "--engine-mode",
                    "eager",
                    *ENVIRONMENT,
                    "--arm",
                    "rloo",
                ]
                if arm != "rloo":
                    arguments += [
                        "--mc-selection",
                        "uniform" if arm == "mc_uniform" else "priority",
                        "--mc-selection-seed",
                        seed + index * 1009,
                        "--mc-max-anchors",
                        2,
                        "--mc-selection-budget",
                        allowance,
                    ]
                self.stage(key + f"_round{index:03d}_collect", COLLECTOR, arguments)
            collected = read(collection / "summary.json")
            if collected.get("admitted_task_groups", 0) <= 0:
                progress["budget_stop_reason"] = "NO_COMPLETE_PARENT_GROUP_ADMITTED"
                break
            baseline = None
            measured = {}
            if arm != "rloo":
                remaining = ARM_CAP - generation_tokens(directory)
                auxiliary_left = max(0, AUX_CAP - generation_tokens(directory, kind="auxiliary"))
                baseline, measured = self.mc(
                    key + f"_round{index:03d}_mc",
                    selected,
                    collection,
                    collection / "prefix_selections.jsonl",
                    actor,
                    rd / "mc",
                    min(allowance, remaining, auxiliary_left),
                    max_anchors=2,
                    seed=seed + index * 1009,
                )
            previous = (
                Path(progress["rounds"][-1]["train"]) / "training_state.pt"
                if progress["rounds"]
                else None
            )
            trained = self.train(
                key + f"_round{index:03d}_train",
                collection,
                baseline,
                actor,
                rd / "train",
                arm,
                seed,
                previous=previous,
            )
            actor = rd / "train/adapter"
            progress["rounds"].append(
                dict(
                    round=index,
                    collect=str(collection),
                    mc=str(rd / "mc") if baseline else None,
                    train=str(rd / "train"),
                    parent_tokens=collected["total_tokens"],
                    auxiliary_tokens=measured.get("total_tokens", 0),
                    completed_baselines=measured.get("completed_baselines", 0),
                    optimizer_updates=trained["optimizer_updates"],
                    completed_optimizer_steps=trained["completed_optimizer_steps"],
                    parent_successes=collected["successes"],
                )
            )
            progress.update(
                final_adapter=str(actor),
                completed_optimizer_steps=trained["completed_optimizer_steps"],
                cumulative_tokens=generation_tokens(directory),
                parent_tokens=generation_tokens(directory, kind="parent"),
                auxiliary_tokens=generation_tokens(directory, kind="auxiliary"),
            )
            if progress["cumulative_tokens"] > ARM_CAP or progress["auxiliary_tokens"] > AUX_CAP:
                raise RuntimeError(f"{key} actual generation exceeded its allowance")
            self.checkpoint_progress(key, path, progress)
            index += 1
        progress.update(
            status="COMPLETE",
            finished_at=now(),
            final_adapter=str(actor),
            cumulative_tokens=generation_tokens(directory),
            unused_budget_tokens=ARM_CAP - generation_tokens(directory),
            equal_budget_ceiling_not_equal_actual_compute=True,
        )
        self.checkpoint_progress(key, path, progress)
        return actor

    def reuse_sft_dev(self, destination):
        source = ROOT / "runs/m3-financial-evidence-20261030-v2/eval/sft_dev"
        previous = read(source / "summary.json")
        expected = dict(
            adapter_path=str(INITIAL),
            environment="financial_evidence",
            temperature=0.0,
            seed=43,
            repetitions=1,
            decode_mode="native",
            diagnostic_mode="standard",
            observation_profile="financial_evidence_conditional_mandate_v2_opaque_public_ids",
            data_sha256=hashlib.sha256(self.data("development.jsonl").read_bytes()).hexdigest(),
            max_context_tokens=16384,
        )
        if previous.get("status") != "COMPLETE" or any(
            previous.get(key) != value for key, value in expected.items()
        ):
            return None
        from src.agents.mandate_repair_train import adapter_digest

        if previous.get("adapter_sha256") != adapter_digest(INITIAL):
            return None
        destination.mkdir(parents=True, exist_ok=True)
        copied = {key: value for key, value in previous.items() if key not in {"results", "groups"}}
        copied.update(
            evaluation_reused=True,
            evaluation_source=str(source),
            total_tokens=0,
            model_requests=0,
            source_sampled_tokens=previous["total_tokens"],
            policy="sft",
            split="dev",
            newly_sampled_trajectories=0,
        )
        save(destination / "summary.json", copied)
        (destination / "trajectory_results.jsonl").write_bytes(
            (source / "trajectory_results.jsonl").read_bytes()
        )
        return copied

    def evaluate(self, key, actor, split="dev"):
        if split == "test":
            frozen = self.output / "frozen_models.json"
            if not frozen.exists() or read(frozen).get("status") != "COMPLETE":
                raise RuntimeError("new Test cannot open before fixed endpoints freeze")
        destination = self.output / "eval" / f"{key}_{split}"
        if (
            (destination / "summary.json").exists()
            or key == "sft"
            and split == "dev"
            and self.reuse_sft_dev(destination) is not None
        ):
            summary = read(destination / "summary.json")
        else:
            self.fresh(destination)
            remaining = EVAL_CAP - generation_tokens(self.output / "eval")
            if remaining < 10 * 16384:
                raise BoundedEvaluationStop("no full evaluation task fits the remaining allowance")
            self.stage(
                f"eval_{key}_{split}",
                COLLECTOR,
                [
                    "--data",
                    self.data("development.jsonl" if split == "dev" else "test.jsonl"),
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    destination,
                    "--seed",
                    43,
                    "--temperature",
                    0,
                    "--repetitions",
                    1,
                    "--engine-mode",
                    "eager",
                    "--token-budget",
                    remaining,
                    *ENVIRONMENT,
                ],
            )
            summary = read(destination / "summary.json")
        summary.update(policy=key, split=split)
        save(destination / "summary.json", summary)
        self.persist(costs=self.costs())
        self.report()
        if (
            summary.get("planned_trajectories") != 100
            or summary.get("complete_trajectories") != 100
        ):
            raise BoundedEvaluationStop(
                f"{key}/{split} did not complete its fixed 100-task denominator"
            )
        return summary

    def comparison(self):
        self.persist(current_step=2, step2_status="RUNNING", step3_status="PENDING")
        values = {"sft": self.evaluate("sft", INITIAL)}
        endpoints = {}
        for arm in ARMS:
            key = arm + "_seed42"
            endpoints[key] = self.train_arm(arm, 42)
            values[key] = self.evaluate(key, endpoints[key])
        gate = dev_gate(values)
        save(self.output / "dev_gate.json", gate)
        self.persist(current_step=3, step2_status="COMPLETE", step3_status="RUNNING", dev_gate=gate)
        if gate["status"] != "PASS":
            self.finish_decision("NO_CLEAR_SEED42_LEARNING_GAIN", fresh_test_opened=False)
            return
        control = gate["strongest_control"]
        confirmation = {}
        confirmation_endpoints = {}
        for arm in (control, "mc_priority"):
            key = arm + "_seed43"
            confirmation_endpoints[key] = self.train_arm(arm, 43)
            confirmation[key] = self.evaluate(key, confirmation_endpoints[key])
        candidate = confirmation["mc_priority_seed43"]["successes"]
        reference = confirmation[control + "_seed43"]["successes"]
        confirmed = candidate > reference and candidate >= values["sft"]["successes"]
        save(
            self.output / "confirmation.json",
            dict(
                status="PASS" if confirmed else "DIRECTION_NOT_CONFIRMED",
                strongest_control=control,
                candidate_successes=candidate,
                control_successes=reference,
                sft_successes=values["sft"]["successes"],
                statistical_significance_claim=False,
            ),
        )
        if not confirmed:
            self.finish_decision("SEED43_DIRECTION_NOT_CONFIRMED", fresh_test_opened=False)
            return
        from src.agents.mandate_repair_train import adapter_digest

        frozen_models = {"sft": INITIAL, **confirmation_endpoints}
        save(
            self.output / "frozen_models.json",
            dict(
                status="COMPLETE",
                frozen_at=now(),
                models={
                    key: dict(
                        adapter=str(actor),
                        adapter_sha256=adapter_digest(actor),
                        base_model=str(MODEL),
                    )
                    for key, actor in frozen_models.items()
                },
                selection="PREDECLARED_SEED43_FIXED_ENDPOINTS_AND_COMMON_SFT",
                test_used_for_selection=False,
            ),
        )
        for key, actor in frozen_models.items():
            self.evaluate(key, actor, split="test")
        self.finish_decision("CONFIRMED_DEV_DIRECTION_TEST_REPORTED", fresh_test_opened=True)

    def finish_decision(self, reason, **details):
        decision = dict(
            status="COMPLETE",
            decision=reason,
            finished_at=now(),
            **details,
            costs=self.costs(),
            credit_method_claim_supported=False,
            full_vineppo_reproduction=False,
            strict_equal_actual_compute_claim=False,
            research_brief_review_required=True,
        )
        save(self.output / "decision.json", decision)
        self.persist(step3_status="COMPLETE", decision=decision)
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
            self.claim_gpu()
            self.wiring()
            if self.step1_only:
                self.persist(status="STEP1_COMPLETE", current_stage=None, finished_at=now())
            else:
                self.comparison()
                self.persist(status="COMPLETE", current_stage=None, finished_at=now())
        except BoundedEvaluationStop as error:
            self.finish_decision(
                "INCOMPLETE_WITHIN_FIXED_EVALUATION_ALLOWANCE",
                incomplete_evaluation=str(error),
                fresh_test_opened=(self.output / "frozen_models.json").exists(),
            )
            self.persist(status="COMPLETE_LIMITED", current_stage=None, finished_at=now())
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
    parser.add_argument("--step1-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(Controller(args.output, args.resume, step1_only=args.step1_only).run())


if __name__ == "__main__":
    main()
