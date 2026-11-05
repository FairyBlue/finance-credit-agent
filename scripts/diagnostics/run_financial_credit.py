#!/usr/bin/env python3
"""Bounded strong-SFT credit-signal experiment, independent of the chat session."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import signal
import subprocess
import time
import uuid
from pathlib import Path

from run_financial_evidence import ENVIRONMENT, GROUP_RESERVATION
from run_financial_evidence import Controller as EvidenceController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, MODEL, ROOT, TRAINER, now, read, save

DEFAULT = ROOT / "runs/m3-financial-credit-20261030-v1"
INITIAL = ROOT / "runs/m3-financial-evidence-20261030-v2/teacher_sft/adapter"
PROBE = "src.agents.financial_credit_probe"
PARENT_BUDGET, PROBE_BUDGET, ARM_BUDGET = 5_000_000, 2_000_000, 2_750_000
ARMS = ("rloo", "sham", "m3", "shuffled")


class Controller(EvidenceController):
    def __init__(self, output, resume):
        super().__init__(output, resume, arm_budget=ARM_BUDGET)
        self.persist(
            experiment="strong_sft_credit_mechanism_v1",
            initial_adapter=str(INITIAL),
            parent_probe_token_cap=PARENT_BUDGET,
            paired_probe_token_cap=PROBE_BUDGET,
            arm_token_cap=ARM_BUDGET,
            total_training_and_probe_token_cap=18_000_000,
            second_seed_automatically_authorized=False,
        )

    def wait_ready(self, path):
        deadline = time.monotonic() + 3600
        while not path.exists():
            self.persist(current_stage=f"WAITING_{path.stem.upper()}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"required artifact not ready: {path}")
            time.sleep(2)
        if path.suffix == ".jsonl":
            if not path.stat().st_size:
                raise RuntimeError(f"empty required artifact: {path}")
        elif read(path).get("status") not in ("COMPLETE", "TRAIN_READY", "READY"):
            raise RuntimeError(f"required artifact incomplete: {path}")

    def report(self):
        script = ROOT / "scripts/diagnostics/read_financial_credit.py"
        if script.exists():
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(script), "--output", str(self.output)],
                cwd=ROOT,
                env=self.environment,
                capture_output=True,
                text=True,
            )
            (self.output / "readout.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                self.persist(readout_error=result.stderr[-3000:])

    def collect_parents(self):
        destination = self.output / "probe/parents"
        if self.fresh(destination):
            self.stage(
                "strong_sft_parent_probe",
                COLLECTOR,
                [
                    "--data",
                    self.output / "data/probe32.jsonl",
                    "--base-model",
                    MODEL,
                    "--adapter",
                    INITIAL,
                    "--output",
                    destination,
                    "--seed",
                    42,
                    "--temperature",
                    1,
                    "--repetitions",
                    4,
                    "--token-budget",
                    PARENT_BUDGET,
                    "--max-groups",
                    32,
                    "--engine-mode",
                    "eager",
                    *ENVIRONMENT,
                ],
            )
        summary = read(destination / "summary.json")
        if summary["total_tokens"] > PARENT_BUDGET:
            raise RuntimeError("parent probe exceeded its fixed token ceiling")
        self.persist(
            parent_probe_tokens=summary["total_tokens"],
            parent_probe_admitted_groups=summary["admitted_task_groups"],
            parent_probe_planned_groups=summary["planned_task_groups"],
        )
        self.report()
        return destination

    def paired_probe(self, parents):
        self.wait_ready(self.output / "probe_worker_ready.json")
        destination = self.output / "probe/paired"
        if (
            destination.exists()
            and any(destination.iterdir())
            and not (destination / "summary.json").exists()
        ):
            requests = (
                rows(destination / "requests.jsonl")
                if (destination / "requests.jsonl").exists()
                else []
            )
            logged_tokens = sum(
                item.get("prompt_tokens", 0) + item.get("response_tokens", 0) for item in requests
            )
            progress = (
                read(destination / "progress.json")
                if (destination / "progress.json").exists()
                else {}
            )
            known_tokens = max(logged_tokens, progress.get("total_tokens", 0))
            archive = self.output / "failed_attempts"
            archive.mkdir(exist_ok=True)
            retained = archive / f"paired_attempt{self.attempt - 1}_{uuid.uuid4().hex[:8]}"
            destination.rename(retained)
            history = self.state.setdefault("retired_paired_attempts", [])
            history.append(
                dict(
                    directory=str(retained),
                    known_generation_tokens=known_tokens,
                    raw_logged_requests=len(requests),
                    token_source="max(raw_request_prompt_plus_response, recorded_progress)",
                    actor_training_data=False,
                )
            )
            self.persist(
                retired_paired_generation_tokens=sum(
                    item["known_generation_tokens"] for item in history
                )
            )
        retired_tokens = self.state.get("retired_paired_generation_tokens", 0)
        remaining = max(0, PROBE_BUDGET - retired_tokens)
        if self.fresh(destination):
            self.stage(
                "paired_information_signal_probe",
                PROBE,
                [
                    "--data",
                    self.output / "data/probe32.jsonl",
                    "--parents",
                    parents,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    INITIAL,
                    "--output",
                    destination,
                    "--seed",
                    42,
                    "--token-budget",
                    remaining,
                    "--max-context-tokens",
                    16384,
                    "--repetitions",
                    2,
                    "--max-anchors",
                    16,
                ],
            )
        summary, gate = read(destination / "summary.json"), read(destination / "gate.json")
        if summary["total_tokens"] + retired_tokens > PROBE_BUDGET:
            raise RuntimeError("paired probe exceeded its fixed token ceiling")
        self.persist(
            paired_probe_tokens=summary["total_tokens"] + retired_tokens,
            completed_paired_probe_tokens=summary["total_tokens"],
            signal_gate=gate,
        )
        self.report()
        return gate

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.persist(
                current_step=1, maximum_tool_calls=10, full_teacher_sft_is_rl_initialization=True
            )
            self.wait_ready(self.output / "data/manifest.json")
            self.wait_ready(self.output / "data/probe32.jsonl")
            self.claim_gpu()
            parents = self.collect_parents()
            gate = self.paired_probe(parents)
            if gate["status"] == "PASS":
                self.persist(
                    current_step=2,
                    step1_status="COMPLETE",
                    current_stage="WAITING_FOR_TRAINING_INTERFACE",
                )
                self.wait_ready(self.output / "training_worker_ready.json")
                self.run_mechanism_comparison()
            elif gate["status"] in ("FAIL", "INCONCLUSIVE"):
                reason = (
                    "NO_REPRODUCIBLE_LOCAL_SIGNAL"
                    if gate["status"] == "FAIL"
                    else "INCONCLUSIVE_SIGNAL_PROBE"
                )
                self.state["arms"] = {
                    arm + "_seed42": dict(status="SKIPPED", reason=reason) for arm in ARMS
                }
                save(
                    self.output / "decision.json",
                    dict(
                        status="COMPLETE",
                        decision=reason,
                        signal_gate=gate,
                        no_credit_training_performed=True,
                        preferred_existing_endpoint=str(INITIAL),
                        credit_method_claim_supported=False,
                        training_stopped=True,
                        fresh_test_opened=False,
                        research_direction_review_required=True,
                        finished_at=now(),
                    ),
                )
                self.persist(
                    step1_status="COMPLETE", step2_status="SKIPPED", step3_status="COMPLETE"
                )
            else:
                raise RuntimeError(f"unknown signal gate status: {gate['status']}")
            self.persist(status="COMPLETE", current_stage=None, finished_at=now())
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
                from run_mandate_repair import gpu

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

    def run_mechanism_comparison(self):
        # Reload at this explicit boundary before the first actor update.
        spec = importlib.util.spec_from_file_location("financial_credit_ready", __file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.mechanism_comparison(self)


def total_training_tokens(controller):
    return (
        controller.state.get("parent_probe_tokens", 0)
        + controller.state.get("paired_probe_tokens", 0)
        + sum(item.get("cumulative_tokens", 0) for item in controller.state["arms"].values())
    )


def store_progress(controller, key, path, progress):
    save(path, progress)
    controller.state["arms"][key] = progress
    actual = total_training_tokens(controller)
    if actual > 18_000_000:
        raise RuntimeError("actual shared probe plus new arm generation exceeded 18M")
    controller.persist(actual_training_and_probe_tokens=actual)
    controller.report()


def train_batch(controller, key, arm, collection, destination, actor, local, previous):
    if controller.fresh(destination):
        arguments = [
            "rl",
            "--data",
            collection,
            "--base-model",
            MODEL,
            "--adapter",
            actor,
            "--output",
            destination,
            "--seed",
            42,
            "--arm",
            arm,
        ]
        if previous is not None:
            arguments += ["--resume-state", previous]
        if arm != "rloo":
            arguments += [
                "--local-credit",
                local,
                "--local-credit-target",
                "business",
                "--local-weight",
                0.25,
            ]
        controller.stage(key + "_train", TRAINER, arguments)
    return read(destination / "summary.json")


def train_credit_arm(controller, arm):
    key = arm + "_seed42"
    directory = controller.output / "training" / key
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    progress = (
        read(progress_path)
        if progress_path.exists()
        else dict(
            arm=arm,
            seed=42,
            status="RUNNING",
            token_budget=ARM_BUDGET,
            auxiliary_token_cap=ARM_BUDGET // 3 if arm != "rloo" else 0,
            cumulative_tokens=0,
            parent_tokens=0,
            auxiliary_tokens=0,
            completed_optimizer_steps=0,
            actor_task_groups=0,
            rounds=[],
            final_adapter=str(INITIAL),
            local_credit_target="business" if arm != "rloo" else None,
            credit_estimator="paired_factual_mask_mean_v1" if arm != "rloo" else None,
            global_shared_generation_is_not_free_compute=True,
        )
    )
    if progress["status"] == "COMPLETE":
        return Path(progress["final_adapter"])

    # All four arms expose the identical on-policy strong-SFT parent batch.
    # Its generation is charged exactly once to the shared probe, never hidden.
    shared = controller.output / "probe/parents"
    shared_local = controller.output / "probe/paired/local_credit.jsonl"
    if not progress.get("shared_initial_batch_complete"):
        destination = directory / "shared_initial/train"
        trained = train_batch(
            controller,
            key + "_shared_initial",
            arm,
            shared,
            destination,
            INITIAL,
            shared_local,
            None,
        )
        progress.update(
            shared_initial_batch_complete=True,
            shared_parent_generation_exposure_tokens=controller.state["parent_probe_tokens"],
            shared_auxiliary_generation_exposure_tokens=(
                controller.state.get("completed_paired_probe_tokens", 0) if arm != "rloo" else 0
            ),
            shared_initial_optimizer_updates=trained["optimizer_updates"],
            completed_optimizer_steps=trained["completed_optimizer_steps"],
            final_adapter=str(destination / "adapter"),
            shared_initial_state=str(destination / "training_state.pt"),
            local_nonzero_request_uses=trained.get("local_nonzero_requests", 0),
        )
        progress["actor_task_groups"] += read(shared / "summary.json")["admitted_task_groups"]
        store_progress(controller, key, progress_path, progress)

    excluded = {task["task_id"] for task in rows(controller.output / "data/probe32.jsonl")}
    tasks = [
        task
        for task in rows(controller.output / "data/train.jsonl")
        if task["task_id"] not in excluded
    ]
    random.Random(42).shuffle(tasks)
    if len(tasks) < 4:
        raise RuntimeError("fresh training stream requires at least four non-probe tasks")
    actor = Path(progress["final_adapter"])
    round_index = len(progress["rounds"])
    while ARM_BUDGET - progress["cumulative_tokens"] >= GROUP_RESERVATION:
        rd = directory / f"round_{round_index:03d}"
        rd.mkdir(parents=True, exist_ok=True)
        selected = rd / "selected.jsonl"
        if not selected.exists():
            offset = round_index * 4
            selected.write_text(
                "".join(
                    json.dumps(tasks[(offset + i) % len(tasks)], ensure_ascii=False) + "\n"
                    for i in range(4)
                )
            )
        collection = rd / "collect"
        remaining = ARM_BUDGET - progress["cumulative_tokens"]
        if controller.fresh(collection):
            controller.stage(
                key + f"_round{round_index:03d}_collect",
                COLLECTOR,
                [
                    "--data",
                    selected,
                    "--base-model",
                    MODEL,
                    "--adapter",
                    actor,
                    "--output",
                    collection,
                    "--seed",
                    42,
                    "--temperature",
                    1,
                    "--repetitions",
                    4,
                    "--token-budget",
                    remaining,
                    "--max-groups",
                    4,
                    "--engine-mode",
                    "eager",
                    *ENVIRONMENT,
                ],
            )
        collected = read(collection / "summary.json")
        parent_tokens = collected["total_tokens"]
        if not parent_tokens:
            progress["budget_stop_reason"] = "NO_FULL_PARENT_GROUP_ADMITTED"
            break
        auxiliary_tokens, local = 0, None
        if arm != "rloo":
            paired = rd / "paired"
            allowance = min(
                progress["auxiliary_token_cap"] - progress["auxiliary_tokens"],
                remaining - parent_tokens,
            )
            if controller.fresh(paired):
                if allowance <= 0:
                    paired.mkdir(parents=True, exist_ok=True)
                    (paired / "local_credit.jsonl").touch(exist_ok=False)
                    save(
                        paired / "summary.json",
                        dict(
                            status="SKIPPED_BUDGET",
                            total_tokens=0,
                            model_requests=0,
                            local_credit_supported_parents=0,
                            reason="NO_REMAINING_AUXILIARY_TOKEN_ALLOWANCE",
                            current_policy_continuations_generated=False,
                        ),
                    )
                else:
                    controller.stage(
                        key + f"_round{round_index:03d}_paired",
                        PROBE,
                        [
                            "--data",
                            selected,
                            "--parents",
                            collection,
                            "--base-model",
                            MODEL,
                            "--adapter",
                            actor,
                            "--output",
                            paired,
                            "--seed",
                            42,
                            "--token-budget",
                            max(0, allowance),
                            "--max-context-tokens",
                            16384,
                            "--repetitions",
                            2,
                            "--max-anchors",
                            2,
                            "--conditions",
                            "factual",
                            "masked",
                            "--local-credit-arm",
                            arm,
                        ],
                    )
            measured = read(paired / "summary.json")
            auxiliary_tokens = measured["total_tokens"]
            if auxiliary_tokens > allowance:
                raise RuntimeError("paired training exceeded its remaining auxiliary allowance")
            local = paired / "local_credit.jsonl"
        previous = (
            Path(progress["rounds"][-1]["train"]) / "training_state.pt"
            if progress["rounds"]
            else Path(progress["shared_initial_state"])
        )
        destination = rd / "train"
        trained = train_batch(
            controller,
            key + f"_round{round_index:03d}",
            arm,
            collection,
            destination,
            actor,
            local,
            previous,
        )
        progress["parent_tokens"] += parent_tokens
        progress["auxiliary_tokens"] += auxiliary_tokens
        progress["cumulative_tokens"] += parent_tokens + auxiliary_tokens
        progress["completed_optimizer_steps"] = trained["completed_optimizer_steps"]
        progress["actor_task_groups"] += collected["admitted_task_groups"]
        progress["local_nonzero_request_uses"] += trained.get("local_nonzero_requests", 0)
        actor = destination / "adapter"
        progress["final_adapter"] = str(actor)
        progress["rounds"].append(
            dict(
                round=round_index,
                collect=str(collection),
                train=str(destination),
                tokens=parent_tokens + auxiliary_tokens,
                parent_tokens=parent_tokens,
                auxiliary_tokens=auxiliary_tokens,
                optimizer_updates=trained["optimizer_updates"],
                completed_optimizer_steps=trained["completed_optimizer_steps"],
                parent_successes=collected["successes"],
                local_nonzero_requests=trained.get("local_nonzero_requests", 0),
            )
        )
        if (
            progress["cumulative_tokens"] > ARM_BUDGET
            or progress["auxiliary_tokens"] > progress["auxiliary_token_cap"]
        ):
            raise RuntimeError("arm actual generation budget exceeded")
        store_progress(controller, key, progress_path, progress)
        round_index += 1
    progress.update(
        status="COMPLETE",
        finished_at=now(),
        final_adapter=str(actor),
        unused_budget_tokens=ARM_BUDGET - progress["cumulative_tokens"],
        endpoints_not_strictly_equal_actual_compute=True,
    )
    store_progress(controller, key, progress_path, progress)
    return actor


def evaluate_endpoint(controller, key, adapter, split="dev"):
    if key == "sft" and split == "dev":
        source = ROOT / "runs/m3-financial-evidence-20261030-v2/eval/sft_dev"
        data = controller.output / "data/development.jsonl"
        original = read(source / "summary.json")
        compatible = (
            original.get("status") == "COMPLETE"
            and original.get("adapter_path") == str(INITIAL)
            and original.get("data_sha256") == hashlib.sha256(data.read_bytes()).hexdigest()
            and original.get("observation_profile")
            == "financial_evidence_conditional_mandate_v2_opaque_public_ids"
        )
        if compatible:
            destination = controller.output / "eval/sft_dev"
            destination.mkdir(parents=True, exist_ok=True)
            if not (destination / "summary.json").exists():
                observed = rows(source / "trajectory_results.jsonl")
                for item in observed:
                    item.update(evaluation_reused=True, evaluation_source=str(source))
                (destination / "trajectory_results.jsonl").write_text(
                    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in observed)
                )
                original.update(
                    evaluation_reused=True,
                    evaluation_source=str(source),
                    source_sampled_tokens=original["total_tokens"],
                    total_tokens=0,
                    model_requests=0,
                    prompt_tokens=0,
                    response_tokens=0,
                    newly_sampled_trajectories=0,
                    policy="sft",
                    split="dev",
                    requested_data_path=str(data),
                )
                save(destination / "summary.json", original)
            return read(destination / "summary.json")
    return controller.evaluate(key, adapter, split=split)


def mechanism_comparison(controller):
    """Called only after the immutable global signal gate passes."""
    if read(controller.output / "probe/paired/gate.json")["status"] != "PASS":
        raise RuntimeError("mechanism training requires the original passed signal gate")
    controller.persist(
        current_step=2,
        step2_status="RUNNING",
        step3_status="PENDING",
        common_shared_parent_batch_is_training_data=True,
        shared_parent_generation_charged_once_globally=True,
    )
    endpoints = {"sft": INITIAL}
    evaluate_endpoint(controller, "sft", INITIAL)
    for arm in ARMS:
        key = arm + "_seed42"
        endpoints[key] = train_credit_arm(controller, arm)
        evaluate_endpoint(controller, key, endpoints[key])
        controller.report()
    controller.persist(current_step=3, step2_status="COMPLETE", step3_status="RUNNING")
    from src.agents.mandate_repair_train import adapter_digest

    frozen = controller.output / "frozen_models.json"
    controller.wait_ready(controller.output / "data/test.jsonl")
    if not frozen.exists():
        save(
            frozen,
            dict(
                status="COMPLETE",
                frozen_at=now(),
                models={
                    key: dict(
                        base_model=str(MODEL),
                        adapter=str(adapter),
                        adapter_sha256=adapter_digest(adapter),
                    )
                    for key, adapter in endpoints.items()
                },
                checkpoint_selection="FIXED_ENDPOINT_ALL_ARMS_INCLUDED",
                extra_seed="NOT_AUTOMATICALLY_STARTED",
                reserved_test_used_for_training_or_allocation=False,
            ),
        )
    for key, adapter in endpoints.items():
        evaluate_endpoint(controller, key, adapter, split="test")
        controller.report()
    totals = total_training_tokens(controller)
    if totals > 18_000_000:
        raise RuntimeError("final actual training and probe generation exceeded 18M")
    save(
        controller.output / "decision.json",
        dict(
            status="COMPLETE",
            decision="REVIEW_FROZEN_CREDIT_MECHANISM_RESULTS",
            actual_training_and_probe_generation_tokens=totals,
            shared_parent_generation_tokens=controller.state["parent_probe_tokens"],
            shared_paired_generation_tokens=controller.state["paired_probe_tokens"],
            new_arm_generation_tokens=sum(
                item.get("cumulative_tokens", 0) for item in controller.state["arms"].values()
            ),
            endpoints=list(endpoints),
            signal_gate=controller.state["signal_gate"],
            strict_equal_actual_compute_claim=False,
            extra_seed_started=False,
            training_stopped=True,
            fresh_test_opened=True,
            finished_at=now(),
        ),
    )
    controller.persist(step3_status="COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(Controller(args.output, args.resume).run())


if __name__ == "__main__":
    main()
