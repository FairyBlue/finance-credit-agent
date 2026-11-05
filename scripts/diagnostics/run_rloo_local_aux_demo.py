#!/usr/bin/env python3
"""Run the bounded sixteen-update exact auxiliary-credit pilot in systemd."""

from __future__ import annotations

import argparse
import csv
import shutil
import signal
import uuid
from pathlib import Path
from types import SimpleNamespace

from run_rloo_terminal_demo import observation
from run_same_prefix_online_demo import (
    COLLECTOR,
    EPISODES,
    MODEL,
    ROOT,
    Controller,
    collection,
    gpu,
    read,
    save,
)

DEFAULT = ROOT / "runs/m2-rloo-local-aux-demo-20261026-v1"
LOCAL = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
RLOO = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
FRESH = ROOT / "runs/m2-rloo-local-credit-fresh-eval-20261026-v1"
INITIAL = LOCAL / "common_initial"
TRAINER = "src.agents.rloo_terminal_train"
ARM = "exact_aux"
TERMINAL = {"COMPLETE", "COMPLETE_WITH_MODEL_FAILURES"}


class Demo(Controller):
    def __init__(self, args):
        self.cli, self.runroot, self.attempt = args, args.output.resolve(), 1
        super().__init__(
            SimpleNamespace(
                output=self.runroot / ARM,
                model=MODEL,
                collect_only=False,
                seed=42,
                episode_ids=EPISODES,
                alpha=0.5,
                eta=0.25,
                stage_timeout_seconds=args.stage_timeout_seconds,
                coordinator_request_timeout_seconds=60,
            )
        )
        self.state_path = self.output / "rloo_state.json"
        self.scratch = self.output / "training_state"
        self.state.update(
            phase="terminal_RLOO_separate_local_auxiliary", rounds=4, arms=[ARM], planned_updates=16
        )
        self.result = dict(
            profile="terminal_Y_RLOO_separate_fixed8_local_PPO_persistent_v1",
            status="RUNNING",
            arm=ARM,
            alpha=0.5,
            eta=0.25,
            seed=42,
            episode_ids=EPISODES,
            parent_repetitions=4,
            planned_parent_trajectories=16,
            planned_updates=16,
            updates_per_round=4,
            baseline=None,
            baseline_path=str(INITIAL),
            rounds=[],
            training_state=str(self.scratch),
            training_state_storage="persistent project disk: FP32 weights and Adam moments",
            actor_checkpoint=str(self.output / "current_actor"),
            common_initialization=str(MODEL),
            critic_updates=0,
            best_round_selection=False,
            shared_initial_reused=True,
            new_initial_generation_cost=0,
            fresh_evaluation=None,
            fresh_collection_path=str(self.output / "fresh_eval"),
            fresh_evaluation_seed_namespace=4,
            reference_results={
                "RLOO": str(RLOO / "rloo_results.json"),
                "plain": str(LOCAL / "plain/rloo_results.json"),
                "exact": str(LOCAL / "exact/rloo_results.json"),
            },
            fresh_reference_results=str(FRESH / "results.json"),
        )

    def persist(self):
        self.update()
        save(self.output / "rloo_results.json", self.result)
        save(self.runroot / "suite_state.json", self.state)
        save(
            self.runroot / "suite_results.json",
            dict(
                profile=self.result["profile"],
                status=self.result["status"],
                arm=ARM,
                result_path=str(self.output / "rloo_results.json"),
                completed_updates=self.result.get("completed_updates", 0),
                planned_updates=16,
                initial_collection=str(INITIAL),
                reference_results=self.result["reference_results"],
                fresh_reference_results=self.result["fresh_reference_results"],
                fresh_collection_path=self.result["fresh_collection_path"],
            ),
        )
        if self.result["baseline"] is None:
            return
        rows = [
            dict(
                updates=0,
                round=0,
                namespace=0,
                status="BASELINE_REUSED",
                **observation(self.result["baseline"]),
            )
        ]
        keys = list(observation(self.result["baseline"]))
        for entry in self.result["rounds"]:
            rows.append(
                dict(
                    updates=entry["round"] * 4,
                    round=entry["round"],
                    namespace=0 if entry["round"] == 4 else entry["round"],
                    status=entry["status"],
                    **(
                        observation(entry["evaluation"])
                        if entry.get("evaluation")
                        else dict.fromkeys(keys)
                    ),
                )
            )
        if self.result["fresh_evaluation"] is not None:
            rows.append(
                dict(
                    updates=16,
                    round=4,
                    namespace=4,
                    status="FRESH_FINAL",
                    **observation(self.result["fresh_evaluation"]),
                )
            )
        with (self.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def check_training(training, number):
        if (
            training.get("status") != "COMPLETE"
            or training.get("optimizer_updates") != 4
            or training.get("completed_updates") != number * 4
            or training.get("arm") != ARM
        ):
            raise RuntimeError("four committed exact_aux optimizer steps required per round")

    def validate_reused_inputs(self):
        initial = collection(INITIAL, MODEL, EPISODES, 42)
        if (
            initial.get("local_credit") is not True
            or initial.get("parents_only") is not False
            or initial.get("parent_repetitions") != 4
            or initial.get("planned_parent_trajectories") != 16
            or initial.get("round_index") != 0
            or observation(initial)["mean_Y"] is None
        ):
            raise RuntimeError("complete original shared sixteen-parent local batch required")
        for path in self.result["reference_results"].values():
            reference = read(Path(path))
            if (
                reference.get("status") != "COMPLETE"
                or reference.get("completed_updates") != 16
                or reference.get("seed") != 42
                or reference.get("episode_ids") != EPISODES
            ):
                raise RuntimeError("completed old16-update references required")
        fresh = read(Path(self.result["fresh_reference_results"]))
        if (
            fresh.get("status") not in TERMINAL
            or fresh.get("round_index") != 4
            or fresh.get("seed") != 42
            or fresh.get("episode_ids") != EPISODES
        ):
            raise RuntimeError("completed four-policy namespace4 reference required")
        self.result["baseline"] = initial

    def prepare(self):
        if not self.state_path.exists():
            if self.scratch.exists():
                raise FileExistsError("fresh exact_aux needs unused state directory")
            return
        if not self.cli.resume_completed_stages:
            raise FileExistsError("existing pilot requires explicit resume")
        state, result = read(self.state_path), read(self.output / "rloo_results.json")
        if (
            state["status"] not in {"FAILED", "PAUSED"}
            or result["arm"] != ARM
            or result["profile"] != self.result["profile"]
            or result["seed"] != 42
            or result["episode_ids"] != EPISODES
            or result["alpha"] != 0.5
            or result["eta"] != 0.25
            or Path(result["training_state"]) != self.scratch
            or Path(result["baseline_path"]) != INITIAL
        ):
            raise RuntimeError("resume requires this stopped exact_aux pilot")
        entries = result["rounds"]
        if [e["round"] for e in entries] != list(range(1, len(entries) + 1)):
            raise RuntimeError("noncontiguous old rounds")
        for entry in entries:
            training = read(self.output / f"round{entry['round']}/train/training_summary.json")
            self.check_training(training, entry["round"])
            entry["training"] = training
        if entries:
            manifest = read(self.scratch / "manifest.json")
            actor = self.output / "current_actor"
            index = read(actor / "model.safetensors.index.json")
            if (
                manifest.get("status") != "COMPLETE"
                or manifest.get("completed_optimizer_steps") != len(entries) * 4
                or (actor / "EXPORT_INCOMPLETE.json").exists()
                or not all((actor / f).is_file() for f in set(index["weight_map"].values()))
            ):
                raise RuntimeError("resume requires committed FP32/Adam and complete BF16 actor")
        elif self.scratch.exists():
            raise RuntimeError("uncommitted scratch state cannot be replayed")
        self.attempt = state.get("controller_attempt", 1) + 1
        archive = self.output / "controller_attempts"
        archive.mkdir(exist_ok=True)
        for filename in ("rloo_state.json", "rloo_results.json", "completion.json", "results.csv"):
            source, target = (
                self.output / filename,
                archive / f"attempt{self.attempt - 1}_{filename}",
            )
            if source.exists():
                if target.exists():
                    raise FileExistsError("attempt snapshot already exists")
                shutil.copy2(source, target)
        self.state, self.result = state, result
        self.state.update(status="RUNNING", current_stage=None, controller_attempt=self.attempt)
        self.result.update(status="RUNNING", controller_attempt=self.attempt)
        self.state.pop("error", None)
        self.result.pop("error", None)

    def collection_path(self, path):
        if (
            not path.exists()
            or not any(path.iterdir())
            or (path / "collection_summary.json").exists()
        ):
            return path
        if not self.cli.resume_completed_stages:
            raise FileExistsError("incomplete collection retained; explicit resume required")
        for filename in ("requests.jsonl", "trajectory_results.jsonl", "head_results.jsonl"):
            artifact = path / filename
            if artifact.exists() and artifact.stat().st_size:
                raise RuntimeError("generated failed collection cannot be resampled")
        fresh = path.with_name(f"{path.name}-attempt{self.attempt}")
        if fresh.exists():
            raise FileExistsError("pre-generation retry path already exists")
        return fresh

    def collect_batch(self, name, path, namespace, local_credit):
        actor = self.output / "current_actor"
        if not (path / "collection_summary.json").exists():
            arguments = [
                "--model",
                str(actor),
                "--output",
                str(path),
                "--episode-ids",
                *EPISODES,
                "--seed",
                "42",
                "--round-index",
                str(namespace),
                "--parent-repetitions",
                "4",
                "--local-credit" if local_credit else "--parents-only",
            ]
            suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
            self.command(name + suffix, COLLECTOR, arguments)
        summary = collection(path, actor, EPISODES, 42)
        if (
            summary.get("parent_repetitions") != 4
            or summary.get("planned_parent_trajectories") != 16
            or summary.get("round_index") != namespace
            or summary.get("parents_only") is local_credit
            or (local_credit and summary.get("local_credit") is not True)
        ):
            raise RuntimeError("fixed four-window grouped evaluation required")
        return summary

    def execute(self):
        self.prepare()
        self.persist()

        def forward(signum, _frame):
            if self.child is not None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        entry = None
        try:
            self.validate_reused_inputs()
            holder = self.cli.holder_directory or Path(
                read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"]
            )
            self.config = gpu.config_for_holder(
                holder,
                lease_token=f"rloo-aux-{uuid.uuid4().hex}",
                evidence_directory=self.runroot / f"gpu_attempt{self.attempt}",
                portfolio_target_bytes=76 * 1024**3,
            )
            gpu.freeze_config(self.runroot / f"gpu_config_attempt{self.attempt}.json", self.config)
            state = gpu.controller_claim(self.config)
            self.claimed = True
            gpu.request(
                str(holder),
                dict(
                    op="release",
                    lease_token=self.config["lease_token"],
                    bytes=state["maximum_bytes"],
                    request_id=f"release-{uuid.uuid4().hex}",
                ),
                timeout=60,
            )
            checked = gpu.reservation_status(self.config)
            if checked["paused"] is not True or any(
                checked[k]
                for k in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
            ):
                raise RuntimeError("holder must be paused and empty")
            data, actor = INITIAL, MODEL
            existing = {e["round"]: e for e in self.result["rounds"]}
            for number in range(1, 5):
                entry = existing.get(number)
                if entry is None:
                    entry = dict(
                        round=number,
                        status="TRAINING",
                        training=None,
                        evaluation=None,
                        training_data_path=str(data),
                    )
                    self.result["rounds"].append(entry)
                    self.persist()
                    arguments = [
                        "--data",
                        str(data),
                        "--output",
                        str(self.output / f"round{number}/train"),
                        "--model",
                        str(actor),
                        "--state-output",
                        str(self.scratch),
                        "--actor-output",
                        str(self.output / "current_actor"),
                        "--updates",
                        "4",
                        "--seed",
                        "42",
                        "--arm",
                        ARM,
                        "--alpha",
                        "0.5",
                        "--eta",
                        "0.25",
                        "--learning-rate",
                        "2e-6",
                        "--clip-epsilon",
                        "0.2",
                    ]
                    if number > 1:
                        arguments += ["--resume-state", str(self.scratch)]
                    suffix = f"_attempt{self.attempt}" if self.attempt > 1 else ""
                    self.command(f"round{number}_train{suffix}", TRAINER, arguments)
                    entry["training"] = read(
                        self.output / f"round{number}/train/training_summary.json"
                    )
                self.check_training(entry["training"], number)
                self.result["completed_updates"] = number * 4
                actor = self.output / "current_actor"
                if entry.get("evaluation") is not None:
                    data = Path(entry["collection_path"])
                    continue
                entry.update(status="COLLECTING")
                entry.pop("error", None)
                self.persist()
                data = self.collection_path(self.output / f"round{number}/eval")
                namespace = 0 if number == 4 else number
                evaluation = self.collect_batch(
                    f"round{number}_parents", data, namespace, number < 4
                )
                entry.update(
                    status="COMPLETE"
                    if observation(evaluation)["mean_Y"] is not None
                    else "COMPLETE_WITH_MODEL_FAILURES",
                    evaluation=evaluation,
                    collection_path=str(data),
                    evaluation_seed_namespace=namespace,
                    paired_with_initial_sampling_seeds=number == 4,
                )
                self.persist()
            entry = None
            if self.result["fresh_evaluation"] is None:
                path = self.collection_path(Path(self.result["fresh_collection_path"]))
                self.result["fresh_collection_path"] = str(path)
                self.persist()
                fresh = self.collect_batch("fresh_final_parents", path, 4, False)
                self.result["fresh_evaluation"] = fresh
                self.persist()
            model_failures = any(
                observation(e["evaluation"])["mean_Y"] is None for e in self.result["rounds"]
            )
            model_failures |= observation(self.result["fresh_evaluation"])["mean_Y"] is None
            status = "COMPLETE_WITH_MODEL_FAILURES" if model_failures else "COMPLETE"
            self.result.update(status=status, completed_updates=16)
            self.update(status=status, current_stage=None)
            return 0
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if entry is not None:
                entry.update(status="FAILED", error=message)
                partial = self.output / f"round{entry['round']}/train/training_summary.json"
                if entry["training"] is None and partial.exists():
                    entry["training"] = read(partial)
            self.result.update(status="FAILED", error=message)
            for stage in self.state["stages"]:
                if stage["status"] == "RUNNING":
                    stage.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            return 1
        finally:
            self.stop_child()
            self.persist()
            evaluations = [e["evaluation"] for e in self.result["rounds"] if e.get("evaluation")]
            if self.result["fresh_evaluation"] is not None:
                evaluations.append(self.result["fresh_evaluation"])
            completion = dict(
                status=self.result["status"],
                arm=ARM,
                seed=42,
                planned_updates=16,
                completed_updates=self.result.get("completed_updates", 0),
                completed_rounds=sum(e["status"] in TERMINAL for e in self.result["rounds"]),
                stage_wall_seconds=sum(s.get("seconds", 0) for s in self.state["stages"]),
                training_state=str(self.scratch),
                actor_checkpoint=str(self.output / "current_actor"),
                training_state_is_host_reboot_volatile=False,
                critic_updates=0,
                shared_initial_reused=True,
                new_initial_generation_cost=0,
                new_collection_cost={
                    k: sum(e.get(k, 0) for e in evaluations)
                    for k in ("model_requests", "prompt_tokens", "response_tokens")
                },
                final_namespace0_collection_path=self.result["rounds"][-1].get("collection_path")
                if len(self.result["rounds"]) == 4
                else None,
                fresh_namespace4_collection_path=self.result["fresh_collection_path"],
                fresh_namespace4_collected=self.result["fresh_evaluation"] is not None,
                best_round_selection=False,
                model_failure_resampling=False,
                old_references_reused=True,
                old_checkpoint_retirement_performed=False,
            )
            save(self.output / "completion.json", completion)
            save(self.runroot / "completion.json", completion)
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.state["holder_restore"] = "REFILLING_76_GIB"
                except (OSError, RuntimeError, ValueError) as error:
                    self.state.update(
                        holder_restore="DEFERRED_TO_EMPTY_CGROUP_WATCHDOG",
                        restore_reason=str(error),
                    )
                self.persist()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume-completed-stages", action="store_true")
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=3600)
    raise SystemExit(Demo(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
