#!/usr/bin/env python3
"""Compare fixed-alpha plain/exact local credit with the completed RLOO demo."""

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

DEFAULT = ROOT / "runs/m2-rloo-local-credit-demo-20261025-v1"
RLOO = ROOT / "runs/m2-rloo-terminal-demo-20261025-v3"
TRAINER = "src.agents.rloo_terminal_train"
ARMS = ("plain", "exact")
RETIRED = "RETIRED_OPTIMIZER_MOMENTS_AFTER_TERMINAL_EVAL"


class ArmRun(Controller):
    def __init__(self, suite, arm):
        self.suite, self.arm, self.attempt = suite, arm, 1
        super().__init__(
            SimpleNamespace(
                output=suite.output / arm,
                model=MODEL,
                collect_only=False,
                seed=42,
                episode_ids=EPISODES,
                alpha=0.5,
                eta=0.25,
                stage_timeout_seconds=suite.args.stage_timeout_seconds,
                coordinator_request_timeout_seconds=60,
            )
        )
        self.config = suite.config
        self.state_path = self.output / "rloo_state.json"
        self.scratch = self.output / "training_state"
        self.state.update(phase="rloo_local_credit", arms=[arm], rounds=4, planned_updates=16)
        self.result = dict(
            profile="persistent_fp32_terminal_RLOO_local_credit_demo_v1",
            status="RUNNING",
            arm=arm,
            alpha=0.5,
            eta=0.25,
            seed=42,
            episode_ids=EPISODES,
            planned_updates=16,
            updates_per_round=4,
            parent_repetitions=4,
            planned_parent_trajectories=16,
            baseline=suite.initial,
            baseline_path=str(suite.output / "common_initial"),
            rounds=[],
            training_state=str(self.scratch),
            training_state_storage="persistent project disk: FP32 weights and Adam moments",
            actor_checkpoint=str(self.output / "current_actor"),
            critic_updates=0,
            best_round_selection=False,
            common_initialization=str(MODEL),
            rloo_reference=str(RLOO),
        )

    def update(self, **values):
        super().update(**values)
        self.suite.state.update(current_arm=self.arm, current_stage=self.state.get("current_stage"))
        self.suite.persist()

    def persist(self):
        save(self.output / "rloo_results.json", self.result)
        rows = [dict(updates=0, round=0, status="BASELINE", **observation(self.result["baseline"]))]
        keys = list(observation(self.result["baseline"]))
        for entry in self.result["rounds"]:
            rows.append(
                dict(
                    updates=entry["round"] * 4,
                    round=entry["round"],
                    status=entry["status"],
                    **(
                        observation(entry["evaluation"])
                        if entry.get("evaluation")
                        else dict.fromkeys(keys)
                    ),
                )
            )
        with (self.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def prepare_resume(self):
        old, result = read(self.state_path), read(self.output / "rloo_results.json")
        if (
            old["status"] not in {"FAILED", "PAUSED"}
            or result["arm"] != self.arm
            or result["episode_ids"] != EPISODES
            or result["alpha"] != 0.5
            or result["eta"] != 0.25
            or Path(result["training_state"]) != self.scratch
        ):
            raise RuntimeError("resume requires this stopped fixed local-credit arm")
        entries = result["rounds"]
        if [e["round"] for e in entries] != list(range(1, len(entries) + 1)):
            raise RuntimeError("noncontiguous committed rounds")
        for entry in entries:
            summary = read(self.output / f"round{entry['round']}/train/training_summary.json")
            self.check_training(summary, entry["round"])
            entry["training"] = summary
        if entries:
            manifest = read(self.scratch / "manifest.json")
            actor = self.output / "current_actor"
            index = read(actor / "model.safetensors.index.json")
            if (
                manifest["status"] != "COMPLETE"
                or manifest["completed_optimizer_steps"] != len(entries) * 4
                or (actor / "EXPORT_INCOMPLETE.json").exists()
                or not all((actor / f).is_file() for f in set(index["weight_map"].values()))
            ):
                raise RuntimeError("resume needs committed FP32/Adam state and BF16 actor")
        elif self.scratch.exists():
            raise RuntimeError("uncommitted scratch state cannot be silently replayed")
        self.attempt = old.get("controller_attempt", 1) + 1
        snapshot(
            self.output,
            self.attempt - 1,
            ("rloo_state.json", "rloo_results.json", "completion.json", "results.csv"),
        )
        self.state, self.result = old, result
        self.state.update(status="RUNNING", current_stage=None, controller_attempt=self.attempt)
        self.result.update(status="RUNNING", controller_attempt=self.attempt)
        self.state.pop("error", None)
        self.result.pop("error", None)

    @staticmethod
    def check_training(summary, number):
        if (
            summary["status"] != "COMPLETE"
            or summary["optimizer_updates"] != 4
            or summary["completed_updates"] != number * 4
        ):
            raise RuntimeError("exactly four persistent optimizer updates required")

    def execute(self):
        if self.state_path.exists():
            if not self.suite.args.resume_completed_stages:
                raise FileExistsError("existing arm requires explicit resume")
            self.prepare_resume()
        elif self.scratch.exists():
            raise FileExistsError("fresh arm needs unused state directory")
        entry = None
        try:
            self.update()
            self.persist()
            data, actor = self.suite.output / "common_initial", MODEL
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
                        self.arm,
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
                actor = self.output / "current_actor"
                if entry.get("evaluation") is not None:
                    data = Path(entry["collection_path"])
                    continue
                entry.update(status="COLLECTING")
                entry.pop("error", None)
                self.persist()
                data = self.suite.fresh_collection_path(
                    self.output / f"round{number}/eval", self.attempt
                )
                namespace = 0 if number == 4 else number
                evaluation = self.suite.collect_batch(
                    self,
                    f"round{number}_parents",
                    actor,
                    data,
                    namespace,
                    local_credit=number < 4,
                )
                entry.update(
                    status="COMPLETE",
                    evaluation=evaluation,
                    collection_path=str(data),
                    evaluation_seed_namespace=namespace,
                    paired_with_initial_sampling_seeds=number == 4,
                )
                self.persist()
            self.result.update(status="COMPLETE", completed_updates=16)
            self.update(status="COMPLETE", current_stage=None)
        except BaseException as error:
            self.stop_child()
            message = f"{type(error).__name__}: {error}"
            if entry is not None:
                entry.update(status="FAILED", error=message)
                partial = self.output / f"round{entry['round']}/train/training_summary.json"
                if entry["training"] is None and partial.exists():
                    entry["training"] = read(partial)
            self.result.update(status="FAILED", error=message)
            self.update(status="FAILED", error=message)
            raise
        finally:
            self.stop_child()
            self.persist()
            evaluations = [e["evaluation"] for e in self.result["rounds"] if e.get("evaluation")]
            save(
                self.output / "completion.json",
                dict(
                    status=self.result["status"],
                    arm=self.arm,
                    seed=42,
                    planned_updates=16,
                    completed_updates=sum(
                        e["training"].get("optimizer_updates", 0)
                        for e in self.result["rounds"]
                        if e.get("training")
                    ),
                    completed_rounds=sum(e["status"] == "COMPLETE" for e in self.result["rounds"]),
                    stage_wall_seconds=sum(s.get("seconds", 0) for s in self.state["stages"]),
                    training_state=str(self.scratch),
                    training_state_is_host_reboot_volatile=False,
                    actor_checkpoint=str(self.output / "current_actor"),
                    critic_updates=0,
                    common_initial_cost_accounting="shared once at suite level",
                    new_post_update_collection_cost={
                        k: sum(e.get(k, 0) for e in evaluations)
                        for k in ("model_requests", "prompt_tokens", "response_tokens")
                    },
                    final_evaluation_shares_initial_sampling_seed_namespace=True,
                ),
            )


def snapshot(directory, attempt, filenames):
    archive = directory / "controller_attempts"
    archive.mkdir(exist_ok=True)
    for name in filenames:
        source, target = directory / name, archive / f"attempt{attempt}_{name}"
        if source.exists():
            if target.exists():
                raise FileExistsError("previous attempt snapshot already exists")
            shutil.copy2(source, target)


class Suite(Controller):
    def __init__(self, args):
        self.cli, self.active, self.initial, self.attempt = args, None, None, 1
        super().__init__(
            SimpleNamespace(
                output=args.output,
                model=MODEL,
                collect_only=args.collect_only,
                seed=42,
                episode_ids=EPISODES,
                alpha=0.5,
                eta=0.25,
                stage_timeout_seconds=args.stage_timeout_seconds,
                coordinator_request_timeout_seconds=60,
            )
        )
        self.args.resume_completed_stages = args.resume_completed_stages
        self.prefix = "initial" if args.collect_only else "suite"
        self.state_path = self.output / f"{self.prefix}_state.json"
        self.state.update(
            phase="terminal_rloo_local_credit",
            arms=list(ARMS),
            rounds=4,
            planned_updates=0 if args.collect_only else 32,
            reference_rloo=str(RLOO),
        )
        reference = read(RLOO / "rloo_results.json")
        if (
            reference["status"] != "COMPLETE"
            or reference["completed_updates"] != 16
            or reference["seed"] != 42
            or reference["episode_ids"] != EPISODES
        ):
            raise RuntimeError("completed fixed four-window seed42 RLOO reference required")
        self.source = Path(reference["baseline_path"])
        self.result = dict(
            profile="fixed_alpha_terminal_RLOO_local_credit_comparison_v1",
            status="RUNNING",
            seed=42,
            episode_ids=EPISODES,
            alpha=0.5,
            eta=0.25,
            planned_new_updates=0 if args.collect_only else 32,
            rloo_reference=str(RLOO),
            rloo_reference_reused=True,
            common_initial=str(self.output / "common_initial"),
            reused_parent_source=str(self.source),
            arms={a: dict(status="NOT_RUN", directory=str(self.output / a)) for a in ARMS},
        )

    def persist(self):
        self.update()
        save(self.output / f"{self.prefix}_results.json", self.result)

    def fresh_collection_path(self, path, attempt):
        if (
            not path.exists()
            or not any(path.iterdir())
            or (path / "collection_summary.json").exists()
        ):
            return path
        if not self.cli.resume_completed_stages:
            raise FileExistsError("incomplete collection retained; explicit resume required")
        for name in ("requests.jsonl", "trajectory_results.jsonl", "head_results.jsonl"):
            artifact = path / name
            if artifact.exists() and artifact.stat().st_size:
                raise RuntimeError("generated partial collection cannot be resampled")
        fresh = path.with_name(f"{path.name}-attempt{attempt}")
        if fresh.exists():
            raise FileExistsError("pre-generation retry path already exists")
        return fresh

    def collect_batch(self, runner, name, actor, path, namespace, *, local_credit, source=None):
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
            ]
            if local_credit:
                arguments.append("--local-credit")
                if source is not None:
                    arguments += ["--local-credit-source", str(source)]
            else:
                arguments.append("--parents-only")
            suffix = f"_attempt{runner.attempt}" if runner.attempt > 1 else ""
            runner.command(name + suffix, COLLECTOR, arguments)
        result = collection(path, actor, EPISODES, 42)
        if (
            result.get("parent_repetitions") != 4
            or result.get("planned_parent_trajectories") != 16
            or result["round_index"] != namespace
            or result["parents_only"] is local_credit
            or (local_credit and result.get("local_credit") is not True)
            or observation(result)["mean_Y"] is None
        ):
            raise RuntimeError("complete fixed four-window grouped collection required")
        if source is not None and result.get("parent_generation_reused") is not True:
            raise RuntimeError("common initial must reuse the original RLOO parent trajectories")
        return result

    def retire_plain_moments(self):
        directory = self.output / "plain"
        result = read(directory / "rloo_results.json")
        if (
            result["status"] != "COMPLETE"
            or result["completed_updates"] != 16
            or len(result["rounds"]) != 4
            or result["rounds"][-1]["status"] != "COMPLETE"
            or observation(result["rounds"][-1]["evaluation"])["mean_Y"] is None
        ):
            raise RuntimeError(
                "plain moment retirement requires all16 updates and final16-parent evaluation"
            )
        state = directory / "training_state"
        manifest_path = state / "manifest.json"
        manifest = read(manifest_path)
        retirement_path = state / "optimizer_moment_retirement.json"
        if manifest["status"] == RETIRED:
            return read(retirement_path)
        if manifest["status"] != "COMPLETE" or manifest["completed_optimizer_steps"] != 16:
            raise RuntimeError("only the completed plain terminal state may retire moments")
        original = state / "manifest.before_optimizer_retirement.json"
        if not original.exists():
            shutil.copy2(manifest_path, original)
        if retirement_path.exists():
            retirement = read(retirement_path)
        else:
            files = []
            for entry in manifest["entries"]:
                for kind in ("exp_avg", "exp_avg_sq"):
                    name = entry["files"].get(kind)
                    if name is None:
                        continue
                    path = state / name
                    if (
                        Path(name).name != name
                        or not name.startswith(kind + "-")
                        or not name.endswith(".pt")
                        or path.is_symlink()
                        or not path.is_file()
                    ):
                        raise RuntimeError(
                            "only this run's regular Adam moment shards may be retired"
                        )
                    files.append(dict(filename=name, bytes=path.stat().st_size))
            retirement = dict(
                status="RETIRING",
                reason="plain terminal16 update evaluation complete; space for exact",
                files=files,
                planned_bytes=sum(f["bytes"] for f in files),
                original_manifest=str(original),
                lossless_training_resume_available=False,
            )
            save(retirement_path, retirement)
        for entry in retirement["files"]:
            path = state / entry["filename"]
            if path.is_symlink():
                raise RuntimeError("moment retirement refuses symlinks")
            if path.exists():
                path.unlink()
        retirement.update(status="COMPLETE", released_bytes=retirement["planned_bytes"])
        save(retirement_path, retirement)
        manifest.update(
            status=RETIRED,
            optimizer_moments_retired=True,
            original_checkpoint_bytes=manifest.get("bytes"),
            lossless_training_resume_available=False,
            retirement_record=str(retirement_path),
        )
        save(manifest_path, manifest)
        result.update(
            optimizer_moments_retired=True,
            lossless_training_resume_available=False,
            optimizer_moment_retirement=retirement,
        )
        save(directory / "rloo_results.json", result)
        completion_path = directory / "completion.json"
        completion = read(completion_path)
        completion.update(
            optimizer_moments_retired=True,
            lossless_training_resume_available=False,
            optimizer_moment_retirement=retirement,
        )
        save(completion_path, completion)
        self.event(
            "PLAIN_TERMINAL_OPTIMIZER_MOMENTS_RETIRED", released_bytes=retirement["released_bytes"]
        )
        return retirement

    def execute(self):
        if self.state_path.exists():
            if not self.cli.resume_completed_stages:
                raise FileExistsError("existing phase requires explicit resume")
            old = read(self.state_path)
            if old["status"] not in {"FAILED", "PAUSED"}:
                raise RuntimeError("only failed/paused suite can resume")
            self.attempt = old.get("controller_attempt", 1) + 1
            snapshot(
                self.output,
                self.attempt - 1,
                (
                    f"{self.prefix}_state.json",
                    f"{self.prefix}_results.json",
                    f"{self.prefix}_completion.json",
                ),
            )
            self.state["controller_attempt"] = self.attempt
        if (
            not self.cli.collect_only
            and (self.output / "initial_state.json").exists()
            and not (self.cli.resume_collected or self.cli.resume_completed_stages)
        ):
            raise RuntimeError("use --resume-collected to continue the collected shared initial")
        holder = self.cli.holder_directory or Path(
            read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"]
        )
        self.config = gpu.config_for_holder(
            holder,
            lease_token=f"rloo-local-{uuid.uuid4().hex}",
            evidence_directory=self.output / f"{self.prefix}_gpu_attempt{self.attempt}",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(
            self.output / f"{self.prefix}_gpu_config_attempt{self.attempt}.json", self.config
        )
        self.persist()

        def forward(signum, _frame):
            runner = self.active or self
            if runner.child is not None:
                runner.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
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
            initial_path = self.fresh_collection_path(self.output / "common_initial", self.attempt)
            self.initial = self.collect_batch(
                self,
                "common_initial",
                MODEL,
                initial_path,
                0,
                local_credit=True,
                source=self.source,
            )
            if initial_path != self.output / "common_initial":
                raise RuntimeError("shared initial path must remain common_initial")
            self.result["baseline"] = self.initial
            self.persist()
            if self.cli.collect_only:
                self.result["status"] = "COLLECTED"
                self.update(status="COMPLETE", current_stage=None)
                return 0
            for arm in ARMS:
                previous = self.output / arm / "rloo_results.json"
                if previous.exists() and read(previous)["status"] == "COMPLETE":
                    if not self.cli.resume_completed_stages:
                        raise FileExistsError("completed arm requires explicit suite resume")
                    self.result["arms"][arm].update(status="COMPLETE", reused_complete=True)
                else:
                    self.active = ArmRun(self, arm)
                    self.result["arms"][arm]["status"] = "RUNNING"
                    self.persist()
                    self.active.execute()
                    self.result["arms"][arm].update(status="COMPLETE", result_path=str(previous))
                if arm == "plain":
                    self.result["plain_optimizer_moment_retirement"] = self.retire_plain_moments()
                self.persist()
            self.state.update(status="COMPLETE", current_arm=None, current_stage=None)
            self.result["status"] = "COMPLETE"
            return 0
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            self.state.update(status="FAILED", error=message)
            self.result.update(status="FAILED", error=message)
            if self.active is not None:
                self.result["arms"][self.active.arm]["status"] = self.active.result["status"]
            return 1
        finally:
            self.stop_child()
            if self.active is not None:
                self.active.stop_child()
            self.persist()
            completed = [
                read(self.output / a / "completion.json")
                for a in ARMS
                if (self.output / a / "completion.json").exists()
            ]
            save(
                self.output / f"{self.prefix}_completion.json",
                dict(
                    status=self.state["status"],
                    planned_new_updates=0 if self.cli.collect_only else 32,
                    completed_new_updates=sum(c["completed_updates"] for c in completed),
                    completed_arms=[c["arm"] for c in completed if c["status"] == "COMPLETE"],
                    rloo_reference_reused=True,
                    rloo_reference=str(RLOO),
                    shared_initial_generation_cost={
                        k: self.initial.get(k, 0)
                        for k in ("model_requests", "prompt_tokens", "response_tokens")
                    }
                    if self.initial
                    else None,
                    shared_initial_parent_trajectories_reused=True,
                    training_state_is_host_reboot_volatile=False,
                    plain_optimizer_moments_retire_only_after_successful_terminal_evaluation=True,
                    costs_scope=(
                        "reused RLOO parents/reference excluded from new GPU generation; "
                        "shared initial branches counted once"
                    ),
                ),
            )
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
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--resume-collected", action="store_true")
    parser.add_argument("--resume-completed-stages", action="store_true")
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--stage-timeout-seconds", type=int, default=3600)
    raise SystemExit(Suite(parser.parse_args()).execute())


if __name__ == "__main__":
    main()
