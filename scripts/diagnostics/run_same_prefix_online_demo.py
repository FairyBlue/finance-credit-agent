#!/usr/bin/env python3
"""Small three-arm online head-GAE demo, run inside its own systemd service."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

import run_same_prefix_training_demo as gpu

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
MODEL = ROOT / "runs/m2-order-recovery-sft-20261011-v1/merged_model"
DEFAULT = ROOT / "runs/m2-same-prefix-online-demo-20261024-v2"
EIGHT_WINDOW_RUN = ROOT / "runs/m2-same-prefix-online-8window-demo-20261024-v1"
EPISODES = [f"formal-train-{i:04d}" for i in (1, 9, 65, 73)]
COLLECTOR = "src.agents.same_prefix_online_collect"
TRAINER = "src.agents.same_prefix_online_train"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collection(path: Path, model: Path, episodes: list[str], seed: int) -> dict:
    result = read(path / "collection_summary.json")
    if result.get("status") not in ("COMPLETE", "PASS"):
        raise RuntimeError("collection has no completed summary")
    actual_model = result.get("model_path", result.get("model"))
    if (
        actual_model is None
        or Path(actual_model).resolve() != model.resolve()
        or result["episode_ids"] != episodes
        or result["seed"] != seed
    ):
        raise RuntimeError("existing collection is not this actor/window/seed batch")
    return result


class Controller:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.phase = "initial" if args.collect_only else "online"
        self.state_path = self.output / f"{self.phase}_state.json"
        self.state = {
            "status": "RUNNING",
            "phase": self.phase,
            "stages": [],
            "model": str(args.model.resolve()),
            "seed": args.seed,
            "episode_ids": args.episode_ids,
            "alpha": args.alpha,
            "eta": args.eta,
            "rounds": 2,
            "arms": ["zero", "plain", "exact"],
            "retain_final_actors_only": getattr(args, "retain_final_actors_only", False),
        }
        self.child = None
        self.config = None
        self.claimed = False
        self.started = time.monotonic()

    def update(self, **values):
        self.state.update(values, elapsed_seconds=time.monotonic() - self.started)
        save(self.state_path, self.state)

    def event(self, event_name, **values):
        print(json.dumps({"event": event_name, **values}, allow_nan=False), flush=True)

    def command(self, name: str, module: str, arguments: list[str]) -> float:
        logs = self.output / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{name}.log"
        command = [str(PYTHON), "-u", "-m", module, *arguments]
        record = {
            "name": name,
            "command": command,
            "log": str(log),
            "status": "RUNNING",
            "started_monotonic": time.monotonic(),
        }
        self.state["stages"].append(record)
        self.update(current_stage=name)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            p
            for p in (
                str(ROOT),
                str(ROOT / "third_party/agent-r1"),
                environment.get("PYTHONPATH", ""),
            )
            if p
        )
        environment.update(
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            CUDA_VISIBLE_DEVICES=str(self.config["cuda_device"]),
        )
        start = time.monotonic()
        with log.open("x") as stream:
            self.child = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            try:
                gpu.request(
                    self.config["directory"],
                    {
                        "op": "attach",
                        "lease_token": self.config["lease_token"],
                        "workers": [gpu.process_identity(self.child.pid)],
                        "request_id": f"online-attach-{uuid.uuid4().hex}",
                    },
                    timeout=getattr(self.args, "coordinator_request_timeout_seconds", 15),
                )
            except (FileNotFoundError, ProcessLookupError):
                if self.child.poll() is None:
                    raise
            self.event("STAGE_STARTED", name=name, pid=self.child.pid, log=str(log))
            returncode = self.child.wait(timeout=self.args.stage_timeout_seconds)
        record.update(
            status="COMPLETE" if returncode == 0 else "FAILED",
            returncode=returncode,
            seconds=time.monotonic() - start,
        )
        self.update()
        self.child = None
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        self.event("STAGE_COMPLETE", name=name, seconds=record["seconds"])
        return record["seconds"]

    def stop_child(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        self.child = None

    def collect(
        self, name: str, model: Path, path: Path, round_index: int, parents_only: bool = False
    ) -> dict:
        args = [
            "--model",
            str(model),
            "--output",
            str(path),
            "--episode-ids",
            *self.args.episode_ids,
            "--seed",
            str(self.args.seed),
            "--round",
            str(round_index),
        ]
        if parents_only:
            args.append("--parents-only")
        self.command(name, COLLECTOR, args)
        return collection(path, model, self.args.episode_ids, self.args.seed)

    def initial(self) -> tuple[Path, dict]:
        path = self.output / "common_initial"
        if (path / "collection_summary.json").exists():
            result = collection(path, self.args.model, self.args.episode_ids, self.args.seed)
            if result["parents_only"] is not False or result["round_index"] != 0:
                raise RuntimeError("only the complete shared initial head/EST batch is reusable")
            self.event("REUSE_COMPLETE_COMMON_INITIAL", path=str(path))
        else:
            result = self.collect("common_initial", self.args.model, path, 0)
        self.update(shared_initial_collection=str(path))
        return path, result

    def train(self, arm: str, number: int, data: Path, actor: Path, critic: Path | None):
        path = self.output / arm / f"round{number}" / "train"
        args = [
            "--data",
            str(data),
            "--output",
            str(path),
            "--model",
            str(actor),
            "--arm",
            "S0" if arm == "zero" else arm,
            "--seed",
            str(self.args.seed),
            "--alpha",
            str(self.args.alpha),
            "--eta",
            str(self.args.eta),
            "--actor-learning-rate",
            str(self.args.actor_lr),
            "--critic-learning-rate",
            str(self.args.critic_lr),
        ]
        if critic is not None:
            args += ["--critic", str(critic)]
        retire_inputs = self.state["retain_final_actors_only"] and number == 2
        if retire_inputs:
            args.append("--retire-input-checkpoints")
        self.command(f"{arm}_round{number}_train", TRAINER, args)
        result = read(path / "training_summary.json")
        if result["status"] not in ("PASS", "COMPLETE"):
            raise RuntimeError("trainer did not complete its actor and critic update")
        new_actor, new_critic = path / "actor", path / "critic"
        if not all(p.is_dir() for p in (new_actor, new_critic)):
            raise RuntimeError("trainer actor/critic HF checkpoint missing")
        if result["actor_updates"] != 1 or result["critic_updates"] != 1:
            raise RuntimeError("one actor and one critic update required per round")
        if critic is not None:
            expected = self.output / arm / "round1" / "train" / "critic"
            if critic != expected or critic.is_symlink():
                raise RuntimeError("refusing to prune any non-demo critic")
            if retire_inputs:
                retired = result.get("retired_input_checkpoints")
                expected_actor = self.output / arm / "round1/train/actor"
                if (
                    actor != expected_actor
                    or retired != [str(critic), str(actor)]
                    or critic.exists()
                    or actor.exists()
                ):
                    raise RuntimeError(
                        "trainer did not receipt retirement of exactly its r1 inputs"
                    )
                self.state.setdefault("trainer_input_retirements", []).append(
                    {"arm": arm, "round": number, "receipt": retired}
                )
            else:
                if not critic.is_dir():
                    raise RuntimeError("previous critic disappeared without input retirement")
                shutil.rmtree(critic)
                self.state.setdefault("pruned_own_intermediate_critics", []).append(str(critic))
        return new_actor, new_critic, result

    def retire_final_critic(self, arm: str, critic: Path) -> dict:
        expected = self.output / arm / "round2" / "train" / "critic"
        if (
            critic != expected
            or critic.is_symlink()
            or not critic.is_dir()
            or not critic.resolve().is_relative_to(self.output)
        ):
            raise RuntimeError("refusing to prune any non-demo final critic")
        files = [p for p in critic.rglob("*") if p.is_file()]
        receipt = {
            "arm": arm,
            "path": str(critic),
            "reason": "completed_final_eval",
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        }
        shutil.rmtree(critic)
        self.state.setdefault("retired_own_final_critics", []).append(receipt)
        self.update()
        return receipt

    def execute(self):
        if self.state_path.exists():
            raise FileExistsError("new controller phase required; no implicit retry")
        holder = self.args.holder_directory
        if holder is None:
            holder = Path(read(ROOT / "runs/GPU_RESERVATION_CURRENT.json")["directory"])
        self.config = gpu.config_for_holder(
            holder,
            lease_token=f"online-demo-{uuid.uuid4().hex}",
            evidence_directory=self.output / f"{self.phase}_gpu_coordinator",
            portfolio_target_bytes=76 * 1024**3,
        )
        gpu.freeze_config(self.output / f"{self.phase}_gpu_config.json", self.config)
        self.update()

        def forward(signum, _frame):
            self.state["stop_signal"] = signum
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            state = gpu.controller_claim(self.config)
            self.claimed = True
            gpu.request(
                str(holder),
                {
                    "op": "release",
                    "lease_token": self.config["lease_token"],
                    "bytes": state["maximum_bytes"],
                    "request_id": f"release-{uuid.uuid4().hex}",
                },
            )
            checked = gpu.reservation_status(self.config)
            if checked["paused"] is not True or any(
                checked[k]
                for k in ("payload_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
            ):
                raise RuntimeError("holder must be paused and empty before starting GPU work")
            initial_path, initial = self.initial()
            failed_arms = 0
            if not self.args.collect_only:
                results = []
                for arm in ("zero", "plain", "exact"):
                    result = {
                        "arm": arm,
                        "status": "RUNNING",
                        "alpha": self.args.alpha,
                        "eta": self.args.eta,
                        "retain_final_actors_only": self.state["retain_final_actors_only"],
                        "baseline": initial,
                        "after_round1": None,
                        "after_round2": None,
                        "round1_training": None,
                        "round2_training": None,
                        "round1_actor": None,
                        "final_actor": None,
                        "final_critic": None,
                        "failed_stage": None,
                        "error": None,
                    }
                    try:
                        actor, critic, first = self.train(
                            arm, 1, initial_path, self.args.model, None
                        )
                        result.update(round1_training=first, round1_actor=str(actor))
                        second_path = self.output / arm / "round2" / "collect"
                        result["after_round1"] = self.collect(
                            f"{arm}_round2_collect", actor, second_path, 1
                        )
                        actor, critic, second = self.train(arm, 2, second_path, actor, critic)
                        result.update(
                            round2_training=second, final_actor=str(actor), final_critic=str(critic)
                        )
                        result["after_round2"] = self.collect(
                            f"{arm}_final_eval", actor, self.output / arm / "final_eval", 2, True
                        )
                        if self.state["retain_final_actors_only"]:
                            result["final_critic_retirement"] = self.retire_final_critic(
                                arm, critic
                            )
                        result["status"] = "COMPLETE"
                    except Exception as error:
                        self.stop_child()
                        if self.state.get("stop_signal"):
                            raise
                        failed_arms += 1
                        result.update(
                            status="FAILED",
                            failed_stage=self.state.get("current_stage"),
                            error=f"{type(error).__name__}: {error}",
                        )
                        partial = self.output / arm / "round2/train/training_summary.json"
                        if result["round2_training"] is None and partial.exists():
                            result["round2_training"] = read(partial)
                        if self.state["stages"] and self.state["stages"][-1]["status"] == "RUNNING":
                            stage = self.state["stages"][-1]
                            stage.update(
                                status="FAILED",
                                error=result["error"],
                                seconds=time.monotonic()
                                - stage.get("started_monotonic", time.monotonic()),
                            )
                        self.event("ARM_FAILED_CONTINUING", arm=arm, error=result["error"])
                    results.append(result)
                    save(
                        self.output / "results.json",
                        {"results": results, "common_initial_executed_once": True},
                    )
                write_results(self.output, results, self.state["stages"])
            self.update(
                status="COMPLETE_WITH_FAILURES" if failed_arms else "COMPLETE",
                failed_arms=failed_arms,
                current_stage=None,
            )
            return 1 if failed_arms else 0
        except BaseException as error:
            self.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.stop_child()
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                    self.update(holder_restore="REFILLING_76_GIB")
                except (OSError, RuntimeError, ValueError) as error:
                    self.update(
                        holder_restore="DEFERRED_TO_EMPTY_CGROUP_WATCHDOG",
                        restore_reason=str(error),
                    )
            for s, handler in handlers.items():
                signal.signal(s, handler)


def write_results(output: Path, results: list[dict], stages: list[dict]):
    import csv

    def mean(summary):
        if summary is None:
            return None
        values = summary["parent_terminal_Y"]
        count = len(summary["episode_ids"])
        if (
            summary["parents_failed"]
            or count not in (4, 8)
            or len(values) != count
            or any(v is None for v in values)
        ):
            return None
        return math.fsum(values) / len(values)

    def net_return(summary):
        if summary is None or mean(summary) is None:
            return None
        metrics = summary.get("parent_terminal_metrics")
        if (
            not metrics
            or len(metrics) != len(summary["episode_ids"])
            or any(m is None for m in metrics)
        ):
            return None
        return math.fsum(m["raw_net_return"] for m in metrics) / len(metrics)

    def training_metric(result, number, key):
        summary = result[f"round{number}_training"]
        path = output / result["arm"] / f"round{number}" / "train/training_summary.json"
        if summary is None and path.exists():
            summary = read(path)
        if summary is None:
            return None
        if key == "order_diverse_groups":
            groups = summary.get("coverage", {}).get("groups")
            return sum(g.get("order_gate") is True for g in groups) if groups is not None else None
        return summary.get(key)

    rows = [
        {
            "arm": r["arm"],
            "status": r["status"],
            "failed_stage": r["failed_stage"],
            "error": r["error"],
            "alpha": r["alpha"],
            "eta": r["eta"],
            "baseline_parent_mean_Y": mean(r["baseline"]),
            "round1_parent_mean_Y": mean(r["after_round1"]),
            "round2_parent_mean_Y": mean(r["after_round2"]),
            "baseline_mean_raw_net_return": net_return(r["baseline"]),
            "round1_mean_raw_net_return": net_return(r["after_round1"]),
            "round2_mean_raw_net_return": net_return(r["after_round2"]),
            **{
                f"round{number}_{key}": training_metric(r, number, key)
                for number in (1, 2)
                for key in (
                    "supported_heads",
                    "actor_updates",
                    "critic_updates",
                    "order_diverse_groups",
                )
            },
            "model_stage_wall_seconds": math.fsum(
                s.get("seconds", 0) for s in stages if s["name"].startswith(r["arm"] + "_")
            ),
            "baseline_failed": r["baseline"]["parents_failed"],
            "round1_failed": r["after_round1"]["parents_failed"] if r["after_round1"] else None,
            "round2_failed": r["after_round2"]["parents_failed"] if r["after_round2"] else None,
            "round1_actor": r["round1_actor"],
            "round1_actor_retained": Path(r["round1_actor"]).is_dir()
            if r["round1_actor"]
            else None,
            "final_actor": r["final_actor"],
            "final_critic_retained": Path(r["final_critic"]).is_dir()
            if r["final_critic"]
            else None,
        }
        for r in results
    ]
    with (output / "results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    initial = read(output / "common_initial/collection_summary.json")
    collections, missing_cost = [initial], []
    for stage in stages:
        if stage["command"][3] != COLLECTOR or stage["name"] == "common_initial":
            continue
        command = stage["command"]
        path = Path(command[command.index("--output") + 1]) / "collection_summary.json"
        if path.exists() and read(path).get("status") == "COMPLETE":
            collections.append(read(path))
        else:
            missing_cost.append(stage["name"])
    known_cost = {
        k: sum(c[k] for c in collections)
        for k in ("model_requests", "prompt_tokens", "response_tokens")
    }
    failed = sum(r["status"] != "COMPLETE" for r in results)
    save(
        output / "demo_summary.json",
        {
            "status": "COMPLETE_WITH_FAILURES" if failed else "COMPLETE",
            "failed_arms": failed,
            "results": rows,
            "shared_initial_collection_executed_once": True,
            "retain_final_actors_only": any(
                r.get("retain_final_actors_only", False) for r in results
            ),
            "actual_collection_totals": known_cost if not missing_cost else None,
            "known_completed_collection_totals": known_cost,
            "failed_collection_stages_without_complete_cost": missing_cost,
            "stage_seconds_this_controller": sum(s.get("seconds", 0) for s in stages),
            "limits": [
                f"Train demo only; {len(initial['episode_ids'])} reused windows, "
                "one seed, two updates per arm.",
                "Head actor/critic updates with gamma=lambda=1; not a full StepPO reproduction.",
                "Fresh Adam each round; no optimizer-state continuation or formal Valid/OOS.",
                "Costs include common collection once; baseline metrics are shared.",
                "Reported stage times are process wall time including loading/export, "
                "not GPU kernel time.",
                "No failure labels imputed; incomplete parent evaluation means stay NA.",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--holder-directory", type=Path)
    parser.add_argument("--episode-ids", nargs="+", default=EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--actor-lr", type=float, default=2e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-6)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--stage-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--retain-final-actors-only",
        action="store_true",
        help="low-disk seed43 or approved eight-window seed42 run: retain only final actors",
    )
    args = parser.parse_args()
    args.model = args.model.resolve()
    if len(args.episode_ids) not in (4, 8) or len(set(args.episode_ids)) != len(args.episode_ids):
        parser.error("demo requires four or eight distinct Train windows")
    eight_window_retention = (
        args.seed == 42 and len(args.episode_ids) == 8 and args.output.resolve() == EIGHT_WINDOW_RUN
    )
    if args.retain_final_actors_only and args.seed != 43 and not eight_window_retention:
        parser.error("final-actor-only retention requires seed43 or the approved eight-window run")
    raise SystemExit(Controller(args).execute())


if __name__ == "__main__":
    main()
