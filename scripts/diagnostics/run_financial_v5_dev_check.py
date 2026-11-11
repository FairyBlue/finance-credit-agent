#!/usr/bin/env python3
"""One frozen V5 greedy Dev100 evaluation; no training or new controls."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

import run_financial_coverage as coverage
from run_financial_learning import BudgetStop, digest, metrics
from run_financial_mc import generation_tokens
from run_mandate_composition import Controller as CompositionController
from run_mandate_composition import rows
from run_mandate_repair import COLLECTOR, ROOT, gpu, now, read, save

DEFAULT = ROOT / "runs/agentic-rl-v5-dev-check-20261110"
PARENT = ROOT / "runs/agentic-rl-invalid-feedback-20261110-v5"
REFERENCE = ROOT / "runs/agentic-rl-learning-20261030-v2/eval/sft_dev"
ADAPTER = PARENT / "training/calculate_and_parse_failures/round_005/train/adapter"
CAP, SECONDS = 3_200_000, 900
IDENTITY = (
    "data_sha256",
    "seed",
    "temperature",
    "repetitions",
    "max_context_tokens",
    "max_response_tokens",
    "environment",
    "decode_mode",
    "observation_profile",
    "action_protocol",
    "engine_mode",
    "base_model_path",
)


def indexed(values):
    result = {v["task_id"]: v for v in values}
    if len(result) != 100 or len(values) != 100:
        raise ValueError("100 unique task outcomes required")
    if any(v.get("status") != "COMPLETE" or v.get("replicate_index") != 0 for v in values):
        raise ValueError("100 complete greedy rep0 outcomes required")
    return result


def prepare(output):
    from src.agents.mandate_repair_train import adapter_digest

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parent = read(PARENT / "state.json")
    ref = read(REFERENCE / "summary.json")
    data = Path(ref["data_path"])
    tasks = rows(data)
    old = indexed(rows(REFERENCE / "trajectory_results.jsonl"))
    if parent["status"] != "COMPLETE" or ref["status"] != "COMPLETE":
        raise RuntimeError("completed V5 and SFT reference required")
    if len(tasks) != 100 or set(old) != {t["task_id"] for t in tasks}:
        raise RuntimeError("fixed Dev100 tasks required")
    if digest(data) != ref["data_sha256"] or ref["successes"] != 82:
        raise RuntimeError("SFT Dev reference identity changed")
    if parent["costs"]["total_new_generation_tokens"] + CAP > 9_000_000:
        raise BudgetStop("V5 and this evaluation must remain within the original9M total")
    rules = dict(
        profile="v5_fixed_dev100_readout",
        data=str(data),
        data_sha256=digest(data),
        adapter=str(ADAPTER),
        adapter_sha256=adapter_digest(ADAPTER),
        reference=str(REFERENCE),
        reference_outcomes_sha256=digest(REFERENCE / "trajectory_results.jsonl"),
        identity={k: ref[k] for k in IDENTITY},
        parent_costs=parent["costs"],
        cap=CAP,
        gpu_stage_seconds_cap=SECONDS,
        actor_training=False,
        reserved_test_read=False,
        historical_market_oos_read=False,
    )
    contract = output / "contract.json"
    if contract.exists() and read(contract) != rules:
        raise RuntimeError("frozen Dev check contract changed")
    if not contract.exists():
        save(contract, rules)
    return rules


class Controller(coverage.Controller):
    def __init__(self, output, resume=False):
        self.contract = prepare(output)
        CompositionController.__init__(self, Path(output), resume)
        # Local process limit; the historical controller source is unchanged.
        coverage.GPU_SECONDS = SECONDS
        self.persist(
            experiment="v5_fixed_dev100_readout",
            actor_training=False,
            reserved_test_access=False,
            contract_sha256=digest(self.output / "contract.json"),
        )
        frozen = self.output / "frozen_sources"
        frozen.mkdir(exist_ok=True)
        (frozen / f"controller_attempt{self.attempt}.py").write_bytes(Path(__file__).read_bytes())

    def costs(self):
        own = generation_tokens(self.output / "eval")
        seconds = sum(
            s.get("seconds", 0) for s in self.state.get("stages", []) if s.get("gpu_worker")
        )
        if own > CAP:
            raise BudgetStop("3.2M fixed evaluation cap exceeded")
        combined = self.contract["parent_costs"]["total_new_generation_tokens"] + own
        if combined > 9_000_000:
            raise BudgetStop("V5 plus Dev check9M cap exceeded")
        return dict(
            total_new_generation_tokens=own,
            gpu_stage_seconds=seconds,
            combined_v5_and_check_generation_tokens=combined,
            failed_requests_included=True,
            training_generation_tokens=0,
        )

    def stage(self, name, module, arguments, gpu_worker=True):
        if module != COLLECTOR:
            raise RuntimeError("Dev check only permits the collector; training is forbidden")
        return super().stage(name, module, arguments, gpu_worker)

    def report(self):
        result = dict(
            status=self.state["status"],
            current_stage=self.state.get("current_stage"),
            reference_successes=82,
            costs=self.costs(),
            comparison=self.state.get("comparison"),
        )
        destination = self.output / "eval/v5_dev"
        if (destination / "summary.json").exists():
            result["evaluation"] = metrics(destination)
        save(self.output / "readout.json", result)
        lines = [
            "# V5固定Dev100终点评估",
            "",
            f"状态：**{result['status']}**。",
            "",
            "仅采样一次固定V5终点；SFT82/100读数复用，没有新训练。",
            "",
            "| 模型 | 完整成功 /100 | 指标正确 |",
            "|---|---:|---:|",
            "| 原SFT（复用） | 82 | 82 |",
        ]
        if "evaluation" in result:
            v = result["evaluation"]
            lines.append(
                f"| V5 | {v['successes']}（完成{v['complete']}/100） | {v['metric_correct']} |"
            )
        lines += [
            "",
            f"新增输入＋输出：{result['costs']['total_new_generation_tokens']:,} token；"
            f"GPU阶段{result['costs']['gpu_stage_seconds']:.1f}秒。",
            "预算320万/15分钟；与原V5合计不超过900万token。旧Dev仅作底座选择的开发读数，非独立确认。",
            "",
            "[实际状态](state.json) · [完整读数](readout.json) · [冻结合同](contract.json)",
            "",
        ]
        (self.output / "REPORT.md").write_text("\n".join(lines))

    def run(self):
        def forward(signum, _frame):
            if self.child is not None and self.child.poll() is None:
                self.child.send_signal(signum)
            else:
                raise KeyboardInterrupt(f"signal {signum}")

        handlers = {s: signal.signal(s, forward) for s in (signal.SIGTERM, signal.SIGINT)}
        code = 0
        try:
            self.claim_gpu()
            destination = self.output / "eval/v5_dev"
            destination.parent.mkdir(parents=True, exist_ok=True)
            _, complete = self.collection(
                "v5_dev100",
                Path(self.contract["data"]),
                ADAPTER,
                destination,
                repetitions=1,
                temperature=0,
                allowance=CAP - generation_tokens(self.output / "eval"),
                seed=43,
            )
            if not complete:
                raise BudgetStop("fixed100task denominator incomplete")
            summary = read(destination / "summary.json")
            if any(summary.get(k) != v for k, v in self.contract["identity"].items()):
                raise RuntimeError(
                    "Dev evaluation configuration differs from the reused SFT reference"
                )
            new = indexed(rows(destination / "trajectory_results.jsonl"))
            old = indexed(rows(REFERENCE / "trajectory_results.jsonl"))
            if set(new) != set(old):
                raise RuntimeError("Dev task identities differ")
            comparison = dict(
                success_difference=sum(v["success"] for v in new.values()) - 82,
                success_increased_tasks=[
                    t for t in sorted(new) if new[t]["success"] and not old[t]["success"]
                ],
                success_decreased_tasks=[
                    t for t in sorted(new) if old[t]["success"] and not new[t]["success"]
                ],
                development_readout_only=True,
            )
            self.persist(
                status="COMPLETE",
                current_stage=None,
                comparison=comparison,
                finished_at=now(),
                costs=self.costs(),
            )
            save(
                self.output / "decision.json",
                dict(
                    status="COMPLETE",
                    comparison=comparison,
                    costs=self.costs(),
                    next_action="CHOOSE_COMMON_INITIALIZATION_WITHOUT_STARTING_NEW_TRAINING",
                ),
            )
        except BudgetStop as error:
            self.persist(status="RESOURCE_LIMIT", error=str(error), finished_at=now())
        except BaseException as error:
            code = 1
            self.persist(
                status="FAILED", error=f"{type(error).__name__}: {error}", finished_at=now()
            )
        finally:
            if self.claimed:
                try:
                    gpu.controller_idle(self.config)
                except Exception as error:
                    self.state["holder_restore_error"] = str(error)
            for s, handler in handlers.items():
                signal.signal(s, handler)
            self.persist()
            self.report()
            save(self.output / "completion.json", self.state)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output)
        print(args.output / "contract.json")
        return
    raise SystemExit(Controller(args.output, args.resume).run())


if __name__ == "__main__":
    main()
