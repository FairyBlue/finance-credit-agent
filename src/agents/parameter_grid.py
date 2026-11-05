"""Fixed two-seed Train parameter grid, using one durable serial controller.

The root contract seals both child experiments before any training. A started
root is never resumed or retried; a failed child stops the entire fixed batch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.agents import learning_restart as restart
from src.agents import low_memory_pilot as runner
from src.agents.screening_launcher import atomic_json

NAME = "common_sft_train_parameter_grid_v2"
PLAN = "docs/2026-10-06_M2_PARAMETER_GRID_EXECUTION_R1.md"
DEFAULT = runner.REPO_ROOT / "runs/m2-parameter-grid-20261006-r1"
SEEDS = (42, 43)
TRAIN_NUMBERS = (1, 65, 9, 73, 17, 149, 25, 157)
EVAL_NUMBERS = (210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 441)
EVALUATION_STEPS = (2, 4)
MODEL = runner.REPO_ROOT / "runs/m2-coldstart-v3-20260914/merged_model"
MERGE_RECEIPT_SHA256 = "cf3b8e5a4e2e45708943f4c71c822d41d60cfae13170e760cb5979e1df2356db"
ROOT_INPUTS = ("grid_settings.json", "source_identity.json", "plan.md")


def _profile(seed):
    rl_cells = (
        runner.CellSpec("m1_lr1e7", "M1", 0.0, 1e-6, 4),
        runner.CellSpec("m2_lr1e7_eta0125", "M2", 0.125, 1e-6, 4),
        runner.CellSpec("m2_lr1e7_eta025", "M2", 0.25, 1e-6, 4),
        runner.CellSpec("m1_lr2e7", "M1", 0.0, 1e-6, 4),
        runner.CellSpec("m2_lr2e7_eta0125", "M2", 0.125, 1e-6, 4),
        runner.CellSpec("m2_lr2e7_eta025", "M2", 0.25, 1e-6, 4),
    )
    specs = (runner.CellSpec("sft0", "M1", 0.0, 1e-6, 0),) + (
        rl_cells if seed == 42 else tuple(reversed(rl_cells))
    )
    return runner.ExperimentProfile(
        name=f"common_sft_parameter_grid_seed{seed}_v2",
        plan=PLAN,
        train_numbers=TRAIN_NUMBERS,
        eval_numbers=EVAL_NUMBERS,
        evaluation_steps=EVALUATION_STEPS,
        rounds=4,
        include_sft=True,
        retire_completed_moments=True,
        archive_completed_actors=True,
        evaluation_incomplete_mode="retain_prefix_v1",
        seed=seed,
        actor_learning_rate=2e-7,
        cell_specs=specs,
        cell_actor_learning_rates=tuple(
            (spec.cell_id, 1e-7 if "lr1e7" in spec.cell_id else 2e-7)
            for spec in specs
        ),
    )


PROFILES = {seed: _profile(seed) for seed in SEEDS}
PROFILE_SEED42 = PROFILES[42]
PROFILE_SEED43 = PROFILES[43]


def profile_for_seed(seed):
    if type(seed) is not int or seed not in PROFILES:
        raise ValueError("parameter grid seed must be 42 or 43")
    return PROFILES[seed]


def settings():
    """Return the fixed root identity; no adaptive ordering or budget extension."""
    return dict(
        profile=NAME,
        plan=PLAN,
        seeds=list(SEEDS),
        profiles={str(seed): asdict(profile) for seed, profile in PROFILES.items()},
        seed_directories={str(seed): f"seed{seed}" for seed in SEEDS},
        planned_trajectories=sum(p.planned_trajectories for p in PROFILES.values()),
        rl_cells=12,
        sft_baselines=2,
        execution="serial_one_shot_v1",
        automatic_retry=False,
        resume_supported=False,
        selection_scope="original_train_only",
        eta_candidates=[0.125, 0.25],
        actor_learning_rates=[1e-7, 2e-7],
        critic_learning_rate=1e-6,
        reserved_seed44_started=False,
        actor_retention="verified_lossless_tensor_xor_v1",
        evaluation_incomplete_mode="retain_prefix_v1",
        preceding_failed_run="runs/m2-parameter-grid-20261006",
        preceding_attempted_trajectories=50,
        cumulative_planned_attempts=1994,
    )


def _same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def child_directory(base, seed):
    profile_for_seed(seed)
    return base / f"seed{seed}"


def prepare(base, *, model_path=MODEL, merge_receipt_sha256=MERGE_RECEIPT_SHA256,
            coordinated_reservation_directory=None):
    if base.exists():
        raise RuntimeError("parameter grid requires a new root directory")
    base.mkdir(parents=True, exist_ok=False)
    atomic_json(base / "grid_settings.json", settings())
    from uuid import uuid4

    coordinator_token = uuid4().hex if coordinated_reservation_directory is not None else None
    for seed, profile in PROFILES.items():
        runner.prepare(
            child_directory(base, seed), model_path=model_path,
            merge_receipt_sha256=merge_receipt_sha256,
            reservation_directory=None, profile=profile,
            **({"coordinated_reservation_directory": coordinated_reservation_directory,
                "coordinator_token": coordinator_token}
               if coordinated_reservation_directory is not None else {}),
        )
    refresh_status(base)


async def input_preflight(base):
    if not _same(restart.read(base / "grid_settings.json"), settings()):
        raise RuntimeError("parameter grid settings differ from the fixed plan")
    for seed, profile in PROFILES.items():
        await runner.input_preflight(child_directory(base, seed), profile=profile)


def seal(base):
    if (base / "run_contract.json").exists() or (base / "state.json").exists():
        raise RuntimeError("parameter grid already sealed or started")
    if not _same(restart.read(base / "grid_settings.json"), settings()):
        raise RuntimeError("parameter grid settings differ from the fixed plan")
    for seed, profile in PROFILES.items():
        runner.seal(child_directory(base, seed), profile=profile)
    shutil.copy2(runner.REPO_ROOT / PLAN, base / "plan.md")
    source_identity = {}
    for relative, source in runner.source_files().items():
        target = base / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_identity[relative] = restart.digest(source)
    atomic_json(base / "source_identity.json", source_identity)
    atomic_json(
        base / "run_contract.json",
        dict(
            profile=NAME, sealed_at_SGT=restart.now(),
            inputs={path: restart.digest(base / path) for path in ROOT_INPUTS},
            child_contracts={
                str(seed): restart.digest(child_directory(base, seed) / "run_contract.json")
                for seed in SEEDS
            },
            planned_trajectories=settings()["planned_trajectories"],
            automatic_retry=False, resume_supported=False,
        ),
    )
    check_contract(base)
    refresh_status(base)


def check_contract(base):
    contract = restart.read(base / "run_contract.json")
    if (
        contract.get("profile") != NAME
        or set(contract.get("inputs", {})) != set(ROOT_INPUTS)
        or set(contract.get("child_contracts", {})) != {str(seed) for seed in SEEDS}
        or contract.get("planned_trajectories") != settings()["planned_trajectories"]
        or contract.get("automatic_retry") is not False
        or contract.get("resume_supported") is not False
    ):
        raise RuntimeError("parameter grid contract identity or budget differs")
    if not _same(restart.read(base / "grid_settings.json"), settings()):
        raise RuntimeError("parameter grid settings differ from the fixed plan")
    for relative, digest in contract["inputs"].items():
        if restart.digest(base / relative) != digest:
            raise RuntimeError("sealed parameter grid input changed: " + relative)
    sources = restart.read(base / "source_identity.json")
    current = runner.source_files()
    if set(sources) != set(current):
        raise RuntimeError("parameter grid source inventory changed")
    for relative, source in current.items():
        digest = sources[relative]
        if (
            restart.digest(source) != digest
            or restart.digest(base / "sources" / relative) != digest
        ):
            raise RuntimeError("parameter grid source changed: " + relative)
    for seed, profile in PROFILES.items():
        child = child_directory(base, seed)
        if child.is_symlink():
            raise RuntimeError("parameter grid child directory may not be a symlink")
        if restart.digest(child / "run_contract.json") != contract["child_contracts"][str(seed)]:
            raise RuntimeError(f"sealed child contract changed: seed{seed}")
        runner.check_contract(child, profile=profile)


def _read_optional(path):
    return restart.read(path) if path.exists() else {}


def _landed_trajectories(cell):
    count = 0
    for path in (cell / "artifacts/screening_evidence").glob("*.jsonl"):
        with path.open() as stream:
            count += sum(bool(line.strip()) for line in stream)
    return count


def progress(base):
    state = _read_optional(base / "state.json")
    rows = []
    for seed, profile in PROFILES.items():
        child = child_directory(base, seed)
        child_state = _read_optional(child / "state.json")
        for spec in runner.cells(profile):
            cell = child / spec.cell_id
            summary = _read_optional(cell / "summary.json")
            planned = (len(profile.eval_ids) * 5 if spec.zero else
                       spec.rounds * 8 + len(profile.evaluation_steps) * len(profile.eval_ids) * 5)
            landed = _landed_trajectories(cell)
            rows.append(dict(
                seed=seed, cell_id=spec.cell_id,
                status=summary.get("status", "PENDING"),
                landed_trajectories=landed, planned_trajectories=planned,
                elapsed_seconds=summary.get("elapsed_seconds"),
                child_status=child_state.get("status", "PREPARED"),
                error=summary.get("error"),
                incomplete_evaluations=summary.get("incomplete_evaluations"),
                financial_status=summary.get("financial_status"),
            ))
    landed = sum(row["landed_trajectories"] for row in rows)
    planned = settings()["planned_trajectories"]
    complete_cells = sum(row["status"] == "COMPLETE" for row in rows)
    elapsed = None
    eta = None
    if state.get("started_at_SGT"):
        start = datetime.fromisoformat(state["started_at_SGT"])
        end = datetime.fromisoformat(state.get("finished_at_SGT") or restart.now())
        elapsed = max(0.0, (end - start).total_seconds())
        if 0 < landed < planned and elapsed > 0:
            eta = elapsed * (planned - landed) / landed
    return dict(
        at_SGT=restart.now(), status=state.get("status", "PREPARED"),
        current_seed=state.get("current_seed"),
        landed_trajectories=landed, planned_trajectories=planned,
        complete_cells=complete_cells, total_cells=len(rows),
        elapsed_seconds=elapsed, estimated_remaining_seconds=eta,
        cells=rows, error=state.get("error"),
        incomplete_evaluations=sum(row["incomplete_evaluations"] or 0 for row in rows),
        financial_status=state.get("financial_status", "PENDING"),
    )


def refresh_status(base):
    snapshot = progress(base)
    atomic_json(base / "progress.json", snapshot)
    current = next((row for row in snapshot["cells"] if row["status"] == "RUNNING"), None)
    status = snapshot["status"]
    child = child_directory(base, current["seed"]) if current else None
    queue = _read_optional(child / "gpu_queue_state.json") if child else {}
    waiting = current and queue.get("cell") == current["cell_id"] and (
        queue.get("status") == "WAITING_FOR_GPU"
    )
    lines = [
        "# 4B M1/M2 参数网格训练", "",
        f"当前进展（新加坡时间）：{snapshot['at_SGT']}", "",
        f"状态：**{'WAITING_FOR_GPU' if waiting else status}**；"
        f"当前 seed：{snapshot['current_seed'] or '尚未开始'}。",
        f"当前臂：{current['cell_id'] if current else '无运行臂'}。",
        f"已落盘轨迹 {snapshot['landed_trajectories']}/{snapshot['planned_trajectories']}；"
        f"已验收臂 {snapshot['complete_cells']}/{snapshot['total_cells']}。", "",
        f"已验收臂中的未完成评估：{snapshot['incomplete_evaluations']}；"
        f"金融覆盖状态：{snapshot['financial_status']}。", "",
        "固定预算：seed42/43；每 seed 一个未 RL 的 SFT 和六个四轮 RL 臂。",
        "Actor LR 1e-7/2e-7；每个 LR 对照 M1、M2 η=0.125/0.25；critic LR 1e-6。",
        "每臂 32 条训练；第 2/4 轮各评估 12 个 Train 窗口，greedy1 + sampled4。",
        "每 seed 共 972 条，全批 1944 条；只使用原 Train，未启动 seed44 或样本外验证。",
        "同一服务串行执行；网络或客户端断开不影响服务；无自动重跑或断点续训。", "",
        "R1 仅隔离真实 256-step 未完成评估，失败保留并单列统计；训练仍须完整。",
        "未完成评估没有有效终局收益；落盘数包含失败尝试，不能当成完整轨迹数。",
        "原失败批次另有 50 次尝试保留，R1 加原批次的计划总数为 1994。", "",
    ]
    if snapshot["elapsed_seconds"] is not None:
        lines.append(f"整批已运行 {snapshot['elapsed_seconds'] / 3600:.2f} 小时。")
    if snapshot["estimated_remaining_seconds"] is not None:
        lines += [
            f"按当前落盘速度粗估剩余 {snapshot['estimated_remaining_seconds'] / 3600:.1f} 小时。",
            "估时包含当前共享 GPU 等待；SFT 与 RL 阶段速度不同，早期估时波动较大。", "",
        ]
    for row in snapshot["cells"]:
        lines.append(
            f"- seed{row['seed']} / {row['cell_id']}：{row['status']}；"
            f"已落盘 {row['landed_trajectories']}/{row['planned_trajectories']}。"
        )
        if row["error"]:
            lines.append("  停止原因：" + row["error"])
        if row["incomplete_evaluations"]:
            lines.append(f"  未完成评估 {row['incomplete_evaluations']}；金融覆盖 INCOMPLETE。")
    if snapshot["error"]:
        lines += ["", "整批停止原因：" + snapshot["error"]]
    lines += [
        "", "落盘条数是进度读数；完整性和收益以验收后的 grid_readout.json 为准。",
        "结果用于检查训练稳定范围和下一步改进线索，不能替代正式性能验证。", "",
    ]
    path = base / "LIVE_STATUS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    if status in {"COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES", "FAILED"}:
        shutil.copy2(path, base / "FINAL_RESULTS.md")


def _monitor(base, stop):
    while not stop.is_set():
        try:
            refresh_status(base)
        except Exception as error:
            atomic_json(base / "monitor_error.json", dict(at_SGT=restart.now(), error=str(error)))
        stop.wait(30)


def _readout(base, *, allow_partial):
    from src.agents.parameter_grid_readout import write_readout

    return write_readout(base, allow_partial=allow_partial)


def after_cell(seed_base, spec, summary):
    """Archive only this batch's accepted actor, then publish partial diagnostics."""
    from src.agents.parameter_grid_archive import archive_completed_cell

    if not spec.zero:
        summary["actor_archive"] = archive_completed_cell(seed_base, spec, summary)
        atomic_json(seed_base / spec.cell_id / "summary.json", summary)
    _readout(seed_base.parent, allow_partial=True)


def run(base, *, post_cell=None):
    """Run exactly the sealed seed order once and preserve every failure."""
    if (base / "state.json").exists():
        raise RuntimeError("new parameter grid state required; no implicit retry or resume")
    check_contract(base)
    state = dict(
        profile=NAME, status="RUNNING", started_at_SGT=restart.now(),
        current_seed=None, completed_seeds=[],
        planned_trajectories=settings()["planned_trajectories"],
        automatic_retry=False, resume_supported=False,
    )
    atomic_json(base / "state.json", state)
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(base, stop), daemon=True)
    monitor.start()
    try:
        for seed, profile in PROFILES.items():
            check_contract(base)
            state["current_seed"] = seed
            atomic_json(base / "state.json", state)
            runner.run(
                child_directory(base, seed), profile=profile,
                post_cell=post_cell or after_cell,
            )
            child_state = restart.read(child_directory(base, seed) / "state.json")
            if child_state.get("status") != "COMPLETE":
                raise RuntimeError(f"seed{seed} did not complete its fixed child budget")
            check_contract(base)
            state["completed_seeds"].append(seed)
            atomic_json(base / "state.json", state)
            _readout(base, allow_partial=True)
        final_readout = _readout(base, allow_partial=False)
        completed_status = final_readout.get("execution_status", "COMPLETE")
        if completed_status not in {"COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES"}:
            raise RuntimeError("parameter grid execution did not validate its fixed budget")
        state.update(status=completed_status, current_seed=None,
                     financial_status=final_readout.get("financial_status", "COMPLETE"))
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        try:
            _readout(base, allow_partial=True)
        except Exception as readout_error:
            state["partial_readout_error"] = f"{type(readout_error).__name__}: {readout_error}"
        raise
    finally:
        state["finished_at_SGT"] = restart.now()
        atomic_json(base / "state.json", state)
        stop.set()
        monitor.join(timeout=20)
        refresh_status(base)
        atomic_json(
            base / "service_completion.json",
            dict(
                at_SGT=restart.now(), status=state["status"],
                training_controller=state["status"],
                completed_seeds=state["completed_seeds"],
                planned_trajectories=state["planned_trajectories"],
                comparison_readout=("PASS" if state["status"] == "COMPLETE" else
                                    "INCOMPLETE" if state["status"]
                                    == "COMPLETE_WITH_EVALUATION_FAILURES" else "PARTIAL"),
                financial_status=state.get("financial_status", "PARTIAL"),
                performance_pass=state["status"] == "COMPLETE",
                original_failure_preserved=True, automatic_retry=False,
                error=state.get("error"),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("prepare", "preflight", "seal", "check", "run"),
    )
    parser.add_argument("--phase", choices=("prepare", "preflight", "seal", "check", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--merge-receipt-sha256", default=MERGE_RECEIPT_SHA256)
    parser.add_argument("--coordinated-reservation-directory", type=Path)
    args = parser.parse_args()
    if args.command and args.phase and args.command != args.phase:
        parser.error("positional command and --phase must agree")
    phase = args.phase or args.command
    if phase is None:
        parser.error("a command or --phase is required")
    base = args.directory.resolve()
    if phase == "prepare":
        coordinated = args.coordinated_reservation_directory
        if coordinated is None:
            coordinated = runner.current_reservation_directory()
        prepare(base, model_path=args.model, merge_receipt_sha256=args.merge_receipt_sha256,
                coordinated_reservation_directory=coordinated)
    elif phase == "preflight":
        asyncio.run(input_preflight(base))
    elif phase == "seal":
        seal(base)
    elif phase == "check":
        check_contract(base)
    else:
        run(base)
    print(json.dumps(dict(phase=phase, status="PASS", at_SGT=restart.now())))


if __name__ == "__main__":
    main()
