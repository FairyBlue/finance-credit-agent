#!/usr/bin/env python3
# ruff: noqa: E501  # Preserve readable Markdown table rows in the README template.
"""Rebuild only this small evidence snapshot from the original local run files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[2]
MAX_BYTES = 5_000_000

REPORTS = [
    "2026-09-13_M2_COLDSTART_REPORT.md",
    "2026-09-13_M2_COLDSTART_V2_REPORT.md",
    "2026-09-14_M2_COLDSTART_V3_REPORT.md",
    "2026-09-14_M2_COMMON_SFT_SHORT_RL_REPORT.md",
    "2026-09-14_M2_TRAINING_TUNING_REPORT.md",
    "2026-09-15_M2_SAMPLING_PROFILE_REPORT.md",
    "2026-09-15_M2_TRAIN_COVERAGE_REPORT.md",
    "2026-09-17_M2_TRAIN_COVERAGE_DIAGNOSTIC_REPORT.md",
    "2026-09-17_M2_COMMON_VALIDITY_PENALTY_PLAN.md",
]
RUN_FILES = {
    "m2-coldstart-20260913": [
        "readout.json",
        "completion_receipt.json",
        "final_validation.json",
        "source_manifest.json",
        "merged_model/merge_receipt.json",
    ],
    "m2-coldstart-v2-20260913": [
        "model_identity.json",
        "exported_model_identity.json",
        "sft_validation.json",
        "source_manifest.json",
        "merged_model/merge_receipt.json",
    ],
    "m2-coldstart-v3-20260914": [
        "model_identity.json",
        "exported_model_identity.json",
        "sft_validation.json",
        "source_manifest.json",
        "merged_model/merge_receipt.json",
    ],
    "m2-common-sft-short-rl-20260914": [
        "completion_receipt.json",
        "goal_completion_validation.json",
        "settings.json",
        "run_contract.json",
        "source_identity.json",
        "model_identity.json",
    ],
    "m2-training-tuning-preparation-20260914": [
        "comparison_summary_evaluation_summary.csv",
        "comparison_summary_paired_curves.csv",
        "comparison_summary_paired_windows.csv",
        "comparison_summary_training_diagnostics.csv",
        "finalization_state.json",
    ],
    "m2-training-tuning-20260914": [
        "FINAL_RESULTS.md",
        "settings.json",
        "run_contract.json",
        "source_identity.json",
        "model_identity.json",
        "selection.json",
    ],
    "m2-sampling-profile-eval-20260915": [
        "FINAL_RESULTS.md",
        "summary.csv",
        "completion_validation.json",
        "settings.json",
        "run_contract.json",
        "source_identity.json",
        "models.json",
    ],
    "m2-train-coverage-20260916-v2": [
        "FINAL_RESULTS.md",
        "summary.csv",
        "completion_validation.json",
        "settings.json",
        "run_contract.json",
        "source_identity.json",
        "model_identity.json",
        "selection.json",
    ],
    "m2-train-coverage-diagnosis-20260917": [
        "completion_validation.json",
        "coverage_diagnosis.png",
        "coverage_diagnosis.svg",
        "behavior/REPORT.md",
        "behavior/group_means.csv",
        "behavior/window_means.csv",
        "behavior/contrast_windows.csv",
        "signals/REPORT.md",
        "signals/trajectory_allocation.csv",
        "sampling/sampling_analysis.json",
        "sampling/within_window_spread.csv",
    ],
    "m2-common-validity-penalty-20260917": [
        "LIVE_STATUS.md",
        "gpu_queue_state.json",
        "run_contract.json",
        "source_identity.json",
        "settings.json",
        "model_identity.json",
        "prelaunch_validation.json",
    ],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def main():
    at = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    records = []
    queue_path = "runs/m2-common-validity-penalty-20260917/gpu_queue_state.json"
    live_path = "runs/m2-common-validity-penalty-20260917/LIVE_STATUS.md"
    fixed_queue_snapshot = {path: (ROOT / path).read_bytes() for path in (queue_path, live_path)}
    queue = json.loads(fixed_queue_snapshot[queue_path])
    if (
        queue.get("status") != "WAITING_FOR_GPU"
        or "0/164" not in fixed_queue_snapshot[live_path].decode()
    ):
        raise RuntimeError("snapshot README requires a fresh review if the queued run has started")

    def save(relative, payload, *, source=None, source_bytes=None, projection=None):
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        records.append(
            {
                "snapshot_path": relative,
                "bytes": len(payload),
                "sha256": digest(payload),
                "source_relative_path": source,
                "source_bytes": len(source_bytes) if source_bytes is not None else None,
                "source_sha256": digest(source_bytes) if source_bytes is not None else None,
                "operation": "exact_byte_copy" if source and projection is None else "generated",
                "projection": projection,
            }
        )

    def read(relative):
        raw = (ROOT / relative).read_bytes()
        return raw, json.loads(raw)

    for relative in ["docs/" + name for name in REPORTS] + [
        "runs/" + run + "/" + name for run, names in RUN_FILES.items() for name in names
    ]:
        raw = fixed_queue_snapshot.get(relative)
        if raw is None:
            raw = (ROOT / relative).read_bytes()
        save(relative, raw, source=relative, source_bytes=raw)

    for run in ("m2-coldstart-v2-20260913", "m2-coldstart-v3-20260914"):
        source = f"runs/{run}/readout.json"
        raw, value = read(source)
        fields = [
            "status",
            "finished_at_SGT",
            "action_protocol",
            "inputs_sha256",
            "gates",
            "limits",
        ]
        result = {name: value[name] for name in fields}
        # These are existing gate results, not a newly pooled financial metric.
        # Keep all suites and both before/after decode/horizon strata verbatim.
        result["by_suite_horizon_sampling"] = {
            suite: {phase: condition[phase]["by_horizon_sampling"] for phase in ("before", "after")}
            for suite, condition in value["suites"].items()
        }
        save(
            f"runs/{run}/gates_and_decode_summary.json",
            encoded(result),
            source=source,
            source_bytes=raw,
            projection={
                "top_level_fields_copied": fields,
                "suite_projection": "suites.<suite>.<before|after>.by_horizon_sampling",
                "omitted": "individual pairs, overall pooled metrics and additional state strata",
                "reaggregated": False,
            },
        )

    run = "m2-common-sft-short-rl-20260914"
    source = f"runs/{run}/readout.json"
    raw, value = read(source)
    fields = [
        "status",
        "at_SGT",
        "controller_status",
        "controller_error",
        "shared_model_identity",
        "initial_critic_heads",
        "remaining_or_failed_checks",
        "input_sha256",
        "limits",
    ]
    cell_fields = ["status", "summary", "groups", "rounds", "training_metrics_as_logged"]
    result = {name: value[name] for name in fields}
    result["cells"] = {
        name: {key: cell[key] for key in cell_fields} for name, cell in value["cells"].items()
    }
    save(
        f"runs/{run}/readout_summary.json",
        encoded(result),
        source=source,
        source_bytes=raw,
        projection={
            "top_level_fields_copied": fields,
            "cell_fields_copied": cell_fields,
            "omitted": "individual trajectory/fill records and evaluation pairs",
            "reaggregated": False,
            "evaluation_decode": "greedy only; step 0 SFT and step 2 M1/M2 retained",
        },
    )

    readme = f"""# 2026-09-17 实验结果备份

快照采集时间：**{at}（新加坡 UTC+8）**。

这是用于 GitHub 讨论和追溯的精简结果快照。模型权重、剩余优化器、完整请求/响应轨迹、
原始市场数据及完整 `runs/` 仍在服务器原位置；本目录不构成可恢复训练的完整备份。
这里的 `runs/` 是少量复制结果，不能作为新的实验运行目录。

所有已完成结果按原实验、checkpoint 轮次、窗口及解码方式保留；没有重新合并 greedy/sample、
不同日期窗口或不同 SFT 起点。不同实验之间的收益不能脱离各自协议直接比较。
“完成”指执行/证据完备，不代表方法有效；冷启动 v2 的 FAIL 和 M2 落后读数均保留。

| 实验 | 阅读入口 | 机器证据 |
|---|---|---|
| 冷启动 v1 | [报告](docs/2026-09-13_M2_COLDSTART_REPORT.md) | [原 readout](runs/m2-coldstart-20260913/readout.json) |
| 冷启动 v2 | [报告](docs/2026-09-13_M2_COLDSTART_V2_REPORT.md) | [门禁和解码分组](runs/m2-coldstart-v2-20260913/gates_and_decode_summary.json) |
| 冷启动 v3 | [报告](docs/2026-09-14_M2_COLDSTART_V3_REPORT.md) | [门禁和解码分组](runs/m2-coldstart-v3-20260914/gates_and_decode_summary.json) |
| 共同 SFT 两轮 RL | [报告](docs/2026-09-14_M2_COMMON_SFT_SHORT_RL_REPORT.md) | [分臂/分轮摘要](runs/m2-common-sft-short-rl-20260914/readout_summary.json) |
| critic/η 训练调参 | [最终结果](runs/m2-training-tuning-20260914/FINAL_RESULTS.md) | [验收曲线](runs/m2-training-tuning-preparation-20260914/comparison_summary_evaluation_summary.csv)、[配对差](runs/m2-training-tuning-preparation-20260914/comparison_summary_paired_curves.csv) |
| 解码配置对照 | [最终结果](runs/m2-sampling-profile-eval-20260915/FINAL_RESULTS.md) | [分模型/解码 CSV](runs/m2-sampling-profile-eval-20260915/summary.csv) |
| 16 窗 Train coverage | [最终结果](runs/m2-train-coverage-20260916-v2/FINAL_RESULTS.md) | [分轮/解码 CSV](runs/m2-train-coverage-20260916-v2/summary.csv)、[完成验收](runs/m2-train-coverage-20260916-v2/completion_validation.json) |
| coverage 收益/学习信号诊断 | [报告](docs/2026-09-17_M2_TRAIN_COVERAGE_DIAGNOSTIC_REPORT.md) | [行为均值](runs/m2-train-coverage-diagnosis-20260917/behavior/group_means.csv)、[采样分析](runs/m2-train-coverage-diagnosis-20260917/sampling/sampling_analysis.json) |

![coverage 诊断](runs/m2-train-coverage-diagnosis-20260917/coverage_diagnosis.png)

## 尚未开始的新实验

共同拒单罚四轮实验在这个快照中是 **NOT STARTED / WAITING_FOR_GPU，0/164 条轨迹**。
[LIVE_STATUS 时间点副本](runs/m2-common-validity-penalty-20260917/LIVE_STATUS.md)和
[队列时间点副本](runs/m2-common-validity-penalty-20260917/gpu_queue_state.json)可能来自相邻刷新，
各自保留原时间戳。它们不会随服务器后续进展自动更新，也不是实验结果。

## 来源与校验

[manifest.json](manifest.json) 对每份输出记录大小与 SHA-256、仓库根目录下的原始相对路径、
原文件 SHA-256。报告、CSV、图片、身份与验收收据均逐字节复制；只有两份冷启动分组摘要和
一份短 RL 摘要是明确字段投影，不重新计算指标；投影规则逐项列在 manifest。
manifest 自身不做递归自哈希，其他全部文件都有输出哈希。

源报告保持原文与原相对链接；未纳入本精简备份的历史计划、完整诊断 JSON 或逐步轨迹，
仍需从原仓库/服务器路径读取。输入/模型哈希可以核对身份，不表示这些大文件已经上传。
可在服务器从仓库根目录运行
`python3 docs/results/2026-09-17-backup/_build_snapshot.py` 重建；如果新实验已经启动，
脚本会要求先更新“未开始”的快照说明。重建会改变快照采集时间；不会改写原实验文件。
"""
    save("README.md", readme.encode(), projection="index and snapshot boundary explanation")
    script = Path(__file__).read_bytes()
    records.append(
        {
            "snapshot_path": "_build_snapshot.py",
            "bytes": len(script),
            "sha256": digest(script),
            "source_relative_path": None,
            "source_sha256": None,
            "operation": "snapshot_builder",
            "projection": None,
        }
    )
    manifest = {
        "schema_version": 1,
        "snapshot_at_SGT": at,
        "scope": "selected completed experiment metrics, provenance, and an unstarted queue snapshot",
        "source_root": "repository root; all source_relative_path values are relative to it",
        "no_training_restart_artifacts": True,
        "no_model_optimizer_market_data_or_full_token_traces": True,
        "queued_experiment_status_at_snapshot": "NOT_STARTED_WAITING_FOR_GPU",
        "files": sorted(records, key=lambda row: row["snapshot_path"]),
        "manifest_self_hash_exclusion": "manifest.json is the only un-hashed output",
        "payload_file_count_excluding_manifest": len(records),
        "payload_bytes_excluding_manifest": sum(row["bytes"] for row in records),
    }
    (DEST / "manifest.json").write_bytes(encoded(manifest))
    files = [p for p in DEST.rglob("*") if p.is_file()]
    expected = {row["snapshot_path"] for row in records} | {"manifest.json"}
    if {str(p.relative_to(DEST)) for p in files} != expected:
        raise RuntimeError("unlisted files exist in snapshot directory")
    total = sum(p.stat().st_size for p in files)
    if total >= MAX_BYTES:
        raise RuntimeError(f"snapshot exceeds {MAX_BYTES} byte bound: {total}")
    for row in records:
        if digest((DEST / row["snapshot_path"]).read_bytes()) != row["sha256"]:
            raise RuntimeError("snapshot output hash differs")
    largest = max(files, key=lambda path: path.stat().st_size)
    print(
        json.dumps(
            {
                "files": len(files),
                "bytes": total,
                "largest": str(largest.relative_to(DEST)),
                "largest_bytes": largest.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
