"""Scalar readout keeps fixed populations, partial evidence, and all checkpoints."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_training_tuning_readout.py"
SPEC = importlib.util.spec_from_file_location("training_tuning_readout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def settings(phase="diagnostic"):
    return {
        "training_tuning": {
            "profile": report.PROFILE,
            "phase": phase,
            "selected_critic_learning_rate": None if phase == "diagnostic" else 1e-6,
            "evaluation_steps": [2] if phase == "diagnostic" else [2, 4, 6, 8],
        }
    }


def raw_trajectory(key, episode, arm="M1", net=0.01):
    steps, interactions, requests = [], [], []
    for step in range(20):
        state = {
            "session_index": step + 1,
            "cash": 100.0,
            "portfolio_equity": 100.0,
            "positions": {},
        }
        steps.append(
            {
                "step_index": step,
                "decision_session_index": step,
                "result_session_index": step + 1,
                "action": "hold",
                "invalid_action": False,
                "done": step == 19,
            }
        )
        interactions.append(
            {
                "observation": {
                    "truncated": False,
                    "text": json.dumps([{"content": json.dumps(state)}]),
                },
                "response": {
                    "truncated": False,
                    "text": '<tool_call>{"name":"hold","arguments":{}}</tool_call>',
                },
                "info": {"fills": [], "invalid_action": False},
            }
        )
        requests.append({"status": "COMPLETE", "response_token_count": 12})
    return {
        "identity": dict(zip(("step", "sample_index", "rollout_n", "validate"), key, strict=True)),
        "episode_id": episode,
        "arm": arm,
        "base_seed": 42,
        "schema_version": 1,
        "complete": True,
        "status": "COMPLETE",
        "observation_profile": "history_features_v1",
        "action_protocol": "single_tool_portfolio_v1",
        "chat_template_kwargs": {"enable_thinking": False},
        "steps": steps,
        "interactions": interactions,
        "requests": requests,
        "terminal_score": net * 100,
        "metrics": {
            "raw_net_return": net,
            "terminal_score": net * 100,
            "transaction_cost": 0.0,
            "max_drawdown": 0.0,
            "turnover": 0.0,
            "maximum_position_weight": 0.0,
            "invalid_count": 0,
            "calibration_sha256": "calibration",
        },
    }


def fit_record(spec, step):
    statistics = {"mean": 1.0, "std_population": 0.5, "min": 0.0, "max": 2.0}
    fit = {
        "values": statistics,
        "returns": statistics,
        "mse": 1.0,
        "half_mse": 0.5,
        "rmse": 1.0,
        "mean_bias_value_minus_return": 0.0,
    }
    return {
        "policy_step": step,
        "status": "COMPLETE",
        "phase": "critic_same_batch_pre_post",
        "arm": spec["arm"],
        "execution_identity_sha256": "execution",
        "update_call_index": step,
        "real_step_count": 160,
        "value_token_count": 160,
        "padding_row_count": 0,
        "extra_forward_passes": 1,
        "pre": fit,
        "post": fit,
        "critic_learning_rates": [spec["critic_lr"]],
        "planned_mini_batch_count": 1,
        "ppo_epochs": 1,
        "returns_recomputed": False,
        "mode_restored": True,
        "dropout": {"any_nonzero": False, "fit_comparison_uses_same_mode": True},
        "optimizer_grad_norms_before_clip": [2.0],
        "pid": 111,
        "extra_forward_seconds_synchronized": 30.0,
        "extra_forward_input_tokens": 1000,
    }


def run_fixture(directory, *, phase="diagnostic", complete=True):
    config = settings(phase)
    specs = report.cell_specs(config)
    model = {"model_snapshot_path": "/model", "model_revision": "revision"}
    manifest = {
        "datasets": {
            "train": {"episode_ids": report.TRAIN_IDS},
            "dev": {"episode_ids": report.EVAL_IDS},
        },
        "calibration": {"payload_sha256": "calibration"},
    }
    write_json(directory / "settings.json", config)
    write_json(directory / "model_identity.json", model)
    write_json(directory / "prepared/manifest.json", manifest)
    write_json(
        directory / "selection.json",
        {
            "phase": phase,
            "selected_critic_learning_rate": config["training_tuning"][
                "selected_critic_learning_rate"
            ],
        },
    )
    write_json(directory / "source_identity.json", {})
    write_json(
        directory / "run_contract.json",
        {
            "inputs": {
                name: report.sha(directory / name)
                for name in (
                    "settings.json",
                    "model_identity.json",
                    "prepared/manifest.json",
                    "selection.json",
                    "source_identity.json",
                )
            }
        },
    )
    write_json(
        directory / "state.json",
        {"status": "COMPLETE" if complete else "RUNNING", "schedule": specs},
    )
    if not complete:
        return config, specs, manifest
    helpers = report.load_helpers()
    for spec in specs:
        cell = directory / spec["cell_id"]
        write_json(
            cell / "summary.json",
            {
                "cell": spec,
                "status": "COMPLETE",
                "actual_rounds": spec["rounds"],
                "model_revision": "revision",
            },
        )
        write_json(
            cell / "command.json",
            [
                "python",
                "-m",
                "trainer",
                *[f"{key}=/model" for key in helpers.MODEL_OPTIONS],
                f"+{helpers.IDENTITY_OPTION}=execution",
            ],
        )
        write_json(
            cell / "artifacts/critic_initialization/head-rank0.json",
            {
                "seed": 42,
                "mode": "cpu_seed_v1",
                "arm": spec["arm"],
                "execution_identity_sha256": "execution",
                "model_path": "/model",
                "head_sha256": "a" * 64,
            },
        )
        grid = report.expected_grid(
            spec, report.TRAIN_IDS, report.EVAL_IDS, config["training_tuning"]["evaluation_steps"]
        )
        write_jsonl(
            cell / "artifacts/screening_evidence/rows.jsonl",
            [
                raw_trajectory(key, episode, spec["arm"], net=0.01 if not spec["rounds"] else -0.01)
                for key, episode in grid.items()
            ],
        )
        diagnostics, updates, fits, logs = [], [], [], []
        for step in range(1, spec["rounds"] + 1):
            diagnostics.append(
                {
                    "policy_step": step,
                    "arm": spec["arm"],
                    "eta": spec["eta"],
                    "run_seed": 42,
                    "real_step_count": 160,
                    "all_real_steps": {"std_U": 1.0, "std_S": spec["eta"]},
                }
            )
            fits.append(fit_record(spec, step))
            for offset, role in enumerate(("critic", "actor"), 1):
                updates.append(
                    {
                        "pid": 111,
                        "role": role,
                        "detected": True,
                        "grad_norm": 2.0,
                        "global_update_count": (step - 1) * 2 + offset,
                        "optimizer_step_count": (step - 1) * 2 + offset,
                    }
                )
            logs.append(f"step:{step} - critic/vf_loss:0.5 - actor/grad_norm:2.0\n")
        write_jsonl(cell / "artifacts/sibling_diagnostics/rows.jsonl", diagnostics)
        write_jsonl(cell / "artifacts/critic_fit/rows.jsonl", fits)
        write_jsonl(cell / "artifacts/update_probe/rows.jsonl", updates)
        (cell / "trainer.log").write_text("".join(logs))
    return config, specs, manifest


def test_pinned_helper_rejects_changed_source(tmp_path):
    copy = tmp_path / "helper.py"
    copy.write_text(report.HELPER_PATH.read_text() + "\n# drift\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        report.load_helpers(copy)


def test_epoch_repeats_sample_indices_but_preserves_policy_identity():
    config = settings("comparison")
    spec = report.cell_specs(config)[1]
    grid = report.expected_grid(spec, report.TRAIN_IDS, report.EVAL_IDS, [2, 4, 6, 8])
    assert len(grid) == 64 + 16
    assert grid[(1, 0, 0, False)] == grid[(5, 0, 0, False)] == report.TRAIN_IDS[0]
    assert grid[(4, 7, 3, False)] == grid[(8, 7, 3, False)] == report.TRAIN_IDS[7]
    assert {key[0] for key in grid if key[3]} == {2, 4, 6, 8}


def test_complete_negative_return_run_is_retained_and_streamed(tmp_path, monkeypatch):
    run_fixture(tmp_path)
    original = Path.read_bytes

    def guarded_read_bytes(path):
        assert path.suffix != ".jsonl", "large raw evidence must be streamed"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    result = report.build(tmp_path)
    assert result["status"] == "COMPLETE", result["problems"] + result["pending"]
    assert result["initial_critic_heads"]["all_equal"]
    cell = result["cells"]["critic_lr_1e5"]
    assert cell["groups"]["train"]["terminal_complete"] == 16
    assert cell["groups"]["eval"]["terminal_complete"] == 4
    assert [row["population"]["terminal_complete"] for row in cell["rounds"]] == [8, 8]
    assert cell["groups"]["train"]["legal_hold_observed"] == 320
    assert cell["extra_critic_forward_seconds"] == 60
    delta = result["evaluation_curves"][0]["points"][0]["fixed_four_window_mean_delta"]
    assert delta["net_return_percentage_points"] == pytest.approx(-2.0)
    assert "interactions" not in json.dumps(result)


def test_duplicate_identity_excludes_both_copies_and_partial_tail_stays_pending(tmp_path):
    helpers = report.load_helpers()
    spec = report.cell_specs(settings())[0]
    grid = report.expected_grid(spec, report.TRAIN_IDS, report.EVAL_IDS, [2])
    key, episode = next(iter(grid.items()))
    row = raw_trajectory(key, episode)
    path = tmp_path / "sft0/artifacts/screening_evidence/rows.jsonl"
    write_jsonl(path, [row, row])
    with path.open("a") as stream:
        stream.write('{"identity":')
    evidence = report.Evidence(tmp_path)
    rows, excluded, count = report.read_trajectories(
        evidence, helpers, tmp_path / "sft0", spec, grid, "calibration"
    )
    assert rows == [] and count == 2
    assert excluded[0]["reason"] == "duplicate identity: all copies excluded"
    assert len(excluded[0]["sources"]) == 2
    assert len(evidence.pending) == 1
    assert not evidence.errors
    assert evidence.hashes[str(path.relative_to(tmp_path))] == report.sha(path)


def test_all_four_budget_points_retained_and_partial_pairs_never_become_four_window_mean(tmp_path):
    run_fixture(tmp_path, phase="comparison")
    result = report.build(tmp_path)
    assert result["status"] == "COMPLETE", result["problems"] + result["pending"]
    assert len(result["evaluation_curves"]) == 5
    curve = next(
        curve
        for curve in result["evaluation_curves"]
        if curve["target"] == "m2_eta025" and curve["reference"] == "m1"
    )
    assert [point["policy_step"] for point in curve["points"]] == [2, 4, 6, 8]
    rows = result["cells"]["m2_eta025"]["trajectory_rows"]
    row = next(row for row in rows if row["identity"]["validate"] and row["identity"]["step"] == 4)
    row["terminal_eligible"] = False
    curves = report.paired_curves(
        report.load_helpers(), result["cells"], report.EVAL_IDS, [2, 4, 6, 8]
    )
    curve = next(
        curve for curve in curves if curve["target"] == "m2_eta025" and curve["reference"] == "m1"
    )
    assert curve["points"][1]["complete_pairs"] == 3
    assert curve["points"][1]["fixed_four_window_mean_delta"] is None
    assert curve["points"][1]["observed_complete_pairs_mean_delta"]["net_return"] == 0
    assert [point["complete_pairs"] for point in curve["points"]] == [4, 3, 4, 4]


def test_live_write_is_repeatable_and_never_claims_missing_grid_complete(tmp_path):
    run_fixture(tmp_path, complete=False)
    for _ in range(2):
        result = report.main(["--directory", str(tmp_path), "--write"])
        assert result["status"] == "INCOMPLETE"
        assert result["cells"]["sft0"]["groups"]["eval"]["missing"] == 4
        assert result["cells"]["sft0"]["groups"]["eval"]["mean_net_return_complete_only"] is None
    assert json.loads((tmp_path / "readout.json").read_text())["status"] == "INCOMPLETE"
    assert "待齐全" in (tmp_path / "LIVE_STATUS.md").read_text()


def test_live_render_prioritizes_running_execution_and_uses_sgt(tmp_path):
    run_fixture(tmp_path, complete=False)
    result = report.build(tmp_path)
    result["controller"]["current_cell"] = "sft0"
    result["generated_at_UTC"] = "2026-09-14T05:27:43+00:00"
    text = report.markdown(result)
    assert result["execution_status"] == "RUNNING"
    assert result["status"] == "INCOMPLETE"
    assert "执行状态：**运行中**" in text
    assert "2026-09-14 13:27:43 SGT（UTC+8）" in text
    assert "**INCOMPLETE**" not in text
    assert "| sft0 | 初始化中 |" in text
    assert "| critic_lr_1e5 | 待启动 |" in text
    assert "当前仍在按计划收集轨迹与更新证据" in text


def test_completed_execution_keeps_unverified_evidence_visible(tmp_path):
    run_fixture(tmp_path, complete=False)
    result = report.build(tmp_path)
    result["execution_status"] = "COMPLETE"
    result["controller"]["status"] = "COMPLETE"
    result["cells"]["sft0"]["summary"]["status"] = "COMPLETE"
    text = report.markdown(result)
    assert "执行状态：**已完成**" in text
    assert "证据尚待核对" in text
    assert "| sft0 | 已完成（证据待核对） |" in text
    result["execution_status"] = "FAILED"
    assert "执行状态：**失败**" in report.markdown(result)


def test_target_drift_and_nonfinite_probe_remain_visible(tmp_path):
    run_fixture(tmp_path)
    path = tmp_path / "critic_lr_1e5/artifacts/critic_fit/rows.jsonl"
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    rows[0]["post"]["returns"]["mean"] = 123
    rows[1]["post"]["half_mse"] = float("nan")
    write_jsonl(path, rows)
    result = report.build(tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert any("critic fit protocol" in value for value in result["problems"])
    value = result["cells"]["critic_lr_1e5"]["rounds"][1]["critic_fit"]["post"]["half_mse"]
    assert value == {"observed_nonfinite": "nan"}
    json.dumps(result, allow_nan=False)
