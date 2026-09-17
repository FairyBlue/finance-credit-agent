"""Fixed-budget, reward-component joins, historical checkpoint and pending-queue tests."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


report = module(
    ROOT / "scripts/diagnostics/m2_common_validity_penalty_readout.py", "coverage_readout"
)
# Reuse the established native 20-session evidence/critic fixtures, not old grids.
fixtures = module(ROOT / "tests/test_m2_training_tuning_readout.py", "coverage_native_fixtures")
write_json, write_jsonl = fixtures.write_json, fixtures.write_jsonl


def settings():
    return {
        "training_coverage": {"profile": report.PROFILE, "evaluation_steps": [2, 4]},
        "training": {"learning_rate": 2e-7, "critic_learning_rate": 1e-6},
        "common_validity_penalty": {
            "profile": "environment_rejection_v1",
            "coefficient": 0.05,
            "applies_to": "training_only",
            "financial_terminal_score_unchanged": True,
        },
    }


def receipt(key, episode, spec):
    mode = "greedy" if key[2] == 0 else "sampled"
    return {
        "schema_version": 1,
        "profile": "greedy1_sample4_v1",
        "status": "COMPLETE",
        "arm": spec["arm"],
        "execution_identity_sha256": "execution",
        "episode_id": episode,
        "identity": dict(zip(("step", "sample_index", "rollout_n", "validate"), key, strict=True)),
        "policy_step": key[0],
        "validation_rollout_n": 5,
        "mode": mode,
        "sampled_sibling": None if mode == "greedy" else key[2] - 1,
        "upstream_sampling_params": {"logprobs": 0, "temperature": 0.3},
        "forwarded_sampling_params": {
            "logprobs": 0,
            "temperature": 0.0 if mode == "greedy" else 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
    }


def actor_record(spec, step):
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "phase": "actor_same_batch_pre_post",
        "arm": spec["arm"],
        "execution_identity_sha256": "execution",
        "policy_step": step,
        "update_call_index": step,
        "real_step_count": 160,
        "padding_row_count": 0,
        "response_token_count": 1920,
        "global_response_token_denominator": 1920,
        "optimizer_step_count": 1,
        "ppo_epochs": 1,
        "planned_minibatch_count": 1,
        "actor_learning_rates": [2e-7],
        "extra_post_forward_micro_batches": 160,
        "optimizer_grad_norms_before_clip": [2.0],
        "groups": {
            "all_real": {"step_count": 160, "response_tokens": 1920},
            "legal": {"step_count": 160, "response_tokens": 1920},
            "invalid": {
                "step_count": 0,
                "response_tokens": 0,
                "mean_advantage_token_weighted": None,
            },
        },
        "rows": [{"details": "retained at source"}] * 160,
    }


def run_fixture(directory, complete=True):
    config = settings()
    specs = report.cell_specs(config)
    model = {"model_snapshot_path": "/model", "model_revision": "revision"}
    write_json(directory / "settings.json", config)
    write_json(directory / "model_identity.json", model)
    write_json(
        directory / "prepared/manifest.json",
        {
            "datasets": {
                "train": {"episode_ids": report.TRAIN_IDS},
                "dev": {"episode_ids": report.EVAL_IDS},
            },
            "calibration": {"payload_sha256": "calibration"},
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
                    "source_identity.json",
                )
            },
        },
    )
    write_json(
        directory / "state.json",
        {
            "status": "COMPLETE" if complete else "RUNNING",
            "schedule": specs,
        },
    )
    if not complete:
        return
    helpers = report.load_helpers()
    for cell_index, spec in enumerate(specs):
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
                *[f"{k}=/model" for k in helpers.MODEL_OPTIONS],
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
        grid = report.expected_grid(spec)
        raw = []
        for key, episode in grid.items():
            # Make greedy/sample differences larger than model differences: pooling is detectable.
            net = (0.5 if key[3] and key[2] == 0 else 0.01 * (key[2] + 1)) + cell_index * 0.001
            trajectory = fixtures.raw_trajectory(key, episode, spec["arm"], net=net)
            trajectory["credits"] = [trajectory["terminal_score"] / 20] * 20
            raw.append(trajectory)
        write_jsonl(cell / "artifacts/screening_evidence/rows.jsonl", raw)
        write_jsonl(
            cell / "artifacts/paired_validation/rows.jsonl",
            [receipt(key, episode, spec) for key, episode in grid.items() if key[3]],
        )
        diagnostics, updates, fits, probes, logs = [], [], [], [], []
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
            fits.append(fixtures.fit_record(spec, step))
            probes.append(actor_record(spec, step))
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
        for diagnostic in diagnostics:
            add_penalty_evidence(diagnostic, raw)
        write_jsonl(cell / "artifacts/sibling_diagnostics/rows.jsonl", diagnostics)
        write_jsonl(cell / "artifacts/critic_fit/rows.jsonl", fits)
        write_jsonl(cell / "artifacts/update_probe/rows.jsonl", updates)
        write_jsonl(cell / "artifacts/actor_update/rows.jsonl", probes)
        (cell / "trainer.log").write_text("".join(logs))


def add_penalty_evidence(diagnostic, raw):
    records, trajectories = [], []
    for trajectory in raw:
        identity = trajectory["identity"]
        if identity["validate"] or identity["step"] != diagnostic["policy_step"]:
            continue
        uid = f"{identity['step']}:{identity['sample_index']}:{identity['rollout_n']}"
        y = trajectory["terminal_score"]
        steps = trajectory["steps"]
        rewards = [
            reward - 0.05 * step["invalid_action"]
            for reward, step in zip(trajectory["credits"], steps, strict=False)
        ]
        for index, (step, reward) in enumerate(zip(steps, trajectory["credits"], strict=False)):
            records.append(
                dict(
                    trajectory_uid=uid,
                    step_index=index,
                    batch_row_index=len(records),
                    episode_id=trajectory["episode_id"],
                    policy_step=identity["step"],
                    rollout_id=identity["rollout_n"],
                    action=step["action"],
                    invalid_action=step["invalid_action"],
                    terminal_Y=y,
                    financial_step_reward=reward,
                    invalid_penalty=0.05 * step["invalid_action"],
                    training_step_reward=rewards[index],
                    unwhitened_U=sum(rewards[index:]),
                    sibling_residual=0.0,
                    final_actor_advantage=1.0,
                    critic_return=sum(rewards[index:]),
                )
            )
        count = sum(step["invalid_action"] for step in steps)
        trajectories.append(
            dict(
                trajectory_uid=uid,
                step_count=len(steps),
                terminal_Y=y,
                invalid_count=count,
                financial_reward_sum=sum(trajectory["credits"]),
                penalty_sum=0.05 * count,
                training_reward_sum=sum(rewards),
                expected_training_reward_sum=y - 0.05 * count,
            )
        )
    diagnostic["reward_evidence"] = records
    diagnostic["common_validity_penalty"] = dict(
        profile="environment_rejection_v1",
        coefficient=0.05,
        invalid_source="trading_invalid_action",
        sibling_residual_outcome="original_financial_terminal_Y",
        trajectories=trajectories,
        **{
            k: sum(t[k] for t in trajectories)
            for k in ("invalid_count", "financial_reward_sum", "penalty_sum", "training_reward_sum")
        },
    )


def test_complete_fixed_164_and_separate_lanes(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "HISTORICAL_RUN", tmp_path / "absent")
    run_fixture(tmp_path)
    result = report.build(tmp_path)
    assert result["status"] == "COMPLETE", (result["problems"], result["pending"])
    assert result["totals"]["planned"] == result["totals"]["terminal_complete"] == 164
    assert result["initial_critic_heads"]["all_equal"]
    assert [len(report.expected_grid(spec)) for spec in report.cell_specs(settings())] == [
        20,
        72,
        72,
    ]
    cell = result["cells"]["m1"]
    assert cell["groups"]["train"]["planned"] == 32
    assert cell["groups"]["eval_greedy"]["planned"] == 8
    assert cell["groups"]["eval_sampled"]["planned"] == 32
    assert cell["common_validity_penalty"]["joined_trajectories"] == 32
    assert cell["evaluations"][0]["population"]["fixed_window_mean"][
        "net_return_pct"
    ] == pytest.approx(50.1)
    assert cell["evaluations"][1]["population"]["fixed_window_mean"][
        "net_return_pct"
    ] == pytest.approx(3.6)
    assert {p["policy_step"] for c in result["evaluation_curves"] for p in c["points"]} == {2, 4}
    report.main(["--directory", str(tmp_path), "--write"])
    assert (tmp_path / "FINAL_RESULTS.md").is_file()
    assert "原 Y" in (tmp_path / "FINAL_RESULTS.md").read_text()


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonfinite"])
def test_partial_never_shrinks_denominator(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(report, "HISTORICAL_RUN", tmp_path / "absent")
    run_fixture(tmp_path)
    path = tmp_path / "m1/artifacts/screening_evidence/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    index = next(
        i
        for i, r in enumerate(rows)
        if r["identity"] == dict(step=2, sample_index=0, rollout_n=1, validate=True)
    )
    if damage == "missing":
        rows.pop(index)
    elif damage == "duplicate":
        rows.append(copy.deepcopy(rows[index]))
    else:
        rows[index]["terminal_score"] = float("nan")
    write_jsonl(path, rows)
    result = report.build(tmp_path)
    pop = result["cells"]["m1"]["evaluations"][1]["population"]
    assert pop["planned"] == 16 and pop["planned_windows"] == 4
    assert pop["fixed_window_mean"] is None and result["status"] != "COMPLETE"
    point = result["evaluation_curves"][1]["points"][0]
    assert point["fixed_four_window_mean_delta"] is None


def test_penalty_rejects_bad_reward_and_preserves_original_y():
    spec = report.cell_specs(settings())[1] | {"rounds": 1}
    raw = [
        fixtures.raw_trajectory(key, episode, "M1", 0.01)
        for key, episode in report.expected_grid(report.cell_specs(settings())[1]).items()
        if key[0] == 1 and not key[3]
    ]
    for row in raw:
        row["credits"] = [row["terminal_score"] / 20] * 20
    raw[0]["steps"][0].update(action="buy", invalid_action=True)
    diagnostic = {"policy_step": 1, "real_step_count": 160}
    add_penalty_evidence(diagnostic, raw)
    rows = [
        {
            "identity": r["identity"],
            "episode_id": r["episode_id"],
            "terminal_eligible": True,
            "Y": r["terminal_score"],
            "financial_step_evidence": [
                {
                    "step_index": s["step_index"],
                    "action": s["action"],
                    "invalid_action": s["invalid_action"],
                    "financial_step_reward": v,
                }
                for s, v in zip(r["steps"], r["credits"], strict=False)
            ],
        }
        for r in raw
    ]
    result = report.penalty_readout(spec, rows, {1: [diagnostic]})
    assert result["status"] == "COMPLETE", result
    assert result["fixed_training_mean"]["financial_reward_sum"] == pytest.approx(1.0)
    assert result["fixed_training_mean"]["training_reward_sum"] == pytest.approx(1 - 0.05 / 8)
    assert result["observed_sums"]["invalid_count"] == 1
    diagnostic["reward_evidence"][0]["invalid_penalty"] = 0
    result = report.penalty_readout(spec, rows, {1: [diagnostic]})
    assert result["problems"] and result["status"] != "COMPLETE"


def test_queue_pending_without_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "HISTORICAL_RUN", tmp_path / "absent")
    run_fixture(tmp_path, complete=False)
    (tmp_path / "state.json").unlink()
    write_json(tmp_path / "gpu_queue_state.json", {"status": "WAITING_FOR_GPU"})
    result = report.build(tmp_path)
    assert result["execution_status"] == "WAITING_FOR_GPU"
    assert result["status"] == "INCOMPLETE" and result["problems"] == []
    assert result["totals"]["planned"] == 164 and result["totals"]["terminal_complete"] == 0
    assert "GPU 排队" in report.markdown(result)


def test_history_matches_only_zero_two_four_not_eight(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "HISTORICAL_RUN", tmp_path / "absent")
    run_fixture(tmp_path)
    current = report.build(tmp_path)
    historical = copy.deepcopy(current)
    historical["profile"] = "common_sft_train_coverage_v1"
    for cell in historical["cells"].values():
        extra = copy.deepcopy(cell["evaluations"][-1])
        extra["policy_step"] = 8
        extra["population"]["fixed_window_mean"]["net_return_pct"] = 999.0
        cell["evaluations"].append(extra)
    old = tmp_path / "historical"
    write_json(old / "readout.json", historical)
    comparison = report.historical_comparison(
        current["cells"], current["shared_model_identity"], old
    )
    assert comparison["status"] == "AVAILABLE"
    assert len(comparison["comparisons"]) == 10
    assert {item["reference_policy_step"] for item in comparison["comparisons"]} == {0, 2, 4}
    assert all(
        item["fixed_four_window_mean_delta"]["net_return_pct"] == 0
        for item in comparison["comparisons"]
    )


@pytest.mark.parametrize("part", ["coefficient", "schedule", "learning_rate"])
def test_reject_changed_contract(part):
    config = settings()
    if part == "coefficient":
        config["common_validity_penalty"]["coefficient"] = 0.1
    elif part == "schedule":
        config["training_coverage"]["evaluation_steps"] = [2, 4, 8]
    else:
        config["training"]["learning_rate"] = 1e-6
    with pytest.raises(ValueError):
        report.cell_specs(config)
