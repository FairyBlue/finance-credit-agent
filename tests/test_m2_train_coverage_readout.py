"""Fixed 308-grid, validation-lane separation, and missing-evidence behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


report = module(ROOT / "scripts/diagnostics/m2_train_coverage_readout.py", "coverage_readout")
# Reuse the established native 20-session evidence/critic fixtures, not old grids.
fixtures = module(ROOT / "tests/test_m2_training_tuning_readout.py", "coverage_native_fixtures")
write_json, write_jsonl = fixtures.write_json, fixtures.write_jsonl


def settings():
    return {
        "training_coverage": {"profile": report.PROFILE, "evaluation_steps": [2, 4, 6, 8]},
        "training": {"learning_rate": 2e-7, "critic_learning_rate": 1e-6},
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
            raw.append(fixtures.raw_trajectory(key, episode, spec["arm"], net=net))
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
        write_jsonl(cell / "artifacts/sibling_diagnostics/rows.jsonl", diagnostics)
        write_jsonl(cell / "artifacts/critic_fit/rows.jsonl", fits)
        write_jsonl(cell / "artifacts/update_probe/rows.jsonl", updates)
        write_jsonl(cell / "artifacts/actor_update/rows.jsonl", probes)
        (cell / "trainer.log").write_text("".join(logs))


def scalar_rows(spec):
    helper = report.load_helpers()
    grid = report.expected_grid(spec)
    return grid, [
        helper.trajectory(
            "fixture",
            fixtures.raw_trajectory(
                key,
                episode,
                spec["arm"],
                net=0.5 if key[2] == 0 else key[2] * 0.01,
            ),
        )
        for key, episode in grid.items()
    ]


def test_exact_grid_uses_sixteen_unique_windows_once_and_five_eval_lanes():
    specs = report.cell_specs(settings())
    grids = [report.expected_grid(spec) for spec in specs]
    assert [len(grid) for grid in grids] == [20, 144, 144]
    assert sum(map(len, grids)) == 308
    grid = grids[1]
    assert grid[(1, 0, 0, False)] == "formal-train-0001"
    assert grid[(1, 1, 3, False)] == "formal-train-0065"
    assert grid[(8, 15, 3, False)] == "formal-train-0189"
    assert (5, 0, 0, False) not in grid
    assert len({episode for key, episode in grid.items() if not key[3]}) == 16
    assert {key[2] for key in grid if key[3]} == set(range(5))
    assert {key[0] for key in grids[0]} == {0}


def test_sampled_window_mean_excludes_greedy_and_preserves_fixed_denominator():
    spec = report.cell_specs(settings())[0]
    grid, rows = scalar_rows(spec)
    helper = report.load_helpers()
    sampled = report.fixed_population(helper, rows, grid, step=0, lane="sampled")
    greedy = report.fixed_population(helper, rows, grid, step=0, lane="greedy")
    assert sampled["planned"] == 16 and greedy["planned"] == 4
    assert sampled["fixed_window_mean"]["net_return_pct"] == pytest.approx(2.5)
    assert greedy["fixed_window_mean"]["net_return_pct"] == 50
    rows.remove(
        next(
            r
            for r in rows
            if r["identity"]["sample_index"] == 0 and r["identity"]["rollout_n"] == 4
        )
    )
    missing = report.fixed_population(helper, rows, grid, step=0, lane="sampled")
    assert missing["fixed_window_mean"] is None and missing["complete_windows"] == 3
    assert missing["windows"][0]["fixed_sibling_mean"] is None
    assert missing["windows"][0]["complete_outcome_siblings"] == 3
    assert (
        report.fixed_population(helper, rows, grid, step=0, lane="greedy")["fixed_window_mean"][
            "net_return_pct"
        ]
        == 50
    )


def test_complete_308_run_retains_separate_curves_and_original_probe_summaries(
    tmp_path, monkeypatch
):
    run_fixture(tmp_path)
    original = Path.read_bytes

    def read_bytes(path):
        assert path.suffix != ".jsonl", "raw evidence must be streamed"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = report.main(["--directory", str(tmp_path), "--write"])
    assert result["status"] == "COMPLETE", result["problems"] + result["pending"]
    assert result["totals"]["terminal_complete"] == 308
    assert result["all_expected_trajectories_observed"]
    sft = result["cells"]["sft0"]
    assert [
        v["population"]["fixed_window_mean"]["net_return_pct"] for v in sft["evaluations"]
    ] == pytest.approx([50, 3.5])
    m1 = result["cells"]["m1"]
    assert m1["groups"]["train"]["planned_windows"] == 16
    assert m1["groups"]["eval_greedy"]["planned"] == 16
    assert m1["groups"]["eval_sampled"]["planned"] == 64
    assert len(result["evaluation_curves"]) == 6
    for curve in result["evaluation_curves"]:
        assert curve["request_pairing_claimed"] is False
        assert [p["policy_step"] for p in curve["points"]] == [2, 4, 6, 8]
        for point in curve["points"]:
            assert point["wins_among_complete_windows"] == 4
            expected = (
                0.2 if curve["target"] == "m2_eta025" and curve["reference"] == "sft0" else 0.1
            )
            assert point["fixed_four_window_mean_delta"][
                "net_return_percentage_points"
            ] == pytest.approx(expected)
            assert point["reference_policy_step"] == (
                0 if curve["reference"] == "sft0" else point["policy_step"]
            )
    probe = m1["rounds"][0]["actor_update_as_recorded"][0]
    assert "rows" not in probe and probe["detail_rows_at_source"] == 160
    assert probe["groups"] == actor_record(report.cell_specs(settings())[1], 1)["groups"]
    assert (tmp_path / "FINAL_RESULTS.md").exists()
    assert len((tmp_path / "summary.csv").read_text().splitlines()) == 19
    assert "SGT（UTC+8）" in (tmp_path / "LIVE_STATUS.md").read_text()


def test_duplicate_or_wrong_window_never_selects_a_convenient_copy(tmp_path):
    spec = report.cell_specs(settings())[0]
    grid = report.expected_grid(spec)
    key, episode = next(iter(grid.items()))
    raw = fixtures.raw_trajectory(key, episode)
    wrong = fixtures.raw_trajectory((0, 0, 1, True), "wrong-window")
    path = tmp_path / "sft0/artifacts/screening_evidence/rows.jsonl"
    write_jsonl(path, [raw, raw, wrong])
    evidence = report.Evidence(tmp_path)
    rows, excluded, count = report.read_trajectories(
        evidence, report.load_helpers(), tmp_path / "sft0", spec, grid, "calibration"
    )
    assert not rows and len(excluded) == 2 and count == 3


def test_nonfinite_or_incomplete_sibling_does_not_become_zero_or_partial_mean():
    spec = report.cell_specs(settings())[0]
    grid, rows = scalar_rows(spec)
    bad = next(r for r in rows if r["identity"]["rollout_n"] == 1)
    bad["final_cash"] = float("nan")
    pop = report.fixed_population(report.load_helpers(), rows, grid, step=0, lane="sampled")
    assert pop["fixed_window_mean"] is None and pop["complete_windows"] == 3
    bad["final_cash"] = 100
    bad["terminal_eligible"], bad["reported_status"] = False, "REQUEST_CAP"
    bad["problems"] = ["trajectory status REQUEST_CAP"]
    pop = report.fixed_population(report.load_helpers(), rows, grid, step=0, lane="sampled")
    assert pop["fixed_window_mean"] is None
    assert pop["incomplete_observed"] == 1 and pop["requests_observed"] == 320
    assert pop["reported_status_counts"]["REQUEST_CAP"] == 1


def test_missing_one_window_only_invalidates_its_lane_and_checkpoint(tmp_path):
    run_fixture(tmp_path)
    result = report.build(tmp_path)
    affected = next(
        item
        for item in result["cells"]["m2_eta025"]["evaluations"]
        if item["sampling"] == "sampled" and item["policy_step"] == 4
    )
    affected["population"]["windows"][0]["fixed_sibling_mean"] = None
    curves = report.paired_curves(result["cells"])
    for curve in curves:
        incomplete = curve["target"] == "m2_eta025" and curve["sampling"] == "sampled"
        assert [p["complete_pairs"] for p in curve["points"]] == (
            [4, 3, 4, 4] if incomplete else [4, 4, 4, 4]
        )
        if incomplete:
            point = curve["points"][1]
            assert point["fixed_four_window_mean_delta"] is None
            assert point["pairs"][0]["delta_target_minus_reference"] is None


def test_wrong_validation_mode_or_sampling_parameters_is_visible(tmp_path):
    spec = report.cell_specs(settings())[0]
    grid = report.expected_grid(spec)
    records = [receipt(key, episode, spec) for key, episode in grid.items()]
    records[1]["mode"] = "greedy"
    records[2]["forwarded_sampling_params"]["temperature"] = 0.7
    write_jsonl(tmp_path / "sft0/artifacts/paired_validation/rows.jsonl", records)
    result = report.paired_validation_readout(
        report.Evidence(tmp_path), tmp_path / "sft0", spec, "execution", grid
    )
    assert len(result["problems"]) == 2 and not result["complete"]
    assert not result["pending"]


def test_missing_actor_probe_not_replaced_by_optimizer_and_bad_totals_visible(tmp_path):
    spec = report.cell_specs(settings())[1]
    path = tmp_path / "m1/artifacts/actor_update/rows.jsonl"
    records = [actor_record(spec, step) for step in range(1, 8)]
    records[0]["groups"]["legal"]["step_count"] = 159
    write_jsonl(path, records)
    result = report.actor_probe_readout(
        report.Evidence(tmp_path), tmp_path / "m1", spec, "execution"
    )
    assert not result["summary_complete"] and result["observed_records"] == 7
    assert any("totals differ" in p for p in result["problems"])
    assert any("not yet complete" in p for p in result["pending"])


def test_live_missing_grid_has_no_final_and_fixed_means_are_null(tmp_path):
    run_fixture(tmp_path, complete=False)
    result = report.main(["--directory", str(tmp_path), "--write"])
    assert result["status"] == "INCOMPLETE" and not result["problems"]
    assert result["totals"]["missing"] == 308
    assert not (tmp_path / "FINAL_RESULTS.md").exists()
    for cell in result["cells"].values():
        assert all(item["population"]["fixed_window_mean"] is None for item in cell["evaluations"])


def test_wrong_profile_or_learning_rate_rejected():
    config = settings()
    config["training_coverage"]["profile"] = "common_sft_training_tuning_v1"
    with pytest.raises(ValueError, match="profile"):
        report.cell_specs(config)
    config = settings()
    config["training"]["learning_rate"] = 1e-6
    with pytest.raises(ValueError, match="learning rates"):
        report.cell_specs(config)
