"""Fresh-SFT pooling contrasts, strict identities and incomplete evidence semantics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import test_parameter_grid_readout as evidence

from src.agents import action_pooling_readout as readout


def profiles():
    return {
        seed: SimpleNamespace(
            seed=seed,
            name=f"shared_recovery_sft_action_pooling_seed{seed}_v1",
            train_ids=readout.TRAIN_IDS,
            eval_ids=readout.EVAL_IDS,
            evaluation_steps=(2, 4),
            rounds=4,
            actor_learning_rate=1e-7,
            planned_trajectories=668,
            evaluation_incomplete_mode="retain_prefix_v1",
            cell_specs=tuple(
                SimpleNamespace(
                    cell_id=cell,
                    arm="M1" if arm == "SFT" else arm,
                    eta=eta,
                    rounds=0 if cell == "sft0" else 4,
                    critic_lr=1e-6,
                )
                for cell, (_, eta, arm) in readout.CELLS.items()
            ),
            cell_actor_learning_rates=tuple(
                (cell, lr or 1e-7) for cell, (lr, _, _) in readout.CELLS.items()
            ),
        )
        for seed in readout.SEEDS
    }


@pytest.fixture
def grid(tmp_path):
    for seed in readout.SEEDS:
        summaries = []
        for cell, (lr, eta, arm) in readout.CELLS.items():
            base = tmp_path / f"seed{seed}" / cell
            original = "m2_lr1e7_eta025" if arm == "M2-plain" else cell
            raw = []
            for validate in (True, False):
                if not validate and cell == "sft0":
                    continue
                for step in ((0,) if cell == "sft0" else (2, 4)) if validate else range(1, 5):
                    episodes = (
                        readout.EVAL_IDS
                        if validate
                        else readout.TRAIN_IDS[(step - 1) * 2 : step * 2]
                    )
                    for episode in episodes:
                        # RLHFDataset's extra_info.index survives batch slicing.
                        dataset_ids = readout.EVAL_IDS if validate else readout.TRAIN_IDS
                        ordinal = dataset_ids.index(episode)
                        for sibling in range(5 if validate else 4):
                            row = evidence.trajectory(
                                seed,
                                original,
                                step,
                                episode,
                                sibling,
                                validate,
                            )
                            row["arm"] = "M1" if arm == "SFT" else arm
                            row["identity"]["sample_index"] = ordinal
                            if arm == "M2-plain":
                                row["metrics"]["raw_net_return"] -= 0.001
                            raw.append(row)
            evidence.put_rows(base / "artifacts/screening_evidence/rollouts.jsonl", raw)
            evidence.put(
                base / "settings.json",
                dict(
                    seeds=[seed],
                    training=dict(learning_rate=lr or 1e-7, critic_learning_rate=1e-6),
                    observation_profile="history_features_recovery_v2",
                    action_protocol="single_tool_portfolio_v1",
                    generation=dict(enable_thinking=False),
                    models=dict(
                        screening=dict(name="shared/merged", merge_receipt_sha256="a" * 64)
                    ),
                ),
            )
            rounds = 0 if cell == "sft0" else 4
            summary = dict(
                status="COMPLETE",
                financial_status="COMPLETE",
                seed=seed,
                returncode=0,
                cell=dict(cell_id=cell, arm="M1" if arm == "SFT" else arm, eta=eta),
                actual_rounds=rounds,
                execution_identity_sha256="identity",
                critic_initial_head_sha256=f"{seed:064x}",
                evaluation_by_step={
                    str(step): {lane: {} for lane in evidence.readout.LANES}
                    for step in ((0,) if cell == "sft0" else (2, 4))
                },
                evidence=dict(
                    trajectories=dict(train=rounds * 8, eval=60 if rounds == 0 else 120),
                    optimizer_evidence=dict(detected_role_counts=dict(actor=rounds, critic=rounds)),
                ),
                actor_update=dict(observed_updates=rounds),
                critic_fit=dict(updates=rounds),
            )
            evidence.put(base / "summary.json", summary)
            summaries.append(summary)
            if rounds:
                for area, rows in evidence.diagnostics(seed, original).items():
                    for row in rows:
                        row["arm"] = arm
                    evidence.put_rows(base / f"artifacts/{area}/updates.jsonl", rows)
        evidence.put(
            tmp_path / f"seed{seed}/state.json",
            dict(
                profile=profiles()[seed].name,
                status="COMPLETE",
                seed=seed,
                planned_trajectories=668,
                cells=summaries,
                train_episode_ids=readout.TRAIN_IDS,
                evaluation_episode_ids=readout.EVAL_IDS,
                evaluation_steps=[2, 4],
            ),
        )
        for cell in readout.CELLS:
            evidence.update_v2_summary(tmp_path, seed, cell)
    return tmp_path


def read(grid):
    return readout.read_grid(grid, expected_profiles=profiles())


def sync_summary(grid, seed, cell, mutate):
    path = grid / f"seed{seed}/state.json"
    state = json.loads(path.read_text())
    summary = next(s for s in state["cells"] if s["cell"]["cell_id"] == cell)
    mutate(summary)
    evidence.put(grid / f"seed{seed}/{cell}/summary.json", summary)
    evidence.put(path, state)


def test_complete_comparison_is_descriptive_and_preserves_all_strata(grid):
    value = read(grid)
    assert value["status"] == "OBSERVED", value["issues"]
    assert value["execution_status"] == value["financial_status"] == "COMPLETE"
    assert value["attempted_trajectories"] == value["complete_trajectories"] == 1336
    assert value["training_coverage"]["complete_trajectories"] == 256
    assert value["evaluation_coverage"]["complete_trajectories"] == 1080
    assert len(value["evaluation_curve"]) == 36
    assert len(value["window_curve"]) == 432
    assert len(value["paired_comparisons"]) == 64
    assert len(value["window_comparisons"]) == 768
    assert len(value["training_diagnostics"]) == 32
    assert value["performance_pass"] is value["formal_performance_claim"] is False
    contrasts = {
        r["comparison"]: r
        for r in value["paired_comparisons"]
        if r["seed"] == 42 and r["policy_step"] == 4 and r["lane"] == "sampled"
    }
    assert contrasts["M2_minus_M2plain"]["net_return_percentage_points"] == pytest.approx(0.1)
    assert contrasts["M2_minus_M1_same_lr"]["reference_cell"] == "m1_lr1e7"
    assert contrasts["M2_minus_M1_high_lr"]["reference_cell"] == "m1_lr2e7"
    assert contrasts["M2plain_minus_M1_same_lr"]["cell_id"] == "m2_plain_lr1e7_eta025"
    for r in value["training_diagnostics"]:
        assert r["residual_scaled_std_ratio"] == pytest.approx(r["residual_std_S"] / 2)


@pytest.mark.parametrize(
    "step,episodes,indices",
    [
        (1, ("formal-train-0001", "formal-train-0065"), (0, 1)),
        (2, ("formal-train-0009", "formal-train-0073"), (2, 3)),
        (3, ("formal-train-0017", "formal-train-0149"), (4, 5)),
        (4, ("formal-train-0025", "formal-train-0157"), (6, 7)),
    ],
)
def test_native_prepared_dataset_indices_survive_all_four_training_batches(step, episodes, indices):
    # These are the actual prepared/train.parquet extra_info.index values from
    # the completed seed42/M1 cell, independent of this fixture's enumeration.
    for episode, index in zip(episodes, indices, strict=True):
        for sibling in range(4):
            readout._identity(
                dict(
                    episode_id=episode,
                    rollout_identity=dict(
                        step=step,
                        sample_index=index,
                        rollout_n=sibling,
                        validate=False,
                    ),
                )
            )


@pytest.mark.parametrize(
    "step,episode,index",
    [
        (1, "formal-train-0009", 2),
        (2, "formal-train-0001", 0),
        (3, "formal-train-0025", 6),
        (4, "formal-train-0017", 4),
    ],
)
def test_correct_global_index_cannot_move_an_episode_to_another_training_round(
    step, episode, index
):
    with pytest.raises(ValueError, match="raw sample_index"):
        readout._identity(
            dict(
                episode_id=episode,
                identity=dict(step=step, sample_index=index, rollout_n=0, validate=False),
            )
        )


@pytest.mark.parametrize(
    "step,episode,index",
    [
        (2, "formal-train-0009", 0),
        (2, "formal-train-0073", 1),
        (3, "formal-train-0017", 0),
        (3, "formal-train-0149", 1),
        (4, "formal-train-0025", 0),
        (4, "formal-train-0157", 1),
        (2, "formal-train-0009", 3),
    ],
)
def test_batch_local_or_another_episode_index_is_rejected(step, episode, index):
    with pytest.raises(ValueError, match="raw sample_index"):
        readout._identity(
            dict(
                episode_id=episode,
                identity=dict(step=step, sample_index=index, rollout_n=0, validate=False),
            )
        )


@pytest.mark.parametrize(
    "damage",
    [
        "raw_sample",
        "train_batch_local_index",
        "wrong_train_round",
        "raw_duplicate",
        "wrong_arm",
        "wrong_lr",
        "missing_update",
        "summary_diverged",
        "wrong_summary",
        "optimizer_count",
        "missing_sft_head",
        "head_differs",
        "old_observation",
        "different_model",
    ],
)
def test_damaged_evidence_cannot_complete(grid, damage):
    cell = "m2_plain_lr1e7_eta025"
    base = grid / f"seed43/{cell}"
    if damage in {
        "raw_sample",
        "raw_duplicate",
        "wrong_arm",
        "train_batch_local_index",
        "wrong_train_round",
    }:
        path = base / "artifacts/screening_evidence/rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "raw_sample":
            rows[0]["identity"]["sample_index"] = 99
        elif damage in {"train_batch_local_index", "wrong_train_round"}:
            target = next(
                row
                for row in rows
                if row["identity"]["validate"] is False and row["identity"]["step"] == 3
            )
            if damage == "train_batch_local_index":
                target["identity"]["sample_index"] = 0
            else:
                target["identity"]["step"] = 1
        elif damage == "raw_duplicate":
            rows.append(rows[0])
        else:
            rows[0]["arm"] = "M2"
        evidence.put_rows(path, rows)
    elif damage in {"wrong_lr", "old_observation", "different_model"}:
        path = base / "settings.json"
        settings = json.loads(path.read_text())
        if damage == "wrong_lr":
            settings["training"]["learning_rate"] = 2e-7
        elif damage == "old_observation":
            settings["observation_profile"] = "history_features_v1"
        else:
            settings["models"]["screening"]["name"] = "another/model"
        evidence.put(path, settings)
    elif damage == "missing_update":
        path = base / "artifacts/actor_update/updates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        evidence.put_rows(path, rows[:-1])
    elif damage == "summary_diverged":
        path = base / "summary.json"
        summary = json.loads(path.read_text())
        summary["actual_rounds"] = 3
        evidence.put(path, summary)
    elif damage == "wrong_summary":
        sync_summary(
            grid,
            43,
            cell,
            lambda s: s["evaluation_by_step"]["4"]["sampled"].update(
                mean_raw_net_return=1,
            ),
        )
    elif damage == "optimizer_count":
        sync_summary(
            grid,
            43,
            cell,
            lambda s: s["evidence"]["optimizer_evidence"]["detected_role_counts"].update(critic=3),
        )
    elif damage == "missing_sft_head":
        sync_summary(grid, 43, "sft0", lambda s: s.pop("critic_initial_head_sha256"))
    else:
        sync_summary(grid, 43, "sft0", lambda s: s.update(critic_initial_head_sha256="f" * 64))
    result = read(grid)
    assert result["status"] == result["execution_status"] == "PARTIAL"
    assert result["financial_status"] == "INCOMPLETE"
    assert result["issues"]
    assert result["performance_pass"] is False


def test_failure_has_null_finance_and_pair_uses_only_common_slots(grid):
    episode = readout.EVAL_IDS[0]
    evidence.fail_slot(grid, 42, "m2_plain_lr1e7_eta025", 4, episode, 1)
    # Its peer's excluded sibling is deliberately extreme, so slot intersection
    # and equal window weighting can be distinguished from unpaired averages.
    path = grid / "seed42/m2_lr1e7_eta025/artifacts/screening_evidence/rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(
        r
        for r in rows
        if r["identity"]["validate"]
        and r["identity"]["step"] == 4
        and r["episode_id"] == episode
        and r["identity"]["rollout_n"] == 1
    )
    target["metrics"]["raw_net_return"] = 100
    evidence.put_rows(path, rows)
    evidence.update_v2_summary(grid, 42, "m2_lr1e7_eta025")
    value = read(grid)
    assert value["status"] == "OBSERVED", value["issues"]
    assert value["execution_status"] == "COMPLETE_WITH_EVALUATION_FAILURES"
    assert value["financial_status"] == "INCOMPLETE"
    assert value["complete_trajectories"] == 1335
    assert value["failure_summary"][0]["raw_net_return"] is None
    pair = next(
        r
        for r in value["paired_comparisons"]
        if r["seed"] == 42
        and r["policy_step"] == 4
        and r["lane"] == "sampled"
        and r["comparison"] == "M2_minus_M2plain"
    )
    assert pair["common_complete_slots"] == 47
    assert pair["common_complete_windows"] == 12
    assert pair["left_complete_slots_excluded_by_pairing"] == 1
    assert pair["net_return_percentage_points"] == pytest.approx(0.1)


def test_missing_reference_never_fills_zero_and_partial_writer_retains_csv(grid):
    path = grid / "seed43/state.json"
    state = json.loads(path.read_text())
    state["cells"] = [r for r in state["cells"] if r["cell"]["cell_id"] != "m1_lr1e7"]
    evidence.put(path, state)
    with pytest.raises(ValueError, match="PARTIAL readout retained"):
        readout.write_readout(grid, expected_profiles=profiles(), allow_partial=False)
    value = json.loads((grid / "stage2_readout.json").read_text())
    missing = [
        r
        for r in value["paired_comparisons"]
        if r["seed"] == 43 and r["comparison"] == "M2plain_minus_M1_same_lr"
    ]
    assert len(missing) == 4
    assert all(r["net_return_percentage_points"] is None for r in missing)
    assert all(
        r["reference_missing_trajectories"] == r["left_planned_trajectories"] for r in missing
    )
    assert (grid / "stage2_window_comparisons.csv").exists()
    assert not (grid / "grid_readout.json").exists()


def test_invalid_profile_is_rejected_before_reading_any_cell(tmp_path):
    expected = profiles()
    expected[42].cell_specs[-1].eta = 0.125
    with pytest.raises(ValueError, match="fixed 1336"):
        readout.read_grid(tmp_path, expected_profiles=expected)


def test_default_entry_checks_sealed_contract_and_actual_profiles(grid, monkeypatch):
    from src.agents import action_pooling_comparison as comparison

    checked = []
    monkeypatch.setattr(comparison, "check_contract", lambda directory: checked.append(directory))
    value = readout.read_grid(grid)
    assert checked == [grid.resolve()]
    assert value["execution_status"] == "COMPLETE", value["issues"]
