"""Fixed-grid completeness, actual-rate checks and honest financial contrasts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agents import parameter_grid_readout as readout


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def put_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def profiles():
    return {
        seed: SimpleNamespace(
            seed=seed,
            name=f"common_sft_parameter_grid_seed{seed}_v1",
            train_ids=readout.TRAIN_IDS,
            eval_ids=readout.EVAL_IDS,
            evaluation_steps=(2, 4),
            rounds=4,
            planned_trajectories=972,
        )
        for seed in readout.SEEDS
    }


def trajectory(seed, cell_id, step, episode, sibling, validate):
    lr, eta, arm = readout.CELLS[cell_id]
    # Eta improves greedy and reverses sampled at step4; LR effect cancels within LR.
    value = 0.01 if cell_id == "sft0" else 0.012 + (lr / 1e-7) * 0.001
    if arm == "M2":
        value += eta * (0.02 if step == 2 or sibling == 0 else -0.02)
    # Window differences must preserve both positive and negative market variation.
    value += readout.EVAL_IDS.index(episode) / 10000 if validate else 0
    metrics = dict.fromkeys(readout.METRICS, 0.1)
    metrics.update(
        raw_net_return=value,
        transaction_cost=100.0,
        terminal_score=-2.0,
        session_count=21,
        step_count=20,
        invalid_count=0,
    )
    state = dict(
        cash=700.0,
        portfolio_equity=1000.0,
        positions={"AAPL": {"weight": 0.2}, "MSFT": {"weight": 0.1}},
    )
    observation = json.dumps(
        [dict(role="user", content="<tool_response>\n" + json.dumps(state) + "\n</tool_response>")]
    )
    interaction = dict(
        decision_session_index=19,
        result_session_index=20,
        info=dict(
            action="buy",
            invalid_action=False,
            equity=1000.0,
            fills=[dict(symbol="AAPL", side="buy")],
        ),
        observation=dict(text=observation),
        response=dict(text='<tool_call>{"name":"buy","arguments":{"symbol":"AAPL"}}</tool_call>'),
    )
    return dict(
        base_seed=seed,
        arm="M1" if arm == "SFT" else arm,
        complete=True,
        status="COMPLETE",
        identity=dict(step=step, rollout_n=sibling, validate=validate, sample_index=0),
        episode_id=episode,
        metrics=metrics,
        interactions=[interaction],
        steps=[
            dict(action="hold" if i < 19 else "buy", invalid_action=False, done=i == 19)
            for i in range(20)
        ],
    )


def diagnostics(seed, cell_id):
    lr, eta, arm = readout.CELLS[cell_id]
    groups = {
        name: dict(
            step_count=20,
            positive_advantage_steps=10,
            negative_advantage_steps=10,
            mean_advantage_token_weighted=0.0,
            mean_delta_logp_token_weighted=0.001,
            advantage_aligned_delta_token_fraction=0.5,
            post_ratio_outside_clip_fraction=0.0,
        )
        for name in ("all_real", "legal", "invalid")
    }
    actor, critic, residual = [], [], []
    for step in range(1, 5):
        common = dict(
            status="COMPLETE", arm=arm, policy_step=step, execution_identity_sha256="identity"
        )
        actor.append(
            dict(
                **common,
                actor_learning_rates=[lr],
                optimizer_step_count=1,
                optimizer_grad_norms_before_clip=[3.0],
                original_training_pg_loss=0.001,
                real_step_count=160,
                groups=groups,
            )
        )
        critic.append(
            dict(
                **common,
                critic_learning_rates=[1e-6],
                optimizer_grad_norms_before_clip=[30.0],
                original_training_vf_loss=3.0,
                pre=dict(rmse=3.0, explained_variance=-0.2, mean_bias_value_minus_return=2.0),
                post=dict(rmse=2.5, explained_variance=-0.1, mean_bias_value_minus_return=1.0),
            )
        )
        residual.append(
            dict(
                **common,
                run_seed=seed,
                eta=eta,
                action_diversity_rate=0.5,
                invalid_step_count=0,
                real_step_count=160,
                whitening_std=2.0,
                pooling_changed_row_rate=0.3,
                all_real_steps=dict(
                    mean_abs_S=eta * 0.2,
                    std_S=eta * 0.3,
                    std_U=2.0,
                    corr_U_S=0.1 if eta else None,
                    nonzero_S_rate=0.4 if eta else 0.0,
                ),
                common_validity_penalty=dict(
                    financial_reward_sum=3.0, training_reward_sum=3.0, penalty_sum=0.0
                ),
            )
        )
    return dict(actor_update=actor, critic_fit=critic, sibling_diagnostics=residual)


@pytest.fixture
def grid(tmp_path):
    for seed in readout.SEEDS:
        summaries = []
        for cell_id, (lr, eta, arm) in readout.CELLS.items():
            base = tmp_path / f"seed{seed}" / cell_id
            evaluation = [
                trajectory(seed, cell_id, step, episode, sibling, True)
                for step in ((0,) if cell_id == "sft0" else (2, 4))
                for episode in readout.EVAL_IDS
                for sibling in range(5)
            ]
            training = (
                []
                if cell_id == "sft0"
                else [
                    trajectory(seed, cell_id, step, episode, sibling, False)
                    for step in range(1, 5)
                    for episode in readout.TRAIN_IDS[(step - 1) * 2 : step * 2]
                    for sibling in range(4)
                ]
            )
            put_rows(base / "artifacts/screening_evidence/rollouts.jsonl", evaluation + training)
            put(
                base / "settings.json",
                dict(
                    seeds=[seed], training=dict(learning_rate=lr or 2e-7, critic_learning_rate=1e-6)
                ),
            )
            summary = dict(
                status="COMPLETE",
                seed=seed,
                returncode=0,
                cell=dict(cell_id=cell_id, arm="M1" if arm == "SFT" else arm, eta=eta),
                actual_rounds=0 if cell_id == "sft0" else 4,
                execution_identity_sha256="identity",
                critic_initial_head_sha256=str(seed)[0] * 64,
                evaluation_by_step={},
                evidence=dict(
                    trajectories=(
                        dict(eval=len(evaluation))
                        if cell_id == "sft0"
                        else dict(train=len(training), eval=len(evaluation))
                    ),
                    optimizer_evidence=dict(
                        detected_role_counts=({} if cell_id == "sft0" else dict(actor=4, critic=4))
                    ),
                ),
                actor_update=dict(observed_updates=0 if cell_id == "sft0" else 4),
                critic_fit=dict(updates=0 if cell_id == "sft0" else 4),
            )
            for step in (0,) if cell_id == "sft0" else (2, 4):
                summary["evaluation_by_step"][str(step)] = {}
                for lane, siblings in readout.LANES.items():
                    selected = [
                        r
                        for r in evaluation
                        if r["identity"]["step"] == step and r["identity"]["rollout_n"] in siblings
                    ]
                    summary["evaluation_by_step"][str(step)][lane] = dict(
                        trajectories=len(selected),
                        decisions=20 * len(selected),
                        invalid_decisions=0,
                        **{
                            "mean_" + name: sum(r["metrics"][name] for r in selected)
                            / len(selected)
                            for name in readout.METRICS
                        },
                    )
            if cell_id != "sft0":
                for area, rows in diagnostics(seed, cell_id).items():
                    put_rows(base / f"artifacts/{area}/updates.jsonl", rows)
            put(base / "summary.json", summary)
            summaries.append(summary)
        put(
            tmp_path / f"seed{seed}/state.json",
            dict(
                status="COMPLETE",
                profile=f"common_sft_parameter_grid_seed{seed}_v1",
                seed=seed,
                planned_trajectories=972,
                cells=summaries,
                train_episode_ids=readout.TRAIN_IDS,
                evaluation_episode_ids=readout.EVAL_IDS,
                evaluation_steps=[2, 4],
            ),
        )
    return tmp_path


def read(base):
    return readout.read_grid(base, expected_profiles=profiles())


def test_complete_grid_preserves_steps_lanes_equal_lr_and_behavior(grid):
    value = read(grid)
    assert value["status"] == "PASS", value["issues"]
    assert value["complete_trajectories"] == 1944
    assert len(value["evaluation_curve"]) == 52
    assert len(value["window_curve"]) == 624
    assert len(value["paired_comparisons"]) == 80
    assert len(value["window_comparisons"]) == 960
    assert len(value["training_diagnostics"]) == 48
    contrasts = {
        (r["seed"], r["cell_id"], r["policy_step"], r["lane"]): r
        for r in value["paired_comparisons"]
        if r["comparison"] == "M2_minus_M1"
    }
    assert contrasts[(42, "m2_lr1e7_eta025", 2, "sampled")][
        "net_return_percentage_points"
    ] == pytest.approx(0.5)
    assert contrasts[(42, "m2_lr2e7_eta025", 4, "sampled")][
        "net_return_percentage_points"
    ] == pytest.approx(-0.5)
    assert contrasts[(42, "m2_lr2e7_eta025", 4, "sampled")]["reference_cell"] == "m1_lr2e7"
    baseline = value["evaluation_curve"][0]
    assert baseline["mean_cash_fraction"] == pytest.approx(0.7)
    assert baseline["mean_position_concentration"] == pytest.approx(0.2)
    assert baseline["mean_held_stock_count"] == 2
    assert baseline["filled_symbols"] == ["AAPL"]
    assert baseline["held_symbols"] == ["AAPL", "MSFT"]
    assert value["training_diagnostics"][0]["critic_rmse_change"] == -0.5
    assert value["training_diagnostics"][0]["residual_scaled_std_ratio"] == 0.0
    assert value["formal_performance_claim"] is False


@pytest.mark.parametrize("cell_id,eta", [("m2_lr1e7_eta0125", 0.125), ("m2_lr1e7_eta025", 0.25)])
def test_residual_strength_uses_already_scaled_runtime_diagnostics_once(grid, cell_id, eta):
    value = read(grid)
    diagnostics = [row for row in value["training_diagnostics"] if row["cell_id"] == cell_id]
    assert len(diagnostics) == 8
    for row in diagnostics:
        assert row["residual_std_S"] == pytest.approx(eta * 0.3)
        assert row["residual_mean_abs_S"] == pytest.approx(eta * 0.2)
        assert row["residual_scaled_std_ratio"] == pytest.approx(eta * 0.3 / 2.0)


def test_residual_strength_with_zero_gae_std_is_undefined(grid):
    path = grid / "seed42/m2_lr1e7_eta025/artifacts/sibling_diagnostics/updates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["all_real_steps"]["std_U"] = 0.0
    put_rows(path, rows)
    value = read(grid)
    diagnostics = [
        row
        for row in value["training_diagnostics"]
        if row["seed"] == 42 and row["cell_id"] == "m2_lr1e7_eta025"
    ]
    assert len(diagnostics) == 4
    assert all(row["residual_scaled_std_ratio"] is None for row in diagnostics)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "wrong_seed",
        "wrong_lr",
        "actual_lr",
        "missing_update",
        "wrong_summary",
        "head",
        "optimizer_budget",
    ],
)
def test_damaged_complete_cell_is_excluded_and_never_passes(grid, damage):
    cell = grid / "seed43/m2_lr2e7_eta025"
    evidence = cell / "artifacts/screening_evidence/rollouts.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    if damage == "missing":
        put_rows(evidence, rows[:-1])
    elif damage == "duplicate":
        put_rows(evidence, rows + rows[:1])
    elif damage == "wrong_seed":
        rows[0]["base_seed"] = 42
        put_rows(evidence, rows)
    elif damage == "wrong_lr":
        put(
            cell / "settings.json",
            dict(seeds=[43], training=dict(learning_rate=1e-7, critic_learning_rate=1e-6)),
        )
    elif damage in {"actual_lr", "missing_update"}:
        path = cell / "artifacts/actor_update/updates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "actual_lr":
            rows[0]["actor_learning_rates"] = [1e-7]
        else:
            rows.pop()
        put_rows(path, rows)
    else:
        path = grid / "seed43/state.json"
        state = json.loads(path.read_text())
        item = state["cells"][-1]
        if damage == "wrong_summary":
            item["evaluation_by_step"]["4"]["sampled"]["mean_raw_net_return"] += 0.1
        elif damage == "optimizer_budget":
            item["evidence"]["optimizer_evidence"]["detected_role_counts"]["critic"] = 3
        else:
            del item["critic_initial_head_sha256"]
        put(path, state)
    value = read(grid)
    assert value["status"] == "PARTIAL"
    assert value["complete_trajectories"] == 1792
    assert not any(
        r["seed"] == 43 and r["cell_id"] == "m2_lr2e7_eta025" for r in value["evaluation_curve"]
    )
    assert any(r.get("cell_id") == "m2_lr2e7_eta025" for r in value["issues"])


def test_first_seed_complete_is_partial_and_finalization_retains_partial_files(grid):
    (grid / "seed43/state.json").unlink()
    value = readout.write_readout(grid, expected_profiles=profiles())
    assert value["status"] == "PARTIAL"
    assert value["complete_trajectories"] == 972
    assert len(value["paired_comparisons"]) == 40
    with pytest.raises(ValueError, match="PARTIAL readout retained"):
        readout.write_readout(grid, expected_profiles=profiles(), allow_partial=False)
    assert json.loads((grid / "grid_readout.json").read_text())["status"] == "PARTIAL"
    assert (grid / "window_comparisons.csv").exists()


def test_complete_writer_and_reverse_cell_order(grid):
    path = grid / "seed43/state.json"
    state = json.loads(path.read_text())
    state["cells"] = list(reversed(state["cells"]))
    put(path, state)
    value = readout.write_grid_readout(grid, expected_profiles=profiles(), allow_partial=False)
    assert value["status"] == "PASS"
    assert len((grid / "evaluation_curve.csv").read_text().splitlines()) == 53


def test_only_complete_cells_enter_partial_readout(grid):
    path = grid / "seed42/state.json"
    state = json.loads(path.read_text())
    state["status"] = "RUNNING"
    state["cells"][-1]["status"] = "RUNNING"
    put(path, state)
    value = read(grid)
    assert value["status"] == "PARTIAL"
    assert value["complete_trajectories"] == 1792


def test_duplicate_cell_or_controller_wrong_seed_cannot_pass(grid):
    path = grid / "seed42/state.json"
    state = json.loads(path.read_text())
    state["cells"].append(state["cells"][0])
    put(path, state)
    assert read(grid)["status"] == "PARTIAL"
    state["seed"] = 43
    put(path, state)
    assert read(grid)["complete_trajectories"] == 972


def test_invalid_profile_is_rejected_before_any_averaging(tmp_path):
    expected = profiles()
    expected[42].eval_ids = readout.EVAL_IDS[:-1]
    with pytest.raises(ValueError, match="outside the fixed"):
        readout.read_grid(tmp_path, expected_profiles=expected)


def v2_profiles():
    expected = profiles()
    for seed, profile in expected.items():
        profile.name = f"common_sft_parameter_grid_seed{seed}_v2"
        profile.evaluation_incomplete_mode = "retain_prefix_v1"
    return expected


def promote_v2(grid):
    for seed in readout.SEEDS:
        path = grid / f"seed{seed}/state.json"
        state = json.loads(path.read_text())
        state["profile"] = f"common_sft_parameter_grid_seed{seed}_v2"
        for summary in state["cells"]:
            summary["financial_status"] = "COMPLETE"
            for lanes in summary["evaluation_by_step"].values():
                for row in lanes.values():
                    n = row["trajectories"]
                    row.update(
                        planned_trajectories=n,
                        attempted_trajectories=n,
                        complete_trajectories=n,
                        incomplete_trajectories=0,
                    )
            put(grid / f"seed{seed}" / summary["cell"]["cell_id"] / "summary.json", summary)
        put(path, state)


def update_v2_summary(grid, seed, cell_id):
    state_path = grid / f"seed{seed}/state.json"
    state = json.loads(state_path.read_text())
    summary = next(row for row in state["cells"] if row["cell"]["cell_id"] == cell_id)
    path = grid / f"seed{seed}" / cell_id / "artifacts/screening_evidence/rollouts.jsonl"
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    summary["financial_status"] = (
        "INCOMPLETE" if any(not row["complete"] for row in raw) else "COMPLETE"
    )
    for step, lanes in summary["evaluation_by_step"].items():
        for lane in lanes:
            attempted = [
                r
                for r in raw
                if r["identity"]["validate"]
                and r["identity"]["step"] == int(step)
                and r["identity"]["rollout_n"] in readout.LANES[lane]
            ]
            completed = [r for r in attempted if r["complete"]]
            lanes[lane] = dict(
                trajectories=len(completed),
                decisions=20 * len(completed),
                invalid_decisions=0,
                planned_trajectories=len(attempted),
                attempted_trajectories=len(attempted),
                complete_trajectories=len(completed),
                incomplete_trajectories=len(attempted) - len(completed),
                **{
                    "mean_" + metric: (
                        sum(row["metrics"][metric] for row in completed) / len(completed)
                        if completed
                        else None
                    )
                    for metric in readout.METRICS
                },
            )
    put(state_path, state)
    put(grid / f"seed{seed}" / cell_id / "summary.json", summary)


def fail_slot(grid, seed, cell_id, step, episode, sibling):
    path = grid / f"seed{seed}" / cell_id / "artifacts/screening_evidence/rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(
        r
        for r in rows
        if r["identity"]["step"] == step
        and r["episode_id"] == episode
        and r["identity"]["rollout_n"] == sibling
    )
    target.update(
        complete=False,
        status="INCOMPLETE",
        metrics=None,
        terminal_score=None,
        evaluation_failure_profile="retain_prefix_v1",
        failure_kind="trajectory_max_steps",
        transport_output_returned=True,
        transport_step_count=256,
        transport_rewards_are_not_financial=True,
        error="max steps reached without terminal",
        requests=[dict(status="COMPLETE") for _ in range(256)],
        steps=[
            dict(
                action="invalid",
                invalid_action=True,
                done=False,
                decision_session_index=3,
                result_session_index=3,
            )
            for _ in range(256)
        ],
        interactions=[
            dict(
                decision_session_index=3,
                result_session_index=3,
                info=dict(
                    invalid_action=True, error="buy target must exceed current weight", fills=[]
                ),
                response=dict(
                    text='<tool_call>{"name":"buy","arguments":{"symbol":"AAPL"}}</tool_call>'
                ),
            )
            for _ in range(256)
        ],
    )
    put_rows(path, rows)
    update_v2_summary(grid, seed, cell_id)
    return path


def test_v2_failed_eval_executes_fixed_budget_but_never_performance_passes(grid):
    promote_v2(grid)
    fail_slot(grid, 42, "sft0", 0, "formal-train-0399", 0)
    value = readout.write_readout(grid, allow_partial=False, expected_profiles=v2_profiles())
    assert value["status"] == "COMPLETE_WITH_EVALUATION_FAILURES", value["issues"]
    assert value["execution_status"] == "COMPLETE_WITH_EVALUATION_FAILURES"
    assert value["financial_status"] == "INCOMPLETE"
    assert value["performance_pass"] is False
    assert value["attempted_trajectories"] == 1944
    assert value["complete_trajectories"] == 1943
    assert value["incomplete_trajectories"] == 1
    assert value["evaluation_coverage"]["planned_trajectories"] == 1560
    curve = next(
        row
        for row in value["evaluation_curve"]
        if row["seed"] == 42 and row["cell_id"] == "sft0" and row["lane"] == "greedy"
    )
    assert curve["planned_trajectories"] == curve["attempted_trajectories"] == 12
    assert curve["trajectories"] == curve["complete_trajectories"] == 11
    assert curve["incomplete_trajectories"] == 1
    assert curve["decisions"] == 220  # Failed prefix's 256 calls are separate behavior evidence.
    failure = value["failure_summary"][0]
    assert failure["raw_net_return"] is None
    assert failure["tool_buy_requests"] == 256
    assert failure["nonadvancing_steps"] == 256
    assert failure["maximum_consecutive_invalid_steps"] == 256
    assert failure["maximum_repeated_response_run"] == 256
    assert (grid / "failure_summary.csv").exists()
    prefix_behavior = next(
        row for row in value["behavior_summary"] if row["trajectory_status"] == "INCOMPLETE"
    )
    assert prefix_behavior["invalid_decisions"] == 256
    assert prefix_behavior["mean_raw_net_return"] is None
    comparison = next(
        row
        for row in value["paired_comparisons"]
        if row["seed"] == 42
        and row["cell_id"] == "m1_lr1e7"
        and row["lane"] == "greedy"
        and row["policy_step"] == 4
    )
    assert comparison["common_complete_slots"] == comparison["common_complete_windows"] == 11
    assert comparison["reference_incomplete_trajectories"] == 1
    assert comparison["left_complete_slots_excluded_by_pairing"] == 1
    empty_window = next(
        row
        for row in value["window_comparisons"]
        if row["seed"] == 42
        and row["cell_id"] == "m1_lr1e7"
        and row["lane"] == "greedy"
        and row["policy_step"] == 4
        and row["episode_id"] == "formal-train-0399"
    )
    assert empty_window["common_complete_slots"] == 0
    assert empty_window["net_return_percentage_points"] is None


@pytest.mark.parametrize("step", [2, 4])
def test_v2_pair_excludes_peer_complete_slot_and_preserves_equal_window_weight(grid, step):
    promote_v2(grid)
    fail_slot(grid, 43, "m2_lr1e7_eta0125", step, readout.EVAL_IDS[0], 1)
    path = grid / "seed43/m1_lr1e7/artifacts/screening_evidence/rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(
        r
        for r in rows
        if r["identity"]["validate"]
        and r["identity"]["step"] == step
        and r["episode_id"] == readout.EVAL_IDS[0]
        and r["identity"]["rollout_n"] == 1
    )
    target["metrics"]["raw_net_return"] = 100.0
    put_rows(path, rows)
    update_v2_summary(grid, 43, "m1_lr1e7")
    value = readout.read_grid(grid, expected_profiles=v2_profiles())
    assert value["execution_status"] == "COMPLETE_WITH_EVALUATION_FAILURES", value["issues"]
    pair = next(
        row
        for row in value["paired_comparisons"]
        if row["seed"] == 43
        and row["cell_id"] == "m2_lr1e7_eta0125"
        and row["policy_step"] == step
        and row["lane"] == "sampled"
        and row["comparison"] == "M2_minus_M1"
    )
    assert pair["common_complete_slots"] == 47
    assert pair["common_complete_windows"] == 12
    assert pair["reference_complete_slots_excluded_by_pairing"] == 1
    assert pair["net_return_percentage_points"] == pytest.approx(0.25 if step == 2 else -0.25)


@pytest.mark.parametrize(
    "damage",
    ["transport_score", "wrong_marker", "request_incomplete", "terminal_present", "missing_slot"],
)
def test_v2_retention_rejects_foreign_failures_and_fabricated_financial_values(grid, damage):
    promote_v2(grid)
    path = fail_slot(grid, 42, "sft0", 0, readout.EVAL_IDS[0], 0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    failed = next(row for row in rows if not row["complete"])
    if damage == "transport_score":
        failed["metrics"] = dict(raw_net_return=0.0)
    elif damage == "wrong_marker":
        failed["evaluation_failure_profile"] = "unreviewed"
    elif damage == "request_incomplete":
        failed["requests"][0]["status"] = "INCOMPLETE"
    elif damage == "terminal_present":
        failed["terminal_score"] = 0.0
    else:
        rows.remove(failed)
    put_rows(path, rows)
    value = readout.read_grid(grid, expected_profiles=v2_profiles())
    assert value["status"] == "PARTIAL"
    assert value["attempted_trajectories"] == 1884


def test_v2_incomplete_training_still_hard_fails_cell(grid):
    promote_v2(grid)
    fail_slot(grid, 42, "m1_lr1e7", 1, readout.TRAIN_IDS[0], 0)
    value = readout.read_grid(grid, expected_profiles=v2_profiles())
    assert value["status"] == "PARTIAL"
    assert value["attempted_trajectories"] == 1792
    assert any("not a retained evaluation prefix" in issue["reason"] for issue in value["issues"])


def test_v1_cannot_opt_into_prefix_by_changing_raw_alone(grid):
    path = fail_slot(grid, 42, "sft0", 0, readout.EVAL_IDS[0], 0)
    assert path.exists()
    assert read(grid)["status"] == "PARTIAL"


def test_v2_all_complete_retains_execution_and_financial_completion(grid):
    promote_v2(grid)
    value = readout.read_grid(grid, expected_profiles=v2_profiles())
    assert value["status"] == "PASS", value["issues"]
    assert value["execution_status"] == value["financial_status"] == "COMPLETE"
    assert value["performance_pass"] is True
    assert value["failure_summary"] == []
