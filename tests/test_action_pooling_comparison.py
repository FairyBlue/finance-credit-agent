"""Fixed ablation budget, real Hydra routing and fail-closed launch boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.agents import action_pooling_comparison as grid
from src.agents.metadata import RunHashes


def test_equal_four_round_budgets_and_paired_reverse_order():
    assert grid.settings()["planned_trajectories"] == sum(
        p.planned_trajectories for p in grid.PROFILES.values()
    ) == 1336
    assert grid.settings()["training_trajectories"] == 8 * 4 * 8 == 256
    assert grid.settings()["evaluation_trajectories"] == 8 * 2 * 12 * 5 + 2 * 12 * 5
    order = [[s.cell_id for s in p.cell_specs] for p in grid.PROFILES.values()]
    assert order[1] == ["sft0", *reversed(order[0][1:])]
    for p in grid.PROFILES.values():
        assert grid.runner.disk_requirement_bytes(p, 1) == 107 * 1024**3
        assert p.observation_profile == "history_features_recovery_v2"


@pytest.mark.parametrize("change", [
    {"rounds": 1}, {"test_freq": 1}, {"evaluation_steps": (1,)},
    {"observation_profile": "history_features_v1"}, {"archive_completed_actors": False},
])
def test_no_one_round_budget_or_old_observation_escape(change):
    with pytest.raises(ValueError):
        replace(grid.PROFILES[42], **change)


@pytest.mark.parametrize("seed", grid.SEEDS)
@pytest.mark.parametrize("cell_id", [s.cell_id for s in grid.PROFILES[42].cell_specs])
def test_real_hydra_fresh_common_sft_arms_rates_and_four_round_protocol(tmp_path, seed, cell_id):
    hydra = pytest.importorskip("hydra")
    p = grid.PROFILES[seed]
    model = dict(source="local_export", name=str(tmp_path / "new-sft"),
                 merge_receipt_sha256="a" * 64)
    settings = grid.runner.settings(model, profile=p)
    grid.atomic_json(tmp_path / "settings.json", settings)
    manifest = dict(
        hashes={"payload_sha256": "b" * 64},
        datasets={n: dict(path=n + ".parquet", episode_ids=ids, row_count=len(ids))
                  for n, ids in (("train", p.train_ids), ("dev", p.eval_ids))},
        calibration=dict(path="reward_calibration.json", payload_sha256="c" * 64, sha256="f" * 64),
    )
    cell = tmp_path / cell_id
    cell.mkdir()
    config = grid.runner.make_config(tmp_path, manifest, cell_id, profile=p)
    command = grid.runner.build_command(
        config=config, run_settings=grid.runner.cell_settings(settings, cell_id, profile=p),
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=cell, profile=p,
    )
    with hydra.initialize_config_dir(
        config_dir=str(grid.runner.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    spec = next(s for s in p.cell_specs if s.cell_id == cell_id)
    assert actual.actor_rollout_ref.actor.optim.lr == p.actor_lr(cell_id)
    assert actual.critic.optim.lr == 1e-6
    assert actual.data.train_max_samples == 8 and actual.data.val_max_samples == 12
    assert actual.data.train_batch_size == 2 and not actual.data.shuffle
    assert actual.actor_rollout_ref.rollout.n == 4
    assert actual.actor_rollout_ref.rollout.val_kwargs.n == 5
    assert actual.trainer.total_training_steps == (1 if spec.zero else 4)
    assert actual.trainer.test_freq == (1 if spec.zero else 2)
    assert actual.trainer.val_only is spec.zero
    assert actual.trainer.resume_mode == "disable"
    assert actual.data.screening_observation_profile == "history_features_recovery_v2"
    assert actual.data.max_prompt_length == 8192 and actual.data.max_response_length == 4096
    assert actual.critic.model.path == actual.actor_rollout_ref.model.path == model["name"]
    assert actual.actor_rollout_ref.ref.model.path == model["name"]
    assert actual.critic.model.fsdp_config.seed == actual.data.seed == seed
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert spec.arm == env.FINANCE_SCREENING_ARM
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == spec.eta
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert env.FINANCE_GRID_EVALUATION_INCOMPLETE == "retain_prefix_v1"


def test_current_disk_shortage_stops_before_any_state_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "check_contract", lambda base: None)
    monkeypatch.setattr(grid.shutil, "disk_usage", lambda path: SimpleNamespace(free=67 * 1024**3))
    monkeypatch.setattr(grid.runner, "run", lambda *a, **kw: pytest.fail("training started"))
    with pytest.raises(RuntimeError, match="107 GiB"):
        grid.run(tmp_path)
    assert not (tmp_path / "state.json").exists()


def test_owned_retirement_wait_uses_actual_free_space(tmp_path, monkeypatch):
    spec = grid.PROFILES[42].cell_specs[1]
    cell = tmp_path / spec.cell_id
    cell.mkdir()
    for name in ("actor_archive_receipt.json", "moment_pruning_receipt.json"):
        grid.atomic_json(cell / name, dict(status="COMPLETE", execution_identity_sha256="e" * 64,
                                           deleted_bytes=1000 * 1024**3))
    samples = iter([67 * 1024**3, 108 * 1024**3])
    monkeypatch.setattr(grid.shutil, "disk_usage", lambda p: SimpleNamespace(free=next(samples)))
    monkeypatch.setattr(grid.time, "sleep", lambda seconds: None)
    grid.wait_archived_disk(tmp_path, spec, {"execution_identity_sha256": "e" * 64})
    receipt = grid.restart.read(cell / "disk_reclaim_receipt.json")
    assert receipt["status"] == "READY" and len(receipt["samples"]) == 2
    assert receipt["counts_deleted_bytes_as_free"] is False


def test_failure_stops_before_second_seed_and_is_not_retried(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(grid, "check_contract", lambda base: None)
    monkeypatch.setattr(grid, "refresh_status", lambda base: None)
    monkeypatch.setattr(grid.shutil, "disk_usage", lambda path: SimpleNamespace(free=108 * 1024**3))
    monkeypatch.setattr(grid.runner, "memory_coordinator_config",
                        lambda base: {"profile": "fixture"})

    def fail(base, **kwargs):
        calls.append(base.name)
        raise RuntimeError("real child failure fixture")

    monkeypatch.setattr(grid.runner, "run", fail)
    with pytest.raises(RuntimeError, match="real child failure"):
        grid.run(tmp_path)
    assert calls == ["seed42"]
    assert grid.restart.read(tmp_path / "state.json")["status"] == "FAILED"
    receipt = grid.restart.read(tmp_path / "service_completion.json")
    assert receipt["method_advantage_claimed"] is False
    with pytest.raises(RuntimeError, match="no resume or retry"):
        grid.run(tmp_path)


@pytest.mark.parametrize("profile", [
    grid.runner.PILOT, grid.recovery.PROFILES[42], grid.PROFILES[42], grid.PROFILES[43],
])
def test_aggregate_guard_dispatch_is_limited_to_new_four_round_profile(
    tmp_path, monkeypatch, profile,
):
    from src.agents import action_pooling_gpu_monitor as monitor

    calls = []

    def capture(name, command, **kwargs):
        calls.append((name, command, kwargs))
        return 17

    monkeypatch.setattr(grid.runner, "capture_process",
                        lambda command, **kwargs: capture("legacy", command, **kwargs))
    monkeypatch.setattr(monitor, "capture_process",
                        lambda command, **kwargs: capture("guarded", command, **kwargs))
    command, environment = ["test-trainer"], {"fixture": "true"}
    assert grid.runner.capture_cell_process(
        command, environment=environment, cell_dir=tmp_path, profile=profile,
    ) == 17
    expected = ("guarded" if isinstance(profile, grid.runner.ActionPoolingExperimentProfile)
                else "legacy")
    assert calls == [(expected, command, dict(environment=environment, cell_dir=tmp_path))]


@pytest.fixture
def partial_grid(tmp_path, monkeypatch):
    """Use genuine raw readout fixtures, then retain the first two completed cells."""
    import test_action_pooling_readout as evidence

    evidence.grid.__wrapped__(tmp_path)
    path = tmp_path / "seed42/state.json"
    state = grid.restart.read(path)
    state.update(status="RUNNING", cells=state["cells"][:2])
    grid.atomic_json(path, state)
    (tmp_path / "seed43/state.json").unlink()
    monkeypatch.setattr(grid, "check_contract", lambda base: None)
    return tmp_path


def test_after_cell_allows_actual_pending_work_and_syncs_archive(partial_grid, monkeypatch):
    from src.agents import parameter_grid_archive

    base = partial_grid / "seed42"
    spec = grid.PROFILES[42].cell_specs[1]
    summary = grid.restart.read(base / spec.cell_id / "summary.json")
    monkeypatch.setattr(parameter_grid_archive, "archive_completed_cell",
                        lambda *args: dict(status="COMPLETE", verified=True))
    waited = []
    monkeypatch.setattr(grid, "wait_archived_disk", lambda *args: waited.append(args[1].cell_id))
    grid.after_cell(base, spec, summary)
    assert waited == [spec.cell_id]
    assert grid.restart.read(base / "state.json")["cells"][-1] == grid.restart.read(
        base / spec.cell_id / "summary.json"
    )
    result = grid.restart.read(partial_grid / "stage2_readout.json")
    assert result["execution_status"] == "PARTIAL"
    assert len(result["cells"]) == 2


@pytest.mark.parametrize("damage", ["current_raw", "previous_raw", "head", "unknown_issue"])
def test_after_cell_readout_errors_stop_before_disk_wait(partial_grid, monkeypatch, damage):
    import json

    from src.agents import parameter_grid_archive

    base = partial_grid / "seed42"
    spec = grid.PROFILES[42].cell_specs[1]
    summary = grid.restart.read(base / spec.cell_id / "summary.json")
    if damage in {"current_raw", "previous_raw"}:
        cell = spec.cell_id if damage == "current_raw" else "sft0"
        path = base / cell / "artifacts/screening_evidence/rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["identity"]["sample_index"] = 99
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif damage == "head":
        summary["critic_initial_head_sha256"] = "f" * 64
    else:
        real_readout = grid._readout

        def unknown(*args, **kwargs):
            result = real_readout(*args, **kwargs)
            result["issues"].append(dict(reason="unexpected new validation error"))
            return result

        monkeypatch.setattr(grid, "_readout", unknown)
    monkeypatch.setattr(parameter_grid_archive, "archive_completed_cell",
                        lambda *args: dict(status="COMPLETE"))
    monkeypatch.setattr(grid, "wait_archived_disk",
                        lambda *args: pytest.fail("readout error must stop before next cell"))
    with pytest.raises(RuntimeError, match="partial readout"):
        grid.after_cell(base, spec, summary)
    result = grid.restart.read(partial_grid / "stage2_readout.json")
    if damage == "head":
        # Both completed cells remain present; coverage alone cannot detect this.
        assert len(result["cells"]) == 2
        assert any("critic head differs" in issue["reason"] for issue in result["issues"])


def test_missing_sft_head_stops_real_runner_before_first_rl(partial_grid, monkeypatch):
    base = partial_grid / "seed42"
    summary = grid.restart.read(base / "sft0/summary.json")
    summary.pop("critic_initial_head_sha256")
    grid.atomic_json(base / "sft0/summary.json", summary)
    (base / "state.json").unlink()
    calls = []

    def run_cell(_base, cell_id, **kwargs):
        calls.append(cell_id)
        return summary

    monkeypatch.setattr(grid.runner, "run_cell", run_cell)
    monkeypatch.setattr(grid.runner, "check_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid.runner, "memory_coordinator_config", lambda *args: None)
    monkeypatch.setattr(grid.runner, "monitor_status", lambda *args: None)
    monkeypatch.setattr(grid.runner, "stop_reservation", lambda *args: None)
    monkeypatch.setattr(grid.runner, "refresh_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid.shutil, "disk_usage", lambda path: SimpleNamespace(free=108 * 1024**3))
    with pytest.raises(RuntimeError, match="partial readout"):
        grid.runner.run(base, profile=grid.PROFILES[42], post_cell=grid.after_cell)
    assert calls == ["sft0"]
    assert grid.restart.read(base / "state.json")["status"] == "FAILED"
