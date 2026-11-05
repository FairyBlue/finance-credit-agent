"""Fixed paired budgets and fail-stop behavior of the two-seed grid controller."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.agents import parameter_grid as grid
from src.agents.metadata import RunHashes


def test_grid_has_paired_rates_and_eta_with_equal_training_budgets():
    assert grid.SEEDS == (42, 43)
    assert tuple(range(210, 442, 21)) == grid.EVAL_NUMBERS
    assert grid.TRAIN_NUMBERS == (1, 65, 9, 73, 17, 149, 25, 157)
    assert grid.settings()["planned_trajectories"] == 1944
    assert grid.settings()["rl_cells"] == 12
    for seed, profile in grid.PROFILES.items():
        assert profile.seed == seed
        assert profile.rounds == 4 and profile.evaluation_steps == (2, 4)
        assert profile.planned_trajectories == 972
        assert len(profile.eval_ids) == 12
        assert profile.train_ids == [f"formal-train-{number:04d}" for number in grid.TRAIN_NUMBERS]
        assert profile.retire_completed_moments and profile.archive_completed_actors
        specs = grid.runner.cells(profile)
        assert len(specs) == 7 and specs[0].zero and specs[0].cell_id == "sft0"
        for lr in (1e-7, 2e-7):
            paired = [
                spec for spec in specs
                if not spec.zero and profile.actor_lr(spec.cell_id) == lr
            ]
            assert {(spec.arm, spec.eta) for spec in paired} == {
                ("M1", 0.0), ("M2", 0.125), ("M2", 0.25),
            }
            assert all(spec.rounds == 4 and spec.critic_lr == 1e-6 for spec in paired)
    order42 = [spec.cell_id for spec in grid.runner.cells(grid.PROFILES[42])]
    order43 = [spec.cell_id for spec in grid.runner.cells(grid.PROFILES[43])]
    assert order43 == ["sft0", *reversed(order42[1:])]


@pytest.mark.parametrize("seed", grid.SEEDS)
@pytest.mark.parametrize("cell_id", [spec.cell_id for spec in grid.runner.cells(grid.PROFILES[42])])
def test_real_hydra_optimizer_rates_eta_eval_grid_and_rng(tmp_path, seed, cell_id):
    hydra = pytest.importorskip("hydra")
    profile = grid.PROFILES[seed]
    model = dict(
        source="local_export", name=str(tmp_path / "sft"), merge_receipt_sha256="a" * 64,
    )
    settings = grid.runner.settings(model, profile=profile)
    grid.atomic_json(tmp_path / "settings.json", settings)
    manifest = dict(
        hashes={"payload_sha256": "b" * 64},
        datasets={
            name: dict(path=name + ".parquet", episode_ids=ids, row_count=len(ids))
            for name, ids in (("train", profile.train_ids), ("dev", profile.eval_ids))
        },
        calibration=dict(path="reward_calibration.json", payload_sha256="c" * 64, sha256="f" * 64),
    )
    spec = next(spec for spec in grid.runner.cells(profile) if spec.cell_id == cell_id)
    cell = tmp_path / cell_id
    cell.mkdir()
    config = grid.runner.make_config(tmp_path, manifest, cell_id, profile=profile)
    command = grid.runner.build_command(
        config=config,
        run_settings=grid.runner.cell_settings(settings, cell_id, profile=profile),
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=cell,
        profile=profile,
    )
    with hydra.initialize_config_dir(
        config_dir=str(grid.runner.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    assert actual.actor_rollout_ref.actor.optim.lr == profile.actor_lr(cell_id)
    assert actual.critic.optim.lr == 1e-6
    assert actual.data.train_max_samples == 8 and actual.data.val_max_samples == 12
    assert actual.data.train_batch_size == 2
    assert actual.actor_rollout_ref.rollout.n == 4
    assert actual.actor_rollout_ref.rollout.val_kwargs.n == 5
    assert actual.actor_rollout_ref.rollout.val_kwargs.do_sample is True
    assert actual.data.apply_chat_template_kwargs.enable_thinking is False
    assert actual.trainer.total_training_steps == (1 if spec.zero else 4)
    assert actual.trainer.test_freq == (1 if spec.zero else 2)
    assert actual.trainer.val_only is spec.zero
    assert actual.trainer.resume_mode == "disable"
    assert actual.data.seed == actual.actor_rollout_ref.actor.data_loader_seed == seed
    assert actual.actor_rollout_ref.actor.fsdp_config.seed == seed
    assert actual.critic.model.fsdp_config.seed == actual.critic.data_loader_seed == seed
    assert actual.actor_rollout_ref.rollout.engine_kwargs.vllm.seed == seed
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert str(seed) == env.FINANCE_SCREENING_RUN_SEED
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == spec.eta
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert env.FINANCE_SCREENING_RESOURCE_PROFILE == "chunked_adam_disk_v1"
    assert env.FINANCE_GRID_EVALUATION_INCOMPLETE == "retain_prefix_v1"


@pytest.mark.parametrize("seed", [True, 44, -1, 42.0, "42"])
def test_unplanned_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed must be"):
        grid.profile_for_seed(seed)


def test_root_prepare_starts_both_profiles_with_disabled_holders(tmp_path, monkeypatch):
    calls = []

    def prepare(base, **kwargs):
        calls.append((base.name, kwargs))
        base.mkdir()

    monkeypatch.setattr(grid.runner, "prepare", prepare)
    base = tmp_path / "new"
    grid.prepare(base, model_path=Path("/model"), merge_receipt_sha256="a" * 64)
    assert [name for name, _ in calls] == ["seed42", "seed43"]
    assert [call["profile"].seed for _, call in calls] == [42, 43]
    assert all(call["reservation_directory"] is None for _, call in calls)
    assert json.loads((base / "grid_settings.json").read_text())["planned_trajectories"] == 1944
    with pytest.raises(RuntimeError, match="new root"):
        grid.prepare(base)


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    base = tmp_path / "root"
    base.mkdir()
    source_root = tmp_path / "repo"
    source_root.mkdir()
    (source_root / "docs").mkdir()
    (source_root / grid.PLAN).write_text("fixed plan\n")
    code = source_root / "source.py"
    code.write_text("stable source\n")
    monkeypatch.setattr(grid.runner, "REPO_ROOT", source_root)
    monkeypatch.setattr(grid.runner, "source_files", lambda: {"source.py": code})
    monkeypatch.setattr(grid.runner, "check_contract", lambda *a, **kw: None)
    grid.atomic_json(base / "grid_settings.json", grid.settings())
    for seed in grid.SEEDS:
        grid.child_directory(base, seed).mkdir()

    def seal_child(child, **kwargs):
        grid.atomic_json(child / "run_contract.json", {"profile": kwargs["profile"].name})

    monkeypatch.setattr(grid.runner, "seal", seal_child)
    grid.seal(base)
    return base, source_root


@pytest.mark.parametrize("changed", ["source", "frozen_source", "child_contract", "grid_settings"])
def test_root_contract_locks_source_child_contract_and_budget(sealed, changed):
    base, source_root = sealed
    if changed == "source":
        (source_root / "source.py").write_text("changed source\n")
    elif changed == "frozen_source":
        (base / "sources/source.py").write_text("changed frozen copy\n")
    elif changed == "child_contract":
        grid.atomic_json(base / "seed43/run_contract.json", {"profile": "different"})
    else:
        value = grid.settings()
        value["planned_trajectories"] += 1
        grid.atomic_json(base / "grid_settings.json", value)
    with pytest.raises(RuntimeError, match="changed|differ"):
        grid.check_contract(base)


def test_progress_contains_running_cell_both_seeds_and_remaining_estimate(tmp_path):
    grid.atomic_json(
        tmp_path / "state.json",
        {"status": "RUNNING", "current_seed": 42,
         "started_at_SGT": "2026-10-06T00:00:00+08:00"},
    )
    cell = tmp_path / "seed42/m1_lr1e7"
    grid.atomic_json(cell / "summary.json", {"status": "RUNNING"})
    evidence = cell / "artifacts/screening_evidence"
    evidence.mkdir(parents=True)
    (evidence / "part.jsonl").write_text('{}\n\n{}\n')
    status = grid.progress(tmp_path)
    assert status["current_seed"] == 42
    assert status["landed_trajectories"] == 2
    assert status["estimated_remaining_seconds"] > 0
    assert len(status["cells"]) == 14
    current = next(row for row in status["cells"] if row["status"] == "RUNNING")
    assert current["cell_id"] == "m1_lr1e7"
    assert current["planned_trajectories"] == 152
    grid.refresh_status(tmp_path)
    text = (tmp_path / "LIVE_STATUS.md").read_text()
    assert "当前进展（新加坡时间）" in text and "seed43" in text
    assert "已落盘轨迹 2/1944" in text


def _stub_run(monkeypatch, calls, *, fail_seed=None):
    monkeypatch.setattr(grid, "check_contract", lambda base: None)

    def run(child, *, profile, post_cell):
        calls.append(profile.seed)
        assert post_cell is not None
        if profile.seed == fail_seed:
            raise RuntimeError("worker failed; retain evidence")
        grid.atomic_json(child / "state.json", {"status": "COMPLETE"})

    monkeypatch.setattr(grid.runner, "run", run)


def test_success_runs_once_in_seed_order_and_requires_full_readout(tmp_path, monkeypatch):
    calls, readouts = [], []
    _stub_run(monkeypatch, calls)

    def readout(base, **kw):
        readouts.append(kw["allow_partial"])
        return {"execution_status": "COMPLETE", "financial_status": "COMPLETE"}

    monkeypatch.setattr(grid, "_readout", readout)
    grid.run(tmp_path)
    assert calls == [42, 43]
    assert readouts == [True, True, False]
    state = json.loads((tmp_path / "state.json").read_text())
    receipt = json.loads((tmp_path / "service_completion.json").read_text())
    assert state["status"] == receipt["status"] == "COMPLETE"
    assert state["completed_seeds"] == [42, 43]
    assert receipt["comparison_readout"] == "PASS"
    with pytest.raises(RuntimeError, match="no implicit retry"):
        grid.run(tmp_path)
    assert calls == [42, 43]


def test_evaluation_failures_complete_execution_without_claiming_financial_pass(
    tmp_path, monkeypatch,
):
    calls = []
    _stub_run(monkeypatch, calls)
    monkeypatch.setattr(grid, "_readout", lambda *a, **kw: {
        "execution_status": "COMPLETE_WITH_EVALUATION_FAILURES",
        "financial_status": "INCOMPLETE",
    })
    grid.run(tmp_path)
    assert calls == [42, 43]
    state = json.loads((tmp_path / "state.json").read_text())
    receipt = json.loads((tmp_path / "service_completion.json").read_text())
    assert state["status"] == "COMPLETE_WITH_EVALUATION_FAILURES"
    assert receipt["comparison_readout"] == "INCOMPLETE"
    assert receipt["performance_pass"] is False
    assert receipt["financial_status"] == "INCOMPLETE"
    assert (tmp_path / "FINAL_RESULTS.md").exists()


def test_worker_failure_stops_next_seed_and_preserves_original_error(tmp_path, monkeypatch):
    calls = []
    _stub_run(monkeypatch, calls, fail_seed=42)

    def readout(base, **kwargs):
        assert kwargs["allow_partial"] is True
        raise ValueError("partial audit error")

    monkeypatch.setattr(grid, "_readout", readout)
    with pytest.raises(RuntimeError, match="worker failed"):
        grid.run(tmp_path)
    assert calls == [42]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "FAILED"
    assert state["completed_seeds"] == []
    assert state["error"].startswith("RuntimeError: worker failed")
    assert state["partial_readout_error"] == "ValueError: partial audit error"
    assert (tmp_path / "FINAL_RESULTS.md").exists()
    assert json.loads((tmp_path / "service_completion.json").read_text())["status"] == "FAILED"


def test_readout_failure_never_reports_complete_training_as_full_pass(tmp_path, monkeypatch):
    calls = []
    _stub_run(monkeypatch, calls)

    def readout(base, *, allow_partial):
        if not allow_partial:
            raise ValueError("incomplete evaluation grid")

    monkeypatch.setattr(grid, "_readout", readout)
    with pytest.raises(ValueError, match="incomplete evaluation"):
        grid.run(tmp_path)
    assert calls == [42, 43]
    receipt = json.loads((tmp_path / "service_completion.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["comparison_readout"] == "PARTIAL"


def test_post_cell_is_forwarded_without_replacing_the_fixed_profiles(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(grid, "check_contract", lambda base: None)
    monkeypatch.setattr(grid, "_readout", lambda *a, **kw: {
        "execution_status": "COMPLETE", "financial_status": "COMPLETE",
    })

    def hook(base, spec, summary):
        pass

    def run(child, *, profile, post_cell):
        assert post_cell is hook
        seen.append(copy.deepcopy(profile))
        grid.atomic_json(child / "state.json", {"status": "COMPLETE"})

    monkeypatch.setattr(grid.runner, "run", run)
    grid.run(tmp_path, post_cell=hook)
    assert seen == [grid.PROFILES[42], grid.PROFILES[43]]


def test_successful_actor_archive_receipt_is_saved_before_partial_readout(tmp_path, monkeypatch):
    from src.agents import parameter_grid_archive

    seed_base = tmp_path / "seed42"
    spec = next(spec for spec in grid.runner.cells(grid.PROFILES[42]) if not spec.zero)
    summary = {"status": "COMPLETE"}
    events = []

    def archive(base, actual_spec, actual_summary):
        assert base == seed_base and actual_spec is spec and actual_summary is summary
        events.append("archive")
        return {"status": "COMPLETE", "receipt": "verified receipt"}

    def readout(base, *, allow_partial):
        assert base == tmp_path and allow_partial is True
        written = json.loads((seed_base / spec.cell_id / "summary.json").read_text())
        assert written["actor_archive"]["status"] == "COMPLETE"
        events.append("readout")

    monkeypatch.setattr(parameter_grid_archive, "archive_completed_cell", archive)
    monkeypatch.setattr(grid, "_readout", readout)
    grid.after_cell(seed_base, spec, summary)
    assert events == ["archive", "readout"]


def test_sft_baseline_does_not_archive_a_model(tmp_path, monkeypatch):
    from src.agents import parameter_grid_archive

    monkeypatch.setattr(
        parameter_grid_archive, "archive_completed_cell",
        lambda *a: pytest.fail("SFT does not export an updated actor"),
    )
    readouts = []
    monkeypatch.setattr(grid, "_readout", lambda base, **kw: readouts.append(base))
    grid.after_cell(tmp_path / "seed42", grid.runner.cells(grid.PROFILES[42])[0], {})
    assert readouts == [tmp_path]
