"""Real command budgets, per-policy evaluation, and owned scratch retirement."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")

from src.agents import actor_export  # noqa: E402
from src.agents import low_memory_comparison as comparison  # noqa: E402
from src.agents import low_memory_comparison_seed43 as replication  # noqa: E402
from src.agents import low_memory_pilot as entry  # noqa: E402
from src.agents.metadata import RunHashes  # noqa: E402

PROFILE = comparison.PROFILE


@pytest.fixture
def prepared(tmp_path):
    model = dict(source="local_export", name=str(tmp_path / "shared-sft"),
                 merge_receipt_sha256="a" * 64)
    settings = entry.settings(model, profile=PROFILE)
    manifest = dict(
        hashes={"payload_sha256": "b" * 64},
        datasets={name: dict(path=name + ".parquet", episode_ids=ids, row_count=len(ids))
                  for name, ids in (("train", PROFILE.train_ids), ("dev", entry.EVAL_IDS))},
        calibration=dict(path="reward_calibration.json", payload_sha256="c" * 64,
                         sha256="f" * 64),
    )
    entry.atomic_json(tmp_path / "settings.json", settings)
    return tmp_path, settings, manifest


@pytest.mark.parametrize("cell_id", ["sft0", "m1", "m2_eta025"])
def test_real_hydra_has_fixed_zero_two_four_eval_and_four_updates(prepared, cell_id):
    base, settings, manifest = prepared
    cell = base / cell_id
    cell.mkdir()
    cfg = entry.make_config(base, manifest, cell_id, profile=PROFILE)
    command = entry.build_command(
        config=cfg, run_settings=entry.cell_settings(settings, cell_id, profile=PROFILE),
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=cell,
        profile=PROFILE,
    )
    with hydra.initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"), version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    zero = cell_id == "sft0"
    assert actual.trainer.val_only is zero and actual.trainer.val_before_train is zero
    assert actual.trainer.total_training_steps == (1 if zero else 4)
    assert actual.trainer.test_freq == (1 if zero else 2)
    assert actual.trainer.save_freq == (1 if zero else 4)
    assert actual.data.train_max_samples == 8 and actual.data.train_batch_size == 2
    assert actual.actor_rollout_ref.rollout.n == 4
    assert actual.actor_rollout_ref.rollout.val_kwargs.n == 5
    assert actual.actor_rollout_ref.rollout.val_kwargs.do_sample is True
    assert actual.trainer.resume_mode == "disable" and actual.trainer.resume_from_path is None
    assert not actual.data.shuffle and not actual.data.validation_shuffle
    assert actual.data.apply_chat_template_kwargs.enable_thinking is False
    assert actual.actor_rollout_ref.actor.optim.lr == 2e-7
    assert actual.critic.optim.lr == 1e-6
    model = settings["models"]["screening"]["name"]
    assert actual.actor_rollout_ref.model.path == actual.actor_rollout_ref.ref.model.path == model
    assert actual.critic.model.path == actual.critic.model.tokenizer_path == model
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.FINANCE_SCREENING_RESOURCE_PROFILE == "chunked_adam_disk_v1"
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == (0.25 if cell_id == "m2_eta025" else 0)
    assert list(actual.actor_rollout_ref.actor.checkpoint.save_contents) == [
        "finance_actor_bf16_hf",
    ]
    assert list(actual.critic.checkpoint.save_contents) == []


def test_profile_has_164_trajectories_and_preserves_pilot_defaults(prepared):
    _, settings, _ = prepared
    assert PROFILE.train_ids == entry.common.TRAIN_IDS == entry.coverage.TRAIN_IDS[:8]
    assert PROFILE.evaluation_steps == (2, 4) and PROFILE.rounds == 4
    assert settings["training_coverage"]["planned_trajectories"] == 20 + 2 * (32 + 40) == 164
    assert [(s.cell_id, s.arm, s.rounds) for s in entry.cells(PROFILE)] == [
        ("sft0", "M1", 0), ("m1", "M1", 4), ("m2_eta025", "M2", 4),
    ]
    assert PROFILE.retire_completed_moments and not entry.PILOT.retire_completed_moments
    assert entry.PILOT.planned_trajectories == 72
    assert entry.PILOT.evaluation_steps == (2,) and entry.PILOT.rounds == 2
    assert entry.cells() == entry.cells(entry.PILOT)


@pytest.mark.parametrize("cell_id", ["sft0", "m1", "m2_eta025"])
def test_seed43_reaches_actual_rng_paths_and_preserves_comparison_budget(prepared, cell_id):
    base, original, manifest = prepared
    profile = replication.PROFILE
    assert profile.seed == 43 and PROFILE.seed == entry.PILOT.seed == 42
    assert profile.planned_trajectories == 164
    assert profile.train_ids == PROFILE.train_ids
    assert profile.evaluation_steps == PROFILE.evaluation_steps
    changed = entry.settings(original["models"]["screening"], profile=profile)
    same = copy.deepcopy(changed)
    same["seeds"] = original["seeds"]
    same["plan"] = original["plan"]
    same["training_coverage"]["profile"] = original["training_coverage"]["profile"]
    assert same == original
    entry.atomic_json(base / "settings.json", changed)
    (base / cell_id).mkdir()
    config = entry.make_config(base, manifest, cell_id, profile=profile)
    assert config["run"]["seed"] == 43 and config["run"]["name"].endswith("-s43")
    condition = entry.cell_settings(changed, cell_id, profile=profile)
    command = entry.build_command(
        config=config, run_settings=condition,
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / cell_id,
        profile=profile,
    )
    with hydra.initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"), version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    assert actual.data.seed == actual.actor_rollout_ref.actor.data_loader_seed == 43
    assert actual.actor_rollout_ref.actor.fsdp_config.seed == 43
    assert actual.critic.model.fsdp_config.seed == actual.critic.data_loader_seed == 43
    assert actual.actor_rollout_ref.rollout.engine_kwargs.vllm.seed == 43
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.FINANCE_SCREENING_RUN_SEED == "43"
    assert env.FINANCE_SCREENING_CRITIC_INIT == "cpu_seed_v1"
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert actual.trainer.total_training_steps == (1 if cell_id == "sft0" else 4)
    assert actual.trainer.val_only is (cell_id == "sft0")
    assert actual.trainer.test_freq == (1 if cell_id == "sft0" else 2)
    config["run"]["seed"] = 42
    with pytest.raises(RuntimeError, match="resolved arm"):
        entry.build_command(
            config=config, run_settings=condition,
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / cell_id,
            profile=profile,
        )
    with pytest.raises(RuntimeError, match="settings differ"):
        entry.read_settings(base, profile=PROFILE)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**32])
def test_invalid_profile_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="experiment seed"):
        replace(PROFILE, seed=seed)


def test_seed43_is_also_forwarded_to_post_training_reward_validation(tmp_path, monkeypatch):
    from src.agents import actor_update_probe, paired_validation

    monkeypatch.setattr(actor_update_probe, "validate_run_evidence", lambda *a, **kw: {})
    monkeypatch.setattr(paired_validation, "validate_run_evidence", lambda *a, **kw: {})
    seen = []

    def penalty(cell, spec, *, expected_seed=42):
        seen.append(expected_seed)
        return {"status": "PASSED"}

    monkeypatch.setattr(entry.common, "validate_common_penalty_evidence", penalty)
    entry.validate_extra_evidence(
        tmp_path, entry.cells(replication.PROFILE)[1], "a" * 64,
        profile=replication.PROFILE,
    )
    assert seen == [43]


def test_date_selection_uses_preserved_full_sequence_then_fixed_prefix(monkeypatch):
    full = [dict(episode_id=episode_id, episode_dates=[f"date-{i}"])
            for i, episode_id in enumerate(entry.coverage.TRAIN_IDS)]
    source = dict(windows=[{"index": i} for i in range(len(full))], preserved="fixture")
    monkeypatch.setattr(entry.coverage, "select_episodes",
                        lambda formal, old: (copy.deepcopy(full), copy.deepcopy(source)))
    selected, selection = entry.select_episodes({}, {}, profile=PROFILE)
    assert selected == full[:8] and selection["windows"] == source["windows"][:8]
    assert selection["unique_training_sessions"] == 8
    assert selection["training_episode_ids"] == PROFILE.train_ids
    assert source["windows"] == [{"index": i} for i in range(len(full))]


def test_old_profile_rejects_new_settings_and_cross_profile_command(prepared):
    base, settings, manifest = prepared
    with pytest.raises(RuntimeError, match="settings differ"):
        entry.read_settings(base)
    cfg = entry.make_config(base, manifest, "m1", profile=PROFILE)
    with pytest.raises(RuntimeError, match="condition differs"):
        entry.build_command(
            config=cfg, run_settings=entry.cell_settings(settings, "m1", profile=PROFILE),
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=base / "m1",
        )


def trajectories(steps):
    rows = []
    for step in steps:
        for episode_id in entry.EVAL_IDS:
            for sibling in range(5):
                rows.append(dict(
                    identity=dict(validate=True, step=step, rollout_n=sibling),
                    episode_id=episode_id,
                    metrics=dict(raw_net_return=step / 100, transaction_cost=step,
                                 terminal_score=-step, max_drawdown=-0.01,
                                 agent_volatility=0.02, benchmark_excess=step / 200,
                                 turnover=1.1, maximum_position_weight=0.3),
                    steps=[dict(invalid_action=sibling == 1)],
                ))
    return rows


def test_evaluation_readout_keeps_policy_two_and_four_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "read_jsonl", lambda path: trajectories([2, 4]))
    two = entry.evaluation_readout(tmp_path, policy_step=2, profile=PROFILE)
    four = entry.evaluation_readout(tmp_path, policy_step=4, profile=PROFILE)
    assert two["greedy"]["trajectories"] == four["greedy"]["trajectories"] == 4
    assert two["sampled"]["trajectories"] == four["sampled"]["trajectories"] == 16
    assert two["sampled"]["mean_raw_net_return"] == 0.02
    assert four["sampled"]["mean_raw_net_return"] == 0.04
    assert two["sampled"]["invalid_decisions"] == 4
    assert four["greedy"]["mean_maximum_position_weight"] == 0.3
    with pytest.raises(RuntimeError, match="lane size"):
        entry.evaluation_readout(tmp_path, profile=PROFILE)


def test_sft_zero_readout_requires_the_zero_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "read_jsonl", lambda path: trajectories([0]))
    result = entry.evaluation_readout(tmp_path, policy_step=0, profile=PROFILE)
    assert result["greedy"]["mean_raw_net_return"] == 0
    with pytest.raises(RuntimeError, match="lane size"):
        entry.evaluation_readout(tmp_path, policy_step=1, profile=PROFILE)


def placement_rows(identity, *, updates=0):
    rows = [dict(profile="chunked_adam_disk_v1", identity_sha256=identity,
                 phase="rollout_mode", gpu_peak_allocated_bytes=1024,
                 gpu_peak_reserved_bytes=2048)]
    if updates:
        rows.append(dict(profile="chunked_adam_disk_v1", identity_sha256=identity,
                         event="exchange", detail=dict(event="optimizer_complete",
                                                       role="actor", role_step=1)))
    return rows


@pytest.mark.parametrize("unexpected", ["updates", "export", "save_phase"])
def test_sft_placement_rejects_actual_updates_or_exports(tmp_path, monkeypatch, unexpected):
    spec = entry.cells(PROFILE)[0]
    rows = placement_rows("a" * 64, updates=unexpected == "updates")
    if unexpected == "export":
        (tmp_path / "checkpoints/global_step_1").mkdir(parents=True)
    if unexpected == "save_phase":
        rows[0]["phase"] = "save_checkpoint:enter"
    monkeypatch.setattr(entry, "read_jsonl", lambda path: rows)
    with pytest.raises(RuntimeError):
        entry.validate_placement(tmp_path, spec, "a" * 64)


def test_sft_placement_requires_identity_and_proves_zero_exchange(tmp_path, monkeypatch):
    rows = placement_rows("a" * 64)
    monkeypatch.setattr(entry, "read_jsonl", lambda path: rows)
    result = entry.validate_placement(tmp_path, entry.cells(PROFILE)[0], "a" * 64)
    assert result["updates"] == {"critic": [], "actor": []} and result["exchanges"] == []
    rows[0]["identity_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="identity"):
        entry.validate_placement(tmp_path, entry.cells(PROFILE)[0], "a" * 64)


@pytest.fixture
def completed_cell(tmp_path, monkeypatch):
    spec = entry.cells(PROFILE)[1]
    cell = tmp_path / spec.cell_id
    root = cell / "artifacts/placement/moment_scratch/999999999"
    summary = dict(status="COMPLETE", returncode=0, actual_rounds=4, cell=asdict(spec),
                   execution_identity_sha256="a" * 64)
    entry.atomic_json(cell / "summary.json", summary)
    for role in ("actor", "critic"):
        directory = root / role
        generation = directory / "generation-000004"
        generation.mkdir(parents=True)
        payload = generation / "p000000-exp_avg-000000.raw"
        payload.write_bytes(b"test-moment-payload")
        chunk = dict(file=payload.name, bytes=payload.stat().st_size,
                     sha256=entry.restart.digest(payload))
        owner = dict(owner="fixture-" + role, identity="a" * 64, role=role)
        entry.atomic_json(directory / "owner.json", owner)
        entry.atomic_json(directory / "manifest.json", dict(
            schema_version=1, **owner, generation=generation.name,
            payload_bytes=chunk["bytes"], parameters=[dict(moments={"exp_avg": [chunk]})],
        ))
    checks = []
    monkeypatch.setattr(entry, "check_contract", lambda *a, **kw: checks.append("contract"))
    monkeypatch.setattr(actor_export, "validate_export_artifact",
                        lambda *a, **kw: checks.append("export"))
    return tmp_path, spec, summary, root, checks


def test_retirement_records_hashes_first_keeps_manifests_and_old_runs(completed_cell, monkeypatch):
    base, spec, summary, root, checks = completed_cell
    old = base / "historical/moment.raw"
    old.parent.mkdir()
    old.write_bytes(b"historical")
    original = Path.unlink
    seen = []

    def unlink(path, *args, **kwargs):
        receipt = entry.restart.read(base / spec.cell_id / "moment_pruning_receipt.json")
        assert receipt["status"] == "PREPARED" and len(receipt["files"]) == 2
        assert all(len(row["sha256"]) == 64 for row in receipt["files"])
        seen.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    result = entry.retire_completed_moments(base, spec, summary, profile=PROFILE)
    assert checks == ["contract", "export"]
    assert result["status"] == "COMPLETE" and result["deleted_files"] == len(seen) == 2
    assert not list(root.rglob("*.raw"))
    assert len(list(root.rglob("manifest.json"))) == 2
    assert len(list(root.rglob("owner.json"))) == 2 and old.read_bytes() == b"historical"
    assert entry.restart.read(Path(result["receipt"]))["status"] == "COMPLETE"


@pytest.mark.parametrize("mutation", ["foreign", "hash", "identity", "symlink", "failed"])
def test_retirement_refuses_unowned_or_failed_scratch_without_deleting(completed_cell, mutation):
    base, spec, summary, root, _ = completed_cell
    raw = next(root.rglob("*.raw"))
    if mutation == "foreign":
        (raw.parent / "foreign.raw").write_bytes(b"keep")
    elif mutation == "hash":
        raw.write_bytes(b"same-length-changed!")
    elif mutation == "identity":
        path = raw.parent.parent / "manifest.json"
        manifest = entry.restart.read(path)
        manifest["identity"] = "b" * 64
        entry.atomic_json(path, manifest)
    elif mutation == "symlink":
        target = base / "must-preserve.raw"
        target.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(target)
    elif mutation == "failed":
        summary["status"] = "FAILED"
        entry.atomic_json(base / spec.cell_id / "summary.json", summary)
    with pytest.raises(RuntimeError):
        entry.retire_completed_moments(base, spec, summary, profile=PROFILE)
    assert raw.exists() and len(list(root.rglob("manifest.json"))) == 2
    assert not (base / spec.cell_id / "moment_pruning_receipt.json").exists()


def test_old_pilot_and_sft0_never_retire_any_moments(completed_cell):
    base, spec, summary, root, checks = completed_cell
    assert entry.retire_completed_moments(base, spec, summary)["status"] == "RETAINED"
    assert entry.retire_completed_moments(
        base, entry.cells(PROFILE)[0], summary, profile=PROFILE,
    )["status"] == "RETAINED"
    assert len(list(root.rglob("*.raw"))) == 2 and checks == []


def test_comparison_live_status_shows_three_cells_and_fixed_164_budget(tmp_path):
    entry.atomic_json(tmp_path / "state.json", dict(status="RUNNING", current_cell="sft0"))
    entry.refresh_status(tmp_path, profile=PROFILE)
    text = (tmp_path / "LIVE_STATUS.md").read_text()
    assert "新加坡时间" in text and "164" in text
    assert "sft0" in text and "0/20" in text and text.count("0/72") == 2
    assert not (tmp_path / "FINAL_RESULTS.md").exists()


def test_four_round_placement_requires_four_updates_and_eight_exchanges(tmp_path, monkeypatch):
    identity = "a" * 64
    rows = placement_rows(identity)
    for step in range(1, 5):
        for index, role in enumerate(("critic", "actor")):
            inactive = "actor" if role == "critic" else "critic"
            events = [
                dict(event="exchange_start", time_ns=(step * 2 + index) * 10**9),
                dict(event="exchange_complete", time_ns=(step * 2 + index) * 10**9 + 1,
                     active_role=role, moment_cpu_bytes_by_role={role: 32, inactive: 0},
                     disk_status={role: "resident", inactive: "offloaded"}),
                dict(event="optimizer_chunked", role=role,
                     detail=dict(event="chunked_adamw_step_complete", chunks=1,
                                 maximum_temporary_bytes=64, moment_transfer_bytes=64)),
                dict(event="optimizer_complete", role=role, role_step=step),
            ]
            rows.extend(dict(profile="chunked_adam_disk_v1", identity_sha256=identity,
                             event="exchange", detail=event) for event in events)
    monkeypatch.setattr(entry, "read_jsonl", lambda path: rows)
    result = entry.validate_placement(tmp_path, entry.cells(PROFILE)[1], identity)
    assert len(result["exchanges"]) == 8
    assert len(result["updates"]["actor"]) == len(result["updates"]["critic"]) == 4
    rows.pop()
    with pytest.raises(RuntimeError, match="fixed real CUDA updates"):
        entry.validate_placement(tmp_path, entry.cells(PROFILE)[1], identity)


@pytest.mark.parametrize("fail_m1", [False, True])
def test_controller_retires_only_completed_training_cells_and_never_retries(
    tmp_path, monkeypatch, fail_m1,
):
    calls, retired = [], []
    monkeypatch.setattr(entry, "check_contract", lambda *a, **kw: None)
    monkeypatch.setattr(entry, "monitor_status", lambda *a: None)
    monkeypatch.setattr(entry, "stop_reservation", lambda *a: None)
    monkeypatch.setattr(entry.shutil, "disk_usage", lambda path: type("Disk", (), {
        "free": 200 * 1024**3,
    })())

    def run_cell(base, cell_id, *, profile):
        assert profile is PROFILE
        calls.append(cell_id)
        if fail_m1 and cell_id == "m1":
            raise RuntimeError("evidence failed; preserve all moments")
        return dict(status="COMPLETE", cell=cell_id)

    def retire(base, spec, summary, *, profile):
        assert profile is PROFILE and summary["status"] == "COMPLETE"
        retired.append(spec.cell_id)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(entry, "run_cell", run_cell)
    monkeypatch.setattr(entry, "retire_completed_moments", retire)
    if fail_m1:
        with pytest.raises(RuntimeError, match="evidence failed"):
            entry.run(tmp_path, profile=PROFILE)
        assert calls == ["sft0", "m1"] and retired == []
    else:
        entry.run(tmp_path, profile=PROFILE)
        assert calls == ["sft0", "m1", "m2_eta025"]
        assert retired == ["m1", "m2_eta025"]
    state = entry.restart.read(tmp_path / "state.json")
    assert state["profile"] == PROFILE.name and state["planned_trajectories"] == 164
    assert state["status"] == ("FAILED" if fail_m1 else "COMPLETE")
    with pytest.raises(RuntimeError, match="no implicit retry"):
        entry.run(tmp_path, profile=PROFILE)
    assert calls == (["sft0", "m1"] if fail_m1 else ["sft0", "m1", "m2_eta025"])


@pytest.mark.parametrize("exits", [False, True])
def test_owned_worker_teardown_wait_is_bounded_without_killing(monkeypatch, exits):
    class Clock:
        value = 0

        def monotonic(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds

    clock = Clock()
    monkeypatch.setattr(entry, "time", clock)
    monkeypatch.setattr(Path, "exists", lambda path: not exits or clock.value < 2)
    if exits:
        entry.wait_moment_worker_exit(123, timeout=3)
        assert clock.value == 2
    else:
        with pytest.raises(RuntimeError, match="has not exited"):
            entry.wait_moment_worker_exit(123, timeout=3)
        assert clock.value == 3
