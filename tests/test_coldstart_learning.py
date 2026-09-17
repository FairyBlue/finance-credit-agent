"""CPU contracts for fixed-budget common-SFT research; no trainer/GPU is started."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from src.agents import coldstart_learning as entry
from src.agents import screening_launcher
from src.agents.metadata import RunHashes


@pytest.fixture
def prepared(tmp_path):
    model = {
        "source": "local_export",
        "name": str(tmp_path / "merged_model"),
        "merge_receipt_sha256": "a" * 64,
    }
    settings = entry.settings(model)
    manifest = {
        "hashes": {"payload_sha256": "b" * 64},
        "datasets": {k: {"path": f"{k}.parquet"} for k in ("train", "dev")},
        "calibration": {
            "path": "reward_calibration.json",
            "sha256": "c" * 64,
            "payload_sha256": "d" * 64,
        },
    }
    entry.atomic_json(tmp_path / "settings.json", settings)
    entry.atomic_json(tmp_path / "prepared/manifest.json", manifest)
    entry.atomic_json(
        tmp_path / "model_identity.json",
        {
            "model_revision": "local-export-sha256:" + "a" * 64,
            "model_snapshot_path": model["name"],
        },
    )
    return tmp_path, settings, manifest


def command(base, settings, manifest, arm, zero=False):
    cfg = entry.make_config(base, manifest, arm, zero=zero)
    result = entry.build_command(
        config=cfg,
        run_settings=settings,
        hashes=RunHashes("e" * 40, "f" * 40, "fixture", True),
        cell_dir=base,
    )
    return cfg, result


@pytest.mark.parametrize("arm,zero", [("M1", True), ("M1", False), ("M2", False)])
def test_real_hydra_has_same_model_roles_fixed_rounds_and_no_gate(prepared, arm, zero):
    base, settings, manifest = prepared
    cfg, argv = command(base, settings, manifest, arm, zero)
    with initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        resolved = compose(config_name="agent_ppo_trainer", overrides=argv[3:])
    model = settings["models"]["screening"]["name"]
    assert resolved.actor_rollout_ref.model.path == model
    assert resolved.actor_rollout_ref.ref.model.path == model
    assert resolved.critic.model.path == resolved.critic.model.tokenizer_path == model
    assert resolved.critic.model.fsdp_config.seed == 42
    assert resolved.actor_rollout_ref.actor.fsdp_config.seed == 42
    assert resolved.data.screening_observation_profile == "history_features_v1"
    assert resolved.data.screening_action_protocol == entry.PROTOCOL
    assert resolved.data.apply_chat_template_kwargs.enable_thinking is False
    assert resolved.trainer.val_only == zero
    assert resolved.trainer.total_training_steps == (1 if zero else 2)
    assert resolved.trainer.test_freq == resolved.trainer.save_freq == (1 if zero else 2)
    assert resolved.trainer.max_actor_ckpt_to_keep == resolved.trainer.max_critic_ckpt_to_keep == 1
    assert resolved.trainer.resume_mode == "disable" and resolved.trainer.resume_from_path is None
    assert resolved.trainer.finance_artifact_profile == entry.RESEARCH_PROFILE
    for checkpoint in (resolved.actor_rollout_ref.actor.checkpoint, resolved.critic.checkpoint):
        assert (
            list(checkpoint.save_contents) == list(checkpoint.load_contents) == ["model", "extra"]
        )
        assert checkpoint.async_save is False
    assert resolved.data.train_max_samples == 8
    assert resolved.data.train_batch_size == 2
    assert resolved.actor_rollout_ref.rollout.n == 4
    assert resolved.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert resolved.actor_rollout_ref.rollout.val_kwargs.do_sample is False
    assert resolved.actor_rollout_ref.rollout.max_model_len == 12288
    assert resolved.actor_rollout_ref.rollout.gpu_memory_utilization == 0.2
    env = resolved.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.FINANCE_SCREENING_CRITIC_INIT == "cpu_seed_v1"
    assert (
        str(base / "artifacts/critic_initialization")
        == env.FINANCE_SCREENING_CRITIC_INIT_EVIDENCE_DIR
    )
    assert all(isinstance(v, str) for v in env.values())
    assert not any(key.startswith("FINANCE_LEARNING_") for key in env)
    assert cfg["screening"]["artifact_profile"] == entry.RESEARCH_PROFILE
    from src.agents.screening_checkpoint import _configuration

    assert _configuration(SimpleNamespace(config=resolved, use_critic=True)) == 1


@pytest.mark.parametrize("damage", ["rounds", "critic", "artifact", "resolved_artifact"])
def test_partial_or_extended_contract_rejected_before_trainer(prepared, damage):
    base, settings, manifest = prepared
    cfg = entry.make_config(base, manifest, "M2")
    if damage == "rounds":
        cfg["training"]["total_training_steps"] = 4
    elif damage == "critic":
        cfg["training"]["critic_initialization"] = "legacy"
    elif damage == "artifact":
        settings["artifact_profile"] = "full"
    else:
        cfg["screening"]["artifact_profile"] = "full"
    with pytest.raises(ValueError):
        entry.build_command(
            config=cfg,
            run_settings=settings,
            hashes=RunHashes("e" * 40, "f" * 40, "fixture", True),
            cell_dir=base,
        )


def test_research_save_cannot_gain_resume_from_generic_launcher(prepared):
    base, settings, manifest = prepared
    cfg = entry.make_config(base, manifest, "M2")
    with pytest.raises(ValueError, match="cannot resume"):
        screening_launcher.build_cell_command(
            config=cfg,
            settings=settings,
            hashes=RunHashes("e" * 40, "f" * 40, "fixture", True),
            cell_dir=base,
            resume_path=base / "global_step_2",
        )


def test_research_profile_changes_identity_and_old_full_remains_default(prepared):
    base, settings, manifest = prepared
    _, candidate = command(base, settings, manifest, "M1")
    full_settings = copy.deepcopy(settings)
    full_settings.pop("artifact_profile")
    cfg = screening_launcher.build_cell_config(
        settings=full_settings,
        manifest=manifest,
        prepared_dir=base,
        arm="M1",
        stage="screening",
        seed=42,
        cell_dir=base,
    )
    full = screening_launcher.build_cell_command(
        config=cfg,
        settings=full_settings,
        hashes=RunHashes("e" * 40, "f" * 40, "fixture", True),
        cell_dir=base,
    )
    key = "+ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256="
    assert next(x for x in candidate if x.startswith(key)) != next(
        x for x in full if x.startswith(key)
    )
    assert not any("finance_artifact_profile" in argument for argument in full)
    assert "artifact_profile" not in cfg["screening"]


def test_saved_settings_cannot_change_budget_or_protocol(prepared):
    base, settings, _ = prepared
    settings["action_protocol"] = "single_tool_v1"
    entry.atomic_json(base / "settings.json", settings)
    with pytest.raises(RuntimeError, match="fixed shared"):
        entry.read_settings(base)


def test_cli_canonicalizes_directory_before_building_ray_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coldstart_learning",
            "prepare",
            "--directory",
            "relative-run",
            "--model",
            str(tmp_path / "export"),
            "--merge-receipt-sha256",
            "a" * 64,
        ],
    )
    observed = []
    monkeypatch.setattr(entry, "prepare", lambda base, **kwargs: observed.append(base))
    entry.main()
    assert observed == [tmp_path / "relative-run"]


def test_sealed_settings_mutation_detected_without_model_loading(prepared, monkeypatch):
    base, _, _ = prepared
    entry.atomic_json(base / "input_preflight.json", {"status": "PASSED"})
    monkeypatch.setattr(entry.restart, "seal", lambda base: None)
    monkeypatch.setattr(entry.restart, "check_sources", lambda base: None)
    entry.seal(base)
    entry.atomic_json(base / "input_preflight.json", {"status": "changed"})
    with pytest.raises(RuntimeError, match="input changed"):
        entry.check_contract(base)


def test_sealed_manifest_rejects_coordinated_parquet_and_manifest_change(prepared, monkeypatch):
    import pandas as pd

    base, settings, manifest = prepared
    for name in ("train", "dev"):
        pd.DataFrame({"episode_id": [entry.TRAIN_IDS[0]], "marker": ["original"]}).to_parquet(
            base / "prepared" / f"{name}.parquet", index=False,
        )
        manifest["datasets"][name]["sha256"] = entry.restart.digest(
            base / "prepared" / f"{name}.parquet"
        )
    entry.atomic_json(base / "prepared/reward_calibration.json", {"fixture": "fixed"})
    manifest["calibration"]["sha256"] = entry.restart.digest(
        base / "prepared/reward_calibration.json"
    )
    entry.atomic_json(base / "prepared/manifest.json", manifest)
    identity = entry.restart.read(base / "model_identity.json")
    entry.atomic_json(base / "input_preflight.json", {
        "status": "PASSED", "model_revision": identity["model_revision"],
        "action_protocol": entry.PROTOCOL, "enable_thinking": False,
    })
    model = settings["models"]["screening"]["name"]
    monkeypatch.setattr(
        entry, "_resolve_exact_model_snapshot",
        lambda config: (model, identity["model_revision"], entry.Path(model)),
    )
    entry.seal(base)
    entry.check_contract(base)

    # An accidental regeneration can replace both real parquet content and its
    # current manifest checksum. The old current-manifest check accepts this.
    pd.DataFrame({"episode_id": [entry.TRAIN_IDS[1]], "marker": ["regenerated"]}).to_parquet(
        base / "prepared/train.parquet", index=False,
    )
    manifest["datasets"]["train"]["sha256"] = entry.restart.digest(
        base / "prepared/train.parquet"
    )
    entry.atomic_json(base / "prepared/manifest.json", manifest)
    entry.restart.check_sources(base)

    monkeypatch.setattr(
        entry, "_resolve_exact_model_snapshot",
        lambda config: pytest.fail("changed prepared input must fail before loading a model"),
    )
    with pytest.raises(RuntimeError, match="input changed after sealing: prepared/manifest.json"):
        entry.check_contract(base)


@pytest.mark.parametrize("fail_arm", [None, "M1"])
def test_schedule_is_baseline_then_two_fixed_arms_without_retry(prepared, monkeypatch, fail_arm):
    base, _, _ = prepared
    calls = []
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(
        entry.shutil, "disk_usage", lambda base: SimpleNamespace(free=128 * 1024**3)
    )

    def run_cell(base, arm, *, zero):
        calls.append((arm, zero))
        if not zero and arm == fail_arm:
            raise RuntimeError("fixture trainer failure")
        return {"status": "COMPLETE", "actual_rounds": 0 if zero else 2}

    monkeypatch.setattr(entry, "run_cell", run_cell)
    if fail_arm:
        with pytest.raises(RuntimeError, match="fixture trainer"):
            entry.run(base)
    else:
        entry.run(base)
    assert calls == (
        [("M1", True), ("M1", False)]
        if fail_arm
        else [
            ("M1", True),
            ("M1", False),
            ("M2", False),
        ]
    )
    state = entry.restart.read(base / "state.json")
    assert state["adaptive_gate"] is False and state["automatic_retry"] is False
    assert state["effective_train_episode_ids"] == entry.TRAIN_IDS[:4]
    with pytest.raises(RuntimeError, match="no implicit retry"):
        entry.run(base)


@pytest.mark.parametrize("zero", [False, True])
def test_run_cell_keeps_compiler_cleans_gate_and_validates_exact_budget(
    prepared, monkeypatch, zero
):
    base, settings, _ = prepared
    expected_model = settings["models"]["screening"]["name"]
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(
        entry,
        "collect_run_hashes",
        lambda **kwargs: RunHashes(
            "e" * 40,
            "f" * 40,
            "fixture",
            True,
        ),
    )
    monkeypatch.setattr(
        entry,
        "verify_gpu_runtime_and_model",
        lambda cfg: {
            "model_revision": "local-export-sha256:" + "a" * 64,
            "model_snapshot_path": expected_model,
            "c_compiler": "/existing/toolchain/gcc",
        },
    )
    monkeypatch.setenv("FINANCE_LEARNING_GATE_DIR", "/old/gate")
    monkeypatch.setenv("FINANCE_LEARNING_RESTART_GATE", "stop2_if_no_signal_v1")
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "old")
    calls = []

    def capture(argv, *, environment, cell_dir):
        assert environment["CC"] == "/existing/toolchain/gcc"
        assert not any(
            k.startswith(("FINANCE_LEARNING_", "FINANCE_SCREENING_")) for k in environment
        )
        assert not any("FINANCE_LEARNING_" in arg for arg in argv)
        assert f"actor_rollout_ref.model.path={expected_model}" in argv
        assert f"critic.model.path={expected_model}" in argv
        assert environment["HF_HUB_OFFLINE"] == "1"
        options = dict(arg.lstrip("+").split("=", 1) for arg in argv[3:])
        identity = options[
            "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"
        ]
        entry.atomic_json(
            cell_dir / "artifacts/critic_initialization/head-rank0.json",
            {
                "phase": "critic_head_after_hf_load_before_fsdp",
                "seed": 42,
                "mode": "cpu_seed_v1",
                "arm": "M1" if zero else "M2",
                "execution_identity_sha256": identity,
                "model_path": expected_model,
                "head_sha256": "1" * 64,
            },
        )
        if not zero:
            entry.atomic_json(base / "base/summary.json", {"critic_initial_head_sha256": "1" * 64})
        return 0

    def validate(cell, **kwargs):
        calls.append(kwargs)
        assert kwargs["artifact_profile"] == entry.RESEARCH_PROFILE
        assert kwargs["expected_rounds"] == (1 if zero else 2)
        assert kwargs["expected_eval_count"] == 4
        assert kwargs["train_episode_ids"] == entry.TRAIN_IDS
        assert kwargs["eval_episode_ids"] == entry.EVAL_IDS
        return {"trajectories": {"train": 0 if zero else 16, "eval": 4}}

    monkeypatch.setattr(entry, "capture_process", capture)
    monkeypatch.setattr(entry, "validate_cell_evidence", validate)
    result = entry.run_cell(base, "M1" if zero else "M2", zero=zero)
    assert result["status"] == "COMPLETE" and len(calls) == 1
    assert result["actual_rounds"] == (0 if zero else 2)
    assert result["resume_supported"] is False and result["optimizer_included"] is False


@pytest.mark.parametrize("damage", [None, "incomplete", "missing_update"])
def test_research_evidence_routes_artifact_validator_but_keeps_learning_checks(
    tmp_path,
    monkeypatch,
    damage,
):
    from test_screening_evidence import _case

    case = _case(tmp_path, monkeypatch)
    case.options.update(
        expected_rounds=2,
        train_episode_ids=entry.TRAIN_IDS,
        eval_episode_ids=entry.EVAL_IDS,
        expected_eval_count=4,
        artifact_profile=entry.RESEARCH_PROFILE,
    )
    case.records = [
        r
        for r in case.records
        if (not r["identity"]["validate"] and r["identity"]["step"] <= 2)
        or (r["identity"]["validate"] and r["identity"]["sample_index"] < 4)
    ]
    for row in case.records:
        if row["identity"]["validate"]:
            row["identity"]["step"] = 2
            row["episode_id"] = entry.EVAL_IDS[row["identity"]["sample_index"]]
    case.diagnostics = case.diagnostics[:2]
    case.updates.update(optimizer_step_count=4, detected_role_counts={"actor": 2, "critic": 2})
    research_calls = []
    monkeypatch.setattr(
        screening_launcher,
        "validate_research_artifact",
        lambda path, **kw: research_calls.append(path),
    )
    if damage == "incomplete":
        case.records[0]["complete"] = False
    elif damage == "missing_update":
        case.updates["detected_role_counts"]["critic"] = 1
    if damage:
        with pytest.raises(RuntimeError, match="INCOMPLETE|optimizer update mismatch"):
            case.validate()
        assert not research_calls
    else:
        assert case.validate()["trajectories"] == {"train": 16, "eval": 4}
        assert research_calls == [tmp_path / "checkpoints/global_step_2"]
    assert not case.checkpoint_calls
