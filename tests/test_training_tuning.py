"""CPU checks for the separately frozen diagnostic and eight-update tuning phases."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from src.agents import training_tuning as entry
from src.agents.metadata import RunHashes

IDENTITY = "e" * 64
MODEL_ID = "local-export-sha256:" + "a" * 64


@pytest.fixture
def prepared(tmp_path):
    def create(phase="comparison", critic_lr=1e-6):
        base = tmp_path / phase
        base.mkdir()
        model = {
            "source": "local_export",
            "name": str(tmp_path / "merged_model"),
            "merge_receipt_sha256": "a" * 64,
        }
        settings = entry.settings(
            model,
            phase=phase,
            critic_lr=critic_lr if phase == "comparison" else None,
        )
        manifest = {
            "hashes": {"payload_sha256": "b" * 64},
            "datasets": {
                name: {"path": f"{name}.parquet", "episode_ids": ids, "row_count": len(ids)}
                for name, ids in (("train", entry.TRAIN_IDS), ("dev", entry.EVAL_IDS))
            },
            "calibration": {"path": "reward_calibration.json", "payload_sha256": "c" * 64},
        }
        for item in [*manifest["datasets"].values(), manifest["calibration"]]:
            path = base / "prepared" / item["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_text(item["path"])
            item["sha256"] = entry.restart.digest(path)
        entry.atomic_json(base / "settings.json", settings)
        entry.atomic_json(base / "prepared/manifest.json", manifest)
        entry.atomic_json(
            base / "model_identity.json",
            {
                "model_revision": MODEL_ID,
                "model_snapshot_path": model["name"],
            },
        )
        entry.atomic_json(
            base / "input_preflight.json",
            {
                "status": "PASSED",
                "model_revision": MODEL_ID,
                "action_protocol": entry.PROTOCOL,
                "enable_thinking": False,
                "windows": [
                    {"episode": episode, "complete": True}
                    for episode in entry.TRAIN_IDS + entry.EVAL_IDS
                ],
            },
        )
        entry.atomic_json(
            base / "selection.json",
            {
                "phase": phase,
                "selected_critic_learning_rate": critic_lr if phase == "comparison" else None,
            },
        )
        return base, settings, manifest

    return create


@pytest.mark.parametrize(
    "phase,cell_id",
    [
        ("diagnostic", "sft0"),
        ("diagnostic", "critic_lr_1e5"),
        ("diagnostic", "critic_lr_1e6"),
        ("comparison", "sft0"),
        ("comparison", "m1"),
        ("comparison", "m2_eta1"),
        ("comparison", "m2_eta025"),
    ],
)
def test_real_hydra_freezes_models_budget_eta_probe_and_artifacts(prepared, phase, cell_id):
    base, settings, manifest = prepared(phase)
    spec = entry._cell(settings, cell_id)
    cell = base / cell_id
    cell.mkdir()
    cfg = entry.make_config(base, manifest, cell_id)
    actual_settings = entry.cell_settings(settings, cell_id)
    command = entry.build_command(
        config=cfg,
        run_settings=actual_settings,
        cell_dir=cell,
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True),
    )
    with initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        resolved = compose(config_name="agent_ppo_trainer", overrides=command[3:])
    model = settings["models"]["screening"]["name"]
    assert (
        resolved.actor_rollout_ref.model.path == resolved.actor_rollout_ref.ref.model.path == model
    )
    assert resolved.critic.model.path == resolved.critic.model.tokenizer_path == model
    assert (
        resolved.critic.model.fsdp_config.seed == resolved.actor_rollout_ref.actor.fsdp_config.seed
    )
    assert resolved.critic.model.fsdp_config.seed == 42
    assert resolved.actor_rollout_ref.actor.optim.lr == 1e-6
    assert resolved.critic.optim.lr == spec.critic_lr
    assert resolved.trainer.total_training_steps == max(1, spec.rounds)
    assert resolved.trainer.total_epochs == (2 if spec.rounds == 8 else 1)
    assert resolved.trainer.test_freq == (1 if spec.zero else 2)
    assert resolved.trainer.save_freq == max(1, spec.rounds)
    assert resolved.trainer.val_only == resolved.trainer.val_before_train == spec.zero
    assert resolved.trainer.critic_warmup == 0
    assert resolved.data.train_max_samples == 8 and resolved.data.train_batch_size == 2
    assert resolved.data.shuffle is False and resolved.data.validation_shuffle is False
    assert resolved.actor_rollout_ref.rollout.n == 4
    assert resolved.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert resolved.actor_rollout_ref.rollout.val_kwargs.do_sample is False
    assert resolved.data.screening_action_protocol == entry.PROTOCOL
    assert resolved.data.apply_chat_template_kwargs.enable_thinking is False
    assert resolved.trainer.resume_mode == "disable" and resolved.trainer.resume_from_path is None
    assert resolved.trainer.finance_artifact_profile == settings["artifact_profile"]
    actor_save = list(resolved.actor_rollout_ref.actor.checkpoint.save_contents)
    assert actor_save == (["finance_actor_bf16_hf"] if phase == "comparison" else [])
    assert list(resolved.actor_rollout_ref.actor.checkpoint.load_contents) == []
    assert list(resolved.critic.checkpoint.save_contents) == []
    assert list(resolved.critic.checkpoint.load_contents) == []
    env = resolved.ray_kwargs.ray_init.runtime_env.env_vars
    assert all(isinstance(value, str) for value in env.values())
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == spec.eta
    assert str(cell / "artifacts/critic_fit") == env.FINANCE_CRITIC_FIT_PROBE_DIR
    assert env.FINANCE_SCREENING_CRITIC_INIT == "cpu_seed_v1"
    assert not any(name.startswith("FINANCE_LEARNING_") for name in env)


@pytest.mark.parametrize("damage", ["rounds", "epochs", "lr", "eta", "artifact", "cell"])
def test_command_rejects_partially_changed_condition_before_dispatch(prepared, damage):
    base, settings, manifest = prepared()
    cfg = entry.make_config(base, manifest, "m2_eta025")
    run_settings = entry.cell_settings(settings, "m2_eta025")
    if damage == "rounds":
        cfg["training"]["total_training_steps"] = 10
    elif damage == "epochs":
        cfg["training"]["total_epochs"] = 1
    elif damage == "lr":
        cfg["training"]["critic_learning_rate"] = 1e-5
    elif damage == "eta":
        cfg["screening"]["residual_eta"] = 1.0
    elif damage == "artifact":
        cfg["screening"]["artifact_profile"] = "research_model_extra_v1"
    else:
        run_settings["training_tuning"]["cell"]["rounds"] = 2
    with pytest.raises(ValueError, match="differ"):
        entry.build_command(config=cfg, run_settings=run_settings, hashes=None, cell_dir=base)


@pytest.mark.parametrize("damage", ["rounds", "eta", "lr", "phase", "eval", "bool"])
def test_saved_phase_settings_cannot_be_adapted(prepared, damage):
    base, settings, _ = prepared()
    if damage == "rounds":
        settings["training"]["total_training_steps"] = 10
    elif damage == "eta":
        settings["m2_residual_eta"] = 0.25
    elif damage == "lr":
        settings["training"]["critic_learning_rate"] = 1e-5
    elif damage == "phase":
        settings["training_tuning"]["phase"] = "freeform"
    elif damage == "bool":
        settings["training"]["max_checkpoints"] = True
    else:
        settings["training_tuning"]["evaluation_steps"] = [8]
    entry.atomic_json(base / "settings.json", settings)
    with pytest.raises(RuntimeError, match="contract"):
        entry.read_settings(base)


def write_fit(cell, spec):
    records, diagnostics, updates = [], [], []
    stats = {"mean": 1.0, "min": 1.0, "max": 1.0, "std_population": 0.0}
    fit = {
        "values": stats,
        "returns": stats,
        "mse": 0.0,
        "half_mse": 0.0,
        "rmse": 0.0,
        "mean_bias_value_minus_return": 0.0,
        "half_squared_mean_bias": 0.0,
        "error_std_population": 0.0,
        "explained_variance": None,
    }
    for step in range(1, spec.rounds + 1):
        records.append(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "phase": "critic_same_batch_pre_post",
                "arm": spec.arm,
                "pid": 99,
                "update_call_index": step,
                "policy_step": step,
                "execution_identity_sha256": IDENTITY,
                "returns_recomputed": False,
                "extra_forward_passes": 1,
                "real_step_count": 160,
                "value_token_count": 160,
                "padding_row_count": 0,
                "planned_mini_batch_count": 1,
                "ppo_epochs": 1,
                "forward_meta_info": {
                    "micro_batch_size": 1,
                    "max_token_len": 32768,
                    "use_dynamic_bsz": False,
                },
                "dropout": {"any_nonzero": False, "fit_comparison_uses_same_mode": True},
                "mode_restored": True,
                "critic_learning_rates": [spec.critic_lr],
                "optimizer_grad_norms_before_clip": [100.0],
                "extra_forward_seconds_synchronized": 1.0,
                "pre": copy.deepcopy(fit),
                "post": copy.deepcopy(fit),
            }
        )
        diagnostics.append(
            {
                "policy_step": step,
                "real_step_count": 160,
                "padding_row_count": 0,
                "eta": spec.eta,
            }
        )
        updates.append(
            {"role": "critic", "pid": 99, "grad_norm": 100.0, "global_update_count": step}
        )
    for name, data in (
        ("critic_fit", records),
        ("sibling_diagnostics", diagnostics),
        ("update_probe", updates),
    ):
        path = cell / "artifacts" / name / "worker.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in data))
    return records


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "missing",
        "step",
        "identity",
        "lr",
        "rows",
        "nonfinite",
        "targets",
        "gradient",
    ],
)
def test_fit_evidence_requires_every_matching_update(prepared, damage):
    base, settings, _ = prepared("diagnostic")
    spec = entry._cell(settings, "critic_lr_1e6")
    cell = base / spec.cell_id
    records = write_fit(cell, spec)
    if damage == "missing":
        records.pop()
    elif damage == "step":
        records[1]["policy_step"] = 1
    elif damage == "identity":
        records[0]["execution_identity_sha256"] = "f" * 64
    elif damage == "lr":
        records[0]["critic_learning_rates"] = [1e-5]
    elif damage == "rows":
        records[0]["value_token_count"] = 159
    elif damage == "nonfinite":
        records[0]["pre"]["mse"] = float("nan")
    elif damage == "targets":
        records[0]["post"]["returns"]["mean"] = 2.0
    elif damage == "gradient":
        records[0]["optimizer_grad_norms_before_clip"] = [200.0]
    path = cell / "artifacts/critic_fit/worker.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    if damage:
        with pytest.raises(RuntimeError, match="critic fit"):
            entry.validate_fit_evidence(cell, spec=spec, execution_identity=IDENTITY)
    else:
        result = entry.validate_fit_evidence(cell, spec=spec, execution_identity=IDENTITY)
        assert result["updates"] == 2 and result["policy_steps"] == [1, 2]
        assert result["source_sha256"] == {
            "artifacts/critic_fit/worker.jsonl": entry.restart.digest(path),
        }


def diagnostic_results(base, settings):
    summaries = []
    for spec in entry.cells(settings):
        summary = {
            "status": "COMPLETE",
            "cell": asdict(spec),
            "actual_rounds": spec.rounds,
            "model_revision": MODEL_ID,
            "execution_identity_sha256": IDENTITY,
        }
        entry.atomic_json(base / spec.cell_id / "summary.json", summary)
        if not spec.zero:
            write_fit(base / spec.cell_id, spec)
        summaries.append(summary)
    entry.atomic_json(base / "state.json", {"status": "COMPLETE", "cells": summaries})
    entry.atomic_json(base / "run_contract.json", {"profile": entry.PROFILE})


def test_worse_fit_is_reportable_evidence_and_sft0_must_have_no_updates(prepared):
    base, settings, _ = prepared("diagnostic")
    spec = entry._cell(settings, "critic_lr_1e6")
    cell = base / spec.cell_id
    records = write_fit(cell, spec)
    for row in records:
        row["post"].update(mse=4.0, half_mse=2.0, rmse=2.0)
        row["post"]["values"] = {
            "mean": 3.0,
            "min": 3.0,
            "max": 3.0,
            "std_population": 0.0,
        }
    (cell / "artifacts/critic_fit/worker.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    assert entry.validate_fit_evidence(cell, spec=spec, execution_identity=IDENTITY)["updates"] == 2
    with pytest.raises(RuntimeError, match="SFT0 unexpectedly"):
        entry.validate_fit_evidence(
            cell,
            spec=entry._cell(settings, "sft0"),
            execution_identity=IDENTITY,
        )


@pytest.mark.parametrize("damage", [None, "unfinished", "model", "unsupported_lr"])
def test_selection_is_explicit_and_freezes_actual_diagnostic_evidence(prepared, damage):
    base, settings, _ = prepared("diagnostic")
    diagnostic_results(base, settings)
    if damage == "unfinished":
        entry.atomic_json(base / "state.json", {"status": "FAILED"})
    kwargs = {
        "phase": "comparison",
        "critic_lr": 1e-6,
        "diagnostic_directory": base,
        "selection_reason": "smaller same-batch prediction shift",
        "model_revision": MODEL_ID,
    }
    if damage == "model":
        kwargs["model_revision"] = "other"
    elif damage == "unsupported_lr":
        kwargs["critic_lr"] = 5e-7
    if damage:
        with pytest.raises((RuntimeError, ValueError)):
            entry._selection(**kwargs)
    else:
        result = entry._selection(**kwargs)
        assert result["selected_critic_learning_rate"] == 1e-6
        assert len([name for name in result["source_sha256"] if "critic_fit" in name]) == 2
        assert result["reason"] == kwargs["selection_reason"]


def test_selection_requires_reason_and_both_candidates():
    with pytest.raises(ValueError, match="selection reason"):
        entry._selection(
            phase="comparison",
            critic_lr=1e-6,
            diagnostic_directory=None,
            selection_reason=None,
            model_revision=MODEL_ID,
        )


@pytest.mark.parametrize(
    "phase,fail_cell",
    [
        ("diagnostic", None),
        ("comparison", None),
        ("comparison", "m2_eta1"),
    ],
)
def test_phase_schedule_is_unique_fresh_and_never_retries(prepared, monkeypatch, phase, fail_cell):
    base, settings, _ = prepared(phase)
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(
        entry.shutil, "disk_usage", lambda base: SimpleNamespace(free=100 * 1024**3)
    )
    calls = []

    def run_cell(base, cell_id):
        calls.append(cell_id)
        if cell_id == fail_cell:
            raise RuntimeError("fixture failed")
        return {"status": "COMPLETE", "cell_id": cell_id}

    monkeypatch.setattr(entry, "run_cell", run_cell)
    if fail_cell:
        with pytest.raises(RuntimeError, match="fixture failed"):
            entry.run(base)
    else:
        entry.run(base)
    expected = [spec.cell_id for spec in entry.cells(settings)]
    assert calls == (expected[: expected.index(fail_cell) + 1] if fail_cell else expected)
    state = entry.restart.read(base / "state.json")
    assert state["automatic_retry"] is False and state["resume_supported"] is False
    assert state["adaptive_gate"] is False
    assert state["train_episode_ids"] == entry.TRAIN_IDS
    with pytest.raises(RuntimeError, match="no implicit retry"):
        entry.run(base)


@pytest.mark.parametrize("cell_id", ["sft0", "m2_eta025"])
def test_run_cell_dispatches_exact_eval_grid_and_validates_probe(prepared, monkeypatch, cell_id):
    base, settings, _ = prepared()
    model = settings["models"]["screening"]["name"]
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(
        entry,
        "collect_run_hashes",
        lambda **kwargs: RunHashes(
            "d" * 40,
            "e" * 40,
            "fixture",
            True,
        ),
    )
    monkeypatch.setattr(
        entry,
        "verify_gpu_runtime_and_model",
        lambda cfg: {
            "model_revision": MODEL_ID,
            "model_snapshot_path": model,
            "c_compiler": "/gcc",
        },
    )
    monkeypatch.setattr(
        entry,
        "_training_environment",
        lambda **kwargs: {
            "CC": kwargs["c_compiler"],
            "FINANCE_LEARNING_GATE_DIR": "/stale",
            "FINANCE_CRITIC_FIT_PROBE_DIR": "/stale",
        },
    )
    if cell_id != "sft0":
        entry.atomic_json(
            base / "sft0/summary.json",
            {
                "status": "COMPLETE",
                "critic_initial_head_sha256": "9" * 64,
            },
        )
    seen = []

    def capture(argv, *, environment, cell_dir):
        assert environment["CC"] == "/gcc"
        assert not any(name.startswith("FINANCE_") for name in environment)
        options = dict(argument.lstrip("+").split("=", 1) for argument in argv[3:])
        identity = options[
            "ray_kwargs.ray_init.runtime_env.env_vars.FINANCE_SCREENING_IDENTITY_SHA256"
        ]
        entry.atomic_json(
            cell_dir / "artifacts/critic_initialization/head-rank0.json",
            {
                "seed": 42,
                "mode": "cpu_seed_v1",
                "arm": "M1" if cell_id == "sft0" else "M2",
                "execution_identity_sha256": identity,
                "model_path": model,
                "phase": "critic_head_after_hf_load_before_fsdp",
                "head_sha256": "9" * 64,
            },
        )
        return 0

    def validate(cell, **kwargs):
        assert kwargs["expected_rounds"] == (1 if cell_id == "sft0" else 8)
        assert kwargs["expected_eval_steps"] == ([0] if cell_id == "sft0" else [2, 4, 6, 8])
        assert kwargs["expected_eta"] == (0.0 if cell_id == "sft0" else 0.25)
        assert kwargs["artifact_profile"] == entry.COMPARISON_ARTIFACT
        seen.append("grid")
        return {"valid": True}

    def validate_fit(cell, *, spec, execution_identity):
        assert spec.cell_id == cell_id and len(execution_identity) == 64
        seen.append("fit")
        return {"updates": spec.rounds}

    monkeypatch.setattr(entry, "capture_process", capture)
    monkeypatch.setattr(entry, "validate_cell_evidence", validate)
    monkeypatch.setattr(entry, "validate_fit_evidence", validate_fit)
    result = entry.run_cell(base, cell_id)
    assert result["status"] == "COMPLETE" and seen == ["grid", "fit"]
    assert result["actual_rounds"] == (0 if cell_id == "sft0" else 8)


@pytest.mark.parametrize("damage", ["settings", "source", "source_manifest", "plan", "parquet"])
def test_seal_detects_input_source_and_plan_drift(prepared, monkeypatch, tmp_path, damage):
    base, settings, manifest = prepared("diagnostic")
    fake_repo = tmp_path / "repo"
    source = fake_repo / "src/agents/training_tuning.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen source")
    plan = fake_repo / settings["plan"]
    plan.parent.mkdir(parents=True)
    plan.write_text("fixed plan")
    monkeypatch.setattr(entry, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(entry, "_source_files", lambda: {"src/agents/training_tuning.py": source})
    monkeypatch.setattr(
        entry.restart,
        "seal",
        lambda base: entry.atomic_json(
            base / "source_identity.json",
            {"src/agents/training_tuning.py": entry.restart.digest(source)},
        ),
    )
    model = settings["models"]["screening"]["name"]
    monkeypatch.setattr(
        entry,
        "_resolve_exact_model_snapshot",
        lambda config: (
            model,
            MODEL_ID,
            entry.Path(model),
        ),
    )
    entry.seal(base)
    entry.check_contract(base)
    if damage == "settings":
        settings["m2_residual_eta"] = 0.25
        entry.atomic_json(base / "settings.json", settings)
    elif damage == "source":
        source.write_text("changed source")
    elif damage == "source_manifest":
        entry.atomic_json(base / "source_identity.json", {})
    elif damage == "plan":
        (base / "plan.md").write_text("changed budget")
    else:
        path = base / "prepared/train.parquet"
        path.write_text("replaced parquet")
        manifest["datasets"]["train"]["sha256"] = entry.restart.digest(path)
        entry.atomic_json(base / "prepared/manifest.json", manifest)
    with pytest.raises(RuntimeError, match="changed"):
        entry.check_contract(base)
    with pytest.raises(RuntimeError, match="already sealed"):
        entry.seal(base)
