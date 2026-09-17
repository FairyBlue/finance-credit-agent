"""The new bounded trial keeps both arms comparable and all validation lanes visible."""

import copy
import json

import pytest
from hydra import compose, initialize_config_dir

from src.agents import common_validity_penalty as entry
from src.agents.actor_update_probe import policy_loss_contract
from src.agents.metadata import RunHashes
from src.agents.screening_launcher import validate_cell_evidence


@pytest.fixture
def prepared(tmp_path):
    model = {
        "source": "local_export",
        "name": str(tmp_path / "sft"),
        "merge_receipt_sha256": "a" * 64,
    }
    settings = entry.settings(model)
    manifest = {
        "hashes": {"payload_sha256": "b" * 64},
        "datasets": {},
        "calibration": {
            "path": "reward_calibration.json",
            "payload_sha256": "c" * 64,
            "sha256": "f" * 64,
        },
    }
    for name, ids in (("train", entry.TRAIN_IDS), ("dev", entry.EVAL_IDS)):
        manifest["datasets"][name] = {
            "path": name + ".parquet",
            "episode_ids": ids,
            "row_count": len(ids),
        }
    entry.atomic_json(tmp_path / "settings.json", settings)
    return tmp_path, settings, manifest


@pytest.mark.parametrize("cell_id", ["sft0", "m1", "m2_eta025"])
def test_real_hydra_shared_conditions_penalty_and_five_validation_lanes(prepared, cell_id):
    base, settings, manifest = prepared
    (base / cell_id).mkdir()
    config = entry.make_config(base, manifest, cell_id)
    spec = next(s for s in entry.cells() if s.cell_id == cell_id)
    command = entry.build_command(
        config=config,
        run_settings=entry.cell_settings(settings, cell_id),
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True),
        cell_dir=base / cell_id,
    )
    with initialize_config_dir(
        config_dir=str(entry.REPO_ROOT / "third_party/agent-r1/agent_r1/config"), version_base=None
    ):
        actual = compose(config_name="agent_ppo_trainer", overrides=command[3:])
    model = settings["models"]["screening"]["name"]
    assert actual.actor_rollout_ref.model.path == actual.actor_rollout_ref.ref.model.path == model
    assert actual.critic.model.path == actual.critic.model.tokenizer_path == model
    assert actual.actor_rollout_ref.actor.optim.lr == 2e-7
    # The probe must accept the actual launcher-selected loss, not just a toy
    # vanilla-PPO fixture. GSPO overrides the policy aggregation internally.
    probe_contract = policy_loss_contract(actual.actor_rollout_ref.actor)
    assert probe_contract["policy_loss_mode"] == "gspo"
    assert probe_contract["effective_policy_loss_aggregation"] == "seq-mean-token-mean"
    assert actual.critic.optim.lr == 1e-6
    assert actual.trainer.total_epochs == 1 and actual.trainer.total_training_steps == max(
        spec.rounds, 1
    )
    assert actual.trainer.test_freq == (1 if spec.zero else 2)
    assert actual.trainer.save_freq == max(spec.rounds, 1)
    assert actual.trainer.val_only == actual.trainer.val_before_train == spec.zero
    assert actual.data.train_max_samples == 8 and actual.data.train_batch_size == 2
    assert not actual.data.shuffle and not actual.data.validation_shuffle
    assert actual.actor_rollout_ref.rollout.n == 4
    assert actual.actor_rollout_ref.rollout.val_kwargs.n == 5
    assert actual.actor_rollout_ref.rollout.val_kwargs.do_sample is True
    assert actual.data.apply_chat_template_kwargs.enable_thinking is False
    assert actual.trainer.resume_mode == "disable"
    assert actual.trainer.resume_from_path is None
    assert (
        actual.actor_rollout_ref.actor.fsdp_config.seed
        == actual.critic.model.fsdp_config.seed
        == 42
    )
    assert list(actual.actor_rollout_ref.actor.checkpoint.save_contents) == [
        "finance_actor_bf16_hf"
    ]
    assert list(actual.critic.checkpoint.save_contents) == []
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == spec.eta
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert config["screening"]["invalid_action_penalty"] == 0.05
    assert str(base / cell_id / "artifacts/actor_update") == env.FINANCE_ACTOR_UPDATE_PROBE_DIR
    assert env.FINANCE_PAIRED_VALIDATION_PROFILE == "greedy1_sample4_v1"
    assert (
        str(base / cell_id / "artifacts/paired_validation")
        == env.FINANCE_PAIRED_VALIDATION_EVIDENCE_DIR
    )
    assert not any(k.startswith("FINANCE_LEARNING_") for k in env)


@pytest.mark.parametrize(
    "field,value",
    [("learning_rate", 1e-6), ("total_epochs", 2), ("total_training_steps", 8), ("episodes", 16)],
)
def test_partially_changed_training_condition_rejected(prepared, field, value):
    base, settings, manifest = prepared
    cfg = entry.make_config(base, manifest, "m2_eta025")
    cfg["training"][field] = value
    with pytest.raises(RuntimeError, match="budget/LR"):
        entry.build_command(
            config=cfg,
            run_settings=entry.cell_settings(settings, "m2_eta025"),
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True),
            cell_dir=base / "m2_eta025",
        )


def validation_rows():
    return [
        dict(
            schema_version=1,
            identity=dict(step=0, sample_index=index, rollout_n=lane, validate=True),
            episode_id=episode,
            arm="M1",
            base_seed=42,
            complete=True,
            terminal_score=1.0,
            metrics={"terminal_score": 1.0, "calibration_sha256": "c" * 64},
        )
        for index, episode in enumerate(entry.EVAL_IDS)
        for lane in range(5)
    ]


@pytest.mark.parametrize("drop_lane", [False, True])
def test_shared_validator_requires_all_twenty_validation_trajectories(tmp_path, drop_lane):
    records = validation_rows()
    if drop_lane:
        records.pop()
    target = tmp_path / "artifacts/screening_evidence"
    target.mkdir(parents=True)
    (target / "sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    kwargs = dict(
        stage="zero-shot",
        expected_rounds=1,
        arm="M1",
        seed=42,
        train_episode_ids=entry.TRAIN_IDS,
        eval_episode_ids=entry.EVAL_IDS,
        execution_identity_sha256="e" * 64,
        calibration_payload_sha256="c" * 64,
        expected_eval_steps=[0],
        eval_rollout_n=5,
    )
    if drop_lane:
        with pytest.raises(RuntimeError, match="trajectory count mismatch"):
            validate_cell_evidence(tmp_path, **kwargs)
    else:
        assert validate_cell_evidence(tmp_path, **kwargs)["trajectories"] == {
            "eval": 20,
        }


@pytest.mark.skipif(
    not (entry.REPO_ROOT / "artifacts/m2_screening/manifest.json").is_file(),
    reason="requires the local prepared M2 screening manifest, absent from a clean clone",
)
def test_date_selection_ignores_benchmark_outcomes():
    import yaml

    formal = yaml.safe_load(
        (entry.REPO_ROOT / "configs/episode_manifest.reviewed.yaml").read_text()
    )
    old = entry.restart.read(entry.REPO_ROOT / "artifacts/m2_screening/manifest.json")
    selected, description = entry.select_episodes(formal, old)
    assert [e["episode_id"] for e in selected] == entry.TRAIN_IDS
    assert description["eligible_start_count"] == 121
    assert description["unique_training_sessions"] < 154
    assert [e["episode_id"] for e in selected] == entry.coverage.TRAIN_IDS[:8]
    for row in formal["splits"]["train"]:
        row["benchmark_closes"] = [-999] * 21
    changed, _ = entry.select_episodes(formal, old)
    assert [e["episode_id"] for e in changed] == entry.TRAIN_IDS


def test_training_window_mutation_cannot_change_settings_silently(prepared):
    base, settings, _ = prepared
    altered = copy.deepcopy(settings)
    altered["training_coverage"]["training_episode_ids"][0] = "formal-train-0441"
    entry.atomic_json(base / "settings.json", altered)
    with pytest.raises(RuntimeError, match="settings differ"):
        entry.read_settings(base)


@pytest.mark.parametrize("where", ["settings", "config"])
def test_penalty_cannot_change_independently_of_sealed_condition(prepared, where):
    base, settings, manifest = prepared
    config = entry.make_config(base, manifest, "m1")
    run_settings = entry.cell_settings(settings, "m1")
    if where == "settings":
        run_settings["common_validity_penalty"]["coefficient"] = 0.1
    else:
        config["screening"]["invalid_action_penalty"] = 0.0
    with pytest.raises(RuntimeError, match="differs"):
        entry.build_command(
            config=config,
            run_settings=run_settings,
            hashes=RunHashes("d" * 40, "e" * 40, "fixture", True),
            cell_dir=base / "m1",
        )


def penalty_records(spec):
    """Small complete requests with a known penalized return at every step."""
    records = []
    for update in range(1, spec.rounds + 1):
        rows, trajectories = [], []
        for sibling in range(8):
            uid = f"{update}-{sibling}"
            rows.extend(
                dict(
                    trajectory_uid=uid,
                    step_index=step,
                    policy_step=update,
                    invalid_action=step == 0,
                    terminal_Y=1.0,
                    financial_step_reward=0.5,
                    invalid_penalty=0.05 if step == 0 else 0.0,
                    training_step_reward=0.45 if step == 0 else 0.5,
                    unwhitened_U=0.0,
                    sibling_residual=0.0,
                    final_actor_advantage=0.0,
                    critic_return=0.95 if step == 0 else 0.5,
                )
                for step in range(2)
            )
            trajectories.append(
                dict(
                    trajectory_uid=uid,
                    step_count=2,
                    invalid_count=1,
                    terminal_Y=1.0,
                    financial_reward_sum=1.0,
                    penalty_sum=0.05,
                    training_reward_sum=0.95,
                    expected_training_reward_sum=0.95,
                )
            )
        records.append(
            dict(
                policy_step=update,
                arm=spec.arm,
                run_seed=42,
                eta=spec.eta,
                real_step_count=16,
                invalid_step_count=8,
                reward_evidence=rows,
                common_validity_penalty=dict(
                    coefficient=0.05,
                    profile="environment_rejection_v1",
                    invalid_source="trading_invalid_action",
                    sibling_residual_outcome="original_financial_terminal_Y",
                    invalid_count=8,
                    financial_reward_sum=8.0,
                    penalty_sum=0.4,
                    training_reward_sum=7.6,
                    trajectories=trajectories,
                ),
            )
        )
    return records


@pytest.mark.parametrize("cell_id", ["sft0", "m1", "m2_eta025"])
def test_new_reward_tensor_receipts_conserve_original_y_and_feed_critic(tmp_path, cell_id):
    spec = next(s for s in entry.cells() if s.cell_id == cell_id)
    path = tmp_path / "artifacts/sibling_diagnostics/advantage-test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in penalty_records(spec)))
    summary = entry.validate_common_penalty_evidence(tmp_path, spec)
    assert summary["updates"] == spec.rounds
    assert summary["trajectories"] == spec.rounds * 8
    assert summary["penalty_sum"] == pytest.approx(spec.rounds * 0.4)


@pytest.mark.parametrize("mutation", ["coefficient", "penalty", "return", "row", "update", "y"])
def test_reward_validator_detects_wrong_sign_critic_target_alignment_and_missing_data(
    tmp_path, mutation
):
    spec = entry.cells()[1]
    records = penalty_records(spec)
    if mutation == "coefficient":
        records[0]["common_validity_penalty"]["coefficient"] = 0.0
    elif mutation == "penalty":
        records[0]["reward_evidence"][0]["invalid_penalty"] = -0.05
    elif mutation == "return":
        records[0]["reward_evidence"][0]["critic_return"] = 1.0
    elif mutation == "row":
        records[0]["reward_evidence"].pop()
    elif mutation == "update":
        records.pop()
    else:
        records[0]["common_validity_penalty"]["trajectories"][0]["terminal_Y"] = 0.95
    path = tmp_path / "artifacts/sibling_diagnostics/advantage-test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(RuntimeError, match="common penalty"):
        entry.validate_common_penalty_evidence(tmp_path, spec)


def test_critic_suffix_accepts_near_zero_float32_cancellation(tmp_path):
    spec = entry.cells()[1]
    records = penalty_records(spec)
    record = records[0]
    # Both raw financial rewards are 0.025: Y=0.05, one invalid request costs
    # 0.05, and the first penalized return is therefore exactly zero.  A small
    # residual can survive GAE's float32 V cancellation without changing r'.
    for row in record["reward_evidence"][:2]:
        row["terminal_Y"] = 0.05
        row["financial_step_reward"] = 0.025
        row["training_step_reward"] = -0.025 if row["step_index"] == 0 else 0.025
        row["critic_return"] = 3e-5 if row["step_index"] == 0 else 0.025
    penalty = record["common_validity_penalty"]
    penalty["financial_reward_sum"] = 7.05
    penalty["training_reward_sum"] = 6.65
    penalty["trajectories"][0].update(
        terminal_Y=0.05,
        financial_reward_sum=0.05,
        training_reward_sum=0.0,
        expected_training_reward_sum=0.0,
    )
    path = tmp_path / "artifacts/sibling_diagnostics/advantage-test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert entry.validate_common_penalty_evidence(tmp_path, spec)["status"] == "PASSED"
    record["reward_evidence"][0]["critic_return"] = 0.05
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(RuntimeError, match="critic returns"):
        entry.validate_common_penalty_evidence(tmp_path, spec)
