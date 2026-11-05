"""Checkpoint interruption and resume guards without importing GPU workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.agents.screening_checkpoint import (
    COMPLETION_MARKER,
    SCREENING_IDENTITY_ENV,
    _label_evaluation_transport_metrics,
    install_checkpoint_hooks,
    validate_checkpoint,
)


def _evaluation_ledger(tmp_path, monkeypatch, *, financial_result=None):
    root = tmp_path / "artifacts/screening_evidence"
    root.mkdir(parents=True)
    monkeypatch.setenv("FINANCE_SCREENING_EVIDENCE_DIR", str(root))
    monkeypatch.setenv("FINANCE_GRID_EVALUATION_INCOMPLETE", "retain_prefix_v1")
    rows = [
        {"identity": {"step": 2, "validate": True}, "status": "COMPLETE"},
        {"identity": {"step": 2, "validate": True}, "status": "INCOMPLETE",
         "evaluation_failure_profile": "retain_prefix_v1",
         "terminal_score": financial_result, "metrics": None},
    ]
    (root / "screening.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def test_opt_in_renames_upstream_zero_transport_scores_and_discloses_incomplete_count(
    tmp_path, monkeypatch,
):
    root = _evaluation_ledger(tmp_path, monkeypatch)
    original = {"val-core/train/reward/mean@5": 0.4, "val-aux/train/reward/std@5": 0.1}
    result = _label_evaluation_transport_metrics(original, step=2)
    assert original == {"val-core/train/reward/mean@5": 0.4,
                        "val-aux/train/reward/std@5": 0.1}
    assert not any(key.startswith(("val-core/", "val-aux/")) for key in result)
    assert result["transport-only-val-core/train/reward/mean@5"] == 0.4
    assert result["finance-grid/evaluation_incomplete_trajectories"] == 1
    receipt = json.loads((root.parent / "evaluation_transport/validation-step-2.json").read_text())
    assert receipt["financial_status"] == "INCOMPLETE"
    assert receipt["failed_financial_results"] is None
    assert receipt["counts"] == {"COMPLETE": 1, "INCOMPLETE": 1}
    assert receipt["upstream_scores_are_transport_only"] is True
    with pytest.raises(FileExistsError):
        _label_evaluation_transport_metrics(original, step=2)


def test_opt_in_metric_hook_rejects_zero_presented_as_failed_financial_result(
    tmp_path, monkeypatch,
):
    _evaluation_ledger(tmp_path, monkeypatch, financial_result=0.0)
    with pytest.raises(RuntimeError, match="claims a financial result"):
        _label_evaluation_transport_metrics({"val-core/reward": 0.0}, step=2)


def test_metric_hook_preserves_exact_old_behavior_without_opt_in(monkeypatch):
    monkeypatch.delenv("FINANCE_GRID_EVALUATION_INCOMPLETE", raising=False)
    metrics = {"val-core/reward": 0.5}
    assert _label_evaluation_transport_metrics(metrics, step=2) is metrics


@pytest.fixture
def trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(SCREENING_IDENTITY_ENV, "a" * 64)

    class BaseTrainer:
        config: Any
        global_steps: int
        save_calls: int
        load_calls: int
        omitted: str | None
        fail_save: bool
        use_critic: bool
        total_training_steps: int
        actor_rollout_wg: object
        critic_wg: object
        validate_calls: int
        fail_validate: bool
        marker_seen_by_validation: dict[str, Any] | None

        def _save_checkpoint(self):
            self.save_calls += 1
            directory = (
                Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            )
            for role in ("actor", "critic"):
                (directory / role).mkdir(parents=True, exist_ok=True)
                for kind in ("model", "optim", "extra_state"):
                    relative = f"{role}/{kind}_world_size_1_rank_0.pt"
                    if relative != self.omitted:
                        (directory / relative).write_bytes((relative + "-state").encode())
            if self.omitted != "data.pt":
                (directory / "data.pt").write_bytes(b"dataloader-cursor-state")
            if self.fail_save:
                raise RuntimeError("interrupted save")
            return "saved"

        def _load_checkpoint(self):
            self.load_calls += 1
            return "loaded"

        def _validate(self):
            self.validate_calls += 1
            if self.global_steps > 0 and self.global_steps == self.total_training_steps:
                directory = (
                    Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
                )
                if (directory / COMPLETION_MARKER).exists():
                    self.marker_seen_by_validation = validate_checkpoint(
                        directory, identity_sha256="a" * 64
                    )
            if self.fail_validate:
                raise RuntimeError("strict final evaluation failed")
            return {"validation": "complete"}

    class ScreeningTrainer(BaseTrainer):
        pass

    checkpoint = {
        "save_contents": ["model", "optimizer", "extra"],
        "load_contents": ["model", "optimizer", "extra"],
        "async_save": False,
    }
    instance = ScreeningTrainer()
    instance.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"actor": {"checkpoint": checkpoint}},
            "critic": {"checkpoint": checkpoint},
            "trainer": {
                "default_local_dir": str(tmp_path / "checkpoints"),
                "default_hdfs_dir": None,
                "nnodes": 1,
                "n_gpus_per_node": 1,
                "resume_mode": "disable",
                "resume_from_path": None,
            },
        }
    )
    instance.use_critic = True
    instance.global_steps = 4
    instance.save_calls = 0
    instance.load_calls = 0
    instance.omitted = None
    instance.fail_save = False
    instance.total_training_steps = 4
    instance.actor_rollout_wg = object()
    instance.critic_wg = object()
    instance.validate_calls = 0
    instance.fail_validate = False
    instance.marker_seen_by_validation = None
    original = BaseTrainer._save_checkpoint
    assert install_checkpoint_hooks(ScreeningTrainer)
    assert BaseTrainer._save_checkpoint is original
    assert not install_checkpoint_hooks(ScreeningTrainer)
    return instance


def _directory(trainer) -> Path:
    return Path(trainer.config.trainer.default_local_dir) / f"global_step_{trainer.global_steps}"


def _resume(trainer):
    trainer.config.trainer.resume_mode = "resume_path"
    trainer.config.trainer.resume_from_path = str(_directory(trainer))
    return trainer._load_checkpoint()


def test_successful_save_marks_both_roles_and_dataloader_then_allows_resume(trainer):
    assert trainer._save_checkpoint() == "saved"
    directory = _directory(trainer)
    marker = validate_checkpoint(directory, identity_sha256="a" * 64)
    assert marker["global_step"] == 4
    assert len(marker["files"]) == 7
    assert "data.pt" in marker["files"]
    assert "critic/optim_world_size_1_rank_0.pt" in marker["files"]
    assert list(directory.glob("*.tmp")) == []
    assert _resume(trainer) == "loaded" and trainer.load_calls == 1


@pytest.mark.parametrize(
    "missing",
    [
        "data.pt",
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "critic/model_world_size_1_rank_0.pt",
        "critic/optim_world_size_1_rank_0.pt",
        "critic/extra_state_world_size_1_rank_0.pt",
    ],
)
def test_partial_successful_upstream_save_never_gets_marker(trainer, missing):
    trainer.omitted = missing
    with pytest.raises(RuntimeError, match="incomplete screening checkpoint"):
        trainer._save_checkpoint()
    assert not (_directory(trainer) / COMPLETION_MARKER).exists()


def test_interrupted_rewrite_invalidates_stale_completion_marker(trainer):
    trainer._save_checkpoint()
    trainer.fail_save = True
    with pytest.raises(RuntimeError, match="interrupted save"):
        trainer._save_checkpoint()
    assert not (_directory(trainer) / COMPLETION_MARKER).exists()
    with pytest.raises(RuntimeError, match="marker is missing"):
        _resume(trainer)
    assert trainer.load_calls == 0


@pytest.mark.parametrize("damage", ["marker", "data", "critic", "size", "identity", "step"])
def test_resume_rejects_damage_before_upstream_can_mutate_state(trainer, damage):
    trainer._save_checkpoint()
    directory = _directory(trainer)
    marker_path = directory / COMPLETION_MARKER
    if damage == "marker":
        marker_path.unlink()
    elif damage == "data":
        (directory / "data.pt").unlink()
    elif damage == "critic":
        (directory / "critic/extra_state_world_size_1_rank_0.pt").unlink()
    elif damage == "size":
        (directory / "actor/model_world_size_1_rank_0.pt").write_bytes(b"short")
    else:
        marker = json.loads(marker_path.read_text())
        marker["identity_sha256" if damage == "identity" else "global_step"] = (
            "b" * 64 if damage == "identity" else 3
        )
        marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError):
        _resume(trainer)
    assert trainer.load_calls == 0


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_async_checkpointing_is_rejected_before_loading_or_saving(trainer, role):
    checkpoint = (
        trainer.config.actor_rollout_ref.actor.checkpoint
        if role == "actor"
        else trainer.config.critic.checkpoint
    )
    checkpoint.async_save = True
    with pytest.raises(RuntimeError, match="async_save"):
        trainer._save_checkpoint()
    with pytest.raises(RuntimeError, match="async_save"):
        trainer._load_checkpoint()
    assert trainer.save_calls == trainer.load_calls == 0


@pytest.mark.parametrize("direction", ["save_contents", "load_contents"])
def test_checkpoint_contract_cannot_omit_optimizer_or_rng(trainer, direction):
    trainer.config.critic.checkpoint[direction] = ["model"]
    with pytest.raises(RuntimeError, match="model/optimizer/extra"):
        trainer._load_checkpoint()
    assert trainer.load_calls == 0


def test_fresh_start_is_allowed_but_automatic_latest_checkpoint_selection_is_not(trainer):
    assert trainer._load_checkpoint() == "loaded"
    trainer.config.trainer.resume_mode = "auto"
    with pytest.raises(RuntimeError, match="auto is forbidden"):
        trainer._load_checkpoint()
    assert trainer.load_calls == 1


def test_missing_run_identity_never_writes_a_checkpoint(trainer, monkeypatch):
    monkeypatch.delenv(SCREENING_IDENTITY_ENV)
    with pytest.raises(RuntimeError, match=SCREENING_IDENTITY_ENV):
        trainer._save_checkpoint()
    assert trainer.save_calls == 0


def test_final_validation_sees_complete_actor_critic_and_data_checkpoint(trainer):
    assert trainer._validate() == {"validation": "complete"}
    assert trainer.save_calls == trainer.validate_calls == 1
    assert trainer.marker_seen_by_validation == validate_checkpoint(
        _directory(trainer), identity_sha256="a" * 64
    )
    assert len(trainer.marker_seen_by_validation["files"]) == 7


def test_failed_final_validation_preserves_updated_checkpoint(trainer):
    trainer.fail_validate = True
    with pytest.raises(RuntimeError, match="strict final evaluation failed"):
        trainer._validate()
    assert trainer.save_calls == trainer.validate_calls == 1
    assert validate_checkpoint(_directory(trainer), identity_sha256="a" * 64)


def test_upstream_save_after_final_validation_is_idempotent(trainer):
    trainer._validate()
    directory = _directory(trainer)
    before = {
        p: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.rglob("*") if p.is_file()
    }
    trainer.fail_save = True  # A second expensive upstream save must not execute.
    assert trainer._save_checkpoint() == "saved"
    assert trainer._save_checkpoint() == "saved"
    assert trainer.save_calls == 1
    assert before == {
        p: (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("step", [0, 2, 5])
def test_zero_shot_intermediate_and_over_budget_validation_do_not_save(trainer, step):
    trainer.global_steps = step
    assert trainer._validate() == {"validation": "complete"}
    assert trainer.save_calls == 0 and trainer.validate_calls == 1
    assert not _directory(trainer).exists()


def test_validation_only_mode_does_not_save_even_at_positive_final_step(trainer):
    trainer.config.trainer.val_only = True
    trainer._validate()
    assert trainer.save_calls == 0


@pytest.mark.parametrize("damage", ["marker", "data.pt", "critic/optim_world_size_1_rank_0.pt"])
def test_final_checkpoint_damage_cannot_be_hidden_by_idempotence(trainer, damage):
    trainer._validate()
    target = COMPLETION_MARKER if damage == "marker" else damage
    (_directory(trainer) / target).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        trainer._save_checkpoint()
    assert trainer.save_calls == 1  # Fail closed; do not overwrite the damaged evidence.


def test_incomplete_save_blocks_final_evaluation(trainer):
    trainer.omitted = "data.pt"
    with pytest.raises(RuntimeError, match="incomplete screening checkpoint"):
        trainer._validate()
    assert trainer.save_calls == 1 and trainer.validate_calls == 0


def test_resume_clears_final_save_idempotence_even_at_same_step(trainer):
    trainer._validate()
    assert _resume(trainer) == "loaded"
    assert trainer._save_checkpoint() == "saved"
    assert trainer.save_calls == 2


@pytest.mark.parametrize("changed", ["step", "identity", "worker"])
def test_final_save_idempotence_requires_same_live_state_identity(trainer, monkeypatch, changed):
    trainer._validate()
    if changed == "step":
        trainer.global_steps += 1
    elif changed == "identity":
        monkeypatch.setenv(SCREENING_IDENTITY_ENV, "b" * 64)
    else:
        trainer.actor_rollout_wg = object()
    assert trainer._save_checkpoint() == "saved"
    assert trainer.save_calls == 2
