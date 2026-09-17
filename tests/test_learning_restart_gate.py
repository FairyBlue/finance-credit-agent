"""Exercise the two-round decision with the actual checkpoint/resource wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.agents.learning_restart_gate import DIRECTORY_ENV, ENABLE_ENV, ENABLE_VALUE, patch_trainer
from src.agents.screening_checkpoint import install_checkpoint_hooks, validate_checkpoint
from src.agents.screening_resources import install_resource_phase_hooks


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    diagnostics = tmp_path / "artifacts/sibling_diagnostics"
    diagnostics.mkdir(parents=True)
    monkeypatch.setenv(ENABLE_ENV, ENABLE_VALUE)
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M2")
    monkeypatch.setenv("FINANCE_SCREENING_RUN_SEED", "42")
    monkeypatch.setenv("FINANCE_SCREENING_DIAGNOSTICS_DIR", str(diagnostics))
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "a" * 64)

    class Trainer:
        def __init__(self) -> None:
            checkpoint = {
                "save_contents": ["model", "optimizer", "extra"],
                "load_contents": ["model", "optimizer", "extra"],
                "async_save": False,
            }
            self.config = OmegaConf.create({
                "actor_rollout_ref": {"actor": {"checkpoint": checkpoint}},
                "critic": {"checkpoint": checkpoint},
                "trainer": {
                    "default_local_dir": str(tmp_path / "checkpoints"),
                    "default_hdfs_dir": None, "nnodes": 1, "n_gpus_per_node": 1,
                    "resume_mode": "disable", "resume_from_path": None,
                    "save_freq": 2, "test_freq": 4, "val_only": False,
                },
            })
            self.use_critic = True
            self.total_training_steps = 4
            self.global_steps = 0
            self.events: list[tuple[str, int]] = []
            self.nonzero = False
            self.missing_step2 = False
            self.fail_save: BaseException | None = None
            self.fail_eval: BaseException | None = None
            self.actor_rollout_wg = SimpleNamespace(
                async_calls_finalize_fn_exec=self._finalize
            )
            self.critic_wg = object()

        def _finalize(self, *, blocking: bool) -> None:
            assert blocking
            self.events.append(("finalize", self.global_steps))

        def _compute_old_log_prob(self, batch: Any) -> Any:
            return batch

        _compute_ref_log_prob = _compute_old_log_prob
        _compute_values = _compute_old_log_prob

        def _update_critic(self, batch: Any) -> Any:
            self.events.append(("critic", self.global_steps))
            return batch

        def _update_actor(self, batch: Any) -> Any:
            self.events.append(("actor", self.global_steps))
            return batch

        def _load_checkpoint(self) -> None:
            return None

        def _save_checkpoint(self) -> None:
            self.events.append(("save", self.global_steps))
            if self.fail_save:
                raise self.fail_save
            directory = Path(self.config.trainer.default_local_dir)
            directory /= f"global_step_{self.global_steps}"
            for role in ("actor", "critic"):
                (directory / role).mkdir(parents=True, exist_ok=True)
                for kind in ("model", "optim", "extra_state"):
                    (directory / role / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"state")
            (directory / "data.pt").write_bytes(b"data cursor")

        def _validate(self) -> dict[str, float]:
            self.events.append(("evaluate", self.global_steps))
            # Actual checkpoint wrapper has already made a complete save at
            # both the manual step-2 boundary and the regular final step 4.
            marker = validate_checkpoint(
                Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}",
                identity_sha256="a" * 64,
            )
            assert marker["global_step"] == self.global_steps
            if self.fail_eval:
                raise self.fail_eval
            return {"train_eval/complete_trajectories": 4.0}

        def fit(self) -> str:
            self._load_checkpoint()
            self.global_steps = 1
            while self.global_steps <= 4:
                # Match upstream's cached is_last_step before generating.
                is_last_step = self.global_steps >= self.total_training_steps
                if not (self.global_steps == 2 and self.missing_step2):
                    row = {
                        "schema_version": 1, "phase": "advantage_pre_update", "arm": "M2",
                        "run_seed": 42, "policy_step": self.global_steps,
                        "trajectory_count": 8, "episode_group_count": 2,
                        "advancing_row_count": 160,
                        "whitening_scope": "logical_rollout_batch_real_step_rows",
                        "advancing_steps": {
                            "row_count": 160, "nonzero_S_count": int(self.nonzero),
                            "exact_nonzero_S_count": int(self.nonzero),
                            "mean_abs_S": 0.01 if self.nonzero else 0.0,
                        },
                    }
                    with (diagnostics / "advantage-1.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                self._update_critic(None)
                self._update_actor(None)
                if is_last_step or self.global_steps % 4 == 0:
                    self._validate()
                if is_last_step or self.global_steps % 2 == 0:
                    self._save_checkpoint()
                self.global_steps += 1
                if is_last_step:
                    self._finalize(blocking=True)
                    return "completed four rounds"
            raise AssertionError("unreachable")

    assert install_checkpoint_hooks(Trainer)
    assert install_resource_phase_hooks(Trainer, diagnostics_dir=diagnostics, enabled=True)
    return Trainer, tmp_path / "artifacts/learning_restart_gate", diagnostics


def test_no_signal_stops_after_two_real_updates_and_evaluation(setup):
    trainer_class, directory, _ = setup
    assert patch_trainer(trainer_class)
    assert not patch_trainer(trainer_class)
    trainer = trainer_class()
    assert trainer.fit() is None
    assert trainer.events == [
        ("critic", 1), ("actor", 1), ("critic", 2), ("actor", 2),
        ("save", 2), ("evaluate", 2), ("finalize", 2),
    ]
    assert trainer.global_steps == 2 and trainer.total_training_steps == 4
    decision = json.loads((directory / "decision.json").read_text())
    assert decision["target_rounds"] == 2
    assert decision["post_update_checkpoint_saved"] is True
    assert decision["final_evaluation_completed"] is True
    assert decision["effective_nonzero_S_count"] == 0


def test_signal_continues_in_same_fit_and_keeps_normal_final_checkpoint_hooks(setup):
    trainer_class, directory, _ = setup
    assert patch_trainer(trainer_class)
    trainer = trainer_class()
    trainer.nonzero = True
    assert trainer.fit() == "completed four rounds"
    assert [step for event, step in trainer.events if event == "actor"] == [1, 2, 3, 4]
    assert [step for event, step in trainer.events if event == "critic"] == [1, 2, 3, 4]
    assert [step for event, step in trainer.events if event == "save"] == [2, 4]
    assert [step for event, step in trainer.events if event == "evaluate"] == [4]
    decision = json.loads((directory / "decision.json").read_text())
    assert decision["target_rounds"] == 4
    assert decision["final_evaluation_completed"] is False


@pytest.mark.parametrize("boundary", ["save", "eval"])
@pytest.mark.parametrize("error", [RuntimeError("failure"), KeyboardInterrupt()])
def test_errors_propagate_without_success_decision(setup, boundary, error):
    trainer_class, directory, _ = setup
    assert patch_trainer(trainer_class)
    trainer = trainer_class()
    setattr(trainer, f"fail_{boundary}", error)
    with pytest.raises(type(error)) as raised:
        trainer.fit()
    assert raised.value is error
    assert not (directory / "decision.json").exists()
    assert not any(event == "finalize" for event, _ in trainer.events)


def test_missing_diagnostics_is_failure_not_zero_signal(setup):
    trainer_class, directory, _ = setup
    assert patch_trainer(trainer_class)
    trainer = trainer_class()
    trainer.missing_step2 = True
    with pytest.raises(RuntimeError, match="exactly policy steps"):
        trainer.fit()
    assert (directory / "checkpoint_step2.json").exists()
    assert not (directory / "decision.json").exists()
    assert not any(event == "evaluate" for event, _ in trainer.events)


def test_disabled_and_wrong_arm_do_not_install(setup, monkeypatch):
    trainer_class, _, _ = setup
    monkeypatch.delenv(ENABLE_ENV)
    original = trainer_class.fit
    assert not patch_trainer(trainer_class)
    assert trainer_class.fit is original
    monkeypatch.setenv(ENABLE_ENV, ENABLE_VALUE)
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M1")
    with pytest.raises(RuntimeError, match="M2-only"):
        patch_trainer(trainer_class)
    assert trainer_class.fit is original


def test_duplicate_diagnostic_step_is_not_another_complete_batch(setup):
    trainer_class, directory, diagnostics = setup
    (diagnostics / "advantage-stale.jsonl").write_text('{"policy_step": 1}\n')
    assert patch_trainer(trainer_class)
    trainer = trainer_class()
    with pytest.raises(RuntimeError, match="exactly policy steps"):
        trainer.fit()
    assert not (directory / "decision.json").exists()


def test_root_launcher_directory_alone_enables_gate(setup, monkeypatch):
    trainer_class, default_directory, _ = setup
    directory = default_directory.parent / "configured-gate"
    monkeypatch.delenv(ENABLE_ENV)
    monkeypatch.setenv(DIRECTORY_ENV, str(directory))
    assert patch_trainer(trainer_class)
    trainer = trainer_class()
    trainer.fit()
    assert json.loads((directory / "decision.json").read_text())["target_rounds"] == 2
    assert not default_directory.exists()


def test_different_training_schedule_fails_before_optimizer_updates(setup):
    trainer_class, directory, _ = setup
    assert patch_trainer(trainer_class)
    trainer = trainer_class()
    trainer.total_training_steps = 2
    with pytest.raises(RuntimeError, match="fresh four-round training"):
        trainer.fit()
    assert trainer.events == []
    assert not directory.exists()
