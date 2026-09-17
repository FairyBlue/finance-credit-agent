"""Tiny CPU artifacts exercise export precision, coverage and final-save ordering."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config

from src.agents.actor_export import (
    ACTOR_EXPORT_PROFILE,
    ACTOR_RECEIPT,
    ACTOR_SAVE_CONTENT,
    DIAGNOSTIC_PROFILE,
    EXPORT_COMPLETION_MARKER,
    _expected_shapes,
    _hf_evidence,
    _write_weight_shards,
    export_completion_document,
    validate_export_artifact,
)
from src.agents.screening_checkpoint import install_checkpoint_hooks

IDENTITY = "a" * 64


def _small_actor(directory: Path, *, step: int = 8):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    hf = directory / "huggingface"
    hf.mkdir(parents=True)
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, max_position_embeddings=64, tie_word_embeddings=True,
        architectures=["Qwen3ForCausalLM"],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
    source = model.state_dict()
    original = {name: tensor.clone() for name, tensor in source.items()}
    admissions = []
    manifest = _write_weight_shards(
        source, hf, admit=admissions.append, max_shard_bytes=1024,
    )
    export_config = copy.deepcopy(config)
    export_config.torch_dtype = torch.bfloat16
    export_config.save_pretrained(hf)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"hold": 0, "[UNK]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]", chat_template="{{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(hf)
    evidence = _hf_evidence(hf, verify_hashes=True)
    assert evidence["tensors"] == manifest
    assert all(torch.equal(source[name], tensor) for name, tensor in original.items())
    assert {name: list(t.shape) for name, t in source.items()} == _expected_shapes(config)
    receipt = {
        "schema_version": 1, "artifact_profile": ACTOR_EXPORT_PROFILE,
        "identity_sha256": IDENTITY, "global_step": step, "world_size": 1,
        "training_dtype": "float32", "export_dtype": "bfloat16",
        "actor_included": True, "critic_included": False,
        "optimizer_included": False, "resume_supported": False,
        "lossy_fp32_to_bf16": True, "source_copy_readback_verified": True,
        "training_storage_version_preserved": True, "hf": evidence,
    }
    (directory / ACTOR_RECEIPT).write_text(json.dumps(receipt))
    return source, admissions


def _completed_actor(tmp_path):
    directory = tmp_path / "global_step_8"
    source, admissions = _small_actor(directory / "actor")
    marker = export_completion_document(
        directory, profile=ACTOR_EXPORT_PROFILE, identity=IDENTITY, step=8,
    )
    (directory / EXPORT_COMPLETION_MARKER).write_text(json.dumps(marker))
    return directory, source, admissions


def test_streamed_actor_loads_exact_bf16_values_and_tokenizer_without_changing_fp32(tmp_path):
    directory, source, admissions = _completed_actor(tmp_path)
    marker = validate_export_artifact(directory, identity_sha256=IDENTITY)
    assert marker["actor_included"] is True and marker["critic_included"] is False
    hf = directory / "actor/huggingface"
    loaded, info = AutoModelForCausalLM.from_pretrained(
        hf, torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True,
    )
    assert not any(info[key] for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    ))
    assert all(
        torch.equal(value, source[name].bfloat16()) for name, value in loaded.state_dict().items()
    )
    assert all(value.dtype == torch.float32 for value in source.values())
    assert AutoTokenizer.from_pretrained(hf, local_files_only=True).encode("hold") == [0]
    assert len(list(hf.glob("*.safetensors"))) > 1
    assert admissions
    assert not list(directory.rglob("*.pt"))


def test_same_size_payload_corruption_is_detected_by_full_hash_readback(tmp_path):
    directory, _, _ = _completed_actor(tmp_path)
    shard = next((directory / "actor/huggingface").glob("*.safetensors"))
    content = bytearray(shard.read_bytes())
    content[-1] ^= 1
    shard.write_bytes(content)
    # The explicitly cheap mode is only a header/metadata check.
    validate_export_artifact(directory, identity_sha256=IDENTITY, verify_hashes=False)
    with pytest.raises(RuntimeError, match="contents changed"):
        validate_export_artifact(directory, identity_sha256=IDENTITY)


@pytest.mark.parametrize("damage", ["metadata", "index", "missing", "training_state", "identity"])
def test_export_rejects_metadata_coverage_state_and_identity_damage(tmp_path, damage):
    directory, _, _ = _completed_actor(tmp_path)
    hf = directory / "actor/huggingface"
    identity = IDENTITY
    if damage == "metadata":
        (hf / "chat_template.jinja").write_text("changed")
    elif damage == "index":
        path = hf / "model.safetensors.index.json"
        index = json.loads(path.read_text())
        index["weight_map"].pop(next(iter(index["weight_map"])))
        path.write_text(json.dumps(index))
    elif damage == "missing":
        next(hf.glob("*.safetensors")).unlink()
    elif damage == "training_state":
        (directory / "data.pt").write_bytes(b"unrequested")
    else:
        identity = "b" * 64
    with pytest.raises(RuntimeError):
        validate_export_artifact(directory, identity_sha256=identity)


def test_shard_bound_and_nonfinite_conversion_fail_before_a_receipt(tmp_path):
    with pytest.raises(RuntimeError, match="shard bound"):
        _write_weight_shards({"large": torch.ones(5)}, tmp_path, admit=lambda n: None,
                             max_shard_bytes=8)
    with pytest.raises(RuntimeError, match="nonfinite"):
        _write_weight_shards({"nan": torch.tensor([float("nan")])}, tmp_path,
                             admit=lambda n: None)
    assert not list(tmp_path.iterdir())


@pytest.fixture
def export_trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", IDENTITY)

    class Trainer:
        def __init__(self):
            self.global_steps = self.total_training_steps = 8
            self.use_critic = True
            self.events = []
            self.fail_validation = False
            self.actor_rollout_wg = SimpleNamespace(save_checkpoint=self.actor_save)
            self.critic_wg = object()
            self.config = OmegaConf.create({
                "actor_rollout_ref": {"actor": {"checkpoint": {
                    "save_contents": [ACTOR_SAVE_CONTENT], "load_contents": [], "async_save": False,
                }}},
                "critic": {"checkpoint": {
                    "save_contents": [], "load_contents": [], "async_save": False,
                }},
                "trainer": {
                    "finance_artifact_profile": ACTOR_EXPORT_PROFILE,
                    "default_local_dir": str(tmp_path), "default_hdfs_dir": None,
                    "resume_mode": "disable", "resume_from_path": None,
                    "n_gpus_per_node": 1, "nnodes": 1,
                },
            })

        def actor_save(self, path, remote, step, max_ckpt_to_keep):
            assert remote is None and max_ckpt_to_keep is None
            self.events.append("actor_saved")
            _small_actor(Path(path), step=step)
            return "actor_receipt"

        def _save_checkpoint(self):
            raise AssertionError("must not save actor/critic/optimizer training state")

        def _load_checkpoint(self):
            self.events.append("fresh_start")

        def _validate(self):
            self.events.append("evaluation")
            validate_export_artifact(tmp_path / f"global_step_{self.global_steps}",
                                     identity_sha256=IDENTITY)
            if self.fail_validation:
                raise RuntimeError("final evaluation failed")
            return "evaluated"

    install_checkpoint_hooks(Trainer)
    return Trainer()


def test_final_export_precedes_evaluation_and_later_save_is_deduplicated(export_trainer):
    trainer = export_trainer
    assert trainer._validate() == "evaluated"
    assert trainer._save_checkpoint() == "actor_receipt"
    assert trainer.events == ["actor_saved", "evaluation"]


def test_failed_final_evaluation_retains_verified_actor_and_forbids_resume(export_trainer):
    trainer = export_trainer
    trainer.fail_validation = True
    with pytest.raises(RuntimeError, match="evaluation failed"):
        trainer._validate()
    directory = Path(trainer.config.trainer.default_local_dir) / "global_step_8"
    validate_export_artifact(directory, identity_sha256=IDENTITY)
    trainer.config.trainer.resume_mode = "resume_path"
    trainer.config.trainer.resume_from_path = str(directory)
    with pytest.raises(RuntimeError, match="fresh-only"):
        trainer._load_checkpoint()


def test_critic_diagnostic_saves_only_receipt_for_two_updates(export_trainer):
    trainer = export_trainer
    trainer.config.trainer.finance_artifact_profile = DIAGNOSTIC_PROFILE
    trainer.config.actor_rollout_ref.actor.checkpoint.save_contents = []
    trainer.global_steps = trainer.total_training_steps = 2
    trainer._validate()
    trainer._save_checkpoint()
    assert trainer.events == ["evaluation"]
    directory = Path(trainer.config.trainer.default_local_dir) / "global_step_2"
    assert [p.name for p in directory.iterdir()] == [EXPORT_COMPLETION_MARKER]
    marker = validate_export_artifact(directory, identity_sha256=IDENTITY)
    assert marker["diagnostic_only"] is True
    assert marker["model_included"] is False and marker["resume_supported"] is False


def test_diagnostic_budget_and_intermediate_actor_saves_are_rejected(export_trainer):
    trainer = export_trainer
    trainer.global_steps = 1
    with pytest.raises(RuntimeError, match="final-update only"):
        trainer._save_checkpoint()
    trainer.config.trainer.finance_artifact_profile = DIAGNOSTIC_PROFILE
    trainer.config.actor_rollout_ref.actor.checkpoint.save_contents = []
    with pytest.raises(RuntimeError, match="exactly two"):
        trainer._save_checkpoint()
    assert trainer.events == []
