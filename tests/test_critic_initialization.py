"""CPU-only reproduction of a causal checkpoint's missing Qwen3 score head."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForTokenClassification, Qwen3Config, Qwen3ForCausalLM


def load_candidate():
    candidate = os.environ.get("FINANCE_CRITIC_CANDIDATE")
    if candidate:
        spec = importlib.util.spec_from_file_location("critic_init_candidate", candidate)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from src.agents import critic_initialization

    return critic_initialization


PATCH = load_candidate()


def settings(path: Path, seed=42):
    return SimpleNamespace(model=SimpleNamespace(path=path, fsdp_config={"seed": seed}))


@pytest.fixture
def tiny_causal_checkpoint(tmp_path: Path):
    # Same architecture and missing-head boundary as the H100 run, tiny CPU sizes.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=32, tie_word_embeddings=True,
            classifier_dropout=0.0, num_labels=1,
        ))
        model.save_pretrained(tmp_path)
    return tmp_path, {k: v.detach().clone() for k, v in model.model.state_dict().items()}


def worker_class():
    class Critic:
        def _build_critic_model_optimizer(self, config):
            return AutoModelForTokenClassification.from_pretrained(
                config.model.path, num_labels=1, classifier_dropout=0.0,
                torch_dtype=torch.float32, attn_implementation="eager",
                local_files_only=True,
            ).eval()

    return Critic


def test_missing_head_depends_on_ambient_rng_without_adapter(tiny_causal_checkpoint):
    path, _ = tiny_causal_checkpoint
    worker = worker_class()()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11)
        first = worker._build_critic_model_optimizer(settings(path))
        torch.random.default_generator.manual_seed(99)
        second = worker._build_critic_model_optimizer(settings(path))
    assert not torch.equal(first.score.weight, second.score.weight)
    assert torch.equal(first.score.bias, second.score.bias)


def test_seeded_missing_head_is_equal_without_changing_backbone_or_caller_rng(
    tiny_causal_checkpoint,
):
    path, backbone = tiny_causal_checkpoint
    worker_type = worker_class()
    assert PATCH.patch_critic_initialization(worker_type, mode=PATCH.MODE)
    assert not PATCH.patch_critic_initialization(worker_type, mode=PATCH.MODE)
    worker = worker_type()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(11)
        before = torch.get_rng_state().clone()
        first = worker._build_critic_model_optimizer(settings(path))
        assert torch.equal(before, torch.get_rng_state())
        torch.random.default_generator.manual_seed(99)
        before = torch.get_rng_state().clone()
        second = worker._build_critic_model_optimizer(settings(path))
        assert torch.equal(before, torch.get_rng_state())
        different = worker._build_critic_model_optimizer(settings(path, 43))
    assert torch.equal(first.score.weight, second.score.weight)
    assert torch.equal(first.score.bias, second.score.bias)
    assert not torch.equal(first.score.weight, different.score.weight)
    for model in (first, second, different):
        assert all(torch.equal(backbone[k], v) for k, v in model.model.state_dict().items())
    ids = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        assert torch.equal(first(ids).logits, second(ids).logits)


def test_loader_failure_restores_caller_rng():
    class BrokenCritic:
        def _build_critic_model_optimizer(self, config):
            torch.rand(32)
            raise RuntimeError("fixture load failure")

    PATCH.patch_critic_initialization(BrokenCritic, mode=PATCH.MODE)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="fixture load failure"):
        BrokenCritic()._build_critic_model_optimizer(settings(Path("unused")))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("seed", [None, True, -1, "42", 2**63])
def test_invalid_initialization_seed_fails_before_loading(seed):
    worker_type = worker_class()
    PATCH.patch_critic_initialization(worker_type, mode=PATCH.MODE)
    with pytest.raises(ValueError, match="fsdp_config.seed"):
        worker_type()._build_critic_model_optimizer(settings(Path("does-not-exist"), seed))
