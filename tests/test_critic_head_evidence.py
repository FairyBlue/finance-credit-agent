"""Record the actual tiny Qwen value head before the worker wraps the model."""

import json

import pytest
import torch
from test_critic_initialization import settings
from test_critic_initialization import tiny_causal_checkpoint as checkpoint_fixture
from transformers import AutoModelForTokenClassification

from src.agents import critic_initialization as candidate

tiny_model = checkpoint_fixture


def loader_and_worker(monkeypatch):
    from verl.utils import model as model_utils

    def cpu_loader(local_path, dtype, model_config, trust_remote_code):
        return AutoModelForTokenClassification.from_pretrained(
            local_path,
            num_labels=1,
            classifier_dropout=0.0,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()

    monkeypatch.setattr(model_utils, "load_valuehead_model", cpu_loader)

    class Worker:
        rank = 0

        def _build_critic_model_optimizer(self, config):
            from verl.utils.model import load_valuehead_model

            module = load_valuehead_model(config.model.path, torch.float32, None, False)
            # This is the exact loader-return boundary before the production
            # builder's apply_monkey_patch/.to/FSDP operations.
            return module

    candidate.patch_critic_initialization(Worker, mode="cpu_seed_v1")
    return model_utils, cpu_loader, Worker


def test_opt_in_records_actual_same_head_preserves_backbone_and_rng(
    tiny_model,
    tmp_path,
    monkeypatch,
):
    path, backbone = tiny_model
    model_utils, loader, Worker = loader_and_worker(monkeypatch)
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "a" * 64)
    hashes = []
    modules = []
    for arm, seed, ambient in (("M1", 42, 11), ("M2", 42, 99), ("M2", 43, 17)):
        directory = tmp_path / f"{arm}-{seed}"
        monkeypatch.setenv(candidate.HEAD_EVIDENCE_ENV, str(directory))
        monkeypatch.setenv("FINANCE_SCREENING_ARM", arm)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(ambient)
            before = torch.get_rng_state().clone()
            module = Worker()._build_critic_model_optimizer(settings(path, seed))
            assert torch.equal(before, torch.get_rng_state())
        record = json.loads((directory / "head-rank0.json").read_text())
        assert record["phase"] == "critic_head_after_hf_load_before_fsdp"
        assert record["seed"] == seed and record["arm"] == arm
        assert record["mode"] == "cpu_seed_v1"
        assert record["execution_identity_sha256"] == "a" * 64
        assert record["model_path"] == str(path.resolve())
        observed = {row["name"]: row for row in record["parameters"]}
        assert observed["weight"]["shape"] == list(module.score.weight.shape)
        assert observed["weight"]["dtype"] == str(module.score.weight.dtype)
        assert sum(row["numel"] for row in record["parameters"]) == sum(
            p.numel() for p in module.score.parameters()
        )
        assert all(torch.equal(backbone[k], v) for k, v in module.model.state_dict().items())
        assert model_utils.load_valuehead_model is loader
        hashes.append(record["head_sha256"])
        modules.append(module)
    assert hashes[0] == hashes[1] and hashes[2] != hashes[0]
    assert torch.equal(modules[0].score.weight, modules[1].score.weight)


def test_recording_does_not_change_head_and_duplicate_failure_restores_loader_rng(
    tiny_model,
    tmp_path,
    monkeypatch,
):
    path, _ = tiny_model
    model_utils, loader, Worker = loader_and_worker(monkeypatch)
    monkeypatch.setenv("FINANCE_SCREENING_IDENTITY_SHA256", "a" * 64)
    monkeypatch.setenv("FINANCE_SCREENING_ARM", "M1")
    monkeypatch.delenv(candidate.HEAD_EVIDENCE_ENV, raising=False)
    plain = Worker()._build_critic_model_optimizer(settings(path))
    monkeypatch.setenv(candidate.HEAD_EVIDENCE_ENV, str(tmp_path / "evidence"))
    recorded = Worker()._build_critic_model_optimizer(settings(path))
    assert all(
        torch.equal(value, recorded.state_dict()[name])
        for name, value in plain.state_dict().items()
    )
    before = torch.get_rng_state().clone()
    with pytest.raises(FileExistsError):
        Worker()._build_critic_model_optimizer(settings(path))
    assert torch.equal(before, torch.get_rng_state())
    assert model_utils.load_valuehead_model is loader
