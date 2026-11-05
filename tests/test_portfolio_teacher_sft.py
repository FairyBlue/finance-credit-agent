"""Completion masks and loss/LoRA-gradient equivalence on a tiny CPU Qwen3."""

import copy
import json

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.agents.coldstart_sft import LORA_TARGETS, train_model
from src.agents.portfolio_teacher_sft import completion_logits_model, encode_record


def test_action_only_encoder_preserves_chat_prefix_and_eos():
    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            assert [m["role"] for m in messages] == ["system", "user"]
            assert kwargs == dict(tokenize=True, enable_thinking=False, add_generation_prompt=True)
            return [3, 4, 5]

        def encode(self, action, **kwargs):
            assert action == "defensive"
            assert kwargs == dict(add_special_tokens=False)
            return [6, 7]

    row = dict(
        state_id="state-1",
        teacher_action="defensive",
        messages=[
            dict(role="system", content="choose"),
            dict(role="user", content="visible state"),
            dict(role="assistant", content="defensive"),
        ],
    )
    encoded = encode_record(row, Tokenizer())
    assert encoded["input_ids"] == [3, 4, 5, 6, 7, 2]
    assert encoded["labels"] == [-100, -100, -100, 6, 7, 2]
    assert encoded["target_tokens"] == 3
    with pytest.raises(ValueError, match="no truncation"):
        encode_record(row, Tokenizer(), max_length=5)


def test_completion_logits_equal_full_loss_gradients_and_old_training_loop(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(43)
        full = get_peft_model(
            Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=32,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=16,
                    eos_token_id=2,
                    pad_token_id=0,
                    attention_dropout=0.0,
                )
            ),
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=list(LORA_TARGETS),
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        # Nonzero B tests A and B gradients rather than the initial zero-A path.
        with torch.no_grad():
            for name, parameter in full.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(mean=0.0, std=0.01)
        compact = copy.deepcopy(full)
        wrapped = completion_logits_model(compact)
        ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 2]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, 8, 9, 2]])
        arguments = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
        full_result = full(**arguments, use_cache=False)
        compact_result = wrapped(**arguments, use_cache=False)
        assert full_result.logits.shape[1] == 8
        assert compact_result.logits.shape[1] == 4
        torch.testing.assert_close(full_result.loss, compact_result.loss, rtol=1e-6, atol=1e-7)
        full_result.loss.backward()
        compact_result.loss.backward()
        full_parameters = dict(full.named_parameters())
        checked = 0
        for name, parameter in compact.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad, full_parameters[name].grad, rtol=1e-5, atol=1e-7
                )
                checked += 1
        assert checked > 0
        examples = [
            dict(
                id=f"teacher-{i}",
                kind="portfolio_action",
                input_ids=ids[0].tolist(),
                labels=labels[0].tolist(),
                prompt_tokens=5,
                target_tokens=3,
            )
            for i in range(3)
        ]
        summary = train_model(
            wrapped,
            examples,
            output=tmp_path,
            pad_token_id=0,
            device="cpu",
            seed=43,
            epochs=1,
            micro_batch=1,
            gradient_accumulation=16,
            lr=5e-5,
        )
        assert summary["optimizer_steps"] == summary["expected_optimizer_steps"] == 1
        assert summary["probe_changed_steps"] == 1
        assert (tmp_path / "adapter/adapter_model.safetensors").is_file()
        assert json.loads((tmp_path / "steps.jsonl").read_text())["examples"] == 3
    finally:
        torch.set_num_threads(previous_threads)
