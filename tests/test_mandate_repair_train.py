"""Tool masking, parent-only credit, and real CPU LoRA optimizer continuation."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.agents.mandate_repair_train import build_batch, encode_record


def test_rl_checkpointing_policy_has_a_shared_strict_length_boundary():
    from src.agents.mandate_repair_train import request_checkpointing_enabled

    assert not request_checkpointing_enabled(4096)
    assert request_checkpointing_enabled(4097)
    assert request_checkpointing_enabled(5, forced=True)
    assert not request_checkpointing_enabled(8192, training=False)
    assert not request_checkpointing_enabled(8192, threshold=None)


def test_raw_tool_json_masks_observations_and_preserves_eos():
    class Tokenizer:
        eos_token, eos_token_id = "<eos>", 2

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return "\n".join(row["content"] for row in messages) + "\nassistant:"

        def encode(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False)
            return [ord(c) + 3 for c in text.removesuffix("<eos>")] + (
                [2] if text.endswith("<eos>") else []
            )

    record = dict(
        id="task/call2",
        task_id="task",
        messages=[
            dict(role="user", content="repair mandate"),
            dict(role="tool", content="invalid solver input; try again"),
        ],
        completion='{"name":"calculate","arguments":{"value":3}}',
    )
    encoded = encode_record(record, Tokenizer())
    n = encoded["prompt_tokens"]
    assert encoded["labels"][:n] == [-100] * n
    assert encoded["labels"][n:] == encoded["input_ids"][n:]
    assert encoded["labels"][-1] == 2
    with pytest.raises(ValueError, match="no truncation"):
        encode_record(record, Tokenizer(), max_length=5)
    with pytest.raises(ValueError, match="raw tool call"):
        encode_record(dict(record, completion='{"answer":3}'), Tokenizer())


def test_k4_rloo_and_m3_train_only_parent_requests(tmp_path):
    parents, requests = [], []
    for replica, reward in enumerate((0.98, -0.01, -0.02, -0.01)):
        key = f"t/rep{replica}"
        parents.append(
            dict(
                task_id="t",
                trajectory_key=key,
                replicate_index=replica,
                terminal_reward=reward,
                success=replica == 0,
                role="parent",
            )
        )
        for call in range(2):
            requests.append(
                dict(
                    trajectory_key=key,
                    request_key=f"{key}/call{call}",
                    prompt_token_ids=[3, 4],
                    response_token_ids=[5, 2],
                    temperature=1.0,
                    decode_mode="native",
                )
            )
    for name, records in (("trajectory_results.jsonl", parents), ("requests.jsonl", requests)):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    local = tmp_path / "local_credit.jsonl"
    local.write_text(
        json.dumps(
            dict(
                trajectory_key="t/rep0",
                request_key="t/rep0/call0",
                status="COMPLETE",
                continuation_policy="current_policy",
                counterfactual_rewards=[-0.02, -0.01],
            )
        )
        + "\n"
    )
    ordinary, _ = build_batch(tmp_path)
    sham, _ = build_batch(tmp_path, "sham", local)
    m3, summary = build_batch(tmp_path, "m3", local)
    assert len(m3) == 4  # Genuine unsuccessful parents remain in the actor batch.
    assert sum(len(group["requests"]) for group in m3.values()) == 8
    assert summary["counterfactual_tokens_trained"] == 0
    assert summary["local_nonzero_requests"] == 1
    for key in ordinary:
        for a, b in zip(ordinary[key]["requests"], sham[key]["requests"], strict=True):
            assert a["advantage"] == b["advantage"]
    anchor, other = m3["t/rep0"]["requests"]
    assert anchor["raw_local_credit"] == pytest.approx(0.995)
    assert other["raw_local_credit"] == 0
    assert anchor["raw_combined_advantage"] - other["raw_combined_advantage"] == pytest.approx(
        0.25 * 0.995
    )


def test_fp32_lora_and_adam_resume_match_uninterrupted_update(tmp_path):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from src.agents.coldstart_sft import LORA_TARGETS
    from src.agents.mandate_repair_train import (
        optimizer_steps,
        parameters,
        restore_checkpoint,
        save_checkpoint,
        token_logprobs,
    )

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(43)
        model = get_peft_model(
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
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=list(LORA_TARGETS),
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        fresh = copy.deepcopy(model)
        args = SimpleNamespace(
            command="rl",
            base_model=tmp_path / "base",
            seed=43,
            arm="rloo",
            learning_rate=2e-6,
            clip_epsilon=0.2,
            local_weight=0.25,
            device="cpu",
            gradient_checkpointing=True,
        )
        optimizer = torch.optim.AdamW(parameters(model).values(), lr=2e-6, weight_decay=0)
        row = dict(prompt_token_ids=[3, 4, 5], response_token_ids=[6, 7, 2])

        def update(actor, opt):
            opt.zero_grad(set_to_none=True)
            (-token_logprobs(actor, row, "cpu", torch).mean()).backward()
            opt.step()

        update(model, optimizer)
        checkpoint = tmp_path / "state.pt"
        save_checkpoint(
            checkpoint, model, optimizer, args, dict(completed_optimizer_steps=1), torch
        )
        args.gradient_checkpointing = False
        recovered = torch.optim.AdamW(parameters(fresh).values(), lr=2e-6, weight_decay=0)
        restore_checkpoint(checkpoint, fresh, recovered, args, torch)
        assert optimizer_steps(recovered) == (1, 1)
        assert all(p.dtype == torch.float32 for p in parameters(fresh).values())
        update(model, optimizer)
        update(fresh, recovered)
        assert optimizer_steps(recovered) == (2, 2)
        for name, parameter in parameters(model).items():
            torch.testing.assert_close(parameter, parameters(fresh)[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
