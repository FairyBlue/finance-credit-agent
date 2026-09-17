"""Real tiny Qwen3 training/export and completion-mask contract checks on CPU."""

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from src.agents.coldstart_sft import (
    IGNORE_INDEX,
    collate_examples,
    encode_record,
    main,
    read_examples,
)


@pytest.fixture
def tokenizer():
    vocab = {
        token: i
        for i, token in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "prompt",
                "long",
                "state",
                "name",
                "arguments",
                "hold",
                "buy",
                "sell",
                "symbol",
                "V",
                "target_weight",
                "0",
                "1",
                "2",
                "3",
                "<tool_call>",
                "</tool_call>",
                "{",
                "}",
                '"',
                ":",
                ",",
                ".",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=["<tool_call>", "</tool_call>"],
    )


def demonstration(index=0, *, long=False):
    return {
        "id": f"demo-{index}",
        "kind": "decision",
        "prompt": "prompt long state " if long else "prompt ",
        "completion": '<tool_call>{"name":"hold","arguments":{}}</tool_call>',
        "metadata": {"source_split": "train"},
    }


def test_completion_mask_padding_eos_and_no_truncation(tokenizer):
    examples = [
        encode_record(demonstration(), tokenizer, 128),
        encode_record(demonstration(1, long=True), tokenizer, 128),
    ]
    batch = collate_examples(examples, tokenizer.pad_token_id)
    for index, example in enumerate(examples):
        length = len(example["input_ids"])
        assert batch["labels"][index, : example["prompt_tokens"]].eq(IGNORE_INDEX).all()
        assert batch["labels"][index, length - 1] == tokenizer.eos_token_id
        assert batch["labels"][index, length:].eq(IGNORE_INDEX).all()
        assert batch["attention_mask"][index, length:].eq(0).all()
        assert (batch["labels"][index, 1:] != IGNORE_INDEX).sum() == example["target_tokens"]
    with pytest.raises(ValueError, match="no truncation"):
        encode_record(demonstration(), tokenizer, 2)
    invalid = demonstration()
    invalid["completion"] = ""
    with pytest.raises(ValueError, match="empty completion"):
        encode_record(invalid, tokenizer, 128)


def test_reject_prefix_retokenization_and_duplicate_ids(tokenizer, tmp_path):
    class MergingTokenizer:
        eos_token = "[EOS]"
        eos_token_id = 2

        def encode(self, value, **_kwargs):
            return [3] if value == "prompt " else [4, 5, 2]

    with pytest.raises(ValueError, match="prompt prefix"):
        encode_record(demonstration(), MergingTokenizer(), 128)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join([json.dumps(demonstration())] * 2) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_examples(data, tokenizer, 128)


def test_cpu_qwen_lora_train_save_load_and_merge(tokenizer, tmp_path: Path):
    from peft import PeftModel

    # The production loop runs unchanged; only the model and device are tiny.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(17)
        base = tmp_path / "base"
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                bos_token_id=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                tie_word_embeddings=True,
            )
        )
        model.save_pretrained(base)
        tokenizer.save_pretrained(base)
        records = [demonstration(i, long=i == 1) for i in range(3)]
        data = tmp_path / "data.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in records))
        output = tmp_path / "sft"
        main(
            [
                "train",
                "--model",
                str(base),
                "--data",
                str(data),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--micro-batch",
                "2",
                "--gradient-accumulation",
                "2",
                "--max-length",
                "128",
            ]
        )
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "COMPLETE"
        assert summary["optimizer_steps"] == summary["probe_changed_steps"] == 1
        assert summary["example_presentations"] == 3
        assert summary["trainable_sha256_before"] != summary["trainable_sha256_after"]
        steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        assert steps[0]["examples"] == 3
        assert set(steps[0]["example_ids"]) == {row["id"] for row in records}
        assert steps[0]["grad_norm_before_clip"] > 0
        assert steps[0]["probe_max_abs_delta"] > 0
        # Local artifact generated above, not an untrusted checkpoint.
        checkpoint = torch.load(output / "optimizer_rng.pt", map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 1
        assert checkpoint["optimizer"]["state"]
        assert checkpoint["rng"]["torch_cpu"].numel() > 0
        adapter = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base), output / "adapter"
        ).eval()
        examples = [encode_record(records[0], tokenizer, 128)]
        batch = collate_examples(examples, tokenizer.pad_token_id)
        with torch.no_grad():
            result = adapter(**batch)
            expected_loss = torch.nn.functional.cross_entropy(
                result.logits[:, :-1].float().reshape(-1, len(tokenizer)),
                batch["labels"][:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        torch.testing.assert_close(result.loss, expected_loss)
        merged = tmp_path / "merged"
        main(
            [
                "merge",
                "--model",
                str(base),
                "--adapter",
                str(output / "adapter"),
                "--output",
                str(merged),
            ]
        )
        receipt = json.loads((merged / "merge_receipt.json").read_text())
        assert receipt["status"] == "COMPLETE" and receipt["device"] == "cpu"
        loaded = AutoModelForCausalLM.from_pretrained(merged).eval()
        assert not any("lora_" in name for name, _ in loaded.named_parameters())
        with torch.no_grad():
            merged_logits = loaded(**batch).logits
        torch.testing.assert_close(
            merged_logits.float(), result.logits.float(), atol=0.004, rtol=0.02
        )
        with pytest.raises(FileExistsError):
            main(
                [
                    "merge",
                    "--model",
                    str(base),
                    "--adapter",
                    str(output / "adapter"),
                    "--output",
                    str(merged),
                ]
            )
    finally:
        torch.set_num_threads(old_threads)
