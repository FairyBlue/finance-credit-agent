"""Completion-only LoRA SFT for the four-action portfolio teacher demo.

The existing coldstart trainer supplies the fixed-budget optimizer loop. This
encoder teaches action IDs, without the old tool-call wrapper. Microbatch one
permits computing vocabulary logits only for the completion and its preceding
position while preserving attention over the entire point-in-time prompt.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from collections import Counter
from pathlib import Path

from src.agents.coldstart_sft import (
    IGNORE_INDEX,
    LORA_TARGETS,
    digest,
    now,
    train_model,
    write_json,
)
from src.agents.portfolio_management import ACTIONS

PROFILE = "portfolio_teacher_completion_sft_v1"


def encode_record(record, tokenizer, max_length=8192):
    """Match the shared inference chat prefix and supervise only action + EOS."""
    identifier = record.get("state_id")
    messages = record.get("messages")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("teacher record needs a nonempty state_id")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [m.get("role") for m in messages] != ["system", "user", "assistant"]
        or messages[-1].get("content") not in ACTIONS
        or messages[-1]["content"] != record.get("teacher_action")
    ):
        raise ValueError(f"{identifier}: expected system/user prefix and one assistant action ID")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer EOS is required")
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, enable_thinking=False, add_generation_prompt=True
    )
    completion = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    targets = [*completion, tokenizer.eos_token_id]
    if not prefix or not completion or len(prefix) + len(targets) > max_length:
        raise ValueError(f"{identifier}: empty or overlong teacher record; no truncation")
    return {
        "id": identifier,
        "kind": "portfolio_action",
        "action_id": messages[-1]["content"],
        "input_ids": [*prefix, *targets],
        "labels": [IGNORE_INDEX] * len(prefix) + targets,
        "prompt_tokens": len(prefix),
        "target_tokens": len(targets),
    }


def read_examples(path, tokenizer, max_length=8192):
    examples, seen = [], set()
    with Path(path).open() as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank teacher JSONL row {index}")
            example = encode_record(json.loads(line), tokenizer, max_length)
            if example["id"] in seen:
                raise ValueError("duplicate supervised teacher state")
            seen.add(example["id"])
            examples.append(example)
    if not examples:
        raise ValueError("nonempty teacher dataset required")
    return examples


def completion_logits_model(model):
    """Thin nn.Module preserving full attention and shifted completion loss."""
    import torch

    class CompletionLogitsModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask, labels, **kwargs):
            if labels.ndim != 2 or labels.shape[0] != 1:
                raise ValueError("completion-logits optimization requires microbatch one")
            count = int((labels != IGNORE_INDEX).sum().item())
            if (
                count == 0
                or count >= labels.shape[1]
                or bool((labels[:, -count:] == IGNORE_INDEX).any().item())
                or bool((labels[:, :-count] != IGNORE_INDEX).any().item())
            ):
                raise ValueError("expected contiguous completion-only labels at the sequence tail")
            # The preceding prompt position predicts the first action token.
            # The final EOS position is retained too, then masked by HF's normal
            # one-position causal shift. Full input/mask is never truncated.
            return self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels[:, -(count + 1) :],
                logits_to_keep=count + 1,
                **kwargs,
            )

        def save_pretrained(self, *args, **kwargs):
            return self.base.save_pretrained(*args, **kwargs)

    return CompletionLogitsModel(model)


def train(args):
    import importlib.metadata

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.seed != 43 or args.micro_batch != 1:
        raise ValueError("the fixed teacher demo uses seed43 and microbatch one")
    args.output.mkdir(parents=True, exist_ok=False)
    started = now()
    write_json(args.output / "state.json", dict(status="PREPARING", started_at_SGT=started))
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        examples = read_examples(args.data, tokenizer, args.max_length)
        label_counts = dict(Counter(example["action_id"] for example in examples))
        preflight = dict(
            status="PASSED",
            examples=len(examples),
            action_counts=label_counts,
            max_tokens=max(len(e["input_ids"]) for e in examples),
            min_prompt_tokens=min(e["prompt_tokens"] for e in examples),
            max_prompt_tokens=max(e["prompt_tokens"] for e in examples),
            min_target_tokens=min(e["target_tokens"] for e in examples),
            max_target_tokens=max(e["target_tokens"] for e in examples),
            total_prompt_tokens=sum(e["prompt_tokens"] for e in examples),
            total_target_tokens=sum(e["target_tokens"] for e in examples),
            supervision="assistant action ID plus EOS only; all prompt labels -100",
            prefix="shared chat template, enable_thinking=False, add_generation_prompt=True",
            truncated_examples=0,
        )
        settings = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        settings.update(
            profile=PROFILE,
            started_at_SGT=started,
            data_sha256=digest(args.data),
            implementation_sha256=digest(Path(__file__)),
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            lora_target_modules=LORA_TARGETS,
            attention="sdpa",
            gradient_checkpointing=True,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            schedule="constant",
            full_prompt_attention=True,
            completion_logits_only=not args.full_logits,
            versions={
                name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
            },
        )
        write_json(args.output / "settings.json", settings)
        write_json(args.output / "data_preflight.json", preflight)
        print(json.dumps(preflight), flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if (
            not args.full_logits
            and "logits_to_keep" not in inspect.signature(model.forward).parameters
        ):
            raise ValueError("this model does not support completion-only vocabulary logits")
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=list(LORA_TARGETS),
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(args.device)
        if not args.full_logits:
            model = completion_logits_model(model)
        write_json(args.output / "state.json", dict(status="TRAINING", started_at_SGT=started))
        summary = train_model(
            model,
            examples,
            output=args.output,
            pad_token_id=tokenizer.pad_token_id,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            micro_batch=args.micro_batch,
            gradient_accumulation=args.gradient_accumulation,
            lr=args.lr,
        )
        tokenizer.save_pretrained(args.output / "adapter")
        summary.update(
            status="COMPLETE",
            profile=PROFILE,
            started_at_SGT=started,
            completed_at_SGT=now(),
            adapter=str(args.output / "adapter"),
            data_sha256=settings["data_sha256"],
            action_counts=label_counts,
            completion_logits_only=not args.full_logits,
            full_prompt_attention=True,
            encoding_stats=preflight,
        )
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "state.json", summary)
    except BaseException as error:
        write_json(
            args.output / "state.json",
            dict(
                status="FAILED",
                started_at_SGT=started,
                failed_at_SGT=now(),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--full-logits", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
