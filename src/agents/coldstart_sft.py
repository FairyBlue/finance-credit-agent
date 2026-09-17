"""Bounded, completion-only LoRA cold start and separate CPU HF model export.

This module does not choose trading actions or alter the environment. Its JSONL
input contains production-rendered prompts and explicitly labelled demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
IGNORE_INDEX = -100


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def encode_record(record: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Check the exact rendered prefix and mask every prompt token, never truncate."""
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("each demonstration needs a nonempty string id")
    if record.get("kind") not in {"decision", "api"}:
        raise ValueError(f"{record['id']}: unsupported demonstration kind")
    if not isinstance(record.get("metadata"), dict):
        raise ValueError(f"{record['id']}: metadata must be an object")
    prompt, completion = record.get("prompt"), record.get("completion")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{record['id']}: empty prompt")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError(f"{record['id']}: empty completion")
    # Only the one tool-call assistant response is taught, including its EOS.
    if not completion.startswith("<tool_call>") or not completion.endswith("</tool_call>"):
        raise ValueError(f"{record['id']}: completion must be a single tool_call")
    try:
        call = json.loads(completion[len("<tool_call>") : -len("</tool_call>")])
    except json.JSONDecodeError as error:
        raise ValueError(f"{record['id']}: invalid tool-call JSON") from error
    if (
        not isinstance(call, dict)
        or set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
        or not isinstance(call["arguments"], dict)
    ):
        raise ValueError(f"{record['id']}: invalid tool-call structure")
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("tokenizer must define EOS")
    if tokenizer.eos_token in completion:
        raise ValueError(f"{record['id']}: completion must not supply its own EOS")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion + tokenizer.eos_token, add_special_tokens=False)
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{record['id']}: tokenized completion changes the production prompt prefix"
        )
    target_ids = full_ids[len(prompt_ids) :]
    if len(target_ids) < 2 or target_ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f"{record['id']}: empty target or missing EOS")
    if len(full_ids) > max_length:
        raise ValueError(
            f"{record['id']}: {len(full_ids)} tokens exceeds {max_length}; no truncation"
        )
    return {
        "id": record["id"],
        "kind": record["kind"],
        "input_ids": full_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
    }


def read_examples(path: Path, tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    examples = []
    seen = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_no}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            example = encode_record(record, tokenizer, max_length)
            if example["id"] in seen:
                raise ValueError(f"duplicate demonstration id: {example['id']}")
            seen.add(example["id"])
            examples.append(example)
    if not examples:
        raise ValueError("empty demonstration dataset")
    return examples


def collate_examples(examples: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Right padding is excluded from both attention and supervision."""
    import torch

    length = max(len(example["input_ids"]) for example in examples)
    ids, labels, masks = [], [], []
    for example in examples:
        padding = length - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_token_id] * padding)
        labels.append(example["labels"] + [IGNORE_INDEX] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
    return {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in (("input_ids", ids), ("labels", labels), ("attention_mask", masks))
    }


def trainable_hash(model: Any) -> str:
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_model(
    model: Any,
    examples: list[dict[str, Any]],
    *,
    output: Path,
    pad_token_id: int,
    device: str,
    seed: int,
    epochs: int,
    micro_batch: int,
    gradient_accumulation: int,
    lr: float,
) -> dict[str, Any]:
    """Run a fixed epoch budget, with token-weighted gradient accumulation."""
    import numpy as np
    import torch

    if min(epochs, micro_batch, gradient_accumulation) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch sizes, and learning rate must be positive")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable adapter parameters")
    probe_name, probe_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    )
    before_hash = trainable_hash(model)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    effective_batch = micro_batch * gradient_accumulation
    expected_steps = epochs * math.ceil(len(examples) / effective_batch)
    step = 0
    changed_steps = 0
    model.train()
    started = time.monotonic()
    with (output / "steps.jsonl").open("x", buffering=1) as log:
        for epoch in range(1, epochs + 1):
            order = torch.randperm(len(examples), generator=sampler).tolist()
            for start in range(0, len(order), effective_batch):
                indexes = order[start : start + effective_batch]
                group = [examples[index] for index in indexes]
                target_count = sum(example["target_tokens"] for example in group)
                optimizer.zero_grad(set_to_none=True)
                probe_before = probe_parameter.detach().clone()
                mean_loss = 0.0
                for offset in range(0, len(group), micro_batch):
                    micro = group[offset : offset + micro_batch]
                    batch = {
                        key: value.to(device)
                        for key, value in collate_examples(micro, pad_token_id).items()
                    }
                    # The first input token is always masked; every target is in the
                    # shifted labels used by HF's causal-LM loss.
                    count = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())
                    if count != sum(example["target_tokens"] for example in micro):
                        raise RuntimeError("supervised-token accounting differs after collation")
                    result = model(**batch, use_cache=False)
                    loss = result.loss
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"nonfinite loss before update {step + 1}")
                    weighted_loss = loss * (count / target_count)
                    mean_loss += float(weighted_loss.detach())
                    weighted_loss.backward()
                    del result, loss, weighted_loss, batch
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"nonfinite gradient before update {step + 1}")
                optimizer.step()
                delta = float((probe_parameter.detach() - probe_before).abs().max())
                if not math.isfinite(delta):
                    raise RuntimeError("nonfinite adapter parameter delta")
                del probe_before
                step += 1
                changed_steps += int(delta > 0)
                record = {
                    "at_SGT": now(),
                    "step": step,
                    "epoch": epoch,
                    "example_ids": [example["id"] for example in group],
                    "examples": len(group),
                    "supervised_tokens": target_count,
                    "loss": mean_loss,
                    "grad_norm_before_clip": grad_norm,
                    "lr": lr,
                    "probe_parameter": probe_name,
                    "probe_max_abs_delta": delta,
                    "adapter_update_detected": delta > 0,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if device.startswith("cuda"):
                    record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                log.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if key not in {"example_ids", "probe_parameter"}
                        }
                    ),
                    flush=True,
                )
    if step != expected_steps:
        raise RuntimeError("completed update count differs from fixed epoch budget")
    after_hash = trainable_hash(model)
    if before_hash == after_hash or changed_steps == 0:
        raise RuntimeError("no actual adapter update detected")
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    rng = {
        "torch_cpu": torch.get_rng_state(),
        "sampler": sampler.get_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
    }
    torch.save(
        {"optimizer": optimizer.state_dict(), "rng": rng, "step": step, "epoch": epochs},
        output / "optimizer_rng.pt",
    )
    return {
        "optimizer_steps": step,
        "expected_optimizer_steps": expected_steps,
        "epochs": epochs,
        "examples_per_epoch": len(examples),
        "example_presentations": epochs * len(examples),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "trainable_sha256_before": before_hash,
        "trainable_sha256_after": after_hash,
        "probe_changed_steps": changed_steps,
        "elapsed_training_seconds": time.monotonic() - started,
    }


def train(args: argparse.Namespace) -> None:
    import importlib.metadata

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    started_at = now()
    write_json(args.output / "state.json", {"status": "PREPARING", "started_at_SGT": started_at})
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        examples = read_examples(args.data, tokenizer, args.max_length)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        config.update(
            started_at_SGT=started_at,
            data_sha256=digest(args.data),
            implementation_sha256=digest(Path(__file__)),
            versions={
                name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
            },
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,
            lora_target_modules=LORA_TARGETS,
            attention="sdpa",
            gradient_checkpointing=True,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            schedule="constant",
        )
        write_json(args.output / "settings.json", config)
        write_json(
            args.output / "data_preflight.json",
            {
                "status": "PASSED",
                "examples": len(examples),
                "kinds": dict(Counter(example["kind"] for example in examples)),
                "max_tokens": max(len(example["input_ids"]) for example in examples),
                "min_target_tokens": min(example["target_tokens"] for example in examples),
                "total_target_tokens": sum(example["target_tokens"] for example in examples),
                "examples_manifest": [
                    {key: example[key] for key in ("id", "kind", "prompt_tokens", "target_tokens")}
                    for example in examples
                ],
            },
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
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
        write_json(args.output / "state.json", {"status": "TRAINING", "started_at_SGT": started_at})
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
            started_at_SGT=started_at,
            completed_at_SGT=now(),
            adapter=str(args.output / "adapter"),
            data_sha256=config["data_sha256"],
            artifacts={
                str(path.relative_to(args.output)): digest(path)
                for path in sorted((args.output / "adapter").glob("*"))
                if path.is_file()
            },
        )
        summary["artifacts"]["optimizer_rng.pt"] = digest(args.output / "optimizer_rng.pt")
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "state.json", summary)
    except BaseException as error:
        write_json(
            args.output / "state.json",
            {
                "status": "FAILED",
                "started_at_SGT": started_at,
                "failed_at_SGT": now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def merge(args: argparse.Namespace) -> None:
    """Run after the training process exits; never allocate a second GPU model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    started_at = now()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map="cpu",
        )
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
        model = model.merge_and_unload(safe_merge=True)
        model.config.use_cache = True
        model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
        tokenizer.save_pretrained(args.output)
        write_json(
            args.output / "merge_receipt.json",
            {
                "status": "COMPLETE",
                "started_at_SGT": started_at,
                "completed_at_SGT": now(),
                "model": str(args.model),
                "adapter": str(args.adapter),
                "device": "cpu",
                "files": {
                    path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in sorted(args.output.iterdir())
                    if path.is_file()
                },
            },
        )
    except BaseException as error:
        write_json(
            args.output / "merge_failure.json",
            {
                "status": "FAILED",
                "started_at_SGT": started_at,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--merge" in argv:
        argv.remove("--merge")
        argv.insert(0, "merge")
    elif not argv or argv[0] not in {"train", "merge", "-h", "--help"}:
        argv.insert(0, "train")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--micro-batch", type=int, default=2)
    training.add_argument("--gradient-accumulation", type=int, default=8)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--max-length", type=int, default=8192)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    exporting = commands.add_parser("merge")
    exporting.add_argument("--model", type=Path, required=True)
    exporting.add_argument("--adapter", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
