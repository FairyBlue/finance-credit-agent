"""Completion-only tool SFT and persistent LoRA PPO for mandate-repair tasks.

The frozen BF16 base is shared by all arms. Only FP32 LoRA parameters and their
Adam state are trained/saved. Counterfactual continuations can supply a scalar
local advantage, and independent current-policy continuations can replace a
selected parent's state baseline. Auxiliary tokens never enter the actor batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

from src.agents.coldstart_sft import (
    IGNORE_INDEX,
    LORA_TARGETS,
    collate_examples,
    digest,
    now,
    trainable_hash,
    write_json,
)
from src.agents.mandate_repair_metrics import business_score
from src.agents.portfolio_teacher_sft import completion_logits_model

PROFILE = "mandate_repair_lora_train_v1"
RL_CHECKPOINT_THRESHOLD = 4096
MC_ARMS = {"mc_uniform", "mc_priority"}
MC_ESTIMATOR = "independent_current_policy_state_value_mean_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def encode_record(record, tokenizer, max_length=16384):
    """Mask the whole production prefix, teaching exact raw tool JSON plus EOS."""
    identifier = record.get("id")
    require(isinstance(identifier, str) and identifier, "nonempty demonstration id")
    messages = record.get("messages")
    completion = record.get("completion")
    if completion is None and messages and messages[-1].get("role") == "assistant":
        completion = messages[-1]["content"]
        messages = messages[:-1]
    prompt = record.get("prompt")
    if prompt is None:
        require(isinstance(messages, list) and messages, f"{identifier}: production messages")
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    require(isinstance(prompt, str) and prompt, f"{identifier}: nonempty prompt")
    require(isinstance(completion, str) and completion, f"{identifier}: exact JSON completion")
    call = json.loads(completion)
    require(
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call["name"], str)
        and isinstance(call["arguments"], dict),
        f"{identifier}: raw tool call must contain name and arguments",
    )
    require(tokenizer.eos_token_id is not None and tokenizer.eos_token, "tokenizer EOS")
    require(tokenizer.eos_token not in completion, "demonstration must not contain EOS itself")
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + completion + tokenizer.eos_token, add_special_tokens=False)
    require(prefix and full[: len(prefix)] == prefix, f"{identifier}: prompt token prefix changed")
    target = full[len(prefix) :]
    require(target and target[-1] == tokenizer.eos_token_id, f"{identifier}: missing target EOS")
    require(len(full) <= max_length, f"{identifier}: sequence exceeds limit; no truncation")
    return dict(
        id=identifier,
        kind="tool",
        task_id=record.get("task_id"),
        input_ids=full,
        labels=[IGNORE_INDEX] * len(prefix) + target,
        prompt_tokens=len(prefix),
        target_tokens=len(target),
    )


def read_examples(path, tokenizer, max_length):
    examples = [encode_record(row, tokenizer, max_length) for row in read_rows(path)]
    require(
        examples and len({row["id"] for row in examples}) == len(examples),
        "nonempty demonstrations with unique ids",
    )
    return examples


def adapter_digest(directory):
    directory = Path(directory)
    files = sorted(directory.glob("adapter_model.*")) + [directory / "adapter_config.json"]
    require(files and all(path.is_file() for path in files), "existing LoRA adapter artifacts")
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode())
        h.update(digest(path).encode())
    return h.hexdigest()


def parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def load_model(args, adapter=None):
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        torch.cuda.manual_seed_all(args.seed)
    base_path = args.model if args.command == "sft" else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if adapter is None:
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
    else:
        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=True, local_files_only=True, autocast_adapter_dtype=True
        )
    model.to(args.device)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require("lora_" in name, "only LoRA parameters may be trained")
            parameter.data = parameter.data.float()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model._mandate_forced_gradient_checkpointing = args.gradient_checkpointing
    model._mandate_auto_checkpoint_threshold = (
        RL_CHECKPOINT_THRESHOLD
        if args.command == "rl" or getattr(args, "auto_request_checkpointing", False)
        else None
    )
    model._mandate_auto_checkpoint_request_uses = 0
    return model, tokenizer, torch


def execution(args, model=None):
    result = dict(
        gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
        rl_auto_checkpoint_input_threshold=(
            RL_CHECKPOINT_THRESHOLD if args.command == "rl" else None
        ),
        rl_auto_checkpoint_request_uses=(
            getattr(model, "_mandate_auto_checkpoint_request_uses", 0)
            if model is not None and args.command == "rl"
            else None
        ),
        attention="sdpa",
        full_prompt_attention=True,
        completion_logits_only=True,
    )
    if args.command == "sft" and getattr(args, "auto_request_checkpointing", False):
        result.update(
            sft_auto_checkpoint_input_threshold=RL_CHECKPOINT_THRESHOLD,
            sft_auto_checkpoint_request_uses=(
                getattr(model, "_mandate_auto_checkpoint_request_uses", 0)
                if model is not None else 0
            ),
        )
    if args.command == "rl" and getattr(args, "diagnostic_request_limit", 0):
        result["diagnostic_request_limit"] = args.diagnostic_request_limit
    if args.command == "rl" and getattr(args, "collection_seed", None) is not None:
        result["collection_seed"] = args.collection_seed
    return result


def cuda_memory(args, torch):
    if not args.device.startswith("cuda"):
        return {}
    return dict(
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )


def recipe(args):
    result = dict(
        profile=PROFILE,
        command=args.command,
        base_model=str((args.model if args.command == "sft" else args.base_model).resolve()),
        seed=args.seed,
        arm=getattr(args, "arm", "sft"),
        learning_rate=args.learning_rate,
        lora_rank=16,
        lora_alpha=32,
        clip_epsilon=getattr(args, "clip_epsilon", None),
        local_weight=getattr(args, "local_weight", None),
        gradient_accumulation=getattr(args, "gradient_accumulation", None),
    )
    # Keep old default recipes byte-for-byte compatible with v1 recovery points.
    if args.command == "sft" and getattr(args, "adapter", None) is not None:
        result["initial_adapter"] = str(args.adapter.resolve())
    if getattr(args, "local_credit_target", None) not in (None, "reward"):
        result["local_credit_target"] = args.local_credit_target
    if getattr(args, "arm", None) in MC_ARMS:
        result["mc_estimator"] = MC_ESTIMATOR
    if args.command == "sft" and getattr(args, "auto_request_checkpointing", False):
        result["sft_auto_checkpoint_input_threshold"] = RL_CHECKPOINT_THRESHOLD
    if args.command == "rl" and (
        getattr(args, "trajectory_minibatch_size", 4) != 4
        or getattr(args, "mixed_task_minibatches", False)
    ):
        result.update(
            trajectory_minibatch_size=getattr(args, "trajectory_minibatch_size", 4),
            mixed_task_minibatches=bool(getattr(args, "mixed_task_minibatches", False)),
        )
    if args.command == "sft" and getattr(args, "max_training_seconds", None) is not None:
        result["max_training_seconds"] = args.max_training_seconds
    return result


def optimizer_steps(optimizer):
    values = [int(state["step"].item()) for state in optimizer.state.values()]
    return (min(values), max(values)) if values else (0, 0)


def save_checkpoint(path, model, optimizer, args, progress, torch):
    """One atomic, small LoRA-only recovery point; no frozen base duplication."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        recipe=recipe(args),
        execution=execution(args),
        progress=progress,
        master={name: p.detach().cpu().clone() for name, p in parameters(model).items()},
        optimizer=optimizer.state_dict(),
        rng=dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
        ),
    )
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(
        path.with_suffix(path.suffix + ".json"),
        dict(
            status="COMPLETE",
            recipe=recipe(args),
            execution=execution(args),
            progress=progress,
            bytes=path.stat().st_size,
            trainable_tensor_count=len(payload["master"]),
            frozen_base_saved=False,
        ),
    )


def restore_checkpoint(path, model, optimizer, args, torch):
    import numpy as np

    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(saved["recipe"] == recipe(args), "resume recipe/base/seed/arm changed")
    trainable = parameters(model)
    require(list(saved["master"]) == list(trainable), "resume LoRA parameter names changed")
    with torch.no_grad():
        for name, parameter in trainable.items():
            master = saved["master"][name]
            require(
                master.dtype == torch.float32 and master.shape == parameter.shape,
                "resume FP32 LoRA master dtype/shape",
            )
            parameter.copy_(master.to(parameter.device))
    optimizer.load_state_dict(saved["optimizer"])
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                require(state[key].dtype == torch.float32, "FP32 Adam moments preserved")
    random.setstate(saved["rng"]["python"])
    np.random.set_state(saved["rng"]["numpy"])
    torch.set_rng_state(saved["rng"]["torch_cpu"])
    if args.device.startswith("cuda") and saved["rng"]["torch_cuda"]:
        torch.cuda.set_rng_state_all(saved["rng"]["torch_cuda"])
    completed = saved["progress"]["completed_optimizer_steps"]
    require(optimizer_steps(optimizer) == (completed, completed), "persistent Adam step counters")
    return saved["progress"]


def export_adapter(model, tokenizer, directory, completed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "EXPORT_INCOMPLETE.json", dict(completed_optimizer_steps=completed))
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    fingerprint = adapter_digest(directory)
    write_json(
        directory / "export_receipt.json",
        dict(
            status="COMPLETE",
            profile=PROFILE,
            completed_optimizer_steps=completed,
            adapter_sha256=fingerprint,
            trainable_sha256=trainable_hash(model),
            base_merged=False,
            precision="FP32 LoRA master; BF16 frozen base",
        ),
    )
    (directory / "EXPORT_INCOMPLETE.json").unlink()
    return fingerprint


def _write_log(stream, row):
    stream.write(canonical(row) + "\n")
    stream.flush()
    print(canonical(row), flush=True)


def sft(args):
    invocation_started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    require(args.epochs == 1 and args.micro_batch == 1, "one epoch, microbatch one")
    time_limit = getattr(args, "max_training_seconds", None)
    require(
        time_limit is None or finite(time_limit) and time_limit > 0,
        "positive finite SFT stage time limit",
    )
    model, tokenizer, torch = load_model(args, getattr(args, "adapter", None))
    examples = read_examples(args.data, tokenizer, args.max_length)
    source_hash = digest(args.data)
    settings = dict(
        recipe(args),
        execution=execution(args),
        data=str(args.data.resolve()),
        data_sha256=source_hash,
        examples=len(examples),
        epochs=1,
        micro_batch=1,
        supervision="exact raw JSON plus EOS; entire prefix masked",
        full_prompt_attention=True,
        total_prompt_tokens=sum(row["prompt_tokens"] for row in examples),
        total_target_tokens=sum(row["target_tokens"] for row in examples),
    )
    write_json(args.output / "settings.json", settings)
    optimizer = torch.optim.AdamW(
        parameters(model).values(), lr=args.learning_rate, weight_decay=0.0
    )
    progress = dict(completed_optimizer_steps=0, cursor=0, data_sha256=source_hash)
    if args.resume_state:
        progress = restore_checkpoint(args.resume_state, model, optimizer, args, torch)
        require(progress["data_sha256"] == source_hash, "SFT resume demonstrations changed")
    previously_elapsed = progress.get("sft_time_budget_seconds_consumed", 0.0)

    def stage_elapsed():
        return previously_elapsed + time.monotonic() - invocation_started

    def record_time_budget():
        if time_limit is not None:
            progress["sft_time_budget_seconds_consumed"] = stage_elapsed()

    state_path = args.state_output or args.output / "training_state.pt"
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    wrapped = completion_logits_model(model)
    model.train()
    started = time.monotonic()
    before = trainable_hash(model)
    with (args.output / "steps.jsonl").open("a", buffering=1) as stream:
        for start in range(progress["cursor"], len(order), args.gradient_accumulation):
            # Preserve a complete optimizer update and exportable checkpoint;
            # a group already in flight may overrun the cap and is reported.
            if time_limit is not None and stage_elapsed() >= time_limit:
                break
            group = [examples[index] for index in order[start : start + args.gradient_accumulation]]
            target_count = sum(row["target_tokens"] for row in group)
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for row in group:
                if getattr(args, "auto_request_checkpointing", False):
                    configure_request_checkpointing(model, len(row["input_ids"]))
                batch = {
                    key: value.to(args.device)
                    for key, value in collate_examples([row], tokenizer.pad_token_id).items()
                }
                loss = wrapped(**batch, use_cache=False).loss * row["target_tokens"] / target_count
                require(torch.isfinite(loss).item(), "finite SFT loss")
                mean_loss += float(loss.detach())
                loss.backward()
                del batch, loss
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters(model).values(), 1.0, error_if_nonfinite=True
                )
            )
            require(norm > 0 and finite(norm), "nonzero finite SFT gradient")
            optimizer.step()
            progress.update(
                completed_optimizer_steps=progress["completed_optimizer_steps"] + 1,
                cursor=start + len(group),
            )
            step = progress["completed_optimizer_steps"]
            require(optimizer_steps(optimizer) == (step, step), "SFT Adam counters preserved")
            _write_log(
                stream,
                dict(
                    step=step,
                    examples=len(group),
                    loss=mean_loss,
                    grad_norm_before_clip=norm,
                    supervised_tokens=target_count,
                    cursor=progress["cursor"],
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            if step % args.checkpoint_every == 0:
                record_time_budget()
                save_checkpoint(state_path, model, optimizer, args, progress, torch)
    progress["exported_adapter_sha256"] = export_adapter(
        model,
        tokenizer,
        args.actor_output or args.output / "adapter",
        progress["completed_optimizer_steps"],
    )
    record_time_budget()
    save_checkpoint(state_path, model, optimizer, args, progress, torch)
    time_stopped = time_limit is not None and progress["cursor"] < len(order)
    summary = dict(
        settings,
        status="COMPLETE_TIME_LIMIT" if time_stopped else "COMPLETE",
        **progress,
        expected_optimizer_steps=math.ceil(len(examples) / args.gradient_accumulation),
        trainable_sha256_before=before,
        trainable_sha256_after=trainable_hash(model),
        state_output=str(state_path),
        adapter=str(args.actor_output or args.output / "adapter"),
        elapsed_training_seconds=time.monotonic() - started,
        actual_supervised_examples=progress["cursor"],
        actual_supervised_prompt_tokens=sum(
            examples[index]["prompt_tokens"] for index in order[: progress["cursor"]]
        ),
        actual_supervised_target_tokens=sum(
            examples[index]["target_tokens"] for index in order[: progress["cursor"]]
        ),
        execution=execution(args, model),
        **cuda_memory(args, torch),
    )
    if time_limit is not None:
        elapsed = stage_elapsed()
        summary.update(
            max_training_seconds=time_limit,
            elapsed_stage_seconds=elapsed,
            time_limit_reached=elapsed >= time_limit,
            time_limit_overrun_seconds=max(0.0, elapsed - time_limit),
            stop_reason="TIME_LIMIT_AT_UPDATE_BOUNDARY" if time_stopped else "ONE_EPOCH_COMPLETE",
            time_limit_includes="model loading, encoding, updates, checkpoints and adapter export",
        )
    write_json(args.output / "summary.json", summary)
    return summary


def request_key(row, index):
    return row.get("request_key", row.get("key", f"request/{index}"))


def build_batch(
    directory, arm="rloo", local_credit=None, local_weight=0.25, local_credit_target=None,
    credit_shuffle_seed=42, mc_baselines=None,
):
    """Keep all genuine terminal outcomes, including unsuccessful tasks."""
    directory = Path(directory)
    require(
        (arm in MC_ARMS) == (mc_baselines is not None),
        "MC arms require --mc-baselines; other arms cannot silently consume MC baselines",
    )
    require(
        arm not in MC_ARMS or local_credit is None,
        "MC state baseline replacement cannot mix with local credit residuals",
    )
    parents = read_rows(directory / "trajectory_results.jsonl")
    requests = read_rows(directory / "requests.jsonl")
    identity = hashlib.sha256(
        (
            digest(directory / "trajectory_results.jsonl")
            + digest(directory / "requests.jsonl")
            + (digest(Path(local_credit)) if local_credit is not None else "")
            + (digest(Path(mc_baselines)) if mc_baselines is not None else "")
        ).encode()
    ).hexdigest()
    parent_map, task_groups, request_groups = {}, {}, {}
    for row in parents:
        key = row["trajectory_key"]
        require(key not in parent_map, "unique parent trajectory keys")
        require(row.get("role", "parent") == "parent", "parent trajectory role required")
        require(not key.startswith("branch/"), "auxiliary trajectories cannot train the actor")
        parent_map[key] = row
        task_groups.setdefault(row["task_id"], []).append(row)
    seen = set()
    for index, source in enumerate(requests):
        row = dict(source, request_key=request_key(source, index))
        require(row["request_key"] not in seen, "unique parent request keys")
        seen.add(row["request_key"])
        require(row.get("role", "parent") == "parent", "auxiliary requests cannot enter actor data")
        require(
            row.get("decode_mode", "native") == "native" and row.get("temperature", 1.0) == 1.0,
            "main PPO data must use native untempered policy sampling",
        )
        require(
            row["trajectory_key"] in parent_map,
            "requests must belong to actual parent trajectories",
        )
        if row.get("prompt_token_ids") and row.get("response_token_ids"):
            request_groups.setdefault(row["trajectory_key"], []).append(row)
    admitted, unsupported = {}, []
    for task_id, group in task_groups.items():
        supported = (
            len(group) == 4
            and len({row["replicate_index"] for row in group}) == 4
            and all(
                finite(row.get("terminal_reward")) and request_groups.get(row["trajectory_key"])
                for row in group
            )
        )
        if not supported:
            unsupported.append(task_id)
            continue
        total = math.fsum(row["terminal_reward"] for row in group)
        for parent in group:
            key = parent["trajectory_key"]
            advantage = parent["terminal_reward"] - (total - parent["terminal_reward"]) / 3
            admitted[key] = dict(
                parent=parent, requests=request_groups[key], raw_advantage=advantage
            )
    require(admitted, "at least one task with four actual terminal parent outcomes")
    mc_records, mc_unsupported, mc_adapter = {}, [], None
    if mc_baselines is not None:
        collection = json.loads((directory / "collection_started.json").read_text())
        mc_adapter = collection.get("adapter_sha256")
        require(
            isinstance(mc_adapter, str) and len(mc_adapter) == 64
            and all(character in "0123456789abcdef" for character in mc_adapter),
            "MC parent collection must declare its actual adapter SHA256",
        )
        parent_requests = {
            request_key(row, index): row for index, row in enumerate(requests)
        }
        mc_seen = set()
        for source in read_rows(mc_baselines):
            anchor = source.get("parent_request_key", source.get("request_key"))
            require(isinstance(anchor, str) and anchor, "MC parent request key required")
            require(anchor not in mc_seen, "unique MC parent request keys")
            mc_seen.add(anchor)
            require(
                source.get("request_key", anchor) == anchor,
                "MC request key aliases identify the same parent request",
            )
            require(anchor in parent_requests, "MC baseline belongs to an actual parent request")
            parent_request = parent_requests[anchor]
            require(
                source.get("trajectory_key") == parent_request["trajectory_key"],
                "MC request and trajectory identify the same actual parent",
            )
            require(
                source.get("record_type") == "state_value_baseline"
                and source.get("continuation_policy") == "current_policy",
                "MC values require independent current-policy state continuations",
            )
            require(
                source.get("adapter_sha256") == mc_adapter,
                "MC continuations must match the actual parent adapter snapshot",
            )
            require(
                source.get("parent_action_excluded") is True
                and source.get("parent_return_excluded") is True
                and source.get("aux_rng_independent") is True,
                "MC baseline must exclude the parent action/return and use independent RNG",
            )
            probability = source.get("selection_probability")
            require(
                finite(probability) and 0 < probability <= 1,
                "MC selection probability must be finite in (0, 1]",
            )
            if source.get("status") != "COMPLETE":
                require(
                    source.get("status") in {"INCOMPLETE", "SKIPPED", "BUDGET_EXHAUSTED"},
                    "known incomplete MC baseline status",
                )
                mc_unsupported.append(anchor)
                continue
            values = source.get("aux_returns", [])
            require(
                type(source.get("m")) is int and source["m"] == 2
                and isinstance(values, list) and len(values) == 2
                and all(finite(value) for value in values),
                "MC baseline requires two actual finite terminal continuation returns",
            )
            value = math.fsum(values) / 2
            require(
                finite(source.get("mc_baseline"))
                and math.isclose(value, source["mc_baseline"], rel_tol=1e-12, abs_tol=1e-12),
                "MC baseline recomputes from its two independent continuation returns",
            )
            aux_keys = source.get("aux_trajectory_keys")
            if aux_keys is not None:
                require(
                    isinstance(aux_keys, list) and len(aux_keys) == 2
                    and all(isinstance(key, str) and key for key in aux_keys)
                    and len(set(aux_keys)) == 2
                    and source["trajectory_key"] not in aux_keys,
                    "MC auxiliary trajectories must be distinct and exclude the parent",
                )
            if source["trajectory_key"] not in admitted:
                mc_unsupported.append(anchor)
                continue
            mc_records[anchor] = source
    anchors, local_unsupported = {}, []
    estimands, estimators = set(), set()
    if local_credit is not None:
        for source in read_rows(local_credit):
            key = source["trajectory_key"]
            require(key not in anchors, "at most one information anchor per parent")
            target = source.get("local_credit_target", "reward")
            require(target in {"reward", "business"}, "known local credit target")
            require(
                local_credit_target is None or target == local_credit_target,
                "local record estimand must match requested target",
            )
            estimands.add(target)
            estimators.add(source.get("credit_estimator", "parent_minus_mask_mean_v1"))
            anchors[key] = source
    require(len(estimands) <= 1, "one consistent local estimand per actor batch")
    require(len(estimators) <= 1, "one consistent local estimator per actor batch")
    shuffled_assignments = []
    for key, group in admitted.items():
        local = anchors.get(key)
        anchor_key, local_value = None, 0.0
        if local is not None:
            rewards = local.get("counterfactual_rewards", [])
            if (
                local.get("status") == "COMPLETE"
                and 1 <= len(rewards) <= 2
                and all(finite(value) for value in rewards)
            ):
                require(
                    local.get("continuation_policy") == "current_policy",
                    "local values require current-policy continuations, not teacher labels",
                )
                anchor_key = local["request_key"]
                require(
                    anchor_key in {row["request_key"] for row in group["requests"]},
                    "local information anchor belongs to its actual parent",
                )
                if local.get("local_credit_target", "reward") == "business":
                    value = business_score(group["parent"])
                    paired = local.get("credit_estimator") == "paired_factual_mask_mean_v1"
                    counterfactual = local.get(
                        "masked_target_values" if paired else "counterfactual_target_values", []
                    )
                    require(
                        value is not None and value in (0.0, 1.0), "actual parent business score"
                    )
                    require(
                        len(counterfactual) == len(rewards)
                        and all(finite(v) and v in (0.0, 1.0) for v in counterfactual),
                        "binary counterfactual business values from actual continuations",
                    )
                    require(
                        local.get("parent_target_value") == value,
                        "local parent value matches actual terminal business verification",
                    )
                    if paired:
                        factual = local.get("factual_target_values", [])
                        factual_rewards = local.get("factual_rewards", [])
                        require(
                            len(factual) == len(counterfactual) == len(factual_rewards) == 2
                            and all(finite(v) and v in (0.0, 1.0) for v in factual)
                            and all(finite(v) for v in factual_rewards),
                            "two actual paired factual/masked business continuations required",
                        )
                        local_value = (
                            math.fsum(factual) - math.fsum(counterfactual)
                        ) / len(factual)
                        require(
                            finite(local.get("unscaled_local_difference"))
                            and math.isclose(
                                local_value, local["unscaled_local_difference"],
                                rel_tol=1e-12, abs_tol=1e-12,
                            ),
                            "paired local credit recomputes from recorded continuations",
                        )
                    else:
                        require(
                            local.get("credit_estimator", "parent_minus_mask_mean_v1")
                            == "parent_minus_mask_mean_v1",
                            "known business local estimator",
                        )
                        local_value = value - math.fsum(counterfactual) / len(counterfactual)
                else:
                    local_value = (
                        group["parent"]["terminal_reward"] - math.fsum(rewards) / len(rewards)
                    )
            else:
                local_unsupported.append(key)
        assigned_key = anchor_key
        if arm == "shuffled" and local_value != 0:
            alternatives = sorted(
                row["request_key"] for row in group["requests"]
                if row["request_key"] != anchor_key
            )
            require(alternatives, "nonzero shuffled credit needs another real parent request")
            seed = hashlib.sha256(canonical([credit_shuffle_seed, key]).encode()).digest()
            assigned_key = alternatives[int.from_bytes(seed[:8], "big") % len(alternatives)]
            shuffled_assignments.append(dict(
                trajectory_key=key, original_request_key=anchor_key,
                assigned_request_key=assigned_key, raw_local_credit=local_value,
            ))
        for row in group["requests"]:
            row["trajectory_advantage"] = group["raw_advantage"]
            row["raw_local_credit"] = local_value if row["request_key"] == assigned_key else 0.0
            row["raw_combined_advantage"] = group["raw_advantage"] + (
                local_weight * row["raw_local_credit"] if arm in {"m3", "shuffled"} else 0.0
            )
            if arm in MC_ARMS:
                mc = mc_records.get(row["request_key"])
                row["mc_baseline_used"] = mc is not None
                row["raw_parent_return"] = group["parent"]["terminal_reward"]
                row["state_value_baseline"] = mc["mc_baseline"] if mc is not None else None
                if mc is not None:
                    # These environments emit only one terminal reward, including
                    # total calls/cost. No earlier reward has to be subtracted.
                    row["raw_combined_advantage"] = (
                        row["raw_parent_return"] - mc["mc_baseline"]
                    )
    all_rows = [row for group in admitted.values() for row in group["requests"]]
    mean = math.fsum(row["raw_combined_advantage"] for row in all_rows) / len(all_rows)
    std = math.sqrt(
        math.fsum((row["raw_combined_advantage"] - mean) ** 2 for row in all_rows) / len(all_rows)
    )
    for row in all_rows:
        row["advantage"] = (row["raw_combined_advantage"] - mean) / std if std > 1e-12 else 0.0
    return admitted, dict(
        data_sha256=identity,
        planned_tasks=len(task_groups),
        supported_tasks=len(admitted) // 4,
        unsupported_task_ids=unsupported,
        parent_trajectories=len(admitted),
        parent_requests=len(all_rows),
        advantages_mean=mean,
        advantages_std=std,
        local_credit_path=str(local_credit) if local_credit else None,
        local_anchor_records=len(anchors),
        local_credit_targets=sorted(estimands),
        local_credit_estimators=sorted(estimators),
        local_assignment=("within_parent_wrong_step" if arm == "shuffled" else "original"),
        shuffled_local_assignments=shuffled_assignments,
        unsupported_local_parents=local_unsupported,
        local_nonzero_requests=sum(row["raw_local_credit"] != 0 for row in all_rows),
        effective_advantage_requests=sum(row["advantage"] != 0 for row in all_rows),
        local_tokens_trained=0,
        counterfactual_tokens_trained=0,
        mc_baseline_path=str(mc_baselines) if mc_baselines is not None else None,
        mc_estimator=MC_ESTIMATOR if arm in MC_ARMS else None,
        mc_adapter_sha256=mc_adapter,
        mc_baseline_requests=len(mc_records),
        unsupported_mc_parent_requests=mc_unsupported,
        mc_return_semantics="terminal_only_environment_reward" if arm in MC_ARMS else None,
        mc_tokens_trained=0,
        parent_prompt_tokens=sum(len(row["prompt_token_ids"]) for row in all_rows),
        parent_response_tokens=sum(len(row["response_token_ids"]) for row in all_rows),
    )


def request_checkpointing_enabled(
    input_length, forced=False, training=True, threshold=RL_CHECKPOINT_THRESHOLD
):
    return bool(forced or (training and threshold is not None and input_length > threshold))


def configure_request_checkpointing(model, input_length):
    """Use the same length rule for financial SFT and parent RL requests."""
    if not model.training:
        return False
    forced = getattr(model, "_mandate_forced_gradient_checkpointing", False)
    threshold = getattr(model, "_mandate_auto_checkpoint_threshold", RL_CHECKPOINT_THRESHOLD)
    enabled = request_checkpointing_enabled(input_length, forced, threshold=threshold)
    if enabled != bool(model.is_gradient_checkpointing):
        if enabled:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_disable()
    if enabled and not forced:
        model._mandate_auto_checkpoint_request_uses = (
            getattr(model, "_mandate_auto_checkpoint_request_uses", 0) + 1
        )
    return enabled


def token_logprobs(model, row, device, torch):
    response = row["response_token_ids"]
    inputs = torch.tensor([row["prompt_token_ids"] + response], dtype=torch.long, device=device)
    configure_request_checkpointing(model, inputs.shape[1])
    result = model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        use_cache=False,
        logits_to_keep=len(response) + 1,
    )
    logits = result.logits[0, :-1, :].float()
    labels = torch.tensor(response, dtype=torch.long, device=device)
    return -torch.nn.functional.cross_entropy(logits, labels, reduction="none")


def trajectory_schedule(admitted, args, data_sha256):
    """Schedule genuine parents, optionally mixing every task at each update.

    The historical default retains its cyclic four-parent scheduling. The new
    protocol takes each task's same replicate once, then advances to the next
    replicate; no parent is repeated or silently omitted.
    """
    size = getattr(args, "trajectory_minibatch_size", 4)
    mixed = bool(getattr(args, "mixed_task_minibatches", False))
    require(type(size) is int and size > 0, "positive integer trajectory minibatch size")
    task_ids = sorted({group["parent"]["task_id"] for group in admitted.values()})
    replicas = sorted({group["parent"]["replicate_index"] for group in admitted.values()})
    require(task_ids and replicas, "nonempty trajectory schedule")
    rng = random.Random(args.seed + int(data_sha256[:12], 16))
    keys, replica_groups = [], []
    for replica in replicas:
        ordered = task_ids.copy()
        rng.shuffle(ordered)
        replica_keys = []
        for task_id in ordered:
            matching = [
                key
                for key, group in admitted.items()
                if group["parent"]["task_id"] == task_id
                and group["parent"]["replicate_index"] == replica
            ]
            if mixed:
                require(len(matching) == 1, "one genuine parent per task and replicate")
            replica_keys.extend(matching)
        keys.extend(replica_keys)
        replica_groups.append(replica_keys)
    requested = getattr(args, "updates", None)
    require(
        requested is None or type(requested) is int and requested > 0,
        "positive integer requested updates",
    )
    if mixed:
        require(size == len(task_ids), "mixed minibatch size must equal supported task count")
        require(
            requested is None or requested == len(replicas),
            "mixed schedule uses every parent exactly once; updates must equal replicate count",
        )
        require(
            len(keys) == len(admitted) and len(set(keys)) == len(keys),
            "mixed schedule covers genuine parents exactly once",
        )
        return replica_groups
    updates = requested or math.ceil(len(keys) / size)
    return [
        [keys[(index * size + offset) % len(keys)] for offset in range(size)]
        for index in range(updates)
    ]


def trajectory_request_weight(selected_count, request_count):
    require(selected_count > 0 and request_count > 0, "nonempty minibatch and trajectory requests")
    return 1 / (selected_count * request_count)


def diagnostic_phase(row):
    action = row.get("action")
    name = action.get("name") if isinstance(action, dict) else None
    if name == "calculate":
        return "calculation"
    if name in {"solve", "submit"}:
        return "execution"
    if name in {"search_docs", "read_document"}:
        return "evidence"
    return "other"


def diagnostic_requests(rows, limit):
    """Fix a small, balanced set using observed actions and pre-update advantages.

    Selection never reads hidden task truth or post-update probabilities. It
    does not change the actor batch or consume the actor's random stream.
    """
    require(type(limit) is int and 0 <= limit <= 16, "diagnostic request limit must be in 0..16")
    if not limit:
        return []
    buckets = {}
    for row in rows:
        raw = row["raw_combined_advantage"]
        sign = "positive" if raw > 0 else "negative" if raw < 0 else "zero"
        buckets.setdefault((diagnostic_phase(row), sign), []).append(row)
    phase_order = {name: index for index, name in enumerate(
        ("calculation", "execution", "evidence", "other")
    )}
    sign_order = {"positive": 0, "negative": 1, "zero": 2}
    ordered = sorted(buckets, key=lambda key: (phase_order[key[0]], sign_order[key[1]]))
    for bucket in buckets.values():
        bucket.sort(key=lambda row: row["request_key"])
    selected, depth = [], 0
    while len(selected) < limit:
        found = False
        for key in ordered:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                found = True
                if len(selected) == limit:
                    break
        if not found:
            break
        depth += 1
    return selected


def probability_delta_record(row, old_mean, new_mean):
    require(finite(old_mean) and finite(new_mean), "finite fixed-response mean logprob")
    return dict(
        request_key=row["request_key"],
        trajectory_key=row["trajectory_key"],
        phase=diagnostic_phase(row),
        raw_combined_advantage=row["raw_combined_advantage"],
        actor_advantage=row["advantage"],
        old_mean_response_logprob=old_mean,
        new_mean_response_logprob=new_mean,
        mean_response_logprob_delta=new_mean - old_mean,
        diagnostic="teacher_forcing_same_recorded_response; not KL or new task success",
        prompt_tokens=len(row["prompt_token_ids"]),
        response_tokens=len(row["response_token_ids"]),
    )


def validate_collection_sampling(collection, args):
    """Separate per-collection randomness from the persistent optimizer seed."""
    expected_seed = getattr(args, "collection_seed", None)
    if expected_seed is None:
        expected_seed = args.seed
    require(
        collection.get("seed") == expected_seed,
        "actual collection seed must match --collection-seed or the legacy --seed",
    )
    require(
        collection.get("repetitions") == 4
        and collection.get("temperature") == 1.0
        and collection.get("decode_mode") == "native",
        "native temperature-one K4 collection required",
    )
    return expected_seed


def rl(args):
    args.output.mkdir(parents=True, exist_ok=True)
    require(
        args.local_weight == 0.25 and args.clip_epsilon == 0.2,
        "fixed local eta=.25 and PPO clip=.2",
    )
    require(
        args.arm == "rloo" or args.arm in MC_ARMS or args.local_credit is not None,
        "sham/M3 must declare their separately sampled local record file",
    )
    admitted, batch = build_batch(
        args.data, args.arm, args.local_credit, args.local_weight,
        getattr(args, "local_credit_target", None), credit_shuffle_seed=args.seed,
        mc_baselines=getattr(args, "mc_baselines", None),
    )
    schedule = trajectory_schedule(admitted, args, batch["data_sha256"])
    updates = len(schedule)
    all_rows = [row for group in admitted.values() for row in group["requests"]]
    diagnostics = diagnostic_requests(all_rows, getattr(args, "diagnostic_request_limit", 0))
    diagnostic_keys = {row["request_key"] for row in diagnostics}
    if diagnostics:
        write_json(args.output / "request_probability_selection.json", dict(
            selected_before_optimizer_updates=True,
            selection="deterministic phase and raw-advantage-sign round robin; no gold",
            request_keys=[row["request_key"] for row in diagnostics],
        ))
    source_adapter = adapter_digest(args.adapter)
    collection = json.loads((args.data / "collection_started.json").read_text())
    require(
        Path(collection["base_model_path"]).resolve() == args.base_model.resolve()
        and Path(collection["adapter_path"]).resolve() == args.adapter.resolve()
        and collection["adapter_sha256"] == source_adapter,
        "old likelihoods must use the actual sampled base and adapter snapshot",
    )
    sampled_collection_seed = validate_collection_sampling(collection, args)
    if args.local_credit is not None:
        for row in read_rows(args.local_credit):
            if row.get("credit_estimator") == "paired_factual_mask_mean_v1":
                require(
                    row.get("adapter_sha256") == source_adapter,
                    "paired continuations must use the actual parent policy snapshot",
                )
    if args.arm in MC_ARMS:
        require(
            batch["mc_adapter_sha256"] == source_adapter,
            "MC continuations must use the actual sampled actor snapshot",
        )
    model, tokenizer, torch = load_model(args, args.adapter)
    write_json(
        args.output / "settings.json",
        dict(
            recipe=recipe(args),
            execution=execution(args),
            data=str(args.data.resolve()),
            source_adapter_sha256=source_adapter,
            sampled_collection_seed=sampled_collection_seed,
        ),
    )
    old_read_rows = [
        row for row in all_rows
        if row["advantage"] != 0 or row["request_key"] in diagnostic_keys
    ]
    # Recompute both likelihoods with the same HF implementation, on genuine
    # native temperature-one samples. No guided or truncated distribution enters.
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(old_read_rows, 1):
            old = token_logprobs(model, row, args.device, torch).detach().float().cpu()
            require(torch.isfinite(old).all().item(), "finite raw old logprobs")
            row["old_logprobs"] = old
            if index % 32 == 0:
                print(
                    canonical(dict(stage="OLD_LOGPROBS", complete=index, total=len(old_read_rows))),
                    flush=True,
                )
    optimizer = torch.optim.AdamW(
        parameters(model).values(), lr=args.learning_rate, weight_decay=0.0
    )
    progress = dict(
        completed_optimizer_steps=0,
        batch_cursor=0,
        batch_data_sha256=batch["data_sha256"],
        source_adapter_sha256=source_adapter,
        batch_initial_optimizer_steps=0,
        batch_executed_trajectory_keys=[],
        batch_skipped_minibatches=[],
    )
    if args.resume_state:
        previous = restore_checkpoint(args.resume_state, model, optimizer, args, torch)
        if previous.get("batch_data_sha256") == batch["data_sha256"]:
            require(
                previous["source_adapter_sha256"] == source_adapter,
                "resumed in-progress batch uses its original sampling adapter",
            )
            progress = previous
        else:
            require(
                previous.get("exported_adapter_sha256") == source_adapter,
                "next collection uses the last exported adapter",
            )
            progress["completed_optimizer_steps"] = previous["completed_optimizer_steps"]
            progress["batch_initial_optimizer_steps"] = previous["completed_optimizer_steps"]
    start_steps = progress["completed_optimizer_steps"]
    state_path = args.state_output or args.output / "training_state.pt"
    model.train()
    before = trainable_hash(model)
    started = time.monotonic()
    executed = progress["batch_executed_trajectory_keys"]
    skipped = progress["batch_skipped_minibatches"]
    with (args.output / "optimizer_updates.jsonl").open("a", buffering=1) as stream:
        for index in range(progress["batch_cursor"], len(schedule)):
            selected = schedule[index]
            optimizer.zero_grad(set_to_none=True)
            losses, ratios, clips = [], [], []
            for key in selected:
                group = admitted[key]
                weight = trajectory_request_weight(len(selected), len(group["requests"]))
                for row in group["requests"]:
                    if row["advantage"] == 0:
                        continue
                    new = token_logprobs(model, row, args.device, torch)
                    old = row["old_logprobs"].to(args.device)
                    ratio = (new - old).exp()
                    require(torch.isfinite(ratio).all().item(), "finite PPO likelihood ratio")
                    advantage = row["advantage"]
                    loss = (
                        -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
                        * weight
                    )
                    require(torch.isfinite(loss).item(), "finite PPO loss")
                    loss.backward()
                    losses.append(float(loss.detach()))
                    ratios.append(float(ratio.detach().mean()))
                    clips.append(float(((ratio.detach() - 1).abs() > 0.2).float().mean()))
                    del new, old, ratio, loss
            progress["batch_cursor"] = index + 1
            if not losses:
                skipped.append(index)
                _write_log(
                    stream,
                    dict(
                        status="SKIPPED_NO_ADVANTAGE",
                        batch_index=index,
                        optimizer_step=progress["completed_optimizer_steps"],
                    ),
                )
                continue
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters(model).values(), 1.0, error_if_nonfinite=True
                )
            )
            if norm == 0:
                skipped.append(index)
                _write_log(
                    stream,
                    dict(
                        status="SKIPPED_ZERO_GRADIENT",
                        batch_index=index,
                        optimizer_step=progress["completed_optimizer_steps"],
                    ),
                )
                continue
            require(finite(norm), "finite adapter gradient")
            optimizer.step()
            progress["completed_optimizer_steps"] += 1
            step = progress["completed_optimizer_steps"]
            require(optimizer_steps(optimizer) == (step, step), "persistent RL Adam counters")
            executed.extend(selected)
            _write_log(
                stream,
                dict(
                    status="UPDATED",
                    batch_index=index,
                    optimizer_step=step,
                    trajectories=selected,
                    trajectory_minibatch_size=len(selected),
                    task_ids=[admitted[key]["parent"]["task_id"] for key in selected],
                    replicate_indices=[
                        admitted[key]["parent"]["replicate_index"] for key in selected
                    ],
                    loss=math.fsum(losses),
                    gradient_norm_before_clip=norm,
                    mean_request_ratio=math.fsum(ratios) / len(ratios),
                    mean_request_clip_fraction=math.fsum(clips) / len(clips),
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            if step % args.checkpoint_every == 0:
                save_checkpoint(state_path, model, optimizer, args, progress, torch)
    probability_diagnostic = None
    if diagnostics:
        records = []
        model.eval()
        with torch.no_grad():
            for row in diagnostics:
                new = token_logprobs(model, row, args.device, torch)
                require(torch.isfinite(new).all().item(), "finite post-update diagnostic logprobs")
                records.append(probability_delta_record(
                    row, float(row["old_logprobs"].mean()), float(new.float().mean())
                ))
        diagnostic_path = args.output / "request_probability_deltas.jsonl"
        diagnostic_path.write_text("".join(canonical(row) + "\n" for row in records))
        probability_diagnostic = dict(
            status="COMPLETE", kind="teacher_forcing_same_recorded_response",
            selected_requests=len(records),
            nonzero_raw_advantage_requests=sum(
                row["raw_combined_advantage"] != 0 for row in records
            ),
            aligned_nonzero_raw_advantage_requests=sum(
                row["raw_combined_advantage"] * row["mean_response_logprob_delta"] > 0
                for row in records
            ),
            nonzero_actor_advantage_requests=sum(row["actor_advantage"] != 0 for row in records),
            aligned_nonzero_actor_advantage_requests=sum(
                row["actor_advantage"] * row["mean_response_logprob_delta"] > 0
                for row in records
            ),
            post_update_prompt_token_reads=sum(row["prompt_tokens"] for row in records),
            post_update_response_token_reads=sum(row["response_tokens"] for row in records),
            records=str(diagnostic_path),
            interpretation="fixed-response probability shift; not KL, rollout success or causality",
        )
    progress["exported_adapter_sha256"] = export_adapter(
        model,
        tokenizer,
        args.actor_output or args.output / "adapter",
        progress["completed_optimizer_steps"],
    )
    save_checkpoint(state_path, model, optimizer, args, progress, torch)
    summary = dict(
        profile=PROFILE,
        status="COMPLETE" if executed else "COMPLETE_NO_EFFECTIVE_UPDATES",
        recipe=recipe(args),
        execution=execution(args, model),
        **batch,
        **progress,
        optimizer_steps_before=progress["batch_initial_optimizer_steps"],
        optimizer_updates=progress["completed_optimizer_steps"]
        - progress["batch_initial_optimizer_steps"],
        optimizer_updates_this_invocation=progress["completed_optimizer_steps"] - start_steps,
        planned_minibatches=updates,
        trajectory_minibatch_size=getattr(args, "trajectory_minibatch_size", 4),
        mixed_task_minibatches=bool(getattr(args, "mixed_task_minibatches", False)),
        planned_trajectory_uses=sum(map(len, schedule)),
        skipped_minibatches=skipped,
        training_trajectory_uses=len(executed),
        unique_training_trajectories=len(set(executed)),
        repeated_training_trajectory_uses=len(executed) - len(set(executed)),
        selected_parent_request_uses=sum(len(admitted[key]["requests"]) for key in executed),
        backprop_request_uses=sum(
            row["advantage"] != 0 for key in executed for row in admitted[key]["requests"]
        ),
        backprop_prompt_token_uses=sum(
            len(row["prompt_token_ids"])
            for key in executed
            for row in admitted[key]["requests"]
            if row["advantage"] != 0
        ),
        backprop_response_token_uses=sum(
            len(row["response_token_ids"])
            for key in executed
            for row in admitted[key]["requests"]
            if row["advantage"] != 0
        ),
        trainable_sha256_before=before,
        trainable_sha256_after=trainable_hash(model),
        old_logprob_source="same HF raw token distribution as updated actor",
        sampling_policy="native temperature1 top_p1 top_k-1 repetition_penalty1",
        sampled_adapter_sha256=source_adapter,
        sampled_collection_seed=sampled_collection_seed,
        old_logprob_request_reads=len(old_read_rows),
        old_logprob_prompt_token_reads=sum(len(row["prompt_token_ids"]) for row in old_read_rows),
        old_logprob_response_token_reads=sum(
            len(row["response_token_ids"]) for row in old_read_rows
        ),
        state_output=str(state_path),
        adapter=str(args.actor_output or args.output / "adapter"),
        elapsed_training_seconds=time.monotonic() - started,
        **cuda_memory(args, torch),
    )
    if probability_diagnostic is not None:
        summary["request_probability_diagnostic"] = probability_diagnostic
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sft", "rl"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=43)
        command.add_argument("--device", default="cuda")
        command.add_argument(
            "--learning-rate", "--lr", type=float, default=5e-5 if name == "sft" else 2e-6
        )
        command.add_argument("--resume-state", type=Path)
        command.add_argument("--state-output", type=Path)
        command.add_argument("--actor-output", type=Path)
        command.add_argument("--checkpoint-every", type=int, default=16)
        command.add_argument(
            "--gradient-checkpointing",
            action="store_true",
            help="Trade extra computation for lower activation memory; default off",
        )
        if name == "sft":
            command.add_argument("--model", type=Path, required=True)
            command.add_argument("--adapter", type=Path, help="Continue an existing LoRA adapter")
            command.add_argument("--epochs", type=int, default=1)
            command.add_argument("--micro-batch", type=int, default=1)
            command.add_argument("--gradient-accumulation", type=int, default=16)
            command.add_argument(
                "--max-training-seconds", type=float,
                help="Cumulative SFT stage time cap including loading; stop at update boundaries",
            )
            command.add_argument("--max-length", type=int, default=16384)
            command.add_argument(
                "--auto-request-checkpointing", action="store_true",
                help="Enable activation checkpointing only for SFT sequences longer than4096",
            )
        else:
            command.add_argument("--base-model", type=Path, required=True)
            command.add_argument("--adapter", type=Path, required=True)
            command.add_argument(
                "--arm", choices=("rloo", "sham", "m3", "shuffled", "mc_uniform", "mc_priority"),
                default="rloo",
            )
            command.add_argument("--updates", type=int)
            command.add_argument("--trajectory-minibatch-size", type=int, default=4)
            command.add_argument(
                "--collection-seed", type=int,
                help="Sampler seed for this batch; --seed remains the optimizer/recovery seed",
            )
            command.add_argument("--diagnostic-request-limit", type=int, default=0)
            command.add_argument(
                "--mixed-task-minibatches", action="store_true",
                help="One parent per task and replicate per update; no parent repetition",
            )
            command.add_argument("--clip-epsilon", type=float, default=0.2)
            command.add_argument("--local-credit", type=Path)
            command.add_argument("--local-weight", type=float, default=0.25)
            command.add_argument("--local-credit-target", choices=("reward", "business"))
            command.add_argument("--mc-baselines", type=Path)
    args = parser.parse_args()
    require(args.learning_rate > 0 and finite(args.learning_rate), "positive finite LR")
    require(args.checkpoint_every > 0, "positive checkpoint interval")
    started = now()
    try:
        summary = sft(args) if args.command == "sft" else rl(args)
        print(
            canonical(
                dict(
                    status=summary["status"], started_at=started, output=str(args.output.resolve())
                )
            ),
            flush=True,
        )
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "failure.json",
            dict(
                status="FAILED",
                started_at=started,
                error=f"{type(error).__name__}: {error}",
            ),
        )
        raise


if __name__ == "__main__":
    main()
