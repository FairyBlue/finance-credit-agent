#!/usr/bin/env python3
"""Persistent native vLLM JSONL worker; no environment or source-tree imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

EOS = (151645, 151643)
MAX_RESPONSE = 4096
MAX_PROMPT = 8192
MAX_REQUESTS = 6904
MAX_BATCH = 32
SAMPLING = dict(
    n=1,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=MAX_RESPONSE,
    ignore_eos=False,
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append(stream: TextIO, value: Any) -> None:
    stream.write(canonical(value) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        append(stream, value)


def emit(protocol: TextIO, value: dict) -> None:
    protocol.write(canonical(value) + "\n")
    protocol.flush()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_metadata(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {
            "__nonfinite_float__": "NaN"
            if math.isnan(value)
            else ("Infinity" if value > 0 else "-Infinity")
        }
    if isinstance(value, (list, tuple)):
        return [safe_metadata(v) for v in value]
    if isinstance(value, dict):
        return {k: safe_metadata(v) for k, v in value.items()}
    return value


def generation_integrity(raw: dict) -> tuple[str, str | None]:
    tokens = raw["response_token_ids"]
    if not tokens:
        return "INCOMPLETE", "EMPTY_RESPONSE"
    if len(tokens) >= MAX_RESPONSE:
        return "INCOMPLETE", "RESPONSE_TOKEN_CAP"
    if raw["returned_finished"] is not True or raw["finish_reason"] != "stop":
        return "INCOMPLETE", "UNKNOWN_OR_NONSTOP_FINISH"
    if tokens[-1] not in EOS or any(t in EOS for t in tokens[:-1]):
        return "INCOMPLETE", "EOS_IDENTITY_INCONSISTENT"
    stop = raw["stop_reason"]
    if stop is not None and (type(stop) is not int or stop != tokens[-1]):
        return "INCOMPLETE", "STOP_REASON_INCONSISTENT"
    return "COMPLETE", None


def engine_settings(model: str) -> dict:
    return dict(
        model=model,
        tokenizer=model,
        dtype="bfloat16",
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=MAX_PROMPT + MAX_RESPONSE,
        max_num_seqs=MAX_BATCH,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.35,
        disable_custom_all_reduce=True,
        enable_sleep_mode=False,
        enforce_eager=False,
        seed=42,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
        override_generation_config={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_new_tokens": MAX_RESPONSE,
        },
    )


def resolve_compiler() -> tuple[str, str]:
    """Same compiler search as launcher, without importing its environment tree."""
    prefix = Path(sys.base_prefix) / "bin"
    candidates = [
        os.environ.get("CC"),
        prefix / "x86_64-conda-linux-gnu-gcc",
        prefix / "x86_64-conda-linux-gnu-cc",
        "cc",
        "gcc",
        "clang",
        "x86_64-conda-linux-gnu-gcc",
        "x86_64-conda-linux-gnu-cc",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(str(candidate))
        if not resolved:
            continue
        try:
            result = subprocess.run(
                [resolved, "--version"], check=True, capture_output=True, text=True, timeout=10
            )
            lines = (result.stdout or result.stderr).splitlines()
            if lines:
                return str(Path(resolved).resolve()), lines[0]
        except (OSError, subprocess.SubprocessError):
            continue
    raise RuntimeError("working host C compiler required for Triton")


def serve(
    output: Path,
    contract: dict,
    llm: Any,
    tokenizer: Any,
    sampling_class: Any,
    incoming: TextIO,
    protocol: TextIO,
) -> None:
    """Run RPC with injected backend objects, also usable for CPU protocol fixtures."""
    limit = contract["request_limit"]
    require(type(limit) is int and 0 < limit <= MAX_REQUESTS, "invalid request_limit")
    require(tokenizer.eos_token_id == EOS[0], "primary EOS differs")
    count = complete = batches = 0
    keys: set[str] = set()
    seeds: set[int] = set()
    last_batch = -1
    try:
        with (
            (output / "submissions.jsonl").open("x") as sh,
            (output / "raw_generations.jsonl").open("x") as rh,
            (output / "generations.jsonl").open("x") as gh,
        ):
            emit(protocol, {"event": "READY"})
            for line in incoming:
                command = json.loads(line)
                require(isinstance(command, dict), "RPC object required")
                if command.get("op") == "close":
                    break
                require(command.get("op") == "generate", "unknown RPC operation")
                batch_id, batch = command["batch_id"], command["requests"]
                require(type(batch_id) is int and batch_id > last_batch, "batch_id not increasing")
                require(
                    isinstance(batch, list) and 0 < len(batch) <= MAX_BATCH, "invalid batch size"
                )
                require(count + len(batch) <= limit, "request limit exceeded")
                new_keys, new_seeds = set(), set()
                for item in batch:
                    require(isinstance(item, dict), "request object required")
                    key, seed, ids = (
                        item["request_key"],
                        item["request_seed"],
                        item["prompt_token_ids"],
                    )
                    require(isinstance(key, str) and bool(key), "invalid request_key")
                    require(key not in keys | new_keys, "duplicate request_key")
                    require(type(seed) is int and 0 <= seed < 2**63, "invalid 63-bit seed")
                    require(seed not in seeds | new_seeds, "duplicate request seed")
                    require(
                        isinstance(ids, list)
                        and 0 < len(ids) <= MAX_PROMPT
                        and all(type(v) is int and v >= 0 for v in ids),
                        "invalid prompt IDs",
                    )
                    canonical(item)
                    new_keys.add(key)
                    new_seeds.add(seed)
                params = [{**SAMPLING, "seed": item["request_seed"]} for item in batch]
                objects = [sampling_class(**v) for v in params]
                require(
                    all(
                        {k: getattr(obj, k) for k in expected} == expected
                        for obj, expected in zip(objects, params, strict=True)
                    ),
                    "sampling parameters changed",
                )
                append(
                    sh,
                    {
                        "batch_id": batch_id,
                        "batch_index": batches,
                        "requests": [
                            {**v, "actual_sampling_params": p}
                            for v, p in zip(batch, params, strict=True)
                        ],
                    },
                )
                start = count
                count += len(batch)
                keys.update(new_keys)
                seeds.update(new_seeds)
                returned = llm.generate(
                    [{"prompt_token_ids": v["prompt_token_ids"]} for v in batch],
                    objects,
                    use_tqdm=False,
                )
                raw = [
                    {
                        "request_id": v.request_id,
                        "finished": v.finished,
                        "prompt_token_ids": list(v.prompt_token_ids),
                        "outputs": [
                            {
                                "token_ids": list(o.token_ids),
                                "engine_text": o.text,
                                "finish_reason": o.finish_reason,
                                "stop_reason": o.stop_reason,
                            }
                            for o in v.outputs
                        ],
                    }
                    for v in returned
                ]
                # Persist the entire native batch before any integrity classification.
                append(
                    rh,
                    {"batch_id": batch_id, "batch_index": batches, "outputs": safe_metadata(raw)},
                )
                require(len(raw) == len(batch), "native result count differs")
                require(
                    [v["request_id"] for v in raw] == [str(i) for i in range(start, count)],
                    "native global request IDs differ",
                )
                records = []
                for inp, result, actual in zip(batch, raw, params, strict=True):
                    require(
                        result["prompt_token_ids"] == inp["prompt_token_ids"],
                        "native prompt differs",
                    )
                    require(len(result["outputs"]) == 1, "native result is not n=1")
                    answer = result["outputs"][0]
                    tokens = answer["token_ids"]
                    record = {
                        **inp,
                        "response_text": tokenizer.decode(tokens, skip_special_tokens=True),
                        "engine_text": answer["engine_text"],
                        "response_token_ids": tokens,
                        "response_token_count": len(tokens),
                        "response_last_token_id": tokens[-1] if tokens else None,
                        "primary_eos_token_id": tokenizer.eos_token_id,
                        "eos_token_ids": list(EOS),
                        "eos_positions": [i for i, t in enumerate(tokens) if t in EOS],
                        "last_token_is_eos": bool(tokens) and tokens[-1] in EOS,
                        "returned_finished": result["finished"],
                        "returned_prompt_token_ids": result["prompt_token_ids"],
                        "finish_reason": answer["finish_reason"],
                        "stop_reason": answer["stop_reason"],
                        "vllm_request_id": result["request_id"],
                        "actual_sampling_params": actual,
                    }
                    record["generation_status"], record["failure_reason"] = generation_integrity(
                        record
                    )
                    record = safe_metadata(record)
                    append(gh, record)
                    records.append(record)
                    complete += record["generation_status"] == "COMPLETE"
                last_batch = batch_id
                batches += 1
                emit(protocol, {"event": "BATCH", "batch_id": batch_id, "records": records})
            else:
                raise EOFError("RPC input ended without explicit close")
        completion = {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
            "model_requests": count,
            "complete_generations": complete,
            "incomplete_generations": count - complete,
            "batches": batches,
            "request_limit": limit,
            "optimizer_updates": 0,
            "automatic_retry": False,
            "output_sha256": {
                name: sha(output / name)
                for name in ("submissions.jsonl", "raw_generations.jsonl", "generations.jsonl")
            },
        }
        write_new(output / "worker_completion.json", completion)
        emit(
            protocol,
            {
                "event": "CLOSED",
                "model_requests": count,
                "complete_generations": complete,
                "incomplete_generations": count - complete,
            },
        )
    except BaseException as exc:
        write_new(
            output / "worker_failure.json",
            {
                "status": "FAILED",
                "model_requests": count,
                "batches": batches,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "automatic_retry": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    # Duplicate the original pipe before redirecting OS descriptors: subprocess and
    # native-library output must never corrupt the RPC stdout stream.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    with (output / "worker.log").open("x", buffering=1) as log:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            contract_path = output / "contract.json"
            contract = json.loads(contract_path.read_text())
            require(
                type(contract["request_limit"]) is int
                and 0 < contract["request_limit"] <= MAX_REQUESTS,
                "invalid request_limit",
            )
            require(isinstance(contract["model_path"], str), "model_path required")
            for name in ("USE_TORCH", "USE_TF", "USE_FLAX"):
                os.environ.pop(name, None)
            os.environ.update(
                USE_TORCH="1",
                USE_TF="0",
                USE_FLAX="0",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                VLLM_USE_V1="1",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                TOKENIZERS_PARALLELISM="false",
            )
            require(
                not any(
                    k.startswith(("FINANCE_SCREENING_", "FINANCE_LEARNING_", "FINANCE_CRITIC_"))
                    for k in os.environ
                ),
                "training hooks in worker environment",
            )
            compiler, version = resolve_compiler()
            os.environ["CC"] = compiler
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams

            engine = engine_settings(contract["model_path"])
            tokenizer = AutoTokenizer.from_pretrained(
                contract["model_path"], local_files_only=True, trust_remote_code=False
            )
            require(tokenizer.eos_token_id == EOS[0], "primary EOS differs")
            write_new(
                output / "worker_config.json",
                {
                    "engine": engine,
                    "sampling": SAMPLING,
                    "compiler": {"CC": compiler, "version": version},
                    "request_limit": contract["request_limit"],
                    "contract_sha256": sha(contract_path),
                    "worker_source_sha256": sha(Path(__file__)),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "versions": {
                        k: importlib.metadata.version(k) for k in ("torch", "vllm", "transformers")
                    },
                },
            )
            llm = LLM(**engine)
            serve(output, contract, llm, tokenizer, SamplingParams, sys.stdin, protocol)
        except BaseException as exc:
            traceback.print_exc()
            emit(protocol, {"event": "ERROR", "error_type": type(exc).__name__, "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
