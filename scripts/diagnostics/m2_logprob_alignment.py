#!/usr/bin/env python3
"""Fixed-output numerical-path checks; never samples or executes a trading decision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/m2-logprob-alignment-20260912"
PLAN = ROOT / "docs/2026-09-12_M2_LOGPROB_ALIGNMENT_PLAN.md"
OLD = ROOT / "runs/m2-action-support-diagnostic-20260912"
OLD_FINAL_SHA = "d534d31a345383db4e63c0fe0d9a56eba9d9613418f1f320c451158c8b6753d1"
JOBS = ("vllm_common", "vllm_serial", "hf_bf16", "hf_fp32")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def ensure(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load_helper() -> Any:
    ensure(sha(OLD / "final_readout.json") == OLD_FINAL_SHA, "Prior final readout changed")
    source = ROOT / "scripts/diagnostics/m2_action_support_probe.py"
    expected = read(OLD / "final_readout.json")["files_sha256"][str(source.relative_to(ROOT))]
    ensure(sha(source) == expected, "Prior helper changed")
    spec = importlib.util.spec_from_file_location("alignment_frozen_helper", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load_helper()


def positions(prompt_length: int, suffix_length: int) -> list[int]:
    ensure(
        type(prompt_length) is int
        and type(suffix_length) is int
        and prompt_length > 0
        and suffix_length > 0,
        "Nonempty integer token lengths required",
    )
    return list(range(prompt_length - 1, prompt_length + suffix_length - 1))


def compare(left: list[float], right: list[float]) -> dict[str, Any]:
    ensure(left and len(left) == len(right), "Probability population mismatch")
    ensure(all(math.isfinite(v) and v <= 1e-6 for v in left + right), "Invalid logprob")
    differences = [abs(a - b) for a, b in zip(left, right, strict=True)]
    total = abs(math.fsum(left) - math.fsum(right))
    return {
        "status": "PASSED" if max(differences) <= 0.1 and total <= 0.5 else "FAILED",
        "max_token_abs_difference": max(differences),
        "max_token_index": differences.index(max(differences)),
        "sum_abs_difference": total,
        "token_tolerance": 0.1,
        "sum_tolerance": 0.5,
    }


def engine_config(model: dict[str, Any], job: str) -> dict[str, Any]:
    ensure(job in JOBS[:2], "Not a vLLM job")
    config = copy.deepcopy(model["engine_kwargs"])
    if job == "vllm_serial":
        config.update(
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=True,
            max_num_seqs=1,
            compilation_config=0,
        )
    return config


def request_grid(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    def request(case: dict[str, Any], mode: str, end: int | None = None) -> dict[str, Any]:
        suffix = case["suffix_ids"] if end is None else case["suffix_ids"][: end + 1]
        selected = list(range(len(suffix))) if end is None else [end]
        return {
            "case_index": case["case_index"],
            "mode": mode,
            "input_ids": case["prompt_ids"] + suffix,
            "selected_suffix_indices": selected,
        }

    batches = [[request(c, "full_batch") for c in cases]]
    batches += [[request(c, "full_single")] for c in cases]
    batches += [
        [request(c, "prefix_single", i)] for c in cases for i in range(len(c["suffix_ids"]))
    ]
    index = 0
    for batch in batches:
        for row in batch:
            row["request_index"] = index
            index += 1
    return batches


def selected_vllm(output: Any, row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    ensure(list(output.prompt_token_ids) == row["input_ids"], "Actual scoring input mismatch")
    ensure(
        output.prompt_logprobs is not None and len(output.prompt_logprobs) == len(row["input_ids"]),
        "Missing prompt logprobs",
    )
    entries = []
    for i in row["selected_suffix_indices"]:
        at = len(case["prompt_ids"]) + i
        distribution = output.prompt_logprobs[at]
        ensure(distribution is not None, "Missing target distribution")
        entries.append(
            {
                "input_token_position": at,
                "suffix_token_index": i,
                "target_token_id": case["suffix_ids"][i],
                "distribution": {
                    str(k): {"logprob": float(v.logprob), "rank": v.rank}
                    for k, v in distribution.items()
                },
            }
        )
    return {
        "request_id": output.request_id,
        "prompt_token_ids": list(output.prompt_token_ids),
        "prompt_logprobs_length": len(output.prompt_logprobs),
        "selected_raw_logprobs": entries,
        "outputs": P.H.raw_output(output)["outputs"],
    }


def prepare() -> None:
    ensure(not (BASE / "manifest.json").exists(), "Manifest already exists")
    os.environ.update(
        USE_TORCH="0",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    from transformers import AutoTokenizer

    old_final = read(OLD / "final_readout.json")
    for name, expected in old_final["files_sha256"].items():
        ensure(sha(ROOT / name) == expected, f"Prior evidence changed: {name}")
    old_manifest = read(OLD / "probe-v1/manifest.json")
    old_result = read(OLD / "probe-v1/model/result.json")
    tokenizer = AutoTokenizer.from_pretrained(
        old_manifest["model"]["snapshot"], local_files_only=True
    )
    cases = []
    for pi, prompt in enumerate(old_manifest["prompts"]):
        sample = next(
            r for r in old_result["natural_rows"] if r["prompt_index"] == pi and r["sibling"] == 0
        )
        control = next(
            r
            for r in old_result["score_rows"]
            if r["kind"] == "calibration" and r["prompt_index"] == pi
        )
        inp = prompt["input"]
        ids = tokenizer.apply_chat_template(
            inp["messages"],
            tools=inp["tools"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ensure(list(ids) == inp["prompt_ids"], "Frozen input tokenizer mismatch")
        ensure(
            sample["status"] == "COMPLETE" and sample["token_ids"] == control["suffix_ids"],
            "Fixed output identity mismatch",
        )
        ensure(
            tokenizer.decode(sample["token_ids"], skip_special_tokens=False) == sample["raw_text"],
            "Original output decode mismatch",
        )
        cases.append(
            {
                "case_index": pi,
                "condition": prompt["condition"],
                "window_index": prompt["window_index"],
                "natural_request_index": sample["request_index"],
                "prompt_ids": list(ids),
                "suffix_ids": sample["token_ids"],
                "raw_text": sample["raw_text"],
                "historical_generation_logprobs": [v["logprob"] for v in sample["sample_logprobs"]],
                "historical_teacher_logprobs": [v["logprob"] for v in control["target_logprobs"]],
            }
        )
    ensure(sum(len(c["suffix_ids"]) for c in cases) == 234, "Fixed 234-token population drift")
    sources = dict(old_manifest["source_sha256"])
    libs = ROOT / ".venv/lib/python3.10/site-packages"
    paths = [
        Path(__file__),
        PLAN,
        ROOT / "tests/test_m2_logprob_alignment.py",
        libs / "vllm/entrypoints/llm.py",
        libs / "vllm/v1/sample/sampler.py",
        libs / "vllm/v1/sample/ops/topk_topp_sampler.py",
        libs / "transformers/models/qwen3/modeling_qwen3.py",
        libs / "transformers/cache_utils.py",
        libs / "transformers/masking_utils.py",
    ]
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    manifest = {
        "status": "PREPARED",
        "at_SGT": P.now(),
        "diagnostic_id": "m2_logprob_alignment_v1",
        "cases": cases,
        "jobs": list(JOBS),
        "model": old_manifest["model"],
        "vllm_configs": {j: engine_config(old_manifest["model"], j) for j in JOBS[:2]},
        "hf_configs": {
            j: {
                "dtype": "bfloat16" if j == "hf_bf16" else "float32",
                "attn_implementation": "eager",
                "allow_tf32": False,
                "batch_size": 1,
                "matmul_precision": "highest",
            }
            for j in JOBS[2:]
        },
        "vllm_batches": request_grid(cases),
        "source_sha256": sources,
        "source_file_sha256": {
            str(OLD.relative_to(ROOT) / "final_readout.json"): OLD_FINAL_SHA,
            str(OLD.relative_to(ROOT) / "probe-v1/manifest.json"): sha(
                OLD / "probe-v1/manifest.json"
            ),
            str(OLD.relative_to(ROOT) / "probe-v1/model/result.json"): sha(
                OLD / "probe-v1/model/result.json"
            ),
        },
        "package_versions": P.H.package_versions(),
        "budget": {
            "vllm_requests": 500,
            "auxiliary_generated_tokens_max": 500,
            "hf_forward_calls": 484,
            "fixed_output_tokens": 234,
            "natural_decisions": 0,
            "environment_actions": 0,
            "training_updates": 0,
        },
        "thresholds": {"token_abs": 0.1, "sum_abs": 0.5},
        "historical_evidence_files_verified": len(old_final["files_sha256"]),
    }
    ensure(
        not any(k in sys.modules for k in ("torch", "vllm", "ray")), "CPU preparation imported GPU"
    )
    BASE.mkdir(exist_ok=True)
    P.write(BASE / "manifest.json", manifest)
    P.write(
        BASE / "preparation.json",
        {
            "status": "PASSED",
            "at_SGT": P.now(),
            "manifest_sha256": sha(BASE / "manifest.json"),
            "fixed_inputs": 8,
            "fixed_output_tokens": 234,
            "new_model_requests": 0,
            "prior_bound_files_verified": 58,
        },
    )
    print(json.dumps({"status": "PREPARED", "manifest_sha256": sha(BASE / "manifest.json")}))


def verify(m: dict[str, Any]) -> None:
    ensure(
        m["jobs"] == list(JOBS) and m["vllm_batches"] == request_grid(m["cases"]),
        "Fixed grid drift",
    )
    for path, expected in {**m["source_sha256"], **m["source_file_sha256"]}.items():
        ensure(sha(ROOT / path) == expected, f"Source/input drift: {path}")
    ensure(P.H.package_versions() == m["package_versions"], "Library version drift")


def run_vllm(m: dict[str, Any], job: str, folder: Path, state: dict[str, Any]) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(**m["vllm_configs"][job])
    state["engine_ready_at_SGT"] = P.now()
    for bi, batch in enumerate(m["vllm_batches"]):
        params = [
            SamplingParams(
                temperature=1.0, top_p=1.0, top_k=-1, seed=42, max_tokens=1, prompt_logprobs=1
            )
            for _ in batch
        ]
        P.write(
            folder / f"batch-{bi:03d}.submitted.json",
            {"plans": batch, "params": [repr(p) for p in params]},
        )
        outputs = llm.generate(
            [{"prompt_token_ids": r["input_ids"]} for r in batch], params, use_tqdm=False
        )
        ensure(len(outputs) == len(batch), "Missing scoring output")
        raw = [
            selected_vllm(o, r, m["cases"][r["case_index"]])
            for o, r in zip(outputs, batch, strict=True)
        ]
        P.write(folder / f"batch-{bi:03d}.raw.json", {"at_SGT": P.now(), "outputs": raw})
        for row, output in zip(batch, raw, strict=True):
            ensure(output["request_id"] == str(row["request_index"]), "Request mapping drift")
            values = [
                v["distribution"][str(v["target_token_id"])]["logprob"]
                for v in output["selected_raw_logprobs"]
            ]
            ensure(
                all(math.isfinite(v) and v <= 1e-6 for v in values), "Nonfinite target probability"
            )
            state["rows"].append(
                {
                    "case_index": row["case_index"],
                    "mode": row["mode"],
                    "request_index": row["request_index"],
                    "suffix_indices": row["selected_suffix_indices"],
                    "logprobs": values,
                }
            )
        if bi % 25 == 0:
            P.H.set_state(folder, state)
            print(
                json.dumps({"job": job, "requests": len(state["rows"]), "at_SGT": P.now()}),
                flush=True,
            )
    comparisons = []
    for case in m["cases"]:
        for mode in ("full_batch", "full_single", "prefix_single"):
            rows = [
                r
                for r in state["rows"]
                if r["case_index"] == case["case_index"] and r["mode"] == mode
            ]
            indices = [i for r in rows for i in r["suffix_indices"]]
            ensure(
                indices == list(range(len(case["suffix_ids"]))),
                "Missing/duplicate scored positions",
            )
            values = [v for r in rows for v in r["logprobs"]]
            comparisons.append(
                {
                    "case_index": case["case_index"],
                    "mode": mode,
                    "against_historical_generation": compare(
                        values, case["historical_generation_logprobs"]
                    ),
                    "against_historical_teacher": compare(
                        values, case["historical_teacher_logprobs"]
                    ),
                }
            )
    state["comparisons"] = comparisons
    state["engine_requests"] = len(state["rows"])


def hf_value(logits: Any, target: int) -> dict[str, Any]:
    import torch

    values = logits.float()
    lp = torch.log_softmax(values, dim=-1)[target]
    return {
        "token_id": target,
        "logprob": float(lp),
        "target_logit": float(values[target]),
        "logsumexp": float(torch.logsumexp(values, dim=-1)),
        "rank": int((values >= values[target]).sum()),
    }


def run_hf(m: dict[str, Any], job: str, folder: Path, state: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    config = m["hf_configs"][job]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        m["model"]["snapshot"],
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to("cuda").eval()
    ensure(all(p.dtype == dtype for p in model.parameters()), "Unexpected model precision")
    state.update(
        engine_ready_at_SGT=P.now(),
        actual_dtype=str(next(model.parameters()).dtype),
        actual_attention=model.config._attn_implementation,
        forward_calls=0,
    )
    with torch.inference_mode():
        for case in m["cases"]:
            prompt, suffix = case["prompt_ids"], case["suffix_ids"]
            p, s = len(prompt), len(suffix)
            full_input = prompt + suffix[:-1]
            selected = positions(p, s)
            P.write(
                folder / f"case-{case['case_index']}.submitted.json",
                {
                    "case_index": case["case_index"],
                    "full_input_ids": full_input,
                    "prediction_positions": selected,
                    "target_ids": suffix,
                    "prompt_ids": prompt,
                    "cache_positions": list(range(p, p + s - 1)),
                    "dtype": config["dtype"],
                },
            )
            inp = torch.tensor([full_input], device="cuda")
            output = model(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                position_ids=torch.arange(len(full_input), device="cuda")[None, :],
                use_cache=False,
                logits_to_keep=torch.tensor(selected, device="cuda"),
            )
            state["forward_calls"] += 1
            full = [hf_value(output.logits[0, i], token) for i, token in enumerate(suffix)]
            del output, inp
            inp = torch.tensor([prompt], device="cuda")
            output = model(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                position_ids=torch.arange(p, device="cuda")[None, :],
                use_cache=True,
                logits_to_keep=1,
            )
            state["forward_calls"] += 1
            cache = output.past_key_values
            ensure(cache.get_seq_length() == p, "Prompt cache length drift")
            incremental = [hf_value(output.logits[0, -1], suffix[0])]
            del output, inp
            for i in range(1, s):
                at = p + i - 1
                output = model(
                    input_ids=torch.tensor([[suffix[i - 1]]], device="cuda"),
                    attention_mask=torch.ones((1, p + i), dtype=torch.long, device="cuda"),
                    position_ids=torch.tensor([[at]], device="cuda"),
                    cache_position=torch.tensor([at], device="cuda"),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                state["forward_calls"] += 1
                cache = output.past_key_values
                ensure(cache.get_seq_length() == p + i, "Incremental KV cache boundary drift")
                incremental.append(hf_value(output.logits[0, -1], suffix[i]))
                del output
            row = {
                "case_index": case["case_index"],
                "full": full,
                "incremental": incremental,
                "full_vs_incremental": compare(
                    [v["logprob"] for v in full], [v["logprob"] for v in incremental]
                ),
                "full_vs_historical_generation": compare(
                    [v["logprob"] for v in full], case["historical_generation_logprobs"]
                ),
                "incremental_vs_historical_generation": compare(
                    [v["logprob"] for v in incremental], case["historical_generation_logprobs"]
                ),
            }
            P.write(folder / f"case-{case['case_index']}.observed.json", row)
            state["rows"].append(row)
            del cache
            P.H.set_state(folder, state)
            print(
                json.dumps({"job": job, "cases": len(state["rows"]), "at_SGT": P.now()}), flush=True
            )
    ensure(state["forward_calls"] == 242, "HF forward budget drift")


def run(job: str, expected: str) -> None:
    ensure(job in JOBS and sha(BASE / "manifest.json") == expected, "Wrong run identity")
    folder = BASE / job
    folder.mkdir(exist_ok=False)
    state: dict[str, Any] = {
        "status": "STARTING",
        "job": job,
        "at_SGT": P.now(),
        "manifest_sha256": expected,
        "rows": [],
        "natural_decisions": 0,
        "environment_actions": 0,
        "parameters_updated": False,
    }
    P.H.set_state(folder, state)
    try:
        m = read(BASE / "manifest.json")
        verify(m)
        for name, digest in {
            **m["model"]["metadata_sha256"],
            **m["model"]["weight_sha256"],
        }.items():
            ensure(sha(Path(m["model"]["snapshot"]) / name) == digest, "Model file drift")
        ensure(os.environ.get("USE_TORCH", "1") != "0", "CPU-only runtime inherited")
        ensure(not any(k.startswith("FINANCE_") for k in os.environ), "Inherited training hooks")
        os.environ.update(VLLM_USE_V1="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        state["status"] = "RUNNING"
        if job.startswith("vllm"):
            run_vllm(m, job, folder, state)
        else:
            run_hf(m, job, folder, state)
        verify(m)
        state.update(status="OBSERVED", finished_at_SGT=P.now())
        state["evidence_files_sha256"] = {
            p.name: sha(p) for p in folder.glob("*.json") if p.name != "state.json"
        }
        P.write(folder / "result.json", state)
        P.H.set_state(folder, state)
    except BaseException as error:
        state.update(status="STOPPED", error=repr(error), finished_at_SGT=P.now())
        P.H.set_state(folder, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--job", choices=JOBS)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "prepare":
        prepare()
    else:
        run(args.job, args.expected_manifest_sha256)


if __name__ == "__main__":
    main()
