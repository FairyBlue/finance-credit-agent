#!/usr/bin/env python3
"""Frozen 4B one-step action support diagnostic: sampling, scoring and CPU replay."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/m2-action-support-diagnostic-20260912"
HERE = BASE / "probe-v1"
PLAN = ROOT / "docs/2026-09-12_M2_ACTION_SUPPORT_DIAGNOSTIC_PLAN.md"
PRIOR = ROOT / "runs/m2-transfer-diagnostic-20260912/trajectory-v1/manifest.json"
PRIOR_SHA = "c3704245113710fbe39cfb0ac60764e2683b1549a2ec7176dfcf647013fb0d17"
TRANSFER = ROOT / "scripts/diagnostics/m2_transfer_probe.py"
TRANSFER_SHA = "4303c58dbd4b89053dcd528bfe7ed93c7d2cfbb15038a93e979e9d870d49e032"
CONDITIONS = ("features", "features_defined", "features_objective", "features_defined_objective")
NAMES = ("buy", "sell", "hold", "abstain", "compute", "retrieve")
COMMON = '<tool_call>\n{"name": "'
EOS = 151645


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if sha(TRANSFER) != TRANSFER_SHA:
    raise RuntimeError("Frozen transfer helper changed")
T = load("support_transfer_helpers", TRANSFER)
H = T.H


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def ensure(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def write(path: Path, value: Any) -> None:
    H.write_new(path, value)


def read(path: Path) -> dict[str, Any]:
    return H.read(path)


def candidate_actions(symbols: list[str]) -> list[dict[str, Any]]:
    ensure(len(symbols) == len(set(symbols)) == 20, "Expected original 20 symbols")
    values: list[tuple[str, dict[str, Any]]] = [("hold", {}), ("abstain", {})]
    values += [("buy", {"symbol": s, "target_weight": 0.1}) for s in symbols]
    values += [("compute", {"metric": "volatility", "symbol": s, "lookback": 5}) for s in symbols]
    return [
        {
            "candidate_index": i,
            "name": name,
            "arguments": args,
            "text": "<tool_call>\n"
            + json.dumps({"name": name, "arguments": args})
            + "\n</tool_call>",
        }
        for i, (name, args) in enumerate(values)
    ]


def grid_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in range(2):
        for condition in CONDITIONS:
            for sibling in range(128):
                identity = {
                    "step": 1,
                    "sample_index": window,
                    "rollout_n": sibling,
                    "validate": False,
                }
                seed = T.E.request_seed({"identity": identity, "request_limit": 1}, 0)
                rows.append(
                    {
                        "request_index": len(rows),
                        "window_index": window,
                        "condition": condition,
                        "sibling": sibling,
                        "prompt_index": window * 4 + CONDITIONS.index(condition),
                        "identity": identity,
                        "seed_env_step": 0,
                        "seed": seed,
                    }
                )
    return rows


def token_records(ids: list[int], distributions: Any) -> list[dict[str, Any]]:
    ensure(distributions is not None and len(ids) == len(distributions), "Missing logprobs")
    records = []
    for token, distribution in zip(ids, distributions, strict=True):
        ensure(distribution is not None and token in distribution, "Missing target token")
        lp = float(distribution[token].logprob)
        ensure(math.isfinite(lp) and lp <= 1e-6, "Invalid target logprob")
        records.append({"token_id": token, "logprob": lp, "rank": distribution[token].rank})
    return records


def suffix_logprobs(prompt: list[int], suffix: list[int], output: Any) -> list[dict[str, Any]]:
    ensure(
        prompt and suffix and list(output.prompt_token_ids) == prompt + suffix,
        "Teacher-forcing boundary or identity mismatch",
    )
    ensure(
        output.prompt_logprobs is not None
        and len(output.prompt_logprobs) == len(prompt) + len(suffix),
        "Truncated prompt logprobs",
    )
    return token_records(suffix, output.prompt_logprobs[len(prompt) :])


def calibration(sample: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    ensure(
        sample and [r["token_id"] for r in sample] == [r["token_id"] for r in scored],
        "Calibration token identity mismatch",
    )
    differences = [abs(a["logprob"] - b["logprob"]) for a, b in zip(sample, scored, strict=True)]
    total = abs(math.fsum(x["logprob"] for x in sample) - math.fsum(x["logprob"] for x in scored))
    return {
        "status": "PASSED" if max(differences) <= 0.1 and total <= 0.5 else "FAILED",
        "maximum_token_abs_difference": max(differences),
        "sum_abs_difference": total,
        "token_tolerance": 0.1,
        "sum_tolerance": 0.5,
    }


def raw_distributions(distributions: Any) -> Any:
    if distributions is None:
        return None
    return [
        None
        if dist is None
        else {str(k): {"logprob": float(v.logprob), "rank": v.rank} for k, v in dist.items()}
        for dist in distributions
    ]


def natural_kwargs(plan: dict[str, Any]) -> dict[str, Any]:
    return {**H.sampling_kwargs(plan), "logprobs": 1}


def effective_natural(llm: Any, sampling: Any) -> dict[str, Any]:
    processor = llm.llm_engine.processor
    eos = processor.input_preprocessor.get_eos_token_id(None)
    effective = sampling.clone()
    effective.update_from_generation_config(processor.generation_config_fields, eos)
    effective.update_from_tokenizer(processor.tokenizer.get_lora_tokenizer(None))
    ensure(
        eos == EOS and set(effective.all_stop_token_ids) == H.EOS_IDS and not effective.ignore_eos,
        "Effective EOS contract drift",
    )
    fields = {key: getattr(effective, key) for key in natural_kwargs({"seed": sampling.seed})}
    ensure(fields == natural_kwargs({"seed": sampling.seed}), "Effective sampling drift")
    return {
        "sampling_params": fields,
        "effective_stop_token_ids": sorted(effective.all_stop_token_ids),
        "tokenizer_eos_token_id": eos,
        "effective_sampling_repr": repr(effective),
        "generation_config_fields": dict(processor.generation_config_fields),
    }


def budget() -> dict[str, Any]:
    return {
        "natural_requests": 1024,
        "fixed_score_requests": 384,
        "maximum_calibration_requests": 8,
        "maximum_engine_requests": 1416,
        "maximum_generated_tokens": 4194696,
        "prompt_cap": 8192,
        "response_cap": 4096,
        "context_cap": 12288,
        "batch_size": 32,
        "full_trajectories": 0,
        "training_updates": 0,
    }


def helpers() -> Any:
    H.add_import_paths()
    return load("support_frozen_conditions", ROOT / "scripts/diagnostics/m2_transfer_conditions.py")


async def build_env(c: Any, rows: Any, prompt: dict[str, Any], tokenizer: Any) -> Any:
    env = c.create_env(rows, prompt["window_index"], prompt["condition"], 4)
    obs, receipt = await c.reset_with_prefix(env)
    T.validate_prefix(env, receipt, {"condition": prompt["condition"], "start_session": 4})
    ensure(
        c.render_prompt(tokenizer, obs, env.tool_schemas) == prompt["input"],
        "Actual fixed state/prompt drift",
    )
    return env


async def apply_once(
    c: Any, rows: Any, prompt: dict[str, Any], tokenizer: Any, text: str, ids: list[int]
) -> dict[str, Any]:
    from agent_r1.env.base import Action

    env = await build_env(c, rows, prompt, tokenizer)
    before = {
        "cash": env.ledger.cash,
        "positions": dict(env.ledger.positions),
        "session": env.current_session_index,
        "date": env.current_date.isoformat(),
    }
    await env.step(Action(text=text, token_ids=ids))
    ensure(len(env.credit_step_trace) == len(env.interaction_trace) == 5, "One-step scope exceeded")
    ensure(not env.episode_complete and env.current_session_index in (4, 5), "Unexpected terminal")
    step = asdict(env.credit_step_trace[-1])
    return {
        "status": "APPLIED",
        "scripted_prefix_actions": 4,
        "model_or_candidate_actions": 1,
        "before": before,
        "after": {
            "cash": env.ledger.cash,
            "positions": dict(env.ledger.positions),
            "session": env.current_session_index,
            "date": env.current_date.isoformat(),
        },
        "step": step,
        "interaction": env.interaction_trace[-1],
        "fills": [asdict(f) for f in env.ledger.fills],
    }


def validate_manifest(m: dict[str, Any]) -> None:
    ensure(
        m["status"] == "PREPARED" and m["diagnostic_id"] == "m2_action_support_v1", "Wrong manifest"
    )
    ensure(m["grid"] == grid_plan() and m["budget"] == budget(), "Population or budget drift")
    ensure(sha(PRIOR) == PRIOR_SHA and m["prior_manifest_sha256"] == PRIOR_SHA, "Prior drift")
    old = read(PRIOR)
    ensure(m["model"] == old["models"]["screening"], "Model/engine drift")
    ensure(len(m["prompts"]) == 8 and len(m["score_grid"]) == 384, "Score population drift")
    for i, prompt in enumerate(m["prompts"]):
        window, ci = divmod(i, 4)
        ensure(
            prompt["window_index"] == window and prompt["condition"] == CONDITIONS[ci],
            "Input order",
        )
        ensure(
            prompt["input"] == m["model"]["initial_prompts"][CONDITIONS[ci]][window], "Prompt drift"
        )
        ensure(0 < len(prompt["input"]["prompt_ids"]) <= 8192, "Input cap")
    for path, expected in m["source_sha256"].items():
        ensure(sha(ROOT / path) == expected, f"Source drift: {path}")


def prepare() -> None:
    ensure(not HERE.exists(), "Refusing existing preparation")
    os.environ.update(
        USE_TORCH="0",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    c = helpers()
    from transformers import AutoTokenizer

    ensure(sha(PRIOR) == PRIOR_SHA, "Prior manifest drift")
    prior = read(PRIOR)
    model = copy.deepcopy(prior["models"]["screening"])
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot"], local_files_only=True)
    rows = H.read_train_rows()
    prompts: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    legal_receipts: list[dict[str, Any]] = []
    for window in range(2):
        for condition in CONDITIONS:
            prompt = {
                "prompt_index": len(prompts),
                "window_index": window,
                "condition": condition,
                "input": model["initial_prompts"][condition][window],
            }
            env = asyncio.run(build_env(c, rows, prompt, tokenizer))
            symbols = list(env.symbols)
            prompts.append(prompt)
            for candidate in candidate_actions(symbols):
                ids = list(tokenizer.encode(candidate["text"], add_special_tokens=False)) + [EOS]
                ensure(tokenizer.decode(ids[:-1]) == candidate["text"], "Candidate token boundary")
                proof = asyncio.run(apply_once(c, rows, prompt, tokenizer, candidate["text"], ids))
                ensure(not proof["step"]["invalid_action"], "Registered candidate is not legal")
                ensure(
                    len(proof["fills"]) == int(candidate["name"] == "buy"),
                    "Candidate fill mismatch",
                )
                legal_receipts.append(
                    {
                        "prompt_index": prompt["prompt_index"],
                        "candidate_index": candidate["candidate_index"],
                        **proof,
                    }
                )
                scores.append(
                    {
                        "score_index": len(scores),
                        "kind": "canonical_action",
                        "prompt_index": prompt["prompt_index"],
                        **candidate,
                        "suffix_ids": ids,
                    }
                )
            common_ids = list(tokenizer.encode(COMMON, add_special_tokens=False))
            for name in NAMES:
                text = COMMON + name + '",'
                ids = list(tokenizer.encode(text, add_special_tokens=False))
                ensure(ids[: len(common_ids)] == common_ids, "Name prefix token boundary")
                scores.append(
                    {
                        "score_index": len(scores),
                        "kind": "tool_name_path",
                        "prompt_index": prompt["prompt_index"],
                        "name": name,
                        "text": text,
                        "suffix_ids": ids,
                        "common_prefix_token_count": len(common_ids),
                    }
                )
    source = dict(prior["source_identity"]["source_sha256"])
    paths = [Path(__file__), PLAN, ROOT / "tests/test_m2_action_support_probe.py"]
    vllm_root = ROOT / ".venv/lib/python3.10/site-packages/vllm"
    paths += [
        vllm_root / p
        for p in (
            "v1/core/kv_cache_manager.py",
            "v1/engine/logprobs.py",
            "v1/engine/processor.py",
            "sampling_params.py",
        )
    ]
    source.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    m = {
        "status": "PREPARED",
        "diagnostic_id": "m2_action_support_v1",
        "at_SGT": now(),
        "prior_manifest_sha256": PRIOR_SHA,
        "model": model,
        "prompts": prompts,
        "grid": grid_plan(),
        "score_grid": scores,
        "budget": budget(),
        "source_sha256": source,
        "source_file_sha256": {
            str(PRIOR.relative_to(ROOT)): PRIOR_SHA,
            **prior["source_file_sha256"],
        },
        "package_versions": H.package_versions(),
        "name_path_common_prefix": COMMON,
        "calibration": {
            "sibling": 0,
            "maximum_token_abs_difference": 0.1,
            "sum_abs_difference": 0.5,
        },
    }
    validate_manifest(m)
    ensure(
        not any(name in sys.modules for name in ("torch", "ray", "vllm")),
        "CPU imported GPU framework",
    )
    HERE.mkdir(parents=True)
    write(HERE / "manifest.json", m)
    write(
        HERE / "candidate_legality.json",
        {
            "status": "PASSED",
            "receipts": legal_receipts,
            "candidate_environment_actions": len(legal_receipts),
            "scripted_prefix_actions": 4 * (len(prompts) + len(legal_receipts)),
            "model_requests": 0,
        },
    )
    write(
        HERE / "preparation.json",
        {
            "status": "PASSED",
            "at_SGT": now(),
            "manifest_sha256": sha(HERE / "manifest.json"),
            "prompts_exact": 8,
            "legal_candidates_checked": 336,
            "score_requests": 384,
            "natural_requests": 1024,
            "model_requests_started": 0,
            "gpu_frameworks_imported": False,
        },
    )
    print(json.dumps({"status": "PREPARED", "manifest_sha256": sha(HERE / "manifest.json")}))


def verify(m: dict[str, Any], *, weights: bool = False) -> None:
    validate_manifest(m)
    for path, expected in m["source_file_sha256"].items():
        ensure(sha(ROOT / path) == expected, f"Input drift: {path}")
    ensure(m["package_versions"] == H.package_versions(), "Runtime package drift")
    inventory = dict(m["model"]["metadata_sha256"])
    if weights:
        inventory.update(m["model"]["weight_sha256"])
    for name, expected in inventory.items():
        ensure(sha(Path(m["model"]["snapshot"]) / name) == expected, "Model file drift")


def score_batch(
    llm: Any,
    sampling_class: Any,
    m: dict[str, Any],
    plans: list[dict[str, Any]],
    directory: Path,
    index: int,
) -> list[dict[str, Any]]:
    inputs = [
        m["prompts"][p["prompt_index"]]["input"]["prompt_ids"] + p["suffix_ids"] for p in plans
    ]
    ensure(all(len(p) <= 12287 for p in inputs), "Scoring context cap")
    params = [
        sampling_class(
            temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1, prompt_logprobs=1, seed=42
        )
        for _ in plans
    ]
    prefix = f"score-{index:03d}"
    write(
        directory / (prefix + ".submitted.json"),
        {
            "at_SGT": now(),
            "plans": plans,
            "inputs": inputs,
            "sampling_params": [repr(p) for p in params],
        },
    )
    outputs = llm.generate([{"prompt_token_ids": p} for p in inputs], params, use_tqdm=False)
    raw = []
    for output in outputs:
        row = H.raw_output(output)
        row["prompt_logprobs"] = raw_distributions(output.prompt_logprobs)
        raw.append(row)
    write(directory / (prefix + ".raw.json"), {"at_SGT": now(), "outputs": raw})
    ensure(len(outputs) == len(plans), "Missing scoring output")
    ensure(len({o.request_id for o in outputs}) == len(outputs), "Duplicate score request ID")
    ensure(
        [o.request_id for o in outputs] == [str(1024 + p["score_index"]) for p in plans],
        "Score request order mismatch",
    )
    observed = []
    for p, output in zip(plans, outputs, strict=True):
        prompt = m["prompts"][p["prompt_index"]]["input"]["prompt_ids"]
        lps = suffix_logprobs(prompt, p["suffix_ids"], output)
        value = {
            **p,
            "vllm_request_id": output.request_id,
            "target_logprobs": lps,
            "logprob_sum": math.fsum(r["logprob"] for r in lps),
        }
        if p["kind"] == "tool_name_path":
            value["conditional_name_logprob"] = math.fsum(
                r["logprob"] for r in lps[p["common_prefix_token_count"] :]
            )
        if p["kind"] == "calibration":
            value["calibration"] = calibration(p["sample_logprobs"], lps)
        observed.append(value)
    write(directory / (prefix + ".observed.json"), {"at_SGT": now(), "rows": observed})
    return observed


def run(expected: str) -> None:
    ensure(sha(HERE / "manifest.json") == expected, "Manifest identity changed")
    directory = HERE / "model"
    directory.mkdir(exist_ok=False)
    m = read(HERE / "manifest.json")
    state: dict[str, Any] = {
        "status": "STARTING",
        "at_SGT": now(),
        "manifest_sha256": expected,
        "natural_rows": [],
        "score_rows": [],
        "budget": budget(),
        "parameters_updated": False,
    }

    def persist() -> None:
        H.set_state(directory, state)

    persist()
    try:
        verify(m, weights=True)
        ensure(not any(k.startswith("FINANCE_") for k in os.environ), "Inherited training hook")
        ensure(os.environ.get("USE_TORCH", "1") != "0", "CPU environment inherited")
        os.environ.update(VLLM_USE_V1="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        H.add_import_paths()
        from vllm import LLM, SamplingParams

        llm = LLM(**m["model"]["engine_kwargs"])
        tokenizer: Any = llm.get_tokenizer()
        for p in m["prompts"]:
            inp = p["input"]
            actual = tokenizer.apply_chat_template(
                inp["messages"],
                tools=inp["tools"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ensure(list(actual) == inp["prompt_ids"], "Runtime actual prompt mismatch")
        state.update(status="RUNNING", engine_ready_at_SGT=now())
        persist()
        for start in range(0, 1024, 32):
            ensure(sha(HERE / "manifest.json") == expected, "Manifest changed during sampling")
            plans = m["grid"][start : start + 32]
            inputs = [m["prompts"][p["prompt_index"]]["input"]["prompt_ids"] for p in plans]
            params = [SamplingParams(**natural_kwargs(p)) for p in plans]
            effective = [effective_natural(llm, param) for param in params]
            prefix = f"natural-{start // 32:03d}"
            write(
                directory / (prefix + ".submitted.json"),
                {
                    "at_SGT": now(),
                    "plans": plans,
                    "inputs": inputs,
                    "effective": effective,
                    "logprobs": 1,
                },
            )
            outputs = llm.generate(
                [{"prompt_token_ids": p} for p in inputs], params, use_tqdm=False
            )
            raw = []
            for output in outputs:
                value = H.raw_output(output)
                value["output_logprobs"] = [raw_distributions(r.logprobs) for r in output.outputs]
                value["cumulative_logprobs"] = [r.cumulative_logprob for r in output.outputs]
                raw.append(value)
            write(directory / (prefix + ".raw.json"), {"at_SGT": now(), "outputs": raw})
            ensure(len(raw) == len(plans), "Missing natural output")
            ensure(len({r["request_id"] for r in raw}) == len(raw), "Duplicate request ID")
            ensure(
                [r["request_id"] for r in raw] == [str(p["request_index"]) for p in plans],
                "Natural request order mismatch",
            )
            observed = []
            for p, inp, r, output in zip(plans, inputs, raw, outputs, strict=True):
                ensure(
                    r["prompt_token_ids"] == inp and len(r["outputs"]) == 1,
                    "Natural output mapping mismatch",
                )
                response = r["outputs"][0]
                ids = response["token_ids"]
                sample_lps = token_records(ids, output.outputs[0].logprobs)
                complete = H.response_complete(ids, response["raw_finish_reason"], H.EOS_IDS)
                ensure(
                    abs(math.fsum(v["logprob"] for v in sample_lps) - r["cumulative_logprobs"][0])
                    <= 1e-5,
                    "Generation logprob sum mismatch",
                )
                row = {
                    **p,
                    "status": "COMPLETE" if complete else "INCOMPLETE",
                    "vllm_request_id": r["request_id"],
                    "token_ids": ids,
                    "raw_finish_reason": response["raw_finish_reason"],
                    "raw_text": tokenizer.decode(ids, skip_special_tokens=False),
                    "action_text": tokenizer.decode(ids, skip_special_tokens=True),
                    "sample_logprobs": sample_lps,
                }
                observed.append(row)
            write(directory / (prefix + ".observed.json"), {"at_SGT": now(), "rows": observed})
            state["natural_rows"].extend(observed)
            persist()
            print(
                json.dumps({"natural_responses": len(state["natural_rows"]), "at_SGT": now()}),
                flush=True,
            )
        score_plans = copy.deepcopy(m["score_grid"])
        for p in m["prompts"]:
            sample = next(
                r
                for r in state["natural_rows"]
                if r["prompt_index"] == p["prompt_index"] and r["sibling"] == 0
            )
            if sample["status"] == "COMPLETE":
                score_plans.append(
                    {
                        "score_index": len(score_plans),
                        "kind": "calibration",
                        "prompt_index": p["prompt_index"],
                        "natural_request_index": sample["request_index"],
                        "suffix_ids": sample["token_ids"],
                        "sample_logprobs": sample["sample_logprobs"],
                    }
                )
        for start in range(0, len(score_plans), 32):
            state["score_rows"].extend(
                score_batch(
                    llm, SamplingParams, m, score_plans[start : start + 32], directory, start // 32
                )
            )
            persist()
            print(json.dumps({"scored": len(state["score_rows"]), "at_SGT": now()}), flush=True)
        verify(m)
        state.update(
            status="OBSERVED",
            finished_at_SGT=now(),
            environment_actions_during_GPU=0,
            full_trajectories=0,
            training_updates=0,
        )
        state["evidence_files_sha256"] = {
            p.name: sha(p) for p in sorted(directory.glob("*.json")) if p.name != "state.json"
        }
        write(directory / "result.json", state)
        persist()
    except BaseException as error:
        state.update(status="STOPPED", finished_at_SGT=now(), error=repr(error))
        persist()
        raise


def replay() -> None:
    os.environ.update(
        USE_TORCH="0", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    m = read(HERE / "manifest.json")
    verify(m)
    result = read(HERE / "model/result.json")
    ensure(result["status"] == "OBSERVED", "GPU population incomplete")
    target = HERE / "applied"
    target.mkdir(exist_ok=False)
    c = helpers()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(m["model"]["snapshot"], local_files_only=True)
    rows = H.read_train_rows()
    summary = []
    for r in result["natural_rows"]:
        if r["status"] == "COMPLETE":
            proof = asyncio.run(
                apply_once(
                    c,
                    rows,
                    m["prompts"][r["prompt_index"]],
                    tokenizer,
                    r["action_text"],
                    r["token_ids"],
                )
            )
        else:
            proof = {"status": "NOT_APPLIED", "reason": "INCOMPLETE_RESPONSE", "fills": []}
        write(target / f"request-{r['request_index']:04d}.json", proof)
        summary.append(
            {
                "request_index": r["request_index"],
                "prompt_index": r["prompt_index"],
                "response_status": r["status"],
                "application_status": proof["status"],
                "action": proof.get("step", {}).get("action"),
                "invalid": proof.get("step", {}).get("invalid_action"),
                "fills": len(proof["fills"]),
            }
        )
    write(
        HERE / "replay.json",
        {
            "status": "PASSED",
            "at_SGT": now(),
            "model_result_sha256": sha(HERE / "model/result.json"),
            "rows": summary,
            "evidence_files_sha256": {p.name: sha(p) for p in target.glob("*.json")},
            "new_model_requests": 0,
            "full_trajectories": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "replay"])
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":

        def interrupted(signum: int, frame: Any) -> None:
            raise InterruptedError(f"Received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        run(args.expected_manifest_sha256)
    else:
        replay()


if __name__ == "__main__":
    main()
