#!/usr/bin/env python3
"""Bounded same-worker natural decode/teacher-score comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "runs/m2-logprob-alignment-20260912"
BASE = ROOT / "runs/m2-decode-paired-20260912-v2"
OLD_SHA = "91cb3d52e768679b3dfd5c8b8266c5b90203df802eb74a2f679be4e2547addd4"
spec = importlib.util.spec_from_file_location(
    "frozen_alignment", ROOT / "scripts/diagnostics/m2_logprob_alignment.py"
)
assert spec and spec.loader
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
P = A.P


def matched_observations(
    records: list[dict[str, Any]], request: str, prompt_length: int, tokens: list[int]
) -> list[dict[str, Any]]:
    selected = []
    all_rows = [r for call in records for r in call["rows"] if r["request_id"] == request]
    for i, token in enumerate(tokens):
        matches = [
            r
            for r in all_rows
            if r["output_prefix"] == tokens[:i]
            and r["last_input_position"] == prompt_length + i - 1
        ]
        A.ensure(len(matches) == 1, "Missing or duplicate final token observation")
        row = matches[0]
        A.ensure(
            row["prompt_length"] == prompt_length and row["token_id"] == token,
            "Token identity mismatch",
        )
        A.ensure(
            row["computed_before"] + row["scheduled_tokens"] == prompt_length + i,
            "KV boundary mismatch",
        )
        if i > 0:
            A.ensure(
                row["scheduled_tokens"] == 1 and row["computed_before"] == prompt_length + i - 1,
                "Not cached single-token decode",
            )
        selected.append(row)
    # Only incomplete-prefill records may be excluded.
    A.ensure(
        all(
            r in selected
            or (r["output_prefix"] == [] and r["last_input_position"] < prompt_length - 1)
            for r in all_rows
        ),
        "Unexplained sampler observation",
    )
    return selected


def observer_callbacks():
    # Agent-R1 owns a top-level scripts package. Use the diagnostic module directly.
    sys.path.insert(0, str(Path(__file__).parent))
    import m2_decode_observer

    A.ensure(
        Path(m2_decode_observer.__file__).resolve()
        == Path(__file__).with_name("m2_decode_observer.py"),
        "Wrong observer module",
    )
    return m2_decode_observer.install_observer, m2_decode_observer.remove_observer


def verify(m: dict[str, Any]) -> None:
    A.ensure(A.sha(OLD / "final_readout.json") == OLD_SHA, "Prior readout changed")
    for name, h in {**m["source_sha256"], **m["source_file_sha256"]}.items():
        A.ensure(A.sha(ROOT / name) == h, f"Source changed: {name}")
    A.ensure(P.H.package_versions() == m["package_versions"], "Version drift")


def prepare() -> None:
    os.environ.update(USE_TORCH="0", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    A.ensure(A.sha(OLD / "final_readout.json") == OLD_SHA, "Prior readout identity")
    prior = A.read(OLD / "final_readout.json")
    for name, h in prior["files_sha256"].items():
        A.ensure(A.sha(ROOT / name) == h, f"Prior evidence changed: {name}")
    old = A.read(OLD / "manifest.json")
    original = A.read(A.OLD / "probe-v1/manifest.json")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(old["model"]["snapshot"], local_files_only=True)
    cases = []
    for c in old["cases"]:
        i = c["case_index"]
        inp = original["prompts"][i]["input"]
        ids = tokenizer.apply_chat_template(
            inp["messages"],
            tools=inp["tools"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        A.ensure(list(ids) == c["prompt_ids"], "Original input changed")
        plan = next(p for p in original["grid"] if p["prompt_index"] == i and p["sibling"] == 0)
        cases.append({**c, "seed": plan["seed"]})
    sources = dict(old["source_sha256"])
    libs = ROOT / ".venv/lib/python3.10/site-packages/vllm"
    extra = [
        ROOT / "docs/2026-09-12_M2_DECODE_PAIRED_PLAN.md",
        Path(__file__),
        ROOT / "scripts/diagnostics/m2_decode_observer.py",
        ROOT / "tests/test_m2_decode_paired_v2.py",
        ROOT / "docs/2026-09-12_M2_DECODE_PAIRED_V2_AMENDMENT.md",
        libs / "v1/worker/gpu_model_runner.py",
        libs / "v1/worker/gpu_input_batch.py",
        libs / "v1/sample/logits_processor.py",
        libs / "worker/worker_base.py",
    ]
    sources.update({str(p.relative_to(ROOT)): A.sha(p) for p in extra})
    m = {
        "status": "PREPARED",
        "at_SGT": P.now(),
        "cases": cases,
        "jobs": ["paired"],
        "model": old["model"],
        "package_versions": P.H.package_versions(),
        "source_sha256": sources,
        "source_file_sha256": {
            "runs/m2-decode-paired-20260912/failure_receipt.json": A.sha(
                ROOT / "runs/m2-decode-paired-20260912/failure_receipt.json"
            ),
            str((OLD / "manifest.json").relative_to(ROOT)): A.sha(OLD / "manifest.json"),
            str((OLD / "final_readout.json").relative_to(ROOT)): OLD_SHA,
        },
        "budget": {
            "natural_requests": 8,
            "teacher_requests": 8,
            "natural_tokens_max": 32768,
            "aux_tokens_max": 8,
            "environment_actions": 0,
            "training_updates": 0,
        },
        "thresholds": {"token_abs": 0.1, "sum_abs": 0.5, "observer_abs": 1e-5},
        "output_population": "new once-only natural outputs from original eight inputs",
    }
    A.ensure("torch" not in sys.modules, "CPU preparation imported torch")
    P.write(BASE / "manifest.json", m)
    P.write(
        BASE / "preparation.json",
        {
            "status": "PASSED",
            "at_SGT": P.now(),
            "manifest_sha256": A.sha(BASE / "manifest.json"),
            "fixed_inputs": 8,
            "prior_bound_files_verified": len(prior["files_sha256"]),
            "new_model_requests": 0,
        },
    )
    print("PREPARED", A.sha(BASE / "manifest.json"))


def run(expected: str) -> None:
    A.ensure(A.sha(BASE / "manifest.json") == expected, "Manifest changed")
    m = A.read(BASE / "manifest.json")
    verify(m)
    folder = BASE / "paired"
    folder.mkdir(exist_ok=False)
    state: dict[str, Any] = {
        "status": "STARTING",
        "at_SGT": P.now(),
        "manifest_sha256": expected,
        "environment_actions": 0,
        "training_updates": 0,
    }
    P.H.set_state(folder, state)
    try:
        for name, h in {**m["model"]["metadata_sha256"], **m["model"]["weight_sha256"]}.items():
            A.ensure(A.sha(Path(m["model"]["snapshot"]) / name) == h, "Weights changed")
        A.ensure(os.environ.get("USE_TORCH", "1") != "0", "CPU environment inherited")
        A.ensure(not any(k.startswith("FINANCE_") for k in os.environ), "Training hooks inherited")
        os.environ.update(VLLM_USE_V1="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        P.H.add_import_paths()
        from vllm import LLM, SamplingParams

        install_observer, remove_observer = observer_callbacks()

        llm = LLM(**m["model"]["engine_kwargs"])
        state.update(status="RUNNING", engine_ready_at_SGT=P.now())
        P.H.set_state(folder, state)
        params = [SamplingParams(**P.natural_kwargs(c)) for c in m["cases"]]
        P.write(
            folder / "natural.submitted.json",
            {"cases": m["cases"], "effective": [P.effective_natural(llm, p) for p in params]},
        )
        installed = llm.collective_rpc(
            install_observer, timeout=30, args=(str(folder / "observer.jsonl"),)
        )
        P.write(folder / "observer_install.json", installed)
        outputs = llm.generate(
            [{"prompt_token_ids": c["prompt_ids"]} for c in m["cases"]], params, use_tqdm=False
        )
        P.write(folder / "observer_remove.json", llm.collective_rpc(remove_observer, timeout=30))
        raw = []
        for o in outputs:
            value = P.H.raw_output(o)
            value["output_logprobs"] = [P.raw_distributions(r.logprobs) for r in o.outputs]
            value["cumulative_logprobs"] = [r.cumulative_logprob for r in o.outputs]
            raw.append(value)
        P.write(folder / "natural.raw.json", {"outputs": raw, "at_SGT": P.now()})
        A.ensure(len(raw) == 8, "Natural request count mismatch")
        records = [
            json.loads(line) for line in (folder / "observer.jsonl").read_text().splitlines()
        ]
        rows = []
        score_cases = []
        for c, r in zip(m["cases"], raw, strict=True):
            i = c["case_index"]
            A.ensure(
                r["request_id"] == str(i) and r["prompt_token_ids"] == c["prompt_ids"],
                "Input mapping",
            )
            response = r["outputs"][0]
            tokens = response["token_ids"]
            A.ensure(
                len(r["outputs"]) == 1
                and response["finish_reason"] == "stop"
                and tokens[-1] in P.H.EOS_IDS,
                "Incomplete natural output",
            )
            obs = matched_observations(records, str(i), len(c["prompt_ids"]), tokens)
            api = [
                dist[str(t)]["logprob"]
                for t, dist in zip(tokens, r["output_logprobs"][0], strict=True)
            ]
            A.ensure(
                max(abs(x["logprob"] - y) for x, y in zip(obs, api, strict=True)) <= 1e-5,
                "Observer/API mismatch",
            )
            rows.append(
                {
                    "case_index": i,
                    "tokens": tokens,
                    "generation_logprobs": api,
                    "observer_logprobs": [x["logprob"] for x in obs],
                    "same_output_as_history": tokens == c["suffix_ids"],
                    "observer_api_max_difference": max(
                        abs(x["logprob"] - y) for x, y in zip(obs, api, strict=True)
                    ),
                }
            )
            score_cases.append({**c, "suffix_ids": tokens})
        batch = [
            {
                "case_index": c["case_index"],
                "mode": "paired_full",
                "request_index": 8 + c["case_index"],
                "input_ids": c["prompt_ids"] + c["suffix_ids"],
                "suffix_indices": list(range(len(c["suffix_ids"]))),
            }
            for c in score_cases
        ]
        score_params = [
            SamplingParams(
                temperature=1, top_p=1, top_k=-1, seed=42, max_tokens=1, prompt_logprobs=1
            )
            for _ in batch
        ]
        P.write(
            folder / "teacher.submitted.json",
            {"plans": batch, "params": [repr(p) for p in score_params]},
        )
        scored = llm.generate(
            [{"prompt_token_ids": p["input_ids"]} for p in batch], score_params, use_tqdm=False
        )
        A.ensure(len(scored) == 8, "Teacher request count mismatch")
        selected = [
            A.selected_vllm(o, p, c) for o, p, c in zip(scored, batch, score_cases, strict=True)
        ]
        P.write(folder / "teacher.raw.json", {"outputs": selected, "at_SGT": P.now()})
        for row, out in zip(rows, selected, strict=True):
            A.ensure(out["request_id"] == str(8 + row["case_index"]), "Teacher request identity")
            vals = [
                v["distribution"][str(v["target_token_id"])]["logprob"]
                for v in out["selected_raw_logprobs"]
            ]
            row.update(
                teacher_logprobs=vals,
                comparison=A.compare(vals, row["generation_logprobs"]),
                first_token_difference=abs(vals[0] - row["generation_logprobs"][0]),
                decode_max_difference=max(
                    abs(x - y)
                    for x, y in zip(vals[1:], row["generation_logprobs"][1:], strict=True)
                ),
            )
        verify(m)
        state.update(
            status="OBSERVED",
            finished_at_SGT=P.now(),
            rows=rows,
            natural_requests=8,
            teacher_requests=8,
            natural_tokens=sum(len(r["tokens"]) for r in rows),
            auxiliary_tokens=8,
            evidence_files_sha256={
                p.name: A.sha(p) for p in folder.iterdir() if p.is_file() and p.name != "state.json"
            },
        )
        P.write(folder / "result.json", state)
        P.H.set_state(folder, state)
        print(
            json.dumps(
                {
                    "status": "OBSERVED",
                    "passed": sum(r["comparison"]["status"] == "PASSED" for r in rows),
                    "total": 8,
                }
            ),
            flush=True,
        )
    except BaseException as e:
        state.update(status="STOPPED", error=repr(e), finished_at_SGT=P.now())
        P.H.set_state(folder, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--job", choices=["paired"])
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.expected_manifest_sha256)
