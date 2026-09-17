"""Validate saved decode evidence and score it once in a separate worker."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/m2-decode-recovery-20260912"
PRIOR = ROOT / "runs/m2-decode-paired-20260912-v3"
spec = importlib.util.spec_from_file_location(
    "decode_recovery_alignment", ROOT / "scripts/diagnostics/m2_logprob_alignment.py"
)
assert spec and spec.loader
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
P = A.P


def completed_response(raw: dict[str, Any], eos: list[int]) -> list[int]:
    A.ensure(len(raw["outputs"]) == 1, "Expected exactly one response")
    out = raw["outputs"][0]
    tokens = out["token_ids"]
    # raw_output intentionally names this raw_finish_reason, not finish_reason.
    A.ensure(P.H.response_complete(tokens, out["raw_finish_reason"], eos), "Incomplete response")
    return tokens


def prepare() -> None:
    m = A.read(PRIOR / "manifest.json")
    for name, h in {**m["source_sha256"], **m["source_file_sha256"]}.items():
        A.ensure(A.sha(ROOT / name) == h, "Prior source changed")
    raw = A.read(PRIOR / "paired/natural.raw.json")["outputs"]
    effective = A.read(PRIOR / "paired/natural.submitted.json")["effective"]
    A.ensure(
        len(raw) == 8 and not (PRIOR / "paired/teacher.submitted.json").exists(),
        "Wrong recovery stage",
    )
    cases = []
    for i, (c, r, e) in enumerate(zip(m["cases"], raw, effective, strict=True)):
        A.ensure(
            r["request_id"] == str(i) and r["prompt_token_ids"] == c["prompt_ids"], "Saved mapping"
        )
        ids = completed_response(r, e["effective_stop_token_ids"])
        cases.append(
            {
                **c,
                "historical_suffix_ids": c["suffix_ids"],
                "suffix_ids": ids,
                "paired_generation_logprobs": [
                    d[str(t)]["logprob"] for t, d in zip(ids, r["output_logprobs"][0], strict=True)
                ],
            }
        )
    sources = dict(m["source_sha256"])
    for p in [
        Path(__file__),
        ROOT / "tests/test_m2_decode_recovery.py",
        ROOT / "docs/2026-09-12_M2_DECODE_RECOVERY_AMENDMENT.md",
    ]:
        sources[str(p.relative_to(ROOT))] = A.sha(p)
    inputs = dict(m["source_file_sha256"])
    for p in PRIOR.rglob("*"):
        if p.is_file() and p.suffix in [".json", ".jsonl", ".log"]:
            inputs[str(p.relative_to(ROOT))] = A.sha(p)
    manifest = {
        "status": "PREPARED",
        "at_SGT": P.now(),
        "jobs": ["score_saved"],
        "model": m["model"],
        "cases": cases,
        "source_sha256": sources,
        "source_file_sha256": inputs,
        "package_versions": m["package_versions"],
        "budget": {
            "natural_requests": 0,
            "teacher_requests": 8,
            "auxiliary_tokens_max": 8,
            "environment_actions": 0,
        },
        "same_worker_pairing": "INCOMPLETE; original worker exited before scoring",
    }
    P.write(BASE / "manifest.json", manifest)
    P.write(
        BASE / "preparation.json",
        {
            "status": "PASSED",
            "at_SGT": P.now(),
            "manifest_sha256": A.sha(BASE / "manifest.json"),
            "fixed_saved_outputs": 8,
            "saved_tokens": sum(len(c["suffix_ids"]) for c in cases),
            "new_natural_requests": 0,
        },
    )


def run(expected: str) -> None:
    A.ensure(A.sha(BASE / "manifest.json") == expected, "Manifest changed")
    m = A.read(BASE / "manifest.json")
    folder = BASE / "score_saved"
    folder.mkdir(exist_ok=False)
    state = {
        "status": "STARTING",
        "at_SGT": P.now(),
        "manifest_sha256": expected,
        "natural_requests": 0,
        "environment_actions": 0,
        "training_updates": 0,
    }
    P.H.set_state(folder, state)
    try:
        for name, h in {**m["source_sha256"], **m["source_file_sha256"]}.items():
            A.ensure(A.sha(ROOT / name) == h, "Frozen file changed: " + name)
        A.ensure(P.H.package_versions() == m["package_versions"], "Version changed")
        for name, h in {**m["model"]["metadata_sha256"], **m["model"]["weight_sha256"]}.items():
            A.ensure(A.sha(Path(m["model"]["snapshot"]) / name) == h, "Model changed")
        A.ensure(not any(k.startswith("FINANCE_") for k in os.environ), "Training hooks inherited")
        P.H.add_import_paths()
        from vllm import LLM, SamplingParams

        llm = LLM(**m["model"]["engine_kwargs"])
        state.update(status="RUNNING", engine_ready_at_SGT=P.now())
        P.H.set_state(folder, state)
        plans = [
            {
                "case_index": i,
                "request_index": i,
                "input_ids": c["prompt_ids"] + c["suffix_ids"],
                "suffix_indices": list(range(len(c["suffix_ids"]))),
            }
            for i, c in enumerate(m["cases"])
        ]
        params = [
            SamplingParams(
                temperature=1, top_p=1, top_k=-1, seed=42, max_tokens=1, prompt_logprobs=1
            )
            for _ in plans
        ]
        P.write(
            folder / "teacher.submitted.json", {"plans": plans, "params": [repr(p) for p in params]}
        )
        outputs = llm.generate(
            [{"prompt_token_ids": p["input_ids"]} for p in plans], params, use_tqdm=False
        )
        A.ensure(len(outputs) == 8, "Missing teacher output")
        raw = [A.selected_vllm(o, p, c) for o, p, c in zip(outputs, plans, m["cases"], strict=True)]
        P.write(folder / "teacher.raw.json", {"outputs": raw, "at_SGT": P.now()})
        rows = []
        for c, o in zip(m["cases"], raw, strict=True):
            i = c["case_index"]
            A.ensure(o["request_id"] == str(i), "Request mapping")
            values = [
                v["distribution"][str(v["target_token_id"])]["logprob"]
                for v in o["selected_raw_logprobs"]
            ]
            A.ensure(
                len(o["outputs"]) == 1 and len(o["outputs"][0]["token_ids"]) == 1,
                "Auxiliary token count",
            )
            rows.append(
                {
                    "case_index": i,
                    "teacher_logprobs": values,
                    "comparison": A.compare(values, c["paired_generation_logprobs"]),
                    "first_token_difference": abs(values[0] - c["paired_generation_logprobs"][0]),
                    "decode_max_difference": max(
                        abs(x - y)
                        for x, y in zip(
                            values[1:], c["paired_generation_logprobs"][1:], strict=True
                        )
                    ),
                }
            )
        for name, h in {**m["source_sha256"], **m["source_file_sha256"]}.items():
            A.ensure(A.sha(ROOT / name) == h, "Source changed after execution")
        state.update(
            status="OBSERVED",
            rows=rows,
            teacher_requests=8,
            auxiliary_tokens=8,
            finished_at_SGT=P.now(),
            evidence_files_sha256={
                p.name: A.sha(p) for p in folder.glob("*.json") if p.name != "state.json"
            },
        )
        P.write(folder / "result.json", state)
        P.H.set_state(folder, state)
    except BaseException as e:
        state.update(status="STOPPED", error=repr(e), finished_at_SGT=P.now())
        P.H.set_state(folder, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--job", choices=["score_saved"])
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.expected_manifest_sha256)
