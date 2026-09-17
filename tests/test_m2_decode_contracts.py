"""CPU end-to-end contract using the actual shared serializers."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "checked_decode_test", ROOT / "scripts/diagnostics/m2_decode_paired_checked.py"
)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def test_full_run_crosses_actual_natural_and_teacher_serializers(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", tmp_path)
    monkeypatch.setattr(D, "verify", lambda m: None)
    monkeypatch.setattr(D.A, "sha", lambda p: "frozen")
    monkeypatch.setattr(D.P.H, "add_import_paths", lambda: None)
    monkeypatch.setattr(D, "observer_extension_name", lambda: "fixture.Extension")
    monkeypatch.setattr(
        D.P, "effective_natural", lambda llm, p: {"effective_stop_token_ids": [151645]}
    )
    monkeypatch.delenv("USE_TORCH", raising=False)
    for name in list(__import__("os").environ):
        if name.startswith("FINANCE_"):
            monkeypatch.delenv(name)
    cases = [
        {"case_index": i, "seed": 42 + i, "prompt_ids": [11, 12], "suffix_ids": [5, 151645]}
        for i in range(8)
    ]
    manifest = {
        "cases": cases,
        "model": {
            "metadata_sha256": {},
            "weight_sha256": {},
            "engine_kwargs": {},
            "snapshot": "unused",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            self.calls = 0

        def collective_rpc(self, method, timeout, args=()):
            if method == "fca_install_decode_observer":
                self.path = Path(args[0])
                return [{"status": "INSTALLED", "mutation": False}]
            return [{"status": "REMOVED", "sampler_calls": 2}]

        def generate(self, inputs, params, use_tqdm):
            self.calls += 1
            outputs = []
            if self.calls == 1:
                records = []
                for k, t in enumerate([5, 151645]):
                    records.append(
                        {
                            "call_index": k,
                            "rows": [
                                {
                                    "request_id": str(i),
                                    "prompt_length": 2,
                                    "output_prefix": [] if k == 0 else [5],
                                    "last_input_position": 1 + k,
                                    "computed_before": 0 if k == 0 else 2,
                                    "scheduled_tokens": 2 if k == 0 else 1,
                                    "token_id": t,
                                    "logprob": -0.1,
                                }
                                for i in range(8)
                            ],
                        }
                    )
                self.path.write_text("".join(json.dumps(r) + "\n" for r in records))
            for i, inp in enumerate(inputs):
                tokens = [5, 151645] if self.calls == 1 else [9]
                probs = [{t: SimpleNamespace(logprob=-0.1, rank=1)} for t in tokens]
                response = SimpleNamespace(
                    index=0,
                    token_ids=tokens,
                    finish_reason="stop" if self.calls == 1 else "length",
                    stop_reason=None,
                    text="fixture",
                    logprobs=probs,
                    cumulative_logprob=-0.1 * len(tokens),
                )
                ids = inp["prompt_token_ids"]
                prompt_probs = [None] * 2 + [
                    {t: SimpleNamespace(logprob=-0.1, rank=1)} for t in ids[2:]
                ]
                outputs.append(
                    SimpleNamespace(
                        request_id=str(i if self.calls == 1 else i + 8),
                        prompt_token_ids=ids,
                        outputs=[response],
                        prompt_logprobs=prompt_probs,
                    )
                )
            return outputs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=Params))
    D.run("frozen")
    result = json.loads((tmp_path / "paired/result.json").read_text())
    assert result["status"] == "OBSERVED" and len(result["rows"]) == 8
    assert all(r["comparison"]["status"] == "PASSED" for r in result["rows"])
    submitted = json.loads((tmp_path / "paired/teacher.submitted.json").read_text())
    assert all(p["selected_suffix_indices"] == [0, 1] for p in submitted["plans"])
    natural = json.loads((tmp_path / "paired/natural.raw.json").read_text())
    assert natural["outputs"][0]["outputs"][0]["raw_finish_reason"] == "stop"
