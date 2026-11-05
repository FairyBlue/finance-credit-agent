"""Small CPU fixtures for the persistent native generation protocol."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnostics/m2_continuation_gpu_worker.py"
SPEC = importlib.util.spec_from_file_location("continuation_worker_fixture", PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class Tokenizer:
    eos_token_id = 151645

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "<tool_call>original native text</tool_call>"


class Engine:
    def __init__(self, output, *, wrong_prompt=False):
        self.output = output
        self.count = 0
        self.wrong_prompt = wrong_prompt

    def generate(self, prompts, params, *, use_tqdm):
        assert use_tqdm is False
        # Submission durability precedes the actual model call.
        assert len((self.output / "submissions.jsonl").read_text().splitlines()) >= 1
        results = []
        for prompt, param in zip(prompts, params, strict=True):
            assert param.seed in {10, 11, 12}
            tokens, finish, stop = ([7, 151645], "stop", None)
            if self.count == 1:
                tokens, finish, stop = ([8, 151643], "stop", 151643)
            if self.count == 2:
                tokens, finish, stop = ([9], "length", None)
            results.append(
                SimpleNamespace(
                    request_id=str(self.count),
                    finished=True,
                    prompt_token_ids=[999] if self.wrong_prompt else prompt["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            token_ids=tokens,
                            text="engine native",
                            finish_reason=finish,
                            stop_reason=stop,
                        )
                    ],
                )
            )
            self.count += 1
        return results


def request(index):
    return {
        "request_key": f"trajectory/r{index}",
        "trajectory_key": "trajectory",
        "request_seed": 10 + index,
        "prompt_token_ids": [1, index + 2],
        "session": index,
        "attempt": 0,
        "max_tokens": 4096,
    }


def rpc(*commands):
    return io.StringIO("".join(json.dumps(v) + "\n" for v in commands))


def test_two_batches_preserve_identity_eos_and_incomplete_without_retry(tmp_path):
    protocol = io.StringIO()
    worker.serve(
        tmp_path,
        {"request_limit": 3},
        Engine(tmp_path),
        Tokenizer(),
        SimpleNamespace,
        rpc(
            {"op": "generate", "batch_id": 0, "requests": [request(0), request(1)]},
            {"op": "generate", "batch_id": 1, "requests": [request(2)]},
            {"op": "close"},
        ),
        protocol,
    )
    events = [json.loads(v) for v in protocol.getvalue().splitlines()]
    assert [v["event"] for v in events] == ["READY", "BATCH", "BATCH", "CLOSED"]
    records = events[1]["records"] + events[2]["records"]
    assert [v["vllm_request_id"] for v in records] == ["0", "1", "2"]
    assert [v["generation_status"] for v in records] == ["COMPLETE", "COMPLETE", "INCOMPLETE"]
    assert records[2]["failure_reason"] == "UNKNOWN_OR_NONSTOP_FINISH"
    for i, record in enumerate(records):
        assert all(record[key] == value for key, value in request(i).items())
        assert record["returned_prompt_token_ids"] == request(i)["prompt_token_ids"]
        assert record["response_text"] == "<tool_call>original native text</tool_call>"
        assert record["actual_sampling_params"] == {**worker.SAMPLING, "seed": 10 + i}
    completion = json.loads((tmp_path / "worker_completion.json").read_text())
    assert completion["model_requests"] == 3
    assert completion["complete_generations"] == 2
    assert completion["incomplete_generations"] == 1
    assert len((tmp_path / "raw_generations.jsonl").read_text().splitlines()) == 2
    for name, digest in completion["output_sha256"].items():
        assert worker.sha(tmp_path / name) == digest


def test_wrong_native_prompt_is_saved_then_hard_fails_before_batch_reply(tmp_path):
    protocol = io.StringIO()
    with pytest.raises(ValueError, match="native prompt differs"):
        worker.serve(
            tmp_path,
            {"request_limit": 3},
            Engine(tmp_path, wrong_prompt=True),
            Tokenizer(),
            SimpleNamespace,
            rpc({"op": "generate", "batch_id": 0, "requests": [request(0)]}),
            protocol,
        )
    assert (tmp_path / "raw_generations.jsonl").read_text()
    assert [json.loads(v)["event"] for v in protocol.getvalue().splitlines()] == ["READY"]
    assert json.loads((tmp_path / "worker_failure.json").read_text())["model_requests"] == 1
    assert not (tmp_path / "worker_completion.json").exists()


def test_limit_rejects_before_model_and_eof_is_not_clean_close(tmp_path):
    engine = Engine(tmp_path)
    with pytest.raises(ValueError, match="limit exceeded"):
        worker.serve(
            tmp_path,
            {"request_limit": 1},
            engine,
            Tokenizer(),
            SimpleNamespace,
            rpc({"op": "generate", "batch_id": 0, "requests": [request(0), request(1)]}),
            io.StringIO(),
        )
    assert engine.count == 0
    eof = tmp_path / "eof"
    eof.mkdir()
    with pytest.raises(EOFError, match="explicit close"):
        worker.serve(
            eof,
            {"request_limit": 1},
            Engine(eof),
            Tokenizer(),
            SimpleNamespace,
            io.StringIO(),
            io.StringIO(),
        )


@pytest.mark.parametrize(
    "tokens,stop,expected",
    [
        ([7, 151645], None, "COMPLETE"),
        ([7, 151643], 151643, "COMPLETE"),
        ([151645, 7, 151645], None, "INCOMPLETE"),
        ([7, 151645], True, "INCOMPLETE"),
        ([7, 151645], 151643, "INCOMPLETE"),
        ([7] * 4095 + [151645], None, "INCOMPLETE"),
    ],
)
def test_eos_boundary_does_not_treat_cap_or_interior_eos_as_complete(tokens, stop, expected):
    status, _ = worker.generation_integrity(
        {
            "response_token_ids": tokens,
            "stop_reason": stop,
            "returned_finished": True,
            "finish_reason": "stop",
        }
    )
    assert status == expected
