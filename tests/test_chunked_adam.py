"""CPU numerical and allocation-contract checks for the opt-in AdamW path."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.agents.chunked_adam import FAILED_ATTRIBUTE, chunked_adamw_step  # noqa: E402


def _parameters():
    return [
        torch.nn.Parameter(torch.linspace(-0.9, 1.1, 15).reshape(3, 5)),
        torch.nn.Parameter(torch.linspace(-0.3, 0.8, 7)),
        torch.nn.Parameter(torch.tensor(0.7)),
        torch.nn.Parameter(torch.empty(0)),
    ]


def _optimizer(parameters, maximize=False):
    return torch.optim.AdamW(
        [{"params": parameters[:1], "lr": 0.003, "weight_decay": 0.13},
         {"params": parameters[1:], "lr": 0.002, "weight_decay": 0.07,
          "betas": (0.8, 0.97), "maximize": maximize}],
        foreach=False, fused=False, eps=1e-7,
    )


@pytest.mark.parametrize("chunk_bytes", [20, 53, 101, 4096])
@pytest.mark.parametrize("maximize", [False, True])
def test_multistep_matches_stock_with_external_accumulation_and_global_clip(chunk_bytes, maximize):
    reference = _parameters()
    candidate = [torch.nn.Parameter(parameter.detach().clone()) for parameter in reference]
    stock = _optimizer(reference, maximize)
    chunked = _optimizer(candidate, maximize)
    identity = id(chunked)
    events = []
    host_admissions = []
    for step in range(5):
        stock.zero_grad(set_to_none=True)
        chunked.zero_grad(set_to_none=True)
        # Two independently computed microbatch contributions, then one global clip.
        for microbatch in range(2):
            for parameters in (reference, candidate):
                loss = sum(
                    ((parameter * (index + 1) + step / 10).square().sum()
                     * (microbatch + 1) / 5)
                    for index, parameter in enumerate(parameters)
                    if index != 1 or step not in {0, 3}
                )
                loss.backward()
        ref_norm = torch.nn.utils.clip_grad_norm_(reference, max_norm=0.7, foreach=False)
        candidate_norm = torch.nn.utils.clip_grad_norm_(candidate, max_norm=0.7, foreach=False)
        assert torch.equal(ref_norm, candidate_norm)
        candidate_gradients = [p.grad.clone() if p.grad is not None else None for p in candidate]
        stock.step()
        assert chunked_adamw_step(
            chunked, chunk_bytes=chunk_bytes, admit_host=host_admissions.append, emit=events.append,
        ) is None
        assert id(chunked) == identity and type(chunked) is torch.optim.AdamW
        comparisons = zip(reference, candidate, candidate_gradients, strict=True)
        for expected, actual, gradient in comparisons:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if gradient is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, gradient, rtol=0, atol=0)
            assert set(stock.state.get(expected, {})) == set(chunked.state.get(actual, {}))
            for name, value in stock.state.get(expected, {}).items():
                observed = chunked.state[actual][name]
                torch.testing.assert_close(observed, value, rtol=0, atol=0)
                assert observed.device.type == "cpu" and not observed.is_pinned()
                assert not observed.requires_grad
    assert len(host_admissions) == len(candidate)
    assert len(events) == 5
    assert all(event["maximum_temporary_bytes"] <= chunk_bytes for event in events)
    assert all(event["moment_transfer_bytes"] == 0 for event in events)
    assert all(event["event"] == "chunked_adamw_step_complete" for event in events)
    assert not getattr(chunked, FAILED_ATTRIBUTE, False)


def test_grad_none_skips_initialization_step_and_weight_decay():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.9)
    admission = []
    before = parameter.detach().clone()
    chunked_adamw_step(optimizer, chunk_bytes=20, admit_host=admission.append)
    assert parameter not in optimizer.state
    assert not admission and torch.equal(parameter, before)
    parameter.grad = torch.ones_like(parameter)
    chunked_adamw_step(optimizer, chunk_bytes=20)
    state = {key: value.clone() for key, value in optimizer.state[parameter].items()}
    before = parameter.detach().clone()
    parameter.grad = None
    chunked_adamw_step(optimizer, chunk_bytes=20, admit_host=admission.append)
    assert torch.equal(parameter, before)
    assert all(torch.equal(optimizer.state[parameter][key], value) for key, value in state.items())
    assert not admission


def test_actual_verl_optimizer_with_omegaconf_betas_matches_stock_without_conversion():
    omegaconf = pytest.importorskip("omegaconf")
    optimizer_module = pytest.importorskip("verl.workers.config.optimizer")
    config = omegaconf.OmegaConf.structured(optimizer_module.FSDPOptimizerConfig(
        lr=0.003, weight_decay=0.07, betas=(0.8, 0.97),
        override_optimizer_config={"foreach": False, "fused": False},
    ))
    reference = torch.nn.Parameter(torch.linspace(-0.9, 1.1, 11))
    candidate = torch.nn.Parameter(reference.detach().clone())
    stock = optimizer_module.build_optimizer([reference], config)
    chunked = optimizer_module.build_optimizer([candidate], config)
    betas = chunked.param_groups[0]["betas"]
    assert isinstance(betas, omegaconf.ListConfig)
    for step in range(3):
        gradient = torch.linspace(-0.7, 0.4, 11) + step / 10
        reference.grad = gradient.clone()
        candidate.grad = gradient.clone()
        stock.step()
        chunked_adamw_step(chunked, chunk_bytes=64)
        assert chunked.param_groups[0]["betas"] is betas
        assert list(betas) == [0.8, 0.97]
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
        for key, value in stock.state[reference].items():
            torch.testing.assert_close(chunked.state[candidate][key], value, rtol=0, atol=0)


def test_functional_receives_only_bounded_views_and_original_step(monkeypatch):
    adamw_module = importlib.import_module("torch.optim.adamw")
    original = adamw_module.adamw
    parameter = torch.nn.Parameter(torch.arange(11, dtype=torch.float32).reshape(1, 11))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], foreach=False, fused=False)
    calls = []

    def inspect_call(params, grads, averages, squares, maxima, steps, **kwargs):
        assert len(params) == len(grads) == len(averages) == len(squares) == len(steps) == 1
        assert params[0].numel() <= 3  # 53 bytes / (4 FP32 temporary arrays).
        assert params[0].untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
        assert grads[0].untyped_storage().data_ptr() == parameter.grad.untyped_storage().data_ptr()
        assert averages[0].untyped_storage().data_ptr() == (
            optimizer.state[parameter]["exp_avg"].untyped_storage().data_ptr()
        )
        assert squares[0].shape == params[0].shape
        assert kwargs["foreach"] is kwargs["fused"] is kwargs["capturable"] is False
        assert kwargs["amsgrad"] is kwargs["differentiable"] is False
        assert not maxima
        calls.append((params[0].numel(), steps[0].item()))
        return original(params, grads, averages, squares, maxima, steps, **kwargs)

    monkeypatch.setattr(adamw_module, "adamw", inspect_call)
    chunked_adamw_step(optimizer, chunk_bytes=53)
    chunked_adamw_step(optimizer, chunk_bytes=53)
    assert calls == [(3, 0), (3, 0), (3, 0), (2, 0), (3, 1), (3, 1), (3, 1), (2, 1)]
    assert optimizer.state[parameter]["step"].item() == 2


@pytest.mark.parametrize("name,value", [
    ("foreach", True), ("fused", True), ("capturable", True), ("differentiable", True),
    ("amsgrad", True), ("decoupled_weight_decay", False), ("maximize", 1),
    ("lr", float("nan")), ("lr", torch.tensor(0.1)), ("weight_decay", -0.1),
    ("eps", float("inf")), ("betas", (0.9, torch.tensor(0.99))), ("betas", (1.0, 0.9)),
])
def test_unsupported_groups_fail_before_any_update(name, value):
    parameters = _parameters()
    optimizer = _optimizer(parameters)
    optimizer.param_groups[1][name] = value
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError):
        chunked_adamw_step(optimizer, chunk_bytes=53)
    assert getattr(optimizer, FAILED_ATTRIBUTE)
    assert not optimizer.state
    assert all(torch.equal(p, old) for p, old in zip(parameters, before, strict=True))


@pytest.mark.parametrize("chunk_bytes", [0, -1, 19, 21.0, True])
def test_invalid_chunk_size(chunk_bytes):
    optimizer = _optimizer(_parameters())
    with pytest.raises(ValueError, match="chunk_bytes"):
        chunked_adamw_step(optimizer, chunk_bytes=chunk_bytes)


@pytest.mark.parametrize(
    "kind", ["parameter_dtype", "parameter_stride", "gradient_stride", "sparse"],
)
def test_invalid_tensor_contract(kind):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    if kind == "parameter_dtype":
        parameter = torch.nn.Parameter(parameter.double())
    elif kind == "parameter_stride":
        parameter = torch.nn.Parameter(parameter.detach().t())
    parameter.grad = torch.ones_like(parameter)
    if kind == "gradient_stride":
        parameter.grad = torch.ones(3, 2).t()
    elif kind == "sparse":
        parameter.grad = parameter.grad.to_sparse()
    optimizer = torch.optim.AdamW([parameter])
    with pytest.raises(ValueError, match="dense contiguous FP32"):
        chunked_adamw_step(optimizer, chunk_bytes=53)


@pytest.mark.parametrize(
    "kind", ["missing_moment", "moment_dtype", "moment_shape", "extra_key", "step"],
)
def test_invalid_or_partially_restored_state_cannot_be_reset(kind):
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter])
    chunked_adamw_step(optimizer, chunk_bytes=53)
    state = optimizer.state[parameter]
    if kind == "missing_moment":
        del state["exp_avg"]
    elif kind == "moment_dtype":
        state["exp_avg"] = state["exp_avg"].double()
    elif kind == "moment_shape":
        state["exp_avg"] = state["exp_avg"].reshape(1, 3)
    elif kind == "extra_key":
        state["extra"] = torch.tensor(1.0)
    else:
        state["step"].fill_(0.5)
    before = parameter.detach().clone()
    with pytest.raises(ValueError):
        chunked_adamw_step(optimizer, chunk_bytes=53)
    assert torch.equal(parameter, before)


def test_host_admission_precedes_allocation_and_failure_is_not_retryable():
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter])
    events = []

    def deny(size):
        assert size == 3 * 4 * 2 + 4
        assert not optimizer.state[parameter]
        raise MemoryError("host denied")

    with pytest.raises(MemoryError, match="host denied"):
        chunked_adamw_step(optimizer, chunk_bytes=53, admit_host=deny, emit=events.append)
    assert not optimizer.state[parameter]
    assert events[-1]["event"] == "chunked_adamw_step_failed"
    with pytest.raises(RuntimeError, match="previously failed"):
        chunked_adamw_step(optimizer, chunk_bytes=53)


def test_partial_functional_failure_marks_optimizer_unusable(monkeypatch):
    adamw_module = importlib.import_module("torch.optim.adamw")
    original = adamw_module.adamw
    parameter = torch.nn.Parameter(torch.ones(9))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter])
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated copy/update failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(adamw_module, "adamw", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        chunked_adamw_step(optimizer, chunk_bytes=53)
    assert getattr(optimizer, FAILED_ATTRIBUTE)
    assert optimizer.state[parameter]["step"].item() == 0
    assert not torch.equal(parameter[:3], torch.ones(3))
    with pytest.raises(RuntimeError, match="previously failed"):
        chunked_adamw_step(optimizer, chunk_bytes=53)


def test_import_does_not_import_torch_or_initialize_cuda():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import sys, json; import src.agents.chunked_adam; "
         "print(json.dumps({'torch_loaded': 'torch' in sys.modules}))"],
        cwd=repository, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == {"torch_loaded": False}
