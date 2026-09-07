"""Real tiny CPU tensors with logical device labels; never initialize CUDA.

This validates tensor values/lifetimes, not GPU copy ordering or FSDP integration.
"""

from __future__ import annotations

import unittest
import weakref
from types import SimpleNamespace

import torch

from src.agents.screening_exchange import ExchangeCoordinator, ExchangeError, Operations, TensorInfo


class CpuTensorOperations:
    def __init__(self):
        self.locations = {}
        self.previous = None
        self.copies = 0

    def label(self, tensor, device):
        self.locations[id(tensor)] = device
        return tensor

    def describe(self, tensor):
        assert tensor.device.type == "cpu"
        return TensorInfo(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            device=self.locations[id(tensor)],
            storage_id=tensor.untyped_storage()._cdata,
            storage_bytes=tensor.untyped_storage().nbytes(),
            payload_bytes=tensor.numel() * tensor.element_size(),
            contiguous=tensor.is_contiguous(),
            storage_offset=tensor.storage_offset(),
            pinned=tensor.is_pinned(),
            is_view=tensor._base is not None,
        )

    def move(self, source, destination, *, non_blocking):
        assert not torch.is_grad_enabled() and non_blocking is False
        assert self.previous is None or self.previous() is None
        self.previous = weakref.ref(source)
        self.copies += 1
        return self.label(source.detach().clone(), destination)

    def interface(self):
        return Operations(
            describe=self.describe,
            scalar_value=lambda tensor: tensor.item(),
            move=self.move,
            synchronize=lambda: None,
            no_grad=torch.no_grad,
            boundary_check=lambda: None,
            admit_transfer=lambda device, size: None,
        )


def fixture():
    ops = CpuTensorOperations()
    coordinator = ExchangeCoordinator(identity="cpu-mock-only", operations=ops.interface())
    optimizers = []
    for role, shapes in (("actor", ((2, 3), (1,))), ("critic", ((4,),))):
        model = torch.nn.ParameterList(
            [ops.label(torch.nn.Parameter(torch.full(shape, 1.5)), "cuda:0") for shape in shapes]
        )
        device = "cuda:0" if role == "actor" else "cpu"
        state = {
            parameter: {
                "step": ops.label(torch.tensor(7.0), "cpu"),
                "exp_avg": ops.label(torch.full_like(parameter, 0.125), device),
                "exp_avg_sq": ops.label(torch.full_like(parameter, 0.25), device),
            }
            for parameter in model.parameters()
        }
        optimizer = SimpleNamespace(
            state=state,
            param_groups=[
                {
                    "params": list(model.parameters()),
                    "foreach": False,
                    "fused": False,
                    "capturable": False,
                    "amsgrad": False,
                }
            ],
        )
        coordinator.register(role, model, optimizer, identity="cpu-mock-only")
        optimizers.append(optimizer)
    return coordinator, ops, optimizers


class RealCpuTensorTests(unittest.TestCase):
    def test_actual_tensors_preserve_exact_values_shapes_dtype_scalar_and_lifetime(self):
        coordinator, ops, optimizers = fixture()
        before = [
            (
                key,
                tensor.tolist(),
                tensor.dtype,
                tuple(tensor.shape),
                id(tensor) if key == "step" else None,
            )
            for optimizer in optimizers
            for state in optimizer.state.values()
            for key, tensor in state.items()
        ]
        coordinator.activate("critic", identity="cpu-mock-only")
        coordinator.activate("actor", identity="cpu-mock-only")
        after = [
            (
                key,
                tensor.tolist(),
                tensor.dtype,
                tuple(tensor.shape),
                id(tensor) if key == "step" else None,
            )
            for optimizer in optimizers
            for state in optimizer.state.values()
            for key, tensor in state.items()
        ]
        self.assertEqual(before, after)
        self.assertEqual(ops.copies, 12)
        self.assertIsNone(ops.previous())
        self.assertTrue(
            all(
                tensor.device.type == "cpu" and not tensor.is_pinned()
                for optimizer in optimizers
                for state in optimizer.state.values()
                for tensor in state.values()
            )
        )

    def test_real_detached_alias_inside_state_rejected_before_copy(self):
        coordinator, ops, optimizers = fixture()
        state = next(iter(optimizers[0].state.values()))
        state["exp_avg_sq"] = ops.label(state["exp_avg"].detach(), "cuda:0")
        with self.assertRaisesRegex(ExchangeError, "shared storage"):
            coordinator.activate("critic", identity="cpu-mock-only")
        self.assertEqual(ops.copies, 0)

    def test_real_external_old_reference_fails_after_replacement(self):
        coordinator, ops, optimizers = fixture()
        old = next(iter(optimizers[0].state.values()))["exp_avg"]
        with self.assertRaisesRegex(ExchangeError, "retained externally"):
            coordinator.activate("critic", identity="cpu-mock-only")
        self.assertTrue(coordinator.failed)
        self.assertEqual(old.tolist(), [[0.125] * 3] * 2)

    def test_real_noncontiguous_and_reduced_dtype_moments_rejected(self):
        for transform in (lambda tensor: tensor.t(), lambda tensor: tensor.bfloat16()):
            coordinator, ops, optimizers = fixture()
            state = next(iter(optimizers[0].state.values()))
            state["exp_avg"] = ops.label(transform(state["exp_avg"]), "cuda:0")
            with self.assertRaises(ExchangeError):
                coordinator.activate("critic", identity="cpu-mock-only")
            self.assertEqual(ops.copies, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
