"""Tiny CPU/mock tests; no torch import, real GPU allocation or CUDA calls."""

from __future__ import annotations

import json
import random
import unittest
import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.screening_exchange import ExchangeCoordinator, ExchangeError, Operations, TensorInfo


class Tensor:
    next_storage = 0

    def __init__(self, size=1, *, device="cuda:0", scalar=False, value=3):
        Tensor.next_storage += 1
        self.info = TensorInfo(
            shape=() if scalar else (size,),
            dtype="float32",
            device=device,
            storage_id=Tensor.next_storage,
            storage_bytes=size * 4,
            payload_bytes=size * 4,
        )
        self.value = value
        self.grad = None


class Model:
    def __init__(self, sizes):
        self.params = [Tensor(size) for size in sizes]

    def parameters(self):
        return iter(self.params)


def optimizer(model, device, empty=False):
    state = {}
    if not empty:
        for parameter in model.params:
            size = parameter.info.shape[0]
            state[parameter] = {
                "step": Tensor(device="cpu", scalar=True, value=7),
                "exp_avg": Tensor(size, device=device, value=11),
                "exp_avg_sq": Tensor(size, device=device, value=23),
            }
    return SimpleNamespace(
        state=state,
        param_groups=[
            {
                "params": model.params,
                "foreach": False,
                "fused": False,
                "capturable": False,
                "amsgrad": False,
            }
        ],
    )


class MockOperations:
    def __init__(self):
        self.move_count = 0
        self.sync_count = 0
        self.in_no_grad = False
        self.previous_source = None
        self.failure_at = None
        self.error = RuntimeError("injected transfer failure")
        self.retained = []
        self.retain_source = False
        self.pin_destination = False
        self.admitted = []

    @contextmanager
    def no_grad(self):
        self.in_no_grad = True
        try:
            yield
        finally:
            self.in_no_grad = False

    def synchronize(self):
        self.sync_count += 1

    def move(self, source, destination, *, non_blocking):
        assert self.in_no_grad and non_blocking is False
        assert self.previous_source is None or self.previous_source() is None
        self.move_count += 1
        if self.move_count == self.failure_at:
            raise self.error
        self.previous_source = weakref.ref(source)
        if self.retain_source:
            self.retained.append(source)
        copied = Tensor(source.info.shape[0], device=destination, value=source.value)
        if self.pin_destination and destination == "cpu":
            copied.info = replace(copied.info, pinned=True)
        return copied

    def interface(self):
        return Operations(
            describe=lambda tensor: tensor.info,
            scalar_value=lambda tensor: tensor.value,
            move=self.move,
            synchronize=self.synchronize,
            no_grad=self.no_grad,
            boundary_check=lambda: None,
            admit_transfer=lambda device, size: self.admitted.append((device, size)),
        )


def fixture(actor_sizes=(6, 1), critic_sizes=(3,), *, empty_actor=False, empty_critic=False):
    ops = MockOperations()
    coordinator = ExchangeCoordinator(identity="candidate-only", operations=ops.interface())
    actor, critic = Model(actor_sizes), Model(critic_sizes)
    actor_opt = optimizer(actor, "cuda:0", empty_actor)
    critic_opt = optimizer(critic, "cpu", empty_critic)
    coordinator.register("actor", actor, actor_opt, identity="candidate-only")
    coordinator.register("critic", critic, critic_opt, identity="candidate-only")
    return coordinator, ops, actor, critic, actor_opt, critic_opt


def values(opt):
    # Snapshot values/metadata only; do not retain tensors or list(state.items()).
    return {
        (id(parameter), key): (tensor.value, tensor.info.dtype, tensor.info.shape)
        for parameter, state in opt.state.items()
        for key, tensor in state.items()
    }


class ExchangeTests(unittest.TestCase):
    def assert_failed(self, coordinator):
        self.assertTrue(coordinator.failed)
        with self.assertRaisesRegex(ExchangeError, "prohibited"):
            coordinator.activate("actor", identity="candidate-only")
        with self.assertRaisesRegex(ExchangeError, "prohibited"):
            coordinator.assert_ready("actor", identity="candidate-only")

    def test_unequal_shapes_counts_exact_values_steps_and_round_trip(self):
        coordinator, ops, actor, critic, actor_opt, critic_opt = fixture()
        initial = (values(actor_opt), values(critic_opt))
        steps = [
            id(state["step"]) for opt in (actor_opt, critic_opt) for state in opt.state.values()
        ]
        first = coordinator.activate("critic", identity="candidate-only")
        self.assertEqual(first["moment_bytes_by_device"], {"cpu": 56, "cuda:0": 24})
        self.assertEqual(coordinator.active_role, "critic")
        coordinator.activate("actor", identity="candidate-only")
        self.assertEqual(initial, (values(actor_opt), values(critic_opt)))
        self.assertEqual(
            steps,
            [id(state["step"]) for opt in (actor_opt, critic_opt) for state in opt.state.values()],
        )
        self.assertEqual(ops.move_count, 12)
        self.assertEqual(ops.sync_count, 4)
        self.assertTrue(all(parameter.grad is None for parameter in actor.params + critic.params))
        json.dumps(coordinator.events)  # Records contain only serializable metadata.

    def test_capacity_and_byte_order_with_empty_and_unequal_queues(self):
        rng = random.Random(731)
        cases = [([], []), ([30, 1], []), ([], [1, 20]), ([100], [1, 1, 1, 3])]
        cases += [
            (
                [rng.randint(1, 1000) for _ in range(rng.randrange(7))],
                [rng.randint(1, 1000) for _ in range(rng.randrange(7))],
            )
            for _ in range(120)
        ]
        for actor_sizes, critic_sizes in cases:
            with self.subTest(actor=actor_sizes, critic=critic_sizes):
                coordinator, ops, *_ = fixture(actor_sizes, critic_sizes)
                before = coordinator.assert_ready("actor", identity="candidate-only")
                after = coordinator.activate("critic", identity="candidate-only")
                largest = before["largest_single_storage_bytes"]
                for device in ("cpu", "cuda:0"):
                    bound = (
                        max(
                            before["moment_bytes_by_device"][device],
                            after["moment_bytes_by_device"][device],
                        )
                        + largest
                    )
                    self.assertTrue(
                        all(
                            event["live_moment_bytes"][device] <= bound
                            for event in coordinator.events
                            if "live_moment_bytes" in event
                        )
                    )
                moves = [event for event in coordinator.events if event["event"] == "move_start"]
                remaining_out, remaining_in = len(actor_sizes) * 2, len(critic_sizes) * 2
                delta = 0
                for event in moves:
                    outgoing = remaining_out and (not remaining_in or delta <= 0)
                    self.assertEqual(event["destination"], "cpu" if outgoing else "cuda:0")
                    if outgoing:
                        remaining_out -= 1
                        delta += event["bytes"]
                    else:
                        remaining_in -= 1
                        delta -= event["bytes"]
                self.assertEqual(ops.move_count, 2 * (len(actor_sizes) + len(critic_sizes)))
                self.assertEqual(len(ops.admitted), ops.move_count)

    def test_lazy_first_use_state_is_reinventoried_without_added_optimizer_steps(self):
        coordinator, ops, actor, critic, actor_opt, critic_opt = fixture(
            empty_actor=True,
            empty_critic=True,
        )
        coordinator.activate("critic", identity="candidate-only")
        self.assertEqual(ops.move_count, 0)
        critic_opt.state.update(optimizer(critic, "cuda:0").state)
        coordinator.activate("actor", identity="candidate-only")
        self.assertEqual(ops.move_count, 2)
        actor_opt.state.update(optimizer(actor, "cuda:0").state)
        coordinator.activate("critic", identity="candidate-only")
        self.assertEqual(ops.move_count, 8)
        self.assertTrue(
            all(
                state["step"].value == 7
                for opt in (actor_opt, critic_opt)
                for state in opt.state.values()
            )
        )

    def test_gradient_gate_does_not_clear_live_gradients_or_transfer(self):
        coordinator, ops, actor, *_ = fixture()
        gradient = object()
        actor.params[0].grad = gradient
        with self.assertRaisesRegex(ExchangeError, "gradient remains"):
            coordinator.activate("critic", identity="candidate-only")
        self.assertIs(actor.params[0].grad, gradient)
        self.assertEqual(ops.move_count, 0)
        self.assert_failed(coordinator)

    def test_full_update_boundary_preserves_calls_and_scheduler_counts(self):
        coordinator, _, actor, critic, actor_opt, critic_opt = fixture()
        calls = []
        for role, model in (("actor", actor), ("critic", critic), ("actor", actor)):
            coordinator.activate(role, identity="candidate-only")
            model.params[0].grad = object()
            calls.extend([role + ":backward", role + ":optimizer", role + ":scheduler"])
            model.params[0].grad = None  # Fake original full update returns after zero_grad.
            calls.append(role + ":returned")
            coordinator.assert_ready(role, identity="candidate-only")
        self.assertEqual(
            calls,
            [
                role + suffix
                for role in ("actor", "critic", "actor")
                for suffix in (":backward", ":optimizer", ":scheduler", ":returned")
            ],
        )

    def test_unsupported_tensor_and_optimizer_contracts_fail_before_any_copy(self):
        changes = [
            {"dtype": "bfloat16"},
            {"contiguous": False},
            {"storage_offset": 1},
            {"is_view": True},
            {"storage_bytes": 999},
            {"shape": (99,)},
            {"device": "cpu"},
        ]
        for change in changes:
            with self.subTest(change=change):
                coordinator, ops, actor, _, actor_opt, _ = fixture()
                state = actor_opt.state[actor.params[0]]
                state["exp_avg"].info = replace(state["exp_avg"].info, **change)
                with self.assertRaises(ExchangeError):
                    coordinator.activate("critic", identity="candidate-only")
                self.assertEqual(ops.move_count, 0)
                self.assert_failed(coordinator)
        for option in ("foreach", "fused", "capturable", "amsgrad"):
            coordinator, ops, _, _, actor_opt, _ = fixture()
            actor_opt.param_groups[0][option] = True
            with self.assertRaisesRegex(ExchangeError, option):
                coordinator.activate("critic", identity="candidate-only")
            self.assertEqual(ops.move_count, 0)

    def test_shared_moment_storage_and_unknown_keys_are_rejected(self):
        for mode in ("same_tensor", "different_tensor_same_storage", "unsupported_key", "partial"):
            with self.subTest(mode=mode):
                coordinator, ops, actor, _, actor_opt, _ = fixture()
                state = actor_opt.state[actor.params[0]]
                if mode == "same_tensor":
                    state["exp_avg_sq"] = state["exp_avg"]
                elif mode == "different_tensor_same_storage":
                    state["exp_avg_sq"].info = state["exp_avg"].info
                elif mode == "unsupported_key":
                    state["max_exp_avg_sq"] = Tensor(6)
                else:
                    del state["exp_avg_sq"]
                with self.assertRaises(ExchangeError):
                    coordinator.activate("critic", identity="candidate-only")
                self.assertEqual(ops.move_count, 0)

    def test_retaining_list_of_old_state_values_invalidates_live_bound(self):
        coordinator, ops, actor, _, actor_opt, _ = fixture()
        retained = list(actor_opt.state[actor.params[0]].items())
        with self.assertRaisesRegex(ExchangeError, "retained externally"):
            coordinator.activate("critic", identity="candidate-only")
        self.assertEqual(ops.move_count, 1)
        self.assertTrue(retained)  # Deliberate regression: captured old tensors stay alive.
        self.assert_failed(coordinator)

    def test_mover_retention_and_pinned_cpu_copy_fail_closed(self):
        for attribute, message in (
            ("retain_source", "retained externally"),
            ("pin_destination", "pageable"),
        ):
            coordinator, ops, *_ = fixture()
            setattr(ops, attribute, True)
            with self.assertRaisesRegex(ExchangeError, message):
                coordinator.activate("critic", identity="candidate-only")
            self.assert_failed(coordinator)

    def test_failure_before_each_replacement_preserves_original_exception(self):
        for failure_at in range(1, 7):
            coordinator, ops, *_ = fixture()
            ops.failure_at = failure_at
            with self.assertRaises(RuntimeError) as caught:
                coordinator.activate("critic", identity="candidate-only")
            self.assertIs(caught.exception, ops.error)
            self.assertEqual(ops.move_count, failure_at)
            self.assert_failed(coordinator)

    def test_failure_after_each_replacement_locks_without_retry(self):
        for failure_at in range(1, 7):
            coordinator, ops, *_ = fixture()
            emit = coordinator._emit
            seen = 0
            error = RuntimeError("injected after replacement")

            def failing_emit(event, emit=emit, failure_at=failure_at, error=error, **fields):
                nonlocal seen
                emit(event, **fields)
                if event == "move_released":
                    seen += 1
                    if seen == failure_at:
                        raise error

            coordinator._emit = failing_emit
            with self.assertRaises(RuntimeError) as caught:
                coordinator.activate("critic", identity="candidate-only")
            self.assertIs(caught.exception, error)
            self.assertEqual(ops.move_count, failure_at)
            self.assertIsNone(ops.previous_source())
            self.assert_failed(coordinator)

    def test_identity_pid_duplicate_registration_and_reentrant_guards(self):
        coordinator, ops, actor, critic, actor_opt, critic_opt = fixture()
        coordinator.register("actor", actor, actor_opt, identity="candidate-only")
        with self.assertRaisesRegex(ExchangeError, "distinct"):
            coordinator.register("actor", Model([1]), object(), identity="candidate-only")
        self.assert_failed(coordinator)
        coordinator, ops, *_ = fixture()
        with self.assertRaisesRegex(ExchangeError, "identity"):
            coordinator.activate("critic", identity="other")
        self.assertEqual(ops.move_count, 0)
        coordinator, ops, *_ = fixture()
        with (
            patch("src.agents.screening_exchange.os.getpid", return_value=coordinator.pid + 1),
            self.assertRaisesRegex(ExchangeError, "PID"),
        ):
            coordinator.activate("critic", identity="candidate-only")
        coordinator, ops, *_ = fixture()
        coordinator.operations = replace(
            coordinator.operations,
            boundary_check=lambda: coordinator.activate("actor", identity="candidate-only"),
        )
        with self.assertRaisesRegex(ExchangeError, "non-reentrant"):
            coordinator.activate("critic", identity="candidate-only")
        self.assertEqual(ops.move_count, 0)
        self.assert_failed(coordinator)

    def test_step_value_or_dtype_drift_and_admission_failure_are_fatal(self):
        for mutation in ("value", "dtype"):
            coordinator, ops, actor, _, actor_opt, _ = fixture()
            sync = coordinator.operations.synchronize

            def drift(sync=sync, ops=ops, actor_opt=actor_opt, actor=actor, mutation=mutation):
                sync()
                if ops.sync_count == 2:
                    step = actor_opt.state[actor.params[0]]["step"]
                    if mutation == "value":
                        step.value += 1
                    else:
                        step.info = replace(step.info, dtype="float64")

            coordinator.operations = replace(coordinator.operations, synchronize=drift)
            with self.assertRaisesRegex(ExchangeError, "scalar steps"):
                coordinator.activate("critic", identity="candidate-only")
            self.assert_failed(coordinator)
        coordinator, ops, *_ = fixture()
        error = MemoryError("insufficient destination reserve")

        def reject(device, size):
            raise error

        coordinator.operations = replace(coordinator.operations, admit_transfer=reject)
        with self.assertRaises(MemoryError) as caught:
            coordinator.activate("critic", identity="candidate-only")
        self.assertIs(caught.exception, error)
        self.assertEqual(ops.move_count, 0)
        self.assert_failed(coordinator)


if __name__ == "__main__":
    unittest.main(verbosity=2)
