"""Standard-library-only worker wrapper tests; no torch imports or CUDA."""

from __future__ import annotations

import asyncio
import inspect
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from test_screening_exchange import MockOperations, Model, optimizer

from src.agents.screening_exchange import ExchangeError
from src.agents.screening_placement import (
    IDENTITY_ENV,
    PROFILE,
    PROFILE_ENV,
    WorkerRegistry,
    install_from_environment,
    patch_worker_classes,
    verify_installed_sources,
)

IDENTITY = "a" * 64


class Backend:
    gpu_device = "cuda:0"

    def __init__(self):
        self.ops = MockOperations()

    def operations(self, boundary_check):
        from dataclasses import replace

        return replace(self.ops.interface(), boundary_check=boundary_check)

    def validate_worker(self, worker, role):
        if worker._is_offload_param or worker._is_offload_optimizer:
            raise ExchangeError("native offload enabled")

    def register_training(self, model, optimizer):
        self.validate_training(model)
        if optimizer.state:
            raise ExchangeError("fresh-only")

    def validate_training(self, model):
        if any(parameter.grad is not None for parameter in model.params):
            raise ExchangeError("live gradient after full update")

    def validate_reference(self, worker):
        if worker.ref_module_fsdp.device != "cpu" or worker.ref_failed:
            raise ExchangeError("reference placement or failure invalid")

    def metrics(self):
        return {"mock_only": True}


def worker_classes():
    class Actor:
        def __init__(self, *, actor=True, ref=False):
            self._is_actor, self._is_ref = actor, ref
            self._is_offload_param = self._is_offload_optimizer = False
            self.calls = []
            self.fail_update = None
            self.leave_gradient = False
            self.result = object()
            self.ref_failed = False

        def init_model(self):
            self.calls.append(("init",))
            if self._is_actor:
                self.actor_module_fsdp = Model([4, 1])
                self.actor_optimizer = optimizer(self.actor_module_fsdp, "cuda:0", empty=True)
            if self._is_ref:
                self.ref_module_fsdp = SimpleNamespace(device="cpu")
            return self.result

        def update_actor(self, data):
            self.calls.append(("update_actor", data))
            if self.fail_update:
                raise self.fail_update
            self.actor_module_fsdp.params[0].grad = object()
            self.calls.extend([("actor_optimizer_step",), ("actor_scheduler_step",)])
            if not self.actor_optimizer.state:
                self.actor_optimizer.state.update(optimizer(self.actor_module_fsdp, "cuda:0").state)
            if not self.leave_gradient:
                self.actor_module_fsdp.params[0].grad = None
            return self.result

        def compute_log_prob(self, data):
            self.calls.append(("compute_log_prob", data))
            return self.result

        def compute_ref_log_prob(self, data):
            self.calls.append(("compute_ref_log_prob", data))
            return self.result

        async def rollout_mode(self):
            self.calls.append(("rollout_mode",))
            return self.result

        def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            self.calls.append(("save", local_path, hdfs_path, global_step, max_ckpt_to_keep))
            return self.result

        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            self.calls.append(("load", local_path))

    class Critic:
        def __init__(self):
            self._is_offload_param = self._is_offload_optimizer = False
            self.calls = []
            self.result = object()

        def init_model(self):
            self.calls.append(("init",))
            self.critic_module = Model([3])
            self.critic_optimizer = optimizer(self.critic_module, "cuda:0", empty=True)
            return self.result

        def update_critic(self, data):
            self.calls.append(("update_critic", data))
            self.critic_module.params[0].grad = object()
            self.calls.extend([("critic_optimizer_step",), ("critic_scheduler_step",)])
            if not self.critic_optimizer.state:
                self.critic_optimizer.state.update(optimizer(self.critic_module, "cuda:0").state)
            self.critic_module.params[0].grad = None
            return self.result

        def compute_values(self, data):
            self.calls.append(("compute_values", data))
            return self.result

        def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            self.calls.append(("save", local_path, hdfs_path, global_step, max_ckpt_to_keep))
            return self.result

        def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
            self.calls.append(("load", local_path))

    Actor.update_actor.dispatch_metadata = {"mode": "original"}
    Actor._finance_manual_reference_profile = {"profile": PROFILE, "identity": IDENTITY}
    return Actor, Critic


def fixture(order=("reference", "critic", "actor")):
    backend = Backend()
    records = []
    registry = WorkerRegistry(
        identity=IDENTITY,
        backend=backend,
        emit=lambda event, **fields: records.append((event, fields)),
    )
    Actor, Critic = worker_classes()
    patch_worker_classes(Actor, Critic, registry)
    actor, critic, reference = Actor(), Critic(), Actor(actor=False, ref=True)
    objects = {"actor": actor, "critic": critic, "reference": reference}
    for role in order:
        objects[role].init_model()
    return registry, backend, actor, critic, reference, records


class WorkerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {PROFILE_ENV: PROFILE, IDENTITY_ENV: IDENTITY})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_original_results_arguments_full_update_and_scheduler_calls_unchanged(self):
        registry, backend, actor, critic, reference, records = fixture()
        data = object()
        for _ in range(2):
            self.assertIs(actor.compute_log_prob(data), actor.result)
            self.assertIs(reference.compute_ref_log_prob(data), reference.result)
            self.assertIs(critic.compute_values(data), critic.result)
            self.assertIs(critic.update_critic(data), critic.result)
            self.assertEqual(registry.coordinator.active_role, "critic")
            self.assertIs(actor.update_actor(data), actor.result)
            self.assertEqual(registry.coordinator.active_role, "actor")
            self.assertIs(actor.save_checkpoint("/mock", global_step=9), actor.result)
            self.assertIs(critic.save_checkpoint("/mock", global_step=9), critic.result)
            self.assertIs(asyncio.run(actor.rollout_mode()), actor.result)
        for worker, role in ((actor, "actor"), (critic, "critic")):
            self.assertEqual(worker.calls.count((role + "_optimizer_step",)), 2)
            self.assertEqual(worker.calls.count((role + "_scheduler_step",)), 2)
            self.assertEqual(worker.calls.count(("update_" + role, data)), 2)
        self.assertEqual(backend.ops.move_count, 14)
        self.assertIs(registry.reference, reference)
        self.assertEqual(set(registry.workers), {"actor", "critic"})
        self.assertTrue(records)

    def test_both_initialization_orders_and_combined_actor_reference(self):
        for order in (("actor", "critic", "reference"), ("critic", "reference", "actor")):
            with self.subTest(order=order):
                registry, _, actor, critic, _, _ = fixture(order)
                critic.update_critic(object())
                actor.update_actor(object())
                self.assertFalse(registry.coordinator.failed)
        backend = Backend()
        registry = WorkerRegistry(identity=IDENTITY, backend=backend, emit=lambda *a, **k: None)
        Actor, Critic = worker_classes()
        patch_worker_classes(Actor, Critic, registry)
        actor, critic = Actor(ref=True), Critic()
        actor.init_model()
        critic.init_model()
        self.assertIs(registry.reference, actor)
        critic.update_critic(object())
        actor.update_actor(object())

    def test_gradients_remaining_after_full_update_lock_all_followup_operations(self):
        registry, backend, actor, critic, reference, _ = fixture()
        critic.update_critic(object())
        actor.leave_gradient = True
        with self.assertRaisesRegex(ExchangeError, "gradient"):
            actor.update_actor(object())
        self.assertIsNotNone(actor.actor_module_fsdp.params[0].grad)
        counts = tuple(len(worker.calls) for worker in (actor, critic, reference))
        for call in (
            lambda: actor.update_actor(object()),
            lambda: actor.compute_log_prob(object()),
            lambda: reference.compute_ref_log_prob(object()),
            lambda: critic.compute_values(object()),
            lambda: actor.save_checkpoint("/mock"),
            lambda: critic.save_checkpoint("/mock"),
            lambda: asyncio.run(actor.rollout_mode()),
        ):
            with self.assertRaises(ExchangeError):
                call()
        self.assertEqual(counts, tuple(len(worker.calls) for worker in (actor, critic, reference)))
        self.assertTrue(registry.coordinator.failed)

    def test_original_training_exception_preserved_no_retry(self):
        registry, _, actor, critic, _, _ = fixture()
        critic.update_critic(object())
        error = KeyboardInterrupt("injected original update")
        actor.fail_update = error
        with self.assertRaises(KeyboardInterrupt) as caught:
            actor.update_actor(object())
        self.assertIs(caught.exception, error)
        self.assertTrue(registry.coordinator.failed)

    def test_load_rejected_for_actor_critic_and_none_without_original_or_transfer(self):
        for role, path in (("actor", "/mock"), ("critic", "/mock"), ("actor", None)):
            registry, backend, actor, critic, _, _ = fixture()
            worker = actor if role == "actor" else critic
            with self.assertRaisesRegex(ExchangeError, "fresh-only"):
                worker.load_checkpoint(path)
            self.assertFalse(any(call[0] == "load" for call in worker.calls))
            self.assertEqual(backend.ops.move_count, 0)
            self.assertTrue(registry.coordinator.failed)

    def test_inactive_checkpoint_ref_only_actor_operation_and_reference_gpu_rejected(self):
        registry, _, actor, critic, _, _ = fixture()
        critic.update_critic(object())
        with self.assertRaisesRegex(ExchangeError, "not active"):
            critic.save_checkpoint("/mock")
        registry, _, _, _, reference, _ = fixture()
        with self.assertRaisesRegex(ExchangeError, "non-actor"):
            reference.update_actor(object())
        registry, _, actor, _, reference, _ = fixture()
        reference.ref_module_fsdp.device = "cuda:0"
        with self.assertRaisesRegex(ExchangeError, "reference placement"):
            actor.compute_log_prob(object())

    def test_native_offload_rejected_before_init_and_identity_drift_before_original(self):
        registry, _, actor, critic, reference, _ = fixture(order=())
        actor._is_offload_optimizer = True
        with self.assertRaisesRegex(ExchangeError, "native offload"):
            actor.init_model()
        self.assertEqual(actor.calls, [])
        registry, _, actor, critic, reference, _ = fixture()
        with (
            patch.dict(os.environ, {IDENTITY_ENV: "b" * 64}),
            self.assertRaisesRegex(ExchangeError, "identity"),
        ):
            actor.compute_log_prob(object())
        self.assertEqual(actor.calls, [("init",)])

    def test_missing_role_different_worker_model_and_duplicate_init_rejected(self):
        registry, _, actor, _, _, _ = fixture(order=("actor", "reference"))
        with self.assertRaisesRegex(ExchangeError, "both roles"):
            actor.compute_log_prob(object())
        registry, _, actor, _, _, _ = fixture()
        actor.actor_module_fsdp = Model([4, 1])
        with self.assertRaisesRegex(ExchangeError, "identity changed"):
            actor.compute_log_prob(object())
        registry, _, actor, _, _, _ = fixture()
        with self.assertRaisesRegex(ExchangeError, "already completed"):
            actor.init_model()
        self.assertEqual(actor.calls, [("init",)])

    def test_wrapper_signature_dispatch_async_and_idempotence_preserved(self):
        Actor, Critic = worker_classes()
        original = Actor.update_actor
        backend = Backend()
        registry = WorkerRegistry(identity=IDENTITY, backend=backend, emit=lambda *a, **k: None)
        self.assertTrue(patch_worker_classes(Actor, Critic, registry))
        self.assertEqual(inspect.signature(Actor.update_actor), inspect.signature(original))
        self.assertIs(Actor.update_actor.__wrapped__, original)
        self.assertIs(Actor.update_actor.dispatch_metadata, original.dispatch_metadata)
        self.assertTrue(inspect.iscoroutinefunction(Actor.rollout_mode))
        wrapper = Actor.update_actor
        self.assertFalse(patch_worker_classes(Actor, Critic, registry))
        self.assertIs(wrapper, Actor.update_actor)

    def test_signature_guard_is_atomic_and_disabled_install_imports_no_torch(self):
        Actor, Critic = worker_classes()
        original = Actor.update_actor
        Critic.update_critic = lambda self, wrong_name: None
        registry = WorkerRegistry(identity=IDENTITY, backend=Backend(), emit=lambda *a, **k: None)
        with self.assertRaisesRegex(ExchangeError, "signature changed"):
            patch_worker_classes(Actor, Critic, registry)
        self.assertIs(Actor.update_actor, original)
        with patch.dict(os.environ, {PROFILE_ENV: ""}):
            self.assertFalse(install_from_environment())

    def test_reference_adapter_install_order_must_match_identity(self):
        for marker in (None, ("manual_ref", IDENTITY), (PROFILE, "b" * 64)):
            Actor, Critic = worker_classes()
            Actor._finance_manual_reference_profile = marker
            original = Actor.update_actor
            registry = WorkerRegistry(
                identity=IDENTITY, backend=Backend(), emit=lambda *a, **k: None
            )
            with self.assertRaisesRegex(ExchangeError, "reference adapter must install before"):
                patch_worker_classes(Actor, Critic, registry)
            self.assertIs(Actor.update_actor, original)

    def test_source_version_guards_validate_current_installed_bytes_without_torch_import(self):
        self.assertEqual(len(verify_installed_sources()), 10)
        with (
            patch("src.agents.screening_placement.metadata.version", return_value="changed"),
            self.assertRaisesRegex(ExchangeError, "version"),
        ):
            verify_installed_sources()

    def test_actual_reference_wrapper_runs_inside_placement_with_cpu_boundaries(self):
        from src.agents import screening_reference

        Actor, Critic = worker_classes()
        del Actor._finance_manual_reference_profile

        def builder(
            self,
            model_path,
            fsdp_config,
            optim_config,
            override_model_config,
            use_remove_padding=False,
            use_fused_kernels=False,
            enable_gradient_checkpointing=False,
            trust_remote_code=False,
            use_liger=False,
            role="actor",
            enable_activation_offload=False,
            use_tiled_mlp=False,
            tiled_mlp_shards=4,
        ):
            raise AssertionError("this test exercises forward wrapper composition only")

        def forward(self, data):
            self.calls.append(("compute_ref_log_prob", data))
            self_state = self._finance_manual_reference_state
            if self.ref_module_fsdp.device != "cuda:0" or not self_state.busy:
                raise AssertionError(
                    "original forward must run within manual load/compute boundary"
                )
            return self.result

        Actor._build_model_optimizer = builder
        Actor.compute_ref_log_prob = forward
        moves = []

        def move(model, destination):
            moves.append(destination)
            model.device = "cpu" if destination == "cpu" else "cuda:0"

        reference_backend = SimpleNamespace(
            move=move,
            no_grad=nullcontext,
            snapshot=lambda model: {"device": model.device},
        )
        self.assertTrue(
            screening_reference.patch_worker_class(
                Actor,
                backend=reference_backend,
                emit=lambda *a, **k: None,
                identity=IDENTITY,
            )
        )
        backend = Backend()
        registry = WorkerRegistry(identity=IDENTITY, backend=backend, emit=lambda *a, **k: None)
        self.assertTrue(patch_worker_classes(Actor, Critic, registry))
        actor, critic, reference = Actor(), Critic(), Actor(actor=False, ref=True)
        for worker in (reference, critic, actor):
            worker.init_model()
        reference._finance_manual_reference_state = screening_reference.ReferenceState(
            reference.ref_module_fsdp,
            IDENTITY,
        )
        data = object()
        self.assertIs(reference.compute_ref_log_prob(data), reference.result)
        self.assertEqual(moves, ["compute", "cpu"])
        self.assertEqual(reference.ref_module_fsdp.device, "cpu")
        self.assertFalse(reference._finance_manual_reference_state.busy)
        self.assertFalse(registry.coordinator.failed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
