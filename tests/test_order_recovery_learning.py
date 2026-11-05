"""CPU-only tests for the separately frozen one-round recovery acceptance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents import low_memory_comparison, low_memory_comparison_seed43, parameter_grid
from src.agents import low_memory_pilot as runner
from src.agents import order_recovery_learning as entry
from src.agents.metadata import RunHashes


def test_legacy_settings_are_byte_identical_to_pre_recovery_profiles():
    model = dict(source="local_export", name="/shared/sft", merge_receipt_sha256="a" * 64)
    expected = {
        runner.PILOT.name: "4917d5f319ee49bdffaba8f01cea9a84435f6937120eb629a83d306ba91ea738",
        low_memory_comparison.PROFILE.name:
            "21bfac8c1e8b7c806a20a5b9d455af97fc39434d72df087561b8eda45d9276eb",
        low_memory_comparison_seed43.PROFILE.name:
            "461da6da7701159e79db9eaab11af38ac04750818bfd3b95473476bf2d1d2fc2",
        parameter_grid.PROFILES[42].name:
            "cb3d59395e0e6b25285dc1a6850090f37ed5cdf736147a496b225e7981191e2b",
        parameter_grid.PROFILES[43].name:
            "09d9a67a828e9fd7c40c6b8f99d4231fd529de68abb43792807a8a3dc168b6f0",
    }
    profiles = [runner.PILOT, low_memory_comparison.PROFILE,
                low_memory_comparison_seed43.PROFILE, *parameter_grid.PROFILES.values()]
    for profile in profiles:
        serialized = json.dumps(runner.settings(model, profile=profile), sort_keys=True)
        assert hashlib.sha256(serialized.encode()).hexdigest() == expected[profile.name]
        assert not hasattr(profile, "disk_budget_profile")
        assert runner.disk_requirement_bytes(profile, 2) == 116 * 1024**3


def test_fixed_one_round_budget_counts_prepared_and_consumed_windows_separately():
    settings = entry.settings()
    assert settings["planned_trajectories"] == 272
    assert settings["training_trajectories"] == 32
    assert settings["evaluation_trajectories"] == 240
    assert settings["native_session_recovery_cap"] is None
    assert settings["native_interaction_cap"] == 256
    assert settings["four_round_performance_evidence"] is False
    assert settings["initial_free_disk_required_bytes"] == 74 * 1024**3
    assert settings["consumed_training_episode_ids"] == ["formal-train-0001", "formal-train-0065"]
    assert len(settings["prepared_training_episode_ids"]) == 8
    for profile in entry.PROFILES.values():
        assert profile.planned_trajectories == 136
        assert profile.rounds == 1 and profile.evaluation_steps == (1,)
        assert len(profile.eval_ids) == 8 and profile.test_freq == 1
        assert profile.retire_completed_moments and not profile.archive_completed_actors
    assert [s.arm for s in runner.cells(entry.PROFILES[42])] == ["M1", "M1", "M2"]
    assert [s.arm for s in runner.cells(entry.PROFILES[43])] == ["M1", "M2", "M1"]


@pytest.mark.parametrize("changed", [
    {"rounds": 2}, {"evaluation_steps": (2,)}, {"test_freq": 2},
    {"observation_profile": "history_features_v1"}, {"disk_budget_profile": "arbitrary"},
    {"retire_completed_moments": False}, {"archive_completed_actors": True},
])
def test_one_round_gate_cannot_be_used_for_another_lifecycle(changed):
    with pytest.raises(ValueError):
        replace(entry.PROFILES[42], **changed)


@pytest.mark.parametrize("seed", entry.SEEDS)
@pytest.mark.parametrize("cell_id", ["sft0", "m1", "m2_eta025"])
def test_real_hydra_routes_new_shared_sft_observation_budget_and_save_profile(
    tmp_path, seed, cell_id,
):
    hydra = pytest.importorskip("hydra")
    profile = entry.PROFILES[seed]
    model = dict(source="local_export", name=str(tmp_path / "new-recovery-sft"),
                 merge_receipt_sha256="a" * 64)
    settings = runner.settings(model, profile=profile)
    runner.atomic_json(tmp_path / "settings.json", settings)
    manifest = dict(
        hashes={"payload_sha256": "b" * 64},
        datasets={name: dict(path=name + ".parquet", episode_ids=ids, row_count=len(ids))
                  for name, ids in (("train", profile.train_ids), ("dev", profile.eval_ids))},
        calibration=dict(path="reward_calibration.json", payload_sha256="c" * 64, sha256="f" * 64),
    )
    cell = tmp_path / cell_id
    cell.mkdir()
    config = runner.make_config(tmp_path, manifest, cell_id, profile=profile)
    command = runner.build_command(
        config=config, run_settings=runner.cell_settings(settings, cell_id, profile=profile),
        hashes=RunHashes("d" * 40, "e" * 40, "fixture", True), cell_dir=cell, profile=profile,
    )
    with hydra.initialize_config_dir(
        config_dir=str(runner.REPO_ROOT / "third_party/agent-r1/agent_r1/config"),
        version_base=None,
    ):
        actual = hydra.compose(config_name="agent_ppo_trainer", overrides=command[3:])
    actor = actual.actor_rollout_ref
    assert actor.model.path == actor.ref.model.path == model["name"]
    assert actual.critic.model.path == actual.critic.model.tokenizer_path == model["name"]
    assert actual.data.screening_observation_profile == "history_features_recovery_v2"
    assert actual.data.train_max_samples == actual.data.val_max_samples == 8
    assert actual.data.train_batch_size == 2 and not actual.data.shuffle
    assert actual.trainer.total_training_steps == 1
    assert actual.trainer.test_freq == actual.trainer.save_freq == 1
    assert actual.trainer.val_only is (cell_id == "sft0")
    assert actual.trainer.resume_mode == "disable" and actual.trainer.resume_from_path is None
    assert actual.data.max_prompt_length == 8192 and actual.data.max_response_length == 4096
    assert actual.actor_rollout_ref.actor.optim.lr == 1e-7
    assert actual.critic.optim.lr == 1e-6
    assert actual.data.seed == actual.critic.model.fsdp_config.seed == seed
    assert actual.actor_rollout_ref.rollout.n == 4
    assert actual.actor_rollout_ref.rollout.val_kwargs.n == 5
    assert list(actor.actor.checkpoint.save_contents) == ["finance_actor_bf16_hf"]
    assert list(actual.critic.checkpoint.save_contents) == []
    env = actual.ray_kwargs.ray_init.runtime_env.env_vars
    assert env.FINANCE_COMMON_INVALID_PENALTY == "0.05"
    assert float(env.FINANCE_SCREENING_RESIDUAL_ETA) == (0.25 if cell_id == "m2_eta025" else 0)
    assert env.FINANCE_GRID_EVALUATION_INCOMPLETE == "retain_prefix_v1"


def _budget_model(path, *, extra=0, dtype="BF16", tied=True):
    path.mkdir()
    runner.atomic_json(path / "config.json", dict(
        model_type="qwen3", hidden_size=2560, tie_word_embeddings=tied,
    ))
    embedding = 151936 * 2560
    shapes = {"model.embed_tokens.weight": [151936, 2560],
              "other.weight": [4022468096 - embedding + extra]}
    cursor, header = 0, {}
    for name, shape in shapes.items():
        size = 2
        for value in shape:
            size *= value
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[cursor, cursor + size])
        cursor += size
    encoded = json.dumps(header).encode()
    # Sparse file tests real byte/header accounting without storing huge payloads.
    with (path / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + cursor)
    return path


def test_one_round_header_budget_counts_critic_head_and_duplicated_export_head(tmp_path):
    proof = runner.verify_one_round_model_budget(_budget_model(tmp_path / "model"))
    assert proof["actor_parameters"] == 4022468096
    assert proof["critic_parameters"] == 4022470657
    assert proof["maximum_moment_bytes"] == 32179765256
    assert proof["actor_export_tensor_bytes"] == 8822848512
    assert proof["role_exchanges"] == 2 and proof["optimizer_included"] is False
    assert runner.disk_requirement_bytes(entry.PROFILES[42], 4) == 74 * 1024**3


@pytest.mark.parametrize("change", [{"extra": 1}, {"dtype": "F32"}, {"tied": False}])
def test_one_round_header_budget_refuses_different_model(tmp_path, change):
    with pytest.raises(RuntimeError, match="budget|Qwen3|parameter"):
        runner.verify_one_round_model_budget(_budget_model(tmp_path / "model", **change))


def _merge_receipt(path):
    path.mkdir(exist_ok=True)
    receipt = dict(status="COMPLETE", model="/old/shared/sft", adapter="/new/recovery/adapter",
                   files={"model.safetensors": {"bytes": 10, "sha256": "f" * 64}})
    runner.atomic_json(path / "merge_receipt.json", receipt)
    return runner.restart.digest(path / "merge_receipt.json")


def test_prepare_pins_inventory_and_same_lease_for_both_seeds(tmp_path, monkeypatch):
    model = tmp_path / "new-sft"
    receipt = _merge_receipt(model)
    calls = []
    monkeypatch.setattr(runner, "_resolve_exact_model_snapshot",
                        lambda cfg: (str(model), "local-export-sha256:" + receipt, model))
    monkeypatch.setattr(runner, "verify_one_round_model_budget", lambda path: {"proof": True})
    monkeypatch.setattr(runner, "prepare", lambda child, **kwargs: calls.append((child, kwargs)))
    monkeypatch.setattr(entry, "_validate_sft_inputs", lambda *args: {"bound": True})
    base = tmp_path / "new-root"
    entry.prepare(base, model_path=model, merge_receipt_sha256=receipt,
                  sft_gate_readout=tmp_path / "sft-gate/readout.json",
                  coordinated_reservation_directory=tmp_path / "holder")
    assert [kw["profile"].seed for _, kw in calls] == [42, 43]
    assert calls[0][1]["coordinator_token"] == calls[1][1]["coordinator_token"]
    assert calls[0][1]["coordinator_token"]
    assert all(kw["model_path"] == model for _, kw in calls)
    assert all(kw["merge_receipt_sha256"] == receipt for _, kw in calls)
    shared = json.loads((base / "shared_sft.json").read_text())
    assert shared["merge_file_inventory"]["model.safetensors"]["sha256"] == "f" * 64
    assert shared["merge_receipt_sha256"] == receipt
    with pytest.raises(RuntimeError, match="new root"):
        entry.prepare(base, model_path=model, merge_receipt_sha256=receipt,
                      sft_gate_readout=tmp_path / "sft-gate/readout.json")
    with pytest.raises(RuntimeError, match="new shared recovery SFT"):
        entry.prepare(tmp_path / "other", model_path=model,
                      merge_receipt_sha256=entry.OLD_SHARED_SFT_RECEIPT,
                      sft_gate_readout=tmp_path / "sft-gate/readout.json")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    base, repo, model = tmp_path / "root", tmp_path / "repo", tmp_path / "new-sft"
    base.mkdir()
    repo.mkdir()
    (repo / "docs").mkdir()
    (repo / entry.PLAN).write_text("fixed stage1 sanity\n")
    code = repo / "source.py"
    code.write_text("stable source\n")
    receipt = _merge_receipt(model)
    files = json.loads((model / "merge_receipt.json").read_text())["files"]
    shared = dict(model=str(model), model_revision="local-export-sha256:" + receipt,
                  model_snapshot_path=str(model), merge_receipt_sha256=receipt,
                  merge_file_inventory=files,
                  merge_source_model="/old/shared/sft",
                  merge_source_adapter="/new/recovery/adapter",
                  disk_budget={"proof": True},
                  recovery_acceptance={"lineage_receipt": "/lineage.json",
                                       "sft_gate_readout": "/gate/readout.json"})
    runner.atomic_json(base / "learning_settings.json", entry.settings())
    runner.atomic_json(base / "shared_sft.json", shared)
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "source_files", lambda: {"source.py": code})
    monkeypatch.setattr(runner, "check_contract", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "verify_one_round_model_budget", lambda path: {"proof": True})
    monkeypatch.setattr(runner, "memory_coordinator_config", lambda child: None)
    monkeypatch.setattr(entry, "_validate_sft_inputs", lambda *args: shared["recovery_acceptance"])
    for seed in entry.SEEDS:
        child = entry.child_directory(base, seed)
        child.mkdir()
        runner.atomic_json(child / "model_identity.json", shared)

    def child_seal(child, **kwargs):
        runner.atomic_json(child / "run_contract.json", {"profile": kwargs["profile"].name})

    monkeypatch.setattr(runner, "seal", child_seal)
    entry.seal(base)
    return base, repo, model


@pytest.mark.parametrize("change", ["source", "source_copy", "child", "budget", "model", "receipt"])
def test_contract_rejects_source_budget_shared_sft_or_receipt_changes(sealed, change):
    base, repo, model = sealed
    if change == "source":
        (repo / "source.py").write_text("changed\n")
    elif change == "source_copy":
        (base / "sources/source.py").write_text("changed\n")
    elif change == "child":
        runner.atomic_json(base / "seed43/run_contract.json", {"changed": True})
    elif change == "budget":
        runner.atomic_json(base / "learning_settings.json", {"planned_trajectories": 999})
    elif change == "model":
        value = json.loads((base / "seed43/model_identity.json").read_text())
        value["model_revision"] = "different-init"
        runner.atomic_json(base / "seed43/model_identity.json", value)
    else:
        (model / "merge_receipt.json").write_text("{}")
    with pytest.raises((RuntimeError, KeyError), match="changed|differ|budget|files|fixed"):
        entry.check_contract(base)


def _stub_run(monkeypatch, *, fail_seed=None, free_gib=100, coordinated=True):
    calls = []
    monkeypatch.setattr(entry, "check_contract", lambda base: None)
    monkeypatch.setattr(runner, "memory_coordinator_config",
                        lambda child: {"holder": True} if coordinated else None)
    monkeypatch.setattr(entry.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=free_gib * 1024**3))

    def run(child, *, profile):
        calls.append(profile.seed)
        if profile.seed == fail_seed:
            raise RuntimeError("single worker failure")
        runner.atomic_json(child / "state.json", {"status": "COMPLETE"})

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(entry, "export_model_spec", lambda base: base / "model_spec_all.json")
    monkeypatch.setattr(entry, "_readout", lambda *a, **kw:
                        {"incomplete_evaluations": 0, "financial_status": "COMPLETE"})
    return calls


def test_success_runs_both_fresh_seeds_once_and_never_claims_bounded_acceptance(
    tmp_path, monkeypatch,
):
    calls = _stub_run(monkeypatch)
    entry.run(tmp_path)
    assert calls == [42, 43]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "COMPLETE" and state["completed_seeds"] == [42, 43]
    assert state["required_free_disk_bytes"] == 74 * 1024**3
    receipt = json.loads((tmp_path / "service_completion.json").read_text())
    assert receipt["four_round_performance_evidence"] is False
    assert receipt["strict_four_followup_acceptance_performed"] is False
    with pytest.raises(RuntimeError, match="no retry or resume"):
        entry.run(tmp_path)


def test_single_failed_seed_stops_batch_and_preserves_failure(tmp_path, monkeypatch):
    calls = _stub_run(monkeypatch, fail_seed=42)
    with pytest.raises(RuntimeError, match="single worker failure"):
        entry.run(tmp_path)
    assert calls == [42]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "FAILED" and state["completed_seeds"] == []
    assert "single worker failure" in state["error"]
    with pytest.raises(RuntimeError, match="no retry or resume"):
        entry.run(tmp_path)


@pytest.mark.parametrize("kwargs,error", [
    ({"free_gib": 73}, "insufficient disk"),
    ({"coordinated": False}, "persistent GPU holder"),
])
def test_run_fails_before_worker_without_disk_or_holder(tmp_path, monkeypatch, kwargs, error):
    calls = _stub_run(monkeypatch, **kwargs)
    with pytest.raises(RuntimeError, match=error):
        entry.run(tmp_path)
    assert calls == [] and not (tmp_path / "state.json").exists()


def test_readout_requires_same_fresh_head_and_preserves_native_incompletes(tmp_path):
    for seed, profile in entry.PROFILES.items():
        for spec in runner.cells(profile):
            path = entry.child_directory(tmp_path, seed) / spec.cell_id / "summary.json"
            runner.atomic_json(path,
                               dict(status="COMPLETE", actual_rounds=spec.rounds,
                                    critic_initial_head_sha256="a" * 64,
                                    incomplete_evaluations=int(spec.cell_id == "m2_eta025")))
    result = entry._readout(tmp_path, allow_partial=False)
    assert result["incomplete_evaluations"] == 2 and result["financial_status"] == "INCOMPLETE"
    path = tmp_path / "seed43/m1/summary.json"
    summary = json.loads(path.read_text())
    summary["critic_initial_head_sha256"] = "b" * 64
    runner.atomic_json(path, summary)
    with pytest.raises(RuntimeError, match="fresh critic heads differ"):
        entry._readout(tmp_path, allow_partial=False)


def test_preflight_uses_real_v2_rejection_and_complete_hold_replays(tmp_path, monkeypatch):
    import exchange_calendars as xcals
    import pandas as pd
    from transformers import AutoTokenizer

    import src.env.trading as trading
    from src.data import MarketObservation, StockBenchAdapter

    profile = entry.PROFILES[42]
    model = dict(source="local_export", name="/new-sft", merge_receipt_sha256="a" * 64)
    runner.atomic_json(tmp_path / "settings.json", runner.settings(model, profile=profile))
    monkeypatch.setattr(runner, "_resolve_exact_model_snapshot",
                        lambda cfg: ("/new-sft", "new-identity", Path("/new-sft")))
    seen_prompts, envs = [], []
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2023-01-03", "2024-10-31")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen_prompts.append(messages)
            return [1] * 100

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    original_env = trading.TradingEnv

    def real_env(**kwargs):
        dates = [datetime.fromisoformat(day) for day in kwargs["episode_dates"]]
        bars = [MarketObservation(
            observation_id=f"recovery-{stamp.isoformat()}", kind="market", symbol="AAA",
            event_at=stamp, available_at=stamp,
            payload={"date": stamp.date().isoformat(), "open": 100.0, "close": 100.0},
        ) for stamp in dates]
        env = original_env(
            adapter=StockBenchAdapter(bars, data_version="recovery-preflight-synthetic-v1"),
            episode_dates=dates,
            episode_open_times=[calendar.session_open(stamp.date().isoformat()).to_pydatetime()
                                for stamp in dates],
            symbols=["AAA"], initial_cash=1000.0, commission_bps=0.0, slippage_bps=0.0,
            drawdown_coefficient=0.0, turnover_coefficient=0.0,
            benchmark_closes=[100.0] * 21, observation_profile=kwargs["observation_profile"],
        )
        envs.append(env)
        return env

    monkeypatch.setattr(trading, "TradingEnv", real_env)
    def rows(path):
        ids = profile.train_ids if path.name == "train.parquet" else profile.eval_ids
        offset = 0 if path.name == "train.parquet" else 8
        return pd.DataFrame([dict(
            extra_info=dict(episode_id=episode, index=index, source_split="train"),
            env_kwargs=json.dumps(dict(episode_dates=[
                calendar.session_close(sessions[(offset + index) * 21 + day]).isoformat()
                for day in range(21)
            ])), prompt=[],
        ) for index, episode in enumerate(ids)])

    monkeypatch.setattr(pd, "read_parquet", rows)
    asyncio.run(runner.input_preflight(tmp_path, profile=profile))
    pre = json.loads((tmp_path / "input_preflight.json").read_text())
    assert pre["observation_profile"] == "history_features_recovery_v2"
    assert len(pre["windows"]) == 16
    assert all(window["complete"] and window["rejection_prompt_tokens"] == 100
               for window in pre["windows"])
    assert any('"order_rejection"' in str(messages) for messages in seen_prompts)
    assert all(env.episode_complete and not env.ledger.fills for env in envs)


def test_cpu_one_round_disk_peak_is_one_critic_payload_and_two_rounds_need_replacement(
    tmp_path, monkeypatch,
):
    from test_chunked_placement import accumulate_and_clip, registry_fixture

    from src.agents.optimizer_state_store import MomentStore

    registry, _, models, _ = registry_fixture(tmp_path, monkeypatch)
    samples = []
    original_write = MomentStore._write_chunk

    def track(store, path, data):
        result = original_write(store, path, data)
        samples.append(sum(file.stat().st_size for file in tmp_path.rglob("*.raw")))
        return result

    monkeypatch.setattr(MomentStore, "_write_chunk", track)

    def one_round(index):
        for role in ("critic", "actor"):
            phase = "update_" + role
            registry.begin(phase, activate=role)
            model, optimizer, scheduler = models[role]
            accumulate_and_clip(model, index)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            registry.finish(phase, role)
        for phase in ("rollout_mode", "compute_values", "save_checkpoint"):
            registry.begin(phase)
            registry.finish(phase, "actor")

    one_round(0)
    critic_bytes = 8 * sum(parameter.numel() for parameter in models["critic"][0].parameters())
    actor_bytes = 8 * sum(parameter.numel() for parameter in models["actor"][0].parameters())
    assert max(samples) == critic_bytes
    manifests = {role: json.loads(store.manifest_path.read_text())
                 for role, store in registry.coordinator.stores.items()}
    assert manifests["actor"]["payload_bytes"] == 0
    assert manifests["critic"]["payload_bytes"] == critic_bytes
    assert registry.coordinator.step_counts == {"actor": 1, "critic": 1}
    one_round(1)
    assert max(samples) == actor_bytes + 2 * critic_bytes


def test_seal_rejects_old_observation_preflight_before_freezing(tmp_path):
    profile = entry.PROFILES[42]
    model = dict(source="local_export", name="/new-sft", merge_receipt_sha256="a" * 64)
    runner.atomic_json(tmp_path / "settings.json", runner.settings(model, profile=profile))
    runner.atomic_json(tmp_path / "input_preflight.json", dict(
        status="PASSED", windows=[dict(episode=episode, complete=True)
                                  for episode in profile.train_ids + profile.eval_ids],
        observation_profile="history_features_v1",
    ))
    with pytest.raises(RuntimeError, match="feedback preflight must match"):
        runner.seal(tmp_path, profile=profile)
    assert not (tmp_path / "run_contract.json").exists()


def _tiny_sft_lineage(tmp_path):
    acceptance = entry.acceptance
    model = tmp_path / "sft"
    model.mkdir()
    for name, text in {"config.json": "{}", "tokenizer_config.json": "{}",
                       "model.safetensors": "genuine test weight bytes"}.items():
        (model / name).write_text(text)
    lineage = tmp_path / "sft-lineage.json"
    data = tmp_path / "training-data.jsonl"
    data.write_text("recovery training evidence\n")
    files = acceptance.hf_inventory(model, excluded={lineage})
    fingerprint = acceptance.fingerprint(files)
    runner.atomic_json(lineage, dict(
        status="COMPLETE", model_key="sft", model_fingerprint=fingerprint,
        action_protocol=acceptance.PROTOCOL, observation_profile=acceptance.OBSERVATION_PROFILE,
        acceptance_gate_sha256=acceptance.fingerprint(acceptance.ENDPOINT_GATE),
        training_updates=72, role="common_sft",
        training_evidence_sha256={str(data): runner.restart.digest(data)},
    ))
    gate = tmp_path / "sft-gate"
    gate.mkdir()
    runner.atomic_json(gate / "manifest.json", dict(
        mode="sft", models={"sft": {"fingerprint": fingerprint}},
    ))
    runner.atomic_json(gate / "readout.json", dict(
        status="OBSERVED", mode="sft", acceptance="PASS",
        endpoint_gate_sha256=acceptance.fingerprint(acceptance.ENDPOINT_GATE),
        manifest_sha256=runner.restart.digest(gate / "manifest.json"),
        raw_audit_sha256="a" * 64,
    ))
    return model, lineage, gate / "readout.json", fingerprint


@pytest.mark.parametrize("change", [None, "weights", "gate", "audit"])
def test_sft_gate_verifies_actual_fingerprint_endpoint_contract_and_raw_audit(
    tmp_path, monkeypatch, change,
):
    model, lineage, readout, fingerprint = _tiny_sft_lineage(tmp_path)
    monkeypatch.setattr(entry.acceptance, "verify_completed", lambda directory, expected, **kwargs:
                        json.loads((directory / "manifest.json").read_text()))
    monkeypatch.setattr(entry.acceptance, "raw_audit_gate", lambda *args, **kwargs: dict(
        status="FAIL" if change == "audit" else "PASS", acceptance_pass=True, sha256="a" * 64,
    ))
    if change == "weights":
        (model / "model.safetensors").write_text("different updated weights")
    elif change == "gate":
        value = json.loads(readout.read_text())
        value["endpoint_gate_sha256"] = "b" * 64
        runner.atomic_json(readout, value)
    if change is not None:
        with pytest.raises((RuntimeError, ValueError), match="gate|lineage"):
            entry._validate_sft_inputs(model, lineage, readout)
    else:
        bound = entry._validate_sft_inputs(model, lineage, readout)
        assert bound["model_fingerprint"] == fingerprint
        assert bound["lineage_receipt_sha256"] == runner.restart.digest(lineage)
        assert bound["sft_gate_readout_sha256"] == runner.restart.digest(readout)


def test_real_actor_lineage_spec_hashes_actual_exports_and_one_common_training_protocol(
    tmp_path, monkeypatch,
):
    from src.agents import actor_export

    sft, sft_lineage, readout, fingerprint = _tiny_sft_lineage(tmp_path)
    base = tmp_path / "learning"
    base.mkdir()
    runner.atomic_json(base / "learning_settings.json", entry.settings())
    runner.atomic_json(base / "run_contract.json", {"actual": "root contract"})
    runner.atomic_json(base / "shared_sft.json", dict(
        model_snapshot_path=str(sft), recovery_acceptance=dict(
            model_fingerprint=fingerprint, lineage_receipt=str(sft_lineage),
            sft_gate_readout=str(readout),
        ),
    ))
    monkeypatch.setattr(entry, "check_contract", lambda path: None)
    export_calls = []
    monkeypatch.setattr(actor_export, "validate_export_artifact",
                        lambda path, **kwargs: export_calls.append((path, kwargs)))
    for seed, profile in entry.PROFILES.items():
        child = entry.child_directory(base, seed)
        for filename in ("run_contract.json", "settings.json"):
            runner.atomic_json(child / filename, {"actual": filename})
        for spec in runner.cells(profile):
            cell = child / spec.cell_id
            runner.atomic_json(cell / "summary.json", dict(
                status="COMPLETE", actual_rounds=spec.rounds,
                execution_identity_sha256="e" * 64, critic_initial_head_sha256="c" * 64,
            ))
            if spec.zero:
                continue
            for filename in ("settings.json", "resolved_config.json", "command.json",
                             "artifacts/critic_initialization/head-rank0.json",
                             "checkpoints/global_step_1/actor_export_complete.json",
                             "checkpoints/global_step_1/actor/actor_export_receipt.json"):
                runner.atomic_json(cell / filename, {"actual": filename, "seed": seed})
            model = cell / "checkpoints/global_step_1/actor/huggingface"
            model.mkdir()
            for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
                (model / filename).write_text(f"{seed}-{spec.arm}-{filename}")
    destination = entry.export_model_spec(base)
    specification = json.loads(destination.read_text())
    assert set(specification) == {"sft", "seed42_m1", "seed42_m2", "seed43_m1", "seed43_m2"}
    assert len(export_calls) == 4 and all(kw["verify_hashes"] for _, kw in export_calls)
    contracts = entry.acceptance.model_contracts(destination, "all")
    protocol_sha = runner.restart.digest(base / "learning_settings.json")
    for key, value in contracts.items():
        if key == "sft":
            continue
        lineage = value["lineage"]
        assert lineage["common_training_protocol_sha256"] == protocol_sha
        assert protocol_sha in lineage["training_evidence_sha256"].values()
        assert lineage["training_updates"] == 1
        assert lineage["common_sft_fingerprint"] == fingerprint
        assert any(path.endswith("actor_export_receipt.json") for path in value["evidence"])
    assert entry.export_model_spec(base) == destination  # Identical evidence is reusable.
    actor = Path(specification["seed42_m1"]["model_path"])
    (actor / "model.safetensors").write_text("tampered actor bytes")
    with pytest.raises(RuntimeError, match="existing actor lineage differs"):
        entry.export_model_spec(base)
