"""CPU acceptance contract tests; every interaction uses the real trading environment."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import exchange_calendars
import pytest
from agent_r1.env.base import Action

from src.agents import order_recovery_acceptance as acceptance
from src.agents import order_recovery_episode as episode
from src.agents import order_recovery_probe as probe
from src.agents.coldstart_data import parse_state, tool_call
from src.data import MarketObservation, StockBenchAdapter
from src.env.trading import TradingEnv

audit_spec = importlib.util.spec_from_file_location(
    "acceptance_audit",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/audit_order_recovery_acceptance.py",
)
auditor = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(auditor)

# Actual historical bank identities include four step-zero anchors. An all-step4
# fixture concealed collisions with fresh source sample indexes 0..23.
KNOWN_SOURCE_ANCHORS = ((4, 0), (0, 2), (2, 6), (0, 9), (4, 10), (4, 11), (0, 1), (0, 3))


@pytest.fixture
def environment_bank(tmp_path):
    calendar = exchange_calendars.get_calendar("XNYS")
    sessions = {
        "known": calendar.sessions_in_range("2024-01-02", "2024-01-31"),
        "fresh": calendar.sessions_in_range("2023-06-01", "2023-06-30"),
    }
    dates = {
        partition: [calendar.session_close(day).to_pydatetime() for day in members]
        for partition, members in sessions.items()
    }
    adapter = StockBenchAdapter(
        [
            MarketObservation(
                observation_id=f"bar-{date.date()}",
                kind="market",
                symbol="AAA",
                event_at=date,
                available_at=date,
                payload={"date": date.date().isoformat(), "open": 100.0, "close": 100.0},
            )
            for group in dates.values()
            for date in group
        ],
        data_version="common-acceptance-fixture",
    )

    def env_class(**kwargs):
        return TradingEnv(adapter=adapter, **kwargs)

    async def replay(case, profile=probe.OLD):
        return await probe.replay_case(case, profile, env_class=env_class)

    async def reset(case):
        return await episode.reset_case(case, env_class=env_class)

    def make_case(key, index, partition, kind):
        kwargs = {
            "episode_dates": dates[partition],
            "episode_open_times": [
                calendar.session_open(day).to_pydatetime() for day in sessions[partition]
            ],
            "symbols": ["AAA"],
            "initial_cash": 1000.0,
            "commission_bps": 0.0,
            "slippage_bps": 0.0,
            "drawdown_coefficient": 0.0,
            "turnover_coefficient": 0.0,
            "benchmark_closes": [100.0] * 21,
            "observation_profile": probe.OLD,
            "reward_mode": "finance_trajectory",
            "finance_epsilon": 0.001,
            "screening": True,
            "finance_scorer_id": "benchmark_scale_v2",
            "finance_volatility_floor": 0.001,
            "screening_calibration_sha256": "f" * 64,
            "risk_lambda": 1.0,
            "turnover_limit": 1.0,
            "position_limit": 0.9,
            "capture_interactions": True,
        }
        prompt = [{"role": "user", "content": "Manage the portfolio."}]
        env = env_class(**kwargs)
        env.reset(raw_prompt=prompt)
        prefix = [tool_call("buy", {"symbol": "AAA", "target_weight": 0.5})]
        if kind == "rejection":
            name, target = (("buy", 0.2), ("sell", 0.7), ("buy", 0.5))[index % 3]
            prefix.append(tool_call(name, {"symbol": "AAA", "target_weight": target}))
        for response in prefix:
            asyncio.run(env.step(Action(text=response)))
        case = {
            "id": key,
            "kind": kind,
            "env_kwargs": kwargs,
            "raw_prompt": prompt,
            "source_base_seed": 44 if partition == "fresh" else 42,
            "source_identity": {
                "step": 0 if partition == "fresh" else KNOWN_SOURCE_ANCHORS[index][0],
                "sample_index": index if partition == "fresh" else KNOWN_SOURCE_ANCHORS[index][1],
                "rollout_n": 0,
                "validate": True,
            },
            "source": {"episode_id": "formal-train-fixture-fresh", "synthetic_actions": True},
            "prefix": prefix,
            "prefix_interactions": env.interaction_trace,
            "prefix_steps": [asdict(step) for step in env.credit_step_trace],
            "expected_start_session_index": env.current_session_index,
            "takeover_ledger": probe.ledger_state(env),
        }
        case = json.loads(probe.canonical(case))
        if partition == "fresh":
            _, obs = asyncio.run(replay(case, acceptance.OBSERVATION_PROFILE))
            state = parse_state(obs)
            expected = acceptance.endpoint_label(state)
            case.update(
                raw_state=state,
                cpu_expected_next_call=tool_call(expected["name"], expected["arguments"]),
                training_example=False,
                teacher_call_is_submitted=False,
            )
        return case

    known = {
        "cases": [
            make_case(key, index, "known", "rejection" if index < 6 else "normal")
            for index, key in enumerate(acceptance.KNOWN_IDS)
        ]
    }
    fresh = {
        "source_split": "train",
        "formal_Valid_or_OOS_used": False,
        "observation_profile": acceptance.OBSERVATION_PROFILE,
        "action_protocol": acceptance.PROTOCOL,
        "teacher_labels_force_actions": False,
        "training_acceptance_known_failure_dates_disjoint": True,
        "training_dataset_sha256": "dataset-placeholder",
        "train_decision_dates": ["2023-01-04"],
        "recovery_training_source_dates": ["2023-01-04"],
        "acceptance_source_dates": [date.date().isoformat() for date in dates["fresh"]],
        "known_failure_source_dates": [date.date().isoformat() for date in dates["known"]],
        "cases": [
            make_case(f"fresh-{index:02}", index, "fresh", "rejection") for index in range(24)
        ],
    }
    return known, fresh, replay, reset


def create_model(tmp_path, *, key="sft", sft_fingerprint=None):
    model = tmp_path / key
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(f"CPU fixture weights {key}".encode())
    evidence = tmp_path / "training.jsonl"
    if not evidence.exists():
        evidence.write_text("real training artifact fixture\n")
    protocol = tmp_path / "training_protocol.json"
    if not protocol.exists():
        protocol.write_text("predeclared identical paired protocol\n")
    files = acceptance.hf_inventory(model, excluded=set())
    lineage = {
        "status": "COMPLETE",
        "model_key": key,
        "model_fingerprint": acceptance.fingerprint(files),
        "action_protocol": acceptance.PROTOCOL,
        "observation_profile": acceptance.OBSERVATION_PROFILE,
        "acceptance_gate_sha256": acceptance.fingerprint(acceptance.ENDPOINT_GATE),
        "training_updates": 1,
        "training_evidence_sha256": {
            str(evidence): probe.sha(evidence),
            str(protocol): probe.sha(protocol),
        },
    }
    if key == "sft":
        lineage["role"] = "common_sft"
    else:
        lineage.update(
            method=key[-2:].upper(),
            training_seed=int(key[4:6]),
            common_sft_fingerprint=sft_fingerprint,
            actor_learning_rate=1e-7,
            common_training_protocol_sha256=probe.sha(protocol),
        )
    receipt = tmp_path / f"{key}_lineage.json"
    probe.save(receipt, lineage)
    return {"model_path": str(model), "lineage_receipt": str(receipt)}


def prepared_fixture(tmp_path, monkeypatch, environment_bank, *, mode="sft"):
    known, fresh, replay, reset = environment_bank
    specification = {"sft": create_model(tmp_path)}
    if mode == "all":
        sft = probe.read(Path(specification["sft"]["lineage_receipt"]))["model_fingerprint"]
        for key in acceptance.MODEL_KEYS[1:]:
            specification[key] = create_model(tmp_path, key=key, sft_fingerprint=sft)
    fresh["training_dataset_sha256"] = probe.sha(tmp_path / "training.jsonl")
    known_path, fresh_path, models_path = (
        tmp_path / name for name in ("known.json", "bank.json", "models.json")
    )
    for path, document in ((known_path, known), (fresh_path, fresh), (models_path, specification)):
        probe.save(path, document)
    original = acceptance.validate_bank

    async def validate(*args):
        return await original(*args, replay=replay, reset=reset)

    monkeypatch.setattr(acceptance, "validate_bank", validate)
    monkeypatch.setattr(
        acceptance,
        "runtime_sources",
        lambda: {str(Path(acceptance.__file__)): probe.sha(Path(acceptance.__file__))},
    )
    directory = tmp_path / "acceptance"
    acceptance.prepare(directory, known_path, fresh_path, models_path, mode)
    return directory, acceptance.verify(directory), replay, reset


class Tokenizer:
    def __init__(self, *, preserve=True, oversized=False):
        self.preserve, self.oversized = preserve, oversized
        self.responses = {}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return probe.canonical(messages)

    def encode(self, prompt, **kwargs):
        token = len(self.responses) + 1
        for message in reversed(json.loads(prompt)):
            text = (
                message["content"]
                .removeprefix("<tool_response>")
                .removesuffix("</tool_response>")
                .strip()
            )
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "session_index" in candidate:
                state = candidate
                break
        else:
            raise AssertionError("Actual observation is missing from prompt")
        call = (
            acceptance.endpoint_label(state)
            if self.preserve and "order_rejection" in state
            else {"name": "hold", "arguments": {}}
        )
        self.responses[token] = tool_call(call["name"], call["arguments"])
        return [token] * (8193 if self.oversized else 1)

    def decode(self, tokens, **kwargs):
        return self.responses[tokens[0]] if tokens else ""


class Engine:
    def __init__(self, *, truncate=False):
        self.count, self.truncate = 0, truncate

    def generate(self, prompts, params, **kwargs):
        outputs = []
        for prompt in prompts:
            self.count += 1
            ids = prompt["prompt_token_ids"]
            tokens = [ids[0]] * (4096 if self.truncate else 1)
            outputs.append(
                SimpleNamespace(
                    request_id=str(self.count),
                    prompt_token_ids=ids,
                    outputs=[
                        SimpleNamespace(
                            token_ids=tokens,
                            finish_reason="length" if self.truncate else "stop",
                            stop_reason=None,
                        )
                    ],
                )
            )
        return outputs


def test_budget_and_endpoint_validator_boundaries():
    assert acceptance.budget("sft")["planned_trajectories"] == 400
    assert acceptance.budget("all")["planned_trajectories"] == 1200
    current, tolerance = 0.5, 1e-12
    for target, name in (
        (current + tolerance, "hold"),
        (current - tolerance, "hold"),
        (current + 2 * tolerance, "buy"),
        (current - 2 * tolerance, "sell"),
    ):
        state = {
            "order_rejection": {
                "current_weight": current,
                "comparison_tolerance": tolerance,
                "rejected_call": {
                    "name": "buy",
                    "arguments": {"symbol": "aaa", "target_weight": target},
                },
            }
        }
        assert acceptance.endpoint_label(state)["name"] == name
    expected = {"name": "sell", "arguments": {"symbol": "AAA", "target_weight": 0.2}}
    assert acceptance.endpoint_matches(
        tool_call("sell", {"symbol": " aaa ", "target_weight": 0.2 + 5e-13}), expected
    )
    assert not acceptance.endpoint_matches(tool_call("hold"), expected)
    assert not acceptance.endpoint_matches(tool_call("abstain"), {"name": "hold", "arguments": {}})


def test_real_prefix_bank_disjoint_and_teacher_labels(environment_bank):
    known, fresh, replay, reset = environment_bank
    original = probe.canonical([known, fresh])
    short, full = asyncio.run(acceptance.validate_bank(known, fresh, replay=replay, reset=reset))
    assert len(short) == 32 and len(full) == 8
    assert probe.canonical([known, fresh]) == original
    assert [case["source_identity"] for case in short] == [
        case["source_identity"] for case in known["cases"] + fresh["cases"]
    ]
    assert [case["request_seed_identity"]["sample_index"] for case in short[8:]] == list(
        range(24, 48)
    )
    assert all("request_seed_identity" not in case for case in short[:8] + full)
    assert {case["endpoint_expected_call"]["name"] for case in short[8:]} == {"buy", "sell", "hold"}
    assert all(case["initial_ledger"]["session_index"] == 0 for case in full)
    assert all(case["initial_ledger"]["positions"] == {} for case in full)
    bad = copy.deepcopy(fresh)
    bad["cases"][0]["cpu_expected_next_call"] = tool_call("hold")
    with pytest.raises(ValueError, match="Offline expected endpoint"):
        asyncio.run(acceptance.validate_bank(known, bad, replay=replay, reset=reset))
    bad = copy.deepcopy(fresh)
    bad["train_decision_dates"] = fresh["acceptance_source_dates"][:1]
    with pytest.raises(ValueError, match="source dates overlap"):
        asyncio.run(acceptance.validate_bank(known, bad, replay=replay, reset=reset))


def test_real_known_collision_namespace_and_legacy_seeds(environment_bank):
    from src.agents.rollout_seed import derive_request_seed

    known, fresh, replay, reset = environment_bank
    originals = known["cases"] + fresh["cases"]
    raw_anchors = [
        (c["source_identity"]["step"], c["source_identity"]["sample_index"]) for c in originals
    ]
    assert len(set(raw_anchors)) == 28  # The four actual production collisions.
    short, full = asyncio.run(acceptance.validate_bank(known, fresh, replay=replay, reset=reset))
    models = {"sft": {"identities": [{"id": f"seed{s}_sft", "base_seed": s} for s in (42, 43)]}}
    manifest = {
        "cases": short,
        "models": models,
        "budget": {"profiles": [acceptance.OBSERVATION_PROFILE]},
    }
    rows = probe.grid(manifest, "sft")
    by_id = {row["id"]: row for row in rows}
    legacy = probe.grid({**manifest, "cases": originals}, "sft")
    full_rows = episode.grid({"cases": full, "models": models}, "sft")
    full_by_id = {row["id"]: row for row in full_rows}
    actual_seed_payloads = {}
    for old_row in legacy:
        row = by_id[old_row["id"]]
        source = next(c for c in originals if c["id"] == row["case_id"])
        sibling = old_row["rollout_identity"]["rollout_n"]
        original_identity = {**source["source_identity"], "rollout_n": sibling, "validate": True}
        # Absence of an explicit override preserves every legacy request seed.
        for offset in range(4):
            env_step = len(source["prefix"]) + offset
            assert derive_request_seed(
                old_row["base_seed"], old_row["rollout_identity"], env_step
            ) == (derive_request_seed(old_row["base_seed"], original_identity, env_step))
            seed = derive_request_seed(row["base_seed"], row["rollout_identity"], env_step)
            payload = (row["base_seed"], tuple(sorted(row["rollout_identity"].items())), env_step)
            assert actual_seed_payloads.setdefault(seed, payload) == payload
        if row["case_id"] in acceptance.KNOWN_IDS:
            assert row["rollout_identity"] == old_row["rollout_identity"]
            assert full_by_id[row["id"]]["rollout_identity"] == original_identity
        else:
            assert row["rollout_identity"] != old_row["rollout_identity"]
    # Every method shares exactly the same identity for a given request seed.
    actor = {"seed42_m1": {"identities": [{"id": "seed42_m1", "base_seed": 42}]}}
    actor_rows = probe.grid({**manifest, "models": actor}, "seed42_m1")
    for row in actor_rows:
        paired = by_id[row["id"].replace("seed42_m1:", "seed42_sft:")]
        assert row["rollout_identity"] == paired["rollout_identity"]
        assert probe.sampling_parameters(row, 2) == probe.sampling_parameters(paired, 2)


def test_namespace_source_binding_is_independently_audited(tmp_path, monkeypatch, environment_bank):
    _, contract, _, _ = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    acceptance.verify_request_seed_namespace(contract)
    auditor.check_request_seed_namespace(contract)
    for kind in ("override", "source", "known", "start", "guard"):
        bad = copy.deepcopy(contract)
        if kind == "override":
            bad["short_cases"][8]["request_seed_identity"]["sample_index"] += 1
        elif kind == "source":
            bad["short_cases"][8]["source_identity"]["sample_index"] += 1
        elif kind == "known":
            bad["short_cases"][0]["request_seed_identity"] = bad["short_cases"][0][
                "source_identity"
            ]
        elif kind == "start":
            bad["request_seed_namespace"]["sample_index_start"] += 1
        else:
            bad["request_seed_namespace"]["seed_collision_check"]["unique_payloads_and_seeds"] += 1
        with pytest.raises(ValueError, match="namespace|provenance|override"):
            acceptance.verify_request_seed_namespace(bad)
        with pytest.raises(ValueError, match="namespace|provenance|override"):
            auditor.check_request_seed_namespace(bad)


def test_actual_seed_hash_collision_blocks_prepare_and_verify(
    tmp_path, monkeypatch, environment_bank
):
    directory, contract, _, _ = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    assert contract["request_seed_namespace"]["seed_collision_check"] == {
        "derived_requests": 21760,
        "unique_payloads_and_seeds": 21440,
        "identical_payload_reuses": 320,
        "distinct_payload_collisions": 0,
    }
    monkeypatch.setattr(probe, "derive_request_seed", lambda *args: 0)
    with pytest.raises(ValueError, match="Distinct request payloads collide"):
        acceptance.verify(directory, check_models=False)
    blocked = tmp_path / "colliding-prepare"
    with pytest.raises(ValueError, match="Distinct request payloads collide"):
        acceptance.prepare(
            blocked,
            tmp_path / "known.json",
            tmp_path / "bank.json",
            tmp_path / "models.json",
            "sft",
        )
    assert not blocked.exists()


def test_absent_models_never_create_prepared_directory(tmp_path):
    specification = tmp_path / "spec.json"
    probe.save(
        specification,
        {
            "sft": {
                "model_path": str(tmp_path / "absent"),
                "lineage_receipt": str(tmp_path / "absent-receipt"),
            }
        },
    )
    with pytest.raises(ValueError, match="lineage receipt does not exist"):
        acceptance.prepare(
            tmp_path / "output", tmp_path / "unknown", tmp_path / "bank", specification, "sft"
        )
    assert not (tmp_path / "output").exists()


def test_dynamic_hf_contract_detects_changed_weight_and_lineage(tmp_path):
    entry = create_model(tmp_path)
    path = tmp_path / "models.json"
    probe.save(path, {"sft": entry})
    contracts = acceptance.model_contracts(path, "sft")
    assert len(contracts["sft"]["identities"]) == 2
    (Path(entry["model_path"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="training lineage"):
        acceptance.model_contracts(path, "sft")


@pytest.mark.parametrize("preserve,expected", [(True, "PENDING_AUDIT"), (False, "FAIL")])
def test_full_400_real_env_population_and_no_all_hold_gate(
    tmp_path, monkeypatch, environment_bank, preserve, expected
):
    directory, contract, replay, reset = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    result = asyncio.run(
        acceptance.execute(
            Engine(),
            Tokenizer(preserve=preserve),
            SimpleNamespace,
            contract,
            "sft",
            directory,
            replay=replay,
            reset=reset,
        )
    )
    assert result["status"] == "OBSERVED"
    readout = acceptance.readout(directory)
    assert readout["acceptance"] == expected
    assert len(readout["short_grid"]) == 320 and len(readout["full_grid"]) == 80
    assert readout["endpoint_adherence"]["legal_advances"] == 240
    assert readout["endpoint_adherence"]["endpoint_adherent"] == (240 if preserve else 80)
    assert all(row["status"] == "COMPLETE" for row in readout["full_grid"])
    assert all(
        not {"terminal_metrics", "full_window_net_return"} & set(row)
        for row in readout["short_grid"]
    )
    assert "cpu_expected" not in "".join(
        record["prompt"]
        for record in acceptance.json_lines(directory / "short/workers/sft/requests.jsonl")
    )
    audited = auditor.audit(directory)
    assert audited["status"] == "PASS" and audited["trajectories"] == 400
    assert audited["acceptance_pass"] is preserve
    probe.save(directory / "raw_audit.json", audited)
    assert acceptance.readout(directory)["acceptance"] == ("PASS" if preserve else "FAIL")


def test_truncation_retains_400_failures_without_finance(tmp_path, monkeypatch, environment_bank):
    directory, contract, replay, reset = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    asyncio.run(
        acceptance.execute(
            Engine(truncate=True),
            Tokenizer(),
            SimpleNamespace,
            contract,
            "sft",
            directory,
            replay=replay,
            reset=reset,
        )
    )
    readout = acceptance.readout(directory)
    assert readout["acceptance"] == "FAIL"
    assert all(
        row["requests"] == 1 and row["failure"] == "RESPONSE_INCOMPLETE"
        for row in readout["short_grid"] + readout["full_grid"]
    )
    assert all(
        not {"terminal_metrics", "full_window_net_return"} & set(row)
        for row in readout["short_grid"] + readout["full_grid"]
    )
    assert all(
        record["environment_applied"] is False
        for phase in ("short", "full")
        for record in acceptance.json_lines(directory / phase / "workers/sft/requests.jsonl")
    )
    audited = auditor.audit(directory)
    assert audited["status"] == "PASS" and audited["acceptance_pass"] is False


def test_missing_worker_keeps_every_planned_cell(tmp_path, monkeypatch, environment_bank):
    directory, _, _, _ = prepared_fixture(tmp_path, monkeypatch, environment_bank)
    readout = acceptance.readout(directory)
    assert readout["status"] == "INCOMPLETE" and readout["acceptance"] == "PENDING"
    assert len(readout["short_grid"]) == 320 and len(readout["full_grid"]) == 80
    assert all(row["status"] == "UNSTARTED" for row in readout["short_grid"] + readout["full_grid"])
