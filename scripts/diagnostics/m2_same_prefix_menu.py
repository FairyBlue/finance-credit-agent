#!/usr/bin/env python3
"""CPU-only, exact-prompt full-prefix branches and visible-state request menus."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Tokenization is local and does not import model backends or contact a hub.
for variable in ("USE_TORCH", "USE_TF", "USE_FLAX"):
    os.environ[variable] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/agent-r1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
from agent_r1.env.base import Action  # noqa: E402
from m2_state_order_oracle import (  # noqa: E402
    BASE,
    HOLD,
    append,
    canonical,
    financial_state,
    full_text,
    live,
    object_sha,
    records_from_manifest,
    sha,
    verify_inventory,
    write_new,
)

from src.agents.screening_prompt import apply_screening_action_protocol  # noqa: E402
from src.env.trading import TradingEnv  # noqa: E402

PREVIOUS = ROOT / "runs/m2-state-order-oracle-20261017-v1"
DEFAULT = ROOT / "runs/m2-same-prefix-menu-20261018-v1"
PLAN = ROOT / "docs/2026-10-18_M2_SAME_PREFIX_MENU_PLAN.md"
READOUT = ROOT / "scripts/diagnostics/m2_same_prefix_menu_readout.py"
TOKENIZER = ROOT / "runs/m2-order-recovery-sft-20261011-v1/merged_model"
GRID = (0.0, 0.1, 0.2)
MARGIN = 1e-6
PROTOCOL = "single_tool_portfolio_v1"
TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}


def action(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    call = {"name": name, "arguments": arguments or {}}
    key = json.dumps(call, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "action_key": key,
        "family": name,
        "symbol": call["arguments"].get("symbol"),
        "target_weight": call["arguments"].get("target_weight"),
        "response_text": "<tool_call>" + key + "</tool_call>",
    }


def menu_from_visible(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Pure function: no environment, future information or outcome argument."""
    symbols = sorted(state["market"])
    assert len(symbols) == 20
    result = [action("hold")]
    for symbol in symbols:
        weight = float(state["positions"].get(symbol, {}).get("weight", 0.0))
        assert math.isfinite(weight) and weight >= 0
        for target in GRID:
            if target > weight + MARGIN:
                result.append(action("buy", {"symbol": symbol, "target_weight": target}))
            elif target < weight - MARGIN:
                result.append(action("sell", {"symbol": symbol, "target_weight": target}))
    assert len({row["action_key"] for row in result}) == len(result) <= 61
    return result


def probes_for(menu: list[dict[str, Any]]) -> list[str]:
    # Order keys, not achieved returns, determine every probe.
    buy = sorted(row["action_key"] for row in menu if row["family"] == "buy")
    sell = sorted(row["action_key"] for row in menu if row["family"] == "sell")
    assert buy, "no buy support: retain failure, do not substitute hold"
    return [menu[0]["action_key"], buy[0], buy[0], sell[-1] if sell else buy[-1]]


def visible(obs: Any) -> dict[str, Any]:
    content = obs.messages[-1]["content"]
    if content.startswith("<tool_response>"):
        content = content.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
    return json.loads(content)


def no_model_backends() -> None:
    assert not any(name in sys.modules for name in ("torch", "vllm", "tensorflow", "jax"))


def prepare(out: Path) -> None:
    old = json.loads((PREVIOUS / "contract.json").read_text())
    verify_inventory(old["source_sha256"])
    done = json.loads((PREVIOUS / "collection_completion.json").read_text())
    assert done["status"] == "COMPLETE" and done["states"] == 256
    assert all(sha(PREVIOUS / name) == value for name, value in done["output_sha256"].items())
    inventory = dict(old["source_sha256"])
    for name in (
        "contract.json",
        "source_manifest.json",
        "collection_completion.json",
        "state_results.jsonl",
    ):
        inventory[str(PREVIOUS / name)] = sha(PREVIOUS / name)
    files = [Path(__file__).resolve(), READOUT, PLAN]
    files += [ROOT / "src/agents/screening_prompt.py", ROOT / "src/agents/coldstart_data.py"]
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    ):
        path = TOKENIZER / name
        if path.exists():
            files.append(path)
    assert (TOKENIZER / "tokenizer.json").exists()
    for path in files:
        inventory[str(path)] = sha(path)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "source_manifest.json").open("xb") as handle:
        handle.write((PREVIOUS / "source_manifest.json").read_bytes())
    write_new(
        out / "contract.json",
        {
            "profile": "same_full_prefix_visible_menu_cpu_v1",
            "status": "SEALED_BEFORE_SNAPSHOT_REPLAY",
            "sealed_at_UTC": datetime.now(timezone.utc),
            "source_sha256": inventory,
            "source_manifest_sha256": sha(out / "source_manifest.json"),
            "previous_run": str(PREVIOUS),
            "anchors": 256,
            "menu_rule": {
                "targets": GRID,
                "visible_weight_margin": MARGIN,
                "symbols": "all frozen 20, sorted",
            },
            "tokenizer_path": str(TOKENIZER),
            "action_protocol": PROTOCOL,
            "chat_template_kwargs": TEMPLATE_KWARGS,
            "maximum_menu_first_steps": 15616,
            "maximum_total_first_steps": 16128,
            "maximum_terminal_branches": 1280,
            "probe_slots": ["hold", "first_buy", "first_buy", "last_sell_else_last_buy"],
            "model_requests": 0,
            "optimizer_updates": 0,
            "GPU_used": False,
            "formal_valid_oos_access": False,
            "branch_cache": False,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pyarrow",
                    "pydantic",
                    "exchange-calendars",
                    "transformers",
                    "tokenizers",
                    "jinja2",
                )
            },
        },
    )
    print("SEALED", out / "contract.json", flush=True)


class Replay:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.records = records_from_manifest(manifest)
        self.manifest = manifest
        self.requests = 0
        self.environments = 0
        self.data: dict[tuple[int, str], dict[str, Any]] = {}
        for seed in (42, 43):
            for row in pq.read_table(BASE / f"seed{seed}/prepared/train.parquet").to_pylist():
                assert row["extra_info"]["source_split"] == "train"
                kwargs = json.loads(row["env_kwargs"])
                assert kwargs["screening"] and "adapter" not in kwargs
                assert kwargs["data_start"] == "2023-01-03" and kwargs["data_end"] == "2024-10-31"
                assert len(kwargs["episode_dates"]) == 21 and kwargs["input_file_count"] == 9220
                kwargs.update(capture_interactions=True, reward_mode="finance_trajectory")
                self.data[seed, row["extra_info"]["episode_id"]] = {
                    "kwargs": kwargs,
                    "prompt": row["prompt"],
                    "sha": object_sha(kwargs),
                }

    async def prefix(self, record_index: int, stop: int) -> tuple[TradingEnv, Any, str]:
        locator = self.manifest["records"][record_index]
        record = self.records[record_index]
        data = self.data[locator["seed"], locator["episode_id"]]
        env = TradingEnv(**data["kwargs"])
        self.environments += 1
        obs = env.reset(raw_prompt=data["prompt"])
        for index in range(stop):
            interaction, credit = record["interactions"][index], record["steps"][index]
            obs, _, _, _ = await env.step(
                Action(text=full_text(interaction["response"], "response"))
            )
            self.requests += 1
            assert env.interaction_trace[-1] == interaction
            assert json.loads(canonical(asdict(env.credit_step_trace[-1]))) == credit
        if stop:
            assert obs.messages == json.loads(
                full_text(record["interactions"][stop - 1]["observation"], "observation")
            )
        assert not env.episode_complete and not env.ledger.pending_orders
        return env, obs, data["sha"]


def first_step_checks(
    before: dict[str, Any], after: dict[str, Any], info: dict[str, Any], env: TradingEnv
) -> None:
    """Independent cash/position/fee identities; preserve score-relevant history."""
    old_ledger, new_ledger = before["ledger"], after["ledger"]
    assert after["initial_cash"] == before["initial_cash"]
    assert new_ledger["session_index"] == old_ledger["session_index"] + 1
    assert after["equity_curve"][:-1] == before["equity_curve"]
    assert after["net_returns"][:-1] == before["net_returns"]
    assert after["turnovers"][:-1] == before["turnovers"]
    assert new_ledger["fills"][: len(old_ledger["fills"])] == old_ledger["fills"]
    assert new_ledger["fills"][len(old_ledger["fills"]) :] == info["fills"]
    assert after["high_water_mark"] == max(before["high_water_mark"], info["equity"])
    assert after["maximum_position_weight"] >= before["maximum_position_weight"]
    cash = old_ledger["cash"]
    positions = dict(old_ledger["positions"])
    for fill in info["fills"]:
        assert fill["side"] in ("buy", "sell")
        direction = 1 if fill["side"] == "buy" else -1
        notional = fill["shares"] * fill["execution_price"]
        assert math.isclose(fill["notional"], notional, rel_tol=1e-12, abs_tol=1e-8)
        assert math.isclose(
            fill["commission"],
            notional * env.ledger.policy.commission_rate,
            rel_tol=1e-12,
            abs_tol=1e-8,
        )
        slippage = fill["shares"] * abs(fill["execution_price"] - fill["raw_open_price"])
        assert math.isclose(fill["slippage_cost"], slippage, rel_tol=1e-12, abs_tol=1e-8)
        assert math.isclose(
            fill["execution_price"],
            fill["raw_open_price"] * (1 + direction * env.ledger.policy.slippage_rate),
            rel_tol=1e-12,
            abs_tol=1e-8,
        )
        assert datetime.fromisoformat(fill["executed_at"]) > datetime.fromisoformat(
            fill["decision_at"]
        )
        cash -= direction * notional + fill["commission"]
        positions[fill["symbol"]] = positions.get(fill["symbol"], 0.0) + direction * fill["shares"]
    assert math.isclose(cash, new_ledger["cash"], rel_tol=1e-12, abs_tol=1e-7)
    assert set(positions) == set(new_ledger["positions"])
    assert all(
        math.isclose(v, new_ledger["positions"].get(k, 0.0), rel_tol=1e-12, abs_tol=1e-8)
        for k, v in positions.items()
    )
    assert new_ledger["cash"] >= 0 and all(v >= 0 for v in new_ledger["positions"].values())
    assert math.isclose(after["equity_curve"][-1], info["equity"], rel_tol=1e-12)
    assert math.isclose(
        after["net_returns"][-1], info["portfolio_return_net"], rel_tol=1e-12, abs_tol=1e-12
    )
    assert after["turnovers"][-1] == info["turnover"]


async def collect(out: Path) -> None:
    from transformers import AutoTokenizer

    started = time.monotonic()
    contract = json.loads((out / "contract.json").read_text())
    verify_inventory(contract["source_sha256"])
    assert sha(out / "source_manifest.json") == contract["source_manifest_sha256"]
    no_model_backends()
    write_new(
        out / "start_receipt.json",
        {
            "started_at_UTC": datetime.now(timezone.utc),
            "contract_sha256": sha(out / "contract.json"),
            "automatic_retry": False,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER), local_files_only=True, trust_remote_code=False
    )
    manifest = json.loads((out / "source_manifest.json").read_text())
    replay = Replay(manifest)
    old_states = [
        json.loads(line) for line in (PREVIOUS / "state_results.jsonl").read_text().splitlines()
    ]
    assert len(old_states) == 256
    live(out, status="RUNNING", phase="STORAGE_PREFLIGHT_AND_ALL_PREFIX_SNAPSHOTS", anchors=0)
    anchors: list[dict[str, Any]] = []
    packets: dict[str, dict[str, Any]] = {}
    with (out / "anchors.jsonl").open("x") as ah, (out / "prompt_packets.jsonl").open("x") as ph:
        for previous in old_states:
            index, stop = previous["record_index"], previous["prefix_requests"]
            env, obs, config_sha = await replay.prefix(index, stop)
            before = json.loads(canonical(financial_state(env, config_sha)))
            assert object_sha(before) == previous["full_state_sha256"]
            assert env.current_session_index == previous["decision_session_index"]
            record = replay.records[index]
            request = record["requests"][stop]
            assert (
                request["request_index"] == stop and request["prompt_token_ids_truncated"] is False
            )
            assert record["action_protocol"] == PROTOCOL
            protocol_messages = apply_screening_action_protocol(obs.messages, PROTOCOL)
            ids = tokenizer.apply_chat_template(
                protocol_messages, tools=env.tool_schemas, tokenize=True, **TEMPLATE_KWARGS
            )
            text = tokenizer.apply_chat_template(
                protocol_messages, tools=env.tool_schemas, tokenize=False, **TEMPLATE_KWARGS
            )
            assert ids == request["prompt_token_ids"]
            prompt_hash = hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest()
            assert (
                prompt_hash == request["prompt_sha256"]
                and len(ids) == request["prompt_token_count"] <= 8192
            )
            state = visible(obs)
            assert sorted(state["market"]) == sorted(env.symbols)
            menu = menu_from_visible(state)
            probes = probes_for(menu)
            anchor_id = f"a{len(anchors):03d}"
            original_slot = next(
                s
                for s in manifest["records"][index]["slots"]
                if s["decision_session_index"] == previous["decision_session_index"]
            )
            diagonal = previous["cells"][previous["sibling_id"]]
            anchor = {
                "anchor_id": anchor_id,
                "record_index": index,
                "source_group_id": previous["group_id"],
                **{
                    k: previous[k]
                    for k in (
                        "seed",
                        "episode_id",
                        "policy_step",
                        "decision_session_index",
                        "sibling_id",
                        "prefix_requests",
                    )
                },
                "financial_state": before,
                "financial_state_sha256": object_sha(before),
                "observation_messages_sha256": object_sha(obs.messages),
                "tool_schemas_sha256": object_sha(env.tool_schemas),
                "historical_prompt_sha256": prompt_hash,
                "prompt_token_count": len(ids),
                "prefix_audit_sha256": object_sha(record["interactions"][:stop]),
                "visible_state": state,
                "menu_input_sha256": object_sha(state),
                "menu": menu,
                "menu_sha256": object_sha(menu),
                "probe_action_keys": probes,
                "old_hold_Y": previous["hold_Y"],
                "old_diagonal_Y": diagonal["Y"],
                "original_action_key": original_slot["canonical_action_json"],
                "original_response_text": original_slot["response_text"],
                "original_first_info": record["interactions"][stop]["info"],
                "original_order_in_menu": original_slot["canonical_action_json"]
                in {m["action_key"] for m in menu},
            }
            packet = {
                "anchor_id": anchor_id,
                "messages": obs.messages,
                "protocol_messages": protocol_messages,
                "tool_schemas": env.tool_schemas,
                "action_protocol": PROTOCOL,
                "chat_template_kwargs": TEMPLATE_KWARGS,
                "rendered_prompt_text": text,
                "prompt_token_ids": ids,
                "historical_prompt_sha256": prompt_hash,
                "prompt_token_count": len(ids),
                "observation_messages_sha256": object_sha(obs.messages),
                "future_sampling_slots": 4,
                "model_requests": 0,
                "menu_injected_into_prompt": False,
            }
            anchors.append(anchor)
            packets[anchor_id] = packet
            append(ah, anchor)
            append(ph, packet)
            if len(anchors) % 16 == 0:
                print("snapshot", len(anchors), "/256", "requests", replay.requests, flush=True)
                live(
                    out,
                    status="RUNNING",
                    phase="SNAPSHOTS_BEFORE_NEW_BRANCHES",
                    anchors=len(anchors),
                    environment_requests=replay.requests,
                )
    assert len(anchors) == 256
    verify_inventory(contract["source_sha256"])
    snapshot_hashes = {name: sha(out / name) for name in ("anchors.jsonl", "prompt_packets.jsonl")}
    write_new(
        out / "snapshot_completion.json",
        {
            "status": "ALL_MENUS_AND_PROMPTS_FROZEN_BEFORE_NEW_BRANCH_OUTCOMES",
            "finished_at_UTC": datetime.now(timezone.utc),
            "anchors": 256,
            "menu_orders": sum(len(a["menu"]) for a in anchors),
            "snapshot_sha256": snapshot_hashes,
            "new_intervention_requests_so_far": 0,
            "environment_requests_in_historical_prefixes": replay.requests,
        },
    )

    counts: Counter[str] = Counter()
    with (out / "branches.jsonl").open("x") as bh:
        for anchor in anchors:
            anchor_id = anchor["anchor_id"]
            selected = set(anchor["probe_action_keys"])
            menu_results: dict[str, dict[str, Any]] = {}

            async def branch(
                order: dict[str, Any],
                role: str,
                terminal: bool,
                anchor: dict[str, Any] = anchor,
                anchor_id: str = anchor_id,
            ) -> dict[str, Any]:
                env, obs, config_sha = await replay.prefix(
                    anchor["record_index"], anchor["prefix_requests"]
                )
                before = json.loads(canonical(financial_state(env, config_sha)))
                assert object_sha(before) == anchor["financial_state_sha256"]
                assert obs.messages == packets[anchor_id]["messages"]
                assert object_sha(obs.messages) == anchor["observation_messages_sha256"]
                assert object_sha(env.tool_schemas) == anchor["tool_schemas_sha256"]
                counts["before_identity_checks"] += 1
                session = env.current_session_index
                _, _, _, info = await env.step(Action(text=order["response_text"]))
                replay.requests += 1
                counts["first_steps"] += 1
                normalized_info = json.loads(canonical(info))
                after = json.loads(canonical(financial_state(env, config_sha)))
                result: dict[str, Any] = {
                    "anchor_id": anchor_id,
                    "branch_role": role,
                    "action_key": order["action_key"],
                    "family": order["family"],
                    "financial_state_sha256": anchor["financial_state_sha256"],
                    "observation_messages_sha256": anchor["observation_messages_sha256"],
                    "historical_prompt_sha256": anchor["historical_prompt_sha256"],
                    "first_info": normalized_info,
                    "first_state_after": after,
                    "first_cost": sum(
                        f["commission"] + f["slippage_cost"] for f in info.get("fills", [])
                    ),
                    "zero_fill": not info.get("fills"),
                    "skipped_count": len(info.get("skipped_orders", [])),
                    "Y": None,
                    "terminal_metrics": None,
                    "suffix_hold_requests": 0,
                    "checks": {"before_identity": True},
                }
                if info["invalid_action"]:
                    assert env.current_session_index == session
                    result["status"] = "INADMISSIBLE_AT_DECISION"
                    counts["invalid_first_steps"] += 1
                else:
                    assert env.current_session_index == session + 1
                    first_step_checks(before, after, normalized_info, env)
                    result["checks"].update(history_append=True, cash_position_fees=True)
                    result["status"] = "LEGAL_ADVANCING"
                    counts["legal_first_steps"] += 1
                    if terminal:
                        while not env.episode_complete:
                            _, _, _, suffix_info = await env.step(Action(text=HOLD))
                            replay.requests += 1
                            assert not suffix_info["invalid_action"] and not suffix_info["fills"]
                            result["suffix_hold_requests"] += 1
                            assert result["suffix_hold_requests"] <= 20
                        metrics = env.screening_terminal_metrics()
                        result.update(Y=metrics["terminal_score"], terminal_metrics=metrics)
                        counts["terminal_branches"] += 1
                append(bh, result)
                return result

            for order in anchor["menu"]:
                result = await branch(order, "menu", order["action_key"] in selected)
                menu_results[order["action_key"]] = result
                counts["menu_first_steps"] += 1
            hold = menu_results[anchor["probe_action_keys"][0]]
            assert hold["Y"] is not None and math.isclose(
                hold["Y"], anchor["old_hold_Y"], abs_tol=1e-12, rel_tol=0
            )
            first_buy = next(
                order
                for order in anchor["menu"]
                if order["action_key"] == anchor["probe_action_keys"][1]
            )
            duplicate = await branch(first_buy, "repeat_buy", True)
            original_buy = menu_results[first_buy["action_key"]]
            assert all(
                duplicate[k] == original_buy[k]
                for k in (
                    "first_info",
                    "first_state_after",
                    "Y",
                    "terminal_metrics",
                    "suffix_hold_requests",
                )
            )
            counts["repeat_buy_exact"] += 1
            original_call = json.loads(anchor["original_action_key"])
            control_order = action(original_call["name"], original_call["arguments"])
            control_order["response_text"] = anchor["original_response_text"]
            control = await branch(control_order, "original_control", True)
            assert control["first_info"] == anchor["original_first_info"]
            assert control["Y"] is not None and math.isclose(
                control["Y"], anchor["old_diagonal_Y"], abs_tol=1e-12, rel_tol=0
            )
            counts["original_control_exact"] += 1
            counts["hold_reference_exact"] += 1
            counts["anchors_complete"] += 1
            assert counts["first_steps"] <= 16128 and counts["terminal_branches"] <= 1280
            live(
                out,
                status="RUNNING",
                phase="ALL_MENU_FIRST_STEPS_AND_SCRIPTED_PROBES",
                elapsed_seconds=time.monotonic() - started,
                environment_requests=replay.requests,
                **counts,
            )
            if counts["anchors_complete"] % 8 == 0:
                print(
                    "anchors",
                    counts["anchors_complete"],
                    "/256",
                    "first_steps",
                    counts["first_steps"],
                    "env_requests",
                    replay.requests,
                    flush=True,
                )
    assert (
        counts["anchors_complete"]
        == counts["repeat_buy_exact"]
        == counts["original_control_exact"]
        == 256
    )
    assert counts["invalid_first_steps"] == 0, "presumed-legal menu failed; evidence preserved"
    assert all(sha(out / name) == value for name, value in snapshot_hashes.items())
    verify_inventory(contract["source_sha256"])
    no_model_backends()
    write_new(
        out / "collection_completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc),
            "elapsed_seconds": time.monotonic() - started,
            "counts": dict(counts),
            "environment_requests": replay.requests,
            "environments_constructed": replay.environments,
            "output_sha256": {
                name: sha(out / name)
                for name in (
                    "anchors.jsonl",
                    "prompt_packets.jsonl",
                    "snapshot_completion.json",
                    "branches.jsonl",
                )
            },
            "model_requests": 0,
            "optimizer_updates": 0,
            "GPU_used": False,
            "formal_valid_oos_access": False,
            "source_and_snapshots_unchanged": True,
            "all_256_historical_prompts_token_exact": True,
            "probe_slot_references": 1024,
            "probe_slots_are_scripted_not_model_samples": True,
        },
    )
    live(out, status="COLLECTION_COMPLETE", phase="AWAITING_READOUT", **counts)
    print("COLLECTION_COMPLETE", dict(counts), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    else:
        try:
            asyncio.run(collect(args.output))
        except Exception as error:
            if args.output.is_dir() and not (args.output / "failure.json").exists():
                write_new(
                    args.output / "failure.json",
                    {
                        "status": "FAILED",
                        "error": repr(error),
                        "failed_at_UTC": datetime.now(timezone.utc),
                        "automatic_retry": False,
                    },
                )
                live(
                    args.output,
                    status="FAILED",
                    phase="STOPPED_NO_AUTOMATIC_RETRY",
                    error=repr(error),
                )
            raise


if __name__ == "__main__":
    main()
