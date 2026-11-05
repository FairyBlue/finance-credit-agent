#!/usr/bin/env python3
"""Train-only, full-prefix CPU interventions with frozen hold continuation."""

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
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/agent-r1"))

import pyarrow.parquet as pq  # noqa: E402
from agent_r1.env.base import Action  # noqa: E402

from src.agents.order_recovery_probe import full_text, ledger_state  # noqa: E402
from src.env.trading import TradingEnv  # noqa: E402

BASE = ROOT / "runs/m2-action-pooling-comparison-20261013-v6"
DEFAULT = ROOT / "runs/m2-state-order-oracle-20261017-v1"
MANIFEST = Path("/tmp/m2-state-order-source-manifest-20261017.json")
PLAN = ROOT / "docs/2026-10-17_M2_STATE_ORDER_ORACLE_PLAN.md"
READOUT = ROOT / "scripts/diagnostics/m2_state_order_readout.py"
EPISODES = (1, 65, 9, 73, 17, 149, 25, 157)
DAYS = (0, 5, 10, 15)
HOLD = '<tool_call>{"name":"hold","arguments":{}}</tool_call>'


def json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=json_default)


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        handle.write(canonical(value) + "\n")


def append(handle: Any, value: Any) -> None:
    handle.write(canonical(value) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def live(out: Path, **values: Any) -> None:
    payload = {"updated_at_UTC": datetime.now(timezone.utc), **values}
    temporary = out / "live_status.tmp"
    temporary.write_text(canonical(payload) + "\n")
    temporary.replace(out / "live_status.json")


def verify_inventory(inventory: dict[str, str]) -> None:
    for name, value in inventory.items():
        assert sha(Path(name)) == value, f"input/source changed: {name}"


def prepare(out: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    assert manifest["selection"]["seeds"] == [42, 43]
    assert manifest["selection"]["decision_session_indices"] == list(DAYS)
    assert manifest["selection"]["episode_ids"] == [f"formal-train-{i:04d}" for i in EPISODES]
    assert len(manifest["records"]) == 64 and len(manifest["groups"]) == 64
    inventory = {name: entry["sha256"] for name, entry in manifest["source_files"].items()}
    verify_inventory(inventory)
    # Production and replay code must still equal the completed Stage 2 snapshot.
    original = json.loads((BASE / "seed42/source_identity.json").read_text())
    for name, value in original.items():
        if name.startswith("src/"):
            assert sha(ROOT / name) == value, f"production source drifted: {name}"
            inventory[str(ROOT / name)] = value
    for file in (Path(__file__).resolve(), READOUT, PLAN):
        inventory[str(file)] = sha(file)
    for file in (ROOT / "third_party/agent-r1/agent_r1").rglob("*.py"):
        if any(part in ("env", "tool") for part in file.relative_to(ROOT).parts):
            inventory[str(file)] = sha(file)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "source_manifest.json").open("xb") as handle:
        handle.write(manifest_path.read_bytes())
    write_new(
        out / "contract.json",
        {
            "profile": "real_train_state_order_hold_oracle_v1",
            "status": "SEALED_BEFORE_REPLAY",
            "sealed_at_UTC": datetime.now(timezone.utc),
            "source_manifest_sha256": sha(out / "source_manifest.json"),
            "source_sha256": inventory,
            "groups": 64,
            "state_instances": 256,
            "transferred_order_slots": 1024,
            "additional_hold_reference_slots": 256,
            "source_arm": "M2",
            "days": DAYS,
            "seeds": [42, 43],
            "continuation": "all hold until original window terminal",
            "target": "original terminal financial Y, never one-day net-return proxy",
            "illegal_transfer": "record actual rejection; Y/q NA, no repair",
            "model_requests": 0,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
            "versions": {
                p: importlib.metadata.version(p)
                for p in (
                    "numpy",
                    "pyarrow",
                    "pydantic",
                    "exchange-calendars",
                )
            },
        },
    )
    print("SEALED", out / "contract.json", flush=True)


def records_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[int, int]] = defaultdict(dict)
    result: list[Any] = [None] * len(manifest["records"])
    for index, locator in enumerate(manifest["records"]):
        by_path[locator["path"]][locator["line"]] = index
    for name, wanted in by_path.items():
        with Path(name).open("rb") as handle:
            for number, line in enumerate(handle, 1):
                if number not in wanted:
                    continue
                index = wanted[number]
                expected = manifest["records"][index]
                assert hashlib.sha256(line).hexdigest() == expected["line_sha256"]
                row = json.loads(line)
                assert row["complete"] and row["status"] == "COMPLETE"
                assert row["identity"] == expected["identity"]
                assert row["episode_id"] == expected["episode_id"]
                assert row["base_seed"] == expected["seed"]
                assert row["terminal_score"] == expected["source_Y"]
                assert len(row["interactions"]) == len(row["steps"])
                result[index] = row
    assert all(row is not None for row in result)
    return result


def financial_state(env: TradingEnv, config_sha: str) -> dict[str, Any]:
    """Reward-state identity, including the path dependence of the financial Y."""
    return {
        "config_sha256": config_sha,
        "ledger": ledger_state(env),
        "date": env.current_date,
        "initial_cash": env.ledger.initial_cash,
        "equity_curve": list(env._equity_curve),
        "net_returns": list(env._net_returns),
        "turnovers": list(env._turnovers),
        "high_water_mark": env._high_water_mark,
        "maximum_position_weight": env._maximum_position_weight,
        "done": env._done,
    }


def context_identity(env: TradingEnv) -> dict[str, Any]:
    """Conservative cache context; distinct from economic state-peer coverage."""
    return {
        "visible_state": json.loads(env._state_text()),
        "tool_calls": env._tool_calls,
        "invalid_calls": env._invalid_calls,
        "abstentions": env._abstentions,
    }


async def replay_exact(env: TradingEnv, record: dict[str, Any], stop: int) -> None:
    for index in range(stop):
        interaction, credit = record["interactions"][index], record["steps"][index]
        await env.step(Action(text=full_text(interaction["response"], "response")))
        assert env.interaction_trace[-1] == interaction, (record["episode_id"], index)
        assert json.loads(canonical(asdict(env.credit_step_trace[-1]))) == credit


async def collect(out: Path) -> None:
    started = time.monotonic()
    contract = json.loads((out / "contract.json").read_text())
    assert sha(out / "source_manifest.json") == contract["source_manifest_sha256"]
    verify_inventory(contract["source_sha256"])
    assert not any(name in sys.modules for name in ("torch", "vllm", "transformers"))
    write_new(
        out / "start_receipt.json",
        {
            "started_at_UTC": datetime.now(timezone.utc),
            "contract_sha256": sha(out / "contract.json"),
            "automatic_retry": False,
        },
    )
    manifest = json.loads((out / "source_manifest.json").read_text())
    records = records_from_manifest(manifest)
    datasets: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in (42, 43):
        path = BASE / f"seed{seed}/prepared/train.parquet"
        for row in pq.read_table(path).to_pylist():
            assert row["extra_info"]["source_split"] == "train"
            kwargs = json.loads(row["env_kwargs"])
            assert kwargs["screening"] and kwargs["data_start"] == "2023-01-03"
            assert kwargs["data_end"] == "2024-10-31" and "adapter" not in kwargs
            assert kwargs["input_file_count"] == 9220 and len(kwargs["episode_dates"]) == 21
            kwargs.update(capture_interactions=True, reward_mode="finance_trajectory")
            datasets[seed, row["extra_info"]["episode_id"]] = {
                "kwargs": kwargs,
                "prompt": row["prompt"],
                "sha": object_sha(kwargs),
            }

    def reset(locator: dict[str, Any]) -> TradingEnv:
        data = datasets[locator["seed"], locator["episode_id"]]
        env = TradingEnv(**data["kwargs"])
        env.reset(raw_prompt=data["prompt"])
        return env

    live(
        out,
        status="RUNNING",
        phase="FULL_HISTORICAL_REPLAY_AND_STORAGE_PREFLIGHT",
        historical_replayed=0,
        states_written=0,
    )
    with (out / "historical_replay.jsonl").open("x") as historical:
        for index, (locator, record) in enumerate(zip(manifest["records"], records, strict=True)):
            begin = time.monotonic()
            env = reset(locator)
            constructor_seconds = time.monotonic() - begin
            await replay_exact(env, record, len(record["interactions"]))
            assert env.episode_complete and env.screening_terminal_metrics() == record["metrics"]
            append(
                historical,
                {
                    "record_index": index,
                    "source_locator": locator,
                    "status": "EXACT_COMPLETE",
                    "interactions": len(record["interactions"]),
                    "constructor_seconds": constructor_seconds,
                    "terminal_metrics": env.screening_terminal_metrics(),
                },
            )
            live(
                out,
                status="RUNNING",
                phase="FULL_HISTORICAL_REPLAY",
                historical_replayed=index + 1,
                states_written=0,
            )
            if index % 8 == 0:
                print("historical replay", index + 1, "/64", flush=True)

    cache: dict[str, dict[str, Any]] = {}
    counts = {
        "states": 0,
        "cells": 0,
        "legal_cells": 0,
        "invalid_cells": 0,
        "unique_branches": 0,
        "environment_requests_after_historical": 0,
    }
    state_path = out / "state_results.jsonl"
    branch_path = out / "branch_results.jsonl"
    with state_path.open("x") as state_stream, branch_path.open("x") as branch_stream:
        for group_index, group in enumerate(manifest["groups"]):
            day = group["decision_session_index"]
            indices = group["record_indices"]
            assert [manifest["records"][i]["sibling_id"] for i in indices] == list(range(4))
            slots = [
                next(
                    s for s in manifest["records"][i]["slots"] if s["decision_session_index"] == day
                )
                for i in indices
            ]
            group_id = f"seed{group['seed']}/{group['episode_id']}/p{group['policy_step']}/d{day}"
            for record_index in indices:
                locator, record = manifest["records"][record_index], records[record_index]
                source_sibling = locator["sibling_id"]
                selected = slots[source_sibling]
                stop = selected["interaction_index"]
                env = reset(locator)
                await replay_exact(env, record, stop)
                counts["environment_requests_after_historical"] += stop
                assert env.current_session_index == day and not env.episode_complete
                config_sha = datasets[locator["seed"], locator["episode_id"]]["sha"]
                before = financial_state(env, config_sha)
                context = context_identity(env)
                state_sha, context_sha = object_sha(before), object_sha(context)

                async def outcome(
                    action_key: str,
                    text: str,
                    captured: tuple[Any, ...] = (
                        state_sha,
                        context_sha,
                        locator,
                        record,
                        stop,
                        config_sha,
                        record_index,
                        before,
                        day,
                    ),
                ) -> dict[str, Any]:
                    (
                        state_sha,
                        context_sha,
                        locator,
                        record,
                        stop,
                        config_sha,
                        record_index,
                        before,
                        day,
                    ) = captured
                    branch_id = object_sha([state_sha, context_sha, action_key])
                    if branch_id in cache:
                        return cache[branch_id]
                    candidate = reset(locator)
                    await replay_exact(candidate, record, stop)
                    counts["environment_requests_after_historical"] += stop
                    assert object_sha(financial_state(candidate, config_sha)) == state_sha
                    assert object_sha(context_identity(candidate)) == context_sha
                    session_before = candidate.current_session_index
                    _, _, _, info = await candidate.step(Action(text=text))
                    counts["environment_requests_after_historical"] += 1
                    normalized_info = json.loads(canonical(info))
                    receipt: dict[str, Any] = {
                        "branch_id": branch_id,
                        "action_key": action_key,
                        "full_state_sha256": state_sha,
                        "cache_context_sha256": context_sha,
                        "source_record_index": record_index,
                        "prefix_requests": stop,
                        "before": before,
                        "first_info": normalized_info,
                        "decision_session_index": day,
                    }
                    if info["invalid_action"]:
                        assert candidate.current_session_index == session_before
                        receipt.update(
                            status="INADMISSIBLE_AT_DECISION",
                            Y=None,
                            suffix_hold_requests=0,
                            terminal_metrics=None,
                        )
                    else:
                        assert candidate.current_session_index == session_before + 1
                        holds = 0
                        while not candidate.episode_complete:
                            _, _, _, suffix_info = await candidate.step(Action(text=HOLD))
                            counts["environment_requests_after_historical"] += 1
                            assert not suffix_info["invalid_action"] and not suffix_info["fills"]
                            holds += 1
                            assert holds <= 20
                        metrics = candidate.screening_terminal_metrics()
                        assert metrics and math.isfinite(metrics["terminal_score"])
                        receipt.update(
                            status="LEGAL_ADVANCING",
                            Y=metrics["terminal_score"],
                            suffix_hold_requests=holds,
                            terminal_metrics=metrics,
                        )
                    cache[branch_id] = receipt
                    append(branch_stream, receipt)
                    counts["unique_branches"] += 1
                    return receipt

                hold_key = json.dumps(
                    {"name": "hold", "arguments": {}}, sort_keys=True, separators=(",", ":")
                )
                hold = await outcome(hold_key, HOLD)
                assert hold["status"] == "LEGAL_ADVANCING"
                cells = []
                for slot_index, slot in enumerate(slots):
                    action_key = slot["canonical_action_json"]
                    result = await outcome(action_key, slot["response_text"])
                    legal = result["status"] == "LEGAL_ADVANCING"
                    info = result["first_info"]
                    q = result["Y"] - hold["Y"] if legal else None
                    if action_key == hold_key:
                        assert legal and abs(q) < 1e-12
                    if slot_index == source_sibling:
                        assert legal, "own historical successful order became illegal"
                        source_info = record["interactions"][stop]["info"]
                        assert info == json.loads(canonical(source_info))
                    counts["cells"] += 1
                    counts["legal_cells" if legal else "invalid_cells"] += 1
                    cells.append(
                        {
                            "slot": slot_index,
                            "action_key": action_key,
                            "family": slot["canonical_action"]["name"],
                            "status": result["status"],
                            "Y": result["Y"],
                            "q": q,
                            "branch_id": result["branch_id"],
                            "zero_fill": legal and not info["fills"],
                            "skipped_count": len(info.get("skipped_orders", [])),
                            "first_cost": sum(
                                f["commission"] + f["slippage_cost"] for f in info.get("fills", [])
                            ),
                            "error": info.get("error"),
                        }
                    )
                append(
                    state_stream,
                    {
                        "group_id": group_id,
                        "seed": group["seed"],
                        "episode_id": group["episode_id"],
                        "policy_step": group["policy_step"],
                        "decision_session_index": day,
                        "sibling_id": source_sibling,
                        "record_index": record_index,
                        "trajectory_uid": locator["trajectory_uid"],
                        "source_Y": locator["source_Y"],
                        "source_locator": {
                            k: locator[k] for k in ("path", "line", "line_sha256", "identity")
                        },
                        "prefix_requests": stop,
                        "full_state_sha256": state_sha,
                        "cache_context_sha256": context_sha,
                        "hold_Y": hold["Y"],
                        "hold_branch_id": hold["branch_id"],
                        "cells": cells,
                    },
                )
                counts["states"] += 1
                live(
                    out,
                    status="RUNNING",
                    phase="COUNTERFACTUAL_HOLD_MATRIX",
                    historical_replayed=64,
                    source_groups_written=group_index,
                    elapsed_seconds=time.monotonic() - started,
                    **counts,
                )
            print(
                "counterfactual group",
                group_index + 1,
                "/64",
                "states",
                counts["states"],
                "branches",
                counts["unique_branches"],
                flush=True,
            )
    assert counts["states"] == 256 and counts["cells"] == 1024
    verify_inventory(contract["source_sha256"])
    assert not any(name in sys.modules for name in ("torch", "vllm", "transformers"))
    write_new(
        out / "collection_completion.json",
        {
            "status": "COMPLETE",
            "finished_at_UTC": datetime.now(timezone.utc),
            "elapsed_seconds": time.monotonic() - started,
            **counts,
            "historical_complete_exact_replays": 64,
            "own_diagonal_cells_complete": 256,
            "source_and_runtime_sha_unchanged": True,
            "output_sha256": {
                p.name: sha(p) for p in (state_path, branch_path, out / "historical_replay.jsonl")
            },
            "model_requests": 0,
            "optimizer_updates": 0,
            "formal_valid_oos_access": False,
        },
    )
    live(out, status="COLLECTION_COMPLETE", phase="AWAITING_READOUT", **counts)
    print("COLLECTION_COMPLETE", counts, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output, args.manifest)
    else:
        try:
            asyncio.run(collect(args.output))
        except Exception as error:
            path = args.output / "failure.json"
            if args.output.is_dir() and not path.exists():
                write_new(
                    path,
                    {
                        "status": "FAILED",
                        "error": repr(error),
                        "failed_at_UTC": datetime.now(timezone.utc),
                        "automatic_retry": False,
                    },
                )
            raise


if __name__ == "__main__":
    main()
