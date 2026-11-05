#!/usr/bin/env python3
"""Engineering checks for sealed same-prefix CPU request-menu branches.

Reads completed evidence only. No environment, model, tokenizer or GPU is loaded.
Outputs are exclusively created; fixed-hold Y is used only for identity checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.screening_prompt import apply_screening_action_protocol  # noqa: E402

PROFILE = "m2_same_prefix_menu_engineering_readout_v1"
GRID = (0.0, 0.1, 0.2)
MARGIN = 1e-6
DAYS = (0, 5, 10, 15)
OUTPUTS = ("readout.json", "menu_coverage.csv", "branch_summary.csv", "checks.json")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def object_sha(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: object, name: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"numeric {name}")
    result = float(value)
    require(math.isfinite(result), f"finite {name}")
    return result


def integer(value: object, name: str) -> int:
    require(type(value) is int and value >= 0, f"nonnegative integer {name}")
    return value


def equal_number(a: object, b: object, name: str) -> None:
    left, right = number(a, name), number(b, name)
    require(math.isclose(left, right, abs_tol=1e-8, rel_tol=1e-11), f"{name}: {left} != {right}")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open() as handle:
        for i, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank line {path}:{i}")
            value = json.loads(line)
            require(isinstance(value, dict), f"JSON object {path}:{i}")
            result.append(value)
    return result


def checked_file(base: Path, name: str, digest: str) -> None:
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"relative evidence name {name}")
    require(sha(base / path) == digest, f"evidence SHA changed: {name}")


def action(name: str, symbol: str | None = None, target: float | None = None) -> dict:
    arguments = {} if symbol is None else {"symbol": symbol, "target_weight": target}
    key = json.dumps({"name": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return {
        "action_key": key,
        "family": name,
        "symbol": symbol,
        "target_weight": target,
        "response_text": f"<tool_call>{key}</tool_call>",
    }


def expected_menu(visible: dict) -> list[dict]:
    market, positions = visible.get("market"), visible.get("positions")
    require(isinstance(market, dict) and len(market) == 20, "exactly 20 visible market symbols")
    require(isinstance(positions, dict), "visible positions mapping")
    result = [action("hold")]
    for symbol in sorted(market):
        require(isinstance(symbol, str) and symbol == symbol.strip().upper(), "normalized symbol")
        position = positions.get(symbol, {})
        require(isinstance(position, dict), "visible position object")
        weight = number(position.get("weight", 0.0), "visible weight")
        require(-MARGIN <= weight <= 1 + MARGIN, "visible unlevered weight")
        for target in GRID:
            if target > weight + MARGIN:
                result.append(action("buy", symbol, target))
            elif target < weight - MARGIN:
                result.append(action("sell", symbol, target))
    return result


def check_menu(anchor: dict) -> list[str]:
    expected = expected_menu(anchor["visible_state"])
    observed = anchor["menu"]
    require(
        isinstance(observed, list) and len(observed) == len(expected), "complete visible-only menu"
    )
    for actual, wanted in zip(observed, expected, strict=True):
        for key in ("action_key", "family", "symbol", "target_weight", "response_text"):
            require(actual.get(key) == wanted[key], f"visible-only menu field {key}")
    require(object_sha(anchor["visible_state"]) == anchor["menu_input_sha256"], "menu input SHA")
    require(object_sha(observed) == anchor["menu_sha256"], "menu SHA")
    buys = sorted(item["action_key"] for item in observed if item["family"] == "buy")
    sells = sorted(item["action_key"] for item in observed if item["family"] == "sell")
    require(bool(buys), "scripted probe requires a visible-legal buy")
    probes = [action("hold")["action_key"], buys[0], buys[0], sells[-1] if sells else buys[-1]]
    require(anchor["probe_action_keys"] == probes, "frozen four-slot scripted probe")
    return probes


def check_packet(anchor: dict, packet: dict) -> None:
    require(packet["anchor_id"] == anchor["anchor_id"], "prompt anchor ID")
    messages, ids, tools = packet["messages"], packet["prompt_token_ids"], packet["tool_schemas"]
    require(isinstance(messages, list) and bool(messages), "complete observation messages")
    require(isinstance(ids, list) and bool(ids), "complete native prompt token IDs")
    require(all(type(i) is int and i >= 0 for i in ids), "valid native prompt tokens")
    require(isinstance(tools, list) and bool(tools), "complete tool schemas")
    require(
        object_sha(messages) == anchor["observation_messages_sha256"], "observation message SHA"
    )
    require(object_sha(tools) == anchor["tool_schemas_sha256"], "tool schema SHA")
    require(
        packet["protocol_messages"]
        == apply_screening_action_protocol(messages, packet["action_protocol"]),
        "exact production action protocol messages",
    )
    require(packet["action_protocol"] == "single_tool_portfolio_v1", "fixed action protocol")
    require(
        packet["chat_template_kwargs"] == {"add_generation_prompt": True, "enable_thinking": False},
        "fixed chat kwargs",
    )
    require(packet["future_sampling_slots"] == 4, "four future sampling slots")
    require(packet["model_requests"] == 0, "no packet model requests")
    require(packet["menu_injected_into_prompt"] is False, "menu did not alter the old prompt")
    require(isinstance(packet["rendered_prompt_text"], str), "rendered prompt text")
    digest = hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest()
    require(digest == anchor["historical_prompt_sha256"], "native prompt token SHA")
    require(packet["historical_prompt_sha256"] == digest, "packet native prompt SHA")
    require(packet["observation_messages_sha256"] == object_sha(messages), "packet observation SHA")
    require(
        len(ids) == packet["prompt_token_count"] == anchor["prompt_token_count"], "prompt length"
    )
    require(len(ids) <= 8192, "unchanged native prompt cap")
    text = messages[-1]["content"]
    require(isinstance(text, str), "last observation text")
    if text.startswith("<tool_response>"):
        text = text.removeprefix("<tool_response>").removesuffix("</tool_response>").strip()
    require(json.loads(text) == anchor["visible_state"], "menu from actual returned observation")


def source_key(row: dict) -> tuple:
    return tuple(
        row[k]
        for k in ("seed", "episode_id", "policy_step", "decision_session_index", "sibling_id")
    )


def full_text(value: dict) -> str:
    text = value["text"]
    require(isinstance(text, str) and value["truncated"] is False, "untruncated historical text")
    require(len(text) == value["char_count"], "historical text count")
    require(hashlib.sha256(text.encode()).hexdigest() == value["sha256"], "historical text SHA")
    return text


def check_references(anchors: list[dict], packets: dict[str, dict], old: Path) -> None:
    manifest = read_json(old / "source_manifest.json")
    previous = {source_key(row): row for row in read_jsonl(old / "state_results.jsonl")}
    require(
        set(previous) == {source_key(a) for a in anchors}, "all original source states retained"
    )
    locators = manifest["records"]
    wanted: dict[str, dict[int, int]] = defaultdict(dict)
    for index, locator in enumerate(locators):
        wanted[locator["path"]][locator["line"]] = index
    historical = {}
    for name, lines in wanted.items():
        with Path(name).open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number not in lines:
                    continue
                index = lines[line_number]
                require(
                    hashlib.sha256(line).hexdigest() == locators[index]["line_sha256"],
                    "raw line SHA",
                )
                historical[index] = json.loads(line)
    require(len(historical) == 64, "all original trajectory records")
    for anchor in anchors:
        index = integer(anchor["record_index"], "record_index")
        require(index < len(locators), "original record index")
        locator, record = locators[index], historical[index]
        for name in ("seed", "episode_id", "policy_step", "sibling_id"):
            require(locator[name] == anchor[name], f"historical {name} binding")
        slot = next(
            s
            for s in locator["slots"]
            if s["decision_session_index"] == anchor["decision_session_index"]
        )
        stop = integer(anchor["prefix_requests"], "prefix_requests")
        require(stop == slot["interaction_index"], "complete original prefix count")
        require(
            anchor["prefix_audit_sha256"] == object_sha(record["interactions"][:stop]),
            "complete original prefix audit SHA",
        )
        request, interaction = record["requests"][stop], record["interactions"][stop]
        require(request["prompt_token_ids_truncated"] is False, "untruncated native prompt")
        require(
            request["prompt_token_ids"] == packets[anchor["anchor_id"]]["prompt_token_ids"],
            "prompt IDs exactly match historical request",
        )
        require(request["prompt_sha256"] == anchor["historical_prompt_sha256"], "source prompt SHA")
        require(anchor["original_action_key"] == slot["canonical_action_json"], "original order")
        require(
            anchor["original_response_text"] == full_text(interaction["response"]),
            "original response",
        )
        require(anchor["original_first_info"] == interaction["info"], "original first info")
        if stop:
            expected_messages = json.loads(
                full_text(record["interactions"][stop - 1]["observation"])
            )
            require(
                packets[anchor["anchor_id"]]["messages"] == expected_messages,
                "full returned historical observation",
            )
        prior = previous[source_key(anchor)]
        require(anchor["source_group_id"] == prior["group_id"], "original source group")
        equal_number(anchor["old_hold_Y"], prior["hold_Y"], "previous hold Y")
        diagonal = next(c for c in prior["cells"] if c["slot"] == anchor["sibling_id"])
        equal_number(anchor["old_diagonal_Y"], diagonal["Y"], "previous own-state fixed-hold Q")
        require(
            anchor["financial_state_sha256"] == prior["full_state_sha256"],
            "previous financial state",
        )


def check_accounting(anchor: dict, branch: dict) -> None:
    info = branch["first_info"]
    before, after = anchor["financial_state"], branch["first_state_after"]
    ledger_before, ledger_after = before["ledger"], after["ledger"]
    require(info.get("invalid_action") is False, "declared-legal menu actually advances")
    require(after["date"] != before["date"], "market date advances")
    require(
        ledger_after["session_index"] == ledger_before["session_index"] + 1, "one session advances"
    )
    for name in ("equity_curve", "net_returns", "turnovers"):
        require(len(after[name]) == len(before[name]) + 1, f"append exactly one {name}")
        require(after[name][:-1] == before[name], f"preserve full prefix {name}")
    require(after["initial_cash"] == before["initial_cash"], "preserve original initial cash")
    require(after["high_water_mark"] >= before["high_water_mark"], "preserve high-water mark")
    require(
        after["maximum_position_weight"] >= before["maximum_position_weight"],
        "preserve risk maximum",
    )
    fills = info["fills"]
    require(ledger_after["fills"] == [*ledger_before["fills"], *fills], "append exact new fills")
    cost = math.fsum(
        number(f["commission"], "commission") + number(f["slippage_cost"], "slippage")
        for f in fills
    )
    equal_number(branch["first_cost"], cost, "first transaction cost")
    require(branch["zero_fill"] is (len(fills) == 0), "zero-fill classification")
    require(branch["skipped_count"] == len(info.get("skipped_orders", [])), "skip classification")
    cash = number(ledger_before["cash"], "before cash")
    positions = {s: number(v, "before shares") for s, v in ledger_before["positions"].items()}
    for fill in fills:
        require(fill["side"] in ("buy", "sell"), "fill side")
        sign = 1 if fill["side"] == "buy" else -1
        shares = number(fill["shares"], "fill shares")
        price = number(fill["execution_price"], "execution price")
        commission = number(fill["commission"], "commission")
        require(shares > 0 and price > 0 and commission >= 0, "nonnegative fill accounting")
        equal_number(fill["notional"], shares * price, "fill notional")
        equal_number(
            fill["slippage_cost"], shares * abs(price - fill["raw_open_price"]), "slippage cost"
        )
        cash -= sign * shares * price + commission
        positions[fill["symbol"]] = positions.get(fill["symbol"], 0.0) + sign * shares
    equal_number(ledger_after["cash"], cash, "cash reconciliation")
    require(number(ledger_after["cash"], "after cash") >= -1e-8, "no leverage")
    require(set(ledger_after["positions"]) == set(positions), "position keys preserved")
    for symbol, shares in positions.items():
        equal_number(ledger_after["positions"][symbol], shares, "position reconciliation")
        require(shares >= -1e-8, "no short positions")
    equal_number(after["equity_curve"][-1], info["equity"], "latest equity")
    equal_number(after["net_returns"][-1], info["portfolio_return_net"], "latest net return")
    equal_number(after["turnovers"][-1], info["turnover"], "latest turnover")


def validate(
    anchors: list[dict], packets: list[dict], branches: list[dict], *, fixed_design=True
) -> dict:
    require(len({a["anchor_id"] for a in anchors}) == len(anchors), "unique anchor IDs")
    require(len({source_key(a) for a in anchors}) == len(anchors), "unique original source states")
    packet_map = {p["anchor_id"]: p for p in packets}
    require(len(packet_map) == len(packets) == len(anchors), "one prompt packet per anchor")
    require(set(packet_map) == {a["anchor_id"] for a in anchors}, "same prompt/anchor population")
    by_anchor: dict[str, list[dict]] = defaultdict(list)
    for branch in branches:
        require(branch["anchor_id"] in packet_map, "branch anchor exists")
        by_anchor[branch["anchor_id"]].append(branch)
    require(set(by_anchor) == set(packet_map), "no missing anchor branches")
    population = Counter()
    coverage = []
    check_counts = Counter()
    for anchor in anchors:
        for name in (
            "seed",
            "policy_step",
            "decision_session_index",
            "sibling_id",
            "prefix_requests",
        ):
            integer(anchor[name], name)
        require(anchor["seed"] in (42, 43), "fixed source seeds")
        require(
            anchor["decision_session_index"] in DAYS and anchor["sibling_id"] in range(4),
            "fixed anchors",
        )
        population[(anchor["seed"], anchor["episode_id"], anchor["decision_session_index"])] += 1
        require(
            object_sha(anchor["financial_state"]) == anchor["financial_state_sha256"],
            "financial state SHA",
        )
        probes = check_menu(anchor)
        check_packet(anchor, packet_map[anchor["anchor_id"]])
        menu_by_key = {item["action_key"]: item for item in anchor["menu"]}
        require(len(menu_by_key) == len(anchor["menu"]), "no duplicate menu entries")
        actual_menu, repeats, controls = {}, [], []
        for branch in by_anchor[anchor["anchor_id"]]:
            role = branch["branch_role"]
            require(role in ("menu", "repeat_buy", "original_control"), "known branch role")
            require(branch["status"] == "LEGAL_ADVANCING", "unexpected request-menu rejection")
            require(
                branch["financial_state_sha256"] == anchor["financial_state_sha256"],
                "same full financial prefix",
            )
            require(
                branch["observation_messages_sha256"] == anchor["observation_messages_sha256"],
                "same visible prefix",
            )
            require(
                branch["historical_prompt_sha256"] == anchor["historical_prompt_sha256"],
                "same exact native prompt identity",
            )
            checks = branch["checks"]
            require(isinstance(checks, dict) and bool(checks), "nonempty branch engineering checks")
            for name in ("before_identity", "history_append", "cash_position_fees"):
                require(checks.get(name) is True, f"mandatory branch check {name}")
            for name, value in checks.items():
                require(value is True, f"branch engineering check {name}")
                check_counts[name] += 1
            check_accounting(anchor, branch)
            if role == "menu":
                key = branch["action_key"]
                require(
                    key in menu_by_key and key not in actual_menu, "one execution per menu request"
                )
                require(branch["family"] == menu_by_key[key]["family"], "menu action family")
                actual_menu[key] = branch
            elif role == "repeat_buy":
                repeats.append(branch)
            else:
                controls.append(branch)
        require(set(actual_menu) == set(menu_by_key), "all finite-menu candidates executed")
        require(len(repeats) == len(controls) == 1, "one fresh repeat and original control")
        selected = set(probes)
        for key, branch in actual_menu.items():
            complete = key in selected
            require((branch["Y"] is not None) == complete, "suffix only for selected menu keys")
            require(
                (branch["terminal_metrics"] is not None) == complete, "selected terminal evidence"
            )
            require(
                branch["suffix_hold_requests"]
                == (19 - anchor["decision_session_index"] if complete else 0),
                "original-end suffix length",
            )
            if complete:
                number(branch["Y"], "terminal Y")
                equal_number(
                    branch["terminal_metrics"]["terminal_score"], branch["Y"], "terminal score"
                )
                require(branch["terminal_metrics"]["session_count"] == 21, "full original window")
        repeat, control = repeats[0], controls[0]
        require(repeat["action_key"] == probes[1] == probes[2], "fresh duplicate buy")
        reference = actual_menu[probes[1]]
        for key in (
            "first_info",
            "first_state_after",
            "Y",
            "terminal_metrics",
            "suffix_hold_requests",
        ):
            require(repeat[key] == reference[key], f"fresh repeat exactly reproduces {key}")
        require(control["action_key"] == anchor["original_action_key"], "original control order")
        require(
            control["first_info"] == anchor["original_first_info"], "original control first info"
        )
        require(
            control["suffix_hold_requests"] == 19 - anchor["decision_session_index"],
            "original control horizon",
        )
        require(isinstance(control["terminal_metrics"], dict), "original control terminal metrics")
        require(control["terminal_metrics"]["session_count"] == 21, "original control full window")
        equal_number(control["Y"], anchor["old_diagonal_Y"], "old fixed-hold diagonal Q reproduced")
        equal_number(
            control["Y"], control["terminal_metrics"]["terminal_score"], "control terminal Y"
        )
        hold = actual_menu[probes[0]]
        equal_number(hold["Y"], anchor["old_hold_Y"], "old hold reference reproduced")
        coverage.append(
            {
                **{
                    k: anchor[k]
                    for k in (
                        "anchor_id",
                        "seed",
                        "episode_id",
                        "policy_step",
                        "decision_session_index",
                        "sibling_id",
                    )
                },
                "menu_candidates": len(actual_menu),
                "menu_legal": len(actual_menu),
                "menu_invalid": 0,
                "menu_buy": sum(b["family"] == "buy" for b in actual_menu.values()),
                "menu_sell": sum(b["family"] == "sell" for b in actual_menu.values()),
                "menu_hold": 1,
                "menu_zero_fill": sum(b["zero_fill"] for b in actual_menu.values()),
                "menu_skipped": sum(b["skipped_count"] > 0 for b in actual_menu.values()),
                "menu_positive_cost": sum(b["first_cost"] > 0 for b in actual_menu.values()),
                "menu_first_cost_sum": math.fsum(b["first_cost"] for b in actual_menu.values()),
                "original_order_in_menu": anchor["original_action_key"] in menu_by_key,
                "scripted_probe_slots": 4,
                "selected_unique_menu_actions": len(selected),
                "fresh_repeat_executions": 1,
                "original_control_executions": 1,
                "actual_executions": len(actual_menu) + 2,
                "actual_complete_hold_branches": len(selected) + 2,
                "scripted_probe_reference_slots_are_independent_samples": False,
            }
        )
    if fixed_design:
        require(len(anchors) == 256 and len(population) == 64, "256 anchors / 64 source groups")
        require(set(population.values()) == {4}, "four siblings in every source group")
        require(set(a["seed"] for a in anchors) == {42, 43}, "two seeds")
        for seed in (42, 43):
            windows = {a["episode_id"] for a in anchors if a["seed"] == seed}
            require(len(windows) == 8, "eight fixed Train windows per seed")
            for window in windows:
                selected = [a for a in anchors if a["seed"] == seed and a["episode_id"] == window]
                require({a["decision_session_index"] for a in selected} == set(DAYS), "fixed dates")
                require({a["sibling_id"] for a in selected} == set(range(4)), "fixed siblings")
    return {"coverage": coverage, "branch_check_counts": dict(sorted(check_counts.items()))}


def branch_summaries(anchors: list[dict], branches: list[dict]) -> list[dict]:
    lookup = {a["anchor_id"]: a for a in anchors}
    bins = defaultdict(list)
    for b in branches:
        a = lookup[b["anchor_id"]]
        scopes = (
            ("all", "all"),
            ("seed", str(a["seed"])),
            ("day", str(a["decision_session_index"])),
            ("episode", f"{a['seed']}:{a['episode_id']}"),
        )
        for scope, value in scopes:
            bins[(scope, value, b["branch_role"], b["family"])].append(b)
    return [
        {
            "scope": scope,
            "scope_value": value,
            "branch_role": role,
            "family": family,
            "actual_executions": len(items),
            "legal_advancing": sum(b["status"] == "LEGAL_ADVANCING" for b in items),
            "zero_fill": sum(b["zero_fill"] for b in items),
            "with_skip": sum(b["skipped_count"] > 0 for b in items),
            "positive_first_cost": sum(b["first_cost"] > 0 for b in items),
            "first_cost_sum": math.fsum(b["first_cost"] for b in items),
            "full_hold_continuations": sum(b["Y"] is not None for b in items),
            "suffix_hold_requests": sum(b["suffix_hold_requests"] for b in items),
        }
        for (scope, value, role, family), items in sorted(bins.items())
    ]


def write_new(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), "nonempty engineering CSV")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def readout(run: Path) -> dict:
    run = run.resolve(strict=True)
    require(
        not any((run / name).exists() for name in OUTPUTS),
        "all engineering output paths must be new",
    )
    contract = read_json(run / "contract.json")
    completion = read_json(run / "collection_completion.json")
    snapshot = read_json(run / "snapshot_completion.json")
    require(completion["status"] == "COMPLETE", "completed collector only")
    require(
        set(snapshot["snapshot_sha256"]) == {"anchors.jsonl", "prompt_packets.jsonl"},
        "complete pre-branch snapshot",
    )
    for name, digest in snapshot["snapshot_sha256"].items():
        checked_file(run, name, digest)
    require("branches.jsonl" in completion["output_sha256"], "completed branch evidence bound")
    for name, digest in completion["output_sha256"].items():
        checked_file(run, name, digest)
    for name, digest in contract["source_sha256"].items():
        require(sha(Path(name)) == digest, f"sealed source/input changed: {name}")
    anchors = read_jsonl(run / "anchors.jsonl")
    packets = read_jsonl(run / "prompt_packets.jsonl")
    branches = read_jsonl(run / "branches.jsonl")
    checked = validate(anchors, packets, branches)
    previous = Path(contract["previous_run"]).resolve(strict=True)
    for name in ("source_manifest.json", "state_results.jsonl"):
        path = previous / name
        require(str(path) in contract["source_sha256"], f"previous oracle {name} pinned")
    check_references(anchors, {p["anchor_id"]: p for p in packets}, previous)
    coverage = checked["coverage"]
    menus = sum(r["menu_candidates"] for r in coverage)
    require(
        snapshot["status"] == "ALL_MENUS_AND_PROMPTS_FROZEN_BEFORE_NEW_BRANCH_OUTCOMES",
        "pre-branch snapshot completion",
    )
    require(snapshot["new_intervention_requests_so_far"] == 0, "no outcomes before snapshot")
    require(snapshot["anchors"] == len(anchors), "snapshot anchor count")
    require(snapshot["menu_orders"] == menus, "snapshot finite-menu denominator")
    expected_counts = {
        "anchors_complete": len(anchors),
        "menu_first_steps": menus,
        "first_steps": len(branches),
        "legal_first_steps": len(branches),
        "before_identity_checks": len(branches),
        "terminal_branches": sum(b["Y"] is not None for b in branches),
        "repeat_buy_exact": len(anchors),
        "original_control_exact": len(anchors),
        "hold_reference_exact": len(anchors),
    }
    for name, count in expected_counts.items():
        require(completion["counts"].get(name) == count, f"collector actual count {name}")
    require(completion["counts"].get("invalid_first_steps", 0) == 0, "zero rejected menu requests")
    for name in ("model_requests", "optimizer_updates"):
        require(completion[name] == contract[name] == 0, f"zero {name}")
    for name in ("GPU_used", "formal_valid_oos_access"):
        require(completion[name] is contract[name] is False, f"no {name}")
    for name in (
        "source_and_snapshots_unchanged",
        "all_256_historical_prompts_token_exact",
        "probe_slots_are_scripted_not_model_samples",
    ):
        require(completion[name] is True, f"collector completion {name}")
    require(
        completion["probe_slot_references"] == len(anchors) * 4, "scripted probe reference count"
    )
    require(
        len(branches) == menus + 2 * len(anchors), "actual executions distinct from slot references"
    )
    require(menus <= 15616 and len(branches) <= 16128, "fixed first-step budget")
    require(expected_counts["terminal_branches"] <= 1280, "fixed terminal-branch budget")
    checks = {
        "status": "PASS",
        "anchors": len(anchors),
        "prompt_packets": len(packets),
        "exact_visible_only_menu_anchors": len(anchors),
        "historical_prompt_token_identity_anchors": len(anchors),
        "financial_and_observation_prefix_identity_executions": len(branches),
        "fresh_buy_repeat_identity_checks": len(anchors),
        "old_hold_reference_checks": len(anchors),
        "old_diagonal_and_original_first_info_checks": len(anchors),
        "branch_check_counts": checked["branch_check_counts"],
    }
    result = {
        "profile": PROFILE,
        "status": "PASS",
        "claim": "CPU engineering support only",
        "counts": {
            "source_anchor_instances": len(anchors),
            "source_groups": 64,
            "source_trajectories": 64,
            "finite_menu_candidates": menus,
            "finite_menu_actual_executions": menus,
            "fresh_repeat_buy_executions": len(anchors),
            "fresh_original_control_executions": len(anchors),
            "total_actual_branch_executions": len(branches),
            "scripted_probe_logical_slots": 4 * len(anchors),
            "selected_unique_menu_hold_continuations": sum(
                r["selected_unique_menu_actions"] for r in coverage
            ),
            "total_actual_complete_hold_branches": sum(
                r["actual_complete_hold_branches"] for r in coverage
            ),
            "menu_invalid": 0,
            "menu_zero_fill": sum(r["menu_zero_fill"] for r in coverage),
            "menu_with_skip": sum(r["menu_skipped"] for r in coverage),
            "original_orders_off_menu": sum(not r["original_order_in_menu"] for r in coverage),
            "model_requests": 0,
            "optimizer_updates": 0,
        },
        "rules": {
            "target_grid": GRID,
            "visible_weight_margin": MARGIN,
            "market_symbols": 20,
            "scripted_probe": "hold,first buy,first buy,last sell if present else last buy",
            "continuation": "explicit hold to original window endpoint; no future-based selection",
            "fixed_hold_Y_usage": "identity/reproduction only; no credit MSE or ordering score",
        },
        "limits": [
            "A scripted legal menu is not a policy sample or evidence of model action support.",
            "Repeated deterministic branches are identity checks, not independent observations.",
            "This does not demonstrate pooling denoising, critic improvement or financial gains.",
            "Legal direction at decision does not guarantee a next-open fill; "
            "skips and zero-fill remain.",
            "Fee totals across mutually exclusive counterfactual branches "
            "are engineering counts only.",
            "Only original Train windows are used; no formal Valid/Final OOS access.",
        ],
        "inputs": {
            name: sha(run / name)
            for name in (
                "contract.json",
                "collection_completion.json",
                "snapshot_completion.json",
                "anchors.jsonl",
                "prompt_packets.jsonl",
                "branches.jsonl",
            )
        },
        "readout_source_sha256": sha(Path(__file__).resolve()),
        "checks": checks,
    }
    # Recheck sealed evidence after calculations, before any output is created.
    for name, digest in result["inputs"].items():
        checked_file(run, name, digest)
    write_csv(run / "menu_coverage.csv", coverage)
    write_csv(run / "branch_summary.csv", branch_summaries(anchors, branches))
    write_new(run / "checks.json", checks)
    result["output_sha256"] = {name: sha(run / name) for name in OUTPUTS if name != "readout.json"}
    write_new(run / "readout.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = readout(args.run)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, allow_nan=False))


if __name__ == "__main__":
    main()
