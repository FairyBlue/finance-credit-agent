#!/usr/bin/env python3
"""Independent, CPU evidence-only readout of the 128 first-request pilot.

No environment/model/tokenizer is imported. Nonadvancing or unsupported slots
are retained, never replaced or used to recenter a smaller credit group.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PROFILE = "m2_same_prefix_first_request_independent_readout_v1"
OUTPUTS = (
    "readout.json",
    "slot_metrics.csv",
    "group_metrics.csv",
    "day_summary.csv",
    "checks.json",
)
FAMILIES = {"buy", "sell", "hold", "abstain"}
INFORMATION = {"retrieve", "compute"}
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
EOS = {151645, 151643}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def object_sha(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open() as handle:
        for i, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank evidence line {path}:{i}")
            value = json.loads(line)
            require(isinstance(value, dict), f"JSON object {path}:{i}")
            result.append(value)
    return result


def number(value: object, name: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"numeric {name}")
    require(math.isfinite(float(value)), f"finite {name}")
    return float(value)


def integer(value: object, name: str) -> int:
    require(type(value) is int and value >= 0, f"nonnegative integer {name}")
    return value


def close(a: object, b: object, name: str) -> None:
    require(math.isclose(number(a, name), number(b, name), abs_tol=1e-12, rel_tol=0), name)


def finite_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        name = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return {"__nonfinite_float__": name}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    return value


def parse_calls(text: str) -> list[dict]:
    """Mirror the pinned Hermes parser, including its permissive bad-tag handling."""
    calls = []
    for match in TOOL_CALL.findall(text):
        try:
            parsed = json.loads(match)
            name, arguments = parsed["name"], parsed["arguments"]
            if not isinstance(arguments, dict):
                arguments = json.loads(arguments) if isinstance(arguments, str) else {}
            calls.append({"name": name, "arguments": arguments})
        except (KeyError, TypeError, ValueError):
            continue
    return calls


def api_key(calls: list[dict]) -> str | None:
    if len(calls) != 1:
        return None
    try:
        return canonical(calls[0])
    except ValueError:
        return None


def economic_key(call: dict) -> str:
    name, arguments = call["name"], call["arguments"]
    require(name in FAMILIES and isinstance(arguments, dict), "advancing parsed action")
    if name in {"hold", "abstain"}:
        return json.dumps([name], allow_nan=False)
    symbol = arguments.get("symbol")
    require(isinstance(symbol, str) and bool(symbol.strip()), "advancing symbol")
    # Pydantic's production float conversion may accept a numeric string/bool.
    # This key follows the accepted numerical action, without repairing its text.
    target = float(arguments["target_weight"])
    require(math.isfinite(target) and 0 <= target <= 1, "accepted target within bounds")
    target = 0.0 if target == 0 else target
    return json.dumps([name, symbol.strip().upper(), target], allow_nan=False)


def menu_keys(anchor: dict) -> set[str]:
    result = set()
    for item in anchor["menu"]:
        result.add(economic_key(json.loads(item["action_key"])))
    return result


def generation_complete(generation: dict) -> bool:
    ids = generation["response_token_ids"]
    require(
        isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids), "response tokens"
    )
    require(generation["response_token_count"] == len(ids), "response token count")
    require(type(generation["returned_finished"]) is bool, "returned finished bool")
    eos = generation["eos_token_ids"]
    require(isinstance(eos, list) and len(eos) == 2 and set(eos) == EOS, "fixed recognized EOS")
    require(generation["primary_eos_token_id"] == 151645, "fixed primary EOS")
    positions = [i for i, token in enumerate(ids) if token in EOS]
    require(generation["eos_positions"] == positions, "independently counted EOS positions")
    require(
        generation["response_last_token_id"] == (ids[-1] if ids else None), "last response token"
    )
    last = bool(ids and ids[-1] in EOS)
    require(generation["last_token_is_eos"] is last, "last-token EOS classification")
    return (
        0 < len(ids) < 4096
        and generation["finish_reason"] == "stop"
        and generation["returned_finished"] is True
        and positions == [len(ids) - 1]
        and (
            generation["stop_reason"] is None
            or (type(generation["stop_reason"]) is int and generation["stop_reason"] == ids[-1])
        )
    )


def check_sampling(params: dict, seed: int) -> None:
    expected = {
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 4096,
        "seed": seed,
        "ignore_eos": False,
    }
    require(isinstance(params, dict), "actual sampling params")
    for key, value in expected.items():
        require(key in params and params[key] == value, f"actual native sampling {key}")
    require(params["ignore_eos"] is False, "EOS not ignored")


def by_key(rows: list[dict], label: str) -> dict[str, dict]:
    result = {row["request_key"]: row for row in rows}
    require(len(result) == len(rows), f"unique {label} request keys")
    return result


def check_generation_sources(
    inputs: list[dict], generations: list[dict], raw: list[dict], submissions: list[dict]
) -> None:
    ins, gens = by_key(inputs, "input"), by_key(generations, "generation")
    require(set(ins) == set(gens), "all planned first requests returned")
    raw_batches = {r["batch_index"]: r for r in raw}
    sub_batches = {r["batch_index"]: r for r in submissions}
    require(
        len(raw_batches) == len(raw) and len(sub_batches) == len(submissions), "unique batch IDs"
    )
    require(set(raw_batches) == set(sub_batches), "raw and submitted batches match")
    seen, response_ids = set(), set()
    for batch_id in sorted(sub_batches):
        submitted, returned = sub_batches[batch_id]["requests"], raw_batches[batch_id]["outputs"]
        require(len(submitted) == len(returned) and bool(submitted), "batch returned all requests")
        numeric_ids = []
        for request, output in zip(submitted, returned, strict=True):
            key = request["request_key"]
            require(key in ins and key not in seen, "submitted planned request exactly once")
            seen.add(key)
            generation = gens[key]
            for name in (
                "anchor_id",
                "slot",
                "request_seed",
                "prompt_token_ids",
                "prompt_sha256",
                "prompt_token_count",
                "source_group",
            ):
                require(request[name] == ins[key][name], f"submitted input binding {name}")
            request_id = output["request_id"]
            require(
                isinstance(request_id, str) and request_id.isdecimal(), "numeric vLLM request ID"
            )
            require(request_id not in response_ids, "globally unique vLLM request ID")
            require(request_id == str(len(response_ids)), "native request IDs exactly 0..127")
            response_ids.add(request_id)
            numeric_ids.append(int(request_id))
            require(
                generation["vllm_request_id"] == request_id, "batch position/request ID mapping"
            )
            require(output["prompt_token_ids"] == ins[key]["prompt_token_ids"], "raw prompt tokens")
            require(
                output["finished"] == generation["returned_finished"],
                "raw returned-finished binding",
            )
            require(len(output["outputs"]) == 1, "one original response per request")
            response = output["outputs"][0]
            for raw_name, generated_name in (
                ("token_ids", "response_token_ids"),
                ("engine_text", "engine_text"),
                ("finish_reason", "finish_reason"),
                ("stop_reason", "stop_reason"),
            ):
                require(
                    response[raw_name] == generation[generated_name],
                    f"raw response binding {raw_name}",
                )
            require(
                request["actual_sampling_params"] == generation["actual_sampling_params"],
                "forwarded sampling evidence",
            )
        require(
            numeric_ids == sorted(numeric_ids) and len(set(numeric_ids)) == len(numeric_ids),
            "strict response ordering per batch",
        )
    require(seen == set(ins), "all 128 submissions observed")


def validate_slots(
    inputs: list[dict],
    generations: list[dict],
    evaluations: list[dict],
    anchors: dict[str, dict],
    *,
    require_design=True,
) -> list[dict]:
    ins, gens, evals = (
        by_key(inputs, "input"),
        by_key(generations, "generation"),
        by_key(evaluations, "evaluation"),
    )
    require(
        set(ins) == set(gens) == set(evals), "planned/generated/evaluated populations identical"
    )
    seeds, result = set(), []
    for key, request in ins.items():
        generation, evaluation = gens[key], evals[key]
        anchor_id, slot = request["anchor_id"], integer(request["slot"], "slot")
        require(
            anchor_id in anchors and slot in range(4) and key == f"{anchor_id}/s{slot}",
            "fixed request identity",
        )
        anchor = anchors[anchor_id]
        for row in (generation, evaluation):
            require(
                row["request_key"] == key and row["anchor_id"] == anchor_id and row["slot"] == slot,
                "response/evaluation identity",
            )
        for field in ("prompt_token_ids", "prompt_sha256", "prompt_token_count", "source_group"):
            require(generation[field] == request[field], f"generated input binding {field}")
        require(generation["request_seed"] == request["request_seed"], "generated request seed")
        seed = integer(request["request_seed"], "request seed")
        require(seed < 2**63 and seed not in seeds, "noncolliding actual request seeds")
        seeds.add(seed)
        prompt_ids = request["prompt_token_ids"]
        require(
            isinstance(prompt_ids, list)
            and bool(prompt_ids)
            and all(type(t) is int and t >= 0 for t in prompt_ids),
            "complete input tokens",
        )
        prompt_sha = hashlib.sha256(json.dumps(prompt_ids).encode("ascii")).hexdigest()
        require(
            prompt_sha == request["prompt_sha256"] == anchor["historical_prompt_sha256"],
            "sealed native prompt SHA",
        )
        require(
            len(prompt_ids)
            == request["prompt_token_count"]
            == anchor["prompt_token_count"]
            <= 8192,
            "input token count/cap",
        )
        require(generation["returned_prompt_token_ids"] == prompt_ids, "returned prompt identity")
        if "prompt_sha256" in generation:
            require(generation["prompt_sha256"] == prompt_sha, "generation prompt SHA")
        source = request["source_group"]
        for source_name, anchor_name in (
            ("source_seed", "seed"),
            ("source_sibling_id", "sibling_id"),
            ("source_episode_id", "episode_id"),
            ("source_policy_step", "policy_step"),
            ("decision_session_index", "decision_session_index"),
        ):
            require(source[source_name] == anchor[anchor_name], f"source anchor {source_name}")
        require(
            source["source_seed"] == 42 and source["source_sibling_id"] == 0,
            "source selection, not RNG seed",
        )
        check_sampling(generation["actual_sampling_params"], seed)
        complete = generation_complete(generation)
        expected_generation = "COMPLETE" if complete else "INCOMPLETE"
        require(
            generation["generation_status"]
            == evaluation["generation_status"]
            == expected_generation,
            "independent finish classification",
        )
        require(
            (generation["failure_reason"] is None)
            if complete
            else isinstance(generation["failure_reason"], str)
            and bool(generation["failure_reason"]),
            "finish failure reason",
        )
        require(
            isinstance(generation["response_text"], str)
            and isinstance(generation["engine_text"], str),
            "original decoded response text",
        )
        calls = parse_calls(generation["response_text"]) if complete else []
        require(
            evaluation["parsed_calls"] == finite_json(calls), "independent original Hermes parsing"
        )
        require(
            evaluation["api_action_key"] == api_key(calls),
            "raw API key before economic normalization",
        )
        for name in ("financial_state_sha256", "observation_messages_sha256"):
            require(evaluation[name] == anchor[name], f"same prefix {name}")
        require(
            evaluation["prefix_requests"] == anchor["prefix_requests"],
            "full original prefix requests",
        )
        close(evaluation["hold_Y"], anchor["old_hold_Y"], "frozen hold baseline reference")
        status, info, trace = (
            evaluation["env_status"],
            evaluation["first_info"],
            evaluation["first_trace"],
        )
        family = None
        if not complete:
            require(
                status == "NOT_APPLIED" and info is None and trace is None,
                "incomplete generation not applied",
            )
        else:
            require(
                isinstance(info, dict) and isinstance(trace, dict),
                "every complete original response applied once",
            )
            require(
                trace["step_index"] == anchor["prefix_requests"],
                "first trace retains original prefix index",
            )
            require(
                trace["decision_session_index"] == anchor["decision_session_index"],
                "same decision session",
            )
            require(trace["done"] is False, "selected first action is not original window endpoint")
            if info.get("invalid_action") is True:
                require(
                    status == "INVALID"
                    and trace["invalid_action"] is True
                    and trace["action"] == "invalid",
                    "invalid environment classification",
                )
                require(
                    trace["result_session_index"] == trace["decision_session_index"],
                    "rejected request does not advance",
                )
            else:
                require(
                    info.get("invalid_action") is not True and trace["invalid_action"] is False,
                    "valid trace flag",
                )
                family = trace["action"]
                require(len(calls) == 1 and calls[0]["name"] == family, "valid call/trace family")
                if family in INFORMATION:
                    require(
                        status == "NONADVANCING" and info.get("tool") == family,
                        "information tool classification",
                    )
                    require(
                        trace["result_session_index"] == trace["decision_session_index"],
                        "information request does not advance",
                    )
                else:
                    require(
                        family in FAMILIES
                        and status == "ADVANCING"
                        and info.get("action") == family,
                        "economic advancing classification",
                    )
                    require(
                        trace["result_session_index"] == trace["decision_session_index"] + 1,
                        "exactly one decision advance",
                    )
        require(evaluation["family"] == family, "environment family, not guessed parsed name")
        order, off_grid, in_menu = None, None, None
        if status == "ADVANCING":
            order = economic_key(calls[0])
            normalized = json.loads(order)
            if family in {"buy", "sell"}:
                off_grid = normalized[2] not in (0.0, 0.1, 0.2)
            in_menu = order in menu_keys(anchor)
            checks = evaluation["first_step_checks"]
            require(
                isinstance(checks, dict)
                and all(
                    checks.get(n) is True
                    for n in ("before_identity", "history_append", "cash_position_fees")
                ),
                "advancing first-step engineering checks",
            )
            require(all(v is True for v in checks.values()), "no failed advancing check")
            require(
                evaluation["terminal_label_status"] == "COMPLETE_FIXED_HOLD",
                "complete original-endpoint label",
            )
            number(evaluation["Y"], "terminal financial Y")
            close(evaluation["q"], evaluation["Y"] - evaluation["hold_Y"], "q=Y-hold")
            metrics = evaluation["terminal_metrics"]
            require(
                isinstance(metrics, dict) and metrics["session_count"] == 21,
                "complete original 21-session financial label",
            )
            require(
                evaluation["suffix_hold_requests"] == 19 - anchor["decision_session_index"],
                "fixed hold to the original window endpoint",
            )
            close(metrics["terminal_score"], evaluation["Y"], "fixed-hold terminal score")
        else:
            require(evaluation["terminal_label_status"] == "NA", "unsupported slot label NA")
            require(
                all(
                    evaluation[n] is None
                    for n in ("Y", "q", "terminal_metrics", "first_step_checks")
                ),
                "no zero-fill or invented target for unsupported slot",
            )
        require(evaluation["economic_order_key"] == order, "independent semantic economic key")
        require(
            evaluation["off_grid"] is off_grid and evaluation["in_anchor_menu"] is in_menu,
            "grid vs legality axes",
        )
        result.append(
            {
                "request_key": key,
                "anchor_id": anchor_id,
                "slot": slot,
                "request_seed": seed,
                "episode_id": anchor["episode_id"],
                "policy_step": anchor["policy_step"],
                "decision_session_index": anchor["decision_session_index"],
                "generation_status": expected_generation,
                "failure_reason": generation["failure_reason"],
                "env_status": status,
                "terminal_label_status": evaluation["terminal_label_status"],
                "family": family,
                "parsed_family": calls[0]["name"]
                if len(calls) == 1 and isinstance(calls[0]["name"], str)
                else None,
                "parsed_call_count": len(calls),
                "api_action_key": evaluation["api_action_key"],
                "economic_order_key": order,
                "off_grid": off_grid,
                "in_anchor_menu": in_menu,
                "response_tokens": len(generation["response_token_ids"]),
                "finish_reason": generation["finish_reason"],
                "zero_fill": not info.get("fills", []) if status == "ADVANCING" else None,
                "skipped_count": len(info.get("skipped_orders", []))
                if status == "ADVANCING"
                else None,
            }
        )
    if require_design:
        require(
            len(inputs) == 128 and len({r["anchor_id"] for r in result}) == 32,
            "fixed128 requests/32 anchors",
        )
    return result


def partition(values: list[str]) -> list[int]:
    return sorted(Counter(values).values(), reverse=True)


def summarize_groups(
    slots: list[dict], evaluations: list[dict], *, require_design=True
) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    ev = by_key(evaluations, "evaluation")
    repeated_checks = 0
    for row in slots:
        groups[row["anchor_id"]].append(row)
    result = []
    for anchor, members in sorted(groups.items()):
        require(
            len(members) == 4 and {r["slot"] for r in members} == set(range(4)),
            "all four planned slots retained",
        )
        advanced = [r for r in members if r["env_status"] == "ADVANCING"]
        supported = len(advanced) == 4
        orders = [r["economic_order_key"] for r in advanced]
        families = [r["family"] for r in advanced]
        for order in set(orders):
            same = [ev[r["request_key"]] for r in advanced if r["economic_order_key"] == order]
            if len(same) > 1:
                repeated_checks += len(same) - 1
                for value in same[1:]:
                    for name in ("first_info", "Y", "q", "terminal_metrics"):
                        require(
                            value[name] == same[0][name],
                            f"same-state/order deterministic label identity: {anchor}:{name}",
                        )
        order_parts, family_parts = partition(orders), partition(families)
        family_gate = len(set(families)) > 1 if supported else None
        order_gate = len(set(orders)) > 1 if supported else None
        result.append(
            {
                "anchor_id": anchor,
                "episode_id": members[0]["episode_id"],
                "policy_step": members[0]["policy_step"],
                "decision_session_index": members[0]["decision_session_index"],
                "planned_slots": 4,
                "generation_complete": sum(r["generation_status"] == "COMPLETE" for r in members),
                "incomplete": sum(r["generation_status"] == "INCOMPLETE" for r in members),
                "invalid": sum(r["env_status"] == "INVALID" for r in members),
                "nonadvancing": sum(r["env_status"] == "NONADVANCING" for r in members),
                "k_advancing": len(advanced),
                "full_n4_support": supported,
                "n4_family_diversity": len(set(families)) if supported else None,
                "n4_order_diversity": len(set(orders)) if supported else None,
                "n4_family_gate": family_gate,
                "n4_order_gate": order_gate,
                "n4_family_partition": family_parts if supported else None,
                "n4_order_partition": order_parts if supported else None,
                "n4_family_pool_opportunity": bool(family_gate and max(family_parts) >= 2)
                if supported
                else None,
                "n4_exact_pool_same_family_gate_opportunity": bool(
                    family_gate and max(order_parts) >= 2
                )
                if supported
                else None,
                "n4_exact_pool_order_gate_opportunity": bool(order_gate and max(order_parts) >= 2)
                if supported
                else None,
                "legal_subset_order_partition": order_parts,
                "legal_subset_unique_orders": len(set(orders)),
                "legal_subset_peer_slots": sum(orders.count(x) > 1 for x in orders),
                "legal_subset_singleton_slots": sum(orders.count(x) == 1 for x in orders),
                "legal_subset_non_hold_orders": sum(
                    r["family"] in {"buy", "sell"} for r in advanced
                ),
                "legal_subset_off_grid_trades": sum(r["off_grid"] is True for r in advanced),
            }
        )
    if require_design:
        require(
            len(result) == 32 and len({r["episode_id"] for r in result}) == 8, "fixed eight windows"
        )
        for episode in {r["episode_id"] for r in result}:
            require(
                {r["decision_session_index"] for r in result if r["episode_id"] == episode}
                == {0, 5, 10, 15},
                "fixed four source days",
            )
    return result, {"same_state_order_duplicate_label_checks": repeated_checks}


def aggregate(slots: list[dict], groups: list[dict]) -> dict:
    advanced = [r for r in slots if r["env_status"] == "ADVANCING"]
    supported = [r for r in groups if r["full_n4_support"]]
    peers = sum(r["legal_subset_peer_slots"] for r in groups)
    singleton = sum(r["legal_subset_singleton_slots"] for r in groups)
    require(peers + singleton == len(advanced), "legal subset peer/singleton partition")
    return {
        "planned_requests": len(slots),
        "planned_groups": len(groups),
        "generation_complete": sum(r["generation_status"] == "COMPLETE" for r in slots),
        "generation_incomplete": sum(r["generation_status"] == "INCOMPLETE" for r in slots),
        "env_status_counts": dict(sorted(Counter(r["env_status"] for r in slots).items())),
        "family_counts_among_valid_requests": dict(
            sorted(Counter(r["family"] for r in slots if r["family"] is not None).items())
        ),
        "k_advancing_histogram_all_groups": {
            str(k): sum(r["k_advancing"] == k for r in groups) for k in range(5)
        },
        "full_n4_support_groups": len(supported),
        "full_n4_support_rate": len(supported) / len(groups) if groups else None,
        "full_n4_family_gate_open_groups": sum(r["n4_family_gate"] for r in supported),
        "full_n4_order_gate_open_groups": sum(r["n4_order_gate"] for r in supported),
        "full_n4_order_partition_counts": dict(
            sorted(Counter("+".join(map(str, r["n4_order_partition"])) for r in supported).items())
        ),
        "full_n4_family_pool_opportunity_groups": sum(
            r["n4_family_pool_opportunity"] for r in supported
        ),
        "full_n4_exact_pool_same_family_gate_opportunity_groups": sum(
            r["n4_exact_pool_same_family_gate_opportunity"] for r in supported
        ),
        "full_n4_exact_pool_order_gate_opportunity_groups": sum(
            r["n4_exact_pool_order_gate_opportunity"] for r in supported
        ),
        "legal_advancing_slots": len(advanced),
        "legal_subset_peer_slots": peers,
        "legal_subset_singleton_slots": singleton,
        "legal_subset_peer_rate_all_planned_slots": peers / len(slots) if slots else None,
        "legal_subset_peer_rate_given_advancing": peers / len(advanced) if advanced else None,
        "legal_trading_slots": sum(r["family"] in {"buy", "sell"} for r in advanced),
        "legal_off_grid_trading_slots": sum(r["off_grid"] is True for r in advanced),
        "legal_orders_in_anchor_menu": sum(r["in_anchor_menu"] is True for r in advanced),
        "legal_orders_outside_anchor_menu": sum(r["in_anchor_menu"] is False for r in advanced),
        "legal_zero_fill_slots": sum(r["zero_fill"] for r in advanced),
        "legal_slots_with_skip": sum(r["skipped_count"] > 0 for r in advanced),
    }


def checked_relative(base: Path, name: str, digest: str) -> None:
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "relative completion evidence path")
    require(sha(base / path) == digest, f"completed evidence changed: {name}")


def write_new(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), "nonempty readout table")
    encoded = [
        {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
        for row in rows
    ]
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(encoded[0]))
        writer.writeheader()
        writer.writerows(encoded)


def readout(run: Path) -> dict:
    run = run.resolve(strict=True)
    require(
        not any((run / name).exists() for name in OUTPUTS), "new readout outputs only; no overwrite"
    )
    contract, proposal = read_json(run / "contract.json"), read_json(run / "proposal.json")
    generation_done, evaluation_done = (
        read_json(run / "generation_completion.json"),
        read_json(run / "evaluation_completion.json"),
    )
    require(
        generation_done["status"] == evaluation_done["status"] == "COMPLETE",
        "completed worker and CPU evaluation",
    )
    require(sha(run / "proposal.json") == contract["proposal_sha256"], "sealed proposal bytes")
    require(sha(run / "inputs.jsonl") == contract["inputs_sha256"], "sealed 128 inputs")
    for name, digest in contract["source_sha256"].items():
        require(sha(Path(name)) == digest, f"sealed source/input changed: {name}")
    required = {
        "generations.jsonl",
        "raw_generations.jsonl",
        "submissions.jsonl",
        "worker_config.json",
    }
    require(required <= set(generation_done["output_sha256"]), "all raw generation evidence bound")
    require("evaluations.jsonl" in evaluation_done["output_sha256"], "all CPU evaluations bound")
    for completion in (generation_done, evaluation_done):
        for name, digest in completion["output_sha256"].items():
            checked_relative(run, name, digest)
    menu_run = Path(contract["source_menu_run"]).resolve(strict=True)
    require(
        sha(run / "proposal.json") == sha(menu_run / "sampling_proposal.json"),
        "exact copied source proposal bytes",
    )
    require(read_json(menu_run / "readout.json")["status"] == "PASS", "prior menu engineering PASS")
    for name, digest in proposal["source_files_sha256"].items():
        checked_relative(menu_run, name, digest)
    anchors_all = {r["anchor_id"]: r for r in read_jsonl(menu_run / "anchors.jsonl")}
    packets = read_jsonl(menu_run / "prompt_packets.jsonl")
    packet_map = {r["anchor_id"]: r for r in packets}
    input_rows = read_jsonl(run / "inputs.jsonl")
    generations = read_jsonl(run / "generations.jsonl")
    evaluations = read_jsonl(run / "evaluations.jsonl")
    selected = {g["anchor_id"] for g in proposal["groups"]}
    require(
        len(selected) == 32 and selected == {r["anchor_id"] for r in input_rows},
        "fixed proposal anchors",
    )
    anchors = {k: anchors_all[k] for k in selected}
    expected = {}
    for group in proposal["groups"]:
        anchor_id = group["anchor_id"]
        for slot in group["slots"]:
            expected[f"{anchor_id}/s{slot['slot']}"] = (group, slot)
    require(
        len(expected) == 128 and set(expected) == set(by_key(input_rows, "input")),
        "fixed 128 proposal slots",
    )
    for row in input_rows:
        group, slot = expected[row["request_key"]]
        require(row["request_seed"] == slot["request_seed"], "frozen actual proposal seed")
        require(
            row["prompt_token_ids"] == packet_map[row["anchor_id"]]["prompt_token_ids"],
            "complete sealed prompt packet",
        )
        require(
            row["prompt_sha256"] == slot["prompt_sha256"] == group["historical_prompt_sha256"],
            "proposal prompt SHA",
        )
    check_generation_sources(
        input_rows,
        generations,
        read_jsonl(run / "raw_generations.jsonl"),
        read_jsonl(run / "submissions.jsonl"),
    )
    slots = validate_slots(input_rows, generations, evaluations, anchors)
    groups, checks = summarize_groups(slots, evaluations)
    counts = aggregate(slots, groups)
    require(
        generation_done["model_requests"] == evaluation_done["model_requests"] == 128,
        "exactly 128 original model requests",
    )
    require(generation_done["followup_model_requests"] == 0, "no followup model requests")
    require(
        generation_done["optimizer_updates"] == evaluation_done["optimizer_updates"] == 0,
        "no optimizer updates",
    )
    require(evaluation_done["slots_evaluated"] == len(slots), "all planned slots CPU classified")
    require(
        evaluation_done["fixed_hold_labels"] == counts["legal_advancing_slots"],
        "labels only for complete legal advancing orders",
    )
    require(evaluation_done["formal_valid_oos_access"] is False, "Train-only label evaluation")
    days = []
    for day in (0, 5, 10, 15):
        ds = [r for r in slots if r["decision_session_index"] == day]
        dg = [r for r in groups if r["decision_session_index"] == day]
        days.append({"decision_session_index": day, **aggregate(ds, dg)})
    checks.update(
        status="PASS",
        planned_requests_verified=len(slots),
        planned_groups_verified=len(groups),
        noncolliding_request_seeds=128,
        partial_groups_not_recentered=True,
        native_finish_and_eos_reclassified=True,
        raw_to_submission_mapping_verified=True,
        original_text_independently_Hermes_parsed=True,
        economic_keys_independently_normalized=True,
    )
    input_names = (
        "contract.json",
        "proposal.json",
        "inputs.jsonl",
        "generation_completion.json",
        "evaluation_completion.json",
        "generations.jsonl",
        "evaluations.jsonl",
        "raw_generations.jsonl",
        "submissions.jsonl",
        "worker_config.json",
    )
    result = {
        "profile": PROFILE,
        "status": "PASS",
        "scope": "descriptive first-request support only",
        "counts": counts,
        "checks": checks,
        "day_summary": days,
        "definitions": {
            "full_n4_support": (
                "all four planned requests generation-complete, legal advancing "
                "and complete fixed-hold labels"
            ),
            "partial_group": (
                "n4 diversity/gate/repetition is NA, never a recentered smaller credit group"
            ),
            "legal_subset": (
                "coverage only, conditional on legal advancing; "
                "all planned-slot denominator separately retained"
            ),
            "economic_order": (
                "buy/sell symbol stripped+uppercased and numerical target; "
                "0/0.0/-0 identical; abstain reason excluded"
            ),
            "off_grid": (
                "legal advancing buy/sell numerical target outside exact {0,.1,.2}; "
                "not an illegality test"
            ),
            "in_anchor_menu": (
                "semantic order match against the finite visible-state menu, separate from legality"
            ),
            "finish": (
                "nonempty response below4096, engine stop+finished, "
                "recognized EOS only at last token"
            ),
        },
        "limits": [
            "Common recovered SFT on M2 historical off-policy states, "
            "not a replay of the original M2 policies.",
            "Only one first request per planned slot; no recovery, followup, retry or resampling.",
            "Information requests, rejected calls and incomplete outputs "
            "have no invented zero/hold financial label.",
            "Fixed-hold deterministic labels and exact repeats do not establish "
            "denoising, learning, gradient correctness or financial gains.",
            "Peers are within one identical-prefix anchor only; "
            "no cross-anchor pooling or independent-sample claim.",
            "The 32 anchors share original windows and dates; "
            "this is descriptive Train evidence, not OOS inference.",
        ],
        "input_sha256": {name: sha(run / name) for name in input_names},
        "readout_source_sha256": sha(Path(__file__).resolve()),
    }
    for name, digest in result["input_sha256"].items():
        checked_relative(run, name, digest)
    write_csv(run / "slot_metrics.csv", slots)
    write_csv(run / "group_metrics.csv", groups)
    write_csv(run / "day_summary.csv", days)
    write_new(run / "checks.json", checks)
    result["output_sha256"] = {name: sha(run / name) for name in OUTPUTS if name != "readout.json"}
    write_new(run / "readout.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = readout(args.run)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, allow_nan=False))


if __name__ == "__main__":
    main()
