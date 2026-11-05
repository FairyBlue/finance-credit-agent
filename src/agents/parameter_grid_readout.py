"""Stream the fixed two-seed Train grid into descriptive performance diagnostics.

Only validated execution-complete cells are read. Missing cells produce PARTIAL.
An explicitly reviewed v2 profile may retain failed evaluation prefixes; their
financial outcomes stay missing and comparisons use common complete slots only.
This module never loads model weights or modifies the training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SEEDS = (42, 43)
EVAL_NUMBERS = (210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 441)
TRAIN_NUMBERS = (1, 65, 9, 73, 17, 149, 25, 157)
EVAL_IDS = tuple(f"formal-train-{n:04d}" for n in EVAL_NUMBERS)
TRAIN_IDS = tuple(f"formal-train-{n:04d}" for n in TRAIN_NUMBERS)
LANES = {"greedy": (0,), "sampled": (1, 2, 3, 4)}
METRICS = (
    "raw_net_return",
    "terminal_score",
    "transaction_cost",
    "max_drawdown",
    "agent_volatility",
    "benchmark_excess",
    "turnover",
    "maximum_position_weight",
    "benchmark_return",
)
# actor learning rate, residual eta, arm; SFT has no optimizer learning rate.
CELLS = {
    "sft0": (None, 0.0, "SFT"),
    "m1_lr1e7": (1e-7, 0.0, "M1"),
    "m2_lr1e7_eta0125": (1e-7, 0.125, "M2"),
    "m2_lr1e7_eta025": (1e-7, 0.25, "M2"),
    "m1_lr2e7": (2e-7, 0.0, "M1"),
    "m2_lr2e7_eta0125": (2e-7, 0.125, "M2"),
    "m2_lr2e7_eta025": (2e-7, 0.25, "M2"),
}
PLANNED_PER_SEED = 972
PLANNED = 1944


def _read(path):
    return json.loads(path.read_text())


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"missing/nonfinite {name}")
    return value


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _records(directory, sources):
    """At most one large evidence record is live; retain only source hashes."""
    for path in sorted(directory.glob("*.jsonl")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if line.strip():
                    yield json.loads(line)
        sources[str(path)] = digest.hexdigest()


def _metadata(seed, cell_id, *, cells=CELLS):
    lr, eta, arm = cells[cell_id]
    return dict(seed=seed, cell_id=cell_id, model=arm, actor_lr=lr, eta=eta)


def _observation(interaction):
    """Read the resulting environment state, never infer fills from requests."""
    text = interaction.get("observation", {}).get("text", "")
    if not text:
        return None
    messages = json.loads(text)
    for message in reversed(messages):
        content = message.get("content", "")
        if "<tool_response>" in content:
            body = content.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
            value = json.loads(body.strip())
            if isinstance(value, dict) and "cash" in value and "positions" in value:
                return value
    return None


def _behavior(record):
    counts = Counter()
    errors = Counter()
    filled = set()
    requested = set()
    held = set()
    cash_fractions, concentrations, holding_counts = [], [], []
    invalid_run = longest_invalid_run = 0
    repeated_run = longest_repeated_run = 0
    previous_response = None
    interactions = record.get("interactions", [])
    steps = record["steps"]
    for step in steps:
        action = step.get("action", "unknown")
        counts[action if action in {"hold", "buy", "sell", "abstain"} else "other"] += 1
        counts["invalid"] += int(step.get("invalid_action") is True)
        invalid_run = invalid_run + 1 if step.get("invalid_action") is True else 0
        longest_invalid_run = max(longest_invalid_run, invalid_run)
        decision, result = step.get("decision_session_index"), step.get("result_session_index")
        if type(decision) is int and type(result) is int:
            counts["nonadvancing"] += int(result == decision)
            counts["advancing"] += int(result > decision)
    for interaction in interactions:
        info = interaction.get("info", {})
        for fill in info.get("fills", []):
            counts["fills"] += 1
            filled.add(fill["symbol"])
            counts["fill_" + fill.get("side", "unknown")] += 1
        if info.get("invalid_action") is True:
            errors[str(info.get("error", info.get("reason", "unspecified")))] += 1
        response = interaction.get("response", {}).get("text", "")
        response_identity = interaction.get("response", {}).get("sha256") or response
        if response_identity:
            repeated_run = repeated_run + 1 if response_identity == previous_response else 1
            longest_repeated_run = max(longest_repeated_run, repeated_run)
            previous_response = response_identity
        if "<tool_call>" in response:
            try:
                call = json.loads(response.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0])
                symbol = call.get("arguments", {}).get("symbol")
                family = call.get("name")
                if family in {"hold", "buy", "sell", "abstain"}:
                    counts["tool_" + family] += 1
                if symbol:
                    requested.add(str(symbol))
            except (ValueError, TypeError):
                # Invalid syntax remains counted as an invalid request by the environment.
                pass
        # Invalid calls do not advance time; avoid duplicating portfolio states.
        if interaction.get("result_session_index") == interaction.get("decision_session_index"):
            continue
        observation = _observation(interaction)
        if observation is None:
            continue
        positions = observation["positions"]
        weights = [_number(p["weight"], "position weight") for p in positions.values()]
        held.update(positions)
        equity = _number(observation.get("portfolio_equity", info.get("equity")), "equity")
        if equity <= 0:
            raise ValueError("nonpositive portfolio equity")
        cash_fractions.append(_number(observation["cash"], "cash") / equity)
        concentrations.append(max(weights, default=0.0))
        holding_counts.append(len(positions))
    return dict(
        decisions=len(steps),
        invalid_decisions=counts["invalid"],
        advancing_steps=counts["advancing"],
        nonadvancing_steps=counts["nonadvancing"],
        maximum_consecutive_invalid_steps=longest_invalid_run,
        maximum_repeated_response_run=longest_repeated_run,
        tool_hold_requests=counts["tool_hold"],
        tool_buy_requests=counts["tool_buy"],
        tool_sell_requests=counts["tool_sell"],
        tool_abstain_requests=counts["tool_abstain"],
        hold_requests=counts["hold"],
        buy_requests=counts["buy"],
        sell_requests=counts["sell"],
        abstain_requests=counts["abstain"],
        other_requests=counts["other"],
        executed_fills=counts["fills"],
        buy_fills=counts["fill_buy"],
        sell_fills=counts["fill_sell"],
        filled_symbols=sorted(filled),
        requested_symbols=sorted(requested),
        held_symbols=sorted(held),
        distinct_filled_symbols=len(filled),
        distinct_requested_symbols=len(requested),
        mean_cash_fraction=_mean(cash_fractions),
        final_cash_fraction=cash_fractions[-1] if cash_fractions else None,
        mean_position_concentration=_mean(concentrations),
        mean_held_stock_count=_mean(holding_counts),
        portfolio_state_samples=len(cash_fractions),
        error_histogram=dict(errors),
    )


def _trajectory(record, seed, cell_id, summary, *, retain_prefix=False, cells=CELLS):
    identity = record.get("rollout_identity", record.get("identity", {}))
    step, sibling, validate = (identity.get(k) for k in ("step", "rollout_n", "validate"))
    if (
        type(step) is not int
        or type(sibling) is not int
        or type(validate) is not bool
        or record.get("base_seed") != seed
        or record.get("arm") != summary["cell"]["arm"]
    ):
        raise ValueError("raw trajectory seed/arm/rollout identity differs")
    steps = record.get("steps", [])
    complete = record.get("complete") is True and record.get("status") == "COMPLETE"
    if not complete:
        requests = record.get("requests", [])
        if (
            not retain_prefix
            or not validate
            or record.get("complete") is not False
            or record.get("status") != "INCOMPLETE"
            or record.get("evaluation_failure_profile") != "retain_prefix_v1"
            or record.get("failure_kind") != "trajectory_max_steps"
            or record.get("transport_output_returned") is not True
            or record.get("transport_step_count") != 256
            or record.get("transport_rewards_are_not_financial") is not True
            or record.get("metrics") is not None
            or record.get("terminal_score") is not None
            or record.get("credits") is not None
            or len(steps) != 256
            or not steps
            or any(s.get("done") is True for s in steps)
            or len(requests) != 256
            or any(r.get("status") != "COMPLETE" for r in requests)
        ):
            raise ValueError("incomplete trajectory is not a retained evaluation prefix")
        metrics = dict.fromkeys(METRICS)
    else:
        metrics = {name: _number(record.get("metrics", {}).get(name), name) for name in METRICS}
        if (
            not steps
            or steps[-1].get("done") is not True
            or record["metrics"].get("step_count") != len(steps)
            or record["metrics"].get("session_count") != 21
        ):
            raise ValueError("trajectory lacks complete 21-session termination")
    behavior = _behavior(record)
    if complete and behavior["invalid_decisions"] != record["metrics"].get("invalid_count"):
        raise ValueError("invalid counter disagrees with raw steps")
    episode = record.get("episode_id")
    expected_steps = (0,) if cell_id == "sft0" else (2, 4)
    if validate:
        if step not in expected_steps or episode not in EVAL_IDS or sibling not in range(5):
            raise ValueError("raw evaluation is outside fixed grid")
        lane = "greedy" if sibling == 0 else "sampled"
    else:
        if (
            cell_id == "sft0"
            or step not in range(1, 5)
            or sibling not in range(4)
            or episode not in TRAIN_IDS[(step - 1) * 2 : step * 2]
        ):
            raise ValueError("raw training is outside fixed four-round grid")
        lane = "train"
    return dict(
        **_metadata(seed, cell_id, cells=cells),
        policy_step=step,
        lane=lane,
        episode_id=episode,
        rollout_n=sibling,
        complete=complete,
        trajectory_status="COMPLETE" if complete else "INCOMPLETE",
        failure_kind=record.get("failure_kind") if not complete else None,
        failure_error=record.get("error") if not complete else None,
        **metrics,
        **behavior,
    )


def _aggregate(rows):
    result = dict(trajectories=len(rows))
    for metric in METRICS:
        result["mean_" + metric] = _mean([r[metric] for r in rows if r["complete"]])
    counters = (
        "decisions",
        "invalid_decisions",
        "hold_requests",
        "buy_requests",
        "sell_requests",
        "abstain_requests",
        "other_requests",
        "executed_fills",
        "buy_fills",
        "sell_fills",
        "advancing_steps",
        "nonadvancing_steps",
        "tool_hold_requests",
        "tool_buy_requests",
        "tool_sell_requests",
        "tool_abstain_requests",
    )
    for name in counters:
        result[name] = sum(r[name] for r in rows)
    for name in (
        "distinct_filled_symbols",
        "distinct_requested_symbols",
        "mean_cash_fraction",
        "final_cash_fraction",
        "mean_position_concentration",
        "mean_held_stock_count",
    ):
        result["mean_" + name.removeprefix("mean_")] = _mean(
            [r[name] for r in rows if r[name] is not None]
        )
    for name in ("filled_symbols", "requested_symbols", "held_symbols"):
        result[name] = sorted(set().union(*(set(r[name]) for r in rows)))
    for name in ("maximum_consecutive_invalid_steps", "maximum_repeated_response_run"):
        result[name] = max((r[name] for r in rows), default=0)
    result["invalid_request_rate"] = (
        result["invalid_decisions"] / result["decisions"] if result["decisions"] else None
    )
    result["hold_request_rate"] = (
        result["hold_requests"] / result["decisions"] if result["decisions"] else None
    )
    result["portfolio_state_trajectories"] = sum(r["portfolio_state_samples"] > 0 for r in rows)
    errors = Counter()
    for row in rows:
        errors.update(row["error_histogram"])
    result["error_histogram"] = dict(errors)
    return result


def _coverage(rows, planned):
    complete = sum(row["complete"] for row in rows)
    return dict(
        planned_trajectories=planned,
        attempted_trajectories=len(rows),
        complete_trajectories=complete,
        incomplete_trajectories=len(rows) - complete,
        missing_trajectories=planned - len(rows),
        financial_coverage=complete / planned,
        financial_status="COMPLETE" if complete == planned else "INCOMPLETE",
    )


def _cell(
    base,
    seed,
    cell_id,
    summary,
    sources,
    *,
    retain_prefix=False,
    cells=CELLS,
    record_validator=None,
):
    lr, eta, arm = cells[cell_id]
    spec = summary.get("cell", {})
    rounds = 0 if cell_id == "sft0" else 4
    if (
        summary.get("status") not in {"COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES"}
        or summary.get("seed") != seed
        or spec.get("cell_id") != cell_id
        or spec.get("eta") != eta
        or spec.get("arm") != ("M1" if arm == "SFT" else arm)
        or summary.get("actual_rounds") != rounds
        or summary.get("returncode") != 0
    ):
        raise ValueError("cell completion/seed/method/update budget differs")
    evidence = summary.get("evidence", {})
    planned_eval = 60 if rounds == 0 else 120
    counts = evidence.get("trajectories", {})
    if (
        set(counts) - {"train", "eval"}
        or counts.get("train", 0) != rounds * 8
        or counts.get("eval") != planned_eval
    ):
        raise ValueError("validated trajectory budget differs")
    if (
        summary.get("actor_update", {}).get("observed_updates") != rounds
        or summary.get("critic_fit", {}).get("updates") != rounds
    ):
        raise ValueError("validated actor/critic update budget differs")
    role_counts = evidence.get("optimizer_evidence", {}).get("detected_role_counts", {})
    if any(role_counts.get(role, 0) != rounds for role in ("actor", "critic")):
        raise ValueError("actual optimizer update budget differs")
    settings = _read(base / "settings.json")
    if (
        settings.get("seeds") != [seed]
        or settings["training"]["critic_learning_rate"] != 1e-6
        or (lr is not None and settings["training"]["learning_rate"] != lr)
    ):
        raise ValueError("cell learning rate or seed differs from fixed grid")
    seen, trajectories = set(), []
    for record in _records(base / "artifacts/screening_evidence", sources):
        if record_validator is not None:
            record_validator(record)
        row = _trajectory(record, seed, cell_id, summary, retain_prefix=retain_prefix, cells=cells)
        key = (row["lane"] == "train", row["policy_step"], row["episode_id"], row["rollout_n"])
        if key in seen:
            raise ValueError("duplicate raw trajectory identity")
        seen.add(key)
        trajectories.append(row)
    expected = {
        (False, step, episode, sibling)
        for step in ((0,) if rounds == 0 else (2, 4))
        for episode in EVAL_IDS
        for sibling in range(5)
    }
    if rounds:
        expected.update(
            (True, step, episode, sibling)
            for step in range(1, 5)
            for episode in TRAIN_IDS[(step - 1) * 2 : step * 2]
            for sibling in range(4)
        )
    if seen != expected:
        raise ValueError(f"raw trajectory grid incomplete: {len(seen)}/{len(expected)}")
    failures = [row for row in trajectories if not row["complete"]]
    if retain_prefix and summary.get("financial_status") != (
        "INCOMPLETE" if failures else "COMPLETE"
    ):
        raise ValueError("cell financial status disagrees with raw coverage")
    curves, windows, behaviors = [], [], []
    groups = defaultdict(list)
    for row in trajectories:
        groups[(row["policy_step"], row["lane"])].append(row)
    for (step, lane), rows in sorted(groups.items()):
        completed = [r for r in rows if r["complete"]]
        aggregate = _aggregate(completed)
        meta = dict(**_metadata(seed, cell_id, cells=cells), policy_step=step, lane=lane)
        for label, selected in (
            ("all_attempts", rows),
            ("COMPLETE", completed),
            ("INCOMPLETE", [r for r in rows if not r["complete"]]),
        ):
            if selected:
                behaviors.append(dict(**meta, trajectory_status=label, **_aggregate(selected)))
        if lane == "train":
            continue
        reported = summary.get("evaluation_by_step", {}).get(str(step), {}).get(lane, {})
        coverage = _coverage(rows, len(EVAL_IDS) * len(LANES[lane]))
        if retain_prefix:
            for name in (
                "planned_trajectories",
                "attempted_trajectories",
                "complete_trajectories",
                "incomplete_trajectories",
            ):
                if reported.get(name) != coverage[name]:
                    raise ValueError(f"raw evaluation coverage disagrees with summary {name}")
        for name in (
            "trajectories",
            "decisions",
            "invalid_decisions",
            *("mean_" + n for n in METRICS if n != "benchmark_return"),
        ):
            actual, claimed = aggregate[name], reported.get(name)
            if actual is None and claimed is None and retain_prefix:
                continue
            if (
                type(claimed) not in (int, float)
                or not math.isfinite(claimed)
                or not math.isclose(actual, claimed, abs_tol=1e-10, rel_tol=1e-9)
            ):
                raise ValueError(f"raw evaluation disagrees with summary {step}/{lane}/{name}")
        curves.append(dict(**meta, **aggregate, **coverage))
        for episode in EVAL_IDS:
            selected = [r for r in rows if r["episode_id"] == episode]
            window_completed = [r for r in selected if r["complete"]]
            windows.append(
                dict(
                    **meta,
                    episode_id=episode,
                    **_aggregate(window_completed),
                    **_coverage(selected, len(LANES[lane])),
                )
            )
    return (
        curves,
        windows,
        behaviors,
        _diagnostics(base, seed, cell_id, summary, sources, cells=cells),
        trajectories,
        failures,
    )


def _diagnostics(base, seed, cell_id, summary, sources, *, cells=CELLS):
    if cell_id == "sft0":
        return []
    result = {}
    lr, eta, arm = cells[cell_id]
    identity = summary["execution_identity_sha256"]
    for directory in ("actor_update", "critic_fit", "sibling_diagnostics"):
        seen = set()
        for row in _records(base / "artifacts" / directory, sources):
            step = row.get("policy_step")
            if (
                type(step) is not int
                or step not in range(1, 5)
                or step in seen
                or row.get("arm") != arm
            ):
                raise ValueError(f"{directory} diagnostic fixed updates differ")
            seen.add(step)
            if directory != "sibling_diagnostics" and (
                row.get("status") != "COMPLETE" or row.get("execution_identity_sha256") != identity
            ):
                raise ValueError(f"{directory} diagnostic identity/completion differs")
            item = result.setdefault(
                step, dict(**_metadata(seed, cell_id, cells=cells), policy_step=step)
            )
            if directory == "actor_update":
                if row.get("actor_learning_rates") != [lr] or row.get("optimizer_step_count") != 1:
                    raise ValueError("actual actor update learning rate/count differs")
                item.update(
                    actor_grad_norm_max=max(row["optimizer_grad_norms_before_clip"]),
                    actor_original_pg_loss=row["original_training_pg_loss"],
                    actor_real_steps=row["real_step_count"],
                )
                for group in ("all_real", "legal", "invalid"):
                    values = row["groups"][group]
                    for name in (
                        "step_count",
                        "positive_advantage_steps",
                        "negative_advantage_steps",
                        "mean_advantage_token_weighted",
                        "mean_delta_logp_token_weighted",
                        "advantage_aligned_delta_token_fraction",
                        "post_ratio_outside_clip_fraction",
                    ):
                        item[f"actor_{group}_{name}"] = values.get(name)
            elif directory == "critic_fit":
                if row.get("critic_learning_rates") != [1e-6]:
                    raise ValueError("actual critic learning rate differs")
                item.update(
                    critic_grad_norm_max=max(row["optimizer_grad_norms_before_clip"]),
                    critic_original_vf_loss=row["original_training_vf_loss"],
                )
                for phase in ("pre", "post"):
                    for metric in ("rmse", "explained_variance", "mean_bias_value_minus_return"):
                        item[f"critic_{phase}_{metric}"] = row[phase][metric]
                item["critic_rmse_change"] = row["post"]["rmse"] - row["pre"]["rmse"]
            else:
                if row.get("run_seed") != seed or row.get("eta") != eta:
                    raise ValueError("residual diagnostic seed/eta differs")
                for name in (
                    "action_diversity_rate",
                    "invalid_step_count",
                    "real_step_count",
                    "whitening_std",
                    "pooling_changed_row_rate",
                ):
                    item["residual_" + name] = row.get(name)
                for name in ("mean_abs_S", "std_S", "std_U", "corr_U_S", "nonzero_S_rate"):
                    item["residual_" + name] = row["all_real_steps"].get(name)
                std_u = row["all_real_steps"].get("std_U")
                std_s = row["all_real_steps"].get("std_S")
                # Runtime std_S already measures the actor residual after eta.
                item["residual_scaled_std_ratio"] = (
                    std_s / std_u if std_u and std_s is not None else None
                )
                penalty = row["common_validity_penalty"]
                for name in ("financial_reward_sum", "training_reward_sum", "penalty_sum"):
                    item[name] = penalty[name]
        if seen != {1, 2, 3, 4}:
            raise ValueError(f"{directory} diagnostic evidence missing updates")
    for item in result.values():
        for name, value in item.items():
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"nonfinite training diagnostic {name}")
    return [result[step] for step in sorted(result)]


def _paired_aggregate(rows):
    """Use equal-weight windows after intersecting actual complete sibling slots."""
    result = _aggregate(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode_id"]].append(row)
    for metric in METRICS:
        result["mean_" + metric] = _mean(
            [_mean([row[metric] for row in group]) for group in grouped.values()]
        )
    return result


def _differences(rows, *, by_window=False, reference_pairs=None):
    groups = defaultdict(list)
    for row in rows:
        if row["lane"] != "train":
            groups[(row["seed"], row["cell_id"], row["policy_step"], row["lane"])].append(row)
    comparisons = []
    for _key, left_rows in sorted(groups.items()):
        row = left_rows[0]
        if row["model"] == "SFT":
            continue
        seed, lane, step = row["seed"], row["lane"], row["policy_step"]
        references = [("RL_minus_SFT", "sft0", 0)]
        if reference_pairs is not None:
            references.extend(
                (kind, reference_id, step)
                for kind, reference_id in reference_pairs.get(row["cell_id"], ())
            )
        elif row["model"] == "M2":
            references.append(
                ("M2_minus_M1", "m1_lr1e7" if row["actor_lr"] == 1e-7 else "m1_lr2e7", step)
            )
        for kind, reference_id, reference_step in references:
            reference_rows = groups.get((seed, reference_id, reference_step, lane), [])
            for episode in EVAL_IDS if by_window else (None,):
                left = [r for r in left_rows if episode is None or r["episode_id"] == episode]
                right = [r for r in reference_rows if episode is None or r["episode_id"] == episode]
                planned = len(LANES[lane]) * (1 if by_window else len(EVAL_IDS))
                left_slots = {(r["episode_id"], r["rollout_n"]): r for r in left}
                right_slots = {(r["episode_id"], r["rollout_n"]): r for r in right}
                common = sorted(
                    slot
                    for slot in left_slots.keys() & right_slots.keys()
                    if left_slots[slot]["complete"] and right_slots[slot]["complete"]
                )
                left_financial = _paired_aggregate([left_slots[slot] for slot in common])
                right_financial = _paired_aggregate([right_slots[slot] for slot in common])
                outcome = {
                    k: row[k]
                    for k in ("seed", "cell_id", "model", "actor_lr", "eta", "policy_step", "lane")
                }
                if by_window:
                    outcome["episode_id"] = episode
                outcome.update(
                    comparison=kind,
                    reference_cell=reference_id,
                    reference_policy_step=reference_step,
                    common_complete_slots=len(common),
                    common_complete_windows=len({slot[0] for slot in common}),
                    financial_status="COMPLETE" if len(common) == planned else "INCOMPLETE",
                    financial_weighting=(
                        "equal common nonempty windows, common sibling slots within window"
                    ),
                )
                for side, population in (("left", left), ("reference", right)):
                    coverage = _coverage(population, planned)
                    for name in (
                        "planned_trajectories",
                        "attempted_trajectories",
                        "complete_trajectories",
                        "incomplete_trajectories",
                        "missing_trajectories",
                    ):
                        outcome[side + "_" + name] = coverage[name]
                    outcome[side + "_complete_slots_excluded_by_pairing"] = coverage[
                        "complete_trajectories"
                    ] - len(common)
                for name in METRICS:
                    lvalue, rvalue = left_financial["mean_" + name], right_financial["mean_" + name]
                    if name == "raw_net_return":
                        outcome["left_net_return_percent"] = (
                            100 * lvalue if lvalue is not None else None
                        )
                        outcome["reference_net_return_percent"] = (
                            100 * rvalue if rvalue is not None else None
                        )
                        outcome["net_return_percentage_points"] = (
                            100 * (lvalue - rvalue)
                            if lvalue is not None and rvalue is not None
                            else None
                        )
                    else:
                        outcome[name + "_difference"] = (
                            lvalue - rvalue if lvalue is not None and rvalue is not None else None
                        )
                for name in (
                    "invalid_request_rate",
                    "hold_request_rate",
                    "mean_distinct_filled_symbols",
                    "mean_cash_fraction",
                    "mean_position_concentration",
                ):
                    lvalue, rvalue = left_financial[name], right_financial[name]
                    outcome[name + "_difference"] = (
                        lvalue - rvalue if lvalue is not None and rvalue is not None else None
                    )
                comparisons.append(outcome)
    return comparisons


def read_grid(base, *, expected_profiles=None):
    """Require all 1944 attempted slots for execution completion.

    PASS additionally requires every trajectory's financial outcome to be
    complete; retained evaluation failures never receive a financial score.
    """
    base = Path(base).resolve()
    if expected_profiles is None:
        from src.agents.parameter_grid import PROFILES

        expected_profiles = PROFILES
    curves, windows, behaviors, diagnostics, sources, issues, cells = [], [], [], [], {}, [], []
    trajectories, failures = [], []
    for seed in SEEDS:
        profile = expected_profiles[seed]
        retain_prefix = getattr(profile, "evaluation_incomplete_mode", None) == "retain_prefix_v1"
        version = "v2" if retain_prefix else "v1"
        if (
            profile.seed != seed
            or profile.name != f"common_sft_parameter_grid_seed{seed}_{version}"
            or getattr(profile, "evaluation_incomplete_mode", None)
            not in {None, "retain_prefix_v1"}
            or tuple(profile.train_ids) != TRAIN_IDS
            or tuple(profile.eval_ids) != EVAL_IDS
            or tuple(profile.evaluation_steps) != (2, 4)
            or profile.rounds != 4
            or profile.planned_trajectories != PLANNED_PER_SEED
        ):
            raise ValueError("readout profile is outside the fixed 1944-trajectory grid")
        seed_base = base / f"seed{seed}"
        state_path = seed_base / "state.json"
        if not state_path.exists():
            issues.append(dict(seed=seed, reason="seed state missing"))
            continue
        state = _read(state_path)
        sources[str(state_path)] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        if (
            state.get("profile") != profile.name
            or state.get("seed") != seed
            or state.get("planned_trajectories") != PLANNED_PER_SEED
            or tuple(state.get("evaluation_episode_ids", [])) != EVAL_IDS
            or tuple(state.get("train_episode_ids", [])) != TRAIN_IDS
            or state.get("evaluation_steps") != [2, 4]
        ):
            issues.append(dict(seed=seed, reason="seed state identity/grid differs"))
            continue
        if state.get("status") not in {"COMPLETE", "COMPLETE_WITH_EVALUATION_FAILURES"}:
            issues.append(dict(seed=seed, reason="seed controller is not COMPLETE"))
        by_id = {}
        for row in state.get("cells", []):
            cell_id = row.get("cell", {}).get("cell_id")
            if cell_id not in CELLS or cell_id in by_id:
                issues.append(dict(seed=seed, reason="duplicate/unknown cell in controller"))
            else:
                by_id[cell_id] = row
        heads = set()
        for cell_id in CELLS:
            summary = by_id.get(cell_id)
            if summary is None or summary.get("status") not in {
                "COMPLETE",
                "COMPLETE_WITH_EVALUATION_FAILURES",
            }:
                issues.append(dict(seed=seed, cell_id=cell_id, reason="cell is not COMPLETE"))
                continue
            try:
                head = summary.get("critic_initial_head_sha256")
                if cell_id != "sft0" and (not isinstance(head, str) or len(head) != 64):
                    raise ValueError("actual initial critic head evidence missing")
                values = _cell(
                    seed_base / cell_id,
                    seed,
                    cell_id,
                    summary,
                    sources,
                    retain_prefix=retain_prefix,
                )
                a, b, c, d, cell_trajectories, cell_failures = values
                curves.extend(a)
                windows.extend(b)
                behaviors.extend(c)
                diagnostics.extend(d)
                trajectories.extend(cell_trajectories)
                failures.extend(cell_failures)
                cells.append(
                    dict(
                        seed=seed,
                        cell_id=cell_id,
                        execution_status="COMPLETE_WITH_EVALUATION_FAILURES"
                        if cell_failures
                        else "COMPLETE",
                        **_coverage(cell_trajectories, 60 if cell_id == "sft0" else 152),
                    )
                )
                if cell_id != "sft0":
                    heads.add(head)
            except (KeyError, TypeError, ValueError, OSError) as error:
                issues.append(dict(seed=seed, cell_id=cell_id, reason=str(error)))
        if len(heads) > 1:
            issues.append(dict(seed=seed, reason="initial critic head differs between RL cells"))
    execution_complete = not issues and len(trajectories) == PLANNED and len(cells) == 14
    financial_complete = execution_complete and not failures
    execution_status = (
        ("COMPLETE_WITH_EVALUATION_FAILURES" if failures else "COMPLETE")
        if execution_complete
        else "PARTIAL"
    )
    eval_rows = [row for row in trajectories if row["lane"] != "train"]
    return dict(
        status=("PASS" if financial_complete else execution_status),
        execution_status=execution_status,
        financial_status="COMPLETE" if financial_complete else "INCOMPLETE",
        performance_pass=financial_complete,
        at_SGT=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        planned_trajectories=PLANNED,
        attempted_trajectories=len(trajectories),
        complete_trajectories=len(trajectories) - len(failures),
        incomplete_trajectories=len(failures),
        missing_trajectories=PLANNED - len(trajectories),
        evaluation_coverage=_coverage(eval_rows, 1560),
        cells=cells,
        issues=issues,
        source_sha256=sources,
        evaluation_curve=curves,
        window_curve=windows,
        paired_comparisons=_differences(trajectories),
        window_comparisons=_differences(trajectories, by_window=True),
        behavior_summary=behaviors,
        failure_summary=failures,
        training_diagnostics=diagnostics,
        formal_performance_claim=False,
        financial_outcomes_complete_only=True,
        automatic_selection=False,
        scope="original Train only; separate seeds, steps, decode lanes and equal-weight windows",
        sampled_rng_note=(
            "request seed includes policy step; SFT0 and steps2/4 "
            "are not exact common random numbers"
        ),
        diagnostic_note=(
            "actor delta logp is teacher-forced token drift, "
            "not generated action probability or causal proof"
        ),
    )


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def write_readout(base, *, allow_partial=True, expected_profiles=None):
    """Publish PARTIAL evidence before refusing an incomplete finalization."""
    base = Path(base).resolve()
    result = read_grid(base, expected_profiles=expected_profiles)
    base.mkdir(parents=True, exist_ok=True)
    for name in (
        "evaluation_curve",
        "window_curve",
        "paired_comparisons",
        "window_comparisons",
        "behavior_summary",
        "training_diagnostics",
        "failure_summary",
    ):
        _write_csv(base / f"{name}.csv", result[name])
    temporary = base / "grid_readout.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(base / "grid_readout.json")
    if result["execution_status"] == "PARTIAL" and not allow_partial:
        raise ValueError("fixed parameter grid is incomplete; PARTIAL readout retained")
    return result


write_grid_readout = write_readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require the full execution grid; report financial failures separately",
    )
    args = parser.parse_args()
    result = write_readout(args.directory, allow_partial=not args.require_complete)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "complete_trajectories", "planned_trajectories")
            }
        )
    )


if __name__ == "__main__":
    main()
