"""Describe the fixed fresh-SFT action-pooling comparison, without a profit gate.

Only execution-complete cells enter the readout. All planned evaluation slots,
including retained failures, are accounted for. Paired financial differences use
common complete sibling slots with equal window weights. This module reads raw
evidence and writes derived summaries only; it never loads policy weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.agents import parameter_grid_readout as shared
from src.agents.rollout_seed import normalize_rollout_identity

PROFILE = "shared_recovery_sft_action_pooling_readout_v1"
SEEDS = shared.SEEDS
TRAIN_IDS, EVAL_IDS = shared.TRAIN_IDS, shared.EVAL_IDS
CELLS = {
    "sft0": (None, 0.0, "SFT"),
    "m1_lr1e7": (1e-7, 0.0, "M1"),
    "m1_lr2e7": (2e-7, 0.0, "M1"),
    "m2_plain_lr1e7_eta025": (1e-7, 0.25, "M2-plain"),
    "m2_lr1e7_eta025": (1e-7, 0.25, "M2"),
}
REFERENCE_PAIRS = {
    "m2_plain_lr1e7_eta025": (("M2plain_minus_M1_same_lr", "m1_lr1e7"),),
    "m2_lr1e7_eta025": (
        ("M2_minus_M2plain", "m2_plain_lr1e7_eta025"),
        ("M2_minus_M1_same_lr", "m1_lr1e7"),
        ("M2_minus_M1_high_lr", "m1_lr2e7"),
    ),
}
PLANNED_PER_SEED, PLANNED_TRAIN, PLANNED_EVAL = 668, 256, 1080
PLANNED = 1336
TABLES = (
    "evaluation_curve",
    "window_curve",
    "paired_comparisons",
    "window_comparisons",
    "behavior_summary",
    "training_diagnostics",
    "failure_summary",
)


def _read(path, sources):
    payload = path.read_bytes()
    sources[str(path)] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def _profile(profile, seed):
    specs = {s.cell_id: s for s in profile.cell_specs}
    if (
        profile.seed != seed
        or profile.name != f"shared_recovery_sft_action_pooling_seed{seed}_v1"
        or tuple(profile.train_ids) != TRAIN_IDS
        or tuple(profile.eval_ids) != EVAL_IDS
        or tuple(profile.evaluation_steps) != (2, 4)
        or profile.rounds != 4
        or profile.planned_trajectories != PLANNED_PER_SEED
        or profile.evaluation_incomplete_mode != "retain_prefix_v1"
        or len(profile.cell_specs) != len(CELLS)
        or set(specs) != set(CELLS)
        or any(
            specs[cell].arm != ("M1" if arm == "SFT" else arm)
            or specs[cell].eta != eta
            or specs[cell].rounds != (0 if cell == "sft0" else 4)
            or specs[cell].critic_lr != 1e-6
            for cell, (_, eta, arm) in CELLS.items()
        )
        or any(
            dict(profile.cell_actor_learning_rates).get(cell, profile.actor_learning_rate) != lr
            for cell, (lr, _, _) in CELLS.items()
            if lr is not None
        )
    ):
        raise ValueError("readout profile is outside the fixed 1336-trajectory comparison")


def _identity(record):
    identity = normalize_rollout_identity(record.get("rollout_identity", record.get("identity")))
    episode = record.get("episode_id")
    if identity.validate:
        expected = EVAL_IDS.index(episode) if episode in EVAL_IDS else None
    else:
        batch = TRAIN_IDS[(identity.step - 1) * 2 : identity.step * 2]
        # Native Agent-R1 keeps prepared extra_info.index across batches.
        # Membership is per round; sample_index is the full dataset ordinal.
        expected = TRAIN_IDS.index(episode) if episode in batch else None
    if expected is None or identity.sample_index != expected:
        raise ValueError("raw sample_index differs from the fixed episode/batch ordinal")


def read_grid(base, *, expected_profiles=None):
    """Read the current snapshot; default entry validates the sealed root contract.

    Explicit expected_profiles supports isolated CPU evidence tests. Missing or
    invalid cells remain issues and cannot be silently counted as complete.
    """
    base = Path(base).resolve()
    if expected_profiles is None:
        from src.agents import action_pooling_comparison as comparison

        comparison.check_contract(base)
        expected_profiles = comparison.PROFILES
    if set(expected_profiles) != set(SEEDS):
        raise ValueError("comparison must contain exactly seeds42/43")
    for seed in SEEDS:
        _profile(expected_profiles[seed], seed)
    sources, issues, cells, heads_by_seed = {}, [], [], {}
    curves, windows, behaviors, diagnostics, trajectories, failures = [], [], [], [], [], []
    shared_models = []
    for filename in (
        "run_contract.json",
        "comparison_settings.json",
        "shared_sft.json",
        "stage1_completion.json",
        "source_identity.json",
        "plan.md",
    ):
        path = base / filename
        if path.exists():
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for seed in SEEDS:
        seed_base = base / f"seed{seed}"
        try:
            state = _read(seed_base / "state.json", sources)
            if (
                state.get("profile") != expected_profiles[seed].name
                or state.get("seed") != seed
                or state.get("planned_trajectories") != PLANNED_PER_SEED
                or tuple(state.get("train_episode_ids", ())) != TRAIN_IDS
                or tuple(state.get("evaluation_episode_ids", ())) != EVAL_IDS
                or state.get("evaluation_steps") != [2, 4]
            ):
                raise ValueError("seed state identity/grid differs")
        except (OSError, KeyError, TypeError, ValueError) as error:
            issues.append(dict(seed=seed, reason=str(error)))
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
        heads = {}
        for cell_id in CELLS:
            summary = by_id.get(cell_id)
            if summary is None or summary.get("status") not in {
                "COMPLETE",
                "COMPLETE_WITH_EVALUATION_FAILURES",
            }:
                issues.append(dict(seed=seed, cell_id=cell_id, reason="cell is not COMPLETE"))
                continue
            cell = seed_base / cell_id
            try:
                if _read(cell / "summary.json", sources) != summary:
                    raise ValueError("cell summary differs from controller evidence")
                head = summary.get("critic_initial_head_sha256")
                if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{64}", head) is None:
                    raise ValueError("actual initial critic head evidence missing")
                settings = _read(cell / "settings.json", sources)
                if (
                    settings.get("observation_profile") != "history_features_recovery_v2"
                    or settings.get("action_protocol") != "single_tool_portfolio_v1"
                    or settings.get("generation", {}).get("enable_thinking") is not False
                    or not settings.get("models", {}).get("screening")
                ):
                    raise ValueError("cell common model/observation/action protocol differs")
                result = shared._cell(
                    cell,
                    seed,
                    cell_id,
                    summary,
                    sources,
                    retain_prefix=True,
                    cells=CELLS,
                    record_validator=_identity,
                )
                a, b, c, d, rows, failed = result
                curves.extend(a)
                windows.extend(b)
                behaviors.extend(c)
                diagnostics.extend(d)
                trajectories.extend(rows)
                failures.extend(failed)
                heads[cell_id] = head
                shared_models.append(settings["models"]["screening"])
                cells.append(
                    dict(
                        seed=seed,
                        cell_id=cell_id,
                        execution_status="COMPLETE_WITH_EVALUATION_FAILURES"
                        if failed
                        else "COMPLETE",
                        **shared._coverage(rows, 60 if cell_id == "sft0" else 152),
                    )
                )
            except (OSError, KeyError, TypeError, ValueError) as error:
                issues.append(dict(seed=seed, cell_id=cell_id, reason=str(error)))
        if len(set(heads.values())) > 1:
            issues.append(dict(seed=seed, reason="initial critic head differs among five cells"))
        heads_by_seed[str(seed)] = heads
    if shared_models and any(model != shared_models[0] for model in shared_models):
        issues.append(dict(reason="common SFT model differs between cells/seeds"))
    execution_complete = not issues and len(trajectories) == PLANNED and len(cells) == 10
    financial_complete = execution_complete and not failures
    execution_status = (
        ("COMPLETE_WITH_EVALUATION_FAILURES" if failures else "COMPLETE")
        if execution_complete
        else "PARTIAL"
    )
    evaluation = [r for r in trajectories if r["lane"] != "train"]
    training = [r for r in trajectories if r["lane"] == "train"]
    return dict(
        profile=PROFILE,
        status="OBSERVED" if execution_complete else "PARTIAL",
        execution_status=execution_status,
        financial_status="COMPLETE" if financial_complete else "INCOMPLETE",
        performance_pass=False,
        formal_performance_claim=False,
        at_SGT=shared.datetime.now(shared.ZoneInfo("Asia/Singapore")).isoformat(),
        planned_trajectories=PLANNED,
        attempted_trajectories=len(trajectories),
        complete_trajectories=len(trajectories) - len(failures),
        incomplete_trajectories=len(failures),
        missing_trajectories=PLANNED - len(trajectories),
        training_coverage=shared._coverage(training, PLANNED_TRAIN),
        evaluation_coverage=shared._coverage(evaluation, PLANNED_EVAL),
        cells=cells,
        issues=issues,
        source_sha256=sources,
        critic_initial_head_by_seed=heads_by_seed,
        evaluation_curve=curves,
        window_curve=windows,
        paired_comparisons=shared._differences(trajectories, reference_pairs=REFERENCE_PAIRS),
        window_comparisons=shared._differences(
            trajectories,
            by_window=True,
            reference_pairs=REFERENCE_PAIRS,
        ),
        behavior_summary=behaviors,
        failure_summary=failures,
        training_diagnostics=diagnostics,
        financial_outcomes_complete_only=True,
        automatic_selection=False,
        scope="original Train development only; separate seeds, steps, lanes and equal windows",
        sampled_rng_note="SFT0 versus RL steps2/4 is not exact CRN; seed includes policy step.",
        residual_scale_note="std_S already measures eta*S; ratio is std_S/std_U, eta applied once.",
        limitations=[
            "Two seeds and twelve previously used Train development windows are not OOS evidence.",
            "M2-plain retains the same diversity gate and GAE; it is not a pure GRPO baseline.",
            "Same-LR contrasts and the high-LR M1 contrast answer different questions.",
            "Nonzero credit, actor token drift and fit diagnostics do not establish useful credit.",
            "Generated actions and finite-window returns are descriptive, not a profit gate.",
        ],
    )


def write_readout(base, *, allow_partial=True, expected_profiles=None):
    """Publish derived stage2 files only; preserve missing/failed evidence explicitly."""
    base = Path(base).resolve()
    result = read_grid(base, expected_profiles=expected_profiles)
    base.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shared._write_csv(base / f"stage2_{name}.csv", result[name])
    temporary = base / "stage2_readout.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(base / "stage2_readout.json")
    if result["execution_status"] == "PARTIAL" and not allow_partial:
        raise ValueError("fixed stage2 grid is incomplete; PARTIAL readout retained")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    value = write_readout(args.directory, allow_partial=not args.require_complete)
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "status",
                    "execution_status",
                    "financial_status",
                    "attempted_trajectories",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
