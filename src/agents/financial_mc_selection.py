"""Prefix-only, independent sampling decisions for MC state baselines.

This fixed pilot heuristic allocates evaluation, never pools rewards across
states. No parent action, terminal outcome or auxiliary outcome is an input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

PROFILE = "financial_mc_prefix_selection_v1"
BASE_PROBABILITY = 0.12
UNIFORM_MIXTURE = 0.25
TARGET_PROFILE = "financial_target_report_parent_quota_v2"
QUOTA_PATH_ENV = "FCA_FINANCIAL_MC_QUOTA_MANIFEST"
QUOTA_SHA_ENV = "FCA_FINANCIAL_MC_QUOTA_SHA256"


def make_parent_quota(task_ids, repetitions, seed, max_anchors, data_path, data_sha256):
    """Draw parent IDs before any rollout, using an independent named RNG."""
    if len(set(task_ids)) != len(task_ids) or not task_ids:
        raise ValueError("unique nonempty task IDs required")
    if repetitions != 4 or not 0 <= max_anchors <= 8:
        raise ValueError("target quota requires four replicas and at most eight parents")
    parents = [f"{task}/rep{replica}" for task in task_ids for replica in range(repetitions)]
    if len(parents) > 32:
        raise ValueError("target quota supports at most 32 parents per batch")
    payload = json.dumps([TARGET_PROFILE, "parent_quota_rng", seed, parents], separators=(",", ":"))
    rng_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    count = min(max_anchors, len(parents))
    selected = set(random.Random(rng_seed).sample(parents, count))
    return dict(
        profile=TARGET_PROFILE,
        seed=seed,
        task_ids=list(task_ids),
        repetitions=repetitions,
        parent_ids=parents,
        selected_parent_ids=[key for key in parents if key in selected],
        parent_quota_size=count,
        parent_population_size=len(parents),
        parent_quota_probability=count / len(parents),
        parent_quota_rng_seed=rng_seed,
        parent_quota_rng_namespace="parent_quota_rng",
        independent_selection_rng=True,
        selected_before_any_parent_action=True,
        parent_outcome_used=False,
        auxiliary_outcome_used=False,
        data_path=str(Path(data_path).resolve()),
        data_sha256=data_sha256,
    )


def _public_json(message):
    raw = message.get("content", "")
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if raw.startswith("<tool_call>") and raw.endswith("</tool_call>"):
        raw = raw[len("<tool_call>") : -len("</tool_call>")].strip()
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def target_report_features(messages, calls_used, max_calls):
    """Use only the public condition and already returned tool observations."""
    condition = None
    for message in messages:
        if message.get("role") == "user":
            value = _public_json(message)
            if isinstance((value or {}).get("condition"), dict):
                condition = value["condition"]
                break
    company = str((condition or {}).get("company", "")).strip()
    year = str((condition or {}).get("report_year", "")).strip()
    valid = bool(company and re.fullmatch(r"(?:19|20)\d{2}", year))
    pattern = (
        re.compile(
            r"^\s*" + re.escape(company) + r"\s+annual\s+report\s+" + re.escape(year) + r"(?:\D|$)",
            re.I,
        )
        if valid
        else None
    )
    previous_action = None
    target_reads, calculations_returned = 0, 0
    for message in messages:
        role = message.get("role")
        value = _public_json(message)
        if role == "assistant":
            previous_action = (value or {}).get("name")
        elif role == "tool":
            if previous_action == "calculate":
                calculations_returned += 1
            if (
                previous_action == "read_document"
                and value
                and value.get("ok") is True
                and pattern
                and pattern.match(str(value.get("title", "")))
            ):
                target_reads += 1
            previous_action = None
    return dict(
        stage="target_report_before_calculation",
        public_condition_available=valid,
        public_target_company=company,
        public_target_report_year=year,
        target_report_reads=target_reads,
        calculation_returns=calculations_returned,
        remaining_calls=max(0, max_calls - calls_used),
    )


def public_features(messages, calls_used, max_calls):
    previous_tools, report_reads, search_candidates = [], 0, 0
    last_feedback = None
    for message in messages:
        if message.get("role") not in ("assistant", "tool"):
            continue
        try:
            value = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        if message["role"] == "assistant":
            if isinstance(value.get("name"), str):
                previous_tools.append(value["name"])
            continue
        last_feedback = value
        if value.get("ok") and "annual report" in str(value.get("title", "")).lower():
            report_reads += 1
        if isinstance(value.get("results"), list):
            search_candidates = max(search_candidates, len(value["results"]))
    calculated = "calculate" in previous_tools
    solved = "solve" in previous_tools
    failed_feedback = bool(last_feedback and last_feedback.get("ok") is False)
    stage = (
        "repair"
        if failed_feedback
        else "post_solve"
        if solved
        else "post_calculation"
        if calculated
        else "evidence_to_calculation"
        if report_reads
        else "information_acquisition"
    )
    weights = dict(
        repair=2.5,
        post_solve=0.4,
        post_calculation=1.0,
        evidence_to_calculation=2.0,
        information_acquisition=0.2,
    )
    ambiguity = min(2.0, max(0, search_candidates - 1) / 4)
    weight = weights[stage] * (1 + 0.25 * ambiguity if report_reads else 1)
    remaining = max_calls - calls_used
    # A public remaining-call estimate, not a suffix observed after the choice.
    relative_cost = max(0.25, remaining / max_calls)
    return dict(
        stage=stage,
        report_reads=report_reads,
        prior_calculation=calculated,
        prior_solve=solved,
        prior_feedback_failed=failed_feedback,
        visible_search_candidates=search_candidates,
        ambiguity=ambiguity,
        remaining_calls=remaining,
        relative_suffix_cost=relative_cost,
        priority_score=weight / relative_cost,
    )


class PrefixSelector:
    def __init__(
        self,
        policy,
        seed,
        max_anchors,
        token_budget,
        max_context,
        max_calls=10,
        *,
        quota_manifest=None,
        quota_sha256=None,
    ):
        if policy not in ("uniform", "priority", "target_quota"):
            raise ValueError("MC prefix policy must be uniform, priority or target_quota")
        if max_anchors < 0 or token_budget < 0 or max_context <= 0 or max_calls <= 0:
            raise ValueError("nonnegative MC limits and positive context/call bounds")
        self.policy, self.seed = policy, seed
        self.max_anchors, self.token_budget = max_anchors, token_budget
        self.max_context, self.max_calls = max_context, max_calls
        self.selected, self.reserved = 0, 0
        self.selected_trajectories = set()
        if policy == "target_quota":
            self._load_parent_quota(quota_manifest, quota_sha256)

    def _load_parent_quota(self, quota_manifest, quota_sha256):
        path = quota_manifest or os.environ.get(QUOTA_PATH_ENV)
        expected = quota_sha256 or os.environ.get(QUOTA_SHA_ENV)
        if not path or not expected:
            raise ValueError("target quota requires a frozen manifest path and SHA256")
        path = Path(path).resolve()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError("target quota manifest SHA256 mismatch")
        manifest = json.loads(raw)
        recomputed = make_parent_quota(
            manifest["task_ids"],
            manifest["repetitions"],
            manifest["seed"],
            manifest["parent_quota_size"],
            manifest["data_path"],
            manifest["data_sha256"],
        )
        if manifest != recomputed or manifest["seed"] != self.seed:
            raise ValueError("target quota is not the declared independent seeded draw")
        if manifest["parent_quota_size"] > self.max_anchors:
            raise ValueError("target parent quota exceeds selector anchor limit")
        if (
            hashlib.sha256(Path(manifest["data_path"]).read_bytes()).hexdigest()
            != (manifest["data_sha256"])
        ):
            raise ValueError("target quota task data SHA256 mismatch")
        self.quota_manifest = manifest
        self.quota_path, self.quota_sha256 = str(path), actual
        self.eligible_parents = set(manifest["selected_parent_ids"])
        self.all_parents = set(manifest["parent_ids"])
        self.considered_trajectories = set()

    def _consider_target(self, request_key, trajectory_key, messages, calls_used):
        if trajectory_key not in self.all_parents:
            raise ValueError("parent identity absent from frozen quota population")
        features = target_report_features(messages, calls_used, self.max_calls)
        reserve = 2 * max(0, self.max_calls - calls_used) * self.max_context
        reason = None
        eligible = trajectory_key in self.eligible_parents
        if not eligible:
            reason = "PARENT_NOT_IN_PRE_ROLLOUT_QUOTA"
        elif calls_used >= self.max_calls:
            reason = "NO_REMAINING_CALLS"
        elif trajectory_key in self.selected_trajectories:
            reason = "ONE_ANCHOR_PER_PARENT"
        elif trajectory_key in self.considered_trajectories:
            reason = "FIRST_ELIGIBLE_PREFIX_ALREADY_CONSIDERED"
        elif not features["public_condition_available"]:
            reason = "PUBLIC_TARGET_CONDITION_MISSING"
        elif not features["target_report_reads"]:
            reason = "TARGET_REPORT_NOT_READ"
        elif features["calculation_returns"]:
            reason = "CALCULATION_ALREADY_RETURNED"
        else:
            self.considered_trajectories.add(trajectory_key)
            if self.selected >= self.max_anchors:
                reason = "ANCHOR_CAP"
            elif self.reserved + reserve > self.token_budget:
                reason = "PAIR_BUDGET_RESERVATION"
        selected = reason is None
        if selected:
            self.selected += 1
            self.reserved += reserve
            self.selected_trajectories.add(trajectory_key)
        nominal = self.quota_manifest["parent_quota_probability"]
        return dict(
            request_key=request_key,
            parent_request_key=request_key,
            trajectory_key=trajectory_key,
            selected=selected,
            selection_probability=nominal if selected else 0.0,
            nominal_probability=nominal,
            conditional_stage_selection_probability=1.0,
            selection_policy=self.policy,
            selection_profile=TARGET_PROFILE,
            selection_seed=self.seed,
            selection_draw=None,
            pre_action_selection=True,
            parent_action_used=False,
            parent_outcome_used=False,
            auxiliary_outcome_used=False,
            observed_calls_used=calls_used,
            score_features=features,
            block_reason=reason,
            required_pair_reservation=reserve,
            reserved_auxiliary_tokens=self.reserved,
            independent_selection_rng=True,
            parent_quota_eligible=eligible,
            parent_quota_size=self.quota_manifest["parent_quota_size"],
            parent_population_size=self.quota_manifest["parent_population_size"],
            parent_quota_probability=nominal,
            quota_manifest_path=self.quota_path,
            quota_manifest_sha256=self.quota_sha256,
            parent_quota_selected_before_rollout=True,
        )

    def consider(self, request_key, trajectory_key, messages, calls_used):
        if self.policy == "target_quota":
            return self._consider_target(request_key, trajectory_key, messages, calls_used)
        features = public_features(messages, calls_used, self.max_calls)
        score = features["priority_score"]
        nominal = (
            BASE_PROBABILITY
            if self.policy == "uniform"
            else min(0.6, BASE_PROBABILITY * (UNIFORM_MIXTURE + (1 - UNIFORM_MIXTURE) * score))
        )
        reserve = 2 * max(0, self.max_calls - calls_used) * self.max_context
        reason = None
        if calls_used >= self.max_calls:
            reason = "NO_REMAINING_CALLS"
        elif trajectory_key in self.selected_trajectories:
            reason = "ONE_ANCHOR_PER_PARENT"
        elif self.selected >= self.max_anchors:
            reason = "ANCHOR_CAP"
        elif self.reserved + reserve > self.token_budget:
            reason = "PAIR_BUDGET_RESERVATION"
        probability = 0.0 if reason else nominal
        raw = json.dumps([PROFILE, self.seed, request_key], separators=(",", ":"))
        draw = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") / 2**64
        selected = reason is None and draw < probability
        if selected:
            self.selected += 1
            self.reserved += reserve
            self.selected_trajectories.add(trajectory_key)
        return dict(
            request_key=request_key,
            parent_request_key=request_key,
            trajectory_key=trajectory_key,
            selected=selected,
            selection_probability=probability,
            nominal_probability=nominal,
            selection_policy=self.policy,
            selection_profile=PROFILE,
            selection_seed=self.seed,
            selection_draw=draw,
            pre_action_selection=True,
            parent_action_used=False,
            parent_outcome_used=False,
            auxiliary_outcome_used=False,
            observed_calls_used=calls_used,
            score_features=features,
            block_reason=reason,
            required_pair_reservation=reserve,
            reserved_auxiliary_tokens=self.reserved,
            uniform_mixture=UNIFORM_MIXTURE,
            base_probability=BASE_PROBABILITY,
            independent_selection_rng=True,
        )


def validate_probability(value):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1
