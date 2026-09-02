"""Process-local JSONL evidence for finance step-credit allocation."""

from __future__ import annotations

import json
import math
import operator
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

from .rollout_seed import RolloutIdentity, normalize_rollout_identity

STEP_CREDIT_EVIDENCE_DIR_ENV = "FINANCE_STEP_CREDIT_EVIDENCE_DIR"
STEP_CREDIT_EVIDENCE_SCHEMA_VERSION = 1
STEP_CREDIT_EVIDENCE_SCHEME = "finance-step-credit-evidence-v1"

_EVIDENCE_FIELDS = frozenset(
    {
        "actions",
        "arm_id",
        "base_seed",
        "complete",
        "conservation_error",
        "credit_sum",
        "credits",
        "mechanism",
        "output_semantics",
        "pid",
        "rollout_n",
        "sample_index",
        "schema_version",
        "scheme",
        "step",
        "step_count",
        "trajectory_advantage",
        "validate",
    }
)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be a non-negative integer") from error
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _evidence_root_from_environment() -> Path | None:
    configured = os.environ.get(STEP_CREDIT_EVIDENCE_DIR_ENV)
    if not configured:
        return None
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ValueError("step-credit evidence directory must be absolute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def record_step_credit_evidence(
    *,
    identity: RolloutIdentity,
    base_seed: int,
    mechanism: str,
    output_semantics: str,
    trajectory_advantage: float,
    credits: list[float],
    actions: list[str],
) -> bool:
    """Append one completed trajectory record when launcher evidence is enabled."""

    root = _evidence_root_from_environment()
    if root is None:
        return False
    if not mechanism or not output_semantics:
        raise ValueError("step-credit mechanism and output semantics must be non-empty")
    normalized_advantage = _finite_float(
        trajectory_advantage, name="trajectory advantage"
    )
    normalized_credits = [
        _finite_float(value, name=f"credit[{index}]")
        for index, value in enumerate(credits)
    ]
    if not normalized_credits or len(normalized_credits) != len(actions):
        raise ValueError("step-credit evidence requires one action per non-empty credit")
    if any(not isinstance(action, str) or not action for action in actions):
        raise ValueError("step-credit evidence actions must be non-empty strings")
    credit_sum = sum(normalized_credits)
    if credit_sum != normalized_advantage:
        raise RuntimeError("step-credit evidence does not conserve the trajectory signal")
    record = {
        "actions": list(actions),
        "arm_id": "A3",
        "base_seed": _nonnegative_integer(base_seed, name="base seed"),
        "complete": True,
        "conservation_error": credit_sum - normalized_advantage,
        "credit_sum": credit_sum,
        "credits": normalized_credits,
        "mechanism": mechanism,
        "output_semantics": output_semantics,
        "pid": os.getpid(),
        "rollout_n": identity.rollout_n,
        "sample_index": identity.sample_index,
        "schema_version": STEP_CREDIT_EVIDENCE_SCHEMA_VERSION,
        "scheme": STEP_CREDIT_EVIDENCE_SCHEME,
        "step": identity.step,
        "step_count": len(normalized_credits),
        "trajectory_advantage": normalized_advantage,
        "validate": identity.validate,
    }
    path = root / f"step-credit-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return True


def summarize_step_credit_evidence(
    evidence_root: str | Path,
    *,
    expected_mechanism: str,
    expected_output_semantics: str,
    expected_base_seed: int,
) -> dict[str, object]:
    """Validate all trajectory records and return a smoke-ready summary."""

    root = Path(evidence_root)
    records: list[dict[str, Any]] = []
    worker_paths = sorted(root.glob("*.jsonl")) if root.is_dir() else []
    identities: set[tuple[int, int, int, bool]] = set()
    for path in worker_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, dict):
                raise RuntimeError(f"invalid step-credit evidence in {path}")
            fields = set(document)
            if fields != _EVIDENCE_FIELDS:
                missing = sorted(_EVIDENCE_FIELDS - fields)
                unexpected = sorted(fields - _EVIDENCE_FIELDS)
                raise RuntimeError(
                    f"step-credit evidence fields changed in {path} "
                    f"(missing={missing}, unexpected={unexpected})"
                )
            if document["schema_version"] != STEP_CREDIT_EVIDENCE_SCHEMA_VERSION:
                raise RuntimeError(f"step-credit evidence schema changed in {path}")
            if document["scheme"] != STEP_CREDIT_EVIDENCE_SCHEME:
                raise RuntimeError(f"step-credit evidence scheme changed in {path}")
            if document["arm_id"] != "A3":
                raise RuntimeError(f"step-credit evidence arm mismatch in {path}")
            if _nonnegative_integer(document["base_seed"], name="base seed") != (
                _nonnegative_integer(expected_base_seed, name="expected base seed")
            ):
                raise RuntimeError(f"step-credit evidence base seed mismatch in {path}")
            if document["complete"] is not True:
                raise RuntimeError(f"step-credit evidence is not complete in {path}")
            if document["mechanism"] != expected_mechanism:
                raise RuntimeError(f"step-credit mechanism mismatch in {path}")
            if document["output_semantics"] != expected_output_semantics:
                raise RuntimeError(f"step-credit output semantics mismatch in {path}")

            identity = normalize_rollout_identity(
                {
                    "step": document["step"],
                    "sample_index": document["sample_index"],
                    "rollout_n": document["rollout_n"],
                    "validate": document["validate"],
                }
            )
            identity_key = (
                identity.step,
                identity.sample_index,
                identity.rollout_n,
                identity.validate,
            )
            if identity_key in identities:
                raise RuntimeError(f"duplicate step-credit trajectory identity in {path}")
            identities.add(identity_key)

            advantage = _finite_float(
                document["trajectory_advantage"], name="trajectory advantage"
            )
            raw_credits = document["credits"]
            raw_actions = document["actions"]
            if not isinstance(raw_credits, list) or not isinstance(raw_actions, list):
                raise RuntimeError(f"invalid step-credit vector evidence in {path}")
            credits = [
                _finite_float(value, name=f"credit[{index}]")
                for index, value in enumerate(raw_credits)
            ]
            if (
                not credits
                or document["step_count"] != len(credits)
                or len(raw_actions) != len(credits)
            ):
                raise RuntimeError(f"step-credit count mismatch in {path}")
            if any(not isinstance(action, str) or not action for action in raw_actions):
                raise RuntimeError(f"invalid step-credit action evidence in {path}")
            credit_sum = sum(credits)
            recorded_sum = _finite_float(document["credit_sum"], name="credit sum")
            conservation_error = _finite_float(
                document["conservation_error"], name="conservation error"
            )
            if (
                credit_sum != advantage
                or recorded_sum != advantage
                or conservation_error != 0.0
            ):
                raise RuntimeError(f"step-credit conservation failed in {path}")
            if expected_mechanism == "m1_uniform":
                share = advantage / len(credits)
                expected_credits = [share] * len(credits)
                expected_credits[-1] = advantage - sum(expected_credits[:-1])
                if credits != expected_credits:
                    raise RuntimeError(f"M1 evidence is not uniform in {path}")
            records.append(document)

    training = [record for record in records if record["validate"] is False]
    validation = [record for record in records if record["validate"] is True]
    training_nonzero = [
        record for record in training if float(record["trajectory_advantage"]) != 0.0
    ]
    validation_nonzero = [
        record for record in validation if float(record["trajectory_advantage"]) != 0.0
    ]
    sorted_records = sorted(
        records,
        key=lambda record: (
            int(record["step"]),
            int(record["sample_index"]),
            int(record["rollout_n"]),
            bool(record["validate"]),
        ),
    )
    canonical_records = [
        {key: value for key, value in record.items() if key != "pid"}
        for record in sorted_records
    ]
    canonical_sha256 = sha256(
        json.dumps(canonical_records, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    sorted_identities = sorted(identities)
    trajectory_identity_sha256 = sha256(
        json.dumps(sorted_identities, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    representative = (
        training_nonzero[0] if training_nonzero else (records[0] if records else None)
    )
    advantages = [float(record["trajectory_advantage"]) for record in records]
    return {
        "schema_version": STEP_CREDIT_EVIDENCE_SCHEMA_VERSION,
        "scheme": STEP_CREDIT_EVIDENCE_SCHEME,
        "base_seed": expected_base_seed if records else None,
        "mechanism": expected_mechanism,
        "output_semantics": expected_output_semantics,
        "trajectory_count": len(records),
        "training_trajectory_count": len(training),
        "validation_trajectory_count": len(validation),
        "training_nonzero_trajectory_advantage_count": len(training_nonzero),
        "validation_nonzero_trajectory_advantage_count": len(validation_nonzero),
        "completed_trajectory_count": len(records),
        "conserved_trajectory_count": len(records),
        "maximum_absolute_conservation_error": 0.0 if records else None,
        "trajectory_advantage_minimum": min(advantages) if advantages else None,
        "trajectory_advantage_maximum": max(advantages) if advantages else None,
        "total_step_count": sum(int(record["step_count"]) for record in records),
        "trajectory_identity_sha256": trajectory_identity_sha256,
        "canonical_sha256": canonical_sha256,
        "worker_files": len(worker_paths),
        "representative_trajectory": representative,
    }
