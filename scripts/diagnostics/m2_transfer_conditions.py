"""Four diagnostic prompt conditions over the exact frozen 4B features environment."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATH = ROOT / "scripts/diagnostics/m2_decision_conditions.py"
LEGACY_SHA = "39c5fda47d09bfb63cd41b1b5b5ea6a12721a6429bafb65de66e7aea1ee3c87f"
REFERENCE_PATH = ROOT / "scripts/diagnostics/m2_comprehension_probe.py"
REFERENCE_SHA = "991b1d536cb1555f10278732731533999d512c17bff39f87f63c5773eca6dede"
OBJECTIVE_PATH = ROOT / "scripts/diagnostics/m2_hold_probe.py"
OBJECTIVE_SHA = "7d610d7ff5343fac95b023470d9af699615ca93572d8bf7b62950e36d1a7153a"
CONDITIONS = ("features", "features_defined", "features_objective", "features_defined_objective")


def frozen_literals(path: Path, expected: str, names: tuple[str, ...]) -> dict[str, str]:
    """Read only the registered text constants, never the quiz execution instructions."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Frozen text source changed: {path.name}")
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                value = ast.literal_eval(node.value)
                if type(value) is not str:
                    raise RuntimeError("Expected literal string")
                result[target.id] = value
    if set(result) != set(names):
        raise RuntimeError("Missing registered text")
    return result


REFERENCES = frozen_literals(
    REFERENCE_PATH, REFERENCE_SHA, ("INTERFACE_REFERENCE", "METRIC_REFERENCE")
)
OBJECTIVE = frozen_literals(OBJECTIVE_PATH, OBJECTIVE_SHA, ("OBJECTIVE",))["OBJECTIVE"]
if hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest() != LEGACY_SHA:
    raise RuntimeError("Frozen features implementation changed")
_spec = importlib.util.spec_from_file_location("transfer_frozen_features", LEGACY_PATH)
assert _spec and _spec.loader
LEGACY = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEGACY)
render_prompt = LEGACY.render_prompt
reset_with_prefix = LEGACY.reset_with_prefix
EPISODES = LEGACY.EPISODES


def validate_condition(condition: object, start_session: object = 4) -> str:
    if type(condition) is not str or condition not in CONDITIONS:
        raise ValueError("Unknown transfer condition")
    if type(start_session) is not int or start_session != 4:
        raise ValueError("Transfer diagnostic starts at integer session 4")
    return condition


def prompt_addition(condition: str) -> str:
    validate_condition(condition)
    parts = []
    if condition in ("features_defined", "features_defined_objective"):
        parts.append(
            "Reference definitions:\n"
            + REFERENCES["INTERFACE_REFERENCE"]
            + "\n"
            + REFERENCES["METRIC_REFERENCE"]
        )
    if condition in ("features_objective", "features_defined_objective"):
        parts.append(OBJECTIVE)
    return "\n\n".join(parts)


def augment_task(prompt: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    addition = prompt_addition(condition)
    if not prompt or any(message.get("role") == "system" for message in prompt):
        raise ValueError("Expected original nonempty task without a system message")
    result = deepcopy(prompt)
    if addition:
        result.insert(0, {"role": "system", "content": addition})
    return result


def create_env(
    rows: list[dict[str, Any]], index: int, condition: str, start_session: int = 4
) -> Any:
    validate_condition(condition, start_session)
    if len(rows) != 2 or type(index) is not int or index not in (0, 1):
        raise ValueError("Expected the two original Train rows")
    copied = deepcopy(rows)
    copied[index]["prompt"] = augment_task(copied[index]["prompt"], condition)
    # The original features condition hides memory and history-count fields.
    # Only its task prefix changes; actual feature computation and all tools stay exact.
    env = LEGACY.create_env(copied, index, "features", 4)
    env.transfer_condition = condition
    return env
