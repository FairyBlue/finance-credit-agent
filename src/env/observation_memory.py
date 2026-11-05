"""Bounded, exact tool-result memory for the optional trading observation profile."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

RECOVERY_OBSERVATION_PROFILE = "history_features_recovery_v1"
RECOVERY_V2_OBSERVATION_PROFILE = "history_features_recovery_v2"
RECOVERY_OBSERVATION_PROFILES = frozenset({
    RECOVERY_OBSERVATION_PROFILE, RECOVERY_V2_OBSERVATION_PROFILE,
})
OBSERVATION_PROFILES = (
    "legacy", "history_memory_v1", "history_features_v1", RECOVERY_OBSERVATION_PROFILE,
    RECOVERY_V2_OBSERVATION_PROFILE,
)
MEMORY_OBSERVATION_PROFILES = frozenset({
    "history_memory_v1", "history_features_v1", *RECOVERY_OBSERVATION_PROFILES,
})
FEATURE_OBSERVATION_PROFILES = frozenset({
    "history_features_v1", *RECOVERY_OBSERVATION_PROFILES,
})


def validate_observation_profile(value: object) -> str:
    if type(value) is not str or value not in OBSERVATION_PROFILES:
        raise ValueError(f"observation_profile must be one of {OBSERVATION_PROFILES}")
    return value


class ToolResultMemory:
    """Keep recent successful results, with explicit capacity and eviction counts.

    Repeated calls at the same as-of time refresh their exact-argument entry.
    A later as-of time is a distinct observation, never an implicit recompute.
    """

    CAPACITY = 20
    RETRIEVAL_CAPACITY = 2
    BYTE_LIMIT = 4096
    COMPUTE_COLUMNS = (
        "symbol", "metric", "lookback_requested", "lookback_used", "as_of", "value",
    )

    def __init__(self) -> None:
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.evicted_count = 0
        self.refreshed_count = 0

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                          separators=(",", ":"))

    def remember(
        self, tool: str, arguments: Mapping[str, Any], result: Mapping[str, Any],
    ) -> None:
        if tool not in {"compute", "retrieve"}:
            raise ValueError("only successful information-tool results enter memory")
        # Validate/copy before altering the bank. Invalid/non-finite tool output
        # must not leave a successful result behind when the environment rejects it.
        copied_result = json.loads(self._json(result))
        normalized = deepcopy(dict(arguments))
        if normalized.get("symbol") is not None:
            normalized["symbol"] = normalized["symbol"].strip().upper()
        if tool == "retrieve":
            normalized["query"] = normalized["query"].casefold()
            normalized.setdefault("limit", 3)
            if normalized.get("symbol") is None:
                normalized.pop("symbol", None)
        key = self._json([tool, copied_result["as_of"], normalized])
        display_arguments = deepcopy(normalized)
        if tool == "retrieve":
            query = display_arguments["query"].encode("unicode_escape").decode("ascii")
            display_arguments["query"] = query[:80]
            display_arguments["query_truncated"] = len(query) > 80
            if "symbol" in display_arguments:
                symbol = display_arguments["symbol"].encode("unicode_escape").decode("ascii")
                display_arguments["symbol"] = symbol[:80]
                display_arguments["symbol_truncated"] = len(symbol) > 80
        entry = {"tool": tool, "arguments": display_arguments, "result": copied_result}
        previous = self._entries.copy(), self.evicted_count, self.refreshed_count
        try:
            self._insert(key, entry)
        except (KeyError, TypeError, ValueError):
            self._entries, self.evicted_count, self.refreshed_count = previous
            raise

    def _insert(self, key: str, entry: dict[str, Any]) -> None:
        if key in self._entries:
            del self._entries[key]
            self.refreshed_count += 1
        self._entries[key] = entry
        while (
            sum(item["tool"] == "retrieve" for item in self._entries.values())
            > self.RETRIEVAL_CAPACITY
        ):
            oldest_retrieval = next(
                name for name, item in self._entries.items() if item["tool"] == "retrieve"
            )
            del self._entries[oldest_retrieval]
            self.evicted_count += 1
        while (
            len(self._entries) > self.CAPACITY
            or len(self._json(self.render()).encode("ascii")) > self.BYTE_LIMIT
        ):
            if len(self._entries) == 1:
                raise ValueError("one tool result exceeds the registered memory byte budget")
            self._entries.popitem(last=False)
            self.evicted_count += 1

    def render(self, current_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        entries = list(self._entries.values())
        duplicate = bool(entries and current_result is not None and
                         entries[-1]["result"] == current_result)
        visible = entries[:-1] if duplicate else entries
        computations = []
        retrievals = []
        for entry in visible:
            args, result = entry["arguments"], entry["result"]
            if entry["tool"] == "compute":
                computations.append([
                    result["symbol"], result["metric"], args["lookback"],
                    result["lookback_used"], result["as_of"], result["value"],
                ])
            else:
                retrievals.append({"arguments": deepcopy(args), "result": deepcopy(result)})
        return {
            "compute_columns": list(self.COMPUTE_COLUMNS),
            "compute_rows": computations,
            "retrievals": retrievals,
            "retained_count": len(entries),
            "evicted_count": self.evicted_count,
            "refreshed_count": self.refreshed_count,
            "capacity": self.CAPACITY,
            "retrieval_capacity": self.RETRIEVAL_CAPACITY,
            "byte_limit": self.BYTE_LIMIT,
            "current_result_in_last_tool_result": duplicate,
        }
