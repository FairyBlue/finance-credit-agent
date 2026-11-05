#!/usr/bin/env python3
"""Bridge frozen EST continuations to a small cached head-only update demo.

This is not an on-policy PPO batch or a financial GAE/critic-target builder.
Both EST repetitions are retained; use repetition zero for a single-copy head
training batch. REF outcomes never enter the label map or output records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "runs/m2-onpolicy-credit-replication-20261019-v1"
ALPHA = 0.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")


def write_rows(path: Path, values) -> None:
    with path.open("x") as stream:
        for value in values:
            stream.write(canonical(value) + "\n")


def exact_signal(scores: list[float], order_keys: list[str]) -> list[float]:
    require(len(scores) == len(order_keys) == 4, "four-slot complete group required")
    require(all(math.isfinite(value) for value in scores), "finite EST terminal scores")
    if len(set(order_keys)) == 1:
        return [0.0] * 4
    mean = math.fsum(scores) / 4
    cells = defaultdict(list)
    for key, score in zip(order_keys, scores, strict=True):
        cells[key].append(score)
    cell_means = {key: math.fsum(values) / len(values) for key, values in cells.items()}
    signal = [cell_means[key] - mean for key in order_keys]
    require(abs(math.fsum(signal)) <= 1e-10, "four-slot multiset-centered exact signal")
    return signal


def build(output: Path | str, source: Path | str = DEFAULT_SOURCE) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    names = (
        "continuation_manifest.json",
        "branch_manifest.json",
        "anchor_snapshots.jsonl",
        "head_results.jsonl",
        "trajectory_results.jsonl",
    )
    source_hashes = {name: sha(source / name) for name in names}
    manifest = json.loads((source / names[0]).read_text())
    branch = json.loads((source / names[1]).read_text())
    require(manifest["formal_valid_oos_access"] is False, "Train-only source")
    policy = manifest["policy_snapshot_sha256"]
    snapshots = {row["anchor_id"]: (line, row) for line, row in rows(source / names[2])}
    anchors = {row["anchor_id"]: row for row in branch["anchors"]}
    heads_by_anchor = defaultdict(list)
    for line, row in rows(source / names[3]):
        heads_by_anchor[row["anchor_id"]].append((line, row))
    estimates = {}
    ignored_reference_rows = 0
    for line, row in rows(source / names[4]):
        if row["role"] == "reference":
            ignored_reference_rows += 1
            continue  # Do not inspect or retain any REF outcome/terminal metric.
        require(row["role"] == "estimate", "only EST/REF terminal records expected")
        key = (row["head_key"], row["replicate"])
        require(key not in estimates, "duplicate EST path")
        require(row["policy_snapshot_sha256"] == policy, "EST policy snapshot")
        estimates[key] = (line, row)

    records, statuses, excluded = [], [], []
    for aid, anchor in sorted(anchors.items()):
        hs = sorted(heads_by_anchor[aid], key=lambda item: item[1]["slot_id"])
        require([row["slot_id"] for _, row in hs] == [0, 1, 2, 3], "original four slots")
        reason = None
        if anchor["status"] != "CAPTURED" or anchor["conditional_state_match"] is not True:
            reason = "UNSUPPORTED_ANCHOR"
        elif not all(head["legal_head"] for _, head in hs):
            reason = "PARTIAL_FOUR_SLOT_GROUP"
        else:
            for _, head in hs:
                for rep in (0, 1):
                    found = estimates.get((head["head_key"], rep))
                    if found is None or found[1]["status"] != "COMPLETE":
                        reason = "INCOMPLETE_EST_GROUP"
        for line, head in hs:
            statuses.append(
                {
                    "anchor_id": aid,
                    "slot_id": head["slot_id"],
                    "head_key": head["head_key"],
                    "status": head["status"],
                    "legal_head": head["legal_head"],
                    "supervised_eligible": reason is None,
                    "exclusion_reason": reason,
                    "source_head_line": line,
                }
            )
        if reason:
            excluded.append({"anchor_id": aid, "reason": reason, "slots": 4})
            continue
        snapshot_line, snapshot = snapshots[aid]
        require(snapshot["policy_snapshot_sha256"] == policy, "anchor policy snapshot")
        prompt = snapshot["prompt_token_ids"]
        require(prompt and all(type(token) is int and token >= 0 for token in prompt), "prompt IDs")
        require(
            hashlib.sha256(json.dumps(prompt).encode("ascii")).hexdigest()
            == snapshot["prompt_sha256"],
            "cached prompt identity",
        )
        keys = [canonical(json.loads(head["economic_order_key"])) for _, head in hs]
        for rep in (0, 1):
            sources = [estimates[head["head_key"], rep] for _, head in hs]
            scores = [float(row["Y"]) for _, row in sources]
            signal = exact_signal(scores, keys)
            for (head_line, head), (est_line, estimate), raw, score, key in zip(
                hs, sources, signal, scores, keys, strict=True
            ):
                require(
                    estimate["anchor_id"] == aid and estimate["slot_id"] == head["slot_id"],
                    "EST/head source identity",
                )
                response = head["response_token_ids"]
                require(
                    response and all(type(token) is int and token >= 0 for token in response),
                    "head response token IDs",
                )
                records.append(
                    {
                        "record_id": f"{head['head_key']}/est-r{rep}",
                        "group_id": f"{aid}/est-r{rep}",
                        "anchor_id": aid,
                        "episode_id": snapshot["episode_id"],
                        "parent_id": head["parent_id"],
                        "slot_id": head["slot_id"],
                        "estimate_replicate": rep,
                        "decision_session_index": head["decision_session_index"],
                        "head_key": head["head_key"],
                        "cell_id": head["cell_id"],
                        "economic_order_key": key,
                        "policy_snapshot_sha256": policy,
                        "prompt_token_ids": prompt,
                        "response_token_ids": response,
                        "response_text": head["first_action_text"],
                        "observation_messages": snapshot["observation_messages"],
                        "financial_state_sha256": snapshot["financial_state_sha256"],
                        "prompt_sha256": snapshot["prompt_sha256"],
                        "financial_Y_est": score,
                        "raw_S_exact": raw,
                        "alpha": ALPHA,
                        "alpha_S_exact": ALPHA * raw,
                        "sample_mask": True,
                        "source_head_line": head_line,
                        "source_snapshot_line": snapshot_line,
                        "source_est_line": est_line,
                        "source_est_trajectory_key": estimate["trajectory_key"],
                    }
                )
    require(records, "no complete supervised cached groups")
    require(
        source_hashes == {name: sha(source / name) for name in names}, "source changed during build"
    )
    for name in ("head_records.jsonl", "head_statuses.jsonl", "dataset_manifest.json"):
        require(not (output / name).exists(), f"refusing overwrite: {name}")
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "head_records.jsonl", records)
    write_rows(output / "head_statuses.jsonl", statuses)
    result = {
        "profile": "cached_est_head_only_exact_shrink_demo_v1",
        "status": "BUILT",
        "source_directory": str(source),
        "source_file_sha256": source_hashes,
        "builder_source_sha256": sha(Path(__file__)),
        "model_path": manifest["model_path"],
        "policy_snapshot_sha256": policy,
        "alpha": ALPHA,
        "formula": "raw_S_i = mean_EST_Y_same_economic_order - mean_EST_Y_original_four_slots; "
        "all-same-order gate=0; alpha_S_exact=0.5*raw_S_exact; no eta/whitening",
        "counts": {
            "source_anchors": len(anchors),
            "source_heads": len(statuses),
            "legal_heads": sum(item["legal_head"] for item in statuses),
            "supervised_rows": len(records),
            "supervised_groups": len(records) // 4,
            "supervised_rows_rep0": sum(row["estimate_replicate"] == 0 for row in records),
            "excluded_anchors": len(excluded),
            "EST_paths": len(estimates),
            "ignored_REF_rows": ignored_reference_rows,
            "nonzero_rows_rep0": sum(
                row["estimate_replicate"] == 0 and row["alpha_S_exact"] != 0 for row in records
            ),
        },
        "excluded_anchors": excluded,
        "head_status_counts": dict(Counter(row["status"] for row in statuses)),
        "recommended_estimate_replicate": 0,
        "training_labels_use_REF": False,
        "financial_GAE_or_critic_targets_constructed": False,
        "on_policy_training_batch": False,
        "scope": "Cached conditional head-only raw-S plumbing demo, with partial groups excluded "
        "and listed. Not full-policy/GAE training, method advantage, or financial return evidence. "
        "Use rep0 only to avoid duplicating head policy weight across EST repetitions.",
        "outputs_sha256": {
            name: sha(output / name) for name in ("head_records.jsonl", "head_statuses.jsonl")
        },
    }
    write_json(output / "dataset_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    result = build(args.output, args.source)
    print(json.dumps({"status": result["status"], **result["counts"]}))


if __name__ == "__main__":
    main()
