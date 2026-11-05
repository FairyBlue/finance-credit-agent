"""V2 target-stage collector wrapper with a frozen pre-rollout parent quota.

The original collector remains unchanged. Its explicit target_quota selector
loads a path/SHA configuration scoped to this invocation; no function is patched.
This module starts GPU work only without --prepare-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from src.agents.financial_mc_selection import QUOTA_PATH_ENV, QUOTA_SHA_ENV, make_parent_quota


def prepare_quota(args):
    data = args.data.resolve()
    tasks = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if not tasks or any(task.get("split") != "train" for task in tasks):
        raise ValueError("target-stage wrapper accepts an explicit Train-only batch")
    if args.repetitions != 4 or args.temperature != 1.0:
        raise ValueError("target-stage collection uses four native temperature-one parents")
    if args.token_budget <= 0 or args.mc_selection_budget < 0:
        raise ValueError("positive parent allowance and nonnegative MC allowance required")
    if args.max_context_tokens not in (8192, 16384) or not 1 <= args.max_response_tokens <= 512:
        raise ValueError("supported context and response bounds required")
    output = args.output.resolve()
    if (output / "collection_started.json").exists():
        raise ValueError("fresh collector output required")
    selection_seed = args.seed if args.mc_selection_seed is None else args.mc_selection_seed
    manifest = make_parent_quota(
        [task["task_id"] for task in tasks],
        args.repetitions,
        selection_seed,
        args.mc_max_anchors,
        data,
        hashlib.sha256(data.read_bytes()).hexdigest(),
    )
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "parent_quota_manifest.json"
    if path.exists() and path.read_text() != raw:
        raise ValueError("cannot replace the frozen pre-rollout quota")
    if not path.exists():
        path.write_text(raw)
    return path, hashlib.sha256(raw.encode()).hexdigest(), manifest


@contextmanager
def quota_environment(path, digest):
    old = {key: os.environ.get(key) for key in (QUOTA_PATH_ENV, QUOTA_SHA_ENV)}
    os.environ[QUOTA_PATH_ENV], os.environ[QUOTA_SHA_ENV] = str(path), digest
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(args):
    path, digest, manifest = prepare_quota(args)
    print(
        json.dumps(
            dict(
                stage="TARGET_QUOTA_FROZEN",
                quota_manifest=str(path),
                quota_sha256=digest,
                selected_parent_ids=manifest["selected_parent_ids"],
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return dict(status="PREPARED_ONLY", quota_manifest=str(path), quota_sha256=digest)
    from src.agents.mandate_repair_collect import run as collect

    options = argparse.Namespace(
        data=args.data,
        model=None,
        base_model=args.base_model,
        adapter=args.adapter,
        output=args.output,
        seed=args.seed,
        temperature=1.0,
        tasks_limit=None,
        max_groups=None,
        repetitions=4,
        max_response_tokens=args.max_response_tokens,
        max_context_tokens=args.max_context_tokens,
        token_budget=args.token_budget,
        decode_mode="native",
        arm="rloo",
        aux_token_budget=0,
        local_credit_target="reward",
        engine_mode="eager",
        environment="financial_evidence",
        diagnostic_mode="standard",
        max_auxiliary_parents=None,
        mc_selection="target_quota",
        mc_selection_seed=manifest["seed"],
        mc_max_anchors=args.mc_max_anchors,
        mc_selection_budget=args.mc_selection_budget,
    )
    with quota_environment(path, digest):
        return collect(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--mc-selection-budget", "--mc-budget", type=int, required=True)
    parser.add_argument("--mc-max-anchors", type=int, default=8)
    parser.add_argument("--mc-selection-seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    result = run(parser.parse_args())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"results", "groups"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
