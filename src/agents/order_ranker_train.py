"""Fit one fixed small CPU MLP to Train-only, same-state candidate Q differences."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from src.agents.order_ranker import FEATURE_NAMES, PROFILE, predict_features

EPOCHS = 800
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN_SIZES = (64, 64)


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_training_rows(directory):
    summary = json.loads((directory / "collection_summary.json").read_text())
    if summary.get("training_split") != "train":
        raise ValueError("ranker only accepts declared Train branch labels")
    if summary.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("branch feature contract changed")
    groups = defaultdict(list)
    with (directory / "candidate_rows.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row["episode_id"].startswith("formal-train-"):
                raise ValueError("non-Train row in scoring data")
            if row.get("training_supported"):
                values = row["features"]
                if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in values):
                    raise ValueError("invalid candidate features")
                if not math.isfinite(row["mean_Q"]):
                    raise ValueError("supported candidate needs finite mean Q")
                groups[row["anchor_id"]].append(row)
    groups = {
        key: sorted(rows, key=lambda row: row["candidate_id"])
        for key, rows in groups.items()
        if len(rows) >= 2
    }
    rows, pairs, targets, weights = [], [], [], []
    for group in groups.values():
        offset = len(rows)
        rows.extend(group)
        pair_count = len(group) * (len(group) - 1) // 2
        for left, right in combinations(range(len(group)), 2):
            pairs.append((offset + left, offset + right))
            targets.append(group[left]["mean_Q"] - group[right]["mean_Q"])
            weights.append(1 / (len(groups) * pair_count))
    return summary, groups, rows, pairs, targets, weights


def run(args):
    directory, output = args.collection.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_summary.json").exists() or (output / "ranker.json").exists():
        raise FileExistsError("fresh ranker output required, no implicit retraining")
    source, groups, rows, pairs, targets, weights = load_training_rows(directory)
    started = time.monotonic()
    summary = dict(
        profile=PROFILE,
        status="STARTING",
        training_split="train",
        collection_path=str(directory),
        frozen_LLM_model_path=source["model_path"],
        supported_anchors=len(groups),
        candidate_rows=len(rows),
        pair_rows=len(pairs),
        seed=args.seed,
        feature_names=list(FEATURE_NAMES),
        hidden_sizes=list(HIDDEN_SIZES),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optimizer="AdamW on CPU, one fixed configuration",
        objective="weighted MSE of same-state predicted score differences and mean-Q differences",
        group_weighting="each supported state equally weighted; pairs equal within state",
        held_out_metric=False,
        LLM_optimizer_updates=0,
        uses_future_at_inference=False,
        valid_or_final_oos_access=False,
        model_path=str(output / "ranker.json"),
    )
    save(output / "training_summary.json", summary)
    if not pairs:
        summary.update(status="NO_TRAINING_SUPPORT", elapsed_seconds=time.monotonic() - started)
        save(output / "training_summary.json", summary)
        return summary
    os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0")
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    means, stds = features.mean(axis=0), features.std(axis=0)
    stds = np.maximum(stds, 1e-6)
    target_scale = max(float(np.std(targets)), 1e-6)
    x = torch.tensor(np.clip((features - means) / stds, -10, 10), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets) / target_scale, dtype=torch.float32)
    pair_index = torch.tensor(pairs, dtype=torch.long)
    pair_weight = torch.tensor(weights, dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_NAMES), HIDDEN_SIZES[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZES[1], 1),
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def diagnostics():
        with torch.no_grad():
            values = model(x).squeeze(-1)
            differences = values[pair_index[:, 0]] - values[pair_index[:, 1]]
            loss = float(((differences - y).square() * pair_weight).sum())
            nonzero = y.abs() > 1e-8
            agreement = float((differences[nonzero].sign() == y[nonzero].sign()).float().mean())
            if not bool(nonzero.any()):
                agreement = None
            return dict(
                weighted_pair_MSE=loss,
                nonzero_pairs=int(nonzero.sum()),
                nonzero_pair_sign_agreement=agreement,
                held_out=False,
            )

    summary["before_fit"] = diagnostics()
    summary["loss_trace"] = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        values = model(x).squeeze(-1)
        differences = values[pair_index[:, 0]] - values[pair_index[:, 1]]
        loss = ((differences - y).square() * pair_weight).sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite scoring loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            summary["loss_trace"].append(dict(epoch=epoch, weighted_pair_MSE=float(loss.detach())))
    summary["after_fit"] = diagnostics()
    payload = dict(
        profile=PROFILE,
        training_split="train",
        feature_names=list(FEATURE_NAMES),
        feature_mean=means.tolist(),
        feature_std=stds.tolist(),
        normalized_features_clip=10,
        target_scale=target_scale,
        score_units="standardized within-state Q difference; common offset is unidentified",
        layers=[
            dict(
                weight=layer.weight.detach().numpy().tolist(),
                bias=layer.bias.detach().numpy().tolist(),
            )
            for layer in model
            if isinstance(layer, torch.nn.Linear)
        ],
        seed=args.seed,
        frozen_LLM_model_path=source["model_path"],
        training_collection=str(directory),
        supported_anchors=len(groups),
        candidate_rows=len(rows),
    )
    save(output / "ranker.json", payload)
    with torch.no_grad():
        expected = model(x[:8]).squeeze(-1).tolist()
    actual = predict_features(features[:8], json.loads((output / "ranker.json").read_text()))
    error = max(abs(a - b) for a, b in zip(actual, expected, strict=True))
    if error > 1e-4:
        raise ValueError("saved NumPy inference differs from CPU training model")
    summary.update(
        status="COMPLETE",
        elapsed_seconds=time.monotonic() - started,
        serialized_inference_max_error=error,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        CPU_only=True,
    )
    save(output / "training_summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.seed != 43:
        raise ValueError("this single fixed ranker uses seed43")
    run(args)


if __name__ == "__main__":
    main()
