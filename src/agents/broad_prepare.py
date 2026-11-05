"""Fast, split-explicit inputs for the approved broad RLOO/order experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "broad_rloo_order_main_v1"
TEMPLATE = ROOT / "runs/m2-action-pooling-comparison-20261013-v6/seed42/prepared/train.parquet"


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("profile") != PROFILE or value.get("seed") != 43:
        raise ValueError("explicit approved broad profile/seed43 required")
    for split, count, maximum in (("train", 32, 441), ("valid", 60, 60)):
        ids = value[f"{split}_episode_ids"]
        if (
            len(ids) != count
            or len(set(ids)) != count
            or any(not key.startswith(f"formal-{split}-") for key in ids)
            or any(not 1 <= int(key.rsplit("-", 1)[1]) <= maximum for key in ids)
        ):
            raise ValueError(f"fixed {split} episode identities required")
    if value.get("formal_final_oos_access") is not False:
        raise ValueError("Final OOS must remain disabled")
    return value


def prepare(path, split):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path).resolve()
    config = load_config(path)
    destination = Path(config[f"prepared_{split}_path"]).resolve()
    ids = config[f"{split}_episode_ids"]
    if destination.exists():
        observed = pq.read_table(destination).to_pylist()
        if [r["extra_info"]["episode_id"] for r in observed] != ids:
            raise ValueError("existing broad prepared identities changed")
        return dict(status="REUSED", split=split, rows=len(observed), path=str(destination))
    template = pq.read_table(TEMPLATE).slice(0, 1).to_pylist()[0]
    settings = json.loads(template["env_kwargs"])
    source = ROOT / "data/processed" / f"{split}.parquet"
    rows = {r["extra_info"]["episode_id"]: r for r in pq.read_table(source).to_pylist()}
    selected = []
    for index, eid in enumerate(ids):
        original = rows[eid]
        actual = json.loads(original["env_kwargs"])
        kwargs = dict(settings)
        for key in ("episode_dates", "episode_open_times", "benchmark_closes"):
            kwargs[key] = actual[key]
        if split == "valid":
            # Reuse the existing history snapshot through February, never StockBench OOS.
            for key in (
                "data_version",
                "data_start",
                "data_end",
                "input_manifest_sha256",
                "input_file_count",
                "input_sessions_per_symbol",
                "manifest_root",
                "storage_root",
            ):
                kwargs[key] = actual[key]
        dates = [str(day)[:10] for day in kwargs["episode_dates"]]
        low, high = (
            ("2023-01-03", "2024-10-31") if split == "train" else ("2024-11-01", "2025-02-28")
        )
        if len(dates) != 21 or not all(low <= day <= high for day in dates):
            raise ValueError("episode dates must stay inside declared history split")
        row = dict(original)
        row["extra_info"] = dict(
            episode_id=eid,
            index=index,
            source_index=original["extra_info"]["index"],
            split=split,
            source_split=split,
            scientific_use="development",
            broad_profile=PROFILE,
        )
        row["env_kwargs"] = json.dumps(kwargs, sort_keys=True)
        selected.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), destination)
    receipt = dict(
        status="PREPARED",
        profile=PROFILE,
        split=split,
        rows=len(selected),
        path=str(destination),
        source=str(source),
        calibration_source=str(TEMPLATE),
        reward_and_observation_reused=True,
        policy_results_read=False,
        final_oos_access=False,
    )
    with destination.with_suffix(".receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.experiment_config, args.split)), flush=True)


if __name__ == "__main__":
    main()
