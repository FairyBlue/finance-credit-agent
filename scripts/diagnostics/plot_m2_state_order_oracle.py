#!/usr/bin/env python3
"""Plot the completed fixed-state oracle readout without replaying environments.

Only readout.json and its SHA-bound CSVs are read. PNG/SVG outputs are new files;
existing figures, raw evidence, the collector and the sealed contract are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DAYS = (0, 5, 10, 15)
METHODS = {
    "plain": ("Plain + family gate", "#2878A5", "o", "-"),
    "family_pool": ("Family pool + family gate", "#D26B35", "s", "--"),
    "exact_same_gate": ("Order pool + family gate", "#358763", "^", ":"),
    "exact_order_gate": ("Order pool + order gate", "#8B60A8", "D", "-."),
    "zero": ("Zero reference", "#747474", "x", ":"),
}


def read_inputs(run: Path) -> tuple[dict, dict[str, list[dict[str, str]]]]:
    document = json.loads((run / "readout.json").read_text(encoding="utf-8"))
    if document.get("status") != "DESCRIPTIVE_READOUT_COMPLETE":
        raise ValueError("plot requires a completed readout")
    if document.get("profile") != "m2_real_train_state_order_readout_v1":
        raise ValueError("unexpected readout profile")
    tables = {}
    for name in ("readout_metrics.csv", "readout_coverage.csv", "readout_algebra.csv"):
        payload = (run / name).read_bytes()
        expected = document["output_csv_sha256"][name]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"readout CSV SHA differs: {name}")
        tables[name] = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    return document, tables


def truth(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid CSV boolean: {value!r}")
    return value == "True"


def number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("readout contains a nonfinite numeric value")
    return result


def coverage_by_day(document: dict, rows: list[dict]) -> dict[int, dict]:
    bins = defaultdict(list)
    for row in rows:
        bins[int(row["decision_session_index"])].append(row)
    if set(bins) != set(DAYS) or len(rows) != document["sample"]["source_states"]:
        raise ValueError("coverage CSV must retain the complete four-day sample")
    output = {}
    for day in DAYS:
        items = bins[day]
        groups = defaultdict(list)
        for row in items:
            groups[row["seed"], row["group_id"]].append(row)
        if any(len(group) != 4 for group in groups.values()):
            raise ValueError("coverage source group has missing or duplicate siblings")
        full = 0
        for group in groups.values():
            flags = {truth(row["group_all_16_legal"]) for row in group}
            if len(flags) != 1:
                raise ValueError("inconsistent full-menu coverage")
            full += flags == {True}
        output[day] = {
            "anchors": len(items),
            "complete": sum(truth(row["complete_anchor"]) for row in items),
            "state_peer": sum(truth(row["own_source_state_has_peer"]) for row in items),
            "state_order_peer": sum(truth(row["own_source_state_order_has_peer"]) for row in items),
            "groups": len(groups),
            "full_groups": full,
        }
    return output


def metric_index(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    index = {}
    for row in rows:
        if row["analysis"] != "same_anchor_oracle" or row["scope"] != "day":
            continue
        key = row["lane"], row["method"], int(row["scope_value"])
        if key in index:
            raise ValueError(f"duplicate same-anchor metric: {key}")
        index[key] = row
    return index


def component_scales(rows: list[dict]) -> dict[str, list[float]]:
    bins = defaultdict(list)
    for row in rows:
        bins[int(row["decision_session_index"])].append(row)
    if set(bins) != set(DAYS):
        raise ValueError("component algebra must retain all four days")
    result = {}
    for key in ("hold_energy", "q_energy", "continuation_energy"):
        values = []
        for day in DAYS:
            energies = [number(row[key]) for row in bins[day]]
            if min(energies) < 0:
                raise ValueError("negative centered component energy")
            values.append(math.sqrt(sum(energies) / len(energies)))
        result[key] = values
    return result


def tau_floor_by_day(document: dict, coverage: dict) -> dict[int, float | None]:
    # Each group appears repeatedly across lane/method combinations in this table.
    floors = {}
    for row in document["matched_tau_decomposition_full_groups"]:
        key = int(row["seed"]), row["group_id"], int(row["decision_session_index"])
        value = float(row["mean_tau_squared"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid zero-sum representation floor")
        if key in floors and not math.isclose(floors[key], value, abs_tol=1e-12, rel_tol=1e-10):
            raise ValueError("tau floor differs across methods or lanes")
        floors[key] = value
    result = {}
    for day in DAYS:
        values = [value for (*_, observed_day), value in floors.items() if observed_day == day]
        if len(values) != coverage[day]["full_groups"]:
            raise ValueError("tau decomposition and full-menu coverage disagree")
        result[day] = sum(values) / len(values) if values else None
    return result


def style_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, pad=12)
    axis.set_xticks(range(len(DAYS)), ["0\nInitial", "5", "10", "15"])
    axis.set_xlabel("Decision session (zero indexed)")
    axis.set_ylabel(ylabel)
    axis.axvspan(-0.35, 0.35, color="#EFF3F6", zorder=-5)
    axis.set_xlim(-0.35, 3.35)
    axis.grid(axis="y", color="#DDE2E6", linewidth=0.6)


def render(run: Path, stem: str) -> tuple[Path, Path]:
    run = run.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stem):
        raise ValueError("--stem must be a plain filename stem")
    outputs = tuple(run / f"{stem}.{suffix}" for suffix in ("png", "svg"))
    if any(path.exists() for path in outputs):
        raise FileExistsError("figure output already exists; choose a new --stem")
    document, tables = read_inputs(run)
    coverage = coverage_by_day(document, tables["readout_coverage.csv"])
    metrics = metric_index(tables["readout_metrics.csv"])
    scales = component_scales(tables["readout_algebra.csv"])
    floors = tau_floor_by_day(document, coverage)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 11.3))
    ax_coverage, ax_error, ax_rank, ax_components = axes.flat
    x = list(range(len(DAYS)))
    style_axis(ax_coverage, "A. Coverage in all source states", "Share of all anchors")
    for key, label, color, marker in (
        ("complete", "Complete four-order anchors", "#2878A5", "o"),
        ("state_peer", "Peer with the same source state", "#D26B35", "s"),
        ("state_order_peer", "Peer with the same state + order", "#358763", "^"),
    ):
        ax_coverage.plot(
            x,
            [coverage[day][key] / coverage[day]["anchors"] for day in DAYS],
            label=label,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            markerfacecolor="white" if key != "complete" else color,
        )
    ax_coverage.set_ylim(-0.04, 1.08)
    ax_coverage.yaxis.set_major_formatter(PercentFormatter(1))
    ax_coverage.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.29))
    counts = "   ".join(
        f"j{day}: {coverage[day]['complete']}/{coverage[day]['anchors']}" for day in DAYS
    )
    ax_coverage.text(
        0, -0.25, f"Complete anchors: {counts}", transform=ax_coverage.transAxes, fontsize=8
    )

    style_axis(ax_error, "B. Same-anchor four-slot oracle error", "MSE (Financial Y squared)")
    style_axis(ax_rank, "C. Same-anchor distinct-order ranking", "Pair-weighted rank score")
    for method, (label, color, marker, linestyle) in METHODS.items():
        mse, rank = [], []
        for day in DAYS:
            slots = metrics.get(("four_original_slots", method, day))
            unique = metrics.get(("unique_order_alternate", method, day))
            mse.append(number(slots["mse_member_weighted"]) if slots else math.nan)
            value = unique["rank_pair_weighted"] if unique else ""
            rank.append(number(value) if value != "" else math.nan)
        for axis, values in ((ax_error, mse), (ax_rank, rank)):
            axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.6,
                markersize=6,
                markerfacecolor="white" if method == "exact_same_gate" else color,
                label=label,
            )
    ax_error.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_error.set_ylim(bottom=-0.02 * max(ax_error.get_ylim()[1], 1e-8))
    ax_error.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.29), ncol=2)
    ax_error.text(
        0,
        -0.25,
        "Plain and order pool with the family gate coincide by algebra.",
        transform=ax_error.transAxes,
        fontsize=8,
    )
    ax_rank.set_ylim(-0.04, 1.06)
    ax_rank.yaxis.set_major_formatter(PercentFormatter(1))
    rank_support = []
    for position, day in enumerate(DAYS):
        row = metrics.get(("unique_order_alternate", "plain", day))
        available = int(row["rank_units_with_nontied_oracle"]) if row else 0
        complete = coverage[day]["complete"]
        rank_support.append(f"j{day}: {available}/{complete}")
        if available == 0:
            ax_rank.text(position, 0.5, "NA", ha="center", color="#646464")
    ax_rank.text(
        0,
        -0.25,
        "Rankable / complete anchors: " + "   ".join(rank_support),
        transform=ax_rank.transAxes,
        fontsize=8,
    )

    style_axis(ax_components, "D. Centered component scales", "Financial Y (centered RMS)")
    for key, label, color, marker in (
        ("hold_energy", "H: hold baseline", "#2878A5", "o"),
        ("q_energy", "q: own-order increment", "#D26B35", "s"),
        ("continuation_energy", "F: continuation difference", "#358763", "^"),
    ):
        ax_components.plot(x, scales[key], label=label, color=color, marker=marker, linewidth=1.8)
    ax_components.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_components.set_ylim(bottom=0)
    ax_components.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.29))
    ax_components.text(
        0,
        -0.25,
        "Square root of mean within-group energy; not variance shares.",
        transform=ax_components.transAxes,
        fontsize=8,
    )

    figure.subplots_adjust(
        left=0.075, right=0.985, top=0.865, bottom=0.22, hspace=0.95, wspace=0.27
    )
    sample = document["sample"]
    figure.suptitle(
        "Fixed-state order oracle | original terminal Financial Y",
        x=0.075,
        ha="left",
        fontsize=16,
        y=0.97,
    )
    figure.text(
        0.075,
        0.925,
        f"M2 Train only | seeds 42 / 43 | {sample['source_groups']} source groups | "
        f"{sample['source_states']} anchors | hold suffix to each original window endpoint",
        fontsize=10,
        color="#454545",
    )
    menu_summary = []
    for day in DAYS:
        value = f"{floors[day]:.3g}" if floors[day] is not None else "NA"
        menu_summary.append(
            f"j{day}: {coverage[day]['full_groups']}/{coverage[day]['groups']}, floor {value}"
        )
    figure.text(
        0.075,
        0.095,
        "Full 16-cell menus and mean zero-sum floor (Financial Y squared): "
        + "  |  ".join(menu_summary),
        fontsize=8.5,
    )
    figure.text(
        0.075,
        0.063,
        "Rank excludes oracle-tied pairs; prediction ties score 1/2. "
        "NA retains anchors with no distinguishable orders. Initial-state control is shaded.",
        fontsize=8.5,
        color="#454545",
    )
    figure.text(
        0.075,
        0.033,
        "Deterministic structural diagnostics, not denoising or RL gains. "
        "H / q / F covary; panel D is not a stacked decomposition. "
        "All days are descriptive and dependent.",
        fontsize=8.5,
        color="#454545",
    )
    try:
        for path in outputs:
            with path.open("xb") as handle:
                figure.savefig(handle, format=path.suffix[1:], dpi=200, facecolor="white")
    finally:
        plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--stem", default="state_order_oracle_summary")
    args = parser.parse_args()
    outputs = render(args.run, args.stem)
    print(json.dumps({"status": "PLOTTED", "outputs": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()
