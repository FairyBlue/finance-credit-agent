"""Render the recorded coverage diagnosis as a standalone research figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "runs/m2-train-coverage-diagnosis-20260917"


def main():
    sampling = json.loads((DIRECTORY / "sampling/sampling_analysis.json").read_text())
    behavior = json.loads((DIRECTORY / "behavior/analysis.json").read_text())
    assert sampling["status"] == behavior["status"] == "PASS"
    groups = {row["group"]: row for row in sampling["groups"]}
    contrasts = {row["contrast_id"]: row["delta"] for row in behavior["contrasts"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.9))
    colors = {"m1": "#266690", "m2_eta025": "#BC6B27"}
    for cell, label, offset, marker in (
        ("m1", "M1", -0.045, "o"),
        ("m2_eta025", "M2 (eta=0.25)", 0.045, "s"),
    ):
        steps = [2, 4, 6, 8]
        rows = [groups[f"{cell}:step{step}"] for step in steps]
        left.errorbar(
            [s + offset for s in steps],
            [r["mean_net_return_pct"] for r in rows],
            yerr=[r["conditional_monte_carlo_standard_error_pp"] for r in rows],
            marker=marker,
            color=colors[cell],
            label=label,
            capsize=4,
            linewidth=1.6,
        )
    left.axhline(
        groups["sft0:step0"]["mean_net_return_pct"],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="SFT sampled mean",
    )
    left.set(
        xticks=[2, 4, 6, 8],
        xlabel="Completed actor updates",
        ylabel="Mean net return (%)",
        ylim=(0.6, 2.5),
    )
    left.set_title("A. Sampled returns on the same four Train windows", loc="left", pad=14)
    left.grid(axis="y", color="#dddddd", linewidth=0.6)
    left.legend(frameon=False, fontsize=9, loc="upper right")
    labels = ["M1: 2 to 8", "M1: 4 to 8", "M2: 2 to 8", "M2: 4 to 8"]
    identifiers = [
        "m1_step2_to8_sampled",
        "m1_step4_to8_sampled",
        "m2_eta025_step2_to8_sampled",
        "m2_eta025_step4_to8_sampled",
    ]
    gross = [contrasts[k]["gross_market_return_pct"] for k in identifiers]
    fees = [-contrasts[k]["cost_pct"] for k in identifiers]
    right.barh(labels, gross, color="#266690", label="Market P&L from actual holdings")
    right.barh(labels, fees, left=gross, color="#BC6B27", label="Additional trading costs")
    for i, key in enumerate(identifiers):
        delta = contrasts[key]["net_return_pct"]
        right.text(delta - 0.015, i, f"{delta:+.3f}", va="center", ha="right", fontsize=10)
    right.axvline(0, color="#555555", linewidth=0.8)
    right.invert_yaxis()
    right.set(xlim=(-0.95, 0.02), xlabel="Change in mean net return (percentage points)")
    right.set_title("B. Exact accounting decomposition of the change", loc="left", pad=14)
    right.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, -0.32))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.83, bottom=0.26, wspace=0.30)
    fig.suptitle(
        "Train coverage diagnosis | one training seed, four sampled siblings per window",
        x=0.065,
        ha="left",
        fontsize=14,
        y=0.965,
    )
    fig.text(
        0.065,
        0.085,
        "A: bars are +/- 1 Monte Carlo SE, conditional on these windows and policies; "
        "evaluation seeds change across checkpoints.",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.065,
        0.045,
        "B: accounting attribution, not a causal effect of training. Source: "
        "m2-train-coverage-20260916-v2, 180 complete evaluation trajectories.",
        fontsize=9,
        color="#444444",
    )
    output = DIRECTORY / "coverage_diagnosis.png"
    fig.savefig(output, dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
