"""The three pre-registered standard figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path


def _prepare_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_reward_vs_step(
    curves: Mapping[str, Sequence[tuple[int, float]]], *, output_path: str | Path
) -> Path:
    """Overlay arm reward learning curves on one axis."""

    import matplotlib.pyplot as plt

    output = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for arm, points in curves.items():
        if not points:
            continue
        steps, rewards = zip(*points, strict=True)
        axis.plot(steps, rewards, label=arm)
    axis.set(xlabel="training step", ylabel="reward", title="Reward vs training step")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_tool_call_success(
    success_rates: Mapping[str, float], *, output_path: str | Path
) -> Path:
    """Compare valid tool-call proportions across arms/baselines."""

    import matplotlib.pyplot as plt

    if any(value < 0.0 or value > 1.0 for value in success_rates.values()):
        raise ValueError("tool success rates must lie in [0, 1]")
    output = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    labels = list(success_rates)
    values = [success_rates[label] for label in labels]
    axis.bar(labels, values)
    axis.set(ylim=(0.0, 1.0), ylabel="successful tool-call rate", title="Tool-call success")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_equity_vs_buy_and_hold(
    *,
    dates: Sequence[datetime | str],
    equity_curves: Mapping[str, Sequence[float]],
    output_path: str | Path,
) -> Path:
    """Plot normalized portfolio equity for arms and buy-and-hold."""

    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num

    output = _prepare_output(output_path)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    parsed_dates = [
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in dates
    ]
    numeric_dates = date2num(parsed_dates)
    for name, values in equity_curves.items():
        if len(values) != len(dates):
            raise ValueError(f"equity curve {name!r} does not align with dates")
        if not values or values[0] <= 0.0:
            raise ValueError(f"equity curve {name!r} must start positive")
        normalized = [value / values[0] for value in values]
        axis.plot(numeric_dates, normalized, label=name)
    axis.xaxis_date()
    axis.set(xlabel="date", ylabel="normalized equity", title="Portfolio equity vs buy-and-hold")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
