#!/usr/bin/env python3
"""Quick visualization helper for Lite3 training logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence


try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "matplotlib is required to visualize logs. Install it inside your environment."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_ROOT / "legged_gym" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize reward and stability metrics from rewards.csv."
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment directory inside legged_gym/logs (e.g. two_leg_stand).",
    )
    parser.add_argument(
        "--run-name",
        help="Specific run suffix to select (matches the tail of the timestamped folder). "
        "If omitted, the most recent run is used.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["total_reward", "two_leg_stability"],
        help="Metrics (column names from rewards.csv) to visualize.",
    )
    parser.add_argument(
        "--save-path",
        help="Optional path to save the generated plot (defaults to <run_dir>/reward_metrics.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window (may require a desktop session).",
    )
    return parser.parse_args()


def select_run_dir(experiment_name: str, run_name: str | None) -> Path:
    exp_dir = LOG_ROOT / experiment_name
    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise SystemExit(f"No runs found inside {exp_dir}")

    if run_name:
        run_dirs = [p for p in run_dirs if p.name.endswith(run_name) or run_name in p.name]
        if not run_dirs:
            raise SystemExit(
                f"No run matching '{run_name}' found inside {exp_dir}. "
                "Check the available folders."
            )

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def load_metrics(csv_path: Path, metrics: Sequence[str]) -> Dict[str, List[float]]:
    if not csv_path.exists():
        raise SystemExit(f"rewards.csv not found at {csv_path}")

    series: Dict[str, List[float]] = {metric: [] for metric in metrics}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"No header row found in {csv_path}")
        for row in reader:
            for metric in metrics:
                value = row.get(metric, "")
                try:
                    series[metric].append(float(value))
                except (TypeError, ValueError):
                    series[metric].append(math.nan)
    return series


def summarize(metric: str, values: Sequence[float]) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return f"{metric}: no numeric samples"
    last = finite[-1]
    mean = sum(finite) / len(finite)
    maximum = max(finite)
    return f"{metric}: last={last:.4f} mean={mean:.4f} max={maximum:.4f} (n={len(finite)})"


def plot_metrics(
    run_dir: Path, series: Dict[str, List[float]], show: bool, save_path: Path | None
) -> None:
    metric_names = list(series.keys())
    steps = range(1, len(next(iter(series.values()))) + 1)
    fig, axes = plt.subplots(
        len(metric_names),
        1,
        sharex=True,
        figsize=(8, max(3, 2 * len(metric_names))),
        constrained_layout=True,
    )
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_names):
        ax.plot(steps, series[metric], label=metric, color="tab:blue")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Reward CSV row (episode)")
    fig.suptitle(f"{run_dir.name} ({run_dir.parent.name})", fontsize=12)

    output = save_path or (run_dir / "reward_metrics.png")
    fig.savefig(output)
    print(f"Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = select_run_dir(args.experiment_name, args.run_name)
    csv_path = run_dir / "rewards.csv"
    print(f"Reading metrics from {csv_path}")

    series = load_metrics(csv_path, args.metrics)
    for metric in args.metrics:
        print(summarize(metric, series[metric]))

    save_path = Path(args.save_path).expanduser() if args.save_path else None
    try:
        plot_metrics(run_dir, series, show=args.show, save_path=save_path)
    except Exception as exc:
        raise SystemExit(f"Failed to generate plot: {exc}") from exc


if __name__ == "__main__":
    main()
