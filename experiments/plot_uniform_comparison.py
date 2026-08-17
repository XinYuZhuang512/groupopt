"""Render the uniform-distribution Original/Ours grouped bar chart from CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MODELS = (
    "am",
    "ptrnet",
    "gpn",
    "reversible_transformer",
    "geometric_transformer",
    "sparse_moe_transformer",
)
MODEL_LABELS = {
    "am": "AM",
    "ptrnet": "PtrNet",
    "gpn": "GPN",
    "reversible_transformer": "Reversible Transformer",
    "geometric_transformer": "Geometric Transformer",
    "sparse_moe_transformer": "Sparse MoE Transformer",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    aggregate = {
        row["model"]: row
        for row in read_rows(args.report_dir / "aggregate_results.csv")
        if row["distribution"] == "uniform"
    }
    paired = [
        row
        for row in read_rows(args.report_dir / "paired_results.csv")
        if row["distribution"] == "uniform"
    ]
    paired_lookup = {(row["model"], int(row["seed"])): row for row in paired}
    seeds = sorted({int(row["seed"]) for row in paired})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    original = np.array([float(aggregate[model]["mean_original_cost"]) for model in MODELS])
    ours = np.array([float(aggregate[model]["mean_ours_cost"]) for model in MODELS])
    improvements = np.array(
        [float(aggregate[model]["mean_relative_improvement_percent"]) for model in MODELS]
    )
    x = np.arange(len(MODELS))
    width = 0.36

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    original_bars = axis.bar(
        x - width / 2, original, width, label="Original", color="#8c8c8c", alpha=0.82
    )
    ours_bars = axis.bar(
        x + width / 2, ours, width, label="Ours", color="#2878b5", alpha=0.88
    )
    for offset, seed, marker in zip(np.linspace(-0.045, 0.045, len(seeds)), seeds, ("o", "D")):
        for center, key in ((x - width / 2, "original_cost"), (x + width / 2, "ours_cost")):
            axis.scatter(
                center + offset,
                [float(paired_lookup[(model, seed)][key]) for model in MODELS],
                marker=marker,
                s=24,
                label=f"seed {seed}" if key == "original_cost" else "_nolegend_",
                facecolors="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
    axis.bar_label(original_bars, fmt="%.2f", padding=2, fontsize=8)
    axis.bar_label(ours_bars, fmt="%.2f", padding=2, fontsize=8)
    annotation_offset = max(original.max(), ours.max()) * 0.06
    for index, improvement in enumerate(improvements):
        axis.text(
            index,
            max(original[index], ours[index]) + annotation_offset,
            f"{improvement:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS], rotation=25, ha="right")
    axis.set_ylabel("Mean tour cost (lower is better)")
    axis.set_title("Uniform TSP50: Original vs Ours (two-seed mean)")
    axis.legend(frameon=False, ncols=4)
    paired_cost_max = max(
        max(float(row["original_cost"]), float(row["ours_cost"])) for row in paired
    )
    axis.set_ylim(0.0, paired_cost_max * 1.12)
    axis.grid(axis="y", linewidth=0.6, alpha=0.25)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(args.report_dir / f"uniform_original_vs_ours_bar{suffix}", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
