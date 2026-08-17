"""Summarize paired distribution-shift tests and render paper-ready figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from groupopt.problems.distributions import TSP_DISTRIBUTIONS, generate_tsp_coordinates

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
MODES = ("native_conditional_fixed", "native_conditional_free")


def load_costs(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"] if isinstance(payload, dict) else payload
    return costs.to(torch.float64).flatten()


def summarize(args: argparse.Namespace) -> None:
    eval_root = Path(args.eval_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for distribution in TSP_DISTRIBUTIONS:
        for model in MODELS:
            for seed in args.train_seeds:
                prefix = f"{model}_{{mode}}_final{args.steps}_seed{seed}"
                fixed_path = eval_root / distribution / prefix.format(mode=MODES[0]) / "costs.pt"
                free_path = eval_root / distribution / prefix.format(mode=MODES[1]) / "costs.pt"
                if not fixed_path.exists() or not free_path.exists():
                    raise FileNotFoundError(
                        f"missing paired costs for {distribution} {model} seed={seed}"
                    )
                fixed = load_costs(fixed_path)
                free = load_costs(free_path)
                if fixed.shape != free.shape:
                    raise ValueError("paired cost tensors have different shapes")
                difference = fixed - free
                standard_error = difference.std(unbiased=True) / math.sqrt(difference.numel())
                rows.append(
                    {
                        "distribution": distribution,
                        "model": model,
                        "seed": seed,
                        "sample_count": difference.numel(),
                        "original_cost": fixed.mean().item(),
                        "ours_cost": free.mean().item(),
                        "absolute_improvement": difference.mean().item(),
                        "relative_improvement_percent": (
                            100.0 * difference.mean() / fixed.mean()
                        ).item(),
                        "paired_ci95_low": (difference.mean() - 1.96 * standard_error).item(),
                        "paired_ci95_high": (difference.mean() + 1.96 * standard_error).item(),
                        "instance_win_rate_percent": (
                            100.0 * (difference > 0).to(torch.float64).mean()
                        ).item(),
                    }
                )

    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "paired_results.csv", rows)
    write_csv(output_dir / "aggregate_results.csv", aggregate)
    (output_dir / "results.json").write_text(
        json.dumps({"paired": rows, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )
    render_plots(rows, aggregate, output_dir, args.distribution_seed)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["distribution"], row["model"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (distribution, model), values in groups.items():
        improvements = [row["relative_improvement_percent"] for row in values]
        aggregate.append(
            {
                "distribution": distribution,
                "model": model,
                "seed_count": len(values),
                "mean_original_cost": sum(row["original_cost"] for row in values) / len(values),
                "mean_ours_cost": sum(row["ours_cost"] for row in values) / len(values),
                "mean_relative_improvement_percent": sum(improvements) / len(values),
                "min_seed_improvement_percent": min(improvements),
                "max_seed_improvement_percent": max(improvements),
                "positive_seed_count": sum(value > 0 for value in improvements),
            }
        )
    return aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_plots(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    output_dir: Path,
    distribution_seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    lookup = {(row["distribution"], row["model"]): row for row in aggregate}
    matrix = np.array(
        [
            [
                lookup[(distribution, model)]["mean_relative_improvement_percent"]
                for distribution in TSP_DISTRIBUTIONS
            ]
            for model in MODELS
        ]
    )
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    limit = max(1.0, float(np.abs(matrix).max()))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(TSP_DISTRIBUTIONS)), TSP_DISTRIBUTIONS, rotation=28, ha="right")
    axis.set_yticks(range(len(MODELS)), [MODEL_LABELS[model] for model in MODELS])
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.1f}%",
                ha="center",
                va="center",
                color="black",
            )
    axis.set_title("Ours relative improvement over Original (two-seed mean)")
    fig.colorbar(image, ax=axis, label="Relative improvement (%)", shrink=0.82)
    save_figure(fig, output_dir / "improvement_heatmap")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)
    x = np.arange(len(MODELS))
    for axis, distribution in zip(axes.flat, TSP_DISTRIBUTIONS):
        for seed_index, marker in enumerate(("o", "D")):
            seed = sorted({int(row["seed"]) for row in rows})[seed_index]
            values = [
                next(
                    row["relative_improvement_percent"]
                    for row in rows
                    if row["distribution"] == distribution
                    and row["model"] == model
                    and row["seed"] == seed
                )
                for model in MODELS
            ]
            axis.scatter(x, values, marker=marker, s=28, label=f"seed {seed}")
        means = [
            lookup[(distribution, model)]["mean_relative_improvement_percent"] for model in MODELS
        ]
        axis.plot(x, means, color="black", linewidth=1.2, alpha=0.7)
        axis.axhline(0.0, color="gray", linewidth=0.8)
        axis.set_title(distribution)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS], rotation=40, ha="right")
    axes[0, 0].set_ylabel("Relative improvement (%)")
    axes[1, 0].set_ylabel("Relative improvement (%)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "seed_consistency_by_distribution")

    fig, axis = plt.subplots(figsize=(7.0, 6.0))
    markers = ("o", "s", "^", "D", "P", "X")
    for distribution, marker in zip(TSP_DISTRIBUTIONS, markers):
        selected = [row for row in rows if row["distribution"] == distribution]
        axis.scatter(
            [row["original_cost"] for row in selected],
            [row["ours_cost"] for row in selected],
            marker=marker,
            s=36,
            label=distribution,
            alpha=0.8,
        )
    bounds = axis.get_xlim()
    lower = min(bounds[0], axis.get_ylim()[0])
    upper = max(bounds[1], axis.get_ylim()[1])
    axis.plot((lower, upper), (lower, upper), color="black", linewidth=1.0)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Original tour cost")
    axis.set_ylabel("Ours tour cost")
    axis.set_title("Original vs Ours across models, seeds, and distributions")
    axis.legend(frameon=False, fontsize=8)
    save_figure(fig, output_dir / "original_vs_ours")

    uniform_rows = [row for row in rows if row["distribution"] == "uniform"]
    uniform_lookup = {
        (row["model"], int(row["seed"])): row for row in uniform_rows
    }
    seeds = sorted({int(row["seed"]) for row in uniform_rows})
    original_means = np.array(
        [lookup[("uniform", model)]["mean_original_cost"] for model in MODELS]
    )
    ours_means = np.array(
        [lookup[("uniform", model)]["mean_ours_cost"] for model in MODELS]
    )
    improvements = np.array(
        [
            lookup[("uniform", model)]["mean_relative_improvement_percent"]
            for model in MODELS
        ]
    )
    x = np.arange(len(MODELS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    original_bars = axis.bar(
        x - width / 2,
        original_means,
        width,
        label="Original",
        color="#8c8c8c",
        alpha=0.82,
    )
    ours_bars = axis.bar(
        x + width / 2,
        ours_means,
        width,
        label="Ours",
        color="#2878b5",
        alpha=0.88,
    )
    seed_offsets = np.linspace(-0.045, 0.045, len(seeds))
    for seed_offset, seed, marker in zip(seed_offsets, seeds, ("o", "D")):
        axis.scatter(
            x - width / 2 + seed_offset,
            [uniform_lookup[(model, seed)]["original_cost"] for model in MODELS],
            marker=marker,
            s=24,
            label=f"seed {seed}",
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        axis.scatter(
            x + width / 2 + seed_offset,
            [uniform_lookup[(model, seed)]["ours_cost"] for model in MODELS],
            marker=marker,
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
    axis.bar_label(original_bars, fmt="%.2f", padding=2, fontsize=8)
    axis.bar_label(ours_bars, fmt="%.2f", padding=2, fontsize=8)
    for index, improvement in enumerate(improvements):
        top = max(original_means[index], ours_means[index])
        axis.text(
            index,
            top + max(original_means.max() * 0.06, 0.5),
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
    uniform_cost_max = max(
        max(row["original_cost"], row["ours_cost"]) for row in uniform_rows
    )
    axis.set_ylim(0.0, uniform_cost_max * 1.12)
    axis.grid(axis="y", linewidth=0.6, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "uniform_original_vs_ours_bar")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    for axis, distribution in zip(axes.flat, TSP_DISTRIBUTIONS):
        coordinates = generate_tsp_coordinates(1, 50, distribution, distribution_seed)[0]
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=12)
        axis.set_title(distribution)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    save_figure(fig, output_dir / "distribution_examples")


def save_figure(figure: Any, base_path: Path) -> None:
    figure.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1234, 4321])
    parser.add_argument("--distribution-seed", type=int, default=20260813)
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
