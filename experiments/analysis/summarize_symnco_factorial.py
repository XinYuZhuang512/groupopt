"""Summarize the AM Original/Ours x REINFORCE/SYM-NCO factorial pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _variant(base_mode: str) -> str:
    if base_mode == "native_conditional_fixed":
        return "original"
    if base_mode == "native_conditional_free":
        return "ours"
    raise ValueError(f"unexpected base mode in factorial results: {base_mode}")


def _paired_stats(differences: torch.Tensor) -> dict[str, float]:
    differences = differences.to(torch.float64)
    count = int(differences.numel())
    mean = differences.mean().item()
    standard_error = differences.std(unbiased=True).item() / math.sqrt(count)
    radius = 1.96 * standard_error
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "standard_error": standard_error,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty summary: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(root: Path, output_dir: Path) -> None:
    records: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for summary_path in sorted(root.rglob("summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("model", "am") != "am":
            continue
        variant = _variant(summary["base_mode"])
        scheme = summary.get("training_scheme", "reinforce")
        seed = int(summary["train_seed"])
        distribution = summary["distribution"]
        costs_path = summary_path.with_name("costs.pt")
        if not costs_path.exists():
            raise FileNotFoundError(costs_path)
        costs = torch.load(costs_path, map_location="cpu", weights_only=False)["costs"]
        key = (variant, scheme, seed, distribution)
        if key in records:
            raise RuntimeError(f"duplicate factorial cell: {key}")
        records[key] = {"summary": summary, "costs": costs, "path": str(summary_path)}

    variants = ("original", "ours")
    schemes = ("reinforce", "symnco_am")
    seeds = sorted({key[2] for key in records})
    distributions = sorted({key[3] for key in records})
    expected = {
        (variant, scheme, seed, distribution)
        for variant in variants
        for scheme in schemes
        for seed in seeds
        for distribution in distributions
    }
    missing = sorted(expected.difference(records))
    if missing:
        raise RuntimeError(f"factorial results are incomplete; missing {missing}")

    cells: list[dict[str, Any]] = []
    ours_comparisons: list[dict[str, Any]] = []
    symmetry_comparisons: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for distribution in distributions:
        for seed in seeds:
            for scheme in schemes:
                for variant in variants:
                    record = records[(variant, scheme, seed, distribution)]
                    summary = record["summary"]
                    cells.append(
                        {
                            "distribution": distribution,
                            "train_seed": seed,
                            "training_scheme": scheme,
                            "variant": variant,
                            "mean_cost": float(summary["mean_cost"]),
                            "standard_error": float(summary["standard_error"]),
                            "test_size": int(summary["test_size"]),
                            "summary_path": record["path"],
                        }
                    )

                fixed = records[("original", scheme, seed, distribution)]["costs"]
                free = records[("ours", scheme, seed, distribution)]["costs"]
                stats = _paired_stats(fixed - free)
                ours_comparisons.append(
                    {
                        "distribution": distribution,
                        "train_seed": seed,
                        "training_scheme": scheme,
                        "original_mean_cost": fixed.mean().item(),
                        "ours_mean_cost": free.mean().item(),
                        "ours_improvement": stats["mean"],
                        "ours_relative_improvement_percent": (
                            100.0 * stats["mean"] / fixed.mean().item()
                        ),
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                    }
                )

            for variant in variants:
                standard = records[(variant, "reinforce", seed, distribution)]["costs"]
                symmetric = records[(variant, "symnco_am", seed, distribution)]["costs"]
                stats = _paired_stats(standard - symmetric)
                symmetry_comparisons.append(
                    {
                        "distribution": distribution,
                        "train_seed": seed,
                        "variant": variant,
                        "reinforce_mean_cost": standard.mean().item(),
                        "symnco_mean_cost": symmetric.mean().item(),
                        "symnco_improvement": stats["mean"],
                        "symnco_relative_improvement_percent": (
                            100.0 * stats["mean"] / standard.mean().item()
                        ),
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                    }
                )

            original_standard = records[
                ("original", "reinforce", seed, distribution)
            ]["costs"]
            ours_standard = records[("ours", "reinforce", seed, distribution)]["costs"]
            original_symnco = records[
                ("original", "symnco_am", seed, distribution)
            ]["costs"]
            ours_symnco = records[("ours", "symnco_am", seed, distribution)]["costs"]
            interaction = (original_symnco - ours_symnco) - (
                original_standard - ours_standard
            )
            stats = _paired_stats(interaction)
            interactions.append(
                {
                    "distribution": distribution,
                    "train_seed": seed,
                    "interaction": stats["mean"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "interpretation": "positive means SYM-NCO increases Ours' advantage",
                }
            )

    aggregate: list[dict[str, Any]] = []
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ours_comparisons:
        grouped[(row["distribution"], row["training_scheme"])].append(
            row["ours_improvement"]
        )
    for (distribution, scheme), improvements in sorted(grouped.items()):
        aggregate.append(
            {
                "distribution": distribution,
                "training_scheme": scheme,
                "mean_improvement_across_train_seeds": sum(improvements) / len(improvements),
                "minimum_seed_improvement": min(improvements),
                "maximum_seed_improvement": max(improvements),
                "positive_seed_count": sum(value > 0 for value in improvements),
                "seed_count": len(improvements),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "factorial_cells.csv", cells)
    _write_csv(output_dir / "ours_vs_original.csv", ours_comparisons)
    _write_csv(output_dir / "symnco_effect.csv", symmetry_comparisons)
    _write_csv(output_dir / "factorial_interaction.csv", interactions)
    payload = {
        "aggregate_ours_vs_original": aggregate,
        "cells": cells,
        "factorial_interactions": interactions,
        "notes": [
            "Positive improvement means lower tour cost for the named method.",
            "Instance-paired CIs condition on a trained checkpoint.",
            "Two training seeds are a pilot consistency check, not a population-level CI.",
        ],
        "ours_vs_original": ours_comparisons,
        "symnco_effect": symmetry_comparisons,
    }
    temporary = output_dir / "factorial_summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "factorial_summary.json")
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summarize(Path(arguments.root).resolve(), Path(arguments.output_dir).resolve())
