"""Aggregate paired AM and Pointer Network learning curves across training seeds."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from itertools import pairwise
from pathlib import Path
from typing import Any

SEEDS = (1234, 2345, 3456, 4567, 5678)
MODES = ("fixed", "adaptive", "adaptive_state")
FAMILIES = {"am": "tsp50", "ptrnet": "ptrnet_tsp50"}


def read_curve(path: Path) -> list[dict[str, float | int]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    curve: list[dict[str, float | int]] = []
    elapsed_offset = 0.0
    previous_raw_elapsed = 0.0
    best_cost = float("inf")
    for row in rows:
        raw_elapsed = float(row["elapsed_seconds"])
        if raw_elapsed < previous_raw_elapsed:
            elapsed_offset += previous_raw_elapsed
        previous_raw_elapsed = raw_elapsed
        best_cost = min(best_cost, float(row["validation_greedy_cost"]))
        curve.append(
            {
                "step": int(row["step"]),
                "elapsed_seconds": elapsed_offset + raw_elapsed,
                "best_cost": best_cost,
            }
        )
    return curve


def value_at_time(
    curve: list[dict[str, float | int]], budget: float
) -> dict[str, float | int]:
    selected = curve[0]
    for point in curve:
        if float(point["elapsed_seconds"]) > budget:
            break
        selected = point
    return selected


def transitions(points: list[dict[str, float]], field: str) -> list[dict[str, float]]:
    crossings: list[dict[str, float]] = []
    for previous, current in pairwise(points):
        if (previous["state_minus_fixed"] <= 0) != (current["state_minus_fixed"] <= 0):
            crossings.append(
                {
                    "left": previous[field],
                    "left_difference": previous["state_minus_fixed"],
                    "right": current[field],
                    "right_difference": current["state_minus_fixed"],
                }
            )
    return crossings


def aggregate_family(formal_dir: Path, prefix: str) -> dict[str, Any]:
    curves = {
        mode: [
            read_curve(formal_dir / f"{prefix}_{mode}_seed{seed}" / "metrics.jsonl")
            for seed in SEEDS
        ]
        for mode in MODES
    }

    common_steps = sorted(
        set.intersection(
            *(
                {int(point["step"]) for point in curve}
                for mode in MODES
                for curve in curves[mode]
            )
        )
    )
    step_points: list[dict[str, float]] = []
    for step in common_steps:
        point: dict[str, float] = {"step": float(step)}
        for mode in MODES:
            values = [
                float(next(item["best_cost"] for item in curve if item["step"] == step))
                for curve in curves[mode]
            ]
            point[mode] = statistics.mean(values)
            point[f"{mode}_sd"] = statistics.stdev(values)
        point["state_minus_fixed"] = point["adaptive_state"] - point["fixed"]
        step_points.append(point)

    common_time_limit = min(
        float(curve[-1]["elapsed_seconds"])
        for mode in MODES
        for curve in curves[mode]
    )
    time_budgets = [float(value) for value in range(0, int(common_time_limit) + 1, 30)]
    if time_budgets[-1] != common_time_limit:
        time_budgets.append(common_time_limit)
    time_points: list[dict[str, float]] = []
    for budget in time_budgets:
        point = {"seconds": budget}
        for mode in MODES:
            selected = [value_at_time(curve, budget) for curve in curves[mode]]
            point[mode] = statistics.mean(float(value["best_cost"]) for value in selected)
            point[f"{mode}_mean_step"] = statistics.mean(
                int(value["step"]) for value in selected
            )
        point["state_minus_fixed"] = point["adaptive_state"] - point["fixed"]
        time_points.append(point)

    return {
        "seeds": list(SEEDS),
        "step_curve": step_points,
        "step_crossings": transitions(step_points, "step"),
        "time_curve": time_points,
        "time_crossings": transitions(time_points, "seconds"),
        "common_time_limit_seconds": common_time_limit,
    }


def run(args: argparse.Namespace) -> None:
    formal_dir = Path(args.formal_dir).resolve()
    output_path = Path(args.output).resolve()
    result = {
        "families": {
            family: aggregate_family(formal_dir, prefix)
            for family, prefix in FAMILIES.items()
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, output_path)
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
