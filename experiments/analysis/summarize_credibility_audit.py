"""Summarize the locked final-checkpoint credibility audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

FAMILIES = {
    "am": "tsp50",
    "ptrnet": "ptrnet_tsp50",
    "gpn": "gpn_tsp50",
}
ARMS = ("original_fixed", "joint_fixed", "joint_free")
SEEDS = (1234, 2345, 3456, 4567)
T_CRITICAL_95_DF3 = 3.182446


def load_cost(root: Path, prefix: str, arm: str, seed: int) -> float:
    path = root / f"{prefix}_{arm}_final10k_seed{seed}" / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["checkpoint_step"]) != 10000:
        raise ValueError(f"not a step-10000 checkpoint: {path}")
    return float(payload["mean_cost"])


def paired_summary(left: list[float], right: list[float]) -> dict[str, object]:
    differences = [a - b for a, b in zip(left, right, strict=True)]
    center = mean(differences)
    margin = T_CRITICAL_95_DF3 * stdev(differences) / math.sqrt(len(differences))
    return {
        "differences": differences,
        "mean_difference": center,
        "seed_level_95_ci": [center - margin, center + margin],
        "wins": sum(value < 0.0 for value in differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "primary_unit": "training_seed",
        "checkpoint": "exact_step_10000",
        "families": {},
    }
    for family, prefix in FAMILIES.items():
        costs = {
            arm: [load_cost(args.root, prefix, arm, seed) for seed in SEEDS]
            for arm in ARMS
        }
        result["families"][family] = {
            "costs": costs,
            "means": {arm: mean(values) for arm, values in costs.items()},
            "joint_free_minus_original": paired_summary(
                costs["joint_free"], costs["original_fixed"]
            ),
            "joint_fixed_minus_original": paired_summary(
                costs["joint_fixed"], costs["original_fixed"]
            ),
            "joint_free_minus_joint_fixed": paired_summary(
                costs["joint_free"], costs["joint_fixed"]
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
