"""汇总 AM head-summary 信息与梯度路由的三种子配对消融。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


DETACHED = "native_conditional_free"
END_TO_END = "native_free_end_to_end_summary"
NO_SUMMARY = "native_free_no_head_summary"
MODES = (DETACHED, END_TO_END, NO_SUMMARY)
COMPARISONS = {
    "detached_vs_end_to_end": (DETACHED, END_TO_END),
    "detached_vs_no_summary": (DETACHED, NO_SUMMARY),
    "end_to_end_vs_no_summary": (END_TO_END, NO_SUMMARY),
}


def _load(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64).flatten()
    if costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError(f"invalid costs: {path}")
    return costs, dict(payload["metadata"])


def _paired(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | int]:
    if first.shape != second.shape:
        raise ValueError("paired costs must have identical shapes")
    difference = first - second
    mean_difference = difference.mean()
    half_width = 1.96 * difference.std(unbiased=True) / math.sqrt(difference.numel())
    first_mean = first.mean()
    second_mean = second.mean()
    return {
        "first_mean": float(first_mean),
        "second_mean": float(second_mean),
        "first_minus_second": float(mean_difference),
        "second_relative_improvement_percent": float(
            100.0 * mean_difference / first_mean
        ),
        "paired_ci95_low": float(mean_difference - half_width),
        "paired_ci95_high": float(mean_difference + half_width),
        "second_win_rate": float((second < first).to(torch.float64).mean()),
        "test_size": int(difference.numel()),
    }


def _path(
    new_root: Path,
    main_root: Path,
    ablation_root: Path,
    mode: str,
    seed: int,
    test_seed: int,
) -> Path:
    if mode == DETACHED:
        root = main_root
    elif mode == NO_SUMMARY:
        root = ablation_root
    else:
        root = new_root
    return (
        root
        / "eval"
        / f"iid_tsp50_seed{test_seed}"
        / "am"
        / mode
        / f"seed{seed}"
        / "costs.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, default=20260904)
    args = parser.parse_args()

    seeds = (1234, 4321, 2468)
    costs: dict[tuple[str, int], torch.Tensor] = {}
    metadata: dict[tuple[str, int], dict[str, object]] = {}
    for mode in MODES:
        for seed in seeds:
            costs[(mode, seed)], metadata[(mode, seed)] = _load(
                _path(
                    args.new_root,
                    args.main_root,
                    args.ablation_root,
                    mode,
                    seed,
                    args.test_seed,
                )
            )

    identity = ("problem", "graph_size", "test_seed", "test_size", "distribution")
    reference = metadata[(DETACHED, seeds[0])]
    for item in metadata.values():
        if any(item.get(key) != reference.get(key) for key in identity):
            raise ValueError("evaluation metadata mismatch")

    mode_rows: list[dict[str, object]] = []
    for mode in MODES:
        for seed in seeds:
            mode_rows.append(
                {
                    "mode": mode,
                    "train_seed": seed,
                    "mean_cost": costs[(mode, seed)].mean().item(),
                }
            )

    comparison_rows: list[dict[str, object]] = []
    comparison_summary: dict[str, object] = {}
    for name, (first_mode, second_mode) in COMPARISONS.items():
        per_seed: dict[str, object] = {}
        for seed in seeds:
            statistics = _paired(costs[(first_mode, seed)], costs[(second_mode, seed)])
            per_seed[str(seed)] = statistics
            comparison_rows.append(
                {
                    "comparison": name,
                    "first_mode": first_mode,
                    "second_mode": second_mode,
                    "train_seed": seed,
                    **statistics,
                }
            )
        comparison_summary[name] = {
            "first_mode": first_mode,
            "second_mode": second_mode,
            "per_seed": per_seed,
            "all_seeds_favor_second": all(
                float(per_seed[str(seed)]["first_minus_second"]) > 0.0
                for seed in seeds
            ),
            "pooled_instance_seed_pairs": _paired(
                torch.cat([costs[(first_mode, seed)] for seed in seeds]),
                torch.cat([costs[(second_mode, seed)] for seed in seeds]),
            ),
        }

    validation = json.loads(
        (args.new_root / "summary" / "gradient_route_validation.json").read_text()
    )
    summary = {
        "experiment": "paper_summary_gradient_am_tsp50_v1",
        "problem": "uniform_euclidean_tsp50",
        "training_seeds": list(seeds),
        "test_seed": args.test_seed,
        "mode_mean_across_seeds": {
            mode: sum(costs[(mode, seed)].mean().item() for seed in seeds) / len(seeds)
            for mode in MODES
        },
        "comparisons": comparison_summary,
        "gradient_route_validation": validation,
        "sign_convention": "positive first-minus-second means the second mode is better",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_gradient_ablation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "mode_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mode_rows[0]))
        writer.writeheader()
        writer.writerows(mode_rows)
    with (args.output_dir / "paired_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
