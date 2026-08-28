"""汇总跨模型 Original 与 GroupOpt 的同实例配对结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def _load_costs(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64)
    metadata = payload["metadata"]
    if costs.ndim != 1 or not torch.isfinite(costs).all():
        raise ValueError(f"非法逐实例 cost：{path}")
    return costs, metadata


def _paired_statistics(original: torch.Tensor, ours: torch.Tensor) -> dict[str, float]:
    if original.shape != ours.shape:
        raise ValueError("Original 与 Ours 的实例数量不一致")
    difference = original - ours
    mean_difference = difference.mean()
    standard_error = difference.std(unbiased=True) / math.sqrt(difference.numel())
    half_width = 1.96 * standard_error
    original_mean = original.mean()
    ours_mean = ours.mean()
    return {
        "original_mean": float(original_mean),
        "ours_mean": float(ours_mean),
        "original_minus_ours": float(mean_difference),
        "relative_improvement_percent": float(100.0 * mean_difference / original_mean),
        "paired_ci95_low": float(mean_difference - half_width),
        "paired_ci95_high": float(mean_difference + half_width),
        "ours_win_rate": float((ours < original).to(torch.float64).mean()),
        "tie_rate": float((ours == original).to(torch.float64).mean()),
        "test_size": int(difference.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--original-mode", default="native_original")
    parser.add_argument("--ours-mode", default="native_conditional_free")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "comparison": f"{args.original_mode}_vs_{args.ours_mode}",
        "lower_cost_is_better": True,
        "families": {},
    }
    family_results = summary["families"]
    assert isinstance(family_results, dict)

    for family in args.families:
        seed_results: dict[str, object] = {}
        original_means: list[float] = []
        ours_means: list[float] = []
        for seed in args.seeds:
            prefix = args.root / f"iid_tsp50_seed{args.test_seed}"
            original_path = (
                prefix
                / f"{family}_tsp50_{args.original_mode}_final{args.steps}_seed{seed}"
                / "costs.pt"
            )
            ours_path = (
                prefix
                / f"{family}_tsp50_{args.ours_mode}_final{args.steps}_seed{seed}"
                / "costs.pt"
            )
            original, original_metadata = _load_costs(original_path)
            ours, ours_metadata = _load_costs(ours_path)
            identity_fields = {
                "test_seed": None,
                "test_size": None,
                "graph_size": None,
                "distribution": "uniform",
            }
            if any(
                original_metadata.get(key, default) != ours_metadata.get(key, default)
                for key, default in identity_fields.items()
            ):
                raise ValueError(f"{family} seed={seed} 不是同一测试集上的配对比较")
            statistics = _paired_statistics(original, ours)
            seed_results[str(seed)] = statistics
            original_means.append(statistics["original_mean"])
            ours_means.append(statistics["ours_mean"])
            rows.append({"family": family, "seed": seed, **statistics})

        original_macro = sum(original_means) / len(original_means)
        ours_macro = sum(ours_means) / len(ours_means)
        family_results[family] = {
            "seeds": seed_results,
            "macro": {
                "seed_count": len(args.seeds),
                "original_mean": original_macro,
                "ours_mean": ours_macro,
                "original_minus_ours": original_macro - ours_macro,
                "relative_improvement_percent": 100.0
                * (original_macro - ours_macro)
                / original_macro,
                "all_seeds_improve": all(
                    result["original_minus_ours"] > 0.0 for result in seed_results.values()
                ),
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
