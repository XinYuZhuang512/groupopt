"""汇总官方 AM 四组桥接实验的同实例配对结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

STUDIES = {
    "bridge": {
        "modes": (
            "official_original",
            "official_conditional_fixed",
            "official_forest_fixed",
            "native_conditional_free",
        ),
        "comparisons": (
            (
                "original_to_conditional_fixed",
                "official_original",
                "official_conditional_fixed",
            ),
            (
                "conditional_fixed_to_forest_fixed",
                "official_conditional_fixed",
                "official_forest_fixed",
            ),
            (
                "forest_fixed_to_groupopt",
                "official_forest_fixed",
                "native_conditional_free",
            ),
            ("original_to_groupopt", "official_original", "native_conditional_free"),
        ),
    },
    "contexts": {
        "modes": (
            "official_original",
            "native_conditional_free",
            "official_groupopt_global_anchor",
            "official_groupopt_graph_tail",
        ),
        "comparisons": (
            ("original_to_local_start", "official_original", "native_conditional_free"),
            (
                "original_to_global_anchor",
                "official_original",
                "official_groupopt_global_anchor",
            ),
            (
                "original_to_graph_tail",
                "official_original",
                "official_groupopt_graph_tail",
            ),
            (
                "local_start_to_global_anchor",
                "native_conditional_free",
                "official_groupopt_global_anchor",
            ),
            (
                "local_start_to_graph_tail",
                "native_conditional_free",
                "official_groupopt_graph_tail",
            ),
            (
                "global_anchor_to_graph_tail",
                "official_groupopt_global_anchor",
                "official_groupopt_graph_tail",
            ),
        ),
    },
    "context_extension": {
        "modes": (
            "official_original",
            "native_conditional_free",
            "official_groupopt_graph_tail",
        ),
        "comparisons": (
            ("original_to_local_start", "official_original", "native_conditional_free"),
            (
                "original_to_graph_tail",
                "official_original",
                "official_groupopt_graph_tail",
            ),
            (
                "local_start_to_graph_tail",
                "native_conditional_free",
                "official_groupopt_graph_tail",
            ),
        ),
    },
}


def _load_costs(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64)
    if costs.ndim != 1 or not torch.isfinite(costs).all():
        raise RuntimeError(f"invalid costs: {path}")
    return costs, dict(payload["metadata"])


def _paired_statistics(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if baseline.shape != candidate.shape:
        raise RuntimeError("paired cost tensors have different shapes")
    difference = baseline - candidate
    mean_difference = difference.mean().item()
    standard_error = difference.std(unbiased=True).item() / math.sqrt(difference.numel())
    return {
        "baseline_mean_cost": baseline.mean().item(),
        "candidate_mean_cost": candidate.mean().item(),
        "mean_improvement": mean_difference,
        "relative_improvement_percent": 100.0 * mean_difference / baseline.mean().item(),
        "paired_ci95_low": mean_difference - 1.96 * standard_error,
        "paired_ci95_high": mean_difference + 1.96 * standard_error,
        "candidate_win_rate": (candidate < baseline).to(torch.float64).mean().item(),
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--study", choices=tuple(STUDIES), default="bridge")
    parser.add_argument("--test-seed", type=int, default=20260823)
    parser.add_argument("--train-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, nargs="+", default=(600, 1200, 2400))
    args = parser.parse_args()
    study = STUDIES[args.study]
    modes = study["modes"]
    comparisons = study["comparisons"]

    evaluation_root = args.output_root / "eval" / f"iid_tsp50_seed{args.test_seed}"
    summary_root = args.output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "test_seed": args.test_seed,
        "train_seed": args.train_seed,
        "study": args.study,
        "steps": {},
    }
    csv_rows: list[dict[str, Any]] = []

    for step in args.steps:
        costs_by_mode: dict[str, torch.Tensor] = {}
        metadata_by_mode: dict[str, dict[str, Any]] = {}
        for mode in modes:
            evaluation_dir = evaluation_root / f"{mode}_step{step}_seed{args.train_seed}"
            legacy_dir = evaluation_root / f"{mode}_seed{args.train_seed}"
            if step == 300 and not evaluation_dir.exists() and legacy_dir.exists():
                evaluation_dir = legacy_dir
            costs, metadata = _load_costs(evaluation_dir / "costs.pt")
            summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
            if int(metadata["checkpoint_step"]) != step or int(summary["checkpoint_step"]) != step:
                raise RuntimeError(f"checkpoint step mismatch: {evaluation_dir}")
            if int(metadata["test_seed"]) != args.test_seed:
                raise RuntimeError(f"test seed mismatch: {evaluation_dir}")
            costs_by_mode[mode] = costs
            metadata_by_mode[mode] = metadata

        sample_counts = {costs.numel() for costs in costs_by_mode.values()}
        if len(sample_counts) != 1:
            raise RuntimeError(f"test-size mismatch at step {step}")

        step_result: dict[str, Any] = {
            "sample_count": sample_counts.pop(),
            "mean_costs": {
                mode: costs_by_mode[mode].mean().item() for mode in modes
            },
            "comparisons": {},
        }
        for name, baseline_mode, candidate_mode in comparisons:
            statistics = _paired_statistics(
                costs_by_mode[baseline_mode], costs_by_mode[candidate_mode]
            )
            comparison = {
                "baseline_mode": baseline_mode,
                "candidate_mode": candidate_mode,
                **statistics,
            }
            step_result["comparisons"][name] = comparison
            csv_rows.append({"step": step, "comparison": name, **comparison})
        result["steps"][str(step)] = step_result

    _atomic_text(
        summary_root / f"{args.study}_summary.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    csv_path = summary_root / f"{args.study}_comparisons.csv"
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(temporary_csv, csv_path)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
