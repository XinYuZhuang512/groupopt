"""Summarize single-seed model-breadth development evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("transformer_ln", "gat", "pointerformer", "gru")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"status": "development_only", "models": {}}
    for model in MODELS:
        costs = {}
        for mode in ("joint_fixed", "joint_free"):
            path = (
                args.root
                / f"{model}_tsp50_{mode}_final10k_seed1234"
                / "summary.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload["checkpoint_step"]) != 10000:
                raise ValueError(f"not a final step-10000 checkpoint: {path}")
            costs[mode] = float(payload["mean_cost"])
        result["models"][model] = {
            **costs,
            "free_minus_fixed": costs["joint_free"] - costs["joint_fixed"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
