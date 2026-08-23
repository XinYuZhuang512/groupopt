"""验证 Original 分支与官方 AM 前向结果完全一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groupopt.models.official_am import OfficialAttentionModelGroupOpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-am-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260823)
    model = OfficialAttentionModelGroupOpt(
        official_root=args.official_am_root,
        embedding_dim=32,
        n_heads=4,
        n_encoder_layers=2,
    ).to(device)
    model.eval()
    coordinates = torch.rand(8, 20, 2, device=device)

    model.native.set_decode_type("greedy")
    with torch.no_grad():
        native_cost, native_ll, native_tour = model.native(coordinates, return_pi=True)
        wrapped = model(
            coordinates,
            decode_type="greedy",
            base_mode="official_original",
        )

    exact_tour = torch.equal(native_tour, wrapped.tails)
    exact_cost = torch.equal(native_cost, wrapped.cost)
    exact_log_likelihood = torch.equal(native_ll, wrapped.log_likelihood)

    model.zero_grad(set_to_none=True)
    training_output = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_original",
    )
    (-training_output.log_likelihood.mean()).backward()
    groupopt_inactive = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if not name.startswith("native.")
    )

    result = {
        "exact_cost": exact_cost,
        "exact_log_likelihood": exact_log_likelihood,
        "exact_tour": exact_tour,
        "groupopt_parameters_inactive_in_original": groupopt_inactive,
        "official_am_root": str(Path(args.official_am_root).resolve()),
    }
    if not all((exact_tour, exact_cost, exact_log_likelihood, groupopt_inactive)):
        raise RuntimeError(f"official AM equivalence failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
