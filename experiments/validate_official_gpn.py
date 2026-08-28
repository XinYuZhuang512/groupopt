"""验证官方 GPN 原生性、Fixed 等价性与 GroupOpt 参数隔离。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groupopt.models.official_gpn import OfficialGraphPointerNetworkGroupOpt


def select(log_p: torch.Tensor, decode_type: str, generator: torch.Generator) -> torch.Tensor:
    if decode_type == "greedy":
        return log_p.argmax(dim=-1)
    return torch.multinomial(log_p.exp(), 1, generator=generator).squeeze(-1)


def official_reference(
    model: OfficialGraphPointerNetworkGroupOpt,
    coordinates: torch.Tensor,
    decode_type: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, node_count, _ = coordinates.shape
    mask = coordinates.new_zeros(batch_size, node_count)
    current = coordinates[:, 0]
    hidden = cell = None
    tour = []
    log_probabilities = []
    generator = torch.Generator(device=coordinates.device).manual_seed(seed)
    for _ in range(node_count):
        probability, hidden, cell, _ = model.native(
            x=current,
            X_all=coordinates,
            h=hidden,
            c=cell,
            mask=mask,
        )
        log_p = torch.log(probability + 1e-15)
        selected = select(log_p, decode_type, generator)
        tour.append(selected)
        log_probabilities.append(log_p.gather(1, selected[:, None]).squeeze(1))
        current = coordinates.gather(
            1, selected[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        mask = mask.scatter(1, selected[:, None], -torch.inf)
    tour_tensor = torch.stack(tour, dim=1)
    ordered = coordinates.gather(1, tour_tensor.unsqueeze(-1).expand(-1, -1, 2))
    cost = (ordered - ordered.roll(-1, 1)).norm(dim=-1).sum(dim=-1)
    likelihood = torch.stack(log_probabilities, dim=1).sum(dim=1)
    return tour_tensor, cost, likelihood


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(20260831)
    model = OfficialGraphPointerNetworkGroupOpt(
        args.official_root, embedding_dim=32
    ).to(device)
    model.eval()
    checks: dict[str, bool] = {}

    for node_count in (2, 12):
        coordinates = torch.rand(4, node_count, 2, device=device)
        for decode_type in ("greedy", "sampling"):
            seed = 20260901 + node_count
            with torch.no_grad():
                reference_tour, reference_cost, reference_ll = official_reference(
                    model, coordinates, decode_type, seed
                )
                original = model(
                    coordinates,
                    decode_type=decode_type,
                    base_mode="official_gpn_original",
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
                fixed = model(
                    coordinates,
                    decode_type=decode_type,
                    base_mode="official_gpn_fixed",
                    generator=torch.Generator(device=device).manual_seed(seed),
                )
            prefix = f"n{node_count}_{decode_type}"
            checks[f"{prefix}_official_tour"] = torch.equal(
                reference_tour, original.tails
            )
            checks[f"{prefix}_official_cost"] = torch.equal(
                reference_cost, original.cost
            )
            checks[f"{prefix}_official_ll"] = torch.equal(
                reference_ll, original.log_likelihood
            )
            checks[f"{prefix}_fixed_tour"] = torch.equal(
                original.tails, fixed.tails
            )
            checks[f"{prefix}_fixed_cost"] = torch.allclose(
                original.cost, fixed.cost, atol=1e-6, rtol=1e-6
            )
            checks[f"{prefix}_fixed_ll"] = torch.allclose(
                original.log_likelihood,
                fixed.log_likelihood,
                atol=1e-6,
                rtol=1e-6,
            )

    # 官方 cuDNN LSTM 只允许在 training mode 保留反传所需缓存；该模型没有
    # dropout/BatchNorm，因此切换模式不会改变前向数值。
    model.train()
    model.zero_grad(set_to_none=True)
    coordinates = torch.rand(4, 12, 2, device=device)
    original = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_gpn_original",
        generator=torch.Generator(device=device).manual_seed(20260902),
    )
    (-original.log_likelihood.mean()).backward()
    tail_prefixes = (
        "project_tail_state.",
        "project_head_summary.",
        "project_tail_query.",
        "tail_pointer.",
    )
    checks["original_tail_parameters_inactive"] = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if name.startswith(tail_prefixes)
    )

    model.zero_grad(set_to_none=True)
    free = model(
        coordinates,
        decode_type="sampling",
        base_mode="official_gpn_free",
        generator=torch.Generator(device=device).manual_seed(20260903),
    )
    (-free.log_likelihood.mean()).backward()
    checks["free_tail_parameters_active"] = any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad) > 0)
        for name, parameter in model.named_parameters()
        if name.startswith(tail_prefixes)
    )
    sorted_successor = free.successor.sort(dim=1).values
    expected = torch.arange(12, device=device).expand_as(sorted_successor)
    checks["free_successor_permutation"] = torch.equal(sorted_successor, expected)
    checks["free_finite"] = bool(
        torch.isfinite(free.cost).all() and torch.isfinite(free.log_likelihood).all()
    )

    if not all(checks.values()):
        raise RuntimeError(f"official GPN validation failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))


if __name__ == "__main__":
    main()
