"""Evaluate one trained checkpoint on a fixed independent TSP test set."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.models.gpn import AdaptiveGraphPointerNetwork
from groupopt.models.ptrnet import AdaptivePointerNetwork


def build_model(config: dict[str, Any]) -> nn.Module:
    model_name = config.get("model", "am")
    if model_name == "ptrnet":
        return AdaptivePointerNetwork(
            embedding_dim=int(config["embedding_dim"]),
            n_encoder_layers=int(config["encoder_layers"]),
        )
    if model_name == "gpn":
        return AdaptiveGraphPointerNetwork(
            embedding_dim=int(config["embedding_dim"]),
            n_encoder_layers=int(config["encoder_layers"]),
        )
    return AdaptiveAttentionModel(
        embedding_dim=int(config["embedding_dim"]),
        n_heads=int(config["heads"]),
        n_encoder_layers=int(config["encoder_layers"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        normalization=config["normalization"],
    )


def load_model(
    checkpoint_path: Path, config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)

    allowed_missing: set[str] = set()
    if config["base_mode"] != "gated_adaptive_state":
        allowed_missing.update(
            {
                "tail_gate.projection.weight",
                "tail_gate.projection.bias",
            }
        )
    if config["base_mode"] not in ("joint_fixed", "joint_free"):
        allowed_missing.update(
            key
            for key in model.state_dict()
            if key.startswith("joint_action_scorer.")
        )
    if config.get("model", "am") == "am" and config["base_mode"] in (
        "fixed",
        "adaptive",
    ):
        allowed_missing.update(
            {
                "project_tail_state.weight",
                "project_state_tail_nodes.weight",
            }
        )
    if (
        not set(incompatible.missing_keys).issubset(allowed_missing)
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model, checkpoint


def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["graph_size"]) != args.graph_size:
        raise ValueError("test graph size differs from checkpoint training graph size")

    device = resolve_device(args.device)
    model, checkpoint = load_model(checkpoint_path, config, device)
    generator = torch.Generator(device="cpu").manual_seed(args.test_seed)
    coordinates = torch.rand(
        args.test_size,
        args.graph_size,
        2,
        generator=generator,
        device="cpu",
    )

    costs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, args.test_size, args.batch_size):
            batch = coordinates[start : start + args.batch_size].to(device)
            output = model(
                batch,
                decode_type="greedy",
                base_mode=config["base_mode"],
            )
            costs.append(output.cost.cpu())
    cost_tensor = torch.cat(costs)
    if cost_tensor.shape != (args.test_size,) or not torch.isfinite(cost_tensor).all():
        raise RuntimeError("independent test produced invalid costs")

    standard_deviation = cost_tensor.std(unbiased=True).item()
    summary = {
        "base_mode": config["base_mode"],
        "best_validation_cost": float(checkpoint["best_cost"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "graph_size": args.graph_size,
        "mean_cost": cost_tensor.mean().item(),
        "model": config.get("model", "am"),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(args.test_size),
        "test_seed": args.test_seed,
        "test_size": args.test_size,
        "train_seed": int(config["seed"]),
    }
    atomic_torch_save(
        {
            "costs": cost_tensor,
            "metadata": summary,
        },
        output_dir / "costs.pt",
    )
    atomic_text_write(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        output_dir / "summary.json",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def atomic_text_write(text: str, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=20260805)
    parser.add_argument("--graph-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if min(arguments.test_size, arguments.graph_size, arguments.batch_size) < 1:
        parser.error("test size, graph size, and batch size must be positive")
    return arguments


if __name__ == "__main__":
    evaluate(parse_args())
