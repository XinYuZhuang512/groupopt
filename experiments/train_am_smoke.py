"""Run a small end-to-end training smoke test for the adaptive AM decoder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.training import reinforce_loss


@dataclass(frozen=True, slots=True)
class SmokeResult:
    device: str
    graph_size: int
    steps: int
    batch_size: int
    validation_size: int
    initial_greedy_cost: float
    final_greedy_cost: float
    peak_gpu_memory_gb: float

    @property
    def relative_change(self) -> float:
        return (self.final_greedy_cost - self.initial_greedy_cost) / self.initial_greedy_cost


def evaluate(model: AdaptiveAttentionModel, coordinates: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return model(coordinates, decode_type="greedy").cost.mean().item()


def run(args: argparse.Namespace) -> SmokeResult:
    if min(args.graph_size, args.steps, args.batch_size, args.validation_size) < 1:
        raise ValueError("sizes and training steps must be positive")

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AdaptiveAttentionModel(
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        feed_forward_dim=args.feed_forward_dim,
        normalization="layer",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    validation = torch.rand(args.validation_size, args.graph_size, 2, device=device)
    initial_cost = evaluate(model, validation)

    for step in range(1, args.steps + 1):
        model.train()
        coordinates = torch.rand(
            args.batch_size, args.graph_size, 2, device=device
        )
        output = model(coordinates, decode_type="sampling")
        loss = reinforce_loss(output.cost, output.log_likelihood)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            greedy_cost = evaluate(model, validation)
            print(
                f"step={step:04d} "
                f"train_cost={output.cost.mean().item():.6f} "
                f"loss={loss.item():.6f} "
                f"val_greedy_cost={greedy_cost:.6f}"
            )

    return SmokeResult(
        device=str(device),
        graph_size=args.graph_size,
        steps=args.steps,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        initial_greedy_cost=initial_cost,
        final_greedy_cost=evaluate(model, validation),
        peak_gpu_memory_gb=(
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--feed-forward-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    payload = asdict(result)
    payload["relative_change"] = result.relative_change
    print(json.dumps(payload, indent=2, sort_keys=True))
