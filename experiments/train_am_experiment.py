"""Train one fixed-base or adaptive-base AM experiment with resumable state."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from groupopt.models.am import AdaptiveAttentionModel
from groupopt.training import reinforce_loss


def evaluate(
    model: AdaptiveAttentionModel,
    coordinates: torch.Tensor,
    base_mode: str,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(coordinates, decode_type="greedy", base_mode=base_mode)
    return output.cost.mean().item()


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    config = _experiment_config(args, device)
    _write_or_validate_config(output_dir / "config.json", config, args.resume is not None)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = AdaptiveAttentionModel(
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        feed_forward_dim=args.feed_forward_dim,
        normalization=args.normalization,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    data_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    validation_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    action_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    validation = torch.rand(
        args.validation_size,
        args.graph_size,
        2,
        device=device,
        generator=validation_generator,
    )

    start_step = 0
    best_cost = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_cost = float(checkpoint["best_cost"])
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
        action_generator.set_state(checkpoint["action_generator_state"].cpu())

    metrics_path = output_dir / "metrics.jsonl"
    started_at = time.monotonic()
    if start_step == 0:
        initial_cost = evaluate(model, validation, args.base_mode)
        _append_metric(
            metrics_path,
            {
                "step": 0,
                "validation_greedy_cost": initial_cost,
                "elapsed_seconds": 0.0,
            },
        )
        best_cost = initial_cost
        initial_payload = _checkpoint_payload(
            model,
            optimizer,
            0,
            best_cost,
            config,
            data_generator,
            action_generator,
        )
        _atomic_torch_save(initial_payload, checkpoint_dir / "best.pt")
        print(f"step=000000 val_greedy_cost={initial_cost:.6f}", flush=True)

    try:
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            coordinates = torch.rand(
                args.batch_size,
                args.graph_size,
                2,
                device=device,
                generator=data_generator,
            )
            output = model(
                coordinates,
                decode_type="sampling",
                base_mode=args.base_mode,
                generator=action_generator,
            )
            loss = reinforce_loss(output.cost, output.log_likelihood)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            should_evaluate = step % args.eval_every == 0 or step == args.steps
            validation_cost: float | None = None
            is_new_best = False
            if should_evaluate:
                validation_cost = evaluate(model, validation, args.base_mode)
                is_new_best = validation_cost < best_cost
                if is_new_best:
                    best_cost = validation_cost
                metric = {
                    "step": step,
                    "train_cost": output.cost.mean().item(),
                    "reinforce_loss": loss.item(),
                    "gradient_norm": float(gradient_norm),
                    "validation_greedy_cost": validation_cost,
                    "best_validation_greedy_cost": best_cost,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "peak_gpu_memory_gb": _peak_memory_gb(device),
                }
                _append_metric(metrics_path, metric)
                print(
                    f"step={step:06d} "
                    f"train_cost={metric['train_cost']:.6f} "
                    f"loss={metric['reinforce_loss']:.6f} "
                    f"val_greedy_cost={validation_cost:.6f} "
                    f"best={best_cost:.6f} "
                    f"peak_gb={metric['peak_gpu_memory_gb']:.3f}",
                    flush=True,
                )

                if is_new_best:
                    best_payload = _checkpoint_payload(
                        model,
                        optimizer,
                        step,
                        best_cost,
                        config,
                        data_generator,
                        action_generator,
                    )
                    _atomic_torch_save(best_payload, checkpoint_dir / "best.pt")

            should_checkpoint = (
                step % args.checkpoint_every == 0 or step == args.steps
            )
            if should_checkpoint:
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    step,
                    best_cost,
                    config,
                    data_generator,
                    action_generator,
                )
                _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
                _atomic_torch_save(
                    payload, checkpoint_dir / f"step-{step:06d}.pt"
                )
    except KeyboardInterrupt:
        payload = _checkpoint_payload(
            model,
            optimizer,
            step,
            best_cost,
            config,
            data_generator,
            action_generator,
        )
        _atomic_torch_save(payload, checkpoint_dir / "interrupted.pt")
        _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        raise


def _checkpoint_payload(
    model: AdaptiveAttentionModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_cost: float,
    config: dict[str, Any],
    data_generator: torch.Generator,
    action_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_cost": best_cost,
        "config": config,
        "data_generator_state": data_generator.get_state(),
        "action_generator_state": action_generator.get_state(),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def _write_or_validate_config(
    path: Path, config: dict[str, Any], resuming: bool
) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("experiment config differs from the existing output directory")
        if not resuming:
            raise FileExistsError(
                "output directory already contains an experiment; use --resume"
            )
        return
    if resuming:
        raise ValueError("resume output directory does not contain config.json")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _experiment_config(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    excluded = {"resume", "output_dir"}
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    config["device"] = str(device)
    return config


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.graph_size,
        args.steps,
        args.batch_size,
        args.validation_size,
        args.eval_every,
        args.checkpoint_every,
    )
    if min(positive) < 1:
        raise ValueError("sizes and step intervals must be positive")
    if args.base_mode not in ("fixed", "adaptive", "adaptive_state"):
        raise ValueError("base_mode must be fixed, adaptive, or adaptive_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-mode",
        choices=("fixed", "adaptive", "adaptive_state"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--graph-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--feed-forward-dim", type=int, default=512)
    parser.add_argument("--normalization", choices=("batch", "layer"), default="batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
