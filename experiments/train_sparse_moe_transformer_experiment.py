"""Train a sparse-MoE Transformer with its sequential attention decoder."""

from __future__ import annotations

import argparse

from torch import nn
from train_am_experiment import parse_args, run

from groupopt.models.encoder_controls import ModernNativeAttentionModel


def build_model(args: argparse.Namespace) -> nn.Module:
    return ModernNativeAttentionModel(
        "sparse_moe_transformer",
        embedding_dim=args.embedding_dim,
        n_encoder_layers=args.encoder_layers,
        n_heads=args.heads,
        feed_forward_dim=args.feed_forward_dim,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "sparse_moe_transformer"
    run(arguments, model_builder=build_model)
