"""Train the LayerNorm Transformer control with the shared joint decoder."""

from __future__ import annotations

import argparse

from torch import nn
from train_am_experiment import parse_args, run

from groupopt.models.am import AdaptiveAttentionModel


def build_transformer_ln(args: argparse.Namespace) -> nn.Module:
    return AdaptiveAttentionModel(
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        feed_forward_dim=args.feed_forward_dim,
        normalization="layer",
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "transformer_ln"
    arguments.normalization = "layer"
    run(arguments, model_builder=build_transformer_ln)
