"""Train the sparse mixture-of-experts Transformer control."""

from __future__ import annotations

import argparse

from torch import nn
from train_am_experiment import parse_args, run

from groupopt.models.encoder_controls import JointEncoderModel


def build_moe_transformer(args: argparse.Namespace) -> nn.Module:
    return JointEncoderModel(
        "moe_transformer",
        embedding_dim=args.embedding_dim,
        n_encoder_layers=args.encoder_layers,
        n_heads=args.heads,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "moe_transformer"
    run(arguments, model_builder=build_moe_transformer)
