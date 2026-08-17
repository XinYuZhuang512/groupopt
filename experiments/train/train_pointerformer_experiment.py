"""Train the Pointerformer-style encoder with the shared joint decoder."""

from __future__ import annotations

import argparse

from torch import nn
from train_am_experiment import parse_args, run

from groupopt.models.encoder_controls import JointEncoderModel


def build_pointerformer(args: argparse.Namespace) -> nn.Module:
    return JointEncoderModel(
        "pointerformer",
        embedding_dim=args.embedding_dim,
        n_encoder_layers=args.encoder_layers,
        n_heads=args.heads,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "pointerformer"
    run(arguments, model_builder=build_pointerformer)
