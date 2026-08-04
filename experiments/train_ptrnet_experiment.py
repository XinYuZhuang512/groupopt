"""Train one fixed-base or adaptive-base Pointer Network experiment."""

from __future__ import annotations

import argparse

from torch import nn

from groupopt.models.ptrnet import AdaptivePointerNetwork

from train_am_experiment import parse_args, run


def build_ptrnet(args: argparse.Namespace) -> nn.Module:
    return AdaptivePointerNetwork(
        embedding_dim=args.embedding_dim,
        n_encoder_layers=args.encoder_layers,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "ptrnet"
    run(arguments, model_builder=build_ptrnet)
