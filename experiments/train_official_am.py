"""训练官方 AM Original 或官方 AM + GroupOpt。"""

from __future__ import annotations

import argparse
import os

from torch import nn
from train import parse_args, run

from groupopt.models.official_am import OfficialAttentionModelGroupOpt


def build_official_am(args: argparse.Namespace) -> nn.Module:
    return OfficialAttentionModelGroupOpt(
        official_root=args.official_am_root,
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        normalization=args.normalization,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "official_am"
    try:
        arguments.official_am_root = os.environ["GROUPOPT_OFFICIAL_AM_ROOT"]
    except KeyError as error:
        raise RuntimeError("必须设置 GROUPOPT_OFFICIAL_AM_ROOT") from error
    run(arguments, model_builder=build_official_am)
