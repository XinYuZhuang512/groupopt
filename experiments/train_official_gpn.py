"""训练官方 GPN Original 或官方 GPN + GroupOpt。"""

from __future__ import annotations

import argparse
import os

from torch import nn
from train import parse_args, run

from groupopt.models.official_gpn import OfficialGraphPointerNetworkGroupOpt


def build_official_gpn(args: argparse.Namespace) -> nn.Module:
    return OfficialGraphPointerNetworkGroupOpt(
        official_root=args.official_gpn_root,
        embedding_dim=args.embedding_dim,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "official_gpn"
    try:
        arguments.official_gpn_root = os.environ["GROUPOPT_OFFICIAL_GPN_ROOT"]
    except KeyError as error:
        raise RuntimeError("必须设置 GROUPOPT_OFFICIAL_GPN_ROOT") from error
    run(arguments, model_builder=build_official_gpn)
