"""Backward-compatible SYM-NCO import; use :mod:`groupopt.objectives`."""

from groupopt.objectives.symnco import (
    SymmetryProjectionHead,
    SymNCOLoss,
    augment_euclidean_symmetries,
    symnco_am_loss,
)

__all__ = [
    "SymNCOLoss",
    "SymmetryProjectionHead",
    "augment_euclidean_symmetries",
    "symnco_am_loss",
]
