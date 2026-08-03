"""Training utilities shared by model adapters."""

from __future__ import annotations

from torch import Tensor


def reinforce_loss(cost: Tensor, log_likelihood: Tensor) -> Tensor:
    """Return a batch-mean REINFORCE objective with a centered batch baseline."""
    if cost.ndim != 1 or log_likelihood.shape != cost.shape:
        raise ValueError("cost and log_likelihood must be vectors with equal shape")
    advantage = (cost - cost.mean()).detach()
    return (advantage * log_likelihood).mean()
