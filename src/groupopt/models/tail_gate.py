"""Shared conservative gate for adaptive TSP tail selection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from groupopt.problems.tsp_tensor import BatchedTSPState


class StateAwareTailGate(nn.Module):
    """Choose how much probability mass may leave the fixed tail policy."""

    def __init__(self, embedding_dim: int, initial_probability: float = 0.1) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must be strictly between zero and one")

        self.projection = nn.Linear(2 * embedding_dim + 1, 1)
        nn.init.zeros_(self.projection.weight)
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        nn.init.constant_(self.projection.bias, initial_logit)

    def forward(
        self,
        state: BatchedTSPState,
        node_embeddings: Tensor,
    ) -> Tensor:
        features = state.path_state_features(node_embeddings)
        legal = ~state.tail_mask()
        legal_count = legal.sum(dim=1, keepdim=True).clamp_min(1)
        summary = (features * legal.unsqueeze(-1)).sum(dim=1) / legal_count
        return torch.sigmoid(self.projection(summary)).squeeze(-1)


def mix_with_fixed_tail(
    adaptive_log_probabilities: Tensor,
    fixed_tail: Tensor,
    gate_probability: Tensor,
) -> Tensor:
    """Mix an adaptive categorical policy with a deterministic fixed action."""
    if adaptive_log_probabilities.ndim != 2:
        raise ValueError("adaptive_log_probabilities must have shape (batch, nodes)")
    batch_size = adaptive_log_probabilities.size(0)
    if fixed_tail.shape != (batch_size,) or fixed_tail.dtype != torch.long:
        raise ValueError("fixed_tail must be a long tensor with shape (batch,)")
    if gate_probability.shape != (batch_size,):
        raise ValueError("gate_probability must have shape (batch,)")
    if ((gate_probability < 0.0) | (gate_probability > 1.0)).any():
        raise ValueError("gate probabilities must be in [0, 1]")

    probabilities = adaptive_log_probabilities.exp() * gate_probability.unsqueeze(1)
    probabilities = probabilities.scatter_add(
        1,
        fixed_tail.unsqueeze(1),
        (1.0 - gate_probability).unsqueeze(1),
    )
    positive = probabilities > 0.0
    safe_log_probabilities = probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    ).log()
    return safe_log_probabilities.masked_fill(~positive, -torch.inf)


def categorical_entropy(log_probabilities: Tensor) -> Tensor:
    """Return entropy without producing ``0 * -inf`` at masked actions."""
    terms = torch.where(
        torch.isfinite(log_probabilities),
        log_probabilities.exp() * log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    return -terms.sum(dim=1)
