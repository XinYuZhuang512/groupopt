"""SYM-NCO's problem-symmetry objective for Euclidean AM training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class SymNCOLoss:
    total: Tensor
    policy: Tensor
    similarity: Tensor
    invariance_penalty: Tensor


class SymmetryProjectionHead(nn.Sequential):
    """Training-only projection head used by the official AM implementation."""

    def __init__(self, embedding_dim: int) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        super().__init__(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )


def augment_euclidean_symmetries(
    coordinates: Tensor,
    factor: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return factor-first random rotations/reflections of each TSP instance."""
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (batch, nodes, 2)")
    if factor < 1:
        raise ValueError("symmetry factor must be positive")
    if factor == 1:
        return coordinates

    batch_size = coordinates.shape[0]
    centered = coordinates - 0.5
    angles = 2.0 * torch.pi * torch.rand(
        factor - 1,
        batch_size,
        1,
        device=coordinates.device,
        dtype=coordinates.dtype,
        generator=generator,
    )
    cosines = angles.cos()
    sines = angles.sin()
    x = centered[..., 0].unsqueeze(0)
    y = centered[..., 1].unsqueeze(0)
    rotated_x = x * cosines - y * sines
    rotated_y = x * sines + y * cosines
    transformed = torch.stack((rotated_x, rotated_y), dim=-1)

    reflect = torch.rand(
        factor - 1,
        batch_size,
        1,
        1,
        device=coordinates.device,
        generator=generator,
    ) < 0.5
    transformed = torch.where(reflect, transformed.flip(-1), transformed) + 0.5
    return torch.cat((coordinates.unsqueeze(0), transformed), dim=0).flatten(0, 1)


def symnco_am_loss(
    cost: Tensor,
    log_likelihood: Tensor,
    node_embeddings: Tensor,
    factor: int,
    alpha: float = 0.1,
) -> SymNCOLoss:
    """Compute the official AM-style SYM-NCO problem-symmetry objective."""
    if factor < 2:
        raise ValueError("SYM-NCO training requires a symmetry factor of at least two")
    if alpha < 0:
        raise ValueError("SYM-NCO alpha must be non-negative")
    if cost.ndim != 1 or log_likelihood.shape != cost.shape:
        raise ValueError("cost and log_likelihood must be equal one-dimensional tensors")
    if cost.shape[0] % factor != 0:
        raise ValueError("sample count must be divisible by the symmetry factor")
    if node_embeddings.ndim != 3 or node_embeddings.shape[0] != cost.shape[0]:
        raise ValueError(
            "node_embeddings must have shape (samples, nodes, embedding_dim)"
        )

    batch_size = cost.shape[0] // factor
    grouped_cost = cost.reshape(factor, batch_size).transpose(0, 1)
    grouped_log_likelihood = log_likelihood.reshape(factor, batch_size).transpose(0, 1)
    advantage = (grouped_cost - grouped_cost.mean(dim=1, keepdim=True)).detach()
    policy = (advantage * grouped_log_likelihood).mean()

    grouped_embeddings = node_embeddings.reshape(
        factor,
        batch_size,
        node_embeddings.shape[1],
        node_embeddings.shape[2],
    )
    reference = grouped_embeddings[0].unsqueeze(0)
    similarity = F.cosine_similarity(
        reference,
        grouped_embeddings[1:],
        dim=-1,
        eps=1e-8,
    ).mean()
    invariance_penalty = 1.0 - similarity
    return SymNCOLoss(
        total=policy - alpha * similarity,
        policy=policy,
        similarity=similarity,
        invariance_penalty=invariance_penalty,
    )
