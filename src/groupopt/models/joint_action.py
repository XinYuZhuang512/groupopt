"""Shared joint-action decoder for stabilizer-chain TSP construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.models.tail_gate import categorical_entropy
from groupopt.problems.tsp_tensor import BatchedTSPState

DecodeType = Literal["greedy", "sampling"]


@dataclass(frozen=True, slots=True)
class JointActionOutput:
    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor
    action_entropy: Tensor


class PairActionScorer(nn.Module):
    """Score complete legal edges from contextual nodes and path state."""

    def __init__(
        self,
        embedding_dim: int,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.tanh_clipping = tanh_clipping
        state_dim = 2 * embedding_dim + 1
        self.project_tail_state = nn.Linear(state_dim, embedding_dim, bias=False)
        self.project_head_state = nn.Linear(state_dim, embedding_dim, bias=False)
        self.project_tails = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_heads = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.tail_normalization = nn.LayerNorm(embedding_dim)
        self.head_normalization = nn.LayerNorm(embedding_dim)
        self.distance_log_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        state: BatchedTSPState,
        mask: Tensor,
        temperature: float,
    ) -> Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        expected_mask_shape = (state.batch_size, state.n, state.n)
        if mask.shape != expected_mask_shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (batch, nodes, nodes)")

        state_features = state.path_state_features(node_embeddings)
        tail_nodes = self.tail_normalization(
            node_embeddings + self.project_tail_state(state_features)
        )
        head_nodes = self.head_normalization(
            node_embeddings + self.project_head_state(state_features)
        )
        tail_keys = self.project_tails(tail_nodes)
        head_keys = self.project_heads(head_nodes)
        logits = torch.matmul(tail_keys, head_keys.transpose(1, 2)) / sqrt(
            self.embedding_dim
        )
        distance_scale = torch.nn.functional.softplus(self.distance_log_scale)
        logits = logits - distance_scale * torch.cdist(coordinates, coordinates)
        if self.tanh_clipping > 0.0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        flat_logits = logits.flatten(1)
        return torch.log_softmax(flat_logits / temperature, dim=1).reshape_as(logits)


def decode_joint_actions(
    coordinates: Tensor,
    node_embeddings: Tensor,
    scorer: PairActionScorer,
    base_mode: Literal["joint_fixed", "joint_free"],
    decode_type: DecodeType,
    anchor: int,
    temperature: float,
    generator: torch.Generator | None,
) -> JointActionOutput:
    """Construct a tour with one categorical legal-edge action per step."""
    if base_mode not in ("joint_fixed", "joint_free"):
        raise ValueError(f"unsupported joint base mode: {base_mode}")

    state = BatchedTSPState.initialize(coordinates)
    tails: list[Tensor] = []
    heads: list[Tensor] = []
    selected_log_probabilities: list[Tensor] = []
    action_entropies: list[Tensor] = []

    while not state.terminal:
        fixed_tail = (
            state.sequential_base(anchor) if base_mode == "joint_fixed" else None
        )
        mask = state.edge_action_mask(fixed_tail)
        action_log_p = scorer(
            coordinates,
            node_embeddings,
            state,
            mask,
            temperature,
        )
        flat_log_p = action_log_p.flatten(1)
        selected_action = _select(flat_log_p, decode_type, generator)
        selected_tail = torch.div(selected_action, state.n, rounding_mode="floor")
        selected_head = selected_action.remainder(state.n)

        selected_log_probabilities.append(
            flat_log_p.gather(1, selected_action[:, None]).squeeze(1)
        )
        action_entropies.append(categorical_entropy(flat_log_p))
        tails.append(selected_tail)
        heads.append(selected_head)
        state = state.update(selected_tail, selected_head)

    tail_tensor = torch.stack(tails, dim=1)
    head_tensor = torch.stack(heads, dim=1)
    return JointActionOutput(
        cost=state.edge_cost(tail_tensor, head_tensor),
        log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
        tails=tail_tensor,
        heads=head_tensor,
        successor=state.successor,
        action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
    )


def _select(
    log_probabilities: Tensor,
    decode_type: DecodeType,
    generator: torch.Generator | None,
) -> Tensor:
    if decode_type == "greedy":
        return log_probabilities.argmax(dim=1)
    if decode_type == "sampling":
        return torch.multinomial(
            log_probabilities.exp(), num_samples=1, generator=generator
        ).squeeze(1)
    raise ValueError(f"unknown decode type: {decode_type}")
