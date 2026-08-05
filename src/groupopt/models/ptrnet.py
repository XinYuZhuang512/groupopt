"""Pointer Network adapter for fixed and adaptive-base TSP construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.models.state_features import (
    ADAPTIVE_BASE_MODES,
    BASE_MODES,
    FEATURE_BASE_MODES,
    select_tail_state_features,
)
from groupopt.problems.tsp_tensor import BatchedTSPState

DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal[
    "adaptive",
    "adaptive_static",
    "adaptive_state_mean",
    "adaptive_state_start",
    "adaptive_state_size",
    "adaptive_state",
    "fixed",
]


@dataclass(frozen=True, slots=True)
class PointerNetworkOutput:
    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor


class _PointerAttention(nn.Module):
    def __init__(self, embedding_dim: int, tanh_clipping: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tanh_clipping = tanh_clipping
        self.project_nodes = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pointer = nn.Parameter(torch.empty(embedding_dim))
        bound = 1.0 / sqrt(embedding_dim)
        nn.init.uniform_(self.pointer, -bound, bound)

    def forward(
        self,
        query: Tensor,
        candidates: Tensor,
        mask: Tensor,
        temperature: float,
    ) -> Tensor:
        compatibility = torch.tanh(
            self.project_nodes(candidates) + self.project_query(query).unsqueeze(1)
        )
        logits = torch.matmul(compatibility, self.pointer)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        return torch.log_softmax(logits / temperature, dim=-1)


class AdaptivePointerNetwork(nn.Module):
    """LSTM Pointer Network with optional learned adaptive base selection."""

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encoder_layers: int = 1,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or n_encoder_layers < 1:
            raise ValueError("embedding_dim and n_encoder_layers must be positive")

        self.embedding_dim = embedding_dim
        self.input_projection = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            embedding_dim,
            num_layers=n_encoder_layers,
            batch_first=True,
        )
        self.decoder_cell = nn.LSTMCell(embedding_dim, embedding_dim)
        self.project_tail_context = nn.Linear(2 * embedding_dim, embedding_dim)
        self.project_head_context = nn.Linear(3 * embedding_dim, embedding_dim)
        self.project_tail_state = nn.Linear(
            2 * embedding_dim + 1, embedding_dim, bias=False
        )
        self.tail_pointer = _PointerAttention(embedding_dim, tanh_clipping)
        self.head_pointer = _PointerAttention(embedding_dim, tanh_clipping)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "adaptive",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> PointerNetworkOutput:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")

        inputs = self.input_projection(coordinates)
        node_embeddings, (encoder_hidden, encoder_cell) = self.encoder(inputs)
        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        graph_embedding = node_embeddings.mean(dim=1)
        state = BatchedTSPState.initialize(coordinates)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []

        while not state.terminal:
            if base_mode in ADAPTIVE_BASE_MODES:
                tail_candidates = node_embeddings
                if base_mode in FEATURE_BASE_MODES:
                    tail_candidates = node_embeddings + self.project_tail_state(
                        select_tail_state_features(base_mode, state, node_embeddings)
                    )
                tail_query = self.project_tail_context(
                    torch.cat((graph_embedding, decoder_hidden), dim=-1)
                )
                tail_log_p = self.tail_pointer(
                    tail_query, tail_candidates, state.tail_mask(), temperature
                )
                selected_tail = _select(tail_log_p, decode_type, generator)
                selected_log_probabilities.append(
                    tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                )
            else:
                selected_tail = state.sequential_base(anchor)

            tail_embedding = _gather_nodes(node_embeddings, selected_tail)
            head_query = self.project_head_context(
                torch.cat((graph_embedding, decoder_hidden, tail_embedding), dim=-1)
            )
            head_log_p = self.head_pointer(
                head_query,
                node_embeddings,
                state.head_mask(selected_tail),
                temperature,
            )
            selected_head = _select(head_log_p, decode_type, generator)
            selected_log_probabilities.append(
                head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )

            tails.append(selected_tail)
            heads.append(selected_head)
            state = state.update(selected_tail, selected_head)
            head_embedding = _gather_nodes(node_embeddings, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        return PointerNetworkOutput(
            cost=state.edge_cost(tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=state.successor,
        )


def _gather_nodes(node_embeddings: Tensor, selected: Tensor) -> Tensor:
    return node_embeddings.gather(
        1, selected[:, None, None].expand(-1, 1, node_embeddings.size(-1))
    ).squeeze(1)


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
