"""Graph Pointer Network adapter for the shared TSP construction process."""

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
class GraphPointerNetworkOutput:
    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor


class _CompleteGraphEmbeddingLayer(nn.Module):
    """One residual message-passing layer on the complete TSP graph."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.project_self = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_neighbors = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, nodes: Tensor) -> Tensor:
        node_count = nodes.size(1)
        neighbor_mean = (nodes.sum(dim=1, keepdim=True) - nodes) / (node_count - 1)
        self_message = self.project_self(nodes)
        neighbor_message = self.project_neighbors(neighbor_mean)
        gate = torch.sigmoid(self.gate_logit)
        update = torch.relu(gate * self_message + (1.0 - gate) * neighbor_message)
        return self.normalization(nodes + update)


class _GraphEmbeddingEncoder(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            _CompleteGraphEmbeddingLayer(embedding_dim) for _ in range(n_layers)
        )

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.input_projection(coordinates)
        for layer in self.layers:
            nodes = layer(nodes)
        return nodes, nodes.mean(dim=1)


class _PointerAttention(nn.Module):
    def __init__(self, embedding_dim: int, tanh_clipping: float) -> None:
        super().__init__()
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


class AdaptiveGraphPointerNetwork(nn.Module):
    """GPN scorer with graph messages and step-relative vector context."""

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encoder_layers: int = 3,
        tanh_clipping: float = 10.0,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or n_encoder_layers < 1:
            raise ValueError("embedding_dim and n_encoder_layers must be positive")

        self.embedding_dim = embedding_dim
        self.encoder = _GraphEmbeddingEncoder(embedding_dim, n_encoder_layers)
        self.relative_projection = nn.Linear(2, embedding_dim, bias=False)
        self.initial_hidden = nn.Linear(embedding_dim, embedding_dim)
        self.initial_cell = nn.Linear(embedding_dim, embedding_dim)
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
    ) -> GraphPointerNetworkOutput:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        if coordinates.size(1) < 2:
            raise ValueError("at least two TSP vertices are required")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")

        node_embeddings, graph_embedding = self.encoder(coordinates)
        decoder_hidden = torch.tanh(self.initial_hidden(graph_embedding))
        decoder_cell = torch.tanh(self.initial_cell(graph_embedding))
        state = BatchedTSPState.initialize(coordinates)
        last_head_coordinates = coordinates.mean(dim=1)

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
                tail_candidates = tail_candidates + self._relative_context(
                    coordinates, last_head_coordinates
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
            tail_coordinates = _gather_nodes(coordinates, selected_tail)
            head_candidates = node_embeddings + self._relative_context(
                coordinates, tail_coordinates
            )
            head_query = self.project_head_context(
                torch.cat((graph_embedding, decoder_hidden, tail_embedding), dim=-1)
            )
            head_log_p = self.head_pointer(
                head_query,
                head_candidates,
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
            last_head_coordinates = _gather_nodes(coordinates, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        return GraphPointerNetworkOutput(
            cost=state.edge_cost(tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=state.successor,
        )

    def _relative_context(self, coordinates: Tensor, current: Tensor) -> Tensor:
        return self.relative_projection(coordinates - current.unsqueeze(1))


def _gather_nodes(values: Tensor, selected: Tensor) -> Tensor:
    return values.gather(
        1, selected[:, None, None].expand(-1, 1, values.size(-1))
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
