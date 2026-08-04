"""Attention Model-style scorer for adaptive-base TSP construction.

The encoder follows the graph self-attention pattern of Kool et al. The decoder uses
two attention decisions per construction step: select a tail (base), then select a
legal head (representative). Feasibility is entirely supplied by ``BatchedTSPState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.problems.tsp_tensor import BatchedTSPState


DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal["adaptive", "adaptive_state", "fixed"]


@dataclass(frozen=True, slots=True)
class AttentionModelOutput:
    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor


class _Normalization(nn.Module):
    def __init__(self, embedding_dim: int, kind: Literal["batch", "layer"]) -> None:
        super().__init__()
        if kind == "batch":
            self.normalizer: nn.Module = nn.BatchNorm1d(embedding_dim)
        elif kind == "layer":
            self.normalizer = nn.LayerNorm(embedding_dim)
        else:
            raise ValueError(f"unknown normalization: {kind}")

    def forward(self, values: Tensor) -> Tensor:
        if isinstance(self.normalizer, nn.BatchNorm1d):
            batch, nodes, embedding = values.shape
            return self.normalizer(values.reshape(batch * nodes, embedding)).reshape(
                batch, nodes, embedding
            )
        return self.normalizer(values)


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        feed_forward_dim: int,
        normalization: Literal["batch", "layer"],
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim, n_heads, batch_first=True, bias=True
        )
        self.attention_norm = _Normalization(embedding_dim, normalization)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.feed_forward_norm = _Normalization(embedding_dim, normalization)

    def forward(self, values: Tensor) -> Tensor:
        attended, _ = self.attention(values, values, values, need_weights=False)
        values = self.attention_norm(values + attended)
        return self.feed_forward_norm(values + self.feed_forward(values))


class _GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        feed_forward_dim: int,
        normalization: Literal["batch", "layer"],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            _EncoderLayer(
                embedding_dim,
                n_heads,
                feed_forward_dim,
                normalization,
            )
            for _ in range(n_layers)
        )

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.input_projection(coordinates)
        for layer in self.layers:
            nodes = layer(nodes)
        return nodes, nodes.mean(dim=1)


class AdaptiveAttentionModel(nn.Module):
    """AM-style encoder with a two-stage adaptive-base attention decoder."""

    def __init__(
        self,
        embedding_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        feed_forward_dim: int = 512,
        tanh_clipping: float = 10.0,
        normalization: Literal["batch", "layer"] = "batch",
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.encoder = _GraphAttentionEncoder(
            embedding_dim,
            n_heads,
            n_encoder_layers,
            feed_forward_dim,
            normalization,
        )

        self.project_graph = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_tail_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_head_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_tail_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_head_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_tail_state = nn.Linear(
            2 * embedding_dim + 1, embedding_dim, bias=False
        )
        self.project_state_tail_nodes = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.first_step_context = nn.Parameter(torch.empty(embedding_dim))
        nn.init.uniform_(self.first_step_context, -1.0, 1.0)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "adaptive",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> AttentionModelOutput:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in ("adaptive", "adaptive_state", "fixed"):
            raise ValueError(f"unknown base mode: {base_mode}")

        node_embeddings, graph_embedding = self.encoder(coordinates)
        state = BatchedTSPState.initialize(coordinates)
        graph_context = self.project_graph(graph_embedding)
        key, value, logit_key = self.project_nodes(node_embeddings).chunk(3, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)

        last_head_embedding = self.first_step_context.unsqueeze(0).expand(
            state.batch_size, self.embedding_dim
        )
        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []

        while not state.terminal:
            if base_mode in ("adaptive", "adaptive_state"):
                tail_query = self.project_tail_context(
                    torch.cat((graph_context, last_head_embedding), dim=-1)
                )
                tail_key, tail_value, tail_logit_key = key, value, logit_key
                if base_mode == "adaptive_state":
                    state_tail_nodes = self._state_aware_tail_embeddings(
                        node_embeddings, state
                    )
                    tail_key, tail_value, tail_logit_key = self.project_state_tail_nodes(
                        state_tail_nodes
                    ).chunk(3, dim=-1)
                    tail_key = self._split_heads(tail_key)
                    tail_value = self._split_heads(tail_value)
                tail_log_p = self._attention_log_probabilities(
                    tail_query,
                    tail_key,
                    tail_value,
                    tail_logit_key,
                    state.tail_mask(),
                    self.project_tail_glimpse,
                    temperature,
                )
                selected_tail = _select(tail_log_p, decode_type, generator)
                selected_log_probabilities.append(
                    tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                )
            else:
                selected_tail = state.sequential_base(anchor)

            tail_embedding = node_embeddings.gather(
                1,
                selected_tail[:, None, None].expand(
                    state.batch_size, 1, self.embedding_dim
                ),
            ).squeeze(1)
            head_query = self.project_head_context(
                torch.cat((graph_context, tail_embedding), dim=-1)
            )
            head_log_p = self._attention_log_probabilities(
                head_query,
                key,
                value,
                logit_key,
                state.head_mask(selected_tail),
                self.project_head_glimpse,
                temperature,
            )
            selected_head = _select(head_log_p, decode_type, generator)

            selected_log_probabilities.append(
                head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )
            tails.append(selected_tail)
            heads.append(selected_head)
            state = state.update(selected_tail, selected_head)
            last_head_embedding = node_embeddings.gather(
                1,
                selected_head[:, None, None].expand(
                    state.batch_size, 1, self.embedding_dim
                ),
            ).squeeze(1)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        log_likelihood = torch.stack(selected_log_probabilities, dim=1).sum(dim=1)
        return AttentionModelOutput(
            cost=state.edge_cost(tail_tensor, head_tensor),
            log_likelihood=log_likelihood,
            tails=tail_tensor,
            heads=head_tensor,
            successor=state.successor,
        )

    def _state_aware_tail_embeddings(
        self, node_embeddings: Tensor, state: BatchedTSPState
    ) -> Tensor:
        """Attach the current open-path structure to every candidate tail.

        A path is summarized by its start endpoint, the mean embedding of its
        vertices, and its normalized size. Together with the candidate tail's own
        embedding, this exposes both endpoints and the current component geometry.
        """
        component_index = state.component.unsqueeze(-1)
        embedding_index = component_index.expand_as(node_embeddings)
        component_sum_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_index, node_embeddings
        )
        component_sum = component_sum_by_label.gather(1, embedding_index)
        ones = torch.ones_like(component_index, dtype=node_embeddings.dtype)
        component_size_by_label = torch.zeros_like(ones).scatter_add(
            1, component_index, ones
        )
        component_size = component_size_by_label.gather(1, component_index)
        component_mean = component_sum / component_size

        is_path_start = state.predecessor < 0
        start_source = node_embeddings * is_path_start.unsqueeze(-1)
        path_start_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_index, start_source
        )
        path_start = path_start_by_label.gather(1, embedding_index)
        normalized_size = component_size / state.n
        state_features = torch.cat(
            (component_mean, path_start, normalized_size), dim=-1
        )
        return node_embeddings + self.project_tail_state(state_features)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch, nodes, _ = values.shape
        head_dim = self.embedding_dim // self.n_heads
        return values.reshape(batch, nodes, self.n_heads, head_dim).permute(0, 2, 1, 3)

    def _attention_log_probabilities(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logit_key: Tensor,
        mask: Tensor,
        glimpse_projection: nn.Linear,
        temperature: float,
    ) -> Tensor:
        batch = query.size(0)
        head_dim = self.embedding_dim // self.n_heads
        head_query = query.reshape(batch, self.n_heads, 1, head_dim)
        compatibility = torch.matmul(head_query, key.transpose(-2, -1)) / sqrt(head_dim)
        compatibility = compatibility.masked_fill(mask[:, None, None, :], -torch.inf)
        attention = torch.softmax(compatibility, dim=-1)
        glimpse = torch.matmul(attention, value)
        glimpse = glimpse.transpose(1, 2).reshape(batch, 1, self.embedding_dim)
        glimpse = glimpse_projection(glimpse)

        logits = torch.matmul(glimpse, logit_key.transpose(-2, -1)).squeeze(1)
        logits = logits / sqrt(self.embedding_dim)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        return torch.log_softmax(logits / temperature, dim=-1)


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
