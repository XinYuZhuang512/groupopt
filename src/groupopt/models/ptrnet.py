"""Pointer Network adapter for fixed and adaptive-base TSP construction."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.models.joint_action import PairActionScorer, decode_joint_actions
from groupopt.models.state_features import (
    ADAPTIVE_BASE_MODES,
    BASE_MODES,
    FEATURE_BASE_MODES,
    JOINT_BASE_MODES,
    select_tail_state_features,
)
from groupopt.models.tail_gate import (
    StateAwareTailGate,
    categorical_entropy,
    mix_with_fixed_tail,
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
    "gated_adaptive_state",
    "joint_fixed",
    "joint_free",
    "fixed",
]


@dataclass(frozen=True, slots=True)
class PointerNetworkOutput:
    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor
    tail_entropy: Tensor
    gate_probability: Tensor
    action_entropy: Tensor


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
        self.tail_gate = StateAwareTailGate(embedding_dim)
        self.joint_action_scorer = PairActionScorer(embedding_dim, tanh_clipping)
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
        if base_mode in JOINT_BASE_MODES:
            joint = decode_joint_actions(
                coordinates,
                node_embeddings,
                self.joint_action_scorer,
                base_mode,
                decode_type,
                anchor,
                temperature,
                generator,
            )
            zeros = torch.zeros_like(joint.cost)
            return PointerNetworkOutput(
                cost=joint.cost,
                log_likelihood=joint.log_likelihood,
                tails=joint.tails,
                heads=joint.heads,
                successor=joint.successor,
                tail_entropy=zeros,
                gate_probability=zeros,
                action_entropy=joint.action_entropy,
            )
        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        graph_embedding = node_embeddings.mean(dim=1)
        state = BatchedTSPState.initialize(coordinates)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        gate_probabilities: list[Tensor] = []

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
                gate_probability = torch.zeros(
                    state.batch_size,
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                )
                if base_mode == "gated_adaptive_state":
                    gate_probability = self.tail_gate(state, node_embeddings)
                    tail_log_p = mix_with_fixed_tail(
                        tail_log_p,
                        state.sequential_base(anchor),
                        gate_probability,
                    )
                selected_tail = _select(tail_log_p, decode_type, generator)
                selected_log_probabilities.append(
                    tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                )
                tail_entropies.append(categorical_entropy(tail_log_p))
                gate_probabilities.append(gate_probability)
            else:
                selected_tail = state.sequential_base(anchor)
                zeros = torch.zeros(
                    state.batch_size,
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                )
                tail_entropies.append(zeros)
                gate_probabilities.append(zeros)

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
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=torch.stack(gate_probabilities, dim=1).mean(dim=1),
            action_entropy=torch.zeros(
                state.batch_size,
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            ),
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
