"""Pointer Network adapter for fixed and adaptive-base TSP construction."""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.framework.neural import BatchedConstructionProcess, ConstructionOutput
from groupopt.models.joint_action import PairActionScorer, decode_joint_actions
from groupopt.models.native_conditional import (
    joint_action_entropy,
    masked_conditional_log_probabilities,
    native_head_summary,
)
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
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState

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
    "native_fixed",
    "native_free",
    "native_conditional_fixed",
    "native_conditional_free",
    "fixed",
]


PointerNetworkOutput = ConstructionOutput


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
        logits = self.logits(query, candidates, mask)
        return torch.log_softmax(logits / temperature, dim=-1)

    def logits(self, query: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        """Return native pointer logits for one or many conditional queries."""
        squeeze_query = query.ndim == 2
        if squeeze_query:
            query = query.unsqueeze(1)
            mask = mask.unsqueeze(1)
        if query.ndim != 3 or mask.ndim != 3:
            raise ValueError("query and mask must describe one or more queries")
        if candidates.ndim == 3:
            candidates = candidates.unsqueeze(1)
        if candidates.ndim != 4:
            raise ValueError(
                "candidates must have shape (batch, nodes, dim) or (batch, queries, nodes, dim)"
            )
        compatibility = torch.tanh(
            self.project_nodes(candidates) + self.project_query(query).unsqueeze(2)
        )
        logits = torch.matmul(compatibility, self.pointer)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        return logits.squeeze(1) if squeeze_query else logits


class AdaptivePointerNetwork(nn.Module):
    """LSTM Pointer Network with optional learned adaptive base selection."""

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encoder_layers: int = 1,
        tanh_clipping: float = 10.0,
        construction_process: BatchedConstructionProcess[
            Tensor, BatchedTSPState
        ] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or n_encoder_layers < 1:
            raise ValueError("embedding_dim and n_encoder_layers must be positive")

        self.embedding_dim = embedding_dim
        self.construction_process = construction_process or BatchedTSPConstruction()
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
        self.project_tail_state = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_native_head_summary = nn.Linear(3, embedding_dim, bias=False)
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
                self.construction_process,
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
        if base_mode == "native_free":
            return self._decode_native_free(
                coordinates,
                node_embeddings,
                encoder_hidden[-1],
                encoder_cell[-1],
                decode_type,
                temperature,
                generator,
            )
        if base_mode == "native_conditional_free":
            return self._decode_native_conditional_free(
                coordinates,
                node_embeddings,
                encoder_hidden[-1],
                encoder_cell[-1],
                decode_type,
                temperature,
                generator,
            )
        if base_mode == "native_fixed":
            base_mode = "fixed"
        if base_mode == "native_conditional_fixed":
            base_mode = "fixed"
        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        graph_embedding = node_embeddings.mean(dim=1)
        process = self.construction_process
        state = process.initial_state(coordinates)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        gate_probabilities: list[Tensor] = []

        while not process.is_terminal(state):
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
                    tail_query, tail_candidates, process.base_mask(state), temperature
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
                        process.fixed_base(state, anchor),
                        gate_probability,
                    )
                selected_tail = _select(tail_log_p, decode_type, generator)
                selected_log_probabilities.append(
                    tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                )
                tail_entropies.append(categorical_entropy(tail_log_p))
                gate_probabilities.append(gate_probability)
            else:
                selected_tail = process.fixed_base(state, anchor)
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
                process.representative_mask(state, selected_tail),
                temperature,
            )
            selected_head = _select(head_log_p, decode_type, generator)
            selected_log_probabilities.append(
                head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )

            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            head_embedding = _gather_nodes(node_embeddings, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        return PointerNetworkOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=torch.stack(gate_probabilities, dim=1).mean(dim=1),
            action_entropy=torch.zeros(
                state.batch_size,
                dtype=node_embeddings.dtype,
                device=node_embeddings.device,
            ),
        )

    def _decode_native_free(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> PointerNetworkOutput:
        """Lift the native PtrNet head pointer over all legal tail choices."""
        graph_embedding = node_embeddings.mean(dim=1)
        process = self.construction_process
        state = process.initial_state(coordinates)
        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not process.is_terminal(state):
            tail_candidates = node_embeddings + self.project_tail_state(
                select_tail_state_features("adaptive_state", state, node_embeddings)
            )
            tail_query = self.project_tail_context(
                torch.cat((graph_embedding, decoder_hidden), dim=-1)
            )
            tail_logits = self.tail_pointer.logits(
                tail_query, tail_candidates, process.base_mask(state)
            )

            expanded_graph = graph_embedding.unsqueeze(1).expand_as(node_embeddings)
            expanded_hidden = decoder_hidden.unsqueeze(1).expand_as(node_embeddings)
            head_queries = self.project_head_context(
                torch.cat((expanded_graph, expanded_hidden, node_embeddings), dim=-1)
            )
            pair_mask = process.action_mask(state)
            head_logits = self.head_pointer.logits(head_queries, node_embeddings, pair_mask)
            pair_logits = (tail_logits.unsqueeze(2) + head_logits).masked_fill(
                pair_mask, -torch.inf
            )
            flat_log_p = torch.log_softmax(pair_logits.flatten(1) / temperature, dim=1)
            action_log_p = flat_log_p.reshape_as(pair_logits)
            selected_action = _select(flat_log_p, decode_type, generator)
            selected_tail = torch.div(selected_action, state.n, rounding_mode="floor")
            selected_head = selected_action.remainder(state.n)

            selected_log_probabilities.append(
                flat_log_p.gather(1, selected_action[:, None]).squeeze(1)
            )
            tail_entropies.append(categorical_entropy(torch.logsumexp(action_log_p, dim=2)))
            action_entropies.append(categorical_entropy(flat_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            head_embedding = _gather_nodes(node_embeddings, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return PointerNetworkOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=zeros,
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _decode_native_conditional_free(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> PointerNetworkOutput:
        """Add base scheduling while preserving PtrNet head conditionals."""
        graph_embedding = node_embeddings.mean(dim=1)
        distances = torch.cdist(coordinates, coordinates)
        process = self.construction_process
        state = process.initial_state(coordinates)
        batch_index = torch.arange(state.batch_size, device=coordinates.device)
        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not process.is_terminal(state):
            expanded_graph = graph_embedding.unsqueeze(1).expand_as(node_embeddings)
            expanded_hidden = decoder_hidden.unsqueeze(1).expand_as(node_embeddings)
            head_queries = self.project_head_context(
                torch.cat((expanded_graph, expanded_hidden, node_embeddings), dim=-1)
            )
            pair_mask = process.action_mask(state)
            head_logits = self.head_pointer.logits(head_queries, node_embeddings, pair_mask)
            head_log_p = masked_conditional_log_probabilities(head_logits, pair_mask, temperature)
            summary = native_head_summary(head_log_p, distances).detach()

            tail_candidates = (
                node_embeddings
                + self.project_tail_state(
                    select_tail_state_features("adaptive_state", state, node_embeddings)
                )
                + self.project_native_head_summary(summary)
            )
            tail_query = self.project_tail_context(
                torch.cat((graph_embedding, decoder_hidden), dim=-1)
            )
            tail_logits = self.tail_pointer.logits(
                tail_query, tail_candidates, process.base_mask(state)
            )
            tail_log_p = torch.log_softmax(tail_logits / temperature, dim=-1)
            selected_tail = _select(tail_log_p, decode_type, generator)
            selected_head_log_p = head_log_p[batch_index, selected_tail]
            selected_head = _select(selected_head_log_p, decode_type, generator)

            selected_log_probabilities.append(
                tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                + selected_head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )
            tail_entropies.append(categorical_entropy(tail_log_p))
            action_entropies.append(joint_action_entropy(tail_log_p, head_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            head_embedding = _gather_nodes(node_embeddings, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return PointerNetworkOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=zeros,
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
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
