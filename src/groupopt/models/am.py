"""Attention Model-style scorer for adaptive-base TSP construction.

The encoder follows the graph self-attention pattern of Kool et al. The decoder uses
two attention decisions per construction step: select a tail (base), then select a
legal head (representative). Feasibility is entirely supplied by ``BatchedTSPState``.
"""

from __future__ import annotations

from dataclasses import replace
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


AttentionModelOutput = ConstructionOutput


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
        self.attention = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True, bias=True)
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
        construction_process: BatchedConstructionProcess[
            Tensor, BatchedTSPState
        ] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.construction_process = construction_process or BatchedTSPConstruction()
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
        self.project_tail_state = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_state_tail_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_native_head_summary = nn.Linear(3, embedding_dim, bias=False)
        self.tail_gate = StateAwareTailGate(embedding_dim)
        self.joint_action_scorer = PairActionScorer(embedding_dim, tanh_clipping)
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
        return_symmetry_embeddings: bool = False,
    ) -> AttentionModelOutput:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")

        node_embeddings, graph_embedding = self.encoder(coordinates)
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
            output = AttentionModelOutput(
                cost=joint.cost,
                log_likelihood=joint.log_likelihood,
                tails=joint.tails,
                heads=joint.heads,
                successor=joint.successor,
                tail_entropy=zeros,
                gate_probability=zeros,
                action_entropy=joint.action_entropy,
            )
            return self._with_symmetry_embeddings(
                output, node_embeddings, return_symmetry_embeddings
            )
        if base_mode == "native_free":
            output = self._decode_native_free(
                coordinates,
                node_embeddings,
                graph_embedding,
                decode_type,
                anchor,
                temperature,
                generator,
            )
            return self._with_symmetry_embeddings(
                output, node_embeddings, return_symmetry_embeddings
            )
        if base_mode == "native_conditional_free":
            output = self._decode_native_conditional_free(
                coordinates,
                node_embeddings,
                graph_embedding,
                decode_type,
                temperature,
                generator,
            )
            return self._with_symmetry_embeddings(
                output, node_embeddings, return_symmetry_embeddings
            )
        if base_mode == "native_fixed":
            # With one legal tail its additive base score cancels exactly.  Use
            # the original fixed decoder directly instead of materializing all
            # conditional tail-head rows.
            base_mode = "fixed"
        if base_mode == "native_conditional_fixed":
            base_mode = "fixed"
        process = self.construction_process
        state = process.initial_state(coordinates)
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
        tail_entropies: list[Tensor] = []
        gate_probabilities: list[Tensor] = []

        while not process.is_terminal(state):
            if base_mode in ADAPTIVE_BASE_MODES:
                tail_query = self.project_tail_context(
                    torch.cat((graph_context, last_head_embedding), dim=-1)
                )
                tail_key, tail_value, tail_logit_key = key, value, logit_key
                if base_mode in FEATURE_BASE_MODES:
                    state_tail_nodes = self._state_aware_tail_embeddings(
                        node_embeddings, state, base_mode
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
                    process.base_mask(state),
                    self.project_tail_glimpse,
                    temperature,
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

            tail_embedding = node_embeddings.gather(
                1,
                selected_tail[:, None, None].expand(state.batch_size, 1, self.embedding_dim),
            ).squeeze(1)
            head_query = self.project_head_context(
                torch.cat((graph_context, tail_embedding), dim=-1)
            )
            head_log_p = self._attention_log_probabilities(
                head_query,
                key,
                value,
                logit_key,
                process.representative_mask(state, selected_tail),
                self.project_head_glimpse,
                temperature,
            )
            selected_head = _select(head_log_p, decode_type, generator)

            selected_log_probabilities.append(
                head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )
            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings.gather(
                1,
                selected_head[:, None, None].expand(state.batch_size, 1, self.embedding_dim),
            ).squeeze(1)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        log_likelihood = torch.stack(selected_log_probabilities, dim=1).sum(dim=1)
        output = AttentionModelOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=log_likelihood,
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
        return self._with_symmetry_embeddings(
            output, node_embeddings, return_symmetry_embeddings
        )

    def _with_symmetry_embeddings(
        self,
        output: AttentionModelOutput,
        node_embeddings: Tensor,
        include: bool,
    ) -> AttentionModelOutput:
        if not include:
            return output
        return replace(output, symmetry_node_embeddings=node_embeddings)

    def _decode_native_free(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        graph_embedding: Tensor,
        decode_type: DecodeType,
        anchor: int,
        temperature: float,
        generator: torch.Generator | None,
    ) -> AttentionModelOutput:
        """Lift the native AM head decoder to legal ``(tail, head)`` actions.

        The original AM glimpse and compatibility projections score every head
        conditional on every candidate tail.  A state-aware native AM tail score
        is added before one softmax over legal pairs.  The parameter-matched
        ``native_fixed`` control is exactly the original fixed-tail decoder,
        because an additive score for its sole legal tail cancels in softmax.
        """
        process = self.construction_process
        state = process.initial_state(coordinates)
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
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not process.is_terminal(state):
            state_tail_nodes = self._state_aware_tail_embeddings(
                node_embeddings, state, "adaptive_state"
            )
            tail_key, tail_value, tail_logit_key = self.project_state_tail_nodes(
                state_tail_nodes
            ).chunk(3, dim=-1)
            tail_query = self.project_tail_context(
                torch.cat((graph_context, last_head_embedding), dim=-1)
            )
            tail_logits = self._attention_logits(
                tail_query,
                self._split_heads(tail_key),
                self._split_heads(tail_value),
                tail_logit_key,
                process.base_mask(state),
                self.project_tail_glimpse,
            )

            expanded_graph = graph_context.unsqueeze(1).expand(
                state.batch_size, state.n, self.embedding_dim
            )
            head_queries = self.project_head_context(
                torch.cat((expanded_graph, node_embeddings), dim=-1)
            )
            pair_mask = process.action_mask(state)
            head_logits = self._attention_logits(
                head_queries,
                key,
                value,
                logit_key,
                pair_mask,
                self.project_head_glimpse,
            )
            pair_logits = tail_logits.unsqueeze(2) + head_logits
            pair_logits = pair_logits.masked_fill(pair_mask, -torch.inf)
            flat_log_p = torch.log_softmax(pair_logits.flatten(1) / temperature, dim=1)
            action_log_p = flat_log_p.reshape_as(pair_logits)
            selected_action = _select(flat_log_p, decode_type, generator)
            selected_tail = torch.div(selected_action, state.n, rounding_mode="floor")
            selected_head = selected_action.remainder(state.n)

            selected_log_probabilities.append(
                flat_log_p.gather(1, selected_action[:, None]).squeeze(1)
            )
            tail_log_p = torch.logsumexp(action_log_p, dim=2)
            tail_entropies.append(categorical_entropy(tail_log_p))
            action_entropies.append(categorical_entropy(flat_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings.gather(
                1,
                selected_head[:, None, None].expand(state.batch_size, 1, self.embedding_dim),
            ).squeeze(1)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return AttentionModelOutput(
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
        graph_embedding: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> AttentionModelOutput:
        """Select a tail without changing the native conditional head policy."""
        process = self.construction_process
        state = process.initial_state(coordinates)
        graph_context = self.project_graph(graph_embedding)
        key, value, logit_key = self.project_nodes(node_embeddings).chunk(3, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)
        distances = torch.cdist(coordinates, coordinates)
        last_head_embedding = self.first_step_context.unsqueeze(0).expand(
            state.batch_size, self.embedding_dim
        )
        batch_index = torch.arange(state.batch_size, device=coordinates.device)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not process.is_terminal(state):
            pair_mask = process.action_mask(state)
            expanded_graph = graph_context.unsqueeze(1).expand(
                state.batch_size, state.n, self.embedding_dim
            )
            head_queries = self.project_head_context(
                torch.cat((expanded_graph, node_embeddings), dim=-1)
            )
            head_logits = self._attention_logits(
                head_queries,
                key,
                value,
                logit_key,
                pair_mask,
                self.project_head_glimpse,
            )
            head_log_p = masked_conditional_log_probabilities(head_logits, pair_mask, temperature)
            summary = native_head_summary(head_log_p, distances).detach()

            state_tail_nodes = self._state_aware_tail_embeddings(
                node_embeddings, state, "adaptive_state"
            ) + self.project_native_head_summary(summary)
            tail_key, tail_value, tail_logit_key = self.project_state_tail_nodes(
                state_tail_nodes
            ).chunk(3, dim=-1)
            tail_query = self.project_tail_context(
                torch.cat((graph_context, last_head_embedding), dim=-1)
            )
            tail_logits = self._attention_logits(
                tail_query,
                self._split_heads(tail_key),
                self._split_heads(tail_value),
                tail_logit_key,
                process.base_mask(state),
                self.project_tail_glimpse,
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
            last_head_embedding = node_embeddings.gather(
                1,
                selected_head[:, None, None].expand(state.batch_size, 1, self.embedding_dim),
            ).squeeze(1)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return AttentionModelOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=zeros,
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _state_aware_tail_embeddings(
        self,
        node_embeddings: Tensor,
        state: BatchedTSPState,
        base_mode: str = "adaptive_state",
    ) -> Tensor:
        """Attach the current open-path structure to every candidate tail.

        A path is summarized by its start endpoint, the mean embedding of its
        vertices, and its normalized size. Together with the candidate tail's own
        embedding, this exposes both endpoints and the current component geometry.
        """
        return node_embeddings + self.project_tail_state(
            select_tail_state_features(base_mode, state, node_embeddings)
        )

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
        logits = self._attention_logits(query, key, value, logit_key, mask, glimpse_projection)
        return torch.log_softmax(logits / temperature, dim=-1)

    def _attention_logits(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logit_key: Tensor,
        mask: Tensor,
        glimpse_projection: nn.Linear,
    ) -> Tensor:
        """Return native AM logits for one or many conditional queries."""
        squeeze_query = query.ndim == 2
        if squeeze_query:
            query = query.unsqueeze(1)
            mask = mask.unsqueeze(1)
        if query.ndim != 3 or mask.ndim != 3:
            raise ValueError("query and mask must describe one or more queries")
        batch, query_count, _ = query.shape
        if mask.shape != (batch, query_count, logit_key.size(1)):
            raise ValueError("attention mask has an incompatible shape")

        head_dim = self.embedding_dim // self.n_heads
        head_query = query.reshape(batch, query_count, self.n_heads, head_dim).permute(0, 2, 1, 3)
        compatibility = torch.matmul(head_query, key.transpose(-2, -1)) / sqrt(head_dim)
        fully_masked = mask.all(dim=-1, keepdim=True)
        safe_mask = mask & ~fully_masked
        compatibility = compatibility.masked_fill(safe_mask[:, None, :, :], -torch.inf)
        attention = torch.softmax(compatibility, dim=-1)
        glimpse = torch.matmul(attention, value)
        glimpse = glimpse.transpose(1, 2).reshape(batch, query_count, self.embedding_dim)
        glimpse = glimpse_projection(glimpse)

        logits = torch.matmul(glimpse, logit_key.transpose(-2, -1))
        logits = logits / sqrt(self.embedding_dim)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        return logits.squeeze(1) if squeeze_query else logits


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
