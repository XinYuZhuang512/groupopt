"""Encoder-family controls for the shared joint-action TSP decoder."""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.framework.neural import BatchedConstructionProcess, ConstructionOutput
from groupopt.models.am import AdaptiveAttentionModel
from groupopt.models.joint_action import PairActionScorer, decode_joint_actions
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState

DecodeType = Literal["greedy", "sampling"]
JointBaseMode = Literal["joint_fixed", "joint_free"]
EncoderType = Literal["gat", "gru", "pointerformer", "geometric", "moe_transformer"]
ModernNativeType = Literal[
    "reversible_transformer",
    "geometric_transformer",
    "sparse_moe_transformer",
]


JointEncoderOutput = ConstructionOutput


class _GRUEncoder(nn.Module):
    """Sequence encoder control matching the PtrNet encoder capacity."""

    def __init__(self, embedding_dim: int, n_layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.recurrent = nn.GRU(
            embedding_dim,
            embedding_dim,
            num_layers=n_layers,
            batch_first=True,
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        nodes, _ = self.recurrent(self.input_projection(coordinates))
        return nodes


class _GATLayer(nn.Module):
    """Multi-head additive graph attention on the complete TSP graph."""

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.project_values = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.target_attention = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        bound = 1.0 / sqrt(self.head_dim)
        nn.init.uniform_(self.source_attention, -bound, bound)
        nn.init.uniform_(self.target_attention, -bound, bound)

    def forward(self, nodes: Tensor) -> Tensor:
        batch, node_count, embedding_dim = nodes.shape
        values = self.project_values(nodes).reshape(
            batch, node_count, self.n_heads, self.head_dim
        )
        source = torch.einsum("bnhd,hd->bnh", values, self.source_attention)
        target = torch.einsum("bnhd,hd->bnh", values, self.target_attention)
        logits = torch.nn.functional.leaky_relu(
            source.unsqueeze(2) + target.unsqueeze(1), negative_slope=0.2
        )
        attention = torch.softmax(logits, dim=2)
        attended = torch.einsum("bijh,bjhd->bihd", attention, values).reshape(
            batch, node_count, embedding_dim
        )
        nodes = self.attention_norm(nodes + self.output_projection(attended))
        return self.feed_forward_norm(nodes + self.feed_forward(nodes))


class _GATEncoder(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            _GATLayer(embedding_dim, n_heads) for _ in range(n_layers)
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        nodes = self.input_projection(coordinates)
        for layer in self.layers:
            nodes = layer(nodes)
        return nodes


class _ReversibleAttentionFunction(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, bias=True
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.normalization(values)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        return attended


class _ReversibleFeedForwardFunction(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(self.normalization(values))


class _ReversibleTransformerLayer(nn.Module):
    """RevNet coupling with attention and feed-forward transformations."""

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("reversible embedding_dim must be even")
        hidden_dim = embedding_dim // 2
        reversible_heads = max(1, n_heads // 2)
        if hidden_dim % reversible_heads != 0:
            raise ValueError("half embedding_dim must be divisible by reversible heads")
        self.attention_function = _ReversibleAttentionFunction(
            hidden_dim, reversible_heads
        )
        self.feed_forward_function = _ReversibleFeedForwardFunction(hidden_dim)

    def forward(self, values: Tensor) -> Tensor:
        left, right = values.chunk(2, dim=-1)
        updated_left = left + self.attention_function(right)
        updated_right = right + self.feed_forward_function(updated_left)
        return torch.cat((updated_left, updated_right), dim=-1)


class _PointerformerStyleEncoder(nn.Module):
    """Clean-room reversible Transformer encoder inspired by Pointerformer."""

    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            _ReversibleTransformerLayer(embedding_dim, n_heads)
            for _ in range(n_layers)
        )
        self.output_normalization = nn.LayerNorm(embedding_dim)

    def forward(self, coordinates: Tensor) -> Tensor:
        nodes = self.input_projection(coordinates)
        for layer in self.layers:
            nodes = layer(nodes)
        return self.output_normalization(nodes)


class _GeometricAttentionLayer(nn.Module):
    """Self-attention with an invariant pairwise-distance bias."""

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.project_queries = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_keys = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_values = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.distance_log_scales = nn.Parameter(torch.zeros(n_heads))
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, nodes: Tensor, distances: Tensor) -> Tensor:
        batch, node_count, _ = nodes.shape
        normalized = self.attention_norm(nodes)

        def split_heads(projection: nn.Linear) -> Tensor:
            return projection(normalized).reshape(
                batch, node_count, self.n_heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(self.project_queries)
        keys = split_heads(self.project_keys)
        values = split_heads(self.project_values)
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(self.head_dim)
        distance_scales = torch.nn.functional.softplus(self.distance_log_scales)
        logits = logits - distance_scales[None, :, None, None] * distances[:, None]
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, values).transpose(1, 2).reshape(
            batch, node_count, self.embedding_dim
        )
        nodes = nodes + self.output_projection(attended)
        return nodes + self.feed_forward(self.feed_forward_norm(nodes))


class _GeometricEncoder(nn.Module):
    """Rigid-motion-invariant graph encoder based only on pairwise distances."""

    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(4, embedding_dim)
        self.layers = nn.ModuleList(
            _GeometricAttentionLayer(embedding_dim, n_heads) for _ in range(n_layers)
        )
        self.output_normalization = nn.LayerNorm(embedding_dim)

    def forward(self, coordinates: Tensor) -> Tensor:
        distances = torch.cdist(coordinates, coordinates)
        node_count = coordinates.size(1)
        diagonal = torch.eye(
            node_count, dtype=torch.bool, device=coordinates.device
        ).unsqueeze(0)
        nearest = distances.masked_fill(diagonal, torch.inf).amin(dim=-1)
        statistics = torch.stack(
            (
                distances.mean(dim=-1),
                distances.std(dim=-1, unbiased=False),
                nearest,
                distances.amax(dim=-1),
            ),
            dim=-1,
        )
        nodes = self.input_projection(statistics)
        for layer in self.layers:
            nodes = layer(nodes, distances)
        return self.output_normalization(nodes)


class _SparseMoEFeedForward(nn.Module):
    """Token-wise top-2 mixture of feed-forward experts."""

    def __init__(self, embedding_dim: int, expert_count: int = 4) -> None:
        super().__init__()
        if expert_count < 2:
            raise ValueError("expert_count must be at least two")
        self.router = nn.Linear(embedding_dim, expert_count, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.GELU(),
                nn.Linear(4 * embedding_dim, embedding_dim),
            )
            for _ in range(expert_count)
        )

    def forward(self, values: Tensor) -> Tensor:
        flat = values.flatten(0, 1)
        router_logits = self.router(flat)
        top_logits, top_experts = router_logits.topk(2, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        mixed = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            assignments = (top_experts == expert_index).nonzero(as_tuple=False)
            if assignments.numel() == 0:
                continue
            token_indices = assignments[:, 0]
            route_slots = assignments[:, 1]
            expert_values = expert(flat[token_indices])
            weighted = expert_values * top_weights[
                token_indices, route_slots
            ].unsqueeze(-1)
            mixed.index_add_(0, token_indices, weighted)
        return mixed.reshape_as(values)


class _MoETransformerLayer(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, n_heads, batch_first=True
        )
        self.moe_norm = nn.LayerNorm(embedding_dim)
        self.moe = _SparseMoEFeedForward(embedding_dim)

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        values = values + attended
        return values + self.moe(self.moe_norm(values))


class _MoETransformerEncoder(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            _MoETransformerLayer(embedding_dim, n_heads) for _ in range(n_layers)
        )
        self.output_normalization = nn.LayerNorm(embedding_dim)

    def forward(self, coordinates: Tensor) -> Tensor:
        nodes = self.input_projection(coordinates)
        for layer in self.layers:
            nodes = layer(nodes)
        return self.output_normalization(nodes)


class _EncoderWithGraphEmbedding(nn.Module):
    """Adapt a node encoder to the native attention-decoder return contract."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.encoder(coordinates)
        return nodes, nodes.mean(dim=1)


class ModernNativeAttentionModel(AdaptiveAttentionModel):
    """Modern encoder families with an untouched sequential attention decoder.

    The paired conditional-free path is inherited from the attention model and
    changes only tail selection; conditional-fixed is exactly sequential decode.
    """

    def __init__(
        self,
        model_type: ModernNativeType,
        embedding_dim: int = 128,
        n_encoder_layers: int = 3,
        n_heads: int = 8,
        feed_forward_dim: int = 512,
        tanh_clipping: float = 10.0,
        construction_process: BatchedConstructionProcess[
            Tensor, BatchedTSPState
        ] | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            feed_forward_dim=feed_forward_dim,
            tanh_clipping=tanh_clipping,
            normalization="layer",
            construction_process=construction_process,
        )
        if model_type == "reversible_transformer":
            encoder: nn.Module = _PointerformerStyleEncoder(
                embedding_dim, n_encoder_layers, n_heads
            )
        elif model_type == "geometric_transformer":
            encoder = _GeometricEncoder(embedding_dim, n_encoder_layers, n_heads)
        elif model_type == "sparse_moe_transformer":
            encoder = _MoETransformerEncoder(
                embedding_dim, n_encoder_layers, n_heads
            )
        else:
            raise ValueError(f"unknown modern native model type: {model_type}")
        self.encoder = _EncoderWithGraphEmbedding(encoder)
        self.model_type = model_type


class JointEncoderModel(nn.Module):
    """GAT or GRU encoder coupled only to the common joint-action scorer."""

    def __init__(
        self,
        encoder_type: EncoderType,
        embedding_dim: int = 128,
        n_encoder_layers: int = 3,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        construction_process: BatchedConstructionProcess[
            Tensor, BatchedTSPState
        ] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or n_encoder_layers < 1:
            raise ValueError("embedding_dim and n_encoder_layers must be positive")
        if encoder_type == "gat":
            self.encoder: nn.Module = _GATEncoder(
                embedding_dim, n_encoder_layers, n_heads
            )
        elif encoder_type == "gru":
            self.encoder = _GRUEncoder(embedding_dim, n_encoder_layers)
        elif encoder_type == "pointerformer":
            self.encoder = _PointerformerStyleEncoder(
                embedding_dim, n_encoder_layers, n_heads
            )
        elif encoder_type == "geometric":
            self.encoder = _GeometricEncoder(
                embedding_dim, n_encoder_layers, n_heads
            )
        elif encoder_type == "moe_transformer":
            self.encoder = _MoETransformerEncoder(
                embedding_dim, n_encoder_layers, n_heads
            )
        else:
            raise ValueError(f"unknown encoder type: {encoder_type}")
        self.encoder_type = encoder_type
        self.construction_process = construction_process or BatchedTSPConstruction()
        self.joint_action_scorer = PairActionScorer(embedding_dim, tanh_clipping)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: JointBaseMode = "joint_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> JointEncoderOutput:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        if base_mode not in ("joint_fixed", "joint_free"):
            raise ValueError("encoder controls support only joint_fixed and joint_free")
        node_embeddings = self.encoder(coordinates)
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
        return JointEncoderOutput(
            cost=joint.cost,
            log_likelihood=joint.log_likelihood,
            tails=joint.tails,
            heads=joint.heads,
            successor=joint.successor,
            tail_entropy=zeros,
            gate_probability=zeros,
            action_entropy=joint.action_entropy,
        )
