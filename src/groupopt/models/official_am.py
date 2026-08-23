"""把 GroupOpt 接入 Kool 等人的官方 Attention Model。

该适配器有意把宿主模型作为外部依赖保留：Original 分支直接调用官方模型的
``forward``，Ours 分支复用同一个官方 encoder、head decoder 投影和 attention
计算，只新增用于选择开放路径 tail 的轻量评分器。
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.framework.neural import ConstructionOutput
from groupopt.models.native_conditional import (
    categorical_entropy,
    joint_action_entropy,
    masked_conditional_log_probabilities,
    native_head_summary,
)
from groupopt.problems.tsp_tensor import BatchedTSPConstruction

DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal["official_original", "native_conditional_free"]
BASE_MODES = ("official_original", "native_conditional_free")


def _load_official_components(official_root: str | Path) -> tuple[type[nn.Module], object]:
    """从官方仓库加载 AM 类与 TSP 问题定义。"""
    root = Path(official_root).expanduser().resolve()
    expected = root / "nets" / "attention_model.py"
    if not expected.is_file():
        raise FileNotFoundError(f"official AM repository is missing {expected}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    attention_module = importlib.import_module("nets.attention_model")
    problems_module = importlib.import_module("problems")
    return attention_module.AttentionModel, problems_module.TSP


class OfficialAttentionModelGroupOpt(nn.Module):
    """官方 AM 与 GroupOpt 共用参数主体的双模式适配器。"""

    def __init__(
        self,
        official_root: str | Path,
        embedding_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        tanh_clipping: float = 10.0,
        normalization: Literal["batch", "layer"] = "batch",
    ) -> None:
        super().__init__()
        official_class, tsp_problem = _load_official_components(official_root)
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.native = official_class(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            problem=tsp_problem,
            n_encode_layers=n_encoder_layers,
            tanh_clipping=tanh_clipping,
            normalization=normalization,
            n_heads=n_heads,
        )
        self.process = BatchedTSPConstruction()

        # Tail Selector 使用 AM 风格的多头 glimpse，但不改动 native 中的任何层。
        self.project_tail_state = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_head_summary = nn.Linear(3, embedding_dim, bias=False)
        self.project_tail_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_tail_graph = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_tail_step = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_tail_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.first_tail_context = nn.Parameter(torch.empty(embedding_dim))
        nn.init.uniform_(self.first_tail_context, -1.0, 1.0)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        return_symmetry_embeddings: bool = False,
    ) -> ConstructionOutput:
        del anchor
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if return_symmetry_embeddings:
            raise ValueError("official AM adapter does not expose SYM-NCO embeddings yet")
        if base_mode == "official_original":
            return self._forward_official_original(coordinates, decode_type, temperature)
        return self._forward_groupopt(coordinates, decode_type, temperature, generator)

    def _forward_official_original(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
    ) -> ConstructionOutput:
        """不经过 GroupOpt 代码路径，直接返回官方 AM 的结果。"""
        self.native.set_decode_type(decode_type, temp=temperature)
        cost, log_likelihood, tour = self.native(coordinates, return_pi=True)
        heads = tour.roll(shifts=-1, dims=1)
        successor = torch.empty_like(tour)
        successor.scatter_(1, tour, heads)
        zeros = cost.new_zeros(cost.shape)
        return ConstructionOutput(
            cost=cost,
            log_likelihood=log_likelihood,
            tails=tour,
            heads=heads,
            successor=successor,
            tail_entropy=zeros,
            gate_probability=zeros,
            action_entropy=zeros,
        )

    def _forward_groupopt(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> ConstructionOutput:
        """复用官方 head decoder，为每条开放路径计算条件 head 分布。"""
        node_embeddings, graph_embedding = self.native.embedder(
            self.native._init_embed(coordinates)
        )
        batch_size, node_count, _ = node_embeddings.shape
        native_fixed = self.native._precompute(node_embeddings, num_steps=node_count)
        distances = torch.cdist(coordinates, coordinates)
        state = self.process.initial_state(coordinates)
        batch_index = torch.arange(batch_size, device=coordinates.device)
        last_head_embedding = self.first_tail_context.unsqueeze(0).expand(
            batch_size, self.embedding_dim
        )

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not self.process.is_terminal(state):
            path_features = state.path_state_features(node_embeddings)
            path_start_embedding = path_features[
                :, :, self.embedding_dim : 2 * self.embedding_dim
            ]
            native_step_context = torch.cat((path_start_embedding, node_embeddings), dim=-1)
            head_query = native_fixed.context_node_projected.expand(
                batch_size, node_count, self.embedding_dim
            ) + self.native.project_step_context(native_step_context)

            pair_mask = self.process.action_mask(state)
            valid_tail = ~pair_mask.all(dim=-1)
            safe_pair_mask = pair_mask & valid_tail.unsqueeze(-1)
            head_logits, _ = self.native._one_to_many_logits(
                head_query,
                native_fixed.glimpse_key,
                native_fixed.glimpse_val,
                native_fixed.logit_key,
                safe_pair_mask,
            )
            head_log_p = masked_conditional_log_probabilities(
                head_logits, pair_mask, temperature
            )
            summary = native_head_summary(head_log_p, distances).detach()

            tail_nodes = (
                node_embeddings
                + self.project_tail_state(path_features)
                + self.project_head_summary(summary)
            )
            tail_logits = self._tail_logits(
                tail_nodes,
                graph_embedding,
                last_head_embedding,
                self.process.base_mask(state),
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
            state = self.process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings[batch_index, selected_head]

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = coordinates.new_zeros(batch_size)
        return ConstructionOutput(
            cost=self.process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=self.process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=zeros,
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _tail_logits(
        self,
        node_embeddings: Tensor,
        graph_embedding: Tensor,
        last_head_embedding: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """按照官方 AM 的 glimpse/logit 结构给候选 tail 打分。"""
        batch_size, node_count, _ = node_embeddings.shape
        key, value, logit_key = self.project_tail_nodes(node_embeddings).chunk(3, dim=-1)
        head_dim = self.embedding_dim // self.n_heads
        key = key.view(batch_size, node_count, self.n_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, node_count, self.n_heads, head_dim).permute(
            0, 2, 1, 3
        )
        query = self.project_tail_graph(graph_embedding) + self.project_tail_step(
            last_head_embedding
        )
        query = query.view(batch_size, self.n_heads, 1, head_dim)
        compatibility = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        compatibility = compatibility.masked_fill(mask[:, None, None, :], -torch.inf)
        glimpse = torch.matmul(torch.softmax(compatibility, dim=-1), value)
        glimpse = glimpse.transpose(1, 2).reshape(batch_size, 1, self.embedding_dim)
        glimpse = self.project_tail_out(glimpse)
        logits = torch.matmul(glimpse, logit_key.transpose(-2, -1)).squeeze(1)
        logits = logits / math.sqrt(self.embedding_dim)
        if self.native.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.native.tanh_clipping
        return logits.masked_fill(mask, -torch.inf)


def _select(
    log_probabilities: Tensor,
    decode_type: DecodeType,
    generator: torch.Generator | None,
) -> Tensor:
    if decode_type == "greedy":
        return log_probabilities.argmax(dim=-1)
    if decode_type == "sampling":
        return torch.multinomial(
            log_probabilities.exp(), 1, generator=generator
        ).squeeze(1)
    raise ValueError(f"unknown decode type: {decode_type}")
