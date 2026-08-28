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
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState

DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal[
    "official_original",
    "official_conditional_fixed",
    "official_forest_fixed",
    "native_conditional_free",
    "official_groupopt_global_anchor",
    "official_groupopt_graph_tail",
    "official_groupopt_forest_context",
    "official_groupopt_amstyle_interface",
    "official_amstyle_fixed",
    "official_amstyle_free",
    "official_amstyle_forest_fixed",
    "official_amstyle_free_no_head_summary",
    "official_amstyle_free_no_path_state",
    "official_amstyle_free_no_last_head",
    "official_groupopt_edge_native",
    "official_groupopt_edge_native_detached",
    "official_edge_native_fixed",
    "official_edge_native_free",
    "official_capacity_single_chain",
    "official_nested_fixed",
    "official_nested_free",
    "official_hybrid_fixed",
    "official_hybrid_free",
]
BASE_MODES = (
    "official_original",
    "official_conditional_fixed",
    "official_forest_fixed",
    "native_conditional_free",
    "official_groupopt_global_anchor",
    "official_groupopt_graph_tail",
    "official_groupopt_forest_context",
    "official_groupopt_amstyle_interface",
    "official_amstyle_fixed",
    "official_amstyle_free",
    "official_amstyle_forest_fixed",
    "official_amstyle_free_no_head_summary",
    "official_amstyle_free_no_path_state",
    "official_amstyle_free_no_last_head",
    "official_groupopt_edge_native",
    "official_groupopt_edge_native_detached",
    "official_edge_native_fixed",
    "official_edge_native_free",
    "official_capacity_single_chain",
    "official_nested_fixed",
    "official_nested_free",
    "official_hybrid_fixed",
    "official_hybrid_free",
)
HeadContextMode = Literal[
    "local_path_start",
    "global_anchor",
    "graph_tail",
    "forest_context",
    "amstyle_interface",
    "edge_native",
    "edge_native_detached",
]


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
        # Forest State Adapter 只改变送入原生 decoder 的状态表示，不改变其
        # multi-head glimpse、pointer logits、mask 或 softmax 结构。
        forest_feature_dim = 3 * embedding_dim + 1
        self.project_decoder_tail_state = nn.Linear(forest_feature_dim, embedding_dim, bias=False)
        self.project_decoder_head_state = nn.Linear(
            forest_feature_dim, 3 * embedding_dim, bias=False
        )
        self.project_decoder_progress = nn.Linear(1, embedding_dim, bias=False)
        # 零初始化保证加入适配器时与 graph-tail 行为完全一致，再由训练逐渐
        # 学会利用 Forest 信息，避免随机残差破坏原生 decoder 的初始策略。
        nn.init.zeros_(self.project_decoder_tail_state.weight)
        nn.init.zeros_(self.project_decoder_head_state.weight)
        nn.init.zeros_(self.project_decoder_progress.weight)
        # 诊断接口：在官方 encoder 上复现旧 AM-style 的专用
        # (graph, tail) -> head scorer，以定位原生 head 接口是否为瓶颈。
        self.project_amstyle_graph = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_amstyle_head_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_amstyle_head_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_amstyle_head_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # 显式多路径 decoder 接口。它不再把 Forest 状态伪装成官方 AM 的
        # “首节点/当前节点”二元上下文，而是分别投影全局、tail、局部路径、
        # 上一步 head 与进度；head 侧也显式接收其所属路径状态。
        edge_query_dim = 6 * embedding_dim + 2
        edge_head_dim = 4 * embedding_dim + 1
        self.project_edge_query = nn.Linear(edge_query_dim, embedding_dim, bias=False)
        self.project_edge_head = nn.Linear(edge_head_dim, 3 * embedding_dim, bias=False)
        self.project_edge_tail_state = nn.Linear(forest_feature_dim, embedding_dim, bias=False)
        self.project_edge_pair = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False),
        )
        # 单链容量对照使用一个参数量与 GroupOpt 活跃接口严格相等的残差 MLP。
        # 最后一层与补齐参数均从零开始，因此初始前向严格退化为原生单链 decoder。
        groupopt_interface_parameters = 27 * embedding_dim**2 + 10 * embedding_dim + 288
        capacity_hidden = groupopt_interface_parameters // (5 * embedding_dim + 2)
        capacity_remainder = groupopt_interface_parameters - capacity_hidden * (
            5 * embedding_dim + 2
        )
        self.project_capacity_in = nn.Linear(4 * embedding_dim + 1, capacity_hidden)
        self.project_capacity_out = nn.Linear(capacity_hidden, embedding_dim, bias=False)
        self.capacity_extra = nn.Parameter(torch.zeros(capacity_remainder))
        nn.init.zeros_(self.project_capacity_out.weight)
        # Native+Forest Hybrid：官方 query 永远作为主干，Forest 仅产生一个
        # 零初始化残差。因而 residual=0 时 Fixed 与官方单链逐动作一致。
        hybrid_feature_dim = 6 * embedding_dim + 2
        self.project_hybrid_in = nn.Linear(hybrid_feature_dim, embedding_dim)
        self.project_hybrid_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.project_hybrid_out.weight)
        self.tanh_clipping = tanh_clipping
        self._initialize_edge_native_from_official()

    def _initialize_edge_native_from_official(self) -> None:
        """让显式接口在初始化时退化为官方 graph-tail 指针。

        新增的 Forest 与边特征通道从零开始，官方 graph/tail/node 投影则被
        原样复制。这样首轮差异来自训练学到的信息，而不是随机接口破坏。
        """
        embedding_dim = self.embedding_dim
        with torch.no_grad():
            self.project_edge_query.weight.zero_()
            self.project_edge_query.weight[:, :embedding_dim].copy_(
                self.native.project_fixed_context.weight
                + self.native.project_step_context.weight[:, :embedding_dim]
            )
            self.project_edge_query.weight[:, embedding_dim : 2 * embedding_dim].copy_(
                self.native.project_step_context.weight[:, embedding_dim:]
            )
            self.project_edge_head.weight.zero_()
            self.project_edge_head.weight[:, :embedding_dim].copy_(
                self.native.project_node_embeddings.weight
            )
            self.project_edge_tail_state.weight.zero_()
            self.project_edge_tail_state.weight[:, : 2 * embedding_dim].copy_(
                self.project_tail_state.weight[:, : 2 * embedding_dim]
            )
            self.project_edge_tail_state.weight[:, -1:].copy_(
                self.project_tail_state.weight[:, -1:]
            )
            self.project_edge_pair[-1].weight.zero_()

    def initialize_edge_native_single_chain(self) -> None:
        """把 edge-native head 接口初始化为官方锚定单链 decoder。

        与默认的 graph-tail 初始化不同，这里把局部路径起点通道映射到官方
        decoder 的 first-node 上下文。只要 tail 仍沿锚定分量延伸，head logits
        就与“强制首节点为 anchor”的官方 AM 逐步一致；Forest 其余通道仍从零
        开始，后续可通过课程训练逐渐启用。
        """
        embedding_dim = self.embedding_dim
        self._initialize_edge_native_from_official()
        with torch.no_grad():
            self.project_edge_query.weight.zero_()
            self.project_edge_query.weight[:, :embedding_dim].copy_(
                self.native.project_fixed_context.weight
            )
            self.project_edge_query.weight[:, embedding_dim : 2 * embedding_dim].copy_(
                self.native.project_step_context.weight[:, embedding_dim:]
            )
            # 输入布局为 graph、tail、component mean、path start、path end、
            # size、last head、progress；把 path start 接到官方 first-node 通道。
            self.project_edge_query.weight[:, 3 * embedding_dim : 4 * embedding_dim].copy_(
                self.native.project_step_context.weight[:, :embedding_dim]
            )

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        return_symmetry_embeddings: bool = False,
        tail_free_probability: float = 1.0,
    ) -> ConstructionOutput:
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= tail_free_probability <= 1.0:
            raise ValueError("tail_free_probability must be in [0, 1]")
        if return_symmetry_embeddings:
            raise ValueError("official AM adapter does not expose SYM-NCO embeddings yet")
        if base_mode == "official_original":
            return self._forward_official_original(coordinates, decode_type, temperature)
        if base_mode == "official_capacity_single_chain":
            return self._forward_capacity_single_chain(
                coordinates, decode_type, temperature, generator, anchor
            )
        if base_mode in ("official_nested_fixed", "official_nested_free"):
            return self._forward_nested_schedule(
                coordinates,
                decode_type,
                temperature,
                generator,
                learned_tail=base_mode == "official_nested_free",
            )
        if base_mode in ("official_hybrid_fixed", "official_hybrid_free"):
            return self._forward_hybrid_schedule(
                coordinates,
                decode_type,
                temperature,
                generator,
                learned_tail=base_mode == "official_hybrid_free",
            )
        return self._forward_groupopt(
            coordinates,
            decode_type,
            temperature,
            generator,
            tail_mode=(
                "anchor"
                if base_mode
                in (
                    "official_conditional_fixed",
                    "official_amstyle_fixed",
                    "official_edge_native_fixed",
                )
                else "deterministic"
                if base_mode
                in (
                    "official_forest_fixed",
                    "official_amstyle_forest_fixed",
                )
                else "learned"
            ),
            head_context_mode=(
                "global_anchor"
                if base_mode == "official_groupopt_global_anchor"
                else "graph_tail"
                if base_mode == "official_groupopt_graph_tail"
                else "forest_context"
                if base_mode == "official_groupopt_forest_context"
                else "amstyle_interface"
                if base_mode
                in (
                    "official_groupopt_amstyle_interface",
                    "official_amstyle_fixed",
                    "official_amstyle_free",
                    "official_amstyle_forest_fixed",
                    "official_amstyle_free_no_head_summary",
                    "official_amstyle_free_no_path_state",
                    "official_amstyle_free_no_last_head",
                )
                else "edge_native"
                if base_mode == "official_groupopt_edge_native"
                else "edge_native_detached"
                if base_mode
                in (
                    "official_groupopt_edge_native_detached",
                    "official_edge_native_fixed",
                    "official_edge_native_free",
                )
                else "local_path_start"
            ),
            anchor=anchor,
            tail_free_probability=tail_free_probability,
            use_head_summary=base_mode != "official_amstyle_free_no_head_summary",
            use_path_state=base_mode != "official_amstyle_free_no_path_state",
            use_last_head=base_mode != "official_amstyle_free_no_last_head",
        )

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
        tail_mode: Literal["anchor", "deterministic", "learned"],
        head_context_mode: HeadContextMode,
        anchor: int,
        tail_free_probability: float,
        use_head_summary: bool = True,
        use_path_state: bool = True,
        use_last_head: bool = True,
    ) -> ConstructionOutput:
        """复用官方 head decoder，并可固定或学习开放路径的 tail。"""
        node_embeddings, graph_embedding = self.native.embedder(
            self.native._init_embed(coordinates)
        )
        batch_size, node_count, _ = node_embeddings.shape
        # 单步的 K/V 可以广播到所有 tail query，避免显式复制 node_count 份。
        native_fixed = self.native._precompute(node_embeddings, num_steps=1)
        distances = torch.cdist(coordinates, coordinates) if tail_mode == "learned" else None
        state = self.process.initial_state(coordinates)
        batch_index = torch.arange(batch_size, device=coordinates.device)
        last_head_embedding = self.first_tail_context.unsqueeze(0).expand(
            batch_size, self.embedding_dim
        )
        amstyle_head_data: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        if head_context_mode == "amstyle_interface":
            amstyle_graph_context = self.project_amstyle_graph(graph_embedding)
            amstyle_key, amstyle_value, amstyle_logit_key = self.project_amstyle_head_nodes(
                node_embeddings
            ).chunk(3, dim=-1)
            head_dim = self.embedding_dim // self.n_heads
            amstyle_key = amstyle_key.view(batch_size, node_count, self.n_heads, head_dim).permute(
                0, 2, 1, 3
            )
            amstyle_value = amstyle_value.view(
                batch_size, node_count, self.n_heads, head_dim
            ).permute(0, 2, 1, 3)
            amstyle_head_data = (
                amstyle_graph_context,
                amstyle_key,
                amstyle_value,
                amstyle_logit_key,
            )

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []
        free_tail_indicators: list[Tensor] = []

        while not self.process.is_terminal(state):
            path_features = state.path_state_features(node_embeddings)
            free_tail_mask: Tensor | None = None
            if tail_mode != "learned":
                if tail_mode == "anchor":
                    selected_tail = self.process.fixed_base(state, anchor)
                else:
                    # 优先选择最短分量的开放端点，得到无参数、可复现且确实
                    # 会并行维护多条局部路径的固定 Forest 调度。
                    selected_tail = _deterministic_forest_tail(
                        state,
                        self.process.base_mask(state),
                    )
                selected_tail_embedding = node_embeddings[batch_index, selected_tail]
                selected_pair_mask = self.process.representative_mask(
                    state, selected_tail
                ).unsqueeze(1)
                if head_context_mode == "amstyle_interface":
                    assert amstyle_head_data is not None
                    (
                        amstyle_graph_context,
                        amstyle_key,
                        amstyle_value,
                        amstyle_logit_key,
                    ) = amstyle_head_data
                    head_query = self.project_amstyle_head_context(
                        torch.cat((amstyle_graph_context, selected_tail_embedding), dim=-1)
                    ).unsqueeze(1)
                    selected_head_logits = self._amstyle_head_logits(
                        head_query,
                        amstyle_key,
                        amstyle_value,
                        amstyle_logit_key,
                        selected_pair_mask,
                    )
                elif head_context_mode in ("edge_native", "edge_native_detached"):
                    # 与 Free 完全共享显式多路径 head 接口，只把 tail 调度
                    # 固定为锚定单链。这样同一 checkpoint 的 Fixed/Free
                    # 唯一差异就是是否学习选择开放路径。
                    forest_features = state.forest_decoder_features(node_embeddings)
                    selected_forest = forest_features[batch_index, selected_tail]
                    progress = coordinates.new_full(
                        (batch_size, 1),
                        (node_count - state.edges_added) / node_count,
                    )
                    head_query = self.project_edge_query(
                        torch.cat(
                            (
                                graph_embedding,
                                selected_tail_embedding,
                                selected_forest,
                                last_head_embedding,
                                progress,
                            ),
                            dim=-1,
                        )
                    ).unsqueeze(1)
                    head_projection = self.project_edge_head(
                        torch.cat((node_embeddings, forest_features), dim=-1)
                    )
                    edge_key, edge_value, edge_logit_key = head_projection.chunk(3, dim=-1)
                    edge_key = self.native._make_heads(edge_key.unsqueeze(1))
                    edge_value = self.native._make_heads(edge_value.unsqueeze(1))
                    selected_head_logits, _ = self.native._one_to_many_logits(
                        head_query,
                        edge_key,
                        edge_value,
                        edge_logit_key.unsqueeze(1),
                        selected_pair_mask,
                    )
                    pair_features = self._edge_pair_features(
                        coordinates, state, forest_features, progress
                    )
                    selected_pair_bias = self.tanh_clipping * torch.tanh(
                        self.project_edge_pair(pair_features).squeeze(-1)[
                            batch_index, selected_tail
                        ]
                    )
                    selected_head_logits = selected_head_logits + selected_pair_bias.unsqueeze(1)
                else:
                    selected_start = path_features[
                        batch_index,
                        selected_tail,
                        self.embedding_dim : 2 * self.embedding_dim,
                    ]
                    native_step_context = torch.cat(
                        (selected_start, selected_tail_embedding), dim=-1
                    ).unsqueeze(1)
                    head_query = (
                        native_fixed.context_node_projected
                        + self.native.project_step_context(native_step_context)
                    )
                    selected_head_logits, _ = self.native._one_to_many_logits(
                        head_query,
                        native_fixed.glimpse_key,
                        native_fixed.glimpse_val,
                        native_fixed.logit_key,
                        selected_pair_mask,
                    )
                selected_head_log_p = masked_conditional_log_probabilities(
                    selected_head_logits, selected_pair_mask, temperature
                ).squeeze(1)
                tail_log_p = None
            else:
                dynamic_glimpse_key = native_fixed.glimpse_key
                dynamic_glimpse_val = native_fixed.glimpse_val
                dynamic_logit_key = native_fixed.logit_key
                pair_mask = self.process.action_mask(state)
                valid_tail = ~pair_mask.all(dim=-1)
                safe_pair_mask = pair_mask & valid_tail.unsqueeze(-1)
                if head_context_mode == "amstyle_interface":
                    assert amstyle_head_data is not None
                    (
                        amstyle_graph_context,
                        amstyle_key,
                        amstyle_value,
                        amstyle_logit_key,
                    ) = amstyle_head_data
                    expanded_graph_context = amstyle_graph_context.unsqueeze(1).expand(
                        batch_size, node_count, self.embedding_dim
                    )
                    head_query = self.project_amstyle_head_context(
                        torch.cat((expanded_graph_context, node_embeddings), dim=-1)
                    )
                    head_logits = self._amstyle_head_logits(
                        head_query,
                        amstyle_key,
                        amstyle_value,
                        amstyle_logit_key,
                        safe_pair_mask,
                    )
                elif head_context_mode in ("edge_native", "edge_native_detached"):
                    forest_features = state.forest_decoder_features(node_embeddings)
                    progress = coordinates.new_full(
                        (batch_size, 1),
                        (node_count - state.edges_added) / node_count,
                    )
                    expanded_graph = graph_embedding.unsqueeze(1).expand(
                        batch_size, node_count, self.embedding_dim
                    )
                    expanded_last_head = last_head_embedding.unsqueeze(1).expand(
                        batch_size, node_count, self.embedding_dim
                    )
                    expanded_progress = progress.unsqueeze(1).expand(batch_size, node_count, 1)
                    head_query = self.project_edge_query(
                        torch.cat(
                            (
                                expanded_graph,
                                node_embeddings,
                                forest_features,
                                expanded_last_head,
                                expanded_progress,
                            ),
                            dim=-1,
                        )
                    )
                    head_projection = self.project_edge_head(
                        torch.cat((node_embeddings, forest_features), dim=-1)
                    )
                    edge_key, edge_value, edge_logit_key = head_projection.chunk(3, dim=-1)
                    edge_key = self.native._make_heads(edge_key.unsqueeze(1))
                    edge_value = self.native._make_heads(edge_value.unsqueeze(1))
                    head_logits, _ = self.native._one_to_many_logits(
                        head_query,
                        edge_key,
                        edge_value,
                        edge_logit_key.unsqueeze(1),
                        safe_pair_mask,
                    )
                    pair_features = self._edge_pair_features(
                        coordinates, state, forest_features, progress
                    )
                    pair_bias = self.tanh_clipping * torch.tanh(
                        self.project_edge_pair(pair_features).squeeze(-1)
                    )
                    head_logits = head_logits + pair_bias
                elif head_context_mode == "forest_context":
                    forest_features = state.forest_decoder_features(node_embeddings)
                    progress = coordinates.new_full(
                        (batch_size, 1),
                        (node_count - state.edges_added) / node_count,
                    )
                    forest_context = graph_embedding + self.project_decoder_progress(progress)
                    tail_context_embedding = node_embeddings + self.project_decoder_tail_state(
                        forest_features
                    )
                    first_context_embedding = forest_context.unsqueeze(1).expand(
                        batch_size, node_count, self.embedding_dim
                    )
                    head_state_projection = self.project_decoder_head_state(forest_features)
                    head_key_delta, head_value_delta, logit_key_delta = head_state_projection.chunk(
                        3, dim=-1
                    )
                    dynamic_glimpse_key = dynamic_glimpse_key + self.native._make_heads(
                        head_key_delta.unsqueeze(1)
                    )
                    dynamic_glimpse_val = dynamic_glimpse_val + self.native._make_heads(
                        head_value_delta.unsqueeze(1)
                    )
                    dynamic_logit_key = dynamic_logit_key + logit_key_delta.unsqueeze(1)
                elif head_context_mode == "local_path_start":
                    first_context_embedding = path_features[
                        :, :, self.embedding_dim : 2 * self.embedding_dim
                    ]
                    tail_context_embedding = node_embeddings
                elif head_context_mode == "global_anchor":
                    first_context_embedding = node_embeddings[:, anchor : anchor + 1].expand(
                        batch_size, node_count, self.embedding_dim
                    )
                    tail_context_embedding = node_embeddings
                else:
                    first_context_embedding = graph_embedding.unsqueeze(1).expand(
                        batch_size, node_count, self.embedding_dim
                    )
                    tail_context_embedding = node_embeddings
                if head_context_mode not in (
                    "amstyle_interface",
                    "edge_native",
                    "edge_native_detached",
                ):
                    native_step_context = torch.cat(
                        (first_context_embedding, tail_context_embedding), dim=-1
                    )
                    head_query = native_fixed.context_node_projected.expand(
                        batch_size, node_count, self.embedding_dim
                    ) + self.native.project_step_context(native_step_context)
                    head_logits, _ = self.native._one_to_many_logits(
                        head_query,
                        dynamic_glimpse_key,
                        dynamic_glimpse_val,
                        dynamic_logit_key,
                        safe_pair_mask,
                    )
                head_log_p = masked_conditional_log_probabilities(
                    head_logits, pair_mask, temperature
                )
                assert distances is not None
                summary = native_head_summary(head_log_p, distances)
                if head_context_mode in ("edge_native", "edge_native_detached"):
                    if head_context_mode == "edge_native_detached":
                        # 旧 AM-style 成功配置只把条件 head 分布作为前向特征，
                        # 不让 tail loss 沿摘要反向扰乱 head scorer。
                        summary = summary.detach()
                    tail_nodes = (
                        node_embeddings
                        + self.project_edge_tail_state(forest_features)
                        + self.project_head_summary(summary)
                    )
                else:
                    # 历史模式保持原行为，避免改变既有实验定义。
                    tail_nodes = node_embeddings
                    if use_path_state:
                        tail_nodes = tail_nodes + self.project_tail_state(path_features)
                    if use_head_summary:
                        tail_nodes = tail_nodes + self.project_head_summary(summary.detach())
                tail_context = (
                    last_head_embedding if use_last_head else torch.zeros_like(last_head_embedding)
                )
                tail_logits = self._tail_logits(
                    tail_nodes,
                    graph_embedding,
                    tail_context,
                    self.process.base_mask(state),
                )
                tail_log_p = torch.log_softmax(tail_logits / temperature, dim=-1)
                learned_tail = _select(tail_log_p, decode_type, generator)
                if tail_free_probability >= 1.0:
                    free_tail_mask = torch.ones(
                        batch_size, dtype=torch.bool, device=coordinates.device
                    )
                elif tail_free_probability <= 0.0:
                    free_tail_mask = torch.zeros(
                        batch_size, dtype=torch.bool, device=coordinates.device
                    )
                else:
                    free_tail_mask = (
                        torch.rand(
                            batch_size,
                            device=coordinates.device,
                            generator=generator,
                        )
                        < tail_free_probability
                    )
                anchored_tail = self.process.fixed_base(state, anchor)
                selected_tail = torch.where(free_tail_mask, learned_tail, anchored_tail)
                selected_head_log_p = head_log_p[batch_index, selected_tail]
            selected_head = _select(selected_head_log_p, decode_type, generator)

            selected_head_log_probability = selected_head_log_p.gather(
                1, selected_head[:, None]
            ).squeeze(1)
            if tail_log_p is None:
                selected_log_probabilities.append(selected_head_log_probability)
                tail_entropies.append(coordinates.new_zeros(batch_size))
                action_entropies.append(categorical_entropy(selected_head_log_p))
            else:
                assert free_tail_mask is not None
                selected_tail_log_probability = tail_log_p.gather(
                    1, selected_tail[:, None]
                ).squeeze(1)
                # 课程阶段的锚定样本只提供稳定的原版轨迹，不更新 Adapter；
                # 否则 REINFORCE 会把已经与官方 decoder 等价的 head 接口推离
                # 初始化点。真正开放 tail 的样本才贡献完整联合策略梯度。
                selected_log_probabilities.append(
                    (selected_head_log_probability + selected_tail_log_probability) * free_tail_mask
                )
                tail_entropy = categorical_entropy(tail_log_p)
                tail_entropies.append(tail_entropy * free_tail_mask)
                action_entropies.append(
                    torch.where(
                        free_tail_mask,
                        joint_action_entropy(tail_log_p, head_log_p),
                        categorical_entropy(selected_head_log_p),
                    )
                )
            free_tail_indicators.append(
                coordinates.new_zeros(batch_size)
                if free_tail_mask is None
                else free_tail_mask.to(coordinates.dtype)
            )
            tails.append(selected_tail)
            heads.append(selected_head)
            state = self.process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings[batch_index, selected_head]

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        return ConstructionOutput(
            cost=self.process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=self.process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            gate_probability=torch.stack(free_tail_indicators, dim=1).mean(dim=1),
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _forward_capacity_single_chain(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        anchor: int,
    ) -> ConstructionOutput:
        """参数量匹配对照：零初始化残差 Adapter + 完整官方单链 rollout。"""
        del anchor
        node_embeddings, graph_embedding = self.native.embedder(
            self.native._init_embed(coordinates)
        )
        batch_size, node_count, _ = node_embeddings.shape
        native_fixed = self.native._precompute(node_embeddings, num_steps=1)
        native_state = self.native.problem.make_state(coordinates)
        last_selected_embedding = coordinates.new_zeros(batch_size, self.embedding_dim)
        selected_nodes: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        action_entropies: list[Tensor] = []

        for step in range(node_count):
            native_step_context = self.native._get_parallel_step_context(
                native_fixed.node_embeddings, native_state
            )
            native_query = native_fixed.context_node_projected + self.native.project_step_context(
                native_step_context
            )
            progress = coordinates.new_full(
                (batch_size, 1),
                (node_count - step) / node_count,
            )
            residual_hidden = torch.relu(
                self.project_capacity_in(
                    torch.cat(
                        (
                            graph_embedding,
                            native_step_context.squeeze(1),
                            last_selected_embedding,
                            progress,
                        ),
                        dim=-1,
                    )
                )
            )
            residual = self.project_capacity_out(residual_hidden)
            if self.capacity_extra.numel() > 0:
                residual = residual + self.capacity_extra.mean()
            selected_query = native_query + residual.unsqueeze(1)
            selected_mask = native_state.get_mask()
            selected_logits, _ = self.native._one_to_many_logits(
                selected_query,
                native_fixed.glimpse_key,
                native_fixed.glimpse_val,
                native_fixed.logit_key,
                selected_mask,
            )
            selected_head_log_p = masked_conditional_log_probabilities(
                selected_logits,
                selected_mask,
                temperature,
            ).squeeze(1)
            selected = _select(selected_head_log_p, decode_type, generator)
            selected_log_probabilities.append(
                selected_head_log_p.gather(1, selected[:, None]).squeeze(1)
            )
            action_entropies.append(categorical_entropy(selected_head_log_p))
            selected_nodes.append(selected)
            native_state = native_state.update(selected)
            last_selected_embedding = node_embeddings.gather(
                1,
                selected[:, None, None].expand(batch_size, 1, self.embedding_dim),
            ).squeeze(1)

        tail_tensor = torch.stack(selected_nodes, dim=1)
        head_tensor = tail_tensor.roll(shifts=-1, dims=1)
        successor = torch.empty_like(tail_tensor)
        successor.scatter_(1, tail_tensor, head_tensor)
        zeros = coordinates.new_zeros(batch_size)
        return ConstructionOutput(
            cost=native_state.get_final_cost().squeeze(-1),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=successor,
            tail_entropy=zeros,
            gate_probability=zeros,
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _forward_nested_schedule(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        learned_tail: bool,
    ) -> ConstructionOutput:
        """在完全相同的增强 head decoder 下只改变 tail 调度。

        Fixed 分支严格复现 ``official_capacity_single_chain``；Free 分支保留
        相同的 encoder、原生 attention K/V 和容量匹配 Adapter，只允许
        Tail Selector 在第一条边之后改选其他开放路径。这样两者的唯一结构
        差异就是单链调度与可学习多路径调度。
        """
        node_embeddings, graph_embedding = self.native.embedder(
            self.native._init_embed(coordinates)
        )
        batch_size, node_count, _ = node_embeddings.shape
        native_fixed = self.native._precompute(node_embeddings, num_steps=1)
        batch_index = torch.arange(batch_size, device=coordinates.device)

        # 两个分支用完全相同的官方首节点分布，避免把首节点策略混入对照。
        native_state = self.native.problem.make_state(coordinates)
        first_step_context = self.native._get_parallel_step_context(
            native_fixed.node_embeddings, native_state
        )
        first_query = native_fixed.context_node_projected + self.native.project_step_context(
            first_step_context
        )
        first_progress = coordinates.new_ones(batch_size, 1)
        first_last = coordinates.new_zeros(batch_size, self.embedding_dim)
        first_residual = self._capacity_residual(
            graph_embedding,
            first_step_context.squeeze(1),
            first_last,
            first_progress,
        )
        first_mask = native_state.get_mask()
        first_logits, _ = self.native._one_to_many_logits(
            first_query + first_residual.unsqueeze(1),
            native_fixed.glimpse_key,
            native_fixed.glimpse_val,
            native_fixed.logit_key,
            first_mask,
        )
        first_log_p = masked_conditional_log_probabilities(
            first_logits, first_mask, temperature
        ).squeeze(1)
        first_node = _select(first_log_p, decode_type, generator)

        state = self.process.initial_state(coordinates)
        last_head_embedding = node_embeddings[batch_index, first_node]
        distances = torch.cdist(coordinates, coordinates) if learned_tail else None
        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities = [first_log_p.gather(1, first_node[:, None]).squeeze(1)]
        tail_entropies: list[Tensor] = [coordinates.new_zeros(batch_size)]
        action_entropies: list[Tensor] = [categorical_entropy(first_log_p)]

        # N-1 次 head 选择把 N 个孤立点合并为一条路径；最后闭环是唯一合法动作。
        while state.edges_added < node_count - 1:
            path_features = state.path_state_features(node_embeddings)
            pair_mask = self.process.action_mask(state)
            valid_tail = ~pair_mask.all(dim=-1)
            safe_pair_mask = pair_mask & valid_tail.unsqueeze(-1)

            path_start = path_features[:, :, self.embedding_dim : 2 * self.embedding_dim]
            native_step_context = torch.cat((path_start, node_embeddings), dim=-1)
            expanded_graph = graph_embedding.unsqueeze(1).expand(
                batch_size, node_count, self.embedding_dim
            )
            expanded_last_head = last_head_embedding.unsqueeze(1).expand(
                batch_size, node_count, self.embedding_dim
            )
            progress = coordinates.new_full(
                (batch_size, 1),
                (node_count - state.edges_added - 1) / node_count,
            )
            expanded_progress = progress.unsqueeze(1).expand(batch_size, node_count, 1)
            residual_hidden = torch.relu(
                self.project_capacity_in(
                    torch.cat(
                        (
                            expanded_graph,
                            native_step_context,
                            expanded_last_head,
                            expanded_progress,
                        ),
                        dim=-1,
                    )
                )
            )
            residual = self.project_capacity_out(residual_hidden)
            if self.capacity_extra.numel() > 0:
                residual = residual + self.capacity_extra.mean()
            head_query = (
                native_fixed.context_node_projected
                + self.native.project_step_context(native_step_context)
                + residual
            )
            head_logits, _ = self.native._one_to_many_logits(
                head_query,
                native_fixed.glimpse_key,
                native_fixed.glimpse_val,
                native_fixed.logit_key,
                safe_pair_mask,
            )
            head_log_p = masked_conditional_log_probabilities(head_logits, pair_mask, temperature)

            # 第一条边固定从共同首节点出发；之后才允许 Free 改变调度。
            if learned_tail and state.edges_added > 0:
                assert distances is not None
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
                tail_probability = tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                tail_entropies.append(categorical_entropy(tail_log_p))
                action_entropies.append(joint_action_entropy(tail_log_p, head_log_p))
            else:
                selected_tail = self._batched_sequential_tail(state, first_node)
                tail_probability = coordinates.new_zeros(batch_size)
                tail_entropies.append(coordinates.new_zeros(batch_size))

            selected_head_log_p = head_log_p[batch_index, selected_tail]
            selected_head = _select(selected_head_log_p, decode_type, generator)
            head_probability = selected_head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            selected_log_probabilities.append(tail_probability + head_probability)
            if not (learned_tail and state.edges_added > 0):
                action_entropies.append(categorical_entropy(selected_head_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = self.process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings[batch_index, selected_head]

        # 此时只剩一条开放路径，闭环 tail/head 均唯一，不引入额外策略概率。
        closing_tail = self._batched_sequential_tail(state, first_node)
        closing_mask = self.process.representative_mask(state, closing_tail)
        closing_head = (~closing_mask).to(torch.long).argmax(dim=-1)
        state = self.process.transition(state, closing_tail, closing_head)
        tails.append(closing_tail)
        heads.append(closing_head)

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

    def _forward_hybrid_schedule(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        learned_tail: bool,
    ) -> ConstructionOutput:
        """官方原生 head 主干 + 可学习 Forest query residual。

        首节点始终由官方 decoder 选择。之后的每个候选 tail 都先得到官方
        ``[path_start, tail]`` query，再叠加 Forest residual。Fixed 与 Free
        共用全部 head 参数；Free 只额外学习开放路径 tail 调度。
        """
        node_embeddings, graph_embedding = self.native.embedder(
            self.native._init_embed(coordinates)
        )
        batch_size, node_count, _ = node_embeddings.shape
        native_fixed = self.native._precompute(node_embeddings, num_steps=1)
        batch_index = torch.arange(batch_size, device=coordinates.device)

        # 首节点完全沿用官方分布，不让 Forest 接口改变原生起点策略。
        native_state = self.native.problem.make_state(coordinates)
        first_step_context = self.native._get_parallel_step_context(
            native_fixed.node_embeddings, native_state
        )
        first_query = native_fixed.context_node_projected + self.native.project_step_context(
            first_step_context
        )
        first_mask = native_state.get_mask()
        first_logits, _ = self.native._one_to_many_logits(
            first_query,
            native_fixed.glimpse_key,
            native_fixed.glimpse_val,
            native_fixed.logit_key,
            first_mask,
        )
        first_log_p = masked_conditional_log_probabilities(
            first_logits, first_mask, temperature
        ).squeeze(1)
        first_node = _select(first_log_p, decode_type, generator)

        state = self.process.initial_state(coordinates)
        last_head_embedding = node_embeddings[batch_index, first_node]
        distances = torch.cdist(coordinates, coordinates) if learned_tail else None
        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities = [first_log_p.gather(1, first_node[:, None]).squeeze(1)]
        tail_entropies: list[Tensor] = [coordinates.new_zeros(batch_size)]
        action_entropies: list[Tensor] = [categorical_entropy(first_log_p)]
        interface_regularization: list[Tensor] = []

        while state.edges_added < node_count - 1:
            forest_features = state.forest_decoder_features(node_embeddings)
            path_features = torch.cat(
                (
                    forest_features[:, :, : 2 * self.embedding_dim],
                    forest_features[:, :, -1:],
                ),
                dim=-1,
            )
            pair_mask = self.process.action_mask(state)
            valid_tail = ~pair_mask.all(dim=-1)
            safe_pair_mask = pair_mask & valid_tail.unsqueeze(-1)

            path_start = forest_features[:, :, self.embedding_dim : 2 * self.embedding_dim]
            native_step_context = torch.cat((path_start, node_embeddings), dim=-1)
            base_query = native_fixed.context_node_projected + self.native.project_step_context(
                native_step_context
            )
            expanded_graph = graph_embedding.unsqueeze(1).expand(
                batch_size, node_count, self.embedding_dim
            )
            expanded_last_head = last_head_embedding.unsqueeze(1).expand(
                batch_size, node_count, self.embedding_dim
            )
            progress = coordinates.new_full(
                (batch_size, 1),
                (node_count - state.edges_added - 1) / node_count,
            )
            expanded_progress = progress.unsqueeze(1).expand(batch_size, node_count, 1)
            hybrid_features = torch.cat(
                (
                    expanded_graph,
                    node_embeddings,
                    forest_features,
                    expanded_last_head,
                    expanded_progress,
                ),
                dim=-1,
            )
            residual = self.project_hybrid_out(torch.relu(self.project_hybrid_in(hybrid_features)))

            base_logits = self._native_head_logits_efficient(
                base_query,
                native_fixed.glimpse_key,
                native_fixed.glimpse_val,
                native_fixed.logit_key,
                safe_pair_mask,
            )
            head_logits = self._native_head_logits_efficient(
                base_query + residual,
                native_fixed.glimpse_key,
                native_fixed.glimpse_val,
                native_fixed.logit_key,
                safe_pair_mask,
            )
            base_head_log_p = masked_conditional_log_probabilities(
                base_logits, pair_mask, temperature
            ).detach()
            head_log_p = masked_conditional_log_probabilities(head_logits, pair_mask, temperature)
            interface_regularization.append(
                self._conditional_kl(base_head_log_p, head_log_p, valid_tail)
            )

            if learned_tail and state.edges_added > 0:
                assert distances is not None
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
                tail_probability = tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                tail_entropies.append(categorical_entropy(tail_log_p))
                action_entropies.append(joint_action_entropy(tail_log_p, head_log_p))
            else:
                selected_tail = self._batched_sequential_tail(state, first_node)
                tail_probability = coordinates.new_zeros(batch_size)
                tail_entropies.append(coordinates.new_zeros(batch_size))

            selected_head_log_p = head_log_p[batch_index, selected_tail]
            selected_head = _select(selected_head_log_p, decode_type, generator)
            head_probability = selected_head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            selected_log_probabilities.append(tail_probability + head_probability)
            if not (learned_tail and state.edges_added > 0):
                action_entropies.append(categorical_entropy(selected_head_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = self.process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings[batch_index, selected_head]

        closing_tail = self._batched_sequential_tail(state, first_node)
        closing_mask = self.process.representative_mask(state, closing_tail)
        closing_head = (~closing_mask).to(torch.long).argmax(dim=-1)
        state = self.process.transition(state, closing_tail, closing_head)
        tails.append(closing_tail)
        heads.append(closing_head)

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
            interface_regularization=torch.stack(interface_regularization, dim=1).mean(dim=1),
        )

    @staticmethod
    def _conditional_kl(
        reference_log_p: Tensor,
        candidate_log_p: Tensor,
        valid_tail: Tensor,
    ) -> Tensor:
        """计算合法 tail 上 ``KL(reference || candidate)`` 的样本均值。"""
        finite = torch.isfinite(reference_log_p) & torch.isfinite(candidate_log_p)
        reference_probability = torch.where(
            finite, reference_log_p.exp(), torch.zeros_like(reference_log_p)
        )
        log_ratio = torch.where(
            finite,
            reference_log_p - candidate_log_p,
            torch.zeros_like(reference_log_p),
        )
        row_kl = (reference_probability * log_ratio).sum(dim=-1)
        return (row_kl * valid_tail.to(row_kl.dtype)).sum(dim=-1) / valid_tail.sum(
            dim=-1
        ).clamp_min(1)

    def _native_head_logits_efficient(
        self,
        query: Tensor,
        glimpse_key: Tensor,
        glimpse_value: Tensor,
        logit_key: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """等价复现官方 head attention，避免为每个 tail 广播复制 K/V。"""
        batch_size, query_count, _ = query.shape
        node_count = mask.size(-1)
        head_dim = self.embedding_dim // self.n_heads
        key = glimpse_key[:, :, 0].permute(1, 0, 2, 3)
        value = glimpse_value[:, :, 0].permute(1, 0, 2, 3)
        pointer_key = logit_key[:, 0]
        if key.shape != (batch_size, self.n_heads, node_count, head_dim):
            raise RuntimeError("unexpected official AM glimpse key shape")

        head_query = query.view(batch_size, query_count, self.n_heads, head_dim).permute(0, 2, 1, 3)
        compatibility = torch.matmul(head_query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        compatibility = compatibility.masked_fill(mask[:, None, :, :], -torch.inf)
        attention = torch.softmax(compatibility, dim=-1)
        glimpse = torch.matmul(attention, value)
        glimpse = glimpse.transpose(1, 2).reshape(batch_size, query_count, self.embedding_dim)
        final_query = self.native.project_out(glimpse)
        logits = torch.matmul(final_query, pointer_key.transpose(-2, -1))
        logits = logits / math.sqrt(self.embedding_dim)
        if self.native.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.native.tanh_clipping
        return logits.masked_fill(mask, -torch.inf)

    def _capacity_residual(
        self,
        graph_embedding: Tensor,
        step_context: Tensor,
        last_selected_embedding: Tensor,
        progress: Tensor,
    ) -> Tensor:
        """计算单链容量对照与嵌套对照共享的残差 Adapter。"""
        hidden = torch.relu(
            self.project_capacity_in(
                torch.cat(
                    (
                        graph_embedding,
                        step_context,
                        last_selected_embedding,
                        progress,
                    ),
                    dim=-1,
                )
            )
        )
        residual = self.project_capacity_out(hidden)
        if self.capacity_extra.numel() > 0:
            residual = residual + self.capacity_extra.mean()
        return residual

    @staticmethod
    def _batched_sequential_tail(state: object, anchors: Tensor) -> Tensor:
        """返回每个样本各自锚定分量的唯一开放尾点。"""
        batch_index = torch.arange(anchors.size(0), device=anchors.device)
        anchor_component = state.component[batch_index, anchors].unsqueeze(1)
        candidates = (state.component == anchor_component) & (state.successor < 0)
        if not torch.all(candidates.sum(dim=1) == 1):
            raise RuntimeError("each anchored component must have exactly one open tail")
        return candidates.to(torch.long).argmax(dim=1)

    @staticmethod
    def _edge_pair_features(
        coordinates: Tensor,
        state: object,
        forest_features: Tensor,
        progress: Tensor,
    ) -> Tensor:
        """构造每个合法候选边 ``(tail, head)`` 的显式关系特征。"""
        tail_coordinates = coordinates.unsqueeze(2)
        head_coordinates = coordinates.unsqueeze(1)
        delta = head_coordinates - tail_coordinates
        distance = delta.norm(p=2, dim=-1, keepdim=True)
        normalized_size = forest_features[:, :, -1:]
        tail_size = normalized_size.unsqueeze(2).expand(-1, -1, coordinates.size(1), -1)
        head_size = normalized_size.unsqueeze(1).expand(-1, coordinates.size(1), -1, -1)
        component = state.component
        same_component = (
            (component.unsqueeze(2) == component.unsqueeze(1)).to(coordinates.dtype).unsqueeze(-1)
        )
        pair_progress = (
            progress[:, None, :].expand(-1, coordinates.size(1), coordinates.size(1)).unsqueeze(-1)
        )
        return torch.cat(
            (
                delta,
                distance,
                tail_size,
                head_size,
                same_component,
                pair_progress,
            ),
            dim=-1,
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
        value = value.view(batch_size, node_count, self.n_heads, head_dim).permute(0, 2, 1, 3)
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

    def _amstyle_head_logits(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logit_key: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """复现旧 AM-style 的条件 head attention，用作宿主接口诊断。"""
        batch_size, query_count, _ = query.shape
        head_dim = self.embedding_dim // self.n_heads
        head_query = query.view(batch_size, query_count, self.n_heads, head_dim).permute(0, 2, 1, 3)
        compatibility = torch.matmul(head_query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        fully_masked = mask.all(dim=-1, keepdim=True)
        safe_mask = mask & ~fully_masked
        compatibility = compatibility.masked_fill(safe_mask[:, None, :, :], -torch.inf)
        attention = torch.softmax(compatibility, dim=-1)
        glimpse = torch.matmul(attention, value)
        glimpse = glimpse.transpose(1, 2).reshape(batch_size, query_count, self.embedding_dim)
        glimpse = self.project_amstyle_head_glimpse(glimpse)
        logits = torch.matmul(glimpse, logit_key.transpose(-2, -1))
        logits = logits / math.sqrt(self.embedding_dim)
        if self.native.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.native.tanh_clipping
        return logits.masked_fill(mask, -torch.inf)


def _deterministic_forest_tail(
    state: BatchedTSPState,
    tail_mask: Tensor,
) -> Tensor:
    """按分量规模和节点编号确定固定 Forest 的下一个开放端点。"""
    node_count = tail_mask.size(1)
    indices = torch.arange(node_count, device=tail_mask.device).expand_as(tail_mask)
    component_size_by_label = torch.zeros_like(state.component).scatter_add(
        1,
        state.component,
        torch.ones_like(state.component),
    )
    component_size = component_size_by_label.gather(1, state.component)
    priority = component_size * (node_count + 1) + indices
    invalid_priority = (node_count + 1) ** 2
    selected = priority.masked_fill(tail_mask, invalid_priority).argmin(dim=1)
    if tail_mask.gather(1, selected[:, None]).any():
        raise ValueError("a nonterminal Forest state has no legal tail")
    return selected


def _select(
    log_probabilities: Tensor,
    decode_type: DecodeType,
    generator: torch.Generator | None,
) -> Tensor:
    if decode_type == "greedy":
        return log_probabilities.argmax(dim=-1)
    if decode_type == "sampling":
        return torch.multinomial(log_probabilities.exp(), 1, generator=generator).squeeze(1)
    raise ValueError(f"unknown decode type: {decode_type}")
