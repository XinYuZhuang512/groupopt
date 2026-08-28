"""将 GroupOpt 接入 Ma 等人的官方 Graph Pointer Network。"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from groupopt.framework.neural import ConstructionOutput
from groupopt.models.native_conditional import (
    categorical_entropy,
    joint_action_entropy,
    native_head_summary,
)
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState

DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal[
    "official_gpn_original",
    "official_gpn_fixed",
    "official_gpn_free",
]
BASE_MODES = (
    "official_gpn_original",
    "official_gpn_fixed",
    "official_gpn_free",
)


def _load_official_gpn(root: str | Path) -> type[nn.Module]:
    source = Path(root).expanduser().resolve() / "tsp_small" / "gpn.py"
    if not source.is_file():
        raise FileNotFoundError(f"official GPN source is missing: {source}")
    spec = importlib.util.spec_from_file_location("groupopt_official_gpn", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official GPN module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GPN


class _OfficialStyleTailPointer(nn.Module):
    """沿用官方 GPN 的 additive pointer 形式给开放路径 tail 打分。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.project_candidates = nn.Linear(embedding_dim, embedding_dim)
        self.project_query = nn.Linear(embedding_dim, embedding_dim)
        self.pointer = nn.Parameter(torch.empty(embedding_dim))
        nn.init.uniform_(
            self.pointer,
            -1.0 / math.sqrt(embedding_dim),
            1.0 / math.sqrt(embedding_dim),
        )

    def forward(self, query: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        compatibility = torch.tanh(
            self.project_candidates(candidates)
            + self.project_query(query).unsqueeze(1)
        )
        logits = 10.0 * torch.tanh(torch.matmul(compatibility, self.pointer))
        return logits.masked_fill(mask, -torch.inf)


class OfficialGraphPointerNetworkGroupOpt(nn.Module):
    """官方 GPN Original 与原生 decoder 内 GroupOpt 调度的双模式模型。"""

    def __init__(self, official_root: str | Path, embedding_dim: int = 128) -> None:
        super().__init__()
        official_class = _load_official_gpn(official_root)
        # 作者实现的参数在构造时直接放到 CUDA；正式适配验证也因此在 GPU 执行。
        self.native = official_class(n_feature=2, n_hidden=embedding_dim)
        self.embedding_dim = embedding_dim
        self.process = BatchedTSPConstruction()
        forest_dim = 3 * embedding_dim + 1
        self.project_tail_state = nn.Linear(forest_dim, embedding_dim, bias=False)
        self.project_head_summary = nn.Linear(3, embedding_dim, bias=False)
        self.project_tail_query = nn.Linear(2 * embedding_dim, embedding_dim)
        self.tail_pointer = _OfficialStyleTailPointer(embedding_dim)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "official_gpn_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> ConstructionOutput:
        del anchor
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        if coordinates.size(1) < 2:
            raise ValueError("at least two TSP vertices are required")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")
        if base_mode == "official_gpn_original":
            return self._forward_original(
                coordinates, decode_type, temperature, generator
            )
        return self._forward_forest(
            coordinates,
            decode_type,
            temperature,
            generator,
            learned_tail=base_mode == "official_gpn_free",
        )

    def _forward_original(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> ConstructionOutput:
        batch_size, node_count, _ = coordinates.shape
        mask = coordinates.new_zeros(batch_size, node_count)
        current = coordinates[:, 0]
        hidden = cell = None
        tour: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        for _ in range(node_count):
            probabilities, hidden, cell, _ = self.native(
                x=current,
                X_all=coordinates,
                h=hidden,
                c=cell,
                mask=mask,
            )
            log_probabilities = self._temperature_log_probabilities(
                probabilities, temperature
            )
            selected = _select(log_probabilities, decode_type, generator)
            tour.append(selected)
            selected_log_probabilities.append(
                log_probabilities.gather(1, selected[:, None]).squeeze(1)
            )
            current = _gather(coordinates, selected)
            mask = mask.scatter(1, selected[:, None], -torch.inf)
        return _tour_output(
            coordinates, torch.stack(tour, dim=1), selected_log_probabilities
        )

    def _forward_forest(
        self,
        coordinates: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        learned_tail: bool,
    ) -> ConstructionOutput:
        batch_size, node_count, _ = coordinates.shape
        batch_index = torch.arange(batch_size, device=coordinates.device)

        initial_mask = coordinates.new_zeros(batch_size, node_count)
        initial_probabilities, hidden, cell, _ = self.native(
            x=coordinates[:, 0],
            X_all=coordinates,
            h=None,
            c=None,
            mask=initial_mask,
        )
        initial_log_p = self._temperature_log_probabilities(
            initial_probabilities, temperature
        )
        first_tail = _select(initial_log_p, decode_type, generator)
        selected_log_probabilities = [
            initial_log_p.gather(1, first_tail[:, None]).squeeze(1)
        ]

        node_embeddings = self._native_node_embeddings(coordinates)
        distances = torch.cdist(coordinates, coordinates)
        state = self.process.initial_state(coordinates)
        last_head_embedding = _gather(node_embeddings, first_tail)
        tails: list[Tensor] = []
        heads: list[Tensor] = []
        tail_entropies: list[Tensor] = [coordinates.new_zeros(batch_size)]
        action_entropies: list[Tensor] = [categorical_entropy(initial_log_p)]

        while state.edges_added < node_count - 1:
            pair_mask = self.process.action_mask(state)
            valid_tail = ~pair_mask.all(dim=-1)
            safe_pair_mask = pair_mask & valid_tail.unsqueeze(-1)
            head_probabilities, next_hidden, next_cell = self._all_tail_native_step(
                coordinates, hidden, cell, safe_pair_mask
            )
            head_log_p = self._temperature_log_probabilities(
                head_probabilities, temperature
            )

            if state.edges_added == 0:
                selected_tail = first_tail
                tail_probability = coordinates.new_zeros(batch_size)
                tail_entropies.append(coordinates.new_zeros(batch_size))
            elif learned_tail:
                summary = native_head_summary(head_log_p, distances).detach()
                forest = state.forest_decoder_features(node_embeddings)
                tail_candidates = (
                    node_embeddings
                    + self.project_tail_state(forest)
                    + self.project_head_summary(summary)
                )
                tail_query = self.project_tail_query(
                    torch.cat((hidden, last_head_embedding), dim=-1)
                )
                tail_logits = self.tail_pointer(
                    tail_query, tail_candidates, self.process.base_mask(state)
                )
                tail_log_p = torch.log_softmax(tail_logits / temperature, dim=-1)
                selected_tail = _select(tail_log_p, decode_type, generator)
                tail_probability = tail_log_p.gather(
                    1, selected_tail[:, None]
                ).squeeze(1)
                tail_entropies.append(categorical_entropy(tail_log_p))
            else:
                selected_tail = self._anchored_tail(state, first_tail)
                tail_probability = coordinates.new_zeros(batch_size)
                tail_entropies.append(coordinates.new_zeros(batch_size))

            selected_head_log_p = head_log_p[batch_index, selected_tail]
            selected_head = _select(selected_head_log_p, decode_type, generator)
            head_probability = selected_head_log_p.gather(
                1, selected_head[:, None]
            ).squeeze(1)
            selected_log_probabilities.append(tail_probability + head_probability)
            if learned_tail and state.edges_added > 0:
                action_entropies.append(joint_action_entropy(tail_log_p, head_log_p))
            else:
                action_entropies.append(categorical_entropy(selected_head_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = self.process.transition(state, selected_tail, selected_head)
            hidden = next_hidden[batch_index, selected_tail]
            cell = next_cell[batch_index, selected_tail]
            last_head_embedding = _gather(node_embeddings, selected_head)

        closing_tail = self._anchored_tail(state, first_tail)
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

    def _all_tail_native_step(
        self,
        coordinates: Tensor,
        hidden: Tensor,
        cell: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, node_count, _ = coordinates.shape
        expanded_coordinates = coordinates[:, None].expand(
            batch_size, node_count, node_count, 2
        ).reshape(batch_size * node_count, node_count, 2)
        expanded_hidden = hidden[:, None].expand(
            batch_size, node_count, self.embedding_dim
        ).reshape(batch_size * node_count, self.embedding_dim)
        expanded_cell = cell[:, None].expand_as(
            hidden[:, None].expand(batch_size, node_count, self.embedding_dim)
        ).reshape(batch_size * node_count, self.embedding_dim)
        additive_mask = coordinates.new_zeros(pair_mask.shape).masked_fill(
            pair_mask, -torch.inf
        ).reshape(batch_size * node_count, node_count)
        probabilities, next_hidden, next_cell, _ = self.native(
            x=coordinates.reshape(batch_size * node_count, 2),
            X_all=expanded_coordinates,
            h=expanded_hidden,
            c=expanded_cell,
            mask=additive_mask,
        )
        return (
            probabilities.reshape(batch_size, node_count, node_count),
            next_hidden.reshape(batch_size, node_count, self.embedding_dim),
            next_cell.reshape(batch_size, node_count, self.embedding_dim),
        )

    def _native_node_embeddings(self, coordinates: Tensor) -> Tensor:
        context = self.native.embedding_all(coordinates).reshape(-1, self.embedding_dim)
        context = self.native.r1 * self.native.W1(context) + (
            1 - self.native.r1
        ) * torch.relu(self.native.agg_1(context))
        context = self.native.r2 * self.native.W2(context) + (
            1 - self.native.r2
        ) * torch.relu(self.native.agg_2(context))
        context = self.native.r3 * self.native.W3(context) + (
            1 - self.native.r3
        ) * torch.relu(self.native.agg_3(context))
        return context.reshape(coordinates.size(0), coordinates.size(1), -1)

    @staticmethod
    def _temperature_log_probabilities(
        probabilities: Tensor, temperature: float
    ) -> Tensor:
        log_probabilities = probabilities.clamp_min(1e-15).log()
        if temperature == 1.0:
            # 与作者训练/测试脚本的 log(output + 1e-15) 完全一致。
            return log_probabilities
        return torch.log_softmax(log_probabilities / temperature, dim=-1)

    @staticmethod
    def _anchored_tail(state: BatchedTSPState, anchors: Tensor) -> Tensor:
        batch_index = torch.arange(anchors.size(0), device=anchors.device)
        component = state.component[batch_index, anchors].unsqueeze(1)
        candidates = (state.component == component) & (state.successor < 0)
        if not torch.all(candidates.sum(dim=1) == 1):
            raise RuntimeError("anchored component must have one open tail")
        return candidates.to(torch.long).argmax(dim=-1)


def _gather(values: Tensor, selected: Tensor) -> Tensor:
    return values.gather(
        1, selected[:, None, None].expand(-1, 1, values.size(-1))
    ).squeeze(1)


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
        ).squeeze(-1)
    raise ValueError(f"unknown decode type: {decode_type}")


def _tour_output(
    coordinates: Tensor,
    tour: Tensor,
    selected_log_probabilities: list[Tensor],
) -> ConstructionOutput:
    ordered = coordinates.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    cost = (ordered - ordered.roll(-1, 1)).norm(dim=-1).sum(dim=-1)
    heads = tour.roll(-1, 1)
    successor = torch.empty_like(tour)
    successor.scatter_(1, tour, heads)
    zeros = cost.new_zeros(cost.shape)
    return ConstructionOutput(
        cost=cost,
        log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
        tails=tour,
        heads=heads,
        successor=successor,
        tail_entropy=zeros,
        gate_probability=zeros,
        action_entropy=zeros,
    )
