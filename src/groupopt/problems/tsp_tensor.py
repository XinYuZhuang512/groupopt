"""Vectorized PyTorch state with the same semantics as the reference TSP process."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BatchedTSPState:
    """A synchronous batch of partial directed TSP constructions.

    ``True`` mask entries are illegal, matching the convention used by attention
    decoders. Component labels encode weakly connected path components.
    """

    coordinates: Tensor
    successor: Tensor
    predecessor: Tensor
    component: Tensor
    edges_added: int

    @classmethod
    def initialize(cls, coordinates: Tensor) -> BatchedTSPState:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        batch_size, n, _ = coordinates.shape
        if batch_size < 1 or n < 2:
            raise ValueError("a batch and at least two TSP vertices are required")

        device = coordinates.device
        return cls(
            coordinates=coordinates,
            successor=torch.full((batch_size, n), -1, dtype=torch.long, device=device),
            predecessor=torch.full((batch_size, n), -1, dtype=torch.long, device=device),
            component=torch.arange(n, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, n)
            .clone(),
            edges_added=0,
        )

    @property
    def batch_size(self) -> int:
        return self.coordinates.size(0)

    @property
    def n(self) -> int:
        return self.coordinates.size(1)

    @property
    def terminal(self) -> bool:
        return self.edges_added == self.n

    def tail_mask(self) -> Tensor:
        """Mask vertices whose outgoing edge has already been fixed."""
        if self.terminal:
            return torch.ones_like(self.successor, dtype=torch.bool)
        return self.successor >= 0

    def head_mask(self, selected_tail: Tensor) -> Tensor:
        """Mask used heads and same-component heads before the closing step."""
        self._validate_action_vector(selected_tail, "selected_tail")
        if self.terminal:
            raise ValueError("a terminal state has no head candidates")

        batch = torch.arange(self.batch_size, device=self.coordinates.device)
        if self.tail_mask()[batch, selected_tail].any():
            raise ValueError("selected_tail contains a masked vertex")

        basic_mask = self.predecessor >= 0
        if self.edges_added == self.n - 1:
            return basic_mask

        tail_component = self.component.gather(1, selected_tail[:, None])
        same_component = self.component == tail_component
        return basic_mask | same_component

    def edge_action_mask(self, fixed_tail: Tensor | None = None) -> Tensor:
        """Mask illegal joint ``(tail, head)`` construction actions."""
        if self.terminal:
            return torch.ones(
                self.batch_size,
                self.n,
                self.n,
                dtype=torch.bool,
                device=self.coordinates.device,
            )

        tail_mask = self.tail_mask().unsqueeze(2)
        head_mask = (self.predecessor >= 0).unsqueeze(1)
        mask = tail_mask | head_mask
        if self.edges_added < self.n - 1:
            same_component = self.component.unsqueeze(2) == self.component.unsqueeze(1)
            mask = mask | same_component

        if fixed_tail is not None:
            self._validate_action_vector(fixed_tail, "fixed_tail")
            batch = torch.arange(self.batch_size, device=self.coordinates.device)
            if self.tail_mask()[batch, fixed_tail].any():
                raise ValueError("fixed_tail contains a masked vertex")
            allowed_tail = torch.zeros_like(self.tail_mask())
            allowed_tail[batch, fixed_tail] = True
            mask = mask | ~allowed_tail.unsqueeze(2)

        if mask.flatten(1).all(dim=1).any():
            raise ValueError("each nonterminal state must have a legal edge action")
        return mask

    def sequential_base(self, anchor: int = 0) -> Tensor:
        """Return the unique tail of each batch item's anchored component."""
        if self.terminal:
            raise ValueError("a terminal state has no sequential base")
        if not 0 <= anchor < self.n:
            raise ValueError(f"anchor must be in [0, {self.n})")

        anchor_component = self.component[:, anchor : anchor + 1]
        candidates = (self.component == anchor_component) & ~self.tail_mask()
        if not torch.all(candidates.sum(dim=1) == 1):
            raise ValueError("each anchored component must have exactly one tail")
        return candidates.to(torch.long).argmax(dim=1)

    def path_state_features(self, node_embeddings: Tensor) -> Tensor:
        """Summarize the open path containing each vertex.

        The result concatenates the component mean embedding, the embedding of the
        unique path start (the vertex without a predecessor), and normalized path
        size. It is model-independent state information for adaptive base selection.
        """
        if node_embeddings.ndim != 3 or node_embeddings.shape[:2] != (
            self.batch_size,
            self.n,
        ):
            raise ValueError("node_embeddings must have shape (batch, nodes, embedding)")

        component_index = self.component.unsqueeze(-1)
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

        is_path_start = self.predecessor < 0
        start_source = node_embeddings * is_path_start.unsqueeze(-1)
        path_start_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_index, start_source
        )
        path_start = path_start_by_label.gather(1, embedding_index)
        normalized_size = component_size / self.n
        return torch.cat((component_mean, path_start, normalized_size), dim=-1)

    def update(self, selected_tail: Tensor, selected_head: Tensor) -> BatchedTSPState:
        """Add one legal edge per batch item and return a new state."""
        self._validate_action_vector(selected_tail, "selected_tail")
        self._validate_action_vector(selected_head, "selected_head")
        batch = torch.arange(self.batch_size, device=self.coordinates.device)

        mask = self.head_mask(selected_tail)
        if mask[batch, selected_head].any():
            raise ValueError("selected edge violates the TSP construction mask")

        successor = self.successor.clone()
        predecessor = self.predecessor.clone()
        successor[batch, selected_tail] = selected_head
        predecessor[batch, selected_head] = selected_tail

        component = self.component
        if self.edges_added < self.n - 1:
            left = component.gather(1, selected_tail[:, None])
            right = component.gather(1, selected_head[:, None])
            merged = torch.minimum(left, right)
            in_merged_component = (component == left) | (component == right)
            component = torch.where(in_merged_component, merged, component)

        return BatchedTSPState(
            coordinates=self.coordinates,
            successor=successor,
            predecessor=predecessor,
            component=component,
            edges_added=self.edges_added + 1,
        )

    def edge_cost(self, tails: Tensor, heads: Tensor) -> Tensor:
        """Return the Euclidean cost of a batch of edge sequences."""
        if tails.shape != heads.shape or tails.ndim != 2:
            raise ValueError("tails and heads must both have shape (batch, steps)")
        if tails.size(0) != self.batch_size:
            raise ValueError("edge batch size does not match the state")

        gather_shape = (*tails.shape, self.coordinates.size(-1))
        tail_coordinates = self.coordinates.gather(
            1, tails.unsqueeze(-1).expand(gather_shape)
        )
        head_coordinates = self.coordinates.gather(
            1, heads.unsqueeze(-1).expand(gather_shape)
        )
        return (tail_coordinates - head_coordinates).norm(p=2, dim=-1).sum(dim=1)

    def _validate_action_vector(self, action: Tensor, name: str) -> None:
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError(f"{name} must be a long tensor with shape (batch,)")
        if ((action < 0) | (action >= self.n)).any():
            raise ValueError(f"{name} contains an out-of-range vertex")
