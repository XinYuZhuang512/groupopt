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
