"""Optional PyTorch bridge between the core paradigm and neural methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from torch import Generator, Tensor

InstanceT_contra = TypeVar("InstanceT_contra", contravariant=True)
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class ConstructionOutput:
    """Common tensor result returned by every neural method adapter."""

    cost: Tensor
    log_likelihood: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor
    tail_entropy: Tensor
    gate_probability: Tensor
    action_entropy: Tensor
    symmetry_node_embeddings: Tensor | None = None


@runtime_checkable
class BatchedConstructionProcess(Protocol[InstanceT_contra, StateT]):
    """Tensorized problem plugin consumed by neural model adapters.

    Boolean masks follow the attention convention: ``True`` means illegal.
    Feasibility, transitions, fixed-base behavior and the objective remain owned by
    the problem plugin rather than a neural model.
    """

    def initial_state(self, instance: InstanceT_contra) -> StateT: ...

    def is_terminal(self, state: StateT) -> bool: ...

    def base_mask(self, state: StateT) -> Tensor: ...

    def representative_mask(self, state: StateT, selected_base: Tensor) -> Tensor: ...

    def action_mask(self, state: StateT, fixed_base: Tensor | None = None) -> Tensor: ...

    def fixed_base(self, state: StateT, anchor: int = 0) -> Tensor: ...

    def transition(
        self, state: StateT, selected_base: Tensor, selected_representative: Tensor
    ) -> StateT: ...

    def objective(
        self, state: StateT, selected_bases: Tensor, selected_representatives: Tensor
    ) -> Tensor: ...

    def solution(self, state: StateT) -> Tensor: ...


@runtime_checkable
class ConstructionModel(Protocol):
    """Stable neural adapter interface shared by all model families."""

    def forward(
        self,
        instance: Tensor,
        decode_type: str = "sampling",
        base_mode: str = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: Generator | None = None,
    ) -> ConstructionOutput:
        """Score framework actions and return one feasible construction."""
        ...
