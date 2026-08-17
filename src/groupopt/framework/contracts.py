"""Protocols forming the dependency boundary of the GroupOpt paradigm."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

InstanceT_contra = TypeVar("InstanceT_contra", contravariant=True)
StateT = TypeVar("StateT")
BaseT = TypeVar("BaseT")
RepresentativeT = TypeVar("RepresentativeT")
SolutionT_co = TypeVar("SolutionT_co", covariant=True)


@runtime_checkable
class ConstructionProcess(
    Protocol[InstanceT_contra, StateT, BaseT, RepresentativeT, SolutionT_co]
):
    """Problem-owned semantics for one constructive optimization process."""

    def initial_state(self, instance: InstanceT_contra) -> StateT:
        """Create an empty construction state."""
        ...

    def base_candidates(self, state: StateT) -> tuple[BaseT, ...]:
        """Return objects that may be stabilized next."""
        ...

    def representative_candidates(
        self, state: StateT, selected_base: BaseT
    ) -> tuple[RepresentativeT, ...]:
        """Return legal representatives conditional on one base."""
        ...

    def transition(
        self,
        state: StateT,
        selected_base: BaseT,
        selected_representative: RepresentativeT,
    ) -> StateT:
        """Apply one legal action; models never own this transition."""
        ...

    def is_terminal(self, state: StateT) -> bool:
        """Return whether a complete solution has been constructed."""
        ...

    def decode(self, state: StateT) -> SolutionT_co:
        """Decode and validate a terminal state."""
        ...
