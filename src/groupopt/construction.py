"""Interfaces shared by group-theoretic construction problems and models."""

from __future__ import annotations

from typing import Protocol, TypeVar


InstanceT = TypeVar("InstanceT", contravariant=True)
StateT = TypeVar("StateT")
BaseT = TypeVar("BaseT")
RepresentativeT = TypeVar("RepresentativeT")
SolutionT = TypeVar("SolutionT", covariant=True)


class ConstructionProcess(
    Protocol[InstanceT, StateT, BaseT, RepresentativeT, SolutionT]
):
    """A model-independent constructive optimization process.

    A model may score the values returned by ``base_candidates`` and
    ``representative_candidates``. Feasibility and state transitions remain owned by
    the process implementation.
    """

    def initial_state(self, instance: InstanceT) -> StateT:
        """Create the empty construction state for an instance."""

    def base_candidates(self, state: StateT) -> tuple[BaseT, ...]:
        """Return the objects that may be stabilized next."""

    def representative_candidates(
        self, state: StateT, selected_base: BaseT
    ) -> tuple[RepresentativeT, ...]:
        """Return legal representative choices after selecting a base."""

    def transition(
        self,
        state: StateT,
        selected_base: BaseT,
        selected_representative: RepresentativeT,
    ) -> StateT:
        """Apply one legal construction action."""

    def is_terminal(self, state: StateT) -> bool:
        """Return whether construction has produced a complete solution."""

    def decode(self, state: StateT) -> SolutionT:
        """Decode a terminal state into a problem solution."""
