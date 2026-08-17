"""Model-independent reference execution of the construction paradigm."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from groupopt.framework.contracts import ConstructionProcess
from groupopt.framework.types import ConstructionAction, ConstructionTrace

InstanceT = TypeVar("InstanceT")
StateT = TypeVar("StateT")
BaseT = TypeVar("BaseT")
RepresentativeT = TypeVar("RepresentativeT")
SolutionT = TypeVar("SolutionT")


def construct(
    process: ConstructionProcess[
        InstanceT, StateT, BaseT, RepresentativeT, SolutionT
    ],
    instance: InstanceT,
    select_base: Callable[[StateT, tuple[BaseT, ...]], BaseT],
    select_representative: Callable[
        [StateT, BaseT, tuple[RepresentativeT, ...]], RepresentativeT
    ],
) -> ConstructionTrace[BaseT, RepresentativeT, SolutionT]:
    """Run a process using arbitrary scorers/selectors supplied by an adapter."""
    state = process.initial_state(instance)
    actions: list[ConstructionAction[BaseT, RepresentativeT]] = []
    while not process.is_terminal(state):
        bases = process.base_candidates(state)
        if not bases:
            raise RuntimeError("a nonterminal construction state has no legal base")
        selected_base = select_base(state, bases)
        if selected_base not in bases:
            raise ValueError("base selector returned an illegal candidate")
        representatives = process.representative_candidates(state, selected_base)
        if not representatives:
            raise RuntimeError("a selected base has no legal representative")
        selected_representative = select_representative(
            state, selected_base, representatives
        )
        if selected_representative not in representatives:
            raise ValueError("representative selector returned an illegal candidate")
        actions.append(ConstructionAction(selected_base, selected_representative))
        state = process.transition(state, selected_base, selected_representative)
    return ConstructionTrace(tuple(actions), process.decode(state))
