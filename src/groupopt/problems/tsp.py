"""A feasibility-preserving construction process for directed TSP tours.

A tour is represented by a permutation. At every non-closing step, the process
chooses a path tail ``x`` and a head ``y`` belonging to another component, then fixes
the partial mapping ``sigma(x) = y``. This is the adaptive-base construction from the
framework document, expressed without any neural-model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Sequence


class InvalidAction(ValueError):
    """Raised when an action violates the construction mask."""


@dataclass(frozen=True, slots=True)
class TSPState:
    """An immutable collection of disjoint directed paths, or one final cycle."""

    n: int
    successor: tuple[int | None, ...]
    predecessor: tuple[int | None, ...]
    component: tuple[int, ...]
    edges_added: int

    @property
    def terminal(self) -> bool:
        return self.edges_added == self.n

    @property
    def heads(self) -> tuple[int, ...]:
        """Vertices without a fixed incoming edge."""
        return tuple(i for i, value in enumerate(self.predecessor) if value is None)

    @property
    def tails(self) -> tuple[int, ...]:
        """Vertices without a fixed outgoing edge."""
        return tuple(i for i, value in enumerate(self.successor) if value is None)

    @property
    def num_components(self) -> int:
        return len(set(self.component))


@dataclass(frozen=True, slots=True)
class DirectedTour:
    """A directed Hamiltonian cycle in successor and visit-order forms."""

    successor: tuple[int, ...]
    order: tuple[int, ...]

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(enumerate(self.successor))


class DirectedTSPConstruction:
    """Construct a single directed Hamiltonian cycle on ``n`` vertices."""

    def initial_state(self, instance: int) -> TSPState:
        """Use the vertex count as the minimal TSP instance description."""
        if instance < 2:
            raise ValueError("directed TSP construction requires at least two vertices")

        state = TSPState(
            n=instance,
            successor=(None,) * instance,
            predecessor=(None,) * instance,
            component=tuple(range(instance)),
            edges_added=0,
        )
        self.validate_state(state)
        return state

    def base_candidates(self, state: TSPState) -> tuple[int, ...]:
        """Return path tails: domains whose image is not fixed yet."""
        if state.terminal:
            return ()
        return state.tails

    def representative_candidates(
        self, state: TSPState, selected_base: int
    ) -> tuple[int, ...]:
        """Return heads that can be the selected base's image.

        While multiple components remain, the same-component head is masked to avoid
        a subtour. With one path left, that head is the unique closing choice.
        """
        if selected_base not in self.base_candidates(state):
            raise InvalidAction(f"{selected_base} is not an available path tail")

        base_component = state.component[selected_base]
        if state.num_components == 1:
            return tuple(
                head for head in state.heads if state.component[head] == base_component
            )

        return tuple(
            head for head in state.heads if state.component[head] != base_component
        )

    def transition(
        self,
        state: TSPState,
        selected_base: int,
        selected_representative: int,
    ) -> TSPState:
        """Add one edge and merge its two path components when applicable."""
        legal_representatives = self.representative_candidates(state, selected_base)
        if selected_representative not in legal_representatives:
            raise InvalidAction(
                f"edge {selected_base}->{selected_representative} is masked in this state"
            )

        successor = list(state.successor)
        predecessor = list(state.predecessor)
        successor[selected_base] = selected_representative
        predecessor[selected_representative] = selected_base

        component = state.component
        if state.num_components > 1:
            left = component[selected_base]
            right = component[selected_representative]
            merged = min(left, right)
            component = tuple(
                merged if label in (left, right) else label for label in component
            )

        next_state = TSPState(
            n=state.n,
            successor=tuple(successor),
            predecessor=tuple(predecessor),
            component=component,
            edges_added=state.edges_added + 1,
        )
        self.validate_state(next_state)
        return next_state

    def is_terminal(self, state: TSPState) -> bool:
        return state.terminal

    def decode(self, state: TSPState) -> DirectedTour:
        """Decode a terminal state and independently verify its single-cycle form."""
        if not state.terminal:
            raise ValueError("cannot decode a non-terminal TSP state")

        successor = tuple(_require_vertex(value) for value in state.successor)
        order: list[int] = []
        current = 0
        for _ in range(state.n):
            if current in order:
                raise ValueError("terminal state contains a premature cycle")
            order.append(current)
            current = successor[current]

        if current != 0 or len(set(order)) != state.n:
            raise ValueError("terminal state is not a Hamiltonian cycle")
        return DirectedTour(successor=successor, order=tuple(order))

    def sequential_base(self, state: TSPState, anchor: int = 0) -> int:
        """Select the current tail of an anchored path for fixed-base decoding.

        This deterministic rule embeds ordinary one-path-at-a-time decoding in the
        same state machine used by adaptive-base decoding.
        """
        if state.terminal:
            raise InvalidAction("a terminal state has no next base")
        if not 0 <= anchor < state.n:
            raise ValueError(f"anchor must be in [0, {state.n})")

        anchor_component = state.component[anchor]
        candidates = tuple(
            tail for tail in state.tails if state.component[tail] == anchor_component
        )
        if len(candidates) != 1:
            raise ValueError("the anchored component does not have exactly one tail")
        return candidates[0]

    def validate_state(self, state: TSPState) -> None:
        """Check representation and graph invariants without trusting transitions."""
        n = state.n
        if n < 2:
            raise ValueError("a TSP state requires at least two vertices")
        if not (
            len(state.successor) == len(state.predecessor) == len(state.component) == n
        ):
            raise ValueError("state vectors must all have length n")

        edge_count = sum(value is not None for value in state.successor)
        if edge_count != state.edges_added:
            raise ValueError("edges_added disagrees with successor entries")
        if sum(value is not None for value in state.predecessor) != edge_count:
            raise ValueError("predecessor and successor edge counts differ")
        if not 0 <= state.edges_added <= n:
            raise ValueError("edges_added is outside its valid range")

        for source, target in enumerate(state.successor):
            if target is not None:
                _check_vertex(target, n)
                if state.predecessor[target] != source:
                    raise ValueError("successor and predecessor are inconsistent")
        for target, source in enumerate(state.predecessor):
            if source is not None:
                _check_vertex(source, n)
                if state.successor[source] != target:
                    raise ValueError("predecessor and successor are inconsistent")

        component_count = state.num_components
        expected_components = max(1, n - state.edges_added)
        if component_count != expected_components:
            raise ValueError(
                f"expected {expected_components} components, found {component_count}"
            )
        graph_partition = _graph_component_partition(state)
        for left in range(n):
            for right in range(n):
                labels_agree = state.component[left] == state.component[right]
                graph_agrees = graph_partition[left] == graph_partition[right]
                if labels_agree != graph_agrees:
                    raise ValueError("component labels disagree with the partial graph")
        if state.terminal:
            self.decode(state)
        elif len(state.heads) != component_count or len(state.tails) != component_count:
            raise ValueError("each open path component must have one head and one tail")


def tour_length(
    tour: DirectedTour, distance_matrix: Sequence[Sequence[float]]
) -> float:
    """Evaluate a tour against a square directed distance matrix."""
    n = len(tour.successor)
    if len(distance_matrix) != n or any(len(row) != n for row in distance_matrix):
        raise ValueError("distance matrix shape must match the tour")
    return fsum(distance_matrix[source][target] for source, target in tour.edges)


def _check_vertex(vertex: int, n: int) -> None:
    if not 0 <= vertex < n:
        raise ValueError(f"vertex {vertex} is outside [0, {n})")


def _require_vertex(vertex: int | None) -> int:
    if vertex is None:
        raise ValueError("complete tour contains an unset successor")
    return vertex


def _graph_component_partition(state: TSPState) -> tuple[int, ...]:
    """Compute weakly connected components directly from the partial graph."""
    labels = [-1] * state.n
    next_label = 0

    for start in range(state.n):
        if labels[start] != -1:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            vertex = stack.pop()
            neighbors = (state.successor[vertex], state.predecessor[vertex])
            for neighbor in neighbors:
                if neighbor is not None and labels[neighbor] == -1:
                    labels[neighbor] = next_label
                    stack.append(neighbor)
        next_label += 1

    return tuple(labels)
