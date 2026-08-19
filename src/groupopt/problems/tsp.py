"""保持可行性的有向 TSP 回路构造过程。

回路用一个置换表示。在每个非闭合步骤中，构造过程选择路径尾点 ``x``，以及属于
另一连通分量的头点 ``y``，随后固定局部映射 ``sigma(x) = y``。这里以不依赖任何
神经模型的方式表达框架中的可学习 base 构造。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum


class InvalidAction(ValueError):
    """当动作违反构造 mask 时抛出。"""


@dataclass(frozen=True, slots=True)
class TSPState:
    """不可变的互不相交有向路径集合，或最终的单一环。"""

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
        """尚未固定入边的顶点。"""
        return tuple(i for i, value in enumerate(self.predecessor) if value is None)

    @property
    def tails(self) -> tuple[int, ...]:
        """尚未固定出边的顶点。"""
        return tuple(i for i, value in enumerate(self.successor) if value is None)

    @property
    def num_components(self) -> int:
        return len(set(self.component))


@dataclass(frozen=True, slots=True)
class DirectedTour:
    """以后继映射和访问顺序两种形式表示的有向 Hamilton 环。"""

    successor: tuple[int, ...]
    order: tuple[int, ...]

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(enumerate(self.successor))


class DirectedTSPConstruction:
    """在 ``n`` 个顶点上构造一个有向 Hamilton 环。"""

    def initial_state(self, instance: int) -> TSPState:
        """使用顶点数作为最简 TSP 实例描述。"""
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
        """返回路径尾点，即映射目标尚未固定的定义域元素。"""
        if state.terminal:
            return ()
        return state.tails

    def representative_candidates(
        self, state: TSPState, selected_base: int
    ) -> tuple[int, ...]:
        """返回可以作为所选 base 映射目标的头点。

        当仍有多个分量时，mask 掉同分量头点以避免子环；只剩一条路径时，
        该头点是唯一的闭合选择。
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
        """加入一条边，并在需要时合并它连接的两个路径分量。"""
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
        """解码终止状态，并独立验证其为单一环。"""
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
        """为固定 base 解码选择锚定路径当前的尾点。

        该确定性规则把普通的逐路径解码嵌入到可学习 base 解码所用的同一状态机中。
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
        """不依赖状态转移的正确性，直接检查表示与图不变量。"""
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
    """根据方形有向距离矩阵计算回路长度。"""
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
    """直接根据部分图计算弱连通分量。"""
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
