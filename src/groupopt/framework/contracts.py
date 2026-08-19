"""构成 GroupOpt 范式依赖边界的协议。"""

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
    """由问题层负责定义的构造式优化过程语义。"""

    def initial_state(self, instance: InstanceT_contra) -> StateT:
        """创建空的构造状态。"""
        ...

    def base_candidates(self, state: StateT) -> tuple[BaseT, ...]:
        """返回下一步可以稳定化的对象。"""
        ...

    def representative_candidates(
        self, state: StateT, selected_base: BaseT
    ) -> tuple[RepresentativeT, ...]:
        """返回给定 base 条件下的合法代表元。"""
        ...

    def transition(
        self,
        state: StateT,
        selected_base: BaseT,
        selected_representative: RepresentativeT,
    ) -> StateT:
        """执行一个合法动作；状态转移不由模型负责。"""
        ...

    def is_terminal(self, state: StateT) -> bool:
        """判断是否已经构造出完整解。"""
        ...

    def decode(self, state: StateT) -> SolutionT_co:
        """解码并验证终止状态。"""
        ...
