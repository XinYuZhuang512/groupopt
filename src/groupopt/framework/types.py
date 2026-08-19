"""由框架定义、在问题实现与模型适配器之间共享的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

BaseT = TypeVar("BaseT")
RepresentativeT = TypeVar("RepresentativeT")
SolutionT = TypeVar("SolutionT")


class BaseSelection(str, Enum):
    """该范式定义的两种科学实验配置。"""

    FIXED = "fixed"
    LEARNED = "learned"


class DecodeStrategy(str, Enum):
    """动作选择策略；特意与 base 选择机制分离。"""

    GREEDY = "greedy"
    SAMPLING = "sampling"


@dataclass(frozen=True, slots=True)
class ConstructionAction(Generic[BaseT, RepresentativeT]):
    base: BaseT
    representative: RepresentativeT


@dataclass(frozen=True, slots=True)
class ConstructionTrace(Generic[BaseT, RepresentativeT, SolutionT]):
    actions: tuple[ConstructionAction[BaseT, RepresentativeT], ...]
    solution: SolutionT
