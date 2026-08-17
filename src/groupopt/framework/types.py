"""Framework-owned values shared across problems and model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

BaseT = TypeVar("BaseT")
RepresentativeT = TypeVar("RepresentativeT")
SolutionT = TypeVar("SolutionT")


class BaseSelection(str, Enum):
    """The two scientific configurations defined by the paradigm."""

    FIXED = "fixed"
    LEARNED = "learned"


class DecodeStrategy(str, Enum):
    """Action selection strategy; intentionally separate from base selection."""

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
