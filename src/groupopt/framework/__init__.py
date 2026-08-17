"""Stable public interface for the GroupOpt construction paradigm.

Problem implementations and neural methods depend on this package.  The framework
package deliberately does not import either of them.
"""

from groupopt.framework.contracts import ConstructionProcess
from groupopt.framework.engine import construct
from groupopt.framework.types import (
    BaseSelection,
    ConstructionAction,
    ConstructionTrace,
    DecodeStrategy,
)

__all__ = [
    "BaseSelection",
    "ConstructionAction",
    "ConstructionProcess",
    "ConstructionTrace",
    "DecodeStrategy",
    "construct",
]
