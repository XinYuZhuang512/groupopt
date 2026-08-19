"""与模型无关的群论构造框架。"""

from groupopt.framework import (
    BaseSelection,
    ConstructionAction,
    ConstructionProcess,
    ConstructionTrace,
    DecodeStrategy,
    construct,
)

__version__ = "0.1.0"

__all__ = [
    "BaseSelection",
    "ConstructionAction",
    "ConstructionProcess",
    "ConstructionTrace",
    "DecodeStrategy",
    "__version__",
    "construct",
]
