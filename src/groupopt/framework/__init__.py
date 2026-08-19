"""GroupOpt 构造范式的稳定公共接口。

问题实现和神经方法都依赖本包；框架包特意不反向导入它们。
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
