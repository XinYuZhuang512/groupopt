"""Backward-compatible import for the public framework contract.

New code should import from :mod:`groupopt.framework`.
"""

from groupopt.framework.contracts import ConstructionProcess

__all__ = ["ConstructionProcess"]
