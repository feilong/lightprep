"""Combining multi-echo data into a single timeseries.

Multi-echo acquisition samples the same signal at several echo times; combining
them well recovers more BOLD contrast than any single echo carries, and does it
differently in each voxel because T2* varies. Methods here share a signature and
return a :class:`CombineResult`.

At least two echoes are required.

To add a method, drop a module here that returns a :class:`CombineResult` and
register it in :data:`METHODS`.
"""

from .base import CombineResult
from .optimal import optimal_combination, optimal_weights

#: Minimum echoes to combine.
MIN_ECHOES = 2

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "optimal": optimal_combination,
}

#: Used when no method is named.
DEFAULT_METHOD = "optimal"


def get_method(name: str | None = None):
    """Look up a combination method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown combination method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["CombineResult", "optimal_combination", "optimal_weights", "MIN_ECHOES",
           "METHODS", "DEFAULT_METHOD", "get_method"]
