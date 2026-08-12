"""Fitting the multi-echo signal decay.

Multi-echo data samples the same excitation at several echo times, so the way
the signal falls off separates two things a single echo confounds: S0, the
signal at TE=0, and T2*, the rate it decays. Methods here share a signature and
return a :class:`DecayResult`.

At least two echoes are required -- a single echo is one point, and a decay
cannot be fitted through it.

To add a method, drop a module here that returns a :class:`DecayResult` and
register it in :data:`METHODS`.
"""

from .base import DecayResult
from .loglinear import T2STAR_MAX_MS, T2STAR_MIN_MS, fit_arrays, loglinear

#: Minimum echoes needed to fit a decay at all.
MIN_ECHOES = 2

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "loglinear": loglinear,
}

#: Used when no method is named.
DEFAULT_METHOD = "loglinear"


def get_method(name: str | None = None):
    """Look up a decay-fitting method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown decay method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["DecayResult", "loglinear", "fit_arrays", "MIN_ECHOES",
           "T2STAR_MIN_MS", "T2STAR_MAX_MS", "METHODS", "DEFAULT_METHOD", "get_method"]
