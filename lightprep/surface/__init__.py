"""Sampling volumetric data onto a cortical surface.

Methods share a signature and return a :class:`SurfaceResult`.

The volume is always read in its own space, through a registration, rather than
being resampled onto an anatomical grid first -- the surface is where the data
is going, so the grid in between is an interpolation nobody keeps.

To add a method, drop a module here that returns a :class:`SurfaceResult` and
register it in :data:`METHODS`.
"""

from .base import SurfaceResult
from .prepare import (PreparedSurfaces, export_binary, prepare_surfaces,
                      separate_hemispheres)
from .concat import ribbon_average_concat
from .ribbon import DEFAULT_DEPTHS, ribbon_average

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "ribbon": ribbon_average,           # from a corrected volume (two interpolations)
    "ribbon-concat": ribbon_average_concat,  # from the original (one)
}

#: Used when no method is named.
DEFAULT_METHOD = "ribbon-concat"


def get_method(name: str | None = None):
    """Look up a surface-sampling method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown surface method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["SurfaceResult", "PreparedSurfaces", "prepare_surfaces", "export_binary",
           "separate_hemispheres",
           "ribbon_average", "ribbon_average_concat", "DEFAULT_DEPTHS",
           "METHODS", "DEFAULT_METHOD", "get_method"]
