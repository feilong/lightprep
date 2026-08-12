"""Head motion correction.

Every method in this step shares one signature::

    method(echoes, out_dir, *, ref=..., ref_echo=0, ...) -> HMCResult

``echoes`` is always a list ordered by echo time (length one for single-echo
data), and the result is always an :class:`HMCResult`, so a caller can swap one
method for another without changing anything around it. Both methods here write
their per-volume transforms as FLIRT matrices, so what comes after them cannot
tell which one ran.

The default is :func:`moco`, niimath's port of the ``3dvolreg`` algorithm: it
needs nothing installed and estimates a 138-frame run in about 20s.
:func:`allineate` is the general-purpose registration, an order of magnitude
slower but able to work against a reference that is not one of the frames
without composing, and to fit more than 6 DOF. :func:`mcflirt` needs FSL, and
its transforms are quarantined -- see :data:`UNREPLAYABLE_METHODS`.

Multi-echo policy, common to every method here: motion is estimated on a single
echo and the resulting per-volume transforms are applied unchanged to all
echoes. The echoes of one TR come from a single excitation and so share a head
position -- estimating each echo separately would add noise and would break the
voxelwise correspondence across echoes that T2*/S0 fitting depends on.

To add a method, drop a module in this subpackage that returns an
:class:`HMCResult`, and register it in :data:`METHODS`.
"""

from .base import (HMCResult, TransformReplayError, UNREPLAYABLE_METHODS,
                   check_transforms_replayable)
from .fsl import mcflirt
from .moco import moco
from .niimath import allineate

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "moco": moco,
    "niimath": allineate,
    "fsl": mcflirt,
}

#: Used when no method is named. -moco is by a wide margin the fastest of the
#: three, and needs nothing beyond niimath itself.
DEFAULT_METHOD = "moco"


def get_method(name: str | None = None):
    """Look up an HMC method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown HMC method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["HMCResult", "moco", "allineate", "mcflirt", "METHODS",
           "DEFAULT_METHOD", "get_method", "TransformReplayError",
           "UNREPLAYABLE_METHODS", "check_transforms_replayable"]
