"""Susceptibility distortion correction.

Methods here share a signature and return an :class:`SDCResult`, so one can be
swapped for another without changing the caller.

MEDIC is the multi-echo method: it reads the field off the phase of the data
itself, per frame. Two implementations of it are registered -- :func:`medic`,
which calls warpkit, and :func:`medic_niimath`, which calls niimath.
:func:`phasediff` is the static, fieldmap-based alternative, for runs with no
usable phase. :func:`pepolar` is the other static one, for the common case of a
single-echo run whose only fieldmap is a reversed-phase-encoding EPI: it needs
no phase at all, recovering the field from the blip-up/blip-down pair with FSL
``topup``.

The default is :func:`medic_niimath`: it needs nothing installed and takes 21
seconds on the pilot's 138-frame run where warpkit takes minutes, and the two
agree to r=0.954 on the native field map. warpkit is still marginally the more
faithful -- against the session's own GRE phase-difference fieldmap it reaches
r=0.835 against niimath's 0.790 -- so :func:`medic` is the one to name when that
margin is worth the dependency.

Note that niimath uses the opposite sign convention for the displacement map.
:mod:`lightprep.sdc.niimath` corrects for it, so the two methods' outputs are
interchangeable here, but a hand-rolled ``niimath --medic`` call is not.

To add a method, drop a module here that returns an :class:`SDCResult` and
register it in :data:`METHODS`.
"""

from .base import SDCResult
from .medic import MIN_ECHOES, medic
from .niimath import medic as medic_niimath
from .pepolar import pepolar
from .phasediff import phasediff

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "medic": medic,
    "niimath": medic_niimath,
    "pepolar": pepolar,
    "phasediff": phasediff,
}

#: Used when no method is named: niimath is the only dependency, so this runs
#: wherever the package does.
DEFAULT_METHOD = "niimath"


def get_method(name: str | None = None):
    """Look up an SDC method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown SDC method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["SDCResult", "medic", "medic_niimath", "pepolar", "phasediff",
           "MIN_ECHOES", "METHODS", "DEFAULT_METHOD", "get_method"]
