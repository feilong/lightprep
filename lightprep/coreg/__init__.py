"""Coregistration of a functional run to its own anatomy.

Methods share a signature and return a :class:`CoregResult`, so one can be
swapped for another without changing the caller. Both write an LTA and a FLIRT
matrix, so surface sampling and FSL resampling read either one.

This step registers a run's *reference* image, not its timeseries: a whole run
is already in one space by the time HMC has run, so the reference stands in for
all of it, and one 6-DOF transform per run is all that is needed.

The default is :func:`allineate`, on niimath, which needs nothing else
installed. :func:`bbregister` is the more accurate of the two where
FreeSurfer is available -- it optimises against the reconstructed white surface,
which is information an intensity cost cannot see -- so prefer it if you have a
recon and the tools to go with it. On the pilot data the two agree to within a
median 0.37-0.45mm across the brain, and the difference is systematic rather
than noise: pick one and keep to it within a study.

To add a method, drop a module here that returns a :class:`CoregResult` and
register it in :data:`METHODS`.
"""

from .base import CoregResult
from .freesurfer import COST_GOOD, COST_SUSPECT, bbregister
from .niimath import allineate

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "niimath": allineate,
    "bbregister": bbregister,
}

#: Used when no method is named: niimath is the only dependency, so this runs
#: wherever the package does.
DEFAULT_METHOD = "niimath"


def get_method(name: str | None = None):
    """Look up a coregistration method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown coregistration method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["CoregResult", "allineate", "bbregister", "COST_GOOD", "COST_SUSPECT",
           "METHODS", "DEFAULT_METHOD", "get_method"]
