"""Moving data into corrected space.

This step is where the transforms other steps *estimate* actually get spent.
Keeping it separate is what makes a single-interpolation pipeline possible: HMC
and SDC each hand over transforms rather than resampled data, and this step
composes them and resamples once.

Methods share a signature and return a :class:`ResampleResult`.

:data:`METHODS` holds only the FSL composition. niimath has no equivalent of
``convertwarp``, so folding a per-frame distortion field together with a rigid
transform -- the whole point of this step -- cannot be done with it, and no
niimath method is offered rather than one that quietly resamples twice.

The *other* function here, :func:`apply_sdc`, does have a niimath version:
:func:`apply_sdc_niimath` undistorts a timeseries with ``-unwarp``, reproducing
``wk-apply-warp`` exactly (r=1.00000 on the pilot run) with no warpkit needed.
It is the one to use for making data to estimate motion on.
"""

from .base import ResampleResult
from .fsl import apply_sdc, compose_and_apply
from .niimath import apply_sdc as apply_sdc_niimath

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "fsl": compose_and_apply,
}

#: Used when no method is named.
DEFAULT_METHOD = "fsl"


def get_method(name: str | None = None):
    """Look up a resample method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown resample method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["ResampleResult", "apply_sdc", "apply_sdc_niimath", "compose_and_apply",
           "METHODS", "DEFAULT_METHOD", "get_method"]
