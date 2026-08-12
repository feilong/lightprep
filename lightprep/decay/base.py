"""The contract every decay-fitting method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DecayResult:
    """Outcome of fitting the multi-echo signal decay.

    The model is mono-exponential, ``S(TE) = S0 * exp(-TE / T2*)``: S0 is the
    signal extrapolated back to TE=0 (proton density and receive sensitivity),
    T2* the rate at which it decays.

    Attributes:
        t2star: Per-vertex T2* in milliseconds.
        s0: Per-vertex S0, in the input's arbitrary units.
        r2: Per-vertex coefficient of determination of the fit, or None. With
            only two echoes a line through two points is exact, so there is no
            residual to measure and nothing to report.
        echo_times: Echo times used, in milliseconds.
        n_vertices: Vertices fitted.
        n_invalid: Vertices with no usable fit -- non-positive signal, or a
            decay running the wrong way. Expect a few; expect them at the edges.
        method: Name of the method that produced this result.
    """

    t2star: Path
    s0: Path
    r2: Path | None
    echo_times: tuple[float, ...]
    n_vertices: int
    n_invalid: int
    method: str
