"""The contract every coregistration method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoregResult:
    """Outcome of aligning a functional run to its own anatomy.

    Attributes:
        registration: The transform, in the method's native format.
        fsl_matrix: The same transform as a FLIRT matrix, or None if the
            method cannot express it that way. This is what lets the
            registration be folded into an FSL resample chain.
        moving: The image that was registered -- for fMRI, the run's reference
            volume rather than the timeseries.
        target: What it was registered to.
        cost: Final cost. Lower is better; the scale is method-specific, so
            compare against the method's own guidance, not across methods.
        method: Name of the method that produced this result.
    """

    registration: Path
    fsl_matrix: Path | None
    moving: Path
    target: str
    cost: float | None
    method: str
