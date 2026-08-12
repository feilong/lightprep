"""The contract every resample method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResampleResult:
    """Outcome of moving data into corrected space.

    Attributes:
        outputs: Corrected images, in the same order as the inputs.
        reference: The grid the outputs live on.
        warps: Per-frame transforms actually applied. For a composed
            correction these are the single combined warps, one per frame.
        method: Name of the method that produced this result.
        n_interpolations: How many times the data was resampled to get here.
            One is the goal; every extra pass costs resolution.
    """

    outputs: tuple[Path, ...]
    reference: Path
    warps: tuple[Path, ...]
    method: str
    n_interpolations: int = 1
