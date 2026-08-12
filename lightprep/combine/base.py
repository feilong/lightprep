"""The contract every echo-combination method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CombineResult:
    """Outcome of merging multi-echo data into one timeseries.

    Attributes:
        output: The combined per-vertex timeseries (one file).
        weights: The per-vertex, per-echo weights that produced it, or None if
            the method does not expose them.
        echo_times: Echo times combined, in milliseconds.
        n_vertices: Vertices combined.
        n_frames: Frames in the output.
        n_fallback: Vertices where the primary weighting could not be formed
            (e.g. no usable T2*) and a fallback was used instead.
        method: Name of the method that produced this result.
    """

    output: Path
    weights: Path | None
    echo_times: tuple[float, ...]
    n_vertices: int
    n_frames: int
    n_fallback: int
    method: str
