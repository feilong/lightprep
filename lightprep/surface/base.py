"""The contract every surface-sampling method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurfaceResult:
    """Outcome of pulling volumetric data onto a cortical surface.

    Attributes:
        output: Per-vertex timeseries, one file (vertices x frames).
        hemi: Which hemisphere, ``lh`` or ``rh``.
        n_vertices: Vertices in the surface.
        n_frames: Frames sampled.
        depths: Fractional depths sampled between white (0) and pial (1).
        n_outside: Vertices whose samples fell outside the volume at any depth.
            Non-zero is normal near the edge of the EPI field of view; large
            counts mean the coverage or the registration is wrong.
        method: Name of the method that produced this result.
    """

    output: Path
    hemi: str
    n_vertices: int
    n_frames: int
    depths: tuple[float, ...]
    n_outside: int
    method: str
