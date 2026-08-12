"""The contract every SDC method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SDCResult:
    """What every susceptibility-distortion-correction method returns.

    The maps are framewise where the method supports it (MEDIC does): one
    field map per volume, rather than a single static estimate.

    Attributes:
        fieldmap_native: B0 field map (Hz) in native, still-distorted space.
        displacement_map: Displacement along the phase-encoding axis (mm).
            This is what a resampler consumes.
        fieldmap: B0 field map (Hz) in undistorted space.
        method: Name of the method that produced this result.
        n_echoes: Echoes used to estimate the field.
        n_frames: Frames the field was estimated for.
        space: Which space the displacement map is valid in, which decides
            where a resampler must apply it relative to head-motion correction.

            ``"native"`` -- the map describes a specific frame as acquired, so
            it is applied before the rigid transform. MEDIC is native: each
            frame's field is read from that frame's own phase.

            ``"reference"`` -- the map is valid once the head is at the
            reference position, so it is applied after the rigid transform. A
            static fieldmap is reference: the B0 perturbation is produced by
            the head's own tissue-air boundaries, so under translation the
            field co-moves with the head exactly (the dipole convolution is
            translation-invariant). The map therefore describes the field in
            the head's frame, not the scanner's.
    """

    fieldmap_native: Path
    displacement_map: Path
    fieldmap: Path
    method: str
    n_echoes: int
    n_frames: int
    space: str = "native"
