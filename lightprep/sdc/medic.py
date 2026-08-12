"""Susceptibility distortion correction with MEDIC (Multi-Echo DIstortion Correction).

MEDIC recovers a B0 field map from the *phase* of a multi-echo acquisition.
Phase accrues linearly with echo time, so the slope of phase against TE is the
field offset::

    phi(TE) = phi_0 + 2*pi*gamma*dB0*TE

Because every frame carries its own phase, the field can be estimated per frame
rather than once for the session -- which is the point of MEDIC: it tracks
distortion as it changes with head position and respiration, instead of assuming
the field measured in a separate fieldmap scan still holds minutes later.

This wraps the reference implementation, warpkit (Van et al., *Imaging
Neuroscience*, https://doi.org/10.1162/IMAG.a.1262).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib

from .._utils import strip_ext
from .base import SDCResult

MIN_ECHOES = 2


def _sidecar(path: Path) -> Path:
    """The BIDS JSON sitting next to a NIfTI."""
    return path.with_name(strip_ext(path) + ".json")


def _check_inputs(magnitude, phase, metadata):
    """Validate the mag/phase pairing and resolve the sidecars."""
    mag = [Path(p).resolve() for p in magnitude]
    pha = [Path(p).resolve() for p in phase]

    if len(mag) != len(pha):
        raise ValueError(
            f"need one magnitude per phase echo, got {len(mag)} magnitude and "
            f"{len(pha)} phase file(s)"
        )
    if len(mag) < MIN_ECHOES:
        raise ValueError(
            f"MEDIC needs at least {MIN_ECHOES} echoes to fit phase against echo "
            f"time, got {len(mag)}. Single-echo data cannot be corrected this way; "
            "use a fieldmap- or pepolar-based method instead."
        )

    missing = [str(p) for p in (*mag, *pha) if not p.exists()]
    if missing:
        raise FileNotFoundError("input(s) not found: " + ", ".join(missing))

    shapes = {nib.load(p).shape for p in (*mag, *pha)}
    if len(shapes) > 1:
        raise ValueError(
            f"magnitude and phase must share a grid and frame count, got {sorted(shapes)}"
        )
    shape = shapes.pop()
    if len(shape) != 4:
        raise ValueError(f"expected 4D timeseries, got shape {shape}")

    meta = [Path(m).resolve() for m in metadata] if metadata else [_sidecar(p) for p in mag]
    if len(meta) != len(mag):
        raise ValueError(f"need one sidecar per echo, got {len(meta)} for {len(mag)} echoes")
    missing = [str(p) for p in meta if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "BIDS sidecar(s) not found: " + ", ".join(missing) + ". MEDIC reads "
            "EchoTime, TotalReadoutTime and PhaseEncodingDirection from them; pass "
            "`metadata=` explicitly if they live elsewhere."
        )
    return mag, pha, meta, shape[3]


def medic(
    magnitude,
    phase,
    out_dir,
    *,
    metadata=None,
    noise_frames: int = 0,
    n_cpus: int = 4,
    wrap_limit: bool = False,
    prefix: str = "medic",
) -> SDCResult:
    """Estimate framewise field maps from multi-echo magnitude and phase.

    Args:
        magnitude: Magnitude echoes, ordered by echo time.
        phase: Phase echoes, in the same echo order as ``magnitude``.
        out_dir: Directory for the field maps and displacement maps.
        metadata: One BIDS sidecar per echo. Defaults to the JSON beside each
            magnitude file. EchoTime is read per echo; TotalReadoutTime and
            PhaseEncodingDirection come from the first.
        noise_frames: Frames to trim from the end of every echo before
            unwrapping (Siemens noise frames, if your protocol appends them).
        n_cpus: Worker processes. Unwrapping is per-frame and the dominant cost.
        wrap_limit: Disable some phase-unwrapping heuristics.
        prefix: Basename stem for the written maps.

    Returns:
        An :class:`~lightprep.sdc.base.SDCResult`.

    Raises:
        ValueError: On fewer than two echoes, or mismatched inputs.
        DependencyError: If warpkit is not installed.

    Note:
        Pass the *raw* phase, not motion-corrected phase. warpkit rescales phase
        to radians itself, inferring the range from the data (so Siemens
        [-4096, 4094] is handled), and it needs the phase in its native,
        still-distorted space to model the field for that frame.
    """
    try:
        from warpkit.api import medic as _warpkit_medic
    except ImportError as exc:  # pragma: no cover - depends on environment
        from .._utils import DependencyError

        raise DependencyError(
            "warpkit is required for MEDIC; install it with `pip install warpkit`"
        ) from exc

    mag, pha, meta, n_frames = _check_inputs(magnitude, phase, metadata)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _warpkit_medic(
        phase=pha,
        magnitude=mag,
        out_prefix=str(out_dir / prefix),
        metadata=meta,
        noise_frames=noise_frames,
        n_cpus=n_cpus,
        wrap_limit=wrap_limit,
    )

    return SDCResult(
        fieldmap_native=Path(result.fieldmap_native),
        displacement_map=Path(result.displacement_map),
        fieldmap=Path(result.fieldmap),
        method="medic",
        n_echoes=len(mag),
        n_frames=n_frames - noise_frames,
        # Each frame's field comes from that frame's own phase, in the space it
        # was acquired in -- so it is applied before the head is moved.
        space="native",
    )
