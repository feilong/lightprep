"""Sampling volumetric data onto the cortical ribbon.

Each vertex is sampled at several fractional depths between its white and pial
positions, and those samples are averaged. Averaging across the ribbon rather
than reading a single depth trades a little spatial specificity for a lot of
noise: a single mid-thickness sample sits one interpolation away from whatever
that voxel happened to contain.

Depth is defined by interpolating between *corresponding* white and pial
vertices::

    p(f) = white + f * (pial - white)

which is not the same as FreeSurfer's ``--projfrac``, which walks along the
surface normal by a fraction of thickness. On this subject the two disagree by a
median of 0.39mm at f=0.5 and up to 1.7mm at p99 -- around 60% of an EPI voxel --
because the white-to-pial vector sits a median 20 degrees off the normal.

The volume is sampled in its own space, through the registration. Nothing is
resampled onto an anatomical grid first: that would spend an interpolation to
produce an intermediate nobody keeps.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from .base import SurfaceResult

#: Five evenly spaced samples strictly inside the ribbon, avoiding the
#: boundaries themselves where partial-volume effects are worst.
DEFAULT_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)

#: Trilinear is the default: it is what the volumetric steps use, and the
#: cost of a sharper kernel is ringing at the tissue edges the ribbon
#: deliberately straddles. The others stay available.
INTERP_ORDERS = {"nearest": 0, "trilinear": 1, "cubic": 3}


def _vertices(path: Path) -> np.ndarray:
    """Vertex coordinates from a GIfTI surface, in its own world space."""
    gii = nib.load(path)
    return np.asarray(gii.darrays[0].data, dtype=np.float64)


def ribbon_average(
    volume,
    white,
    pial,
    registration,
    out,
    *,
    hemi: str,
    depths=DEFAULT_DEPTHS,
    interp: str = "trilinear",
) -> SurfaceResult:
    """Average a volume across the cortical ribbon, per vertex.

    Args:
        volume: 3D or 4D image, in its own (functional) space.
        white: White surface, in the anatomical space ``registration`` targets.
        pial: Pial surface, with vertices corresponding 1:1 to ``white``.
        registration: FreeSurfer LTA mapping the functional reference to the
            anatomy -- what ``bbregister`` writes.
        out: Where to write the per-vertex timeseries (``.func.gii``).
        hemi: ``lh`` or ``rh``, recorded on the result.
        depths: Fractional depths between white (0) and pial (1).
        interp: Interpolation, one of :data:`INTERP_ORDERS`.

    Returns:
        A :class:`~lightprep.surface.base.SurfaceResult`.

    Raises:
        ValueError: If the surfaces disagree, or an argument is invalid.
    """
    if interp not in INTERP_ORDERS:
        raise ValueError(f"interp must be one of {sorted(INTERP_ORDERS)}, got {interp!r}")
    depths = tuple(float(d) for d in depths)
    if not depths:
        raise ValueError("need at least one depth")
    if not all(0.0 <= d <= 1.0 for d in depths):
        raise ValueError(f"depths must lie in [0, 1] (white to pial), got {depths}")

    wv, pv = _vertices(Path(white)), _vertices(Path(pial))
    if wv.shape != pv.shape:
        raise ValueError(
            f"white and pial must have corresponding vertices, got {wv.shape} and "
            f"{pv.shape}. Depth is only meaningful between matching vertices."
        )

    img = nib.load(volume)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    n_frames = data.shape[3]

    # The LTA nominally reads functional -> anatomical, but nitransforms hands
    # back the mapping in the resampling direction, i.e. anatomical ->
    # functional -- which is what carries surface points into the volume.
    from nitransforms.linear import Affine

    xfm = Affine.from_filename(str(registration), fmt="fs")
    inv_affine = np.linalg.inv(img.affine)

    ribbon = pv - wv
    # All depths at once: (nDepth * nVert, 3) voxel coordinates.
    pts = np.concatenate([wv + f * ribbon for f in depths], axis=0)
    ras = np.asarray(xfm.map(pts))
    ijk = (inv_affine[:3, :3] @ ras.T).T + inv_affine[:3, 3]

    order = INTERP_ORDERS[interp]
    coords = ijk.T
    n_vert = wv.shape[0]

    inside = np.all(
        (ijk >= 0) & (ijk <= (np.array(data.shape[:3]) - 1)), axis=1
    ).reshape(len(depths), n_vert)
    n_outside = int((~inside.all(axis=0)).sum())

    out_ts = np.empty((n_vert, n_frames), dtype=np.float32)
    for t in range(n_frames):
        vals = map_coordinates(
            data[..., t], coords, order=order, mode="constant", cval=np.nan
        )
        # Depths that left the field of view must not drag the mean down.
        out_ts[:, t] = np.nanmean(vals.reshape(len(depths), n_vert), axis=0)

    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    gii = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                out_ts[:, t].astype(np.float32),
                intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32",
                encoding="GIFTI_ENCODING_B64GZ",
            )
            for t in range(n_frames)
        ]
    )
    gii.to_filename(str(out))

    return SurfaceResult(
        output=out,
        hemi=hemi,
        n_vertices=n_vert,
        n_frames=n_frames,
        depths=depths,
        n_outside=n_outside,
        method="ribbon",
    )
