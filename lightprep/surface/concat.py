"""Sampling the surface straight from the acquired data, in one interpolation.

Every resample costs resolution, and the losses compound. Correcting distortion
and motion into a volume, then sampling that volume onto the surface, spends two
-- and the intermediate volume is not the deliverable, so the first one is spent
on something nobody keeps.

This walks each surface point backwards through the whole chain instead, and
touches the data once, at the end::

    vertex (anatomical)
      --[coregistration]-->  functional reference
      --[rigid, frame t]-->  where the head was in frame t
      --[field warp]------>  where the readout put the signal
      --> sample the original volume

Only the last step interpolates image data. The transforms are composed on the
coordinates themselves, so nothing is resampled on the way.

Where the field warp enters depends on ``sdc_result.space``, for the reason set
out in :mod:`lightprep.resample.fsl`: MEDIC measures the field in each frame as
acquired, whereas a static fieldmap describes it with the head at the reference
position.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from ..hmc.base import check_transforms_replayable
from .base import SurfaceResult
from .ribbon import DEFAULT_DEPTHS, INTERP_ORDERS, _vertices

PE_AXIS_INDEX = {"i": 0, "x": 0, "j": 1, "y": 1, "k": 2, "z": 2}


def _affine_from(path, fmt, reference=None, moving=None):
    from nitransforms.linear import Affine

    kw = {}
    if reference is not None:
        kw["reference"] = str(reference)
    if moving is not None:
        kw["moving"] = str(moving)
    return Affine.from_filename(str(path), fmt=fmt, **kw)


def ribbon_average_concat(
    volume,
    white,
    pial,
    coreg,
    hmc_result,
    sdc_result,
    out,
    *,
    hemi: str,
    depths=DEFAULT_DEPTHS,
    pe_direction: str = "j",
    interp: str = "trilinear",
) -> SurfaceResult:
    """Average the cortical ribbon, reading the *original* volume once.

    Args:
        volume: The original, uncorrected 4D echo -- not a corrected derivative.
        white: White surface, in the anatomy ``coreg`` targets.
        pial: Pial surface, vertices corresponding to ``white``.
        coreg: FreeSurfer LTA from the functional reference to the anatomy.
        hmc_result: Supplies the per-frame rigid transforms and the reference
            grid they are defined on.
        sdc_result: Supplies the displacement map and, through ``space``, where
            in the chain it belongs.
        out: Output path for the per-vertex timeseries (``.func.gii``).
        hemi: ``lh`` or ``rh``.
        depths: Fractional depths between white (0) and pial (1).
        pe_direction: The run's PhaseEncodingDirection; sets which axis the
            displacement runs along.
        interp: Interpolation for the single data read.

    Returns:
        A :class:`~lightprep.surface.base.SurfaceResult`.

    Raises:
        TransformReplayError: If ``hmc_result`` came from a method whose
            transforms do not replay faithfully -- see
            :data:`lightprep.hmc.base.UNREPLAYABLE_METHODS`.
    """
    if interp not in INTERP_ORDERS:
        raise ValueError(f"interp must be one of {sorted(INTERP_ORDERS)}, got {interp!r}")
    axis = PE_AXIS_INDEX.get(pe_direction.rstrip("-"))
    if axis is None:
        raise ValueError(f"unrecognised pe_direction {pe_direction!r}")
    space = getattr(sdc_result, "space", "native")
    if space not in ("native", "reference"):
        raise ValueError(f"sdc_result.space must be 'native' or 'reference', got {space!r}")
    # Every vertex is carried through these transforms; matrices that do not
    # replay would draw the timeseries from the wrong tissue.
    check_transforms_replayable(hmc_result, step="surface.ribbon_average_concat")

    wv, pv = _vertices(Path(white)), _vertices(Path(pial))
    if wv.shape != pv.shape:
        raise ValueError(f"white {wv.shape} and pial {pv.shape} must correspond")

    img = nib.load(str(volume))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        data = data[..., None]
    n_frames = data.shape[3]
    transforms = list(hmc_result.transforms)
    if len(transforms) != n_frames:
        raise ValueError(f"{len(transforms)} rigid transforms for {n_frames} frames")

    dmap_img = nib.load(str(sdc_result.displacement_map))
    dmap = dmap_img.get_fdata(dtype=np.float32)
    static = dmap.ndim == 3
    if not static and dmap.shape[3] not in (1, n_frames):
        raise ValueError(
            f"displacement map has {dmap.shape[3]} frames, expected 1 or {n_frames}"
        )
    if not np.allclose(dmap_img.affine, img.affine, atol=1e-3):
        raise ValueError("displacement map and volume must share a grid")

    aff = img.affine
    inv_aff = np.linalg.inv(aff)
    # The displacement is millimetres along the readout's phase-encoding axis;
    # that axis points along this column of the affine in world space.
    pe_world = aff[:3, axis] / np.linalg.norm(aff[:3, axis])

    # anatomy -> functional reference, once: it does not vary with frame.
    to_ref = _affine_from(coreg, "fs")
    ribbon = pv - wv
    n_vert = wv.shape[0]
    pts_anat = np.concatenate([wv + f * ribbon for f in depths], axis=0)
    pts_ref = np.asarray(to_ref.map(pts_anat))

    def to_vox(p):
        return (inv_aff[:3, :3] @ p.T).T + inv_aff[:3, 3]

    ref_vox = to_vox(pts_ref)
    order = INTERP_ORDERS[interp]
    out_ts = np.empty((n_vert, n_frames), dtype=np.float32)
    ever_outside = np.zeros(len(pts_anat), dtype=bool)
    hi = np.array(data.shape[:3]) - 1

    # Resolve every rigid transform up front. from_filename re-reads the
    # reference image on each call, which otherwise dominates the whole run.
    rigids = [
        np.asarray(
            _affine_from(m, "fsl", reference=hmc_result.reference,
                         moving=hmc_result.reference).matrix,
            dtype=np.float64,
        )
        for m in transforms
    ]

    # A static field read at fixed reference coordinates gives the same shift
    # for every frame, so it is sampled once rather than 138 times.
    static_shift = None
    if space == "reference" and static:
        static_shift = map_coordinates(dmap, ref_vox.T, order=1, mode="nearest")

    for t in range(n_frames):
        M = rigids[t]
        pts_nat = (M[:3, :3] @ pts_ref.T).T + M[:3, 3]

        if static_shift is not None:
            shift = static_shift
        else:
            frame_map = dmap if static else dmap[..., 0 if dmap.shape[3] == 1 else t]
            # A field measured in the frame as acquired is read at the frame's
            # own coordinates; one measured at the reference position is read
            # there, before the head was moved.
            probe = ref_vox if space == "reference" else to_vox(pts_nat)
            shift = map_coordinates(frame_map, probe.T, order=1, mode="nearest")
        pts_acq = pts_nat + shift[:, None] * pe_world[None, :]

        vox = to_vox(pts_acq)
        ever_outside |= ~np.all((vox >= 0) & (vox <= hi), axis=1)
        vals = map_coordinates(data[..., t], vox.T, order=order,
                               mode="constant", cval=np.nan)
        out_ts[:, t] = np.nanmean(vals.reshape(len(depths), n_vert), axis=0)

    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                out_ts[:, t].astype(np.float32),
                intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32",
                encoding="GIFTI_ENCODING_B64GZ",
            )
            for t in range(n_frames)
        ]
    ).to_filename(str(out))

    return SurfaceResult(
        output=out,
        hemi=hemi,
        n_vertices=n_vert,
        n_frames=n_frames,
        depths=tuple(depths),
        n_outside=int(ever_outside.reshape(len(depths), n_vert).any(axis=0).sum()),
        method="ribbon-concat",
    )
