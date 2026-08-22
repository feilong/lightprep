"""Susceptibility distortion correction from a reversed-phase-encoding pair.

The PEPOLAR approach, and the one to reach for on single-echo data with no GRE
fieldmap: two EPI acquisitions that differ only in the polarity of the phase
encoding distort in *opposite* directions, so the displacement that maps one
onto the other is twice the distortion, and the field can be recovered from the
images alone. No phase is needed, which is what makes it work where
:mod:`lightprep.sdc.medic` (multi-echo phase) and :mod:`lightprep.sdc.phasediff`
(a separate GRE scan) both do not.

The estimate comes from FSL ``topup``. niimath has no equivalent -- ``--medic``
reads a field off phase it is given and cannot solve the blip-up/blip-down
inverse problem -- so unlike most steps in this package there is no niimath
method to prefer here. What niimath *does* still do is spend the result:
:func:`lightprep.resample.apply_sdc_niimath` consumes the displacement map this
returns.

Like :mod:`lightprep.sdc.phasediff` this is a static, single estimate, and it is
converted to a shift along the phase-encoding axis with the *functional* run's
readout time::

    displacement(mm) = B0(Hz) * TotalReadoutTime(s) * voxel_size_PE(mm)

so the map is interchangeable with the other methods' and carries the same sign
convention.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .._utils import run, strip_ext
from .base import SDCResult
from .phasediff import _pe_axis_index, _pe_sign_and_axis

#: topup's stock warp-resolution schedule. Its first levels subsample by 2, so
#: every spatial dimension of the input must be even.
DEFAULT_CONFIG = "b02b0.cnf"

#: BIDS axis letter -> the row topup's ``--datain`` expects for that axis.
_AXIS_ROW = {"i": (1, 0, 0), "j": (0, 1, 0), "k": (0, 0, 1)}


def _flip_polarity(pe_direction: str) -> str:
    """The opposite phase-encoding polarity."""
    return pe_direction[:-1] if pe_direction.endswith("-") else pe_direction + "-"


def _acq_row(pe_direction: str, readout_time: float) -> str:
    """One ``--datain`` line: the PE unit vector, then the readout time."""
    sign, axis = _pe_sign_and_axis(pe_direction)
    axis = {"x": "i", "y": "j", "z": "k"}.get(axis, axis)
    vec = _AXIS_ROW[axis]
    return " ".join(f"{sign * v:d}" for v in vec) + f" {readout_time:.6f}"


def _load_4d(path: Path):
    """Load an image as 4D, promoting a 3D volume to a single-frame series."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float64)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    elif data.ndim != 4:
        raise ValueError(f"{path.name}: expected a 3D or 4D image, got {data.ndim}D")
    return img, data


def _config_path(config: str) -> str:
    """Resolve a topup config, preferring FSL's own copy when it is a bare name."""
    if os.sep in str(config):
        return str(config)
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        candidate = Path(fsldir) / "etc" / "flirtsch" / config
        if candidate.exists():
            return str(candidate)
    return str(config)


def pepolar(
    epi,
    opposite,
    target,
    out_dir,
    *,
    pe_direction: str,
    total_readout_time: float,
    opposite_pe_direction: str | None = None,
    epi_readout_time: float | None = None,
    opposite_readout_time: float | None = None,
    config: str = DEFAULT_CONFIG,
    register: bool = False,
    cost: str = "corratio",
    dof: int = 6,
    frames=None,
    max_frames: int = 5,
    keep_workdir: bool = False,
) -> SDCResult:
    """Build a displacement map for ``target`` from a reversed-PE EPI pair.

    Args:
        epi: EPI acquired with the *same* phase encoding as the functional run
            -- typically the run's own reference, or the fieldmap's matching
            polarity image. 3D or 4D.
        opposite: EPI acquired with the reversed polarity (in BIDS, the
            ``fmap/*_epi`` whose ``dir-`` entity is the other way round). 3D or
            4D; it need not have the same number of frames as ``epi``.
        target: The functional reference to produce the displacement map for.
            The output lands on this grid, so pass the reference of the run
            being corrected -- runs may differ in geometry.
        out_dir: Where to write the fieldmap and displacement map.
        pe_direction: The *functional* run's PhaseEncodingDirection (e.g. ``j``,
            ``j-``), which is also ``epi``'s. Sets the axis and the sign of the
            shift.
        total_readout_time: The *functional* run's TotalReadoutTime in seconds.
            This scales the distortion, and is not necessarily either input's
            own readout -- a fieldmap may be acquired with a different readout
            than the run it corrects.
        opposite_pe_direction: ``opposite``'s PhaseEncodingDirection. Defaults
            to ``pe_direction`` reversed, which is the whole premise of the
            method; pass it explicitly only to assert what the sidecar says.
        epi_readout_time: ``epi``'s own TotalReadoutTime, for topup's
            ``--datain``. Defaults to ``total_readout_time``.
        opposite_readout_time: ``opposite``'s own TotalReadoutTime. Defaults to
            ``epi_readout_time``.
        config: topup configuration. A bare name is looked up in
            ``$FSLDIR/etc/flirtsch`` before being passed through. The default
            subsamples by 2, so every spatial dimension must be even.
        register: Register the estimated field to ``target`` before using it.
            Off by default, unlike :func:`~lightprep.sdc.phasediff.phasediff`:
            a reversed-PE fieldmap is normally acquired in the same session at
            the same prescription as the runs it corrects, so it already shares
            their grid. Turn it on if it does not.
        cost: FLIRT cost function, used only when ``register``.
        dof: FLIRT degrees of freedom, used only when ``register``. 6 (rigid)
            is what a within-session fieldmap warrants.
        frames: Which frames of ``epi`` to use, as indices. The default takes
            the first ``max_frames``, which is the earliest the run offers and
            so the closest in time to a field map acquired before it. Pass
            indices to choose on some other ground -- head position being the
            one that matters, since a blip pair estimated across a pose
            difference asks topup to explain a rigid displacement as a field.
            ``opposite`` is always taken from the front: it is seconds long,
            with nothing to choose between its frames.
        max_frames: Frames to take from each input. topup's cost is per frame
            and the field it solves for is static, so a handful of volumes per
            polarity buys everything more would. ``0`` means all of them.
        keep_workdir: Keep the intermediate images and topup's own outputs.

    Returns:
        An :class:`~lightprep.sdc.base.SDCResult` whose ``displacement_map`` is a
        single static 3D map (``n_frames == 1``), on ``target``'s grid.

    Raises:
        ValueError: If the two inputs do not share a voxel grid, if their phase
            encoding is not actually opposed, or if a dimension is odd and the
            chosen config subsamples.
    """
    epi, opposite, target = (Path(p).resolve() for p in (epi, opposite, target))
    for p in (epi, opposite, target):
        if not p.exists():
            raise FileNotFoundError(f"input not found: {p}")
    if total_readout_time <= 0:
        raise ValueError(f"total_readout_time must be positive, got {total_readout_time}")

    sign, axis = _pe_sign_and_axis(pe_direction)
    if opposite_pe_direction is None:
        opposite_pe_direction = _flip_polarity(pe_direction)
    if _flip_polarity(pe_direction) != opposite_pe_direction:
        raise ValueError(
            f"pepolar needs opposed phase encoding, got {pe_direction!r} and "
            f"{opposite_pe_direction!r}. These distort the same way, so there is "
            f"no blip-up/blip-down difference to solve for."
        )
    epi_readout_time = total_readout_time if epi_readout_time is None else epi_readout_time
    opposite_readout_time = (
        epi_readout_time if opposite_readout_time is None else opposite_readout_time
    )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    # 1. Both polarities into one series. topup solves for a single field, so
    #    the two must already sit on the same grid -- fail closed rather than
    #    silently resampling and attributing the mismatch to susceptibility.
    up_img, up = _load_4d(epi)
    down_img, down = _load_4d(opposite)
    if up.shape[:3] != down.shape[:3]:
        raise ValueError(
            f"{epi.name} is {up.shape[:3]} but {opposite.name} is {down.shape[:3]}; "
            f"a reversed-PE pair must share a voxel grid"
        )
    if not np.allclose(up_img.affine, down_img.affine, atol=1e-4):
        raise ValueError(
            f"{epi.name} and {opposite.name} share a shape but not an affine; "
            f"they are not the same prescription"
        )
    if frames is not None:
        picked = np.asarray(frames, dtype=int)
        if picked.ndim != 1 or picked.size == 0:
            raise ValueError(f"frames must be a non-empty 1-D index array, got {frames!r}")
        if picked.min() < 0 or picked.max() >= up.shape[3]:
            raise ValueError(
                f"frames index {picked.min()}..{picked.max()} outside "
                f"{epi.name}'s {up.shape[3]} frames"
            )
        up = up[..., picked]
        if max_frames:
            down = down[..., :max_frames]
    elif max_frames:
        up = up[..., :max_frames]
        down = down[..., :max_frames]

    if "--subsamp" not in str(config) and any(d % 2 for d in up.shape[:3]):
        raise ValueError(
            f"topup config {config!r} subsamples by 2 but the input is "
            f"{up.shape[:3]}; every spatial dimension must be even"
        )

    merged = work / "blip_pair.nii.gz"
    nib.Nifti1Image(
        np.concatenate([up, down], axis=3), up_img.affine, up_img.header
    ).to_filename(merged)

    # 2. One --datain row per frame, in the order they were concatenated.
    acqparams = work / "acqparams.txt"
    acqparams.write_text(
        "\n".join(
            [_acq_row(pe_direction, epi_readout_time)] * up.shape[3]
            + [_acq_row(opposite_pe_direction, opposite_readout_time)] * down.shape[3]
        )
        + "\n"
    )

    # 3. Solve. --fout is the field in Hz, which is what the shift is built from.
    fmap_hz = work / "topup_field_hz.nii.gz"
    run(["topup",
         f"--imain={merged}",
         f"--datain={acqparams}",
         f"--config={_config_path(config)}",
         f"--out={work / 'topup'}",
         f"--fout={work / strip_ext(fmap_hz)}",
         f"--iout={work / 'topup_corrected'}"])

    # 4. Onto the target grid. Normally a no-op -- see `register`.
    if register:
        xfm = out_dir / "fmap2target.mat"
        corrected_mean = work / "topup_corrected_mean.nii.gz"
        run(["fslmaths", work / "topup_corrected", "-Tmean", corrected_mean])
        run(["flirt", "-in", corrected_mean, "-ref", target, "-omat", xfm,
             "-dof", dof, "-cost", cost, "-out", work / "fmap_on_target.nii.gz"])
        fmap_on_target = work / "field_hz_on_target.nii.gz"
        run(["flirt", "-in", fmap_hz, "-ref", target, "-applyxfm", "-init", xfm,
             "-out", fmap_on_target, "-interp", "trilinear"])
    else:
        target_img = nib.load(target)
        field_img = nib.load(fmap_hz)
        if field_img.shape[:3] != target_img.shape[:3]:
            raise ValueError(
                f"the estimated field is {field_img.shape[:3]} but {target.name} is "
                f"{target_img.shape[:3]}; pass register=True to resample it"
            )
        fmap_on_target = fmap_hz

    # 5. Hz -> mm of shift along the PE axis, using the *target's* readout.
    ref_img = nib.load(target)
    voxel_pe = float(ref_img.header.get_zooms()[_pe_axis_index(axis)])
    field = nib.load(fmap_on_target)
    # float64 for the Hz -> mm conversion; the images are written float32,
    # which is the NIfTI convention, but the arithmetic is not done in it.
    hz = field.get_fdata(dtype=np.float64)

    fieldmap = out_dir / "fieldmap_hz.nii.gz"
    nib.Nifti1Image(
        hz.astype(np.float32), field.affine, field.header
    ).to_filename(fieldmap)

    displacement = sign * hz * total_readout_time * voxel_pe
    dmap = out_dir / "displacementmap.nii.gz"
    nib.Nifti1Image(
        displacement.astype(np.float32), field.affine, field.header
    ).to_filename(dmap)

    if not keep_workdir:
        shutil.rmtree(work, ignore_errors=True)

    return SDCResult(
        fieldmap_native=fieldmap,
        displacement_map=dmap,
        fieldmap=fieldmap,
        method="pepolar",
        n_echoes=1,
        n_frames=1,
        # Static, like phasediff: the field is measured once, with the head at
        # the position the fieldmap was acquired in, so it is applied only once
        # motion correction has put every frame at the reference.
        space="reference",
    )
