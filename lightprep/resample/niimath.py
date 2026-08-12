"""Applying a distortion map with niimath, via ``-unwarp``.

This is the niimath counterpart to :func:`lightprep.resample.fsl.apply_sdc`: it
undistorts a timeseries with a framewise displacement map, needing neither
warpkit nor FSL. It reproduces ``wk-apply-warp`` to the last bit -- on the
pilot's three-echo run the two outputs correlate 1.00000 -- so it is a true
substitution and not merely a similar one.

As in :mod:`lightprep.sdc.niimath`, the sign has to be dealt with. ``-unwarp``
reads a displacement map in niimath's own polarity, which runs opposite to the
convention the rest of this package uses (warpkit's, and the one
:mod:`lightprep.sdc.phasediff` writes). The map is therefore negated on the way in.
This is not a cosmetic detail: applying the map unnegated leaves the data
correlating 0.977 with the correct result, where leaving it *uncorrected*
correlates 0.991 -- the wrong sign is worse than doing nothing, and it does not
look wrong.

The same caveat as the FSL version applies to what this is for. The output has
been interpolated once already, so feeding it through HMC and keeping that
result costs a second pass; use it to produce data to *estimate* motion on, and
:func:`lightprep.resample.compose_and_apply` for the data you keep.
"""

from __future__ import annotations

from pathlib import Path

from .._niimath import niimath
from .._utils import strip_ext
from .fsl import PE_AXES, _check_axis


def apply_sdc(images, displacement_map, out_dir, *, pe_axis: str = "j",
              keep_workdir: bool = False) -> tuple[Path, ...]:
    """Undistort images with a framewise displacement map.

    Args:
        images: 4D timeseries to undistort.
        displacement_map: Displacement map in *this package's* sign convention,
            i.e. what :mod:`lightprep.sdc` writes. 3D maps are broadcast over
            frames; 4D maps must match the frame count.
        out_dir: Where to write the undistorted images.
        pe_axis: Axis the displacement map runs along, from the sidecar's
            PhaseEncodingDirection. A trailing ``-`` is accepted and ignored:
            the sign lives in the map, not in the axis.
        keep_workdir: Keep the sign-corrected copy of the displacement map.

    Returns:
        Paths of the undistorted images, in input order.

    Raises:
        ValueError: If ``pe_axis`` is not one of :data:`lightprep.resample.fsl.PE_AXES`,
            or an output would overwrite its input.
    """
    _check_axis(pe_axis)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    displacement_map = Path(displacement_map).resolve()
    if not displacement_map.exists():
        raise FileNotFoundError(f"displacement map not found: {displacement_map}")

    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    # Into niimath's polarity. See the module docstring: this is load-bearing.
    flipped = work / "displacementmap_niimath.nii.gz"
    niimath(displacement_map, "-mul", -1, flipped)

    # -unwarp takes a bare axis; the sign is already carried by the map.
    axis = pe_axis.rstrip("-")

    outputs = []
    for img in images:
        img = Path(img).resolve()
        dst = out_dir / f"{strip_ext(img)}.nii.gz"
        if dst == img:
            raise ValueError(f"out_dir would overwrite the input: {img}")
        niimath(img, "-unwarp", flipped, axis, dst)
        outputs.append(dst)

    if not keep_workdir:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    return tuple(outputs)


__all__ = ["apply_sdc", "PE_AXES"]
