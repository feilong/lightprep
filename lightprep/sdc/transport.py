"""Carry a static field from the head position it was measured at onto another.

A ``space="reference"`` map is head-fixed: the B0 perturbation is produced by
the head's own tissue-air boundaries, so the map describes the field in the
*head's* frame. A resampler uses that by probing the map at reference
coordinates, before the rigid transform has moved anything -- see
:func:`lightprep.resample.compose_and_apply` and
:func:`lightprep.surface.ribbon_average_concat`, which do the same thing in
field form and in coordinate form respectively.

Head-fixed relative to *which* head position, though. A field estimated from
one run's frames is indexed by that run's reference pose. Handed unchanged to a
run whose subject has since moved, it is read at the wrong anatomy: the grids
match, so nothing complains, but voxel ``x`` is no longer the tissue the field
was measured on. Over a session this is the difference between the field map
and where the head ended up -- millimetres and degrees, not a rounding error.

This module closes that gap by rigidly registering the two references and
resampling the map between them. What it fixes is *where* the field sits. It
cannot fix what the field *is*: susceptibility depends on how the head sits in
B0, so a rotated head has a genuinely different field, and no rigid transform
recovers information that was never measured. Transport is exact under
translation and approximate under rotation.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from .._niimath import niimath
from .._utils import strip_ext
from .base import SDCResult


def transport(sdc_result: SDCResult, measured_at, onto, out_dir, *,
              cost: str = "ls", interp: str = "linear",
              keep_workdir: bool = False) -> SDCResult:
    """Re-index a head-fixed field onto a different reference pose.

    Args:
        sdc_result: A result with ``space == "reference"``. A ``"native"`` map
            describes a frame as acquired and has no head frame to move
            between, so passing one is an error rather than a no-op.
        measured_at: The image the field is currently indexed by -- the
            reference the estimate was made against.
        onto: The reference to re-index it onto, normally a run's HMC
            reference. Both are single volumes of the same anatomy, and both
            are acquired with the *same* phase encoding, so this registration
            never crosses a blip pair.
        out_dir: Where to write the moved map.
        cost: Registration cost for the rigid fit.
        interp: Interpolation for resampling the map. The map is smooth and
            in millimetres, so linear is enough and does not overshoot.
        keep_workdir: Keep the rigid fit and the resampled intermediate.

    Returns:
        A copy of ``sdc_result`` whose ``displacement_map`` and ``fieldmap``
        are indexed by ``onto``. ``fieldmap_native`` is passed through: it
        describes the acquisition, which no head position changes.

    Raises:
        ValueError: If the field is not a reference-space one.
    """
    if getattr(sdc_result, "space", "native") != "reference":
        raise ValueError(
            f"only a reference-space field has a head frame to move between; "
            f"this one is {sdc_result.space!r}"
        )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    measured_at = Path(measured_at).resolve()
    onto = Path(onto).resolve()

    # -allineate writes fixed -> moving, so registering `measured_at` against
    # `onto` gives exactly the pull we need: for each coordinate of `onto`,
    # where to read it in the frame the field was measured in.
    xfm = work / "measured_at_to_onto.json"
    niimath(measured_at, "-allineate", onto, "-cost", cost, "-warp", "shr",
            "-savemat", xfm, work / "measured_at_on_onto.nii.gz")

    moved = {}
    for name in ("displacement_map", "fieldmap"):
        src = Path(getattr(sdc_result, name)).resolve()
        dst = out_dir / f"{strip_ext(src)}.nii.gz"
        if dst == src:
            raise ValueError(f"out_dir would overwrite the input: {src}")
        niimath(src, "-allineate", onto, "-applymat", xfm, "-final", interp, dst)
        moved[name] = dst

    if not keep_workdir:
        shutil.rmtree(work, ignore_errors=True)
    return replace(sdc_result, **moved)
