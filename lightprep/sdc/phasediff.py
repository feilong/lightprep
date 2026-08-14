"""Susceptibility distortion correction from a GRE phase-difference fieldmap.

The classic approach: a separate dual-echo gradient-echo scan measures B0 once,
and that single static map is assumed to hold for the whole run. Unlike
:mod:`lightprep.sdc.medic` this needs no phase in the functional data, so it works
on single-echo runs -- but it cannot track the field as the head moves, and it
must be registered into each run's space because it is a different acquisition.

The field comes from the phase accrued between the two echoes::

    B0(Hz) = phase_difference(rad) / (2*pi*dTE)

which is turned into a voxel shift along the phase-encoding axis using the
*functional* run's readout time, not the fieldmap's::

    displacement(mm) = B0(Hz) * TotalReadoutTime(s) * voxel_size_PE(mm)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .._utils import run, strip_ext
from .base import SDCResult

#: FSL's SIEMENS branch expects the phase difference on this scale.
FSL_PHASE_MAX = 4095


def _rescale_phase(src: Path, dst: Path) -> tuple[float, float]:
    """Put a Siemens phase-difference image on FSL's expected 0..4095 scale.

    ``fsl_prepare_fieldmap`` converts with ``(x/2048 - 1)*pi``, which is only
    correct for input spanning 0..4096. Siemens exports vary (0..4095 and
    -4096..4094 both occur), and the script's own "MRIcron double range"
    heuristic mis-handles the signed convention -- it halves -4096..4094 to
    -2048..2047 and yields [-2*pi, 0] rather than [-pi, pi]. Rescaling from the
    observed range sidesteps the guesswork.
    """
    img = nib.load(src)
    data = img.get_fdata(dtype=np.float64)
    lo, hi = float(data.min()), float(data.max())
    if hi <= lo:
        raise ValueError(f"phase image {src.name} has no range: min==max=={lo}")
    scaled = (data - lo) / (hi - lo) * FSL_PHASE_MAX
    nib.Nifti1Image(scaled, img.affine, img.header).to_filename(dst)
    return lo, hi


def _pe_sign_and_axis(pe_direction: str) -> tuple[int, str]:
    """Split a BIDS PhaseEncodingDirection into a sign and a bare axis."""
    axis = pe_direction.rstrip("-")
    if axis not in ("i", "j", "k", "x", "y", "z"):
        raise ValueError(f"unrecognised PhaseEncodingDirection {pe_direction!r}")
    return (-1 if pe_direction.endswith("-") else 1), axis


def _pe_axis_index(axis: str) -> int:
    return {"i": 0, "x": 0, "j": 1, "y": 1, "k": 2, "z": 2}[axis]


def phasediff(
    phase,
    magnitude,
    target,
    out_dir,
    *,
    delta_te: float,
    total_readout_time: float,
    pe_direction: str,
    register: bool = True,
    bet_frac: float = 0.5,
    cost: str = "corratio",
    dof: int = 6,
    keep_workdir: bool = False,
) -> SDCResult:
    """Build a displacement map for ``target`` from a GRE phase-difference scan.

    Args:
        phase: The phase-difference image (Siemens native units).
        magnitude: A magnitude image from the same fieldmap scan. Used for
            brain extraction and, if registering, as the moving image.
        target: The functional reference to produce the displacement map for.
            The output lands on this grid, so pass the reference of the run
            being corrected -- runs may differ in geometry.
        out_dir: Where to write the fieldmap and displacement map.
        delta_te: Echo time difference of the fieldmap scan, in seconds
            (``EchoTime2 - EchoTime1``).
        total_readout_time: The *functional* run's TotalReadoutTime in seconds.
            This scales the distortion, and it differs between protocols -- a
            faster multi-echo readout distorts less than a single-echo one.
        pe_direction: The *functional* run's PhaseEncodingDirection (e.g.
            ``j``, ``j-``). Sets the axis and the sign of the shift.
        register: Register the fieldmap to ``target`` before resampling. Off
            only if you know the two are already in register; the fieldmap is a
            separate acquisition, so leaving it on is the safe default.
        bet_frac: Fractional intensity threshold for brain extraction.
        cost: FLIRT cost function. The default suits the differing contrast
            between a GRE magnitude and an EPI.
        dof: FLIRT degrees of freedom; 6 (rigid) is what a within-session
            fieldmap warrants.
        keep_workdir: Keep the intermediate images.

    Returns:
        An :class:`~lightprep.sdc.base.SDCResult` whose ``displacement_map`` is a
        single static 3D map (``n_frames == 1``), on ``target``'s grid.
    """
    phase, magnitude, target = (Path(p).resolve() for p in (phase, magnitude, target))
    for p in (phase, magnitude, target):
        if not p.exists():
            raise FileNotFoundError(f"input not found: {p}")
    if delta_te <= 0:
        raise ValueError(f"delta_te must be positive, got {delta_te}")
    if total_readout_time <= 0:
        raise ValueError(f"total_readout_time must be positive, got {total_readout_time}")
    sign, axis = _pe_sign_and_axis(pe_direction)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    # 1. Phase onto FSL's scale, then brain-extract the magnitude for prelude.
    phase_fsl = work / "phasediff_fsl.nii.gz"
    _rescale_phase(phase, phase_fsl)
    mag_brain = work / "mag_brain.nii.gz"
    run(["bet", magnitude, mag_brain, "-f", bet_frac, "-m"])
    mag_mask = work / "mag_brain_mask.nii.gz"

    # 2. Unwrap and convert to rad/s. deltaTE is given to FSL in milliseconds.
    fmap_rads = work / "fmap_rads.nii.gz"
    run(["fsl_prepare_fieldmap", "SIEMENS", phase_fsl, mag_brain, fmap_rads, delta_te * 1000])

    # 3. Onto the target grid, registering first unless told not to.
    fmap_on_target = work / "fmap_rads_on_target.nii.gz"
    mask_on_target = work / "mask_on_target.nii.gz"
    if register:
        xfm = out_dir / "fmap2target.mat"
        run(["flirt", "-in", mag_brain, "-ref", target, "-omat", xfm,
             "-out", work / "mag_on_target.nii.gz", "-dof", dof, "-cost", cost])
    else:
        xfm = work / "identity.mat"
        xfm.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    for src, dst, interp in ((fmap_rads, fmap_on_target, "trilinear"),
                             (mag_mask, mask_on_target, "nearestneighbour")):
        run(["flirt", "-in", src, "-ref", target, "-applyxfm", "-init", xfm,
             "-out", dst, "-interp", interp])

    # 4. rad/s -> Hz -> mm of shift along the PE axis, using the *target's* readout.
    ref_img = nib.load(target)
    voxel_pe = float(ref_img.header.get_zooms()[_pe_axis_index(axis)])
    rads = nib.load(fmap_on_target)
    # float64 for the rad/s -> Hz -> mm conversion; written out as float32.
    hz = rads.get_fdata(dtype=np.float64) / (2 * np.pi)
    mask = nib.load(mask_on_target).get_fdata() > 0.5
    hz *= mask

    fieldmap = out_dir / "fieldmap_hz.nii.gz"
    nib.Nifti1Image(
        hz.astype(np.float32), rads.affine, rads.header
    ).to_filename(fieldmap)

    displacement = sign * hz * total_readout_time * voxel_pe
    dmap = out_dir / "displacementmap.nii.gz"
    nib.Nifti1Image(displacement.astype(np.float32), rads.affine, rads.header).to_filename(dmap)

    if not keep_workdir:
        shutil.rmtree(work, ignore_errors=True)

    return SDCResult(
        fieldmap_native=fieldmap,
        displacement_map=dmap,
        fieldmap=fieldmap,
        method="phasediff",
        n_echoes=2,
        n_frames=1,
        # The perturbation is generated by the head's own tissue-air boundaries,
        # so it travels with the head rather than sitting still in the scanner.
        # This map describes the field with the head at the reference position
        # (where it was measured and registered to), so it is applied only once
        # motion correction has put every frame there.
        space="reference",
    )
