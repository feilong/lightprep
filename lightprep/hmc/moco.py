"""Head motion correction with niimath, via ``-moco``.

``-moco`` is niimath's clean-room implementation of Cox & Jesmanowicz (*MRM*
42:1014-1018, 1999) -- the algorithm behind AFNI's ``3dvolreg``, written for
exactly this job. A rigid transform is factored into four 3D shears, each of
which displaces data along a single index axis, so every interpolation is a
constant shift of a contiguous row; and the six parameters are fitted by
repeated linearization rather than by searching.

That is what makes it the default. :func:`~lightprep.hmc.niimath.allineate` runs a
general-purpose registration per frame, and its coarse pass -- a 64-point grid
plus 101 random trials -- is nearly all of its cost, spent rediscovering that
frames of one run are already almost aligned. On the pilot's 138-frame run that
is roughly 700s of estimation against **18-25s** here.

The catch is that ``-moco`` reports its estimate as six numbers per volume, in
AFNI's ``-1Dfile`` layout, and never writes a matrix. Everything downstream in
this package consumes per-frame transforms, so this module reconstructs them.
The convention is taken from niimath's own source rather than guessed, and each
piece was checked:

* rotation ``R = Ry(yaw) . Rx(pitch) . Rz(roll)``, in AFNI DICOM (LPS) axes and
  degrees -- round-trips against the source's own ``moco_unrot`` to 3e-15 deg;
* translation ``s = (dL, dP, dS)``, DICOM millimetres -- a reconstruction
  reproduces the reported values exactly;
* the transform acts about the grid centre, ``T(x) = R(x - c) + c + s``. The
  source pads by ``MOCO_PAD = 4`` voxels a side and takes ``c`` at the centre of
  the *padded* grid, which in the original grid's index units is exactly
  ``(dim - 1) / 2``;
* the resampling direction is ``out = in . T^-1``, so the pull transform -- the
  one nitransforms and :mod:`lightprep.surface.concat` want -- is ``T^-1``.

``-moco`` also always registers onto volume 0, which ``ref`` would otherwise
choose. That costs nothing once the transforms exist: re-referencing to any
frame is composition, ``P'_t = P_t . P_ref^-1``, done here in closed form. A
reference that is not one of the frames (``"mean"``, or an image of your own)
takes one extra registration to place it against volume 0.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .._niimath import (FINAL_INTERPS, merge_frames, motion_parameters, niimath,
                        read_savemat, split_frames, world_to_fsl,
                        write_fsl_matrix, write_savemat)
from .._utils import strip_ext
from .base import HMCResult
from .fsl import _check_echoes

#: ``-final`` interpolations, for applying the estimated transforms.
INTERPOLATIONS = tuple(FINAL_INTERPS)

#: DICOM (LPS) <-> NIfTI world (RAS).
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def _rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """DICOM-axis rotation matrix, as niimath's ``moco_rot`` builds it."""
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return ry @ rx @ rz


def parameters_to_pull(row, affine, shape) -> np.ndarray:
    """One row of a ``-moco`` ``.1D`` file as a world-space pull transform.

    Args:
        row: ``(roll, pitch, yaw, dS, dL, dP)`` -- degrees counter-clockwise
            about I-S, R-L and A-P, then millimetres toward Superior, Left and
            Posterior. That is AFNI's ``-1Dfile`` layout, which is what
            ``-moco`` writes.
        affine: The series' voxel-to-world (RAS) affine.
        shape: The series' spatial dimensions.

    Returns:
        The 4x4 world-millimetre affine mapping a coordinate in volume 0 to the
        coordinate in this volume it is sampled from -- the same direction
        niimath's ``-savemat`` writes, and the direction nitransforms returns.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (6,):
        raise ValueError(f"expected 6 motion parameters, got shape {row.shape}")

    rot = _LPS_TO_RAS @ _rotation(*row[:3]) @ _LPS_TO_RAS
    # The file orders the shifts S, L, P; DICOM axis order is L, P, S.
    shift = _LPS_TO_RAS @ np.array([row[4], row[5], row[3]])

    affine = np.asarray(affine, dtype=np.float64)
    centre_idx = (np.asarray(shape[:3], dtype=np.float64) - 1.0) / 2.0
    centre = affine[:3, :3] @ centre_idx + affine[:3, 3]

    push = np.eye(4)
    push[:3, :3] = rot
    push[:3, 3] = centre - rot @ centre + shift
    # niimath resamples as out = in . T^-1, so the pull transform is T^-1.
    return np.linalg.inv(push)


def _reference_pull(ref, pulls, echo: Path, out_dir: Path, work: Path):
    """The reference image, and the pull that carries volume 0 onto it."""
    n_volumes = len(pulls)

    if isinstance(ref, (str, Path)) and str(ref) not in ("middle", "mean"):
        ref_path = Path(ref).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(f"reference image not found: {ref_path}")
        return ref_path, _register_to_volume0(ref_path, echo, work)

    if ref == "mean":
        target = out_dir / "reference.nii.gz"
        niimath(echo, "-Tmean", target)
        return target, _register_to_volume0(target, echo, work)

    index = n_volumes // 2 if ref == "middle" else int(ref)
    if not 0 <= index < n_volumes:
        raise ValueError(
            f"reference volume {index} is out of range for {n_volumes} volumes"
        )
    target = out_dir / "reference.nii.gz"
    niimath(echo, "-crop", index, 1, target)
    # The reference is one of the frames, so its transform is already known and
    # nothing needs registering.
    return target, pulls[index]


def _register_to_volume0(image: Path, echo: Path, work: Path) -> np.ndarray:
    """Place an image that is not one of the frames against volume 0."""
    volume0 = work / "volume0.nii.gz"
    if not volume0.exists():
        niimath(echo, "-crop", 0, 1, volume0)
    savemat = work / "reference_to_volume0.json"
    niimath(image, "-allineate", volume0, "-cost", "ls", "-warp", "shr",
            "-savemat", savemat, work / "reference_on_volume0.nii.gz")
    # -allineate gives fixed(volume0) -> moving(image); we want volume0 -> ref.
    return read_savemat(savemat)


def moco(
    echoes,
    out_dir,
    *,
    ref="middle",
    ref_echo: int = 0,
    interp: str = "linear",
    keep_workdir: bool = False,
) -> HMCResult:
    """Estimate head motion on one echo and apply it to every echo.

    Motion is estimated once, on ``ref_echo`` (the first echo by default: it has
    the shortest TE, so the most signal and the least dropout). The resulting
    per-volume transforms are then replayed unchanged onto all echoes. Echoes
    come from a single excitation and therefore share a head position, so a
    per-echo estimate would only add noise -- and would break the voxelwise
    correspondence across echoes that T2*/S0 fitting relies on.

    Args:
        echoes: Paths to the echo timeseries, ordered by echo time. A
            single-echo run is just a list of length one.
        out_dir: Directory for the realigned echoes and the motion estimates.
        ref: Registration target. ``"middle"`` (default) uses the middle volume,
            ``"mean"`` the mean volume, an int a specific volume index, or pass
            a path to use an existing image. ``-moco`` itself always registers
            onto volume 0; anything else is reached by composition afterwards,
            which is exact, and for ``"mean"`` or a path costs one extra
            registration.
        ref_echo: Index into ``echoes`` to estimate motion on. Defaults to the
            first echo; pass ``1`` for the mid-TE echo that afni_proc.py favours.
        interp: Interpolation used when applying the transforms, one of
            :data:`INTERPOLATIONS`. Note that this is *not* the kernel ``-moco``
            uses internally (an 8-tap Lagrange): its own corrected output is
            discarded, so that every echo is resampled once, the same way, by
            the same call.
        keep_workdir: Keep the per-frame volumes, the ``.1D`` file and the
            reconstructed JSON transforms.

    Returns:
        An :class:`~lightprep.hmc.base.HMCResult`. ``transforms`` are FLIRT
        matrices, as the other methods' are, so this is a drop-in for them.
        ``parameters`` is a six-column ``motion.par`` in MCFLIRT's convention --
        rotations (radians) then translations (mm) -- *not* the AFNI-ordered
        degrees that ``-moco`` itself writes, so the traces of all three methods
        can be compared directly.

    Raises:
        ValueError: If the echoes disagree in shape, or an argument is invalid.
        DependencyError: If no niimath binary can be found.

    Note:
        Phase images must not be passed here. Interpolating wrapped phase across
        the +/-pi boundary produces nonsense. Estimate on magnitude, then apply
        these transforms to the real and imaginary parts instead.
    """
    if interp not in INTERPOLATIONS:
        raise ValueError(f"interp must be one of {INTERPOLATIONS}, got {interp!r}")

    paths, n_volumes = _check_echoes(echoes)
    if not 0 <= ref_echo < len(paths):
        raise ValueError(
            f"ref_echo {ref_echo} is out of range for {len(paths)} echo(s)"
        )
    if n_volumes < 2:
        raise ValueError(
            f"-moco needs at least 2 volumes to have anything to align, got {n_volumes}"
        )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [out_dir / f"{strip_ext(p)}.nii.gz" for p in paths]
    clobbered = [str(p) for p, o in zip(paths, outputs) if p == o]
    if clobbered:
        raise ValueError(
            "out_dir would overwrite the input echo(es): " + ", ".join(clobbered)
        )

    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    transform_dir = out_dir / "transforms"
    transform_dir.mkdir(exist_ok=True)

    # 1. Estimate, in one pass over the whole series.
    par_1d = work / "moco.1D"
    niimath(paths[ref_echo], "-moco", "-1Dfile", par_1d, work / "moco_corrected.nii.gz")
    rows = np.atleast_2d(np.loadtxt(par_1d))
    if rows.shape != (n_volumes, 6):
        raise RuntimeError(
            f"-moco wrote {rows.shape} motion parameters for {n_volumes} volumes"
        )

    img = nib.load(str(paths[ref_echo]))
    pulls = [parameters_to_pull(row, img.affine, img.shape) for row in rows]

    # 2. Re-reference. -moco works against volume 0; composing with the
    #    reference's own transform moves the whole series onto whatever `ref`
    #    asked for, exactly.
    reference, ref_pull = _reference_pull(ref, pulls, paths[ref_echo], out_dir, work)
    inv_ref = np.linalg.inv(ref_pull)
    pulls = [pull @ inv_ref for pull in pulls]

    transforms, parameters = [], []
    for t, pull in enumerate(pulls):
        transforms.append(
            write_fsl_matrix(world_to_fsl(pull, reference, moving=paths[ref_echo]),
                             transform_dir / f"MAT_{t:04d}")
        )
        parameters.append(
            motion_parameters(pull, reference, moving=paths[ref_echo])
        )
        write_savemat(pull, work / f"xfm{t:04d}.json",
                      fixed=reference, moving=paths[ref_echo])

    # 3. Replay the one set of transforms onto every echo.
    for src, dst in zip(paths, outputs):
        apply_dir = work / f"apply_{strip_ext(src)}"
        corrected = []
        for t, frame in enumerate(split_frames(src, apply_dir, prefix="vol")):
            out = apply_dir / f"corr{t:04d}.nii.gz"
            niimath(frame, "-allineate", reference,
                    "-applymat", work / f"xfm{t:04d}.json",
                    "-final", interp, out)
            corrected.append(out)
        merge_frames(corrected, dst, template=src)

    par = out_dir / "motion.par"
    np.savetxt(par, np.asarray(parameters), fmt="%.8g", delimiter="  ")

    if not keep_workdir:
        shutil.rmtree(work, ignore_errors=True)

    return HMCResult(
        outputs=tuple(outputs),
        reference=reference,
        transforms=tuple(transforms),
        parameters=par,
        ref_echo=ref_echo,
        method="moco",
    )
