"""Finding the niimath binary, and the affine bookkeeping around it.

niimath (Rorden, https://github.com/rordenlab/niimath) is the one thing the
methods in this package need -- no FSL, no FreeSurfer, no warpkit. That is the
point of preferring it: those steps run on a bare Python environment.

The binary is *not* in the repository. It is a platform-specific build artifact,
and a committed one would be wrong for every machine but the one that built it,
failing in a way that reads as a lightprep bug rather than a missing dependency.
:func:`niimath_path` therefore looks in two places, in order:

1. ``lightprep/niimath``, beside this file -- drop a build there and it wins,
   which keeps a checkout self-contained and pinned to a known version;
2. ``niimath`` on PATH.

``BUILD.md`` has the invocation used to produce the reference build. Whichever
is found, :func:`version` reports it; the methods here need a build recent
enough to carry ``-moco``, ``--medic`` and ``-unwarp``.

Two things need care when using it as a registration engine.

**Affine conventions.** ``-savemat`` writes the transform in *world* millimetres
(the NIfTI sform frame), as the pull transform: it maps a coordinate in the
fixed image to the coordinate in the moving image it should be sampled from.
That is the same direction nitransforms hands back, and so the same direction
:mod:`lightprep.surface.concat` composes with -- but the rest of the package also
has to hand transforms to FSL tools, which want FLIRT matrices in FSL's
scaled-voxel frame. :func:`world_to_fsl` and :func:`fsl_to_world` convert
between the two, so a niimath method is a drop-in for an FSL one.

**4D data.** niimath registers volumes, not timeseries: given a 4D input,
``-allineate`` silently uses the first frame. Anything framewise here therefore
splits with ``-crop``, works per frame, and merges the frames back with nibabel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from ._utils import DependencyError

#: Where a checkout-local build is looked for, beside this file. Not in the
#: repository -- see the module docstring and ``BUILD.md``.
NIIMATH = Path(__file__).resolve().parent / "niimath"

#: niimath features the methods here rely on, and the release that introduced
#: them. Named in the error so an old build on PATH is diagnosable.
REQUIRED_FEATURES = ("-moco", "--medic", "-unwarp")

#: Registration costs that accept ``-warp``, and so can be constrained to rigid.
#: ``fast``/``fastcr`` are excluded on purpose: they run a fixed 12-DOF schedule,
#: which is the wrong model for anything within a single head.
RIGID_COSTS = ("ls", "hel", "lpc", "lpa")

#: ``-warp`` transform types, in increasing degrees of freedom.
WARPS = {"sho": 3, "shr": 6, "srs": 9, "aff": 12}

#: ``-final`` interpolation for resampling, and the scipy spline order it means.
FINAL_INTERPS = {"NN": 0, "linear": 1, "cubic": 3}


def niimath_path() -> Path:
    """Locate the niimath binary: beside this file first, then PATH.

    Raises:
        DependencyError: If neither is usable, with what to do about it.
    """
    if NIIMATH.exists():
        if not os.access(NIIMATH, os.X_OK):
            raise DependencyError(
                f"the niimath build at {NIIMATH} is not executable. "
                f"Run `chmod +x {NIIMATH}`."
            )
        return NIIMATH

    found = shutil.which("niimath")
    if found is not None:
        return Path(found)

    raise DependencyError(
        "niimath was not found. lightprep does not ship it -- a binary is "
        "platform-specific, so it is built rather than committed.\n"
        f"  * build one and put it at {NIIMATH}, or install it on PATH;\n"
        "  * BUILD.md has the invocation used for the reference build;\n"
        "  * sources: https://github.com/rordenlab/niimath\n"
        f"It must be recent enough to carry {', '.join(REQUIRED_FEATURES)}."
    )


def niimath(*args, quiet: bool = True) -> subprocess.CompletedProcess:
    """Run niimath, raising with captured output on failure.

    Args:
        *args: Arguments after the binary itself. Paths and numbers are
            stringified, so they can be passed as-is.
        quiet: Keep niimath's progress chatter out of the caller's terminal.
            It still comes back on the exception if the call fails.
    """
    cmd = [str(niimath_path()), *(str(a) for a in args)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "niimath failed ({}): {}\n--- stdout ---\n{}\n--- stderr ---\n{}".format(
                proc.returncode, " ".join(cmd[1:]), proc.stdout, proc.stderr
            )
        )
    if not quiet and proc.stderr:
        print(proc.stderr)
    return proc


def version() -> str:
    """The version banner of whichever binary :func:`niimath_path` finds."""
    # niimath reports its version on the usage screen, which it prints when it
    # cannot make sense of the arguments.
    proc = subprocess.run(
        [str(niimath_path()), "-h"], capture_output=True, text=True
    )
    for line in proc.stdout.splitlines():
        if "niimath version" in line:
            return line.strip()
    return "unknown"


# --------------------------------------------------------------------------
# Affine conventions
# --------------------------------------------------------------------------


def _fsl_frame(image) -> np.ndarray:
    """Voxel -> FSL scaled-millimetre matrix for an image.

    FSL measures in millimetres from the corner of the volume, one axis at a
    time, and flips the first axis when the sform's determinant is positive --
    its way of making every image look radiological before it does any
    arithmetic. Reproducing that flip is the whole of the FLIRT/world
    conversion; get it wrong and left and right silently swap.
    """
    img = nib.load(str(image))
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    scale = np.diag([*zooms, 1.0])
    if np.linalg.det(img.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = (img.shape[0] - 1) * zooms[0]
        return flip @ scale
    return scale


def world_to_fsl(pull, reference, moving=None) -> np.ndarray:
    """Convert a world-space pull transform into a FLIRT matrix.

    Args:
        pull: 4x4 world-millimetre affine mapping a *reference* coordinate to
            the *moving* coordinate it is sampled from -- niimath's
            ``fixed_to_moving``, and what nitransforms returns.
        reference: The fixed/target image, whose grid the result lands on.
        moving: The image being registered. Defaults to ``reference``, which is
            the framewise case: every frame shares one grid.

    Returns:
        The 4x4 FLIRT matrix, mapping moving to reference in FSL's frame --
        what ``flirt -omat`` writes and ``applywarp``/``convertwarp`` read.
    """
    pull = np.asarray(pull, dtype=np.float64)
    if pull.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine, got shape {pull.shape}")
    moving = reference if moving is None else moving

    a_ref = nib.load(str(reference)).affine
    a_mov = nib.load(str(moving)).affine
    q_ref, q_mov = _fsl_frame(reference), _fsl_frame(moving)
    # world moving->reference is the inverse of the pull; FSL wants it in the
    # scaled-voxel frame at each end.
    push = np.linalg.inv(pull)
    return q_ref @ np.linalg.inv(a_ref) @ push @ a_mov @ np.linalg.inv(q_mov)


def fsl_to_world(matrix, reference, moving=None) -> np.ndarray:
    """Convert a FLIRT matrix into a world-space pull transform.

    The exact inverse of :func:`world_to_fsl`; see it for the arguments.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 affine, got shape {matrix.shape}")
    moving = reference if moving is None else moving

    a_ref = nib.load(str(reference)).affine
    a_mov = nib.load(str(moving)).affine
    q_ref, q_mov = _fsl_frame(reference), _fsl_frame(moving)
    push = a_ref @ np.linalg.inv(q_ref) @ matrix @ q_mov @ np.linalg.inv(a_mov)
    return np.linalg.inv(push)


def read_savemat(path) -> np.ndarray:
    """The world-space pull transform from a niimath ``-savemat`` JSON."""
    doc = json.loads(Path(path).read_text())
    try:
        pull = np.asarray(doc["fixed_to_moving"], dtype=np.float64)
    except KeyError:
        raise ValueError(
            f"{Path(path).name} is not a niimath -savemat file "
            "(no 'fixed_to_moving' matrix)"
        ) from None
    if doc.get("space") not in (None, "world"):
        raise ValueError(f"{Path(path).name}: expected a world-space matrix")
    return pull


def write_savemat(pull, path, *, fixed=None, moving=None) -> Path:
    """Write a world-space pull transform in niimath's ``-savemat`` JSON layout.

    This is what ``-allineate -applymat`` reads, so it is how a transform this
    package computed itself gets replayed by niimath.
    """
    pull = np.asarray(pull, dtype=np.float64)
    doc = {
        "type": "allineate_affine", "version": 1, "space": "world", "units": "mm",
        "engine": "lightprep", "dof": 6,
        "fixed": str(fixed) if fixed is not None else "",
        "moving": str(moving) if moving is not None else "",
        "fixed_to_moving": pull.tolist(),
        "moving_to_fixed": np.linalg.inv(pull).tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    return path


def write_fsl_matrix(matrix, path) -> Path:
    """Write a 4x4 FLIRT matrix in the plain-text layout FSL reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=np.float64), fmt="%.10f")
    return path


def _euler_xyz(rot: np.ndarray) -> tuple[float, float, float]:
    """Rx, Ry, Rz (radians) of the rotation nearest to ``rot``.

    The nearest rotation rather than ``rot`` itself, so that a hair of
    numerical shear cannot throw the angle extraction off.
    """
    u, _, vt = np.linalg.svd(rot)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    ry = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    if np.abs(r[2, 0]) < 1.0 - 1e-9:
        return float(np.arctan2(r[2, 1], r[2, 2])), float(ry), float(np.arctan2(r[1, 0], r[0, 0]))
    # Gimbal lock: pitch is +/-90 degrees and roll and yaw are degenerate.
    return float(np.arctan2(-r[1, 2], r[1, 1])), float(ry), 0.0


def motion_parameters(pull, reference, moving=None) -> np.ndarray:
    """A frame's motion as six numbers, in MCFLIRT's ``.par`` convention.

    Args:
        pull: The frame's world-space pull transform (reference -> frame).
        reference: The image the transform is defined against.
        moving: The image that moved. Defaults to ``reference``; pass it when
            the registration target is on a grid of its own.

    Returns:
        Rotations about x, y, z in radians, then translations in millimetres --
        MCFLIRT's column order. Both are read in FSL's scaled-voxel frame, from
        the reference-to-frame transform, with the translation measured at the
        centre of the volume rather than at FSL's corner origin (a rotation
        about the corner would otherwise report metres of "translation").

        This is MCFLIRT's convention, so the two traces can be compared
        directly. They are not bit-identical: MCFLIRT linearises the
        composition of rotation and translation, and against its own ``.par``
        on this data the difference is 0.026mm RMS -- second-order terms, an
        order of magnitude below the motion itself.
    """
    reference = str(reference)
    # world_to_fsl gives frame -> reference; invert for the pull direction.
    m = np.linalg.inv(world_to_fsl(pull, reference, moving))
    img = nib.load(reference)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    centre = (np.asarray(img.shape[:3], dtype=np.float64) - 1.0) * zooms / 2.0
    trans = -(m[:3, 3] + (m[:3, :3] - np.eye(3)) @ centre)
    return np.array([*_euler_xyz(m[:3, :3]), *trans], dtype=np.float64)


# --------------------------------------------------------------------------
# 4D helpers
# --------------------------------------------------------------------------


def n_frames(image) -> int:
    """Frames in an image; 1 for a 3D volume."""
    shape = nib.load(str(image)).shape
    return shape[3] if len(shape) > 3 else 1


def split_frames(image, out_dir, *, prefix: str = "vol") -> tuple[Path, ...]:
    """Split a 4D image into per-frame volumes, with ``niimath -crop``."""
    image = Path(image).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in range(n_frames(image)):
        dst = out_dir / f"{prefix}{t:04d}.nii.gz"
        niimath(image, "-crop", t, 1, dst)
        frames.append(dst)
    return tuple(frames)


def merge_frames(frames, out, *, template=None) -> Path:
    """Concatenate per-frame volumes into one 4D image.

    niimath has no merge operation -- it is a per-image calculator -- so this is
    the one place the split/apply/merge cycle steps outside it. The header comes
    from ``template`` (the original timeseries) when given, so that ``pixdim[4]``
    and the rest of the timing survive the round trip.
    """
    frames = [Path(f) for f in frames]
    if not frames:
        raise ValueError("no frames to merge")
    first = nib.load(str(frames[0]))
    data = np.stack(
        [np.asanyarray(nib.load(str(f)).dataobj, dtype=np.float32) for f in frames],
        axis=-1,
    )
    header = first.header.copy()
    if template is not None:
        src = nib.load(str(template)).header
        header["pixdim"][4] = src["pixdim"][4]
        header.set_xyzt_units(*src.get_xyzt_units())
    header.set_data_dtype(np.float32)
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(data, first.affine, header).to_filename(str(out))
    return out
