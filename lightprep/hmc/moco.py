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

The ``-1Dfile`` text is AFNI's frozen ``%8.4f`` layout, parsed downstream by
column position and so unwidenable, which rounds the fit to 1e-4 degrees and
1e-4 mm. Every transform here is *rebuilt* from those numbers rather than read
from a matrix, so that rounding would propagate into the resampling. Where
niimath offers ``-bin`` this module takes the float64 companion instead and
never sees the text; :func:`_read_parameters` falls back to it only for older
binaries.

Which volume everything is fitted to is the other thing this module takes a
position on. ``-moco`` aims at volume 0 unless told otherwise, and volume 0 is
a poor default: it is the frame most likely to be disturbed, and a disturbed
target contaminates every row fitted to it. Where the binary carries ``-ref``
the target is handed to the estimator, so the fit really is made against it;
where it does not, re-referencing is still exact as a change of coordinates --
``P'_t = P_t . P_ref^-1`` -- but the registration underneath remains against
volume 0, and no composition can undo a bad base.

``ref="stable"`` picks the target from the data rather than by convention.
:func:`relative_motion` runs ``-moco -relative``, which fits each volume onto
its predecessor and writes no image, giving raw frame-to-frame displacement
before anything is corrected; :func:`best_reference` then takes the volume with
the smallest displacement on *both* sides. It costs a few seconds for a couple
of hundred frames, and the same measurement doubles as the run's motion trace
for QC -- one that no choice of reference can flatter.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .._niimath import (FINAL_INTERPS, merge_frames, motion_parameters, niimath,
                        niimath_path,
                        read_savemat, split_frames, world_to_fsl,
                        write_fsl_matrix, write_savemat)
from .._utils import DependencyError, save_trace, strip_ext
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


def _help_text() -> str:
    """niimath's own help, for feature detection."""
    import subprocess
    try:
        return subprocess.run([str(niimath_path()), "-h"], capture_output=True,
                              text=True).stdout
    except Exception:                                   # noqa: BLE001
        return ""


def supports_ref() -> bool:
    """Whether the niimath on this machine can estimate onto a chosen target.

    ``-moco -ref`` is recent. Without it, motion can only be estimated against
    volume 0 and moved onto another reference afterwards -- which changes the
    coordinate frame and the output grid, but not what the registration was
    fitted to. That distinction matters when volume 0 is itself a bad frame.
    """
    return "-ref <n|img>" in _help_text()


def supports_relative() -> bool:
    """Whether the niimath on this machine can measure frame-to-frame motion.

    ``-moco -relative`` fits each volume onto its *predecessor* and writes
    nothing but the parameters. It is what :data:`REF_STABLE` is built on.
    """
    return "-relative" in _help_text()


def supports_bin() -> bool:
    """Whether this niimath can write the float64 companion to a ``.1D``.

    The ``-1Dfile`` layout is AFNI's, six ``%8.4f`` fields parsed by column
    position, so it cannot be widened -- it rounds the fit to 1e-4 degrees and
    1e-4 mm. ``-bin`` writes the unrounded parameters alongside it, and
    everything here reads those where they exist.
    """
    return "-bin adds" in _help_text()


def _read_parameters(par_1d: Path, n_volumes: int) -> np.ndarray:
    """The six-column estimate, at the best precision this niimath wrote it.

    Prefers the ``-bin`` companion -- ``nt * 6`` little-endian float64,
    row-major, no header -- and falls back to parsing the text, which is all an
    older binary leaves behind.
    """
    companion = Path(str(par_1d) + ".bin")
    if companion.exists():
        rows = np.fromfile(companion, dtype="<f8").reshape(-1, 6)
    else:
        rows = np.atleast_2d(np.loadtxt(par_1d)).astype(np.float64)
    if rows.shape != (n_volumes, 6):
        raise RuntimeError(
            f"-moco wrote {rows.shape} motion parameters for {n_volumes} volumes"
        )
    return rows


#: Radius converting rotation to surface displacement, following Power et al.
#: (2012). Their 50mm is roughly the cortex-to-centre distance, so a radian of
#: rotation moves cortex by 50mm.
FD_RADIUS_MM = 50.0

#: ``ref`` value asking for the least-disturbed volume, chosen by a ``-relative``
#: measurement pass over the raw series before anything is registered.
REF_STABLE = "stable"

#: ``ref`` value asking for a groupwise target: the average of the corrected
#: series, rebuilt from the original data each round. See
#: :func:`groupwise_reference`.
REF_GROUPWISE = "groupwise"

#: Seconds over which a movement still counts against a later frame's spin
#: history. Defined in time rather than frames so it means the same at any TR.
#: Linear decay to zero here stands in for the exponential recovery T1 gives;
#: at 3T (T1 ~ 1.4s) six seconds is about four T1, by which point the
#: magnetisation has largely forgotten.
SPIN_HISTORY_S = 6.0

#: Frames of recent history a seed volume is asked to have been quiet through.
#: A movement disturbs more than the frame it happens in: the spins carry the
#: perturbed history for several TR before the steady state recovers, so a
#: volume acquired shortly after a lurch is wrong in a way no rigid transform
#: undoes. Six is a few TR at typical timings -- long enough to cover the
#: recovery, short enough that a run still offers candidates.
SEED_HISTORY = 6

#: Most of a run the frame selection may discard. Every frame is still
#: estimated and still corrected -- this bounds what the *target* is built from,
#: not what comes out. A parameter rather than a constant because the right
#: value is a property of the dataset: a cohort that barely moves should not be
#: halved to match one that does.
SELECT_MAX_DROP = 0.5

#: Flag a run for review when the selection scores agree this well about which
#: frames to reject. Motion and CDTM see different failures -- one cannot see a
#: spike or a dropped slice, the other cannot see a volume that is smeared but
#: still looks like the run -- so on a sound run they overlap little and the
#: rule spends its budget on frames that are merely each score's worst. Agreeing
#: means the badness is real and more widespread than the ceiling can remove.
#:
#: Not a reason to drop a run automatically; a reason for someone to look at it.
#: The statistic moves with how many scores there are and what they measure, so
#: it needs recalibrating whenever those change: on the current three it reads
#: 0.16 for a clean run against 0.42 for the worst one here, and two runs is
#: thin evidence for where between those the line belongs.
#:
#: Nothing but this warning depends on it. The per-frame scores are written
#: beside each reference, so the agreement of a finished run can be recomputed
#: in seconds and the threshold revised without reprocessing anything --
#: calibrate it from a cohort after the fact rather than guessing before.
GROUPWISE_AGREEMENT = 0.25

#: Voxels of dilation on the mask used for the intensity score. The brain/air
#: boundary is where motion produces the largest intensity change, so a mask
#: tight to the brain excludes exactly the voxels carrying the signal. The
#: geometric score gets the *undilated* mask -- there the mask only sets a
#: second moment, and padding it just lengthens the rotational lever arm.
GROUPWISE_DILATE = 2

#: Cap on groupwise rounds. The loop's real stopping rule is the kept set
#: repeating; this only bounds the damage if two sets alternate forever.
GROUPWISE_MAX_ROUNDS = 8


def relative_motion(volume, out=None) -> np.ndarray:
    """Frame-to-frame motion of a series, measured without correcting it.

    Runs ``-moco -relative``, which fits every volume onto the one before it
    and writes no image. The base of each pair is the *original* predecessor,
    never a corrected one, so the numbers are raw motion rather than residual
    drift after a correction -- and each pair is independent of every other.

    This is the honest way to ask how much a subject moved, because it needs no
    reference volume and so cannot be flattered or punished by the choice of
    one. Ordinary ``-moco`` fits everything onto a single base, and a base that
    is itself disturbed contaminates every row it produces.

    Args:
        volume: A 4D series.
        out: Where to write the ``.1D`` (and its ``.bin`` companion). A
            temporary is used and discarded when this is ``None``.

    Returns:
        ``(n_frames, 6)`` float64: ``(roll, pitch, yaw, dS, dL, dP)`` --
        degrees counter-clockwise about I-S, R-L and A-P, then millimetres
        toward Superior, Left and Posterior, the same convention ``-1Dfile``
        uses. Row 0 is all zeros: volume 0 has no predecessor.

    Raises:
        RuntimeError: If this niimath cannot measure relative motion, or wrote
            the wrong number of rows.
    """
    if not supports_relative():
        raise RuntimeError(
            f"{niimath_path()} has no '-moco -relative'; rebuild niimath from a "
            "revision that carries it, or choose a reference volume explicitly"
        )

    volume = Path(volume).resolve()
    with contextlib.ExitStack() as stack:
        if out is None:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            out = Path(tmp) / "relative.1D"
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        extra = ["-bin"] if supports_bin() else []
        niimath(volume, "-moco", "-relative", *extra, out)
        rows = _read_parameters(out, nib.load(str(volume)).shape[-1])
    return rows


def relative_displacement(rows, radius: float = FD_RADIUS_MM) -> np.ndarray:
    """Per-frame displacement in mm, from :func:`relative_motion` parameters.

    This is Power framewise displacement, but computed from a fit made
    *directly* between neighbouring frames rather than by differencing two fits
    against a common base. The two agree closely on quiet runs and diverge
    where it matters, since differencing inherits the error of both estimates
    and of whatever the base was doing.

    Args:
        rows: ``(n_frames, 6)`` as :func:`relative_motion` returns, or a path
            to its ``.1D``.
        radius: Sphere radius converting rotation to displacement.

    Returns:
        ``(n_frames,)`` mm. Element ``t`` is how far the head moved between
        volume ``t - 1`` and volume ``t``; element 0 is zero by construction,
        not by measurement.
    """
    rows = np.asarray(rows if not isinstance(rows, (str, Path))
                      else np.loadtxt(rows), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(f"expected (n_frames, 6) parameters, got {rows.shape}")
    return (np.abs(rows[:, 3:6]).sum(1)
            + radius * np.abs(np.radians(rows[:, :3])).sum(1))



def relative_rms(rows, image, mask=None) -> np.ndarray:
    """Per-frame RMS tissue displacement, from :func:`relative_motion` parameters.

    The same steps :func:`relative_displacement` scores, measured the other way.
    Power framewise displacement sums absolute translations and converts
    rotation with an assumed 50mm sphere; this takes the RMS distance the
    subject's own brain voxels actually travel, with the rotational lever arm
    coming from their inertia tensor rather than from a constant. It is
    therefore in the same units as :func:`within_tr_motion` and
    :func:`step_motion`, so a within-TR and a between-TR threshold set on it
    mean the same thing.

    Neither replaces the other. Power FD is comparable across studies and needs
    no mask; this is comparable across the *metrics here* and needs one. Report
    both if you are publishing a number.

    Args:
        rows: ``(n_frames, 6)`` as :func:`relative_motion` returns, or a path to
            its ``.1D``. Each row is the fit of one frame onto its predecessor,
            so the transform it describes *is* the step.
        image: A volume on the series' grid, for the affine and the geometry.
        mask: Brain mask, undilated. Without one the geometry falls back to an
            intensity threshold that tracks the field of view rather than the
            head -- see :func:`brain_geometry`.

    Returns:
        ``(n_frames,)`` mm. Element ``t`` is how far tissue moved between volume
        ``t - 1`` and volume ``t``; element 0 is zero by construction.
    """
    rows = np.asarray(rows if not isinstance(rows, (str, Path))
                      else np.loadtxt(rows), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(f"expected (n_frames, 6) parameters, got {rows.shape}")
    img = nib.load(str(image))
    centroid, moment, count = brain_geometry(image, mask=mask)
    identity = np.eye(4)
    return np.array([
        pose_distance(parameters_to_pull(row, img.affine, img.shape), identity,
                      centroid, moment, count)
        for row in rows])


def best_reference(displacement) -> int:
    """The volume that moved least from both of its neighbours.

    A registration target should be a volume the scanner caught while the head
    was still. Motion *during* a volume's acquisition blurs it, and blur in the
    target propagates into every frame fitted to it; a frame flanked by two
    small displacements was almost certainly acquired in one position.

    Scored as ``max(d[t], d[t + 1])`` -- the worse of the two neighbouring
    displacements -- with their sum breaking ties. Minimising the *worse* one
    is deliberate: a frame can sit at the end of a long still stretch and be
    followed immediately by a lurch, which a sum would forgive and a max will
    not.

    The first and last volumes are never chosen. They have one neighbour each,
    so their score is not comparable with the rest, and they are also where a
    run is least trustworthy -- the start before the subject settles, the end
    once they can feel it coming.

    Args:
        displacement: ``(n_frames,)`` from :func:`relative_displacement`.

    Returns:
        The chosen volume index.

    Note:
        This finds a volume that is *locally* still, which is not the same as
        one in the run's *typical* head position. A subject who moves once and
        then holds the new position gives that whole stretch near-zero
        displacement, and a target picked from it is sharp but off-centre.
        Where that matters, score the frames against the series instead --
        :func:`lightprep.qc.motion.cdtm` asks how unlike the run each frame is,
        which is the complementary question.
    """
    d = np.asarray(displacement, dtype=np.float64).ravel()
    if d.size < 3:
        return int(d.size // 2)
    pair = np.column_stack([d[1:-1], d[2:]])
    return 1 + int(np.lexsort((pair.sum(1), pair.max(1)))[0])


def stable_reference(volume, out=None, radius: float = FD_RADIUS_MM):
    """``(index, displacement)`` for the least-disturbed volume of a series.

    The measurement pass this runs is cheap -- a few seconds for a couple of
    hundred frames -- so there is little reason to guess a reference when the
    data can be asked.
    """
    d = relative_displacement(relative_motion(volume, out), radius=radius)
    return best_reference(d), d


def brain_geometry(image, percentile: float = 40.0, mask=None):
    """``(centroid, inertia tensor, count)`` of the brain, in world millimetres.

    The pose arithmetic below needs to know what is being moved, not just how.
    Every distance here is the distance *tissue* travels, so it is weighted by
    where the tissue is: the second moment of the brain about its centroid.

    A brain is not a sphere, and that matters. For a spherical object the
    inertia tensor is a multiple of the identity and the pose average below
    collapses to the textbook rotation average on SO(3); for a real brain,
    longer front-to-back than top-to-bottom, a given rotation costs more
    displacement about the short axis, and the average shifts accordingly.

    Args:
        image: Any volume on the series' grid -- only the geometry is used.
        percentile: Used only when ``mask`` is None. Intensity percentile,
            among non-zero voxels, above which a voxel counts as brain.

            Be aware of what this fallback actually selects. Air in an EPI is
            noisy rather than zero, so nearly every voxel passes ``> 0`` and the
            threshold keeps a fixed *fraction of the field of view* -- measured
            at 59% of the matrix on two subjects whose heads differ, giving an
            RMS radius of 71mm against 50-54mm for a real brain. It is stable
            but it is not anatomy, and it lengthens the lever arm every rotation
            is weighted by.
        mask: Boolean brain mask on ``image``'s grid, used in preference to
            ``percentile``. Pass the real thing wherever one is available. It
            should NOT be dilated: dilation is for intensity scores, where the
            brain/air boundary carries the signal, whereas here the mask only
            sets a second moment and padding it merely inflates the radius.

    Returns:
        ``(centroid (3,), second moment (3, 3), n voxels)``, all float64.
    """
    img = nib.load(str(image))
    if mask is None:
        data = np.asarray(img.dataobj, dtype=np.float64)
        if data.ndim > 3:
            data = data[..., 0]
        positive = data[data > 0]
        if positive.size == 0:
            raise ValueError(f"no positive voxels in {image}, cannot locate a brain")
        mask = data > np.percentile(positive, percentile)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != img.shape[:3]:
        raise ValueError(
            f"mask is {mask.shape} but {image} is {img.shape[:3]}"
        )
    if not mask.any():
        raise ValueError("mask selects no voxels")
    index = np.array(np.nonzero(mask), dtype=np.float64).T
    affine = np.asarray(img.affine, dtype=np.float64)
    points = index @ affine[:3, :3].T + affine[:3, 3]
    centroid = points.mean(0)
    centred = points - centroid
    return centroid, centred.T @ centred, len(points)


def pose_distance(pull_a, pull_b, centroid, moment, count) -> float:
    """RMS millimetres of tissue travel between two head poses.

    Expanding the mean squared displacement over the brain leaves only traces,
    so this costs nothing per pair once :func:`brain_geometry` has run::

        D^2 = [2 tr(S) - 2 tr(R_a S R_b^T)] / N + || p_a - p_b ||^2

    with ``S`` the inertia tensor and ``p`` the centroid's position under each
    pose. It is the honest version of framewise displacement: no 50mm rotation
    radius is assumed, the subject's own head supplies the lever arm.
    """
    a, b = np.asarray(pull_a, dtype=np.float64), np.asarray(pull_b, dtype=np.float64)
    rot = np.trace(a[:3, :3] @ moment @ b[:3, :3].T)
    gap = (a[:3, :3] @ centroid + a[:3, 3]) - (b[:3, :3] @ centroid + b[:3, 3])
    squared = (2.0 * np.trace(moment) - 2.0 * rot) / count + gap @ gap
    return float(np.sqrt(max(squared, 0.0)))


def pose_components(pull_a, pull_b, centroid, moment, count):
    """Split :func:`pose_distance` into its rotation and translation halves.

    ``D^2`` above is a sum of two independent terms, and they are not
    interchangeable for every purpose. Susceptibility distortion is the case
    that forces the distinction: the field depends on how the head sits in
    B0, so rotating the head changes the field itself, while translating it
    mostly carries the same field along. Two poses an equal number of
    millimetres apart are therefore not equally damaging to a field map -- the
    rotated one is worse, and a single RMS number hides which it was.

    Returns:
        ``(rotation mm, translation mm, angle degrees)``. The first two are
        the two terms of ``D^2``, so ``hypot`` of them is
        :func:`pose_distance`. The angle is the rotation between the poses,
        independent of the brain's shape, for comparing with a specification.
    """
    a, b = np.asarray(pull_a, dtype=np.float64), np.asarray(pull_b, dtype=np.float64)
    rot = np.trace(a[:3, :3] @ moment @ b[:3, :3].T)
    spin = np.sqrt(max((2.0 * np.trace(moment) - 2.0 * rot) / count, 0.0))
    gap = (a[:3, :3] @ centroid + a[:3, 3]) - (b[:3, :3] @ centroid + b[:3, 3])
    relative = a[:3, :3].T @ b[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
    return float(spin), float(np.linalg.norm(gap)), float(angle)


def frechet_mean_pose(pulls, centroid, moment, weights=None) -> np.ndarray:
    """The rigid pose minimising total squared tissue displacement, in closed form.

    Minimising ``sum_t w_t sum_i || A_t x_i - M x_i ||^2`` over rigid ``M``
    separates once the brain is centred, because the cross terms carry a
    ``sum_i y_i`` that is zero by construction:

    * the translation is the weighted mean of the centroid's positions;
    * ``tr(R S R^T) = tr(S)`` is constant over rotations, so what is left is
      ``max_R tr(R S (sum_t w_t R_t)^T)`` -- orthogonal Procrustes, one SVD.

    No iteration, no tangent-space linearisation, no starting guess.

    Args:
        pulls: ``(T, 4, 4)`` world-space pulls, as :func:`parameters_to_pull`
            returns them.
        centroid: Brain centroid, from :func:`brain_geometry`.
        moment: Brain inertia tensor, from :func:`brain_geometry`.
        weights: Optional per-frame weights. Frames the fit should not be
            allowed to drag the average toward get a small one.

    Returns:
        The 4x4 mean pose, in the same frame as ``pulls``.
    """
    pulls = np.asarray(pulls, dtype=np.float64)
    rot = pulls[:, :3, :3]
    position = rot @ centroid + pulls[:, :3, 3]

    if weights is None:
        weights = np.ones(len(pulls))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(pulls),):
        raise ValueError(f"expected {len(pulls)} weights, got {weights.shape}")
    if not np.all(weights >= 0) or not weights.sum() > 0:
        raise ValueError("weights must be non-negative and not all zero")
    weights = weights / weights.sum()

    u, _, vt = np.linalg.svd(moment @ np.einsum("t,tij->ij", weights, rot).T)
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    mean_rot = vt.T @ flip @ u.T

    mean = np.eye(4)
    mean[:3, :3] = mean_rot
    mean[:3, 3] = weights @ position - mean_rot @ centroid
    return mean


def centre_pulls(pulls, mean_pose) -> np.ndarray:
    """Re-express pulls so the series' mean pose is exactly the identity.

    Right-composing with the mean's inverse moves the reference frame onto the
    mean pose in one algebraic step -- there is nothing to converge. Writing
    ``G = M*^-1``, the brain points in the new frame are ``z_i = M* x_i``, so

        sum_t || (A_t G) z_i - M z_i ||^2 = sum_t || A_t x_i - (M M*) x_i ||^2

    which is minimised at ``M = I``. This is what stops a groupwise reference
    inheriting the arbitrary pose of whichever volume seeded the first round.
    """
    inverse = np.linalg.inv(np.asarray(mean_pose, dtype=np.float64))
    return np.asarray(pulls, dtype=np.float64) @ inverse



def _strip_or_threshold(image):
    """MindGrab a volume, falling back to an intensity cut if it will not run.

    The workflow has to keep going without a person watching. The fallback is
    measurably worse -- an intensity cut keeps a fixed fraction of the field of
    view rather than the brain, a 71mm RMS radius against 50-54mm for a real
    mask, which lengthens the lever arm every rotation is weighted by and
    inflated a rotation-heavy run's scores by 28% when measured. So it warns,
    and it is built here rather than left as None, which would only send the
    next caller back to the extraction that just failed.
    """
    from ..qc.motion import brain_mask
    try:
        return brain_mask(image)
    except (RuntimeError, DependencyError) as exc:              # noqa: BLE001
        warnings.warn(
            f"brain extraction failed on {Path(image).name} ({exc}); falling "
            f"back to an intensity threshold. Treat this run's motion scores "
            f"as approximate.", stacklevel=3)
        vol = np.asarray(nib.load(str(image)).dataobj, dtype=np.float64)
        if vol.ndim > 3:
            vol = vol[..., 0]
        return vol > np.percentile(vol[vol > 0], 40.0)


def _estimate_pulls(echo: Path, ref_arg: str, work: Path, n_volumes: int,
                    tag: str):
    """``(pulls, corrected series)`` from one ``-moco`` pass.

    The corrected series is niimath's own, resampled with its internal 8-tap
    Lagrange kernel rather than this module's ``-final``. That is good enough
    to score frames against each other -- which is all it is used for -- and it
    saves resampling the run a second time just to look at it.
    """
    par = work / f"{tag}.1D"
    corrected = work / f"{tag}_corrected.nii.gz"
    precise = ["-bin"] if supports_bin() else []
    niimath(echo, "-moco", "-ref", ref_arg, "-1Dfile", par, *precise, corrected)
    img = nib.load(str(echo))
    pulls = np.array([parameters_to_pull(row, img.affine, img.shape)
                      for row in _read_parameters(par, n_volumes)])
    return pulls, corrected


def quiet_reference(displacement, history: int = SEED_HISTORY) -> int:
    """Index of the volume with the quietest recent past and immediate future.

    ``displacement`` is frame-to-frame motion, so entry ``t`` is the step from
    ``t-1`` into ``t``. A volume is a good place to start from when the head
    was already still for a while before it (the spin history has recovered),
    was still during it, and had not begun moving by the next frame -- so the
    window averaged here runs from ``t - history`` through ``t + 1``.

    This asks for something different from :func:`best_reference`, which looks
    only at the two immediate neighbours and takes the worst of them. That
    finds a frame in a momentary lull; this one finds a frame in a quiet
    stretch. The distinction matters on runs that move, where lulls are common
    and quiet stretches are not.

    Args:
        displacement: Per-frame displacement, e.g. from
            :func:`relative_displacement`. Entry 0 is a placeholder -- there is
            no step into the first frame -- and is excluded from every window.
        history: Frames of recent past the window covers.

    Returns:
        The chosen index, always one with a full window available.
    """
    d = np.asarray(displacement, dtype=np.float64).ravel()
    width = history + 2
    if d.size < width + 1:
        return int(d.size // 2)
    means = np.convolve(d, np.ones(width) / width, mode="valid")
    # means[k] averages d[k : k+width], which is the window for t = k + history.
    # k starts at 1 so the placeholder d[0] never enters a window.
    return int(np.argmin(means[1:]) + 1 + history)


def _flags_at(order, k: int, n: int):
    """Each score's worst `k` frames, from a ranking sorted once."""
    return {name: np.isin(np.arange(n), o[:k]) for name, o in order.items()}


def select_frames(scores, max_drop: float = SELECT_MAX_DROP):
    """Keep the frames no score condemns, dropping as few as the scores allow.

    Every score is "bigger is worse" and each condemns its own worst ``k``
    frames. Raising ``k`` by one adds at most one frame per score, so the union
    grows by at most ``len(scores)`` -- which is what makes
    ``k = budget // len(scores)`` the largest count that is safe no matter how
    little the scores agree. From there ``k`` rises while the *union* stays
    within budget.

    That the scores overlap is the point. Views of a frame disagreeing means
    the run is fine and ``k`` stops early; views condemning the same frames mean
    ``k`` can go much further and still spend the budget on genuinely bad frames
    rather than arbitrary ones. On the current three scores that lifts ``k``
    from 0.22 to 0.30 of the run between a clean subject and the worst one.

    The union is nested in ``k``, so the largest feasible ``k`` is found by
    bisection over the counts rather than by stepping one frame at a time, and
    each score is ranked once rather than per step.

    Args:
        scores: Mapping of name -> per-frame score, all "bigger is worse" and
            all the same length.
        max_drop: Ceiling on the fraction of frames dropped.

    Returns:
        ``(keep, fraction, flags, agreement)``. ``agreement`` is the mean
        pairwise Jaccard overlap of the flags, which is the useful diagnostic:
        the rule always spends its whole budget, so *how much* it drops says
        nothing, but *whether the scores agree about what to drop* separates a
        run whose worst frames are merely its worst from one where three
        independent views condemn the same frames. Measured here: 0.04-0.08 on
        clean runs against 0.4-0.6 on a run that needs a human.

    Note:
        With mutually uninformative scores this lands on the ceiling
        immediately, discarding ``max_drop`` of even a flawless run. That is
        intended -- half a good run still makes a good target -- but it means
        the drop count is not evidence of anything. Read ``agreement`` for that.
    """
    if not scores:
        raise ValueError("need at least one score")
    if not 0.0 < max_drop < 1.0:
        raise ValueError(f"max_drop must be in (0, 1), got {max_drop}")
    lengths = {len(v) for v in scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"scores disagree in length: {lengths}")

    n = lengths.pop()
    budget = int(np.floor(max_drop * n))
    order = {name: np.argsort(np.asarray(v, dtype=np.float64))[::-1]
             for name, v in scores.items()}

    def union_at(k):
        u = np.zeros(n, dtype=bool)
        for o in order.values():
            u[o[:k]] = True
        return u

    # A score's k frames are distinct, so the union is never smaller than k:
    # nothing above `budget` can fit, and `budget // len(scores)` always does.
    lo, hi = budget // len(scores), budget
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if union_at(mid).sum() <= budget:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1

    flags = _flags_at(order, best, n)
    pairs, names = [], sorted(flags)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = (flags[a] | flags[b]).sum()
            pairs.append((flags[a] & flags[b]).sum() / union if union else 0.0)
    agreement = float(np.mean(pairs)) if pairs else 1.0
    return ~union_at(best), best / n, flags, agreement


def _substack(img, data, parity, out: Path) -> Path:
    """One slice parity as its own series, keeping the original world space."""
    scale = np.eye(4)
    scale[2, 2] = 2.0            # every second slice
    scale[2, 3] = float(parity)  # ... starting at 0 or 1
    nib.Nifti1Image(data[:, :, parity::2, :], np.asarray(img.affine) @ scale
                    ).to_filename(out)
    return out



def interleaved_slices(slice_timing, tr: float) -> bool:
    """Whether odd and even slices are acquired in different halves of the TR.

    This is the property :func:`within_tr_motion` lives on, so it is what to
    test -- not whether the order "looks interleaved". Multiband acquires
    several slices at once and Siemens orders them by parity within each band,
    so pattern-matching the acquisition order is fiddly, while the question that
    matters is simply whether splitting on parity separates the two halves in
    time.

    A sequential acquisition puts consecutive slices one slice-interval apart,
    so the parity means differ by about ``tr / n_slices`` -- negligible. An
    interleaved one separates them by roughly half a TR, which is what makes
    the odd and even stacks two views of the head a half-TR apart.

    Args:
        slice_timing: Seconds from the volume's start, one per slice. BIDS
            ``SliceTiming``.
        tr: Repetition time in seconds.

    Returns:
        True when the parity means are more than a quarter of a TR apart.
    """
    times = np.asarray(slice_timing, dtype=np.float64).ravel()
    if times.size < 4 or not tr > 0:
        return False
    separation = abs(times[1::2].mean() - times[0::2].mean())
    return bool(separation > 0.25 * tr)


def within_tr_pulls(echo, ref: int, work):
    """Motion *inside* each TR, from the interleaved slice stacks.

    An interleaved acquisition takes the odd slices in one half of the TR and
    the even slices in the other, so a volume is two half-volumes about a second
    apart. Motion-correcting each stack on its own gives two poses per TR, and
    the distance between them is motion no rigid transform can undo: the volume
    is internally smeared, not merely displaced.

    Volume-level FD cannot see this -- measured across a clean cohort the two
    correlate at rho ~ 0.08 -- because when the head moves mid-TR no single
    rigid transform describes the volume, and the estimator returns an
    unremarkable compromise.

    Each stack is only ever registered against itself, never against the other:
    the two sample planes 2mm apart, and comparing them as *images* is badly
    conditioned along z -- tried, and it reported up to 2mm of motion on a
    series with none. The comparison happens in parameter space instead, whose
    noise floor measures at 0.008mm.

    Returns the two pose traces rather than the distance between them, so a
    caller that later improves its mask can rescore without repeating the two
    ``-moco`` passes -- the registrations do not depend on the mask, only the
    conversion to millimetres does.

    Args:
        echo: The series to measure.
        ref: Reference volume index, shared by both stacks.
        work: Scratch directory.

    Returns:
        ``(odd, even)``, each a list of world-space pulls, one per volume.

    Note:
        Both stacks are referenced to the same TR, so this returns
        ``W_t - W_ref``: if the head moved inside the reference TR, every value
        carries that offset. Choose ``ref`` from a quiet stretch.
    """
    echo = Path(echo)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(echo))
    data = np.asarray(img.dataobj, dtype=np.float32)
    precise = ["-bin"] if supports_bin() else []

    pulls = {}
    for parity, name in ((1, "odd"), (0, "even")):
        src = _substack(img, data, parity, work / f"{name}.nii.gz")
        par = work / f"{name}.1D"
        niimath(src, "-moco", "-ref", str(ref), "-1Dfile", par, *precise,
                work / f"{name}_mc.nii.gz")
        sub = nib.load(str(src))
        rows = _read_parameters(par, data.shape[3])
        pulls[name] = [parameters_to_pull(r, sub.affine, sub.shape) for r in rows]

    return pulls["odd"], pulls["even"]


def within_tr_motion(pulls, centroid, moment, count) -> np.ndarray:
    """Millimetres between the two half-TR poses, from :func:`within_tr_pulls`."""
    odd, even = pulls
    return np.array([pose_distance(o, e, centroid, moment, count)
                     for o, e in zip(odd, even)])


def motion_history(steps, tr: float, span: float = SPIN_HISTORY_S,
                   within=None, min_lags: int = 0) -> np.ndarray:
    """Recent motion preceding each frame, weighted by how long ago it was.

    A movement disturbs the spin history for a while afterwards: the slices are
    excited having last been excited in a different position, and the
    magnetisation takes of order T1 to forget it. So a frame can be perfectly
    still itself and still be wrong, and that is a different defect from being
    smeared -- measured here, the two correlate at only 0.06-0.44, which is why
    they are separate criteria rather than one averaged score.

    The weight falls linearly from 1 at the immediately preceding step to 0 at
    ``span`` seconds, and the window is defined in *seconds* so it means the
    same thing at any TR. Linear is a stand-in for the exponential recovery T1
    actually gives; at 3T (T1 ~ 1.4s) a 6s span reaches about four T1.

    Movement *during* an earlier TR counts as much as movement between two of
    them -- the magnetisation does not care which side of a volume boundary it
    happened -- so ``within`` enters the window on the same footing, at the
    weight for its own lag. Entry 0 of ``steps`` is a placeholder and never
    enters; entry 0 of ``within`` is a real measurement and does.

    This does double-count a little: a movement inside TR ``t-k`` shows up in
    ``within[t-k]`` and again in the steps either side of it, so the result is
    not literally the displacement over the window. It is a score to rank on,
    and the two are near-independent in practice, so the overlap is small.

    Args:
        steps: Frame-to-frame motion; entry ``t`` is the step into ``t``, entry
            0 a placeholder that never enters a window.
        tr: Repetition time in seconds.
        span: How far back motion still counts.
        within: Within-TR motion per frame, when the acquisition interleaves.
            Omitted for a sequential one, which has none to measure.
        min_support: Fraction of the window's total weight that must be
            available, below which the frame scores infinite instead of
            whatever little evidence there is.

            The default of 0 scores an unmeasurable history as 0, which is
            right when this is a criterion for *rejecting* frames: the opening
            frames of a run have no history, and that is not a reason to throw
            them out. It is wrong when the score is minimised to *choose* a
            reference, because 0 is the best attainable value and the opening
            frames would win on the absence of evidence -- measured, that
            elected frame 0 or 1 in 14 of 59 runs here, and frame 0 is the one
            most likely to be disturbed. Pass 1.0 to require a full window.

    Returns:
        ``(n_frames,)`` weighted RMS of the preceding motion.
    """
    d = np.asarray(steps, dtype=np.float64).ravel()
    n = d.size
    w_in = None if within is None else np.asarray(within, dtype=np.float64).ravel()
    if w_in is not None and w_in.size != n:
        raise ValueError(f"within has {w_in.size} frames, steps has {n}")
    lags = np.arange(1, max(2, int(np.ceil(span / tr)) + 1))
    weights = np.maximum(0.0, 1.0 - lags * tr / span)
    out = np.zeros(n)
    for t in range(n):
        num = den = 0.0
        complete = 0
        for lag, weight in zip(lags, weights):
            if weight <= 0.0:
                continue
            # steps[0] is a placeholder -- there is no step into the first
            # frame -- while within[0] is a real measurement, so the step is
            # the binding constraint and a lag with one has both.
            if t - lag >= 1:
                num += weight * d[t - lag] ** 2
                den += weight
                complete += 1
                if w_in is not None:
                    num += weight * w_in[t - lag] ** 2
                    den += weight
        out[t] = np.sqrt(num / den) if den > 0.0 else 0.0
        if complete < min_lags:
            out[t] = np.inf
    return out


def current_motion(steps, within=None) -> np.ndarray:
    """Motion at the time of each frame: the steps either side, and within it.

    Two levels, not three peers. The steps into and out of a frame are two
    measurements of one thing -- how much the head moved between this volume
    and its neighbours -- so they are combined first, by
    :func:`neighbour_motion`. Only then does the within-TR term join, at equal
    weight, because it is one measurement of a different thing. Pooling all
    three at once would give between-TR twice the influence merely because
    there happen to be two of it:

        between[t] = sqrt( (d[t]^2 + d[t+1]^2) / 2 )
        current[t] = sqrt( (W[t]^2 + between[t]^2) / 2 )

    ``within`` is absent for a sequential acquisition, where the slice stacks
    are acquired together and there is no within-TR motion to measure; the
    score is then the between-TR term alone.
    """
    between = neighbour_motion(steps)
    if within is None:
        return between
    return combine_rms(np.asarray(within, dtype=np.float64).ravel(), between)

def step_motion(pulls, centroid, moment, count) -> np.ndarray:
    """Motion between each volume and the one before it, in the same units."""
    steps = np.zeros(len(pulls))
    for t in range(1, len(pulls)):
        steps[t] = pose_distance(pulls[t], pulls[t - 1], centroid, moment, count)
    return steps




def combine_rms(*scores) -> np.ndarray:
    """Combine RMS displacement scores in quadrature.

    ``sqrt(mean(d_i^2))`` -- the RMS over the pooled displacements, so the
    result stays an RMS. Use wherever two of these are being weighed together:
    the steps either side of a frame, or within-TR against between-TR motion.
    """
    stack = np.vstack([np.asarray(s, dtype=np.float64).ravel() for s in scores])
    return np.sqrt((stack ** 2).mean(axis=0))


def neighbour_motion(steps) -> np.ndarray:
    """Per-frame local motion: the mean of the step into a frame and out of it.

    ``steps`` is frame-to-frame motion, so entry ``t`` is the step from ``t-1``
    into ``t``. A frame is compromised by movement on either side of it -- one
    that arrives still and is left immediately afterwards is smeared just as
    surely as the reverse -- so scoring a frame by the step *into* it alone
    misses half the cases.

    Combined in quadrature, not arithmetically. Each step is already an RMS
    over voxels, so ``sqrt((a^2 + b^2)/2)`` is the RMS over both of them pooled
    and the result is still an RMS of something; an arithmetic mean is not.
    It also behaves better here: by AM-QM the quadratic mean is the larger, and
    by more the further the two steps disagree, so a frame flanked by one still
    neighbour and one lurch scores closer to the lurch. One movement
    compromises a frame whatever the other side was doing.

    Unlike sum against mean, this changes the ranking, so it is a real choice
    and not a scaling. The first and last frames have one neighbour each and
    take that step as it stands.
    """
    steps = np.asarray(steps, dtype=np.float64).ravel()
    n = steps.size
    if n < 2:
        return np.zeros(n)
    local = np.empty(n)
    local[0] = steps[1]
    local[-1] = steps[-1]
    if n > 2:
        local[1:-1] = np.sqrt((steps[1:-1] ** 2 + steps[2:] ** 2) / 2.0)
    return local



def window_rms(steps, history: int = SEED_HISTORY) -> np.ndarray:
    """RMS of the steps over ``[t - history, t + 1]``, per frame.

    The neighbourhood a *reference* wants, which is not the one a frame wants.
    A frame is judged on its immediate neighbours (:func:`neighbour_motion`) --
    was it smeared. A reference is judged on a stretch, because a movement
    disturbs the spin history for several TR afterwards, so a volume acquired
    shortly after one is wrong in a way no rigid transform undoes even if it
    sits in a momentary lull.

    RMS over the window rather than a mean, to stay in quadrature with
    everything else these scores are combined with.

    Args:
        steps: Frame-to-frame motion; entry ``t`` is the step into ``t``, and
            entry 0 is a placeholder excluded from every window.
        history: Frames of recent past the window covers.

    Returns:
        ``(n_frames,)``, infinite where a full window is not available so that
        an argmin cannot pick an edge on partial evidence.
    """
    d = np.asarray(steps, dtype=np.float64).ravel()
    width = history + 2
    out = np.full(d.size, np.inf)
    if d.size < width + 1:
        return out
    power = np.convolve(d ** 2, np.ones(width) / width, mode="valid")
    # power[k] covers d[k : k+width], the window for t = k + history; k starts
    # at 1 so the placeholder d[0] never enters one.
    for k in range(1, len(power)):
        out[k + history] = np.sqrt(power[k])
    return out


def _cdtm_values(corrected, mask):
    """Per-frame correlation distance to the run's iteratively refined mean.

    CDTM asks whether a frame *looks like* the run, which catches what pose
    arithmetic cannot see: a spike, a dropped slice, signal lost to a movement
    that happened mid-acquisition. Its own outlier rule is a threshold or a
    ratio; here a flat fraction is wanted instead, so the iterated distances
    are used as a score and cut at the quantile.
    """
    from ..qc.motion import cdtm

    return np.asarray(cdtm(corrected, mask=mask).values, dtype=np.float64)


def groupwise_reference(echo, out_dir, *, seed=None,
                       max_drop: float = SELECT_MAX_DROP,
                       dilate: int = GROUPWISE_DILATE,
                       max_rounds: int = GROUPWISE_MAX_ROUNDS,
                       tr: float | None = None,
                       span: float = SPIN_HISTORY_S,
                       interleaved: bool = True,
                       interp: str = "linear",
                       mask=None, work=None) -> Path:
    """Build a groupwise registration target: the average of the best frames.

    A single volume is one sample of the noise, and every estimate made against
    it inherits that sample. Averaging the corrected series removes almost all
    of it: measured split-half on corrected runs here, a single frame carries
    2.7-11.2% noise against its own mean signal and a half-series average
    carries 0.27-1.4%, a factor of 8-11 against a white-noise ceiling of about
    10. Averaging is running at roughly 95% of the best it could do, so the
    limiting noise in a single-frame target really is the kind that averages
    away.

    What that buys is reproducibility. Seeded from six different volumes, the
    resulting frame-to-frame estimates scattered by 40 micrometres (p95 218) on
    the worst run here when each seed was used as the target directly, and by
    0.5 micrometres (p95 3.6) when each was first turned into an average. The
    groupwise fixed point is, to that precision, unique -- the answer stops
    depending on a choice nobody has a principled way to make.

    It does not make the correction dramatically *better*: on a clean run the
    output tSNR was unchanged against ``ref="middle"`` (15.40 vs 15.41) for
    twice the runtime. Prefer it when the estimates themselves are the object
    of study, or when a run's frames disagree enough that the choice of target
    would otherwise be doing real work.

    The loop:

    0. ``-moco -relative`` over the raw series, and seed from the volume in the
       quietest stretch (:func:`quiet_reference`). Nothing is registered yet,
       so this choice is made on the data as acquired.
    1. ``-moco -ref <target>``, keeping niimath's own corrected series.
    2. Score every frame by CDTM against that series and keep the best
       ``keep_fraction``.
    3. Stop if that set is one already seen. Otherwise place the target on the
       kept frames' mean pose (:func:`frechet_mean_pose`, :func:`centre_pulls`),
       rebuild the average **from the original frames**, and go back to 1.

    Two rules keep the iteration from eating what it gains:

    * **Resample only from the original.** Each round rebuilds the average from
      ``echo`` with a composed transform, never by resampling the previous
      round's output, so the target carries exactly one interpolation however
      many rounds run.
    * **Put the target on the mean pose.** Otherwise the reference keeps the
      pose of whatever volume seeded round 0, and the series is corrected to an
      arbitrary position rather than a central one.

    Selection and averaging use the same kept set, so the target's pose and its
    content agree about which frames are trusted.

    Args:
        echo: The series to build the target from -- the same echo motion will
            be estimated on.
        out_dir: Where ``reference.nii.gz``, ``boldref_frames.npy`` (the final
            kept set) and ``boldref_cdtm.npy`` (the final distances) go.
        seed: Volume to aim round 0 at. ``None`` measures the run and asks
            :func:`quiet_reference`; an int forces one.
        max_drop: Ceiling on the fraction of frames the selection may discard.
            See :data:`SELECT_MAX_DROP` and :func:`select_frames`.
        dilate: Voxels of dilation on the mask used for the intensity score
            only. See :data:`GROUPWISE_DILATE`.
        max_rounds: Safety cap. See :data:`GROUPWISE_MAX_ROUNDS`.
        tr: Repetition time in seconds, for the spin-history window. Read from
            the header when omitted.
        span: Seconds over which past motion still counts. See
            :data:`SPIN_HISTORY_S`.
        interleaved: Whether slices are acquired interleaved, so that the odd
            and even stacks sample different halves of the TR and within-TR
            motion is measurable. False for a sequential acquisition, where the
            two stacks are acquired together and the split measures nothing:
            the motion criterion then reduces to the between-TR term. Read it
            from SliceTiming rather than assuming.
        interp: Interpolation used to build the average, one of
            :data:`INTERPOLATIONS`. ``linear`` is niimath's name for the
            trilinear kernel every other resample in this package uses, and
            matching them is worth more than the sharper reference a cubic
            average would give.

            A reference on a different kernel from the data is a reference that
            differs from it in smoothness as well as in content, and that
            confounds anything measured between the two -- a coregistration
            cost read across the pair reflects the kernels as much as the
            registration. Cubic does make a visibly sharper average, so this is
            a real cost, knowingly paid for comparability.
        mask: Brain mask for the CDTM scoring, on ``echo``'s grid. Omitted,
            CDTM strips the first round's corrected series itself and the
            result is reused for every later round. Pass one to skip that --
            useful when a mask already exists, and when the strip is the least
            reliable thing in the loop.
        work: Scratch directory. A temporary one is used and removed if omitted.

    Returns:
        Path to the boldref, on ``echo``'s voxel grid -- which is what
        ``-moco -ref`` requires of an image reference.

    Warns:
        UserWarning: If the kept set is still changing at ``max_rounds``, or if
            it settles into a cycle rather than a fixed point. The reference is
            still returned; it is the last one built.

    Note:
        The average is over time, so task-driven and physiological signal
        average out with the noise: the target ends up an essentially
        anatomical EPI. That is a second reason to prefer it, independent of
        SNR -- a single-frame target carries whatever the BOLD signal was doing
        at that instant into every fit made against it.
    """
    if not supports_ref():
        raise RuntimeError(
            "a groupwise reference needs niimath with '-moco -ref <n|img>'; "
            "this build has no -ref, so there is no way to fit onto an image"
        )
    if not 0.0 < max_drop < 1.0:
        raise ValueError(f"max_drop must be in (0, 1), got {max_drop}")
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")

    echo = Path(echo).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(echo))
    n_volumes = img.shape[-1]
    if tr is None:
        tr = float(img.header.get_zooms()[3]) if img.ndim > 3 else 0.0
    if not tr > 0:
        raise ValueError(
            "a repetition time is needed to weight the spin-history window in "
            f"seconds; pass tr= (the header gives {tr!r}, which is not usable)")

    with contextlib.ExitStack() as stack:
        if work is None:
            work = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)

        # 0a. A first reference on Power FD alone -- the one score needing no
        # mask, which is the whole reason it goes first. Same two terms the
        # frames are later judged on, just in FD units: motion at the time of
        # the frame, and motion still echoing from before it.
        steps = relative_motion(echo, work / "relative.1D")
        fd = relative_displacement(steps)
        save_trace(fd, out_dir / "relative_fd.txt")
        if seed is None:
            first = int(np.argmin(combine_rms(
                current_motion(fd),
                motion_history(fd, tr, span, min_lags=1))))
        else:
            first = int(seed)
        if not 0 <= first < n_volumes:
            raise ValueError(
                f"seed volume {first} is out of range for {n_volumes} volumes")
        first_vol = work / "first.nii.gz"
        niimath(echo, "-crop", first, 1, first_vol)

        # 0b. Strip that one volume for a mask. A single EPI frame is noisier
        # than the multi-frame mean brain_mask would normally take, but the mask
        # is only consumed as a second moment and the noise averages out of
        # that: Dice 0.96-0.995 against the multi-frame mask, RMS radius within
        # 0.4mm. On a run that moves it is the better of the two, a temporal
        # mean there being a smeared union of head positions.
        mask = _strip_or_threshold(first_vol)
        first_geom = brain_geometry(first_vol, mask=mask)

        # 0c. The same two terms again, now in the units everything downstream
        # uses. The odd/even registrations are done once here and rescored
        # later against the final mask -- they do not depend on it.
        stacks = within_tr_pulls(echo, first, work / "within_tr") if interleaved else None
        rms_steps = relative_rms(steps, first_vol, mask=mask)
        within = None if stacks is None else within_tr_motion(stacks, *first_geom)
        index = first if seed is not None else int(np.argmin(combine_rms(
            current_motion(rms_steps, within),
            motion_history(rms_steps, tr, span, within=within, min_lags=1))))

        # 0d. The reference everything is registered to, and its own mask: the
        # brain sits differently in this frame than in `first`, so the geometry
        # is re-derived rather than carried over. Both motion scores are
        # rescored against it, which costs arithmetic and no registration.
        grid = work / "seed.nii.gz"
        niimath(echo, "-crop", index, 1, grid)
        mask = _strip_or_threshold(grid)
        dilated = ndimage.binary_dilation(mask, iterations=dilate)
        geom = brain_geometry(grid, mask=mask)
        rms_steps = relative_rms(steps, grid, mask=mask)
        within = None if stacks is None else within_tr_motion(stacks, *geom)
        motion = {"current": current_motion(rms_steps, within),
                  "history": motion_history(rms_steps, tr, span, within=within)}

        frames = list(split_frames(echo, work / "frames", prefix="vol"))
        target, boldref = str(index), out_dir / "reference.nii.gz"
        seen, keep, values = [], None, None

        for round_index in range(max_rounds):
            pulls, corrected = _estimate_pulls(echo, target, work, n_volumes,
                                               f"groupwise{round_index}")
            # CDTM is the one score read off the corrected series, so it is
            # the only one recomputed per round -- and it is not read once but
            # iterated, because the reference it scores against is built from
            # the very frames being chosen.
            #
            # The frames feeding that reference are the *jointly* selected
            # ones, not CDTM's own survivors. A frame acquired while the head
            # moved is smeared and mispositioned however much it still
            # resembles the run, and averaging it into the yardstick blurs the
            # yardstick -- which then correlates acceptably with everything and
            # flattens the very discrimination CDTM exists to provide.
            #
            # It starts from a motion-only selection: within-TR and between-TR
            # are known before any CDTM exists, so the first reference is
            # already clear of the worst frames.
            from ..qc.motion import correlation_distance, masked_series
            series = masked_series(corrected, dilated)
            keep = select_frames(motion, max_drop=max_drop)[0]
            values = fraction = flags = agreement = None
            for _ in range(GROUPWISE_MAX_ROUNDS):
                values = correlation_distance(series, keep)
                scores = dict(motion, cdtm=np.nan_to_num(values, nan=np.inf))
                chosen, fraction, flags, agreement = select_frames(
                    scores, max_drop=max_drop)
                if np.array_equal(chosen, keep):
                    break
                keep = chosen
            del series

            if agreement > GROUPWISE_AGREEMENT:
                warnings.warn(
                    f"{Path(echo).name}: the selection scores agree on "
                    f"{100 * agreement:.0f}% of the frames they "
                    f"reject (typical is under "
                    f"{100 * GROUPWISE_AGREEMENT:.0f}%). The selection still "
                    f"kept {100 * keep.mean():.0f}% because that is the "
                    f"ceiling, so the reference is being built partly from bad "
                    f"frames -- this run wants a look.",
                    stacklevel=2)

            # The fixed point is the kept set repeating, which is the whole
            # stopping rule. Comparing against every earlier set, not just the
            # last, also catches a two-cycle instead of spinning until the cap.
            packed = np.packbits(keep).tobytes()
            if packed in seen:
                if packed != (seen[-1] if seen else None):
                    warnings.warn(
                        f"frame selection cycled rather than converging after "
                        f"{round_index + 1} rounds; using the last average",
                        stacklevel=2,
                    )
                break
            seen.append(packed)

            pulls = centre_pulls(pulls, frechet_mean_pose(pulls, *geom[:2],
                                                          weights=keep.astype(float)))
            _write_average(frames, pulls, keep, grid, boldref, work, interp)
            target = str(boldref)
        else:
            warnings.warn(
                f"frame selection had not converged after {max_rounds} rounds; "
                f"using the last average",
                stacklevel=2,
            )

        # The three scores the selection was made on, and the CDTM values as
        # they stood at the end -- kept raw, with NaN where a frame had no
        # variance, because these are for a person to look at rather than for
        # the rule to re-read.
        np.save(out_dir / "boldref_frames.npy", keep)
        np.save(out_dir / "boldref_cdtm.npy", np.asarray(values))
        for name, score in motion.items():
            np.save(out_dir / f"boldref_{name}.npy", np.asarray(score))
        # The within-TR term behind `current`, so a reader can see whether a
        # frame was condemned for being smeared or merely for moving. Absent
        # for a sequential acquisition, which has no within-TR term.
        if within is not None:
            np.save(out_dir / "boldref_within_tr.npy", np.asarray(within))
        np.save(out_dir / "boldref_between_tr.npy", np.asarray(rms_steps))

    return boldref


def _write_average(frames, pulls, keep, grid: Path, out: Path, work: Path,
                   interp: str) -> Path:
    """Resample the original frames onto ``grid`` and average the kept ones.

    Each frame is scaled to a common brain mean first. A run drifts by a few
    percent over its length, and without this the average is quietly weighted
    toward whichever end was brighter.
    """
    grid_img = nib.load(str(grid))
    reference = np.asarray(grid_img.dataobj, dtype=np.float64)
    if reference.ndim > 3:
        reference = reference[..., 0]
    positive = reference[reference > 0]
    mask = reference > np.percentile(positive, 40.0)

    total = np.zeros(reference.shape, dtype=np.float64)
    scales, used = [], 0
    for t, (frame, pull) in enumerate(zip(frames, pulls)):
        if not keep[t]:
            continue
        write_savemat(pull, work / f"tpl{t:04d}.json", fixed=grid, moving=frame)
        dst = work / f"tpl{t:04d}.nii.gz"
        niimath(frame, "-allineate", grid, "-applymat", work / f"tpl{t:04d}.json",
                "-final", interp, dst)
        data = np.asarray(nib.load(str(dst)).dataobj, dtype=np.float64)
        if data.ndim > 3:
            data = data[..., 0]
        scale = data[mask].mean()
        if not scale > 0:
            continue
        total += data / scale
        scales.append(scale)
        used += 1

    if not used:
        raise RuntimeError("every frame was excluded from the reference average")
    average = total / used * float(np.mean(scales))
    nib.save(nib.Nifti1Image(average.astype(np.float32), grid_img.affine,
                             grid_img.header), out)
    return out


def _reference_image(ref, n_volumes: int, echo: Path, out_dir: Path):
    """``(reference image, the -ref argument for it)``.

    niimath reads an all-digit argument as a volume number, so a file whose
    name is digits must be given as ``./7``; paths are passed through
    unchanged otherwise.
    """
    target = out_dir / "reference.nii.gz"
    if isinstance(ref, (str, Path)) and str(ref) not in ("middle", "mean"):
        ref_path = Path(ref).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(f"reference image not found: {ref_path}")
        arg = str(ref_path)
        return ref_path, (f"./{arg}" if Path(arg).name.isdigit() else arg)

    if ref == "mean":
        niimath(echo, "-Tmean", target)
        return target, str(target)

    index = n_volumes // 2 if ref == "middle" else int(ref)
    if not 0 <= index < n_volumes:
        raise ValueError(
            f"reference volume {index} is out of range for {n_volumes} volumes"
        )
    niimath(echo, "-crop", index, 1, target)
    return target, str(index)


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
    tr: float | None = None,
    interleaved: bool | None = None,
    groupwise_mask=None,
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
        ref: Registration target. ``"middle"`` (default) uses the middle
            volume, ``"stable"`` the volume that moved least from both of its
            neighbours (measured first by :func:`stable_reference`, a few
            seconds), ``"groupwise"`` a groupwise average of the corrected
            series built by :func:`groupwise_reference` (several times the cost,
            and the only option whose result does not depend on picking a
            volume), ``"mean"`` the mean volume -- which is *not* the same
            thing, being an average of the series as acquired, motion and all
            -- an int a specific volume index, or a path to an existing image. Where niimath
            supports ``-ref`` the target is handed to the estimator; otherwise
            it is reached by composition afterwards, which puts the output in
            the same frame but leaves the fit itself against volume 0.
            ``"stable"`` also writes ``relative_fd.txt`` beside the outputs:
            the raw frame-to-frame displacement, which is a motion trace of the
            *input* and so is not affected by anything this function then does
            to it.
        ref_echo: Index into ``echoes`` to estimate motion on. Defaults to the
            first echo; pass ``1`` for the mid-TE echo that afni_proc.py favours.
        interp: Interpolation used when applying the transforms, one of
            :data:`INTERPOLATIONS`. Note that this is *not* the kernel ``-moco``
            uses internally (an 8-tap Lagrange): its own corrected output is
            discarded, so that every echo is resampled once, the same way, by
            the same call.
        tr: Repetition time in seconds, for the spin-history window of
            ``ref="groupwise"``. Read from the header when omitted, which is
            worth overriding from the sidecar -- a NIfTI pixdim[4] is not
            always the TR.
        interleaved: Whether slices are acquired in two interleaved halves, so
            within-TR motion is measurable. Pass
            :func:`interleaved_slices(SliceTiming, tr)` rather than assuming;
            omitted, ``groupwise_reference`` assumes True.
        groupwise_mask: Only for ``ref="groupwise"``: a brain mask to score
            frames in, passed through to :func:`groupwise_reference`. Omitted,
            it is derived from the data.
        keep_workdir: Keep the per-frame volumes, the ``.1D`` file and the
            reconstructed JSON transforms.

    Returns:
        An :class:`~lightprep.hmc.base.HMCResult`. ``transforms`` are FLIRT
        matrices, as the other methods' are, so this is a drop-in for them.
        ``parameters`` is a six-column ``motion.npy`` in MCFLIRT's convention --
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
    #
    # Where niimath supports it, the target is handed to the estimator so the
    # fit is made against it. The older route -- estimate onto volume 0, then
    # compose with the reference's own transform -- gives the same coordinate
    # frame but a different registration: every frame was still fitted to
    # volume 0, so a corrupted volume 0 degrades every estimate and no amount
    # of re-referencing recovers it.
    img = nib.load(str(paths[ref_echo]))
    par_1d = work / "moco.1D"
    native_ref = supports_ref()

    # 0. Ask the data which volume to aim at, before anything is registered.
    if isinstance(ref, str) and ref == REF_STABLE:
        ref, relative = stable_reference(paths[ref_echo], work / "relative.1D")
        save_trace(relative, out_dir / "relative_fd.txt")
    elif isinstance(ref, str) and ref == REF_GROUPWISE:
        # Build the target out of the data first. This estimates the series a
        # couple of extra times; the pass below then re-estimates against the
        # finished boldref, so what is returned was fitted to it and not to
        # some intermediate.
        extra = {} if interleaved is None else {"interleaved": interleaved}
        ref = groupwise_reference(paths[ref_echo], out_dir, work=work / "groupwise",
                                  mask=groupwise_mask, tr=tr, **extra)

    # -bin, where it exists, is what keeps the estimate in float64: the -1Dfile
    # text is AFNI's frozen %8.4f layout, so without it every transform below
    # is rebuilt from parameters rounded to 1e-4 deg and 1e-4 mm.
    precise = ["-bin"] if supports_bin() else []

    if native_ref:
        reference, ref_arg = _reference_image(ref, n_volumes, paths[ref_echo],
                                              out_dir)
        niimath(paths[ref_echo], "-moco", "-ref", ref_arg,
                "-1Dfile", par_1d, *precise, work / "moco_corrected.nii.gz")
    else:
        niimath(paths[ref_echo], "-moco", "-1Dfile", par_1d, *precise,
                work / "moco_corrected.nii.gz")

    rows = _read_parameters(par_1d, n_volumes)
    pulls = [parameters_to_pull(row, img.affine, img.shape) for row in rows]

    if not native_ref:
        # 2. Re-reference algebraically: same frame, different origin.
        reference, ref_pull = _reference_pull(ref, pulls, paths[ref_echo],
                                              out_dir, work)
        pulls = [pull @ np.linalg.inv(ref_pull) for pull in pulls]

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

    par = save_trace(parameters, out_dir / "motion.par")

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
