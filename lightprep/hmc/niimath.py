"""Head motion correction with niimath, via ``-allineate``.

``-allineate`` is niimath's port of AFNI's ``3dAllineate``. Constrained to
``shift_rotate`` it is a 6-DOF rigid registration, which is the model head
motion warrants, and it runs entirely inside niimath -- so this
method needs nothing else installed, where
:func:`~lightprep.hmc.fsl.mcflirt` needs a working FSL install.

niimath registers volumes, not timeseries: handed a 4D image, ``-allineate``
takes its first frame and warns. So each frame is cut out with ``-crop``,
registered on its own, and the corrected frames are stacked back up at the end.
Estimation dominates the cost -- see :func:`allineate` on choosing ``cost``.

The transforms are written as FLIRT matrices, not as niimath's own JSON, so that
everything downstream (``convertwarp`` in :mod:`lightprep.resample.fsl`,
nitransforms in :mod:`lightprep.surface.concat`) reads this method's output exactly
as it reads MCFLIRT's. The two are interchangeable, and measurably agree. Run
against MCFLIRT on the pilot's 138-frame three-echo run, estimating on the same
echo against the same reference:

===========================  ==========================================
motion trace vs ``.par``     max 9.4e-04 rad / 0.103mm, rms 2.9e-04 / 0.036mm
transforms                   max 0.129mm displacement across the brain
                             (median frame 0.070mm)
corrected timeseries         r = 0.9939, 0.9948, 0.9956 by echo
===========================  ==========================================

Worth keeping in proportion: the head barely moved in this run. MCFLIRT's own
trace spans 0.345mm at most and never sits more than 0.201mm from the reference,
so a 0.103mm disagreement is a third of the displacement being corrected, not a
thousandth of it. What this shows is that the two agree on a still head; how they
compare on a moving one is not measured here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .._niimath import (FINAL_INTERPS, RIGID_COSTS, WARPS, merge_frames,
                        motion_parameters, niimath, read_savemat, split_frames,
                        world_to_fsl, write_fsl_matrix)
from .._utils import strip_ext
from .base import HMCResult
from .fsl import _check_echoes

#: ``-final`` interpolations, for applying the estimated transforms.
INTERPOLATIONS = tuple(FINAL_INTERPS)

#: Costs that can be constrained to 6 DOF. ``ls`` (least squares) is the
#: default: frames of one run are the same tissue in the same sequence, so
#: matching intensities directly is both the right model and the cheapest.
COSTS = RIGID_COSTS


def _make_reference(echo: Path, ref, n_volumes: int, out_dir: Path) -> Path:
    """Materialise the registration target as a file on disk.

    Writing it out explicitly means the very same image is the fixed image for
    every frame, and is reused when the transforms are replayed onto the other
    echoes.
    """
    if isinstance(ref, (str, Path)) and str(ref) not in ("middle", "mean"):
        ref = Path(ref).resolve()
        if not ref.exists():
            raise FileNotFoundError(f"reference image not found: {ref}")
        return ref

    target = out_dir / "reference.nii.gz"
    if ref == "mean":
        niimath(echo, "-Tmean", target)
        return target

    index = n_volumes // 2 if ref == "middle" else int(ref)
    if not 0 <= index < n_volumes:
        raise ValueError(
            f"reference volume {index} is out of range for {n_volumes} volumes"
        )
    niimath(echo, "-crop", index, 1, target)
    return target


def allineate(
    echoes,
    out_dir,
    *,
    ref="middle",
    ref_echo: int = 0,
    cost: str = "ls",
    dof: int = 6,
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
            a path to use an existing image.
        ref_echo: Index into ``echoes`` to estimate motion on. Defaults to the
            first echo; pass ``1`` for the mid-TE echo that afni_proc.py favours.
        cost: Registration cost, one of :data:`COSTS`. ``ls`` is the default and
            the fastest; ``hel`` (Hellinger) is the AFNI-style information
            measure and takes roughly 2.5x as long for no benefit within a
            single run. Note that niimath's ``fast``/``fastcr`` costs are *not*
            offered here: they run a fixed 12-DOF schedule, and letting scale
            and shear float is not head motion.
        dof: Degrees of freedom; 6 (rigid body) is what head motion warrants and
            anything else here should be justified.
        interp: Interpolation used when applying the transforms, one of
            :data:`INTERPOLATIONS`. ``linear`` is the default, matching the
            trilinear default of the MCFLIRT method for the same reason: a
            sharper kernel rings at the tissue edges.
        keep_workdir: Keep the per-frame volumes and niimath's own JSON
            transforms.

    Returns:
        An :class:`~lightprep.hmc.base.HMCResult`. ``transforms`` are FLIRT
        matrices, as MCFLIRT's are, so the two methods are interchangeable
        downstream. ``parameters`` is a six-column ``motion.par``: rotations
        (radians) then translations (mm), in MCFLIRT's convention -- see
        :func:`lightprep._niimath.motion_parameters`.

    Raises:
        ValueError: If the echoes disagree in shape, or an argument is invalid.
        DependencyError: If no niimath binary can be found.

    Note:
        Estimation is one registration per frame and is the dominant cost by a
        wide margin; replaying the transforms afterwards is cheap, on the order
        of 0.1s a frame. A single 6-DOF registration of a 76x76x46 volume runs
        in a few seconds on 8 threads and roughly 9s on one, so a 138-frame run
        is minutes of estimation where MCFLIRT spends a fraction of a second a
        frame -- one to two orders of magnitude, depending on threads. Treat
        those figures as ballpark: they were measured on a machine that was not
        idle, and repeated identical registrations varied by a factor of two.

        Set ``OMP_NUM_THREADS`` to control threading. niimath's own ``-p`` flag
        is *not* accepted in the position its help implies -- it is parsed as an
        input filename -- so the environment variable is the way. Reach for this
        method when avoiding an FSL dependency is worth the wall clock, not when
        it isn't.

        Phase images must not be passed here. Interpolating wrapped phase across
        the +/-pi boundary produces nonsense. Estimate on magnitude, then apply
        these transforms to the real and imaginary parts instead.
    """
    if cost not in COSTS:
        raise ValueError(f"cost must be one of {COSTS}, got {cost!r}")
    warp = {v: k for k, v in WARPS.items()}.get(dof)
    if warp is None:
        raise ValueError(f"dof must be one of {sorted(WARPS.values())}, got {dof}")
    if interp not in INTERPOLATIONS:
        raise ValueError(f"interp must be one of {INTERPOLATIONS}, got {interp!r}")

    paths, n_volumes = _check_echoes(echoes)
    if not 0 <= ref_echo < len(paths):
        raise ValueError(
            f"ref_echo {ref_echo} is out of range for {len(paths)} echo(s)"
        )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [out_dir / f"{strip_ext(p)}.nii.gz" for p in paths]
    clobbered = [str(p) for p, o in zip(paths, outputs) if p == o]
    if clobbered:
        raise ValueError(
            "out_dir would overwrite the input echo(es): " + ", ".join(clobbered)
        )

    reference = _make_reference(paths[ref_echo], ref, n_volumes, out_dir)

    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    transform_dir = out_dir / "transforms"
    transform_dir.mkdir(exist_ok=True)

    # 1. Estimate, one frame at a time. -allineate would otherwise silently
    #    register only the first frame of a 4D input.
    est_dir = work / "estimate"
    frames = split_frames(paths[ref_echo], est_dir, prefix="est")
    transforms, parameters = [], []
    for t, frame in enumerate(frames):
        savemat = est_dir / f"xfm{t:04d}.json"
        # The resampled volume is a byproduct: it is discarded so that every
        # echo, this one included, is resampled once by the same call below.
        niimath(frame, "-allineate", reference, "-cost", cost, "-warp", warp,
                "-savemat", savemat, est_dir / f"reg{t:04d}.nii.gz")
        pull = read_savemat(savemat)
        # `moving` is the echo, not the reference: a caller may hand in a
        # target on a grid of its own, and FSL's frame is per-image.
        transforms.append(
            write_fsl_matrix(world_to_fsl(pull, reference, moving=paths[ref_echo]),
                             transform_dir / f"MAT_{t:04d}")
        )
        parameters.append(motion_parameters(pull, reference, moving=paths[ref_echo]))

    # 2. Replay the one set of transforms onto every echo.
    for src, dst in zip(paths, outputs):
        apply_dir = work / f"apply_{strip_ext(src)}"
        corrected = []
        for t, frame in enumerate(split_frames(src, apply_dir, prefix="vol")):
            out = apply_dir / f"corr{t:04d}.nii.gz"
            niimath(frame, "-allineate", reference,
                    "-applymat", est_dir / f"xfm{t:04d}.json",
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
        method="niimath",
    )
