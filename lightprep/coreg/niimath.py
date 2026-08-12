"""Coregistration to the subject's anatomy with niimath.

``-allineate`` (niimath's port of AFNI's ``3dAllineate``) registers the run's
reference volume to the subject's own T1, using the local Pearson correlation
cost that AFNI built for exactly this pairing. LPC does not assume EPI and T1w
look alike -- it correlates them *locally* and rewards the anticorrelation that
inverted EPI/T1 contrast produces -- which is what makes a plain intensity
registration workable across these two modalities at all.

The point of this method is that it needs no FreeSurfer binaries and no FSL:
only niimath and, if you are registering to a recon, the recon's
``orig.mgz`` on disk. :func:`~lightprep.coreg.freesurfer.bbregister` remains the
more accurate choice where FreeSurfer is available -- it optimises against the
reconstructed white surface rather than against voxel intensities, so it uses
information this method cannot see.

The two do not disagree by much. On the pilot session's two multi-echo runs
they land a median 0.37 and 0.45mm apart across the brain (p95 0.67mm, max
0.85mm) -- a fraction of the 2.8mm EPI voxel -- and sampling the cortical
ribbon through one registration or the other gives per-vertex values correlated
at r=0.997. Which is not the same as saying they are interchangeable: the
disagreement is systematic, not noise, so do not mix the two within a study.

The result carries both an LTA and a FLIRT matrix, as ``bbregister``'s does, so
surface sampling and FSL resampling read it without knowing which method ran.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .._niimath import (RIGID_COSTS, WARPS, niimath, read_savemat,
                        world_to_fsl, write_fsl_matrix)
from .base import CoregResult

CONTRASTS = ("bold", "t2", "t1", "dti")

#: Registration costs. ``lpc`` is AFNI's local Pearson correlation, built for
#: EPI-to-T1; ``lpa`` is its absolute-value sibling, for same-contrast pairs.
COSTS = RIGID_COSTS

#: Default cost per contrast: the EPI-like contrasts get the cross-modality
#: cost, a T1-like moving image the same-contrast one.
CONTRAST_COSTS = {"bold": "lpc", "t2": "lpc", "dti": "lpc", "t1": "lpa"}

#: Rough guidance for the final LPC cost. It is a (negated) correlation, so it
#: runs from -1 (perfect) upwards, and the scale has nothing to do with
#: bbregister's boundary-based cost -- do not compare the two numbers.
COST_GOOD = -0.15
COST_SUSPECT = -0.05

_COST_RE = re.compile(r"Fine cost\s*=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def _final_cost(proc) -> float | None:
    """The final cost niimath reports, from its progress output."""
    matches = _COST_RE.findall((proc.stdout or "") + (proc.stderr or ""))
    return float(matches[-1]) if matches else None


def _as_nifti(path: Path, out_dir: Path) -> Path:
    """Give niimath a NIfTI, converting FreeSurfer's .mgz if need be.

    The affine is carried across untouched, so the transform this produces is
    defined against the original volume's world space and the LTA written from
    it lines up with the surfaces.
    """
    path = Path(path)
    if path.name.endswith((".nii", ".nii.gz")):
        return path
    img = nib.load(str(path))
    dst = out_dir / (path.stem + ".nii.gz")
    nib.Nifti1Image(
        np.asanyarray(img.dataobj, dtype=np.float32), img.affine
    ).to_filename(str(dst))
    return dst


def _anat_target(subject: str, subjects_dir) -> Path:
    """The volume a FreeSurfer recon's surfaces are defined against."""
    subjects_dir = Path(subjects_dir).resolve()
    orig = subjects_dir / subject / "mri" / "orig.mgz"
    if not orig.exists():
        raise ValueError(
            f"no FreeSurfer recon for {subject!r} in {subjects_dir} (expected "
            f"{orig}). Pass target= to register to a plain anatomical instead."
        )
    return orig


def allineate(
    moving,
    subject: str,
    out_dir,
    *,
    subjects_dir=None,
    target=None,
    contrast: str = "bold",
    cost: str | None = None,
    dof: int = 6,
    source_automask: bool = True,
    freesurfer_home=None,
    save_resampled: bool = True,
) -> CoregResult:
    """Register a functional reference to the subject's anatomy.

    Args:
        moving: The volume to register -- the run's reference/mean image, not
            the timeseries. Should already be distortion-corrected: a rigid fit
            cannot absorb the residual stretch of an uncorrected EPI, and will
            trade it off against position instead, landing slightly wrong
            everywhere.
        subject: FreeSurfer subject name, as found in ``subjects_dir``. Ignored
            when ``target`` is given.
        out_dir: Where to write the registration.
        subjects_dir: The SUBJECTS_DIR holding ``subject``. The registration
            target is that subject's ``mri/orig.mgz`` -- the volume its
            surfaces are defined against, so the LTA written here is directly
            usable by :mod:`lightprep.surface`. Required unless ``target`` is given.
        target: Register to this volume instead of a recon. Use it when there is
            no FreeSurfer output, e.g. a raw T1w. Note that a transform to a raw
            T1w will *not* line up with FreeSurfer surfaces.
        contrast: Contrast of ``moving``, one of :data:`CONTRASTS`. This only
            picks the default ``cost`` -- see :data:`CONTRAST_COSTS`.
        cost: Registration cost, one of :data:`COSTS`. Defaults by ``contrast``.
        dof: Degrees of freedom: 6 (rigid), 9, or 12. 6 is right for
            same-subject registration; more only lets the fit absorb error it
            shouldn't.
        source_automask: Mask the moving image to its own brain before matching.
            AFNI recommends this with the LPC costs, and it is what keeps
            out-of-brain signal from steering the fit.
        freesurfer_home: Accepted so that this method can stand in for
            :func:`~lightprep.coreg.freesurfer.bbregister` unchanged. It is not
            used: no FreeSurfer binary is run, only ``orig.mgz`` is read.
        save_resampled: Also write ``moving`` resampled into anatomical space,
            for eyeballing the result.

    Returns:
        A :class:`~lightprep.coreg.base.CoregResult`. ``registration`` is an LTA,
        as ``bbregister`` writes, and ``fsl_matrix`` the equivalent FLIRT
        matrix. ``cost`` is niimath's final cost -- on a different scale from
        bbregister's, so read it against :data:`COST_GOOD` here and not against
        the boundary-based thresholds.

    Raises:
        ValueError: On an invalid argument, or a missing subject.
        DependencyError: If no niimath binary can be found.
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, got {contrast!r}")
    cost = CONTRAST_COSTS[contrast] if cost is None else cost
    if cost not in COSTS:
        raise ValueError(f"cost must be one of {COSTS}, got {cost!r}")
    warp = {v: k for k, v in WARPS.items()}.get(dof)
    if warp is None:
        raise ValueError(f"dof must be one of {sorted(WARPS.values())}, got {dof}")

    moving = Path(moving).resolve()
    if not moving.exists():
        raise FileNotFoundError(f"moving image not found: {moving}")

    if target is not None:
        anat = Path(target).resolve()
        if not anat.exists():
            raise FileNotFoundError(f"target image not found: {anat}")
        target_name = anat.name
    else:
        if subjects_dir is None:
            raise ValueError("pass either subjects_dir= (with subject) or target=")
        anat = _anat_target(subject, subjects_dir)
        target_name = f"{subject} (freesurfer)"

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    fixed = _as_nifti(anat, work)
    savemat = work / "boldref2anat.json"
    resampled = (out_dir / "boldref_in_anat.nii.gz" if save_resampled
                 else work / "resampled.nii.gz")

    args = [moving, "-allineate", fixed, "-cost", cost, "-warp", warp]
    if source_automask:
        args.append("-source_automask")
    args += ["-savemat", savemat, resampled]
    proc = niimath(*args)

    # niimath's world-space pull transform is already the direction nitransforms
    # and the surface code want: anatomy -> functional.
    pull = read_savemat(savemat)

    from nitransforms.linear import Affine

    lta = out_dir / "boldref2anat.lta"
    Affine(pull, reference=str(fixed)).to_filename(
        str(lta), fmt="fs", moving=str(moving)
    )
    fslmat = write_fsl_matrix(
        world_to_fsl(pull, reference=fixed, moving=moving),
        out_dir / "boldref2anat.mat",
    )

    shutil.rmtree(work, ignore_errors=True)

    return CoregResult(
        registration=lta,
        fsl_matrix=fslmat,
        moving=moving,
        target=target_name,
        cost=_final_cost(proc),
        method="niimath",
    )
