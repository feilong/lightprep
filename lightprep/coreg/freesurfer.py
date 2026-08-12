"""Coregistration to a FreeSurfer recon, via boundary-based registration.

BBR aligns an EPI to the subject's own anatomy by maximising the intensity
contrast across the white-matter surface that FreeSurfer already reconstructed,
rather than by matching image intensities directly. That is what makes it robust
across modalities: it never has to assume EPI and T1w look alike, only that the
grey/white boundary is in the same place.

Register a *distortion-corrected* reference. BBR is 6-DOF rigid, so it cannot
absorb the residual stretch of an uncorrected EPI -- it will instead trade that
stretch off against position and land slightly wrong everywhere.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .._utils import DependencyError, run
from .base import CoregResult

CONTRASTS = ("bold", "t2", "t1", "dti")
INITS = ("coreg", "header", "fsl")

#: bbregister's own rule of thumb for the final boundary-based cost.
COST_GOOD = 0.5
COST_SUSPECT = 0.8


def _freesurfer_env(freesurfer_home, subjects_dir) -> dict:
    """Build the environment FreeSurfer tools need."""
    home = Path(freesurfer_home or os.environ.get("FREESURFER_HOME", "")).resolve()
    if not (home / "bin").is_dir():
        raise DependencyError(
            "FreeSurfer not found. Set FREESURFER_HOME or pass freesurfer_home= "
            f"(looked for a bin/ directory under {home})"
        )
    return {
        "FREESURFER_HOME": str(home),
        "SUBJECTS_DIR": str(Path(subjects_dir).resolve()),
        "PATH": f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def _read_cost(reg: Path) -> float | None:
    """bbregister writes the final cost alongside the registration."""
    mincost = reg.with_name(reg.name + ".mincost")
    if not mincost.exists():
        return None
    first = mincost.read_text().split()
    return float(first[0]) if first else None


def bbregister(
    moving,
    subject: str,
    out_dir,
    *,
    subjects_dir,
    contrast: str = "bold",
    dof: int = 6,
    init: str = "coreg",
    freesurfer_home=None,
    save_resampled: bool = True,
) -> CoregResult:
    """Register a functional reference to a subject's FreeSurfer recon.

    Args:
        moving: The volume to register -- the run's reference/mean image, not
            the timeseries. Should already be distortion-corrected.
        subject: FreeSurfer subject name, as found in ``subjects_dir``.
        out_dir: Where to write the registration.
        subjects_dir: The FreeSurfer SUBJECTS_DIR holding ``subject``.
        contrast: Contrast of ``moving``, one of :data:`CONTRASTS`. ``bold``
            (equivalently ``t2``) means grey matter brighter than white, which
            is what an EPI looks like.
        dof: Degrees of freedom. 6 (rigid) is right for same-subject
            registration; more only lets the fit absorb error it shouldn't.
        init: How to initialise, one of :data:`INITS`.
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        save_resampled: Also write ``moving`` resampled into anatomical space,
            for eyeballing the result.

    Returns:
        A :class:`~lightprep.coreg.base.CoregResult`. ``cost`` is the final
        boundary-based cost: below ~0.5 is good, above ~0.8 deserves a look.

    Raises:
        ValueError: On an invalid argument or a missing subject.
        DependencyError: If FreeSurfer is not available.
    """
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, got {contrast!r}")
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")

    moving = Path(moving).resolve()
    if not moving.exists():
        raise FileNotFoundError(f"moving image not found: {moving}")

    subjects_dir = Path(subjects_dir).resolve()
    subject_dir = subjects_dir / subject
    if not (subject_dir / "surf" / "lh.white").exists():
        raise ValueError(
            f"no FreeSurfer recon for {subject!r} in {subjects_dir} "
            "(expected surf/lh.white -- BBR needs the reconstructed surfaces)"
        )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _freesurfer_env(freesurfer_home, subjects_dir)

    reg = out_dir / "boldref2anat.dat"
    lta = out_dir / "boldref2anat.lta"
    fslmat = out_dir / "boldref2anat.mat"
    cmd = [
        "bbregister",
        "--s", subject,
        "--mov", moving,
        "--reg", reg,
        "--lta", lta,
        "--fslmat", fslmat,
        f"--{contrast}",
        f"--{dof}",
        f"--init-{init}",
    ]
    if save_resampled:
        cmd += ["--o", out_dir / "boldref_in_anat.nii.gz"]
    run(cmd, extra_env=env)

    return CoregResult(
        registration=lta,
        fsl_matrix=fslmat,
        moving=moving,
        target=f"{subject} (freesurfer)",
        cost=_read_cost(reg),
        method="bbregister",
    )
