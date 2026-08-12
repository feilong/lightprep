"""Fitting the GLM per vertex and writing the contrast maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class GLMResult:
    """Outcome of a per-vertex GLM.

    Attributes:
        effect: A - B effect size (the task_A coefficient), per vertex.
        tstat: t-statistic of that effect.
        zstat: z-statistic of that effect.
        design: The design matrix that was fitted.
        column_names: Names of the design columns.
        n_vertices: Vertices fitted.
        n_frames: Timepoints.
        dof: Model residual degrees of freedom.
    """

    effect: Path
    tstat: Path
    zstat: Path
    design: Path
    column_names: tuple
    n_vertices: int
    n_frames: int
    dof: float


def _read_timeseries(path):
    return np.stack([np.asarray(d.data, dtype=np.float64)
                     for d in nib.load(str(path)).darrays], axis=1)   # (V, T)


def _write_shape(values, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
        np.asarray(values, dtype=np.float32), intent="NIFTI_INTENT_SHAPE",
        datatype="NIFTI_TYPE_FLOAT32", encoding="GIFTI_ENCODING_B64GZ")]).to_filename(str(path))
    return path


def fit(surface_timeseries, design, column_names, out_prefix, *,
        contrast="task_A", noise_model="ols"):
    """Fit ``design`` to a surface timeseries and write the contrast maps.

    Args:
        surface_timeseries: per-vertex ``.func.gii`` (vertices x frames).
        design: ``(n_frames, n_regressors)`` design matrix, task column first.
        column_names: names of the design columns.
        out_prefix: path stem for the output maps.
        contrast: which column to test (default the task regressor).
        noise_model: passed to nilearn's ``run_glm``. Ordinary least squares by
            default. ``ar1`` would prewhiten the residual serial correlation, but
            OLS is used here by choice; note its t/z are then anticonservative on
            autocorrelated fMRI.

    Returns:
        A :class:`GLMResult`.
    """
    from nilearn.glm import compute_contrast
    from nilearn.glm.first_level import run_glm

    Y = _read_timeseries(surface_timeseries)          # (V, T)
    if Y.shape[1] != design.shape[0]:
        raise ValueError(f"{Y.shape[1]} frames but design has {design.shape[0]} rows")
    names = list(column_names)
    if contrast not in names:
        raise ValueError(f"contrast {contrast!r} not among columns {names}")

    # nilearn wants (n_scans, n_vertices); vertices with no signal are skipped
    good = np.all(np.isfinite(Y), axis=1) & (Y.std(axis=1) > 0)
    Yg = Y[good].T                                    # (T, nGood)

    labels, results = run_glm(Yg, design, noise_model=noise_model)
    con_vec = np.array([1.0 if n == contrast else 0.0 for n in names])
    con = compute_contrast(labels, results, con_vec, stat_type="t")

    n_vert = Y.shape[0]
    eff = np.full(n_vert, np.nan); tval = np.full(n_vert, np.nan); zval = np.full(n_vert, np.nan)
    eff[good] = con.effect_size().ravel()
    tval[good] = con.stat().ravel()
    zval[good] = con.z_score().ravel()

    out_prefix = Path(out_prefix)
    eff_p = _write_shape(eff, out_prefix.with_name(out_prefix.name + "_effect.shape.gii"))
    t_p = _write_shape(tval, out_prefix.with_name(out_prefix.name + "_tstat.shape.gii"))
    z_p = _write_shape(zval, out_prefix.with_name(out_prefix.name + "_zstat.shape.gii"))
    d_p = out_prefix.with_name(out_prefix.name + "_design.tsv")
    d_p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(d_p, design, delimiter="\t", header="\t".join(names), comments="")

    dof = float(design.shape[0] - np.linalg.matrix_rank(design))
    return GLMResult(eff_p, t_p, z_p, d_p, tuple(names), n_vert, design.shape[0], dof)
