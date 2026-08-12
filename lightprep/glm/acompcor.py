"""Anatomical CompCor nuisance components (Behzadi et al., 2007).

Principal components of the BOLD timeseries within white matter and CSF -- tissue
that carries no signal of interest, so its dominant temporal patterns are
physiological and scanner noise shared with grey matter.

The noise ROI is built the way fMRIPrep builds it: take the grey-matter mask,
*dilate* it, and subtract that from the WM and CSF masks. Any voxel within one
dilation step of grey matter is dropped, so partial-volume BOLD never enters the
ROI. This is done in the functional grid, at 2.8 mm, where a single-voxel
dilation already clears a wide margin.

The masks come from FreeSurfer's aparc+aseg, resampled into the functional space
through the coregistration, nearest-neighbour so labels stay intact.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# aparc+aseg label groups. In *aparc*+aseg the cortical ribbon is labelled with
# the parcellation numbers (1000-2999), not the aseg 3/42 -- so cortex is matched
# by range below, not listed here.
GM_SUBCORTICAL = (
    8, 47,                       # cerebellar cortex
    10, 11, 12, 13, 17, 18, 26,  # L thalamus, caudate, putamen, pallidum, hipp, amyg, accumbens
    49, 50, 51, 52, 53, 54, 58,  # R
)
CORTEX_LABEL_RANGE = (1000, 3000)                          # [lo, hi)
WM_LABELS = (2, 41, 7, 46, 77, 251, 252, 253, 254, 255)    # cerebral+cerebellar WM, CC, hypo
# Ventricular CSF only. The generic "CSF" label (24) is the extracerebral rim --
# ~385 cm3 of non-brain tissue dominated by motion and pulsation -- and has no
# place in a physiological-noise ROI; the ventricles are the clean CSF signal.
CSF_LABELS = (4, 43, 5, 44, 14, 15, 72)                    # lateral, inf-lat, 3rd, 4th, 5th


def _tissue_masks(labels):
    lo, hi = CORTEX_LABEL_RANGE
    gm = ((labels >= lo) & (labels < hi)) | np.isin(labels, GM_SUBCORTICAL)
    return gm, np.isin(labels, WM_LABELS), np.isin(labels, CSF_LABELS)


def _resample_labels_to_func(aseg_path, func_img, coreg_lta):
    """Put aparc+aseg on the functional grid, nearest-neighbour."""
    from nitransforms.linear import Affine

    aseg = nib.load(str(aseg_path))
    aseg_data = np.asarray(aseg.dataobj)
    aseg_inv = np.linalg.inv(aseg.affine)

    # LTA (fs) maps anat world -> func world; we need func -> anat to pull labels.
    to_func = Affine.from_filename(str(coreg_lta), fmt="fs")
    func_to_anat = ~to_func

    shape = func_img.shape[:3]
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], axis=0)
    func_world = func_img.affine @ vox
    anat_world = np.asarray(func_to_anat.map(func_world[:3].T))
    aseg_vox = np.rint(aseg_inv[:3, :3] @ anat_world.T + aseg_inv[:3, 3:4]).astype(int)

    out = np.zeros(ii.size, dtype=aseg_data.dtype)
    ok = np.all((aseg_vox >= 0) & (aseg_vox < np.array(aseg_data.shape)[:, None]), axis=0)
    idx = aseg_vox[:, ok]
    out[ok] = aseg_data[idx[0], idx[1], idx[2]]
    return out.reshape(shape)


def noise_roi(aseg_path, func_img, coreg_lta, *, gm_dilation=1):
    """WM+CSF noise ROI on the functional grid, GM-adjacent voxels removed.

    Returns ``(roi, wm, csf, gm_dilated)`` boolean volumes for inspection.
    """
    labels = _resample_labels_to_func(aseg_path, func_img, coreg_lta)
    gm, wm, csf = _tissue_masks(labels)

    if gm_dilation < 1:
        raise ValueError("gm_dilation must be >= 1 (scipy treats 0 as dilate-to-convergence)")
    gm_dil = ndimage.binary_dilation(gm, iterations=gm_dilation)
    wm_roi = wm & ~gm_dil
    csf_roi = csf & ~gm_dil
    return (wm_roi | csf_roi), wm_roi, csf_roi, gm_dil


def _detrend_and_normalise(ts, order=2):
    """Remove Legendre trends 0..order from each voxel, then unit-variance it."""
    from .design import legendre_trends

    n = ts.shape[0]
    X = legendre_trends(n, order)
    beta, *_ = np.linalg.lstsq(X, ts, rcond=None)
    resid = ts - X @ beta
    sd = resid.std(axis=0)
    good = sd > 0
    resid[:, good] /= sd[good]
    return resid[:, good]


def compcor_components(volume_timeseries, roi, n_components=6, trend_order=2):
    """Top temporal principal components of the noise ROI.

    Args:
        volume_timeseries: ``(X, Y, Z, T)`` functional data.
        roi: boolean ``(X, Y, Z)`` noise ROI.
        n_components: how many components to return.
        trend_order: Legendre order removed before the decomposition.

    Returns:
        ``(T, n_components)`` array, and the fraction of ROI variance each
        component explains.
    """
    ts = volume_timeseries[roi].T                       # (T, nVox)
    ts = ts[:, np.all(np.isfinite(ts), axis=0) & (ts.std(0) > 0)]
    ts = _detrend_and_normalise(ts, trend_order)
    # temporal PCs: left singular vectors of (time x voxel)
    U, S, _ = np.linalg.svd(ts, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    k = min(n_components, U.shape[1])
    return U[:, :k], var[:k]


def optimal_combination_volume(echo_paths, echo_times_ms):
    """Volumetric optimal combination, for computing aCompCor on multi-echo data.

    Reuses the same weighting as the surface combination, so the nuisance ROI
    sees the same kind of data the analysis does.
    """
    from ..combine.optimal import optimal_weights

    imgs = [nib.load(str(p)) for p in echo_paths]
    data = np.stack([im.get_fdata(dtype=np.float32) for im in imgs], axis=-1)  # (X,Y,Z,T,E)
    shape = data.shape[:3]
    flat = data.reshape(-1, data.shape[3], data.shape[4])                     # (V,T,E)
    mean_sig = flat.mean(axis=1)                                              # (V,E)
    mean_sig[mean_sig < 0] = 0
    w, _ = optimal_weights(mean_sig, echo_times_ms)
    comb = np.einsum("vte,ve->vt", flat, w)
    return comb.reshape(*shape, data.shape[3]), imgs[0]
