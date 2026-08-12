"""Mono-exponential decay fitting by linear regression in log space.

Taking logs turns the model into a straight line::

    S(TE) = S0 * exp(-TE / T2*)
    ln S(TE) = ln S0 - TE / T2*

so ordinary least squares recovers ``ln S0`` as the intercept and ``1/T2*`` as
the slope against ``-TE``. It is fast, has a closed form, and with two echoes it
is exact -- a line through two points.

Logging does distort the noise -- the log-space variance of an echo goes as
(sigma/S)^2, so late, weak echoes are noisier in log space -- which suggests
weighting each echo by its signal. Measured against simulated Rician data at
these echo times, that suggestion is wrong: weighting makes the bias *worse* at
every SNR, and w=S^2 is worse still (at SNR 10, bias +1.30ms unweighted, +1.49
weighted, +2.81 with S^2). Weighting toward the early echoes drags the
regression's centroid toward TE1, which lengthens the late echo's lever arm and
hands the noisiest point *more* influence over the slope. Unweighted is
therefore the default; ``weighted=True`` remains available but is not
recommended.

What survives is the residual positive bias itself: magnitude data is
rectified, so weak late echoes read high and T2* is overestimated. It is
negligible at good SNR (+0.01ms at SNR 100) and worth knowing about in dropout
regions (+4.9ms at SNR 5).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .base import DecayResult

#: Physically plausible bounds for T2* (ms) in brain tissue at 3T. Fits landing
#: outside are noise, not measurement, and are marked invalid rather than kept.
T2STAR_MIN_MS = 1.0
T2STAR_MAX_MS = 500.0


def _read_surface_mean(path: Path) -> np.ndarray:
    """Time-average of a per-vertex timeseries GIfTI."""
    darrays = nib.load(str(path)).darrays
    if not darrays:
        raise ValueError(f"{Path(path).name}: no data arrays")
    # (nVert, nFrames) -> mean over frames
    return np.mean(np.stack([np.asarray(d.data, dtype=np.float64) for d in darrays], axis=1), axis=1)


def _write_shape(values: np.ndarray, out: Path) -> Path:
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    gii = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(values, dtype=np.float32),
                intent="NIFTI_INTENT_SHAPE",
                datatype="NIFTI_TYPE_FLOAT32",
                encoding="GIFTI_ENCODING_B64GZ",
            )
        ]
    )
    gii.to_filename(str(out))
    return out


def fit_arrays(signal: np.ndarray, echo_times_ms, *, weighted: bool = False):
    """Fit S0 and T2* from per-echo signal.

    Args:
        signal: ``(nVertex, nEcho)`` non-negative signal, one column per echo.
        echo_times_ms: Echo times in milliseconds, matching ``signal``'s columns.
        weighted: Weight each echo by its own signal. Off by default: it
            measurably increases bias rather than reducing it (see the module
            docstring).

    Returns:
        ``(s0, t2star_ms, r2, valid)``. ``r2`` is None for two echoes, where the
        fit is exact. Invalid vertices carry NaN.
    """
    te = np.asarray(echo_times_ms, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != te.size:
        raise ValueError(f"signal {signal.shape} does not match {te.size} echo times")
    if te.size < 2:
        raise ValueError(f"need at least 2 echoes to fit a decay, got {te.size}")
    if len(set(np.round(te, 6))) != te.size:
        raise ValueError(f"echo times must be distinct, got {te}")

    n_vert = signal.shape[0]
    usable = np.all(signal > 0, axis=1)          # log needs positive signal
    y = np.full((n_vert, te.size), np.nan)
    y[usable] = np.log(signal[usable])

    # design: ln S = ln S0 + (-TE) * (1/T2*)
    X = np.column_stack([np.ones_like(te), -te])          # (nEcho, 2)
    w = signal if weighted else np.ones_like(signal)

    s0 = np.full(n_vert, np.nan)
    t2s = np.full(n_vert, np.nan)
    r2 = np.full(n_vert, np.nan) if te.size > 2 else None

    idx = np.flatnonzero(usable)
    if idx.size:
        # Solve each vertex's weighted 2-parameter system in closed form; this
        # is a normal-equations solve done vectorised rather than per vertex.
        W = w[idx]
        Y = y[idx]
        s_w = W.sum(1)
        s_x = (W * X[:, 1]).sum(1)
        s_xx = (W * X[:, 1] ** 2).sum(1)
        s_y = (W * Y).sum(1)
        s_xy = (W * X[:, 1] * Y).sum(1)
        det = s_w * s_xx - s_x ** 2
        good = np.abs(det) > 1e-12
        b0 = np.full(idx.size, np.nan)
        b1 = np.full(idx.size, np.nan)
        b0[good] = (s_xx[good] * s_y[good] - s_x[good] * s_xy[good]) / det[good]
        b1[good] = (s_w[good] * s_xy[good] - s_x[good] * s_y[good]) / det[good]

        with np.errstate(divide="ignore", invalid="ignore"):
            t2 = 1.0 / b1                                  # b1 = 1/T2*
        s0[idx] = np.exp(b0)
        t2s[idx] = t2

        if r2 is not None:
            pred = b0[:, None] + np.outer(b1, X[:, 1])
            ss_res = (W * (Y - pred) ** 2).sum(1)
            mu = (W * Y).sum(1) / np.maximum(s_w, 1e-12)
            ss_tot = (W * (Y - mu[:, None]) ** 2).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2[idx] = 1.0 - ss_res / ss_tot

    # A decay that grows with TE, or one absurdly fast or slow, is noise.
    valid = np.isfinite(t2s) & (t2s >= T2STAR_MIN_MS) & (t2s <= T2STAR_MAX_MS) & np.isfinite(s0)
    s0[~valid] = np.nan
    t2s[~valid] = np.nan
    if r2 is not None:
        r2[~valid] = np.nan
    return s0, t2s, r2, valid


def loglinear(
    echoes,
    echo_times_ms,
    out_prefix,
    *,
    weighted: bool = False,
) -> DecayResult:
    """Fit S0 and T2* per vertex from per-echo surface timeseries.

    Each echo is averaged over time first, so the result is one static estimate
    per vertex for the run rather than a per-frame timeseries.

    Args:
        echoes: Per-echo surface timeseries (``.func.gii``), ordered by echo time.
        echo_times_ms: Echo times in milliseconds, matching ``echoes``.
        out_prefix: Path stem; ``_T2starmap`` / ``_S0map`` / ``_R2map`` are
            appended.
        weighted: Weight echoes by signal. Off by default; see the module
            docstring for why the obvious choice is the wrong one.

    Returns:
        A :class:`~lightprep.decay.base.DecayResult`.
    """
    echoes = [Path(e) for e in echoes]
    te = tuple(float(t) for t in echo_times_ms)
    if len(echoes) != len(te):
        raise ValueError(f"{len(echoes)} echoes but {len(te)} echo times")

    signal = np.column_stack([_read_surface_mean(e) for e in echoes])
    s0, t2s, r2, valid = fit_arrays(signal, te, weighted=weighted)

    out_prefix = Path(out_prefix)
    t2_path = _write_shape(t2s, out_prefix.with_name(out_prefix.name + "_T2starmap.shape.gii"))
    s0_path = _write_shape(s0, out_prefix.with_name(out_prefix.name + "_S0map.shape.gii"))
    r2_path = None
    if r2 is not None:
        r2_path = _write_shape(r2, out_prefix.with_name(out_prefix.name + "_R2map.shape.gii"))

    return DecayResult(
        t2star=t2_path,
        s0=s0_path,
        r2=r2_path,
        echo_times=te,
        n_vertices=int(signal.shape[0]),
        n_invalid=int((~valid).sum()),
        method="loglinear",
    )
