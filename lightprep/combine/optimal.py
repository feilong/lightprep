"""Optimal combination of multi-echo data (Posse et al., MRM 1999).

The BOLD signal change scales with echo time as ``dS/S ~ TE * exp(-TE / T2*)``,
so the echo that carries the most BOLD contrast is the one nearest ``TE = T2*``,
and it differs from voxel to voxel. Weighting each echo by::

    w_n = TE_n * exp(-TE_n / T2*)      (normalised so the weights sum to 1)

is the matched filter for that signal: it maximises BOLD CNR, and equivalently
tSNR, at every vertex. This is what tedana does by default.

The weights need a per-vertex T2*, fitted here from the same echoes being
combined (their time-average), so the combination is self-consistent with the
data rather than borrowing a T2* estimated elsewhere. The weights are static --
one set per vertex, applied to every frame -- which is the standard choice;
re-fitting T2* per frame would chase noise.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ..decay.loglinear import fit_arrays
from .base import CombineResult


def _read_timeseries(path: Path) -> np.ndarray:
    """(nVertex, nFrame) from a per-vertex timeseries GIfTI."""
    darrays = nib.load(str(path)).darrays
    if not darrays:
        raise ValueError(f"{Path(path).name}: no data arrays")
    return np.stack([np.asarray(d.data, dtype=np.float64) for d in darrays], axis=1)


def optimal_weights(signal_per_echo, echo_times_ms):
    """Static optimal-combination weights from per-echo mean signal.

    Args:
        signal_per_echo: ``(nVertex, nEcho)`` non-negative mean signal.
        echo_times_ms: Echo times in ms, matching the columns.

    Returns:
        ``(weights, fitted)``. ``weights`` is ``(nVertex, nEcho)`` summing to 1
        along the echo axis. ``fitted`` marks vertices whose T2* fit succeeded;
        the rest fall back to the matched filter's zero-contrast limit, a plain
        mean.
    """
    te = np.asarray(echo_times_ms, dtype=np.float64)
    n_vert, n_echo = signal_per_echo.shape
    if te.size != n_echo:
        raise ValueError(f"{n_echo} echoes but {te.size} echo times")

    _, t2star, _, valid = fit_arrays(signal_per_echo, te)

    w = np.empty((n_vert, n_echo), dtype=np.float64)
    # matched filter where T2* is trustworthy
    t2 = t2star[valid][:, None]
    wv = te[None, :] * np.exp(-te[None, :] / t2)
    w[valid] = wv / wv.sum(axis=1, keepdims=True)
    # fallback: equal weights. As T2* -> infinity the matched filter tends to
    # TE-proportional and then flat; with no reliable T2* a plain mean is the
    # honest default rather than a guessed decay.
    w[~valid] = 1.0 / n_echo
    return w, valid


def optimal_combination(
    echoes,
    echo_times_ms,
    out,
    *,
    save_weights: bool = True,
) -> CombineResult:
    """Combine per-echo surface timeseries into one, by optimal combination.

    Args:
        echoes: Per-echo surface timeseries (``.func.gii``), ordered by echo
            time. At least two are required.
        echo_times_ms: Echo times in milliseconds, matching ``echoes``.
        out: Output path for the combined timeseries (``.func.gii``).
        save_weights: Also write the per-vertex weights beside the output.

    Returns:
        A :class:`~lightprep.combine.base.CombineResult`.

    Raises:
        ValueError: On fewer than two echoes or mismatched inputs.
    """
    echoes = [Path(e) for e in echoes]
    te = tuple(float(t) for t in echo_times_ms)
    if len(echoes) < 2:
        raise ValueError(f"optimal combination needs >=2 echoes, got {len(echoes)}")
    if len(echoes) != len(te):
        raise ValueError(f"{len(echoes)} echoes but {len(te)} echo times")

    series = [_read_timeseries(e) for e in echoes]
    shapes = {s.shape for s in series}
    if len(shapes) > 1:
        raise ValueError(f"echoes must share vertex and frame counts, got {sorted(shapes)}")
    n_vert, n_frames = series[0].shape

    stack = np.stack(series, axis=2)                 # (nVert, nFrame, nEcho)
    mean_signal = np.nanmean(stack, axis=1)          # (nVert, nEcho)
    mean_signal[~np.isfinite(mean_signal)] = 0.0
    mean_signal[mean_signal < 0] = 0.0

    weights, fitted = optimal_weights(mean_signal, te)   # (nVert, nEcho)
    combined = np.einsum("vte,ve->vt", stack, weights).astype(np.float32)

    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                combined[:, t], intent="NIFTI_INTENT_TIME_SERIES",
                datatype="NIFTI_TYPE_FLOAT32", encoding="GIFTI_ENCODING_B64GZ",
            )
            for t in range(n_frames)
        ]
    ).to_filename(str(out))

    weights_path = None
    if save_weights:
        # Swap the BIDS suffix: the weights are their own thing, not bold data.
        base = out.name
        if base.endswith("_bold.func.gii"):
            weights_name = base[: -len("_bold.func.gii")] + "_weights.shape.gii"
        else:
            weights_name = base.replace(".func.gii", "") + "_weights.shape.gii"
        weights_path = out.with_name(weights_name)
        nib.gifti.GiftiImage(
            darrays=[
                nib.gifti.GiftiDataArray(
                    weights[:, e].astype(np.float32), intent="NIFTI_INTENT_SHAPE",
                    datatype="NIFTI_TYPE_FLOAT32", encoding="GIFTI_ENCODING_B64GZ",
                )
                for e in range(len(te))
            ]
        ).to_filename(str(weights_path))

    return CombineResult(
        output=out,
        weights=weights_path,
        echo_times=te,
        n_vertices=n_vert,
        n_frames=n_frames,
        n_fallback=int((~fitted).sum()),
        method="optimal",
    )
