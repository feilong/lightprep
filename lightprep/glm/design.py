"""Building the GLM design matrix: task and nuisance regressors.

The experiment is a block design that tiles the whole run: alternating B and A
blocks, 20 s each, starting and ending with B. Because every timepoint belongs
to a block, there is no rest baseline to model against -- and modelling both A
and B boxcars alongside a constant would be perfectly collinear with it. So only
condition A is modelled (convolved with the HRF); B is absorbed by the intercept
(Legendre order 0). The A regressor's coefficient is then the A - B contrast
directly.

Nuisance regressors, as specified:
  * 6 head-motion parameters and their temporal derivatives (12)
  * Legendre polynomial trends up to 2nd order (3; order 0 is the intercept)
  * framewise displacement (1)
  * 6 aCompCor components (added by :mod:`lightprep.glm.acompcor`)
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval

BLOCK_SECONDS = 20.0
RUN_SECONDS = 260.0


def block_onsets(run_seconds=RUN_SECONDS, block_seconds=BLOCK_SECONDS):
    """Onsets (s) of the A and B blocks for a B-starting, B-ending sequence."""
    n_blocks = int(round(run_seconds / block_seconds))
    if n_blocks % 2 == 0:
        raise ValueError(
            f"{n_blocks} blocks would not both start and end with B; "
            "the sequence needs an odd count"
        )
    starts = np.arange(n_blocks) * block_seconds
    return {"B": starts[0::2], "A": starts[1::2]}   # B first, then alternating


def task_regressor(n_frames, t_r, *, hrf_model="glover", condition="A"):
    """Convolved boxcar for one condition, sampled at the run's frame times.

    Returns ``(regressor, frame_times)``. Uses nilearn's HRF so the convolution
    matches the rest of the ecosystem.
    """
    from nilearn.glm.first_level import compute_regressor

    onsets = block_onsets()[condition]
    frame_times = np.arange(n_frames) * t_r
    exp = np.vstack([onsets, np.full(onsets.size, BLOCK_SECONDS), np.ones(onsets.size)])
    signal, _ = compute_regressor(exp, hrf_model, frame_times, con_id=condition)
    return signal[:, 0], frame_times


def legendre_trends(n_frames, order=2):
    """Legendre polynomials 0..order over the run, as columns (n_frames, order+1).

    Order 0 is the constant term, so it serves as the model intercept.
    """
    x = np.linspace(-1.0, 1.0, n_frames)
    cols = [legval(x, [0] * k + [1]) for k in range(order + 1)]
    return np.column_stack(cols)


def motion_regressors(motion_par):
    """6 motion parameters and their temporal derivatives -> (n_frames, 12).

    ``motion_par`` is MCFLIRT's ``.par``: rotations (rad) then translations (mm).
    The derivative is a backward difference, first row zero.
    """
    par = np.atleast_2d(np.loadtxt(motion_par))
    if par.shape[1] != 6:
        raise ValueError(f"expected 6 motion columns, got {par.shape[1]}")
    deriv = np.vstack([np.zeros((1, 6)), np.diff(par, axis=0)])
    return np.column_stack([par, deriv])


def framewise_displacement(motion_par, radius_mm=50.0):
    """Power et al. framewise displacement -> (n_frames,). First frame is 0.

    Rotations (radians) are converted to millimetres of arc on a sphere of the
    given radius before summing with the translations.
    """
    par = np.atleast_2d(np.loadtxt(motion_par)).astype(float)
    d = np.vstack([np.zeros((1, 6)), np.abs(np.diff(par, axis=0))])
    d[:, :3] *= radius_mm                 # rotations rad -> mm of arc
    return d.sum(axis=1)


def assemble(task, trends, motion, fd, acompcor):
    """Stack the pieces into (design, column_names), task first."""
    blocks = [
        (task[:, None], ["task_A"]),
        (trends, [f"legendre{ i }" for i in range(trends.shape[1])]),
        (motion[:, :6], [f"motion{i}" for i in range(6)]),
        (motion[:, 6:], [f"motion_d{i}" for i in range(6)]),
        (fd[:, None], ["fd"]),
        (acompcor, [f"acompcor{i}" for i in range(acompcor.shape[1])]),
    ]
    mats = [b[0] for b in blocks]
    names = [n for b in blocks for n in b[1]]
    return np.column_stack(mats), names
