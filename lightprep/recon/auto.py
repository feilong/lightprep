"""Choosing a recon method from the voxel geometry.

The fake-affine trick of :mod:`lightprep.recon.fake` buys exactness at a
price: FreeSurfer insists on an isotropic conformed grid, so a volume whose
voxels are not isotropic must claim one voxel size where it truly has two or
three. ``M`` removes that from every coordinate afterwards, exactly -- but it
cannot remove it from what FreeSurfer *did* while the recon was running.
Skull stripping, the Talairach registration, the atlas priors and the surface
deformation all ran on a head that was the wrong shape by

    anisotropy - 1  =  max(voxel size) / min(voxel size)  -  1

At 2% (a 0.85 x 0.868 x 0.868 mm acquisition) that is well inside the range
those algorithms tolerate. At 50% it is not, and the surfaces themselves --
not merely their coordinates -- become suspect. No amount of correcting
afterwards recovers a recon that placed the white surface in the wrong place.

So the default is to use ``native`` while the distortion is small and to fall
back to interpolating when it is not: past :data:`MAX_ANISOTROPY`, one
interpolation is the lesser harm. The threshold is a judgement call, not a
theorem, which is why it is a parameter.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

#: Largest max/min voxel-size ratio for which ``native`` is used by default.
#: 1.2 is a 20% distortion of FreeSurfer's view of the anatomy.
MAX_ANISOTROPY = 1.2

#: What :func:`auto` falls back to when the input is too anisotropic. ``hires``
#: rather than ``std`` because it conforms to the smallest native voxel, so it
#: keeps whatever resolution the acquisition had on its best axis.
DEFAULT_FALLBACK = "hires"

#: Relative slack on the comparison. NIfTI stores voxel sizes as float32, so a
#: volume that is 1.0 x 1.0 x 1.2 mm by construction reads back with a ratio of
#: 1.2000000476837158 -- fractionally over a 1.2 threshold, for no reason a
#: user could see or act on. The threshold is a judgement call to one decimal
#: place; it should not turn on the seventh.
ANISOTROPY_TOL = 1e-6


def within(ratio: float, max_anisotropy: float) -> bool:
    """Is this ratio inside the threshold, allowing for float32 headers?"""
    return ratio <= max_anisotropy * (1.0 + ANISOTROPY_TOL)


class AnisotropyWarning(UserWarning):
    """The fake-geometry trick is being used on strongly anisotropic data."""


def anisotropy(t1) -> float:
    """The ratio of largest to smallest voxel dimension.

    ``1.0`` for isotropic data, which is the case the trick handles for free.

    Args:
        t1: A volume, or anything :func:`nibabel.load` accepts.

    Returns:
        ``max(zooms) / min(zooms)`` over the three spatial axes.
    """
    zooms = np.asarray(nib.load(t1).header.get_zooms()[:3], dtype=float)
    return float(zooms.max() / zooms.min())


def choose_method(t1, *, max_anisotropy: float = MAX_ANISOTROPY,
                  fallback: str = DEFAULT_FALLBACK) -> tuple[str, float]:
    """Which recon method suits this volume, and why.

    Pure: it inspects the header and returns a name, warning about nothing and
    running nothing. :func:`auto` is the one that acts on the answer.

    Args:
        t1: The T1-weighted volume.
        max_anisotropy: Largest max/min voxel ratio for which ``native`` is
            chosen. Raise it to accept more distortion, or pass ``inf`` to
            always choose ``native``.
        fallback: Method to name when the ratio is exceeded.

    Returns:
        ``(method_name, ratio)``.
    """
    ratio = anisotropy(t1)
    return (("native" if within(ratio, max_anisotropy) else fallback), ratio)


def auto(t1, subject: str, subjects_dir, *, method: str | None = None,
         max_anisotropy: float = MAX_ANISOTROPY,
         fallback: str = DEFAULT_FALLBACK, **kwargs):
    """Reconstruct, choosing the method from the voxel geometry.

    Isotropic or near-isotropic data goes through
    :func:`~lightprep.recon.fake.native`, which never interpolates the
    anatomy. Strongly anisotropic data warns and falls back to a method that
    does interpolate, because past that point the fake grid distorts
    FreeSurfer's view of the head more than one resampling would.

    Args:
        t1: The T1-weighted volume, or several. Several means they must be
            averaged, which ``native`` cannot do, so ``fallback`` is chosen.
        subject: FreeSurfer subject name.
        subjects_dir: SUBJECTS_DIR to write into.
        method: Override the choice entirely -- ``"native"``, ``"hires"`` or
            ``"std"``. The geometry check still runs, so forcing ``native`` on
            anisotropic data warns rather than passing silently.
        max_anisotropy: Largest max/min voxel ratio for which ``native`` is
            chosen. Pass ``float("inf")`` to always choose ``native``.
        fallback: Method used when the ratio is exceeded.
        **kwargs: Passed to the chosen method.

    Returns:
        The chosen method's :class:`~lightprep.recon.base.ReconResult`.

    Raises:
        ValueError: If ``method`` or ``fallback`` is not a real method, or is
            ``"auto"`` itself.

    Warns:
        AnisotropyWarning: When the input is too anisotropic for ``native``.
    """
    from . import METHODS               # imported late: METHODS holds `auto`

    for name, what in ((method, "method"), (fallback, "fallback")):
        if name is None:
            continue
        if name == "auto":
            raise ValueError(f"{what}='auto' would recurse; name a real method")
        if name not in METHODS:
            raise ValueError(f"unknown {what} {name!r}; "
                             f"available: {sorted(set(METHODS) - {'auto'})}")

    multiple = not isinstance(t1, (str, Path)) and len(list(t1)) > 1
    probe = list(t1)[0] if not isinstance(t1, (str, Path)) else t1
    chosen, ratio = choose_method(probe, max_anisotropy=max_anisotropy,
                                  fallback=fallback)
    if multiple and chosen == "native":
        # Several structurals can be averaged into one anatomy, which is worth
        # more than avoiding the interpolation: recon-all's -motioncor stage
        # aligns them with mri_robust_template and takes a median, buying real
        # SNR. native cannot do it -- averaging interpolates by construction.
        chosen = fallback
        warnings.warn(
            f"{len(list(t1))} structurals given, so they will be averaged into "
            f"one anatomy by FreeSurfer's -motioncor stage. That is an "
            f"interpolation, which 'native' exists to avoid, so {fallback!r} "
            f"is used instead. Pass a single volume to keep 'native'.",
            AnisotropyWarning, stacklevel=2)
    elif method is not None and method != chosen:
        chosen = method
    elif chosen != "native":
        warnings.warn(
            f"voxels are anisotropic by {ratio:.3f} (max/min), above the "
            f"{max_anisotropy} threshold: the fake-geometry recon would make "
            f"FreeSurfer see the head stretched by {100 * (ratio - 1):.0f}%, "
            f"which distorts skull stripping, the Talairach registration and "
            f"the surface placement itself -- not just the coordinates, which "
            f"the correction would fix. Falling back to {chosen!r}, which "
            f"interpolates the anatomy once. Pass method='native' to override, "
            f"or raise max_anisotropy.",
            AnisotropyWarning, stacklevel=2)

    return METHODS[chosen](t1, subject, subjects_dir, **kwargs)
