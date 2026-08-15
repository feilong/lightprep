"""Surface reconstruction QC from the Euler number.

Before FreeSurfer fixes the topology it produces ``?h.orig.nofix``, a surface
that still carries every handle and hole the segmentation put there. Its Euler
characteristic counts them, and that count turns out to be a good proxy for
how good the input image was: Rosen et al. (2018) found it tracks manual
quality ratings across large samples, which no intensity SNR measure does as
well. A clean scan gives a surface with few defects; a moving or noisy one
gives a segmentation that connects things that should not be connected, and
each of those is a handle.

    chi = V - E + F,   holes = (2 - chi) / 2

More negative is worse. The absolute value is not comparable across sites or
FreeSurfer versions, so it is used relatively -- an outlier within a study,
typically more than a few median absolute deviations below the rest.

This reads the surface directly rather than shelling out to
``mris_euler_number``: chi is a count of vertices, edges and faces, so it needs
no FreeSurfer install and no geometry. That last point matters here -- a recon
run through :mod:`lightprep.recon.fake` carries a deliberately wrong affine,
and the Euler number is untouched by it, being topological.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

HEMIS = ("lh", "rh")

#: The surface to measure: before topology fixing, which is the point.
DEFAULT_SURFACE = "orig.nofix"


@dataclass(frozen=True)
class SurfaceQC:
    """Euler numbers for one subject.

    Attributes:
        subject: FreeSurfer subject name.
        euler: ``{hemi: chi}``.
        holes: ``{hemi: (2 - chi) // 2}``, the handle count.
        surface: Which surface was measured.
    """

    subject: str
    euler: dict
    holes: dict
    surface: str = DEFAULT_SURFACE

    @property
    def mean_euler(self) -> float:
        """Mean over hemispheres -- the form Rosen et al. report."""
        return float(np.mean(list(self.euler.values()))) if self.euler else float("nan")

    @property
    def total_holes(self) -> int:
        return int(sum(self.holes.values()))


def euler_number(surface) -> int:
    """Euler characteristic of a FreeSurfer surface.

    Args:
        surface: Path to a surface such as ``lh.orig.nofix``.

    Returns:
        ``V - E + F``. A defect-free closed surface gives 2.

    Raises:
        FileNotFoundError: If the surface does not exist.
    """
    surface = Path(surface)
    if not surface.exists():
        raise FileNotFoundError(f"no such surface: {surface}")
    v, f = nib.freesurfer.read_geometry(str(surface))
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1)
    n_edges = len(np.unique(edges, axis=0))
    return int(len(v) - n_edges + len(f))


def surface_qc(subject: str, subjects_dir, surface: str = DEFAULT_SURFACE) -> SurfaceQC:
    """Euler number per hemisphere for one subject.

    Args:
        subject: FreeSurfer subject name.
        subjects_dir: The SUBJECTS_DIR holding it.
        surface: Surface to measure; the pre-fix one by default.

    Returns:
        A :class:`SurfaceQC`. Hemispheres whose surface is missing are absent
        rather than guessed -- a corrected derivative folder may legitimately
        not carry ``orig.nofix``, in which case point this at the recon it was
        derived from.
    """
    surf_dir = Path(subjects_dir) / subject / "surf"
    euler, holes = {}, {}
    for hemi in HEMIS:
        path = surf_dir / f"{hemi}.{surface}"
        if not path.exists():
            continue
        chi = euler_number(path)
        euler[hemi] = chi
        holes[hemi] = (2 - chi) // 2
    return SurfaceQC(subject=subject, euler=euler, holes=holes, surface=surface)


def flag_outliers(results, n_mad: float = 3.0):
    """Which subjects sit far below the rest on mean Euler number.

    Absolute Euler numbers are not comparable across sites or FreeSurfer
    versions, so the useful question is relative: does this subject's surface
    carry far more defects than its cohort's? Median absolute deviation is used
    rather than the standard deviation because the outliers being looked for
    would inflate an SD and hide themselves.

    Args:
        results: Iterable of :class:`SurfaceQC`.
        n_mad: How many scaled MADs below the median counts as an outlier.

    Returns:
        ``(flagged, threshold, median)`` -- flagged is a list of subject names.
    """
    results = [r for r in results if r.euler]
    if len(results) < 3:
        return [], float("nan"), float("nan")
    vals = np.array([r.mean_euler for r in results], dtype=float)
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median))) * 1.4826    # -> sigma units
    if mad == 0:
        return [], float("-inf"), median
    threshold = median - n_mad * mad
    return ([r.subject for r in results if r.mean_euler < threshold],
            threshold, median)
