"""``recon-all``, on the grid FreeSurfer wants to put the anatomy on.

Two methods live here, differing only in what conform resamples to:

``std``
    ``recon-all -all``. Conform to 1 mm isotropic. Anything not already 1 mm
    isotropic and axis-aligned is interpolated once, on the way in.

``hires``
    ``recon-all -all -hires``. Conform to the smallest native voxel size
    instead of 1 mm, so submillimetre detail survives -- but the grid is still
    isotropic and axis-aligned, so anisotropic or obliquely prescribed data is
    still interpolated.

:mod:`lightprep.recon.fake` is the third option, which avoids the
interpolation altogether.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .._utils import run
from ..coreg.freesurfer import _freesurfer_env
from .base import ReconResult

#: FreeSurfer's recommended expert option for submillimetre data: the hi-res
#: surfaces carry more vertices, so inflation needs more iterations.
HIRES_EXPERT = "mris_inflate -n 15"


def field_strength_args(field_strength) -> list:
    """``recon-all`` flags for the scanner field strength.

    FreeSurfer has one field-strength preset, ``-3T``, which selects the 3T
    Talairach atlas and a stronger N3 non-uniformity correction. There is no
    ``-7T``: for 7T the honest thing is to *not* claim 3T tuning, so this
    returns no flag and leaves the bias correction to the caller's expert
    options. 7T structurals usually need more aggressive intensity
    normalisation than any preset provides.

    Args:
        field_strength: ``1.5``, ``3`` or ``7`` (or the strings ``"1.5T"``,
            ``"3T"``, ``"7T"``), or ``None`` to pass no flag at all.

    Returns:
        A list of flags to splice into the ``recon-all`` command line.

    Raises:
        ValueError: If the field strength is not one this maps.
    """
    if field_strength is None:
        return []
    key = str(field_strength).upper().rstrip("T")
    if key in ("1.5", "1_5"):
        return []                      # FreeSurfer's defaults are the 1.5T ones
    if key == "3":
        return ["-3T"]
    if key == "7":
        return []                      # see the docstring: no preset exists
    raise ValueError(
        f"unsupported field_strength {field_strength!r}; expected 1.5, 3 or 7")


def _recon_all(
    t1,
    subject: str,
    subjects_dir,
    *,
    extra: list | None = None,
    field_strength=3,
    threads: int = 8,
    parallel: bool = False,
    expert: str | None = None,
    freesurfer_home=None,
    force: bool = False,
) -> Path:
    """Run ``recon-all -all`` and return the subject directory."""
    subjects_dir = Path(subjects_dir).resolve()
    subjects_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = subjects_dir / subject

    if (subject_dir / "scripts" / "recon-all.done").exists() and not force:
        return subject_dir
    # -all cannot resume a directory left behind by a killed run, and
    # `recon-all -i` refuses to start if the subject directory exists at all.
    if subject_dir.exists():
        shutil.rmtree(subject_dir)

    if expert:
        # NOT recon-all's -expert flag. recon-all 8.2.0 line 4220 reads
        #     if($XOptsFile) set cmd = ($cmd --expert $XOptsFile)
        # where every other site correctly tests $#XOptsFile. XOptsFile is a
        # tcsh list holding a path, so the bare $XOptsFile makes tcsh evaluate
        # a path arithmetically and abort with "if: Expression Syntax." right
        # after the Sphere stage -- killing any run that reaches surface
        # registration. The global expert file sets GlobXOptsFile instead,
        # which mris_inflate honours and which that buggy line never inspects.
        (subjects_dir / "global-expert-options.txt").write_text(expert + "\n")

    cmd = ["recon-all", "-all", *field_strength_args(field_strength),
           *(extra or []), "-s", subject, "-i", str(t1),
           "-sd", str(subjects_dir), "-threads", str(threads)]
    if parallel:
        cmd.append("-parallel")
    run(cmd, extra_env=_freesurfer_env(freesurfer_home, subjects_dir))
    return subject_dir


def std(t1, subject: str, subjects_dir, **kwargs) -> ReconResult:
    """``recon-all -all``, conformed to 1 mm isotropic.

    Args:
        t1: The T1-weighted volume.
        subject: FreeSurfer subject name.
        subjects_dir: SUBJECTS_DIR to write into.
        **kwargs: Passed to the shared runner -- ``field_strength`` (default
            3), ``threads``, ``parallel``, ``expert``, ``freesurfer_home``,
            ``force``.

    Returns:
        A :class:`~lightprep.recon.base.ReconResult` with
        ``interpolated=True``: unless the input is already 1 mm isotropic and
        axis-aligned, conform resampled it.
    """
    subject_dir = _recon_all(t1, subject, subjects_dir, **kwargs)
    return ReconResult(subject=subject, subjects_dir=subject_dir.parent,
                       method="std", interpolated=True, conform="1mm",
                       input_volume=Path(t1))


def hires(t1, subject: str, subjects_dir, *, expert: str | None = HIRES_EXPERT,
          **kwargs) -> ReconResult:
    """``recon-all -all -hires``, conformed to the smallest native voxel size.

    Keeps submillimetre detail that the 1 mm conform of :func:`std` would
    throw away, but the target grid is still isotropic and axis-aligned, so
    anisotropic or oblique data is still interpolated once.

    Args:
        t1: The T1-weighted volume.
        subject: FreeSurfer subject name.
        subjects_dir: SUBJECTS_DIR to write into.
        expert: Expert options file contents. Defaults to
            :data:`HIRES_EXPERT`; pass ``None`` for none.
        **kwargs: As :func:`std`.

    Returns:
        A :class:`~lightprep.recon.base.ReconResult` with ``interpolated=True``.
    """
    subject_dir = _recon_all(t1, subject, subjects_dir, extra=["-hires"],
                             expert=expert, **kwargs)
    return ReconResult(subject=subject, subjects_dir=subject_dir.parent,
                       method="hires", interpolated=True, conform="min",
                       input_volume=Path(t1))
