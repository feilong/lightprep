"""Head motion correction with FSL's MCFLIRT."""

from __future__ import annotations

import shutil
from pathlib import Path

import nibabel as nib

from .._utils import run, strip_ext
from .base import HMCResult

INTERPOLATIONS = ("nearestneighbour", "trilinear", "spline", "sinc")
COSTS = ("mutualinfo", "woods", "corratio", "normcorr", "normmi", "leastsquares")


def _check_echoes(echoes) -> tuple[list[Path], int]:
    """Validate the echo list and return the paths plus the volume count."""
    paths = [Path(e).resolve() for e in echoes]
    if not paths:
        raise ValueError("no echoes given")

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("echo file(s) not found: " + ", ".join(missing))

    shapes = {nib.load(p).shape for p in paths}
    if len(shapes) > 1:
        raise ValueError(
            "echoes must be on the same grid with the same number of volumes, got "
            f"{sorted(shapes)}. A shared set of transforms cannot be applied to "
            "echoes that disagree."
        )

    shape = shapes.pop()
    if len(shape) != 4:
        raise ValueError(f"expected 4D echo timeseries, got shape {shape}")
    return paths, shape[3]


def _make_reference(echo: Path, ref, n_volumes: int, out_dir: Path) -> Path:
    """Materialise the registration target as a file on disk.

    Writing the reference out explicitly (rather than leaning on mcflirt's
    ``-refvol``) means the very same image is reused when the transforms are
    replayed onto the other echoes.
    """
    if isinstance(ref, (str, Path)) and str(ref) not in ("middle", "mean"):
        ref = Path(ref).resolve()
        if not ref.exists():
            raise FileNotFoundError(f"reference image not found: {ref}")
        return ref

    target = out_dir / "reference.nii.gz"
    if ref == "mean":
        run(["fslmaths", echo, "-Tmean", target])
        return target

    index = n_volumes // 2 if ref == "middle" else int(ref)
    if not 0 <= index < n_volumes:
        raise ValueError(
            f"reference volume {index} is out of range for {n_volumes} volumes"
        )
    run(["fslroi", echo, target, index, 1])
    return target


def mcflirt(
    echoes,
    out_dir,
    *,
    ref="middle",
    ref_echo: int = 0,
    cost: str = "normcorr",
    dof: int = 6,
    stages: int = 3,
    interp: str = "trilinear",
    keep_workdir: bool = False,
) -> HMCResult:
    """Estimate head motion on one echo and apply it to every echo.

    Motion is estimated once, on ``ref_echo`` (the first echo by default: it has
    the shortest TE, so the most signal and the least dropout). The resulting
    per-volume transforms are then replayed unchanged onto all echoes. Echoes
    come from a single excitation and therefore share a head position, so a
    per-echo estimate would only add noise -- and would break the voxelwise
    correspondence across echoes that T2*/S0 fitting relies on.

    Args:
        echoes: Paths to the echo timeseries, ordered by echo time. A
            single-echo run is just a list of length one.
        out_dir: Directory for the realigned echoes and the motion estimates.
        ref: Registration target. ``"middle"`` (default) uses the middle volume,
            ``"mean"`` the mean volume, an int a specific volume index, or pass a
            path to use an existing image.
        ref_echo: Index into ``echoes`` to estimate motion on. Defaults to the
            first echo; pass ``1`` for the mid-TE echo that afni_proc.py favours.
        cost: MCFLIRT cost function, one of :data:`COSTS`.
        dof: Degrees of freedom; 6 (rigid body) is what head motion warrants.
        stages: MCFLIRT search levels. Pass 4 for a slower, finer final stage.
        interp: Interpolation used when applying the transforms, one of
            :data:`INTERPOLATIONS`. The default matches what MCFLIRT applies
            natively. Prefer it: on this data ``spline`` rings badly enough to
            give up most of the correction (DVARS -0.9% vs -4.0% for trilinear),
            while trilinear reproduces MCFLIRT's own output exactly.
        keep_workdir: Keep MCFLIRT's intermediate estimation output.

    Returns:
        An :class:`~lightprep.hmc.base.HMCResult`. ``parameters`` points at
        MCFLIRT's ``.par`` file: six columns, rotations (radians) then
        translations (mm).

    Raises:
        ValueError: If the echoes disagree in shape, or an argument is invalid.
        DependencyError: If FSL is not on PATH.

    Note:
        Phase images must not be passed here. Interpolating wrapped phase across
        the +/-pi boundary produces nonsense. Estimate on magnitude, then apply
        these transforms to the real and imaginary parts instead.
    """
    if cost not in COSTS:
        raise ValueError(f"cost must be one of {COSTS}, got {cost!r}")
    if interp not in INTERPOLATIONS:
        raise ValueError(f"interp must be one of {INTERPOLATIONS}, got {interp!r}")

    paths, n_volumes = _check_echoes(echoes)
    if not 0 <= ref_echo < len(paths):
        raise ValueError(
            f"ref_echo {ref_echo} is out of range for {len(paths)} echo(s)"
        )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [out_dir / f"{strip_ext(p)}.nii.gz" for p in paths]
    clobbered = [str(p) for p, o in zip(paths, outputs) if p == o]
    if clobbered:
        raise ValueError(
            "out_dir would overwrite the input echo(es): " + ", ".join(clobbered)
        )

    reference = _make_reference(paths[ref_echo], ref, n_volumes, out_dir)

    # Estimate. MCFLIRT always resamples the echo it estimates on; that output is
    # a byproduct we discard, so that every echo -- including this one -- is
    # resampled the same way, by the same call below.
    workdir = out_dir / "_estimate"
    workdir.mkdir(exist_ok=True)
    estimate = workdir / "estimate"
    run(
        [
            "mcflirt",
            "-in", paths[ref_echo],
            "-out", estimate,
            "-reffile", reference,
            "-cost", cost,
            "-dof", dof,
            "-stages", stages,
            "-mats",
            "-plots",
        ]
    )

    mat_dir = Path(f"{estimate}.mat")
    transforms = sorted(mat_dir.glob("MAT_*"))
    if len(transforms) != n_volumes:
        raise RuntimeError(
            f"mcflirt wrote {len(transforms)} transforms for {n_volumes} volumes"
        )

    # Apply the one set of transforms to every echo.
    for src, dst in zip(paths, outputs):
        run(
            ["applyxfm4D", src, reference, dst, mat_dir, "-fourdigit",
             "-interp", interp]
        )

    # Keep the estimates next to the outputs, so the workdir can go away.
    parameters = out_dir / "motion.par"
    shutil.move(f"{estimate}.par", parameters)

    transform_dir = out_dir / "transforms"
    transform_dir.mkdir(exist_ok=True)
    saved = tuple(Path(shutil.move(str(t), transform_dir / t.name)) for t in transforms)

    if not keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    return HMCResult(
        outputs=tuple(outputs),
        reference=reference,
        transforms=saved,
        parameters=parameters,
        ref_echo=ref_echo,
        method="fsl",
    )
