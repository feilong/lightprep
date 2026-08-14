"""``recon-all`` with NO interpolation of the anatomy.

FreeSurfer conforms every input to an isotropic, axis-aligned grid before it
does anything else. A T1 that is anisotropic, or prescribed with a tilt, is
therefore resampled once on the way in -- and every surface, label and
statistic thereafter describes the resampled copy rather than what was
acquired. It is a small effect and an avoidable one.

The way out is to hand FreeSurfer a volume that is *already* on its target
grid, so conform has nothing to do. The voxel array is untouched -- only
reoriented by axis permutation and padded, both exact -- and carries a **fake
affine** equal to the conform target. Conform then becomes bit-exact:
trilinear interpolation evaluated at coincident grid points is the identity.
What makes this safe rather than a lie is that the fake geometry is undone
afterwards, analytically::

    M = A_true @ inv(A_fake)

applied to surface *vertices* (exact -- transforming points interpolates
nothing) and to volume *headers* (also exact -- it moves where a grid sits in
the world without touching a voxel). :func:`lightprep.recon.correct` does that
and writes a derivative folder holding only what survives it.

What this costs
---------------
FreeSurfer requires an isotropic conformed volume, so the fake grid claims one
voxel size where the data may truly have two or three. ``M`` removes that
exactly from anything coordinate-based, but NOT from FreeSurfer's own derived
morphometry: ``?h.thickness``, surface area and the ``.stats`` files are
computed in the fake metric, and because the scale is anisotropic while
thickness is measured along the local normal, no single scalar rescales them.
Treat them as invalid and recompute from the corrected surfaces --
:func:`lightprep.recon.correct` does. Sampled timeseries are unaffected: they
need only vertex positions, which ``M`` fixes perfectly.

The size of the lie is reported as :attr:`FakeGeometry.scale`. For a 0.850 x
0.868 x 0.868 mm acquisition it is ``(1.0000, 1.0212, 1.0212)`` -- a 2.1%
anisotropic stretch. An already-isotropic input gives exactly ``(1, 1, 1)``
and the whole correction becomes the identity.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import (apply_orientation, inv_ornt_aff,
                                  io_orientation, ornt_transform)

from .._utils import run
from ..coreg.freesurfer import _freesurfer_env
from .base import CONFORM_FLAGS, FakeGeometry, ReconResult
from .freesurfer import HIRES_EXPERT, _recon_all

#: Below this voxel size ``recon-all`` needs ``-hires`` to accept the input.
SUBMM_MM = 0.999


def conform_target(t1, *, conform: str = "min", freesurfer_home=None,
                   subjects_dir=None) -> tuple[tuple[int, int, int], np.ndarray]:
    """Learn FreeSurfer's conform target grid by asking it, not guessing.

    ``mri_convert`` is run for its geometry only; the resampled data it
    produces is discarded. Asking rather than reimplementing the rule is the
    point -- ``--conform_min`` clamps, rounds and caps in ways that have
    changed between releases, and a target computed from a stale rule would
    silently reintroduce the interpolation this module exists to avoid.

    Args:
        t1: The T1-weighted volume.
        conform: Which target, a key of
            :data:`~lightprep.recon.base.CONFORM_FLAGS` (``1mm`` or ``min``).
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        subjects_dir: Only used to build the FreeSurfer environment.

    Returns:
        ``(shape, affine)`` of the grid FreeSurfer would conform to.

    Raises:
        ValueError: If ``conform`` is not a known target.
    """
    if conform not in CONFORM_FLAGS:
        raise ValueError(f"unknown conform target {conform!r}; "
                         f"expected one of {sorted(CONFORM_FLAGS)}")
    env = _freesurfer_env(freesurfer_home, subjects_dir or Path.cwd())
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "conform_probe.mgz"
        run(["mri_convert", str(t1), str(probe), CONFORM_FLAGS[conform]],
            extra_env=env)
        img = nib.load(probe)
        return tuple(int(s) for s in img.shape[:3]), img.affine.copy()


def fake_input(t1, out, *, conform: str = "min", freesurfer_home=None,
               subjects_dir=None) -> FakeGeometry:
    """Place a volume on FreeSurfer's conform grid without resampling it.

    The array is reoriented by axis permutation and flips, then padded to the
    target shape. Both are exact: no voxel value is read, interpolated or
    combined with another. The written file carries the conform target's
    affine, which is what makes conform a no-op.

    Args:
        t1: The T1-weighted volume.
        out: Where to write the faked volume.
        conform: Which conform target to imitate.
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        subjects_dir: Only used to build the FreeSurfer environment.

    Returns:
        The :class:`~lightprep.recon.base.FakeGeometry` needed to undo it.

    Raises:
        ValueError: If the conform grid is smaller than the data in some axis
            and centring it would crop away nonzero voxels. That happens with
            very high-resolution input, where FreeSurfer caps the conformed
            dimensions; there is no lossless placement, so this refuses rather
            than quietly discarding anatomy.
    """
    shape, a_fake = conform_target(t1, conform=conform,
                                   freesurfer_home=freesurfer_home,
                                   subjects_dir=subjects_dir)
    img = nib.load(t1)
    data = np.asanyarray(img.dataobj)
    a_true = img.affine.copy()

    # 1. Reorient by axis permutation/flip only -- no interpolation. The affine
    #    is transformed by the same operation so it keeps describing the data.
    ornt = ornt_transform(io_orientation(a_true), io_orientation(a_fake))
    data = apply_orientation(data, ornt)
    a_true = a_true @ inv_ornt_aff(ornt, img.shape[:3])

    # 2. Pad (or crop) to the target shape, centred. Padding adds voxels; it
    #    does not alter existing ones. Cropping would remove them, so check
    #    first that anything cropped is background.
    offsets = [(want - have) // 2 for have, want in zip(data.shape, shape)]
    src = [slice(max(0, -o), max(0, -o) + min(s, t - max(o, 0)))
           for o, s, t in zip(offsets, data.shape, shape)]
    if any(sl != slice(0, s) for sl, s in zip(src, data.shape)):
        kept = np.zeros(data.shape, dtype=bool)
        kept[tuple(src)] = True
        lost = int(np.count_nonzero(data[~kept]))
        if lost:
            raise ValueError(
                f"the {conform!r} conform grid {shape} is smaller than the "
                f"{data.shape} input on some axis, and centring it would crop "
                f"{lost} nonzero voxels. There is no lossless placement: "
                f"either downsample the input first, or accept the "
                f"interpolation and use the 'hires' method.")

    padded = np.zeros(shape, dtype=data.dtype)
    dst = [slice(max(0, o), max(0, o) + (sl.stop - sl.start))
           for o, sl in zip(offsets, src)]
    padded[tuple(dst)] = data[tuple(src)]

    # Shifting the array origin by +offset shifts the affine origin by -offset.
    shift = np.eye(4)
    shift[:3, 3] = -np.asarray(offsets, dtype=float)
    a_true_on_fake = a_true @ shift

    # 3. Write with the FAKE affine: identical to the conform target, so
    #    conform finds nothing to do.
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(padded, a_fake), out)

    M = a_true_on_fake @ np.linalg.inv(a_fake)
    return FakeGeometry(
        M=M, a_true_on_fake=a_true_on_fake, a_fake=a_fake, shape=shape,
        conform=conform,
        scale=tuple(np.linalg.norm(M[:3, :3], axis=0).tolist()),
        pad=tuple(int(o) for o in offsets),
    )


def check_conform_lossless(faked, *, conform: str = "min",
                           freesurfer_home=None,
                           subjects_dir=None) -> tuple[int, int]:
    """Confirm that conform introduces no voxel value that was not there.

    The test that matters is not "did the geometry match" but "did any new
    number appear". Interpolation between grid points produces values that
    were not in the input; an exact resampling cannot. Counting novel values
    catches a near-miss that a header comparison would pass.

    Args:
        faked: The volume written by :func:`fake_input`.
        conform: The conform target it was built for.
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        subjects_dir: Only used to build the FreeSurfer environment.

    Returns:
        ``(n_novel, n_total)`` over the nonzero voxels of the conformed
        output. ``n_novel == 0`` means conform was bit-exact.
    """
    env = _freesurfer_env(freesurfer_home, subjects_dir or Path.cwd())
    img = nib.load(faked)
    src = np.asanyarray(img.dataobj).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        as_float, conf = tmp / "in.nii.gz", tmp / "out.mgz"
        nib.save(nib.Nifti1Image(src, img.affine), as_float)
        # --no_scale 1 and float output keep mri_convert from rescaling
        # intensities, which would make every value look novel.
        run(["mri_convert", str(as_float), str(conf), CONFORM_FLAGS[conform],
             "-odt", "float", "--no_scale", "1"], extra_env=env)
        got = np.asanyarray(nib.load(conf).dataobj).astype(np.float64)
    nz = got[got != 0]
    novel = int(np.isin(nz, np.unique(src.astype(np.float64)),
                        invert=True).sum())
    return novel, int(nz.size)


def native(t1, subject: str, subjects_dir, *, conform: str = "min",
           work_dir=None, verify: bool = True, expert: str | None = HIRES_EXPERT,
           field_strength=3, freesurfer_home=None,
           max_anisotropy: float | None = None, **kwargs) -> ReconResult:
    """``recon-all`` on a volume whose conform is a bit-exact no-op.

    Args:
        t1: The T1-weighted volume. Exactly one -- see Raises.
        subject: FreeSurfer subject name.
        subjects_dir: SUBJECTS_DIR to write into.
        conform: Which conform grid to imitate. ``min`` keeps the native
            resolution; ``1mm`` fakes the default grid, which still avoids
            interpolation but discards detail by claiming a coarser voxel.
        work_dir: Where the faked input and the geometry JSON are written.
            Defaults to ``<subjects_dir>/_inputs``. Deliberately NOT the
            subject directory: ``recon-all -i`` refuses to start if that
            already exists, so nothing may be written there first.
        verify: Check that conform really is lossless, and refuse to run if it
            is not. Turning this off defeats the purpose of the method.
        expert: Expert options file contents, as for
            :func:`~lightprep.recon.freesurfer.hires`.
        field_strength: Scanner field strength; see
            :func:`~lightprep.recon.freesurfer.field_strength_args`.
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        max_anisotropy: Warn if the voxels are more anisotropic than this,
            since the fake grid then distorts FreeSurfer's view of the head by
            more than the correction can undo -- see
            :mod:`lightprep.recon.auto`. Defaults to
            :data:`~lightprep.recon.auto.MAX_ANISOTROPY`; pass
            ``float("inf")`` to check nothing.
        **kwargs: ``threads``, ``parallel``, ``force``.

    Returns:
        A :class:`~lightprep.recon.base.ReconResult` with ``interpolated=False``
        and a :class:`~lightprep.recon.base.FakeGeometry` attached. Pass it to
        :func:`lightprep.recon.correct` to get usable derivatives.

    Raises:
        ValueError: If more than one structural is given. Merging them is an
            interpolation, which this method exists to avoid.
        RuntimeError: If ``verify`` is set and conform is not bit-exact -- in
            which case the anatomy was interpolated after all and the whole
            point is lost.

    Warns:
        AnisotropyWarning: If the voxels exceed ``max_anisotropy``. The recon
            still runs -- asking for ``native`` explicitly is taken as meaning
            it -- but the warning is not suppressible by asking, because the
            distortion is there either way.
    """
    from .auto import MAX_ANISOTROPY, AnisotropyWarning, anisotropy, within

    if not isinstance(t1, (str, Path)) and len(list(t1)) > 1:
        raise ValueError(
            "the 'native' method takes one structural. Averaging several is "
            "what FreeSurfer's -motioncor stage does, and it cannot be done "
            "here: the average is an interpolation of at least one input, "
            "which is the very thing this method exists to avoid, and it "
            "would be computed in the fake anisotropic metric, where a rigid "
            "head movement between scans is not rigid. Use recon.hires (or "
            "recon.auto, which picks it) to merge them.")
    if not isinstance(t1, (str, Path)):
        t1 = list(t1)[0]

    limit = MAX_ANISOTROPY if max_anisotropy is None else max_anisotropy
    ratio = anisotropy(t1)
    if not within(ratio, limit):
        warnings.warn(
            f"running the fake-geometry recon on voxels that are "
            f"anisotropic by {ratio:.3f} (max/min): FreeSurfer will see the "
            f"head stretched by {100 * (ratio - 1):.0f}%, which affects skull "
            f"stripping, the Talairach registration and where the surfaces "
            f"are placed. The correction fixes the coordinates, not the "
            f"recon. Consider recon.hires instead.",
            AnisotropyWarning, stacklevel=2)

    subjects_dir = Path(subjects_dir).resolve()
    work = Path(work_dir) if work_dir else subjects_dir / "_inputs"
    work.mkdir(parents=True, exist_ok=True)

    faked = work / f"{subject}_faked.nii.gz"
    geom = fake_input(t1, faked, conform=conform,
                      freesurfer_home=freesurfer_home,
                      subjects_dir=subjects_dir)

    if verify:
        novel, total = check_conform_lossless(
            faked, conform=conform, freesurfer_home=freesurfer_home,
            subjects_dir=subjects_dir)
        if novel:
            raise RuntimeError(
                f"conform introduced {novel} novel values across {total} "
                f"nonzero voxels, so the anatomy was interpolated after all. "
                f"The faked input is not on FreeSurfer's grid.")

    meta = geom.to_json(work / f"{subject}_native2true.json")

    # A submillimetre grid is only accepted with -hires, whatever put it there.
    voxel = float(np.min(np.linalg.norm(geom.a_fake[:3, :3], axis=0)))
    extra = ["-hires", "-cubic"] if voxel < SUBMM_MM else ["-cubic"]

    subject_dir = _recon_all(
        faked, subject, subjects_dir, extra=extra, expert=expert,
        field_strength=field_strength, freesurfer_home=freesurfer_home,
        **kwargs)
    # Now that recon-all has made the subject directory, the geometry can live
    # beside the recon it belongs to.
    shutil.copy2(meta, subject_dir / "native2true.json")

    return ReconResult(subject=subject, subjects_dir=subjects_dir,
                       method="native", interpolated=False, conform="none",
                       geometry=geom, inputs=(faked,))
