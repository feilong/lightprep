"""A self-contained motion QC page: corrected volumes, scrubbable, over FD.

Motion correction is judged by watching it. A number says a run had a mean
framewise displacement of 2.5mm; only looking at the frames tells you whether
what remains is a head that now sits still, or a head that is still moving with
the estimator chasing it. So the page puts the corrected series next to the FD
trace and lets you drag through time on either.

It renders with NiiVue, which reads NIfTI in the browser and gives real
three-plane views with intensity windowing, rather than a strip of baked PNGs.

Comparing estimators is the other reason this exists. Pass more than one
corrected volume -- ``{"niimath": ..., "fsl": ...}`` -- and they scrub together
frame for frame, which is what shows where two methods disagree. Pass one and
the page collapses to a single viewer, which is the ordinary case.

Where the volumes come from is a choice, because a corrected run is ~100MB and
a single-file page cannot carry that at full resolution:

``embed``  (default)
    Quantised, downsampled copies inlined as base64. One file, opens from
    disk, no server, no network -- at the cost of 4mm display voxels.
``link``
    The real files at full resolution, staged next to the report and served
    from it -- ``serve(report)`` starts the server and prints the URL. The
    page is ~2MB and nothing is quantised. Staging uses hard links where the
    filesystem allows, so it costs no disk.
``pick``
    The same, without a server, for when you cannot run one -- viewing on
    another machine, say. The page opens empty and you hand it each file.
    ``link`` is preferable whenever serving is possible: the tool already
    knows which files it wants, so it should not ask.
"""

from __future__ import annotations

import base64
import gc
import gzip
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .._utils import load_trace, run
from ..hmc.moco import relative_displacement, relative_motion

#: Radius used to convert rotations to surface displacement in FD, following
#: Power et al. (2012). 50mm is their value: roughly the cortex-to-centre
#: distance, so a radian of rotation moves cortex by 50mm.
FD_RADIUS_MM = 50.0

#: In-plane downsampling for the embedded copies. The volumes are for looking
#: at, not measuring: a corrected run is ~100MB, which no single-file page can
#: carry, and gross displacement is perfectly visible at half resolution.
DEFAULT_DOWNSAMPLE = 2

#: Percentiles the display intensity is clipped to before quantisation.
DEFAULT_CLIP = (0.5, 99.5)


def framewise_displacement(parameters, radius: float = FD_RADIUS_MM) -> np.ndarray:
    """Power framewise displacement from a six-column motion parameter file.

    Args:
        parameters: Path to a motion trace -- either twin, the float64 ``.npy``
            being preferred -- or an ``(n_frames, 6)`` array in MCFLIRT's
            convention: three rotations in radians, then three translations in
            mm.
        radius: Sphere radius converting rotation to displacement.

    Returns:
        ``(n_frames - 1,)`` displacement in mm. The first frame has no
        predecessor and so no FD.

    Raises:
        ValueError: If the parameters are not six columns.
    """
    m = np.asarray(parameters if not isinstance(parameters, (str, Path))
                   else load_trace(parameters), dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 6:
        raise ValueError(f"expected (n_frames, 6) motion parameters, got {m.shape}")
    d = np.abs(np.diff(m, axis=0))
    return d[:, 3:6].sum(1) + radius * d[:, :3].sum(1)


def relative_fd(volume, radius: float = FD_RADIUS_MM) -> np.ndarray:
    """Framewise displacement of a raw series, with nothing corrected first.

    :func:`framewise_displacement` differences a motion trace, so it can only
    be computed after head motion correction has run and inherits whatever that
    run's reference volume did. This measures the same quantity directly --
    each volume fitted onto its predecessor, by
    :func:`lightprep.hmc.relative_motion` -- which needs no reference and no
    corrected image, and so is available *before* the pipeline commits to
    anything.

    That makes it the right trace for deciding what to do with a run: which
    frames to flag, and which volume to aim the correction at (see
    :func:`lightprep.hmc.best_reference`).

    Args:
        volume: A 4D series.
        radius: Sphere radius converting rotation to displacement.

    Returns:
        ``(n_frames,)`` mm; element 0 is zero, having no predecessor.

    Raises:
        RuntimeError: If niimath cannot measure relative motion.
    """
    return relative_displacement(relative_motion(volume), radius=radius)


#: Displacement above which Power et al. (2012) call a frame contaminated.
#: A fixed millimetre threshold rather than a spread rule, because unlike a
#: correlation this quantity has a physical meaning: half a millimetre is half
#: a millimetre whether the rest of the run was still or not.
FD_THRESHOLD_MM = 0.5


def fd_outliers(displacement, threshold: float = FD_THRESHOLD_MM):
    """Frames the head was moving through, by displacement alone.

    Args:
        displacement: ``(n_frames,)`` mm, from :func:`relative_fd` or
            :func:`framewise_displacement`.
        threshold: Millimetres. Frames at or above it are flagged.

    Returns:
        ``(indices, fraction)`` -- the flagged frames and what share of the run
        they are. A run where that fraction is large is not one to scrub; it is
        one to reconsider.

    Note:
        This asks whether a frame was *acquired* while the head moved, which is
        not the same question as :func:`cdtm`'s -- whether the frame *looks
        like* the run. A frame can move and still resemble the run (a small
        translation, well corrected), and a frame can sit perfectly still and
        not resemble it at all (a spike, a dropout, a stretch spent in a
        displaced position). Where both are available, flag on either.
    """
    d = np.asarray(displacement, dtype=np.float64).ravel()
    idx = np.flatnonzero(d >= threshold)
    return idx, float(idx.size) / max(d.size, 1)


def dvars(volume, mask=None, standardize: bool = True,
          sample_frames: int = 20) -> np.ndarray:
    """DVARS: the RMS intensity change from one frame to the next.

    FD says how far the head moved; DVARS says how much the picture changed.
    They are complementary because either can move without the other -- a
    spike from a gradient glitch shifts DVARS with no head motion, and a slow
    drift moves the head with little frame-to-frame intensity change. Power et
    al. (2012) introduced them together for that reason.

    ``standardize`` divides by the run's own robust temporal standard
    deviation, estimated per voxel as ``IQR / 1.349`` and averaged over the
    mask, following the convention fMRIPrep reports. A value near 1 is then
    "as much change as this run's noise usually produces", and it is
    comparable between runs and subjects, which raw intensity units are not.

    Frames are streamed one at a time rather than loaded as a block: a 4D run
    is ~440MB in float32 and twice that in the float64 the arithmetic wants.

    Args:
        volume: 4D image, or a path to one.
        mask: Boolean array or path to a mask. Defaults to voxels above the
            70th percentile of the mean image, which is a brain.
        standardize: Divide by the robust temporal SD (see above).
        sample_frames: How many evenly spaced frames estimate the mean image
            and the per-voxel SD. The whole run is not needed for either.

    Returns:
        ``(n_frames - 1,)``. Like FD, the first frame has no predecessor.

    Raises:
        ValueError: If the image is not 4D or has fewer than two frames.
    """
    img = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    if img.ndim != 4 or img.shape[3] < 2:
        raise ValueError(f"need a 4D image with >=2 frames, got shape {img.shape}")
    n = img.shape[3]

    idx = np.unique(np.linspace(0, n - 1, min(sample_frames, n)).astype(int))
    sample = np.stack([np.asanyarray(img.dataobj[..., i], dtype=np.float64)
                       for i in idx], axis=-1)

    if mask is None:
        mean_img = sample.mean(-1)
        keep = mean_img > np.percentile(mean_img, 70)
    else:
        keep = (np.asarray(mask, dtype=bool) if not isinstance(mask, (str, Path))
                else np.asanyarray(nib.load(str(mask)).dataobj) > 0)
    if not keep.any():
        raise ValueError("the mask selects no voxels")

    scale = 1.0
    if standardize:
        inside = sample[keep]                     # (n_vox, n_sample)
        q75, q25 = np.percentile(inside, [75, 25], axis=1)
        sd = (q75 - q25) / 1.349
        scale = float(np.mean(sd[sd > 0])) if np.any(sd > 0) else 1.0

    out = np.empty(n - 1, dtype=np.float64)
    prev = np.asanyarray(img.dataobj[..., 0], dtype=np.float64)[keep]
    for t in range(1, n):
        cur = np.asanyarray(img.dataobj[..., t], dtype=np.float64)[keep]
        out[t - 1] = np.sqrt(np.mean((cur - prev) ** 2))
        prev = cur
    return out / scale if scale else out


#: Seconds a MindGrab strip may take before it is treated as wedged. A real
#: strip is 18-24s on a 2mm EPI. Kept as a backstop rather than a workaround:
#: the hangs that motivated it were brainchop's first-run "Optimize now? [y/n]"
#: prompt blocking on an inherited stdin, which run() now closes.
STRIP_TIMEOUT_S = 120.0

#: How many times a wedged strip is retried. It has always cleared on the next
#: attempt when tried by hand.
STRIP_RETRIES = 2


def brain_mask(volume, out=None, brainchop: str = "brainchop",
               sample_frames: int = 20) -> np.ndarray:
    """A brain mask for a 4D series, from MindGrab on its temporal mean.

    A single frame is too noisy to strip well, so the mean of evenly spaced
    frames is stripped instead. MindGrab is trained across modalities and
    handles EPI, which intensity-threshold methods do poorly -- an EPI has no
    skull to threshold away, but it does have neck, eyes and ghosts, and those
    are exactly what a naive mask lets in.

    Args:
        volume: 4D image or path to one.
        out: Where to keep the mask. Temporary if omitted.
        brainchop: The brainchop executable.
        sample_frames: How many frames the mean is taken over.

    Returns:
        Boolean array on the input grid.

    Raises:
        RuntimeError: If the mask does not come back on the input grid, which
            would silently mask the wrong voxels.
    """
    import tempfile

    img = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    n = img.shape[3] if img.ndim == 4 else 1
    idx = np.unique(np.linspace(0, n - 1, min(sample_frames, n)).astype(int))
    if img.ndim == 4:
        # trimmed, for the same reason cdtm's first reference is: one badly
        # corrupted frame among the samples would blur what gets stripped
        sample = np.stack([np.asanyarray(img.dataobj[..., i], dtype=np.float64)
                           for i in idx], axis=-1)
        mean_img = trimmed_mean(sample, CDTM_TRIM, axis=-1)
    else:
        sample = None
        mean_img = np.asanyarray(img.dataobj, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = tmp / "mean.nii.gz"
        nib.Nifti1Image(mean_img.astype(np.float32), img.affine).to_filename(ref)
        mask_path = Path(out) if out else tmp / "mask.nii.gz"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        # The mean is on disk; the parent has no reason to keep holding it.
        del sample, mean_img
        gc.collect()

        # --inverse-conform: brainchop conforms internally, and without this
        # the mask comes back on its own grid rather than the data's.
        # A strip that has not started moving in five minutes is not going to;
        # kill it and try again rather than stall a batch overnight.
        run([brainchop, str(ref), "-m", "mindgrab", "--mask", str(mask_path),
             "--inverse-conform", "-o", str(tmp / "stripped.nii.gz")],
            timeout=STRIP_TIMEOUT_S, retries=STRIP_RETRIES)
        m = nib.load(str(mask_path))
        arr = np.asanyarray(m.dataobj) > 0
        if arr.shape != img.shape[:3]:
            raise RuntimeError(
                f"MindGrab returned a {arr.shape} mask for {img.shape[:3]} "
                f"data; it did not come back on the input grid")
        if out:
            nib.Nifti1Image(arr.astype(np.uint8), img.affine).to_filename(out)
    return arr


def frame_distance(volume, mask=None, brainchop: str = "brainchop") -> np.ndarray:
    """Correlation distance of every frame from the run's average, in the brain.

    Where FD and DVARS measure change from one frame to the next, this
    measures each frame against the run as a whole. The difference matters for
    a head that moves and then stays put: DVARS spikes once at the step and
    returns to baseline, reporting the run as fine again, while this stays
    depressed for as long as the head is somewhere new. Sustained
    displacement and single-frame corruption look different here, and the same
    under a difference measure.

    Args:
        volume: 4D image or path to one.
        mask: Boolean array or mask path. Defaults to :func:`brain_mask`,
            i.e. MindGrab on the temporal mean.
        brainchop: Passed to :func:`brain_mask` when no mask is given.

    Returns:
        ``(n_frames,)`` of ``1 - r``. Every QC series in this module points the
        same way -- larger is worse -- so that none of them can be fed to
        something expecting another and silently invert it. Unlike FD and
        DVARS this has a value for every frame, the first included, because it
        references the mean rather than a predecessor.

        This is :func:`cdtm` without the refinement; use that unless you
        specifically want a single pass against the raw mean.

    Raises:
        ValueError: If the image is not 4D.
    """
    img = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    if img.ndim != 4:
        raise ValueError(f"need a 4D image, got shape {img.shape}")
    if mask is None:
        keep = brain_mask(img, brainchop=brainchop)
    elif isinstance(mask, (str, Path)):
        keep = np.asanyarray(nib.load(str(mask)).dataobj) > 0
    else:
        keep = np.asarray(mask, dtype=bool)
    if not keep.any():
        raise ValueError("the mask selects no voxels")

    n = img.shape[3]
    # Two passes, one frame at a time: the average must exist before anything
    # can be compared to it, and the run does not fit in memory twice.
    total = np.zeros(int(keep.sum()), dtype=np.float64)
    for t in range(n):
        total += np.asanyarray(img.dataobj[..., t], dtype=np.float64)[keep]
    avg = total / n
    avg_c = avg - avg.mean()
    avg_n = np.sqrt((avg_c ** 2).sum())

    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        f = np.asanyarray(img.dataobj[..., t], dtype=np.float64)[keep]
        f_c = f - f.mean()
        denom = np.sqrt((f_c ** 2).sum()) * avg_n
        out[t] = float((f_c * avg_c).sum() / denom) if denom else np.nan
    return 1.0 - out


#: AFNI's rule for calling a volume suspect: outside median +/- 3.5 MAD of the
#: run's own quality indices. Under normality that is about 1% of volumes.
AQI_MAD = 3.5


def quality_index(volume, mask=None, method: str = "spearman",
                  reference: str = "median", brainchop: str = "brainchop"):
    """AFNI's per-volume quality index: 1 - correlation with the median volume.

    This is ``3dTqual``, which MRIQC reports as AQI. Each frame is correlated
    against the run's median volume inside the brain and the distance ``1 - r``
    is returned, so small is good.

    Both defaults are deliberate and both differ from the obvious choices:

    * **median, not mean.** The reference is built from the same frames being
      judged, so a corrupted frame pulls a mean reference towards itself and
      partly excuses its own corruption. A median barely moves.
    * **Spearman, not Pearson.** Rank correlation is unmoved by the intensity
      scaling that motion and spin-history introduce, which a Pearson
      correlation would read as a change in the picture.

    The mask is the one departure from AFNI, and an improvement: ``3dTqual``
    thresholds intensity (``-autoclip``), which on EPI keeps neck, eyes and
    ghosts because there is no skull to threshold away. :func:`brain_mask`
    asks MindGrab instead.

    Args:
        volume: 4D image or path to one.
        mask: Boolean array or mask path. Defaults to :func:`brain_mask`.
        method: ``spearman`` (AFNI's default) or ``pearson``.
        reference: ``median`` (AFNI's default) or ``mean``.
        brainchop: Passed to :func:`brain_mask` when no mask is given.

    Returns:
        ``(n_frames,)`` of ``1 - r``. One value per frame, the first included,
        since each is judged against the run rather than its predecessor.

    Raises:
        ValueError: If the image is not 4D, or an option is unknown.
    """
    from scipy.stats import rankdata

    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be spearman or pearson, got {method!r}")
    if reference not in ("median", "mean"):
        raise ValueError(f"reference must be median or mean, got {reference!r}")

    img = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    if img.ndim != 4:
        raise ValueError(f"need a 4D image, got shape {img.shape}")
    if mask is None:
        keep = brain_mask(img, brainchop=brainchop)
    elif isinstance(mask, (str, Path)):
        keep = np.asanyarray(nib.load(str(mask)).dataobj) > 0
    else:
        keep = np.asarray(mask, dtype=bool)
    if not keep.any():
        raise ValueError("the mask selects no voxels")

    # Masked, this is a few hundred MB rather than the whole run, and a median
    # needs every frame at once anyway.
    X = np.stack([np.asanyarray(img.dataobj[..., t], dtype=np.float64)[keep]
                  for t in range(img.shape[3])], axis=1)      # (n_vox, n_time)
    ref = np.median(X, axis=1) if reference == "median" else X.mean(axis=1)

    if method == "spearman":
        X = np.apply_along_axis(rankdata, 0, X)
        ref = rankdata(ref)

    Xc = X - X.mean(axis=0, keepdims=True)
    rc = ref - ref.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((rc ** 2).sum())
    r = np.divide((Xc * rc[:, None]).sum(axis=0), denom,
                  out=np.full(X.shape[1], np.nan), where=denom > 0)
    return 1.0 - r


def aqi_outliers(index, n_mad: float = AQI_MAD):
    """Frames AFNI would call suspect: outside median +/- n_mad MAD.

    Returns ``(indices, lo, hi)``. AFNI prints these bounds alongside the
    index for exactly this purpose; under normality about 1% of volumes fall
    outside them.
    """
    a = np.asarray(index, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    lo, hi = med - n_mad * mad, med + n_mad * mad
    return np.flatnonzero((a < lo) | (a > hi)), lo, hi


#: A frame is an outlier when its correlation distance exceeds this multiple
#: of the typical distance among the frames still kept, where typical is the
#: trimmed mean.
#:
#: Deliberately not a MAD rule. MAD collapses when a run is uniformly good, so
#: ordinary variation clears any fixed multiple of it: measured here, AFNI's
#: 3.5-MAD rule flagged 25% of a clean run and 17% of a badly corrupted one --
#: the wrong way round. A ratio against the median is scale-free and says
#: something a reader can check: this frame is three times as unlike the run
#: as a typical frame is.
CDTM_RATIO = 3.0

#: Fraction of frames discarded when building the FIRST reference, split
#: between the tails: 0.5 keeps the middle half, per voxel. The first pass is
#: the only one whose reference has not been cleaned, so it is where a bad
#: frame does the most damage -- it pulls the mean towards itself and then
#: scores well against it. A trimmed mean resists up to 25% contamination in
#: either tail, which is more than any run worth keeping should have.
CDTM_TRIM = 0.5


def trimmed_mean(x: np.ndarray, trim: float = CDTM_TRIM, axis: int = -1):
    """Mean of the middle ``1 - trim`` of the values, along one axis.

    ``trim=0.5`` gives the interquartile mean. ``trim=0`` is the plain mean.
    """
    if not 0.0 <= trim < 1.0:
        raise ValueError(f"trim must be in [0, 1), got {trim}")
    if trim == 0.0:
        return x.mean(axis=axis)
    n = x.shape[axis]
    cut = int(np.floor(n * trim / 2.0))
    if n - 2 * cut < 1:
        cut = (n - 1) // 2
    lo, hi = cut, n - cut
    return np.sort(x, axis=axis).take(range(lo, hi), axis=axis).mean(axis=axis)


@dataclass(frozen=True)
class CDTMResult:
    """Correlation distance to mean, after iterative refinement.

    Attributes:
        values: ``(n_frames,)`` final CDTM, computed against the refined mean.
        outliers: Frame indices excluded from the reference.
        keep: Boolean ``(n_frames,)``, the complement of ``outliers``.
        threshold: The cut the last iteration applied.
        history: ``[(n_kept, trimmed_mean, threshold, n_changed), ...]``.
        reference: The final reference image, the mean of the kept frames, on
            the full grid with zeros outside the mask.
        mask: The brain mask the last iteration used.
        converged: Whether it stopped finding outliers rather than running out
            of iterations.
    """

    values: np.ndarray
    outliers: np.ndarray
    keep: np.ndarray
    threshold: float
    history: tuple
    reference: np.ndarray
    mask: np.ndarray
    converged: bool



def masked_series(volume, mask) -> np.ndarray:
    """``(n_voxels, n_frames)`` float64 of a series inside a mask.

    Read once so a selection can be iterated against it without touching disk
    again -- the frames never change, only which of them the reference is built
    from.
    """
    img = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    if img.ndim != 4:
        raise ValueError(f"need a 4D image, got shape {img.shape}")
    if isinstance(mask, (str, Path)):
        mask = np.asanyarray(nib.load(str(mask)).dataobj) > 0
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != img.shape[:3]:
        raise ValueError(f"mask is {mask.shape} but data is {img.shape[:3]}")
    return np.stack([np.asanyarray(img.dataobj[..., t], dtype=np.float64)[mask]
                     for t in range(img.shape[3])], axis=1)


def correlation_distance(series, keep=None, trim: float = CDTM_TRIM) -> np.ndarray:
    """Each frame's correlation distance to the mean of the kept frames.

    The scoring half of :func:`cdtm`, with the choosing half removed. cdtm
    decides for itself which frames build its reference; this takes that
    decision from the caller, so a frame can be kept out of the yardstick for
    reasons correlation cannot see -- it was acquired while the head moved, and
    is smeared or mispositioned however much it still resembles the run.

    Every frame is scored, including excluded ones: they are judged, not
    forgotten.

    Args:
        series: ``(n_voxels, n_frames)`` from :func:`masked_series`.
        keep: Boolean over frames, the reference set. ``None`` uses the
            per-voxel trimmed mean of everything, which is the right start when
            nothing has been excluded yet and the frames that need finding are
            still in the average.
        trim: Fraction trimmed when ``keep`` is None.

    Returns:
        ``(n_frames,)`` correlation distance; bigger is less like the reference.
    """
    X = np.asarray(series, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (n_voxels, n_frames), got {X.shape}")
    n = X.shape[1]
    if keep is None:
        avg = trimmed_mean(X, trim, axis=1)
    else:
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (n,):
            raise ValueError(f"keep is {keep.shape}, expected ({n},)")
        if keep.sum() < 2:
            raise ValueError("the reference needs at least 2 frames")
        avg = X[:, keep].mean(axis=1)
    ac = avg - avg.mean()
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((ac ** 2).sum())
    return 1.0 - np.divide((Xc * ac[:, None]).sum(axis=0), denom,
                           out=np.full(n, np.nan), where=denom > 0)


def cdtm(volume, mask=None, *, threshold: float | None = None,
         ratio: float = CDTM_RATIO, trim: float = CDTM_TRIM,
         max_iter: int = 20, remask: bool = False, save_reference=None,
         brainchop: str = "brainchop") -> CDTMResult:
    """Correlation distance to the mean image, with the mean refined iteratively.

    A frame is scored by how unlike the run's average it is. The difficulty is
    that the average is built from the same frames, so a corrupted one drags
    the reference toward itself and partly excuses its own corruption -- the
    reason AFNI's 3dTqual uses a median instead.

    This takes the other route: keep the mean, but stop letting bad frames
    into it. The first reference is a per-voxel trimmed mean, so the pass that
    has to find the outliers is not itself distorted by them. Each pass then
    drops the outliers found so far, rebuilds the average (and, unless
    ``remask`` is off, the brain mask, since a sharper average strips better)
    and rescores every frame against it. Frames already
    excluded keep being scored -- they are judged, not forgotten -- so a frame
    dropped early can be seen to be far worse than the survivors.

    It stops when the set of outliers stops changing -- a fixed point, not a
    one-way filter. Frames are re-judged every pass against the current
    reference, so a frame condemned by a contaminated early reference can be
    reinstated once the reference improves. That is the difference between
    iterating and simply removing in stages, and it is why an early mistake
    is not permanent.

    Args:
        volume: 4D image or path to one.
        mask: Initial mask; MindGrab on the temporal mean by default.
        threshold: An absolute cut on the distance. Fixed for a dataset, it
            makes runs directly comparable -- but it is tied to the
            acquisition, since voxel size, coverage and how much anatomy
            dominates the correlation all move the scale. ``None`` uses
            ``ratio`` instead, which is scale-free but relative to each run.
        ratio: Used when ``threshold`` is None: a frame is an outlier when its
            distance exceeds this multiple of the trimmed mean distance among
            the frames still kept. See :data:`CDTM_RATIO`.
        save_reference: Write the final reference image here.
        trim: Fraction trimmed, used in both places the middle 50% is wanted:
            building the FIRST reference image, per voxel across frames, and
            summarising the survivors' distances into the yardstick the ratio
            multiplies. See :data:`CDTM_TRIM`. Later reference images use a
            plain mean of the kept frames, which is unbiased once the outliers
            are out and wastes none of the good data.
        max_iter: Give up after this many passes.
        remask: Recompute the brain mask from each refined average.
        brainchop: Passed to :func:`brain_mask`.

    Returns:
        A :class:`CDTMResult`.

    Raises:
        ValueError: If the image is not 4D, or every frame would be dropped.
    """
    img = volume if hasattr(volume, "dataobj") else nib.load(str(volume))
    if img.ndim != 4:
        raise ValueError(f"need a 4D image, got shape {img.shape}")
    n = img.shape[3]

    if mask is None:
        keep_vox = brain_mask(img, brainchop=brainchop)
    elif isinstance(mask, (str, Path)):
        keep_vox = np.asanyarray(nib.load(str(mask)).dataobj) > 0
    else:
        keep_vox = np.asarray(mask, dtype=bool)

    keep = np.ones(n, dtype=bool)
    history, values, thr, converged = [], None, float("nan"), False
    avg = None
    seen = set()

    for it in range(max_iter):
        X = np.stack([np.asanyarray(img.dataobj[..., t], dtype=np.float64)[keep_vox]
                      for t in range(n)], axis=1)
        # First pass: the central 50% per voxel, because nothing has been
        # excluded yet and the frames that need finding are still in the
        # average. After that the outliers are out and a plain mean of the
        # survivors is unbiased and wastes none of the good data.
        avg = trimmed_mean(X, trim, axis=1) if it == 0 else X[:, keep].mean(axis=1)
        ac = avg - avg.mean()
        Xc = X - X.mean(axis=0, keepdims=True)
        denom = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((ac ** 2).sum())
        values = 1.0 - np.divide((Xc * ac[:, None]).sum(axis=0), denom,
                                 out=np.full(n, np.nan), where=denom > 0)

        centre = float(trimmed_mean(values[keep], trim))
        thr = float(threshold) if threshold is not None else ratio * centre
        # Every frame is re-judged, not just the survivors: with the reference
        # improving each pass, a frame condemned early can come back.
        new_keep = np.nan_to_num(values, nan=np.inf) <= thr
        history.append((int(keep.sum()), centre, thr,
                        int((new_keep != keep).sum())))
        if new_keep.sum() < 2:
            raise ValueError(
                f"only {int(new_keep.sum())} frames fall under {thr:.5g}; the "
                f"threshold is too tight for this run, or it has no usable "
                f"frames")
        if np.array_equal(new_keep, keep):
            converged = True
            break
        signature = new_keep.tobytes()
        if signature in seen:
            # Two states alternating: report the current one rather than spin.
            keep = new_keep
            break
        seen.add(signature)
        keep = new_keep

        if remask:
            # A mean over cleaner frames is sharper, so the strip improves too.
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                ref = Path(tmp) / "mean.nii.gz"
                full = np.zeros(img.shape[:3], dtype=np.float64)
                full[keep_vox] = X[:, keep].mean(axis=1)
                nib.Nifti1Image(full.astype(np.float32), img.affine).to_filename(ref)
                keep_vox = brain_mask(ref, brainchop=brainchop)

    reference = np.zeros(img.shape[:3], dtype=np.float64)
    reference[keep_vox] = X[:, keep].mean(axis=1)
    if save_reference is not None:
        Path(save_reference).parent.mkdir(parents=True, exist_ok=True)
        nib.Nifti1Image(reference.astype(np.float32),
                        img.affine).to_filename(str(save_reference))

    return CDTMResult(values=values, outliers=np.flatnonzero(~keep), keep=keep,
                      threshold=thr, history=tuple(history),
                      reference=reference, mask=keep_vox, converged=converged)


def find_niivue() -> Path | None:
    """A NiiVue UMD bundle on this machine, if one is installed.

    nilearn ships one, which spares this module a vendored copy and a network
    fetch. Returns None if nothing is found, in which case the caller must say
    where to get it.
    """
    try:
        import nilearn
    except ImportError:
        return None
    candidate = Path(nilearn.__file__).parent / "_assets" / "js" / "niivue.umd.js"
    return candidate if candidate.exists() else None


def _display_volume(src, downsample: int, clip) -> bytes:
    """A small uint8 copy of a 4D series, gzipped, for embedding.

    Quantisation is done once over the whole run rather than per frame, so a
    frame that darkens on screen darkened in the data -- per-frame windowing
    would hide exactly the intensity steps that mark a swallow or a spike.
    """
    img = nib.load(str(src))
    data = img.get_fdata(dtype=np.float64)
    if data.ndim == 3:
        data = data[..., None]
    step = max(1, int(downsample))
    data = data[::step, ::step, ::step]

    lo, hi = np.percentile(data[np.isfinite(data)], clip)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((data - lo) / (hi - lo), 0, 1)
    out = np.rint(scaled * 255).astype(np.uint8)

    affine = img.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag([step, step, step])
    small = nib.Nifti1Image(out, affine)
    small.header.set_xyzt_units(*img.header.get_xyzt_units())
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fh:
        fh.write(small.to_bytes())
    return buf.getvalue()


def _relative_url(target: Path, start: Path) -> str:
    """A URL from the report to a file, relative where possible.

    Relative keeps the pair movable together; an absolute file:// URL would
    break the moment the directory was copied anywhere else.
    """
    import os
    from urllib.request import pathname2url
    try:
        rel = os.path.relpath(target.resolve(), start.resolve())
    except ValueError:                          # different drive, Windows
        return "file://" + pathname2url(str(target.resolve()))
    return pathname2url(rel)


#: Fallback mesh colours, for surfaces that are not a white/pial pair.
_MESH_COLOURS = [[0, 229, 255, 255], [124, 252, 0, 255], [255, 128, 255, 255],
                 [255, 165, 0, 255]]

#: Colourmaps for contour overlays that are not a white/pial pair.
_OVERLAY_COLOURMAPS = ["blue", "green", "violet", "cool"]


def _stage(src: Path, out_html: Path, label: str) -> str:
    """Put a volume beside the report and return its relative URL.

    Hard link first, symlink next, copy last. A 100MB run should not be
    duplicated just to be served, but it must be reachable from one directory
    so that serving the report does not mean serving the whole filesystem.
    """
    room = out_html.parent / f"{out_html.stem}_data"
    room.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    dst = room / f"{safe}__{src.name}"
    if not dst.exists():
        try:
            dst.hardlink_to(src.resolve())
        except (OSError, AttributeError):
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
    from urllib.request import pathname2url
    return pathname2url(f"{room.name}/{dst.name}")


def serve(report, port: int = 0, open_browser: bool = True):
    """Serve a ``link`` report and print its URL.

    Browsers refuse cross-origin reads from ``file://``, so a linked report
    needs HTTP. Only the report's own directory is exposed -- which is why
    ``stage`` puts the volumes there rather than serving whatever common
    ancestor the originals happened to share.

    Args:
        report: The HTML written by :func:`motion_report`.
        port: TCP port; 0 picks a free one.
        open_browser: Open the page once the server is up.

    Blocks until interrupted.
    """
    import functools
    import http.server
    import socketserver
    import webbrowser

    report = Path(report).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(report.parent))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/{report.name}"
        print(f"serving {report.parent}\n{url}\nCtrl-C to stop", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def _as_traces(fd) -> dict:
    """Normalise the fd argument to {label: list of floats}."""
    if fd is None:
        return {}
    if isinstance(fd, dict):
        items = fd.items()
    elif isinstance(fd, (str, Path)) or np.ndim(fd) >= 1:
        items = [("FD", fd)]
    else:
        raise TypeError(f"cannot read FD from {type(fd)}")
    out = {}
    for label, series in items:
        if isinstance(series, (str, Path)):    # a trace on disk
            series = load_trace(series)
        arr = np.asarray(series, dtype=np.float64)
        if arr.ndim == 2:                      # six columns, not FD yet
            arr = framewise_displacement(arr)
        if arr.ndim != 1:
            raise ValueError(
                f"FD series {label!r} has shape {arr.shape}; expected a 1D "
                f"trace or an (n_frames, 6) motion parameter array")
        out[str(label)] = [float(v) for v in arr]
    return out


def motion_report(volumes, out_html, *, fd=None, title: str = "motion QC",
                  subtitle: str = "", dvars_traces=None, distance=None,
                  outliers=None, distance_max: float | None = None,
                  panels=None,
                  surfaces=None, mesh_thickness_mm: float = 1.0,
                  contours=None, niivue_js=None,
                  source: str = "embed", stage: bool = True,
                  downsample: int = DEFAULT_DOWNSAMPLE, clip=DEFAULT_CLIP,
                  fd_threshold: float = 0.5) -> Path:
    """Write a self-contained motion QC page.

    Args:
        volumes: ``{label: path}`` of motion-corrected 4D series, one per
            method being shown. A single path or a one-entry mapping gives the
            ordinary single-viewer page.
        out_html: Where to write the report.
        fd: ``{label: series}``, a single series, or a path to a ``motion.par``
            (an ``(n, 6)`` array is converted with
            :func:`framewise_displacement`). Optional.
        distance: Correlation distance per frame, from :func:`cdtm` (its
            ``.values``) or :func:`frame_distance`. Drawn in a third panel. It
            has one value per frame where FD and DVARS have one per
            transition; the page aligns them.
        distance_max: Fixed y-axis top for the distance panel, overriding the
            usual scaling. It used to default to 0.05, because the axis was
            once set by the maximum and one corrupted frame at 0.2 would
            flatten the 0.01 range everything else lives in. The panel now
            scales to a high quantile, which fixes that at the source, and the
            fixed ceiling had become the worst offender on the page -- across a
            59-run cohort the trace occupied a median 10% of its height, and
            under 20% in 46 of them. Left None unless a run really must be read
            against a fixed scale.
        surfaces: ``{label: path}`` of GIfTI meshes to draw over every viewer,
            or a list of paths. In the 2D slice views NiiVue draws a mesh as
            its intersection with the slice, so a white and a pial surface come
            out as contours -- which is the only way to see whether the
            coregistration actually landed, rather than trusting a cost number.
            The meshes must already be in the *volumes'* world space: a surface
            still in anatomical coordinates will draw, confidently, in the
            wrong place. Labels containing "pial" and "white" pick the
            conventional colours, anything else cycles.
        panels: Extra traces to draw, ``{title: series}`` or
            ``{title: {label: series}}`` for several on one panel. A value may
            also be ``(series, threshold, ymax)`` to set the dashed line and fix
            the axis. Drawn after FD, DVARS and the distance panel, each with a
            checkbox: a page carrying five measures is unreadable all at once
            and useless with any of them missing, so which are shown is the
            reader's choice rather than the writer's.
        contours: ``{label: path}`` of binary volumes drawn as coloured
            overlays on every viewer, on the volumes' own grid. This is the way
            to get a real surface contour: NiiVue has no mesh-plane
            intersection mode -- ``surfaces`` clips a slab of shaded 3D
            geometry, which reads as a render rather than an outline -- so a
            one-voxel shell rasterised from the surface is drawn instead, and
            the line you see in a slice is exactly where the surface crosses
            it. Labels containing "pial"/"white" pick the conventional colours.
        mesh_thickness_mm: How far from the slice plane, in millimetres, mesh
            geometry is still drawn. NiiVue's default is infinity, which paints
            the whole surface onto every slice and looks like a 3D render
            rather than a contour. About a voxel gives a thin outline; large
            enough that triangles crossing the plane are not missed, small
            enough that it reads as a line.
        outliers: Frame indices to call out in the frame readout, typically
            the union of whatever the caller counts as bad. It no longer draws
            the bars: each panel shades where its own trace crosses its own
            threshold, so a bar means *this* measure objected. A frame in
            ``outliers`` is still labelled OUTLIER as you scrub past it, which
            is what says a frame is bad by some measure not currently shown.
        dvars_traces: The same shapes, drawn in a second panel below FD.
            Separate rather than overlaid because DVARS is in standardised
            intensity units and FD in millimetres; one axis would make the
            comparison of their shapes a comparison of their scales.
        title: Page heading.
        subtitle: Smaller line under it -- subject and run, typically.
        niivue_js: NiiVue UMD bundle to inline. Defaults to
            :func:`find_niivue`.
        source: Where the viewer gets its voxels -- ``embed`` (default),
            ``link`` or ``pick``. See the module docstring; ``downsample`` and
            ``clip`` apply only to ``embed``.
        stage: For ``link``, put the volumes in a folder beside the report so
            the pair is self-contained and a server needs to expose only that
            folder. Hard-linked where possible, so it costs no disk. Turn it
            off to link the originals where they lie.
        downsample: In-plane/through-plane step for the embedded copies.
        clip: Percentiles to window the display intensities to.
        fd_threshold: Drawn as a line on the trace, and used for the
            percent-over-threshold readout.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If a volume or the NiiVue bundle is missing.
    """
    if isinstance(volumes, (str, Path)):
        volumes = {"corrected": volumes}
    volumes = {str(k): Path(v) for k, v in dict(volumes).items()}
    for label, path in volumes.items():
        if not path.exists():
            raise FileNotFoundError(f"volume for {label!r} not found: {path}")

    if surfaces is None:
        surfaces = {}
    elif isinstance(surfaces, (str, Path)):
        surfaces = {Path(surfaces).stem: surfaces}
    elif not hasattr(surfaces, "items"):
        surfaces = {Path(p).stem: p for p in surfaces}
    surfaces = {str(k): Path(v) for k, v in dict(surfaces).items()}
    for label, path in surfaces.items():
        if not path.exists():
            raise FileNotFoundError(f"surface for {label!r} not found: {path}")

    if contours is None:
        contours = {}
    elif isinstance(contours, (str, Path)):
        contours = {Path(contours).stem: contours}
    elif not hasattr(contours, "items"):
        contours = {Path(p).stem: p for p in contours}
    contours = {str(k): Path(v) for k, v in dict(contours).items()}
    for label, path in contours.items():
        if not path.exists():
            raise FileNotFoundError(f"contour for {label!r} not found: {path}")

    bundle = Path(niivue_js) if niivue_js else find_niivue()
    if bundle is None or not Path(bundle).exists():
        raise FileNotFoundError(
            "no NiiVue bundle found. Install nilearn, which ships one, or pass "
            "niivue_js=<path to niivue.umd.js>.")

    if source not in ("embed", "link", "pick"):
        raise ValueError(f"source must be embed, link or pick, got {source!r}")

    out_html = Path(out_html)
    payload, n_frames = {}, 0
    for label, path in volumes.items():
        img = nib.load(str(path))
        n_frames = max(n_frames, img.shape[3] if img.ndim == 4 else 1)
        if source == "embed":
            blob = _display_volume(path, downsample, clip)
            payload[label] = {"mode": "embed", "name": path.name,
                              "data": base64.b64encode(blob).decode("ascii")}
        elif source == "link":
            url = (_stage(path, out_html, label) if stage
                   else _relative_url(path, out_html.parent))
            payload[label] = {"mode": "link", "name": path.name, "url": url}
        else:
            payload[label] = {"mode": "pick", "name": path.name,
                              "hint": str(path)}

    traces = _as_traces(fd)
    dtraces = _as_traces(dvars_traces)
    # Already a distance; just dropped to the transition grid the other two
    # live on, so all three panels share one x axis and one cursor.
    ctraces = {k: list(v[1:]) for k, v in _as_traces(distance).items()}
    # Meshes are always staged, never embedded: NiiVue reads them by URL, and
    # a surface is small enough that a link costs nothing and a base64 copy of
    # every vertex would dominate the page.
    meshes = []
    for i, (label, path) in enumerate(surfaces.items()):
        low = label.lower()
        if "pial" in low:
            rgba = [255, 64, 64, 255]
        elif "white" in low:
            rgba = [255, 212, 0, 255]
        else:
            rgba = _MESH_COLOURS[i % len(_MESH_COLOURS)]
        meshes.append({"label": label, "name": path.name, "rgba": rgba,
                       "url": _stage(path, out_html, label)})

    overlays = []
    for i, (label, path) in enumerate(contours.items()):
        low = label.lower()
        cmap = ("red" if "pial" in low else
                "warm" if "white" in low else
                _OVERLAY_COLOURMAPS[i % len(_OVERLAY_COLOURMAPS)])
        overlays.append({"label": label, "name": path.name, "colormap": cmap,
                         "url": _stage(path, out_html, label)})

    extra = []
    for title, value in dict(panels or {}).items():
        threshold, ymax = None, None
        if isinstance(value, tuple):
            value, threshold, ymax = (list(value) + [None, None])[:3]
        series = value if hasattr(value, "items") else {str(title): value}
        extra.append([str(title),
                      {k: list(np.asarray(v, dtype=np.float64).ravel()[1:])
                       for k, v in dict(series).items()},
                      threshold, ymax])

    html = _HTML.replace("/*NIIVUE*/", Path(bundle).read_text(encoding="utf-8"))
    html = html.replace("/*DATA*/", json.dumps({
        "volumes": payload,
        "meshes": meshes,
        "meshThickness": float(mesh_thickness_mm),
        "overlays": overlays,
        "traces": traces,
        "panels": [["FD (mm)", traces, fd_threshold, None],
                   ["DVARS (standardised)", dtraces, 1.5, None],
                   ["CD to mean image", ctraces, 0.01, distance_max]] + extra,
        "nFrames": n_frames,
        "title": title,
        "subtitle": subtitle,
        "threshold": fd_threshold,
        "source": source,
        "outliers": sorted(int(i) for i in (outliers if outliers is not None
                                            else [])),
    }))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>motion QC</title>
<style>
:root{--bg:#0e1116;--panel:#171b22;--line:#2a313c;--ink:#e6edf3;--dim:#8b949e;
      --accent:#4a9eff;--warn:#f0883e}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);overflow:hidden;
     display:flex;flex-direction:column;
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:10px 18px;border-bottom:1px solid var(--line);flex:0 0 auto}
h1{margin:0;font-size:15px;font-weight:600}
.sub{color:var(--dim);font-size:12px;margin-top:1px}
/* the viewers take whatever the header, readout and trace leave */
#viewers{display:flex;gap:8px;padding:8px 18px;flex:1 1 auto;min-height:0}
.viewer{flex:1 1 0;min-width:0;background:var(--panel);display:flex;
        flex-direction:column;border:1px solid var(--line);border-radius:8px;
        overflow:hidden}
.viewer h2{margin:0;padding:5px 10px;font-size:11px;font-weight:600;
           letter-spacing:.04em;text-transform:uppercase;color:var(--dim);
           border-bottom:1px solid var(--line);flex:0 0 auto}
canvas{flex:1 1 auto;min-height:0;width:100%;display:block;cursor:crosshair}
#bar{padding:2px 18px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;
     flex:0 0 auto}
#frame{font-variant-numeric:tabular-nums;font-weight:600}
#stats{color:var(--dim);font-size:13px}
#stats b{color:var(--ink);font-weight:600}
#plot{padding:0 18px 10px;flex:0 0 auto}
svg.trace{width:100%;height:76px;display:block;background:var(--panel);
    border:1px solid var(--line);border-radius:8px;cursor:ew-resize;
    touch-action:none;margin-bottom:6px}
.hint{padding:0 18px 10px;color:var(--dim);font-size:12px;flex:0 0 auto}
.pick{padding:7px 11px;border-bottom:1px solid var(--line);display:flex;
      gap:9px;align-items:center;font-size:12px;color:var(--dim)}
.pick span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.err{padding:9px 11px;color:var(--warn);font-size:12px;line-height:1.45}
.legend{display:inline-flex;align-items:center;gap:5px;margin-right:12px}
#toggles{padding:4px 18px 0;display:flex;flex-wrap:wrap;gap:4px 16px;
  font-size:12px;color:var(--dim)}
.toggle{display:inline-flex;align-items:center;gap:5px;cursor:pointer;
  user-select:none}
.toggle input{cursor:pointer;margin:0}
.toggle.empty{opacity:.45;cursor:default}
.toggle.empty input{cursor:default}
.swatch{width:11px;height:3px;border-radius:2px;display:inline-block}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div></header>
<div id="viewers"></div>
<div id="bar"><span id="frame"></span><span id="stats"></span></div>
<div id="toggles"></div>
<div id="plot"></div>
<div class="hint" id="hint">Tick a measure to draw it. Drag on a trace to move
  through frames; drag on a volume to move the crosshair, which both viewers
  follow. Arrow keys step frames; Home/End jump to the ends.</div>
<script>/*NIIVUE*/</script>
<script>
const D = /*DATA*/;
const NV = (window.niivue || window.Niivue || {});
const Niivue = NV.Niivue || window.Niivue;
const COLORS = ["#4a9eff","#f0883e","#3fb950","#db61a2"];
let frame = 0, viewers = [];

function b64ToBlobUrl(b64){
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) buf[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([buf], {type:"application/gzip"}));
}

async function build(){
  document.getElementById("title").textContent = D.title;
  document.getElementById("subtitle").textContent = D.subtitle;
  const host = document.getElementById("viewers");
  for (const [label, spec] of Object.entries(D.volumes)){
    const card = document.createElement("div");
    card.className = "viewer";
    const h = document.createElement("h2"); h.textContent = label;
    const cv = document.createElement("canvas");
    card.appendChild(h);
    if (spec.mode === "pick"){
      const bar = document.createElement("div");
      bar.className = "pick";
      const btn = document.createElement("input");
      btn.type = "file"; btn.accept = ".nii,.nii.gz";
      const note = document.createElement("span");
      note.textContent = spec.hint || spec.name;
      note.title = spec.hint || "";
      bar.appendChild(btn); bar.appendChild(note);
      card.appendChild(bar);
      btn.onchange = async e => {
        const f = e.target.files[0];
        if (!f) return;
        // a File is loadable exactly like an embedded blob: NiiVue picks the
        // format from the name, and nothing is copied or converted
        await load(nv, URL.createObjectURL(f), f.name);
        note.textContent = f.name;
        setFrame(frame);
      };
    }
    card.appendChild(cv); host.appendChild(card);
    // NEVER (0) drops the 3D render tile that multiplanar shows by default,
    // and ROW (3) puts axial/coronal/sagittal side by side -- three views of
    // the selected frame, one short wide strip per volume.
    const nv = new Niivue({backColor:[0.06,0.07,0.09,1], show3Dcrosshair:false,
                           crosshairColor:[0.29,0.62,1,0.85], textHeight:0.03,
                           multiplanarShowRender:0, multiplanarLayout:3,
                           // Left-drag on a slice moves the crosshair, which
                           // is what dragging a volume should do. dragMode 0
                           // (none) only suppresses the right-drag contrast
                           // gesture, so a stray drag cannot silently rewindow
                           // one viewer and not the other.
                           dragMode:0,
                           dragAndDropEnabled: spec.mode === "pick"});
    nv.attachToCanvas(cv);
    nv.onImageLoaded = () => setFrame(frame);
    try {
      if (spec.mode === "embed")
        await load(nv, b64ToBlobUrl(spec.data), spec.name || label + ".nii.gz");
      else if (spec.mode === "link")
        await load(nv, spec.url, spec.name);
    } catch (err) {
      // The path is known and was tried; only now is it worth asking. Report
      // what actually happened rather than assuming: this fires for any load
      // failure, and blaming file:// when the page is on http sends whoever
      // reads it looking in the wrong place.
      const msg = document.createElement("div");
      msg.className = "err";
      const why = location.protocol === "file:"
        ? "The page was opened from file://, which browsers refuse to read " +
          "other files from. Serve it instead: lightprep.qc.serve(report)."
        : "The page is served over " + location.protocol + ", so this is not " +
          "the file:// restriction. Check the server log and the browser " +
          "console; a large 4D volume can also exhaust memory while decoding.";
      msg.textContent = "Could not read " + (spec.url || spec.name) + ". " +
        why + " (" + (err && err.message ? err.message : err) + ")";
      const btn = document.createElement("input");
      btn.type = "file"; btn.accept = ".nii,.nii.gz";
      btn.onchange = async e => {
        const f = e.target.files[0];
        if (!f) return;
        await load(nv, URL.createObjectURL(f), f.name);
        msg.remove(); btn.remove(); setFrame(frame);
      };
      card.appendChild(msg); card.appendChild(btn);
      nv.opts.dragAndDropEnabled = true;
    }
    viewers.push(nv);
  }
  // Same subject, same run, same space: a location found in one is the same
  // location in the other, so the crosshairs follow each other. Frames are
  // shared through setFrame; this shares where in the head you are looking.
  viewers.forEach(nv => {
    const others = viewers.filter(o => o !== nv);
    if (others.length && nv.broadcastTo) nv.broadcastTo(others, {"2d":true});
  });
  drawToggles();
  drawPlot();
  setFrame(0);
}

async function load(nv, url, name){
  await nv.loadVolumes([{url, name}]);
  nv.setSliceType(nv.sliceTypeMultiplanar);
  nv.opts.multiplanarShowRender = 0;   // NEVER
  nv.opts.multiplanarLayout = 3;       // ROW
  await loadMeshes(nv);
  await loadContours(nv);
  if (nv.resizeListener) nv.resizeListener();
  nv.drawScene();
}

// A rasterised one-voxel shell, drawn opaque over the volume. In a slice this
// IS the surface's intersection with that plane -- NiiVue has no mesh-plane
// intersection mode, and a clipped slab of 3D mesh is not the same thing.
async function loadContours(nv){
  const specs = (D.overlays || []);
  if (!specs.length || !nv.addVolumeFromUrl) return;
  for (const o of specs){
    try {
      await nv.addVolumeFromUrl({url: o.url, name: o.name, colormap: o.colormap,
                                 opacity: 1.0, cal_min: 0.5, cal_max: 1.0});
    } catch (err) {
      console.warn("contour overlay failed:", o.label, err);
    }
  }
  if (nv.updateGLVolume) nv.updateGLVolume();
}

// In a 2D slice NiiVue draws a mesh as its intersection with the slice plane,
// so a white and a pial surface become the contours you would check a
// coregistration with. Thickness 1 keeps them a line rather than a slab.
async function loadMeshes(nv){
  const specs = (D.meshes || []);
  if (!specs.length || !nv.loadMeshes) return;
  try {
    await nv.loadMeshes(specs.map(m => ({url: m.url, name: m.name,
                                         rgba255: m.rgba})));
    // Must go through the setter, not opts: it is setMeshThicknessOn2D that
    // calls updateGLVolume, and without that the shader keeps the default --
    // which is Infinity, i.e. the whole surface painted on every slice.
    if (nv.setMeshThicknessOn2D) nv.setMeshThicknessOn2D(D.meshThickness);
    else nv.opts.meshThicknessOn2D = D.meshThickness;
  } catch (err) {
    // A missing or unreadable surface must not cost the volume viewer: the
    // page is still useful without contours, and the console says why.
    console.warn("surface overlay failed:", err);
  }
}

// --- the FD trace, as SVG so it scales and stays crisp -------------------
const PAD = {l:44, r:12, t:12, b:24};
let plotW = 900, plotH = 88, nPts = 1;

// Which panels are drawn. A page carrying five measures is unreadable with all
// of them and useless with the wrong ones missing, so the reader chooses --
// and a panel with no data anywhere starts off, since an empty axis is just
// noise competing for height.
const shown = D.panels.map(([, tr]) =>
  Object.values(tr).some(a => a && a.length));

function drawToggles(){
  const host = document.getElementById("toggles");
  if (!host) return;
  host.innerHTML = "";
  D.panels.forEach(([name, tr], i) => {
    const has = Object.values(tr).some(a => a && a.length);
    const id = "panel" + i;
    const label = document.createElement("label");
    label.className = "toggle" + (has ? "" : " empty");
    const box = document.createElement("input");
    box.type = "checkbox"; box.id = id; box.checked = shown[i]; box.disabled = !has;
    box.onchange = () => { shown[i] = box.checked; drawPlot(); setFrame(frame); };
    label.appendChild(box);
    label.appendChild(document.createTextNode(has ? name : name + " (no data)"));
    host.appendChild(label);
  });
}

function drawPlot(){
  const host = document.getElementById("plot");
  host.innerHTML = "";
  let bar = "";
  D.panels.forEach(([name, traces, thr, ymax], i) => {
    if (!shown[i]) return;
    panel(host, traces, name, thr, ymax);
    bar += summarise(traces, name, thr);
  });
  document.getElementById("stats").innerHTML = bar;
}

function summarise(traces, name, thr){
  const short = name.split(" ")[0];
  return Object.keys(traces).map((k,i) => {
    const a = traces[k], mean = a.reduce((p,c)=>p+c,0)/a.length;
    const over = 100*a.filter(v=>v>thr).length/a.length;
    return `<span class="legend"><i class="swatch"
      style="background:${COLORS[i%4]}"></i>${short} ${k}:
      <b>${mean.toFixed(mean<0.1?4:2)}</b>,
      <b>${over.toFixed(0)}%</b>&gt;${thr}</span>`;
  }).join("");
}

function panel(host, traces, name, thr, ymax){
  const labels = Object.keys(traces);
  if (!labels.length) return;
  nPts = Math.max(nPts, ...labels.map(k => traces[k].length));
  // Two different ways a trace ends up a flat line at the bottom, and the axis
  // has to dodge both. Scaling to the maximum lets one bad frame squash the
  // range the rest of the run lives in; scaling to the threshold does the same
  // whenever the run never goes near it. So: a high quantile rather than the
  // max, and the threshold only gets a say when the data come within reach of
  // it.
  const all = labels.flatMap(k => traces[k]).filter(v => Number.isFinite(v));
  const sorted = all.slice().sort((a, b) => a - b);
  const q = f => sorted.length ? sorted[Math.floor(f*(sorted.length-1))] : 0;
  const robust = Math.max(q(0.995), 1e-9);
  // The threshold may stretch the axis, but only so far: past twice the
  // robust maximum it is buying a dashed line at the cost of the trace, and a
  // run nowhere near its limit is better read on its own scale. Capping rather
  // than switching keeps this smooth -- no cliff where one frame changes the
  // axis by a factor.
  let maxY;
  if (ymax != null) maxY = ymax;
  else if (thr == null) maxY = robust*1.3;
  else maxY = Math.max(robust*1.15, Math.min(thr*1.15, robust*2.0));
  const trueMax = sorted.length ? sorted[sorted.length-1] : 0;
  const clipped = trueMax > maxY*1.001;
  const D_ = traces, maxFD = maxY;
  const x = i => PAD.l + i*(plotW-PAD.l-PAD.r)/Math.max(1,D.nFrames-2);
  const y = v => plotH-PAD.b -
        (Math.min(v, maxFD)/maxFD)*(plotH-PAD.t-PAD.b);   // clamp to the top
  let s = `<svg viewBox="0 0 ${plotW} ${plotH}" preserveAspectRatio="none"
           class="trace">`;
  s += `<text x="${PAD.l}" y="11" fill="#8b949e" font-size="11">${name}` +
       (clipped ? ` <tspan fill="#f0883e">peaks ${fmt(trueMax)}</tspan>` : "") +
       `</text>`;
  // Marked where THIS measure exceeds ITS OWN threshold. Shading one panel
  // with another's verdict says a frame is bad here when it is bad elsewhere,
  // which is the opposite of what a per-measure panel is for -- the whole
  // reason to draw five of them is to see which one objects.
  if (thr != null) {
    const bad = new Set();
    labels.forEach(k => traces[k].forEach((v, i) => { if (v > thr) bad.add(i); }));
    bad.forEach(i => {
      const xf = x(i);
      s += `<line x1="${xf}" y1="${PAD.t}" x2="${xf}" y2="${plotH-PAD.b}"
            stroke="#f85149" stroke-width="1.2" opacity=".45"/>`;
    });
  }
  s += `<line x1="${PAD.l}" y1="${y(0)}" x2="${plotW-PAD.r}" y2="${y(0)}"
        stroke="#2a313c"/>`;
  if (thr != null && thr <= maxY) {
    s += `<line x1="${PAD.l}" y1="${y(thr)}" x2="${plotW-PAD.r}"
          y2="${y(thr)}" stroke="#f0883e" stroke-dasharray="4 4"
          opacity=".7"/>`;
    s += `<text x="4" y="${y(thr)+4}" fill="#8b949e" font-size="11">${thr}</text>`;
  }
  s += `<text x="4" y="${y(maxFD)+9}" fill="#8b949e" font-size="11">
        ${maxFD < 0.1 ? maxFD : maxFD.toFixed(1)}</text>`;
  labels.forEach((k,i) => {
    const pts = D_[k].map((v,j) => `${x(j)},${y(v)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${COLORS[i%4]}"
          stroke-width="1.4" vector-effect="non-scaling-stroke"/>`;
  });
  s += `<line class="cursor" x1="0" y1="${PAD.t}" x2="0" y2="${plotH-PAD.b}"
        stroke="#e6edf3" stroke-width="1" opacity=".9"/>`;
  s += `</svg>`;
  const wrap = document.createElement("div");
  wrap.innerHTML = s;
  const svg = wrap.firstElementChild;
  host.appendChild(svg);
  scrubbable(svg);
}

// --- scrubbing ------------------------------------------------------------
function scrubbable(el){
  const move = ev => {
    const r = el.getBoundingClientRect();
    const cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    // Traces hold n-1 points for n frames -- both are differences against the
    // preceding frame -- drawn inside the axis margins, so point i belongs to
    // frame i+1. Only traces scrub; volumes navigate.
    const l = PAD.l*r.width/plotW;
    const w = r.width - (PAD.l+PAD.r)*r.width/plotW;
    const u = Math.min(1, Math.max(0, (cx-l)/Math.max(1,w)));
    setFrame(1 + Math.round(u*(D.nFrames-2)));
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
  };
  el.addEventListener("pointerdown", ev => {
    ev.preventDefault(); move(ev);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}

function setFrame(f){
  frame = Math.max(0, Math.min(D.nFrames-1, f|0));
  viewers.forEach(nv => {
    if (nv.volumes && nv.volumes.length) nv.setFrame4D(nv.volumes[0].id, frame);
  });
  const px = PAD.l + ((frame-1)/Math.max(1,D.nFrames-2))*(plotW-PAD.l-PAD.r);
  document.querySelectorAll(".cursor").forEach(c => {
    c.setAttribute("x1", px); c.setAttribute("x2", px);
  });
  // both are differences, so frame f pairs with trace[f-1]
  // Adaptive precision: CD runs around 0.01, so two decimals would print
  // every frame of a run as the same number.
  const fmt = v => Math.abs(v) < 0.1 ? v.toFixed(4) : v.toFixed(2);
  const at = (tr) => Object.keys(tr).map(k => {
    const v = tr[k][frame-1];
    return v === undefined ? `${k} --` : `${k} ${fmt(v)}`;
  }).join(" ");
  const parts = D.panels.map(([name, tr], i) => {
    if (!shown[i]) return "";
    const t = at(tr);
    return t ? `${name.split(" ")[0]} ${t}` : "";
  }).filter(Boolean).join("   ");
  const flagged = (D.outliers || []).includes(frame);
  const el = document.getElementById("frame");
  el.textContent = `frame ${frame+1} / ${D.nFrames}` +
    (flagged ? "  OUTLIER" : "") + `   ${parts}`;
  el.style.color = flagged ? "#f85149" : "";
}

addEventListener("keydown", e => {
  const k = {ArrowLeft:-1, ArrowRight:1, ArrowDown:-10, ArrowUp:10}[e.key];
  if (k){ e.preventDefault(); setFrame(frame+k); }
  else if (e.key === "Home") setFrame(0);
  else if (e.key === "End") setFrame(D.nFrames-1);
});
addEventListener("resize", () => {
  drawPlot(); setFrame(frame);
  viewers.forEach(nv => { if (nv.resizeListener) nv.resizeListener(); });
});
if (D.source === "pick")
  document.getElementById("hint").textContent =
    "Choose each file above, or drop it on its viewer -- full resolution, " +
    "nothing copied. Then drag a trace for frames, a volume for coordinates.";
build();
</script></body></html>
"""
