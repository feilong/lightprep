"""Tell sessions apart by brain shape -- match each to its 1-nearest-neighbour.

DICOM PatientName is not trusted (a generic 'crlab' / operator placeholders), so
we verify who-is-who from the anatomy itself. Each session's T1w is brain-
extracted and affinely aligned to a common reference; the pairwise similarity of
the aligned brains then clusters sessions by subject -- the same brain matches
almost perfectly, different brains keep sulcal/ventricle differences an affine
cannot remove.

This is intentionally simple (affine + correlation, 1NN). It is a verification
aid, not a biometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ._utils import run


def _robustfov(img: Path, out: Path) -> Path:
    """Crop the neck/oversized FOV before bet.

    Full-head MPRAGE (e.g. fm) has a large field of view that leaves bet
    including skull and neck; robustfov standardises the FOV to a brain-sized
    region. It is a near-no-op on the already-cropped EP3d structurals.
    """
    run(["robustfov", "-i", img, "-r", out])
    return out


def _bet(t1: Path, out: Path) -> Path:
    run(["bet", t1, out, "-f", "0.5", "-R"])
    return out


def _mindgrab(img: Path, out: Path, brainchop: str = "brainchop") -> Path:
    """Brain-extract with MindGrab (brainchop) -- the default here.

    bet thresholds intensities, which fails outright on MP2RAGE UNI: its
    background is filled with noise rather than dark, so bet keeps skull and air
    (measured: 2400-3000 ml of "brain", against a real 1100-1500). MindGrab is a
    small CNN trained across modalities, so it strips UNI, T2w and even EPI
    alike -- on the pilot's SM it agrees with FreeSurfer's own brainmask to 3%
    (1088 vs 1120 ml), in ~20 s on CPU.

    Returns the masked image (intensities kept, background zeroed).
    """
    run([brainchop, img, "-m", "mindgrab", "-o", out])
    return out


#: brain-extraction backends, by name
STRIP_METHODS = {"mindgrab": _mindgrab, "bet": _bet}


def _flirt(moving: Path, ref: Path, out: Path) -> Path:
    run(["flirt", "-in", moving, "-ref", ref, "-out", out,
         "-dof", 12, "-cost", "corratio", "-interp", "trilinear"])
    return out


def _downsample(img: Path, out: Path, mm: float) -> Path:
    run(["flirt", "-in", img, "-ref", img, "-applyisoxfm", mm, "-out", out,
         "-interp", "trilinear"])
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, accumulated in float64.

    Images are read as float32 -- that is what the scanner wrote -- but the
    sums here run over millions of voxels, and float32 accumulation over that
    many terms is not accurate enough to be compared at three decimal places.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


def pairwise_similarity(t1_paths, labels=None, work_dir=None, downsample_mm=2.0,
                        strip="mindgrab"):
    """Directional (asymmetric) brain similarity between every ordered pair.

    Each T1w is brain-extracted and downsampled to a common resolution. Then,
    for every ordered pair (i, j), image i is affinely registered onto image j
    and their overlap is scored over j's brain mask. Because the moving image and
    the reference mask both change with direction, ``M[i, j] != M[j, i]`` -- the
    matrix is asymmetric, as requested.

    Interpretation: two sessions of the same person align near-perfectly in both
    directions (both M[i,j] and M[j,i] high); different people leave residual
    shape differences an affine cannot absorb.

    Args:
        strip: brain-extraction backend, ``"mindgrab"`` (default) or ``"bet"``.
            bet is kept only for the 3T T1w/T2starw it was tuned on -- it cannot
            strip MP2RAGE UNI (see :func:`_mindgrab`).

    Returns:
        dict with ``labels`` and ``matrix`` (NxN; row i = image i registered onto
        each column j; diagonal = 1).
    """
    t1_paths = [Path(p) for p in t1_paths]
    if labels is None:
        labels = [p.parents[1].name for p in t1_paths]
    work = Path(work_dir or "/tmp/identity_pw")
    work.mkdir(parents=True, exist_ok=True)

    prepped = []
    for i, p in enumerate(t1_paths):
        if strip == "mindgrab":
            # strip at native resolution (~20 s), then downsample: MindGrab is a
            # CNN, so full-res costs little, and it needs the head intact --
            # robustfov/bet's crop-then-threshold dance is unnecessary.
            brain = _mindgrab(p, work / f"{i:02d}_brain.nii.gz")
            prepped.append(_downsample(brain, work / f"{i:02d}_ds.nii.gz", downsample_mm))
        else:
            # bet is slow at full res (up to 500x600x416), so downsample first;
            # identity needs only a coarse brain.
            ds = _downsample(p, work / f"{i:02d}_ds.nii.gz", downsample_mm)
            crop = _robustfov(ds, work / f"{i:02d}_rfov.nii.gz")
            prepped.append(_bet(crop, work / f"{i:02d}_brain.nii.gz"))

    n = len(prepped)
    M = np.eye(n)
    for j in range(n):                                   # reference (fixed) = j
        ref = nib.load(prepped[j]).get_fdata(dtype=np.float64)
        mask = ref > np.percentile(ref, 70)
        for i in range(n):
            if i == j:
                continue
            moved = _flirt(prepped[i], prepped[j], work / f"{i:02d}_to_{j:02d}.nii.gz")
            mov = nib.load(moved).get_fdata(dtype=np.float64)
            M[i, j] = _corr(mov[mask], ref[mask])
    return {"labels": labels, "matrix": M}


def format_matrix(result) -> str:
    """Render the asymmetric matrix (row i registered onto column j)."""
    labels, M = result["labels"], result["matrix"]
    short = [l.replace("ses-", "").replace("sub-", "")[:12] for l in labels]
    w = max(len(s) for s in short)
    head = " " * (w + 6) + " ".join(f"{s[:8]:>8}" for s in short)
    lines = ["row i -> aligned ONTO column j (higher = i looks like j)", head]
    for i, s in enumerate(short):
        row = " ".join(f"{M[i, j]:8.3f}" for j in range(len(short)))
        lines.append(f"{s:>{w}}  ->  {row}")
    return "\n".join(lines)


def brain_similarity(t1_paths, labels=None, work_dir=None, reference_idx=0):
    """Affinely align T1w images and score their pairwise brain similarity.

    Args:
        t1_paths: one T1w per session.
        labels: names for each (defaults to the parent session directory).
        work_dir: scratch for the brain-extracted / aligned intermediates.
        reference_idx: which image everything is aligned to (its own space).

    Returns:
        dict with ``labels``, ``similarity`` (NxN Pearson r over the shared brain
        mask), and ``nearest`` -- ``[(label, nn_label, r), …]``, each session's
        most similar other session.
    """
    t1_paths = [Path(p) for p in t1_paths]
    if labels is None:
        labels = [p.parents[1].name for p in t1_paths]      # the ses-* dir
    work = Path(work_dir or "/tmp/identity"); work.mkdir(parents=True, exist_ok=True)

    # brain-extract, then affine-align each to the reference's space
    brains = [_bet(p, work / f"{i:02d}_brain.nii.gz") for i, p in enumerate(t1_paths)]
    ref = brains[reference_idx]
    aligned = []
    for i, b in enumerate(brains):
        aligned.append(b if i == reference_idx
                       else _flirt(b, ref, work / f"{i:02d}_aligned.nii.gz"))

    vols = [nib.load(a).get_fdata(dtype=np.float64) for a in aligned]
    stack = np.stack(vols, 0)                                # (N, X, Y, Z)
    mask = stack.mean(0) > np.percentile(stack.mean(0), 70)  # shared brain
    # z-score each brain within the mask, then correlate. float64 throughout:
    # X @ X.T sums over every masked voxel -- 5.2M of them on a 1mm T1 -- and
    # BLAS sgemm has no pairwise summation to fall back on, so in float32 the
    # error reaches 1% here. It shows up as a diagonal that is not 1 (measured:
    # 1.0106) and, less visibly, as the same size of error on the off-diagonal
    # similarities this function exists to report.
    X = stack[:, mask]
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    sim = (X @ X.T) / X.shape[1]

    n = len(labels)
    nearest = []
    for i in range(n):
        others = [(sim[i, j], j) for j in range(n) if j != i]
        r, j = max(others)
        nearest.append((labels[i], labels[j], float(r)))
    return {"labels": labels, "similarity": sim, "nearest": nearest}


def format_report(result) -> str:
    """A compact text report: the matrix and each session's nearest neighbour."""
    labels, sim = result["labels"], result["similarity"]
    short = [l.replace("ses-", "")[:14] for l in labels]
    w = max(len(s) for s in short)
    lines = [" " * (w + 2) + " ".join(f"{s[:6]:>6}" for s in short)]
    for i, s in enumerate(short):
        row = " ".join(f"{sim[i, j]:6.2f}" for j in range(len(short)))
        lines.append(f"{s:>{w}}  {row}")
    lines.append("")
    for label, nn, r in result["nearest"]:
        lines.append(f"  {label:<28} 1NN -> {nn:<28} r={r:.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------- QC workflow --
@dataclass(frozen=True)
class IdentityQC:
    """Verdict of checking session labels against brain shape.

    Attributes:
        labels: One name per scan, in matrix order.
        subjects: The subject each scan claims to belong to.
        matrix: Directional similarity, row i registered onto column j.
        within: Similarity over ordered pairs claiming the same subject.
        between: Similarity over ordered pairs claiming different subjects.
        separation: ``min(within) - max(between)``. Positive means the two
            distributions do not overlap, which is what makes the labelling
            checkable rather than merely plausible.
        nearest: ``[(label, nn_label, r, ok)]`` -- each scan's most similar
            other scan, and whether it belongs to the same subject.
        mismatches: Scans whose nearest neighbour is a different subject.
        singletons: Scans that are the only one for their subject, so they
            have no within-subject pair and cannot be checked this way.
    """

    labels: tuple[str, ...]
    subjects: tuple[str, ...]
    matrix: np.ndarray
    within: np.ndarray
    between: np.ndarray
    separation: float
    nearest: tuple
    mismatches: tuple[str, ...]
    singletons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """No scan looks more like somebody else than like its own subject."""
        return not self.mismatches

    @property
    def asymmetry(self) -> np.ndarray:
        """``|M[i,j] - M[j,i]|`` over unordered pairs.

        A running check on the method rather than the data: it should be small,
        but if it is comparable to ``separation`` then direction matters more
        than identity does and the threshold is not meaningful.
        """
        n = len(self.labels)
        return np.array([abs(self.matrix[i, j] - self.matrix[j, i])
                         for i in range(n) for j in range(i + 1, n)])


def collect_bids_anat(bids_dir, suffix: str = "T1w"):
    """Every anatomical of a given suffix in a BIDS tree.

    Args:
        bids_dir: BIDS dataset root.
        suffix: BIDS suffix to collect, e.g. ``T1w``, ``T2w``, ``UNIT1``.

    Returns:
        ``(paths, labels, subjects)``. Labels are ``sub/ses`` where sessions
        exist and ``sub`` where they do not.
    """
    bids_dir = Path(bids_dir)
    paths, labels, subjects = [], [], []
    for sub_dir in sorted(p for p in bids_dir.glob("sub-*") if p.is_dir()):
        sub = sub_dir.name.replace("sub-", "")
        for img in sorted(sub_dir.glob(f"**/anat/*_{suffix}.nii*")):
            ses = next((p[4:] for p in img.name.split("_")
                        if p.startswith("ses-")), None)
            paths.append(img)
            labels.append(f"{sub}/{ses}" if ses else sub)
            subjects.append(sub)
    return paths, labels, subjects


def identity_qc(t1_paths, subjects, labels=None, **kwargs) -> IdentityQC:
    """Check that scans claiming the same subject really are the same brain.

    Wraps :func:`pairwise_similarity` with the comparison that makes it a QC
    rather than a matrix: within-subject similarity against between-subject,
    and a 1-nearest-neighbour verdict per scan.

    Mislabelling is the failure this exists to catch, and nothing downstream
    catches it. A swapped session produces surfaces that fit one brain and
    timeseries sampled from another, and every later stage runs happily.

    Args:
        t1_paths: One structural per scan.
        subjects: The subject each scan claims to belong to, same length.
        labels: Display names; defaults to ``subjects``.
        **kwargs: Passed to :func:`pairwise_similarity` (``work_dir``,
            ``downsample_mm``, ``strip``).

    Returns:
        An :class:`IdentityQC`.

    Raises:
        ValueError: If fewer than two scans are given, or the lengths disagree.
    """
    t1_paths = [Path(p) for p in t1_paths]
    subjects = list(subjects)
    if len(t1_paths) < 2:
        raise ValueError(f"need at least two scans to compare, got {len(t1_paths)}")
    if len(subjects) != len(t1_paths):
        raise ValueError(
            f"{len(subjects)} subject labels for {len(t1_paths)} scans")
    labels = list(labels) if labels is not None else list(subjects)

    M = pairwise_similarity(t1_paths, labels=labels, **kwargs)["matrix"]
    n = len(labels)

    # Ordered pairs: a directional matrix has two measurements per pair, and
    # both are evidence.
    within = np.array([M[i, j] for i in range(n) for j in range(n)
                       if i != j and subjects[i] == subjects[j]])
    between = np.array([M[i, j] for i in range(n) for j in range(n)
                        if i != j and subjects[i] != subjects[j]])
    separation = (float(within.min() - between.max())
                  if within.size and between.size else float("nan"))

    nearest, mismatches, singletons = [], [], []
    for i in range(n):
        j = int(next(k for k in np.argsort(M[i])[::-1] if k != i))
        ok = subjects[j] == subjects[i]
        alone = not any(k != i and subjects[k] == subjects[i] for k in range(n))
        nearest.append((labels[i], labels[j], float(M[i, j]), ok))
        if alone:
            singletons.append(labels[i])
        elif not ok:
            mismatches.append(labels[i])

    return IdentityQC(
        labels=tuple(labels), subjects=tuple(subjects), matrix=M,
        within=within, between=between, separation=separation,
        nearest=tuple(nearest), mismatches=tuple(mismatches),
        singletons=tuple(singletons))


def format_qc(qc: IdentityQC) -> str:
    """The matrix, the within/between contrast, and the per-scan verdict."""
    lines = [format_matrix({"labels": list(qc.labels), "matrix": qc.matrix}), ""]
    for name, vals in (("within-subject", qc.within), ("between-subject", qc.between)):
        if vals.size:
            lines.append(f"{name:<16s} n={vals.size:>3d}  mean {vals.mean():.3f}  "
                         f"min {vals.min():.3f}  max {vals.max():.3f}")
    if np.isfinite(qc.separation):
        verdict = "cleanly separated" if qc.separation > 0 else "OVERLAP -- inspect"
        lines.append(f"{'separation':<16s} lowest within - highest between = "
                     f"{qc.separation:+.3f}  ({verdict})")
    asym = qc.asymmetry
    if asym.size:
        lines.append(f"{'asymmetry':<16s} |M[i,j]-M[j,i]| median "
                     f"{np.median(asym):.4f}, max {asym.max():.4f}")

    lines += ["", "nearest neighbour of each scan:"]
    for label, nn, r, ok in qc.nearest:
        note = ("(only scan for this subject)" if label in qc.singletons
                else "same subject" if ok else "DIFFERENT SUBJECT")
        lines.append(f"  {label:<16s} -> {nn:<16s} r={r:.3f}   {note}")
    lines.append("")
    lines.append("PASS: every scan's nearest neighbour is the same subject."
                 if qc.passed else
                 f"FAIL: {len(qc.mismatches)} scan(s) look more like another "
                 f"subject: {', '.join(qc.mismatches)}")
    return "\n".join(lines)
