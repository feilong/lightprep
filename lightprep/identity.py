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

