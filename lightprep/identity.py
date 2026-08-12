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


def _bet(t1: Path, out: Path) -> Path:
    run(["bet", t1, out, "-f", "0.4", "-R"])
    return out


def _flirt(moving: Path, ref: Path, out: Path) -> Path:
    run(["flirt", "-in", moving, "-ref", ref, "-out", out,
         "-dof", 12, "-cost", "corratio", "-interp", "trilinear"])
    return out


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

    vols = [nib.load(a).get_fdata(dtype=np.float32) for a in aligned]
    stack = np.stack(vols, 0)                                # (N, X, Y, Z)
    mask = stack.mean(0) > np.percentile(stack.mean(0), 70)  # shared brain
    # z-score each brain within the mask, then correlate
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
