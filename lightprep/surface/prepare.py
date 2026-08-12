"""Getting FreeSurfer surfaces into the space the volumes live in.

FreeSurfer writes surfaces in tkrRAS ("surface RAS"), which is offset from the
scanner RAS of its own ``orig.mgz`` by that volume's centre, ``c_ras``. Load a
surface next to a NIfTI without accounting for it and everything sits a few
millimetres out -- on this subject, ``(-0.92, 0.91, -4.58) mm``. That is small
enough to look plausible and large enough to be wrong, which is the dangerous
size: a correct registration renders as a bad one, and sampled data is quietly
drawn from the neighbouring tissue.

``mris_convert --to-scanner`` applies the shift. This module wraps it and then
*checks* the result against the c_ras the volume header reports, so a silent
convention change cannot pass unnoticed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .._utils import run
from ..coreg.freesurfer import _freesurfer_env

HEMIS = ("lh", "rh")
SURFACES = ("white", "pial")

#: How far the check may drift from the header's c_ras before we call it broken.
CRAS_TOL_MM = 1e-3


@dataclass(frozen=True)
class PreparedSurfaces:
    """Surfaces converted into their volume's scanner RAS.

    Attributes:
        paths: ``{(hemi, surface): path}`` of the written GIfTI files.
        subject: FreeSurfer subject the surfaces came from.
        c_ras: The offset that was applied, read from the volume header.
    """

    paths: dict
    subject: str
    c_ras: tuple[float, float, float]

    def __getitem__(self, key):
        return self.paths[key]


def _c_ras(subject_dir: Path) -> np.ndarray:
    """The scanner-RAS coordinate of the volume centre, from orig.mgz."""
    orig = subject_dir / "mri" / "orig.mgz"
    if not orig.exists():
        raise FileNotFoundError(f"no orig.mgz for this subject: {orig}")
    return np.asarray(nib.load(orig).header["Pxyz_c"], dtype=float).ravel()


def prepare_surfaces(
    subject: str,
    subjects_dir,
    out_dir,
    *,
    hemis=HEMIS,
    surfaces=SURFACES,
    freesurfer_home=None,
    overwrite: bool = False,
) -> PreparedSurfaces:
    """Convert a subject's FreeSurfer surfaces to scanner RAS GIfTI.

    Args:
        subject: FreeSurfer subject name.
        subjects_dir: The SUBJECTS_DIR holding it.
        out_dir: Where to write the GIfTI surfaces.
        hemis: Hemispheres to convert.
        surfaces: Surface names to convert, e.g. ``white``, ``pial``.
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        overwrite: Redo the conversion even if the output already exists.

    Returns:
        A :class:`PreparedSurfaces` mapping ``(hemi, surface)`` to paths.

    Raises:
        FileNotFoundError: If a source surface or orig.mgz is missing.
        RuntimeError: If the applied shift does not match the header's c_ras,
            which would mean the surfaces are not where we think they are.
    """
    subjects_dir = Path(subjects_dir).resolve()
    subject_dir = subjects_dir / subject
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"no such FreeSurfer subject: {subject_dir}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _freesurfer_env(freesurfer_home, subjects_dir)
    cras = _c_ras(subject_dir)

    paths = {}
    for hemi in hemis:
        for surf in surfaces:
            src = subject_dir / "surf" / f"{hemi}.{surf}"
            if not src.exists():
                # FreeSurfer 7+ sometimes ships only the .T1 variant.
                alt = src.with_name(f"{hemi}.{surf}.T1")
                if not alt.exists():
                    raise FileNotFoundError(f"no {hemi}.{surf} (or .T1) in {src.parent}")
                src = alt
            dst = out_dir / f"{subject}_hemi-{hemi[0].upper()}_space-scanner_{surf}.surf.gii"
            if overwrite or not dst.exists():
                run(["mris_convert", "--to-scanner", src, dst], extra_env=env)

            # Trust nothing: confirm the file really is the source plus c_ras.
            raw, _ = nib.freesurfer.read_geometry(src)
            got = np.asarray(nib.load(dst).darrays[0].data, dtype=float)
            if got.shape != raw.shape:
                raise RuntimeError(f"{dst.name}: {got.shape} vertices, source has {raw.shape}")
            shift = (got - raw).mean(axis=0)
            if not np.allclose(shift, cras, atol=CRAS_TOL_MM):
                raise RuntimeError(
                    f"{dst.name}: applied shift {shift} does not match the volume's "
                    f"c_ras {cras}. The surfaces would not line up with the volumes."
                )
            paths[(hemi, surf)] = dst

    return PreparedSurfaces(paths=paths, subject=subject, c_ras=tuple(map(float, cras)))


def separate_hemispheres(lh, rh, out_lh, out_rh, *, gap_mm: float = 6.0):
    """Pull two centred display surfaces apart so they sit side by side.

    ``mris_inflate`` centres each hemisphere on the origin independently, so
    ``lh.inflated`` and ``rh.inflated`` occupy the same volume and render as one
    interpenetrating blob. Anatomical surfaces do not have this problem -- white
    and pial keep their real left/right positions -- so this is only for the
    inflated (or spherical) geometries used for display.

    Each surface is translated along x until they abut with ``gap_mm`` between
    them, left hemisphere at negative x as in RAS. Only x is touched: vertex
    order, faces and every other coordinate are untouched, so per-vertex
    overlays still line up.

    Args:
        lh: Left surface (GIfTI).
        rh: Right surface (GIfTI).
        out_lh: Where to write the shifted left surface.
        out_rh: Where to write the shifted right surface.
        gap_mm: Space to leave between the two.

    Returns:
        ``(out_lh, out_rh)``.
    """
    outs = []
    imgs = {side: nib.load(str(p)) for side, p in (("lh", lh), ("rh", rh))}
    verts = {side: np.asarray(img.darrays[0].data, dtype=np.float64)
             for side, img in imgs.items()}
    # lh's right edge goes to -gap/2; rh's left edge to +gap/2
    shifts = {
        "lh": -gap_mm / 2.0 - verts["lh"][:, 0].max(),
        "rh": gap_mm / 2.0 - verts["rh"][:, 0].min(),
    }
    for side, out in (("lh", out_lh), ("rh", out_rh)):
        img = imgs[side]
        moved = verts[side].copy()
        moved[:, 0] += shifts[side]
        img.darrays[0].data = moved.astype(np.float32)
        out = Path(out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        img.to_filename(str(out))
        outs.append(out)
    return tuple(outs)


#: Magic for :func:`export_binary`; bumped if the layout ever changes.
BINARY_MAGIC = b"SURF"


def export_binary(surface, out) -> Path:
    """Repack a GIfTI surface as a flat binary a browser can use directly.

    GIfTI is XML with base64 payloads, which JavaScript would have to parse
    before it could draw anything. This writes the same geometry -- no
    decimation, no smoothing -- as::

        magic "SURF" | uint32 nVert | uint32 nTri | float32 xyz[nVert] | uint32 tri[nTri]

    so a fetch lands straight in a Float32Array/Uint32Array.
    """
    gii = nib.load(str(surface))
    verts = np.ascontiguousarray(gii.darrays[0].data, dtype=np.float32)
    faces = np.ascontiguousarray(gii.darrays[1].data, dtype=np.uint32)
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(struct.pack("<4sII", BINARY_MAGIC, verts.shape[0], faces.shape[0]))
        fh.write(verts.tobytes())
        fh.write(faces.tobytes())
    return out
