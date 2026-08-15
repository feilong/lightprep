"""Turning a fake-geometry recon into derivatives valid in true geometry.

:func:`lightprep.recon.native` (in :mod:`lightprep.recon.fake`) runs the recon on a volume carrying a fake
affine, so FreeSurfer's conform is a no-op and the anatomy is never
interpolated. Everything that recon produces therefore lives in a space whose
metric is wrong by ``M = A_true @ inv(A_fake)``.

:func:`correct` writes a second folder holding ONLY what is valid once ``M``
is applied:

* **volumes** -- voxel data is untouched by the trick (it is computed from
  intensities on a grid that is exactly the native array), so they are copied
  verbatim with the affine rewritten to ``M @ A_fake``. No resampling.
* **surfaces** -- vertex coordinates transformed by ``M``, which is exact.
  FreeSurfer stores surfaces in tkreg coordinates derived from the volume's
  dims and voxel sizes, so correcting a volume's affine also moves tkreg; the
  surfaces are rewritten as ``s_new = tkr_corr @ inv(tkr_fake) @ s_old`` to
  stay consistent with the corrected volumes. That is what lets ``bbregister``
  run against this folder.
* **spherical parameterisations** -- ``?h.sphere.reg`` says which vertex sits
  where on the registration sphere, not where anything is in the head, so
  ``M`` does not apply and it is copied byte-for-byte.
* **annotations** -- per-vertex labels, carrying no geometry.
* **morphometry** -- RECOMPUTED from the corrected coordinates. FreeSurfer's
  own ``?h.thickness``/``?h.area``/``.stats`` are computed in the fake metric
  and cannot be rescaled, because the scale is anisotropic while thickness is
  measured along the local normal.

Anything not in that list is deliberately absent -- see :data:`EXCLUDED` and
the MANIFEST written into the output folder. The rule is that a file appears
only if it is either unaffected by the trick or corrected in a way that is
exactly justified.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from ..coreg.freesurfer import _freesurfer_env
from .base import CorrectedRecon, FakeGeometry

HEMIS = ("lh", "rh")

#: Volumes whose voxel values are intensity- or label-derived on the untouched
#: grid, so only the affine needs fixing. Ordered roughly by pipeline stage.
VOLUMES = (
    "orig.mgz",             # the native voxels, conformed as a no-op
    "nu.mgz",               # bias-corrected
    "T1.mgz",               # intensity-normalised
    "brainmask.mgz",        # skull-stripped (bbregister's default target)
    "brain.finalsurfs.mgz",
    "wm.mgz",
    "aseg.mgz",             # subcortical labels
    "aparc+aseg.mgz",       # + cortical parcellation (Desikan-Killiany)
    "aparc.a2009s+aseg.mgz",  # + cortical parcellation (Destrieux)
    "wmparc.mgz",           # white matter parcellated by overlying gyrus
    "ribbon.mgz",
)

#: Surfaces whose vertices sit in anatomical space and so transform exactly.
SURFACES = ("white", "pial")

#: Surfaces copied byte-for-byte -- spherical parameterisations, not
#: anatomical coordinates.
VERBATIM_SURFACES = ("sphere.reg",)

#: Why each omitted class is omitted. Written into the MANIFEST.
EXCLUDED = {
    "?h.thickness, ?h.area, ?h.volume, ?h.curv, ?h.sulc":
        "computed by FreeSurfer in the fake metric. The scale error is "
        "anisotropic while these are measured along local normals/tangents, "
        "so no scalar rescales them. Recomputed versions are provided under "
        "morphometry/.",
    "*.stats (aseg.stats, ?h.aparc.stats, wmparc.stats)":
        "aggregate the above in the fake metric. Recomputed aseg volumes are "
        "provided as stats/aseg_volumes.tsv; cortical stats are not "
        "reproduced because they depend on FreeSurfer's own thickness "
        "definition rather than a quantity recoverable from coordinates.",
    "?h.inflated, ?h.sphere, ?h.orig, ?h.smoothwm":
        "derived shapes in the fake metric, not anatomical coordinates -- M "
        "does not apply to them and they are not reproduced here. Use the "
        "uncorrected recon if you need them.",
    "transforms/ (talairach*, synthmorph*)":
        "estimated in the fake space. An affine registration absorbs the "
        "scale error, which is why aseg labels remain valid, but the "
        "transforms themselves are not true-space Talairach transforms.",
}


def vox2ras_tkr(shape, zooms) -> np.ndarray:
    """FreeSurfer's tkreg vox2ras: built from dims and voxel sizes only."""
    ds = np.asarray(zooms, dtype=float)
    ns = np.asarray(shape, dtype=float) * ds / 2.0
    return np.array([
        [-ds[0], 0, 0, ns[0]],
        [0, 0, ds[2], -ns[2]],
        [0, -ds[1], 0, ns[1]],
        [0, 0, 0, 1],
    ])


def vertex_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Barycentric vertex area: a third of each incident triangle."""
    tri = v[f]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(v))
    for k in range(3):
        np.add.at(out, f[:, k], area / 3.0)
    return out


def _correct_volumes(src_dir: Path, M: np.ndarray, out: Path, volumes):
    """Copy volumes with the affine rewritten. Voxel data is untouched."""
    (out / "mri").mkdir(parents=True, exist_ok=True)
    written, a_corr = [], None
    for name in volumes:
        src = src_dir / "mri" / name
        if not src.exists():
            continue
        img = nib.load(src)
        new_affine = M @ img.affine
        # MGH stores direction cosines + voxel sizes + c_ras, so it represents
        # an oblique anisotropic grid natively. The data array passes through.
        nib.save(nib.MGHImage(np.asanyarray(img.dataobj), new_affine),
                 out / "mri" / name)
        written.append(name)
        if a_corr is None:
            a_corr = new_affine
    if a_corr is None:
        raise FileNotFoundError(
            f"none of {list(volumes)} exist under {src_dir / 'mri'}")
    return written, a_corr


def _correct_surfaces(src_dir: Path, M: np.ndarray, out: Path, prefix: str,
                      surfaces, verbatim):
    """Rewrite surfaces in the corrected tkreg, and export true-space GIfTI."""
    (out / "surf").mkdir(parents=True, exist_ok=True)
    (out / "surf_gifti").mkdir(parents=True, exist_ok=True)

    fake_orig = nib.load(src_dir / "mri" / "orig.mgz")
    corr_orig = nib.load(out / "mri" / "orig.mgz")
    tkr_fake = vox2ras_tkr(fake_orig.shape[:3], fake_orig.header.get_zooms()[:3])
    tkr_corr = vox2ras_tkr(corr_orig.shape[:3], corr_orig.header.get_zooms()[:3])
    # s_new = tkr_corr @ inv(tkr_fake) @ s_old keeps the surfaces consistent
    # with the corrected volumes under FreeSurfer's own convention...
    S = tkr_corr @ np.linalg.inv(tkr_fake)
    # ...and this puts a fake-tkreg vertex straight into TRUE scanner RAS.
    tk2true = M @ fake_orig.affine @ np.linalg.inv(tkr_fake)

    written, geom = [], {}
    for hemi in HEMIS:
        for name in surfaces:
            src = src_dir / "surf" / f"{hemi}.{name}"
            if not src.exists():
                continue
            v, f = nib.freesurfer.read_geometry(str(src))
            h = np.hstack([v, np.ones((len(v), 1))])

            nib.freesurfer.write_geometry(
                str(out / "surf" / f"{hemi}.{name}"), (S @ h.T).T[:, :3], f)
            written.append(f"{hemi}.{name}")

            true_xyz = (tk2true @ h.T).T[:, :3]
            label = "L" if hemi == "lh" else "R"
            nib.save(nib.gifti.GiftiImage(darrays=[
                nib.gifti.GiftiDataArray(true_xyz.astype(np.float32),
                                         intent="NIFTI_INTENT_POINTSET"),
                nib.gifti.GiftiDataArray(f.astype(np.int32),
                                         intent="NIFTI_INTENT_TRIANGLE"),
            ]), out / "surf_gifti" /
                f"{prefix}_hemi-{label}_space-scanner_{name}.surf.gii")
            geom[(hemi, name)] = (true_xyz, f)

    for hemi in HEMIS:
        for name in verbatim:
            src = src_dir / "surf" / f"{hemi}.{name}"
            if src.exists():
                shutil.copy2(src, out / "surf" / f"{hemi}.{name}")
                written.append(f"{hemi}.{name} (verbatim)")
    return written, geom


def _freesurfer_thickness(subject: str, out: Path, freesurfer_home) -> list:
    """Run FreeSurfer's own mris_thickness on the corrected surfaces.

    Worth attempting because it gives thickness in FreeSurfer's definition --
    the symmetric closest-distance between the surfaces, which is what every
    other study reports -- but now measured in true geometry.

    Non-fatal: if it will not run against a corrected (oblique, anisotropic)
    subject directory, the file is simply absent rather than wrong.
    """
    env = _freesurfer_env(freesurfer_home, out.parent)
    written = []
    for hemi in HEMIS:
        dst = out / "surf" / f"{hemi}.thickness"
        proc = subprocess.run(["mris_thickness", subject, hemi, str(dst)],
                              capture_output=True, text=True,
                              env={**os.environ, **env})
        if proc.returncode == 0 and dst.exists():
            written.append(f"{hemi}.thickness")
        else:
            dst.unlink(missing_ok=True)
    return written


def _recompute_morphometry(geom: dict, out: Path, prefix: str) -> dict:
    """Thickness, area and volume from the corrected coordinates."""
    (out / "morphometry").mkdir(parents=True, exist_ok=True)
    summary = {}
    for hemi in HEMIS:
        if (hemi, "white") not in geom or (hemi, "pial") not in geom:
            continue
        w, f = geom[(hemi, "white")]
        p, _ = geom[(hemi, "pial")]
        label = "L" if hemi == "lh" else "R"

        # Thickness as the white->pial vertex displacement. This is the metric
        # the ribbon sampling itself uses (p(f) = white + f*(pial - white)),
        # and unlike mris_thickness it is recoverable exactly from
        # coordinates. It is NOT identical to ?h.thickness.
        thickness = np.linalg.norm(p - w, axis=1)
        area_w, area_p = vertex_areas(w, f), vertex_areas(p, f)
        volume = vertex_areas((w + p) / 2.0, f) * thickness

        for arr, name in ((thickness, "thickness"), (area_w, "area"),
                          (area_p, "area-pial"), (volume, "volume")):
            nib.save(nib.gifti.GiftiImage(darrays=[
                nib.gifti.GiftiDataArray(arr.astype(np.float32),
                                         intent="NIFTI_INTENT_SHAPE")]),
                out / "morphometry" / f"{prefix}_hemi-{label}_{name}.shape.gii")

        summary[hemi] = {
            "n_vertices": int(len(w)),
            "thickness_mean_mm": float(thickness.mean()),
            "thickness_median_mm": float(np.median(thickness)),
            "white_area_mm2": float(area_w.sum()),
            "pial_area_mm2": float(area_p.sum()),
            "cortical_volume_mm3": float(volume.sum()),
        }
    return summary


def _recompute_aseg_volumes(out: Path, voxel_mm3: float,
                            freesurfer_home) -> int:
    """Label volumes = voxel count x TRUE voxel volume."""
    aseg_path = out / "mri" / "aseg.mgz"
    if not aseg_path.exists():
        return 0
    labels = np.asanyarray(nib.load(aseg_path).dataobj).astype(int)

    lut, home = {}, Path(freesurfer_home or os.environ.get("FREESURFER_HOME", ""))
    lut_file = home / "FreeSurferColorLUT.txt"
    if lut_file.exists():
        for line in lut_file.read_text().splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                lut[int(parts[0])] = parts[1]

    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    (out / "stats").mkdir(parents=True, exist_ok=True)
    with open(out / "stats" / "aseg_volumes.tsv", "w") as fh:
        fh.write("label_id\tlabel_name\tn_voxels\tvolume_mm3\n")
        for i, c in zip(ids, counts):
            fh.write(f"{i}\t{lut.get(int(i), 'unknown')}\t{c}\t"
                     f"{c * voxel_mm3:.3f}\n")
    return len(ids)


def correct(recon, out_dir, *, geometry=None, prefix: str | None = None,
            volumes=VOLUMES, surfaces=SURFACES, verbatim=VERBATIM_SURFACES,
            freesurfer_home=None, run_mris_thickness: bool = True,
            overwrite: bool = True) -> CorrectedRecon:
    """Write a true-geometry derivative folder from a fake-geometry recon.

    Args:
        recon: A :class:`~lightprep.recon.base.ReconResult` from
            :func:`lightprep.recon.native`, or the subject directory itself.
        out_dir: The corrected subject directory to create.
        geometry: The :class:`~lightprep.recon.base.FakeGeometry`. Defaults to
            the ``ReconResult``'s, or to ``native2true.json`` in the subject
            directory.
        prefix: Filename prefix for the GIfTI outputs. Defaults to the subject
            name; pass a BIDS label such as ``sub-01`` to match a dataset.
        volumes: Volume filenames to correct. See :data:`VOLUMES`.
        surfaces: Surfaces to transform. See :data:`SURFACES`.
        verbatim: Surfaces to copy unchanged. See :data:`VERBATIM_SURFACES`.
        freesurfer_home: FreeSurfer install. Defaults to $FREESURFER_HOME.
        run_mris_thickness: Also attempt FreeSurfer's own thickness on the
            corrected surfaces. Non-fatal if it refuses.
        overwrite: Replace ``out_dir`` if it exists.

    Returns:
        A :class:`~lightprep.recon.base.CorrectedRecon`.

    Raises:
        FileNotFoundError: If the recon or its geometry cannot be found.
    """
    if hasattr(recon, "subject_dir"):
        subject, src_dir = recon.subject, Path(recon.subject_dir)
        geometry = geometry or recon.geometry
    else:
        src_dir = Path(recon)
        subject = src_dir.name
    if not src_dir.is_dir():
        raise FileNotFoundError(f"no such subject directory: {src_dir}")
    if geometry is None:
        meta = src_dir / "native2true.json"
        if not meta.exists():
            raise FileNotFoundError(
                f"no geometry given and no native2true.json in {src_dir}. "
                f"This folder does not look like a fake-geometry recon.")
        geometry = FakeGeometry.from_json(meta)

    prefix = prefix or subject
    out = Path(out_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    vols, a_corr = _correct_volumes(src_dir, geometry.M, out, volumes)
    surfs, geom = _correct_surfaces(src_dir, geometry.M, out, prefix,
                                    surfaces, verbatim)

    # Annotations are per-vertex labels; no geometry, valid as they stand.
    (out / "label").mkdir(exist_ok=True)
    n_annot = 0
    for src in sorted((src_dir / "label").glob("*.annot")):
        shutil.copy2(src, out / "label" / src.name)
        n_annot += 1

    if run_mris_thickness:
        _freesurfer_thickness(subject, out, freesurfer_home)

    morph = _recompute_morphometry(geom, out, prefix)
    voxel_mm3 = float(abs(np.linalg.det(a_corr[:3, :3])))
    _recompute_aseg_volumes(out, voxel_mm3, freesurfer_home)

    geometry.to_json(out / "native2true.json")
    manifest = out / "MANIFEST.md"
    manifest.write_text(_manifest(subject, geometry, vols, surfs, n_annot,
                                  morph, voxel_mm3))

    return CorrectedRecon(
        subject=subject, out_dir=out, volumes=tuple(vols),
        surfaces=tuple(surfs), annotations=n_annot, morphometry=morph,
        voxel_volume_mm3=voxel_mm3, manifest=manifest)


def _manifest(subject, geometry, vols, surfs, n_annot, morph, voxel_mm3) -> str:
    n_verbatim = sum(1 for s in surfs if "verbatim" in s)
    lines = [
        f"# {subject} -- corrected FreeSurfer derivatives (true geometry)",
        "",
        "Produced by `lightprep.recon.correct` from a fake-geometry recon,",
        "which was run on a volume carrying a deliberately false affine so",
        "that FreeSurfer's conform step was a bit-exact no-op and the anatomy",
        "was never interpolated.",
        "",
        f"Correction `M = A_true @ inv(A_fake)`, scale "
        f"`{[round(s, 6) for s in geometry.scale]}`.",
        "",
        "## Included, and why it is valid",
        "",
        "| item | count | basis |",
        "| --- | --- | --- |",
        f"| `mri/*.mgz` | {len(vols)} | voxel data computed on the untouched "
        "native array; only the affine was wrong, and it is rewritten to "
        "`M @ A_fake`. No resampling. |",
        f"| `surf/?h.sphere.reg` | {n_verbatim} | spherical parameterisation, "
        "not an anatomical coordinate: identical in the uncorrected and "
        "corrected folders. |",
        f"| `surf/?h.{{white,pial}}` | {len(surfs) - n_verbatim} | vertices are "
        "anatomical coordinates, so `M` applies exactly. Rewritten into the "
        "corrected tkreg (`tkr_corr @ inv(tkr_fake)`) so they stay consistent "
        "with the corrected volumes -- this is what lets bbregister run "
        "here. |",
        "| `surf_gifti/*.surf.gii` | 4 | the same surfaces in true scanner "
        "RAS, for analysis and visualisation. |",
        f"| `label/*.annot` | {n_annot} | per-vertex labels; carry no "
        "geometry. |",
        "| `morphometry/*.shape.gii` | 8 | recomputed from corrected "
        "coordinates. |",
        f"| `stats/aseg_volumes.tsv` | 1 | label voxel counts x the true voxel "
        f"volume ({voxel_mm3:.6f} mm3). |",
        "",
        "## Recomputed morphometry",
        "",
        "`thickness` is the white->pial displacement per corresponding "
        "vertex,",
        "`||pial_v - white_v||`. This is the metric the ribbon sampling "
        "itself",
        "uses (`p(f) = white + f*(pial - white)`) and is recoverable exactly",
        "from coordinates. It is **not** FreeSurfer's `?h.thickness`, which "
        "is a",
        "symmetric closest-distance measure and is generally slightly "
        "smaller.",
        "Areas are barycentric; `volume` is mid-surface area x thickness.",
        "",
    ]
    for hemi, s in morph.items():
        lines.append(f"- **{hemi}**: {s['n_vertices']} vertices, thickness "
                     f"{s['thickness_mean_mm']:.3f} mm mean / "
                     f"{s['thickness_median_mm']:.3f} median, white area "
                     f"{s['white_area_mm2'] / 100:.0f} cm2, cortical volume "
                     f"{s['cortical_volume_mm3'] / 1000:.1f} cm3")
    lines += ["", "## Deliberately excluded", ""]
    for what, why in EXCLUDED.items():
        lines.append(f"- **`{what}`** -- {why}")
    lines += [
        "",
        "The uncorrected recon is internally consistent; use it if you need",
        "anything not listed above, bearing in mind its metric is wrong by",
        "the scale given.",
        "",
    ]
    return "\n".join(lines) + "\n"
