"""The contract every recon method implements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: ``--conform`` flag -> the mri_convert argument that selects it.
CONFORM_FLAGS = {
    "1mm": "--conform",       # FreeSurfer's default 1mm isotropic grid
    "min": "--conform_min",   # isotropic at the smallest native voxel size
}

#: How far the fake voxel size may sit from the true one and still be called
#: exact. Well below anything that matters anatomically.
METRIC_TOL = 1e-6


@dataclass(frozen=True)
class FakeGeometry:
    """The lie told to FreeSurfer, and the transform that undoes it.

    A volume is written carrying an affine equal to FreeSurfer's conform
    target, so that conform becomes a bit-exact no-op and the anatomy is never
    interpolated. :attr:`M` maps that fake world back to the true one.

    Attributes:
        M: 4x4. Fake scanner RAS -> true scanner RAS. Apply to surface
            vertices (exact: transforming points interpolates nothing), or to
            a volume affine as ``M @ affine`` (also exact: it moves where a
            grid sits in the world without touching a voxel).
        a_true_on_fake: The affine giving the TRUE world position of each
            voxel of the written volume.
        a_fake: The affine actually written into it -- the conform target.
        shape: Grid the volume was placed on.
        conform: Which conform target was used, a key of :data:`CONFORM_FLAGS`.
        scale: Column norms of ``M``: the per-axis metric error the trick
            introduces, which ``M`` removes from anything coordinate-based.
            Anisotropic in general, which is why FreeSurfer's own morphometry
            cannot simply be rescaled.
        pad: Voxels added before the data on each axis when it was centred on
            the target grid.
    """

    M: np.ndarray
    a_true_on_fake: np.ndarray
    a_fake: np.ndarray
    shape: tuple[int, int, int]
    conform: str
    scale: tuple[float, float, float]
    pad: tuple[int, int, int]

    @property
    def is_identity(self) -> bool:
        """True when ``M`` is the identity: same grid, same origin.

        Rare, and not the useful test -- centring the data on the conform grid
        introduces a translation even when the metric is perfect. See
        :attr:`metric_exact`.
        """
        return bool(np.allclose(self.M, np.eye(4), atol=1e-9))

    @property
    def scale_error(self) -> float:
        """How far the fake metric stretches the anatomy, as a fraction.

        ``0.0212`` means the fake grid claims a voxel size that is 2.12% wrong
        on its worst axis.
        """
        return float(np.max(np.abs(np.asarray(self.scale) - 1.0)))

    @property
    def metric_exact(self) -> bool:
        """True when the fake grid's voxel size matches the real one.

        This is what decides whether FreeSurfer's own morphometry survives.
        The fake grid must be isotropic, so an isotropic input gives an exact
        metric and only a rigid offset -- ``?h.thickness`` and the ``.stats``
        files are then valid as FreeSurfer wrote them. An anisotropic input
        cannot, and they must be recomputed from the corrected coordinates.
        """
        return self.scale_error <= METRIC_TOL

    def to_json(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps({
            "M": self.M.tolist(),
            "A_true_on_fake_grid": self.a_true_on_fake.tolist(),
            "A_fake": self.a_fake.tolist(),
            "conform_shape": list(self.shape),
            "conform": self.conform,
            "scale": list(self.scale),
            "pad": list(self.pad),
            "note": "M maps FAKE scanner RAS -> TRUE scanner RAS. Apply to "
                    "surface vertices, or to volume affines as M @ A.",
        }, indent=4) + "\n")
        return path

    @classmethod
    def from_json(cls, path) -> "FakeGeometry":
        meta = json.loads(Path(path).read_text())
        M = np.asarray(meta["M"], dtype=float)
        return cls(
            M=M,
            a_true_on_fake=np.asarray(meta["A_true_on_fake_grid"], dtype=float),
            a_fake=np.asarray(meta["A_fake"], dtype=float),
            shape=tuple(meta["conform_shape"]),
            conform=meta.get("conform", "min"),
            scale=tuple(meta.get(
                "scale", np.linalg.norm(M[:3, :3], axis=0).tolist())),
            pad=tuple(meta.get("pad", (0, 0, 0))),
        )


@dataclass(frozen=True)
class ReconResult:
    """Outcome of a FreeSurfer reconstruction.

    Attributes:
        subject: FreeSurfer subject name.
        subjects_dir: The SUBJECTS_DIR it was written into.
        method: Name of the method that produced this result.
        interpolated: Whether conform resampled the anatomy. ``False`` only
            for the ``native`` method.
        conform: The grid the recon ran on: ``1mm``, ``min``, or ``none`` when
            conform was made a no-op.
        geometry: For ``native``, the :class:`FakeGeometry` needed to undo the
            fake affine. ``None`` for the methods that do not fake one.
        input_volume: The volume actually handed to ``recon-all``.
    """

    subject: str
    subjects_dir: Path
    method: str
    interpolated: bool
    conform: str
    geometry: FakeGeometry | None = None
    input_volume: Path | None = None

    @property
    def subject_dir(self) -> Path:
        return Path(self.subjects_dir) / self.subject

    @property
    def complete(self) -> bool:
        return (self.subject_dir / "scripts" / "recon-all.done").exists()


@dataclass(frozen=True)
class CorrectedRecon:
    """A derivative folder holding only what is valid in true geometry.

    Attributes:
        subject: FreeSurfer subject name.
        out_dir: The corrected subject directory.
        volumes: Volume filenames copied with a rewritten affine.
        surfaces: Surface filenames written.
        annotations: Annotation files copied verbatim.
        morphometry: Per-hemisphere morphometry recomputed from the corrected
            coordinates.
        voxel_volume_mm3: True voxel volume, used for the label volumes.
        manifest: Path to the MANIFEST describing what is included and why.
    """

    subject: str
    out_dir: Path
    volumes: tuple[str, ...]
    surfaces: tuple[str, ...]
    annotations: int
    morphometry: dict
    voxel_volume_mm3: float
    manifest: Path
