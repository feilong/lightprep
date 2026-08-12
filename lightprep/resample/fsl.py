"""Resampling into corrected space, via warpkit + FSL.

The point of this step is to spend exactly one interpolation. Distortion
correction and head-motion correction are each a resample; doing them in
sequence blurs the data twice. Instead the per-frame distortion warp and the
per-frame rigid transform are composed into a single warp, which is applied
once to the *original* data.

That matters most for multi-echo: T2*/S0 fitting reads voxelwise across echoes,
and repeated smoothing biases the fit differently at each echo.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import nibabel as nib

from .._utils import run, strip_ext
from ..hmc.base import check_transforms_replayable
from .base import ResampleResult

INTERPOLATIONS = ("nn", "trilinear", "sinc", "spline")
PE_AXES = ("x", "y", "z", "x-", "y-", "z-", "i", "j", "k", "i-", "j-", "k-")


def _check_axis(pe_axis: str) -> str:
    if pe_axis not in PE_AXES:
        raise ValueError(f"pe_axis must be one of {PE_AXES}, got {pe_axis!r}")
    return pe_axis


def apply_sdc(images, displacement_map, out_dir, *, pe_axis: str = "j") -> tuple[Path, ...]:
    """Undistort images with a framewise MEDIC displacement map.

    This exists to produce data to *estimate* motion on. The result is not
    meant to be the deliverable: it has been interpolated once already, so
    feeding it through HMC and keeping that output would cost a second pass.
    Use :func:`compose_and_apply` for the data you actually keep.

    Args:
        images: 4D timeseries to undistort.
        displacement_map: MEDIC's framewise displacement map. Frame i of each
            image is resampled with frame i of the map.
        out_dir: Where to write the undistorted images.
        pe_axis: Axis the displacement map runs along, from the sidecar's
            PhaseEncodingDirection.

    Returns:
        Paths of the undistorted images, in input order.
    """
    _check_axis(pe_axis)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    displacement_map = Path(displacement_map).resolve()

    outputs = []
    for img in images:
        img = Path(img).resolve()
        dst = out_dir / f"{strip_ext(img)}.nii.gz"
        if dst == img:
            raise ValueError(f"out_dir would overwrite the input: {img}")
        run([
            "wk-apply-warp",
            "--input", img,
            "--transform", displacement_map,
            "--transform-type", "map",
            "--phase-encoding-axis", pe_axis,
            "--output", dst,
        ])
        outputs.append(dst)
    return tuple(outputs)


def compose_and_apply(
    images,
    sdc_result,
    hmc_result,
    out_dir,
    *,
    pe_axis: str = "j",
    interp: str = "trilinear",
    keep_workdir: bool = False,
) -> ResampleResult:
    """Compose distortion + motion correction and apply them in one pass.

    Where the distortion warp belongs in that composition depends on which
    space the field was measured in -- ``sdc_result.space`` decides it::

        space="native"     (e.g. MEDIC, a per-frame field read from that
                            frame's own phase)
        acquired --[warp, frame t]--> undistorted --[rigid, frame t]--> reference

        space="reference"  (e.g. a static GRE fieldmap)
        acquired --[rigid, frame t]--> aligned --[warp]--> reference

    The second is not merely a convention. A susceptibility field is produced by
    the head's own tissue-air boundaries, so it travels with the head: under
    translation the field co-moves exactly. A fieldmap measured once therefore
    describes the field in the *head's* frame, and is only valid once motion
    correction has put the head back where it was measured. Applying it in the
    scanner frame instead would hold the field still while the anatomy that
    creates it moves.

    Args:
        images: The *original* echoes -- raw and distorted, not the undistorted
            copies used for estimation. Ordered by echo time.
        sdc_result: From :func:`lightprep.sdc.medic`, supplying the framewise
            displacement map.
        hmc_result: From an HMC method, supplying the per-frame rigid
            transforms and the reference grid. Estimate this on undistorted
            data so the rigid model isn't absorbing distortion changes.
        out_dir: Where to write the corrected images.
        pe_axis: Axis the displacement map runs along.
        interp: Interpolation for the single resample, one of
            :data:`INTERPOLATIONS`.
        keep_workdir: Keep the intermediate per-frame fields and warps.

    Returns:
        A :class:`~lightprep.resample.base.ResampleResult` with
        ``n_interpolations == 1``.

    Raises:
        ValueError: If the frame counts disagree or an argument is invalid.
        TransformReplayError: If ``hmc_result`` came from a method whose
            transforms do not replay faithfully -- see
            :data:`lightprep.hmc.base.UNREPLAYABLE_METHODS`.
    """
    _check_axis(pe_axis)
    if interp not in INTERPOLATIONS:
        raise ValueError(f"interp must be one of {INTERPOLATIONS}, got {interp!r}")
    # These transforms get composed into the one warp the data is resampled
    # through, so a method whose matrices do not replay would corrupt the whole
    # point of this step, silently.
    check_transforms_replayable(hmc_result, step="resample.compose_and_apply")

    images = [Path(p).resolve() for p in images]
    if not images:
        raise ValueError("no images given")

    n_frames = len(hmc_result.transforms)
    dmap = Path(sdc_result.displacement_map).resolve()
    dmap_shape = nib.load(dmap).shape
    # A framewise method (MEDIC) gives one map per frame; a static one
    # (a GRE fieldmap) gives a single map that applies to every frame.
    n_dmap = 1 if len(dmap_shape) == 3 else dmap_shape[3]
    if n_dmap not in (1, n_frames):
        raise ValueError(
            f"displacement map has {n_dmap} frames but HMC has {n_frames} "
            "transforms; a map must be either static (1) or framewise "
            f"({n_frames}) to describe this run"
        )
    for img in images:
        n_img = nib.load(img).shape[3]
        if n_img != n_frames:
            raise ValueError(
                f"{img.name} has {n_img} frames but the transforms describe {n_frames}"
            )

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    (work / "fields").mkdir(parents=True, exist_ok=True)
    (work / "warps").mkdir(parents=True, exist_ok=True)
    reference = Path(hmc_result.reference).resolve()

    # 1. 1-channel displacement map(s) -> 3-channel FSL field(s). A static map
    #    converts once and is then reused for every frame.
    if n_dmap == 1:
        field = work / "fields" / "field_static.nii.gz"
        outputs_arg = [field]
        fields = [field] * n_frames
    else:
        fields = [work / "fields" / f"field_{t:04d}.nii.gz" for t in range(n_frames)]
        outputs_arg = fields
    run([
        "wk-convert-warp",
        "--input", dmap,
        "--from", "map", "--to", "field",
        "--to-format", "fsl",
        "--axis", pe_axis,
        "--output", *outputs_arg,
    ])

    # 2. Fold each frame's rigid transform together with the distortion warp,
    #    in the way the field's space demands (see the docstring).
    space = getattr(sdc_result, "space", "native")
    if space not in ("native", "reference"):
        raise ValueError(
            f"sdc_result.space must be 'native' or 'reference', got {space!r}"
        )
    warps = []
    for field, mat, t in zip(fields, hmc_result.transforms, range(n_frames)):
        warp = work / "warps" / f"warp_{t:04d}.nii.gz"
        if space == "native":
            # The field describes this frame as acquired, so unwarp there and
            # move the head afterwards. --postmat leaves the shift on the
            # scanner PE axis, which is where the readout put it.
            run([
                "convertwarp",
                f"--ref={reference}",
                f"--warp1={field}",
                f"--postmat={mat}",
                f"--out={warp}",
                "--relout",
            ])
        else:
            # Two things must hold at once, and no single convertwarp
            # composition delivers both:
            #   * the shift's MAGNITUDE is head-fixed -- the field travels with
            #     the tissue boundaries that create it, so it is read at the
            #     reference (head) coordinate;
            #   * the shift's DIRECTION is scanner-fixed -- the readout always
            #     displaces along the PE axis, whatever the head is doing.
            # --premat would rotate the shift along with the head (a 30 deg
            # rotation tilts it a full 30 deg off PE). Since both are relative
            # displacement fields in the same frame, adding them gives exactly
            #     pull(x) = rigid(x) + phi(x) * TRT * e_PE
            # keeping the magnitude head-fixed and the direction on PE. For a
            # pure translation this reduces to --premat identically.
            rigid = work / "warps" / f"rigid_{t:04d}.nii.gz"
            run([
                "convertwarp",
                f"--ref={reference}",
                f"--premat={mat}",
                f"--out={rigid}",
                "--relout",
            ])
            run(["fslmaths", rigid, "-add", field, warp])
        warps.append(warp)

    # 3. One resample of the original data, through the combined warp.
    outputs = []
    for img in images:
        dst = out_dir / f"{strip_ext(img)}.nii.gz"
        if dst == img:
            raise ValueError(f"out_dir would overwrite the input: {img}")
        split = work / f"split_{strip_ext(img)}"
        split.mkdir(exist_ok=True)
        run(["fslsplit", img, f"{split}/vol", "-t"])
        frames = sorted(split.glob("vol*.nii.gz"))
        if len(frames) != n_frames:
            raise RuntimeError(f"fslsplit produced {len(frames)} frames, expected {n_frames}")
        corrected = []
        for frame, warp, t in zip(frames, warps, range(n_frames)):
            out = split / f"corr{t:04d}.nii.gz"
            run([
                "applywarp",
                "-i", frame,
                "-r", reference,
                "-w", warp,
                "-o", out,
                f"--interp={interp}",
            ])
            corrected.append(out)
        run(["fslmerge", "-t", dst, *corrected])
        outputs.append(dst)

    if not keep_workdir:
        shutil.rmtree(work, ignore_errors=True)
        warps = ()

    return ResampleResult(
        outputs=tuple(outputs),
        reference=reference,
        warps=tuple(warps),
        method="fsl",
        n_interpolations=1,
    )
