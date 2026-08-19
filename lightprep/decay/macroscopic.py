"""Undoing the macroscopic field gradient's contribution to the measured decay.

Not all of the measured R2* belongs to the tissue. Split it three ways::

    R2*_measured = R2 + R2'_meso + R2'_macro

R2 is spin-spin relaxation, and R2'_meso is dephasing around structures far
smaller than a voxel -- capillaries carrying deoxyhaemoglobin, iron, myelin.
Both are properties of the tissue and both are scale-invariant: they do not care
how the voxel was drawn around them, and R2'_meso is the term BOLD actually
modulates. R2'_macro is a different animal. It is dephasing across the voxel
*itself*, driven by the background field that air-tissue interfaces impose on
the head, so it scales with the voxel rather than with anything biological. It
is why T2* is short in orbitofrontal and inferior temporal cortex, why a thinner
slice recovers signal there, and why a T2* reported without its voxel size is
not a number anyone can compare against.

This module estimates that term from a B0 fieldmap and divides it back out.

The model
---------

Let the background field vary linearly across the voxel. Every position in the
voxel then precesses at its own offset, and by echo time TE their phases have
fanned out; the receiver measures their vector sum, not their number. Averaging
``exp(2j*pi*f*TE)`` over a box of uniform spins gives a sinc::

    S(TE) = S0 * exp(-TE / T2*) * prod_a sinc(df_a * TE)

with ``sinc(x) = sin(pi*x) / (pi*x)`` -- numpy's convention, which is why no
factor of pi appears in the code below. ``df_a`` is the total frequency spread
across the voxel along axis ``a``, in Hz.

That spread needs no voxel dimension in millimetres. The fieldmap is sampled on
the voxel grid itself, so its finite difference along an axis is already Hz per
voxel step -- which is Hz across one voxel. A gradient in Hz/mm times a width in
mm is the same quantity computed the long way round.

Two consequences worth keeping in view. First, a sinc times an exponential is
not an exponential, so an uncorrected mono-exponential fit is not merely biased,
it is fitting the wrong function, and how wrong depends on which echo times were
sampled. Second, the first zero sits at ``df * TE = 1``: past it there is no
signal left to rescue, and magnitude data cannot even tell a side lobe from the
main one because the sign flip is discarded along with the phase. Vertices are
therefore refused below :data:`DEFAULT_MIN_FACTOR` rather than divided by a
number close to zero.

Which axes
----------

For EPI the answer is the slice axis, and only the slice axis. The three
directions of the readout do quite different things with a background gradient:

- **Frequency-encode.** The background gradient adds to a very strong readout
  gradient, and each line's k-space centre is still traversed, so the
  intravoxel phase is refocused at every echo in the train. What survives is a
  slight mis-sizing of the voxel, not signal loss.

- **Phase-encode.** The effective gradient along this axis is tiny, so the
  background field displaces signal a long way -- this is EPI's geometric
  distortion, and it is what distortion correction exists to undo. The
  intravoxel phase is still refocused as the trajectory crosses k_y = 0.
  Correcting it here as though it were dephasing would double-count the SDC
  step that already handled it.

- **Slice.** Nothing encodes along the slice axis during the readout, so the
  phase spread laid down at excitation simply accumulates with TE and is never
  refocused. This is the whole of the effect, and the reason z-shimming is a
  slice-direction trick.

:func:`voxel_spread` therefore defaults to the slice axis alone. The in-plane
axes remain available for non-EPI multi-echo data -- a multi-echo GRE, where the
argument above does not apply and all three terms are real.

Masked fieldmaps
----------------

A fieldmap is usually zero outside the brain, and the step from tissue to that
zero is a cliff a finite difference reads as an enormous gradient. Differencing
it unfilled would put a spurious correction on exactly the outermost voxel layer
the cortical ribbon lives in. The field is therefore continued outside its mask
by nearest neighbour before differencing, which makes the boundary gradient
roughly zero instead: the correction there under-reaches rather than inventing a
dropout that is not present.

References:
    Yablonskiy & Haacke (1994), Magn Reson Med 32:749 -- the static dephasing
    regime, and where the mesoscopic/macroscopic split comes from.

    Fernandez-Seara & Wehrli (2000), Magn Reson Med 44:358 -- correcting R2* for
    background gradients from an image-based field estimate.

    Dahnke & Schaeffter (2005), Magn Reson Med 53:1202 -- the same correction
    with the through-slice term treated properly.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

#: Which image axis is the slice axis when the header does not say. Every 2D
#: EPI this package has seen stores slices along k.
DEFAULT_SLICE_AXIS = 2

#: Refuse the correction below this dephasing factor. The division amplifies
#: noise by 1/F, so 0.3 already costs a factor of 3.3; more importantly, F is
#: only single-valued on the main lobe, and by the first zero (F = 0) the
#: measurement carries no recoverable signal at all.
DEFAULT_MIN_FACTOR = 0.3


def slice_axis(img, default: int = DEFAULT_SLICE_AXIS) -> int:
    """Which image axis slices were acquired along.

    Reads NIfTI's ``dim_info``, which a converter fills in from the DICOM and
    most other tools then drop when they write a derivative. Falls back to
    ``default`` when it is absent.
    """
    try:
        k = img.header.get_dim_info()[2]
    except (AttributeError, KeyError, ValueError):
        k = None
    return int(default if k is None else k)


def _fill_outside(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Continue ``field`` outside ``mask`` by nearest neighbour.

    So that the mask boundary does not read as a gradient; see the module
    docstring. Returns ``field`` untouched when the mask covers everything.
    """
    if mask.all():
        return field
    if not mask.any():
        raise ValueError("fieldmap mask is empty: no voxel carries a field estimate")
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def voxel_spread(field, axes=(DEFAULT_SLICE_AXIS,), mask=None) -> np.ndarray:
    """Frequency spread across one voxel, per axis, in Hz.

    Args:
        field: B0 fieldmap in Hz, 3D, sampled on the grid the data was acquired
            on. A 4D framewise map must be reduced to 3D first -- see
            :func:`static_field`.
        axes: Image axes to account for. The slice axis alone by default; see
            the module docstring for why in-plane axes do not belong here for
            EPI.
        mask: Where the field is an estimate rather than a fill value. Defaults
            to the finite, non-zero voxels.

    Returns:
        Array of shape ``field.shape + (len(axes),)``: the field difference
        across one voxel along each axis, in Hz.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError(f"fieldmap must be 3D, got shape {field.shape}")
    axes = tuple(int(a) for a in axes)
    if not axes:
        raise ValueError("need at least one axis to difference along")
    if any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"axes must index a 3D volume, got {axes}")

    if mask is None:
        mask = np.isfinite(field) & (field != 0.0)
    filled = _fill_outside(np.nan_to_num(field, nan=0.0), np.asarray(mask, dtype=bool))
    # Central differences: the slope over two voxels, which is noticeably
    # steadier than a forward difference on a fieldmap and is the local slope
    # the linear-across-the-voxel model assumes anyway.
    return np.stack([np.gradient(filled, axis=a) for a in axes], axis=-1)


def dephasing_factor(spread, te_ms: float) -> np.ndarray:
    """The signal that survives intravoxel dephasing at one echo time.

    Args:
        spread: Per-axis frequency spread across the voxel in Hz, as
            :func:`voxel_spread` returns it -- axes on the last dimension.
        te_ms: Echo time in milliseconds.

    Returns:
        The factor ``F`` multiplying ``S0 * exp(-TE/T2*)``, with the axis
        dimension contracted away. Signed: it goes negative beyond the first
        zero, which callers should treat as no measurement rather than as a
        small one.
    """
    spread = np.asarray(spread, dtype=np.float64)
    te_s = float(te_ms) / 1000.0
    # np.sinc(x) is sin(pi x)/(pi x), so the spread-time product goes in bare.
    return np.prod(np.sinc(spread * te_s), axis=-1)


def static_field(fieldmap) -> tuple[np.ndarray, "nib.Nifti1Image"]:
    """Read a fieldmap as a single 3D volume in Hz.

    MEDIC estimates the field per frame, so its map is 4D. The decay is fitted
    on time-averaged echoes, so what is wanted here is one static field: the
    temporal median, which ignores the occasional frame the phase unwrapping
    lost. Breathing moves the field by a few Hz across a run, against
    background gradients of tens of Hz across a voxel, so little is given up.
    """
    img = nib.load(str(fieldmap))
    data = img.get_fdata(dtype=np.float64)
    if data.ndim == 4:
        data = np.median(data, axis=3)
    elif data.ndim != 3:
        raise ValueError(f"fieldmap must be 3D or 4D, got shape {data.shape}")
    return data, img


def dephasing_volumes(
    fieldmap,
    echo_times_ms,
    out_dir,
    *,
    axes=None,
    reference=None,
) -> tuple[list[Path], Path, dict]:
    """Write one dephasing-factor volume per echo, on the fieldmap's grid.

    These are sampled onto the surface exactly as the echo data is, so that the
    factor divided out of a vertex is the one that applied to the voxels that
    vertex was drawn from. Averaging the factor over the ribbon and then
    dividing is not the same as dividing voxel by voxel -- the sinc is not
    linear -- but it is a far better approximation than evaluating the sinc at
    an averaged gradient, which is what dividing a per-vertex spread would do.

    Args:
        fieldmap: B0 fieldmap in Hz, 3D or framewise 4D.
        echo_times_ms: Echo times in milliseconds.
        out_dir: Where to write the volumes.
        axes: Image axes to account for. Defaults to the slice axis, taken from
            ``reference``'s header if it names one.
        reference: An image from the run, consulted only for its ``dim_info``.

    Returns:
        ``(factor_paths, spread_path, stats)``. ``stats`` reports the spread
        distribution and, per echo, the fraction of in-mask voxels the
        correction has to refuse.
    """
    field, img = static_field(fieldmap)
    if axes is None:
        axes = (slice_axis(reference if reference is not None else img),)
    axes = tuple(int(a) for a in axes)

    mask = np.isfinite(field) & (field != 0.0)
    spread = voxel_spread(field, axes=axes, mask=mask)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Total spread across the voxel, for QC: the quadrature sum over axes is
    # what sets how fast the sinc falls, when more than one axis is in play.
    total = np.sqrt((spread ** 2).sum(axis=-1))
    spread_path = out_dir / "voxel_spread_hz.nii.gz"
    nib.Nifti1Image(total.astype(np.float32), img.affine, img.header).to_filename(
        str(spread_path)
    )

    paths, refused = [], []
    for i, te in enumerate(echo_times_ms, start=1):
        f = dephasing_factor(spread, te)
        p = out_dir / f"dephasing_echo-{i}.nii.gz"
        nib.Nifti1Image(f.astype(np.float32), img.affine, img.header).to_filename(str(p))
        paths.append(p)
        refused.append(float(np.mean(f[mask] < DEFAULT_MIN_FACTOR)))

    stats = {
        "axes": list(axes),
        "echo_times_ms": [float(t) for t in echo_times_ms],
        "spread_hz_percentiles": {
            str(q): float(v)
            for q, v in zip(
                (50, 75, 90, 95, 99), np.percentile(np.abs(total[mask]), [50, 75, 90, 95, 99])
            )
        },
        "fraction_below_min_factor": refused,
        "min_factor": DEFAULT_MIN_FACTOR,
        "n_frames_reduced": int(nib.load(str(fieldmap)).ndim == 4),
    }
    return paths, spread_path, stats


__all__ = [
    "DEFAULT_MIN_FACTOR",
    "DEFAULT_SLICE_AXIS",
    "dephasing_factor",
    "dephasing_volumes",
    "slice_axis",
    "static_field",
    "voxel_spread",
]
