"""MEDIC with niimath, via ``--medic``.

Same method as :mod:`lightprep.sdc.medic` -- a B0 field map fitted per frame from
multi-echo phase -- but computed by niimath rather than by warpkit. niimath's is a clean-room reimplementation from the paper (Van
et al., *Imaging Neuroscience* 4, 2026), with the MIT ROMEO port doing the phase
unwrapping, so it carries no GPL-licensed code and needs nothing installed.

It is also *much* faster: 21 seconds for the pilot's 138-frame three-echo run,
against minutes for warpkit, single-threaded either way.

The two implementations agree closely. On the pilot's three-echo run their
native field maps correlate 0.954, with a median voxelwise difference of 0.06 Hz
against a field standard deviation of 23.9 Hz (p95 9.5 Hz). Measured against the
session's own GRE phase-difference fieldmap -- an independent measurement of the
same B0 -- warpkit reaches r=0.835 and niimath r=0.790, so warpkit remains
marginally the more faithful of the two; the gap is small enough that speed and
the absent dependency generally win, which is why this is the default.

**One thing is not a silent drop-in, and is handled here rather than left to the
caller: sign.** niimath and warpkit use opposite polarity for the displacement
map given the same BIDS ``PhaseEncodingDirection``. With echoes correctly paired
to their echo times, raw ``niimath --medic ... -phase-encoding-direction j``
output correlates -0.948 with warpkit's and -0.795 with what
:mod:`lightprep.sdc.phasediff` derives from the GRE fieldmap. The rest of this
package is built around the latter convention, so this module hands niimath the
*opposite* polarity and what lands on disk is in lightprep's convention (then
+0.917 against warpkit). Feed a hand-rolled ``niimath --medic`` output straight
into :func:`lightprep.resample.compose_and_apply` and it will add the distortion
rather than remove it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .._niimath import niimath
from .base import SDCResult
from .medic import MIN_ECHOES, _check_inputs

#: ROMEO weighting presets niimath accepts.
WEIGHTS = ("romeo", "romeo2", "romeo3", "romeo4", "romeo6")

#: Phase-offset correction modes.
PHASE_OFFSETS = ("mcpc", "none")

PE_DIRECTIONS = ("i", "j", "k", "i-", "j-", "k-")


def _flip_polarity(pe_direction: str) -> str:
    """The opposite phase-encoding polarity.

    niimath's displacement map runs the other way from warpkit's and from
    :mod:`lightprep.sdc.phasediff` for a given direction, so the direction is
    flipped on the way in and the output lands in this package's convention.
    """
    return pe_direction[:-1] if pe_direction.endswith("-") else pe_direction + "-"


def _read_metadata(meta, echo_times_ms, total_readout_time, pe_direction):
    """Echo times, readout and PE direction, from the sidecars or the caller."""
    docs = [json.loads(Path(m).read_text()) for m in meta]

    if echo_times_ms is None:
        missing = [str(m) for m, d in zip(meta, docs) if "EchoTime" not in d]
        if missing:
            raise ValueError("no EchoTime in sidecar(s): " + ", ".join(missing))
        echo_times_ms = [float(d["EchoTime"]) * 1000.0 for d in docs]
    echo_times_ms = [float(t) for t in echo_times_ms]
    if len(echo_times_ms) != len(meta):
        raise ValueError(
            f"{len(echo_times_ms)} echo times for {len(meta)} echoes"
        )
    if len(set(echo_times_ms)) != len(echo_times_ms):
        raise ValueError(f"echo times must be distinct, got {echo_times_ms}")

    if total_readout_time is None:
        total_readout_time = docs[0].get("TotalReadoutTime")
        if total_readout_time is None:
            raise ValueError(
                f"no TotalReadoutTime in {Path(meta[0]).name}; pass "
                "total_readout_time= explicitly"
            )
    total_readout_time = float(total_readout_time)
    if total_readout_time <= 0:
        raise ValueError(
            f"total_readout_time must be positive, got {total_readout_time}"
        )

    if pe_direction is None:
        pe_direction = docs[0].get("PhaseEncodingDirection")
        if pe_direction is None:
            raise ValueError(
                f"no PhaseEncodingDirection in {Path(meta[0]).name}; pass "
                "pe_direction= explicitly"
            )
    if pe_direction not in PE_DIRECTIONS:
        raise ValueError(
            f"pe_direction must be one of {PE_DIRECTIONS}, got {pe_direction!r}"
        )
    return echo_times_ms, total_readout_time, pe_direction


def _find_output(prefix: Path, suffix: str) -> Path:
    """Locate one of niimath's three outputs, whichever extension it chose."""
    for ext in (".nii.gz", ".nii"):
        candidate = prefix.with_name(prefix.name + f"_{suffix}{ext}")
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"niimath --medic did not write {prefix.name}_{suffix}.nii[.gz]"
    )


def medic(
    magnitude,
    phase,
    out_dir,
    *,
    metadata=None,
    echo_times_ms=None,
    total_readout_time=None,
    pe_direction=None,
    noise_frames: int = 0,
    n_cpus: int = 4,
    rank: int = 10,
    temporal_correction: bool = True,
    phase_offset: str = "mcpc",
    weights: str = "romeo4",
    mask=None,
    save_intermediates: bool = False,
    prefix: str = "medic",
) -> SDCResult:
    """Estimate framewise field maps from multi-echo magnitude and phase.

    Args:
        magnitude: Magnitude echoes. Any order -- see ``echo_times_ms``.
        phase: Phase echoes, in the same echo order as ``magnitude``.
        out_dir: Directory for the field maps and displacement maps.
        metadata: One BIDS sidecar per echo. Defaults to the JSON beside each
            magnitude file.
        echo_times_ms: Echo times in milliseconds, matching the input order.
            Defaults to EchoTime from each sidecar. The echoes are sorted into
            ascending echo time before niimath sees them -- it requires that,
            and a BIDS ``echo-N`` entity does not always ascend with TE.
        total_readout_time: Seconds. Defaults to the first sidecar's.
        pe_direction: BIDS PhaseEncodingDirection, e.g. ``j`` or ``j-``.
            Defaults to the first sidecar's. Polarity matters: it sets the sign
            of the displacement map.
        noise_frames: Frames to drop from the end of the outputs (Siemens noise
            frames, if your protocol appends them).
        n_cpus: OpenMP threads, if niimath was built with OpenMP (the reference
            build in BUILD.md is), in which case this
            does take effect -- the full pilot run took roughly 12s against 22s
            single-threaded, though the machine was not idle when that was
            measured.
        rank: Low-rank truncation of the field-map series; 0 disables it. 10 is
            what the paper specifies, and what niimath flags as the least
            reference-faithful stage of its pipeline. On the pilot run it barely
            matters: rank 0 moves agreement with warpkit by about 0.02.
        temporal_correction: Temporal 2*pi consistency correction.
        phase_offset: MCPC-3D-S phase-offset correction, one of
            :data:`PHASE_OFFSETS`.
        weights: ROMEO weighting preset, one of :data:`WEIGHTS`.
        mask: Use this mask verbatim for both unwrapping stages, instead of
            ROMEO's robust mask of the first echo's magnitude.
        save_intermediates: Also write per-echo unwrapped phase and the masks.
        prefix: Basename stem for the written maps.

    Returns:
        An :class:`~lightprep.sdc.base.SDCResult`, with the displacement map in
        this package's sign convention rather than niimath's raw one -- see the
        module docstring.

    Raises:
        ValueError: On fewer than two echoes, or mismatched inputs.
        DependencyError: If no niimath binary can be found.

    Note:
        Pass the *raw* phase, not motion-corrected phase: niimath rescales phase
        to radians itself, and it needs the phase in its native, still-distorted
        space to model the field for that frame.
    """
    if phase_offset not in PHASE_OFFSETS:
        raise ValueError(
            f"phase_offset must be one of {PHASE_OFFSETS}, got {phase_offset!r}"
        )
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    if noise_frames < 0:
        raise ValueError(f"noise_frames must be non-negative, got {noise_frames}")

    mag, pha, meta, n_frames = _check_inputs(magnitude, phase, metadata)
    echo_times_ms, total_readout_time, pe_direction = _read_metadata(
        meta, echo_times_ms, total_readout_time, pe_direction
    )

    # niimath requires strictly increasing echo times, and rejects the call
    # outright otherwise. Nothing guarantees the caller's order ascends -- the
    # BIDS echo-N entity usually does, but it is a label, not a measurement --
    # so sort on the echo times themselves. Which echo comes first is only
    # bookkeeping: the fit is over all of them, and the pairing of each image
    # with its own TE is what has to hold.
    order = sorted(range(len(mag)), key=lambda i: echo_times_ms[i])
    mag = [mag[i] for i in order]
    pha = [pha[i] for i in order]
    echo_times_ms = [echo_times_ms[i] for i in order]

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / prefix

    args = [
        "--medic",
        "--magnitude", *mag,
        "--phase", *pha,
        "--te-ms", ",".join(f"{t:g}" for t in echo_times_ms),
        "--total-readout-time", total_readout_time,
        # Flipped on purpose: see the module docstring on sign.
        "--phase-encoding-direction", _flip_polarity(pe_direction),
        "--out-prefix", out_prefix,
        "--rank", rank,
        "--temporal-correction", int(temporal_correction),
        "--phase-offset", phase_offset,
        "--weights", weights,
        "--noise-frames", noise_frames,
        "--n-cpus", n_cpus,
        "--gz", 1,
    ]
    if mask is not None:
        args += ["--mask", Path(mask).resolve()]
    if save_intermediates:
        args.append("--save-intermediates")
    niimath(*args)

    return SDCResult(
        fieldmap_native=_find_output(out_prefix, "fieldmaps_native"),
        displacement_map=_find_output(out_prefix, "displacementmaps"),
        fieldmap=_find_output(out_prefix, "fieldmaps"),
        method="niimath",
        n_echoes=len(mag),
        n_frames=n_frames - noise_frames,
        # Each frame's field comes from that frame's own phase, in the space it
        # was acquired in -- so it is applied before the head is moved.
        space="native",
    )


__all__ = ["medic", "MIN_ECHOES", "WEIGHTS", "PHASE_OFFSETS", "PE_DIRECTIONS"]
