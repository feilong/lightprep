"""The contract every HMC method implements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HMCResult:
    """What every HMC method returns, whichever tool produced it.

    Attributes:
        outputs: Realigned echoes, in the same order as the inputs.
        reference: The target volume that motion was estimated against.
        transforms: Per-volume rigid transforms, in acquisition order. One
            series of transforms, shared by every echo.
        parameters: Six-column motion trace, if the method produces one. This
            is the float64 ``.npy``; a text twin sits beside it under the same
            stem, for reading rather than parsing. Either can be handed to
            :func:`lightprep._utils.load_trace`, which prefers the ``.npy``.
            Ordering and units are tool-specific -- see the method's docstring.
        ref_echo: Index of the echo motion was estimated on.
        method: Name of the method that produced this result.
    """

    outputs: tuple[Path, ...]
    reference: Path
    transforms: tuple[Path, ...]
    parameters: Path | None
    ref_echo: int
    method: str

    @property
    def n_volumes(self) -> int:
        return len(self.transforms)


class TransformReplayError(RuntimeError):
    """A result's transforms will not resample the data faithfully."""


#: Methods whose saved transforms do not reproduce the resampling their own tool
#: performs, and so must not be replayed onto other data.
#:
#: ``fsl`` is here because MCFLIRT's ``-mats`` do not reproduce MCFLIRT's own
#: ``-out``. Measured on the pilot's 138-frame run under FSL 6.0.7.22, within a
#: single invocation: ``-out`` lowers DVARS 7.7% and halves the centre-of-mass
#: trajectory spread (0.143mm -> 0.071mm), while replaying the ``.mat`` files it
#: wrote raises DVARS 55% and *doubles* that spread (to 0.279mm) -- worse than no
#: correction at all. It reproduces with ``-reffile`` and ``-refvol`` alike.
#:
#: The estimates are sound; only the replay is not. The matrices and the ``.par``
#: imply the same frame-to-frame displacement to within 2%, and ``applyxfm4D``
#: is faithful to whatever matrix it is handed (identity matrices return the
#: input bit-for-bit, and it matches ``flirt -applyxfm`` exactly). The defect is
#: in the correspondence between what MCFLIRT writes and what FSL reads back.
#:
#: :func:`lightprep.hmc.mcflirt` is left as it was and still replays these
#: internally; this set only stops them reaching a *different* step, where the
#: damage would be silent. Remove ``fsl`` from this set to override.
UNREPLAYABLE_METHODS = {"fsl"}


def check_transforms_replayable(hmc_result, *, step: str) -> None:
    """Raise if ``hmc_result``'s transforms must not be replayed.

    Args:
        hmc_result: The result whose ``transforms`` are about to be applied.
        step: What is about to use them, named for the error message.

    Raises:
        TransformReplayError: If the method is in :data:`UNREPLAYABLE_METHODS`.
    """
    method = getattr(hmc_result, "method", None)
    if method in UNREPLAYABLE_METHODS:
        raise TransformReplayError(
            f"{step} was handed transforms from the {method!r} HMC method, which "
            "cannot be replayed: MCFLIRT's saved .mat files do not reproduce "
            "MCFLIRT's own resampling, and applying them makes the data worse "
            "than leaving it uncorrected (see lightprep.hmc.base.UNREPLAYABLE_METHODS "
            "for the measurements). Re-run head motion correction with "
            "lightprep.hmc.moco (the default) or lightprep.hmc.allineate, whose "
            "transforms are built from niimath's own estimates and verified "
            f"against its resampling -- or, if you know what you are doing, "
            f"remove {method!r} from lightprep.hmc.base.UNREPLAYABLE_METHODS."
        )
