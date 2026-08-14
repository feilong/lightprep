"""FreeSurfer reconstruction, with control over what conform resamples.

``recon-all`` conforms every input to an isotropic, axis-aligned grid before it
does anything else. For a 1 mm isotropic axis-aligned T1 that costs nothing;
for anything else -- an anisotropic acquisition, a tilted prescription, the
0.7 mm submillimetre scans that come off a 7T -- it is an interpolation, and
every surface, label and statistic afterwards describes the resampled copy
rather than what was acquired.

Four methods, sharing a signature and returning a :class:`ReconResult`::

    from lightprep import recon

    result = recon.auto(t1, "sub-01", subjects_dir)        # picks for you
    corrected = recon.correct(result, "derivatives/fs-corrected/sub-01")

    # ...or name one, e.g. from a config file:
    result = recon.get_method("hires")(t1, "sub-01", subjects_dir)

=========  ==========================  ==============  ====================
method     conform target              interpolates?   use when
=========  ==========================  ==============  ====================
``auto``   whichever the geometry       only if it     the default: you have
           calls for                    has to         not measured the
                                                       voxels yourself
``std``    1 mm isotropic              yes             the input is already
                                                       1 mm isotropic, or you
                                                       want the conventional
                                                       output
``hires``  smallest native voxel       yes             submillimetre and
                                                       isotropic input
``native`` none -- made a no-op        **no**          anything anisotropic
                                                       or obliquely
                                                       prescribed
=========  ==========================  ==============  ====================

``auto`` is the default because the right answer is a property of the data,
not of the study. It measures ``max/min`` voxel size and uses ``native`` up to
:data:`~lightprep.recon.auto.MAX_ANISOTROPY` (1.2), beyond which the fake
isotropic grid would distort FreeSurfer's *view* of the head by more than one
interpolation costs -- so it warns and falls back to ``hires``. Every part of
that is overridable: ``method=`` forces a choice, ``max_anisotropy=`` moves
the threshold, ``fallback=`` changes what it retreats to. Forcing
``method="native"`` still warns, because asking for the distortion does not
remove it.

``native`` works by handing FreeSurfer a volume that is already on the conform
grid, carrying a deliberately false affine, so conform becomes the identity;
:mod:`lightprep.recon.fake` explains the mechanism and what it costs. Its
output is in a space whose metric is wrong, so it is only half a workflow:
:func:`correct` undoes the fake geometry analytically and writes a derivative
folder holding only what survives that exactly.

Field strength is a parameter rather than a hardcoded ``-3T``; see
:func:`~lightprep.recon.freesurfer.field_strength_args` for what each setting
does, and note that FreeSurfer has no 7T preset.

To add a method, drop a module here that returns a :class:`ReconResult` and
register it in :data:`METHODS`.
"""

from .base import CONFORM_FLAGS, CorrectedRecon, FakeGeometry, ReconResult
from .correct import (EXCLUDED, SURFACES, VERBATIM_SURFACES, VOLUMES, correct,
                      vertex_areas, vox2ras_tkr)
from .freesurfer import HIRES_EXPERT, field_strength_args, hires, std
from .fake import (check_conform_lossless, conform_target, fake_input,
                   native)
from .auto import (ANISOTROPY_TOL, DEFAULT_FALLBACK, MAX_ANISOTROPY,
                   AnisotropyWarning, anisotropy, auto, choose_method, within)

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "auto": auto,        # native, or hires if the voxels are too anisotropic
    "std": std,          # 1mm conform (interpolates)
    "hires": hires,      # conform to the min native voxel (interpolates)
    "native": native,    # conform made a bit-exact no-op
}

#: Used when no method is named.
DEFAULT_METHOD = "auto"


def get_method(name: str | None = None):
    """Look up a recon method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown recon method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["ReconResult", "FakeGeometry", "CorrectedRecon", "CONFORM_FLAGS",
           "auto", "std", "hires", "native", "correct",
           "anisotropy", "choose_method", "AnisotropyWarning",
           "MAX_ANISOTROPY", "DEFAULT_FALLBACK", "ANISOTROPY_TOL", "within",
           "conform_target", "fake_input", "check_conform_lossless",
           "field_strength_args", "HIRES_EXPERT",
           "VOLUMES", "SURFACES", "VERBATIM_SURFACES", "EXCLUDED",
           "vox2ras_tkr", "vertex_areas",
           "METHODS", "DEFAULT_METHOD", "get_method"]
