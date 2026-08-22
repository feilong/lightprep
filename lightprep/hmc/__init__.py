"""Head motion correction.

Every method in this step shares one signature::

    method(echoes, out_dir, *, ref=..., ref_echo=0, ...) -> HMCResult

``echoes`` is always a list ordered by echo time (length one for single-echo
data), and the result is always an :class:`HMCResult`, so a caller can swap one
method for another without changing anything around it. Both methods here write
their per-volume transforms as FLIRT matrices, so what comes after them cannot
tell which one ran.

The default is :func:`moco`, niimath's port of the ``3dvolreg`` algorithm: it
needs nothing installed and estimates a 138-frame run in about 20s.
:func:`allineate` is the general-purpose registration, an order of magnitude
slower but able to work against a reference that is not one of the frames
without composing, and to fit more than 6 DOF. :func:`mcflirt` needs FSL, and
its transforms are quarantined -- see :data:`UNREPLAYABLE_METHODS`.

Multi-echo policy, common to every method here: motion is estimated on a single
echo and the resulting per-volume transforms are applied unchanged to all
echoes. The echoes of one TR come from a single excitation and so share a head
position -- estimating each echo separately would add noise and would break the
voxelwise correspondence across echoes that T2*/S0 fitting depends on.

``ref`` is shared too, and takes the same names everywhere -- with one addition
only :func:`moco` can serve. ``ref="stable"`` measures frame-to-frame motion
before registering anything (:func:`relative_motion`, a few seconds) and aims
at the volume flanked by the two smallest displacements
(:func:`best_reference`). It is worth preferring to ``"middle"`` on data that
moves: the middle volume is chosen by counting frames, and a target the subject
was moving through blurs every fit made against it.

``ref="groupwise"`` gives up on picking a volume at all. It seeds from the volume
in the quietest stretch of the raw run (:func:`quiet_reference`), then estimates,
selects the frames that most look like the run by CDTM, averages those onto
their own mean pose, and re-estimates against that average -- repeating until
the selected set stops changing (:func:`groupwise_reference`). The target
therefore carries the noise of no single frame, and the loop's stopping rule is
a fixed point rather than a round count.

Its argument is reproducibility rather than accuracy: seeded from six different
volumes, the resulting estimates agreed to under a micrometre, where using
those volumes directly as targets left tens of micrometres of scatter. It costs
several passes over the series, and it is the only option here whose answer
does not depend on a choice made before the data was looked at.

To add a method, drop a module in this subpackage that returns an
:class:`HMCResult`, and register it in :data:`METHODS`.
"""

from .base import (HMCResult, TransformReplayError, UNREPLAYABLE_METHODS,
                   check_transforms_replayable)
from .fsl import mcflirt
from .moco import (REF_STABLE, REF_GROUPWISE, best_reference, brain_geometry,
                   centre_pulls, frechet_mean_pose, moco, pose_components,
                   pose_distance,
                   quiet_reference, relative_displacement, relative_motion, relative_rms,
                   stable_reference, supports_bin, supports_ref,
                   supports_relative, select_frames, step_motion,
                   neighbour_motion, current_motion, motion_history,
                   within_tr_pulls, combine_rms, interleaved_slices,
                   groupwise_reference, within_tr_motion)
from .niimath import allineate

#: Method name -> callable. Extend this as methods are added.
METHODS = {
    "moco": moco,
    "niimath": allineate,
    "fsl": mcflirt,
}

#: Used when no method is named. -moco is by a wide margin the fastest of the
#: three, and needs nothing beyond niimath itself.
DEFAULT_METHOD = "moco"


def get_method(name: str | None = None):
    """Look up an HMC method by name, for config-driven pipelines.

    Passing ``None`` gives :data:`DEFAULT_METHOD`.
    """
    name = DEFAULT_METHOD if name is None else name
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown HMC method {name!r}; available: {sorted(METHODS)}"
        ) from None


__all__ = ["HMCResult", "moco", "allineate", "mcflirt", "METHODS",
           "DEFAULT_METHOD", "get_method", "TransformReplayError",
           "UNREPLAYABLE_METHODS", "check_transforms_replayable",
           "relative_motion", "relative_displacement", "relative_rms",
           "best_reference",
           "stable_reference", "REF_STABLE", "supports_ref",
           "supports_relative", "supports_bin", "groupwise_reference",
           "REF_GROUPWISE", "frechet_mean_pose", "centre_pulls",
           "brain_geometry", "pose_distance", "pose_components",
           "quiet_reference",
           "select_frames", "within_tr_motion", "step_motion",
           "neighbour_motion", "current_motion", "motion_history",
           "within_tr_pulls", "combine_rms", "interleaved_slices"]
