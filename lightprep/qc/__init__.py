"""Quality control: looking at what the pipeline produced.

Every other subpackage is a step that transforms data. These are the checks on
it -- the questions worth asking before a result is believed, packaged so they
are run rather than reinvented per project.

:mod:`lightprep.qc.motion` renders a self-contained NiiVue page: the corrected
series, scrubbable frame by frame, over its framewise-displacement trace. Pass
several corrected volumes and they scrub together, which is how two motion
estimators are compared where it matters -- on the frames they disagree about.

The identity check (are two sessions really the same head?) lives in
:mod:`lightprep.identity`, beside the registration primitives it is built from.
"""

from .motion import (DEFAULT_CLIP, DEFAULT_DOWNSAMPLE, FD_RADIUS_MM,
                     find_niivue, framewise_displacement, motion_report, serve)

__all__ = ["motion_report", "serve", "framewise_displacement", "find_niivue",
           "FD_RADIUS_MM", "DEFAULT_DOWNSAMPLE", "DEFAULT_CLIP"]
