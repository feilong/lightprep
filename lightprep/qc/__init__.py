"""Quality control: looking at what the pipeline produced.

Every other subpackage is a step that transforms data. These are the checks on
it -- the questions worth asking before a result is believed, packaged so they
are run rather than reinvented per project.

:mod:`lightprep.qc.surface` reads the Euler number of the pre-topology-fix
surface, which is the cheapest honest proxy for how good a structural was.

:mod:`lightprep.qc.motion` renders a self-contained NiiVue page: the corrected
series, scrubbable frame by frame, over its framewise-displacement trace. Pass
several corrected volumes and they scrub together, which is how two motion
estimators are compared where it matters -- on the frames they disagree about.

Its measurements are worth having on their own, and two of them run before the
pipeline commits to anything. :func:`~lightprep.qc.motion.relative_fd` fits
each volume onto its predecessor, so it reports how much a subject moved
without a reference volume to be flattered by, and
:func:`~lightprep.qc.motion.cdtm` scores each frame against the run itself.
They ask different questions -- was this frame *acquired* cleanly, and does it
*look like* the rest of the run -- and a frame can fail either alone.

The identity check (are two sessions really the same head?) lives in
:mod:`lightprep.identity`, beside the registration primitives it is built from.
"""

from .surface import (DEFAULT_SURFACE, SurfaceQC, euler_number,
                      flag_outliers, surface_qc)
from .table import SEVERE, QUIET, summary_table
from .motion import (DEFAULT_CLIP, DEFAULT_DOWNSAMPLE, FD_RADIUS_MM,
                     brain_mask, dvars, find_niivue, framewise_displacement,
                     correlation_distance, frame_distance, masked_series,
                     motion_report, quality_index,
                     aqi_outliers, AQI_MAD, cdtm, CDTMResult,
                     CDTM_RATIO, CDTM_TRIM, relative_fd, fd_outliers,
                     FD_THRESHOLD_MM,
                     trimmed_mean, serve)

__all__ = ["motion_report", "summary_table", "serve", "dvars", "frame_distance",
           "masked_series", "correlation_distance",
           "relative_fd", "fd_outliers", "FD_THRESHOLD_MM",
           "brain_mask", "quality_index", "aqi_outliers", "AQI_MAD",
           "cdtm", "CDTMResult", "CDTM_RATIO", "CDTM_TRIM",
           "trimmed_mean",
           "surface_qc", "euler_number",
           "flag_outliers", "SurfaceQC", "DEFAULT_SURFACE", "framewise_displacement", "find_niivue",
           "FD_RADIUS_MM", "DEFAULT_DOWNSAMPLE", "DEFAULT_CLIP"]
