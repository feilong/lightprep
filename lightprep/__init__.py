"""lightprep -- lightNIIng-based preprocessing workflows.

Preprocessing for (multi-echo) fMRI, built on niimath. Each step lives in its
own subpackage (``lightprep.hmc``, and more to come), and exposes one function
per method, all sharing a signature and a return type, so that methods can be
mixed and matched freely across steps::

    from lightprep import hmc

    result = hmc.moco(echoes, out_dir="derivatives/hmc")
    # ...or select a method by name, e.g. from a config file:
    result = hmc.get_method("fsl")(echoes, out_dir="derivatives/hmc")
    # ...or take the step's default:
    result = hmc.get_method()(echoes, out_dir="derivatives/hmc")

Wherever a step can be done with niimath, that method is registered alongside
the others and is the step's :data:`DEFAULT_METHOD` -- so head motion
correction, coregistration and distortion correction all run on a bare Python
environment, with no FSL, no FreeSurfer and no warpkit to install and no version
of any of them to account for. That is where the speed comes from as much as the
portability: head motion correction estimates a 138-frame run in about 20s, and
MEDIC field maps in about 12s. The original methods are unchanged and still
selectable by name; each step's docstring sets out what its niimath method
trades away.

:mod:`lightprep.recon` is the odd one out: it is anatomical rather than
functional, and it wraps FreeSurfer rather than replacing it. It exists because
``recon-all`` conforms every input to an isotropic axis-aligned grid, which
interpolates anything anisotropic or obliquely prescribed before the pipeline
has even started. Its ``native`` method makes that conform a bit-exact no-op,
and :func:`lightprep.recon.correct` undoes the deception analytically.

Two things niimath does not do are left alone rather than approximated.
:mod:`lightprep.resample` needs warp *composition* to spend a single
interpolation, which niimath has no equivalent of -- though its ``-unwarp`` does
give an exact substitute for the simpler
:func:`~lightprep.resample.apply_sdc`. And combination, decay fitting and
surface sampling work on per-vertex GIfTI, which is not image arithmetic at all.

niimath itself is not in this repository: a binary is a platform-specific build
artifact, and a committed one would be wrong for every machine but the one that
built it. ``BUILD.md`` has the invocation for the reference build, and
:func:`lightprep._niimath.niimath_path` explains where a build is looked for --
beside the package first, then PATH. A build recent enough to carry ``-moco``,
``--medic`` and ``-unwarp`` is required.

Build it with OpenMP if you can. Threading is what makes the per-frame
registration in :mod:`lightprep.hmc.niimath` tolerable -- a few seconds a frame
across 8 threads against roughly 9s on one -- though :mod:`lightprep.hmc.moco`,
the default, is fast either way. Set ``OMP_NUM_THREADS`` to control it.
"""

from . import (combine, coreg, decay, glm, hmc, qc, recon, resample, sdc,
               surface)
from ._niimath import NIIMATH

__version__ = "0.1.0"
__all__ = ["coreg", "combine", "decay", "glm", "hmc", "qc", "recon",
           "resample", "sdc", "surface", "NIIMATH"]
