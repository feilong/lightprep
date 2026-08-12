"""First-level GLM on surface data.

Builds a design matrix from the block-design task plus the specified nuisance
model (motion + derivatives, Legendre trends, framewise displacement, aCompCor),
fits it per vertex, and writes the A - B contrast.

The pieces are separate so a caller can inspect or swap any of them:
:mod:`~lightprep.glm.design` builds the regressors, :mod:`~lightprep.glm.acompcor`
the anatomical CompCor components, :mod:`~lightprep.glm.model` fits and contrasts.
"""

from . import acompcor, design
from .model import GLMResult, fit

__all__ = ["design", "acompcor", "fit", "GLMResult"]
