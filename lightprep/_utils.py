"""Helpers shared by the step subpackages."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

NIFTI_EXTS = (".nii.gz", ".nii")


class DependencyError(RuntimeError):
    """A required external program could not be found."""


def require(program: str, path: str | None = None) -> str:
    """Return the full path to ``program``, or raise if it is not installed."""
    found = shutil.which(program, path=path)
    if found is None:
        raise DependencyError(
            f"{program!r} was not found on PATH. If this is an FSL tool, check that "
            f"FSL is installed and that FSLDIR and PATH are set; if it is a "
            f"FreeSurfer tool, check FREESURFER_HOME."
        )
    return found


def run(cmd: list, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run an external command, raising with captured output on failure.

    ``extra_env`` is merged over the inherited environment, and is used to
    resolve the executable -- so a caller can point at a toolbox that is not on
    the ambient PATH.
    """
    cmd = [str(c) for c in cmd]
    env = dict(os.environ)
    # Without this, the extension of FSL outputs depends on the caller's shell.
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    require(cmd[0], path=env.get("PATH"))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed ({}): {}\n--- stdout ---\n{}\n--- stderr ---\n{}".format(
                proc.returncode, " ".join(cmd), proc.stdout, proc.stderr
            )
        )
    return proc


def strip_ext(path: Path) -> str:
    """Filename of ``path`` without its NIfTI extension."""
    name = Path(path).name
    for ext in NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


#: Suffix of the authoritative copy of a numeric trace. ``.npy`` rather than a
#: raw ``.bin`` because it carries its own dtype and shape: a raw dump needs an
#: out-of-band contract to be read back, and a contract that lives in a
#: docstring is one a future reader can get wrong.
TRACE_SUFFIX = ".npy"


def save_trace(array, path) -> Path:
    """Save a numeric trace as float64 ``.npy``, with a text twin beside it.

    The ``.npy`` is the artefact. Text cannot hold a float64 -- ``%.17g``
    round-trips but is unreadable, anything narrower quietly rounds -- and a
    trace written by one step is read back by another, so a lossy hop between
    them is a computation done at less than float64 for no reason.

    The text twin is kept because a motion trace is something people look at:
    ``head motion.par``, a column pasted into a plot, an eyeball comparison
    against another tool's output. Nothing in this package parses it when the
    ``.npy`` is there, so its precision is a display choice rather than a
    contract, and it is written at a width meant to be read.

    Args:
        array: The trace. Cast to float64.
        path: Where the *text* twin goes -- conventionally ``motion.par``. The
            ``.npy`` takes the same stem.

    Returns:
        The path to the ``.npy``, which is what a result should carry.
    """
    import numpy as np

    array = np.asarray(array, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = path.with_suffix(TRACE_SUFFIX)
    np.save(binary, array)
    np.savetxt(path, array, fmt="%.8f", delimiter="  ")
    return binary


def load_trace(path):
    """Read a trace, preferring the ``.npy`` twin over the text.

    Either twin's path gives the same numbers, so a caller does not have to
    know which one a result carries. Falling back to the text also means
    derivatives written before the ``.npy`` existed still load -- at the
    precision they were written with, which is all there is to recover.

    Args:
        path: Either twin, or any text file of numbers.

    Returns:
        A float64 array.
    """
    import numpy as np

    path = Path(path)
    binary = path if path.suffix == TRACE_SUFFIX else path.with_suffix(TRACE_SUFFIX)
    if binary.exists():
        return np.load(binary).astype(np.float64, copy=False)
    if path.suffix == TRACE_SUFFIX:
        raise FileNotFoundError(path)
    return np.loadtxt(path).astype(np.float64, copy=False)
