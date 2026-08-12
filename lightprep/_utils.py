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
