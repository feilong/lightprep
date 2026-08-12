"""Just enough BIDS awareness to find the runs in a func directory.

This is deliberately small: entity parsing by regex, no index, no validation.
It exists so the driver scripts agree on what "a run" is rather than each
rolling their own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ._utils import strip_ext


def sidecar(path: Path) -> Path:
    """The BIDS JSON sitting next to a NIfTI."""
    path = Path(path)
    return path.with_name(strip_ext(path) + ".json")


def run_key(path: Path) -> str:
    """Collapse an echo filename to the run it belongs to.

    Drops the entities that vary *within* a run (echo, part) so that every
    image of one acquisition maps to the same key.
    """
    stem = strip_ext(Path(path))
    stem = re.sub(r"_echo-\d+", "", stem)
    stem = re.sub(r"_part-[a-zA-Z]+", "", stem)
    return re.sub(r"_bold$", "", stem)


def echo_index(path: Path) -> int:
    """Echo number, or 1 for single-echo runs that carry no echo entity."""
    match = re.search(r"_echo-(\d+)", Path(path).name)
    return int(match.group(1)) if match else 1


def _meta(path: Path, key: str):
    js = sidecar(path)
    if not js.exists():
        return None
    return json.loads(js.read_text()).get(key)


def echo_time(path: Path) -> float | None:
    """EchoTime in seconds from the sidecar, or None if there isn't one."""
    return _meta(path, "EchoTime")


def phase_encoding_direction(path: Path) -> str | None:
    """PhaseEncodingDirection (e.g. ``j``, ``j-``) from the sidecar."""
    return _meta(path, "PhaseEncodingDirection")


def total_readout_time(path: Path) -> float | None:
    """TotalReadoutTime in seconds from the sidecar."""
    return _meta(path, "TotalReadoutTime")


def find_runs(func_dir, part: str = "mag", suffix: str = "bold") -> dict[str, list[Path]]:
    """Map each run to its echoes, ordered by echo time.

    Args:
        func_dir: A BIDS func directory.
        part: The ``part-`` entity to collect, e.g. ``mag`` or ``phase``.
        suffix: The BIDS suffix to collect.

    Returns:
        ``{run_key: [echo paths in echo order]}``, sorted by run.
    """
    runs: dict[str, list[Path]] = {}
    for path in Path(func_dir).glob(f"*_part-{part}_{suffix}.nii.gz"):
        runs.setdefault(run_key(path), []).append(path)
    return {key: sorted(paths, key=echo_index) for key, paths in sorted(runs.items())}
