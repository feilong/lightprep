"""The corrections themselves. See CORRECTIONS.md for the rationale of each."""

from __future__ import annotations

import gzip
import json
import re
import struct
import subprocess
from pathlib import Path

# R2: station -> human scanner name
STATION_TO_SCANNER = {"AWP177514": "Vida", "AWP237021": "TerraX"}


def _write(f: Path, text: str) -> None:
    """Write text to ``f``, making it writable first (heudiconv marks 0444)."""
    import os
    f.chmod(f.stat().st_mode | 0o200)
    f.write_text(text)
    os.chmod(f, f.stat().st_mode & ~0o222)   # restore read-only, BIDS convention


def scanner_from_station(station: str) -> str:
    """Map a DICOM StationName to the scanner label (R2)."""
    return STATION_TO_SCANNER.get(station.strip(), station.strip() or "Unknown")


def subject_label(patient_name: str) -> str:
    """R1: BIDS subject from PatientName, non-alphanumerics stripped, case kept.

    Provisional -- PatientName is not fully trusted (see TODO: verify by brain
    shape). PatientID is deliberately not used (it is a generic 'crlab' for some).
    """
    return re.sub(r"[^0-9A-Za-z]", "", patient_name)


def session_label(study_date: str, study_time: str, station: str) -> str:
    """R2: ``ses-<scanner><YYMMDD>T<HHmm>`` from StudyDate/StudyTime/StationName."""
    yymmdd = study_date.strip()[2:8]
    hhmm = study_time.strip()[:4]
    return f"{scanner_from_station(station)}{yymmdd}T{hhmm}"


def _n_volumes(nii: Path) -> int:
    with gzip.open(nii, "rb") as fh:
        hdr = fh.read(352)
    dim = struct.unpack("<8h", hdr[40:56])
    return dim[4] if dim[0] >= 4 else 1


def _part_from_imagetype(image_type) -> str | None:
    it = [str(x).upper() for x in (image_type or [])]
    if "PHASE" in it or "P" in it:
        return "phase"
    if "MAGNITUDE" in it or "M" in it:
        return "mag"
    return None


def _reconstruct_func_name(base: str, part: str, suffix: str) -> str:
    """Rebuild a clean func name from heudiconv's disambiguated output.

    heudiconv 1.4.0 emits e.g. ``..._run-02_echo-1_bold``,
    ``..._run-02_bold__echo-1_dup-01``, ``..._run-01_bold__dup-02``,
    ``..._run-01_sbref`` -- the echo may sit before ``_bold`` or inside a
    ``__echo-N_dup-NN`` tail. Keep the entities through ``run-``, recover the
    echo number from anywhere in the name, and drop everything heudiconv added
    (``_bold``, ``__dup-NN``, ``__echo-N``, ``_sbref``).
    """
    prefix = re.match(r"(.+_run-\d+)", base).group(1)
    echo = re.search(r"echo-(\d+)", base)
    name = prefix + (f"_echo-{int(echo.group(1))}" if echo else "")
    return f"{name}_part-{part}_{suffix}"


def correct_func_naming(func_dir) -> list[tuple[str, str]]:
    """C2: give every func image its correct ``part-`` and suffix.

    ``part`` comes from the JSON ``ImageType`` (magnitude vs phase); the suffix
    from the volume count (1 -> sbref, many -> bold). heudiconv's disambiguation
    tails (``bold``, ``__dup-NN``, ``__echo-N``) are dropped. Each run/echo's
    four images map to the four distinct ``part-{mag,phase}_{bold,sbref}`` names,
    so no collision. Physio and non-mag/phase files are left alone.

    Returns the (old_base, new_base) mappings applied.
    """
    func_dir = Path(func_dir)
    renames = []
    targets = sorted(func_dir.glob("*_bold*.nii.gz")) + sorted(func_dir.glob("*_sbref*.nii.gz"))
    planned = {}
    for nii in targets:
        base = nii.name[: -len(".nii.gz")]
        js = func_dir / f"{base}.json"
        if not js.exists():
            continue
        meta = json.loads(js.read_text())
        part = _part_from_imagetype(meta.get("ImageType"))
        if part is None:
            continue
        # Suffix from the Siemens SeriesDescription '_SBRef' marker, not the
        # volume count: an aborted run leaves a 1-volume bold that the count
        # would misread as an sbref (and collide with the real one).
        sd = meta.get("SeriesDescription", "")
        if "SBRef" in sd:
            suffix = "sbref"
        elif sd:
            suffix = "bold"
        else:
            suffix = "sbref" if _n_volumes(nii) == 1 else "bold"   # fallback
        new_base = _reconstruct_func_name(base, part, suffix)
        if new_base == base:
            continue
        if new_base in planned.values():
            raise ValueError(f"func name collision: {base} and another both -> {new_base}")
        planned[base] = new_base
    for base, new_base in planned.items():
        for ext in (".nii.gz", ".json"):
            src = func_dir / f"{base}{ext}"
            if src.exists():
                src.rename(func_dir / f"{new_base}{ext}")
        renames.append((base, new_base))
    return renames


def normalize_func_acq(session_dir) -> int:
    """Undo reproin's ``-``->``X`` in the func acq label: ``acq-2dX`` -> ``acq-2d``.

    The protocol's ``acq-2d-1echo`` becomes ``acq-2dX1echo`` because ``-`` is not
    allowed inside a BIDS value; dropping it gives the pilot's ``acq-2d1echo``.
    Applied to names and to sidecar string references (physio, IntendedFor, scans).
    Returns the number of paths renamed.
    """
    session_dir = Path(session_dir)
    for f in session_dir.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".tsv"):
            t = f.read_text()
            if "acq-2dX" in t:
                _write(f, t.replace("acq-2dX", "acq-2d"))
    renamed = 0
    for p in sorted(session_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if "acq-2dX" in p.name:
            p.rename(p.with_name(p.name.replace("acq-2dX", "acq-2d")))
            renamed += 1
    return renamed


def rename_task(session_dir, old="rest", new="naming") -> int:
    """C1: rename the ``task-<old>`` entity to ``task-<new>`` throughout a session.

    File names, sidecar string references (IntendedFor/B0Field), and the
    ``TaskName`` field. Returns the number of paths renamed.
    """
    session_dir = Path(session_dir)
    old_tok, new_tok = f"task-{old}", f"task-{new}"
    # contents first (names still contain old token, but content edit is by string)
    for f in session_dir.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".tsv"):
            t = f.read_text()
            t2 = t.replace(old_tok, new_tok)
            t2 = re.sub(rf'("TaskName"\s*:\s*")\s*{re.escape(old)}(")', rf"\1{new}\2", t2)
            if t2 != t:
                _write(f, t2)
    renamed = 0
    for p in sorted(session_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if old_tok in p.name:
            p.rename(p.with_name(p.name.replace(old_tok, new_tok)))
            renamed += 1
    return renamed


def recover_phase_sbref(func_dir, dicom_dir, dcm2niix="dcm2niix") -> list[Path]:
    """C3: recover phase single-band references that reproin dropped.

    On a run where several series collide on the ``bold`` stem, reproin drops
    one -- the phase SBRef (SeriesDescription ``*_SBRef_Pha``, ImageType PHASE,
    one volume). Find each such DICOM series that has no ``part-phase_sbref`` in
    ``func_dir`` yet, convert it, and place it.

    Returns the recovered NIfTI paths.
    """
    from .dicom import group_series  # local import; light DICOM header scan

    func_dir, dicom_dir = Path(func_dir), Path(dicom_dir)
    recovered = []
    have = {f.name for f in func_dir.glob("*_part-phase_sbref.nii.gz")}
    for sd, files in group_series(dicom_dir).items():
        if not sd.endswith("_SBRef_Pha"):
            continue
        # parse the reproin-style SeriesDescription into BIDS entities
        m = re.search(r"acq-(\S+?)_run-(\d+)", sd)
        if not m:
            continue
        acq = m.group(1).replace("-", "")
        run = int(m.group(2))
        # does a matching part-phase_sbref already exist?
        stem_pat = re.compile(rf"acq-{acq}.*run-0?{run}.*_part-phase_sbref\.nii\.gz$")
        if any(stem_pat.search(h) for h in have):
            continue
        # derive the output name from a sibling corrected file for this run/acq
        sib = next(func_dir.glob(f"*acq-{acq}*run-0?{run}*_part-mag_sbref.nii.gz"), None)
        if sib is None:
            sib = next(func_dir.glob(f"*acq-{acq}*run-0?{run}*_part-mag_bold.nii.gz"), None)
        if sib is None:
            continue
        out_base = re.sub(r"_echo-\d+", "", sib.name[: -len(".nii.gz")])
        out_base = re.sub(r"_part-mag_(bold|sbref)$", "_part-phase_sbref", out_base)
        # convert the dropped series with dcm2niix into a temp, then move
        tmp = func_dir / "_recover"
        tmp.mkdir(exist_ok=True)
        for f in files:
            (tmp / Path(f).name).write_bytes(Path(f).read_bytes())
        subprocess.run([dcm2niix, "-o", str(tmp), "-f", "recovered", "-z", "y", str(tmp)],
                       capture_output=True)
        nii = next(tmp.glob("recovered*.nii.gz"), None)
        if nii:
            nii.rename(func_dir / f"{out_base}.nii.gz")
            j = next(tmp.glob("recovered*.json"), None)
            if j:
                j.rename(func_dir / f"{out_base}.json")
            recovered.append(func_dir / f"{out_base}.nii.gz")
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return recovered
