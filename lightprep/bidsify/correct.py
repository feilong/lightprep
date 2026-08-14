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

#: R1: session -> finalized subject, for identities confirmed from the anatomy.
#:
#: Keyed by **session**, not by PatientName: the name is whatever an operator
#: typed that day, so one string can cover different people (`CR_TEST` and
#: `CR_test04` are not self-evidently the same person) and one person several
#: strings. The evidence is per-session -- each session's brain was matched --
#: so the record is per-session too. A session that is not listed keeps the
#: provisional label from its PatientName.
#:
#: `Ro`/`crtest` -> CR resolved 2026-07-28 (EP3d magnitude, r 0.92-0.95);
#: the 7T rows 2026-08-14 (mindgrab + affine, mutual r 0.97-0.99 within subject
#: against 0.66 between). `ses-TerraX260605T1307` (`CR_test04`) is absent on
#: purpose: it has no MP2RAGE, so nothing has verified it.
RESOLVED_SESSION = {
    # pilot 3T Vida, PatientName `Ro`
    "ses-Vida260702T1603": "CR",
    "ses-Vida260707T1241": "CR",
    "ses-Vida260708T1251": "CR",
    "ses-Vida260708T1401": "CR",
    "ses-Vida260709T0907": "CR",
    # pilot 7T Terra.X, PatientName `crtest`
    "ses-TerraX260702T1518": "CR",
    # pilot 7T Terra.X, PatientName `Moore` -- a subject new to the project
    "ses-TerraX260812T1608": "SM",
    # pilot 3T Vida, names already correct but verified as their own subjects
    "ses-Vida260709T1021": "CD",
    "ses-Vida260714T1157": "FM",
    "ses-Vida260707T0942": "phantom",  # not a person
    # fmlab 7T Terra.X
    "ses-TerraX260528T1531": "FM",     # PatientName `FM`, confirmed not assumed
    "ses-TerraX260601T1433": "CD",     # PatientName `CD`, confirmed not assumed
    "ses-TerraX260529T1249": "CR",     # PatientName `CR_TEST`
    "ses-TerraX260603T1601": "FM",     # PatientName `FM2`
    "ses-TerraX260608T1349": "CD",     # PatientName `cd`
    "ses-TerraX260608T1437": "FM",     # PatientName `fm`
    "ses-TerraX260608T1514": "CR",     # PatientName `cr`
}


def _write(f: Path, text: str) -> None:
    """Write text to ``f``, making it writable first (heudiconv marks 0444)."""
    import os
    if f.exists():
        f.chmod(f.stat().st_mode | 0o200)
    f.write_text(text)
    os.chmod(f, f.stat().st_mode & ~0o222)   # restore read-only, BIDS convention


def scanner_from_station(station: str) -> str:
    """Map a DICOM StationName to the scanner label (R2)."""
    return STATION_TO_SCANNER.get(station.strip(), station.strip() or "Unknown")


def subject_label(patient_name: str, session: str | None = None) -> str:
    """R1: BIDS subject from PatientName, non-alphanumerics stripped, case kept.

    Provisional -- PatientName is not fully trusted. PatientID is deliberately
    not used (it is a generic 'crlab' for some). Pass ``session`` (the
    ``ses-<scanner><YYMMDD>T<HHmm>`` label from :func:`session_label`) and a
    session whose identity has been confirmed from the anatomy is resolved
    through :data:`RESOLVED_SESSION` instead; without it, or for a session not
    yet verified, the provisional label stands.
    """
    if session and session in RESOLVED_SESSION:
        return RESOLVED_SESSION[session]
    label = re.sub(r"[^0-9A-Za-z]", "", patient_name)
    if session:
        # Unverified: keep the session apart. Two sessions sharing a PatientName
        # are not evidence of one person -- the name is typed per session, and
        # `CR_TEST`/`CR_test04`/`FM2` show how loosely it is used -- so an
        # unresolved session gets its own date-stamped label rather than being
        # merged into whoever else answered to that name. It joins the real
        # subject only once RESOLVED_SESSION says so, on anatomical evidence.
        m = re.match(r"ses-[A-Za-z]+(\d{6})T", session)
        if m:
            return f"{label}{m.group(1)}"
    return label


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


def _func_stem(base: str):
    """The BIDS entity prefix (through any ``run-``) and echo of a func name.

    heudiconv 1.4.0 emits e.g. ``..._run-02_echo-1_bold``,
    ``..._run-02_bold__echo-1_dup-01``, ``..._run-01_bold__dup-02``,
    ``..._run-01_sbref`` -- the echo may sit before ``_bold`` or inside a
    ``__echo-N_dup-NN`` tail. Keep the entities through ``run-`` (3T), or, when
    there is no ``run-`` (7T Terra.X), strip heudiconv's tails and the
    suffix/echo. Returns ``(prefix, echo_or_None)``.
    """
    m = re.match(r"(.+_run-\d+)", base)
    if m:
        prefix = m.group(1)                       # 3T path, unchanged
    else:
        prefix = re.sub(r"(__dup-\d+|__echo-\d+|_heudiconv\d+)", "", base)
        prefix = re.sub(r"_(bold|sbref)$", "", prefix)
        prefix = re.sub(r"_echo-\d+", "", prefix)
    echo = re.search(r"echo-(\d+)", base)
    return prefix, (int(echo.group(1)) if echo else None)


def correct_func_naming(func_dir) -> list[tuple[str, str]]:
    """C2: give every func image its correct ``part-``, ``run-`` and suffix.

    ``part`` comes from the JSON ``ImageType`` (magnitude vs phase); the suffix
    from the Siemens ``_SBRef`` marker (else the volume count). heudiconv's
    disambiguation tails (``bold``, ``__dup-NN``, ``__echo-N``) are dropped.

    Images are grouped by (prefix, echo, part, suffix). A singleton keeps its
    name as-is -- the 3T case, where run-/echo/part/suffix already separate a
    run's four ``part-{mag,phase}_{bold,sbref}`` images (no ``run-`` added). When
    a group has *several* images that reproin collapsed onto one stem -- the 7T
    Terra.X case, where the protocol name carries no run number so distinct runs
    collide -- they are separated by ``run-N`` ordered by ``SeriesNumber``.

    Returns the (old_base, new_base) mappings applied.
    """
    from collections import defaultdict

    func_dir = Path(func_dir)
    targets = sorted(func_dir.glob("*_bold*.nii.gz")) + sorted(func_dir.glob("*_sbref*.nii.gz"))
    groups = defaultdict(list)
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
        prefix, echo = _func_stem(base)
        groups[(prefix, echo, part, suffix)].append((base, meta.get("SeriesNumber", 0)))

    planned = {}
    for (prefix, echo, part, suffix), members in groups.items():
        multi = len(members) > 1
        members.sort(key=lambda t: t[1])          # by SeriesNumber
        for i, (base, _sn) in enumerate(members, 1):
            name = prefix
            if multi and "_run-" not in prefix:   # distinct runs collapsed on one stem
                name += f"_run-{i}"
            if echo:
                name += f"_echo-{echo}"
            planned[base] = f"{name}_part-{part}_{suffix}"

    renames = []
    for base, new_base in planned.items():
        if new_base == base:
            continue
        for ext in (".nii.gz", ".json"):
            src = func_dir / f"{base}{ext}"
            if src.exists():
                src.rename(func_dir / f"{new_base}{ext}")
        renames.append((base, new_base))
    return renames


#: marker of the dzne T2*-weighted 3D-EPI structural (reproin mislabels it T1w)
EP3D_ACQ_MARK = "Ep3d"


def disambiguate_anat(anat_dir) -> list[tuple[str, str]]:
    """C5: rename each EP3d structural image with a distinct ``part-``/``run-``.

    The dzne ``Ep3d`` 3D-EPI structural emits magnitude + phase (and often two
    repeats) per phase-encoding direction, but reproin collapses them onto the
    same ``_T1w`` stem with ``__dup-NN`` tails -- so magnitude and phase silently
    overwrite, and the base ``_T1w`` name is left as whichever series wrote last
    (phase for some sessions). This is the anat analogue of C2.

    Rebuild each name from metadata that says so unambiguously:
    - **part** from JSON ``ImageType`` (``M`` -> ``part-mag``, ``P`` -> ``part-phase``).
    - **run** ordered by ``SeriesNumber`` within each (dir, part) group, added
      only when that group has more than one image (a genuine repeat).
    - **dir-REV** kept as-is; all ``__dup-NN`` tails dropped.
    - **suffix** ``T1w`` -> ``T2starw``: reproin inherits ``T1w`` from the
      protocol name, but this acquisition is T2*-weighted (TE~56 ms; bright CSF
      and venous susceptibility, matching the BOLD func). fm's true MPRAGE
      (TE~2.3 ms, no ``acq`` label) is a real ``T1w`` and is not touched here.

    No collision by construction; nothing is lost.

    Returns the (old_base, new_base) mappings applied.
    """
    from collections import defaultdict

    anat_dir = Path(anat_dir)
    targets = [p for p in sorted(anat_dir.glob("*_T1w*.nii.gz"))
               if EP3D_ACQ_MARK in p.name]
    recs = []
    for nii in targets:
        base = nii.name[: -len(".nii.gz")]
        js = anat_dir / f"{base}.json"
        if not js.exists():
            continue
        meta = json.loads(js.read_text())
        part = _part_from_imagetype(meta.get("ImageType"))
        if part is None:
            continue
        recs.append((base, part, "dir-REV" in base, meta.get("SeriesNumber", 0)))

    groups = defaultdict(list)
    for base, part, is_rev, sn in recs:
        groups[(is_rev, part)].append((base, sn))
    plan = {}
    for (is_rev, part), items in groups.items():
        items.sort(key=lambda t: t[1])                     # by SeriesNumber
        multi = len(items) > 1
        for i, (base, _sn) in enumerate(items, 1):
            prefix = re.match(r"(sub-[^_]+_ses-[^_]+_acq-[^_]+)", base).group(1)
            ents = [prefix]
            if is_rev:
                ents.append("dir-REV")
            if multi:
                ents.append(f"run-{i}")
            ents.append(f"part-{part}")
            plan[base] = "_".join(ents) + "_T2starw"     # T2*-weighted, not T1w

    renames = []
    for old, new in plan.items():
        if old == new:
            continue
        for ext in (".nii.gz", ".json"):
            src = anat_dir / f"{old}{ext}"
            if src.exists():
                src.rename(anat_dir / f"{new}{ext}")
        renames.append((old, new))
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


def regenerate_scans_tsv(session_dir) -> Path:
    """Rebuild ``*_scans.tsv`` from the session's actual files + sidecar times.

    heudiconv writes scans.tsv once, *before* the renaming corrections (C2/C5),
    so it goes stale -- its rows point at pre-correction ``__dup`` names that no
    longer exist. Rebuild it from disk: one row per NIfTI, ``acq_time`` from the
    JSON ``AcquisitionTime`` plus the session date, sorted by time. Keeps the
    heudiconv 4-column shape (filename, acq_time, operator, randstr); randstr is
    a stable hash of the name (nothing downstream depends on its value).

    Run this last, after every rename, so the manifest matches the tree.
    """
    import hashlib

    session_dir = Path(session_dir)
    m = re.search(r"ses-[A-Za-z]+(\d{6})T", session_dir.name)   # <scanner><YYMMDD>T<HHmm>
    date = f"20{m.group(1)[:2]}-{m.group(1)[2:4]}-{m.group(1)[4:6]}" if m else ""
    rows = []
    for nii in sorted(session_dir.rglob("*.nii.gz")):
        rel = nii.relative_to(session_dir).as_posix()
        js = Path(str(nii)[: -len(".nii.gz")] + ".json")
        at = json.loads(js.read_text()).get("AcquisitionTime", "") if js.exists() else ""
        acq_time = f"{date}T{at}" if (date and at) else (at or "n/a")
        rand = hashlib.md5(rel.encode()).hexdigest()[:8]
        rows.append((acq_time, rel, rand))
    rows.sort()
    tsv = next(session_dir.glob("*_scans.tsv"), None) or \
        session_dir / f"{session_dir.parent.name}_{session_dir.name}_scans.tsv"
    lines = ["filename\tacq_time\toperator\trandstr"]
    lines += [f"{rel}\t{at}\tn/a\t{rand}" for at, rel, rand in rows]
    _write(tsv, "\n".join(lines) + "\n")
    return tsv


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
