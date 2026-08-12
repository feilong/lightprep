"""Minimal DICOM header reading -- just the tags the corrections need.

A tiny partial parser so the pipeline has no hard pydicom dependency. Reads
only the first block of each file (the wanted tags precede the pixel data and
the large private CSA blocks).
"""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path

_WANT = {
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0030): "StudyTime",
    (0x0008, 0x1030): "StudyDescription",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0008, 0x1010): "StationName",
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0011): "SeriesNumber",
}


def read_header(path, nbytes=16384) -> dict:
    """Return the wanted tags from one DICOM file (empty dict if not DICOM)."""
    with open(path, "rb") as fh:
        buf = fh.read(nbytes)
    out = {}
    if buf[128:132] != b"DICM":
        return out
    i, n = 132, len(buf)
    while i + 8 <= n:
        group, elem = struct.unpack_from("<HH", buf, i)
        i += 4
        vr = buf[i:i + 2]
        i += 2
        if vr in (b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN"):
            i += 2
            if i + 4 > n:
                break
            length = struct.unpack_from("<I", buf, i)[0]
            i += 4
        else:
            if i + 2 > n:
                break
            length = struct.unpack_from("<H", buf, i)[0]
            i += 2
        if length == 0xFFFFFFFF:
            break
        if (group, elem) in _WANT:
            try:
                out[_WANT[(group, elem)]] = buf[i:i + length].decode("ascii", "replace").strip("\x00 ")
            except Exception:
                pass
        i += length
        if group > 0x0020:
            break
    return out


def session_meta(dicom_dir) -> dict:
    """Session-level tags, read from the first parseable DICOM in the directory."""
    for f in sorted(Path(dicom_dir).iterdir()):
        if f.is_file():
            h = read_header(f)
            if h:
                return h
    return {}


def group_series(dicom_dir) -> dict:
    """Map SeriesDescription -> list of file paths, for a directory of DICOMs."""
    groups = defaultdict(list)
    for f in sorted(Path(dicom_dir).iterdir()):
        if not f.is_file():
            continue
        h = read_header(f, 4096)
        sd = h.get("SeriesDescription")
        if sd:
            groups[sd].append(str(f))
    return dict(groups)
