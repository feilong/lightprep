"""Deterministic DICOM->BIDS corrections, reproducible from metadata.

Everything here is decided from DICOM/sidecar fields, never by hand, so the
corrected BIDS can be regenerated from the DICOMs. The corrections are documented
in the repo's ``CORRECTIONS.md``; this module is their executable form.

Driven by ``scripts/dicom2bids.py``.
"""

from .correct import (
    correct_func_naming,
    disambiguate_anat,
    normalize_func_acq,
    recover_phase_sbref,
    regenerate_scans_tsv,
    rename_task,
    scanner_from_station,
    session_label,
    subject_label,
)

__all__ = [
    "correct_func_naming", "disambiguate_anat", "normalize_func_acq",
    "recover_phase_sbref", "regenerate_scans_tsv", "rename_task",
    "scanner_from_station", "session_label", "subject_label",
]
