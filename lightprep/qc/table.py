"""A cohort at a glance: one row per run, one column per measure.

A per-run report answers "what happened in this run"; nothing answers "which
runs should I open". This does, by putting the fraction of frames each measure
objects to in a grid and colouring it, so a cohort is scanned rather than read.

The colouring is deliberately non-linear. Across a healthy cohort the median
run flags well under 1% of its frames, so a linear ramp to 50% would leave
almost every real problem the same shade of nothing; the scale below reaches
half intensity by a few percent and saturates at :data:`SEVERE`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

#: Percentage at which a cell is fully saturated. Beyond this the run is not
#: marginal and the exact number stops mattering to the decision.
SEVERE = 25.0

#: Below this a cell is left unpainted. A single bad frame in a long run is
#: not a finding, and colouring it spends the reader's attention on noise.
QUIET = 0.5


def summary_table(table, out_html, *, title: str = "QC summary",
                  subtitle: str = "", links=None, notes=None, marks=None,
                  severe: float = SEVERE, quiet: float = QUIET,
                  note: str = "") -> Path:
    """Write a coloured grid of per-run, per-measure outlier percentages.

    Args:
        table: ``{row: {column: percentage}}``. Columns are taken from the
            first row and every row is read in that order, so a missing entry
            shows as blank rather than shifting the grid.
        out_html: Where to write.
        title, subtitle: Page heading and the line under it.
        links: ``{row: url}`` making a row label clickable -- normally its own
            report, so the grid is a way in rather than a dead end.
        notes: ``{row: text}`` shown dim beside the label, for something that
            is not a percentage and must not share the colour scale.
        marks: ``{row: bool}`` drawing attention to a row's note.
        severe: Value at which colour saturates. A dict keyed by column sets
            it per column, for a grid whose columns are not all percentages --
            a millimetre and a percent should not share a ramp.
        quiet: Value below which nothing is painted; also per-column by dict.
        note: Free text under the table, for what the thresholds were.

    Returns:
        The path written.
    """
    table = {str(k): dict(v) for k, v in dict(table).items()}
    if not table:
        raise ValueError("nothing to tabulate")
    columns = list(next(iter(table.values())))

    def scale(value, default):
        if isinstance(value, dict):
            return [float(value.get(c, default)) for c in columns]
        return [float(value)] * len(columns)

    rows = [{"run": r, "url": (links or {}).get(r, ""),
             "note": (notes or {}).get(r, ""), "mark": bool((marks or {}).get(r)),
             "values": [v.get(c) for c in columns]} for r, v in table.items()]

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        _HTML.replace("/*DATA*/", json.dumps(
            {"columns": columns, "rows": rows,
             "severe": scale(severe, SEVERE), "quiet": scale(quiet, QUIET),
             "title": title, "subtitle": subtitle, "note": note})),
        encoding="utf-8")
    return out_html


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QC summary</title>
<style>
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#21262d;--warn:#f85149}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 -apple-system,
  BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:14px 18px 8px}
h1{margin:0;font-size:15px;font-weight:600}
.sub{color:var(--dim);font-size:12px;margin-top:3px}
.wrap{overflow-x:auto;padding:0 18px 6px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:3px 9px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--bg);color:var(--dim);font-weight:500;
  font-size:12px;cursor:pointer;user-select:none}
th:hover{color:var(--fg)}
th.run,td.run{text-align:left}
td.run a{color:var(--fg);text-decoration:none;border-bottom:1px dotted #3d444d}
td.run a:hover{color:#58a6ff;border-bottom-color:#58a6ff}
tr:hover td{background:#161b22}
tr:hover td.paint{background-image:linear-gradient(rgba(255,255,255,.06),
  rgba(255,255,255,.06))}
.note{padding:8px 18px 20px;color:var(--dim);font-size:12px;max-width:70em}
.arrow{color:#58a6ff}
.gap{color:#6e7681;font-size:11px;margin-left:7px}
.gap.long{color:#f0883e}
tr.placeholder td{color:#6e7681;font-style:italic}
tr.placeholder td.run a{color:#6e7681}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div></header>
<div class="wrap"><table id="grid"></table></div>
<div class="note" id="note"></div>
<script>
const D = /*DATA*/;
document.getElementById("title").textContent = D.title;
document.getElementById("subtitle").textContent = D.subtitle;
document.getElementById("note").textContent = D.note;

// Colour has to earn its place: a healthy run flags a fraction of a percent, so
// a linear ramp would paint every real problem the same near-nothing. Square
// root reaches half intensity by a quarter of the severe mark.
function paint(v, i){
  if (v == null || v < D.quiet[i]) return "";
  const t = Math.min(1, Math.sqrt(v / D.severe[i]));
  // amber -> red, lightening the text as the ground darkens under it
  const r = Math.round(120 + 100*t), g = Math.round(90 - 60*t), b = 40;
  return `background:rgba(${r},${g},${b},${0.25 + 0.65*t})`;
}

let sortCol = -1, desc = true;
function draw(){
  const rows = D.rows.slice();
  if (sortCol >= 0) rows.sort((a, b) => {
    const x = a.values[sortCol], y = b.values[sortCol];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return desc ? y - x : x - y;
  });   // no sort column: leave the order given, which is acquisition order
        // when times were supplied and alphabetical when they were not
  const back = sortCol < 0 ? "" : " <span class=\"arrow\">\u21ba</span>";
  let h = `<thead><tr><th class="run" data-i="-1" title="back to the`
        + ` original order">run${back}</th>`;
  D.columns.forEach((c, i) => {
    h += `<th data-i="${i}">${c}` +
         (sortCol === i ? ` <span class="arrow">${desc ? "↓" : "↑"}</span>` : "") +
         "</th>";
  });
  h += "</tr></thead><tbody>";
  for (const r of rows){
    // A row with nothing measured is a placeholder -- a field map or an
    // anatomical, which has no frames to score but does have a place in the
    // order, and whose neighbours are the point of showing it.
    const empty = r.values.every(v => v == null);
    const label = r.url ? `<a href="${r.url}">${r.run}</a>` : r.run;
    const note = r.note
      ? ` <span class="gap${r.mark ? " long" : ""}">${r.note}</span>` : "";
    h += `<tr class="${empty ? "placeholder" : ""}">` +
         `<td class="run">${label}${note}</td>`;
    r.values.forEach((v, i) => {
      const s = paint(v, i);
      h += `<td class="${s ? "paint" : ""}" style="${s}">` +
           (v == null ? "" : (v === 0 ? "·" : v.toFixed(v < 10 ? 1 : 0))) +
           "</td>";
    });
    h += "</tr>";
  }
  document.getElementById("grid").innerHTML = h + "</tbody>";
  document.querySelectorAll("th[data-i]").forEach(th => {
    th.onclick = () => {
      const i = +th.dataset.i;
      // Largest first, then smallest, then back to the order the table was
      // written in -- which is the one carrying the acquisition sequence, and
      // was unreachable once any column had been clicked.
      if (i < 0) sortCol = -1;
      else if (i !== sortCol) { sortCol = i; desc = true; }
      else if (desc) desc = false;
      else sortCol = -1;
      draw();
    };
  });
}
draw();
</script></body></html>
"""


def _seconds(value):
    """Seconds from a number, or from a BIDS ``AcquisitionTime`` string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"cannot read a time from {value!r}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def outlier_fractions(series, thresholds):
    """``{measure: percent of frames over threshold}`` plus ``any``.

    Every measure is put on one frame grid first. A per-transition measure --
    framewise displacement, DVARS -- has one value fewer than a per-frame one,
    and scoring each as it comes divides by different totals: the percentages
    stop being comparable and a column can exceed the union that contains it.
    A leading zero pads the short ones, which is the honest value there anyway
    since no transition precedes the first frame.

    Args:
        series: ``{measure: 1-D array}``. A measure absent here, or empty, is
            reported as None rather than as zero -- not measured and not
            exceeded are different facts.
        thresholds: ``{measure: value}``. Its order sets the column order.

    Returns:
        ``{measure: percent}`` in the order of ``thresholds``, then ``any``,
        the union over the measures actually present.
    """
    import numpy as np

    present = {k: np.asarray(v, dtype=np.float64).ravel()
               for k, v in dict(series).items() if np.size(v)}
    if not present:
        return {**{k: None for k in thresholds}, "any": None}
    n = max(v.size for v in present.values())

    row, union = {}, np.zeros(n, dtype=bool)
    for measure, threshold in dict(thresholds).items():
        v = present.get(measure)
        if v is None:
            row[measure] = None
            continue
        if v.size == n - 1:
            v = np.concatenate([[0.0], v])
        if v.size != n:
            row[measure] = None
            continue
        over = v > threshold
        row[measure] = float(100.0 * over.mean())
        union |= over
    row["any"] = float(100.0 * union.mean())
    return row


def outlier_summary(runs, thresholds, out_html, *, acquisition=None,
                    sessions=None, duration=None, gap_warn: float = 300.0,
                    links=None, placeholders=(), annotate=None, columns=None,
                    **kwargs) -> Path:
    """A coloured grid of how often each measure objects, one row per run.

    Args:
        runs: ``{label: {measure: series}}``.
        thresholds: ``{measure: value}``, in the order the columns should read.
        out_html: Where to write.
        acquisition: ``{label: time}`` -- seconds, or a BIDS ``AcquisitionTime``
            string. Given, the rows are ordered as they were acquired rather
            than alphabetically, and each carries the gap since the one before.
            A cohort reads differently in that order: drift, fatigue and the
            run someone was repositioned before all show as a sequence.
        sessions: ``{label: session}``. Acquisition times usually come from
            BIDS ``AcquisitionTime``, which is a clock time with no date, so
            ordering by it alone interleaves sessions that happened on
            different days and invents gaps between them. Given this, rows are
            ordered within a session and sessions keep the order they first
            appear here; a gap is only ever measured between two runs of the
            same session.
        duration: ``{label: seconds}``. With it a gap is the real dead time
            between one row ending and the next starting; without it the
            interval runs start to start, which is the same thing plus a scan,
            and is written with a leading ``\u2264`` to say so. Give durations for
            the placeholder rows too -- a field map is short but not zero.
        gap_warn: Seconds after which a gap is called out. A long one is where
            the subject was spoken to, repositioned, or left alone -- and
            wherever a field map sits far from what it corrects.
        links: ``{label: url}`` to make a row clickable.
        placeholders: Labels to show as rows with no measures -- field maps and
            anatomicals, which have no frames to score but do have a place in
            the order, and whose neighbours are what a reader wants to see.
        columns: ``{label: {name: value}}`` of things that are not outlier
            percentages -- a pose difference in millimetres, a temperature --
            added after the measure columns. Pass ``severe`` and ``quiet`` as
            dicts to give them their own colour scale, since a millimetre and
            a percent do not belong on one ramp.
        annotate: ``{label: text}`` appended to that row's interval. An
            interval is only ever measured to the previous row *given here*,
            so where scans exist that this table does not list, they sit
            silently inside it; this is how to say so.
        **kwargs: Passed to :func:`summary_table`.

    Returns:
        The path written.
    """
    runs = {str(k): dict(v) for k, v in dict(runs).items()}
    for label in placeholders:
        runs.setdefault(str(label), {})

    table = {label: outlier_fractions(series, thresholds)
             for label, series in runs.items()}
    columns = {str(k): dict(v) for k, v in dict(columns or {}).items()}
    if columns:
        names = list(dict.fromkeys(n for v in columns.values() for n in v))
        for label, row in table.items():
            row.update({n: columns.get(label, {}).get(n) for n in names})

    times = {k: _seconds(v) for k, v in dict(acquisition or {}).items()}
    notes, marks = {}, {}
    if times:
        groups = {k: str(v) for k, v in dict(sessions or {}).items()}
        rank = {}
        for label in table:                     # first appearance sets the order
            rank.setdefault(groups.get(label, ""), len(rank))

        def key(label):
            t = times.get(label)
            return (rank[groups.get(label, "")], t is None, t if t is not None else 0.0)

        ordered = sorted(table, key=key)
        table = {k: table[k] for k in ordered}
        extra = {str(k): str(v) for k, v in dict(annotate or {}).items()}
        last = {}                    # per session: (end of the previous row,
        for label in ordered:        #  whether that end was actually known)
            now, group = times.get(label), groups.get(label, "")
            if now is not None and group in last:
                end, known = last[group]
                gap = now - end
                notes[label] = ("" if known else "\u2264") + _duration(gap)
                # An upper bound over the threshold is not evidence of a long
                # gap -- most of it may be the previous scan. Mark only what
                # the times actually establish.
                marks[label] = known and gap > gap_warn
            if label in extra:
                notes[label] = (notes.get(label, "") + " " + extra[label]).strip()
            if now is not None:
                length = (duration or {}).get(label)
                last[group] = (now + float(length or 0.0), length is not None)
    return summary_table(table, out_html, links=links, notes=notes, marks=marks,
                         **kwargs)


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s" if minutes else f"{rest}s"
