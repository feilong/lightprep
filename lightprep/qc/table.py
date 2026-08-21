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
                  subtitle: str = "", links=None, severe: float = SEVERE,
                  quiet: float = QUIET, note: str = "") -> Path:
    """Write a coloured grid of per-run, per-measure outlier percentages.

    Args:
        table: ``{row: {column: percentage}}``. Columns are taken from the
            first row and every row is read in that order, so a missing entry
            shows as blank rather than shifting the grid.
        out_html: Where to write.
        title, subtitle: Page heading and the line under it.
        links: ``{row: url}`` making a row label clickable -- normally its own
            report, so the grid is a way in rather than a dead end.
        severe: Percentage at which colour saturates.
        quiet: Percentage below which nothing is painted.
        note: Free text under the table, for what the thresholds were.

    Returns:
        The path written.
    """
    table = {str(k): dict(v) for k, v in dict(table).items()}
    if not table:
        raise ValueError("nothing to tabulate")
    columns = list(next(iter(table.values())))
    rows = [{"run": r, "url": (links or {}).get(r, ""),
             "values": [v.get(c) for c in columns]} for r, v in table.items()]

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(
        _HTML.replace("/*DATA*/", json.dumps(
            {"columns": columns, "rows": rows, "severe": float(severe),
             "quiet": float(quiet), "title": title, "subtitle": subtitle,
             "note": note})),
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
function paint(v){
  if (v == null || v < D.quiet) return "";
  const t = Math.min(1, Math.sqrt(v / D.severe));
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
  }); else rows.sort((a, b) => a.run.localeCompare(b.run));
  let h = "<thead><tr><th class='run'>run</th>";
  D.columns.forEach((c, i) => {
    h += `<th data-i="${i}">${c}` +
         (sortCol === i ? ` <span class="arrow">${desc ? "↓" : "↑"}</span>` : "") +
         "</th>";
  });
  h += "</tr></thead><tbody>";
  for (const r of rows){
    const label = r.url ? `<a href="${r.url}">${r.run}</a>` : r.run;
    h += `<tr><td class="run">${label}</td>`;
    r.values.forEach(v => {
      const s = paint(v);
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
      if (i === sortCol) desc = !desc; else { sortCol = i; desc = true; }
      draw();
    };
  });
}
draw();
</script></body></html>
"""
