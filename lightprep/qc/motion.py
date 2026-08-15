"""A self-contained motion QC page: corrected volumes, scrubbable, over FD.

Motion correction is judged by watching it. A number says a run had a mean
framewise displacement of 2.5mm; only looking at the frames tells you whether
what remains is a head that now sits still, or a head that is still moving with
the estimator chasing it. So the page puts the corrected series next to the FD
trace and lets you drag through time on either.

It renders with NiiVue, which reads NIfTI in the browser and gives real
three-plane views with intensity windowing, rather than a strip of baked PNGs.

Comparing estimators is the other reason this exists. Pass more than one
corrected volume -- ``{"niimath": ..., "fsl": ...}`` -- and they scrub together
frame for frame, which is what shows where two methods disagree. Pass one and
the page collapses to a single viewer, which is the ordinary case.

Where the volumes come from is a choice, because a corrected run is ~100MB and
a single-file page cannot carry that at full resolution:

``embed``  (default)
    Quantised, downsampled copies inlined as base64. One file, opens from
    disk, no server, no network -- at the cost of 4mm display voxels.
``link``
    The real files at full resolution, staged next to the report and served
    from it -- ``serve(report)`` starts the server and prints the URL. The
    page is ~2MB and nothing is quantised. Staging uses hard links where the
    filesystem allows, so it costs no disk.
``pick``
    The same, without a server, for when you cannot run one -- viewing on
    another machine, say. The page opens empty and you hand it each file.
    ``link`` is preferable whenever serving is possible: the tool already
    knows which files it wants, so it should not ask.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

#: Radius used to convert rotations to surface displacement in FD, following
#: Power et al. (2012). 50mm is their value: roughly the cortex-to-centre
#: distance, so a radian of rotation moves cortex by 50mm.
FD_RADIUS_MM = 50.0

#: In-plane downsampling for the embedded copies. The volumes are for looking
#: at, not measuring: a corrected run is ~100MB, which no single-file page can
#: carry, and gross displacement is perfectly visible at half resolution.
DEFAULT_DOWNSAMPLE = 2

#: Percentiles the display intensity is clipped to before quantisation.
DEFAULT_CLIP = (0.5, 99.5)


def framewise_displacement(parameters, radius: float = FD_RADIUS_MM) -> np.ndarray:
    """Power framewise displacement from a six-column motion parameter file.

    Args:
        parameters: Path to a ``motion.par``, or an ``(n_frames, 6)`` array in
            MCFLIRT's convention -- three rotations in radians, then three
            translations in mm.
        radius: Sphere radius converting rotation to displacement.

    Returns:
        ``(n_frames - 1,)`` displacement in mm. The first frame has no
        predecessor and so no FD.

    Raises:
        ValueError: If the parameters are not six columns.
    """
    m = np.asarray(parameters if not isinstance(parameters, (str, Path))
                   else np.loadtxt(parameters), dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 6:
        raise ValueError(f"expected (n_frames, 6) motion parameters, got {m.shape}")
    d = np.abs(np.diff(m, axis=0))
    return d[:, 3:6].sum(1) + radius * d[:, :3].sum(1)


def find_niivue() -> Path | None:
    """A NiiVue UMD bundle on this machine, if one is installed.

    nilearn ships one, which spares this module a vendored copy and a network
    fetch. Returns None if nothing is found, in which case the caller must say
    where to get it.
    """
    try:
        import nilearn
    except ImportError:
        return None
    candidate = Path(nilearn.__file__).parent / "_assets" / "js" / "niivue.umd.js"
    return candidate if candidate.exists() else None


def _display_volume(src, downsample: int, clip) -> bytes:
    """A small uint8 copy of a 4D series, gzipped, for embedding.

    Quantisation is done once over the whole run rather than per frame, so a
    frame that darkens on screen darkened in the data -- per-frame windowing
    would hide exactly the intensity steps that mark a swallow or a spike.
    """
    img = nib.load(str(src))
    data = img.get_fdata(dtype=np.float64)
    if data.ndim == 3:
        data = data[..., None]
    step = max(1, int(downsample))
    data = data[::step, ::step, ::step]

    lo, hi = np.percentile(data[np.isfinite(data)], clip)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((data - lo) / (hi - lo), 0, 1)
    out = np.rint(scaled * 255).astype(np.uint8)

    affine = img.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag([step, step, step])
    small = nib.Nifti1Image(out, affine)
    small.header.set_xyzt_units(*img.header.get_xyzt_units())
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fh:
        fh.write(small.to_bytes())
    return buf.getvalue()


def _relative_url(target: Path, start: Path) -> str:
    """A URL from the report to a file, relative where possible.

    Relative keeps the pair movable together; an absolute file:// URL would
    break the moment the directory was copied anywhere else.
    """
    import os
    from urllib.request import pathname2url
    try:
        rel = os.path.relpath(target.resolve(), start.resolve())
    except ValueError:                          # different drive, Windows
        return "file://" + pathname2url(str(target.resolve()))
    return pathname2url(rel)


def _stage(src: Path, out_html: Path, label: str) -> str:
    """Put a volume beside the report and return its relative URL.

    Hard link first, symlink next, copy last. A 100MB run should not be
    duplicated just to be served, but it must be reachable from one directory
    so that serving the report does not mean serving the whole filesystem.
    """
    room = out_html.parent / f"{out_html.stem}_data"
    room.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    dst = room / f"{safe}__{src.name}"
    if not dst.exists():
        try:
            dst.hardlink_to(src.resolve())
        except (OSError, AttributeError):
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
    from urllib.request import pathname2url
    return pathname2url(f"{room.name}/{dst.name}")


def serve(report, port: int = 0, open_browser: bool = True):
    """Serve a ``link`` report and print its URL.

    Browsers refuse cross-origin reads from ``file://``, so a linked report
    needs HTTP. Only the report's own directory is exposed -- which is why
    ``stage`` puts the volumes there rather than serving whatever common
    ancestor the originals happened to share.

    Args:
        report: The HTML written by :func:`motion_report`.
        port: TCP port; 0 picks a free one.
        open_browser: Open the page once the server is up.

    Blocks until interrupted.
    """
    import functools
    import http.server
    import socketserver
    import webbrowser

    report = Path(report).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(report.parent))
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/{report.name}"
        print(f"serving {report.parent}\n{url}\nCtrl-C to stop", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def _as_traces(fd) -> dict:
    """Normalise the fd argument to {label: list of floats}."""
    if fd is None:
        return {}
    if isinstance(fd, dict):
        items = fd.items()
    elif isinstance(fd, (str, Path)) or np.ndim(fd) >= 1:
        items = [("FD", fd)]
    else:
        raise TypeError(f"cannot read FD from {type(fd)}")
    out = {}
    for label, series in items:
        if isinstance(series, (str, Path)):    # a motion.par on disk
            series = np.loadtxt(series)
        arr = np.asarray(series, dtype=np.float64)
        if arr.ndim == 2:                      # six columns, not FD yet
            arr = framewise_displacement(arr)
        if arr.ndim != 1:
            raise ValueError(
                f"FD series {label!r} has shape {arr.shape}; expected a 1D "
                f"trace or an (n_frames, 6) motion parameter array")
        out[str(label)] = [float(v) for v in arr]
    return out


def motion_report(volumes, out_html, *, fd=None, title: str = "motion QC",
                  subtitle: str = "", niivue_js=None, source: str = "embed",
                  stage: bool = True,
                  downsample: int = DEFAULT_DOWNSAMPLE, clip=DEFAULT_CLIP,
                  fd_threshold: float = 0.5) -> Path:
    """Write a self-contained motion QC page.

    Args:
        volumes: ``{label: path}`` of motion-corrected 4D series, one per
            method being shown. A single path or a one-entry mapping gives the
            ordinary single-viewer page.
        out_html: Where to write the report.
        fd: ``{label: series}``, a single series, or a path to a ``motion.par``
            (an ``(n, 6)`` array is converted with
            :func:`framewise_displacement`). Optional.
        title: Page heading.
        subtitle: Smaller line under it -- subject and run, typically.
        niivue_js: NiiVue UMD bundle to inline. Defaults to
            :func:`find_niivue`.
        source: Where the viewer gets its voxels -- ``embed`` (default),
            ``link`` or ``pick``. See the module docstring; ``downsample`` and
            ``clip`` apply only to ``embed``.
        stage: For ``link``, put the volumes in a folder beside the report so
            the pair is self-contained and a server needs to expose only that
            folder. Hard-linked where possible, so it costs no disk. Turn it
            off to link the originals where they lie.
        downsample: In-plane/through-plane step for the embedded copies.
        clip: Percentiles to window the display intensities to.
        fd_threshold: Drawn as a line on the trace, and used for the
            percent-over-threshold readout.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If a volume or the NiiVue bundle is missing.
    """
    if isinstance(volumes, (str, Path)):
        volumes = {"corrected": volumes}
    volumes = {str(k): Path(v) for k, v in dict(volumes).items()}
    for label, path in volumes.items():
        if not path.exists():
            raise FileNotFoundError(f"volume for {label!r} not found: {path}")

    bundle = Path(niivue_js) if niivue_js else find_niivue()
    if bundle is None or not Path(bundle).exists():
        raise FileNotFoundError(
            "no NiiVue bundle found. Install nilearn, which ships one, or pass "
            "niivue_js=<path to niivue.umd.js>.")

    if source not in ("embed", "link", "pick"):
        raise ValueError(f"source must be embed, link or pick, got {source!r}")

    out_html = Path(out_html)
    payload, n_frames = {}, 0
    for label, path in volumes.items():
        img = nib.load(str(path))
        n_frames = max(n_frames, img.shape[3] if img.ndim == 4 else 1)
        if source == "embed":
            blob = _display_volume(path, downsample, clip)
            payload[label] = {"mode": "embed", "name": path.name,
                              "data": base64.b64encode(blob).decode("ascii")}
        elif source == "link":
            url = (_stage(path, out_html, label) if stage
                   else _relative_url(path, out_html.parent))
            payload[label] = {"mode": "link", "name": path.name, "url": url}
        else:
            payload[label] = {"mode": "pick", "name": path.name,
                              "hint": str(path)}

    traces = _as_traces(fd)
    html = _HTML.replace("/*NIIVUE*/", Path(bundle).read_text(encoding="utf-8"))
    html = html.replace("/*DATA*/", json.dumps({
        "volumes": payload,
        "traces": traces,
        "nFrames": n_frames,
        "title": title,
        "subtitle": subtitle,
        "threshold": fd_threshold,
        "source": source,
    }))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>motion QC</title>
<style>
:root{--bg:#0e1116;--panel:#171b22;--line:#2a313c;--ink:#e6edf3;--dim:#8b949e;
      --accent:#4a9eff;--warn:#f0883e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:14px 18px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:16px;font-weight:600}
.sub{color:var(--dim);font-size:13px;margin-top:2px}
#viewers{display:flex;gap:10px;padding:10px 18px;flex-wrap:wrap}
.viewer{flex:1 1 380px;min-width:320px;background:var(--panel);
        border:1px solid var(--line);border-radius:8px;overflow:hidden}
.viewer h2{margin:0;padding:7px 11px;font-size:12px;font-weight:600;
           letter-spacing:.04em;text-transform:uppercase;color:var(--dim);
           border-bottom:1px solid var(--line)}
canvas{width:100%;height:300px;display:block;cursor:ew-resize}
#bar{padding:4px 18px 2px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#frame{font-variant-numeric:tabular-nums;font-weight:600}
#stats{color:var(--dim);font-size:13px}
#stats b{color:var(--ink);font-weight:600}
#plot{padding:0 18px 16px}
svg{width:100%;height:190px;display:block;background:var(--panel);
    border:1px solid var(--line);border-radius:8px;cursor:ew-resize;
    touch-action:none}
.hint{padding:0 18px 16px;color:var(--dim);font-size:12px}
.pick{padding:7px 11px;border-bottom:1px solid var(--line);display:flex;
      gap:9px;align-items:center;font-size:12px;color:var(--dim)}
.pick span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.err{padding:9px 11px;color:var(--warn);font-size:12px;line-height:1.45}
.legend{display:inline-flex;align-items:center;gap:5px;margin-right:12px}
.swatch{width:11px;height:3px;border-radius:2px;display:inline-block}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="subtitle"></div></header>
<div id="viewers"></div>
<div id="bar"><span id="frame"></span><span id="stats"></span></div>
<div id="plot"></div>
<div class="hint" id="hint">Drag anywhere on a viewer or on the trace to scrub
  frames. Arrow keys step; Home/End jump to the ends.</div>
<script>/*NIIVUE*/</script>
<script>
const D = /*DATA*/;
const NV = (window.niivue || window.Niivue || {});
const Niivue = NV.Niivue || window.Niivue;
const COLORS = ["#4a9eff","#f0883e","#3fb950","#db61a2"];
let frame = 0, viewers = [];

function b64ToBlobUrl(b64){
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) buf[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([buf], {type:"application/gzip"}));
}

async function build(){
  document.getElementById("title").textContent = D.title;
  document.getElementById("subtitle").textContent = D.subtitle;
  const host = document.getElementById("viewers");
  for (const [label, spec] of Object.entries(D.volumes)){
    const card = document.createElement("div");
    card.className = "viewer";
    const h = document.createElement("h2"); h.textContent = label;
    const cv = document.createElement("canvas");
    card.appendChild(h);
    if (spec.mode === "pick"){
      const bar = document.createElement("div");
      bar.className = "pick";
      const btn = document.createElement("input");
      btn.type = "file"; btn.accept = ".nii,.nii.gz";
      const note = document.createElement("span");
      note.textContent = spec.hint || spec.name;
      note.title = spec.hint || "";
      bar.appendChild(btn); bar.appendChild(note);
      card.appendChild(bar);
      btn.onchange = async e => {
        const f = e.target.files[0];
        if (!f) return;
        // a File is loadable exactly like an embedded blob: NiiVue picks the
        // format from the name, and nothing is copied or converted
        await load(nv, URL.createObjectURL(f), f.name);
        note.textContent = f.name;
        setFrame(frame);
      };
    }
    card.appendChild(cv); host.appendChild(card);
    const nv = new Niivue({backColor:[0.06,0.07,0.09,1], show3Dcrosshair:false,
                           crosshairColor:[0.29,0.62,1,0.85], textHeight:0.04,
                           dragAndDropEnabled: spec.mode === "pick"});
    nv.attachToCanvas(cv);
    nv.onImageLoaded = () => setFrame(frame);
    try {
      if (spec.mode === "embed")
        await load(nv, b64ToBlobUrl(spec.data), spec.name || label + ".nii.gz");
      else if (spec.mode === "link")
        await load(nv, spec.url, spec.name);
    } catch (err) {
      // The path is known and was tried; only now is it worth asking. This is
      // the file:// case -- the browser refused the read, not the disk.
      const msg = document.createElement("div");
      msg.className = "err";
      msg.textContent = "Could not read " + (spec.url || spec.name) +
        ". Opened from file://, browsers refuse this. Serve it instead -- " +
        "lightprep.qc.serve(report) -- or choose the file here:";
      const btn = document.createElement("input");
      btn.type = "file"; btn.accept = ".nii,.nii.gz";
      btn.onchange = async e => {
        const f = e.target.files[0];
        if (!f) return;
        await load(nv, URL.createObjectURL(f), f.name);
        msg.remove(); btn.remove(); setFrame(frame);
      };
      card.appendChild(msg); card.appendChild(btn);
      nv.opts.dragAndDropEnabled = true;
    }
    viewers.push(nv);
    scrubbable(cv);
  }
  drawPlot();
  setFrame(0);
}

async function load(nv, url, name){
  await nv.loadVolumes([{url, name}]);
  nv.setSliceType(nv.sliceTypeMultiplanar);
}

// --- the FD trace, as SVG so it scales and stays crisp -------------------
const PAD = {l:44, r:12, t:12, b:24};
let plotW = 900, plotH = 190, maxFD = 1, nPts = 1;

function drawPlot(){
  const labels = Object.keys(D.traces);
  const host = document.getElementById("plot");
  if (!labels.length){ host.innerHTML = ""; return; }
  nPts = Math.max(...labels.map(k => D.traces[k].length));
  maxFD = Math.max(D.threshold*1.4,
                   ...labels.map(k => Math.max(...D.traces[k])));
  const x = i => PAD.l + i*(plotW-PAD.l-PAD.r)/Math.max(1,nPts-1);
  const y = v => plotH-PAD.b - (v/maxFD)*(plotH-PAD.t-PAD.b);
  let s = `<svg viewBox="0 0 ${plotW} ${plotH}" preserveAspectRatio="none" id="svg">`;
  s += `<line x1="${PAD.l}" y1="${y(0)}" x2="${plotW-PAD.r}" y2="${y(0)}"
        stroke="#2a313c"/>`;
  s += `<line x1="${PAD.l}" y1="${y(D.threshold)}" x2="${plotW-PAD.r}"
        y2="${y(D.threshold)}" stroke="#f0883e" stroke-dasharray="4 4"
        opacity=".7"/>`;
  s += `<text x="4" y="${y(D.threshold)+4}" fill="#8b949e" font-size="11">
        ${D.threshold}</text>`;
  s += `<text x="4" y="${y(maxFD)+9}" fill="#8b949e" font-size="11">
        ${maxFD.toFixed(1)}</text>`;
  labels.forEach((k,i) => {
    const pts = D.traces[k].map((v,j) => `${x(j)},${y(v)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="${COLORS[i%4]}"
          stroke-width="1.4" vector-effect="non-scaling-stroke"/>`;
  });
  s += `<line id="cursor" x1="0" y1="${PAD.t}" x2="0" y2="${plotH-PAD.b}"
        stroke="#e6edf3" stroke-width="1" opacity=".9"/>`;
  s += `</svg>`;
  host.innerHTML = s;
  scrubbable(document.getElementById("svg"));

  const bar = document.getElementById("stats");
  bar.innerHTML = labels.map((k,i) => {
    const a = D.traces[k], mean = a.reduce((p,c)=>p+c,0)/a.length;
    const over = 100*a.filter(v=>v>D.threshold).length/a.length;
    return `<span class="legend"><i class="swatch"
      style="background:${COLORS[i%4]}"></i>${k}: mean <b>${mean.toFixed(3)}</b>
      mm, max <b>${Math.max(...a).toFixed(2)}</b>,
      <b>${over.toFixed(0)}%</b> over ${D.threshold}</span>`;
  }).join("");
}

// --- scrubbing ------------------------------------------------------------
function scrubbable(el){
  const move = ev => {
    const r = el.getBoundingClientRect();
    const cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    // the plot has margins; map through them so the cursor tracks the data
    const isSvg = el.tagName.toLowerCase() === "svg";
    const l = isSvg ? PAD.l*r.width/plotW : 0;
    const w = isSvg ? r.width - (PAD.l+PAD.r)*r.width/plotW : r.width;
    setFrame(Math.round(((cx-l)/Math.max(1,w))*(D.nFrames-1)));
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
  };
  el.addEventListener("pointerdown", ev => {
    ev.preventDefault(); move(ev);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}

function setFrame(f){
  frame = Math.max(0, Math.min(D.nFrames-1, f|0));
  viewers.forEach(nv => {
    if (nv.volumes && nv.volumes.length) nv.setFrame4D(nv.volumes[0].id, frame);
  });
  const cur = document.getElementById("cursor");
  if (cur){
    const x = PAD.l + (frame/Math.max(1,D.nFrames-1))*(plotW-PAD.l-PAD.r);
    cur.setAttribute("x1", x); cur.setAttribute("x2", x);
  }
  const labels = Object.keys(D.traces);
  // FD is a difference, so frame f pairs with fd[f-1]
  const here = labels.map(k => {
    const v = D.traces[k][frame-1];
    return v === undefined ? `${k} --` : `${k} ${v.toFixed(2)}`;
  }).join("   ");
  document.getElementById("frame").textContent =
    `frame ${frame+1} / ${D.nFrames}` + (here ? `    FD: ${here}` : "");
}

addEventListener("keydown", e => {
  const k = {ArrowLeft:-1, ArrowRight:1, ArrowDown:-10, ArrowUp:10}[e.key];
  if (k){ e.preventDefault(); setFrame(frame+k); }
  else if (e.key === "Home") setFrame(0);
  else if (e.key === "End") setFrame(D.nFrames-1);
});
addEventListener("resize", () => drawPlot() || setFrame(frame));
if (D.source === "pick")
  document.getElementById("hint").textContent =
    "Choose each file above, or drop it on its viewer -- full resolution, " +
    "nothing copied. Then drag on a viewer or the trace to scrub frames.";
build();
</script></body></html>
"""
