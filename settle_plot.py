"""Render the settle-threshold grid (settle_analyze.py output) as one self-contained HTML page.

    python3 settle_plot.py settle/grid.json --out settle/plots.html

The page shows, for every (time t, speed threshold v) cell, how well the early "slow shapes still inside" count
predicts the shapes left at 10 s: a 2D heatmap (click a cell for its numbers), a 3D surface, the per-t baseline
(inside at t, any speed) against the best v per t, and the top-10 table. Plotly comes from cdnjs (pinned).
"""

import argparse
import json
import math
import os
import sys

PLOTLY_URL = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"
METRIC_ORDER = ["spearman", "pearson", "f1", "precision", "recall"]
SKIP_KEYS = {"spearman_lo", "spearman_hi"}


def _clean(x):
    """NaN / inf -> None, recursively, so the embedded JSON is valid."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _is_grid(m, nt, nv):
    return isinstance(m, list) and len(m) == nt and all(isinstance(r, list) and len(r) == nv for r in m)


def normalize(grid):
    """Validate the grid.json schema and reduce it to what the page needs."""
    t = [float(x) for x in grid["t"]]
    v = [float(x) for x in grid["v"]]
    nt, nv = len(t), len(v)
    raw = grid.get("metrics", {})
    metrics = {}
    for name in METRIC_ORDER + sorted(k for k in raw if k not in METRIC_ORDER):
        if name in raw and name not in SKIP_KEYS:
            if not _is_grid(raw[name], nt, nv):
                raise ValueError(f"metrics.{name} is not a {nt}x{nv} t-major grid")
            metrics[name] = raw[name]
    if not metrics:
        raise ValueError("grid.json has no metrics")
    interval = None
    if _is_grid(raw.get("spearman_lo"), nt, nv) and _is_grid(raw.get("spearman_hi"), nt, nv):
        interval = {"lo": raw["spearman_lo"], "hi": raw["spearman_hi"]}

    # baseline: {metric: [per t]} (preferred), a bare per-t list (taken as spearman), or with its own "t" key.
    base_raw = grid.get("baseline") or {}
    baseline = {}
    if isinstance(base_raw, list):
        base_raw = {"spearman": base_raw}
    if isinstance(base_raw, dict):
        for k, val in base_raw.items():
            if k != "t" and isinstance(val, list) and len(val) == nt:
                baseline[k] = val

    side_w = {}
    for w, block in (grid.get("side_w") or {}).items():
        if isinstance(block, dict):
            side_w[str(w)] = {k: m for k, m in block.items() if _is_grid(m, nt, nv)}

    best = [b for b in (grid.get("best") or []) if isinstance(b, dict) and "t" in b and "v" in b]
    return _clean({
        "t": t, "v": v, "w": grid.get("w", 0.5), "n_runs": grid.get("n_runs"),
        "n_shapes": grid.get("n_shapes"), "metrics": metrics, "interval": interval,
        "baseline": baseline, "side_w": side_w, "best": best[:10],
    })


def render(data, source_name=""):
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return (PAGE.replace("__PLOTLY_URL__", PLOTLY_URL)
            .replace("__SOURCE__", source_name.replace("<", "&lt;"))
            .replace("__DATA__", payload))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grid", help="grid.json from settle_analyze.py")
    ap.add_argument("--out", default=None, help="output HTML (default: plots.html next to the grid)")
    args = ap.parse_args(argv)
    with open(args.grid) as f:
        grid = json.load(f)
    data = normalize(grid)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.grid)), "plots.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(render(data, os.path.basename(args.grid)))
    print(f"wrote {out} ({len(data['t'])} t x {len(data['v'])} v, metrics: {', '.join(data['metrics'])})")
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Settle Threshold Search</title>
<script src="__PLOTLY_URL__"></script>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1f; --muted: #6b6b70; --border: #e2e2de;
  --accent: #2b6cb0; --best: #d9480f; --grid: #e6e6e2; --hl: #fff4e6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141416; --panel: #1d1d20; --text: #ececf0; --muted: #9a9aa3; --border: #2e2e33;
    --accent: #6ea8fe; --best: #ff8c42; --grid: #2c2c31; --hl: #3a2a1c;
  }
}
:root[data-theme="dark"] {
  --bg: #141416; --panel: #1d1d20; --text: #ececf0; --muted: #9a9aa3; --border: #2e2e33;
  --accent: #6ea8fe; --best: #ff8c42; --grid: #2c2c31; --hl: #3a2a1c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 16px 48px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 15px; margin: 0 0 6px; }
.meta { color: var(--muted); }
.spacer { flex: 1; }
button, .seg label { font: inherit; color: var(--text); background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.explain { margin: 10px 0 14px; color: var(--muted); font-size: 13px; }
.explain b { color: var(--text); font-weight: 600; }
.explain p { margin: 2px 0; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
.seg { display: inline-flex; flex-wrap: wrap; gap: 4px; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg input:checked + label { background: var(--accent); border-color: var(--accent); color: #fff; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; min-width: 0; }
.row { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 12px; margin-bottom: 12px; }
.row2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; margin-bottom: 12px; }
@media (max-width: 900px) { .row, .row2 { grid-template-columns: minmax(0, 1fr); } }
.plot { width: 100%; height: 460px; }
.plot.short { height: 360px; }
.hint { color: var(--muted); font-size: 12px; margin: 0 0 4px; }
#detail table, #top table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
#detail td { padding: 3px 4px; border-bottom: 1px solid var(--border); }
#detail td:last-child { text-align: right; }
#detail .big { font-size: 18px; font-weight: 600; }
#top { overflow-x: auto; }
#top th, #top td { padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
#top th:first-child, #top td:first-child { text-align: left; }
#top th { color: var(--muted); font-weight: 500; }
#top tbody tr { cursor: pointer; }
#top tbody tr:hover, #top tbody tr.sel { background: var(--hl); }
.rank1 { color: var(--best); font-weight: 700; }
#err { display: none; color: #c92a2a; margin: 8px 0; }
</style>
</head>
<body>
<main>
<header>
  <h1>Settle threshold search</h1>
  <span class="meta" id="meta"></span>
  <span class="spacer"></span>
  <button id="theme" type="button" title="Switch light / dark">Theme: auto</button>
</header>
<div class="explain">
  <p><b>Time t</b> is when the early score is taken (s into the chunk test); <b>speed threshold v</b> (px/s): a shape
  still inside the chunk at t counts as stuck if its centroid speed over the last <span class="wv"></span> s is below v.</p>
  <p><b>Spearman &rho;</b> / <b>Pearson r</b>: rank / linear correlation across runs between that stuck count and the
  shapes still inside at 10 s (1 = perfect). <b>F1, precision, recall</b>: per shape, "stuck at t" vs "still inside at
  10 s". <b>Baseline</b>: count every shape inside at t, whatever its speed.</p>
</div>
<div id="err"></div>
<div class="controls">
  <span class="meta">Metric</span>
  <span class="seg" id="metrics"></span>
</div>
<div class="row">
  <div class="card">
    <h2>Heatmap: <span class="mname"></span> by t and v</h2>
    <p class="hint">Numbered rings mark the 10 best cells for this metric (1 = best). Hover for numbers, click to pin a cell.</p>
    <div id="heat" class="plot"></div>
  </div>
  <div class="card" id="detail"></div>
</div>
<div class="row2">
  <div class="card">
    <h2>Surface: <span class="mname"></span></h2>
    <p class="hint">Drag to rotate. Dots are the top-10 cells.</p>
    <div id="surf" class="plot short"></div>
  </div>
  <div class="card">
    <h2>Baseline vs best speed threshold</h2>
    <p class="hint">Per t: the baseline (any speed) and the best v for <span class="mname"></span>.</p>
    <div id="base" class="plot short"></div>
  </div>
</div>
<div class="card" id="top"></div>
</main>
<script id="grid-data" type="application/json">__DATA__</script>
<script>
"use strict";
window.addEventListener("error", ev => {
  const el = document.getElementById("err");
  el.style.display = "block"; el.textContent = "Error: " + ev.message;
});
const D = JSON.parse(document.getElementById("grid-data").textContent);
const NAMES = { spearman: "Spearman ρ", pearson: "Pearson r", f1: "F1", precision: "Precision", recall: "Recall" };
const nm = k => NAMES[k] || k;
const T = D.t, V = D.v, NT = T.length, NV = V.length;
const metricKeys = Object.keys(D.metrics);
let metric = metricKeys.includes("spearman") ? "spearman" : metricKeys[0];
let sel = null;  // [ti, vi]

const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const fv = x => String(Number(x.toPrecision(3)));
const ft = x => x.toFixed(2);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const val = (k, ti, vi) => { const m = D.metrics[k]; return m ? m[ti][vi] : null; };
const nearest = (arr, x) => arr.reduce((b, a, i) => Math.abs(a - x) < Math.abs(arr[b] - x) ? i : b, 0);

function topCells(k, n = 10) {
  const cells = [];
  for (let i = 0; i < NT; i++) for (let j = 0; j < NV; j++) {
    const z = D.metrics[k][i][j];
    if (z !== null && Number.isFinite(z)) cells.push([z, i, j]);
  }
  cells.sort((a, b) => b[0] - a[0]);
  return cells.slice(0, n);
}

function layoutBase() {
  const text = css("--text");
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: text, size: 12 }, margin: { l: 56, r: 12, t: 8, b: 44 },
    hoverlabel: { bgcolor: css("--panel"), bordercolor: css("--border"), font: { color: text } },
  };
}

function cellText(i, j) {
  let s = `t = ${ft(T[i])} s<br>v = ${fv(V[j])} px/s`;
  for (const k of metricKeys) {
    const x = val(k, i, j);
    s += `<br>${k === metric ? "<b>" : ""}${nm(k)}: ${fmt(x)}${k === metric ? "</b>" : ""}`;
    if (k === "spearman" && D.interval) s += `  [${fmt(D.interval.lo[i][j])}, ${fmt(D.interval.hi[i][j])}] 90%`;
  }
  return s;
}

function drawHeat() {
  const L = layoutBase();
  const z = [], text = [];
  for (let j = 0; j < NV; j++) {
    z.push(T.map((_, i) => val(metric, i, j)));
    text.push(T.map((_, i) => cellText(i, j)));
  }
  const ylab = V.map(fv);
  const top = topCells(metric);
  const traces = [{
    type: "heatmap", x: T, y: ylab, z, text, hoverinfo: "text", colorscale: "Viridis",
    colorbar: { title: { text: nm(metric), side: "right" }, thickness: 12 }, xgap: 1, ygap: 1,
  }, {
    type: "scatter", mode: "markers+text", x: top.map(c => T[c[1]]), y: top.map(c => ylab[c[2]]),
    text: top.map((_, r) => String(r + 1)), textfont: { color: "#fff", size: 10 },
    marker: { size: top.map((_, r) => r === 0 ? 20 : 15), color: css("--best"),
              line: { color: "#fff", width: 1.5 }, symbol: top.map((_, r) => r === 0 ? "star" : "circle") },
    hoverinfo: "skip", showlegend: false,
  }];
  if (sel) traces.push({
    type: "scatter", mode: "markers", x: [T[sel[0]]], y: [ylab[sel[1]]], hoverinfo: "skip", showlegend: false,
    marker: { size: 24, symbol: "square-open", color: L.font.color, line: { width: 2.5 } },
  });
  Plotly.react("heat", traces, {
    ...L,
    xaxis: { title: "time t (s)", gridcolor: css("--grid"), dtick: 0.5 },
    yaxis: { title: "speed threshold v (px/s)", type: "category", gridcolor: css("--grid") },
  }, { responsive: true, displaylogo: false });
}

function drawSurf() {
  const L = layoutBase();
  const z = [], text = [];
  for (let j = 0; j < NV; j++) {
    z.push(T.map((_, i) => val(metric, i, j)));
    text.push(T.map((_, i) => cellText(i, j)));
  }
  const top = topCells(metric);
  const ax = t => ({ title: t, gridcolor: css("--grid"), backgroundcolor: "rgba(0,0,0,0)", color: L.font.color });
  Plotly.react("surf", [{
    type: "surface", x: T, y: V, z, text, hoverinfo: "text", colorscale: "Viridis", showscale: false,
    contours: { z: { show: true, usecolormap: true, project: { z: true } } },
  }, {
    type: "scatter3d", mode: "markers", x: top.map(c => T[c[1]]), y: top.map(c => V[c[2]]),
    z: top.map(c => c[0]), marker: { size: 4, color: css("--best") }, hoverinfo: "skip", showlegend: false,
  }], {
    ...L, margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: { xaxis: ax("t (s)"), yaxis: { ...ax("v (px/s)"), type: "log" }, zaxis: ax(nm(metric)),
             camera: { eye: { x: 1.3, y: -1.3, z: 0.7 } } },
  }, { responsive: true, displaylogo: false });
}

function drawBase() {
  const L = layoutBase();
  const traces = [];
  const bestV = T.map((_, i) => {
    let b = -1;
    for (let j = 0; j < NV; j++) { const x = val(metric, i, j); if (x !== null && (b < 0 || x > val(metric, i, b))) b = j; }
    return b;
  });
  traces.push({
    type: "scatter", mode: "lines+markers", name: `best v (${nm(metric)})`, x: T,
    y: bestV.map((j, i) => j < 0 ? null : val(metric, i, j)), line: { color: css("--accent") },
    text: bestV.map(j => j < 0 ? "" : `v = ${fv(V[j])} px/s`),
    hovertemplate: "t = %{x:.2f} s<br>%{text}<br>" + nm(metric) + " = %{y:.3f}<extra>best v</extra>",
  });
  if (metric === "spearman" && D.interval) {
    traces.push({
      type: "scatter", mode: "lines", x: T.concat([...T].reverse()), showlegend: false, hoverinfo: "skip",
      y: bestV.map((j, i) => j < 0 ? null : D.interval.hi[i][j])
        .concat(bestV.map((j, i) => j < 0 ? null : D.interval.lo[i][j]).reverse()),
      fill: "toself", fillcolor: "rgba(110,140,200,0.18)", line: { width: 0 }, name: "90% interval",
    });
  }
  const bk = D.baseline[metric] ? metric : null;
  if (bk) traces.push({
    type: "scatter", mode: "lines+markers", name: "baseline (any speed)", x: T, y: D.baseline[bk],
    line: { color: css("--muted"), dash: "dash" },
    hovertemplate: "t = %{x:.2f} s<br>" + nm(bk) + " = %{y:.3f}<extra>baseline</extra>",
  });
  const note = bk ? "" : (Object.keys(D.baseline).length ? `no baseline for ${nm(metric)}` : "no baseline in grid.json");
  Plotly.react("base", traces, {
    ...L,
    xaxis: { title: "time t (s)", gridcolor: css("--grid") }, yaxis: { title: nm(metric), gridcolor: css("--grid") },
    legend: { orientation: "h", y: 1.12, x: 0 },
    annotations: note ? [{ text: note, xref: "paper", yref: "paper", x: 1, y: 0, showarrow: false,
                           xanchor: "right", font: { color: css("--muted") } }] : [],
  }, { responsive: true, displaylogo: false });
}

function drawDetail() {
  const el = document.getElementById("detail");
  if (!sel) { el.innerHTML = "<h2>Cell</h2><p class='hint'>Click a heatmap cell or a table row.</p>"; return; }
  const [i, j] = sel;
  const rank = topCells(metric, NT * NV).findIndex(c => c[1] === i && c[2] === j) + 1;
  let h = `<h2>t = ${ft(T[i])} s, v = ${fv(V[j])} px/s</h2>`;
  h += `<p class="hint">Rank ${rank || "–"} of ${NT * NV} by ${nm(metric)}</p><table>`;
  for (const k of metricKeys) {
    h += `<tr><td>${nm(k)}</td><td class="${k === metric ? "big" : ""}">${fmt(val(k, i, j))}</td></tr>`;
    if (k === "spearman" && D.interval)
      h += `<tr><td>&nbsp;&nbsp;90% interval</td><td>[${fmt(D.interval.lo[i][j])}, ${fmt(D.interval.hi[i][j])}]</td></tr>`;
  }
  for (const k of Object.keys(D.baseline)) {
    const b = D.baseline[k][i], x = val(k, i, j);
    const d = (b !== null && x !== null) ? ` (${x - b >= 0 ? "+" : ""}${fmt(x - b)})` : "";
    h += `<tr><td>Baseline ${nm(k)} at t</td><td>${fmt(b)}${d}</td></tr>`;
  }
  for (const [w, blk] of Object.entries(D.side_w)) for (const [k, m] of Object.entries(blk))
    h += `<tr><td>${nm(k)}, window ${w} s</td><td>${fmt(m[i][j])}</td></tr>`;
  el.innerHTML = h + "</table>";
}

function drawTop() {
  const el = document.getElementById("top");
  // grid.json's own "best" list when looking at spearman; otherwise the top 10 of the chosen metric.
  let rows;
  if (metric === "spearman" && D.best.length) rows = D.best.map(b => [nearest(T, b.t), nearest(V, b.v)]);
  else rows = topCells(metric).map(c => [c[1], c[2]]);
  const cols = metricKeys;
  let h = `<h2>Top 10 by ${nm(metric)}</h2><p class="hint">Click a row to show it on the heatmap.</p>`;
  h += "<table><thead><tr><th>#</th><th>t (s)</th><th>v (px/s)</th>";
  for (const k of cols) h += `<th>${nm(k)}</th>`;
  if (D.interval) h += "<th>ρ 90% interval</th>";
  h += "</tr></thead><tbody>";
  rows.forEach(([i, j], r) => {
    const s = sel && sel[0] === i && sel[1] === j ? " class='sel'" : "";
    h += `<tr data-i="${i}" data-j="${j}"${s}><td class="${r === 0 ? "rank1" : ""}">${r + 1}</td>` +
         `<td>${ft(T[i])}</td><td>${fv(V[j])}</td>`;
    for (const k of cols) h += `<td${k === metric ? " style='font-weight:600'" : ""}>${fmt(val(k, i, j))}</td>`;
    if (D.interval) h += `<td>[${fmt(D.interval.lo[i][j])}, ${fmt(D.interval.hi[i][j])}]</td>`;
    h += "</tr>";
  });
  el.innerHTML = h + "</tbody></table>";
  el.querySelectorAll("tbody tr").forEach(tr => tr.addEventListener("click", () => {
    select(+tr.dataset.i, +tr.dataset.j);
  }));
}

function select(i, j) { sel = [i, j]; drawHeat(); drawDetail(); drawTop(); }

function drawAll() {
  document.querySelectorAll(".mname").forEach(e => e.textContent = nm(metric));
  drawHeat(); drawSurf(); drawBase(); drawDetail(); drawTop();
}

function init() {
  const meta = [];
  if (D.n_runs !== null && D.n_runs !== undefined) meta.push(`${D.n_runs} runs`);
  if (D.n_shapes) meta.push(`${D.n_shapes} shapes`);
  meta.push(`${NT} t × ${NV} v`, `speed window ${D.w} s`);
  document.getElementById("meta").textContent = meta.join(" · ");
  document.querySelectorAll(".wv").forEach(e => e.textContent = D.w);
  const seg = document.getElementById("metrics");
  for (const k of metricKeys) {
    seg.insertAdjacentHTML("beforeend",
      `<input type="radio" name="m" id="m-${k}" value="${k}"${k === metric ? " checked" : ""}><label for="m-${k}">${nm(k)}</label>`);
  }
  seg.addEventListener("change", e => { metric = e.target.value; drawAll(); });
  const top = topCells(metric, 1)[0];
  if (top) sel = [top[1], top[2]];
  const heatEl = document.getElementById("heat");
  const modes = ["auto", "light", "dark"];
  const btn = document.getElementById("theme");
  let mode = "auto";
  try { mode = localStorage.getItem("settle-theme") || "auto"; } catch (e) {}
  const applyTheme = () => {
    if (mode === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", mode);
    btn.textContent = "Theme: " + mode;
  };
  btn.addEventListener("click", () => {
    mode = modes[(modes.indexOf(mode) + 1) % 3];
    try { localStorage.setItem("settle-theme", mode); } catch (e) {}
    applyTheme(); drawAll();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
  applyTheme();
  drawAll();
  // Plotly adds .on() to the div on its first plot, so bind clicks after drawAll.
  heatEl.on("plotly_click", ev => {
    const p = ev.points.find(q => q.data.type === "heatmap") || ev.points[0];
    if (!p) return;
    const j = V.map(fv).indexOf(String(p.y));
    if (j >= 0) select(nearest(T, +p.x), j);
  });
}

if (typeof Plotly === "undefined") {
  const e = document.getElementById("err");
  e.style.display = "block";
  e.textContent = "Plotly failed to load from __PLOTLY_URL__ (offline?). The page needs network access once.";
} else {
  try { init(); } catch (e) {
    const el = document.getElementById("err"); el.style.display = "block"; el.textContent = "Error: " + e;
    throw e;
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
