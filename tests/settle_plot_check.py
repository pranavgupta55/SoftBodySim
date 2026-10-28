"""Check settle_plot.py against a synthetic grid.json (same schema as settle_analyze.py output).

    python3 tests/settle_plot_check.py [--outdir DIR] [--no-browser]

Writes DIR/grid.json and DIR/plots.html, checks the embedded data, then (if Google Chrome is installed and Plotly
can be fetched) loads the page headless, clicks a table row and switches metric, and checks the DOM for drawn plots
and no JS errors. Also saves DIR/plots.png.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import settle_plot  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def synthetic_grid(seed=0):
    """A plausible grid: rho peaks around t ~ 3 s, v ~ 4 px/s, with noise; f1 peaks nearby."""
    import random
    rng = random.Random(seed)
    t = [0.5 + 0.25 * i for i in range(19)]
    v = [round(0.5 * (80 ** (k / 29)), 4) for k in range(30)]
    def surf(ti, vi, t0, lv0, peak, st, sv):
        lv = math.log(v[vi])
        return peak * math.exp(-((t[ti] - t0) / st) ** 2 - ((lv - lv0) / sv) ** 2)
    def g(f):
        return [[f(i, j) for j in range(len(v))] for i in range(len(t))]
    sp = g(lambda i, j: 0.25 + surf(i, j, 3.2, math.log(4), 0.6, 2.0, 1.4) + rng.gauss(0, 0.015))
    pe = g(lambda i, j: sp[i][j] - 0.05 + rng.gauss(0, 0.02))
    pr = g(lambda i, j: min(1, 0.3 + surf(i, j, 4.0, math.log(2), 0.6, 2.5, 1.2)))
    rc = g(lambda i, j: min(1, 0.4 + 0.5 * (j / 29) + 0.02 * i))
    f1 = g(lambda i, j: 2 * pr[i][j] * rc[i][j] / (pr[i][j] + rc[i][j]))
    lo = g(lambda i, j: sp[i][j] - 0.08 - 0.03 * rng.random())
    hi = g(lambda i, j: min(1, sp[i][j] + 0.07 + 0.03 * rng.random()))
    sp[0][0] = float("nan")  # analysis may emit NaN for constant columns; the page must cope
    cells = sorted(((sp[i][j], i, j) for i in range(len(t)) for j in range(len(v)) if sp[i][j] == sp[i][j]),
                   reverse=True)[:10]
    return {
        "t": t, "v": v, "w": 0.5,
        "metrics": {"spearman": sp, "pearson": pe, "f1": f1, "precision": pr, "recall": rc,
                    "spearman_lo": lo, "spearman_hi": hi},
        "baseline": {"spearman": [0.2 + 0.05 * i ** 0.5 for i in range(len(t))],
                     "pearson": [0.15 + 0.05 * i ** 0.5 for i in range(len(t))],
                     "f1": [0.4 + 0.01 * i for i in range(len(t))]},
        "side_w": {"0.25": {"spearman": g(lambda i, j: sp[i][j] - 0.03)},
                   "1.0": {"spearman": g(lambda i, j: sp[i][j] - 0.01)}},
        "n_runs": 412,
        "best": [{"t": t[i], "v": v[j], "spearman": z, "f1": f1[i][j]} for z, i, j in cells],
    }


def check_static(html, grid):
    m = re.search(r'<script id="grid-data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "embedded data block missing"
    data = json.loads(m.group(1))  # strict: NaN would fail here
    assert data["t"] == grid["t"] and len(data["v"]) == len(grid["v"])
    assert data["metrics"]["spearman"][0][0] is None, "NaN should become null"
    assert set(data["metrics"]) == {"spearman", "pearson", "f1", "precision", "recall"}
    assert data["interval"] and len(data["interval"]["lo"]) == len(grid["t"])
    assert len(data["best"]) == 10 and data["n_runs"] == 412
    assert settle_plot.PLOTLY_URL in html and "2.27.0" in html
    for el in ('id="heat"', 'id="surf"', 'id="base"', 'id="top"', 'id="detail"', "prefers-color-scheme"):
        assert el in html, el
    # schema variants: bare-list baseline, no interval, no side_w
    g2 = dict(grid, baseline=grid["baseline"]["spearman"], side_w={})
    g2["metrics"] = {k: grid["metrics"][k] for k in ("spearman", "f1")}
    d2 = settle_plot.normalize(g2)
    assert d2["interval"] is None and list(d2["baseline"]) == ["spearman"]
    try:
        settle_plot.normalize(dict(grid, metrics={"spearman": [[1.0]]}))
        raise AssertionError("bad grid shape accepted")
    except ValueError:
        pass


# Appended to a copy of the page: exercise click-to-select and the metric switch, then report into the DOM.
DRIVER = """<script>
setTimeout(() => {
  const out = {};
  try {
    out.svgs = document.querySelectorAll(".main-svg").length;
    out.gl = document.querySelectorAll("#surf canvas").length;
    document.querySelectorAll("#top tbody tr")[3].click();
    out.detail1 = document.querySelector("#detail h2").textContent;
    document.getElementById("m-f1").click();
    out.topTitle = document.querySelector("#top h2").textContent;
    out.mname = document.querySelector(".mname").textContent;
    document.getElementById("theme").click();
    out.theme = document.documentElement.getAttribute("data-theme");
    document.getElementById("theme").click();
    document.getElementById("theme").click();  // back to auto (localStorage is shared with the screenshot run)
  } catch (e) { out.driverError = String(e); }
  out.err = document.getElementById("err").style.display === "block" ? document.getElementById("err").textContent : "";
  const pre = document.createElement("pre"); pre.id = "driver-out"; pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
}, 1500);
</script>"""


def check_browser(html_path, outdir):
    if not os.path.exists(CHROME):
        print("skip browser check: Google Chrome not found")
        return
    html = open(html_path).read()
    drv = os.path.join(outdir, "plots_driver.html")
    with open(drv, "w") as f:
        f.write(html.replace("</body>", DRIVER + "</body>"))
    prof = tempfile.mkdtemp(prefix="settle_plot_chrome_")
    base = [CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run", "--hide-scrollbars",
            f"--user-data-dir={prof}", "--window-size=1300,2100", "--virtual-time-budget=15000"]
    try:
        r = subprocess.run(base + ["--enable-logging=stderr", "--v=0", "--dump-dom", "file://" + drv],
                           capture_output=True, text=True, timeout=90)
        m = re.search(r'<pre id="driver-out">(.*?)</pre>', r.stdout, re.S)
        if 'Plotly failed to load' in r.stdout and not m:
            print("skip browser check: Plotly could not be fetched")
            return
        assert m, "driver output missing; stderr tail:\n" + r.stderr[-2000:]
        out = json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&"))
        print("browser:", out)
        if "Plotly failed to load" in out.get("err", ""):
            print("skip browser checks: Plotly could not be fetched (offline?)")
            return
        console_errs = [ln for ln in r.stderr.splitlines()
                        if "CONSOLE" in ln and ("Error" in ln or "Uncaught" in ln)]
        assert not out.get("err") and not out.get("driverError"), out
        assert not console_errs, console_errs
        assert out["svgs"] >= 3, "plots not drawn"
        assert out["topTitle"] == "Top 10 by F1" and out["mname"] == "F1"
        assert out["detail1"].startswith("t = ") and out["theme"] == "light"
        png = os.path.join(outdir, "plots.png")
        subprocess.run(base + [f"--screenshot={png}", "file://" + os.path.abspath(html_path)],
                       capture_output=True, timeout=90)
        if os.path.exists(png):
            print("screenshot:", png)
    finally:
        shutil.rmtree(prof, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    outdir = a.outdir or tempfile.mkdtemp(prefix="settle_plot_check_")
    os.makedirs(outdir, exist_ok=True)
    grid = synthetic_grid()
    gpath = os.path.join(outdir, "grid.json")
    with open(gpath, "w") as f:
        json.dump(grid, f)  # writes NaN like Python's default json does
    hpath = os.path.join(outdir, "plots.html")
    assert settle_plot.main([gpath, "--out", hpath]) == 0
    check_static(open(hpath).read(), grid)
    print("static checks ok")
    if not a.no_browser:
        check_browser(hpath, outdir)
    print("settle_plot_check OK")


if __name__ == "__main__":
    main()
