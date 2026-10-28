"""Settle-threshold search: which early score best predicts the 10 s chunk-test outcome.

    .venv/bin/python3 settle_analyze.py settle/runs.jsonl --out settle/grid.json

Reads chunk runs (one JSON object per line, written by settle_collect.py) with each shape's centroid sampled every
sample_dt and its exit time. For every (t, v) on a grid (t = 0.5..5.0 s step 0.25, v = 30 log-spaced thresholds
0.5..40 px/s) the early score of a run is

    pred(t, v) = #shapes still inside at t (exit_t null or > t) whose speed at t is under v,

speed(t) = |xy(t) - xy(t - w)| / w (w = 0.5 s by default; the window is clipped at t = 0). Scored against the run's
left_10 (shapes still inside at 10 s): Spearman rho and Pearson r across runs, a 90% bootstrap interval of rho
(resampling runs), and a per-shape classification (pred slow-and-inside vs still inside at 10 s): precision, recall,
F1 pooled over all shapes. The baseline is "inside at t, any speed". Side check: Spearman/F1 for w = 0.25 and 1.0.

Everything is vectorised over shapes and grid cells; bootstrap Spearman is computed exactly from resample counts
(weighted mid-ranks), so it equals ranking each resampled run list but avoids re-sorting.
"""
import argparse
import itertools
import json
import math
import sys
import time
import warnings

import numpy as np

T_GRID = tuple(round(0.5 + 0.25 * i, 2) for i in range(19))  # 0.5 .. 5.0
V_GRID = tuple(float(f"{x:.3g}") for x in np.geomspace(0.5, 40.0, 30))
SIDE_W = (0.25, 1.0)


# ---------------------------------------------------------------- loading
def load_runs(path, map_name=None):
    """Runs from a jsonl file. Skips blank and unparsable lines (e.g. a partial last line while the collector is
    still appending) with a note on stderr."""
    runs, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if map_name is not None and run.get("map") != map_name:
                continue
            runs.append(run)
    if bad:
        print(f"note: skipped {bad} unparsable line(s) in {path}", file=sys.stderr)
    return runs


def shape_table(runs, t_grid=T_GRID, windows=(0.5,) + SIDE_W):
    """Flatten runs into per-shape arrays.

    Returns dict: run (S,) int run index (non-decreasing), still (S,) bool (inside at 10 s), exit_t (S,) float (inf
    when still inside), speed {w: (S, T) float, nan where a position is missing}, left (R,) int from left_10."""
    t_grid = np.asarray(t_grid, float)
    run_idx, still, exit_t = [], [], []
    speed = {w: [] for w in windows}
    left = np.empty(len(runs), np.int64)
    for r, run in enumerate(runs):
        left[r] = run["left_10"]
        shapes = run["shapes"]
        n = len(shapes)
        if n == 0:
            continue
        dt = float(run.get("sample_dt", 0.1))
        kt = np.rint(t_grid / dt).astype(np.int64)
        K = int(kt.max()) + 1
        # one conversion for all of the run's needed samples, then scatter into a NaN-padded (n, K, 2) block
        cut = [s["xy"][:K] for s in shapes]
        lens = np.fromiter((len(c) for c in cut), np.int64, n)
        P = np.full((n, K, 2), np.nan)
        if lens.sum():
            flat = np.asarray(list(itertools.chain.from_iterable(cut)), float).reshape(-1, 2)
            rows = np.repeat(np.arange(n), lens)
            cols = np.arange(len(flat)) - np.repeat(np.cumsum(lens) - lens, lens)
            P[rows, cols] = flat
        for w in windows:
            k0 = np.maximum(kt - int(round(w / dt)), 0)
            span = (kt - k0) * dt
            d = P[:, kt] - P[:, k0]
            with np.errstate(invalid="ignore", divide="ignore"):
                speed[w].append(np.hypot(d[..., 0], d[..., 1]) / span)
        et = np.array([math.inf if s["exit_t"] is None else float(s["exit_t"]) for s in shapes])
        run_idx.append(np.full(n, r, np.int64))
        still.append(np.isinf(et))
        exit_t.append(et)
    cat = (lambda xs, dt, shape=(0,): np.concatenate(xs) if xs else np.zeros(shape, dt))
    T = len(t_grid)
    return {"run": cat(run_idx, np.int64), "still": cat(still, bool), "exit_t": cat(exit_t, float),
            "speed": {w: cat(speed[w], float, (0, T)) for w in windows}, "left": left}


# ---------------------------------------------------------------- predictions
def predictions(tab, t_grid, v_grid, w):
    """pred (T, V, R) per-run counts of inside-and-slow shapes; tp, pos (T, V) pooled shape counts (true positive:
    predicted and still inside at 10 s; predicted positives). Baseline (v = inf) as pred_b (T, R), tp_b, pos_b."""
    t_grid = np.asarray(t_grid, float)
    v_grid = np.asarray(v_grid, float)
    R = len(tab["left"])
    T, V = len(t_grid), len(v_grid)
    run, still = tab["run"], tab["still"]
    spd = tab["speed"][w]
    inside = tab["exit_t"][:, None] > t_grid[None, :]  # (S, T)
    pred = np.zeros((T, V, R), np.int64)
    tp = np.zeros((T, V), np.int64)
    pos = np.zeros((T, V), np.int64)
    # bin each shape by the first threshold it is under, then cumsum over v
    for j in range(T):
        m = inside[:, j]
        s = spd[m, j]
        rr = run[m]
        st = still[m]
        b = np.searchsorted(v_grid, s, side="right")  # first v index with s < v; V when none (or nan)
        b[np.isnan(s)] = V
        h = np.bincount(rr * (V + 1) + b, minlength=R * (V + 1)).reshape(R, V + 1)
        pred[j] = np.cumsum(h[:, :V], axis=1).T
        hs = np.bincount(b[st], minlength=V + 1)[:V]
        ha = np.bincount(b, minlength=V + 1)[:V]
        tp[j] = np.cumsum(hs)
        pos[j] = np.cumsum(ha)
    pred_b = np.zeros((T, R), np.int64)
    tp_b = np.zeros(T, np.int64)
    pos_b = np.zeros(T, np.int64)
    for j in range(T):
        m = inside[:, j]
        pred_b[j] = np.bincount(run[m], minlength=R)
        tp_b[j] = np.count_nonzero(still[m])
        pos_b[j] = np.count_nonzero(m)
    return pred, tp, pos, pred_b, tp_b, pos_b


# ---------------------------------------------------------------- statistics
def _midranks(X, w):
    """X (C, R) non-negative ints, w (R,) non-negative counts -> (C, R) mid-ranks of each X[c, r] within row c of
    the multiset where run r appears w[r] times (ties averaged, as scipy.stats.rankdata 'average')."""
    C, R = X.shape
    K = int(X.max()) + 1 if X.size else 1
    idx = X + (np.arange(C) * K)[:, None]
    h = np.bincount(idx.ravel(), weights=np.broadcast_to(w, (C, R)).ravel(), minlength=C * K)
    below = np.cumsum(h) - h  # flat cumsum is fine: subtracting the row's start offset below
    below = below.reshape(C, K) - below.reshape(C, K)[:, :1]
    mid = below + (h.reshape(C, K) + 1.0) / 2.0
    return np.take_along_axis(mid, X, axis=1)


def _wpearson(A, y, w):
    """Weighted Pearson of each row of A (C, R) with y (R,), weights w (R,). nan where either side is constant."""
    sw = w.sum()
    ma = A @ w / sw
    my = y @ w / sw
    da = A - ma[:, None]
    dy = y - my
    cov = (da * w) @ dy
    va = (da * da) @ w
    vy = (dy * dy) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vy)
    r[(va <= 1e-12 * sw) | (vy <= 1e-12 * sw)] = np.nan
    return r


def spearman(X, y, w=None):
    """Spearman rho of each row of X (C, R) with y (R,), runs weighted by integer counts w (default all 1)."""
    w = np.ones(X.shape[1]) if w is None else np.asarray(w, float)
    rx = _midranks(X, w)
    ry = _midranks(np.asarray(y)[None, :], w)[0]
    return _wpearson(rx, ry, w)


def pearson(X, y):
    return _wpearson(X.astype(float), np.asarray(y, float), np.ones(X.shape[1]))


def bootstrap_spearman(X, y, n_boot=200, seed=0, q=(5.0, 95.0)):
    """90% (default) percentile interval of Spearman rho, resampling runs with replacement n_boot times."""
    R = X.shape[1]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, X.shape[0]))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        out[b] = spearman(X, y, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan cells (constant pred) -> nan interval
        lo, hi = np.nanpercentile(out, q, axis=0)
    return lo, hi


def classification(tp, pos, n_still):
    """Precision, recall, F1 from pooled true positives, predicted positives and actual positives."""
    tp = np.asarray(tp, float)
    pos = np.asarray(pos, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(pos > 0, tp / np.maximum(pos, 1), np.nan)
        rec = tp / n_still if n_still else np.full_like(tp, np.nan)
        f1 = np.where(pos + n_still > 0, 2 * tp / np.maximum(pos + n_still, 1), np.nan)
    return prec, rec, f1


# ---------------------------------------------------------------- analysis
def analyze(runs, t_grid=T_GRID, v_grid=V_GRID, w=0.5, side_w=SIDE_W, n_boot=200, seed=0, tab=None):
    t_grid, v_grid = list(t_grid), list(v_grid)
    T, V = len(t_grid), len(v_grid)
    windows = (w,) + tuple(x for x in side_w if x != w)
    if tab is None:
        tab = shape_table(runs, t_grid, windows)
    left = tab["left"]
    R = len(left)
    n_still = int(np.count_nonzero(tab["still"]))

    pred, tp, pos, pred_b, tp_b, pos_b = predictions(tab, t_grid, v_grid, w)
    X = pred.reshape(T * V, R)
    rho = spearman(X, left).reshape(T, V)
    r = pearson(X, left).reshape(T, V)
    prec, rec, f1 = classification(tp, pos, n_still)
    # one bootstrap over grid cells and baseline rows together (same resamples)
    lo, hi = bootstrap_spearman(np.vstack([X, pred_b]), left, n_boot, seed)
    lo_g, hi_g = lo[:T * V].reshape(T, V), hi[:T * V].reshape(T, V)
    lo_b, hi_b = lo[T * V:], hi[T * V:]

    pb, rb, fb = classification(tp_b, pos_b, n_still)
    baseline = {"spearman": spearman(pred_b, left), "pearson": pearson(pred_b, left), "f1": fb,
                "precision": pb, "recall": rb, "spearman_lo": lo_b, "spearman_hi": hi_b}

    side = {}
    for sw in side_w:
        p2, tp2, pos2 = predictions(tab, t_grid, v_grid, sw)[:3]
        side[str(float(sw))] = {"spearman": spearman(p2.reshape(T * V, R), left).reshape(T, V),
                                "f1": classification(tp2, pos2, n_still)[2]}

    cells = [(i, j) for i in range(T) for j in range(V) if np.isfinite(rho[i, j])]
    cells.sort(key=lambda c: (-rho[c], -np.nan_to_num(f1[c], nan=-1.0)))
    best = [{"t": t_grid[i], "v": v_grid[j], "spearman": rho[i, j], "f1": f1[i, j], "pearson": r[i, j],
             "precision": prec[i, j], "recall": rec[i, j], "spearman_lo": lo_g[i, j], "spearman_hi": hi_g[i, j]}
            for i, j in cells[:10]]

    return {"t": t_grid, "v": v_grid, "w": w,
            "metrics": {"spearman": rho, "pearson": r, "f1": f1, "precision": prec, "recall": rec,
                        "spearman_lo": lo_g, "spearman_hi": hi_g},
            "baseline": baseline, "side_w": side, "n_runs": R, "n_shapes": int(len(tab["run"])),
            "n_still_10": n_still, "n_boot": n_boot,
            "maps": sorted({str(x.get("map")) for x in runs}) if runs else [],
            "best": best}


def to_jsonable(x):
    """numpy -> lists, nan/inf -> None, rounded to 5 decimals."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return round(x, 5) if math.isfinite(x) else None
    return x


def _fmt(x):
    return "  nan" if x is None or not np.isfinite(x) else f"{x:5.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", help="runs.jsonl from settle_collect.py")
    ap.add_argument("--out", required=True, help="grid.json to write")
    ap.add_argument("--w", type=float, default=0.5, help="speed window in seconds (default 0.5)")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map", default=None, help="only runs of this map")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    runs = load_runs(a.runs, a.map)
    if len(runs) < 3:
        sys.exit(f"need at least 3 runs, got {len(runs)}")
    t1 = time.perf_counter()
    res = analyze(runs, w=a.w, n_boot=a.boot, seed=a.seed)
    t2 = time.perf_counter()
    with open(a.out, "w") as f:
        json.dump(to_jsonable(res), f)

    print(f"{res['n_runs']} runs, {res['n_shapes']} shapes ({res['n_still_10']} still in at 10 s); "
          f"load {t1 - t0:.1f} s, analyse {t2 - t1:.1f} s -> {a.out}")
    print(f"top (w = {a.w}):    t      v   rho [90% CI]          r     F1  prec   rec")
    for b in res["best"]:
        print(f"           {b['t']:5.2f} {b['v']:6.2f} {_fmt(b['spearman'])} [{_fmt(b['spearman_lo'])},"
              f" {_fmt(b['spearman_hi'])}]  {_fmt(b['pearson'])} {_fmt(b['f1'])} {_fmt(b['precision'])}"
              f" {_fmt(b['recall'])}")
    bl = res["baseline"]
    print("baseline (inside at t, any speed): " + "  ".join(
        f"{t:g}:{_fmt(x)}" for t, x in zip(res["t"], bl["spearman"]) if t in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)))


if __name__ == "__main__":
    main()
