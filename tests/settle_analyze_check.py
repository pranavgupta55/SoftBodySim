"""Checks for settle_analyze.py against synthetic runs with a planted best (t, v). Run from the repo root:

    .venv/bin/python3 tests/settle_analyze_check.py          # ~1 min
    .venv/bin/python3 tests/settle_analyze_check.py --full   # also a 3000-run jsonl round trip (~0.5 GB, temp)

Synthetic chunk runs (contract format, sample_dt 0.1, 10 s): every shape "settles" at a time uniform in 0.3..2.0 s,
moving ~60 px/s before. After settling, shapes that will still be inside at 10 s crawl at ~2 px/s and shapes that
leave move at ~20 px/s (they exit at a uniform time in 0.3..9.5 s). From 2.5 s on, stuck shapes start getting
jostled (~30 px/s) at a rate of 1/6 per second. So with w = 0.5 the clean window is t = 2.5 exactly, and the best
threshold sits between 2 and 20 px/s (geometric middle ~6.3).

1. Statistics: Spearman (mid-ranks, ties) and Pearson match a direct reference; weighted (bootstrap-count)
   Spearman equals Spearman of the explicitly resampled list.
2. Hand-built run: pred counts, exit_t boundary (inside means exit_t > t), window clipped at t = 0, missing samples.
3. Planted structure through the CLI: grid.json schema, best[0] at t = 2.5 with v in 3..13, peak above its
   neighbours in t, above the baseline, F1 high, CI brackets rho; side windows peak where expected.
4. Speed: analysis of 3000 runs x 160 shapes (in memory) well under a minute.
Exit status 1 on any failure.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import settle_analyze as sa  # noqa: E402

FAILS = []
T_STAR, V_LO, V_HI = 2.5, 3.0, 13.0


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


# ---------------------------------------------------------------- synthetic data
def synth_run(rng, n=None, dt=0.1, seconds=10.0):
    """One synthetic run in the contract format (see module docstring)."""
    K = int(round(seconds / dt)) + 1
    n = int(rng.integers(120, 201)) if n is None else n
    p_stuck = rng.uniform(0.05, 0.4)
    stuck = rng.random(n) < p_stuck
    t = np.arange(K) * dt
    settle = rng.uniform(0.3, 2.0, n)
    jostle = np.where(stuck, T_STAR + rng.exponential(6.0, n), np.inf)
    after = np.where(stuck, 2.0, 20.0) * rng.lognormal(0, 0.4, n)
    spd = np.where(t[None, :] < settle[:, None], 60.0 * rng.lognormal(0, 0.3, n)[:, None], after[:, None])
    spd = np.where(t[None, :] >= jostle[:, None], 30.0, spd) * rng.lognormal(0, 0.25, (n, K))
    ang = rng.uniform(0, 2 * np.pi, n)[:, None] + np.cumsum(rng.normal(0, 0.1, (n, K)), axis=1)
    step = spd * dt
    x = rng.uniform(0, 400, n)[:, None] + np.cumsum(step * np.cos(ang), axis=1) - step[:, :1] * np.cos(ang[:, :1])
    y = rng.uniform(0, 400, n)[:, None] + np.cumsum(step * np.sin(ang), axis=1) - step[:, :1] * np.sin(ang[:, :1])
    exit_t = np.where(stuck, np.nan, np.round(rng.uniform(0.3, 9.5, n), 3))
    keep = np.where(stuck, K, np.floor(np.nan_to_num(exit_t) / dt + 1e-9).astype(int) + 1)
    xy = np.round(np.stack([x, y], axis=2), 2).tolist()
    shapes = [{"r": 10.0, "area": 300.0, "exit_t": None if stuck[i] else float(exit_t[i]), "xy": xy[i][:keep[i]]}
              for i in range(n)]
    return {"map": "synthetic", "chunk": "grid_0_0", "seed": int(rng.integers(1 << 30)), "layout": "base",
            "fill": 1.0, "seconds": seconds, "sample_dt": dt, "n": n, "shapes": shapes, "left_10": int(stuck.sum())}


def synth_runs(n_runs, seed=0, n=None):
    rng = np.random.default_rng(seed)
    return [synth_run(rng, n) for _ in range(n_runs)]


# ---------------------------------------------------------------- reference stats
def ref_rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        m = a == v
        r[m] = r[m].mean()
    return r


def ref_spearman(x, y):
    return float(np.corrcoef(ref_rank(x), ref_rank(y))[0, 1])


def check_stats():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 8, (6, 50))
    y = (X[0] + rng.integers(0, 4, 50))
    rho = sa.spearman(X, y)
    ref = [ref_spearman(x, y) for x in X]
    check(np.allclose(rho, ref), "spearman matches mid-rank reference (with ties)")
    check(np.allclose(sa.pearson(X, y), [np.corrcoef(x, y)[0, 1] for x in X]), "pearson matches np.corrcoef")
    ok = True
    for _ in range(20):
        idx = rng.integers(0, 50, 50)
        w = np.bincount(idx, minlength=50).astype(float)
        got = sa.spearman(X, y, w)
        exp = [ref_spearman(x[idx], y[idx]) for x in X]
        ok &= np.allclose(got, exp)
    check(ok, "weighted spearman == spearman of the explicitly resampled runs")
    const = sa.spearman(np.zeros((1, 10), int), np.arange(10))
    check(np.isnan(const[0]), "constant prediction gives nan rho")


def check_hand():
    # one run, dt 0.1: positions are x = speed * time (straight line along x)
    def line(speed, k_end=101):
        return [[speed * 0.1 * k, 0.0] for k in range(k_end)]
    shapes = [
        {"r": 1, "area": 1, "exit_t": None, "xy": line(1.0)},           # stuck, 1 px/s
        {"r": 1, "area": 1, "exit_t": None, "xy": line(10.0)},          # stays in, 10 px/s
        {"r": 1, "area": 1, "exit_t": 1.0, "xy": line(1.0, 11)},        # slow, leaves exactly at 1.0
        {"r": 1, "area": 1, "exit_t": 1.05, "xy": line(1.0, 11)},       # slow, leaves after 1.0 (still in at 1.0)
        {"r": 1, "area": 1, "exit_t": 3.0, "xy": line(1.0, 5)},         # samples missing after 0.4 s -> nan speed
    ]
    run = {"map": "m", "chunk": "c", "seed": 0, "layout": "base", "fill": 1.0, "seconds": 10.0, "sample_dt": 0.1,
           "n": 5, "shapes": shapes, "left_10": 2}
    tab = sa.shape_table([run], [0.5, 1.0], (0.5, 1.0))
    check(np.allclose(tab["speed"][0.5][:2, 0], [1.0, 10.0]) and np.allclose(tab["speed"][1.0][:2, 0], [1.0, 10.0]),
          "speed = displacement / window, window clipped at t = 0 (w = 1.0 at t = 0.5)")
    check(np.isnan(tab["speed"][0.5][4, 0]), "missing sample gives nan speed")
    pred, tp, pos, pred_b, tp_b, pos_b = sa.predictions(tab, [0.5, 1.0], [5.0, 20.0], 0.5)
    # t = 1.0: inside = shapes 0, 1, 3, 4; slow (< 5) = 0, 3 (4 has nan speed); < 20: 0, 1, 3
    check(pred[1, :, 0].tolist() == [2, 3] and pred_b[1, 0] == 4, f"pred counts at t = 1.0 ({pred[1, :, 0].tolist()},"
          f" baseline {pred_b[1, 0]})")
    check(tp[1].tolist() == [1, 2] and pos[1].tolist() == [2, 3], "pooled shape tp / positives")


def check_planted(tmp):
    runs = synth_runs(600, seed=2)
    path = os.path.join(tmp, "runs.jsonl")
    with open(path, "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
        f.write('{"map": "synthetic", "partial')  # a half-written last line must be skipped
    out = os.path.join(tmp, "grid.json")
    t0 = time.perf_counter()
    sa.main([path, "--out", out])
    print(f"     (600 runs via CLI incl. json: {time.perf_counter() - t0:.1f} s)")
    g = json.load(open(out))
    T, V = len(g["t"]), len(g["v"])
    check(g["t"][0] == 0.5 and g["t"][-1] == 5.0 and T == 19, "t grid 0.5..5.0 step 0.25")
    check(V == 30 and g["v"][0] == 0.5 and g["v"][-1] == 40.0 and all(np.diff(g["v"]) > 0), "v grid 30 log values")
    shapes_ok = all(np.shape(g["metrics"][k]) == (T, V) for k in
                    ("spearman", "pearson", "f1", "precision", "recall", "spearman_lo", "spearman_hi"))
    shapes_ok &= all(len(g["baseline"][k]) == T for k in ("spearman", "pearson", "f1", "precision", "recall"))
    shapes_ok &= set(g["side_w"]) == {"0.25", "1.0"} and np.shape(g["side_w"]["1.0"]["spearman"]) == (T, V)
    check(shapes_ok and g["n_runs"] == 600 and g["w"] == 0.5 and len(g["best"]) == 10, "grid.json schema")

    rho = np.array(g["metrics"]["spearman"], float)
    b = g["best"][0]
    print(f"     best: t {b['t']} v {b['v']} rho {b['spearman']:.4f} f1 {b['f1']:.4f}")
    check(b["t"] == T_STAR and V_LO <= b["v"] <= V_HI, f"planted best found (t = {T_STAR}, v in {V_LO}..{V_HI})")
    i = g["t"].index(T_STAR)
    peak = np.nanmax(rho, axis=1)
    check(peak[i] > peak[i - 1] and peak[i] > peak[i + 1] and peak[i] > peak[0] + 0.05,
          f"peak over v is highest at t = {T_STAR} (t-1: {peak[i - 1]:.4f}, t: {peak[i]:.4f}, t+1: {peak[i + 1]:.4f})")
    check(b["spearman"] > g["baseline"]["spearman"][i] + 0.2, f"beats baseline at same t "
          f"({b['spearman']:.3f} vs {g['baseline']['spearman'][i]:.3f})")
    check(b["f1"] > 0.97 and b["precision"] > 0.97 and b["recall"] > 0.97, "F1 / precision / recall high at best")
    lo, hi = np.array(g["metrics"]["spearman_lo"], float), np.array(g["metrics"]["spearman_hi"], float)
    fin = np.isfinite(rho)
    check(np.mean((lo[fin] <= rho[fin] + 1e-9) & (rho[fin] <= hi[fin] + 1e-9)) > 0.95 and np.all(hi[fin] - lo[fin] < 0.5),
          "90% bootstrap interval brackets rho")
    check(not rho[0, 0] > 0.3, f"tiny v at t = 0.5 is poor or undefined (rho {rho[0, 0]:.3f})")
    check(np.nanmax(rho[0]) < peak[i] - 0.05, f"t = 0.5 well below the peak (max rho {np.nanmax(rho[0]):.3f})")
    s25 = np.array(g["side_w"]["0.25"]["spearman"], float)
    s10 = np.array(g["side_w"]["1.0"]["spearman"], float)
    t25 = g["t"][int(np.nanargmax(np.nanmax(s25, axis=1)))]
    t10 = g["t"][int(np.nanargmax(np.nanmax(s10, axis=1)))]
    check(t25 in (2.25, 2.5) and t10 in (2.75, 3.0, 3.25), f"side windows peak where expected (w 0.25 at t {t25}, "
          f"w 1.0 at t {t10})")


def check_speed(full, tmp):
    t0 = time.perf_counter()
    runs = synth_runs(3000, seed=3, n=160)
    t1 = time.perf_counter()
    res = sa.analyze(runs)
    t2 = time.perf_counter()
    print(f"     (generate {t1 - t0:.1f} s, analyse {t2 - t1:.1f} s, best {res['best'][0]['t']}, "
          f"{res['best'][0]['v']})")
    check(t2 - t1 < 30, f"3000 runs x 160 shapes analysed in {t2 - t1:.1f} s (< 30)")
    if full:
        path = os.path.join(tmp, "big.jsonl")
        with open(path, "w") as f:
            for r in runs:
                f.write(json.dumps(r) + "\n")
        del runs
        print(f"     (big.jsonl {os.path.getsize(path) / 1e6:.0f} MB)")
        t0 = time.perf_counter()
        sa.main([path, "--out", os.path.join(tmp, "big.json")])
        dt = time.perf_counter() - t0
        check(dt < 60, f"3000-run jsonl round trip via CLI in {dt:.1f} s (< 60)")


def main():
    full = "--full" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        check_stats()
        check_hand()
        check_planted(tmp)
        check_speed(full, tmp)
    print(f"\n{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
