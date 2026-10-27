"""Fixed-tick determinism check (World.step runs whole 1 / (60 * substeps) s ticks). Run from the repo root:

    .venv/bin/python3 tests/tick_determinism_check.py [backend] [seconds] [preset ...]

backend: "cpu" (default) or "gpu". For each preset (default: penrose/evolved 1, fluid, looping):

1. headless: step(1/60) x (seconds x 60), the way evolve_layout.py scores layouts.
2. chunked: the app's variable frame times, random chunks uniform in 1/200 .. 1/25 s (seeded; the ones over max_dt
   are clamped, so fewer ticks run), then flushed with half-tick steps to run 1's tick count.
3. slow: the same chunks at time scale 0.5, then a headless run flushed to its tick count.
4. split: calls of 1, 2, 3, 5, 1, 2, ... ticks (step(k * tick length)), so CCD spans (World.span_ticks ticks,
   aligned to the tick counter) begin in one call and end in a later one, against 1's tick count.
Each pair must end with bitwise equal positions and velocities (np.array_equal). On the GPU backend the MLX solves
use float32 atomics, so it reports the largest difference instead (and run-to-run with identical chunking).
Exit status 1 if a CPU comparison differs.
"""
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from entities import EnvConfig  # noqa: E402
import scene  # noqa: E402

SIZE = (1800, 1169)


def build(name):
    return scene.PRESETS[name](*SIZE)


def flush(w, ticks):
    """Run w up to `ticks` ticks: half-tick steps add at most one tick each."""
    while w.ticks < ticks:
        w.step(w.tick_length() / 2)
    assert w.ticks == ticks, (w.ticks, ticks)


def headless(name, frames):
    w = build(name)
    for _ in range(frames):
        w.step(1 / 60)
    return w


def chunked(name, seconds, scale, seed):
    w = build(name)
    rng = random.Random(seed)
    left = seconds
    while left > 1e-12:
        dt = min(rng.uniform(1 / 200, 1 / 25), left)
        left -= dt
        w.step(dt, scale)
    return w


def split(name, ticks, sizes=(1, 2, 3, 5)):
    w = build(name)
    k = 0
    while w.ticks < ticks:
        n = min(sizes[k % len(sizes)], ticks - w.ticks)
        w.step(n * w.tick_length())
        k += 1
    flush(w, ticks)
    return w


def diff(a, b):
    """(equal, max |dP|, max |dV|)"""
    eq = np.array_equal(a.P, b.P) and np.array_equal(a.V, b.V)
    return eq, float(np.max(np.abs(a.P - b.P))), float(np.max(np.abs(a.V - b.V)))


def main(argv):
    EnvConfig.backend = argv[0] if argv else "cpu"
    seconds = float(argv[1]) if len(argv) > 1 else 10.0
    names = argv[2:] or ["penrose/evolved 1", "fluid", "looping"]
    frames = int(round(seconds * 60))
    bad = 0
    for name in names:
        a = headless(name, frames)
        b = chunked(name, seconds, 1.0, seed=1)
        clamped = b.ticks
        flush(b, a.ticks)
        rows = [("chunked vs headless", a, b, f"chunked ran {clamped} of {a.ticks} ticks before the flush")]
        c = chunked(name, seconds, 0.5, seed=2)
        d = headless(name, 0)
        flush(d, c.ticks)
        rows.append(("scale 0.5 vs headless", c, d, f"{c.ticks} ticks"))
        e = split(name, a.ticks)
        rows.append(("1/2/3/5 split vs headl.", e, a, f"{e.ticks} ticks, spans of {a.span_ticks(a.tick_length())}"))
        if EnvConfig.backend != "cpu":
            a2 = headless(name, frames)
            rows.append(("headless run-to-run", a, a2, f"{a.ticks} ticks"))
        for label, x, y, note in rows:
            eq, dp, dv = diff(x, y)
            bad += not eq and EnvConfig.backend == "cpu"
            print(f"{name:20s} {EnvConfig.backend} {label:22s} {'EQUAL' if eq else 'DIFF '} max|dP| {dp:.3g} px, "
                  f"max|dV| {dv:.3g} px/s  ({note}; {x.P.shape[1]} pts, backend {type(x.backend).__name__})",
                  flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
