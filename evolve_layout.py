"""Evolve a diamonds (or penrose) layout unattended and save the winner as a new world in its folder.

    python3 evolve_layout.py NAME [--tiling diamonds|penrose] [--base random|LAYOUT] [--rounds 2] [--children 100]
                             [--keep 5] [--seconds 60] [--deadline "YYYY-MM-DD HH:MM"] ...

Round 1 runs `children` variants of the base layout (each drops `remove` random tiles and adds `add` new ones,
diamonds_map.pick_diamonds), later rounds give each of the previous round's top `keep` an equal share of `children`
variants. Every run is the physics only (no display, no drawing) for `seconds` of sim time, scored over its last 10 s:

    score = w1 x shape spread (0.7 x-entropy + 0.3 y-entropy of the shapes' centers, 1 = perfectly even)
          + w2 x unstuck (per cell of a 4x4 grid of the world: each shape's average speed over the past 3 s; in each
                 cell holding at least 3 shapes' centers, the mean of its slowest 20% (at least 1) over the global
                 median shape speed, capped at 1; the mean of the worst 4 such cells (all of them if fewer qualify),
                 averaged over the scored samples: 1 = nothing lags anywhere, ~0 = a pocket of stuck shapes)
          + w3 x tile evenness (1 / (1 + std/mean) of the obstacle tile counts in a 4x4 grid of the world)

with --weights w1,w2,w3 (default 0.7,0.2,0.1).

Then the last round's top `finalists` and the base (not with --no-final-base) run `final_seconds` each, on
--final-seeds fresh spawn seeds (ranked by their mean), and the best is written to
maps/<tiling>/NAME.json (with the spawn settings it was evolved under, and for penrose the tiling's settings: pentagrid
seed, edge, rotation, min slope, cup rule ("cups") and, for the "pair" rule, its gap, see scene._layout_tiling),
and the base's archetype "clusters" settings, if any: the variants' added tiles follow them too), so it shows up as "<tiling>/NAME" in the Worlds panel. Progress goes to stdout (one line per round) and every run to
LOG_DIR/runs.jsonl.

Four search improvements, each on by default (--no-targeted --no-halving --no-paired --no-crossover together give the
original search, run for run):
  --targeted: each run records where shapes were stuck (3-s speed under STUCK_SHARE x the median, once a second over
      the scored window) and its per-cell unstuck; a child of a parent with run data drops tiles weighted by the stuck
      shapes within STUCK_REACH tile edges and adds tiles weighted toward sparse, stuck-free spots (cells without
      stuck shapes, none within STUCK_REACH), each draw targeted with probability --targeted-share, uniform
      otherwise. pick_diamonds' rules still veto illegal adds. The base gets one full run first (round 0) for its
      data.
  --halving STAGES ("auto": seconds/6, seconds/2, seconds; or e.g. "10,30,60", the last one = --seconds): every child
      runs the first stage, the best half (at least `keep`) go on to the next one (rerun from scratch); only the last
      stage's scores rank them. --stages "SECONDS:SURVIVORS,...,SECONDS" (e.g. "10:50,30:25,60") instead sets each
      stage's survivors explicitly (a stage without ":N" halves as above) and the last stage's seconds = --seconds.
  --paired: all runs of a round (its children and their parents) share one fresh shape-spawn seed, so score
      differences come from the layout; the kept parents are re-run each round on it and rank by the mean of all
      their scores (parents and children compete for `keep`). The final phase uses one more fresh seed for all.
  --crossover SHARE: that share of a round's children (with 2+ parents) take each 4x4 cell's tiles from whichever of
      two random parents had the better unstuck in that cell (with probability CROSS_BETTER; either didn't score it:
      a coin flip); pick_diamonds then re-checks by the rules the tiles within the rules' reach of a cell border,
      after the others (each parent's own, trusted), so a border conflict drops the border tile.

--tiling diamonds (default) picks from the tumbling-blocks lattice; penrose from a Penrose P3 rhombus tiling
(diamonds_map.penrose_tiles, pentagrid offsets seeded by --pentagrid-seed). --base random starts from a seeded random
layout (--seed) at the current spawn settings' density (scene.DIAMONDS_SPAWN / PENROSE_SPAWN, or --density);
--base LAYOUT (a name in maps/<tiling>) starts from that layout, its tiling and its own spawn settings.

--rounds is a maximum with --deadline (local time): before each round after the first, the run checks that one more
round (the longest one so far, or the stage plan's estimate if longer) plus the final phase (ceil((finalists [+ 1
for the base]) x final_seeds / workers) waves of final runs, timed from the last round's runs per sim second) and a
margin still end before the deadline, and goes straight to the final phase if not, so the map is saved in time.

Runs use `workers` processes (each imports and compiles once); launch it at low priority, e.g. `nice -n 10`, with
OMP_NUM_THREADS=1.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
EVERY, BX, BY = 15, 10, 6  # sample every 15 frames; 10 x / 6 y histogram bins
SCORE_WINDOW = 600  # frames (10 s) at the end of each run that are scored
SPEED_WINDOW = 180  # frames (3 s): each shape's speed is averaged over this rolling window
SLOW_SHARE = 0.2  # the slowest share of shapes (per cell) the unstuck score looks at
CELLS = 4  # unstuck cells: a CELLS x CELLS grid of the world
CELL_MIN_SHAPES = 3  # a cell needs this many shapes' centers to be scored
WORST_CELLS = 4  # unstuck is the mean of this many worst cells
STUCK_SHARE = 0.25  # targeted: a shape is stuck when its 3-s speed is under this share of the median
STUCK_EVERY = 60  # frames: stuck shapes are recorded once a second over the scored window
CROSS_BETTER = 0.8  # crossover: the chance a cell's tiles come from the parent that scored better there
STUCK_REACH = 2.5  # tile edges: tiles this close to stuck shapes are removal targets; additions keep away
DEADLINE_MARGIN = 300.0  # s kept spare before --deadline (plus 15% on the final phase's estimate)
_SETUP = {}  # per worker: the world size, spawn settings, weights and tiling (init)


def init(size, spawn, weights, tiling=None):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from entities import EnvConfig
    EnvConfig.backend = "cpu"
    _SETUP.update(size=size, spawn=spawn, weights=weights, tiling=tiling or {"tiling": "diamonds"})


def entropy(counts):
    import numpy as np
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def cell_values(pos, avg, W, H):
    """{cell (iy * CELLS + ix): its unstuck} of one sample, for the cells that qualify (see cell_unstuck)."""
    import numpy as np
    med = max(float(np.median(avg)), 1e-9)
    ix = np.clip((pos[:, 0] % W / W * CELLS).astype(int), 0, CELLS - 1)
    iy = np.clip((pos[:, 1] % H / H * CELLS).astype(int), 0, CELLS - 1)
    cell = iy * CELLS + ix
    vals = {}
    for k in range(CELLS * CELLS):
        v = np.sort(avg[cell == k])
        if len(v) >= CELL_MIN_SHAPES:
            vals[k] = min(1.0, float(v[:max(1, int(round(SLOW_SHARE * len(v))))].mean()) / med)
    return vals


def cell_unstuck(pos, avg, W, H, vals=None):
    """The unstuck score of one sample: shape centers `pos` (n, 2, any wrap) and their 3-s average speeds `avg` (n):
    per cell of a CELLS x CELLS grid with at least CELL_MIN_SHAPES shapes, the mean of its slowest SLOW_SHARE (at
    least 1) over the global median speed, capped at 1; then the mean of the WORST_CELLS lowest (all if fewer).
    None if no cell qualifies. `vals`: cell_values' result, if already at hand."""
    import numpy as np
    vals = list((cell_values(pos, avg, W, H) if vals is None else vals).values())
    if not vals:
        return None
    return float(np.mean(sorted(vals)[:WORST_CELLS]))


def run(job):
    """job: {"tag", "keys" (the layout), "frames", optional: "add_seed" (None: the layout itself, else a variant of it
    with "remove" / "add" tiles changed, steered by "remove_order" / "add_order" if given), "check" (tile keys of the
    layout to re-pick by the rules, see pick_diamonds), "seed" (the shapes' spawn seed; None: the scene's own)} -> result dict."""
    import numpy as np
    import scene
    from entities import SoftBody
    from diamonds_map import tile_key
    tag, keys, frames, add_seed = job["tag"], job["keys"], job["frames"], job.get("add_seed")
    remove, add = job.get("remove", 0), job.get("add", 0)
    W, H = _SETUP["size"]
    t = time.time()
    vary = add_seed is not None
    tiling = _SETUP["tiling"]
    w = scene.build_diamonds_scene(W, H, layout=(keys, (W, H)), remove=remove if vary else 0, add=add if vary else 0,
                                   add_seed=add_seed or 0, spawn=_SETUP["spawn"], tiling=tiling,
                                   shape_seed=job.get("seed"), remove_order=job.get("remove_order") if vary else None,
                                   add_order=job.get("add_order") if vary else None, check=job.get("check", False))
    dia = [b.points for b in w.bodies if getattr(b, "palette_role", "") == "diamond"]
    c = np.array([p.mean(axis=1) for p in dia])
    counts = np.histogram2d(c[:, 0], c[:, 1], (4, 4), ((0, W), (0, H)))[0].ravel()
    even = 1.0 / (1.0 + counts.std() / counts.mean())
    shapes = [b for b in w.bodies if isinstance(b, SoftBody)]
    hx, hy = np.zeros(BX), np.zeros(BY)
    track_from = frames - SCORE_WINDOW - SPEED_WINDOW  # speeds are kept from 3 s before the scored window
    speeds = np.zeros((SPEED_WINDOW, len(shapes)))  # rolling: the last 3 s of each shape's speed (px/s)
    prev = None
    unstuck, slow_speed, stuck_xy = [], [], []
    cell_sum, cell_n = np.zeros(CELLS * CELLS), np.zeros(CELLS * CELLS)
    n_slow = max(1, int(round(SLOW_SHARE * len(shapes))))
    for f in range(frames):
        w.step(1 / 60)
        if f < track_from:
            continue
        c = np.array([b.points.mean(axis=1) for b in shapes])
        if prev is not None:
            d = c - prev
            d -= np.array([[W, H]]) * np.round(d / np.array([[W, H]]))  # a wrap isn't a jump across the world
            speeds[f % SPEED_WINDOW] = np.hypot(d[:, 0], d[:, 1]) * 60.0
        prev = c
        if f >= frames - SCORE_WINDOW and f % EVERY == 0:
            hx += np.histogram(c[:, 0] % W, BX, (0, W))[0]
            hy += np.histogram(c[:, 1] % H, BY, (0, H))[0]
            avg = speeds.mean(axis=0)  # each shape's average speed over the past 3 s
            slow_speed.append(np.sort(avg)[:n_slow].mean())
            vals = cell_values(c, avg, W, H)
            for k, v in vals.items():
                cell_sum[k] += v
                cell_n[k] += 1
            u = cell_unstuck(c, avg, W, H, vals)
            if u is not None:
                unstuck.append(u)
            if f % STUCK_EVERY == 0:
                slow = avg < STUCK_SHARE * np.median(avg)
                stuck_xy += [[round(float(x % W), 1), round(float(y % H), 1)] for x, y in c[slow]]
    spread = 0.7 * entropy(hx) + 0.3 * entropy(hy)
    stuck = float(np.mean(unstuck)) if unstuck else 0.0
    w1, w2, w3 = _SETUP["weights"]
    return {"tag": tag, "score": float(w1 * spread + w2 * stuck + w3 * even), "spread": spread, "unstuck": stuck,
            "slow_speed": float(np.mean(slow_speed)), "even": float(even),
            "diamonds": len(dia), "keys": [tile_key(p) for p in dia], "secs": round(time.time() - t, 1),
            "sim_seconds": frames / 60, "seed": job.get("seed"), "stuck_xy": stuck_xy,
            "cells": [round(float(a / n), 4) if n else None for a, n in zip(cell_sum, cell_n)]}


def tiling_dir(tiling):
    sys.path.insert(0, REPO)
    import scene
    return scene.PENROSE_DIR if tiling == "penrose" else scene.DIAMONDS_DIR


def base_layout(args, size):
    """(keys, spawn settings, tiling info, description) of the starting layout."""
    sys.path.insert(0, REPO)
    import scene
    from diamonds_map import layout_tiling, load_layout, tile_key
    if args.base == "random":
        tiling = scene._layout_tiling(None, args.tiling, args.pentagrid_seed)
        spawn = scene.diamonds_spawn(spawn={"density": args.density} if args.density else None,
                                     tiling=tiling["tiling"])
        w = scene.build_diamonds_scene(*size, seed=args.seed, layout=None, spawn=spawn, tiling=tiling)
        keys = [tuple(tile_key(b.points)) for b in w.bodies if getattr(b, "palette_role", "") == "diamond"]
        return keys, spawn, tiling, f"random base (seed {args.seed}, {len(keys)} tiles)"
    path = os.path.join(tiling_dir(args.tiling), args.base + ".json")
    keys, _ = load_layout(path)
    return keys, scene.diamonds_spawn(path), scene._layout_tiling(path), f"base {args.base}"


def all_tiles(size, tiling):
    """(tile keys = centers (n, 2) of every tile of the tiling lying in the world, tile edge length), as
    scene.build_diamonds_scene builds them."""
    import numpy as np
    sys.path.insert(0, REPO)
    import scene
    from diamonds_map import lattice_diamonds, penrose_tiles, steep_tiles, tile_key
    W, H = size
    if tiling["tiling"] == "penrose":
        side = float(tiling["edge"])
        tiles = steep_tiles(penrose_tiles(W, H, side, seed=int(tiling["pentagrid_seed"]),
                                          rotation=float(tiling["rotation"])), float(tiling["min_slope"]))
    else:
        side = scene.DIAMOND_SIDE
        tiles = lattice_diamonds(W, H, side, seed=scene.DIAMONDS_SEED, y0=side)
    eps = 1e-6
    keys = [tile_key(t[0]) for t in tiles if t[0][0].min() >= -eps and t[0][0].max() <= W + eps
            and t[0][1].min() >= -eps and t[0][1].max() <= H + eps]
    return np.array(keys, dtype=float), side


def _near_counts(centers, points, reach, size):
    """How many of `points` (m, 2) lie within `reach` of each of `centers` (n, 2), across the wraps."""
    import numpy as np
    if len(points) == 0 or len(centers) == 0:
        return np.zeros(len(centers))
    wh = np.array(size)
    d = centers[:, None, :] - np.asarray(points, dtype=float)[None, :, :]
    d -= wh * np.round(d / wh)
    return (np.hypot(d[..., 0], d[..., 1]) <= reach).sum(axis=1).astype(float)


def _weighted_order(rng, weights, share):
    """A random permutation of range(len(weights)) where each draw is weighted by `weights` with probability
    `share` and uniform otherwise (Efraimidis-Spirakis keys over the mixture)."""
    import numpy as np
    n = len(weights)
    w = np.asarray(weights, dtype=float)
    p = np.full(n, 1.0 / n)
    if w.sum() > 0:
        p = share * w / w.sum() + (1 - share) * p
    keys = np.log(rng.random(n)) / np.maximum(p, 1e-12)
    return np.argsort(-keys)


def targeted_orders(rng, keys, res, tiles, side, size, share):
    """(remove_order, add_order) tile keys for a child of the layout `keys` from its run result `res`: removal weighted
    by the stuck shapes within STUCK_REACH edges of each tile; additions by sparseness (1 / (1 + the layout's tiles
    within 2 x STUCK_REACH edges)), zero within STUCK_REACH of a stuck shape and x0.25 in a cell that has any."""
    import numpy as np
    W, H = size
    reach = STUCK_REACH * side
    own = np.array(keys, dtype=float).reshape(-1, 2)
    stuck = np.array(res.get("stuck_xy") or [], dtype=float).reshape(-1, 2)
    rem_w = _near_counts(own, stuck, reach, size)
    remove_order = own[_weighted_order(rng, rem_w, share)].tolist()
    in_layout = {(round(x, 1), round(y, 1)) for x, y in own.tolist()}
    cand = np.array([t for t in tiles.tolist() if (round(t[0], 1), round(t[1], 1)) not in in_layout]).reshape(-1, 2)
    sparse = 1.0 / (1.0 + _near_counts(cand, own, 2 * reach, size))
    calm = (_near_counts(cand, stuck, reach, size) == 0).astype(float)
    cell = lambda p: (np.clip((p[:, 1] / H * CELLS).astype(int), 0, CELLS - 1) * CELLS  # noqa: E731
                      + np.clip((p[:, 0] / W * CELLS).astype(int), 0, CELLS - 1))
    dirty = set(cell(stuck).tolist()) if len(stuck) else set()
    clean = np.array([0.25 if k in dirty else 1.0 for k in cell(cand).tolist()])
    add_order = cand[_weighted_order(rng, sparse * calm * clean, share)].tolist()
    return remove_order, add_order


def crossover_keys(rng, a, b, size, reach):
    """(layout, keys to re-check): each CELLS x CELLS cell's tiles from whichever of parents a / b ({"keys", "res"})
    had the better unstuck in that cell (with probability CROSS_BETTER, else the other; a coin flip when either didn't
    score it), ordered so the tiles farthest inside their cells come first. The tiles within `reach` (the rules'
    reach) of a tile the other parent doesn't have are re-checked by pick_diamonds' rules; the others (whose
    neighbourhood one parent had as a whole) are trusted, so a border conflict drops the later tile."""
    import numpy as np
    W, H = size
    ca, cb = a["res"]["cells"], b["res"]["cells"]
    pick = []
    for k in range(CELLS * CELLS):
        if ca[k] is None or cb[k] is None or ca[k] == cb[k]:
            pick.append(bool(rng.random() < 0.5))
        else:
            pick.append((ca[k] > cb[k]) == bool(rng.random() < CROSS_BETTER))
    out = []
    for take_a, parent in ((True, a), (False, b)):
        for x, y in parent["keys"]:
            fx, fy = x / W * CELLS, y / H * CELLS
            ix, iy = min(CELLS - 1, max(0, int(fx))), min(CELLS - 1, max(0, int(fy)))
            if pick[iy * CELLS + ix] == take_a:
                inner = min(fx - ix, ix + 1 - fx) * W / CELLS, min(fy - iy, iy + 1 - fy) * H / CELLS
                out.append((min(inner), [x, y], take_a))
    out.sort(key=lambda t: -t[0])
    has = {True: {tuple(k) for k in a["keys"]}, False: {tuple(k) for k in b["keys"]}}
    pts = np.array([k for _, k, _ in out], dtype=float).reshape(-1, 2)
    # foreign: a union tile its neighbours' parent lacks (from the other parent and not shared)
    foreign = {True: np.array([tuple(k) not in has[True] for _, k, _ in out], dtype=bool),
               False: np.array([tuple(k) not in has[False] for _, k, _ in out], dtype=bool)}
    near = {side: _near_counts(pts, pts[foreign[side]], reach, size) > 0 for side in (True, False)}
    check = [k for i, (_, k, side) in enumerate(out) if near[side][i] or foreign[side][i]]
    return [k for _, k, _ in out], check


def parse_deadline(text):
    """'YYYY-MM-DD HH:MM' local time -> epoch seconds (None for None)."""
    return None if text is None else time.mktime(time.strptime(text, "%Y-%m-%d %H:%M"))


def survivors(n, si, args):
    """How many of the n children run in stage si go on to stage si + 1: --stages' explicit count (at most n), else
    the best half, at least `keep`."""
    k = args.survivors[si]
    return min(n, k) if k is not None else max(math.ceil(n / 2), min(n, args.keep))


def stage_plan(args):
    """[(sim seconds, runs)] of one round: the halving stages (children only; the parents join the last one when
    paired)."""
    n, plan = args.children, []
    for i, secs in enumerate(args.stages):
        last = i == len(args.stages) - 1
        plan.append((secs, n + (args.keep if last and args.paired else 0)))
        if not last:
            n = survivors(n, i, args)
    return plan


def final_runs(args):
    """The final phase's runs: (finalists [+ the base]) x final_seeds."""
    return (args.finalists + (1 if args.final_base else 0)) * args.final_seeds


def fits_deadline(deadline, now, round_secs, run_secs_per_sim, args):
    """(fits, estimate in s) of one more round (the longest so far, or the stage plan's waves at the measured wall
    seconds per sim second if longer) plus the final phase: ceil(final_runs / workers) waves of final_seconds runs at
    that rate, +15%, + DEADLINE_MARGIN."""
    waves = math.ceil(final_runs(args) / args.workers)
    final = waves * args.final_seconds * run_secs_per_sim * 1.15
    planned = sum(math.ceil(n / args.workers) * secs * run_secs_per_sim for secs, n in stage_plan(args))
    need = max(max(round_secs), planned) + final + DEADLINE_MARGIN
    return now + need <= deadline, need


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name", help="the new layout's name (maps/<tiling>/NAME.json, world <tiling>/NAME)")
    p.add_argument("--tiling", choices=("diamonds", "penrose"), default="diamonds",
                   help="the tiling the obstacles are picked from (and the maps folder)")
    p.add_argument("--pentagrid-seed", type=int, default=None,
                   help="penrose random base: the pentagrid offsets' seed (default scene.PENROSE_SEED)")
    p.add_argument("--base", default="random", help="random, or a layout name in maps/<tiling>")
    p.add_argument("--seed", type=int, default=int(time.time()) % 100000,
                   help="random base's seed, and the search's own (round seeds, targeting, crossover; default: clock)")
    p.add_argument("--rounds", type=int, default=2, help="rounds (a maximum with --deadline)")
    p.add_argument("--children", type=int, default=100, help="variants per round")
    p.add_argument("--keep", type=int, default=5, help="parents kept per round")
    p.add_argument("--remove", type=int, default=3)
    p.add_argument("--add", type=int, default=3)
    p.add_argument("--seconds", type=float, default=60, help="sim seconds per run")
    p.add_argument("--finalists", type=int, default=8, help="the last round's top N re-run in the final phase")
    p.add_argument("--final-seconds", type=float, default=180)
    p.add_argument("--final-seeds", type=int, default=1,
                   help="fresh spawn seeds each finalist runs on in the final phase (ranked by the mean)")
    p.add_argument("--final-base", action=argparse.BooleanOptionalAction, default=True,
                   help="also re-run the base in the final phase (for the comparison in the note); "
                        "--no-final-base saves its runs, the saved map is the best finalist either way")
    p.add_argument("--size", default="1800x1169", help="world size WxH (the screen's: window px / SCALE_DOWN / zoom)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--weights", default="0.7,0.2,0.1", help="w1,w2,w3: spread, unstuck, tile evenness")
    p.add_argument("--density", type=float, default=None,
                   help="random base's tiles per square world unit (default: scene.DIAMONDS_SPAWN's / PENROSE_SPAWN's)")
    p.add_argument("--deadline", default=None,
                   help='"YYYY-MM-DD HH:MM" local time: stop starting rounds so the map is saved before it')
    p.add_argument("--log-dir", default=None, help="default: <tmp>/evolve_layout/NAME")
    p.add_argument("--targeted", action=argparse.BooleanOptionalAction, default=True,
                   help="steer removals to tiles near stuck shapes, additions to sparse stuck-free spots")
    p.add_argument("--targeted-share", type=float, default=0.5, help="share of targeted (vs uniform) draws")
    p.add_argument("--halving", default="auto",
                   help='successive halving stages in sim s, e.g. "10,30,60" (the last = --seconds); "auto": '
                        'seconds/6, seconds/2, seconds')
    p.add_argument("--no-halving", dest="halving", action="store_const", const="off", help="one stage: --seconds")
    p.add_argument("--stages", default=None,
                   help='explicit stage plan "SECONDS:SURVIVORS,...,SECONDS", e.g. "10:50,30:25,60" (overrides '
                        '--halving; a stage without ":N" keeps the best half, at least keep; the last stage\'s '
                        'seconds = --seconds)')
    p.add_argument("--paired", action=argparse.BooleanOptionalAction, default=True,
                   help="one shape seed per round for children and parents; parents re-scored and averaged")
    p.add_argument("--crossover", type=float, default=0.25, help="share of children made by cell crossover")
    p.add_argument("--no-crossover", dest="crossover", action="store_const", const=0.0)
    args = p.parse_args(argv)
    args.survivors = None
    if args.stages is not None:
        parts = [v.strip().split(":") for v in args.stages.split(",")]
        if any(len(v) > 2 for v in parts) or len(parts[-1]) != 1:
            p.error('--stages: "SECONDS:SURVIVORS,...,SECONDS" (the last stage has no survivor count)')
        args.stages = [float(v[0]) for v in parts]
        args.survivors = [int(v[1]) if len(v) == 2 else None for v in parts[:-1]]
        if any(k is not None and k < 1 for k in args.survivors):
            p.error("--stages: survivor counts must be at least 1")
        args.seconds = args.stages[-1]
    elif args.halving == "off":
        args.stages = [args.seconds]
    elif args.halving == "auto":
        args.stages = [args.seconds / 6, args.seconds / 2, args.seconds]
    else:
        args.stages = [float(v) for v in args.halving.split(",")]
        if abs(args.stages[-1] - args.seconds) > 1e-9:
            p.error(f"--halving's last stage ({args.stages[-1]:g}) must be --seconds ({args.seconds:g})")
    if args.survivors is None:
        args.survivors = [None] * (len(args.stages) - 1)
    if args.final_seeds < 1:
        p.error("--final-seeds must be at least 1")
    import numpy as np
    size = tuple(float(v) for v in args.size.lower().split("x"))
    weights = tuple(float(v) for v in args.weights.split(","))
    deadline = parse_deadline(args.deadline)
    out_dir = tiling_dir(args.tiling)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, args.name + ".json")
    log_dir = args.log_dir or os.path.join(tempfile.gettempdir(), "evolve_layout", args.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "runs.jsonl")
    open(log_path, "w").close()

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    base, spawn, tiling, base_desc = base_layout(args, size)
    frames, final_frames = int(args.seconds * 60), int(args.final_seconds * 60)
    rng = np.random.default_rng([args.seed, 11])  # the search's own draws (round seeds, targeting, crossover)
    tiles, side = all_tiles(size, tiling)
    # pick_diamonds' reach: tiles whose centers are farther apart can't cup or link (cup gap / near-miss distance)
    import scene
    rule_reach = 2 * side + max(float(tiling.get("cup_gap", scene.CUP_GAP)), float(tiling.get("near", 0.0)))
    features = (f"targeted {args.targeted_share:g}" if args.targeted else "untargeted") + \
        (f", stages {' / '.join(f'{s:g} s' + (f' -> {k}' if k is not None else '') for s, k in zip(args.stages, args.survivors + [None]))}"
         if len(args.stages) > 1 else "") + \
        (", paired seeds + elite re-scoring" if args.paired else "") + \
        (f", crossover {args.crossover:g}" if args.crossover > 0 else "")
    print(f"{args.name}: {tiling}, {base_desc}, spawn {spawn}, up to {args.rounds} rounds x {args.children} runs of "
          f"{args.seconds:g} s ({features}){', deadline ' + args.deadline if deadline else ''}, log {log_path}",
          flush=True)
    t0 = time.time()
    # parents: {"tag", "keys", "scores" (one per scored run on a different seed), "res" (its latest run), "fresh"}
    parents, ranking, round_secs, rounds_done = [{"tag": "base", "keys": base, "scores": [], "res": None}], [], [], 0
    run_secs_per_sim = None

    def mean(e):
        return sum(e["scores"]) / len(e["scores"])

    def run_all(pool, jobs, r, stage=None):
        out = []
        for res in pool.imap_unordered(run, jobs):
            res["round"] = r
            if stage is not None:
                res["stage"] = stage
            log(res)
            out.append(res)
        return out

    with mp.Pool(args.workers, initializer=init, initargs=(size, spawn, weights, tiling)) as pool:
        for r in range(1, args.rounds + 1):
            if deadline is not None and round_secs:
                ok, need = fits_deadline(deadline, time.time(), round_secs, run_secs_per_sim, args)
                if not ok:
                    print(f"deadline: another round + final need ~{need:.0f} s, {deadline - time.time():.0f} s left: "
                          f"going to the final after {rounds_done} rounds", flush=True)
                    break
            t_round = time.time()
            seed = int(rng.integers(1, 1 << 30)) if args.paired else None
            for e in parents:
                e["fresh"] = False
            if args.targeted and any(e["res"] is None for e in parents):  # parents without run data: run them first
                todo = [e for e in parents if e["res"] is None]
                for res in run_all(pool, [{"tag": e["tag"], "keys": e["keys"], "frames": frames, "seed": seed}
                                          for e in todo], r - 1):
                    e = next(e for e in todo if e["tag"] == res["tag"])
                    e["res"], e["fresh"] = res, True
                    e["scores"].append(res["score"])
            n_cross = int(round(args.crossover * args.children)) if len(parents) >= 2 else 0
            per = max(1, (args.children - n_cross) // len(parents))
            jobs = []
            for i, e in enumerate(parents):
                for k in range(per):
                    job = {"tag": f"r{r}.{i}.{k}", "keys": e["keys"], "add_seed": r * 100000 + i * 1000 + k,
                           "remove": args.remove, "add": args.add}
                    if args.targeted and e["res"] is not None:
                        job["remove_order"], job["add_order"] = targeted_orders(
                            rng, e["keys"], e["res"], tiles, side, size, args.targeted_share)
                    jobs.append(job)
            for j in range(n_cross):
                a, b = (parents[int(x)] for x in rng.choice(len(parents), 2, replace=False))
                keys, check = crossover_keys(rng, a, b, size, rule_reach)
                jobs.append({"tag": f"r{r}.x{j}", "keys": keys, "check": check, "parents": [a["tag"], b["tag"]]})
            sim_secs, wall_secs, stage_note, n_runs = 0.0, 0.0, [], 0
            for si, secs in enumerate(args.stages):
                last = si == len(args.stages) - 1
                todo = [dict(j, frames=int(round(secs * 60)), seed=seed) for j in jobs]
                if last and args.paired:  # the parents re-run on this round's seed (unless just run on it)
                    todo += [{"tag": e["tag"], "keys": e["keys"], "frames": frames, "seed": seed, "parent": True}
                             for e in parents if not e["fresh"]]
                results = run_all(pool, todo, r, secs if len(args.stages) > 1 else None)
                sim_secs += sum(x["sim_seconds"] for x in results)
                wall_secs += sum(x["secs"] for x in results)
                n_runs += len(results)
                kids = sorted((x for x in results if x["tag"] not in {e["tag"] for e in parents}),
                              key=lambda x: -x["score"])
                if not last:
                    n = survivors(len(kids), si, args)
                    stage_note.append(f"{secs:g} s: {len(kids)} -> {n}, best {kids[0]['score']:.4f}")
                    jobs = [{"tag": x["tag"], "keys": x["keys"]} for x in kids[:n]]  # survivors: their own layouts
            for res in results:  # the parents' re-runs
                e = next((e for e in parents if e["tag"] == res["tag"]), None)
                if e is not None:
                    e["res"] = res
                    e["scores"].append(res["score"])
            round_secs.append(time.time() - t_round)
            run_secs_per_sim = wall_secs / sim_secs
            rounds_done = r
            pool_entries = [{"tag": x["tag"], "keys": x["keys"], "scores": [x["score"]], "res": x} for x in kids]
            if args.paired:
                pool_entries += parents
            ranking = sorted(pool_entries, key=lambda e: -mean(e))
            top = ranking[:args.keep]
            best = ranking[0]["res"]
            elite = ""
            if args.paired:
                elite = ", elite " + " ".join(f"{e['tag']}={mean(e):.4f}/{len(e['scores'])}" for e in top)
            print(f"round {r}: {n_runs} runs, best {mean(ranking[0]):.4f} {ranking[0]['tag']} (spread "
                  f"{best['spread']:.4f}, unstuck {best['unstuck']:.3f}, even {best['even']:.3f}), top {args.keep} "
                  f"mean {sum(mean(e) for e in top) / len(top):.4f}, median "
                  f"{mean(ranking[len(ranking) // 2]):.4f}{elite}"
                  f"{' [' + '; '.join(stage_note) + ']' if stage_note else ''}, {time.time() - t0:.0f} s", flush=True)
            parents = [{"tag": e["tag"], "keys": e["keys"], "scores": e["scores"], "res": e["res"]} for e in top]
        finalists = [(e["tag"], e["keys"]) for e in ranking[:args.finalists] if e["tag"] != "base"]
        if args.final_base or not finalists:  # (no round ran: the base is the only candidate)
            finalists.insert(0, ("base", base))
        jobs = []
        for _ in range(args.final_seeds):
            seed = int(rng.integers(1, 1 << 30)) if args.paired else None
            jobs += [{"tag": tag, "keys": keys, "frames": final_frames, "seed": seed} for tag, keys in finalists]
        runs = run_all(pool, jobs, "final")
    final = []  # one entry per finalist: its score fields averaged over the final seeds
    for tag, _ in finalists:
        mine = [x for x in runs if x["tag"] == tag]
        x = dict(mine[0])
        for k in ("score", "spread", "unstuck", "slow_speed", "even"):
            x[k] = float(sum(m[k] for m in mine) / len(mine))
        x["scores"] = [m["score"] for m in mine]
        x["seeds"] = [m["seed"] for m in mine]
        final.append(x)
    final.sort(key=lambda x: -x["score"])
    for x in final:
        each = f" = mean of {' '.join(f'{v:.4f}' for v in x['scores'])}" if len(x["scores"]) > 1 else ""
        print(f"final {x['tag']}: {x['score']:.4f}{each} (spread {x['spread']:.4f}, unstuck {x['unstuck']:.3f}, "
              f"even {x['even']:.3f})", flush=True)
    best = final[0]
    base_score = next((x["score"] for x in final if x["tag"] == "base"), None)
    vs_base = f" vs base {base_score:.4f}" if base_score is not None else ""
    sys.path.insert(0, REPO)
    from diamonds_map import save_layout
    stopped = f" (deadline {args.deadline}: stopped after {rounds_done})" if rounds_done < args.rounds else ""
    note = (f"{args.name} ({time.strftime('%Y-%m-%d')}): {tiling['tiling']} tiling, {base_desc}; {rounds_done} rounds"
            f"{stopped} of {args.children} remove-{args.remove}/add-{args.add} variants run {args.seconds:g} s (top "
            f"{args.keep} kept per round; {features}), score {weights[0]:g} spread + {weights[1]:g} unstuck (worst "
            f"{WORST_CELLS} of {CELLS}x{CELLS} cells) + {weights[2]:g} tile evenness; last round's top "
            f"{args.finalists}{' + base' if base_score is not None else ''} re-run {args.final_seconds:g} s"
            f"{f' on {args.final_seeds} seeds' if args.final_seeds > 1 else ''}: {best['tag']} {best['score']:.4f}"
            f"{vs_base}")
    save_layout(out, None, size, note=note, spawn=spawn, keys=best["keys"],
                tiling=tiling if tiling["tiling"] != "diamonds" else None)
    drop = ("keys", "stuck_xy")
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump({"name": args.name, "out": out, "note": note, "features": features,
                   "final": [{k: v for k, v in x.items() if k not in drop} for x in final]}, f, indent=1)
    print(f"saved {out} ({best['tag']}, {best['score']:.4f}{vs_base}) in {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
