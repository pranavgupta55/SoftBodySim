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

Then the last round's top `finalists` and the base run `final_seconds` each, and the best is written to
maps/<tiling>/NAME.json (with the spawn settings it was evolved under, and for penrose the pentagrid seed and edge),
so it shows up as "<tiling>/NAME" in the Worlds panel. Progress goes to stdout (one line per round) and every run to
LOG_DIR/runs.jsonl.

--tiling diamonds (default) picks from the tumbling-blocks lattice; penrose from a Penrose P3 rhombus tiling
(diamonds_map.penrose_tiles, pentagrid offsets seeded by --pentagrid-seed). --base random starts from a seeded random
layout (--seed) at the current spawn settings' density (scene.DIAMONDS_SPAWN / PENROSE_SPAWN, or --density);
--base LAYOUT (a name in maps/<tiling>) starts from that layout, its tiling and its own spawn settings.

--rounds is a maximum with --deadline (local time): before each round after the first, the run checks that one more
round (the longest one so far) plus the final phase (ceil((finalists + 1) / workers) waves of final runs, timed from
the last round's runs per sim second) and a margin still end before the deadline, and goes straight to the final
phase if not, so the map is saved in time.

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


def cell_unstuck(pos, avg, W, H):
    """The unstuck score of one sample: shape centers `pos` (n, 2, any wrap) and their 3-s average speeds `avg` (n):
    per cell of a CELLS x CELLS grid with at least CELL_MIN_SHAPES shapes, the mean of its slowest SLOW_SHARE (at
    least 1) over the global median speed, capped at 1; then the mean of the WORST_CELLS lowest (all if fewer).
    None if no cell qualifies."""
    import numpy as np
    med = max(float(np.median(avg)), 1e-9)
    ix = np.clip((pos[:, 0] % W / W * CELLS).astype(int), 0, CELLS - 1)
    iy = np.clip((pos[:, 1] % H / H * CELLS).astype(int), 0, CELLS - 1)
    cell = iy * CELLS + ix
    vals = []
    for k in range(CELLS * CELLS):
        v = np.sort(avg[cell == k])
        if len(v) >= CELL_MIN_SHAPES:
            vals.append(min(1.0, float(v[:max(1, int(round(SLOW_SHARE * len(v))))].mean()) / med))
    if not vals:
        return None
    return float(np.mean(sorted(vals)[:WORST_CELLS]))


def run(job):
    """job: (tag, layout keys, add_seed or None (the layout itself), remove, add, frames) -> result dict."""
    tag, keys, add_seed, remove, add, frames = job
    import numpy as np
    import scene
    from entities import SoftBody
    from diamonds_map import tile_key
    W, H = _SETUP["size"]
    t = time.time()
    vary = add_seed is not None
    tiling = _SETUP["tiling"]
    w = scene.build_diamonds_scene(W, H, layout=(keys, (W, H)), remove=remove if vary else 0, add=add if vary else 0,
                                   add_seed=add_seed or 0, spawn=_SETUP["spawn"], tiling=tiling["tiling"],
                                   pentagrid_seed=tiling.get("pentagrid_seed"))
    dia = [b.points for b in w.bodies if getattr(b, "palette_role", "") == "diamond"]
    c = np.array([p.mean(axis=1) for p in dia])
    counts = np.histogram2d(c[:, 0], c[:, 1], (4, 4), ((0, W), (0, H)))[0].ravel()
    even = 1.0 / (1.0 + counts.std() / counts.mean())
    shapes = [b for b in w.bodies if isinstance(b, SoftBody)]
    hx, hy = np.zeros(BX), np.zeros(BY)
    track_from = frames - SCORE_WINDOW - SPEED_WINDOW  # speeds are kept from 3 s before the scored window
    speeds = np.zeros((SPEED_WINDOW, len(shapes)))  # rolling: the last 3 s of each shape's speed (px/s)
    prev = None
    unstuck, slow_speed = [], []
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
            u = cell_unstuck(c, avg, W, H)
            if u is not None:
                unstuck.append(u)
    spread = 0.7 * entropy(hx) + 0.3 * entropy(hy)
    stuck = float(np.mean(unstuck)) if unstuck else 0.0
    w1, w2, w3 = _SETUP["weights"]
    return {"tag": tag, "score": float(w1 * spread + w2 * stuck + w3 * even), "spread": spread, "unstuck": stuck,
            "slow_speed": float(np.mean(slow_speed)), "even": float(even),
            "diamonds": len(dia), "keys": [tile_key(p) for p in dia], "secs": round(time.time() - t, 1)}


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
        w = scene.build_diamonds_scene(*size, seed=args.seed, layout=None, spawn=spawn, tiling=tiling["tiling"],
                                       pentagrid_seed=tiling.get("pentagrid_seed"))
        keys = [tuple(tile_key(b.points)) for b in w.bodies if getattr(b, "palette_role", "") == "diamond"]
        return keys, spawn, tiling, f"random base (seed {args.seed}, {len(keys)} tiles)"
    path = os.path.join(tiling_dir(args.tiling), args.base + ".json")
    keys, _ = load_layout(path)
    return keys, scene.diamonds_spawn(path), scene._layout_tiling(path), f"base {args.base}"


def parse_deadline(text):
    """'YYYY-MM-DD HH:MM' local time -> epoch seconds (None for None)."""
    return None if text is None else time.mktime(time.strptime(text, "%Y-%m-%d %H:%M"))


def fits_deadline(deadline, now, round_secs, run_secs_per_sim, args):
    """(fits, estimate in s) of one more round (the longest so far) plus the final phase: ceil((finalists + 1) /
    workers) waves of final_seconds runs at the measured wall seconds per sim second, +15%, + DEADLINE_MARGIN."""
    waves = math.ceil((args.finalists + 1) / args.workers)
    final = waves * args.final_seconds * run_secs_per_sim * 1.15
    need = max(round_secs) + final + DEADLINE_MARGIN
    return now + need <= deadline, need


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name", help="the new layout's name (maps/<tiling>/NAME.json, world <tiling>/NAME)")
    p.add_argument("--tiling", choices=("diamonds", "penrose"), default="diamonds",
                   help="the tiling the obstacles are picked from (and the maps folder)")
    p.add_argument("--pentagrid-seed", type=int, default=None,
                   help="penrose random base: the pentagrid offsets' seed (default scene.PENROSE_SEED)")
    p.add_argument("--base", default="random", help="random, or a layout name in maps/<tiling>")
    p.add_argument("--seed", type=int, default=int(time.time()) % 100000, help="random base's seed (default: clock)")
    p.add_argument("--rounds", type=int, default=2, help="rounds (a maximum with --deadline)")
    p.add_argument("--children", type=int, default=100, help="variants per round")
    p.add_argument("--keep", type=int, default=5, help="parents kept per round")
    p.add_argument("--remove", type=int, default=3)
    p.add_argument("--add", type=int, default=3)
    p.add_argument("--seconds", type=float, default=60, help="sim seconds per run")
    p.add_argument("--finalists", type=int, default=8)
    p.add_argument("--final-seconds", type=float, default=180)
    p.add_argument("--size", default="1800x1169", help="world size WxH (the screen's: window px / SCALE_DOWN / zoom)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--weights", default="0.7,0.2,0.1", help="w1,w2,w3: spread, unstuck, tile evenness")
    p.add_argument("--density", type=float, default=None,
                   help="random base's tiles per square world unit (default: scene.DIAMONDS_SPAWN's / PENROSE_SPAWN's)")
    p.add_argument("--deadline", default=None,
                   help='"YYYY-MM-DD HH:MM" local time: stop starting rounds so the map is saved before it')
    p.add_argument("--log-dir", default=None, help="default: <tmp>/evolve_layout/NAME")
    args = p.parse_args(argv)
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
    print(f"{args.name}: {tiling}, {base_desc}, spawn {spawn}, up to {args.rounds} rounds x {args.children} runs of "
          f"{args.seconds:g} s{', deadline ' + args.deadline if deadline else ''}, log {log_path}", flush=True)
    t0 = time.time()
    parents, results, round_secs, rounds_done = [("base", base)], [], [], 0
    with mp.Pool(args.workers, initializer=init, initargs=(size, spawn, weights, tiling)) as pool:
        for r in range(1, args.rounds + 1):
            if deadline is not None and round_secs:
                ok, need = fits_deadline(deadline, time.time(), round_secs, run_secs_per_sim, args)
                if not ok:
                    print(f"deadline: another round + final need ~{need:.0f} s, {deadline - time.time():.0f} s left: "
                          f"going to the final after {rounds_done} rounds", flush=True)
                    break
            t_round = time.time()
            per = max(1, args.children // len(parents))
            jobs = [(f"r{r}.{i}.{k}", keys, r * 100000 + i * 1000 + k, args.remove, args.add, frames)
                    for i, (_, keys) in enumerate(parents) for k in range(per)]
            results = []
            for res in pool.imap_unordered(run, jobs):
                res["round"] = r
                log(res)
                results.append(res)
            round_secs.append(time.time() - t_round)
            run_secs_per_sim = sum(x["secs"] for x in results) / len(results) / args.seconds
            rounds_done = r
            results.sort(key=lambda x: -x["score"])
            top = results[:args.keep]
            print(f"round {r}: {len(results)} runs, best {results[0]['score']:.4f} (spread {results[0]['spread']:.4f}, "
                  f"unstuck {results[0]['unstuck']:.3f}, even {results[0]['even']:.3f}), top {args.keep} mean "
                  f"{sum(x['score'] for x in top) / len(top):.4f}, median {results[len(results) // 2]['score']:.4f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
            parents = [(x["tag"], x["keys"]) for x in top]
        jobs = [("base", base, None, 0, 0, final_frames)]
        jobs += [(x["tag"], x["keys"], None, 0, 0, final_frames) for x in results[:args.finalists]]
        final = []
        for res in pool.imap_unordered(run, jobs):
            res["round"] = "final"
            log(res)
            final.append(res)
    final.sort(key=lambda x: -x["score"])
    for x in final:
        print(f"final {x['tag']}: {x['score']:.4f} (spread {x['spread']:.4f}, unstuck {x['unstuck']:.3f}, "
              f"even {x['even']:.3f})", flush=True)
    best = final[0]
    base_score = next(x["score"] for x in final if x["tag"] == "base")
    sys.path.insert(0, REPO)
    from diamonds_map import save_layout
    stopped = f" (deadline {args.deadline}: stopped after {rounds_done})" if rounds_done < args.rounds else ""
    note = (f"{args.name} ({time.strftime('%Y-%m-%d')}): {tiling['tiling']} tiling, {base_desc}; {rounds_done} rounds"
            f"{stopped} of {args.children} remove-{args.remove}/add-{args.add} variants run {args.seconds:g} s (top "
            f"{args.keep} kept per round), score {weights[0]:g} spread + {weights[1]:g} unstuck (worst {WORST_CELLS} "
            f"of {CELLS}x{CELLS} cells) + {weights[2]:g} tile evenness; last round's top {args.finalists} + base "
            f"re-run {args.final_seconds:g} s: {best['tag']} {best['score']:.4f} vs base {base_score:.4f}")
    save_layout(out, None, size, note=note, spawn=spawn, keys=best["keys"],
                tiling=tiling if tiling["tiling"] != "diamonds" else None)
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump({"name": args.name, "out": out, "note": note, "final": [{k: v for k, v in x.items() if k != "keys"}
                                                                          for x in final]}, f, indent=1)
    print(f"saved {out} ({best['tag']}, {best['score']:.4f} vs base {base_score:.4f}) in {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
