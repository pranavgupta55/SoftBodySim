"""Evolve a diamonds layout unattended and save the winner as a new world in the diamonds folder.

    python3 evolve_layout.py NAME [--base random|LAYOUT] [--rounds 2] [--children 100] [--keep 5] [--seconds 60] ...

Round 1 runs `children` variants of the base layout (each drops `remove` random diamonds and adds `add` new ones,
diamonds_map.pick_diamonds), later rounds give each of the previous round's top `keep` an equal share of `children`
variants. Every run is the physics only (no display, no drawing) for `seconds` of sim time, scored over its last 10 s:

    score = 0.9 x shape spread (0.7 x-entropy + 0.3 y-entropy of the shapes' centers, 1 = perfectly even)
          + 0.1 x diamond evenness (1 / (1 + std/mean) of the diamond counts in a 4x4 grid of the world)

Then the last round's top `finalists` and the base run `final_seconds` each, and the best is written to
maps/diamonds/NAME.json (with the spawn settings it was evolved under), so it shows up as "diamonds/NAME" in the Worlds
panel. Progress goes to stdout (one line per round) and every run to LOG_DIR/runs.jsonl.

--base random starts from a seeded random layout (--seed) at the current spawn settings' density (scene.DIAMONDS_SPAWN);
--base LAYOUT (a name in maps/diamonds) starts from that layout and its own spawn settings. Runs use `workers` processes
(each imports and compiles once); launch it at low priority, e.g. `nice -n 10`, with OMP_NUM_THREADS=1.
"""
import argparse
import json
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
_SETUP = {}  # per worker: the world size and spawn settings (init)


def init(size, spawn):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from entities import EnvConfig
    EnvConfig.backend = "cpu"
    _SETUP.update(size=size, spawn=spawn)


def entropy(counts):
    import numpy as np
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


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
    w = scene.build_diamonds_scene(W, H, layout=(keys, (W, H)), remove=remove if vary else 0, add=add if vary else 0,
                                   add_seed=add_seed or 0, spawn=_SETUP["spawn"])
    dia = [b.points for b in w.bodies if getattr(b, "palette_role", "") == "diamond"]
    c = np.array([p.mean(axis=1) for p in dia])
    counts = np.histogram2d(c[:, 0], c[:, 1], (4, 4), ((0, W), (0, H)))[0].ravel()
    even = 1.0 / (1.0 + counts.std() / counts.mean())
    shapes = [b for b in w.bodies if isinstance(b, SoftBody)]
    hx, hy = np.zeros(BX), np.zeros(BY)
    for f in range(frames):
        w.step(1 / 60)
        if f >= frames - SCORE_WINDOW and f % EVERY == 0:
            s = np.array([b.points.mean(axis=1) for b in shapes])
            hx += np.histogram(s[:, 0] % W, BX, (0, W))[0]
            hy += np.histogram(s[:, 1] % H, BY, (0, H))[0]
    spread = 0.7 * entropy(hx) + 0.3 * entropy(hy)
    return {"tag": tag, "score": float(0.9 * spread + 0.1 * even), "spread": spread, "even": float(even),
            "diamonds": len(dia), "keys": [tile_key(p) for p in dia], "secs": round(time.time() - t, 1)}


def base_layout(args, size):
    """(keys, spawn settings, description) of the starting layout."""
    sys.path.insert(0, REPO)
    import scene
    from diamonds_map import load_layout, tile_key
    if args.base == "random":
        spawn = scene.diamonds_spawn()
        w = scene.build_diamonds_scene(*size, seed=args.seed, layout=None, spawn=spawn)
        keys = [tuple(tile_key(b.points)) for b in w.bodies if getattr(b, "palette_role", "") == "diamond"]
        return keys, spawn, f"random base (seed {args.seed}, {len(keys)} diamonds)"
    path = os.path.join(scene.DIAMONDS_DIR, args.base + ".json")
    keys, _ = load_layout(path)
    return keys, scene.diamonds_spawn(path), f"base {args.base}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name", help="the new layout's name (maps/diamonds/NAME.json, world diamonds/NAME)")
    p.add_argument("--base", default="random", help="random, or a layout name in maps/diamonds")
    p.add_argument("--seed", type=int, default=int(time.time()) % 100000, help="random base's seed (default: clock)")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--children", type=int, default=100, help="variants per round")
    p.add_argument("--keep", type=int, default=5, help="parents kept per round")
    p.add_argument("--remove", type=int, default=3)
    p.add_argument("--add", type=int, default=3)
    p.add_argument("--seconds", type=float, default=60, help="sim seconds per run")
    p.add_argument("--finalists", type=int, default=8)
    p.add_argument("--final-seconds", type=float, default=180)
    p.add_argument("--size", default="1800x1169", help="world size WxH (the screen's: window px / SCALE_DOWN / zoom)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-dir", default=None, help="default: <tmp>/evolve_layout/NAME")
    args = p.parse_args(argv)
    size = tuple(float(v) for v in args.size.lower().split("x"))
    out = os.path.join(REPO, "maps", "diamonds", args.name + ".json")
    log_dir = args.log_dir or os.path.join(tempfile.gettempdir(), "evolve_layout", args.name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "runs.jsonl")
    open(log_path, "w").close()

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    base, spawn, base_desc = base_layout(args, size)
    frames, final_frames = int(args.seconds * 60), int(args.final_seconds * 60)
    print(f"{args.name}: {base_desc}, spawn {spawn}, {args.rounds} rounds x {args.children} runs of {args.seconds:g} s, "
          f"log {log_path}", flush=True)
    t0 = time.time()
    parents, results = [("base", base)], []
    with mp.Pool(args.workers, initializer=init, initargs=(size, spawn)) as pool:
        for r in range(1, args.rounds + 1):
            per = max(1, args.children // len(parents))
            jobs = [(f"r{r}.{i}.{k}", keys, r * 100000 + i * 1000 + k, args.remove, args.add, frames)
                    for i, (_, keys) in enumerate(parents) for k in range(per)]
            results = []
            for res in pool.imap_unordered(run, jobs):
                res["round"] = r
                log(res)
                results.append(res)
            results.sort(key=lambda x: -x["score"])
            top = results[:args.keep]
            print(f"round {r}: {len(results)} runs, best {results[0]['score']:.4f}, top {args.keep} mean "
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
        print(f"final {x['tag']}: {x['score']:.4f} (spread {x['spread']:.4f}, even {x['even']:.3f})", flush=True)
    best = final[0]
    base_score = next(x["score"] for x in final if x["tag"] == "base")
    sys.path.insert(0, REPO)
    from diamonds_map import save_layout
    note = (f"{args.name} ({time.strftime('%Y-%m-%d')}): {base_desc}; {args.rounds} rounds of {args.children} "
            f"remove-{args.remove}/add-{args.add} variants run {args.seconds:g} s (top {args.keep} kept per round), "
            f"score 0.9 x shape spread + 0.1 x diamond evenness; last round's top {args.finalists} + base re-run "
            f"{args.final_seconds:g} s: {best['tag']} {best['score']:.4f} vs base {base_score:.4f}")
    save_layout(out, None, size, note=note, spawn=spawn, keys=best["keys"])
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump({"name": args.name, "out": out, "note": note, "final": [{k: v for k, v in x.items() if k != "keys"}
                                                                          for x in final]}, f, indent=1)
    print(f"saved {out} ({best['tag']}, {best['score']:.4f} vs base {base_score:.4f}) in {time.time() - t0:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
