"""Collect chunk-test runs with sampled shape tracks, for the settle-threshold search (settle_analyze.py).

    python3 settle_collect.py --minutes 60 --workers 6 --out settle/runs.jsonl [--map "evolved 1 halloween"]
                              [--mutate-share 0.3] [--seed S] [--fill 0.6] [--seconds 10] [--sample-dt 0.1]

Runs random (chunk, seed) jobs, the chunk drawn among all 64 of chunk_test.chunk_rects and the seed in 1..2^31,
through a chunk_test.ChunkPool, appending one JSON line per run as it finishes, until --minutes have passed (then
it stops queueing and waits for the runs in flight). A share (--mutate-share) of the jobs runs on a mutated layout:
the base (the map's own tiles) with chunk_mutate.mutate_chunk applied, stuck=[], to every grid chunk the test chunk
overlaps (1 for a "grid" chunk, 2 for shift_x / shift_y, 4 for shift_xy: the same edit density per area), each with
3-6 removes and 3-6 adds. Each mutated layout is written to <out dir>/layouts/<hash>.json ({"map", "tiles":
[[x, y], ...], "mutated": [grid chunk ids]}) and named in the line's "layout" ("base" otherwise).

Line format (settle_contract.md):
    {"map", "chunk", "seed", "layout", "fill", "seconds", "sample_dt", "n",
     "shapes": [{"r", "area", "exit_t" (s, null: still in at the end), "xy": [[x, y], ...]}], "left_10",
     "fill_achieved", "wall"}
xy: each shape's centroid (mean of its points) at t = k x sample_dt (k = 0, 1, ...) while it is in the chunk, map
coordinates, not wrapped (continuous across a seam chunk's world edge).

Appending: an existing --out is kept and added to (a torn last line is cut first). Ctrl-C / a crash terminates the
workers; workers of a killed parent exit by themselves (ChunkPool). Run it as a script file (spawn pool).
"""
import argparse
import hashlib
import json
import os
import queue
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import chunk_test as ct  # noqa: E402

MAP = "evolved 1 halloween"
MUTATE_SHARE = 0.3
EDITS = (3, 6)  # removes and adds per mutated grid chunk, each uniform in this range (inclusive)
SAMPLE_DT = 0.1


def repair_tail(path):
    """Cut a torn last line (no trailing newline) off the JSONL file; the number of whole lines left."""
    if not os.path.isfile(path):
        return 0
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            f.truncate(cut)
            data = data[:cut]
    return data.count(b"\n")


def overlapping_grid_chunks(size, rect):
    """The "grid" chunk ids whose rects the chunk rect overlaps (across the world's wrap)."""
    W, H = size
    x0, y0, w, h = rect
    out = []
    for g in ct.chunk_rects(size, grids=("grid",)):
        gx, gy, gw, gh = g["rect"]
        for sx in (-W, 0.0, W):
            for sy in (-H, 0.0, H):
                if gx + sx < x0 + w - 1e-6 and x0 < gx + sx + gw - 1e-6 and gy + sy < y0 + h - 1e-6 \
                        and y0 < gy + sy + gh - 1e-6:
                    out.append(g["id"])
    return sorted(set(out))


def mutated_layout(rng, uni, base, size, chunk):
    """(layout keys, grid chunk ids mutated): base with mutate_chunk (stuck=[]) on every grid chunk the test chunk
    overlaps, EDITS removes and adds each."""
    from chunk_mutate import mutate_chunk
    keys = base
    ids = overlapping_grid_chunks(size, chunk["rect"])
    for gid in ids:
        n_remove, n_add = (int(rng.integers(EDITS[0], EDITS[1] + 1)) for _ in range(2))
        keys = mutate_chunk(rng, uni, keys, gid, [], n_remove, n_add)
    return keys, ids


def save_layout(folder, map_name, keys, ids):
    """Write the layout to folder/<hash>.json (if not there yet); its file name."""
    tiles = sorted(ct.norm_key(k) for k in keys)
    name = hashlib.sha1(json.dumps(tiles).encode()).hexdigest()[:16] + ".json"
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"map": map_name, "tiles": [list(k) for k in tiles], "mutated": ids}, f)
        os.replace(tmp, path)
    return name


def to_line(map_name, layout, res, fill, seconds):
    """A ChunkPool result (run with sample_dt) -> the contract's JSONL record."""
    tr = res["track"]
    shapes = tr["shapes"]
    return {"map": map_name, "chunk": res["chunk"], "seed": int(res["seed"]), "layout": layout, "fill": fill,
            "seconds": seconds, "sample_dt": tr["sample_dt"], "n": len(shapes), "shapes": shapes,
            "left_10": sum(s["exit_t"] is None for s in shapes),
            "fill_achieved": res["fill_achieved"], "wall": res["wall"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--minutes", type=float, required=True, help="stop queueing new runs after this long")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True, help="JSONL file to append to")
    p.add_argument("--map", default=MAP)
    p.add_argument("--mutate-share", type=float, default=MUTATE_SHARE)
    p.add_argument("--seed", type=int, default=None, help="job draw seed (default: fresh entropy)")
    p.add_argument("--fill", type=float, default=ct.FILL)
    p.add_argument("--seconds", type=float, default=10.0)  # the ground truth: shapes still in at 10 s
    p.add_argument("--sample-dt", type=float, default=SAMPLE_DT)
    p.add_argument("--max-runs", type=int, default=None, help="also stop queueing after this many runs (tests)")
    args = p.parse_args(argv)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    had = repair_tail(out)
    layouts_dir = os.path.join(os.path.dirname(out), "layouts")
    rng = np.random.default_rng(args.seed)
    info = ct.MapInfo(args.map)
    chunks = ct.chunk_rects(info.size)
    uni = None
    if args.mutate_share > 0:
        from chunk_mutate import TileUniverse
        uni = TileUniverse(info)
    kw = {"fill": args.fill, "seconds": args.seconds, "sample_dt": args.sample_dt}
    print(f"settle_collect: {info.name}, {args.workers} workers, {args.minutes:g} min, mutate share "
          f"{args.mutate_share:g}, appending to {out} ({had} runs there)", flush=True)

    def draw_job():
        chunk = chunks[int(rng.integers(len(chunks)))]
        seed = int(rng.integers(1, 2 ** 31))
        if uni is not None and rng.random() < args.mutate_share:
            keys, ids = mutated_layout(rng, uni, info.keys, info.size, chunk)
            layout = save_layout(layouts_dir, info.name, keys, ids)
        else:
            keys, layout = info.keys, "base"
        return (keys, chunk["id"], seed), layout

    done = queue.Queue()
    t0 = time.monotonic()
    deadline = t0 + 60.0 * args.minutes
    queued = written = 0
    in_flight = 0
    with ct.ChunkPool(info.path, args.workers, **kw) as pool, open(out, "a") as f:
        print("worker pids: " + " ".join(str(v) for v in pool.pids()), flush=True)
        def more():
            return time.monotonic() < deadline and (args.max_runs is None or queued < args.max_runs)

        while True:
            while in_flight < args.workers + 1 and more():  # keep every worker busy, one queued
                job, layout = draw_job()
                pool.submit(job, callback=lambda res, layout=layout: done.put((layout, res)),
                            error_callback=lambda e, job=job: done.put((None, (job[1:], e))))
                queued += 1
                in_flight += 1
            if in_flight == 0:
                break
            try:
                layout, res = done.get(timeout=1.0)
            except queue.Empty:
                continue
            in_flight -= 1
            if layout is None:
                raise RuntimeError(f"run {res[0]} failed: {res[1]!r}")
            f.write(json.dumps(to_line(info.name, layout, res, args.fill, args.seconds),
                               separators=(",", ":")) + "\n")
            f.flush()
            written += 1
            if written % 10 == 0 or in_flight == 0:
                mins = (time.monotonic() - t0) / 60.0
                print(f"{written} runs in {mins:.1f} min ({written / max(mins, 1e-9):.1f}/min, "
                      f"{written / max(mins, 1e-9) / args.workers:.2f}/min/worker)", flush=True)
    mins = (time.monotonic() - t0) / 60.0
    print(f"done: {written} runs in {mins:.1f} min; {out} holds {had + written}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
