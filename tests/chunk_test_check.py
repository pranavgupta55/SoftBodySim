"""Checks for chunk_test.py (the per-chunk stuckness test). Run from the repo root:

    .venv/bin/python3 tests/chunk_test_check.py

1. chunk_rects: 4 grids x 16 chunks tiling the world; the shifted grids' last row / column wraps.
2. A chunk's world builds: only the chunk's tiles (all overlapping its rect, across the wrap for a seam chunk),
   shapes inside the rect, none overlapping each other or a tile, fill near the target.
3. Determinism: the same (chunk, seed) twice gives bitwise equal positions after 1 s and the same result; another
   seed gives other shapes.
4. A run: scores in [0, 1], cleared + stuck = shapes, cleared shapes are gone from the world.
5. Layouts: the map's own keys rebuild its tiles exactly (order, polygons, palette picks); run_chunk ==
   run_chunk_keys(info's own keys) bit for bit (a full 10 s test); another layout changes only that chunk's tiles;
   keys off the tiling raise KeyError.
6. ChunkPool: the same results as serial run_chunk_keys, in job order; no worker left after the block; workers of
   a SIGKILLed parent exit by themselves.
Exit status 1 on any failure.
"""
import os
import signal
import subprocess
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import chunk_test as ct  # noqa: E402

FAILS = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


def run_frames(test, frames):
    for _ in range(frames):
        test.step()
    return test


def main():
    info = ct.MapInfo("evolved 1 halloween")
    W, H = info.size
    rects = ct.chunk_rects(info.size)
    check(len(rects) == 64 and len({r["id"] for r in rects}) == 64, "64 distinct chunk ids (4 grids x 16)")
    grid = [r for r in rects if r["grid"] == "grid"]
    check(abs(sum(r["rect"][2] * r["rect"][3] for r in grid) - W * H) < 1e-6, "the grid's chunks cover the world")
    last = ct.find_chunk(info.size, "shift_xy r3c3")["rect"]
    check(last[0] + last[2] > W and last[1] + last[3] > H, "shift_xy r3c3 reaches past the world's corner (wraps)")

    for cid in ("grid r1c1", "shift_xy r3c3"):
        t = ct.ChunkTest(info, ct.find_chunk(info.size, cid), 1)
        x0, y0, w, h = t.rect
        tiles, _ = ct.chunk_tiles(info, t.chunk["rect"])
        check(len(t.tile_bodies) == len(tiles) > 0, f"{cid}: {len(tiles)} tiles, only the chunk's")
        check(all(b.points[0].max() > x0 and b.points[0].min() < x0 + w and b.points[1].max() > y0
                  and b.points[1].min() < y0 + h for b in t.tile_bodies), f"{cid}: every tile overlaps the rect")
        c = np.array([b.chunk_c for b in t.shapes])  # placement circles (a pumpkin's isn't its point mean)
        r = np.array([b.chunk_r for b in t.shapes])
        inside = all(x0 <= b.points[0].min() and b.points[0].max() <= x0 + w and y0 <= b.points[1].min()
                     and b.points[1].max() <= y0 + h for b in t.shapes)
        check(len(t.shapes) > 50 and inside, f"{cid}: {len(t.shapes)} shapes, all inside the rect")
        d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1)) + np.eye(len(c)) * 1e9
        check(bool(np.all(d >= (r[:, None] + r[None, :]) * 0.999)), f"{cid}: placement circles don't overlap")
        from diamonds_map import _dist_to_polygon
        clear = min(_dist_to_polygon(p, tb.points) - ri for p, ri in zip(c, r) for tb in t.tile_bodies)
        check(clear >= 0, f"{cid}: shapes clear of tiles (min clearance {clear:.2f})")
        check(0.5 < t.fill_achieved <= t.fill + 0.02, f"{cid}: fill achieved {t.fill_achieved:.3f} of {t.fill}")

    chunk = ct.find_chunk(info.size, "grid r0c1")
    a = run_frames(ct.ChunkTest(info, chunk, 3), 60)
    b = run_frames(ct.ChunkTest(info, chunk, 3), 60)
    check(np.array_equal(a.world.P, b.world.P) and np.array_equal(a.world.V, b.world.V),
          "same seed: bitwise equal state after 1 s")
    o = ct.ChunkTest(info, chunk, 4)
    check(len(o.shapes) != len(a.shapes) or not np.allclose(o.shapes[0].points, a.shapes[0].points),
          "another seed: other shapes")
    r1 = ct.run_chunk(info, chunk, 3, seconds=2.0)
    r2 = ct.run_chunk(info, chunk, 3, seconds=2.0)
    strip = ("wall", "build_wall")
    check({k: v for k, v in r1.items() if k not in strip} == {k: v for k, v in r2.items() if k not in strip},
          f"same seed: same result (score {r1['score']}, stuck {r1['stuck_count']})")
    check(all(0.0 <= r1[k] <= 1.0 for k in ("score", "area_score", "flow")), "scores in [0, 1]")
    check(r1["cleared"] + r1["stuck_count"] == r1["shapes"] and len(r1["stuck"]) == r1["stuck_count"],
          "cleared + stuck = shapes")
    t = run_frames(ct.ChunkTest(info, chunk, 3), 120)
    ids = {id(x) for x in t.world.bodies}
    check(all(id(x) not in ids for x, _ in t.cleared) and all(id(x) in ids for x in t.remaining),
          f"cleared shapes removed from the world ({len(t.cleared)} cleared)")

    check_layouts(info)
    check_pool(info)
    print("FAILED: " + "; ".join(FAILS) if FAILS else "all passed")
    return 1 if FAILS else 0


def strip(res):
    return {k: v for k, v in res.items() if k not in ("wall", "build_wall")}


def other_layout(info, chunk_id="grid r1c1", drop=6):
    """The map's layout minus `drop` tiles whose centres lie in the chunk."""
    x0, y0, w, h = ct.find_chunk(info.size, chunk_id)["rect"]
    mine = sorted(k for k in info.keys if x0 <= k[0] < x0 + w and y0 <= k[1] < y0 + h)
    return info.keys - frozenset(mine[::max(1, len(mine) // drop)][:drop])


def check_layouts(info):
    own = info.layout_tiles(info.keys)
    check(len(own) == len(info.tiles) and all(np.array_equal(a, b) for (a, _), b in zip(own, info.tiles))
          and [p for _, p in own] == info.tile_picks, f"own keys rebuild the map's {len(own)} tiles exactly")
    cid = "grid r1c1"
    chunk = ct.find_chunk(info.size, cid)
    a = run_frames(ct.ChunkTest(info, chunk, 2), 60)
    b = run_frames(ct.ChunkTest(info, chunk, 2, keys=info.keys), 60)
    check(np.array_equal(a.world.P, b.world.P) and np.array_equal(a.world.V, b.world.V),
          "own keys: bitwise equal state after 1 s")
    r1 = ct.run_chunk(info, chunk, 2)
    r2 = ct.run_chunk_keys(info, sorted(info.keys), cid, 2)  # a list, any order: the same layout
    check(strip(r1) == strip(r2), f"run_chunk == run_chunk_keys(own keys), full {r1['sim_seconds']:g} s "
                                  f"(flow {r1['flow']}, stuck {r1['stuck_count']})")
    alt = other_layout(info, cid)
    t_alt = ct.ChunkTest(info, chunk, 2, keys=alt, seconds=0.1)
    t_far = ct.ChunkTest(info, ct.find_chunk(info.size, "grid r3c3"), 2, keys=alt, seconds=0.1)
    t_far0 = ct.ChunkTest(info, ct.find_chunk(info.size, "grid r3c3"), 2, seconds=0.1)
    check(len(t_alt.tile_bodies) < len(a.tile_bodies)
          and len(t_far.tile_bodies) == len(t_far0.tile_bodies)
          and np.array_equal(t_far.world.P, t_far0.world.P),
          f"another layout: {len(a.tile_bodies)} -> {len(t_alt.tile_bodies)} tiles in {cid}, grid r3c3 unchanged")
    try:
        info.layout_tiles(set(info.keys) | {(1.23, 4.56)})
        check(False, "a key off the tiling raises KeyError")
    except KeyError:
        check(True, "a key off the tiling raises KeyError")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def check_pool(info):
    alt = other_layout(info)
    jobs = [(info.keys, "grid r1c1", 1), (alt, "grid r1c1", 1), (alt, "shift_xy r3c3", 5), (info.keys, "grid r0c2", 2),
            (sorted(alt), "grid r1c1", 1, {"seconds": 1.0})]
    kw = {"seconds": 3.0}
    serial = [ct.run_chunk_keys(info, k, c, s, **dict(kw, **(j[3] if len(j) > 3 else {})))
              for j in jobs for k, c, s in [j[:3]]]
    with ct.ChunkPool(info.path, 3, **kw) as pool:
        pids = pool.pids()
        got = pool.score(jobs)
    check([strip(r) for r in got] == [strip(r) for r in serial], f"pool == serial on {len(jobs)} jobs, in order")
    check(got[0]["flow"] != got[1]["flow"] or got[0]["tiles"] != got[1]["tiles"], "layouts differ in the pool")
    time.sleep(0.5)
    check(not any(alive(p) for p in pids), f"no pool worker left after the block ({len(pids)} workers)")
    # a parent killed with SIGKILL: its workers notice (watchdog) and exit
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--pool-child"], stdout=subprocess.PIPE,
                             text=True, cwd=os.getcwd())
    line = child.stdout.readline()
    pids = [int(v) for v in line.split()]
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    deadline = time.time() + 10
    while time.time() < deadline and any(alive(p) for p in pids):
        time.sleep(0.25)
    left = [p for p in pids if alive(p)]
    for p in left:
        os.kill(p, signal.SIGKILL)
    check(len(pids) == 2 and not left, f"workers of a SIGKILLed parent exit by themselves ({len(pids)} workers, "
                                       f"{len(left)} left)")


def pool_child():
    with ct.ChunkPool("evolved 1 halloween", 2) as pool:
        pool.score([(pool_child.keys, "grid r0c0", 1, {"seconds": 0.1})] * 2)  # workers up and initialised
        print(" ".join(str(p) for p in pool.pids()), flush=True)
        time.sleep(120)


if __name__ == "__main__":
    if "--pool-child" in sys.argv:
        pool_child.keys = ct.MapInfo("evolved 1 halloween").keys
        pool_child()
        sys.exit(0)
    sys.exit(main())
