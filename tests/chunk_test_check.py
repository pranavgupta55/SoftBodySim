"""Checks for chunk_test.py (the per-chunk stuckness test). Run from the repo root:

    .venv/bin/python3 tests/chunk_test_check.py

1. chunk_rects: 4 grids x 16 chunks tiling the world; the shifted grids' last row / column wraps.
2. A chunk's world builds: only the chunk's tiles (all overlapping its rect, across the wrap for a seam chunk),
   shapes inside the rect, none overlapping each other or a tile, fill near the target.
3. Determinism: the same (chunk, seed) twice gives bitwise equal positions after 1 s and the same result; another
   seed gives other shapes.
4. A run: scores in [0, 1], cleared + stuck = shapes, cleared shapes are gone from the world.
Exit status 1 on any failure.
"""
import os
import sys

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
    print("FAILED: " + "; ".join(FAILS) if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
