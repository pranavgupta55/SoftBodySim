"""Checks for chunk_test's sampling hook (sample_dt) and settle_collect.py. Run from the repo root:

    .venv/bin/python3 tests/settle_collect_check.py

1. Sampling doesn't change the simulation: a full 10 s chunk test with and without sample_dt gives bitwise equal
   world state and the same result (minus "track" and wall times); on a mutated layout too (2 s).
2. The track: 101 samples (t = 0..10 s) for a shape still in at the end, ceil(exit frame / 6) for a cleared one;
   sample 0 is the shape's start centroid in map coordinates; exit_t matches the cleared list; a bad sample_dt
   raises.
3. Mutated layouts: overlapping grid chunks are 1 / 2 / 2 / 4 for grid / shift_x / shift_y / shift_xy; a mutated
   layout differs from the base only in those chunks' tiles and is legal (chunk_mutate.legal).
4. settle_collect end to end (2 workers, short runs): lines in the contract's format, left_10 = shapes with
   exit_t null, mutated runs name an existing layout file holding their tiles; a second call appends; a torn last
   line is cut first; no worker left after it exits.
Exit status 1 on any failure.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import chunk_test as ct  # noqa: E402
import settle_collect as sc  # noqa: E402

FAILS = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


def strip(res):
    return {k: v for k, v in res.items() if k not in ("wall", "build_wall", "track")}


def run(test):
    while not test.done:
        test.step()
    return test


def check_identical(info, chunk, seed, keys=None, seconds=ct.SECONDS):
    a = run(ct.ChunkTest(info, chunk, seed, keys=keys, seconds=seconds))
    b = run(ct.ChunkTest(info, chunk, seed, keys=keys, seconds=seconds, sample_dt=0.1))
    same = np.array_equal(a.world.P, b.world.P) and np.array_equal(a.world.V, b.world.V)
    ra, rb = a.result(), b.result()
    check(same and strip(ra) == strip(rb) and "track" not in ra and "track" in rb,
          f"{chunk['id']} seed {seed} {seconds:g} s{' (mutated)' if keys is not None else ''}: bit-identical "
          f"with sampling (stuck {ra['stuck_count']} of {ra['shapes']})")
    return b


def check_track(t):
    tr = t.result()["track"]
    shapes = tr["shapes"]
    check(len(shapes) == len(t.shapes) and tr["sample_dt"] == 0.1, f"track: {len(shapes)} shapes, sample_dt 0.1")
    start = [list(np.round(b.chunk_c + t.offset, 2)) for b in t.shapes]  # placement centre, not the centroid
    exit_t = {id(b): s for b, s in t.cleared}
    ok_len, ok_exit = True, True
    for b, s in zip(t.shapes, shapes):
        if id(b) in exit_t:
            frame = int(round(exit_t[id(b)] * ct.FPS))
            ok_len &= len(s["xy"]) == math.ceil(frame / 6)
            ok_exit &= s["exit_t"] is not None and abs(s["exit_t"] - exit_t[id(b)]) < 1e-3
        else:
            ok_len &= len(s["xy"]) == 101
            ok_exit &= s["exit_t"] is None
    check(ok_len, "track lengths: 101 samples if still in, ceil(exit frame / 6) if cleared")
    check(ok_exit, "exit_t matches the cleared list (null for the ones still in)")
    # sample 0: centroid at t = 0, map coordinates (polygons' centroid == placement centre; pumpkins' isn't)
    d = [math.hypot(s["xy"][0][0] - c[0], s["xy"][0][1] - c[1]) for s, c in zip(shapes, start)]
    check(float(np.median(d)) < 0.05 and max(d) < 30, f"sample 0 at the start centroid in map coordinates "
                                                       f"(median offset {np.median(d):.3f})")
    last = t.remaining[0] if t.remaining else None
    if last is not None:
        k = t.shapes.index(last)
        c = last.points.mean(axis=1) + t.offset
        check(np.allclose(shapes[k]["xy"][-1], c, atol=0.006), "last sample = centroid at t = 10 s")
    try:
        ct.ChunkTest(t.info, t.chunk, 1, seconds=0.1, sample_dt=0.025)
        check(False, "sample_dt not a whole number of frames raises")
    except ValueError:
        check(True, "sample_dt not a whole number of frames raises")


def check_mutation(info):
    from chunk_mutate import TileUniverse, legal
    uni = TileUniverse(info)
    counts = {g: len(sc.overlapping_grid_chunks(info.size, ct.find_chunk(info.size, f"{g} r3c3")["rect"]))
              for g in ct.GRIDS}
    check(counts == {"grid": 1, "shift_x": 2, "shift_y": 2, "shift_xy": 4}, f"overlapping grid chunks {counts}")
    rng = np.random.default_rng(5)
    ok = True
    for cid in ("grid r1c2", "shift_xy r3c3", "shift_x r0c1"):
        keys, ids = sc.mutated_layout(rng, uni, info.keys, info.size, ct.find_chunk(info.size, cid))
        diff = keys ^ info.keys
        owners = {uni.chunk_ids[uni.owner[uni.index[k]]] for k in diff}
        ok &= bool(diff) and owners <= set(ids) and not legal(uni, keys)
        print(f"     {cid}: mutated {ids}, {len(info.keys - keys)} removed, {len(keys - info.keys)} added")
    check(ok, "mutated layouts differ from the base only in the overlapping grid chunks, and are legal")
    keys, _ = sc.mutated_layout(rng, uni, info.keys, info.size, ct.find_chunk(info.size, "grid r2c1"))
    return keys


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def collect(out, *extra):
    cmd = [sys.executable, os.path.join(os.getcwd(), "settle_collect.py"), "--minutes", "5", "--workers", "2",
           "--out", out, "--seconds", "2", "--seed", "3", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    pids = [int(v) for line in p.stdout.splitlines() if line.startswith("worker pids:")
            for v in line.split(":")[1].split()]
    if p.returncode:
        print(p.stdout, p.stderr)
    return p.returncode, pids


def check_collect(info):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "settle", "runs.jsonl")
        rc, pids = collect(out, "--max-runs", "3", "--mutate-share", "0.5")
        time.sleep(0.5)
        check(rc == 0 and len(pids) == 2 and not any(alive(p) for p in pids),
              f"collector ran (exit {rc}) and left no worker ({len(pids)} workers)")
        lines = [json.loads(v) for v in open(out)]
        fields = {"map", "chunk", "seed", "layout", "fill", "seconds", "sample_dt", "n", "shapes", "left_10"}
        ok = len(lines) == 3 and all(fields <= set(r) for r in lines)
        ok &= all(r["n"] == len(r["shapes"]) and r["left_10"] == sum(s["exit_t"] is None for s in r["shapes"])
                  and all(set(s) == {"r", "area", "exit_t", "xy"} for s in r["shapes"]) for r in lines)
        ok &= all(r["seconds"] == 2.0 and r["sample_dt"] == 0.1 and r["map"] == info.name for r in lines)
        ok &= all(len(s["xy"]) == 21 for r in lines for s in r["shapes"] if s["exit_t"] is None)
        check(ok, f"3 lines in the contract format ({[(r['chunk'], r['layout'][:8], r['left_10']) for r in lines]})")
        rc, _ = collect(out, "--max-runs", "2", "--mutate-share", "1")
        lines = [json.loads(v) for v in open(out)]
        mut = [r for r in lines if r["layout"] != "base"]
        files_ok = bool(mut) and all(os.path.isfile(os.path.join(d, "settle", "layouts", r["layout"])) for r in mut)
        lay = json.load(open(os.path.join(d, "settle", "layouts", mut[0]["layout"]))) if mut else {}
        check(rc == 0 and len(lines) == 5 and files_ok and len(lay.get("tiles", [])) > 100
              and frozenset(ct.norm_key(k) for k in lay["tiles"]) != info.keys,
              f"second call appended (5 lines, {len(mut)} mutated with layout files)")
        with open(out, "a") as f:
            f.write('{"map": "torn')
        rc, _ = collect(out, "--max-runs", "1", "--mutate-share", "0")
        lines = [json.loads(v) for v in open(out)]  # raises on a torn line
        check(rc == 0 and len(lines) == 6 and lines[-1]["layout"] == "base", "torn last line cut, then appended")


def main():
    info = ct.MapInfo("evolved 1 halloween")
    t = check_identical(info, ct.find_chunk(info.size, "grid r1c1"), 2)
    check_track(t)
    keys = check_mutation(info)
    check_identical(info, ct.find_chunk(info.size, "grid r2c1"), 4, keys=keys, seconds=2.0)
    check_collect(info)
    print("FAILED: " + "; ".join(FAILS) if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
