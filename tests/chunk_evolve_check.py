"""Checks for chunk_evolve.py's round loop, against a fake map backend (no physics; runs in seconds). From the repo
root:

    .venv/bin/python3 tests/chunk_evolve_check.py

The fake: a 400 x 400 world, candidate tiles on a 20-unit grid; a chunk's flow is 0.5 + 0.03 x (good tiles - bad
tiles in its rect) + a small noise that depends only on (chunk, seed), so paired comparisons are exact. "Poison" tiles
along grid r0c0's right border count +3 in grid chunks but -6 in shifted ones: grid r0c0 wants them, the seam test
must take them back out. Mutation removes / adds random tiles owned by the chunk.

1. Geometry: ownership, the 8 shifted chunks over each grid chunk, the wrap-aware window; allocate().
2. A 4-round run: flow improves; every job of a round uses that round's seeds (all of them per chunk / layout),
   fresh seeds each round; a chunk switches only when its best variant beats the incumbent by the margin, and
   unchanged chunks keep their tiles; the poison is reverted at the seams; no duplicate jobs (cache); spot checks on
   rounds 2 and 4 with the base run once; the map saved every round; the pool opened and closed once.
3. margin 1: nothing changes.
4. Deadline (fake clock): stops starting rounds that would overrun; every round ends before the deadline.
5. CLI: main(argv, backend=fake) runs and saves; --estimate prints.
Exit status 1 on any failure.
"""
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import chunk_evolve as ce  # noqa: E402
from diamonds_map import load_layout, save_layout  # noqa: E402

FAILS = []
SIZE = (400.0, 400.0)
CANDS = [(10.0 + 20 * i, 10.0 + 20 * j) for i in range(20) for j in range(20)]
POISON = {(90.0, 10.0 + 20 * j) for j in range(5)}  # grid r0c0's right column
GOOD = {t for t in CANDS if t not in POISON and (int(t[0]) // 20 * 7 + int(t[1]) // 20 * 3) % 5 < 2}


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def value(t, grid):
    if t in POISON:
        return 3.0 if grid else -6.0
    return 1.0 if t in GOOD else -1.0


def fake_result(keys, chunk_id, seed):
    rect = next(c["rect"] for c in ce.chunk_rects(SIZE, grids=ce.ALL_GRIDS) if c["id"] == chunk_id)
    inside = ce.window(SIZE, rect, keys, 0.0)
    grid = chunk_id.startswith("grid ")
    noise = 0.02 * np.random.default_rng([seed, sum(map(ord, chunk_id))]).random()
    flow = float(np.clip(0.5 + 0.03 * sum(value(t, grid) for t in inside) + noise, 0.0, 1.0))
    stuck = [[t[0], t[1], 0.0, 5.0] for t in inside if value(t, grid) < 0]
    return {"chunk": chunk_id, "seed": seed, "flow": flow, "score": min(1.0, flow + 0.1), "stuck": stuck,
            "wall": 1.4}


class FakePool:
    def __init__(self, be, workers):
        self.be, self.workers = be, workers

    def __enter__(self):
        self.be.opened += 1
        return self

    def __exit__(self, *exc):
        self.be.closed += 1

    def score(self, jobs):
        self.be.jobs.append(list(jobs))
        self.be.clock.now += 1.4 * -(-len(jobs) // self.workers)  # waves of 1.4 s tests
        return [fake_result(k, c, s) for k, c, s in jobs]


class FakeBackend:
    name = "fake base"

    def __init__(self, folder, seed=5):
        rng = np.random.default_rng(seed)
        self.size, self.pad, self.folder, self.path = SIZE, 0.0, folder, os.path.join(folder, "fake base.json")
        base = [t for t in CANDS if t not in POISON and rng.random() < 0.5]
        self.base_keys = frozenset(base)
        self.clock = Clock()
        self.opened = self.closed = 0
        self.jobs, self.saves, self.stuck_seen, self.spot_jobs = [], [], [], []

    def pool(self, workers):
        return FakePool(self, workers)

    def mutate(self, rng, keys, chunk_id, stuck, n_remove, n_add, share):
        self.stuck_seen.append(len(stuck))
        own = [t for t in CANDS if ce.owner(SIZE, t) == chunk_id]
        have = sorted(t for t in own if t in keys)
        free = sorted(t for t in own if t not in keys)
        drop = {have[i] for i in rng.permutation(len(have))[:n_remove]}
        add = {free[i] for i in rng.permutation(len(free))[:n_add]}
        return frozenset((set(keys) - drop) | add)

    def assemble(self, picks):
        out = set()
        for g, keys in picks.items():
            out |= {t for t in keys if ce.owner(SIZE, t) == g}
        return frozenset(out)

    def spot(self, jobs, seconds, workers):
        self.spot_jobs.append(len(jobs))
        self.clock.now += seconds * 0.4
        return [{"score": float(np.mean([value(t, True) for t in k])) + 0.01 * (s % 7), "unstuck": 0.5,
                 "keys": sorted(k)} for k, s in jobs]

    def save(self, path, keys, note):
        self.saves.append(path)
        save_layout(path, None, SIZE, note=note, spawn={"shapes": 1}, keys=sorted(keys), tiling={"tiling": "penrose"})


def read_log(log_dir):
    with open(os.path.join(log_dir, "runs.jsonl")) as f:
        return [json.loads(line) for line in f]


def run(tmp, tag, **kw):
    be = FakeBackend(tmp)
    log_dir = os.path.join(tmp, "log " + tag)
    out = os.path.join(tmp, f"out {tag}.json")
    args = dict(variants=4, seeds=2, workers=6, spot_every=2, spot_seconds=10.0, spot_seeds=2, seed=3)
    args.update(kw)
    with redirect_stdout(io.StringIO()) as buf:
        res = ce.evolve(be, out, log_dir, args.pop("rounds", 4), clock=be.clock, **args)
    return be, res, read_log(log_dir), out, log_dir, buf.getvalue()


def check_geometry():
    check(ce.owner(SIZE, (10.0, 10.0)) == "grid r0c0" and ce.owner(SIZE, (390.0, 110.0)) == "grid r1c3",
          "owner: the grid chunk holding the key")
    rects = {c["id"]: c["rect"] for c in ce.chunk_rects(SIZE, grids=ce.ALL_GRIDS)}
    shifted = [c for c in rects if not c.startswith("grid ")]
    counts = [sum(ce.overlaps(SIZE, rects[g], rects[s]) for s in shifted) for g in ce.grid_ids()]
    check(counts == [8] * 16, f"each grid chunk under 8 shifted chunks (2 x, 2 y, 4 xy): {set(counts)}")
    w = ce.window(SIZE, rects["shift_xy r3c3"], [(395.0, 5.0), (200.0, 200.0), (360.0, 390.0)], 0.0)
    check(w == {(395.0, 5.0), (360.0, 390.0)}, "window wraps across the world's edge")
    check(ce.window(SIZE, rects["grid r0c0"], [(103.0, 50.0)], 5.0) == {(103.0, 50.0)}
          and not ce.window(SIZE, rects["grid r0c0"], [(103.0, 50.0)], 0.0), "window pad")
    flows = {g: 0.9 for g in ce.grid_ids()}
    flows["grid r0c0"], flows["grid r3c3"] = 0.2, 1.0
    a = ce.allocate(flows, 16 * 6)
    check(sum(a.values()) == 96 and min(a.values()) >= 1 and a["grid r0c0"] == max(a.values())
          and a["grid r3c3"] == 1, f"allocate: total 96, weakest most, perfect chunk 1 ({a['grid r0c0']})")
    check(sum(ce.allocate({g: 1.0 for g in ce.grid_ids()}, 32).values()) == 32, "allocate: all perfect -> even")


def check_run(tmp):
    be, res, log, out, log_dir, text = run(tmp, "main", rounds=4)
    rounds = [x for x in log if x["type"] == "round"]
    jobs = [x for x in log if x["type"] == "job"]
    check(res["rounds"] == 4 and len(rounds) == 4, "4 rounds ran and logged")
    check(rounds[-1]["flow_in"] > rounds[0]["flow_in"] + 0.02,
          f"flow improves: {rounds[0]['flow_in']:.3f} -> {rounds[-1]['flow_in']:.3f}")
    check(text.count("\nround ") + text.startswith("round ") == 4, "one stdout line per round")
    # paired seeds
    ok_seeds, all_seeds = True, []
    for rec in rounds:
        mine = [x for x in jobs if x["round"] == rec["round"] and not x["role"].endswith("_recheck")]
        ok_seeds &= {x["seed"] for x in mine} == set(rec["seeds"]) and len(set(rec["seeds"])) == 2
        groups = {}
        for x in mine:
            groups.setdefault((x["chunk"], x["layout"], x["role"]), set()).add(x["seed"])
        ok_seeds &= all(v == set(rec["seeds"]) for v in groups.values())
        all_seeds += rec["seeds"]
    check(ok_seeds, "every job of a round on that round's seeds, every chunk / layout on all of them")
    check(len(set(all_seeds)) == len(all_seeds), "fresh seeds each round")
    dup = len(jobs) - len({(x["round"], x["chunk"], x["seed"], x["layout"]) for x in jobs})
    check(dup == 0, f"no job run twice within a round (cache): {dup}")
    # selection and margin
    ok_sel, ok_own = True, True
    prev = be.base_keys
    for rec in rounds:
        r = rec["round"]
        mine = [x for x in jobs if x["round"] == r]
        for g in ce.grid_ids():
            inc = np.mean([x["flow"] for x in mine if x["chunk"] == g and x["role"] == "inc"])
            per = {}
            for x in mine:
                if x["chunk"] == g and x["role"] == "var":
                    per.setdefault(x["layout"], []).append(x["flow"])
            best = max((np.mean(v) for v in per.values()), default=-1.0)
            ok_sel &= (best > inc + 0.005) == (g in rec["changed"])
        cur = frozenset(tuple(k) for k in load_layout(os.path.join(log_dir, "layouts", f"r{r}.json"))[0])
        kept = set(rec["changed"]) - set(rec["reverted"])
        moved = {ce.owner(SIZE, t) for t in prev ^ cur}
        ok_own &= moved <= kept
        prev = cur
    check(ok_sel, "a chunk switches iff its best variant beats the incumbent by the margin")
    check(ok_own, "only kept changed chunks' tiles move between rounds")
    check(not (res["keys"] & POISON), "no poison tile in the final layout")
    seams = [s for rec in rounds for s in rec["seams"]]
    check(any(s["chunk"] == "grid r0c0" and s["revert"] for s in seams),
          f"seam reverts took grid r0c0 back ({sum(s['revert'] for s in seams)} reverts)")
    check(max(be.stuck_seen) > 0,
          "mutate gets the chunk's stuck shapes")
    shift_inc = [x for x in jobs if x["role"] == "shift_inc"]
    check(all(len([x for x in shift_inc if x["round"] == r["round"]]) <= 48 * 2 for r in rounds),
          "K's shifted chunks scored at most once per seed")
    # spot checks and saves
    spots = [x for x in log if x["type"] == "spot"]
    check(sorted({x["round"] for x in spots}) == [2, 4] and sum(x["who"] == "base" for x in spots) == 2
          and be.spot_jobs == [4, 2], f"spot checks rounds 2 and 4, base once ({be.spot_jobs})")
    check(len(res["spots"]) == 2 and "unstuck" in res["spots"][0], "spot records")
    check(be.saves.count(out) >= 4 and all(os.path.exists(os.path.join(log_dir, "layouts", f"r{r}.json"))
                                           for r in range(1, 5)), "saved every round (map + layouts/rN.json)")
    saved = frozenset(tuple(k) for k in load_layout(out)[0])
    with open(out) as f:
        data = json.load(f)
    check(saved == res["keys"] and data["tiling"] == "penrose" and "spot r4" in data["note"],
          "the saved map is the final layout, with the base's settings and the last spot in the note")
    check(be.opened == 1 and be.closed == 1, "pool opened and closed once")
    return rounds


def check_margin(tmp):
    be, res, log, *_ = run(tmp, "margin", rounds=2, margin=1.0, spot_every=0)
    rounds = [x for x in log if x["type"] == "round"]
    check(res["keys"] == be.base_keys and all(not r["changed"] for r in rounds), "margin 1: nothing changes")


def check_deadline(tmp, rounds):
    be = FakeBackend(tmp)
    secs = max(r["secs"] for r in rounds)
    deadline = be.clock.now + ce.DEADLINE_MARGIN + 3.3 * secs
    log_dir = os.path.join(tmp, "log deadline")
    with redirect_stdout(io.StringIO()) as buf:
        res = ce.evolve(be, os.path.join(tmp, "out deadline.json"), log_dir, 10, variants=4, seeds=2, workers=6,
                        spot_every=0, seed=3, deadline=deadline, clock=be.clock)
    log = read_log(log_dir)
    ends = [x for x in log if x["type"] == "round"]
    check(1 <= res["rounds"] < 10 and "deadline:" in buf.getvalue(),
          f"deadline stops starting rounds ({res['rounds']} of 10)")
    check(be.clock.now <= deadline, f"the last round ended before the deadline ({deadline - be.clock.now:.0f} s spare)")
    check(len(ends) == res["rounds"], "rounds logged")
    be2 = FakeBackend(tmp)
    with redirect_stdout(io.StringIO()):
        res2 = ce.evolve(be2, os.path.join(tmp, "out late.json"), os.path.join(tmp, "log late"), 3, spot_every=0,
                         deadline=be2.clock.now + 10, clock=be2.clock)
    check(res2["rounds"] == 0 and os.path.exists(os.path.join(tmp, "out late.json")),
          "no time for a round: the base is saved as is")


def check_cli(tmp):
    be = FakeBackend(tmp)
    with redirect_stdout(io.StringIO()) as buf:
        res = ce.main(["cli out", "--rounds", "2", "--variants", "2", "--seeds", "1", "--spot-every", "1",
                       "--spot-seconds", "5", "--seed", "1", "--log-dir", os.path.join(tmp, "log cli")], backend=be)
    check(res["rounds"] == 2 and os.path.exists(os.path.join(tmp, "cli out.json")) and "saved" in buf.getvalue(),
          "CLI runs through main(argv, backend)")
    with redirect_stdout(io.StringIO()) as buf:
        ce.main(["x", "--estimate"])
    check("per round" in buf.getvalue(), "--estimate prints the plan")


def main():
    tmp = tempfile.mkdtemp(prefix="chunk_evolve_check_")
    try:
        check_geometry()
        rounds = check_run(tmp)
        check_margin(tmp)
        check_deadline(tmp, rounds)
        check_cli(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(FAILS)} failures" if FAILS else "all ok")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
