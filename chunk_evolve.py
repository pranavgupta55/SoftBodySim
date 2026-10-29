"""Evolve a penrose / diamonds map chunk by chunk: per-chunk variants scored by the chunk stuck test (chunk_test.py).

    python3 chunk_evolve.py OUT_NAME --base "evolved 1 halloween" [--rounds 10] [--variants 6] [--seeds 2]
                            [--workers 6] [--deadline "YYYY-MM-DD HH:MM"] [--spot-every 2] [--spot-seconds 120]
                            [--log-dir DIR] [--estimate]

A layout is a frozenset of tile keys (tile centres, diamonds_map.tile_key). The world splits into the 4 x 4 "grid"
chunks of chunk_test.chunk_rects; a tile belongs to the grid chunk whose rect holds its key, and a chunk variant
changes only the tiles its chunk owns. The three half-chunk shifted grids straddle the seams and only score them.

One round, incumbent layout K, S fresh seeds shared by every job of the round (paired):
  1. Score the 16 grid chunks of K on the S seeds. A chunk's metric (--metric) is its mean `free` over the seeds
     (1 - the share of shapes settled in the chunk at the end of the 5 s test, chunk_test's default; or `flow`).
     score is logged too.
  2. Make ~16 x V chunk variants (chunk_mutate.mutate_chunk, steered by the chunk's stuck shapes on all seeds:
     the settled ones for `free`), allotted in proportion to 1 - metric (weaker chunks get more, every chunk at
     least one); score each variant's chunk on the same seeds.
  3. Per chunk, the best variant replaces the incumbent's tiles only if its metric beats the incumbent's by more than
     --margin.
  4. Assemble K' (chunk_mutate.assemble, which drops border tiles that break the placement rules). Score the
     shifted chunks whose tiles differ between K and K' on the same seeds, for both. A changed grid chunk's seam
     damage is the summed metric change of those shifted chunks over it; a chunk past -(--seam-threshold) is re-scored
     on --recheck-seeds fresh seeds and judged on all of them; if still past it, the worst is reverted,
     K' re-assembled and re-scored. (One chunk's flow varies ~0.02 between seeds, free ~0.027, so the threshold sits
     at ~2x.)
  5. Save K' to maps/<base's folder>/OUT_NAME.json (the base's spawn, tiling, clusters and lighting kept) and to
     LOG_DIR/layouts/rN.json.
  6. Every --spot-every rounds and after the last round: one whole-map run of K' per spot seed (chunk_test.
     whole_map_score, --spot-seconds), against the base map on the same seeds (fixed for the whole run, so the base
     runs once and every spot check is comparable).

A chunk test depends only on the tiles overlapping its rect, so results are cached within a round by (chunk, seed,
the layout's keys within one tile edge of the rect): K's shifted chunks are scored once, unchanged ones not at all.

Output: one line per round on stdout (mean chunk flow before / after, worst chunk, chunks changed, seam reverts,
seconds); every job, round and spot check to LOG_DIR/runs.jsonl (default <tmp>/chunk_evolve/OUT_NAME).

--rounds is a maximum with --deadline: a round only starts if the longest round so far (before the first, the
--estimate plan at --test-wall s per chunk test) plus a margin ends before the deadline; a spot check is skipped if it
would overrun. The map is saved after every round, so stopping early loses nothing.

Uses a persistent pool of `workers` processes (chunk_test.ChunkPool); launch it at low priority, e.g.
`nice -n 10`, with OMP_NUM_THREADS=1.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from chunk_test import CHUNKS, SLOW_SPEED, chunk_rects, map_path  # noqa: E402

TEST_WALL = 0.9  # s: one 5 s chunk test on one core (evolved 1 halloween, measured)
ALL_GRIDS = ("grid", "shift_x", "shift_y", "shift_xy")
DEADLINE_MARGIN = 120.0  # s kept spare before --deadline (plus 10% on the round estimate)
SPOT_WALL_PER_SIM = 0.45  # s wall per sim second of a whole-map run (measured 0.40), until one is timed


# ---------------------------------------------------------------------------------------------------------------
# The adapter: the one place that names chunk_test's / chunk_mutate's evolution API (run in other modules). Tests
# pass a fake with the same attributes and methods to evolve() instead.
class MapBackend:
    """The base map (size, keys, spawn / tiling / lighting to save with) and the scoring and mutation calls."""

    def __init__(self, base):
        import chunk_test
        import scene
        from diamonds_map import layout_lighting, layout_spawn, layout_tiling, load_layout
        self.path = map_path(base)
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.folder = os.path.dirname(self.path)
        self.info = chunk_test.MapInfo(self.path)
        self.size = self.info.size
        keys, _ = load_layout(self.path)
        self.base_keys = frozenset(tuple(float(v) for v in k) for k in keys)
        self.tiling = layout_tiling(self.path)  # incl. "clusters" (the rules mutate_chunk's adds follow)
        self.spawn = layout_spawn(self.path) or scene.diamonds_spawn(self.path)
        self.lighting = layout_lighting(self.path) or None
        # a tile's corners lie within one edge length of its key, so keys this far outside a rect can't touch it
        self.pad = float(self.tiling.get("edge", 30.0))
        from chunk_mutate import TileUniverse
        self.uni = TileUniverse(self.info)

    def pool(self, workers):
        """A context manager with .score([(keys, chunk_id, seed)]) -> [result dict] in order."""
        from chunk_test import ChunkPool
        return ChunkPool(self.path, workers)

    def mutate(self, rng, keys, chunk_id, stuck, n_remove, n_add, share):
        from chunk_mutate import mutate_chunk
        return frozenset(mutate_chunk(rng, self.uni, keys, chunk_id, stuck, n_remove, n_add, targeted_share=share))

    def assemble(self, picks):
        from chunk_mutate import assemble
        out = assemble(self.uni, picks)
        return frozenset(out[0] if isinstance(out, tuple) else out)  # (keys, dropped) or keys

    def spot(self, jobs, seconds, workers):
        """[whole_map_score dict] of jobs [(keys, seed)], in parallel processes (the chunk pool idles meanwhile)."""
        from chunk_test import whole_map_score
        if workers <= 1 or len(jobs) <= 1:
            return [whole_map_score(self.path, k, seconds, s) for k, s in jobs]
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(whole_map_score, self.path, k, seconds, s) for k, s in jobs]
            return [f.result() for f in futures]

    def save(self, path, keys, note):
        from diamonds_map import save_layout
        save_layout(path, None, self.size, note=note, spawn=self.spawn, keys=sorted(keys),
                    tiling=self.tiling if self.tiling.get("tiling") != "diamonds" else None, lighting=self.lighting)


# ---------------------------------------------------------------------------------------------------------------
# Geometry (wrap-aware: the world loops)
def grid_ids():
    return [f"grid r{r}c{c}" for r in range(CHUNKS) for c in range(CHUNKS)]


def owner(size, key):
    """The id of the grid chunk owning a tile key."""
    W, H = size
    c = min(CHUNKS - 1, max(0, int(key[0] % W / W * CHUNKS)))
    r = min(CHUNKS - 1, max(0, int(key[1] % H / H * CHUNKS)))
    return f"grid r{r}c{c}"


def _span_overlap(a0, aw, b0, bw, period):
    return (b0 - a0) % period < aw or (a0 - b0) % period < bw


def overlaps(size, a, b):
    """Do rects a and b (x0, y0, w, h) share area, across the wrap?"""
    return _span_overlap(a[0], a[2], b[0], b[2], size[0]) and _span_overlap(a[1], a[3], b[1], b[3], size[1])


def window(size, rect, keys, pad):
    """The keys within `pad` of the rect (across the wrap): every tile that can overlap it."""
    W, H = size
    x0, y0, w, h = rect
    out = []
    for k in keys:
        dx, dy = (k[0] - x0) % W, (k[1] - y0) % H
        if (dx <= w + pad or dx >= W - pad) and (dy <= h + pad or dy >= H - pad):
            out.append(k)
    return frozenset(out)


def layout_id(keys):
    return hashlib.sha1(repr(sorted(keys)).encode()).hexdigest()[:10]


def allocate(flows, total):
    """{chunk: variants}: at least one each, the rest in proportion to 1 - flow (largest remainder; even split if
    every chunk is perfect)."""
    ids = list(flows)
    total = max(total, len(ids))
    w = np.array([max(0.0, 1.0 - flows[c]) for c in ids])
    w = np.ones(len(ids)) if w.sum() <= 0 else w
    share = (total - len(ids)) * w / w.sum()
    n = np.floor(share).astype(int)
    for i in np.argsort(-(share - n), kind="stable")[: total - len(ids) - int(n.sum())]:
        n[i] += 1
    return {c: 1 + int(k) for c, k in zip(ids, n)}


def mean_of(results, key="flow"):
    return float(np.mean([r[key] for r in results]))


def plan_seconds(variants, seeds, workers, test_wall=TEST_WALL, affected=48, reverts=1):
    """Estimated wall s of one round: waves of chunk tests (incumbent grid, variants, K and K' shifted chunks
    (`affected` of the 48 changed), and per seam revert the 8 shifted chunks over the reverted chunk), each wave
    test_wall long. Mutation and assembly time in this process are not counted."""
    def waves(n):
        return math.ceil(n / workers)
    n = CHUNKS * CHUNKS
    return test_wall * (waves(n * seeds) + waves(n * variants * seeds) + waves(2 * affected * seeds)
                        + reverts * waves(8 * seeds))


# ---------------------------------------------------------------------------------------------------------------
class Scorer:
    """One round's chunk scoring on its seeds through the pool, cached by (chunk, seed, the layout's window)."""

    def __init__(self, pool, size, pad, seeds, log, round_no):
        self.pool, self.size, self.pad, self.seeds, self.log, self.round = pool, size, pad, seeds, log, round_no
        self.rects = {c["id"]: c["rect"] for c in chunk_rects(size, grids=ALL_GRIDS)}
        self.cache = {}
        self.jobs = 0
        self.wall = 0.0

    def sig(self, keys, chunk_id):
        return window(self.size, self.rects[chunk_id], keys, self.pad)

    def score(self, reqs):
        """reqs [(keys, chunk_id, role, extra log fields)] -> [[result per seed]] in order."""
        todo, queued, keyed = [], set(), []
        for keys, cid, role, extra in reqs:
            sig = self.sig(keys, cid)
            row = []
            for s in self.seeds:
                ck = (cid, s, sig)
                row.append(ck)
                if ck not in self.cache and ck not in queued:
                    queued.add(ck)
                    todo.append((ck, (keys, cid, s), dict(extra, role=role, layout=layout_id(keys))))
            keyed.append(row)
        if todo:
            results = self.pool.score([job for _, job, _ in todo])
            for (ck, job, meta), res in zip(todo, results):
                self.cache[ck] = res
                self.jobs += 1
                self.wall += float(res.get("wall", 0.0))
                self.log(dict(res, type="job", round=self.round, chunk=job[1], seed=job[2], **meta))
        return [[self.cache[ck] for ck in row] for row in keyed]


def evolve(be, out_path, log_dir, rounds, variants=6, seeds=2, workers=6, margin=0.005, seam_threshold=0.04,
           recheck_seeds=2, n_remove=3, n_add=3, share=0.5, seed=0, deadline=None, spot_every=2, spot_seconds=120.0,
           spot_seeds=2, test_wall=TEST_WALL, metric="free", clock=time.time, say=print):
    """Run up to `rounds` rounds from be.base_keys (see the module doc); returns {"keys", "rounds", "history",
    "spots"}. `be`: a MapBackend or a fake with the same members."""
    os.makedirs(os.path.join(log_dir, "layouts"), exist_ok=True)
    log_path = os.path.join(log_dir, "runs.jsonl")
    open(log_path, "w").close()

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    size, pad = be.size, be.pad
    grids = grid_ids()
    shifted = [c["id"] for c in chunk_rects(size, grids=("shift_x", "shift_y", "shift_xy"))]
    rects = {c["id"]: c["rect"] for c in chunk_rects(size, grids=ALL_GRIDS)}
    over = {g: [s for s in shifted if overlaps(size, rects[g], rects[s])] for g in grids}
    rng = np.random.default_rng([int(seed), 23])  # the search's own draws: round seeds, mutations
    spot_rng = np.random.default_rng([int(seed), 29])
    spot_seed_list = [int(spot_rng.integers(1, 1 << 30)) for _ in range(spot_seeds)]
    base = K = frozenset(be.base_keys)
    t0 = clock()
    history, spots, round_secs, base_spot, spot_secs = [], [], [], None, None
    say(f"chunk_evolve -> {out_path}: base {len(base)} tiles, up to {rounds} rounds x ~{16 * variants} variants, "
        f"{seeds} seeds, metric {metric}, margin {margin:g}, seam threshold {seam_threshold:g}, {workers} workers"
        f"{', deadline in %.0f s' % (deadline - t0) if deadline else ''}; log {log_path}")
    log({"type": "start", "out": out_path, "tiles": len(base), "rounds": rounds, "variants": variants,
         "seeds": seeds, "metric": metric, "margin": margin, "seam_threshold": seam_threshold, "remove": n_remove,
         "add": n_add, "share": share, "seed": seed, "spot_seeds": spot_seed_list})

    def note(r):
        last = spots[-1] if spots else None
        sp = (f"; spot r{last['round']}: {last['score']:.4f} vs base {last['base_score']:.4f} ({spot_seconds:g} s, "
              f"{spot_seeds} seeds)") if last else ""
        return (f"{os.path.splitext(os.path.basename(out_path))[0]} ({time.strftime('%Y-%m-%d')}): chunk_evolve from "
                f"{getattr(be, 'name', 'base')}, {r} rounds of ~{16 * variants} chunk variants (remove {n_remove} / "
                f"add {n_add}) x {seeds} paired seeds, {metric} margin {margin:g}, seam threshold {seam_threshold:g}"
                f"{sp}")

    def spot_check(r):
        nonlocal base_spot, spot_secs
        est = spot_secs if spot_secs is not None else (
            spot_seconds * SPOT_WALL_PER_SIM * math.ceil(spot_seeds * (1 if base_spot else 2) / max(workers, 1)))
        if deadline is not None and clock() + est > deadline:
            say(f"spot r{r}: skipped, ~{est:.0f} s left would overrun the deadline")
            return
        ts = clock()
        jobs = [(K, s) for s in spot_seed_list] + ([] if base_spot else [(base, s) for s in spot_seed_list])
        res = be.spot(jobs, spot_seconds, workers)
        mine, theirs = res[:spot_seeds], res[spot_seeds:]
        with_base = base_spot is None
        if with_base:
            base_spot = theirs
            for x, s in zip(theirs, spot_seed_list):
                log(dict({k: v for k, v in x.items() if k != "keys"}, type="spot", round=r, who="base", seed=s))
        for x, s in zip(mine, spot_seed_list):
            log(dict({k: v for k, v in x.items() if k != "keys"}, type="spot", round=r, who="K", seed=s,
                     layout=layout_id(K)))
        if not with_base:  # the later checks run K only: time those
            spot_secs = clock() - ts
        rec = {"round": r, "score": mean_of(mine, "score"), "base_score": mean_of(base_spot, "score")}
        for k in ("unstuck", "spread"):
            if all(k in x for x in mine + base_spot):
                rec[k], rec["base_" + k] = mean_of(mine, k), mean_of(base_spot, k)
        spots.append(rec)
        say(f"spot r{r}: whole map {rec['score']:.4f} vs base {rec['base_score']:.4f}"
            + (f" (unstuck {rec['unstuck']:.3f} vs {rec['base_unstuck']:.3f})" if "unstuck" in rec else "")
            + f", {clock() - ts:.0f} s")
        be.save(out_path, K, note(r))

    done = 0
    with be.pool(workers) as pool:
        for r in range(1, rounds + 1):
            if deadline is not None:
                est = max(round_secs) if round_secs else plan_seconds(variants, seeds, workers, test_wall)
                need = 1.1 * est + DEADLINE_MARGIN
                if clock() + need > deadline:
                    say(f"deadline: a round needs ~{need:.0f} s, {deadline - clock():.0f} s left: stopping after "
                        f"{done} rounds")
                    break
            t_round = clock()
            round_seeds = [int(rng.integers(1, 1 << 30)) for _ in range(seeds)]
            sc = Scorer(pool, size, pad, round_seeds, log, r)
            # 1. the incumbent's grid chunks
            inc = dict(zip(grids, sc.score([(K, g, "inc", {}) for g in grids])))
            flow = {g: mean_of(inc[g], metric) for g in grids}
            # 2. variants, more for the weaker chunks
            alloc = allocate(flow, len(grids) * variants)
            var = {}
            for g in grids:
                stuck = [s for res in inc[g] for s in res.get("stuck", []) if metric != "free" or s[2] < SLOW_SPEED]
                var[g] = [be.mutate(rng, K, g, stuck, n_remove, n_add, share) for _ in range(alloc[g])]
            reqs = [(v, g, "var", {"variant": i}) for g in grids for i, v in enumerate(var[g])]
            vres = sc.score(reqs)
            # 3. per chunk: the best variant, if it beats the incumbent by the margin
            picks, gain = {}, {}
            k = 0
            for g in grids:
                best = None
                for v in var[g]:
                    f, s = mean_of(vres[k], metric), mean_of(vres[k], "score")
                    k += 1
                    if best is None or (f, s) > best[0]:
                        best = ((f, s), v)
                if best is not None and best[1] != K and best[0][0] > flow[g] + margin:
                    picks[g], gain[g] = best[1], best[0][0] - flow[g]
            # 4. assemble, then revert changed chunks whose seams got worse in total (re-checked on more seeds)
            changed = sorted(picks)
            reverted, seam_log, Kn = [], [], K
            recheck = Scorer(pool, size, pad, [int(rng.integers(1, 1 << 30)) for _ in range(recheck_seeds)], log, r) \
                if recheck_seeds else None

            def deltas(scorer, Kn, chunks, suffix=""):
                res = scorer.score([(K, s, "shift_inc" + suffix, {}) for s in chunks]
                                   + [(Kn, s, "shift_new" + suffix, {"reverted": list(reverted)}) for s in chunks])
                n = len(chunks)
                return {s: mean_of(res[n + i], metric) - mean_of(res[i], metric) for i, s in enumerate(chunks)}

            while True:
                live = [g for g in changed if g not in reverted]
                if not live:
                    Kn = K
                    break
                Kn = be.assemble({g: picks.get(g, K) if g not in reverted else K for g in grids})
                affected = [s for s in shifted if sc.sig(Kn, s) != sc.sig(K, s)]
                delta = deltas(sc, Kn, affected)
                # a changed chunk's seam damage: the summed change of the affected shifted chunks over it
                damage = {g: sum(delta[s] for s in over[g] if s in delta) for g in live}
                victim = None
                for d, g in sorted((d, g) for g, d in damage.items() if d < -seam_threshold):
                    mine = [s for s in over[g] if s in delta]
                    if recheck is not None:  # suspect: rescore its seams on fresh seeds, judge on all of them
                        more = deltas(recheck, Kn, mine, "_recheck")
                        w0, w1 = len(sc.seeds), len(recheck.seeds)
                        d = sum((delta[s] * w0 + more[s] * w1) / (w0 + w1) for s in mine)
                    seam_log.append({"chunk": g, "damage": round(sum(delta[s] for s in mine), 4),
                                     "rechecked": round(d, 4), "shifted": mine, "revert": d < -seam_threshold})
                    if d < -seam_threshold:
                        victim = g
                        break
                if victim is None:
                    break
                reverted.append(victim)
            kept = [g for g in changed if g not in reverted]
            expected = set(K)
            for g in kept:  # the tiles the picks asked for: dropped ones went to fix the seams
                expected = {t for t in expected if owner(size, t) != g} | {t for t in picks[g] if owner(size, t) == g}
            dropped = sorted(expected - set(Kn))
            K = Kn
            out_flow = {g: (flow[g] + gain[g] if g in kept else flow[g]) for g in grids}
            worst = min(grids, key=lambda g: out_flow[g])
            secs = clock() - t_round
            round_secs.append(secs)
            done = r
            rec = {"type": "round", "round": r, "seeds": round_seeds, "flow_in": float(np.mean(list(flow.values()))),
                   "flow_out_est": float(np.mean(list(out_flow.values()))), "worst": worst,
                   "worst_flow": out_flow[worst], "score_in": float(np.mean([mean_of(inc[g], "score") for g in grids])),
                   "alloc": alloc, "changed": changed, "reverted": reverted, "seams": seam_log,
                   "dropped": [list(t) for t in dropped], "tiles": len(K), "jobs": sc.jobs, "secs": round(secs, 1),
                   "layout": layout_id(K)}
            log(rec)
            history.append(rec)
            be.save(out_path, K, note(r))
            be.save(os.path.join(log_dir, "layouts", f"r{r}.json"), K, note(r))
            say(f"round {r}: flow {rec['flow_in']:.4f} -> ~{rec['flow_out_est']:.4f}, worst {worst} "
                f"{out_flow[worst]:.3f}, {len(kept)}/{len(changed)} chunks changed ({len(reverted)} seam reverts"
                f"{f', {len(dropped)} border tiles dropped' if dropped else ''}), {sc.jobs} tests, {len(K)} tiles, "
                f"{secs:.0f} s (total {clock() - t0:.0f} s)")
            if spot_every and spot_seconds > 0 and (r % spot_every == 0 or r == rounds):
                spot_check(r)
    if done and spot_every and spot_seconds > 0 and (not spots or spots[-1]["round"] != done):
        spot_check(done)  # stopped early by the deadline: the saved map's spot check, if it fits
    if not done:
        be.save(out_path, K, note(0))
    say(f"saved {out_path} after {done} rounds ({len(K)} tiles) in {clock() - t0:.0f} s")
    return {"keys": K, "rounds": done, "history": history, "spots": spots}


def parse_deadline(text):
    """'YYYY-MM-DD HH:MM' local time -> epoch seconds (None for None)."""
    return None if text is None else time.mktime(time.strptime(text, "%Y-%m-%d %H:%M"))


def main(argv=None, backend=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name", help="the new map's name (saved next to the base: maps/<folder>/NAME.json)")
    p.add_argument("--base", default="evolved 1 halloween", help="the starting map (a name in maps/penrose or a path)")
    p.add_argument("--rounds", type=int, default=10, help="rounds (a maximum with --deadline)")
    p.add_argument("--variants", type=int, default=6, help="chunk variants per grid chunk per round, on average")
    p.add_argument("--seeds", type=int, default=10, help="paired seeds per round (fresh each round)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--metric", choices=("free", "flow"), default="free",
                   help="chunk metric: free (1 - settled share at the end) or flow")
    p.add_argument("--margin", type=float, default=0.005, help="a variant must beat the chunk's metric by more")
    p.add_argument("--seam-threshold", type=float, default=0.04,
                   help="a changed chunk is reverted if the shifted chunks over it lose more than this in total")
    p.add_argument("--recheck-seeds", type=int, default=2,
                   help="fresh seeds a suspect seam is re-scored on before reverting (0: no re-check)")
    p.add_argument("--remove", type=int, default=3, help="tiles removed per chunk variant")
    p.add_argument("--add", type=int, default=3, help="tiles added per chunk variant")
    p.add_argument("--targeted-share", type=float, default=0.5, help="share of stuck-targeted (vs uniform) draws")
    p.add_argument("--seed", type=int, default=int(time.time()) % 100000, help="the search's seed (default: clock)")
    p.add_argument("--deadline", default=None, help='"YYYY-MM-DD HH:MM" local time: stop starting rounds before it')
    p.add_argument("--spot-every", type=int, default=2, help="whole-map spot check every N rounds (0: never)")
    p.add_argument("--spot-seconds", type=float, default=120.0)
    p.add_argument("--spot-seeds", type=int, default=2)
    p.add_argument("--test-wall", type=float, default=TEST_WALL,
                   help="s per chunk test on one core, for --estimate and the first round's deadline check")
    p.add_argument("--estimate", action="store_true", help="print the per-round wall time estimate and exit")
    p.add_argument("--log-dir", default=None, help="default: <tmp>/chunk_evolve/NAME")
    args = p.parse_args(argv)
    if args.estimate:
        for aff, rev in ((48, 0), (48, 2), (32, 1)):
            secs = plan_seconds(args.variants, args.seeds, args.workers, args.test_wall, aff, rev)
            print(f"variants {args.variants}, seeds {args.seeds}, {args.workers} workers, {aff} shifted chunks "
                  f"affected, {rev} reverts: ~{secs:.0f} s per round (+ mutation time)")
        spot = args.spot_seconds * SPOT_WALL_PER_SIM
        print(f"spot check: ~{spot * math.ceil(args.spot_seeds / args.workers):.0f} s (the first, with the base: "
              f"~{spot * math.ceil(2 * args.spot_seeds / args.workers):.0f} s)")
        return None
    be = backend or MapBackend(args.base)
    folder = getattr(be, "folder", os.path.join(REPO, "maps", "penrose"))
    out = os.path.join(folder, args.name + ".json")
    if os.path.abspath(out) == os.path.abspath(getattr(be, "path", "")):
        p.error("OUT_NAME is the base map: pick a new name")
    log_dir = args.log_dir or os.path.join(tempfile.gettempdir(), "chunk_evolve", args.name)
    return evolve(be, out, log_dir, args.rounds, variants=args.variants, seeds=args.seeds, workers=args.workers,
                  margin=args.margin, seam_threshold=args.seam_threshold,
                  recheck_seeds=args.recheck_seeds, n_remove=args.remove, n_add=args.add,
                  share=args.targeted_share, seed=args.seed, deadline=parse_deadline(args.deadline),
                  spot_every=args.spot_every, spot_seconds=args.spot_seconds, spot_seeds=args.spot_seeds,
                  test_wall=args.test_wall, metric=args.metric)


if __name__ == "__main__":
    main()
