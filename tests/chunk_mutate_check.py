"""Checks for chunk_mutate.py (chunk variants + assembly). Run from the repo root:

    .venv/bin/python3 tests/chunk_mutate_check.py

1. The universe: every base tile is in it, the 16 grid chunks partition it by key, the base is legal.
2. mutate_chunk: changes only tiles the chunk owns, stays legal (also chained over many chunks), keeps the
   chunk's count within +-20% of its base count, is deterministic for a given rng seed.
3. Targeted removals (share 1) land near the given stuck points, far nearer than uniform ones.
4. assemble of the base's own chunks is the base exactly; of independent variants of every chunk: legal and
   deterministic; a planted cross-seam conflict is found by legal() and fixed by dropping one tile.
5. _cycles finds an order cycle; fix_seams breaks it.
6. Time per mutate_chunk call (well under 0.1 s).
Exit status 1 on any failure.
"""
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

import chunk_mutate as cm  # noqa: E402
from chunk_test import map_path  # noqa: E402

FAILS = []


def check(ok, what):
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILS.append(what)


class Info:  # what TileUniverse reads of chunk_test.MapInfo (skips building the scene)
    def __init__(self, name):
        self.path = map_path(name)
        self.size = (1800.0, 1169.0)


def owners(uni, keys):
    return {uni.chunk_ids[uni.owner[uni.index[k]]] for k in keys}


def main():
    t = time.time()
    uni = cm.TileUniverse(Info("evolved 1 halloween"))
    print(f"universe: {len(uni.keys)} tiles, {len(uni.base)} in the base, {time.time() - t:.2f} s")
    W, H = uni.size

    # 1. universe
    check(all(k in uni.index for k in uni.base), "every base tile is in the universe")
    parts = sorted(i for c in uni.chunk_ids for i in uni.owned[c].tolist())
    check(parts == list(range(len(uni.keys))), "the 16 grid chunks partition the universe")
    inside = True
    for c in uni.chunk_ids:
        x0, y0, w, h = uni.chunk_rect[c]
        for i in uni.owned[c].tolist():
            x, y = uni.keys[i]
            inside &= x0 - 1e-9 <= x <= x0 + w + 1e-9 and y0 - 1e-9 <= y <= y0 + h + 1e-9
    check(inside, "each tile's key lies in its owner's rect")
    check(cm.legal(uni, uni.base) == [], "the base map is legal")

    # 2. mutate_chunk
    times, ok_own, ok_legal, ok_band, changed = [], True, True, True, 0
    for s in range(48):
        cid = uni.chunk_ids[s % 16]
        rng = np.random.default_rng(s)
        stuck = [[float(x), float(y), 1.0, 8.0] for x, y in
                 rng.uniform([0, 0], [W, H], (3, 2))]
        t = time.time()
        k = cm.mutate_chunk(rng, uni, uni.base, cid, stuck, 3, 3)
        times.append(time.time() - t)
        diff = k ^ uni.base
        changed += bool(diff)
        ok_own &= owners(uni, diff) <= {cid}
        ok_legal &= cm.legal(uni, k) == []
        n = len(cm.owned_keys(uni, k, cid))
        ok_band &= 0.8 * uni.base_count[cid] - 1e-9 <= n <= 1.2 * uni.base_count[cid] + 1e-9
    check(ok_own, "mutate_chunk changes only the chunk's own tiles (48 variants)")
    check(ok_legal, "every variant is legal")
    check(ok_band, "every variant keeps the chunk's count within +-20%")
    check(changed == 48, f"every variant differs from the base ({changed}/48)")
    a = cm.mutate_chunk(np.random.default_rng(5), uni, uni.base, "grid r1c1", [], 3, 3)
    b = cm.mutate_chunk(np.random.default_rng(5), uni, uni.base, "grid r1c1", [], 3, 3)
    c = cm.mutate_chunk(np.random.default_rng(6), uni, uni.base, "grid r1c1", [], 3, 3)
    check(a == b and a != c, "mutate_chunk is deterministic per rng seed, other seeds differ")
    k, rng, ok = uni.base, np.random.default_rng(99), True
    for step in range(64):  # chained: each variant the next one's parent, all chunks in turn
        k = cm.mutate_chunk(rng, uni, k, uni.chunk_ids[step % 16], [], 3, 3, targeted_share=0.5)
        ok &= cm.legal(uni, k) == []
    check(ok, f"64 chained mutations stay legal ({len(k ^ uni.base)} tiles differ from the base)")
    band = cm.mutate_chunk(np.random.default_rng(1), uni, uni.base, "grid r0c0", [], 1000, 0)
    n0 = uni.base_count["grid r0c0"]
    check(len(cm.owned_keys(uni, band, "grid r0c0")) == int(np.ceil(0.8 * n0)),
          f"a huge removal stops at 80% of the chunk's base count ({n0} -> {int(np.ceil(0.8 * n0))})")

    # 3. targeted removals
    cid = "grid r2c1"
    own = np.array([k for k in uni.base if uni.chunk_ids[uni.owner[uni.index[k]]] == cid])
    reach = 2.5 * uni.side
    near_n = (np.hypot(*(own[:, None, :] - own[None, :, :]).transpose(2, 0, 1)) <= reach).sum(axis=1)
    sp = own[int(np.argmax(near_n))]  # the stuck point: at the chunk's densest tile (3+ tiles within reach)
    stuck = [[float(sp[0]), float(sp[1]), 0.5, 9.0]] * 3

    def removed_dist(share, seeds=40):
        d = []
        for s in range(seeds):
            k = cm.mutate_chunk(np.random.default_rng(s), uni, uni.base, cid, stuck, 3, 0, targeted_share=share)
            d += [float(np.hypot(*(np.array(r) - sp))) for r in uni.base - k]
        return np.array(d)
    dt, du = removed_dist(1.0), removed_dist(0.0)
    check(np.mean(dt <= reach) == 1.0 and np.mean(dt) < 0.3 * np.mean(du),
          f"targeted removals concentrate at the stuck point (mean distance {np.mean(dt):.0f} vs uniform "
          f"{np.mean(du):.0f}, {np.mean(dt <= reach):.0%} within {reach:.0f})")
    dh = removed_dist(0.5)
    check(np.mean(dt) < np.mean(dh) < np.mean(du), f"share 0.5 lies between ({np.mean(dh):.0f})")
    k = cm.mutate_chunk(np.random.default_rng(3), uni, uni.base, cid, stuck * 4, 0, 3, targeted_share=1.0)
    added = np.array(sorted(k - uni.base))
    check(len(added) > 0 and np.all(np.hypot(*(added - sp).T) > reach),
          f"targeted additions keep {reach:.0f} away from stuck shapes ({len(added)} added)")

    # 4. assemble
    check(cm.assemble(uni, {c: uni.base for c in uni.chunk_ids}) == uni.base,
          "assemble of the base's own chunks is the base")
    check(cm.assemble(uni, {}, default=uni.base) == uni.base, "assemble with the base as default is the base")
    picks = {c: cm.mutate_chunk(np.random.default_rng([7, i]), uni, uni.base, c, [], 4, 4)
             for i, c in enumerate(uni.chunk_ids)}
    rep = []
    m1 = cm.assemble(uni, picks, report=rep)
    m2 = cm.assemble(uni, dict(reversed(list(picks.items()))))
    check(cm.legal(uni, m1) == [], f"assemble of 16 independent variants is legal ({len(rep)} dropped)")
    check(m1 == m2, "assemble is deterministic (and independent of the picks' order)")
    check(all(cm.owned_keys(uni, m1, c) <= cm.owned_keys(uni, picks[c], c) for c in uni.chunk_ids),
          "each chunk of the assembly is (a subset of) its pick's tiles")

    # a planted cross-seam conflict: two tiles, each addable alone, in neighbouring chunks, conflicting together
    mem = uni.indices(uni.base)
    addable = [i for i in range(len(uni.keys)) if i not in mem and uni.add_ok(i, mem)[0]]
    pair = None
    for i in addable:
        for j, dx, dy, g in uni.nbr[i]:
            if j in addable and uni.owner[j] != uni.owner[i] and j > i:
                ij, ji = uni.cups(i, j, dx, dy) if g <= uni.add_gap else (False, False)
                if (ij and ji) or (dx and g <= uni.gap):
                    pair = (i, j)
                    break
        if pair:
            break
    if pair is None:
        check(False, "found a planted conflict pair across a chunk border")
    else:
        i, j = pair
        ki, kj = uni.keys[i], uni.keys[j]
        ci, cj = uni.chunk_ids[uni.owner[i]], uni.chunk_ids[uni.owner[j]]
        bad = uni.base | {ki, kj}
        check(any(v["rule"] in ("cup", "seam") for v in cm.legal(uni, bad)),
              f"legal() flags the planted pair {ki} ({ci}) / {kj} ({cj})")
        rep = []
        m = cm.assemble(uni, {ci: uni.base | {ki}, cj: uni.base | {kj}}, default=uni.base, report=rep)
        check(cm.legal(uni, m) == [] and len(rep) == 1 and len(m) == len(uni.base) + 1,
              f"assemble drops exactly one of them ({rep})")

    # 5. order cycles
    check(cm._cycles({1, 2, 3, 4}, [(1, 2), (2, 3), (3, 1), (3, 4)]) == [[1, 2, 3]]
          and cm._cycles({1, 2, 3}, [(1, 2), (2, 3)]) == [], "_cycles finds a 3-cycle and nothing in a DAG")
    # three mutually close tiles off the base, each addable alone; their pairwise cups are faked into a cycle
    tri = None
    for i in addable:
        near = {j for j, dx, _, g in uni.nbr[i] if not dx and g <= uni.add_gap and j > i and j in addable}
        for j in sorted(near):
            ks = [k for k, dx, _, g in uni.nbr[j] if not dx and g <= uni.add_gap and k > j and k in near]
            if ks:
                tri = (i, j, ks[0])
                break
        if tri:
            break
    if tri is None:
        check(False, "found three mutually close addable tiles")
    else:
        saved = dict(uni._cups)
        a, b, c3 = tri
        for p, q in ((a, b), (b, c3), (c3, a)):  # p cups when tested after q: p must come first
            uni._cups[(p, q)], uni._cups[(q, p)] = (True, False), (False, True)
        lay = uni.base | {uni.keys[v] for v in tri}
        rules = sorted({v["rule"] for v in cm.legal(uni, lay)})
        fixed, dropped = cm.fix_seams(uni, lay)
        check("order" in rules and cm.legal(uni, fixed) == [] and len(dropped) == 1,
              f"legal() flags a faked order cycle ({rules}), fix_seams drops one tile ({len(dropped)})")
        uni._cups.clear()
        uni._cups.update(saved)

    # 6. timing
    print(f"mutate_chunk: mean {np.mean(times) * 1000:.1f} ms, max {np.max(times) * 1000:.1f} ms")
    check(np.mean(times) < 0.05 and np.max(times) < 0.1, "mutate_chunk well under 0.1 s per call")
    t = time.time()
    cm.assemble(uni, picks)
    print(f"assemble: {(time.time() - t) * 1000:.1f} ms")

    print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall ok")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
