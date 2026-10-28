"""Chunk variants of a penrose / diamonds layout, and their assembly into one map (for chunk_evolve.py).

A layout is a frozenset of tile keys ((x, y) centers rounded as diamonds_map.tile_key, the keys a map JSON stores
under "tiles"). TileUniverse(info) holds every tile of the map's tiling that lies wholly in the world (the candidates,
as pick_diamonds sees them), the grid chunk owning each (the "grid rRcC" chunk of chunk_test.chunk_rects whose rect
holds its key: the 16 grid chunks partition the tiles) and the placement rules' data.

The rules are pick_diamonds' for a variant's added tiles (evolve_layout's remove / add on a trusted parent layout),
pair cup rule only (the "outline" rule isn't supported here):
  * cup: diamonds_map.forms_cup(new, existing) with the random fill's gap (cup_gap x clusters' "random_gap"),
    across the wraps. forms_cup isn't symmetric and pick_diamonds tests each new tile against the ones already
    there (trusting the parent layout), so legality depends on the order tiles came in. Here a pair of tiles of the
    base layout (the map's own: they coexisted in a legal layout) is trusted; any other pair close enough must not
    cup both ways ("cup"), and a pair that cups one way only fixes which of the two came first: the layout is legal
    when some order satisfies all of them (no cycle: "order"), i.e. pick_diamonds would accept its tiles added in
    that order. mutate_chunk adds a tile exactly as pick_diamonds does (tested after all the others).
  * seam: no new tile links (corners within cup_gap) to another across the left / right wrap.
  * width: a cluster (tiles linked by corners within cup_gap) holding a pair that isn't trusted is at most
    scene.DIAMOND_CHAIN_WIDTH wide.
  * min slope: the universe holds only steep_tiles of the map's min_slope.
The pair rule is monotone: removing tiles never breaks it. Adds are random-fill tiles only (no archetype cluster
growth: a chunk variant changes a handful of tiles).

mutate_chunk(rng, uni, keys, chunk_id, stuck, n_remove, n_add): a new layout differing from keys only in tiles owned
by chunk_id: removals weighted toward stuck shapes, additions toward sparse stuck-free spots (evolve_layout's
targeted_orders weights), `targeted_share` of the draws targeted. assemble(uni, picks): each grid chunk's owned
tiles from its picked layout, then fix_seams drops the fewest tiles that make the union legal.
"""
import itertools
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from chunk_test import CHUNKS, chunk_rects  # noqa: E402

COUNT_BAND = 0.2  # mutate_chunk keeps a chunk's tile count within this share of its base count (None: unbounded)
EXACT_COVER = 16  # fix_seams: conflict components up to this many tiles get an exact minimum vertex cover


def _key(k):
    return (round(float(k[0]), 1), round(float(k[1]), 1))


class TileUniverse:
    """The map's candidate tiles and rule data. info: chunk_test.MapInfo (or anything with .path and .size).
    base: the trusted layout (default: the map's own tiles).

    Attributes: size, side (tile edge), keys [(x, y)], index {key: i}, corners (n, 2, k), centers (n, 2), owner (n,)
    grid chunk index, chunk_ids [16 "grid rRcC"], owned {chunk_id: tile indices}, base (frozenset of keys),
    base_count {chunk_id: base tiles owned}, gap (link / cluster gap), add_gap (cup gap of added tiles), max_width,
    nbr[i]: [(j, dx, dy, corner gap)] of tiles whose corners come within add_gap of i's (j shifted by dx, dy)."""

    def __init__(self, info, base=None):
        import scene
        from diamonds_map import cluster_settings, load_layout, penrose_tiles, steep_tiles, lattice_diamonds
        self.size = W, H = tuple(float(v) for v in info.size)
        tiling = scene._layout_tiling(info.path)
        self.tiling = tiling
        cups = tiling.get("cups", scene.DIAMONDS_CUPS)
        if cups != "pair":
            raise NotImplementedError(f"chunk_mutate: cup rule {cups!r} (only 'pair')")
        if tiling["tiling"] == "penrose":
            self.side = float(tiling["edge"])
            tiles = steep_tiles(penrose_tiles(W, H, self.side, seed=int(tiling["pentagrid_seed"]),
                                              rotation=float(tiling["rotation"])), float(tiling["min_slope"]))
        else:
            self.side = scene.DIAMOND_SIDE
            tiles = lattice_diamonds(W, H, self.side, seed=scene.DIAMONDS_SEED, y0=self.side)
        eps = 1e-6
        inside = [t[0] for t in tiles if t[0][0].min() >= -eps and t[0][0].max() <= W + eps
                  and t[0][1].min() >= -eps and t[0][1].max() <= H + eps]
        self.corners = np.array(inside, dtype=float)  # (n, 2, k)
        self.centers = self.corners.mean(axis=2)
        # tile_key of each (the same means, rounded the same way)
        self.keys = [(round(x, 1), round(y, 1)) for x, y in zip(self.corners[:, 0].mean(axis=1).tolist(),
                                                                self.corners[:, 1].mean(axis=1).tolist())]
        self.index = {k: i for i, k in enumerate(self.keys)}
        n = len(self.keys)
        # ownership: the grid chunk whose rect holds the key
        cw, ch = W / CHUNKS, H / CHUNKS
        kx = np.array([k[0] for k in self.keys])
        ky = np.array([k[1] for k in self.keys])
        col = np.clip(np.floor(kx / cw).astype(int), 0, CHUNKS - 1)
        row = np.clip(np.floor(ky / ch).astype(int), 0, CHUNKS - 1)
        self.owner = row * CHUNKS + col
        self.chunk_ids = [c["id"] for c in chunk_rects(self.size, grids=("grid",))]
        self.chunk_index = {c: i for i, c in enumerate(self.chunk_ids)}
        self.chunk_rect = {c["id"]: c["rect"] for c in chunk_rects(self.size, grids=("grid",))}
        self.owned = {c: np.flatnonzero(self.owner == i) for i, c in enumerate(self.chunk_ids)}
        # rules
        self.gap = float(tiling.get("cup_gap", scene.CUP_GAP))
        settings = cluster_settings(tiling.get("clusters"))
        random_gap = float(tiling["clusters"].get("random_gap", 1.0)) if isinstance(tiling.get("clusters"), dict) \
            else 1.0
        self.add_gap = self.gap * random_gap if settings is not None else self.gap
        self.max_width = scene.DIAMOND_CHAIN_WIDTH
        self.reach = 2 * self.side + max(self.gap, self.add_gap)  # centers farther apart can't cup or link
        self._neighbours()
        self._cups = {}
        # trusted base
        if base is None:
            base, _ = load_layout(info.path)
        self.base = frozenset(_key(k) for k in base)
        missing = [k for k in self.base if k not in self.index]
        if missing:
            raise ValueError(f"{len(missing)} base tiles aren't in the tiling, e.g. {missing[:3]}")
        self.in_base = np.zeros(n, dtype=bool)
        self.in_base[[self.index[k] for k in self.base]] = True
        self.base_count = {c: int(self.in_base[self.owned[c]].sum()) for c in self.chunk_ids}
        # the base's clusters: a cluster of trusted tiles only is a subset of one of these (no wider)
        self.base_cluster = np.full(n, -1)
        for cid, comp in enumerate(self._components(set(np.flatnonzero(self.in_base).tolist()))):
            self.base_cluster[list(comp)] = cid

    def _neighbours(self):
        """nbr[i]: [(j, dx, dy, corner gap)] for every tile j (shifted by (dx, dy) across the wraps) whose corners
        come within max(gap, add_gap) of tile i's."""
        W, H = self.size
        n = len(self.keys)
        lim = max(self.gap, self.add_gap)
        self.nbr = [[] for _ in range(n)]
        wh = np.array([W, H])
        for a in range(0, n, 512):
            d = self.centers[a:a + 512, None, :] - self.centers[None, :, :]
            d -= wh * np.round(d / wh)
            close = (np.abs(d[..., 0]) <= self.reach) & (np.abs(d[..., 1]) <= self.reach)
            ii, jj = np.nonzero(close)
            ii = ii + a
            keep = ii != jj
            ii, jj = ii[keep], jj[keep]
            # j's shift so that it lies next to i: c_j + s = c_i - d  =>  s = c_i - d - c_j
            s = self.centers[ii] - d[ii - a, jj] - self.centers[jj]
            s = np.round(s / wh) * wh
            q = self.corners[jj] + s[:, :, None]
            p = self.corners[ii]
            cg = np.hypot(p[:, 0, :, None] - q[:, 0, None, :], p[:, 1, :, None] - q[:, 1, None, :]).min(axis=(1, 2))
            k = cg <= lim
            for i, j, dx, dy, g in zip(ii[k].tolist(), jj[k].tolist(), s[k, 0].tolist(), s[k, 1].tolist(),
                                       cg[k].tolist()):
                self.nbr[i].append((j, dx, dy, g))

    def cups(self, i, j, dx, dy):
        """(forms_cup(i, j shifted), forms_cup(j shifted, i)) at add_gap, cached."""
        from diamonds_map import forms_cup
        c = self._cups.get((i, j))
        if c is None:
            p, q = self.corners[i], self.corners[j] + np.array([[dx], [dy]])
            c = (forms_cup(p, q, self.side, self.add_gap), forms_cup(q, p, self.side, self.add_gap))
            self._cups[(i, j)] = c
            self._cups[(j, i)] = c[::-1]
        return c

    def trusted(self, i, j):
        return bool(self.in_base[i] and self.in_base[j])

    def links(self, i, members):
        """The tiles of `members` (a set of indices) linked to i (corner gap within gap, no left / right wrap)."""
        return [j for j, dx, _, g in self.nbr[i] if g <= self.gap and not dx and j in members]

    def _components(self, members):
        seen, out = set(), []
        for s in sorted(members):
            if s in seen:
                continue
            comp, todo = [s], [s]
            seen.add(s)
            while todo:
                for j in self.links(todo.pop(), members):
                    if j not in seen:
                        seen.add(j)
                        comp.append(j)
                        todo.append(j)
            out.append(comp)
        return out

    def cluster(self, i, members):
        """Tile i's cluster (indices) in `members` (a set, i included or not)."""
        members = members | {i}
        comp, todo, seen = [i], [i], {i}
        while todo:
            for j in self.links(todo.pop(), members):
                if j not in seen:
                    seen.add(j)
                    comp.append(j)
                    todo.append(j)
        return comp

    def width(self, comp):
        xs = self.corners[comp, 0]
        return float(xs.max() - xs.min())

    def width_ok(self, comp):
        """A cluster passes the width rule: at most max_width, or within one base cluster (trusted)."""
        if self.width(comp) <= self.max_width + 1e-9:
            return True
        ids = set(self.base_cluster[comp].tolist())
        return len(ids) == 1 and -1 not in ids

    def indices(self, keys):
        """Layout keys -> set of indices (KeyError for a key not in the tiling)."""
        return {self.index[_key(k)] for k in keys}

    def add_ok(self, i, members):
        """Can tile i join the layout `members` (indices, i not in it) by the rules, as pick_diamonds adds a tile
        to a trusted layout (tested after all the others)? (ok, reason or None)"""
        for j, dx, dy, g in self.nbr[i]:
            if j not in members or self.trusted(i, j):
                continue
            if g <= self.add_gap and self.cups(i, j, dx, dy)[0]:  # i tested after j, as pick_diamonds adds it
                return False, "cup"
            if dx and g <= self.gap:
                return False, "seam"
        if not self.width_ok(self.cluster(i, members)):
            return False, "width"
        return True, None


def _pairs(uni, members):
    """(conflicts [(i, j, "cup" | "seam")], order [(a, b): a must be placed before b]) of the untrusted tile pairs
    of `members` (indices): a cup both ways or a left / right seam link is a conflict; a cup one way only fixes
    their order (the tile tested later must be the one that doesn't cup)."""
    bad, order = [], []
    for i in sorted(members):
        for j, dx, dy, g in uni.nbr[i]:
            if j <= i or j not in members or uni.trusted(i, j):
                continue
            ij, ji = uni.cups(i, j, dx, dy) if g <= uni.add_gap else (False, False)
            if ij and ji:
                bad.append((i, j, "cup"))
            elif dx and g <= uni.gap:
                bad.append((i, j, "seam"))
            elif ij:  # i can't be tested after j: i first
                order.append((i, j))
            elif ji:
                order.append((j, i))
    return bad, order


def _cycles(nodes, edges):
    """The strongly connected components with 2+ nodes of the directed graph (Tarjan, iterative): order cycles."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    index, low, on, stack, out, n = {}, {}, set(), [], [], [0]
    for root in sorted(nodes):
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, k = work.pop()
            if k == 0:
                index[v] = low[v] = n[0]
                n[0] += 1
                stack.append(v)
                on.add(v)
            nxt = adj.get(v, [])
            if k < len(nxt):
                work.append((v, k + 1))
                w = nxt[k]
                if w not in index:
                    work.append((w, 0))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            for w in nxt:  # children done: fold their low links in
                if w in on:
                    low[v] = min(low[v], low[w])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return out


def legal(uni, keys):
    """The rule violations of layout `keys`: [{"rule": "unknown" | "cup" | "seam" | "order" | "width", "tiles":
    [keys]}] ([] = legal). "order": tiles whose one-way cups admit no order to add them in (a cycle). See the module
    docstring for the rules."""
    out = []
    members = set()
    for k in keys:
        i = uni.index.get(_key(k))
        if i is None:
            out.append({"rule": "unknown", "tiles": [_key(k)]})
        else:
            members.add(i)
    bad, order = _pairs(uni, members)
    out += [{"rule": r, "tiles": [uni.keys[i], uni.keys[j]]} for i, j, r in bad]
    out += [{"rule": "order", "tiles": [uni.keys[m] for m in c]}
            for c in _cycles({m for e in order for m in e}, order)]
    for comp in uni._components(members):
        if not uni.width_ok(comp):
            out.append({"rule": "width", "tiles": sorted(uni.keys[m] for m in comp)})
    return out


def _stuck_xy(stuck):
    return np.array([[float(s[0]), float(s[1])] for s in (stuck or [])], dtype=float).reshape(-1, 2)


def mutate_chunk(rng, uni, keys, chunk_id, stuck, n_remove, n_add, targeted_share=0.5, count_band=COUNT_BAND):
    """A new layout (frozenset of keys) differing from `keys` only in tiles owned by grid chunk `chunk_id`: up to
    n_remove of its tiles dropped, then up to n_add new ones added by the rules (checked against every tile of the
    layout, other chunks' included; never one just dropped). Steering (evolve_layout.targeted_orders' weights,
    each draw targeted with probability targeted_share, uniform otherwise): removal weighted by the stuck shapes
    (`stuck`: [[x, y, speed, r], ...] map coordinates, a chunk test result's "stuck") within STUCK_REACH tile edges
    of each tile; additions by sparseness (1 / (1 + layout tiles within 2 x STUCK_REACH edges)), zero within
    STUCK_REACH of a stuck shape. The chunk's tile count stays within count_band (share) of its base count (the
    request is clipped; None: no limit). rng: a numpy Generator."""
    from evolve_layout import STUCK_REACH, _near_counts, _weighted_order
    if chunk_id not in uni.owned:
        raise ValueError(f"mutate_chunk: {chunk_id!r} isn't a grid chunk")
    members = uni.indices(keys)
    own_all = uni.owned[chunk_id]
    own = np.array(sorted(m for m in members if uni.owner[m] == uni.chunk_index[chunk_id]), dtype=int)
    n_now = len(own)
    if count_band is None:
        lo, hi = 0, n_now + n_add
    else:
        base_n = uni.base_count[chunk_id]
        lo, hi = math.ceil((1 - count_band) * base_n), math.floor((1 + count_band) * base_n)
    n_remove = max(0, min(n_remove, n_now - lo))
    reach = STUCK_REACH * uni.side
    st = _stuck_xy(stuck)
    drop = []
    if n_remove and n_now:
        w = _near_counts(uni.centers[own], st, reach, uni.size)
        drop = own[_weighted_order(rng, w, targeted_share)[:n_remove]].tolist()
    members -= set(drop)
    n_add = max(0, min(n_add, hi - (n_now - len(drop))))
    if n_add:
        cand = np.array([c for c in own_all.tolist() if c not in members and c not in drop], dtype=int)
        if len(cand):
            # the layout's tiles within 2 x reach of the chunk (sparseness), across the wraps
            near_pts = uni.centers[sorted(members)] if members else np.zeros((0, 2))
            sparse = 1.0 / (1.0 + _near_counts(uni.centers[cand], near_pts, 2 * reach, uni.size))
            calm = (_near_counts(uni.centers[cand], st, reach, uni.size) == 0).astype(float)
            added = 0
            for c in cand[_weighted_order(rng, sparse * calm, targeted_share)].tolist():
                if uni.add_ok(c, members)[0]:
                    members.add(c)
                    added += 1
                    if added == n_add:
                        break
    return frozenset(uni.keys[m] for m in members)


def _min_cover(nodes, edges, prefer):
    """A minimum vertex cover of the graph (exact for up to EXACT_COVER nodes, else greedy), ties broken by
    `prefer` (a sort key per node: smaller is dropped first)."""
    nodes = sorted(nodes, key=prefer)
    if len(nodes) <= EXACT_COVER:
        for r in range(1, len(nodes) + 1):
            for combo in itertools.combinations(nodes, r):  # in `prefer` order: the first cover found is preferred
                s = set(combo)
                if all(a in s or b in s for a, b in edges):
                    return list(combo)
    out, left = [], set(edges)
    while left:
        deg = {}
        for a, b in left:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        v = min(deg, key=lambda v: (-deg[v], prefer(v)))
        out.append(v)
        left = {e for e in left if v not in e}
    return out


def fix_seams(uni, keys):
    """(legal layout, dropped): `keys` with the fewest tiles dropped to clear its rule violations. Pair conflicts
    (cup, seam) are a graph: per connected component its minimum vertex cover (exact up to EXACT_COVER tiles, else
    greedy), preferring to drop tiles not in the base, then the larger key; then each too-wide cluster drops, one at
    a time, the non-base tile whose removal leaves the narrowest cluster. dropped: [{"key", "rule", "chunk"}] in
    drop order. Deterministic."""
    members = set()
    dropped = []
    for k in sorted(_key(k) for k in keys):
        i = uni.index.get(k)
        if i is None:
            dropped.append({"key": k, "rule": "unknown", "chunk": None})
        else:
            members.add(i)
    bad, _ = _pairs(uni, members)
    edges = {(i, j) for i, j, _ in bad}

    def prefer(v):  # drop new tiles first, then a fixed order
        return (bool(uni.in_base[v]), uni.keys[v])

    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = set()
    for s in sorted(adj):
        if s in seen:
            continue
        comp, todo = {s}, [s]
        while todo:
            for t in adj[todo.pop()]:
                if t not in comp:
                    comp.add(t)
                    todo.append(t)
        seen |= comp
        for v in _min_cover(comp, [e for e in edges if e[0] in comp], prefer):
            members.discard(v)
            dropped.append({"key": uni.keys[v], "rule": "pair", "chunk": uni.chunk_ids[uni.owner[v]]})
    while True:  # order cycles: drop the tile on the most cycle edges (new tiles first) until none is left
        _, order = _pairs(uni, members)
        cyc = _cycles({m for e in order for m in e}, order)
        if not cyc:
            break
        comp = set(cyc[0])
        deg = {v: sum(v in e and e[0] in comp and e[1] in comp for e in order) for v in comp}
        v = min(comp, key=lambda v: (-deg[v], prefer(v)))
        members.discard(v)
        dropped.append({"key": uni.keys[v], "rule": "order", "chunk": uni.chunk_ids[uni.owner[v]]})
    while True:
        wide = [c for c in uni._components(members) if not uni.width_ok(c)]
        if not wide:
            break
        comp = wide[0]
        cand = [v for v in comp if not uni.in_base[v]] or comp

        def after(v):
            rest = set(comp) - {v}
            return (max((uni.width(c) for c in uni._components(rest)), default=0.0), prefer(v))
        v = min(cand, key=after)
        members.discard(v)
        dropped.append({"key": uni.keys[v], "rule": "width", "chunk": uni.chunk_ids[uni.owner[v]]})
    return frozenset(uni.keys[m] for m in members), dropped


def owned_keys(uni, keys, chunk_id):
    """The tiles of layout `keys` owned by grid chunk chunk_id."""
    i = uni.chunk_index[chunk_id]
    return frozenset(k for k in (_key(k) for k in keys) if k in uni.index and uni.owner[uni.index[k]] == i)


def assemble(uni, picks, default=None, report=None):
    """The layout made of each grid chunk's owned tiles from its picked layout (picks: {chunk_id: keys}; a chunk
    left out takes `default`'s, or raises KeyError without one), made legal by fix_seams. report: a list that gets
    fix_seams' dropped entries (also kept as assemble.last_dropped). Deterministic."""
    union = set()
    for cid in uni.chunk_ids:
        if cid in picks:
            src = picks[cid]
        elif default is not None:
            src = default
        else:
            raise KeyError(f"assemble: no pick for {cid}")
        union |= owned_keys(uni, src, cid)
    out, dropped = fix_seams(uni, union)
    assemble.last_dropped = dropped
    if report is not None:
        report.extend(dropped)
    return out


assemble.last_dropped = []
