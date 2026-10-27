"""The "diamonds" preset's geometry: a tumbling-blocks rhombus tiling (the lattice of an isometric image, see
maps/diamonds.json, which also gives the palette) laid over the world at any size, a seeded random pick of its
diamonds that never forms a cup, plus placement of the falling hexagons. Pure geometry; scene.py turns the results
into bodies. penrose_tiles gives a Penrose P3 rhombus tiling (de Bruijn's pentagrid) in the same [(corners, rgb, kind)]
form, for the "penrose" worlds: the picking rules work on any convex tiles.
"""
import json
import math
import os

import numpy as np

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "diamonds.json")
SQRT3_2 = math.sqrt(3) / 2


def load_tiling(path=MAP_PATH):
    """{'image_size': [w, h], 'lattice': {...}, 'palette': [[r, g, b], ...], 'diamonds': [...]} (corners in image px)."""
    with open(path) as f:
        return json.load(f)


def cube_faces(c, side):
    """The 3 faces of the isometric cube whose front corner (where its faces meet) is c: {face: (2, 4) corners}.
    The top face points left / right; the left and right faces each point down (their lowest corner is below c)."""
    a, cx, cy = side * SQRT3_2, c[0], c[1]
    pts = {"top": [(0, 0), (-a, -side / 2), (0, -side), (a, -side / 2)],
           "left": [(0, 0), (0, side), (-a, side / 2), (-a, -side / 2)],
           "right": [(0, 0), (a, -side / 2), (a, side / 2), (0, side)]}
    return {face: np.array([[cx + dx for dx, _ in v], [cy + dy for _, dy in v]]) for face, v in pts.items()}


def lattice_diamonds(world_w, world_h, side, seed=7, palette=None, y0=0.0):
    """Every diamond of a tumbling-blocks tiling with edge length `side` covering the world: [(corners (2, 4),
    rgb, face)], each a random palette color (default: the image's, maps/diamonds.json, minus its dark navy, which
    vanishes against the background). Cubes sit in rows 1.5 side apart, alternate rows shifted half a cube; the
    whole tiling is shifted down by `y0` (y0 = side: the top faces of the first row touch the top edge)."""
    if palette is None:
        palette = [c for c in load_tiling()["palette"] if sum(c) > 230]
    rng = np.random.default_rng(seed)
    w = 2 * side * SQRT3_2  # one cube's width
    # every cube's front corner, row by row, then all faces' corners at once (cube_faces' arithmetic, element-wise)
    fronts = [(i * w + (w / 2 if j % 2 else 0.0), y0 + j * 1.5 * side)
              for j in range(-1, int(world_h / (1.5 * side)) + 3) for i in range(-1, int(world_w / w) + 3)]
    offsets = cube_faces((0.0, 0.0), side)  # {face: (2, 4) corners around a front corner at the origin}
    faces = list(offsets)
    c = np.array(fronts, dtype=float).reshape(-1, 1, 2, 1)
    corners = (c + np.array([offsets[f] for f in faces])[None]).reshape(-1, 2, 4)  # (cubes * 3, 2, 4)
    # one draw per face in order: the same stream as drawing them one at a time
    picks = rng.integers(len(palette), size=len(corners)).tolist() if len(corners) else []
    return [(corners[k], list(palette[p]), faces[k % 3]) for k, p in enumerate(picks)]


def pentagrid_offsets(seed):
    """de Bruijn's five pentagrid offsets (gamma_0..4) for `seed`: random in [0, 1), shifted to sum exactly to 0, so
    the dual tiling is a true Penrose (P3) tiling. Random offsets are regular (no three grid lines meet) almost surely."""
    g = np.random.default_rng(seed).uniform(0.0, 1.0, 5)
    g -= g.mean()
    g[4] = -g[:4].sum()
    return g


def penrose_tiles(world_w, world_h, side, seed=1, palette=None, margin=None, rotation=18.0):
    """Every rhombus of a Penrose P3 tiling with edge `side` that reaches into the world grown by `margin` (default
    2 sides) on every side: [(corners (2, 4), rgb, kind)], kind "fat" (72 / 108 degree corners) or "thin" (36 / 144),
    sorted by center (rows of y, then x), each a random palette color as lattice_diamonds gives them.
    Edges run in 5 directions 36 degrees apart (mod 180), the first at `rotation` degrees: the default 18 leaves no
    edge horizontal (slopes 18, 54, 90); at 0, 4 of the 10 rhombus orientations have flat tops (see scene.py's
    PENROSE_ROTATION notes). Any rotation has edges sloping 18 degrees or less.
    de Bruijn's pentagrid: grid lines x.e_j + gamma_j = k (e_j five unit vectors 72 degrees apart, gamma_j from
    pentagrid_offsets(seed), summing to 0); each crossing of a family-r line with a family-s line is one rhombus,
    whose corners are side * sum_j K_j e_j for the integer K_j = ceil(x.e_j + gamma_j) of the four grid cells around
    the crossing. The tiling depends only on `seed` and `side` (never on the world size), so tile keys are stable."""
    if palette is None:
        palette = [c for c in load_tiling()["palette"] if sum(c) > 230]
    margin = 2 * side if margin is None else margin
    a0 = math.radians(rotation)
    e = np.array([[math.cos(a0 + 2 * math.pi * j / 5), math.sin(a0 + 2 * math.pi * j / 5)] for j in range(5)])  # (5, 2)
    gamma = pentagrid_offsets(seed)
    # a grid point p lands near side * (2.5 p + sum_j gamma_j e_j) (sum_j (p.e_j) e_j = 2.5 p): the grid region that
    # covers the grown world, padded by 2 grid units
    shift = gamma @ e
    lo = (np.array([-margin, -margin]) / side - shift) / 2.5 - 2
    hi = (np.array([world_w + margin, world_h + margin]) / side - shift) / 2.5 + 2
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]])
    proj = box @ e.T + gamma  # (4, 5): each family's line index at the region's corners
    ks = [np.arange(math.floor(proj[:, j].min()) - 1, math.ceil(proj[:, j].max()) + 2) for j in range(5)]
    corners, kinds = [], []
    for r in range(5):
        for s in range(r + 1, 5):
            kr, kss = (a.ravel() for a in np.meshgrid(ks[r], ks[s], indexing="ij"))
            a = np.array([e[r], e[s]])  # p.e_r = kr - gamma_r, p.e_s = ks - gamma_s
            p = np.linalg.solve(a, np.array([kr - gamma[r], kss - gamma[s]])).T  # (n, 2) crossings
            k = np.ceil(p @ e.T + gamma)  # (n, 5) the cell's indices; families r and s are on their lines
            k[:, r], k[:, s] = kr, kss
            dr, ds = np.eye(5)[r], np.eye(5)[s]
            quad = np.stack([k, k + dr, k + dr + ds, k + ds], axis=1)  # (n, 4, 5) the 4 cells around the crossing
            corners.append(side * np.einsum("nqj,jd->ndq", quad, e))  # (n, 2, 4)
            kinds += ["fat" if s - r in (1, 4) else "thin"] * len(kr)
    corners = np.concatenate(corners)
    kinds = np.array(kinds)
    lo_c, hi_c = corners.min(axis=2), corners.max(axis=2)
    keep = np.flatnonzero((hi_c[:, 0] >= -margin) & (lo_c[:, 0] <= world_w + margin)
                          & (hi_c[:, 1] >= -margin) & (lo_c[:, 1] <= world_h + margin))
    centers = corners[keep].mean(axis=2)
    keep = keep[np.lexsort((np.round(centers[:, 0], 1), np.round(centers[:, 1], 1)))]
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(palette), size=len(keep)).tolist()
    return [(corners[i], list(palette[c]), str(kinds[i])) for i, c in zip(keep, picks)]


def edge_slope(corners):
    """The shallowest slope (degrees from horizontal, 0..90) of a polygon's (2, n) edges."""
    d = np.roll(corners, -1, axis=1) - corners
    return float(np.degrees(np.arctan2(np.abs(d[1]), np.abs(d[0]))).min())


def steep_tiles(tiles, min_slope):
    """The tiles [(corners, ...)] none of whose edges slope less than `min_slope` degrees: every face a falling shape
    can land on is steep enough to slide off (a rhombus has each edge's parallel twin facing up or down, so a shallow
    edge is always a shallow top somewhere)."""
    return [t for t in tiles if edge_slope(t[0]) >= min_slope - 1e-9]


def _corner_arc(poly, k):
    """The directions (start, end) radians, counterclockwise with y up, that convex poly (2, n) covers at corner k."""
    v = poly[:, k]
    e1, e2 = poly[:, k - 1] - v, poly[:, (k + 1) % poly.shape[1]] - v
    a1, a2 = math.atan2(-e1[1], e1[0]), math.atan2(-e2[1], e2[0])
    if (a2 - a1) % (2 * math.pi) > math.pi:  # the interior angle is under 180 degrees: take the short way round
        a1, a2 = a2, a1
    return a1 % (2 * math.pi), a2 % (2 * math.pi)


def forms_cup(p, q, side, gap=0.0):
    """Do diamonds p and q (2, 4) touch at a corner with an open valley above it, walled by p on one side and q on
    the other? The gap between them there is a wedge of directions; it's a cup when that wedge opens upward and both
    of its walls point up (the two faces of a cube without its top face, two top faces tip to tip, ...): it catches
    what falls in. With `gap` > 0, corners up to that far apart count too (q is slid onto p's corner for the
    test): a pocket narrower than `gap` catches anything wider than it just the same."""
    eps = 1e-6 * side
    for i in range(p.shape[1]):
        d = np.hypot(*(q - p[:, i:i + 1]))
        if d.min() > max(gap, eps):
            continue  # no corner of q at (or near) this one
        arcs = [_corner_arc(p, i) + ("p",), _corner_arc(q, int(d.argmin())) + ("q",)]
        # the free wedges are between one arc's end and the other's start (counterclockwise)
        for (s1, e1, n1), (s2, e2, n2) in (arcs, arcs[::-1]):
            lo, hi = e1, s2  # free from the end of arc 1 to the start of arc 2
            width = (hi - lo) % (2 * math.pi)
            if width < 1e-9:
                continue
            if 0 < lo < math.pi and 0 < lo + width < math.pi:  # both walls point up: an upward-opening valley
                return True
    return False


def cup_at(v, tiles, side):
    """The "outline" cup rule at point v: the tiles ((2, n) each) with a corner at v cover wedges of directions around
    it; the union's outline runs along the wedges' unshared edges, so each free wedge between two solid ones is a
    notch in the outline at v (several notches when tiles meet only at v: the outline passes v once per notch). v is
    a cup when a notch is under 180 degrees (a reflex outline vertex) and both of its edges lead up from v (both
    outline neighbours are higher, screen y down): a U / V that opens upward and catches what falls in. A single
    convex tile's bottom tip is never one (its free wedge is over 180 degrees). Returns None or the cup's
    ((start, end) radians, counterclockwise with y up, of the wedge on the notch's clockwise side, and of the one on
    its counterclockwise side, the notch (start, width), and the indices into `tiles` of those two)."""
    eps = 1e-6 * side
    arcs = []
    for j, q in enumerate(tiles):
        d = np.hypot(q[0] - v[0], q[1] - v[1])
        k = int(d.argmin())
        if d[k] <= eps:
            arcs.append(_corner_arc(q, k) + (j,))
    if len(arcs) < 2:
        return None
    arcs.sort()
    for a, b in zip(arcs, arcs[1:] + arcs[:1]):
        lo = a[1]
        width = (b[0] - lo) % (2 * math.pi)  # a shared edge: ~0 (or ~2 pi from rounding): never a cup
        if width > 1e-9 and 0 < lo < math.pi and 0 < lo + width < math.pi:
            return (a[0], a[1]), (b[0], b[1]), (lo, width), a[2], b[2]
    return None


def outline_cup(p, others, side):
    """Would adding tile p (2, n) to the picked tiles `others` (those touching it, already shifted across the wraps)
    make a cup (cup_at) at one of p's corners? Only p's corners can change: the outline anywhere else stays the same.
    Returns None or (p's corner index, cup_at's result)."""
    tiles = [p] + list(others)
    for i in range(p.shape[1]):
        c = cup_at(p[:, i], tiles, side)
        if c is not None:
            return i, c
    return None


def cup_pattern(cup):
    """A cup_at result as a hashable local pattern: the two wedges walling the notch, (start, end) in whole degrees
    (counterclockwise from +x, y up), which give each tile's corner angle (fat 72 / 108, thin 36 / 144) and its
    orientation."""
    (a, b, _, _, _) = cup
    deg = lambda w: tuple(int(round(math.degrees(x))) % 360 for x in w)  # noqa: E731
    return deg(a), deg(b)


def _corner_gap(p, q):
    """The smallest distance between a corner of p and a corner of q ((2, n) each)."""
    return float(np.hypot(*(p[:, :, None] - q[:, None, :])).min())


def _snap(q, p, near):
    """q (2, n) with each corner closer than `near` to a corner of p moved onto that corner: a near miss counts as
    touching, so the outline cup test sees a U / V across the small gap."""
    d = np.hypot(*(q[:, :, None] - p[:, None, :]))  # (q corner, p corner)
    j = d.argmin(axis=1)
    hit = d[np.arange(q.shape[1]), j] < near
    if not hit.any():
        return q
    q = q.copy()
    q[:, hit] = p[:, j[hit]]
    return q


def _stack(diamonds):
    """The diamonds' corners as one (n, 2, k) float array, or None if their corner arrays differ in shape."""
    shapes = {np.shape(d[0]) for d in diamonds}
    if len(shapes) != 1:
        return None
    return np.array([d[0] for d in diamonds], dtype=float)


def tile_key(corners):
    """A tiling diamond's id: its center, rounded to 0.1 world units (the same at any world size)."""
    return [round(float(corners[0].mean()), 1), round(float(corners[1].mean()), 1)]


def load_layout(path):
    """A saved diamond layout: ([tile_key, ...], (world_w, world_h) it was made for, or None) (see save_layout)."""
    with open(path) as f:
        data = json.load(f)
    return [tuple(k) for k in data["tiles"]], tuple(data["size"]) if "size" in data else None


def layout_spawn(path):
    """A saved layout's own shape spawning settings ({} if it has none; see scene.DIAMONDS_SPAWN for the keys)."""
    with open(path) as f:
        return dict(json.load(f).get("spawn", {}))


def layout_shift(path):
    """A saved layout's own offset (dx, dy) for its picked diamonds, in world units ((0, 0) if it has none): the
    whole layout slid over after picking, e.g. half a row down off the lattice."""
    with open(path) as f:
        return tuple(float(v) for v in json.load(f).get("shift", (0.0, 0.0)))


def layout_tiling(path):
    """A saved layout's tiling: {"tiling": "diamonds"} (the default, for files without one) or {"tiling": "penrose",
    "pentagrid_seed": int, "edge": float, "rotation": float, "min_slope": float, "cups": "pair" | "outline", "cup_gap": float} (penrose_tiles' seed, edge length and rotation,
    steep_tiles' minimum slope, pick_diamonds' cup rule and its gap (pair rule only), so it rebuilds identically).
    A penrose file without "cups" predates the outline rule: "pair"."""
    with open(path) as f:
        data = json.load(f)
    keys = ("tiling", "pentagrid_seed", "edge", "rotation", "min_slope", "cups", "cup_gap", "near", "bias")
    info = {k: data[k] for k in keys if k in data} or {"tiling": "diamonds"}
    if info["tiling"] == "penrose":
        info.setdefault("cups", "pair")
        if info["cups"] == "outline":
            info.setdefault("near", 0.0)  # saved before the near-miss ban
    return info


def shift_diamonds(diamonds, shift, world_w, world_h):
    """[(corners, rgb)] translated by shift (dx, dy); one that ends up wholly past the right / bottom edge comes back
    one world width / height over (the world loops on every side). Ones straddling an edge stay put: the seams are
    periodic, so they collide and draw across it."""
    dx, dy = shift
    if not dx and not dy:
        return diamonds
    out = []
    for corners, rgb in diamonds:
        c = corners + np.array([[dx], [dy]])
        c[0] -= world_w * (c[0].min() >= world_w)
        c[1] -= world_h * (c[1].min() >= world_h)
        out.append((c, rgb))
    return out


def save_layout(path, diamonds, size, note="", spawn=None, keys=None, tiling=None):
    """Write the picked diamonds' tile keys ([(corners, rgb)] from pick_diamonds, or `keys` directly), the world size
    they were picked for, (optional) the layout's own shape spawning settings and (optional) its tiling (a dict, see
    layout_tiling; left out for the default diamonds tiling) to `path` as JSON."""
    keys = [tile_key(c) for c, _ in diamonds] if keys is None else [list(k) for k in keys]
    data = {"note": note, "size": list(size), **(tiling or {}), **({"spawn": spawn} if spawn else {}), "tiles": keys}
    with open(path, "w") as f:
        json.dump(data, f, indent=0)


def tile_neighbours(tiles, reach=30.0):
    """For each tile, the indices of the tiles with a corner within `reach` of one of its corners (sharing a corner
    or nearly)."""
    cs = [np.asarray(t[0], float) for t in tiles]
    centers = np.array([c.mean(axis=1) for c in cs])
    radius = max(float(np.hypot(*(c - c.mean(axis=1, keepdims=True))).max()) for c in cs)
    adj = np.zeros((len(tiles), len(tiles)), bool)
    for i, j in zip(*np.nonzero(np.triu(np.hypot(*(centers[:, None] - centers[None]).transpose(2, 0, 1))
                                        <= 2 * radius + reach, 1))):
        d = cs[i][:, :, None] - cs[j][:, None, :]
        adj[i, j] = adj[j, i] = bool((np.hypot(d[0], d[1]) <= reach).any())
    return adj


def clustered_order(tiles, seed, bias, target=None, rounds=5, first=0.4, reach=30.0):
    """A pick order (a permutation of tile indices) that mildly favours clusters: round 1 draws `first` of `target`
    (default: all tiles) uniformly at random; each later round draws (target - round 1) / (rounds - 1) more with
    weights 1 + bias * (number of already-drawn neighbours, see tile_neighbours), until every tile is drawn
    (pick_diamonds' rules may veto some, so the order runs past the target)."""
    n = len(tiles)
    rng = np.random.default_rng([seed, 3])
    target = n if target is None else min(n, target)
    adj = tile_neighbours(tiles, reach).astype(float)
    first_n = max(1, int(round(first * target)))
    step = max(1, int(math.ceil((target - first_n) / max(1, rounds - 1))))
    order = rng.choice(n, size=min(first_n, n), replace=False).tolist()
    drawn = np.zeros(n, bool)
    drawn[order] = True
    while len(order) < n:
        w = (1.0 + bias * (adj @ drawn)) * ~drawn
        k = min(step, n - len(order))
        pick = rng.choice(n, size=k, replace=False, p=w / w.sum()).tolist()
        order += pick
        drawn[pick] = True
    return order


def _random_order(tiles, seed, bias, target):
    """pick_diamonds' order: a plain seeded permutation, or clustered_order when `bias` is set."""
    if not bias:
        return np.random.default_rng(seed).permutation(len(tiles))
    return clustered_order(tiles, seed, bias, target)


def pick_diamonds(diamonds, world_w, world_h, seed=7, fraction=0.15, size=None, gap=0.0, max_width=None,
                  layout=None, layout_size=None, remove=0, add=0, add_seed=0, count=None, cups="pair", cup_log=None,
                  near=30.0, bias=0.0):
    """A seeded random `fraction` (or `count`) of the diamonds (corners, rgb, face) lying wholly inside the world (0, 0, world_w,
    world_h), which wraps on every side (looping, no walls). A diamond that would form a cup with one already picked
    (forms_cup; corners up to `gap` apart count as touching) is skipped and the next random one tried, as is one that
    would make a cluster (picked diamonds linked by corners up to `gap` apart) wider than `max_width`: clusters may
    grow tall, never into long horizontal chains. Both rules also see across the wraps (each picked diamond's copies
    a world width / height away); a cluster can't link across the left / right seam (its width would be ambiguous).
    `size`: the tiling's edge length (for the cup test; default: from the first diamond). `layout`: tile keys
    (tile_key) to pick instead of random ones (those outside the world are left out; the rest are trusted to follow
    the rules: forms_cup isn't symmetric, so re-testing them in another order could drop some). In a world bigger
    than `layout_size` (the world it was made for), the area past it is topped up with random diamonds by the rules,
    to the same `fraction` of its tiles.
    Then a variant of that layout: `remove` of its diamonds dropped at random and `add` new ones picked by the same
    rules (never one just removed), both seeded by (seed, add_seed). Returns [(corners, rgb)] in tiling order.
    cups: "pair" (the above: forms_cup on each pair, with `gap`) or "outline": only tiles that touch (share a corner;
    `gap` is unused) link into clusters, and a diamond is skipped when the outline of the union of it and the picked
    ones touching it would get a cup at one of its corners (outline_cup / cup_at: a reflex outline vertex whose two
    outline neighbours are both higher). The outline rule is order-independent but not monotone (a third tile can
    fill a cup), so `remove` only drops diamonds whose removal leaves no cup. `cup_log`: a list that gets
    cup_pattern() of every cup the outline rule turned down."""
    eps = 1e-6
    stack = _stack(diamonds)  # (n, 2, corners) when they all have the same corner count: whole-array tests
    if stack is None:
        inside = [d for d in diamonds if d[0][0].min() >= -eps and d[0][0].max() <= world_w + eps
                  and d[0][1].min() >= -eps and d[0][1].max() <= world_h + eps]
        corners = _stack(inside)
    else:
        lo, hi = stack.min(axis=2), stack.max(axis=2)
        keep = np.flatnonzero((lo[:, 0] >= -eps) & (hi[:, 0] <= world_w + eps)
                              & (lo[:, 1] >= -eps) & (hi[:, 1] <= world_h + eps))
        inside, corners = [diamonds[k] for k in keep], stack[keep]
    if not inside:
        return []
    size = float(np.hypot(*(inside[0][0][:, 1] - inside[0][0][:, 0]))) if size is None else size
    fraction = fraction if count is None else count / len(inside)
    outline = cups == "outline"
    if cups not in ("pair", "outline"):
        raise ValueError(f"cups: {cups!r}")
    gap = 0.0 if outline else gap  # the outline rule links only tiles that touch
    near = float(near) if outline else 0.0  # near-miss ban: corners closer than this count as touching (snapped)
    link = max(gap, near, 1e-6 * size)
    reach = 2 * size + max(gap, near)  # centers farther apart than this can't touch, cup or link
    # each diamond's center = its corners' mean, the same reduction as d[0].mean(axis=1), for all of them at once
    centers = corners.mean(axis=2) if corners is not None else np.array([d[0].mean(axis=1) for d in inside])
    shifts = [(dx, dy) for dx in (0.0, -world_w, world_w) for dy in (0.0, -world_h, world_h)]
    shift_x, shift_y = np.array([s[0] for s in shifts]), np.array([s[1] for s in shifts])
    chosen, cluster, banned = [], {}, set()  # cluster: picked index -> its cluster (a shared list of picked indices)

    def fill(order, target, trusted=False):  # trusted: already checked (a saved layout), just rebuild the clusters
        # the picked diamonds' centers, in pick order (rows), for the reach test of every (picked, shift) pair at once:
        # the same float operations as testing each pair in turn, so the same pairs go on to the exact tests below,
        # in the same order (picked diamonds in pick order, each one's shifts in order)
        cx_picked, cy_picked = np.empty(len(inside)), np.empty(len(inside))
        for i, m in enumerate(chosen):
            cx_picked[i], cy_picked[i] = centers[m]
        for k in order:
            if len(chosen) == target:
                break
            if k in cluster or k in banned:
                continue
            p, linked, touching, ok = inside[k][0], [], [], True
            n = len(chosen)
            far = ((np.abs(cx_picked[:n, None] + shift_x - centers[k][0]) > reach)
                   | (np.abs(cy_picked[:n, None] + shift_y - centers[k][1]) > reach))
            for i, s in zip(*np.nonzero(~far)):
                m, (dx, dy) = chosen[i], shifts[s]
                q = inside[m][0] + np.array([[dx], [dy]])
                if not trusted and not outline and forms_cup(p, q, size, gap):
                    ok = False
                elif _corner_gap(p, q) <= link:
                    if dx:
                        ok = trusted  # no cluster through the left / right seam
                    linked.append(m)
                    touching.append(_snap(q, p, near) if near > 0 else q)
                if not ok:
                    break
            if ok and outline and not trusted:
                cup = outline_cup(p, touching, size)
                if cup is not None:
                    ok = False
                    if cup_log is not None:
                        cup_log.append(cup_pattern(cup[1]))
            if not ok:
                continue
            merged = {id(c): c for c in (cluster[m] for m in linked)}.values()
            members = list(dict.fromkeys([int(k)] + [m for c in merged for m in c]))
            if max_width is not None and not trusted:
                xs = np.concatenate([inside[m][0][0] for m in members])
                if xs.max() - xs.min() > max_width:
                    continue
            for m in members:
                cluster[m] = members
            cx_picked[n], cy_picked[n] = centers[k]
            chosen.append(int(k))

    if layout is None:
        target = max(1, int(round(fraction * len(inside))))
        fill(_random_order(inside, seed, bias, target), target)
    else:
        if corners is not None:  # tile_key of every diamond at once (the same means, rounded the same way)
            mx, my = corners[:, 0].mean(axis=1).tolist(), corners[:, 1].mean(axis=1).tolist()
            index = {(round(x, 1), round(y, 1)): k for k, (x, y) in enumerate(zip(mx, my))}
        else:
            index = {tuple(tile_key(d[0])): k for k, d in enumerate(inside)}
        order = [index[tuple(key)] for key in layout if tuple(key) in index]
        fill(order, len(order), trusted=True)
        if layout_size is not None:  # top up the area past the layout's world (right / bottom strips)
            lw, lh = layout_size
            extra = [k for k, d in enumerate(inside) if d[0][0].max() > lw + eps or d[0][1].max() > lh + eps]
            if extra:
                rng0 = np.random.default_rng([seed, 1])
                fill([extra[i] for i in rng0.permutation(len(extra))],
                     len(chosen) + max(0, int(round(fraction * len(extra)))))
    rng = np.random.default_rng([seed, add_seed])
    if remove and outline:  # drop some whose removal leaves no cup (a third tile can fill one), re-pick the rest
        drop, kept_set = set(), set(chosen)
        for k in rng.permutation(chosen).tolist():
            if len(drop) == min(remove, len(chosen)):
                break
            kept_set.discard(k)
            p = inside[k][0]
            rest = [inside[m][0] + np.array([[dx], [dy]]) for m in kept_set for dx, dy in shifts
                    if abs(centers[m][0] + dx - centers[k][0]) <= reach
                    and abs(centers[m][1] + dy - centers[k][1]) <= reach]
            if any(cup_at(p[:, i], rest, size) is not None for i in range(p.shape[1])):
                kept_set.add(k)  # k fills a cup: keep it
            else:
                drop.add(k)
        kept = [k for k in chosen if k not in drop]
        chosen.clear()
        cluster.clear()
        fill(kept, len(kept), trusted=True)
        banned |= drop
    elif remove:  # drop some, then re-pick the rest (a subset of a valid layout is valid)
        drop = set(rng.choice(chosen, size=min(remove, len(chosen)), replace=False).tolist())
        kept = [k for k in chosen if k not in drop]
        chosen.clear()
        cluster.clear()
        fill(kept, len(kept), trusted=True)
        banned |= drop
    if add:  # `add` more on top, by the same rules, in a second seeded order (an offshoot of the layout)
        fill(rng.permutation(len(inside)), len(chosen) + add)
    return [inside[k][:2] for k in sorted(chosen)]


def _dist_to_polygon(p, poly):
    """Distance from point p to the closed polygon poly (2, n); 0 if p is inside."""
    a, b = poly, np.roll(poly, -1, axis=1)
    ab, ap = b - a, p[:, None] - a
    t = np.clip(np.sum(ap * ab, axis=0) / np.maximum(np.sum(ab * ab, axis=0), 1e-12), 0, 1)
    d = float(np.hypot(*(ap - t * ab)).min())
    x, y = p
    straddle = (a[1] > y) != (b[1] > y)
    xc = a[0] + (y - a[1]) * (b[0] - a[0]) / np.where(straddle, b[1] - a[1], 1.0)
    return 0.0 if np.count_nonzero(straddle & (x < xc)) % 2 else d


def place_hexagons(obstacles, world_w, world_h, count, seed=7, r_min=8.0, r_max=20.0, side=5.0, gap=6.0, tries=200):
    """Up to `count` non-overlapping circles (cx, cy, r, angle) for hexagons of circumradius r in [r_min, r_max],
    anywhere in the world between the side walls, each at least `gap` from every obstacle polygon (2, n) and from
    the others. A hexagon lies inside its circle, so circle clearance is hexagon clearance. Seeded."""
    rng = np.random.default_rng(seed + 1)
    placed = []
    # Speed only, same results: each test is first narrowed to the circles / obstacles whose bounding box comes
    # within the clearance (plus a margin far above float error), then decided by the exact test as before. Anything
    # farther than that in x or y can't be closer than the clearance, so the ones skipped could never fail it.
    eps = 1e-6
    obstacles = list(obstacles)
    boxes = (np.array([[np.min(p[0]), np.max(p[0]), np.min(p[1]), np.max(p[1])] for p in obstacles])
             if obstacles else np.empty((0, 4)))
    circles = np.empty((count, 3))  # the placed circles' x, y, r
    for _ in range(count):
        for _ in range(tries):
            r = float(rng.uniform(r_min, r_max))
            lo_x, hi_x = side + r + gap, world_w - side - r - gap
            c = np.array([rng.uniform(lo_x, hi_x), rng.uniform(r + gap, world_h - r - gap)])
            cx, cy, n = float(c[0]), float(c[1]), len(placed)
            if n:
                reach = circles[:n, 2] + (r + gap + eps)
                near = np.flatnonzero((np.abs(circles[:n, 0] - cx) < reach) & (np.abs(circles[:n, 1] - cy) < reach))
                if any(math.hypot(c[0] - placed[i][0], c[1] - placed[i][1]) < r + placed[i][2] + gap for i in near):
                    continue
            m = r + gap + eps
            near = np.flatnonzero((boxes[:, 0] - m <= cx) & (cx <= boxes[:, 1] + m)
                                  & (boxes[:, 2] - m <= cy) & (cy <= boxes[:, 3] + m))
            if any(_dist_to_polygon(c, obstacles[i]) < r + gap for i in near):
                continue
            circles[n] = cx, cy, r
            placed.append((cx, cy, r, float(rng.uniform(0, math.pi / 3))))
            break
    return placed
