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
    keys = ("tiling", "pentagrid_seed", "edge", "rotation", "min_slope", "cups", "cup_gap", "near", "clusters")
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


# pick_diamonds' archetype clusters (its `clusters` dict; any key left out takes this default): "share" of the tiles
# to place is grown as clusters of touching tiles (the rest are random as before), each one either "round" (a compact
# blob: every step adds the touching tile that keeps the tile centers' radius of gyration smallest, with some noise) or
# a "line" (a chain along a random direction at least "line_min_angle" degrees from horizontal, up to vertical, leaning
# either way: every step adds the touching tile past either end lying closest to the axis), "round_share" of them
# round; target sizes are drawn from "round_size" / "line_size" (inclusive). Every tile still has to pass the rules; a
# cluster stops growing after "max_fails" vetoed or missing candidates.
CLUSTER_DEFAULTS = {"share": 0.0, "round_share": 0.75, "round_size": [4, 8], "line_size": [4, 7],
                    "line_min_angle": 45.0, "max_fails": 3}
# Optional size tiers (clusters["tiers"]; without it the single round_size / line_size pool above is used, as before):
# each tier grows its "share" of the target tiles as clusters drawn from "size" (round; lines from "line_size", default
# the same), largest tier first; the rest of the target is random fill (`share` is replaced by the tiers' sum). A
# cluster gives up after max(max_fails, the tier's "max_fails" or wanted size // 2) vetoes; one under half its wanted
# size is undone and reseeded (up to the tier's "retries", default 40), after which a short one is
# kept and its tiles count toward the tier. "max_width" (optional) widens the cluster width limit for that tier only.
TIER_DEFAULTS = [{"name": "large", "share": 0.10, "size": [18, 30]},
                 {"name": "medium", "share": 0.20, "size": [8, 14]},
                 {"name": "small", "share": 0.30, "size": [3, 6]}]


def cluster_settings(clusters):
    """CLUSTER_DEFAULTS updated with `clusters` (a dict, or a bare share), or None when its share is 0."""
    if not clusters:
        return None
    out = dict(CLUSTER_DEFAULTS)
    out.update(clusters if isinstance(clusters, dict) else {"share": float(clusters)})
    if out.get("tiers"):
        out["share"] = sum(t["share"] for t in out["tiers"])
    return out if out["share"] > 0 else None


def pick_diamonds(diamonds, world_w, world_h, seed=7, fraction=0.15, size=None, gap=0.0, max_width=None,
                  layout=None, layout_size=None, remove=0, add=0, add_seed=0, count=None, cups="pair", cup_log=None,
                  near=30.0, clusters=None, cluster_log=None, remove_order=None, add_order=None, check=False):
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
    cup_pattern() of every cup the outline rule turned down.
    clusters: archetype cluster settings (see CLUSTER_DEFAULTS; None or share 0: all random) for a random layout and
    for `add`: that share of the new tiles is grown as round / line clusters of touching tiles, each decision based on
    the tiles actually picked. `cluster_log`: a list that gets {"kind": "round" | "line", "tiles": [index into the
    returned list...], "angle": degrees from horizontal, + leaning right (lines only)} for each archetype cluster.
    Optional steering (evolve_layout's targeted mutation and crossover; left out, nothing changes): `remove_order`:
    tile keys, the order `remove` tries the layout's tiles in (instead of a seeded random one; those not in the layout
    are skipped); `add_order`: tile keys, the order `add` tries tiles in (instead of a seeded random permutation;
    archetype clusters are still grown first); `check`: True: `layout`'s tiles are not trusted but picked by the
    rules in the given order (any a rule vetoes is left out); a list of tile keys: only those are, after all the
    others (trusted)."""
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
    # clusters "random_gap": the cup rule's gap for the random fill (not the archetype clusters), x gap
    random_gap = float(clusters.get("random_gap", 1.0)) if isinstance(clusters, dict) else 1.0
    cup_gap = [gap]  # the cup gap in force: gap, or gap x random_gap while the random fill runs
    reach = 2 * size + max(gap * max(random_gap, 1.0), near)  # centers farther apart can't touch, cup or link
    # each diamond's center = its corners' mean, the same reduction as d[0].mean(axis=1), for all of them at once
    centers = corners.mean(axis=2) if corners is not None else np.array([d[0].mean(axis=1) for d in inside])
    shifts = [(dx, dy) for dx in (0.0, -world_w, world_w) for dy in (0.0, -world_h, world_h)]
    shift_x, shift_y = np.array([s[0] for s in shifts]), np.array([s[1] for s in shifts])
    chosen, cluster, banned = [], {}, set()  # cluster: picked index -> its cluster (a shared list of picked indices)

    # the picked diamonds' centers, in pick order (rows), for the reach test of every (picked, shift) pair at once:
    # the same float operations as testing each pair in turn, so the same pairs go on to the exact tests below,
    # in the same order (picked diamonds in pick order, each one's shifts in order)
    cx_picked, cy_picked = np.empty(len(inside)), np.empty(len(inside))

    def fill(order, target, trusted=False):  # trusted: already checked (a saved layout), just rebuild the clusters
        for k in order:
            if len(chosen) == target:
                break
            if k in cluster or k in banned:
                continue
            try_add(k, trusted)

    def try_add(k, trusted=False):  # pick diamond k if the rules allow it: True if picked
        p, linked, touching, ok = inside[k][0], [], [], True
        own = []  # members of the growing cluster touching p (inner_cups "outline": tested by outline_cup, not pairs)
        why[0] = None
        n = len(chosen)
        far = ((np.abs(cx_picked[:n, None] + shift_x - centers[k][0]) > reach)
               | (np.abs(cy_picked[:n, None] + shift_y - centers[k][1]) > reach))
        for i, s in zip(*np.nonzero(~far)):
            m, (dx, dy) = chosen[i], shifts[s]
            q = inside[m][0] + np.array([[dx], [dy]])
            if inner_outline and m in growing and not trusted:
                if _corner_gap(p, q) <= touch:
                    own.append(q)
                if _corner_gap(p, q) <= link:
                    if dx:
                        ok = False
                    linked.append(m)
            elif not trusted and not outline and forms_cup(p, q, size, cup_gap[0]):
                ok, why[0] = False, "outer pair cup"
            elif _corner_gap(p, q) <= link:
                if dx:
                    ok = trusted  # no cluster through the left / right seam
                linked.append(m)
                touching.append(_snap(q, p, near) if near > 0 else q)
            if not ok:
                break
        if ok and own and outline_cup(p, own, size) is not None:
            ok, why[0] = False, "inner outline cup"
        if ok and outline and not trusted:
            cup = outline_cup(p, touching, size)
            if cup is not None:
                ok = False
                if cup_log is not None:
                    cup_log.append(cup_pattern(cup[1]))
        if ok and not trusted and sealed and any(m in sealed and m not in growing for m in linked):
            ok, why[0] = False, "sealed neighbour"  # finished archetype clusters keep their shape: nothing else links onto them
        if not ok:
            return False
        merged = {id(c): c for c in (cluster[m] for m in linked)}.values()
        members = list(dict.fromkeys([int(k)] + [m for c in merged for m in c]))
        cap = width_cap[0] if width_cap[0] is not None and growing else max_width
        if cap is not None and not trusted:
            xs = np.concatenate([inside[m][0][0] for m in members])
            if xs.max() - xs.min() > cap:
                why[0] = "width limit"
                return False
        for m in members:
            cluster[m] = members
        cx_picked[n], cy_picked[n] = centers[k]
        chosen.append(int(k))
        return True

    settings = cluster_settings(clusters)
    touch = 1e-3 * size  # tiles of the tiling that share an edge or a corner have corners this close
    archetypes = []  # (kind, [picked indices], angle or None)
    sealed, growing = set(), set()  # picked tiles of archetype clusters; those of the one growing now
    width_cap = [None]  # a tier's own max_width while its clusters grow (None: max_width)
    # inner_cups: "pair" (forms_cup against every picked tile) or "outline": a candidate for the growing archetype
    # cluster is tested against that cluster's own members by the outline rule (outline_cup), the pair rule only
    # against all other tiles
    inner_outline = bool(settings) and settings.get("inner_cups", "pair") == "outline" and not outline
    why = [None]  # the rule that vetoed the last try_add
    vetoes = {}  # per tier: reason -> vetoed growth candidates (debug: pick_diamonds.last_vetoes)
    pick_diamonds.last_vetoes = vetoes

    def touching_tiles(k):  # tiles of the tiling (inside the world) sharing an edge or a corner with tile k
        near_k = np.flatnonzero(np.hypot(*(centers - centers[k]).T) <= 2 * size + touch)
        out = []
        for j in near_k.tolist():
            if j != k:
                d = inside[k][0][:, :, None] - inside[j][0][:, None, :]
                if (np.hypot(d[0], d[1]) <= touch).any():
                    out.append(j)
        return out

    def grow(rng, target, budget, tier=None):  # archetype clusters until `budget` tiles are in them (or `target` picked)
        placed, tries, redo = 0, 0, 0
        name = tier["name"] if tier else None
        round_size = tier["size"] if tier else settings["round_size"]
        line_size = tier.get("line_size", tier["size"]) if tier else settings["line_size"]
        width_cap[0] = tier.get("max_width") if tier else None
        while placed < budget and len(chosen) < target and tries < 50 * max(1, budget):
            tries += 1
            free = [k for k in range(len(inside)) if k not in cluster and k not in banned]
            if not free:
                break
            k = free[int(rng.integers(len(free)))]
            growing.clear()
            if not try_add(k):
                continue
            members, tried, fails = [k], {k}, 0
            growing.add(k)
            # the kind behind its round_share of the tiles placed so far (round ones lose more to the cup veto)
            n_round = sum(len(m) for kd, m, _, tn in archetypes if kd == "round" and tn == name)
            n_all = sum(len(m) for _, m, _, tn in archetypes if tn == name)
            rs = tier.get("round_share", settings["round_share"]) if tier else settings["round_share"]
            gap_round = rs * (n_all + 1) - n_round
            if gap_round > 0.5 or (abs(gap_round) <= 0.5 and rng.random() < rs):
                kind, angle = "round", None
                want = int(rng.integers(round_size[0], round_size[1] + 1))
            else:
                kind = "line"
                tilt = float(rng.uniform(settings["line_min_angle"], 90.0))
                angle = tilt if rng.random() < 0.5 else -tilt  # + leaning right (down-right)
                want = int(rng.integers(line_size[0], line_size[1] + 1))
                axis = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
                normal = np.array([-axis[1], axis[0]])
            want = min(want, budget - placed, target - len(chosen) + 1)
            max_fails = (max(settings["max_fails"], tier.get("max_fails", want // 2)) if tier
                         else settings["max_fails"])
            while len(members) < want and fails < max_fails and len(chosen) < target:
                cand = sorted({j for m in members for j in touching_tiles(m)
                               if j not in tried and j not in cluster and j not in banned})
                if kind == "line":  # only tiles past either end of the line so far
                    along = (centers[members] - centers[k]) @ axis
                    t = (centers[cand] - centers[k]) @ axis if cand else np.zeros(0)
                    reach_ok = (t > along.max() + 0.25 * size) | (t < along.min() - 0.25 * size)
                    cand = [j for j, ok in zip(cand, reach_ok.tolist()) if ok]
                if not cand:
                    break
                if kind == "round":
                    pts = centers[members]
                    score = [float(((np.vstack([pts, centers[j]]) - np.vstack([pts, centers[j]]).mean(axis=0)) ** 2)
                                   .sum(axis=1).mean() ** 0.5) for j in cand]
                else:
                    score = np.abs((centers[cand] - centers[k]) @ normal).tolist()
                # round: radii of gyration differ little, so a small noise keeps blobs circular; settings
                # "round_noise" (x edge, default 0.1) larger lets them grow irregular (still touching, so connected)
                noise = settings.get("round_noise", 0.1) if kind == "round" else 0.35
                score = np.array(score) + rng.uniform(0.0, noise * size, len(cand))
                j = cand[int(np.argmin(score))]
                tried.add(j)
                if try_add(j):
                    members.append(j)
                    growing.add(j)
                else:
                    fails += 1
                    v = vetoes.setdefault(name, {})
                    v[why[0] or "other"] = v.get(why[0] or "other", 0) + 1
            if tier and len(members) < want // 2 and redo < tier.get("retries", 40):
                redo += 1  # stalled far short: undo it (its tiles are the last picked, linked to nothing else), reseed
                for m in members:
                    del cluster[m]
                del chosen[-len(members):]
                continue
            redo = 0
            placed += len(members)
            archetypes.append((kind, members, angle, name))
            sealed.update(members)
        growing.clear()
        width_cap[0] = None
        return placed

    def key_index():  # tile_key -> index into `inside`
        if corners is not None:  # tile_key of every diamond at once (the same means, rounded the same way)
            mx, my = corners[:, 0].mean(axis=1).tolist(), corners[:, 1].mean(axis=1).tolist()
            return {(round(x, 1), round(y, 1)): k for k, (x, y) in enumerate(zip(mx, my))}
        return {tuple(tile_key(d[0])): k for k, d in enumerate(inside)}

    def steered(keys, among=None):  # tile keys -> indices into `inside` (those in `among`, if given), in order
        index = key_index()
        out = list(dict.fromkeys(index[tuple(key)] for key in keys if tuple(key) in index))
        return out if among is None else [k for k in out if k in among]

    if layout is None:
        target = max(1, int(round(fraction * len(inside))))
        if settings is not None:
            rng_c = np.random.default_rng([seed, 3])
            if settings.get("tiers"):  # largest first: big clusters need room
                for t in sorted(settings["tiers"], key=lambda t: -t["size"][1]):
                    grow(rng_c, target, int(round(t["share"] * target)), t)
            else:
                grow(rng_c, target, int(round(settings["share"] * target)))
            if settings.get("random_share") is not None:  # random tiles = that share of all picked ones
                rs = settings["random_share"]
                target = min(target, len(chosen) + int(round(len(chosen) * rs / (1.0 - rs))))
            cup_gap[0] = gap * random_gap
            fill(rng_c.permutation(len(inside)), target)
            cup_gap[0] = gap
        else:
            fill(np.random.default_rng(seed).permutation(len(inside)), target)
    else:
        index = key_index()
        order = [index[tuple(key)] for key in layout if tuple(key) in index]
        if check is True or not check:
            fill(order, len(order), trusted=not check)
        else:  # re-check only those in `check`, after all the others (trusted)
            recheck = set(steered(check))
            fill([k for k in order if k not in recheck], len(order), trusted=True)
            fill([k for k in order if k in recheck], len(order))
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
        tries = rng.permutation(chosen).tolist() if remove_order is None else steered(remove_order, kept_set)
        for k in tries:
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
        if remove_order is None:
            drop = set(rng.choice(chosen, size=min(remove, len(chosen)), replace=False).tolist())
        else:
            drop = set(steered(remove_order, set(chosen))[:remove])
        kept = [k for k in chosen if k not in drop]
        chosen.clear()
        cluster.clear()
        fill(kept, len(kept), trusted=True)
        banned |= drop
    if add:  # `add` more on top, by the same rules, in a second seeded order (an offshoot of the layout)
        target = len(chosen) + add
        if settings is not None:  # the same share of them as archetype clusters (stochastic rounding)
            for t in sorted(settings.get("tiers") or [None], key=lambda t: -t["size"][1] if t else 0):
                want = (t or settings)["share"] * add
                grow(rng, target, int(want) + int(rng.random() < want - int(want)), t)
        cup_gap[0] = gap * random_gap if settings is not None else gap
        fill(rng.permutation(len(inside)) if add_order is None else steered(add_order), target)
        cup_gap[0] = gap
    out = sorted(chosen)
    if cluster_log is not None:
        pos = {k: i for i, k in enumerate(out)}
        for kind, members, angle, name in archetypes:
            if all(m in pos for m in members):  # (not dropped by `remove` since)
                cluster_log.append({"kind": kind, "tiles": [pos[m] for m in members],
                                    **({"angle": round(angle, 1)} if angle is not None else {}),
                                    **({"tier": name} if name is not None else {})})
    return [inside[k][:2] for k in out]


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
