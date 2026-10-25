"""The "diamonds" preset's geometry: a tumbling-blocks rhombus tiling (the lattice of an isometric image, see
maps/diamonds.json, which also gives the palette) laid over the world at any size, a seeded random pick of its
diamonds that never forms a cup, plus placement of the falling hexagons. Pure geometry; scene.py turns the results
into bodies.
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


def lattice_diamonds(world_w, world_h, side, seed=7, palette=None):
    """Every diamond of a tumbling-blocks tiling with edge length `side` covering the world: [(corners (2, 4),
    rgb, face)], each a random palette color (default: the image's, maps/diamonds.json, minus its dark navy, which
    vanishes against the background). Cubes sit in rows 1.5 side apart, alternate rows shifted half a cube."""
    if palette is None:
        palette = [c for c in load_tiling()["palette"] if sum(c) > 230]
    rng = np.random.default_rng(seed)
    w = 2 * side * SQRT3_2  # one cube's width
    out = []
    for j in range(-1, int(world_h / (1.5 * side)) + 3):
        for i in range(-1, int(world_w / w) + 3):
            c = (i * w + (w / 2 if j % 2 else 0.0), j * 1.5 * side)
            for face, corners in cube_faces(c, side).items():
                out.append((corners, list(palette[int(rng.integers(len(palette)))]), face))
    return out


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


def _corner_gap(p, q):
    """The smallest distance between a corner of p and a corner of q ((2, n) each)."""
    return float(np.hypot(*(p[:, :, None] - q[:, None, :])).min())


def tile_key(corners):
    """A tiling diamond's id: its center, rounded to 0.1 world units (the same at any world size)."""
    return [round(float(corners[0].mean()), 1), round(float(corners[1].mean()), 1)]


def load_layout(path):
    """A saved diamond layout: ([tile_key, ...], (world_w, world_h) it was made for, or None) (see save_layout)."""
    with open(path) as f:
        data = json.load(f)
    return [tuple(k) for k in data["tiles"]], tuple(data["size"]) if "size" in data else None


def save_layout(path, diamonds, size, note=""):
    """Write the picked diamonds' tile keys ([(corners, rgb)] from pick_diamonds) and the world size they were picked
    for to `path` as JSON."""
    with open(path, "w") as f:
        json.dump({"note": note, "size": list(size), "tiles": [tile_key(c) for c, _ in diamonds]}, f, indent=0)


def pick_diamonds(diamonds, world_w, world_h, seed=7, fraction=0.15, side=5.0, margin=10.0, top=20.0, size=None,
                  gap=0.0, max_width=None, layout=None, layout_size=None, remove=0, add=0, add_seed=0):
    """A seeded random `fraction` of the diamonds (corners, rgb, face) in the playfield: at least `margin` in from the
    side walls (at x = side and world_w - side), at least `top` below the top edge (keeps the band where looping
    bodies re-enter clear), and centered above the bottom edge (there's no floor, so one may hang off the bottom).
    A diamond that would form a cup with one already picked (forms_cup; corners up to `gap` apart count as touching)
    is skipped and the next random one tried, as is one that would make a cluster (picked diamonds linked by
    corners up to `gap` apart) wider than `max_width`: clusters may grow tall, never into long horizontal chains.
    `size`: the tiling's edge length (for the cup test; default: from the first diamond). `layout`: tile keys
    (tile_key) to pick instead of random ones (those outside the playfield are left out; the rest are trusted to
    follow the rules: forms_cup isn't symmetric, so re-testing them in another order could drop some). In a world
    bigger than `layout_size` (the world it was made for), the area past it is topped up with random diamonds by the
    rules, to the same `fraction` of its tiles.
    Then a variant of that layout: `remove` of its diamonds dropped at random and `add` new ones picked by the same
    rules (never one just removed), both seeded by (seed, add_seed). Returns [(corners, rgb)] in tiling order."""
    inside = [d for d in diamonds
              if d[0][0].min() >= side + margin and d[0][0].max() <= world_w - side - margin
              and d[0][1].min() >= top and d[0][1].mean() <= world_h]
    if not inside:
        return []
    size = float(np.hypot(*(inside[0][0][:, 1] - inside[0][0][:, 0]))) if size is None else size
    link = max(gap, 1e-6 * size)
    chosen, cluster, banned = [], {}, set()  # cluster: picked index -> its cluster (a shared list of picked indices)

    def fill(order, target, trusted=False):  # trusted: already checked (a saved layout), just rebuild the clusters
        for k in order:
            if len(chosen) == target:
                break
            if k in cluster or k in banned:
                continue
            p = inside[k][0]
            if not trusted and any(forms_cup(p, inside[m][0], size, gap) for m in chosen):
                continue
            near = [m for m in chosen if _corner_gap(p, inside[m][0]) <= link]
            merged = {id(c): c for c in (cluster[m] for m in near)}.values()
            members = [int(k)] + [m for c in merged for m in c]
            if max_width is not None and not trusted:
                xs = np.concatenate([inside[m][0][0] for m in members])
                if xs.max() - xs.min() > max_width:
                    continue
            for m in members:
                cluster[m] = members
            chosen.append(int(k))

    if layout is None:
        fill(np.random.default_rng(seed).permutation(len(inside)), max(1, int(round(fraction * len(inside)))))
    else:
        index = {tuple(tile_key(d[0])): k for k, d in enumerate(inside)}
        order = [index[tuple(key)] for key in layout if tuple(key) in index]
        fill(order, len(order), trusted=True)
        if layout_size is not None:  # top up the area past the layout's world (right / bottom strips)
            lw, lh = layout_size
            extra = [k for k, d in enumerate(inside) if d[0][0].max() > lw - side - margin or d[0][1].mean() > lh]
            if extra:
                rng0 = np.random.default_rng([seed, 1])
                fill([extra[i] for i in rng0.permutation(len(extra))],
                     len(chosen) + max(0, int(round(fraction * len(extra)))))
    rng = np.random.default_rng([seed, add_seed])
    if remove:  # drop some, then re-pick the rest (a subset of a valid layout is valid)
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
    for _ in range(count):
        for _ in range(tries):
            r = float(rng.uniform(r_min, r_max))
            lo_x, hi_x = side + r + gap, world_w - side - r - gap
            c = np.array([rng.uniform(lo_x, hi_x), rng.uniform(r + gap, world_h - r - gap)])
            if any(math.hypot(c[0] - x, c[1] - y) < r + q + gap for x, y, q, _ in placed):
                continue
            if any(_dist_to_polygon(c, poly) < r + gap for poly in obstacles):
                continue
            placed.append((float(c[0]), float(c[1]), r, float(rng.uniform(0, math.pi / 3))))
            break
    return placed
