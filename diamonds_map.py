"""The "diamonds" preset's geometry: the rhombus tiling recovered from a tumbling-blocks image (maps/diamonds.json,
made offline by a lattice fit, so nothing here needs the image or image processing), fitted to the world, plus
placement of the falling hexagons. Pure geometry; scene.py turns the results into bodies.
"""
import json
import math
import os

import numpy as np

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "diamonds.json")


def load_tiling(path=MAP_PATH):
    """{'image_size': [w, h], 'palette': [[r, g, b], ...], 'diamonds': [{'corners': 4 x [x, y], 'color': k}, ...]}
    with corners in image px (y down), one entry per rhombus of the image."""
    with open(path) as f:
        return json.load(f)


def fit_to_world(tiling, world_w, world_h):
    """Every diamond as ((2, 4) world corners, rgb). The image is scaled uniformly to cover the world (the larger
    of the width and height ratios, so the width is always covered) and centered; the overflow is cropped."""
    img_w, img_h = tiling["image_size"]
    scale = max(world_w / img_w, world_h / img_h)
    off = np.array([[(world_w - img_w * scale) / 2], [(world_h - img_h * scale) / 2]])
    palette = tiling["palette"]
    return [(np.array(d["corners"], dtype=float).T * scale + off, list(palette[d["color"]])) for d in tiling["diamonds"]]


def pick_diamonds(fitted, world_w, world_h, seed=7, fraction=0.10, side=5.0, margin=10.0, top=20.0):
    """A seeded random `fraction` of the fitted diamonds in the playfield: at least `margin` in from the side walls
    (at x = side and world_w - side), at least `top` below the top edge (keeps the band where looping bodies re-enter
    clear), and centered above the bottom edge (there's no floor, so one may hang off the bottom). Returns the chosen
    (corners, rgb) in tiling order."""
    inside = [d for d in fitted
              if d[0][0].min() >= side + margin and d[0][0].max() <= world_w - side - margin
              and d[0][1].min() >= top and d[0][1].mean() <= world_h]
    count = max(1, int(round(fraction * len(inside)))) if inside else 0
    keep = np.random.default_rng(seed).choice(len(inside), size=count, replace=False) if count else []
    return [inside[k] for k in sorted(keep)]


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
