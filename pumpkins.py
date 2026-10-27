"""The pumpkin: a soft body shaped like a small pumpkin, a falling shape kind of the diamonds / penrose maps (a map's
spawn "pumpkin_share", scene.build_diamonds_scene) next to the hexagons and heptagons.

Shape (TEMPLATE, in units of the half width, screen y down): a squat round body, a little wider than tall, with
shallow lobes (one dip in the middle of the bottom, two rounded shoulders either side of the stem) and a short stem
as part of the outline: a narrow 4-point protrusion out of the top dip, leaning a little to the right. 15 points.

Springs (SPRINGS): what a plain SoftBody's skip_sets=3 ring gives (each point to its 1st..4th neighbours round the
outline), plus springs across the body (each point to the two most nearly opposite ones) that keep it round under
load, plus the stem's braces (BRACES: its tip points to the far sides; the ring already ties them to the stem
base, both shoulders and the upper sides), so the stem is a triangulated truss on the body. 79 springs, all ordinary
SoftBody springs (the body is a SoftBody with its springs set, stiffness STIFFNESS), so the solver batches them like
any other and scene_io saves them as they are. Measured (60 s, 1x and 4x, closed-box pile of 63 pumpkins + 67
polygons with an area grab stirring it and point grabs yanking stems; and the halloween map with an area grab): no
outline self-crossing in any frame; impacts and grabs squash a stem flat for a moment, it always comes back.

Sizes: the outline must not touch itself at rest. A point collides with its own body's edges that aren't its or its
neighbours' (solver.World._filter_pairs) when closer than EnvConfig.contact_radius (3 world units); the closest such
pairs (the stem's tip and the shoulder it leans over, the stem's two sides) are MIN_GAP (0.365) x the half width
apart, so at the smallest size (SIZES[0] = 10) 3.65: a resting pumpkin has no self-contacts and a squeezed one is
held apart by them.

A pumpkin is marked `shape_kind` = "pumpkin" (scene_io.BODY_EXTRAS, so saved scenes keep it): pumpkin_eyes.py gives
every such body two black eyes, in any palette.
"""
import math

import numpy as np

from entities import SoftBody

KIND = "pumpkin"
# x, y per point (half width 1, centre 0, y down), clockwise on screen from the stem's tip
TEMPLATE = np.array([
    (0.01, -1.15),  # 0 stem tip, left
    (0.30, -1.11),  # 1 stem tip, right (the stem leans right)
    (0.20, -0.68),  # 2 stem base, right
    (0.52, -0.76),  # 3 right shoulder (lobe)
    (0.88, -0.55),  # 4
    (1.00, -0.08),  # 5 right side
    (0.88, 0.42),   # 6
    (0.50, 0.72),   # 7 bottom right lobe
    (0.00, 0.62),   # 8 bottom dip
    (-0.50, 0.72),  # 9 bottom left lobe
    (-0.88, 0.42),  # 10
    (-1.00, -0.08),  # 11 left side
    (-0.88, -0.55),  # 12
    (-0.52, -0.76),  # 13 left shoulder (lobe)
    (-0.20, -0.68),  # 14 stem base, left
]).T
N = TEMPLATE.shape[1]
STEM = (0, 1)  # the stem's tip points
SKIP_SETS = 3  # ring springs to the 1st .. (SKIP_SETS + 1)th neighbour
# stem braces past the ring's reach: each tip point to the far sides (the ring already ties it to the stem base,
# both shoulders and the upper sides, the across springs to the bottom)
BRACES = ((0, 5), (0, 6), (1, 10), (1, 11))
# spring stiffness (SoftBody's): firmer than the other shapes' 0.1. With 15 points and ~11 springs per point, each
# spring is softer (solver.spring_compliance divides by the count), and at 0.1 a pumpkin in a pile ended dented to
# 0.6 of its area (polygons: 0.7-0.8); at 0.3 it ends at 0.86+, rounder than the polygons, and its stem stands
STIFFNESS = 0.3
SIZES = (10.0, 11.0)  # half width range (world units) of the spawned pumpkins (a map's spawn may give its own)
EYES = np.array([(-0.34, -0.2), (0.34, -0.2)]).T  # eye centres (template units): on the face, under the stem
EYE_RADIUS = 0.17  # x the half width


def _springs():
    n, pairs = N, set()
    for i in range(n):
        for k in range(1, SKIP_SETS + 2):
            pairs.add(tuple(sorted((i, (i + k) % n))))
        for k in (n // 2, n - n // 2):  # across the body
            pairs.add(tuple(sorted((i, (i + k) % n))))
    pairs.update(tuple(sorted(p)) for p in BRACES)
    return np.array(sorted(p for p in pairs if p[0] != p[1]), dtype=int).T


SPRINGS = _springs()


def outline(cx, cy, size, tilt=0.0):
    """(2, N) points of a pumpkin of half width `size` centred on (cx, cy), turned `tilt` radians (clockwise on
    screen; 0: stem up)."""
    c, s = math.cos(tilt), math.sin(tilt)
    x, y = TEMPLATE * size
    return np.array([cx + c * x - s * y, cy + s * x + c * y])


def make(cx, cy, size, tilt=0.0, stiffness=STIFFNESS, **style):
    """A pumpkin SoftBody (see the module doc); style: Body keyword arguments (colours, point_radius)."""
    body = SoftBody(outline(cx, cy, size, tilt), stiffness=stiffness, skip_sets=SKIP_SETS, **style)
    set_springs(body)
    body.shape_kind = KIND
    return body


def set_springs(body):
    """Give a SoftBody built from a pumpkin outline the pumpkin's springs, rest lengths from its points now."""
    body.springs = body.draw_edges = SPRINGS.copy()
    i, j = body.springs
    body.rest_len = np.hypot(*(body.points[:, j] - body.points[:, i]))
    body.spring_count = np.maximum(np.bincount(body.springs.ravel(), minlength=N), 1)


def is_pumpkin(body):
    return getattr(body, "shape_kind", None) == KIND and np.shape(body.points)[1] == N


def size_of(body):
    """A pumpkin's half width at rest, from its spring rest lengths (so squashing it doesn't change it)."""
    rest = np.asarray(body.rest_len, dtype=float)
    if rest.shape != (SPRINGS.shape[1],):
        pts = np.asarray(body.points, dtype=float)
        return float(np.ptp(pts[0])) / 2
    i, j = SPRINGS
    return float(rest.sum() / np.hypot(*(TEMPLATE[:, j] - TEMPLATE[:, i])).sum())


def self_gap(points=TEMPLATE):
    """Smallest distance from a point to one of its own outline's edges that it may collide with (not its own or
    its neighbours': solver.World._filter_pairs' contact rule), in the units of `points`."""
    n, best = points.shape[1], np.inf
    for i in range(n):
        p = points[:, i]
        for a in range(n):
            b = (a + 1) % n
            if min((i - a) % n, (a - i) % n) <= 1 or min((i - b) % n, (b - i) % n) <= 1:
                continue
            u, v = points[:, a], points[:, b]
            t = np.clip(np.dot(p - u, v - u) / max(np.dot(v - u, v - u), 1e-12), 0.0, 1.0)
            best = min(best, float(np.hypot(*(p - u - t * (v - u)))))
    return best


MIN_GAP = self_gap()
REACH = float(np.hypot(*TEMPLATE).max())  # the outline's farthest point from its centre (the stem's tip)
