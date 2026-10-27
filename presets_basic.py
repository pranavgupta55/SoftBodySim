"""Simple preset worlds for scene.PRESETS: a blank box, a box with random wall segments, and an open world that
loops on all four sides. Each builder takes (world_width, world_height) in world units and returns a World that
starts intersection-free (validate_layout checks it).

scene imports this module, so the palette and world box helper are imported inside the builders.
"""
import math

import numpy as np

from entities import EnvConfig, FixedBody, RigidBody, SoftBody, polygon_points, rect_points
from solver import World

SIDES = ("left", "right", "top", "bottom")


def _set_looping(world, sides):
    """Initial looping for a preset: the sides in `sides` on, the rest off."""
    for side in SIDES:
        world.set_looping(side, side in sides)


def _seg_cross(a, b, c, d):
    """True when segments ab and cd properly cross (touching at an endpoint or overlapping collinearly is not)."""
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    return cr(a, b, c) * cr(a, b, d) < 0 and cr(c, d, a) * cr(c, d, b) < 0


def _point_seg_dist(p, a, b):
    ab, ap = b - a, p - a
    t = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(ap - t * ab)))


def _seg_dist(s, t):
    """Shortest distance between two segments, each a (2, 2) array of endpoints as rows."""
    if _seg_cross(s[0], s[1], t[0], t[1]):
        return 0.0
    return min(_point_seg_dist(s[0], *t), _point_seg_dist(s[1], *t), _point_seg_dist(t[0], *s), _point_seg_dist(t[1], *s))


def random_wall_segments(world_width, world_height, seed=0, inset=5, length=(60.0, 130.0), gap=70.0, margin=40.0,
                         crossings=None):
    """Short random wall segments spread over the box, as a list of (2, 2) endpoint arrays (rows = points).

    Segments stay `margin` world units inside the world box walls, out of a clear band at the top (room to spawn or
    drop things in) and one at the bottom (room to land). Segments that don't cross stay at least `gap` apart, so
    bodies smaller than that can fall between any two. Exactly `crossings` pairs cross (default: 2 or 3, picked by
    the seed), each an X made by adding a segment through the middle of an existing one.
    """
    rng = np.random.default_rng(seed)
    if crossings is None:
        crossings = int(rng.integers(2, 4))
    x0, x1 = inset + margin, world_width - inset - margin
    y0, y1 = inset + 0.18 * world_height, world_height - inset - 90.0
    target = int(np.clip((x1 - x0) * (y1 - y0) / 55000.0, 6, 24))

    def inside(seg):
        return bool(np.all((seg[:, 0] >= x0) & (seg[:, 0] <= x1) & (seg[:, 1] >= y0) & (seg[:, 1] <= y1)))

    def make(center, angle, ln, s=0.5):
        """Segment of length ln along angle, with `center` at fraction s along it."""
        u = np.array([math.cos(angle), math.sin(angle)])
        return np.array([center - s * ln * u, center + (1 - s) * ln * u])

    segs, angles = [], []
    for _ in range(4000):  # spread-out, non-crossing segments by rejection sampling
        if len(segs) >= target:
            break
        angle = rng.uniform(0, math.pi)
        seg = make(np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)]), angle, rng.uniform(*length))
        if inside(seg) and all(_seg_dist(seg, t) >= gap for t in segs):
            segs.append(seg)
            angles.append(angle)

    crossed = set()
    for _ in range(4000):  # X shapes: a new segment through the middle of an existing, not yet crossed one
        if len(crossed) >= crossings:
            break
        free = [k for k, a in enumerate(angles) if a is not None and k not in crossed]
        if not free:
            break
        k = int(rng.choice(free))
        base = segs[k]
        center = base[0] + rng.uniform(0.3, 0.7) * (base[1] - base[0])
        angle = angles[k] + rng.choice([-1, 1]) * rng.uniform(math.radians(45), math.radians(135))
        seg = make(center, angle, rng.uniform(*length), rng.uniform(0.3, 0.7))
        if inside(seg) and all(_seg_dist(seg, t) >= gap for j, t in enumerate(segs) if j != k):
            segs.append(seg)
            crossed.add(k)
            angles.append(None)  # a crosser is never crossed again
    return segs


def _style(col):
    return dict(point_col=col, stick_col=col, fill_col=col)


def _drop_bodies(world_width, world_height, inset, cols):
    """A few small soft and rigid bodies spread along the top of the world, at rest."""
    y = inset + 0.09 * world_height
    shapes = [
        lambda x: SoftBody(polygon_points(x, y, 18, 12), stiffness=0.1, skip_sets=2, point_radius=3, **_style(cols[0])),
        lambda x: RigidBody(rect_points(x, y, 36, 36), point_radius=3, **_style(cols[1])),
        lambda x: SoftBody(polygon_points(x, y, 20, 6), stiffness=0.1, point_radius=3, **_style(cols[2])),
        lambda x: RigidBody(polygon_points(x, y, 22, 3, angle=math.pi / 2), point_radius=3, **_style(cols[3])),
        lambda x: SoftBody(polygon_points(x, y, 16, 10), stiffness=0.05, skip_sets=2, point_radius=3, **_style(cols[4])),
    ]
    xs = np.linspace(0.15, 0.85, len(shapes)) * world_width
    return [shape(x) for shape, x in zip(shapes, xs)]


def build_blank(world_width, world_height):
    """A completely blank world: just the four world box walls."""
    from scene import Endesga, INSET, add_world_box
    world = World()
    _set_looping(world, ())
    add_world_box(world, world_width, world_height, INSET, col=Endesga.greyD)
    return world


def build_random_walls(world_width, world_height, seed=0, bodies=True):
    """The world box plus short random fixed wall segments (2 or 3 crossing pairs) and, with `bodies`, a few soft and
    rigid bodies along the top that fall through them."""
    from scene import Endesga, INSET, add_world_box
    world = World()
    _set_looping(world, ())
    add_world_box(world, world_width, world_height, INSET, col=Endesga.greyD)
    for seg in random_wall_segments(world_width, world_height, seed=seed, inset=INSET):
        world.add(FixedBody(seg.T, point_radius=2, stick_width=2, **_style(Endesga.steel_grey)))
    if bodies:
        cols = [Endesga.cream, Endesga.debug_red, Endesga.network_green, Endesga.sky_blue, Endesga.orchid]
        for body in _drop_bodies(world_width, world_height, INSET, cols):
            world.add(body)
    return world


def build_looping(world_width, world_height):
    """No walls and every side looping: a few bodies fall forever (reaching terminal velocity), some drifting
    sideways so the left / right wrap shows too."""
    from scene import Endesga, INSET, add_world_box
    world = World()
    _set_looping(world, SIDES)
    add_world_box(world, world_width, world_height, INSET, col=Endesga.greyD)  # bounds; no walls, every side loops
    cols = [Endesga.cream, Endesga.debug_red, Endesga.network_green, Endesga.sky_blue, Endesga.orchid]
    drift = [-150.0, 0.0, 120.0, -60.0, 200.0]  # px/s sideways
    for body, vx in zip(_drop_bodies(world_width, world_height, INSET, cols), drift):
        body.vels[0] = vx
        world.add(body)
    return world


FLUID_COUNT = 675  # dodecagons in the "fluid" preset
FLUID_SIDES = 12  # their side count (springs: every point to every other, the most a shape can have)
FLUID_DEPTH = 0.4  # the share of the world's height they fill, from the bottom
FLUID_BLUES = ([18, 60, 130], [90, 200, 240])  # darkest and lightest shade: each dodecagon is a random mix
FLUID_FRICTION = 0.05  # slippery, so the pile flows (this world's own setting, World.env)


def build_fluid(world_width, world_height, count=FLUID_COUNT, seed=0):
    """A few hundred small soft dodecagons in shades of blue, packed into the bottom FLUID_DEPTH of the world, with low
    friction so they slosh around like a fluid. Left / right looping is on (no side walls), so the x axis is periodic
    (looping.py): the pile is one continuous ring, dodecagons near one edge press on those near the other, and what
    flows out one side comes in the other at the same height and speed."""
    from scene import Endesga, INSET, add_world_box
    world = World()
    add_world_box(world, world_width, world_height, INSET, col=Endesga.greyD)
    _set_looping(world, ("left", "right"))
    bodies = []  # added in one go at the end (one repack instead of one per body)
    world.env["friction"] = FLUID_FRICTION
    rng = np.random.default_rng(seed)
    x0, x1 = INSET, world_width - INSET
    y1 = world_height - INSET
    depth = FLUID_DEPTH * world_height
    # hex packing of circles (radius R, one per dodecagon) that fits `count` in the box: rows sqrt(3) R apart
    R = math.sqrt((x1 - x0) * depth / (2 * math.sqrt(3) * count))
    r = R - 3.0  # each dodecagon's circumradius: a 6 unit gap to its neighbours and the walls (> contact_radius)
    lo, hi = np.array(FLUID_BLUES[0], dtype=float), np.array(FLUID_BLUES[1], dtype=float)
    cols = int((x1 - x0) // (2 * R))
    left = x0 + ((x1 - x0) - cols * 2 * R) / 2  # center the rows (looping sides: no walls to touch)
    row = 0
    while y1 - R - row * math.sqrt(3) * R > y1 - depth + R:
        cy = y1 - R - row * math.sqrt(3) * R
        shift = R if row % 2 else 0.0
        for k in range(cols - (1 if row % 2 else 0)):
            cx = left + R + shift + k * 2 * R
            col = [int(c) for c in lo + rng.uniform() * (hi - lo)]
            size = r * rng.uniform(0.85, 1.0)  # not all the same size, so they don't lock into a crystal
            angle = rng.uniform(0, 2 * math.pi / FLUID_SIDES)
            bodies.append(SoftBody(polygon_points(cx, cy, size, FLUID_SIDES, angle=angle), stiffness=0.1,
                                   skip_sets=FLUID_SIDES // 2 - 1, point_radius=2, **_style(col)))
        row += 1
    world.add_many(bodies)
    return world


def validate_layout(world):
    """Raise ValueError unless the world is intersection-free: no movable body's edge crosses another body's edge
    (or its own non-neighbouring edges), no movable body sits inside another, and no point of a movable body is
    within contact_radius of a fixed edge. Fixed bodies may cross each other (they never collide)."""
    r = EnvConfig.contact_radius
    bodies = world.bodies

    def segs(b):
        p = b.points.T
        return [(p[i], p[j]) for i, j in b.edges.T]

    def inside(pt, b):
        if b.points.shape[1] < 3:
            return False
        x, y = b.points
        qx, qy = np.roll(x, -1), np.roll(y, -1)
        st = (y > pt[1]) != (qy > pt[1])
        xc = x + (pt[1] - y) * (qx - x) / np.where(st, qy - y, 1.0)
        return np.count_nonzero(st & (pt[0] < xc)) % 2 == 1

    for i, a in enumerate(bodies):
        sa = segs(a)
        if a.self_collide:
            n = len(sa)
            for k in range(n):
                for m in range(n):
                    if min((k - m) % n, (m - k) % n) > 1 and _seg_cross(*sa[k], *sa[m]):
                        raise ValueError(f"body {i} crosses itself")
        for j in range(i + 1, len(bodies)):
            b = bodies[j]
            if a.fixed and b.fixed:
                continue
            sb = segs(b)
            if any(_seg_cross(*s, *t) for s in sa for t in sb):
                raise ValueError(f"bodies {i} and {j} cross")
            if inside(a.points[:, 0], b) or inside(b.points[:, 0], a):
                raise ValueError(f"bodies {i} and {j} overlap")
            for mov, fix in ((a, b), (b, a)):
                if fix.fixed and not mov.fixed:
                    if any(_point_seg_dist(p, *s) < r for p in mov.points.T for s in segs(fix)):
                        raise ValueError(f"bodies {i} and {j} start in contact")


PRESETS = {
    "blank": build_blank,
    "random walls": build_random_walls,
    "looping": build_looping,
    "fluid": build_fluid,
}
