"""Looping (World.looping) and the world box walls it replaces.

World.looping = {"left": bool, "right": bool, "top": bool, "bottom": bool}. A body that leaves the world rectangle
World.bounds = (x0, y0, x1, y1) fully through a side whose looping is on, while moving outward through it, comes
back in through the opposite side: same other coordinate (bottom -> top keeps x, right -> left keeps y) and the same
velocities. It is placed just outside the opposite edge (fully off screen, so it slides in), or, if the opposite
side still has its wall, just inside that wall so it isn't trapped outside the box. Works for any gravity direction:
only position + direction of motion decide.

World box walls: add_world_box gives a World four 2-point FixedBody walls, tagged `wall_side` and kept in
World.walls[side] whether or not they are currently in World.bodies. Each wall runs `ext` past both ends of its
side (a "#" frame, ends off screen), so bodies passing through an open side's off-screen band stay between the
other walls. set_looping(world, side, True) removes that side's wall right away. set_looping(world, side, False)
re-adds it only once no movable body lies beyond or straddles its line (every point at least contact_radius inside);
until then the side is "pending": looping[side] already reads False, the wall is retried every frame, and wrapping
through that side keeps working meanwhile so whatever is out there can come back. Returns True if applied now.

No-intersection guarantee: World.step calls loop_step once per frame, before the substeps, so a wrap is never part
of a substep's straight-line motion that CCD checks. A body is only moved into a spot with no edge crossings, no
containment and every point at least clearance() from other bodies' edges (and theirs from its edges). If the spot
is taken, up to STACK further spots (one body size + clearance apart, away from the screen, or further inside when
placing inside a wall) are tried; if all are taken the body is left where it is and retried next frame.

Ropes / chains: bodies tied together by ropes form a group that wraps together by one offset (rope lengths
unchanged), once the whole group is past the edge. A group with any fixed end (rope to a Hook, (x, y) anchor or
FixedBody) never wraps, nor does the group holding the dragged point. Fixed bodies never wrap.
"""
import numpy as np

from entities import EnvConfig, FixedBody

SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
STACK = 4  # destination spots tried per wrap


def clearance():
    """Minimum gap (world units) between a wrapped body and anything else at its destination."""
    return 2 * EnvConfig.contact_radius


# ---------------- World box walls
def add_world_box(world, width, height, inset, col=(100, 100, 100), ext=None):
    """Give `world` its bounds (0, 0, width, height) and four tagged wall segments `inset` in from each edge,
    each running `ext` (default max(width, height)) past the box corners. Walls go first in World.bodies (drawn
    under everything); sides already looping get no wall."""
    ext = max(width, height) if ext is None else ext
    lo_x, hi_x, lo_y, hi_y = inset, width - inset, inset, height - inset
    ends = {"left": [[lo_x, lo_x], [lo_y - ext, hi_y + ext]], "right": [[hi_x, hi_x], [lo_y - ext, hi_y + ext]],
            "top": [[lo_x - ext, hi_x + ext], [lo_y, lo_y]], "bottom": [[lo_x - ext, hi_x + ext], [hi_y, hi_y]]}
    world.bounds = (0.0, 0.0, float(width), float(height))
    for side in SIDES:
        wall = FixedBody(np.array(ends[side], dtype=float), point_col=col, stick_col=col, point_radius=1, stick_width=2,
                         fill_col=col)
        wall.wall_side = side
        world.walls[side] = wall
    present = [world.walls[s] for s in SIDES if not world.looping[s]]
    _edit_bodies(world, lambda: world.bodies.__setitem__(slice(0, 0), present))


def _edit_bodies(world, mutate):
    """Run mutate() on world.bodies, repack, and keep a held grab on the same point."""
    ref = None
    if world.grabbed is not None and world.bodies:
        k = int(world.point_body[world.grabbed])
        ref = (world.bodies[k], world.grabbed - int(world.offsets[k]))
    mutate()
    if world.bodies:
        world._rebuild()
    world.grabbed = None
    if ref is not None and any(b is ref[0] for b in world.bodies):
        k = next(i for i, b in enumerate(world.bodies) if b is ref[0])
        world.grabbed = int(world.offsets[k]) + ref[1]


def _wall_line(wall):
    """(axis, coordinate) of a wall's line: axis 0 for a vertical wall (x = c), 1 for a horizontal one (y = c)."""
    return (0, float(wall.points[0, 0])) if wall.wall_side in ("left", "right") else (1, float(wall.points[1, 0]))


def wall_clear(world, side):
    """True when every movable body lies fully inside `side`'s wall line, each point at least contact_radius
    from it (so the wall can be added without touching or crossing anything)."""
    axis, c = _wall_line(world.walls[side])
    movable = world.W > 0 if world.bodies else np.zeros(0, bool)
    coords = world.P[axis, movable] if world.bodies else np.empty(0)
    gap = EnvConfig.contact_radius
    if side in ("left", "top"):
        return bool(np.all(coords >= c + gap))
    return bool(np.all(coords <= c - gap))


def set_looping(world, side, on):
    """Turn looping for one side on (removes its wall now) or off (re-adds its wall when clear, see module doc).
    Returns True if the wall change happened now, False if it's pending."""
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}, expected one of {SIDES}")
    world.looping[side] = bool(on)
    wall = world.walls.get(side)
    if on:
        world.pending_walls.discard(side)
        if wall is not None and any(b is wall for b in world.bodies):
            _edit_bodies(world, lambda: world.bodies.remove(wall))
        return True
    if wall is None or any(b is wall for b in world.bodies):
        world.pending_walls.discard(side)
        return True
    world.pending_walls.add(side)
    return _try_pending_walls(world, only=side)


def _try_pending_walls(world, only=None):
    """Add the pending walls that are clear now. Returns whether `only` (if given) got its wall."""
    done = False
    for side in [s for s in SIDES if s in world.pending_walls and (only is None or s == only)]:
        if wall_clear(world, side):
            world.pending_walls.discard(side)
            wall = world.walls[side]
            _edit_bodies(world, lambda: world.bodies.insert(0, wall))
            done = done or side == only
    return done


# ---------------- Wrapping
def rope_groups(world):
    """{root: [body indices]} of movable bodies tied together by ropes (singletons for free bodies), and the set of
    roots of groups that are anchored (a rope end on a hook / world anchor / fixed body)."""
    index = {id(b): k for k, b in enumerate(world.bodies)}
    parent = list(range(len(world.bodies)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    anchored_bodies = set()
    for rope in world.ropes:
        ends = [index.get(id(body)) if body is not None else None for body, _ in rope.ends]
        movable = [k for k in ends if k is not None and not world.bodies[k].fixed]
        if len(movable) == 2:
            parent[find(movable[0])] = find(movable[1])
        else:  # the other end is a hook, a world anchor or a fixed body
            anchored_bodies.update(movable)
    groups = {}
    for k, b in enumerate(world.bodies):
        if not b.fixed:
            groups.setdefault(find(k), []).append(k)
    anchored = {root for root, members in groups.items() if any(k in anchored_bodies for k in members)}
    return groups, anchored


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _segments(points):
    """(a, b) endpoint arrays (2, m) of a body's outline: one segment for a 2-point body, else the closed loop."""
    if points.shape[1] == 2:
        return points[:, :1], points[:, 1:]
    return points, np.roll(points, -1, axis=1)


def _point_seg_dist(p, a, b):
    """(n, m) distances from points p (2, n) to segments a -> b (2, m)."""
    ab = b - a
    ap = p[:, :, None] - a[:, None, :]
    t = np.clip(np.sum(ap * ab[:, None, :], axis=0) / np.maximum(np.sum(ab * ab, axis=0), 1e-12), 0.0, 1.0)
    d = ap - t[None] * ab[:, None, :]
    return np.hypot(d[0], d[1])


def _segs_cross(a, b, c, d):
    """Any proper crossing between segments a -> b (2, n) and c -> d (2, m). Touching / collinear overlaps put an
    endpoint on the other segment, which the distance check in bodies_clear already rejects."""
    ab, cd = (b - a)[:, :, None], (d - c)[:, None, :]
    ac, ad = c[:, None, :] - a[:, :, None], d[:, None, :] - a[:, :, None]
    ca, cb = a[:, :, None] - c[:, None, :], b[:, :, None] - c[:, None, :]
    return bool(np.any((_cross(ab, ac) * _cross(ab, ad) < 0) & (_cross(cd, ca) * _cross(cd, cb) < 0)))


def _inside(pt, poly):
    """Even-odd point in polygon; poly (2, n) closed loop."""
    x, y = pt
    px, py = poly
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    straddle = (py > y) != (qy > y)
    xc = px + (y - py) * (qx - px) / np.where(straddle, qy - py, 1.0)
    return np.count_nonzero(straddle & (x < xc)) % 2 == 1


def bodies_clear(pts_a, pts_b, gap):
    """True when two bodies (2, n) / (2, m) neither cross nor contain each other and every point of each is at least
    `gap` from the other's edges."""
    a0, a1 = _segments(pts_a)
    b0, b1 = _segments(pts_b)
    if _point_seg_dist(pts_a, b0, b1).min() < gap or _point_seg_dist(pts_b, a0, a1).min() < gap:
        return False
    if _segs_cross(a0, a1, b0, b1):
        return False
    if pts_a.shape[1] > 2 and pts_b.shape[1] > 2 and (_inside(pts_a[:, 0], pts_b) or _inside(pts_b[:, 0], pts_a)):
        return False
    return True


def spot_free(world, members, shift, gap=None):
    """Whether the bodies `members` (indices), translated by shift (dx, dy), are clear of every other body."""
    gap = clearance() if gap is None else gap
    member_set = set(members)
    d = np.reshape(np.asarray(shift, dtype=float), (2, 1))
    moved = [world.bodies[k].points + d for k in members]
    lo = np.min([m.min(axis=1) for m in moved], axis=0) - gap
    hi = np.max([m.max(axis=1) for m in moved], axis=0) + gap
    starts = world.offsets[:-1]
    o_lo = np.stack([np.minimum.reduceat(world.P[k], starts) for k in range(2)])
    o_hi = np.stack([np.maximum.reduceat(world.P[k], starts) for k in range(2)])
    near = np.flatnonzero(np.all(o_lo <= hi[:, None], axis=0) & np.all(o_hi >= lo[:, None], axis=0)
                          & (world.body_counts > 0))  # boxes within the gap of the moved group's box
    for k in near:
        if k in member_set:
            continue
        other = world.bodies[k]
        if not all(bodies_clear(m, other.points, gap) for m in moved):
            return False
    return True


def _exit_side(lo, hi, vel, bounds, active):
    """The side a group with box lo..hi and mean velocity vel has fully left through while moving outward, among
    the `active` sides, or None."""
    x0, y0, x1, y1 = bounds
    if "bottom" in active and lo[1] > y1 and vel[1] > 0:
        return "bottom"
    if "top" in active and hi[1] < y0 and vel[1] < 0:
        return "top"
    if "right" in active and lo[0] > x1 and vel[0] > 0:
        return "right"
    if "left" in active and hi[0] < x0 and vel[0] < 0:
        return "left"
    return None


def _spots(world, entry, lo, hi, gap):
    """Candidate translations along the entry side's axis that put the box lo..hi just past the `entry` side:
    outside the screen edge when that side has no wall, else just inside its wall; then further from the screen
    (or deeper inside) one box size + gap at a time."""
    axis = 0 if entry in ("left", "right") else 1
    size = hi[axis] - lo[axis] + gap
    wall = world.walls.get(entry)
    walled = wall is not None and any(b is wall for b in world.bodies)
    near_side = entry in ("left", "top")  # the entry edge is at the low coordinate
    if walled:
        c = _wall_line(wall)[1]
        base = (c + gap - lo[axis]) if near_side else (c - gap - hi[axis])
        step = size if near_side else -size  # deeper into the box
    else:
        edge = world.bounds[axis] if near_side else world.bounds[axis + 2]
        base = (edge - hi[axis]) if near_side else (edge - lo[axis])
        step = -size if near_side else size  # further off screen
    out = []
    for k in range(STACK):
        d = np.zeros(2)
        d[axis] = base + k * step
        out.append(d)
    return out


def loop_step(world):
    """One looping pass (World.step calls it before each frame's substeps): add pending walls that are clear, then
    wrap every group that fully left through a looping side. Moves P and PREV only; velocities untouched.
    Returns [(body, (dx, dy)), ...] for the bodies moved, e.g. so motion trails don't draw the jump."""
    if world.pending_walls:
        _try_pending_walls(world)
    active = {s for s in SIDES if world.looping[s] or s in world.pending_walls}
    if not active or not world.bodies or world.bounds is None:
        return []
    # cheap vectorized pre-check: is any movable body fully outside the bounds at all?
    starts = world.offsets[:-1]
    nonempty = world.body_counts > 0
    box_lo = np.stack([np.minimum.reduceat(world.P[k], starts) for k in range(2)])  # (2, bodies)
    box_hi = np.stack([np.maximum.reduceat(world.P[k], starts) for k in range(2)])
    x0, y0, x1, y1 = world.bounds
    outside = nonempty & ~world.body_fixed & ((box_lo[0] > x1) | (box_hi[0] < x0) | (box_lo[1] > y1) | (box_hi[1] < y0))
    if not outside.any():
        return []
    groups, anchored = rope_groups(world)
    grabbed_body = None if world.grabbed is None else int(world.point_body[world.grabbed])
    gap = clearance()
    wraps = []
    for root, members in groups.items():
        if root in anchored or grabbed_body in members or not all(outside[k] for k in members):
            continue
        idx = np.concatenate([np.arange(world.offsets[k], world.offsets[k + 1]) for k in members])
        pts = world.P[:, idx]
        lo, hi = pts.min(axis=1), pts.max(axis=1)
        side = _exit_side(lo, hi, world.V[:, idx].mean(axis=1), world.bounds, active)
        if side is None:
            continue
        for d in _spots(world, OPPOSITE[side], lo, hi, gap):
            if spot_free(world, members, d, gap):
                world.P[:, idx] += d[:, None]
                world.PREV[:, idx] += d[:, None]
                wraps.extend((world.bodies[k], (float(d[0]), float(d[1]))) for k in members)
                break
        # every spot taken: leave the group where it is and try again next frame
    return wraps
