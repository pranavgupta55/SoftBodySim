"""Looping (World.looping) and the world box walls it replaces.

World.looping = {"left": bool, "right": bool, "top": bool, "bottom": bool}; World.bounds = (x0, y0, x1, y1). A side
is "active" when its looping is on or its wall is pending (see below). Each axis (left / right = x, top / bottom = y)
is in one of two modes:

Periodic (both sides of the axis active): the world repeats along that axis with period L = x1 - x0 (or y1 - y0).
Contacts, CCD (time of impact, impulses, inelastic projection, impact zones) and the broad phases treat it as a
torus: a point near one edge meets edges near the opposite edge through the minimum-image offset (+-L), see
World._image_pairs. Wrapping is then a pure translation by exactly one period: a rope group (below) whose bounding
box centre has left [x0, x1) (or [y0, y1)) moves by the whole periods that bring the centre back; the other
coordinate and every velocity are kept, nothing is searched or lifted, and coordinates stay within half a group's
size of the bounds. On a torus a translation by L changes no distance, so the no-intersection guarantee holds
through it. An axis only becomes periodic once that is true of the state too (_images_clear: no movable body
crosses or holds another body's image, e.g. bodies parked off screen by the walled mode); until then it stays
walled-mode and the check is retried every RETRY_FRAMES frames. World.period is (Lx, Ly) (0 on an axis that isn't
periodic), or None when neither is: then nothing in the solver changes at all.

Walled mode (an axis with at most one side active, i.e. no looping, or mixed: one side looping, the other walled):
exactly the behaviour from before periodic collisions. A body that leaves the world rectangle fully through an
active side, while moving outward through it, comes back in through the opposite side: same other coordinate and
the same velocities, placed just outside the opposite edge (fully off screen, so it slides in), or, if the opposite
side still has its wall, just inside that wall so it isn't trapped outside the box; only into a spot with clearance
from every other body (spot_free, which also looks across a periodic other axis), trying further spots (and for left
/ right a few lifted ones) if it's taken, else waiting.

World box walls: add_world_box gives a World four 2-point FixedBody walls, tagged `wall_side` and kept in
World.walls[side] whether or not they are currently in World.bodies. Each wall runs `ext` past both ends of its
side (a "#" frame, ends off screen). The walls of a walled axis are parallel to the other axis: a periodic axis
never pairs anything with them through an image (they run past its seams, so the direct contact is the same one),
see World._image_topology. set_looping(world, side, True) removes that side's wall right away.
set_looping(world, side, False) re-adds it only once no movable body lies beyond or straddles its line (every point
at least contact_radius inside; the same rule in both modes); until then the side is "pending": looping[side]
already reads False, the wall is retried every frame, and the side stays active meanwhile (wrapping through it, and
a periodic axis stays periodic) so whatever is out there can come back. On a periodic axis whose other side still
loops, free groups reaching within contact_radius of the pending wall's line are moved one period over to the
looping side (_periodic_wraps), so the wall comes back within a frame or two (anchored groups don't move and can
keep it pending). When the wall goes in, that axis drops to walled mode (mixed). Returns True if applied now.

Ropes / chains: bodies tied together by ropes form a group that wraps together by one offset (rope lengths
unchanged). A group with any fixed end (rope to a Hook, (x, y) anchor or FixedBody) never wraps: on a periodic axis
it still collides across the seam, it just keeps its coordinates. Fixed bodies never wrap. The group holding the
dragged point wraps like any other on a periodic axis, and World.grab_shift / grab_target move with it so the hand
keeps pulling toward the cursor's image in the body's frame; in walled mode it never wraps (as before).

World.step calls loop_step once per frame, before the substeps, so a wrap is never part of a substep's straight-line
motion that CCD checks. On a periodic axis World._substeps also wraps between the frame's CCD spans (span_wraps): at
high time scales one frame holds many spans, and a fast body could otherwise get more than a period past the bounds,
where the +-L image pairs no longer reach what it should hit. Each span ends intersection-free, so this is the same
safe translation as the per-frame one.
"""
import numpy as np

from entities import EnvConfig, FixedBody

SIDES = ("left", "right", "top", "bottom")
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
REACH = 0.5  # destination spots are searched up to this fraction of the band's length past the first one
MAX_SPOTS = 64
LIFT_SPOTS = 16  # a left / right wrap with no free spot level with the body also tries this many spots stacked above
RETURN_SPEED = 5.0  # px/s: a group fully past a left / right edge wraps unless it's heading back in faster than this
RETRY_FRAMES = 8  # a group with no free spot at all waits this many frames before searching again (and an axis
                  # that can't turn periodic yet waits this long before _images_clear checks again)
AXIS_SIDES = (("left", "right"), ("top", "bottom"))
AXIS_OF = {"left": 0, "right": 0, "top": 1, "bottom": 1}


def clearance():
    """Minimum gap (world units) between a wrapped body and anything else at its destination."""
    return 2 * EnvConfig.contact_radius


# ---------------- Periodic axes
def _active(world, side):
    """Looping on, or its wall pending (the side still wraps)."""
    return bool(world.looping[side]) or side in world.pending_walls


def wanted_axes(world):
    """(x, y): whether each axis has both sides active, i.e. should be periodic (once _images_clear allows it)."""
    if world.bounds is None:
        return (False, False)
    return tuple(all(_active(world, s) for s in sides) for sides in AXIS_SIDES)


def periods(world):
    """(Lx, Ly): the bounds' width and height."""
    x0, y0, x1, y1 = world.bounds
    return np.array([x1 - x0, y1 - y0], dtype=float)


def image_shifts(period):
    """The nonzero image offsets (2,) for a World.period (Lx, Ly; 0 = that axis isn't periodic), in a fixed order:
    +Lx, -Lx, +Ly, -Ly, then the four diagonal ones (only when both axes are periodic). [] for None."""
    if period is None:
        return []
    kx = (1, -1) if period[0] else ()
    ky = (1, -1) if period[1] else ()
    combos = [(a, 0) for a in kx] + [(0, b) for b in ky] + [(a, b) for a in kx for b in ky]
    return [np.array([a * period[0], b * period[1]], dtype=float) for a, b in combos]


def _set_periodic(world, axes):
    world.periodic = tuple(bool(a) for a in axes)
    world.period = np.where(world.periodic, periods(world), 0.0) if any(world.periodic) else None


def clear_periodic(world):
    """Leave periodic mode on both axes (looping off everywhere, or a new scene)."""
    world.periodic, world.period = (False, False), None


def is_world_wall(world, body):
    """One of the world box walls (World.walls), whether or not it's in World.bodies right now."""
    side = getattr(body, "wall_side", None)
    return side is not None and world.walls.get(side) is body


def _images_clear(world, period, axis):
    """Is the state intersection-free on the torus `period` across `axis` (0 / 1)? No movable body, moved by an
    image offset that wraps that axis, may cross or contain another body (gap 0: plain intersection). Bodies that
    merely touch across the seam are fine (CCD separates them). World walls never pair through images."""
    shifts = [s for s in image_shifts(period) if s[axis] != 0]
    if not shifts or not world.bodies:
        return True
    lo, hi = body_boxes(world)
    nonempty = world.body_counts > 0
    movable = nonempty & ~world.body_fixed
    other = nonempty & ~np.array([is_world_wall(world, b) for b in world.bodies])
    mk, ok = np.flatnonzero(movable), np.flatnonzero(other)
    if mk.size == 0 or ok.size == 0:
        return True
    for s in shifts:
        slo, shi = lo[:, mk] + s[:, None], hi[:, mk] + s[:, None]
        hit = np.all((slo[:, :, None] <= hi[:, None, ok]) & (shi[:, :, None] >= lo[:, None, ok]), axis=0)
        for a, b in zip(*np.nonzero(hit)):
            k, j = int(mk[a]), int(ok[b])
            if k != j and not bodies_clear(world.bodies[k].points + s[:, None], world.bodies[j].points, 0.0):
                return False
    return True


def _update_periodic(world):
    """Set World.periodic / World.period for this frame: an axis whose two sides are no longer both active leaves
    periodic mode at once; one that newly has both active enters it once _images_clear says the state allows it."""
    want = wanted_axes(world)
    have = getattr(world, "periodic", (False, False))
    axes = [h and w for h, w in zip(have, want)]
    new = [a for a in (0, 1) if want[a] and not have[a]]
    if new:
        if world.__dict__.get("_periodic_wait", 0) > 0:
            world._periodic_wait -= 1
        else:
            for a in new:
                trial = list(axes)
                trial[a] = True
                if _images_clear(world, np.where(trial, periods(world), 0.0), a):
                    axes = trial
            if any(not axes[a] for a in new):
                world._periodic_wait = RETRY_FRAMES
    _set_periodic(world, axes)


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


def body_boxes(world):
    """Every body's bounding box: (lo, hi), each (2, bodies)."""
    starts = world.offsets[:-1]
    return (np.stack([np.minimum.reduceat(world.P[k], starts) for k in range(2)]),
            np.stack([np.maximum.reduceat(world.P[k], starts) for k in range(2)]))


def spot_free(world, members, shift, gap=None, boxes=None):
    """Whether the bodies `members` (indices), translated by shift (dx, dy), are clear of every other body, and of
    its images across a periodic axis (World.period). boxes: body_boxes(world), if the caller already has them (they
    must be current)."""
    gap = clearance() if gap is None else gap
    member_set = set(members)
    d = np.reshape(np.asarray(shift, dtype=float), (2, 1))
    moved = [world.bodies[k].points + d for k in members]
    lo = np.min([m.min(axis=1) for m in moved], axis=0) - gap
    hi = np.max([m.max(axis=1) for m in moved], axis=0) + gap
    o_lo, o_hi = body_boxes(world) if boxes is None else boxes
    for image in [None] + image_shifts(getattr(world, "period", None)):
        s = np.zeros((2, 1)) if image is None else image[:, None]
        near = np.flatnonzero(np.all(o_lo + s <= hi[:, None], axis=0) & np.all(o_hi + s >= lo[:, None], axis=0)
                              & (world.body_counts > 0))  # boxes within the gap of the moved group's box
        for k in near:
            if k in member_set:
                continue
            other = world.bodies[k]
            if image is None and _parallel_wall_ok(other, members, world, d):
                continue
            if image is not None and is_world_wall(world, other):
                continue  # walls never pair through images
            pts = other.points if image is None else other.points + s
            if not all(bodies_clear(m, pts, gap) for m in moved):
                return False
    return True


def _parallel_wall_ok(wall, members, world, d):
    """A wall along the wrap direction (e.g. a side wall for a vertical wrap): a body sliding past it in contact
    is closer than the clearance and always would be, so instead of the gap rule check that every point of the group
    is on the box side of the wall's line (strictly, more than ccd_gap): a move parallel to the line keeps each
    point's distance to it, and the wall spans the whole loop band, so the moved group stays clear of it too."""
    side = getattr(wall, "wall_side", None)
    if side is None or world.walls.get(side) is not wall:
        return False
    axis, c = _wall_line(wall)
    if d[axis] != 0:
        return False  # not a parallel move
    coords = np.concatenate([world.bodies[k].points[axis] for k in members])
    lo_end, hi_end = wall.points[1 - axis].min(), wall.points[1 - axis].max()
    span = np.concatenate([world.bodies[k].points[1 - axis] for k in members]) + d[1 - axis]
    if span.min() < lo_end or span.max() > hi_end:
        return False  # the destination runs past the wall's end: use the general check
    gap = EnvConfig.ccd_gap
    return bool(np.all(coords > c + gap)) if side in ("left", "top") else bool(np.all(coords < c - gap))


def _exit_side(lo, hi, vel, bounds, active):
    """The side a group with box lo..hi and mean velocity vel has fully left through while moving outward, among
    the `active` sides, or None. Left / right also take a group that has stopped out there (not heading back in
    faster than RETURN_SPEED): bodies come to rest off screen on a floor that runs past the side edges."""
    x0, y0, x1, y1 = bounds
    if "bottom" in active and lo[1] > y1 and vel[1] > 0:
        return "bottom"
    if "top" in active and hi[1] < y0 and vel[1] < 0:
        return "top"
    if "right" in active and lo[0] > x1 and vel[0] > -RETURN_SPEED:
        return "right"
    if "left" in active and hi[0] < x0 and vel[0] < RETURN_SPEED:
        return "left"
    return None


def _spots(world, entry, lo, hi, gap):
    """Candidate translations along the entry side's axis that put the box lo..hi just past the `entry` side:
    outside the screen edge when that side has no wall, else just inside its wall; then further from the screen
    (or deeper inside) half a box size + gap at a time, up to REACH of the band's length."""
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
    step /= 2  # half a body apart, so gaps in a stream of bodies are found
    reach = REACH * (world.bounds[axis + 2] - world.bounds[axis])
    out = []
    for k in range(int(min(MAX_SPOTS, reach // abs(step) + 1))):
        d = np.zeros(2)
        d[axis] = base + k * step
        out.append(d)
    return out


def _lifted(world, entry, spots, lo, hi, gap):
    """For a left / right wrap: spots just inside the entry edge (an unwalled entry's spots are all outside it),
    from level with the body up by a box height + gap at a time, up to LIFT_SPOTS, staying below the top of the
    bounds. A pile resting against the entry edge (e.g. on a floor that runs past the screen) blocks every spot level
    with it; above it there's room, and the body falls in from there."""
    rise = hi[1] - lo[1] + gap
    inward = 0.0 if world.walls.get(entry) in world.bodies else (hi[0] - lo[0] + gap) * (1 if entry == "left" else -1)
    out = []
    for k in range(LIFT_SPOTS + 1):
        d = spots[0] + (inward, -k * rise)
        if lo[1] + d[1] < world.bounds[1]:
            break
        out.append(d)
    return out


def _group_index(world, members):
    return np.concatenate([np.arange(world.offsets[k], world.offsets[k + 1]) for k in members])


def _periodic_wraps(world):
    """Translate every free group whose box centre has left the bounds along a periodic axis by the whole periods
    that bring it back (module doc). A side whose wall is pending while the opposite side still loops also keeps
    groups off its wall line: one reaching within contact_radius of the line goes one more period over to the
    looping side (an isometry of the torus like any wrap), so the wall can come back. Returns the wraps."""
    L, on = world.period, world.period > 0
    origin = np.array(world.bounds[:2], dtype=float)
    span = np.where(on, L, 1.0)
    box_lo, box_hi = body_boxes(world)
    centre = 0.5 * (box_lo + box_hi)
    movable = (world.body_counts > 0) & ~world.body_fixed
    out = movable & np.any(on[:, None] & (np.floor((centre - origin[:, None]) / span[:, None]) != 0), axis=0)
    r = EnvConfig.contact_radius
    pend = []  # (axis, wall on the high side?, wall line) of pending walls facing a looping side
    for side in sorted(world.pending_walls):
        a = AXIS_OF[side]
        if on[a] and world.looping[OPPOSITE[side]] and world.walls.get(side) is not None:
            high = side in ("right", "bottom")
            c = _wall_line(world.walls[side])[1]
            pend.append((a, high, c))
            out |= movable & ((box_hi[a] > c - r) if high else (box_lo[a] < c + r))
    if not out.any():
        return []
    if world.ropes:
        groups, anchored = rope_groups(world)
    else:  # no ropes: every body is its own group (only those to move matter)
        groups, anchored = {int(k): [int(k)] for k in np.flatnonzero(out)}, set()
    grabbed_body = None if world.grabbed is None else int(world.point_body[world.grabbed])
    wraps = []
    for root, members in groups.items():
        if root in anchored or not any(out[k] for k in members):
            continue
        g_lo, g_hi = box_lo[:, members].min(axis=1), box_hi[:, members].max(axis=1)
        d = np.where(on, -np.floor((0.5 * (g_lo + g_hi) - origin) / span) * span, 0.0) + 0.0  # (+ 0.0: no -0.0)
        for a, high, line in pend:  # (the same every frame, so a group doesn't go back and forth)
            if high and g_hi[a] + d[a] > line - r:
                d[a] -= L[a]
            elif not high and g_lo[a] + d[a] < line + r:
                d[a] += L[a]
        if not d.any():
            continue
        idx = _group_index(world, members)
        world.P[:, idx] += d[:, None]
        world.PREV[:, idx] += d[:, None]
        if grabbed_body in members:  # the hand's target moves into the body's new frame with it
            world.grab_target = np.asarray(world.grab_target, dtype=float) + d
            world.grab_shift = world.grab_shift - d
        wraps.extend((world.bodies[k], (float(d[0]), float(d[1]))) for k in members)
    return wraps


def loop_step(world):
    """One looping pass (World.step calls it before each frame's substeps): add pending walls that are clear, update
    the periodic axes, translate groups across periodic seams, then the walled-mode wraps through the other active
    sides. Moves P and PREV only; velocities untouched. Returns [(body, (dx, dy)), ...] for the bodies moved, e.g.
    so motion trails don't draw the jump."""
    if world.pending_walls:
        _try_pending_walls(world)
    if world.bounds is None or not world.bodies:
        clear_periodic(world)
        return []
    _update_periodic(world)
    wraps = _periodic_wraps(world) if world.period is not None else []
    active = {s for s in SIDES if _active(world, s) and not world.periodic[AXIS_OF[s]]}
    if active:
        wraps += _walled_wraps(world, active)
    return wraps


def span_wraps(world):
    """Periodic wraps between two CCD spans of one frame (World._substeps; World.period is set): the per-frame
    _periodic_wraps translations only (pending-wall rule included); adding pending walls, switching axes in or out
    of periodic mode and the walled-mode wraps stay once per frame in loop_step. Returns the wraps.

    No position-derived cache survives a wrap: the span's contact lists (World._span_contacts, its image pairs
    included) and reach are cleared when each span ends, and the CCD box cache is dropped here (it is keyed on the
    identity of a D array that never comes back, so this is only belt and braces)."""
    wraps = _periodic_wraps(world)
    if wraps:
        world._ccd_box_cache = None
    return wraps


def _walled_wraps(world, active):
    """Walled-mode wrapping (module doc) through the `active` sides, all on axes that aren't periodic."""
    # cheap vectorized pre-check: is any movable body fully outside the bounds across one of those axes at all?
    nonempty = world.body_counts > 0
    box_lo, box_hi = body_boxes(world)  # (2, bodies); kept current below as groups move
    x0, y0, x1, y1 = world.bounds
    across_x, across_y = not world.periodic[0], not world.periodic[1]  # (both, as before periodic mode existed)
    outside = nonempty & ~world.body_fixed & ((((box_lo[0] > x1) | (box_hi[0] < x0)) & across_x)
                                              | (((box_lo[1] > y1) | (box_hi[1] < y0)) & across_y))
    if not outside.any():
        return []
    groups, anchored = rope_groups(world)
    grabbed_body = None if world.grabbed is None else int(world.point_body[world.grabbed])
    gap = clearance()
    wraps = []
    # groups that found no free spot recently: id(first body) -> frames left before they search again
    waiting = world.__dict__.setdefault("_wrap_wait", {})
    for key in list(waiting):
        waiting[key] -= 1
        if waiting[key] <= 0:
            del waiting[key]
    for root, members in groups.items():
        if root in anchored or grabbed_body in members or not all(outside[k] for k in members):
            continue
        key = id(world.bodies[members[0]])
        if key in waiting:
            continue
        idx = _group_index(world, members)
        pts = world.P[:, idx]
        lo, hi = pts.min(axis=1), pts.max(axis=1)
        side = _exit_side(lo, hi, world.V[:, idx].mean(axis=1), world.bounds, active)
        if side is None:
            continue
        spots = _spots(world, OPPOSITE[side], lo, hi, gap)
        if side in ("left", "right") and spots:  # just past the edge, then above a pile there, then further out
            spots = spots[:1] + _lifted(world, OPPOSITE[side], spots, lo, hi, gap) + spots[1:]
        waiting[key] = RETRY_FRAMES  # cleared below when a spot is found
        for d in spots:
            if spot_free(world, members, d, gap, (box_lo, box_hi)):
                world.P[:, idx] += d[:, None]
                world.PREV[:, idx] += d[:, None]
                box_lo[:, members] += d[:, None]
                box_hi[:, members] += d[:, None]
                del waiting[key]
                wraps.extend((world.bodies[k], (float(d[0]), float(d[1]))) for k in members)
                break
        # every spot taken: leave the group where it is and try again next frame
    return wraps


def _reaching(world, lo, hi, s, margin):
    """Mask of boxes lo..hi (2, m) that, moved by s, reach into the bounds grown by margin."""
    x0, y0, x1, y1 = world.bounds
    return ((lo[0] + s[0] <= x1 + margin) & (hi[0] + s[0] >= x0 - margin)
            & (lo[1] + s[1] <= y1 + margin) & (hi[1] + s[1] >= y0 - margin))


def edges_with_images(world, a, b, margin=0.0):
    """Edge endpoint arrays a, b (2, m) plus their copies one period over across the periodic axes that reach into
    the bounds (grown by margin): every edge an object placed inside the bounds can come near (editor checks)."""
    shifts = image_shifts(getattr(world, "period", None))
    if not shifts or a.shape[1] == 0:
        return a, b
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    out_a, out_b = [a], [b]
    for s in shifts:
        k = _reaching(world, lo, hi, s, margin)
        if k.any():
            out_a.append(a[:, k] + s[:, None])
            out_b.append(b[:, k] + s[:, None])
    return np.concatenate(out_a, axis=1), np.concatenate(out_b, axis=1)


def points_with_images(world, pts, margin=0.0):
    """Points (2, m) plus their copies one period over that land in the bounds (grown by margin)."""
    shifts = image_shifts(getattr(world, "period", None))
    out = [pts]
    for s in shifts:
        k = _reaching(world, pts, pts, s, margin)
        if k.any():
            out.append(pts[:, k] + s[:, None])
    return np.concatenate(out, axis=1) if len(out) > 1 else pts


def polygon_images(world, pts, margin=0.0):
    """[pts] plus the copies of a body's points (2, n) one period over whose box reaches into the bounds."""
    out = [pts]
    if pts.shape[1]:
        lo, hi = pts.min(axis=1, keepdims=True), pts.max(axis=1, keepdims=True)
        out += [pts + s[:, None] for s in image_shifts(getattr(world, "period", None))
                if _reaching(world, lo, hi, s, margin)[0]]
    return out


def image_copies(world):
    """[(body index, (2,) shift)] of the extra copies to draw: every body (world walls aside) whose box sticks out
    past an edge of a periodic axis shows again one period over on the other side, and in the opposite corner when
    it sticks out across both axes. [] when nothing is periodic."""
    period = getattr(world, "period", None)
    if period is None or not world.bodies:
        return []
    x0, y0, x1, y1 = world.bounds
    lo, hi = body_boxes(world)
    out = []
    for s in image_shifts(period):
        need = world.body_counts > 0  # a copy at +Lx shows the part of a body left of x0, at -Lx right of x1, etc.
        for a, (e0, e1) in enumerate(((x0, x1), (y0, y1))):
            if s[a] > 0:
                need = need & (lo[a] < e0)
            elif s[a] < 0:
                need = need & (hi[a] > e1)
        out += [(int(k), s) for k in np.flatnonzero(need) if not is_world_wall(world, world.bodies[k])]
    return out
