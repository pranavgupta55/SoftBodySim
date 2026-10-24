import itertools
import math

import numpy as np
import pygame

DEFAULT_COL = (127, 127, 127)
_CHAIN_IDS = itertools.count(1)


def new_chain_id():
    """A fresh id for Chain.chain_id (copies of a chain get a new one, so they are a chain of their own)."""
    return next(_CHAIN_IDS)


class EnvConfig:
    gravity = np.array([0.0, 500.0])  # px/s^2, +y is down on screen
    substeps = 4  # physics substeps per frame
    iterations = 4  # constraint + contact solver passes per substep
    contact_radius = 3.0  # px; a point collides with an edge when closer than this
    friction = 0.5  # 0 = ice, 1 = sticky
    damping = 0.2  # fraction of velocity lost per second
    # px/s; no point moves faster than this (None = no limit; per world via World.env).
    # A fall down a whole 982-unit-tall world peaks near 850 px/s, so only longer falls (looping worlds) reach it.
    terminal_velocity = 1000.0
    cell_size = 24.0  # px; spatial hash grid cell size, keep >= longest edge you care about
    max_dt = 1 / 30  # clamp for frame hitches
    # Substeps per CCD pass. 1: CCD after every substep, contact candidates rebuilt every substep. k > 1: k substeps
    # run without CCD, then one CCD pass checks the straight path from their start to their end, and the contact
    # candidates come from swept boxes over those k substeps. None: once per frame (all substeps).
    ccd_stride = None
    ccd_passes = 8  # collision-impulse passes per CCD pass before falling back to rigid impact zones
    ccd_gap = 0.01  # px; minimum gap continuous collision detection keeps between a point and an edge
    ccd_push = 0.25  # px; gap a collision impulse aims for, larger than ccd_gap so chains of contacts settle quickly
    grab_radius = 15.0  # px; how close the cursor must be to a point to pick it up
    # 1/s; the gap to the cursor shrinks as exp(-grab_stiffness * t) (85.6: 30% per 1/240 s substep)
    grab_stiffness = 85.6
    grab_max_speed = 960.0  # px/s; fastest the hand pulls a point (4 px per 1/240 s substep)
    peg_spacing = 15.0  # world units (= screen px at the default zoom); hooks snap to a grid of pegs this far apart, aligned to the world origin
    # "cpu": all physics in numpy (fastest for small scenes). "gpu": broad phases + PBD solves on the Apple GPU via
    # MLX, exact CCD still float64 on the CPU (gpu_backend.py); faster from roughly 600 points (M4). Read in World.add.
    backend = "gpu"
    # True: the CPU physics' hot paths run as numba kernels (solver_jit.py), bit-identical to the numpy code and
    # several times faster; ignored (numpy) when numba can't be imported. Per world via World.env like the rest.
    jit = True


def rect_points(cx, cy, w, h):
    """(2, 4) corners: top-left, top-right, bottom-right, bottom-left."""
    return np.array([[cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2],
                     [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2]])


def polygon_points(cx, cy, radius, n, angle=-math.pi / 2):
    """(2, n) regular polygon, first vertex at `angle` (default: straight up on screen)."""
    angles = angle + 2 * math.pi * np.arange(n) / n
    return np.array([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])


class Body:
    """A closed polygon of points. Collision outline is the loop 0 -> 1 -> ... -> n-1 -> 0."""
    fixed = False
    self_collide = False  # whether a point can hit this body's own non-neighboring edges
    # Opt-in bounce between two bodies that both have restitution > 0 (World._bounce): 0 = the usual inelastic
    # contacts, 1 = perfectly elastic. Treats the bodies as balls (impulses through the centres, no spin), e.g. the
    # Newton's cradle. Set it before World.add.
    restitution = 0.0

    def __init__(self, points=None, vels=None, mass=1.0, point_col=None, stick_col=None, point_radius=2, stick_width=1,
                 fill_col=None):
        # shape (2, n): row 0 = x, row 1 = y, one column per point
        # NOTE: World rebinds these arrays to views of its own big arrays, so only ever modify them in place
        self.points = np.empty((2, 0), dtype=float) if points is None else np.array(points, dtype=float)
        n = self.points.shape[1]
        self.vels = np.zeros_like(self.points) if vels is None else np.array(vels, dtype=float)
        self.prev = self.points.copy()  # positions before this substep's prediction
        self.inv_mass = np.full(n, 1.0 / mass)
        self.point_col = DEFAULT_COL if point_col is None else point_col
        self.stick_col = DEFAULT_COL if stick_col is None else stick_col
        self.fill_col = self.point_col if fill_col is None else fill_col  # used by the filled shape view
        self.point_radius = point_radius
        self.stick_width = stick_width

        idx = np.arange(n)
        if n >= 3:
            self.edges = np.array([idx, np.roll(idx, -1)])
        elif n == 2:  # a single segment (e.g. a thin fixed wall): one edge, no interior
            self.edges = np.array([[0], [1]])
        else:
            self.edges = np.empty((2, 0), dtype=int)
        self.draw_edges = self.edges
        # +1 / -1 depending on loop direction, so the solver knows which side of an edge is outside
        x, y = self.points
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        self.winding = 1.0 if area >= 0 else -1.0

    def predict(self, h):
        movable = self.inv_mass > 0
        self.vels += EnvConfig.gravity[:, None] * h * movable
        self.prev[:] = self.points
        self.points += self.vels * h

    def solve_constraints(self):
        pass

    def draw(self, surface, zoom=1.0):
        """zoom maps world units to surface pixels; radii and line widths stay in pixels."""
        if self.points.shape[1] == 0:
            return
        pts = self.points * zoom
        for i, j in self.draw_edges.T:
            pygame.draw.line(surface, self.stick_col, pts[:, i], pts[:, j], self.stick_width)
        for pos in pts.T:
            pygame.draw.circle(surface, self.point_col, pos, self.point_radius)

    def draw_filled(self, surface, border_col=(0, 0, 0), border_width=2, zoom=1.0):
        """Filled view: the perimeter loop as a solid polygon with a border, no points or springs."""
        pts = self.points * zoom
        if pts.shape[1] == 2:  # a segment: a thick line in the fill colour with a black outline
            a, b = pts[:, 0], pts[:, 1]
            pygame.draw.line(surface, border_col, a, b, max(self.stick_width, 1) + 2 * border_width)
            for end in (a, b):
                pygame.draw.circle(surface, border_col, end, (max(self.stick_width, 1) + 2 * border_width) / 2)
            pygame.draw.line(surface, self.fill_col, a, b, max(self.stick_width, 1))
            return
        if self.edges.shape[1] < 3:  # not a closed polygon (e.g. a rope), nothing to fill
            self.draw(surface, zoom)
            return
        loop = np.round(pts[:, self.edges[0]]).astype(int)  # perimeter in loop order, whole pixels
        pygame.draw.polygon(surface, self.fill_col, loop.T.tolist())
        draw_border(surface, loop, border_col, border_width, self.winding)


def draw_border(surface, loop, col=(0, 0, 0), width=2, winding=None):
    """Border of a filled polygon whose corners are the whole-pixel (2, n) `loop`: the 1px outline plus copies
    shifted 1px at a time outward along each edge's minor axis. pygame's own thick lines grow to one side only, and
    its fill spills up to 1px past the outline on some edges, so a stroke on (and just outside) the outline is what
    fully covers the fill's rim. winding: +1 / -1 for the loop's direction (computed from the loop if None)."""
    loop = np.asarray(loop, dtype=int)
    if winding is None:
        x, y = loop
        winding = 1.0 if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) >= 0 else -1.0
    d = np.roll(loop, -1, axis=1) - loop
    steep = np.abs(d[1]) >= np.abs(d[0])
    outward = winding * np.array([d[1], -d[0]])  # interior is left of each edge for positive shoelace area
    shift = np.where(steep, [np.sign(outward[0]), np.zeros_like(d[0])], [np.zeros_like(d[0]), np.sign(outward[1])])
    a, b = loop.T.tolist(), np.roll(loop, -1, axis=1).T.tolist()
    for k in range(width):
        offs = (shift * k).T.tolist()
        for (ax, ay), (bx, by), (sx, sy) in zip(a, b, offs):
            pygame.draw.line(surface, col, (ax + sx, ay + sy), (bx + sx, by + sy))


class FixedBody(Body):
    """Never moves: zero inverse mass, so gravity and the solver ignore it."""
    fixed = True

    def __init__(self, points=None, **kwargs):
        super().__init__(points, **kwargs)
        self.inv_mass[:] = 0.0


class RigidBody(Body):
    """Shape matching: each pass, snap points toward the best-fit rotation + translation of the rest shape."""

    def __init__(self, points=None, stiffness=1.0, **kwargs):
        super().__init__(points, **kwargs)
        self.stiffness = stiffness
        self.rest = self.points - self.points.mean(axis=1, keepdims=True)

    def solve_constraints(self):
        com = self.points.mean(axis=1, keepdims=True)
        a = (self.points - com) @ self.rest.T  # 2x2 covariance between current and rest shape
        theta = math.atan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1])
        c, s = math.cos(theta), math.sin(theta)
        goal = np.array([[c, -s], [s, c]]) @ self.rest + com
        self.points += self.stiffness * (goal - self.points)


class SoftBody(Body):
    """Springs (as PBD distance constraints) around the perimeter plus `skip_sets` sets of longer-range springs.

    skip_sets=0: perimeter only. 1: also each point to its 2nd neighbor (both directions). 2: also its 3rd neighbor, etc.
    Capped at n // 2, past which the sets would only repeat springs that already exist.
    """
    self_collide = True  # a dented soft body can fold over itself otherwise

    def __init__(self, points=None, stiffness=1.0, skip_sets=1, **kwargs):
        super().__init__(points, **kwargs)
        self.stiffness = stiffness
        n = self.points.shape[1]
        self.skip_sets = max(0, min(skip_sets, n // 2))
        pairs = set()
        for i in range(n):
            for k in range(1, self.skip_sets + 2):
                j = (i + k) % n
                if i != j:
                    pairs.add((min(i, j), max(i, j)))  # sorted tuple so (i, j) and (j, i) are one spring
        self.springs = np.array(sorted(pairs), dtype=int).T.reshape(2, -1)
        self.draw_edges = self.springs
        i, j = self.springs
        self.rest_len = np.hypot(*(self.points[:, j] - self.points[:, i]))
        self.spring_count = np.maximum(np.bincount(self.springs.ravel(), minlength=n), 1)

    def solve_constraints(self):
        i, j = self.springs
        d = self.points[:, j] - self.points[:, i]
        dist = np.maximum(np.hypot(d[0], d[1]), 1e-9)
        wi, wj = self.inv_mass[i], self.inv_mass[j]
        corr = self.stiffness * (dist - self.rest_len) / (dist * (wi + wj)) * d
        acc = np.zeros_like(self.points)
        for k in range(2):
            np.add.at(acc[k], i, wi * corr[k])
            np.add.at(acc[k], j, -wj * corr[k])
        self.points += acc / self.spring_count  # Jacobi: average the corrections hitting each point


class Hook:
    """A single fixed point (world units) for ropes to hang from. Not a body: no mass, no edges, never collides and
    isn't packed into the World's collision arrays. Pass the Hook itself as a Rope end."""

    def __init__(self, pos, col=None, radius=3):
        self.pos = np.array(pos, dtype=float)  # shape (2,); ropes keep a reference to this array, so edit it in place
        self.col = DEFAULT_COL if col is None else col
        self.radius = radius  # marker radius in surface pixels

    def draw(self, surface, zoom=1.0, filled=False):
        """A ring with a hole: black-bordered disc in the filled view, a 1px ring around a dot in the points view."""
        c = self.pos * zoom
        r = self.radius
        if filled:
            pygame.draw.circle(surface, (0, 0, 0), c, r + 1)
            pygame.draw.circle(surface, self.col, c, r - 0.5)
            pygame.draw.circle(surface, (0, 0, 0), c, 1)
        else:
            pygame.draw.circle(surface, self.col, c, r, 1)
            pygame.draw.circle(surface, self.col, c, 1)


class Rope:
    """A tension-only spring between two ends: pulls them together when stretched past rest_length, never pushes.

    Each end is either (body, point_index) to attach to a body's point, a Hook, or (x, y) for a fixed world anchor.
    It's a constraint, not a body: it has no points of its own and doesn't collide with anything.
    rest_length defaults to the starting distance between the ends. stiffness is the fraction of the stretch
    removed per solver pass (1.0 = inextensible).
    """

    def __init__(self, end_a, end_b, rest_length=None, stiffness=1.0, col=None, width=1):
        self.ends = [self._parse_end(end_a), self._parse_end(end_b)]
        self.stiffness = stiffness
        self.col = DEFAULT_COL if col is None else col
        self.width = width
        self.rest_length = float(np.hypot(*(self.end_pos(1) - self.end_pos(0)))) if rest_length is None else float(rest_length)

    @staticmethod
    def _parse_end(end):
        if isinstance(end, Hook):  # a fixed anchor that shares the hook's position array (inverse mass 0)
            return None, end.pos
        if isinstance(end[0], Body):
            body, idx = end
            return body, int(idx)
        return None, np.array(end, dtype=float)  # fixed world anchor

    def end_pos(self, k):
        body, ref = self.ends[k]
        # read body.points at call time: World rebinds it to a view of its shared array
        return body.points[:, ref].copy() if body is not None else ref.copy()

    def end_inv_mass(self, k):
        body, ref = self.ends[k]
        return float(body.inv_mass[ref]) if body is not None else 0.0

    def length(self):
        return float(np.hypot(*(self.end_pos(1) - self.end_pos(0))))

    def solve_constraints(self):
        pa, pb = self.end_pos(0), self.end_pos(1)
        wa, wb = self.end_inv_mass(0), self.end_inv_mass(1)
        d = pb - pa
        dist = float(np.hypot(*d))
        if dist <= self.rest_length or dist < 1e-9 or wa + wb <= 0:
            return  # slack (or nothing can move): no force at all
        corr = self.stiffness * (dist - self.rest_length) / (dist * (wa + wb)) * d
        for k, w, sign in ((0, wa, 1.0), (1, wb, -1.0)):
            body, ref = self.ends[k]
            if body is not None and w > 0:
                body.points[:, ref] += sign * w * corr  # in place, so the World's shared array sees it

    def draw(self, surface, zoom=1.0):
        pygame.draw.line(surface, self.col, self.end_pos(0) * zoom, self.end_pos(1) * zoom, self.width)


def end_point(end):
    """World position of a rope end spec: (body, point_index), a Hook, or (x, y)."""
    if isinstance(end, Hook):
        return end.pos.copy()
    if isinstance(end[0], Body):
        return np.array(end[0].points[:, int(end[1])], dtype=float)
    return np.array(end, dtype=float)


def chain_box_points(center, u, size):
    """(2, 6) square of side `size` centred on `center`, sides along the unit vector u: the four corners plus the
    midpoints of its back (-u) and front (+u) sides, where the chain's ropes attach (indices 0 and 3)."""
    c, u = np.asarray(center, dtype=float), np.asarray(u, dtype=float)
    v, h = np.array([-u[1], u[0]]), size / 2
    pts = [c - h * u, c - h * u + h * v, c + h * u + h * v, c + h * u, c + h * u - h * v, c - h * u - h * v]
    return np.array(pts).T


class Chain:
    """Builder for a chain: alternating ropes and small square RigidBodies in a straight line from end_a to end_b.

    segments counts ropes + boxes; `start` / `end` ("rope" or "box") say what the chain begins and ends with, so it
    must be odd when they match and even when they differ (see chain_counts). A rope end is any Rope end spec (body
    point, Hook or (x, y) anchor); a box end is the (x, y) centre of that end's box. Ropes get rest length = their
    starting length, so the chain starts exactly taut. Not a body itself: add .bodies and .ropes to a World (or use
    World.add_chain). Box points are the corners plus the back / front side midpoints the ropes attach to.
    """
    BOX_SIZE = 10.0  # world units (5 px at the default zoom of 0.5)

    def __init__(self, end_a, end_b, segments, start="rope", end="rope", box_size=BOX_SIZE, box_col=None, rope_col=None,
                 stiffness=1.0, rope_width=1, point_radius=3):
        a, b = end_point(end_a), end_point(end_b)
        centers, u, _ = chain_layout(a, b, segments, start, end, box_size)
        style = dict(point_col=box_col, stick_col=box_col, fill_col=box_col, point_radius=point_radius)
        self.bodies = [RigidBody(chain_box_points(c, u, box_size), **style) for c in centers]
        # the pieces in order (a box, or None for a rope); a rope joins the previous box's front midpoint (or end_a)
        # to the next box's back midpoint (or end_b)
        boxes = iter(self.bodies)
        pieces = [next(boxes) if _chain_is_box(k, start) else None for k in range(segments)]
        self.ropes = []
        for k, p in enumerate(pieces):
            if p is None:
                ra = (pieces[k - 1], 3) if k > 0 else end_a
                rb = (pieces[k + 1], 0) if k + 1 < segments else end_b
                self.ropes.append(Rope(ra, rb, stiffness=stiffness, col=rope_col, width=rope_width))
        # every piece carries the chain's id, so the editor can select, move, copy and delete the chain as one object
        self.chain_id = new_chain_id()
        for piece in self.bodies + self.ropes:
            piece.chain_id = self.chain_id


def _chain_is_box(k, start):
    return (k % 2 == 0) == (start == "box")


def chain_counts(segments, start="rope", end="rope"):
    """(ropes, boxes) in a chain of `segments` alternating pieces. The count must be odd when both ends are the same
    kind and even when they differ, and at least 3 (same ends) or 2 (mixed); raises ValueError otherwise."""
    if (segments % 2 == 1) != (start == end) or segments < (3 if start == end else 2):
        raise ValueError(f"{segments} segments can't start with a {start} and end with a {end}")
    boxes = (segments + 1) // 2 if start == "box" else segments // 2
    return segments - boxes, boxes


def chain_layout(a, b, segments, start="rope", end="rope", box_size=Chain.BOX_SIZE):
    """Box centres (k, 2), the unit direction a -> b, and the rope length, for a straight chain from a to b. A box
    end's box is centred on that end. The rope length can come out short or negative when a and b are too close."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ropes, boxes = chain_counts(segments, start, end)
    dist = float(np.hypot(*(b - a)))
    u = (b - a) / dist if dist > 1e-9 else np.array([1.0, 0.0])
    covered = boxes * box_size - box_size / 2 * ((start == "box") + (end == "box"))  # box length between a and b
    rope = (dist - covered) / ropes
    centers, s = [], (-box_size / 2 if start == "box" else 0.0)  # s: distance along the chain so far
    for k in range(segments):
        if _chain_is_box(k, start):
            centers.append(a + (s + box_size / 2) * u)
            s += box_size
        else:
            s += rope
    return np.array(centers).reshape(-1, 2), u, rope
