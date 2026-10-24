import math

import numpy as np
import pygame

DEFAULT_COL = (127, 127, 127)


class EnvConfig:
    gravity = np.array([0.0, 500.0])  # px/s^2, +y is down on screen
    substeps = 4  # physics substeps per frame
    iterations = 4  # constraint + contact solver passes per substep
    contact_radius = 3.0  # px; a point collides with an edge when closer than this
    friction = 0.5  # 0 = ice, 1 = sticky
    damping = 0.2  # fraction of velocity lost per second
    cell_size = 24.0  # px; spatial hash grid cell size, keep >= longest edge you care about
    max_dt = 1 / 30  # clamp for frame hitches
    ccd_passes = 8  # collision-impulse passes per substep before falling back to rigid impact zones
    ccd_gap = 0.01  # px; minimum gap continuous collision detection keeps between a point and an edge
    ccd_push = 0.25  # px; gap a collision impulse aims for, larger than ccd_gap so chains of contacts settle quickly
    grab_radius = 15.0  # px; how close the cursor must be to a point to pick it up
    grab_stiffness = 0.3  # fraction of the gap to the cursor closed per substep
    grab_max_step = 4.0  # px; max pull per substep


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

    def __init__(self, points=None, vels=None, mass=1.0, point_col=None, stick_col=None, point_radius=2, stick_width=1):
        # shape (2, n): row 0 = x, row 1 = y, one column per point
        # NOTE: World rebinds these arrays to views of its own big arrays, so only ever modify them in place
        self.points = np.empty((2, 0), dtype=float) if points is None else np.array(points, dtype=float)
        n = self.points.shape[1]
        self.vels = np.zeros_like(self.points) if vels is None else np.array(vels, dtype=float)
        self.prev = self.points.copy()  # positions before this substep's prediction
        self.inv_mass = np.full(n, 1.0 / mass)
        self.point_col = DEFAULT_COL if point_col is None else point_col
        self.stick_col = DEFAULT_COL if stick_col is None else stick_col
        self.point_radius = point_radius
        self.stick_width = stick_width

        idx = np.arange(n)
        self.edges = np.array([idx, np.roll(idx, -1)]) if n >= 3 else np.empty((2, 0), dtype=int)
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

    def draw(self, surface):
        if self.points.shape[1] == 0:
            return
        for i, j in self.draw_edges.T:
            pygame.draw.line(surface, self.stick_col, self.points[:, i], self.points[:, j], self.stick_width)
        for pos in self.points.T:
            pygame.draw.circle(surface, self.point_col, pos, self.point_radius)


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


class Rope:
    """A tension-only spring between two ends: pulls them together when stretched past rest_length, never pushes.

    Each end is either (body, point_index) to attach to a body's point, or (x, y) for a fixed world anchor.
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

    def draw(self, surface):
        pygame.draw.line(surface, self.col, self.end_pos(0), self.end_pos(1), self.width)
