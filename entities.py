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

    def update_velocities(self, h):
        self.vels[:] = (self.points - self.prev) / h * max(0.0, 1.0 - EnvConfig.damping * h)

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
    """Springs (as PBD distance constraints) between neighbors and neighbors-of-neighbors."""

    def __init__(self, points=None, stiffness=1.0, **kwargs):
        super().__init__(points, **kwargs)
        self.stiffness = stiffness
        n = self.points.shape[1]
        pairs = set()
        for i in range(n):
            for k in (1, 2):
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
