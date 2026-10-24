from collections import defaultdict

import numpy as np

from entities import EnvConfig


class World:
    """Owns all bodies and runs the global PBD loop: predict -> (constraints + contacts) x N -> velocities."""

    def __init__(self):
        self.bodies = []

    def add(self, body):
        self.bodies.append(body)
        self._rebuild()

    def _rebuild(self):
        """Pack every body into shared arrays; bodies keep views into them so they edit in place."""
        bodies = self.bodies
        self.P = np.concatenate([b.points for b in bodies], axis=1)
        self.PREV = np.concatenate([b.prev for b in bodies], axis=1)
        self.V = np.concatenate([b.vels for b in bodies], axis=1)
        self.W = np.concatenate([b.inv_mass for b in bodies])
        counts = [b.points.shape[1] for b in bodies]
        offsets = np.cumsum([0] + counts)
        for b, start, end in zip(bodies, offsets[:-1], offsets[1:]):
            b.points, b.prev, b.vels, b.inv_mass = self.P[:, start:end], self.PREV[:, start:end], self.V[:, start:end], self.W[start:end]
        self.point_body = np.repeat(np.arange(len(bodies)), counts)
        self.edges = np.concatenate([b.edges + off for b, off in zip(bodies, offsets[:-1])], axis=1)
        self.edge_body = np.repeat(np.arange(len(bodies)), [b.edges.shape[1] for b in bodies])
        self.edge_sign = np.concatenate([np.full(b.edges.shape[1], b.winding) for b in bodies])
        self.body_fixed = np.array([b.fixed for b in bodies])

    def step(self, dt):
        dt = min(dt, EnvConfig.max_dt)
        h = dt / EnvConfig.substeps
        for _ in range(EnvConfig.substeps):
            for b in self.bodies:
                b.predict(h)
            pts, edge_ids = self._candidates()
            for _ in range(EnvConfig.iterations):
                for b in self.bodies:
                    b.solve_constraints()
                self._solve_contacts(pts, edge_ids)
            for b in self.bodies:
                b.update_velocities(h)

    def draw(self, surface):
        for b in self.bodies:
            b.draw(surface)

    # ---------------- Broad phase: spatial hash grid
    def _candidates(self):
        """(point, edge) pairs that share a grid cell. Edges are registered in every cell their padded AABB touches."""
        cell = EnvConfig.cell_size
        pad = 2 * EnvConfig.contact_radius
        a, b = self.P[:, self.edges[0]], self.P[:, self.edges[1]]
        lo = np.floor((np.minimum(a, b) - pad) / cell).astype(int)
        hi = np.floor((np.maximum(a, b) + pad) / cell).astype(int)
        grid = defaultdict(list)
        for e in range(self.edges.shape[1]):
            for cx in range(lo[0, e], hi[0, e] + 1):
                for cy in range(lo[1, e], hi[1, e] + 1):
                    grid[(cx, cy)].append(e)

        point_cells = np.floor(self.P / cell).astype(int)
        pairs = set()
        for i in range(self.P.shape[1]):
            for e in grid.get((point_cells[0, i], point_cells[1, i]), ()):
                pb, eb = self.point_body[i], self.edge_body[e]
                if pb != eb and not (self.body_fixed[pb] and self.body_fixed[eb]):  # no self-collision, no static-vs-static
                    pairs.add((i, e))
        if not pairs:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        pts, edge_ids = np.array(sorted(pairs)).T
        return pts, edge_ids

    # ---------------- Narrow phase + resolution: point vs edge capsule
    def _solve_contacts(self, pts, edge_ids):
        if pts.size == 0:
            return
        r, mu = EnvConfig.contact_radius, EnvConfig.friction
        ia, ib = self.edges[0, edge_ids], self.edges[1, edge_ids]
        P, PREV, W = self.P, self.PREV, self.W

        p, a, b = P[:, pts], P[:, ia], P[:, ib]
        e = b - a
        len2 = np.maximum(np.sum(e * e, axis=0), 1e-12)
        t = np.clip(np.sum((p - a) * e, axis=0) / len2, 0.0, 1.0)
        c = a + t * e
        d = p - c
        dist = np.hypot(d[0], d[1])

        # outward normal of the edge (depends on the body's loop direction)
        n_edge = self.edge_sign[edge_ids] * np.array([e[1], -e[0]]) / np.sqrt(len2)
        inside = np.sum(d * n_edge, axis=0) < 0  # point already crossed to the inner side of the edge
        use_edge_normal = inside | (dist < 1e-9)
        n = np.where(use_edge_normal, n_edge, d / np.maximum(dist, 1e-9))
        depth = r - np.sum(d * n, axis=0)

        wp, wa, wb = W[pts], W[ia], W[ib]
        denom = wp + wa * (1 - t) ** 2 + wb * t ** 2
        active = (dist < r) & (denom > 1e-12)
        if not active.any():
            return

        # friction: cancel part of the point's sliding relative to the contact point on the edge
        c_prev = PREV[:, ia] + t * (PREV[:, ib] - PREV[:, ia])
        rel = (p - PREV[:, pts]) - (c - c_prev)
        tang = rel - np.sum(rel * n, axis=0) * n
        tang_len = np.hypot(tang[0], tang[1])
        fric = np.minimum(1.0, mu * depth / np.maximum(tang_len, 1e-9))
        delta = depth * n - tang * fric

        lam = delta / np.where(active, denom, 1.0)
        acc = np.zeros_like(P)
        count = np.zeros(P.shape[1])
        for idx, share in ((pts, wp), (ia, -wa * (1 - t)), (ib, -wb * t)):
            for k in range(2):
                np.add.at(acc[k], idx[active], (share * lam[k])[active])
            np.add.at(count, idx[active], 1.0)
        P += acc / np.maximum(count, 1.0)  # Jacobi: average when a point is hit by several contacts
