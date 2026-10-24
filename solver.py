from collections import defaultdict

import numpy as np
import pygame

from entities import EnvConfig


class World:
    """Owns all bodies and steps them. Per substep:

    1. PBD: predict, drag, then springs / shape matching + soft contacts (thickness and friction).
    2. Turn that motion into velocities and run continuous collision detection (CCD) on the straight-line paths
       from the start positions, with impulses until no point crosses an edge (Bridson et al. 2002).
    3. If impulses don't converge, merge colliding points into rigid impact zones until nothing collides.
    Since each substep starts intersection-free and its whole motion is checked, it also ends intersection-free.
    """

    def __init__(self):
        self.bodies = []
        self.ropes = []  # tension-only constraints between body points / world anchors; not collidable
        self.grabbed = None  # global index of the point being dragged
        self.grab_target = np.zeros(2)
        self.zone_count = 0  # how many substeps needed the rigid impact zone fallback (for debugging)

    def add(self, body):
        self.bodies.append(body)
        self._rebuild()

    def add_rope(self, rope):
        self.ropes.append(rope)

    def _rebuild(self):
        """Pack every body into shared arrays; bodies keep views into them so they edit in place."""
        bodies = self.bodies
        self.P = np.concatenate([b.points for b in bodies], axis=1)
        self.PREV = np.concatenate([b.prev for b in bodies], axis=1)
        self.V = np.concatenate([b.vels for b in bodies], axis=1)
        self.W = np.concatenate([b.inv_mass for b in bodies])
        counts = [b.points.shape[1] for b in bodies]
        offsets = np.cumsum([0] + counts)
        self.offsets = offsets
        for b, start, end in zip(bodies, offsets[:-1], offsets[1:]):
            b.points, b.prev, b.vels, b.inv_mass = self.P[:, start:end], self.PREV[:, start:end], self.V[:, start:end], self.W[start:end]
        self.point_body = np.repeat(np.arange(len(bodies)), counts)
        self.edges = np.concatenate([b.edges + off for b, off in zip(bodies, offsets[:-1])], axis=1)
        self.edge_body = np.repeat(np.arange(len(bodies)), [b.edges.shape[1] for b in bodies])
        self.edge_sign = np.concatenate([np.full(b.edges.shape[1], b.winding) for b in bodies])
        self.body_fixed = np.array([b.fixed for b in bodies])
        self.body_self = np.array([b.self_collide for b in bodies])
        self.body_counts = np.array(counts)

    # ---------------- Mouse dragging
    def grab(self, pos):
        """Pick up the nearest movable point within grab_radius of pos (if any)."""
        dist = np.hypot(*(self.P - np.reshape(pos, (2, 1))))
        dist[self.W <= 0] = np.inf  # fixed points can't be dragged
        i = int(np.argmin(dist))
        if dist[i] <= EnvConfig.grab_radius:
            self.grabbed = i
            self.grab_target = np.array(pos, dtype=float)

    def drag_to(self, pos):
        self.grab_target = np.array(pos, dtype=float)

    def release(self):
        self.grabbed = None

    def _apply_grab(self):
        """Nudge the held point toward the cursor once per substep, so contacts (solved after) can out-vote the hand."""
        if self.grabbed is None:
            return
        gap = self.grab_target - self.P[:, self.grabbed]
        step = gap * EnvConfig.grab_stiffness
        length = np.hypot(*step)
        if length > EnvConfig.grab_max_step:
            step *= EnvConfig.grab_max_step / length
        self.P[:, self.grabbed] += step

    def step(self, dt):
        dt = min(dt, EnvConfig.max_dt)
        h = dt / EnvConfig.substeps
        for _ in range(EnvConfig.substeps):
            for b in self.bodies:
                b.predict(h)
            self._apply_grab()
            pts, edge_ids = self._candidates()
            for _ in range(EnvConfig.iterations):
                for b in self.bodies:
                    b.solve_constraints()
                for r in self.ropes:
                    r.solve_constraints()
                self._solve_contacts(pts, edge_ids)
            self.V[:] = (self.P - self.PREV) / h
            self._resolve_ccd(h)
            self.P[:] = self.PREV + self.V * h
            self.V *= max(0.0, 1.0 - EnvConfig.damping * h)

    def draw(self, surface, filled=False):
        """Bodies in the order they were added (later ones on top). filled: solid polygons instead of points + springs."""
        for b in self.bodies:
            if filled:
                b.draw_filled(surface)
            else:
                b.draw(surface)
        for r in self.ropes:
            r.draw(surface)
        if self.grabbed is not None:
            pygame.draw.line(surface, (255, 255, 255), self.P[:, self.grabbed], self.grab_target, 1)

    def draw_grid(self, surface, line_col=(52, 58, 76), edge_cell_col=(37, 42, 55), point_cell_col=(45, 51, 68)):
        """Debug view of the broad phase grid: cell (i, j) covers [i, i + 1) * cell_size, same as the solver's floor().

        Cells an edge is registered in (its AABB padded like _candidates does) get edge_cell_col,
        cells holding a point get point_cell_col (drawn over), then the grid lines on top.
        """
        cell = EnvConfig.cell_size
        w, h = surface.get_size()
        cols, rows = int(np.ceil(w / cell)), int(np.ceil(h / cell))
        if self.bodies:
            pad = 2 * EnvConfig.contact_radius
            a, b = self.P[:, self.edges[0]], self.P[:, self.edges[1]]
            lo = np.clip(np.floor((np.minimum(a, b) - pad) / cell).astype(int), 0, [[cols], [rows]])
            hi = np.clip(np.floor((np.maximum(a, b) + pad) / cell).astype(int), -1, [[cols - 1], [rows - 1]])
            span = np.maximum(hi - lo + 1, 0)  # visible cells per axis per edge (0 if fully off screen)
            counts = span[0] * span[1]
            # expand every edge into its covered cells without a Python loop
            e = np.repeat(np.arange(counts.size), counts)
            k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
            occupied = np.zeros((cols, rows), dtype=np.int8)
            occupied[lo[0, e] + k % span[0, e], lo[1, e] + k // span[0, e]] = 1
            pc = np.floor(self.P / cell).astype(int)
            on = (pc[0] >= 0) & (pc[0] < cols) & (pc[1] >= 0) & (pc[1] < rows)
            occupied[pc[0, on], pc[1, on]] = 2
            for (i, j), kind in zip(np.argwhere(occupied).tolist(), occupied[occupied > 0].tolist()):
                x0, y0 = int(np.floor(i * cell)), int(np.floor(j * cell))
                rect = (x0, y0, int(np.floor((i + 1) * cell)) - x0, int(np.floor((j + 1) * cell)) - y0)
                surface.fill(point_cell_col if kind == 2 else edge_cell_col, rect)
        for i in range(1, cols):
            x = int(np.floor(i * cell))
            pygame.draw.line(surface, line_col, (x, 0), (x, h - 1))
        for j in range(1, rows):
            y = int(np.floor(j * cell))
            pygame.draw.line(surface, line_col, (0, y), (w - 1, y))

    # ---------------- Broad phase: spatial hash grid
    def _edge_grid(self, pad):
        """Spatial hash: cell -> ids of edges whose AABB (grown by pad) touches that cell."""
        cell = EnvConfig.cell_size
        a, b = self.P[:, self.edges[0]], self.P[:, self.edges[1]]
        lo = np.floor((np.minimum(a, b) - pad) / cell).astype(int)
        hi = np.floor((np.maximum(a, b) + pad) / cell).astype(int)
        grid = defaultdict(list)
        for e in range(self.edges.shape[1]):
            for cx in range(lo[0, e], hi[0, e] + 1):
                for cy in range(lo[1, e], hi[1, e] + 1):
                    grid[(cx, cy)].append(e)
        return grid

    def _candidates(self):
        """(point, edge) pairs that share a grid cell. Edges are registered in every cell their padded AABB touches."""
        cell = EnvConfig.cell_size
        grid = self._edge_grid(2 * EnvConfig.contact_radius)

        point_cells = np.floor(self.P / cell).astype(int)
        pairs = set()
        for i in range(self.P.shape[1]):
            for e in grid.get((point_cells[0, i], point_cells[1, i]), ()):
                pb, eb = self.point_body[i], self.edge_body[e]
                if pb != eb:
                    if not (self.body_fixed[pb] and self.body_fixed[eb]):  # no static-vs-static
                        pairs.add((i, e))
                elif self.body_self[pb]:
                    # self-collision: skip edges touching the point or its immediate neighbors (they're always close)
                    n, base = self.body_counts[pb], self.offsets[pb]
                    li = i - base  # point index within its body
                    ring_gap = lambda v: min((li - v) % n, (v - li) % n)
                    if ring_gap(self.edges[0, e] - base) > 1 and ring_gap(self.edges[1, e] - base) > 1:
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

        # Edge normal facing the side the point started this substep on. CCD guarantees no point ever crosses an
        # edge, so that's always the free side: outside for neighboring bodies, inside for a body's contents.
        # (Only a point exactly on the edge's line at the start falls back to the body's outward side.)
        pa0 = PREV[:, ia]
        e0 = PREV[:, ib] - pa0
        q0 = PREV[:, pts] - pa0
        cross0 = e0[0] * q0[1] - e0[1] * q0[0]
        side = np.where(cross0 != 0, np.sign(cross0), -self.edge_sign[edge_ids])
        n_edge = -side * np.array([e[1], -e[0]]) / np.sqrt(len2)
        inside = np.sum(d * n_edge, axis=0) < 0  # point has moved past the edge's line since the start of the substep
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

    # ---------------- Continuous collision detection (the no-intersection guarantee)
    def _ccd_pairs(self, D):
        """(point, edge) pairs whose swept boxes (start -> start + D) share a grid cell."""
        cell, X0 = EnvConfig.cell_size, self.PREV
        X1 = X0 + D
        lo_pt = np.floor(np.minimum(X0, X1) / cell).astype(int)
        hi_pt = np.floor(np.maximum(X0, X1) / cell).astype(int)
        a0, b0, a1, b1 = X0[:, self.edges[0]], X0[:, self.edges[1]], X1[:, self.edges[0]], X1[:, self.edges[1]]
        lo = np.floor((np.minimum(np.minimum(a0, b0), np.minimum(a1, b1)) - 1.0) / cell).astype(int)
        hi = np.floor((np.maximum(np.maximum(a0, b0), np.maximum(a1, b1)) + 1.0) / cell).astype(int)
        grid = defaultdict(list)
        for e in range(self.edges.shape[1]):
            for cx in range(lo[0, e], hi[0, e] + 1):
                for cy in range(lo[1, e], hi[1, e] + 1):
                    grid[(cx, cy)].append(e)

        pairs = set()
        for i in range(X0.shape[1]):
            for cx in range(lo_pt[0, i], hi_pt[0, i] + 1):
                for cy in range(lo_pt[1, i], hi_pt[1, i] + 1):
                    for e in grid.get((cx, cy), ()):
                        pb, eb = self.point_body[i], self.edge_body[e]
                        if pb != eb:
                            if not (self.body_fixed[pb] and self.body_fixed[eb]):
                                pairs.add((i, e))
                        elif self.body_self[pb] and i != self.edges[0, e] and i != self.edges[1, e]:
                            pairs.add((i, e))
        if not pairs:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        pts, edge_ids = np.array(sorted(pairs)).T
        return pts, edge_ids

    def _ccd_detect(self, D, pairs=None):
        """Points that touch an edge at some time in the substep, moving in straight lines from PREV by D.

        A point p(t) is on the line through a(t), b(t) when cross(b - a, p - a) = 0, a quadratic in t in [0, 1];
        it's a hit if the point is also within the segment then. Ending closer than ccd_gap also counts, so states
        never get so close that rounding decides which side a point is on. Returns (pts, edge_ids, t, s, side).
        `pairs` reuses an earlier broad phase; only a fresh one (the default) can prove there are no collisions.
        """
        pts, eids = self._ccd_pairs(D) if pairs is None else pairs
        empty = np.empty(0)
        if pts.size == 0:
            return pts, eids, empty, empty, empty
        ia, ib = self.edges[0, eids], self.edges[1, eids]
        X0 = self.PREV
        e0, de = X0[:, ib] - X0[:, ia], D[:, ib] - D[:, ia]
        q0, dq = X0[:, pts] - X0[:, ia], D[:, pts] - D[:, ia]

        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]

        c0, c1, c2 = cross(e0, q0), cross(e0, dq) + cross(de, q0), cross(de, dq)
        side = np.where(c0 != 0, np.sign(c0), 1.0)  # which side of the edge the point started on

        # both roots of c2 t^2 + c1 t + c0 (falls back to the linear root when c2 ~ 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            disc = c1 * c1 - 4 * c2 * c0
            # only rounding-level negatives count as a grazing touch; real negatives mean the point never reaches the line
            disc = np.where((disc < 0) & (disc > -1e-9 * (c1 * c1 + np.abs(4 * c2 * c0))), 0.0, disc)
            sq = np.sqrt(disc)  # NaN where there's no real root, which fails the range checks below
            quad = np.abs(c2) > 1e-12
            r1 = np.where(quad, (-c1 - sq) / (2 * c2), -c0 / c1)
            r2 = np.where(quad, (-c1 + sq) / (2 * c2), np.nan)

        t_hit = np.full(pts.size, np.inf)
        s_hit = np.zeros(pts.size)
        for r in (r1, r2):
            with np.errstate(invalid="ignore"):  # NaN / inf roots just fail the range checks
                e, q = e0 + r * de, q0 + r * dq
                s = np.sum(q * e, axis=0) / np.maximum(np.sum(e * e, axis=0), 1e-12)
                ok = (r >= 0) & (r <= 1) & (s >= 0) & (s <= 1) & (r < t_hit)
            t_hit = np.where(ok, r, t_hit)
            s_hit = np.where(ok, s, s_hit)

        # ending within ccd_gap of the edge counts as a hit at t = 1
        e1, q1 = e0 + de, q0 + dq
        len2 = np.maximum(np.sum(e1 * e1, axis=0), 1e-12)
        s1 = np.clip(np.sum(q1 * e1, axis=0) / len2, 0, 1)
        end_dist = np.hypot(*(q1 - s1 * e1))
        near = np.isinf(t_hit) & (end_dist < EnvConfig.ccd_gap)
        t_hit = np.where(near, 1.0, t_hit)
        s_hit = np.where(near, s1, s_hit)

        hit = np.isfinite(t_hit)
        return pts[hit], eids[hit], t_hit[hit], s_hit[hit], side[hit]

    def _resolve_ccd(self, h):
        """Make the straight-line motion PREV -> PREV + V h collision-free by editing V."""
        V, W, X0, E = self.V, self.W, self.PREV, self.edges
        push = EnvConfig.ccd_push
        pairs = self._ccd_pairs(V * h)
        for _ in range(EnvConfig.ccd_passes):
            pts, eids, ts, ss, sides = self._ccd_detect(V * h, pairs)
            if pts.size == 0:
                # impulses changed the paths, so confirm with a fresh broad phase before declaring the substep clean
                pairs = self._ccd_pairs(V * h)
                pts, eids, ts, ss, sides = self._ccd_detect(V * h, pairs)
                if pts.size == 0:
                    return
            # sequential (Gauss-Seidel) impulses, earliest collision first
            for k in np.argsort(ts):
                i, ia, ib, t, s = pts[k], E[0, eids[k]], E[1, eids[k]], ts[k], ss[k]
                # push direction: from the closest point on the edge to the point, at the start of the substep.
                # Mid-edge that's the edge normal; near a corner it's corner -> point, which both edges meeting
                # there agree on (per-edge normals would ping-pong the point between them).
                e0 = X0[:, ib] - X0[:, ia]
                u = min(max((X0[:, i] - X0[:, ia]) @ e0 / max(e0 @ e0, 1e-12), 0.0), 1.0)
                sep = X0[:, i] - (X0[:, ia] + u * e0)
                if np.hypot(*sep) > 1e-9:
                    n = sep / np.hypot(*sep)
                else:
                    e = (X0[:, ib] + t * h * V[:, ib]) - (X0[:, ia] + t * h * V[:, ia])
                    n = sides[k] * np.array([-e[1], e[0]]) / max(np.hypot(*e), 1e-12)  # points to the start side
                v_rel = V[:, i] - ((1 - s) * V[:, ia] + s * V[:, ib])
                # signed gap at the end of the substep, measured at the contact point
                end_p = X0[:, i] + h * V[:, i]
                end_c = (1 - s) * (X0[:, ia] + h * V[:, ia]) + s * (X0[:, ib] + h * V[:, ib])
                dvn = max((push - (end_p - end_c) @ n) / h, -(v_rel @ n), 0.0)
                denom = W[i] + W[ia] * (1 - s) ** 2 + W[ib] * s ** 2
                if dvn == 0.0 or denom < 1e-12:
                    continue
                j = dvn / denom * n
                V[:, i] += W[i] * j
                V[:, ia] -= W[ia] * (1 - s) * j
                V[:, ib] -= W[ib] * s * j

        self.zone_count += 1
        self._impact_zones(h)

    def _impact_zones(self, h):
        """Fail-safe: points that keep colliding are merged into zones that move rigidly for this substep.

        A rigid 2D motion of a zone (here a similarity transform over the substep) can't make its own members
        collide, so each round either finishes or merges zones; at worst everything stops, which is the valid start.
        """
        V, W, X0, E = self.V, self.W, self.PREV, self.edges
        n_pts = X0.shape[1]
        parent = np.arange(n_pts)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _ in range(n_pts):
            pts, eids, *_ = self._ccd_detect(V * h)
            if pts.size == 0:
                return
            merged = False
            for i, e in zip(pts, eids):
                for j in (E[0, e], E[1, e]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
                        merged = True
            roots = np.array([find(x) for x in range(n_pts)])
            for root in np.unique(roots[pts]):
                members = np.flatnonzero(roots == root)
                if (W[members] == 0).any():  # touching something fixed: the zone holds still this substep
                    V[:, members] = 0.0
                    continue
                m = 1.0 / W[members]
                x, v = X0[:, members], V[:, members]
                com, v_com = (x * m).sum(axis=1) / m.sum(), (v * m).sum(axis=1) / m.sum()
                r, u = x - com[:, None], v - v_com[:, None]
                inertia = (m * (r * r).sum(axis=0)).sum()
                omega = (m * (r[0] * u[1] - r[1] * u[0])).sum() / inertia if inertia > 1e-12 else 0.0
                V[:, members] = v_com[:, None] + omega * np.array([-r[1], r[0]])
            if not merged:
                break  # collisions inside a rigid zone can only be rounding; stop everything below
        V[:, W > 0] = 0.0  # last resort: no motion this substep, i.e. the (valid) start positions
