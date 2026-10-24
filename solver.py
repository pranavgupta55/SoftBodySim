from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pygame

from entities import Body, EnvConfig, RigidBody, SoftBody
import looping

_NO_SECTION = nullcontext()  # stands in for profiler sections when no profiler is attached


def _make_backend(world):
    """Helper object for EnvConfig.backend: None runs everything in this file ("cpu"), "gpu" moves the broad
    phases and the PBD solves to the Apple GPU (gpu_backend.py)."""
    if EnvConfig.backend == "cpu":
        return None
    if EnvConfig.backend == "gpu":
        from gpu_backend import GpuBackend
        return GpuBackend(world)
    raise ValueError(f"unknown EnvConfig.backend {EnvConfig.backend!r}, expected 'cpu' or 'gpu'")


def _outline_gap(a, b):
    """Smallest distance from a point of body a to an edge of body b (World._bounce)."""
    p, q0, q1 = a.points, b.points[:, b.edges[0]], b.points[:, b.edges[1]]
    e = q1 - q0  # (2, m)
    d = p[:, :, None] - q0[:, None, :]  # (2, n, m)
    t = np.clip(np.sum(d * e[:, None, :], axis=0) / np.maximum(np.sum(e * e, axis=0), 1e-12), 0.0, 1.0)
    off = d - t * e[:, None, :]
    return float(np.sqrt(np.min(off[0] ** 2 + off[1] ** 2)))


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
        self.hooks = []  # entities.Hook: fixed points for ropes; drawn only, never stepped or collided
        self.grabbed = None  # global index of the point being dragged
        self.grab_target = np.zeros(2)
        self.zone_count = 0  # how many substeps needed the rigid impact zone fallback (for debugging)
        self.profiler = None  # optional profiler.Profiler: step() then times its phases (see _section)
        self.backend = None  # EnvConfig.backend helper, None for "cpu" (see _make_backend)
        # Looping (looping.py): a body that fully leaves `bounds` through a looping side re-enters through the opposite
        # one, same other coordinate and velocities. Done between frames, never inside a substep's CCD-checked motion.
        # Change it with set_looping, which also removes / re-adds that side's wall.
        self.looping = {side: False for side in looping.SIDES}
        self.bounds = None  # (x0, y0, x1, y1) world rectangle bodies wrap around; set by looping.add_world_box
        self.walls = {}  # side -> its world box wall (a FixedBody), whether or not it's in bodies right now
        self.pending_walls = set()  # sides whose looping was turned off, waiting for room to re-add the wall
        self.last_wraps = []  # [(body, (dx, dy))] moved by looping in the last step(), e.g. to break motion trails
        self.env = {}  # per-world EnvConfig overrides (attribute name -> value), in effect only during step()
        self.bounce_bodies = []  # indices of bodies with restitution > 0 (see _bounce)

    def add(self, body, index=None):
        """Add a body (at the end, or at list position `index`: bodies draw in list order, so a container goes
        before what it holds). Repacks the arrays, so global point indices (e.g. a held grab) shift."""
        if index is None:
            self.bodies.append(body)
        else:
            self.bodies.insert(index, body)
            self.grabbed = None
        self._rebuild()

    def set_looping(self, side, on):
        """Looping for one side ("left", "right", "top", "bottom"): on removes that side's wall now; off re-adds it
        once nothing is beyond its line (pending until then). True if done now. See looping.py."""
        return looping.set_looping(self, side, on)

    def add_rope(self, rope):
        self.ropes.append(rope)

    def add_hook(self, hook):
        self.hooks.append(hook)

    def add_chain(self, chain):
        """An entities.Chain: its boxes (one repack) and then its ropes, in order along the chain."""
        self.bodies.extend(chain.bodies)
        self._rebuild()
        self.ropes.extend(chain.ropes)

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
        # per point / per edge lookups for the broad phase filters (index within the body, body size, endpoints)
        self.point_local = np.arange(self.P.shape[1]) - offsets[self.point_body]
        self.point_ring = self.body_counts[self.point_body]
        self.edge_local = self.edges - offsets[self.edge_body]
        self.bounce_bodies = [k for k, b in enumerate(bodies) if b.restitution > 0 and (b.inv_mass > 0).all()]
        self._pack_constraints()
        self.backend = _make_backend(self)

    def _pack_constraints(self):
        """Gather the bodies whose constraints World solves in one batch (stock SoftBody springs, RigidBody shape
        matching, Body.predict). Bodies whose class overrides those methods keep getting called one by one.
        The batches copy stiffness / rest shapes when a body is added, so change those before World.add."""
        bodies, offsets, n_pts = self.bodies, self.offsets, self.P.shape[1]
        self.batched_predict = all(type(b).predict is Body.predict for b in bodies)
        soft = [k for k, b in enumerate(bodies) if type(b).solve_constraints is SoftBody.solve_constraints]
        rigid = [k for k, b in enumerate(bodies) if type(b).solve_constraints is RigidBody.solve_constraints]
        batched = set(soft) | set(rigid)
        self.solo_bodies = [b for k, b in enumerate(bodies) if k not in batched
                            and type(b).solve_constraints is not Body.solve_constraints]

        # all springs as one Jacobi pass; the averaging count is 1 for points without springs (they get + 0)
        springs = [bodies[k].springs + offsets[k] for k in soft]
        self.springs = np.concatenate(springs, axis=1) if springs else np.empty((2, 0), dtype=int)
        self.spring_rest = np.concatenate([bodies[k].rest_len for k in soft]) if soft else np.empty(0)
        self.spring_stiffness = np.concatenate([np.full(bodies[k].springs.shape[1], float(bodies[k].stiffness))
                                                for k in soft]) if soft else np.empty(0)
        self.spring_count = np.ones(n_pts)
        for k in soft:
            self.spring_count[offsets[k]:offsets[k + 1]] = bodies[k].spring_count

        # rigid bodies grouped by point count, so each group is a (bodies, points) block
        groups = defaultdict(list)
        for k in rigid:
            groups[self.body_counts[k]].append(k)
        self.rigid_groups = [(np.array([np.arange(offsets[k], offsets[k + 1]) for k in ks]),  # (B, n) point ids
                              np.array([bodies[k].rest for k in ks]),  # (B, 2, n)
                              np.array([float(bodies[k].stiffness) for k in ks]))
                             for ks in groups.values()]

    def _solve_body_constraints(self):
        """Every body's own constraints, as if each body's solve_constraints ran (bodies don't share points)."""
        P, W = self.P, self.W
        if self.springs.shape[1]:  # SoftBody.solve_constraints for all soft bodies at once
            i, j = self.springs
            d = P[:, j] - P[:, i]
            dist = np.maximum(np.hypot(d[0], d[1]), 1e-9)
            wi, wj = W[i], W[j]
            corr = self.spring_stiffness * (dist - self.spring_rest) / (dist * (wi + wj)) * d
            ij = np.concatenate([i, j])
            for k in range(2):  # bincount adds in input order: i's shares then j's, like the per-body np.add.at
                acc = np.bincount(ij, np.concatenate([wi * corr[k], -wj * corr[k]]), minlength=P.shape[1])
                P[k] += acc / self.spring_count
        for idx, rest, stiffness in self.rigid_groups:  # RigidBody.solve_constraints, one group of same-size bodies
            pts = P[:, idx]  # (2, B, n)
            com = pts.mean(axis=2, keepdims=True)
            a = np.matmul((pts - com).transpose(1, 0, 2), rest.transpose(0, 2, 1))  # (B, 2, 2) covariances
            theta = np.arctan2(a[:, 1, 0] - a[:, 0, 1], a[:, 0, 0] + a[:, 1, 1])
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]]).transpose(2, 0, 1)  # (B, 2, 2)
            goal = np.matmul(rot, rest).transpose(1, 0, 2) + com
            P[:, idx] += stiffness[:, None] * (goal - pts)
        for b in self.solo_bodies:
            b.solve_constraints()

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

    def _section(self, name):
        """A profiler timing section, or a shared no-op context when no profiler is attached."""
        return _NO_SECTION if self.profiler is None else self.profiler.section(name)

    def step(self, dt):
        if not self.env:
            self._step(dt)
            return
        saved = {name: getattr(EnvConfig, name) for name in self.env}
        try:  # this world's overrides apply for the whole step (backend included), then the globals come back
            for name, value in self.env.items():
                setattr(EnvConfig, name, value)
            self._step(dt)
        finally:
            for name, value in saved.items():
                setattr(EnvConfig, name, value)

    def _step(self, dt):
        # before the substeps, where the state is settled and intersection-free
        self.last_wraps = looping.loop_step(self) if self.pending_walls or any(self.looping.values()) else []
        dt = min(dt, EnvConfig.max_dt)
        h = dt / EnvConfig.substeps
        sec = self._section
        for _ in range(EnvConfig.substeps):
            if self.bounce_bodies:
                self._bounce(h)
            if self.backend is None or not self.backend.pbd_substep(h):  # False: the backend can't run this scene
                self._pbd_substep(h)
            self.V[:] = (self.P - self.PREV) / h
            with sec("ccd"):
                self._resolve_ccd(h)
            self.P[:] = self.PREV + self.V * h
            self.V *= max(0.0, 1.0 - EnvConfig.damping * h)

    # ---------------- Opt-in restitution (Body.restitution > 0), e.g. the Newton's cradle
    def _bounce(self, h):
        """Elastic velocity pass between bodies that both have restitution > 0, at the start of a substep.

        PBD contacts and CCD impulses are inelastic, and between touching bodies the Jacobi contacts spread a hit
        over the whole row. So before the substep's PBD, pairs that are touching, or would get within
        contact_radius during this substep at their current closing speed, swap momentum like balls: an impulse
        along the line between the centres of mass with restitution min(e_a, e_b), applied as one velocity change to
        every point of the body (no spin). Gauss-Seidel sweeps until no such pair is closing, so a hit travels down
        a row of touching balls within one substep. Only velocities change here; the substep's PBD and CCD still
        move the positions, so the no-intersection guarantee is untouched.
        """
        r = EnvConfig.contact_radius
        bodies = [b for b in (self.bodies[k] for k in self.bounce_bodies) if (b.inv_mass > 0).all()]
        if len(bodies) < 2:
            return
        mass = [float(np.sum(1.0 / b.inv_mass)) for b in bodies]
        com = [(b.points / b.inv_mass).sum(axis=1) / m for b, m in zip(bodies, mass)]
        vel = [(b.vels / b.inv_mass).sum(axis=1) / m for b, m in zip(bodies, mass)]
        reach = [float(np.hypot(*(b.points - c[:, None])).max()) for b, c in zip(bodies, com)]
        # pairs that could touch this substep, allowing the impulses below to speed bodies up to 2x the fastest
        # (equal masses never pass it). A pair missed here still gets the usual inelastic contacts, never a crossing
        far = 1.1 * r + 4.0 * h * max(float(np.hypot(*v)) for v in vel)
        pairs = []
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                d = com[j] - com[i]
                dist = float(np.hypot(*d))
                if dist < 1e-9 or dist > reach[i] + reach[j] + far:
                    continue  # bounding circles too far apart to touch this substep
                gap = min(_outline_gap(bodies[i], bodies[j]), _outline_gap(bodies[j], bodies[i]))
                if gap < far:
                    e = min(bodies[i].restitution, bodies[j].restitution)
                    pairs.append((i, j, d / dist, gap, e, 1.0 / mass[i] + 1.0 / mass[j]))
        if not pairs:
            return
        before = [v.copy() for v in vel]
        for _ in range(4 * len(pairs) + 4):
            hit = False
            for i, j, n, gap, e, w in pairs:
                closing = float(n @ (vel[i] - vel[j]))
                # touching now, or closing in to within contact_radius during this substep at the current speeds
                if closing > 1e-9 and gap - h * closing < 1.1 * r:
                    jn = (1.0 + e) * closing / w * n
                    vel[i] = vel[i] - jn / mass[i]
                    vel[j] = vel[j] + jn / mass[j]
                    hit = True
            if not hit:
                break
        for b, v0, v in zip(bodies, before, vel):
            if (v != v0).any():
                b.vels += (v - v0)[:, None]  # in place: a view of self.V

    def _pbd_substep(self, h):
        """Predict, drag, then iterations of body constraints, ropes and soft contacts (moves P, from PREV)."""
        sec = self._section
        with sec("predict+grab"):
            if self.batched_predict:  # Body.predict for every body at once
                self.V += EnvConfig.gravity[:, None] * h * (self.W > 0)
                self.PREV[:] = self.P
                self.P += self.V * h
            else:
                for b in self.bodies:
                    b.predict(h)
            self._apply_grab()
        with sec("contacts"):
            candidates = self._candidates()
            reach = None  # (candidates within reach, positions when they were picked), see _solve_contacts
        for _ in range(EnvConfig.iterations):
            with sec("constraints"):
                self._solve_body_constraints()
                for r in self.ropes:
                    r.solve_constraints()
            with sec("contacts"):
                reach = self._solve_contacts(candidates, reach)

    def draw(self, surface, filled=False, zoom=1.0):
        """Bodies in the order they were added (later ones on top). filled: solid polygons instead of points + springs.
        zoom maps world units to surface pixels."""
        for b in self.bodies:
            if filled:
                b.draw_filled(surface, zoom=zoom)
            else:
                b.draw(surface, zoom)
        for r in self.ropes:
            r.draw(surface, zoom)
        for hk in self.hooks:  # over the ropes, so a rope ends inside its hook
            hk.draw(surface, zoom, filled)
        if self.grabbed is not None:
            pygame.draw.line(surface, (255, 255, 255), self.P[:, self.grabbed] * zoom, self.grab_target * zoom, 1)

    def draw_grid(self, surface, line_col=(52, 58, 76), edge_cell_col=(37, 42, 55), point_cell_col=(45, 51, 68), zoom=1.0):
        """Debug view of the broad phase grid: cell (i, j) covers [i, i + 1) * cell_size, same as the solver's floor().

        Cells an edge is registered in (its AABB padded like _candidates does) get edge_cell_col,
        cells holding a point get point_cell_col (drawn over), then the grid lines on top.
        """
        cell = EnvConfig.cell_size
        w, h = surface.get_size()
        cols, rows = int(np.ceil(w / zoom / cell)), int(np.ceil(h / zoom / cell))
        cell_px = cell * zoom  # cells are hashed in world units, drawn in surface pixels
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
                x0, y0 = int(np.floor(i * cell_px)), int(np.floor(j * cell_px))
                rect = (x0, y0, int(np.floor((i + 1) * cell_px)) - x0, int(np.floor((j + 1) * cell_px)) - y0)
                surface.fill(point_cell_col if kind == 2 else edge_cell_col, rect)
        for i in range(1, cols):
            x = int(np.floor(i * cell_px))
            pygame.draw.line(surface, line_col, (x, 0), (x, h - 1))
        for j in range(1, rows):
            y = int(np.floor(j * cell_px))
            pygame.draw.line(surface, line_col, (0, y), (w - 1, y))

    # ---------------- Broad phase: spatial hash grid (vectorized: cells are sorted keys instead of a dict)
    @staticmethod
    def _cells(lo, hi, origin, rows):
        """Every grid cell of the cell-index boxes lo..hi (inclusive, shape (2, m)): (box id, cell key) per cell.
        Keys number the cells column by column from `origin`, `rows` cells per column."""
        span = hi - lo + 1
        counts = span[0] * span[1]
        owner = np.repeat(np.arange(counts.size), counts)
        k = np.arange(owner.size) - np.repeat(np.cumsum(counts) - counts, counts)  # cell number within its box
        cx = lo[0, owner] + k % span[0, owner]
        cy = lo[1, owner] + k // span[0, owner]
        return owner, (cx - origin[0]) * rows + (cy - origin[1])

    def _cell_pairs(self, pt_lo, pt_hi, edge_lo, edge_hi):
        """(point, edge) pairs whose cell boxes share at least one cell, sorted by (point, edge), no duplicates."""
        if pt_lo.shape[1] == 0 or edge_lo.shape[1] == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        origin = np.minimum(pt_lo.min(axis=1), edge_lo.min(axis=1))
        rows = max(pt_hi[1].max(), edge_hi[1].max()) - origin[1] + 1
        pt_owner, pt_keys = self._cells(pt_lo, pt_hi, origin, rows)
        edge_owner, edge_keys = self._cells(edge_lo, edge_hi, origin, rows)
        n_edges = edge_lo.shape[1]
        # edge cells sorted by (cell, edge id): one sort of a combined key is faster than a stable argsort
        edge_keys, edge_owner = np.divmod(np.sort(edge_keys * n_edges + edge_owner), n_edges)
        start = np.searchsorted(edge_keys, pt_keys, "left")
        hits = np.searchsorted(edge_keys, pt_keys, "right") - start  # edges sharing each point cell
        first = np.repeat(start - (np.cumsum(hits) - hits), hits)
        i, e = np.repeat(pt_owner, hits), edge_owner[first + np.arange(first.size)]
        if pt_owner.size == pt_lo.shape[1]:
            return i, e  # one cell per point: already in (point, edge) order, and a pair can't repeat
        # a pair sharing several cells shows up once per cell: sort and dedupe
        code = np.sort(i * n_edges + e)
        code = code[np.r_[True, code[1:] != code[:-1]]]
        return np.divmod(code, n_edges)

    def _filter_pairs(self, i, e, contact):
        """Drop pairs that can't or needn't collide. Different bodies always pair unless both are fixed. Same body
        only if it self-collides, and not the point's own edges; for contacts (contact=True) also not its
        neighbors' edges, which are always close."""
        pb, eb = self.point_body[i], self.edge_body[e]
        keep = (pb != eb) & ~(self.body_fixed[pb] & self.body_fixed[eb])
        s = np.flatnonzero((pb == eb) & self.body_self[pb])  # self-collision candidates (usually few)
        si, se = i[s], e[s]
        if contact:
            li, n = self.point_local[si], self.point_ring[si]
            va, vb = self.edge_local[0, se], self.edge_local[1, se]
            gap_a = np.minimum((li - va) % n, (va - li) % n)
            gap_b = np.minimum((li - vb) % n, (vb - li) % n)
            keep[s] = (gap_a > 1) & (gap_b > 1)
        else:
            keep[s] = (si != self.edges[0, se]) & (si != self.edges[1, se])
        return i[keep], e[keep]

    def _candidates(self):
        """(point, edge) pairs that share a grid cell. Edges are registered in every cell their padded AABB touches."""
        cell, pad = EnvConfig.cell_size, 2 * EnvConfig.contact_radius
        a, b = self.P[:, self.edges[0]], self.P[:, self.edges[1]]
        lo = np.floor((np.minimum(a, b) - pad) / cell).astype(np.int64)
        hi = np.floor((np.maximum(a, b) + pad) / cell).astype(np.int64)
        point_cells = np.floor(self.P / cell).astype(np.int64)
        return self._filter_pairs(*self._cell_pairs(point_cells, point_cells, lo, hi), contact=True)

    # ---------------- Narrow phase + resolution: point vs edge capsule
    def _solve_contacts(self, candidates, reach=None):
        """One Jacobi pass over the (pts, edge_ids) candidates. Returns a `reach` to pass to the next pass of the
        same substep: the candidates within contact_radius + slack, and P when they were picked. A point's distance
        to a segment changes by at most its own motion plus the farther endpoint's, so while no point has moved
        slack / 2 since then, the other candidates are out of range and are skipped (same result, less work)."""
        r, mu = EnvConfig.contact_radius, EnvConfig.friction
        P, PREV, W = self.P, self.PREV, self.W
        slack = r
        if reach is not None and np.hypot(*(P - reach[2])).max() < 0.5 * slack - 1e-6:
            pts, edge_ids = reach[0], reach[1]
        else:
            pts, edge_ids = candidates
            reach = None
        if pts.size == 0:
            return reach
        ia, ib = self.edges[0, edge_ids], self.edges[1, edge_ids]

        p, a, b = P[:, pts], P[:, ia], P[:, ib]
        e = b - a
        len2 = np.maximum(np.sum(e * e, axis=0), 1e-12)
        t = np.clip(np.sum((p - a) * e, axis=0) / len2, 0.0, 1.0)
        c = a + t * e
        d = p - c
        dist = np.hypot(d[0], d[1])
        if reach is None:
            keep = dist < r + slack
            reach = (pts[keep], edge_ids[keep], P.copy())

        # most candidates are out of range: keep only the touching ones (everything below is per pair)
        near = np.flatnonzero(dist < r)
        if near.size == 0:
            return reach
        pts, edge_ids, ia, ib, t, len2, dist = (x[near] for x in (pts, edge_ids, ia, ib, t, len2, dist))
        p, c, d, e = (x[:, near] for x in (p, c, d, e))

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
        active = denom > 1e-12
        if not active.any():
            return reach

        # friction: cancel part of the point's sliding relative to the contact point on the edge
        c_prev = PREV[:, ia] + t * (PREV[:, ib] - PREV[:, ia])
        rel = (p - PREV[:, pts]) - (c - c_prev)
        tang = rel - np.sum(rel * n, axis=0) * n
        tang_len = np.hypot(tang[0], tang[1])
        fric = np.minimum(1.0, mu * depth / np.maximum(tang_len, 1e-9))
        delta = depth * n - tang * fric

        lam = delta / np.where(active, denom, 1.0)
        # Jacobi: average when a point is hit by several contacts. bincount sums in input order (the point's
        # shares, then edge start, then edge end), the same order as one np.add.at per term
        idx = np.concatenate([pts[active], ia[active], ib[active]])
        shares = [s[active] for s in (wp, -wa * (1 - t), -wb * t)]
        count = np.maximum(np.bincount(idx, minlength=P.shape[1]), 1.0)
        for k in range(2):
            acc = np.bincount(idx, np.concatenate([s * lam[k][active] for s in shares]), minlength=P.shape[1])
            P[k] += acc / count
        return reach

    # ---------------- Continuous collision detection (the no-intersection guarantee)
    def _ccd_boxes(self, D):
        """Swept boxes of the straight-line paths PREV -> PREV + D: (point lo, point hi, edge lo, edge hi), each
        (2, m). Edge boxes are grown by 1.0 px, which also covers the ccd_gap end check."""
        X0 = self.PREV
        X1 = X0 + D
        a0, b0, a1, b1 = X0[:, self.edges[0]], X0[:, self.edges[1]], X1[:, self.edges[0]], X1[:, self.edges[1]]
        return (np.minimum(X0, X1), np.maximum(X0, X1),
                np.minimum(np.minimum(a0, b0), np.minimum(a1, b1)) - 1.0,
                np.maximum(np.maximum(a0, b0), np.maximum(a1, b1)) + 1.0)

    def _ccd_cells(self, D):
        """Grid cell ranges of the swept boxes: (point lo, point hi, edge lo, edge hi) cell indices."""
        return [np.floor(box / EnvConfig.cell_size).astype(int) for box in self._ccd_boxes(D)]

    def _ccd_pairs(self, D, cells=None):
        """(point, edge) pairs whose swept boxes (start -> start + D) share a grid cell. `cells`: _ccd_cells(D)."""
        cells = self._ccd_cells(D) if cells is None else cells
        pairs = None if self.backend is None else self.backend.ccd_pairs(cells)  # None: use the CPU code below
        if pairs is not None:
            return pairs
        return self._filter_pairs(*self._cell_pairs(*cells), contact=False)

    def _ccd_new_pairs(self, old_cells, cells):
        """The pairs among _ccd_pairs for `cells` that may be missing from the pairs for `old_cells`: a pair can
        only be new if its point's or its edge's cell range grew, so pair just those against everything."""
        p_lo, p_hi, e_lo, e_hi = cells
        grew_p = np.flatnonzero(((p_lo < old_cells[0]) | (p_hi > old_cells[1])).any(axis=0))
        grew_e = np.flatnonzero(((e_lo < old_cells[2]) | (e_hi > old_cells[3])).any(axis=0))
        i1, e1 = self._cell_pairs(p_lo[:, grew_p], p_hi[:, grew_p], e_lo, e_hi)
        i2, e2 = self._cell_pairs(p_lo, p_hi, e_lo[:, grew_e], e_hi[:, grew_e])
        return self._filter_pairs(np.concatenate([grew_p[i1], i2]), np.concatenate([e1, grew_e[e2]]), contact=False)

    def _ccd_detect(self, D, pairs=None):
        """Points that touch an edge at some time in the substep, moving in straight lines from PREV by D.

        A point p(t) is on the line through a(t), b(t) when cross(b - a, p - a) = 0, a quadratic in t in [0, 1];
        it's a hit if the point is also within the segment then. Ending closer than ccd_gap also counts, so states
        never get so close that rounding decides which side a point is on. Returns (pts, edge_ids, t, s, side).
        `pairs` reuses an earlier broad phase; only a fresh one (the default) can prove there are no collisions.
        """
        pts, eids = self._ccd_pairs(D) if pairs is None else pairs
        empty = np.empty(0)
        X0 = self.PREV
        if pts.size:
            # Cheap pre-reject: a hit puts the point on the edge at some time t, so the point's swept box and the
            # edge's swept box (grown by 1.0 px, more than ccd_gap) overlap. Grid cells are much coarser than that,
            # so this drops most broad phase pairs before the quadratic, without changing any result. Except: an edge
            # that shrinks to (nearly) zero length mid-substep makes the cross product vanish for every point, which
            # the math below reports as hits even for far points; keep those pairs so the results stay the same.
            p_lo, p_hi, e_lo, e_hi = self._ccd_boxes(D)
            ea, eb = self.edges
            e0, de = X0[:, eb] - X0[:, ea], D[:, eb] - D[:, ea]
            t_min = np.clip(-np.sum(e0 * de, axis=0) / np.maximum(np.sum(de * de, axis=0), 1e-300), 0.0, 1.0)
            shortest = e0 + t_min * de  # each edge at its shortest during the substep
            collapses = np.sum(shortest * shortest, axis=0) < 1e-6
            overlap = np.all((p_lo[:, pts] <= e_hi[:, eids]) & (p_hi[:, pts] >= e_lo[:, eids]), axis=0)
            keep = overlap | collapses[eids]
            pts, eids = pts[keep], eids[keep]
        if pts.size == 0:
            return pts, eids, empty, empty, empty
        ia, ib = self.edges[0, eids], self.edges[1, eids]
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
        V, sec = self.V, self._section
        with sec("broad"):
            D = V * h
            cells = self._ccd_cells(D)
            pairs = self._ccd_pairs(D, cells)
        with sec("narrow+impulses"):
            hits = self._ccd_detect(D, pairs)
        fresh = True  # V hasn't changed since the broad phase
        for _ in range(EnvConfig.ccd_passes):
            if hits[0].size == 0:
                if fresh:
                    return
                # Impulses changed the paths, so confirm with a fresh broad phase before declaring the substep clean.
                # Every current pair already in `pairs` was just checked (no hits), so first look only at the pairs
                # the fresh broad phase could add; if none of them hits either, the fresh check would come up empty.
                D = V * h
                with sec("broad"):
                    new_cells = self._ccd_cells(D)
                    extra = self._ccd_new_pairs(cells, new_cells)
                with sec("narrow+impulses"):
                    if self._ccd_detect(D, extra)[0].size == 0:
                        return
                with sec("broad"):
                    cells = new_cells
                    pairs = self._ccd_pairs(D, cells)
                with sec("narrow+impulses"):
                    hits = self._ccd_detect(D, pairs)
                if hits[0].size == 0:
                    return
            with sec("narrow+impulses"):
                V_before = V.copy()
                self._ccd_impulses(*hits, h)
                hits = self._ccd_redetect(V * h, pairs, hits, (V != V_before).any(axis=0))
            fresh = False

        self.zone_count += 1
        with sec("impact zones"):
            self._impact_zones(h)

    def _ccd_redetect(self, D, pairs, hits, changed):
        """_ccd_detect(D, pairs) after impulses that changed only the `changed` points' velocities. A pair's result
        depends only on its three points, so pairs not touching a changed point keep their result from `hits`."""
        E = self.edges

        def touches(pts, eids):
            return changed[pts] | changed[E[0, eids]] | changed[E[1, eids]]

        pts, eids = pairs
        moved = touches(pts, eids)
        new = self._ccd_detect(D, (pts[moved], eids[moved]))
        kept = ~touches(hits[0], hits[1])
        merged = [np.concatenate([old[kept], fresh]) for old, fresh in zip(hits, new)]
        order = np.argsort(merged[0] * E.shape[1] + merged[1], kind="stable")  # back to (point, edge) order
        return tuple(a[order] for a in merged)

    def _ccd_impulses(self, pts, eids, ts, ss, sides, h):
        """One pass of sequential (Gauss-Seidel) impulses over detected collisions, earliest first, editing V."""
        V, W, X0, E = self.V, self.W, self.PREV, self.edges
        push = EnvConfig.ccd_push
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
