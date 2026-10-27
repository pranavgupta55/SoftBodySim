from collections import defaultdict
from contextlib import nullcontext
import copy
import math
import time

import numpy as np
import pygame

from entities import Body, EnvConfig, RigidBody, SoftBody
import looping
import solver_jit

_NO_SECTION = nullcontext()  # stands in for profiler sections when no profiler is attached


def _jit():
    """Whether the hot CPU paths run as compiled kernels (solver_jit.py, bit-identical to the numpy code here):
    EnvConfig.jit on and numba importable. The numpy code stays the reference and the fallback."""
    return EnvConfig.jit and solver_jit.enabled()


# EnvConfig.backend = "auto": a world runs on the GPU once it has AUTO_GPU_POINTS movable points (inverse mass > 0),
# and goes back to the CPU only below AUTO_CPU_POINTS (hysteresis, so spawning / deleting near the line doesn't flip
# it). Re-measured 2026-10-24 with the numba CPU path (solver_jit.py) and frame-level CCD on both backends, Apple M4
# (10-core GPU), world 2244x1440, physics ms/frame CPU / GPU (best of 2 interleaved runs, 120 frames from 0.5 s in;
# fluidN = the fluid preset with N hexagons):
#   random walls 35 movable pts 0.55 / 2.16, fluid30 114 0.68 / 2.11, fluid60 324 1.12 / 2.23, fluid100 528
#   1.55 / 2.63, fluid150 774 2.16 / 2.40, diamonds 840 4.15 / 3.27, fluid450 2478 9.5 / 4.4, fluid1000 5766
#   45.6 / 11.4, fluid2000 11592 191 / 44
# The GPU has a ~2 ms floor per frame; it pulls ahead between ~780 and ~840 movable points and wins 2-4x on big
# scenes. (The first numpy-vs-GPU measurement put the line at 900: dense piles crossed at ~550, diamonds lost.)
AUTO_GPU_POINTS = 800
AUTO_CPU_POINTS = 650


# XPBD spring compliance from a SoftBody's PBD stiffness (EnvConfig.xpbd). A PBD spring in a Jacobi pass removes
# about stiffness / count of its error per pass (count = springs at its points, the averaging divisor), so
# f = 1 - (1 - stiffness / count)^I per substep of I passes. Against a steady load that balances like a spring of
# compliance alpha = (w_i + w_j) h^2 / f (w = inverse masses): stiffer with shorter substeps, softer with longer ones.
# XPBD fixes alpha at its value for today's reference schedule (h = 1/240 s, 4 passes), so a body feels the same as
# with PBD there and keeps that stiffness at any other substep length or count. The (w_i + w_j) factor keeps PBD's
# mass independence (a heavy body sags as far as a light one). XPBD_CALIBRATION scales the result; measured, 1.0
# (the formula as is) matches PBD at 4 x 4 best: rest sag (height / rest height) of a resting 12-gon 0.809 vs 0.810,
# hexagon 0.978 vs 0.980, heptagon 0.989 vs 0.990, 8-point box 0.994 vs 0.995 (stiffness 0.1, skip 2; 0.7 / 0.5 give
# 0.838 / 0.858 on the 12-gon). The Jacobi passes don't converge the solve at long substeps, so a body is still
# softer there than at 1/240 s (12-gon at 1/60 s: 0.64 vs PBD's 0.42), but far less so than with PBD.
XPBD_REF_H = 1 / 240
XPBD_REF_ITERATIONS = 4
XPBD_CALIBRATION = 1.0


def spring_compliance(stiffness, count):
    """Per-spring XPBD compliance per unit inverse mass (alpha / (w_i + w_j)) for PBD stiffness and the springs' mean
    point spring count (see XPBD_REF_H). Stiffness <= 0 gives a near-infinite compliance (no pull)."""
    f = 1.0 - (1.0 - np.clip(stiffness / count, 0.0, 1.0)) ** XPBD_REF_ITERATIONS
    return XPBD_CALIBRATION * XPBD_REF_H * XPBD_REF_H / np.maximum(f, 1e-12)


def _backend_kind(world):
    """"cpu" or "gpu": this world's backend setting (world.env, else EnvConfig.backend), with "auto" decided from the
    scene. Scenes GpuBackend.pbd_substep can't run (ropes, custom predict / solve_constraints, no edges) are "cpu"."""
    kind = getattr(world, "env", {}).get("backend", EnvConfig.backend)
    if kind != "auto":
        return kind
    if world.ropes or world.solo_bodies or not world.batched_predict or world.edges.shape[1] == 0:
        return "cpu"
    limit = AUTO_CPU_POINTS if world.backend is not None else AUTO_GPU_POINTS  # hysteresis: stay where it is
    return "gpu" if np.count_nonzero(world.W) >= limit else "cpu"


def _make_backend(world):
    """Helper object for the world's backend (_backend_kind): None runs everything in this file ("cpu"), "gpu" moves
    the broad phases and the PBD solves to the Apple GPU (gpu_backend.py)."""
    kind = _backend_kind(world)
    if kind == "cpu":
        return None
    if kind == "gpu":
        from gpu_backend import GpuBackend
        return GpuBackend(world)
    raise ValueError(f"unknown backend {kind!r}, expected 'auto', 'cpu' or 'gpu'")


def _outline_gap(a, b):
    """Smallest distance from a point of body a to an edge of body b (World._bounce)."""
    p, q0, q1 = a.points, b.points[:, b.edges[0]], b.points[:, b.edges[1]]
    e = q1 - q0  # (2, m)
    d = p[:, :, None] - q0[:, None, :]  # (2, n, m)
    t = np.clip(np.sum(d * e[:, None, :], axis=0) / np.maximum(np.sum(e * e, axis=0), 1e-12), 0.0, 1.0)
    off = d - t * e[:, None, :]
    return float(np.sqrt(np.min(off[0] ** 2 + off[1] ** 2)))


class World:
    """Owns all bodies and steps them. Per CCD span (EnvConfig.ccd_stride substeps, by default the whole frame):

    1. PBD substeps: predict, drag, then springs / shape matching + soft contacts (thickness and friction).
    2. Turn the span's motion into velocities and run continuous collision detection (CCD) on the straight-line
       paths from the span's start positions, with impulses until no point crosses an edge (Bridson et al. 2002).
    3. If impulses don't converge, merge colliding points into rigid impact zones until nothing collides.
    Since each span starts intersection-free and its whole motion is checked, it also ends intersection-free. The
    substeps inside a span are only a proposal for that motion and may overlap.
    """

    def __init__(self):
        self.bodies = []
        self.ropes = []  # tension-only constraints between body points / world anchors; not collidable
        self.hooks = []  # entities.Hook: fixed points for ropes; drawn only, never stepped or collided
        self.grabbed = None  # global index of the point being dragged
        self.grab_target = np.zeros(2)
        # area grab (grab with a radius): (point indices (k,), offsets (2, k), radius); point j is pulled toward
        # grab_target + its offset (in its own frame), so the held points keep their layout around the cursor
        self.grab_set = None
        self.zone_count = 0  # how many CCD passes needed the rigid impact zone fallback (for debugging)
        self.projection_count = 0  # how many CCD passes the inelastic projection fallback fixed (for debugging)
        self.profiler = None  # optional profiler.Profiler: step() then times its phases (see _section)
        self.backend = None  # EnvConfig.backend helper, None for "cpu" (see _make_backend); re-picked on every repack
        # Looping (looping.py): a body that fully leaves `bounds` through a looping side re-enters through the opposite
        # one, same other coordinate and velocities. Done between frames, never inside a substep's CCD-checked motion.
        # Change it with set_looping, which also removes / re-adds that side's wall.
        self.looping = {side: False for side in looping.SIDES}
        self.bounds = None  # (x0, y0, x1, y1) world rectangle bodies wrap around; set by looping.add_world_box
        self.walls = {}  # side -> its world box wall (a FixedBody), whether or not it's in bodies right now
        self.pending_walls = set()  # sides whose looping was turned off, waiting for room to re-add the wall
        self.last_wraps = []  # [(body, (dx, dy))] moved by looping in the last step(), e.g. to break motion trails
        # Periodic axes (looping.py): an axis whose two sides both loop is a torus for contacts and CCD. period is
        # (Lx, Ly), 0 on an axis that isn't periodic, or None when neither is (then nothing below changes at all).
        # Pairs found through an image carry the offset added to their point (_image_pairs). Set by loop_step.
        self.periodic = (False, False)
        self.period = None
        self.grab_shift = np.zeros(2)  # the held point's frame minus the cursor's (an image offset; see grab)
        self._image_topo = None  # (edges array, points, edges) that take part in image pairs (_image_topology)
        self._ccd_box_cache = None  # (D, _ccd_float_boxes(D)) with a periodic axis
        self.env = {}  # per-world EnvConfig overrides (attribute name -> value), in effect only during step()
        self.background = None  # per-world background color (rgb), None for the default (main_opengl.py)
        self.palette = None  # the palettes.py palette this world's colors come from (None: it has none)
        self.bounce_bodies = []  # indices of bodies with restitution > 0 (see _bounce)
        # Optional debug recorder (debug_overlay.StepRecorder), set only while an overlay needs solver data. It is
        # told about each step's CCD hits and impact zone points; None (the default) costs one attribute check
        self.debug = None
        # During a multi-substep CCD span (see _ccd_span): positions at its start, which are intersection-free, so
        # contacts take their sides from them. None when CCD runs after every substep (contacts then use PREV)
        self._span_start = None
        self._span_left = 0.0  # time from the current substep's start to the end of the span
        self._span_contacts = None  # (candidates, point box lo, point box hi), shared by the span's substeps
        self._span_reach = None  # _solve_contacts' reach, carried across the span's substeps
        self.contact_builds = 0  # contact candidate broad phases run (for debugging)
        self.last_sim_dt = 0.0  # seconds the last step() simulated (dt * scale after the max_dt clamp)
        self._h = XPBD_REF_H  # the current substep's length (XPBD springs), set by _pbd_substep
        self._substep_cost = None  # running mean of seconds per substep (EnvConfig.physics_budget)
        self.clock = time.perf_counter  # what _substep_cost is measured with (a benchmark may swap in thread time)

    def add(self, body, index=None):
        """Add a body (at the end, or at list position `index`: bodies draw in list order, so a container goes
        before what it holds). Repacks the arrays, so global point indices (e.g. a held grab) shift."""
        if index is None:
            self.bodies.append(body)
        else:
            self.bodies.insert(index, body)
            self.grabbed = self.grab_set = None
        self._rebuild()

    def set_looping(self, side, on):
        """Looping for one side ("left", "right", "top", "bottom"): on removes that side's wall now; off re-adds it
        once nothing is beyond its line (pending until then). True if done now. See looping.py."""
        return looping.set_looping(self, side, on)

    def add_rope(self, rope):
        self.ropes.append(rope)
        self._recheck_backend()

    def add_hook(self, hook):
        self.hooks.append(hook)

    def add_chain(self, chain):
        """An entities.Chain: its boxes and then its ropes, in order along the chain (one repack)."""
        self.bodies.extend(chain.bodies)
        self.ropes.extend(chain.ropes)
        self._rebuild()

    # ---------------- Editing (the editor's select / delete / paste, while paused; separate from step)
    def add_many(self, bodies=(), ropes=(), hooks=()):
        """Add bodies (at the end, in order), ropes and hooks with one repack."""
        self.bodies.extend(bodies)
        self.ropes.extend(ropes)
        self.hooks.extend(hooks)
        if bodies:
            self._rebuild()
        elif ropes:
            self._recheck_backend()

    def remove(self, bodies=(), ropes=(), hooks=()):
        """Remove bodies, ropes and hooks (by identity); ropes attached to a removed body or hook go too. One repack.
        Removed bodies get their own copies of their arrays, so they stay valid objects (e.g. to add back later).
        At least one body must remain (the packed arrays can't be empty)."""
        gone_b, gone_h, gone_r = {id(b) for b in bodies}, {id(h.pos) for h in hooks}, {id(r) for r in ropes}

        def attached(rope):  # an end on a removed body, or sharing a removed hook's position array
            return any(id(body) in gone_b if body is not None else id(ref) in gone_h for body, ref in rope.ends)

        self.ropes = [r for r in self.ropes if id(r) not in gone_r and not attached(r)]
        self.hooks = [hk for hk in self.hooks if id(hk.pos) not in gone_h]
        if gone_b:
            for b in self.bodies:
                if id(b) in gone_b:
                    b.points, b.prev, b.vels, b.inv_mass = b.points.copy(), b.prev.copy(), b.vels.copy(), b.inv_mass.copy()
            self.bodies = [b for b in self.bodies if id(b) not in gone_b]
            self.grabbed = self.grab_set = None
            self._rebuild()
        else:
            self._recheck_backend()

    def remove_body(self, body):
        self.remove(bodies=[body])

    def remove_rope(self, rope):
        self.remove(ropes=[rope])

    def remove_hook(self, hook):
        self.remove(hooks=[hook])

    def reorder(self, bodies):
        """Set the draw order: `bodies` is self.bodies in a new order (containers go before what they hold). Repacks."""
        assert sorted(map(id, bodies)) == sorted(map(id, self.bodies))
        self.bodies = list(bodies)
        self.grabbed = self.grab_set = None
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

    def _recheck_backend(self):
        """After a change that doesn't repack (ropes added / removed): switch backends only if the kind changed."""
        if self.bodies and (_backend_kind(self) == "gpu") != (self.backend is not None):
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
        # XPBD (EnvConfig.xpbd): compliance per unit inverse mass, and the mass-splitting counts of both ends
        i, j = self.springs
        self.spring_ci, self.spring_cj = self.spring_count[i], self.spring_count[j]
        self.spring_comp = spring_compliance(self.spring_stiffness, 0.5 * (self.spring_ci + self.spring_cj))
        self.spring_lambda = np.zeros(self.springs.shape[1])  # Lagrange multipliers, reset every substep

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
        if _jit():
            solver_jit.solve_body_constraints(self)
            for b in self.solo_bodies:
                b.solve_constraints()
            return
        P, W = self.P, self.W
        if self.springs.shape[1] and EnvConfig.xpbd:
            self._xpbd_springs()
        elif self.springs.shape[1]:  # SoftBody.solve_constraints for all soft bodies at once
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
            q, rx, ry = pts - com, rest[:, 0], rest[:, 1]
            # best-fit rotation: cos / sin of atan2(num, den) straight from the covariance sums, without matmul
            # (BLAS uses FMA) or trig (np.sin isn't libm's sin), so solver_jit's kernel matches it bit for bit
            num = (q[1] * rx).sum(axis=1) - (q[0] * ry).sum(axis=1)
            den = (q[0] * rx).sum(axis=1) + (q[1] * ry).sum(axis=1)
            norm = np.hypot(num, den)
            ok = norm > 0
            norm = np.where(ok, norm, 1.0)
            c = np.where(ok, den / norm, 1.0)[:, None]  # no rotation at all when num = den = 0 (atan2(0, 0) = 0)
            s = np.where(ok, num / norm, 0.0)[:, None]
            goal = np.array([c * rx - s * ry, s * rx + c * ry]) + com
            P[:, idx] += stiffness[:, None] * (goal - pts)
        for b in self.solo_bodies:
            b.solve_constraints()

    def _xpbd_springs(self):
        """One pass of every SoftBody spring as an XPBD distance constraint (Macklin, Müller, Chentanez 2016), Jacobi
        with mass splitting (Tonge et al. 2012: each end's inverse mass counted once per spring at that point, so the
        summed corrections need no averaging and the multipliers stay consistent), plus Rayleigh damping along the
        spring (XPBD eq. 26, EnvConfig.spring_damping). The multipliers carry across a substep's passes; _pbd_substep
        resets them. Same arithmetic, in the same order, as solver_jit.xpbd_springs."""
        P, W, h = self.P, self.W, self._h
        i, j = self.springs
        d = P[:, j] - P[:, i]
        dist = np.maximum(np.hypot(d[0], d[1]), 1e-9)
        nx, ny = d[0] / dist, d[1] / dist
        wi, wj = W[i], W[j]
        at = self.spring_comp * (wi + wj) / (h * h)  # alpha~ = alpha / h^2
        gamma = at * EnvConfig.spring_damping * h
        # how fast the spring is stretching this substep (relative motion since the substep's start, along it)
        rel = nx * ((P[0, j] - self.PREV[0, j]) - (P[0, i] - self.PREV[0, i])) + \
            ny * ((P[1, j] - self.PREV[1, j]) - (P[1, i] - self.PREV[1, i]))
        lam = self.spring_lambda
        num = -(dist - self.spring_rest) - at * lam - gamma * rel
        den = (1.0 + gamma) * (self.spring_ci * wi + self.spring_cj * wj) + at
        ok = den > 0  # both ends fixed: no correction
        dl = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
        lam += dl
        ij = np.concatenate([i, j])
        a, b = -wi * dl, wj * dl
        for k, n in enumerate((nx, ny)):  # bincount adds in input order: i's shares then j's
            P[k] += np.bincount(ij, np.concatenate([a * n, b * n]), minlength=P.shape[1])

    def _solve_ropes(self):
        """Every rope's solve_constraints, in list order (Gauss-Seidel)."""
        if self.ropes and _jit() and solver_jit.solve_ropes(self):
            return
        for r in self.ropes:
            r.solve_constraints()

    # ---------------- Mouse dragging
    def grab(self, pos, radius=None):
        """Pick up the nearest movable point within grab_radius of pos (if any). Across a periodic axis a point's
        images count too (a body straddling the seam can be grabbed on either visible copy): grab_shift then holds
        the image offset, and grab_target is the cursor moved into the point's frame.
        radius: an area grab instead, every movable point within `radius` of pos (grab_area)."""
        if radius is not None:
            return self.grab_area(pos, radius)
        dist = np.hypot(*(self.P - np.reshape(pos, (2, 1))))
        dist[self.W <= 0] = np.inf  # fixed points can't be dragged
        i = int(np.argmin(dist))
        shift = np.zeros(2)
        for s in looping.image_shifts(self.period):  # the point's copy at P + s under the cursor
            d = np.hypot(*(self.P + s[:, None] - np.reshape(pos, (2, 1))))
            d[self.W <= 0] = np.inf
            k = int(np.argmin(d))
            if d[k] < dist[i]:
                i, dist, shift = k, d, s
        if dist[i] <= EnvConfig.grab_radius:
            self.grabbed = i
            self.grab_shift = shift
            self.grab_target = np.array(pos, dtype=float) - shift if shift.any() else np.array(pos, dtype=float)

    def grab_area(self, pos, radius):
        """Pick up every movable point within `radius` of pos, of any bodies (on either copy across a periodic seam):
        grab_set. Each is pulled toward the cursor plus its offset from the cursor at the moment of the grab, so the
        held region moves with the cursor like one piece while springs and contacts act on it."""
        c = np.reshape(np.asarray(pos, dtype=float), (2, 1))
        dist = np.hypot(*(self.P - c))
        for s in looping.image_shifts(self.period):  # a point counts if any of its visible copies is in the circle
            dist = np.minimum(dist, np.hypot(*(self.P + s[:, None] - c)))
        # target = cursor + offset starts at the point itself and moves with the cursor (in the point's own frame,
        # so a copy across the seam is held where it is)
        off = self.P - c
        idx = np.flatnonzero((dist <= radius) & (self.W > 0))
        if idx.size:
            self.grabbed = None
            self.grab_shift = np.zeros(2)
            self.grab_target = np.array(pos, dtype=float)
            self.grab_set = (idx, off[:, idx].copy(), float(radius))

    def drag_to(self, pos):
        shift = self.grab_shift
        self.grab_target = np.array(pos, dtype=float) - shift if shift.any() else np.array(pos, dtype=float)

    def release(self):
        self.grabbed = self.grab_set = None
        self.grab_shift = np.zeros(2)

    def _apply_grab(self, h):
        """Nudge the held point toward the cursor once per substep of length h, so contacts (solved after) can
        out-vote the hand. The pull is set per second, so dragging feels the same at any substep count."""
        if self.grab_set is not None:  # an area grab: every held point toward its own target, each capped
            idx, off, _ = self.grab_set
            gap = self.grab_target[:, None] + off - self.P[:, idx]
            step = gap * -math.expm1(-EnvConfig.grab_stiffness * h)
            length = np.hypot(*step)
            max_step = EnvConfig.grab_max_speed * h
            step *= np.minimum(1.0, max_step / np.maximum(length, 1e-12))
            self.P[:, idx] += step
            return
        if self.grabbed is None:
            return
        gap = self.grab_target - self.P[:, self.grabbed]
        step = gap * -math.expm1(-EnvConfig.grab_stiffness * h)
        length = np.hypot(*step)
        max_step = EnvConfig.grab_max_speed * h
        if length > max_step:
            step *= max_step / length
        self.P[:, self.grabbed] += step

    def _section(self, name):
        """A profiler timing section, or a shared no-op context when no profiler is attached."""
        return _NO_SECTION if self.profiler is None else self.profiler.section(name)

    def step(self, dt, scale=1.0, frames=1.0):
        """Advance dt * scale seconds. scale is the time scale (slow motion, timectl.py): the step then runs
        round(substeps * scale) substeps (at least 1) and clamps at max_dt * scale, so the substep length stays what
        it is at scale 1 (substeps / max_dt from EnvConfig, or this world's env overrides). frames: how many 1/60 s
        frames this step stands for (EnvConfig.substeps are per 1/60 s frame), e.g. 0.5 at 120 fps: half the
        substeps, so the substep length doesn't depend on the frame rate. Above EnvConfig.max_substeps per 1/60 s
        (high time scales) the substeps get longer instead, up to EnvConfig.max_substep_dt. Sets last_sim_dt to the
        seconds actually simulated."""
        if not self.env:
            self._step(dt, scale, frames)
            return
        saved = {name: getattr(EnvConfig, name) for name in self.env}
        try:  # this world's overrides apply for the whole step (backend included), then the globals come back
            for name, value in self.env.items():
                setattr(EnvConfig, name, value)
            self._step(dt, scale, frames)
        finally:
            for name, value in saved.items():
                setattr(EnvConfig, name, value)

    def _step(self, dt, scale=1.0, frames=1.0):
        # before the substeps, where the state is settled and intersection-free
        if self.pending_walls or any(self.looping.values()):
            self.last_wraps = looping.loop_step(self)
        else:
            self.last_wraps = []
            if self.period is not None:
                looping.clear_periodic(self)
        if scale == 1.0 and frames == 1.0:
            dt, n = min(dt, EnvConfig.max_dt), EnvConfig.substeps
        else:
            dt, n = min(dt, EnvConfig.max_dt) * scale, max(1, round(EnvConfig.substeps * scale * frames))
        # High time scales: fewer, longer substeps than round(substeps * scale * frames) when over a cap (see EnvConfig).
        # Never below the scale-1 count, so steps at scale <= 1 are untouched (and deterministic)
        base = max(1, round(EnvConfig.substeps * frames))
        if EnvConfig.max_substeps is not None:
            n = min(n, max(base, round(EnvConfig.max_substeps * frames)))
        if EnvConfig.physics_budget is not None and self._substep_cost:
            n = min(n, max(base, int(EnvConfig.physics_budget * frames / self._substep_cost)))
        if EnvConfig.max_substep_dt is not None:  # but never longer than max_substep_dt
            n = max(n, math.ceil(dt / EnvConfig.max_substep_dt - 1e-9))
        self.last_sim_dt = dt  # seconds this step simulates (after the max_dt clamp and the time scale)
        t0 = self.clock()
        self._substeps(n, dt / n)
        # running mean of the time a substep takes, for physics_budget (the first one includes no compile stalls:
        # solver_jit compiles at import / warmup)
        cost = (self.clock() - t0) / n
        self._substep_cost = cost if self._substep_cost is None else 0.7 * self._substep_cost + 0.3 * cost

    def _substeps(self, n, h):
        """n substeps of length h, with CCD per substep or per span (ccd_stride, ccd_max_span)."""
        sec = self._section
        if self.debug is not None:
            self.debug.begin_step(self)
        stride = n if EnvConfig.ccd_stride is None else max(1, min(int(EnvConfig.ccd_stride), n))
        if EnvConfig.ccd_max_span is not None:  # no CCD span longer than ccd_max_span seconds (see EnvConfig)
            stride = max(1, min(stride, int(EnvConfig.ccd_max_span / h + 1e-9)))
        if stride > 1:
            for k in range(0, n, stride):
                if k and self.period is not None:  # keep wrapping between spans (looping.span_wraps)
                    self.last_wraps += looping.span_wraps(self)
                self._ccd_span(h, min(stride, n - k))
            return
        for k in range(n):
            if k and self.period is not None:
                self.last_wraps += looping.span_wraps(self)
            if self.bounce_bodies:
                self._bounce(h)
            if self.backend is None or not self.backend.pbd_substep(h):  # False: the backend can't run this scene
                self._pbd_substep(h)
            self.V[:] = (self.P - self.PREV) / h
            self._limit_speed()  # before CCD, which then checks the (shorter) clamped paths
            with sec("ccd"):
                self._resolve_ccd(h)
            self.P[:] = self.PREV + self.V * h
            self.V *= max(0.0, 1.0 - EnvConfig.damping * h)

    def _ccd_span(self, h, m):
        """m substeps of length h without CCD, then one CCD pass over the span's straight-line motion (Bridson et al.
        2002, sec. 5: any internal dynamics propose the end positions, CCD then edits the average velocity).

        The span starts intersection-free and CCD makes its straight path from there collision-free, the same
        argument as per-substep CCD with h = m h. Contacts in the substeps take their sides from the span start and
        share one swept-box candidate list (_span_candidates). Points CCD leaves alone keep their last substep's
        velocity; points it edited take the corrected average velocity."""
        sec = self._section
        T = m * h
        X0 = self.P.copy()
        self._span_start, self._span_contacts, self._span_reach = X0, None, None
        damp = max(0.0, 1.0 - EnvConfig.damping * h)
        try:
            for k in range(m):
                self._span_left = T - k * h
                if self.bounce_bodies:
                    self._bounce(h)
                if self.backend is None or not self.backend.pbd_substep(h):  # False: the backend can't run this scene
                    self._pbd_substep(h)
                self.V[:] = (self.P - self.PREV) / h
                self._limit_speed()
                self.P[:] = self.PREV + self.V * h
                self.V *= damp
        finally:
            self._span_start, self._span_contacts, self._span_reach = None, None, None
        v_last = self.V.copy()
        self.PREV[:] = X0
        self.V[:] = v_avg = (self.P - X0) / T
        with sec("ccd"):
            self._resolve_ccd(T)
        changed = (self.V != v_avg).any(axis=0)
        self.P[:] = X0 + self.V * T
        self.V[:] = np.where(changed, self.V * damp ** m, v_last)

    def _limit_speed(self):
        """Terminal velocity: scale down any point velocity faster than the limit (direction kept). It runs on V
        before CCD, so it only shortens the straight-line paths CCD then makes collision-free; it can't cause a
        crossing. Shared by both backends (the GPU backend only replaces the PBD part of a substep)."""
        vmax = EnvConfig.terminal_velocity  # per world: World.env["terminal_velocity"]
        if vmax is None:
            return
        V = self.V
        speed2 = V[0] * V[0] + V[1] * V[1]
        fast = speed2 > vmax * vmax
        if fast.any():
            V[:, fast] *= vmax / np.sqrt(speed2[fast])

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
        if _jit():
            solver_jit.bounce(self, h, r)
            return
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
        self._h = h  # the XPBD springs' substep length, and a fresh set of multipliers for this substep
        self.spring_lambda.fill(0.0)
        with sec("predict+grab"):
            if self.batched_predict:  # Body.predict for every body at once
                self.V += EnvConfig.gravity[:, None] * h * (self.W > 0)
                self.PREV[:] = self.P
                self.P += self.V * h
            else:
                for b in self.bodies:
                    b.predict(h)
            self._apply_grab(h)
        with sec("contacts"):
            if self._span_start is None:
                candidates = self._candidates()
                reach = None  # (candidates within reach, positions when they were picked), see _solve_contacts
            else:
                candidates, reach = self._span_candidates()
        for _ in range(EnvConfig.iterations):
            with sec("constraints"):
                self._solve_body_constraints()
                self._solve_ropes()
            with sec("contacts"):
                reach = self._solve_contacts(candidates, reach)
        if self._span_start is not None:
            self._span_reach = reach

    def draw(self, surface, filled=False, zoom=1.0):
        """Bodies in the order they were added (later ones on top). filled: solid polygons instead of points + springs.
        zoom maps world units to surface pixels. A body sticking out past a periodic edge also draws its copy one
        period over (looping.image_copies), right after itself."""
        copies = defaultdict(list)
        for k, s in looping.image_copies(self):
            copies[k].append(s)
        for k, b in enumerate(self.bodies):
            for s in [None] + copies.get(k, []):
                if s is not None:  # a shallow copy with moved points draws the image (the body's arrays untouched)
                    b = copy.copy(self.bodies[k])
                    b.points = self.bodies[k].points + s[:, None]
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
        if self.grab_set is not None:  # an area grab: each held point to its target
            idx, off, _ = self.grab_set
            for a, b in zip(self.P[:, idx].T, (self.grab_target[:, None] + off).T):
                pygame.draw.line(surface, (255, 255, 255), a * zoom, b * zoom, 1)

    def draw_grid(self, surface, line_col=(52, 58, 76), edge_cell_col=(37, 42, 55), point_cell_col=(45, 51, 68), zoom=1.0):
        """Debug view of the broad phase grid: cell (i, j) covers [i, i + 1) * cell_size, same as the solver's floor().

        Cells an edge is registered in (its AABB padded like _candidates does) get edge_cell_col,
        cells holding a point get point_cell_col (drawn over), then the grid lines on top.
        """
        cell = self.env.get("cell_size", EnvConfig.cell_size)  # this world's own grid size, if it has one
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
        if pt_owner.size == pt_lo.shape[1] or i.size == 0:
            # one cell per point: already in (point, edge) order, and a pair can't repeat. No pairs at all: nothing to
            # dedupe (the mask below would be one element longer than the empty array it indexes: IndexError)
            return i, e
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

    # ---------------- Periodic axes: pairs through images (looping.py; only while World.period is set)
    def _image_topology(self):
        """(point mask, edge ids) that take part in image pairs: everything but the world box walls (World.walls).
        Those of a walled axis run along the periodic one and past its seams, so their direct contacts already are
        the image ones. Cached until the next repack."""
        t = self._image_topo
        if t is None or t[0] is not self.edges:
            wall = np.array([looping.is_world_wall(self, b) for b in self.bodies], dtype=bool)
            t = (self.edges, ~wall[self.point_body], ~wall[self.edge_body], None, None)
            self._image_topo = t
        return t[1], t[2]

    def _image_shifts(self):
        """looping.image_shifts(self.period) as one (k, 2) array, cached per period (the kernels' form)."""
        self._image_topology()  # (rebuilt on a repack)
        t = self._image_topo
        if t[3] is None or not np.array_equal(t[3], self.period):
            t = self._image_topo = t[:3] + (self.period.copy(),
                                             np.array(looping.image_shifts(self.period), dtype=float).reshape(-1, 2))
        return t[4]

    def _image_pairs(self, plo, phi, elo, ehi):
        """(point, edge, offset (2, m)) pairs that meet only through an image: the point's float box plo..phi moved
        by an image offset (looping.image_shifts) shares a grid cell with the edge's box elo..ehi (the same boxes
        the direct broad phase floors, pads included). Only points whose moved box reaches the edges' overall extent
        are tried, so this costs O(points near a seam). Same-body pairs are left out (a body meets its own image
        only if it is bigger than the period). solver_jit.image_pairs gives the same pairs per offset (the merge then
        orders them the same), so the two CPU paths stay identical."""
        empty = np.empty(0, dtype=np.int64)
        point_ok, edge_ok = self._image_topology()
        if _jit():
            return solver_jit.image_pairs(self, plo, phi, elo, ehi, point_ok, edge_ok, self._image_shifts(),
                                          EnvConfig.cell_size)
        out_i, out_e, out_o = [], [], []
        if edge_ok.any():
            cell = EnvConfig.cell_size
            ext_lo = np.min(elo, axis=1, where=edge_ok[None], initial=np.inf)[:, None]
            ext_hi = np.max(ehi, axis=1, where=edge_ok[None], initial=-np.inf)[:, None]
            for s in looping.image_shifts(self.period):
                # points whose moved box reaches the edges' extent: those near the seam opposite the offset
                idx = np.flatnonzero(point_ok & (plo[0] + s[0] <= ext_hi[0]) & (phi[0] + s[0] >= ext_lo[0])
                                     & (plo[1] + s[1] <= ext_hi[1]) & (phi[1] + s[1] >= ext_lo[1]))
                if idx.size == 0:
                    continue
                lo, hi = plo[:, idx] + s[:, None], phi[:, idx] + s[:, None]
                u_lo, u_hi = lo.min(axis=1), hi.max(axis=1)
                eids = np.flatnonzero(edge_ok & (elo[0] <= u_hi[0]) & (ehi[0] >= u_lo[0]) & (elo[1] <= u_hi[1])
                                      & (ehi[1] >= u_lo[1]))
                if eids.size == 0:
                    continue
                i, e = self._cell_pairs(np.floor(lo / cell).astype(np.int64), np.floor(hi / cell).astype(np.int64),
                                        np.floor(elo[:, eids] / cell).astype(np.int64),
                                        np.floor(ehi[:, eids] / cell).astype(np.int64))
                i, e = idx[i], eids[e]
                pb, eb = self.point_body[i], self.edge_body[e]
                keep = (pb != eb) & ~(self.body_fixed[pb] & self.body_fixed[eb])
                out_i.append(i[keep])
                out_e.append(e[keep])
                out_o.append(np.repeat(s[:, None], np.count_nonzero(keep), axis=1))
        if not out_i:
            return empty, empty, np.empty((2, 0))
        return np.concatenate(out_i), np.concatenate(out_e), np.concatenate(out_o, axis=1)

    def _image_extras(self, plo, phi, elo, ehi):
        """The pairs _merge_images would add to the direct broad phase of the same float boxes, without that direct
        list: image pairs (_image_pairs) whose direct cell ranges don't overlap (a direct pair needs just that, since
        image pairs already join two bodies that aren't both fixed), the first image of each (point, edge) only.
        (pts, eids, off (2, m)), in no particular order: for the GPU backend, which runs them as a second contact
        list next to its own direct pairs (its atomics don't keep an order anyway)."""
        if _jit():
            return solver_jit.image_extras(self, plo, phi, elo, ehi, *self._image_topology(), self._image_shifts(),
                                           EnvConfig.cell_size)
        i, e, off = self._image_pairs(plo, phi, elo, ehi)
        if i.size == 0:
            return i, e, off
        cell = EnvConfig.cell_size
        a, b = (np.floor(box / cell).astype(np.int64) for box in (plo[:, i], phi[:, i]))
        c, d = (np.floor(box / cell).astype(np.int64) for box in (elo[:, e], ehi[:, e]))
        fresh = np.flatnonzero(~np.all((a <= d) & (b >= c), axis=0))  # not met directly
        _, first = np.unique(i[fresh] * max(self.edges.shape[1], 1) + e[fresh], return_index=True)
        keep = fresh[first]
        return i[keep], e[keep], off[:, keep]

    def _with_images(self, direct, boxes):
        """_merge_images(direct, _image_pairs(*boxes)): with the kernels in one call (the same result)."""
        if _jit():
            return solver_jit.merge_with_images(self, direct, *boxes, *self._image_topology(), self._image_shifts(),
                                                EnvConfig.cell_size)
        return self._merge_images(direct, self._image_pairs(*boxes))

    def _merge_images(self, direct, image):
        """Direct pairs (sorted by (point, edge), unique) plus image pairs -> (pts, eids, off), sorted by (point,
        edge) with off = 0 for the direct ones. Minimum image: a (point, edge) pair met directly keeps only that, and
        of several images of one pair the first (image_shifts order) is kept, so (point, edge) stays unique."""
        i_d, e_d = direct
        i_d, e_d = i_d.astype(np.int64, copy=False), e_d.astype(np.int64, copy=False)
        i_m, e_m, o_m = image
        if i_m.size == 0:
            return i_d, e_d, np.zeros((2, i_d.size))
        n_edges = max(self.edges.shape[1], 1)
        keys_d = i_d * n_edges + e_d
        keys_m = i_m.astype(np.int64) * n_edges + e_m
        if keys_d.size:  # drop the pairs met directly
            at = np.searchsorted(keys_d, keys_m)
            fresh = (at >= keys_d.size) | (keys_d[np.minimum(at, keys_d.size - 1)] != keys_m)
        else:
            fresh = np.ones(keys_m.size, dtype=bool)
        keys_m, first = np.unique(keys_m[fresh], return_index=True)  # sorted; the first image of a pair is kept
        o_m = o_m[:, fresh][:, first]
        at = np.searchsorted(keys_d, keys_m)  # insertion points: no sort of the (long) direct list
        i_m, e_m = np.divmod(keys_m, n_edges)
        pts, eids = np.insert(i_d, at, i_m), np.insert(e_d, at, e_m)
        off = np.zeros((2, pts.size))
        off[:, at + np.arange(at.size)] = o_m
        return pts, eids, off

    def _contact_boxes(self):
        """Float boxes behind _candidates' cells: points, and edges grown by 2 contact_radius."""
        pad = 2 * EnvConfig.contact_radius
        a, b = self.P[:, self.edges[0]], self.P[:, self.edges[1]]
        return self.P, self.P, np.minimum(a, b) - pad, np.maximum(a, b) + pad

    def _candidates(self):
        """(point, edge) pairs that share a grid cell. Edges are registered in every cell their padded AABB touches.
        With a periodic axis: (point, edge, offset) including the pairs met through images (_merge_images)."""
        direct = self._direct_candidates()
        if self.period is None:
            return direct
        return self._with_images(direct, self._contact_boxes())

    def _direct_candidates(self):
        if _jit():
            return solver_jit.candidates(self, EnvConfig.cell_size, EnvConfig.contact_radius)
        cell, pad = EnvConfig.cell_size, 2 * EnvConfig.contact_radius
        a, b = self.P[:, self.edges[0]], self.P[:, self.edges[1]]
        lo = np.floor((np.minimum(a, b) - pad) / cell).astype(np.int64)
        hi = np.floor((np.maximum(a, b) + pad) / cell).astype(np.int64)
        point_cells = np.floor(self.P / cell).astype(np.int64)
        self.contact_builds += 1
        return self._filter_pairs(*self._cell_pairs(point_cells, point_cells, lo, hi), contact=True)

    def _span_boxes(self):
        """Per-point boxes over the predicted motion to the end of the CCD span, x .. x + v t + g t^2 / 2 (t =
        _span_left), grown by contact_radius, plus the edge boxes around them grown by 2 contact_radius more:
        (point lo, point hi, edge lo, edge hi). While every point stays in its box, any point within 2
        contact_radius of an edge has overlapping boxes with it, so the pairs are a superset of _candidates'."""
        r = EnvConfig.contact_radius
        plo, phi = self._span_point_boxes()
        ea, eb = self.edges
        return (plo, phi, np.minimum(plo[:, ea], plo[:, eb]) - 2 * r, np.maximum(phi[:, ea], phi[:, eb]) + 2 * r)

    def _span_point_boxes(self):
        """The point boxes of _span_boxes: (point lo, point hi)."""
        r, t = EnvConfig.contact_radius, self._span_left
        end = self.P + self.V * t + 0.5 * EnvConfig.gravity[:, None] * t * t * (self.W > 0)
        if self.grabbed is not None:  # the hand pulls toward the cursor at up to grab_max_speed
            g = self.grabbed
            pull = self.grab_target - self.P[:, g]
            reach = EnvConfig.grab_max_speed * t
            dist = float(np.hypot(*pull))
            end[:, g] = self.P[:, g] + self.V[:, g] * t + (pull if dist <= reach else pull * (reach / dist))
        if self.grab_set is not None:  # an area grab: the same reach for each held point
            idx, off, _ = self.grab_set
            pull = self.grab_target[:, None] + off - self.P[:, idx]
            reach = EnvConfig.grab_max_speed * t
            dist = np.hypot(*pull)
            end[:, idx] = self.P[:, idx] + self.V[:, idx] * t + pull * np.minimum(1.0, reach / np.maximum(dist, 1e-12))
        return np.minimum(self.P, end) - r, np.maximum(self.P, end) + r

    def _span_candidates(self):
        """Contact candidates and reach for a substep inside a CCD span: the span's swept-box candidate list
        (_span_boxes), rebuilt only when a point has left its box, and the reach carried over from the last
        substep (it is checked against how far points moved, so it holds across substeps)."""
        sc = self._span_contacts
        if sc is not None and not ((self.P < sc[1]).any() or (self.P > sc[2]).any()):
            return sc[0], self._span_reach
        if _jit():  # same pairs in the same order, the edge boxes never built as floats
            boxes = self._span_point_boxes()
            candidates = solver_jit.span_candidates(self, *boxes, EnvConfig.cell_size, EnvConfig.contact_radius)
        else:
            boxes = self._span_boxes()
            cells = [np.floor(box / EnvConfig.cell_size).astype(np.int64) for box in boxes]
            candidates = self._filter_pairs(*self._cell_pairs(*cells), contact=True)
        if self.period is not None:  # plus the pairs through images, from the same float boxes (_span_boxes)
            plo, phi = boxes[0], boxes[1]
            r, (ea, eb) = EnvConfig.contact_radius, self.edges
            elo, ehi = np.minimum(plo[:, ea], plo[:, eb]) - 2 * r, np.maximum(phi[:, ea], phi[:, eb]) + 2 * r
            candidates = self._with_images(candidates, (plo, phi, elo, ehi))
        self.contact_builds += 1
        self._span_contacts = (candidates, boxes[0], boxes[1])
        return candidates, None

    # ---------------- Narrow phase + resolution: point vs edge capsule
    def _solve_contacts(self, candidates, reach=None):
        """One Jacobi pass over the (pts, edge_ids) candidates. Returns a `reach` to pass to the next pass of the
        same substep: the candidates within contact_radius + slack, and P when they were picked. A point's distance
        to a segment changes by at most its own motion plus the farther endpoint's, so while no point has moved
        slack / 2 since then, the other candidates are out of range and are skipped (same result, less work)."""
        r, mu = EnvConfig.contact_radius, EnvConfig.friction
        if _jit():
            return solver_jit.solve_contacts(self, candidates, reach, r, mu)
        P, PREV, W = self.P, self.PREV, self.W
        slack = r
        # reach: (pts, edge ids, P snapshot[, offsets]); candidates: (pts, edge ids[, offsets]). Offsets (2, m) come
        # with a periodic axis (_merge_images): the image offset added to each pair's point
        if reach is not None and np.hypot(*(P - reach[2])).max() < 0.5 * slack - 1e-6:
            pts, edge_ids, off = reach[0], reach[1], (reach[3] if len(reach) > 3 else None)
        else:
            pts, edge_ids, off = candidates[0], candidates[1], (candidates[2] if len(candidates) > 2 else None)
            reach = None
        if pts.size == 0:
            return reach
        ia, ib = self.edges[0, edge_ids], self.edges[1, edge_ids]

        p, a, b = P[:, pts], P[:, ia], P[:, ib]
        if off is not None:
            p = p + off
        e = b - a
        len2 = np.maximum(np.sum(e * e, axis=0), 1e-12)
        t = np.clip(np.sum((p - a) * e, axis=0) / len2, 0.0, 1.0)
        c = a + t * e
        d = p - c
        dist = np.hypot(d[0], d[1])
        if reach is None:
            keep = dist < r + slack
            reach = (pts[keep], edge_ids[keep], P.copy()) + (() if off is None else (off[:, keep],))

        # most candidates are out of range: keep only the touching ones (everything below is per pair)
        near = np.flatnonzero(dist < r)
        if near.size == 0:
            return reach
        pts, edge_ids, ia, ib, t, len2, dist = (x[near] for x in (pts, edge_ids, ia, ib, t, len2, dist))
        p, c, d, e = (x[:, near] for x in (p, c, d, e))
        if off is not None:
            off = off[:, near]

        # Edge normal facing the side the point started this substep on. CCD guarantees no point ever crosses an
        # edge, so that's always the free side: outside for neighboring bodies, inside for a body's contents.
        # (Only a point exactly on the edge's line at the start falls back to the body's outward side.) Inside a
        # multi-substep CCD span only the span start is CCD-checked, so the side comes from there.
        X_side = PREV if self._span_start is None else self._span_start
        pa0 = X_side[:, ia]
        e0 = X_side[:, ib] - pa0
        q0 = (X_side[:, pts] if off is None else X_side[:, pts] + off) - pa0
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
        rel = (p - (PREV[:, pts] if off is None else PREV[:, pts] + off)) - (c - c_prev)
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
    def _ccd_float_boxes(self, D):
        """_ccd_boxes, from the compiled kernel when it runs (the same values). With a periodic axis the boxes of
        the latest D are kept: _ccd_cells, _ccd_pairs and _ccd_new_pairs all need them for the same D (PREV doesn't
        change while a D is in use, and every D is a fresh array)."""
        c = getattr(self, "_ccd_box_cache", None)
        if c is not None and c[0] is D:
            return c[1]
        if _jit():
            boxes = solver_jit.ccd_boxes(self.PREV, D, *solver_jit._topology(self)[1:3])
        else:
            boxes = self._ccd_boxes(D)
        if self.period is not None:
            self._ccd_box_cache = (D, boxes)
        return boxes

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
        if self.period is not None:  # (the float boxes are kept for the image pairs)
            boxes = self._ccd_float_boxes(D)
            if _jit():
                return [solver_jit.floor_cells(box, EnvConfig.cell_size) for box in boxes]
            return [np.floor(box / EnvConfig.cell_size).astype(int) for box in boxes]
        if _jit():
            return solver_jit.ccd_cells(self, D, EnvConfig.cell_size)
        return [np.floor(box / EnvConfig.cell_size).astype(int) for box in self._ccd_boxes(D)]

    def _ccd_pairs(self, D, cells=None):
        """(point, edge) pairs whose swept boxes (start -> start + D) share a grid cell. `cells`: _ccd_cells(D)."""
        cells = self._ccd_cells(D) if cells is None else cells
        if self.period is not None:  # direct pairs on the CPU, plus the pairs through images (_merge_images)
            direct = (solver_jit.ccd_pairs(self, cells) if _jit()
                      else self._filter_pairs(*self._cell_pairs(*cells), contact=False))
            return self._with_images(direct, self._ccd_float_boxes(D))
        pairs = None if self.backend is None else self.backend.ccd_pairs(cells)  # None: use the CPU code below
        if pairs is not None:
            return pairs
        if _jit():
            return solver_jit.ccd_pairs(self, cells)
        return self._filter_pairs(*self._cell_pairs(*cells), contact=False)

    def _ccd_new_pairs(self, old_cells, cells, D=None):
        """The pairs among _ccd_pairs for `cells` that may be missing from the pairs for `old_cells`: a pair can
        only be new if its point's or its edge's cell range grew, so pair just those against everything. D (the
        motion `cells` came from) is needed only with a periodic axis."""
        p_lo, p_hi, e_lo, e_hi = cells
        grew_p = np.flatnonzero(((p_lo < old_cells[0]) | (p_hi > old_cells[1])).any(axis=0))
        grew_e = np.flatnonzero(((e_lo < old_cells[2]) | (e_hi > old_cells[3])).any(axis=0))
        i1, e1 = self._cell_pairs(p_lo[:, grew_p], p_hi[:, grew_p], e_lo, e_hi)
        i2, e2 = self._cell_pairs(p_lo, p_hi, e_lo[:, grew_e], e_hi[:, grew_e])
        i, e = self._filter_pairs(np.concatenate([grew_p[i1], i2]), np.concatenate([e1, grew_e[e2]]), contact=False)
        if self.period is None:
            return i, e
        # with a periodic axis: plus every pair through an image for D (few; this only needs to be a superset of the
        # new pairs, see _resolve_ccd)
        i_m, e_m, o_m = self._image_pairs(*self._ccd_float_boxes(D))
        return (np.concatenate([i, i_m]), np.concatenate([e, e_m]),
                np.concatenate([np.zeros((2, i.size)), o_m], axis=1))

    def _ccd_detect(self, D, pairs=None):
        """Points that touch an edge at some time in the substep, moving in straight lines from PREV by D.

        A point p(t) is on the line through a(t), b(t) when cross(b - a, p - a) = 0, a quadratic in t in [0, 1];
        it's a hit if the point is also on the segment then. Ending closer than ccd_gap also counts, so states
        never get so close that rounding decides which side a point is on. Returns (pts, edge_ids, t, s, side).
        `pairs` reuses an earlier broad phase; only a fresh one (the default) can prove there are no collisions.

        Every root is verified geometrically: a hit needs the point within ccd_tau of the segment at time t, which
        rejects the spurious roots of an edge that collapses to ~zero length (the cross product then vanishes for
        far points too). The textbook roots with an exact 0 <= s <= 1 test are tried first; where they find no
        verified touch, a float64-robust solve runs: cancellation-free roots q / c2 and c0 / q, and s clamped to
        the segment with the ccd_tau distance test, so a path through (or within rounding of) an endpoint is not
        missed when s rounds to just outside [0, 1] on both edges meeting there (research/02 F3). An exactly
        collinear point and edge (c2 = c1 = 0) get the times the point passes either end as candidates. Hits the
        textbook test does verify keep its values bit for bit, so results only change where it was wrong.
        """
        pairs = self._ccd_pairs(D) if pairs is None else pairs
        # with a periodic axis pairs carry (2, m) image offsets added to their points, and so do the hits returned
        pts, eids, img = pairs[0], pairs[1], (pairs[2] if len(pairs) > 2 else None)
        if _jit():
            return solver_jit.ccd_detect(self, D, pts, eids, EnvConfig.ccd_gap, EnvConfig.ccd_tau, img)
        empty = np.empty(0)
        X0 = self.PREV
        if pts.size:
            # Cheap pre-reject: a hit puts the point within ccd_tau of the edge at some time t, so the point's swept
            # box and the edge's swept box (grown by 1.0 px, more than ccd_gap and ccd_tau) overlap. Grid cells are
            # much coarser than that, so this drops most broad phase pairs before the quadratic.
            p_lo, p_hi, e_lo, e_hi = self._ccd_boxes(D)
            if img is None:
                keep = np.all((p_lo[:, pts] <= e_hi[:, eids]) & (p_hi[:, pts] >= e_lo[:, eids]), axis=0)
            else:  # the point's image path p0 -> p0 + D (as solver_jit.ccd_narrow forms it)
                p0 = X0[:, pts] + img
                p1 = p0 + D[:, pts]
                keep = np.all((np.minimum(p0, p1) <= e_hi[:, eids]) & (np.maximum(p0, p1) >= e_lo[:, eids]), axis=0)
                img = img[:, keep]
            pts, eids = pts[keep], eids[keep]
        if pts.size == 0:
            return (pts, eids, empty, empty, empty) + (() if img is None else (np.empty((2, 0)),))
        ia, ib = self.edges[0, eids], self.edges[1, eids]
        e0, de = X0[:, ib] - X0[:, ia], D[:, ib] - D[:, ia]
        q0, dq = (X0[:, pts] if img is None else X0[:, pts] + img) - X0[:, ia], D[:, pts] - D[:, ia]

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
        tau2 = EnvConfig.ccd_tau * EnvConfig.ccd_tau
        found = np.flatnonzero(np.isfinite(t_hit))
        if found.size:  # verify: the point must be on the segment (within ccd_tau) at the root
            r = t_hit[found]
            e, q = e0[:, found] + r * de[:, found], q0[:, found] + r * dq[:, found]
            off = q - s_hit[found] * e
            far = found[off[0] * off[0] + off[1] * off[1] > tau2]
            t_hit[far], s_hit[far] = np.inf, 0.0

        # float64-robust solve where the textbook one found nothing
        k = np.flatnonzero(np.isinf(t_hit))
        if k.size:
            e0k, dek, q0k, dqk = e0[:, k], de[:, k], q0[:, k], dq[:, k]
            c0k, c1k, c2k, sqk = c0[k], c1[k], c2[k], sq[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                qq = -0.5 * (c1k + np.where(c1k >= 0, sqk, -sqk))  # no cancellation
                quad, lin = c2k != 0, (c2k == 0) & (c1k != 0)
                # collinear all substep (c2 = c1 = 0): the times the point passes either end of the segment
                w = np.sum(dqk * e0k, axis=0)
                ta = -np.sum(q0k * e0k, axis=0) / w
                tb = -np.sum((q0k - e0k) * e0k, axis=0) / (w - np.sum(dek * e0k, axis=0))
                ra = np.where(quad, qq / c2k, np.where(lin, -c0k / c1k, ta))
                rb = np.where(quad, c0k / qq, np.where(lin, np.nan, tb))
            t_rob = np.full(k.size, np.inf)
            s_rob = np.zeros(k.size)
            for r in (ra, rb):
                with np.errstate(invalid="ignore", over="ignore"):
                    e, q = e0k + r * dek, q0k + r * dqk
                    s = np.clip(np.sum(q * e, axis=0) / np.maximum(np.sum(e * e, axis=0), 1e-300), 0.0, 1.0)
                    off = q - s * e
                    ok = (r >= 0) & (r <= 1) & (off[0] * off[0] + off[1] * off[1] <= tau2) & (r < t_rob)
                t_rob = np.where(ok, r, t_rob)
                s_rob = np.where(ok, s, s_rob)
            t_hit[k], s_hit[k] = t_rob, s_rob

        # ending within ccd_gap of the edge counts as a hit at t = 1
        e1, q1 = e0 + de, q0 + dq
        len2 = np.maximum(np.sum(e1 * e1, axis=0), 1e-12)
        s1 = np.clip(np.sum(q1 * e1, axis=0) / len2, 0, 1)
        end_dist = np.hypot(*(q1 - s1 * e1))
        near = np.isinf(t_hit) & (end_dist < EnvConfig.ccd_gap)
        t_hit = np.where(near, 1.0, t_hit)
        s_hit = np.where(near, s1, s_hit)

        hit = np.isfinite(t_hit)
        return (pts[hit], eids[hit], t_hit[hit], s_hit[hit], side[hit]) + (() if img is None else (img[:, hit],))

    def _resolve_ccd(self, h):
        """Make the straight-line motion PREV -> PREV + V h collision-free by editing V."""
        V, sec = self.V, self._section
        with sec("broad"):
            D = V * h
            cells = self._ccd_cells(D)
            pairs = self._ccd_pairs(D, cells)
        with sec("narrow+impulses"):
            hits = self._ccd_detect(D, pairs)
        if self.debug is not None:
            self.debug.ccd_hits(self, hits, D)
        fresh = True  # V hasn't changed since the broad phase
        V_pre = None
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
                    extra = self._ccd_new_pairs(cells, new_cells, D)
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
                if self.debug is not None:
                    self.debug.ccd_hits(self, hits, D)
            with sec("narrow+impulses"):
                V_before = V.copy()
                if fresh:
                    V_pre = V_before  # before any CCD response: the inelastic projection's reference
                self._ccd_impulses(*hits[:5], h, *hits[5:])
                hits = self._ccd_redetect(V * h, pairs, hits, (V != V_before).any(axis=0))
            fresh = False

        with sec("impact zones"):
            if self._inelastic_zones(h, V_pre):
                return
            self.zone_count += 1
            self._impact_zones(h)

    def _ccd_normal(self, i, ia, ib, t, side, h, off=None):
        """_ccd_impulses' push direction for point i against edge (ia, ib): from the closest point on the edge to the
        point at the start of the substep, or the edge normal at the hit time t (towards `side`) if they touch.
        off: the point's image offset (periodic axis), or None."""
        V, X0 = self.V, self.PREV
        xi = X0[:, i] if off is None else X0[:, i] + off
        e0 = X0[:, ib] - X0[:, ia]
        u = min(max((xi - X0[:, ia]) @ e0 / max(e0 @ e0, 1e-12), 0.0), 1.0)
        sep = xi - (X0[:, ia] + u * e0)
        if np.hypot(*sep) > 1e-9:
            return sep / np.hypot(*sep)
        e = (X0[:, ib] + t * h * V[:, ib]) - (X0[:, ia] + t * h * V[:, ia])
        return side * np.array([-e[1], e[0]]) / max(np.hypot(*e), 1e-12)

    def _inelastic_zones(self, h, V_pre, pgs_iters=300):
        """Fallback when the impulse passes leave hits: inelastic projection (Harmon et al. 2008, "Robust treatment
        of simultaneous collisions", Algorithm 1), tried before the rigid impact zones. True if the motion ends up
        collision-free (checked with a fresh broad phase); False leaves V as it found it.

        Each hit (point i, edge (a, b), s, normal n) becomes a constraint G V >= b with G = +n on i, -(1 - s) n on
        a and -s n on b, and b = (ccd_push - start gap along n) / h: the same end gap _ccd_impulses aims for.
        Constraints accumulate over rounds; each connected group of them is reset to its velocities from before
        any CCD response and projected onto them in the mass metric, min |V' - V_pre|_M subject to G V' >= b, by
        projected Gauss-Seidel on the small dual (Delassus) system. Only velocities along the constraint normals
        change, so sliding survives, where the impact zones stop all relative motion. Sequential impulses converge
        only geometrically for a point wedged between two edges (Bridson et al. 2002 name the case); solving the
        group's constraints together fixes that. Every round is re-verified by _ccd_detect, so the guarantee still
        rests on the impact zones as the last resort (research/02 F4)."""
        rounds = EnvConfig.ccd_projection_rounds
        if not rounds:
            return False
        V, W, X0, E = self.V, self.W, self.PREV, self.edges
        push = EnvConfig.ccd_push
        V_entry = V.copy()
        V_pre = V_entry if V_pre is None else V_pre
        cons = {}  # (point, edge) -> (i, a, b, s, n, rhs), in first-seen order
        projected = False
        for _ in range(rounds):
            hits = self._ccd_detect(V * h)
            pts, eids, ts, ss, sides = hits[:5]
            if pts.size == 0:
                self.projection_count += projected
                return True
            offs = hits[5].T if len(hits) > 5 else [None] * pts.size  # image offsets (periodic axis)
            for i, e, t, s, side, o in zip(pts.tolist(), eids.tolist(), ts.tolist(), ss.tolist(), sides.tolist(), offs):
                ia, ib = int(E[0, e]), int(E[1, e])
                n = self._ccd_normal(i, ia, ib, t, side, h, o)
                xi = X0[:, i] if o is None else X0[:, i] + o
                gap0 = (xi - ((1 - s) * X0[:, ia] + s * X0[:, ib])) @ n
                cons[(i, e)] = (i, ia, ib, s, n, (push - gap0) / h)
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i, ia, ib, *_ in cons.values():
                for j in (ia, ib):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri
            groups = {}
            for c in cons.values():
                groups.setdefault(find(c[0]), []).append(c)
            for cs in groups.values():
                members = sorted({p for c in cs for p in c[:3]})
                loc = {p: k for k, p in enumerate(members)}
                m, k = len(members), len(cs)
                G = np.zeros((k, 2 * m))
                rhs = np.empty(k)
                for row, (i, ia, ib, s, n, b) in enumerate(cs):
                    for p, wgt in ((i, 1.0), (ia, -(1 - s)), (ib, -s)):
                        G[row, 2 * loc[p]:2 * loc[p] + 2] += wgt * n
                    rhs[row] = b
                Minv = np.repeat(W[members], 2)
                diag = np.einsum("ij,ij->i", G * Minv, G)
                lam = np.zeros(k)
                v = V_pre[:, members].T.reshape(-1).copy()
                for _ in range(pgs_iters):
                    worst = 0.0
                    for row in range(k):
                        if diag[row] < 1e-12:
                            continue
                        viol = rhs[row] - G[row] @ v
                        new = max(lam[row] + viol / diag[row], 0.0)
                        if new != lam[row]:
                            v += Minv * G[row] * (new - lam[row])
                            lam[row] = new
                        if lam[row] > 0 or viol > 0:
                            worst = max(worst, abs(viol))
                    if worst < 1e-9:
                        break
                V[:, members] = v.reshape(m, 2).T
            projected = True
        V[:] = V_entry
        return False

    def _ccd_redetect(self, D, pairs, hits, changed):
        """_ccd_detect(D, pairs) after impulses that changed only the `changed` points' velocities. A pair's result
        depends only on its three points, so pairs not touching a changed point keep their result from `hits`."""
        E = self.edges

        def touches(pts, eids):
            return changed[pts] | changed[E[0, eids]] | changed[E[1, eids]]

        pts, eids = pairs[0], pairs[1]
        moved = np.flatnonzero(touches(pts, eids))  # (indices: a take is much faster than a mask on (2, m) offsets)
        new = self._ccd_detect(D, tuple(a.take(moved, axis=-1) for a in pairs))  # (offsets (2, m) too, if periodic)
        kept = np.flatnonzero(~touches(hits[0], hits[1]))
        merged = [np.concatenate([old.take(kept, axis=-1), fresh], axis=-1) for old, fresh in zip(hits, new)]
        order = np.argsort(merged[0] * E.shape[1] + merged[1], kind="stable")  # back to (point, edge) order
        return tuple(a[..., order] for a in merged)

    def _ccd_impulses(self, pts, eids, ts, ss, sides, h, off=None):
        """One pass of sequential (Gauss-Seidel) impulses over detected collisions, earliest first, editing V.
        off: (2, m) image offsets of the hits' points (periodic axis), or None."""
        if _jit():
            solver_jit.ccd_impulses(self, pts, eids, ts, ss, sides, h, EnvConfig.ccd_push, off)
            return
        V, W, X0, E = self.V, self.W, self.PREV, self.edges
        push = EnvConfig.ccd_push
        for k in np.argsort(ts):
            i, ia, ib, t, s = pts[k], E[0, eids[k]], E[1, eids[k]], ts[k], ss[k]
            xi = X0[:, i] if off is None else X0[:, i] + off[:, k]  # the point (its image, through a seam)
            # push direction: from the closest point on the edge to the point, at the start of the substep.
            # Mid-edge that's the edge normal; near a corner it's corner -> point, which both edges meeting
            # there agree on (per-edge normals would ping-pong the point between them).
            e0 = X0[:, ib] - X0[:, ia]
            u = min(max((xi - X0[:, ia]) @ e0 / max(e0 @ e0, 1e-12), 0.0), 1.0)
            sep = xi - (X0[:, ia] + u * e0)
            if np.hypot(*sep) > 1e-9:
                n = sep / np.hypot(*sep)
            else:
                e = (X0[:, ib] + t * h * V[:, ib]) - (X0[:, ia] + t * h * V[:, ia])
                n = sides[k] * np.array([-e[1], e[0]]) / max(np.hypot(*e), 1e-12)  # points to the start side
            v_rel = V[:, i] - ((1 - s) * V[:, ia] + s * V[:, ib])
            # signed gap at the end of the substep, measured at the contact point
            end_p = xi + h * V[:, i]
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
        if self.period is not None:
            self._impact_zones_periodic(h)
            return
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
                if self.debug is not None:
                    self.debug.zone_points(self, members)
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
        if self.debug is not None:
            self.debug.zone_points(self, np.flatnonzero(W > 0))

    def _impact_zones_periodic(self, h):
        """_impact_zones with a periodic axis. A zone can reach across a seam, so its rigid motion is computed on
        unwrapped positions: the union-find also tracks each point's offset to its root's frame (a hit of point i
        through image offset o against an edge means i + o sits next to the edge's points). A zone whose offsets
        disagree (it closes a loop around the torus) has no common frame and holds still for the substep."""
        V, W, X0, E = self.V, self.W, self.PREV, self.edges
        n_pts = X0.shape[1]
        parent = np.arange(n_pts)
        shift = np.zeros((2, n_pts))  # offset from a point's frame to its parent's
        wrapped = set()  # roots of zones with inconsistent offsets

        def find(x):
            path = []
            while parent[x] != x:
                path.append(x)
                x = parent[x]
            for node in reversed(path):  # compress; the node next to the root first, so its parent's shift is final
                p = parent[node]
                if p != x:
                    shift[:, node] += shift[:, p]
                    parent[node] = x
            return x

        for _ in range(n_pts):
            pts, eids, _, _, _, offs = self._ccd_detect(V * h)
            if pts.size == 0:
                return
            merged = False
            for i, e, o in zip(pts, eids, offs.T):
                for j in (E[0, e], E[1, e]):
                    ri, rj = find(i), find(j)
                    if ri != rj:  # put j's zone in i's frame: S(i) - S'(j) = o
                        parent[rj] = ri
                        shift[:, rj] = shift[:, i] - o - shift[:, j]
                        if rj in wrapped:
                            wrapped.discard(rj)
                            wrapped.add(ri)
                        merged = True
                    elif np.abs(shift[:, i] - shift[:, j] - o).max() > 1.0:  # (sums of whole periods: exact)
                        wrapped.add(ri)
            roots = np.array([find(x) for x in range(n_pts)])
            for root in np.unique(roots[pts]):
                members = np.flatnonzero(roots == root)
                if self.debug is not None:
                    self.debug.zone_points(self, members)
                if (W[members] == 0).any() or root in wrapped:  # touching something fixed, or no common frame
                    V[:, members] = 0.0
                    continue
                m = 1.0 / W[members]
                x, v = X0[:, members] + shift[:, members], V[:, members]
                com, v_com = (x * m).sum(axis=1) / m.sum(), (v * m).sum(axis=1) / m.sum()
                r, u = x - com[:, None], v - v_com[:, None]
                inertia = (m * (r * r).sum(axis=0)).sum()
                omega = (m * (r[0] * u[1] - r[1] * u[0])).sum() / inertia if inertia > 1e-12 else 0.0
                V[:, members] = v_com[:, None] + omega * np.array([-r[1], r[0]])
            if not merged:
                break
        V[:, W > 0] = 0.0
        if self.debug is not None:
            self.debug.zone_points(self, np.flatnonzero(W > 0))
