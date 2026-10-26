"""Compiled (numba) versions of World's hot CPU paths (research/05-compiled-loops.md).

solver.World keeps its numpy code as the reference and calls into this module when EnvConfig.jit is on and numba can
be imported (AVAILABLE). Each kernel replicates the numpy expression order of solver.py, so the two paths give
bit-identical states (and the CCD makes exactly the same decisions):
- no fastmath: no FMA contraction or reassociation; error_model="numpy" so x / 0 gives inf / nan like numpy
- the same accumulation order as np.bincount (all point shares, then edge starts, then edge ends, in pair order)
- numpy's pairwise summation (_pw) wherever numpy reduces a contiguous run, a plain sum where it doesn't
- pairs in the same (point, edge) order as World._cell_pairs, impulses in np.argsort order
The kernels are cached on disk (__pycache__, keyed by source content and CPU); warmup() compiles or loads them all
once, so no first-use compile stalls a frame mid-simulation. No recursion in kernels: a self-recursive cached numba
function segfaulted when loaded from the cache (research 05, F3).
"""
import math

import numpy as np

try:
    from numba import njit
    AVAILABLE = True
except ImportError:  # no numba: World runs its numpy code
    AVAILABLE = False

    def njit(*args, **kwargs):
        return (lambda f: f) if not args or not callable(args[0]) else args[0]

NB = dict(cache=True, error_model="numpy", nogil=True)
_warm = False


def enabled():
    """True if the kernels can run; compiles / loads every kernel the first time (see warmup)."""
    if AVAILABLE and not _warm:
        warmup()
    return AVAILABLE


# ================================================================ glue: World state -> kernel arguments
def _topology(world):
    """Contiguous, fixed-dtype topology arrays for the kernels, rebuilt when World._rebuild repacks (new edges)."""
    t = getattr(world, "_jit_topology", None)
    if t is None or t[0] is not world.edges:
        bodies = world.bodies
        t = (world.edges, np.ascontiguousarray(world.edges[0]), np.ascontiguousarray(world.edges[1]),
             np.ascontiguousarray(world.edge_local[0]), np.ascontiguousarray(world.edge_local[1]),
             world.body_fixed.astype(np.bool_), world.body_self.astype(np.bool_),
             world.edge_sign.astype(np.float64),
             np.concatenate([[0], np.cumsum([b.edges.shape[1] for b in bodies])]).astype(np.int64),  # edge offsets
             {id(b): k for k, b in enumerate(bodies)})  # body -> index (rope ends)
        world._jit_topology = t
    return t


def _pairs(world, plo, phi, elo, ehi, contact):
    """World._filter_pairs(*World._cell_pairs(plo, phi, elo, ehi), contact) in one kernel."""
    _, ea, eb, el0, el1, fixed, self_coll, *_ = _topology(world)
    return cell_pairs(plo, phi, elo, ehi, ea, eb, world.point_body, world.edge_body, fixed, self_coll,
                      world.point_local, world.point_ring, el0, el1, contact, 4 * (plo.shape[1] + elo.shape[1]) + 65536)


def candidates(world, cell, radius):
    """World._candidates."""
    _, ea, eb, *_ = _topology(world)
    pc, lo, hi = contact_cells(world.P, ea, eb, cell, 2 * radius)
    return _pairs(world, pc, pc, lo, hi, True)


def span_candidates(world, plo, phi, cell, radius):
    """The candidate list of World._span_candidates from its point boxes (World._span_point_boxes)."""
    _, ea, eb, *_ = _topology(world)
    return _pairs(world, *span_cells(plo, phi, ea, eb, 2 * radius, cell), True)


def solve_contacts(world, candidates, reach, r, mu):
    """World._solve_contacts (same `reach` protocol)."""
    P = world.P
    slack = r
    if reach is not None and moved_max(P, reach[2]) < 0.5 * slack - 1e-6:
        pts, edge_ids, off = reach[0], reach[1], (reach[3] if len(reach) > 3 else None)
    else:
        pts, edge_ids, off = candidates[0], candidates[1], (candidates[2] if len(candidates) > 2 else None)
        reach = None
    if pts.size == 0:
        return reach
    _, ea, eb, _, _, _, _, esign, *_ = _topology(world)
    snap = P.copy() if reach is None else None  # P when the candidates were picked, before this pass moves it
    XS = world.PREV if world._span_start is None else world._span_start
    ox, oy = _offsets(off)
    keep = contact_pass(P, world.PREV, XS, world.W, ea, eb, esign, pts, edge_ids, r, mu, slack, reach is None, ox, oy)
    if reach is None:
        return (pts[keep], edge_ids[keep], snap) + (() if off is None else (off[:, keep],))
    return reach


_NO_OFFSETS = np.empty(0)


def _offsets(off):
    """Contiguous x / y rows of (2, m) image offsets (World._merge_images), or two empty arrays for none (the kernels
    then run their offset-free code)."""
    if off is None:
        return _NO_OFFSETS, _NO_OFFSETS
    return np.ascontiguousarray(off[0], dtype=np.float64), np.ascontiguousarray(off[1], dtype=np.float64)


def solve_body_constraints(world):
    """The batched springs and shape matching of World._solve_body_constraints (solo bodies stay with World)."""
    from entities import EnvConfig
    P = world.P
    if world.springs.shape[1] and EnvConfig.xpbd:
        xpbd_springs(P, world.PREV, world.W, world.springs[0], world.springs[1], world.spring_rest, world.spring_comp,
                     world.spring_ci, world.spring_cj, world.spring_lambda, world._h, float(EnvConfig.spring_damping))
    elif world.springs.shape[1]:
        springs(P, world.W, world.springs[0], world.springs[1], world.spring_rest, world.spring_stiffness,
                world.spring_count)
    for idx, rest, stiffness in world.rigid_groups:
        shape_match(P, idx, rest, stiffness)


def solve_ropes(world):
    """Every rope's Rope.solve_constraints in list order (Gauss-Seidel). False (nothing done) if a rope's class
    overrides solve_constraints. Ends, anchors, rest lengths and stiffness are read every call (the editor can move
    hooks or ropes between frames)."""
    from entities import Rope
    ropes = world.ropes
    if any(type(rope).solve_constraints is not Rope.solve_constraints for rope in ropes):
        return False
    offsets, index = world.offsets, _topology(world)[-1]
    n = len(ropes)
    ends = np.full((2, n), -1, np.int64)  # global point index per end, -1 for a fixed anchor
    anchors = np.zeros((n, 2, 2))
    for r, rope in enumerate(ropes):
        for k, (body, ref) in enumerate(rope.ends):
            if body is None:
                anchors[r, k] = ref
            else:
                ends[k, r] = offsets[index[id(body)]] + ref
    rest = np.array([rope.rest_length for rope in ropes], dtype=float)
    stiffness = np.array([rope.stiffness for rope in ropes], dtype=float)
    rope_pass(world.P, world.W, ends[0], ends[1], anchors, rest, stiffness)
    return True


def image_pairs(world, plo, phi, elo, ehi, point_ok, edge_ok, S, cell):
    """World._image_pairs: the same (point, edge) pairs for each image offset (S: (k, 2), World._image_shifts),
    grouped by offset in the same order (within an offset sorted by (point, edge), like World._cell_pairs)."""
    if S.shape[0] == 0:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty((2, 0))
    i, e, k = image_pair_kernel(*_image_args(world, plo, phi, elo, ehi, point_ok, edge_ok, S, cell))
    return i, e, np.ascontiguousarray(S[k].T)


def _image_args(world, plo, phi, elo, ehi, point_ok, edge_ok, S, cell):
    c = np.ascontiguousarray
    return (c(plo), c(phi), c(elo), c(ehi), point_ok.astype(np.bool_, copy=False),
            edge_ok.astype(np.bool_, copy=False), S, cell, world.point_body, world.edge_body, _topology(world)[5])


def merge_with_images(world, direct, plo, phi, elo, ehi, point_ok, edge_ok, S, cell):
    """World._merge_images(direct, World._image_pairs(plo, phi, elo, ehi)) in one kernel call: the same result."""
    i_d, e_d = direct
    if S.shape[0] == 0:
        return i_d, e_d, np.zeros((2, i_d.size))
    pts, eids, ox, oy = image_merge_kernel(np.ascontiguousarray(i_d, dtype=np.int64),
                                           np.ascontiguousarray(e_d, dtype=np.int64), max(world.edges.shape[1], 1),
                                           *_image_args(world, plo, phi, elo, ehi, point_ok, edge_ok, S, cell))
    return pts, eids, np.stack([ox, oy])


def image_extras(world, plo, phi, elo, ehi, point_ok, edge_ok, S, cell):
    """World._image_extras: the same pairs (in another order, which it leaves open)."""
    if S.shape[0] == 0:
        empty = np.empty(0, np.int64)
        return empty, empty, np.empty((2, 0))
    i, e, k = image_extras_kernel(max(world.edges.shape[1], 1),
                                  *_image_args(world, plo, phi, elo, ehi, point_ok, edge_ok, S, cell))
    return i, e, np.ascontiguousarray(S[k].T)


@njit(**NB)
def image_merge_kernel(i_d, e_d, n_edges, plo, phi, elo, ehi, point_ok, edge_ok, S, cell, point_body, edge_body,
                       body_fixed):
    """merge_with_images: image_pair_kernel, then merge_kernel."""
    i, e, k = image_pair_kernel(plo, phi, elo, ehi, point_ok, edge_ok, S, cell, point_body, edge_body, body_fixed)
    if i.size == 0:
        return i_d.copy(), e_d.copy(), np.zeros(i_d.size), np.zeros(i_d.size)
    ox, oy = np.empty(i.size), np.empty(i.size)
    for q in range(i.size):
        ox[q], oy[q] = S[k[q], 0], S[k[q], 1]
    return merge_kernel(i_d, e_d, i, e, ox, oy, n_edges)


@njit(**NB)
def image_extras_kernel(n_edges, plo, phi, elo, ehi, point_ok, edge_ok, S, cell, point_body, edge_body, body_fixed):
    """image_extras: the image pairs whose direct cell ranges don't overlap, the first image of each (point, edge)
    only. Returns (points, edges, offset index)."""
    i, e, k = image_pair_kernel(plo, phi, elo, ehi, point_ok, edge_ok, S, cell, point_body, edge_body, body_fixed)
    keep = np.empty(i.size, np.int64)
    n = 0
    for q in range(i.size):
        p, d = i[q], e[q]
        if not (math.floor(plo[0, p] / cell) <= math.floor(ehi[0, d] / cell)
                and math.floor(phi[0, p] / cell) >= math.floor(elo[0, d] / cell)
                and math.floor(plo[1, p] / cell) <= math.floor(ehi[1, d] / cell)
                and math.floor(phi[1, p] / cell) >= math.floor(elo[1, d] / cell)):
            keep[n] = q
            n += 1
    keep = keep[:n]
    keys = i[keep] * n_edges + e[keep]
    order = np.argsort(keys, kind="mergesort")  # stable: the first image of a key comes first
    sel = np.empty(n, np.int64)
    m = 0
    for r in range(n):
        if m == 0 or keys[order[r]] != keys[order[r - 1]]:
            sel[m] = keep[order[r]]
            m += 1
    sel = sel[:m]
    return i[sel], e[sel], k[sel]


@njit(**NB)
def image_pair_kernel(plo, phi, elo, ehi, point_ok, edge_ok, S, cell, point_body, edge_body, body_fixed):
    """Pairs of image_pairs: per offset s, points whose box moved by s reaches the extent of the edges with
    edge_ok, edges with edge_ok overlapping those points' union box, and the (point, edge) pairs among them whose
    floored cell ranges overlap, other bodies only and not both fixed. Returns (points, edges, offset index)."""
    N, E = plo.shape[1], elo.shape[1]
    ext = np.array([np.inf, np.inf, -np.inf, -np.inf])  # lo x, lo y, hi x, hi y
    anyedge = False
    for e in range(E):
        if edge_ok[e]:
            anyedge = True
            for c in range(2):
                ext[c] = min(ext[c], elo[c, e])
                ext[2 + c] = max(ext[2 + c], ehi[c, e])
    empty = np.empty(0, np.int64)
    if not anyedge:
        return empty, empty, empty
    K = S.shape[0]
    # Points near a seam first: an offset's test below passes only if the test of its nonzero x or y part does, so
    # points failing those for every offset (most of them) are skipped; the rest get the same float tests as before
    xs, ys = np.empty(K), np.empty(K)  # the distinct nonzero x and y parts of the offsets
    nx = ny = 0
    for k in range(K):
        if S[k, 0] != 0.0 and not (S[k, 0] == xs[:nx]).any():
            xs[nx] = S[k, 0]
            nx += 1
        if S[k, 1] != 0.0 and not (S[k, 1] == ys[:ny]).any():
            ys[ny] = S[k, 1]
            ny += 1
    near = np.empty(N, np.int64)
    nn = 0
    for i in range(N):
        if not point_ok[i]:
            continue
        hit = False
        for j in range(nx):
            if plo[0, i] + xs[j] <= ext[2] and phi[0, i] + xs[j] >= ext[0]:
                hit = True
        for j in range(ny):
            if plo[1, i] + ys[j] <= ext[3] and phi[1, i] + ys[j] >= ext[1]:
                hit = True
        if hit:
            near[nn] = i
            nn += 1
    pts = np.empty((K, nn), np.int64)  # per offset: its points, ascending
    npts = np.zeros(K, np.int64)
    ub = np.empty((K, 4))  # per offset: union box of its moved points (lo x, lo y, hi x, hi y)
    for k in range(K):
        sx, sy = S[k, 0], S[k, 1]
        ulx, uly, uhx, uhy = np.inf, np.inf, -np.inf, -np.inf
        for q in range(nn):
            i = near[q]
            lx, ly, hx, hy = plo[0, i] + sx, plo[1, i] + sy, phi[0, i] + sx, phi[1, i] + sy
            if lx <= ext[2] and hx >= ext[0] and ly <= ext[3] and hy >= ext[1]:
                pts[k, npts[k]] = i
                npts[k] += 1
                ulx, uly, uhx, uhy = min(ulx, lx), min(uly, ly), max(uhx, hx), max(uhy, hy)
        ub[k, 0], ub[k, 1], ub[k, 2], ub[k, 3] = ulx, uly, uhx, uhy
    eds = np.empty(E, np.int64)
    ec = np.empty((4, E), np.int64)  # cell ranges of the selected edges: x lo, x hi, y lo, y hi
    cap = 1024
    oi, oe, ok = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64)
    m = 0
    for k in range(K):
        sx, sy = S[k, 0], S[k, 1]
        npt = npts[k]
        if npt == 0:
            continue
        ulx, uly, uhx, uhy = ub[k, 0], ub[k, 1], ub[k, 2], ub[k, 3]
        ned = 0
        for e in range(E):
            if edge_ok[e] and elo[0, e] <= uhx and ehi[0, e] >= ulx and elo[1, e] <= uhy and ehi[1, e] >= uly:
                eds[ned] = e
                # the edge's cell range, floored once per offset (the pair loop below only compares integers)
                ec[0, ned] = np.int64(math.floor(elo[0, e] / cell))
                ec[1, ned] = np.int64(math.floor(ehi[0, e] / cell))
                ec[2, ned] = np.int64(math.floor(elo[1, e] / cell))
                ec[3, ned] = np.int64(math.floor(ehi[1, e] / cell))
                ned += 1
        for q in range(npt):
            i = pts[k, q]
            cx0 = np.int64(math.floor((plo[0, i] + sx) / cell))
            cy0 = np.int64(math.floor((plo[1, i] + sy) / cell))
            cx1 = np.int64(math.floor((phi[0, i] + sx) / cell))
            cy1 = np.int64(math.floor((phi[1, i] + sy) / cell))
            pb = point_body[i]
            for r in range(ned):
                if not (ec[0, r] <= cx1 and ec[1, r] >= cx0 and ec[2, r] <= cy1 and ec[3, r] >= cy0):
                    continue
                e = eds[r]
                eb = edge_body[e]
                if pb == eb or (body_fixed[pb] and body_fixed[eb]):
                    continue
                if m == cap:
                    cap *= 2
                    ni, ne, nk = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64)
                    ni[:m], ne[:m], nk[:m] = oi[:m], oe[:m], ok[:m]
                    oi, oe, ok = ni, ne, nk
                oi[m], oe[m], ok[m] = i, e, k
                m += 1
    return oi[:m].copy(), oe[:m].copy(), ok[:m].copy()


@njit(**NB)
def merge_kernel(i_d, e_d, i_m, e_m, ox_m, oy_m, n_edges):
    """merge_with_images' merge (World._merge_images): drop image pairs whose key i * n_edges + e is a direct one,
    keep the first image (in list order) of each remaining key, and merge them into the sorted direct list (direct
    keys are sorted and unique)."""
    nd, nm = i_d.size, i_m.size
    keys_m = i_m * n_edges + e_m
    fresh = np.empty(nm, np.int64)
    nf = 0
    for k in range(nm):  # not met directly (binary search in the direct keys)
        key = keys_m[k]
        lo, hi = 0, nd
        while lo < hi:
            mid = (lo + hi) // 2
            if i_d[mid] * n_edges + e_d[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        if lo >= nd or i_d[lo] * n_edges + e_d[lo] != key:
            fresh[nf] = k
            nf += 1
    fresh = fresh[:nf]
    order = fresh[np.argsort(keys_m[fresh], kind="mergesort")]  # stable: the first image of a key comes first
    sel = np.empty(nf, np.int64)
    ns = 0
    for q in range(nf):
        if ns == 0 or keys_m[order[q]] != keys_m[sel[ns - 1]]:
            sel[ns] = order[q]
            ns += 1
    n = nd + ns
    pts, eids = np.empty(n, np.int64), np.empty(n, np.int64)
    ox, oy = np.zeros(n), np.zeros(n)
    a, b = 0, 0
    for k in range(n):  # merge; an image key never equals a direct one
        if b < ns and (a >= nd or keys_m[sel[b]] < i_d[a] * n_edges + e_d[a]):
            j = sel[b]
            pts[k], eids[k], ox[k], oy[k] = i_m[j], e_m[j], ox_m[j], oy_m[j]
            b += 1
        else:
            pts[k], eids[k] = i_d[a], e_d[a]
            a += 1
    return pts, eids, ox, oy


def ccd_cells(world, D, cell):
    """World._ccd_cells."""
    _, ea, eb, *_ = _topology(world)
    return [floor_cells(box, cell) for box in ccd_boxes(world.PREV, D, ea, eb)]


def ccd_pairs(world, cells):
    """The CPU part of World._ccd_pairs."""
    return _pairs(world, *cells, False)


def ccd_detect(world, D, pts, eids, gap, tau, off=None):
    """World._ccd_detect for the given broad phase pairs (off: their (2, m) image offsets, or None)."""
    _, ea, eb, *_ = _topology(world)
    ox, oy = _offsets(off)
    out = ccd_narrow(world.PREV, D, ea, eb, pts.astype(np.int64, copy=False), eids.astype(np.int64, copy=False), gap,
                     tau, ox, oy)
    if off is None:
        return out[:5]
    return out[:5] + (out[5],)


def ccd_impulses(world, pts, eids, ts, ss, sides, h, push, off=None):
    """World._ccd_impulses. The order comes from np.argsort itself (ties then break exactly as in numpy)."""
    _, ea, eb, *_ = _topology(world)
    ox, oy = _offsets(off)
    impulse_pass(world.V, world.W, world.PREV, ea, eb, pts, eids, ts, ss, sides, np.argsort(ts), h, push, ox, oy)


def bounce(world, h, r):
    """World._bounce."""
    _, ea, eb, *_, edge_offsets, _ = _topology(world)
    rest = np.array([world.bodies[k].restitution for k in world.bounce_bodies], dtype=float)
    bounce_pass(world.P, world.V, world.W, world.offsets.astype(np.int64, copy=False), edge_offsets, ea, eb,
                np.array(world.bounce_bodies, dtype=np.int64), rest, r, h)


# ================================================================ contacts (World._solve_contacts)
@njit(**NB)
def moved_max(P, P0):
    """Largest point displacement between P0 and P (np.hypot(*(P - P0)).max())."""
    m = 0.0
    for i in range(P.shape[1]):
        d = math.hypot(P[0, i] - P0[0, i], P[1, i] - P0[1, i])
        if d > m:
            m = d
    return m


@njit(**NB)
def contact_pass(P, PREV, XS, W, ea, eb, esign, pts, eids, r, mu, slack, want_reach, ox, oy):
    """One Jacobi contact pass, in place on P. Returns the mask dist < r + slack over the candidates if want_reach,
    else an empty array. One loop that skips non-touching pairs right after the distance (numpy computes everything
    for every candidate, then filters). ox / oy: per pair image offsets added to the point (periodic axis), or empty
    for none."""
    m = pts.size
    has_off = ox.size > 0
    keep = np.zeros(m if want_reach else 0, np.bool_)
    # per active contact: point, edge ends, the three shares and lam
    cp = np.empty(m, np.int64)
    ca = np.empty(m, np.int64)
    cb = np.empty(m, np.int64)
    s_p, s_a, s_b = np.empty(m), np.empty(m), np.empty(m)
    lx, ly = np.empty(m), np.empty(m)
    nc = 0
    for k in range(m):
        ip = pts[k]
        e = eids[k]
        ia = ea[e]
        ib = eb[e]
        px, py, ax, ay = P[0, ip], P[1, ip], P[0, ia], P[1, ia]
        if has_off:
            px += ox[k]
            py += oy[k]
        ex, ey = P[0, ib] - ax, P[1, ib] - ay
        len2 = max(ex * ex + ey * ey, 1e-12)
        t = min(max(((px - ax) * ex + (py - ay) * ey) / len2, 0.0), 1.0)
        cx, cy = ax + t * ex, ay + t * ey
        dx, dy = px - cx, py - cy
        dist = math.hypot(dx, dy)
        if want_reach:
            keep[k] = dist < r + slack
        if not dist < r:
            continue
        # edge normal facing the side the point started on: XS is PREV (this substep's start), or the CCD span's
        # start when CCD runs once per frame (World._span_start)
        pax, pay = XS[0, ia], XS[1, ia]
        e0x, e0y = XS[0, ib] - pax, XS[1, ib] - pay
        if has_off:
            q0x, q0y = (XS[0, ip] + ox[k]) - pax, (XS[1, ip] + oy[k]) - pay
        else:
            q0x, q0y = XS[0, ip] - pax, XS[1, ip] - pay
        cross0 = e0x * q0y - e0y * q0x
        if cross0 != 0:
            side = 1.0 if cross0 > 0 else -1.0
        else:
            side = -esign[e]
        sq = math.sqrt(len2)
        nex, ney = -side * ey / sq, -side * (-ex) / sq
        if (dx * nex + dy * ney) < 0 or dist < 1e-9:
            nx, ny = nex, ney
        else:
            dd = max(dist, 1e-9)
            nx, ny = dx / dd, dy / dd
        depth = r - (dx * nx + dy * ny)
        wp, wa, wb = W[ip], W[ia], W[ib]
        denom = wp + wa * ((1 - t) * (1 - t)) + wb * (t * t)
        if not denom > 1e-12:
            continue
        # friction
        cpx = PREV[0, ia] + t * (PREV[0, ib] - PREV[0, ia])
        cpy = PREV[1, ia] + t * (PREV[1, ib] - PREV[1, ia])
        if has_off:
            relx, rely = (px - (PREV[0, ip] + ox[k])) - (cx - cpx), (py - (PREV[1, ip] + oy[k])) - (cy - cpy)
        else:
            relx, rely = (px - PREV[0, ip]) - (cx - cpx), (py - PREV[1, ip]) - (cy - cpy)
        rn = relx * nx + rely * ny
        tx, ty = relx - rn * nx, rely - rn * ny
        fric = min(1.0, mu * depth / max(math.hypot(tx, ty), 1e-9))
        cp[nc], ca[nc], cb[nc] = ip, ia, ib
        lx[nc], ly[nc] = (depth * nx - tx * fric) / denom, (depth * ny - ty * fric) / denom
        s_p[nc], s_a[nc], s_b[nc] = wp, -wa * (1 - t), -wb * t
        nc += 1
    if nc == 0:
        return keep
    n = P.shape[1]
    cnt, accx, accy = np.zeros(n), np.zeros(n), np.zeros(n)
    # the order of np.bincount over concatenate([pts, ia, ib])
    for c in range(nc):
        cnt[cp[c]] += 1.0
        accx[cp[c]] += s_p[c] * lx[c]
        accy[cp[c]] += s_p[c] * ly[c]
    for c in range(nc):
        cnt[ca[c]] += 1.0
        accx[ca[c]] += s_a[c] * lx[c]
        accy[ca[c]] += s_a[c] * ly[c]
    for c in range(nc):
        cnt[cb[c]] += 1.0
        accx[cb[c]] += s_b[c] * lx[c]
        accy[cb[c]] += s_b[c] * ly[c]
    for i in range(n):
        P[0, i] += accx[i] / max(cnt[i], 1.0)
    for i in range(n):
        P[1, i] += accy[i] / max(cnt[i], 1.0)
    return keep


# ================================================================ body constraints
@njit(**NB)
def springs(P, W, si, sj, rest, stiff, count):
    """All SoftBody springs as one Jacobi pass, in place on P (World._solve_body_constraints)."""
    m = si.size
    cx, cy = np.empty(m), np.empty(m)
    for k in range(m):
        i, j = si[k], sj[k]
        dx, dy = P[0, j] - P[0, i], P[1, j] - P[1, i]
        dist = max(math.hypot(dx, dy), 1e-9)
        f = stiff[k] * (dist - rest[k]) / (dist * (W[i] + W[j]))
        cx[k], cy[k] = f * dx, f * dy
    n = P.shape[1]
    ax, ay = np.zeros(n), np.zeros(n)
    for k in range(m):  # bincount order: i's shares, then j's
        ax[si[k]] += W[si[k]] * cx[k]
        ay[si[k]] += W[si[k]] * cy[k]
    for k in range(m):
        ax[sj[k]] += -W[sj[k]] * cx[k]
        ay[sj[k]] += -W[sj[k]] * cy[k]
    for i in range(n):
        P[0, i] += ax[i] / count[i]
    for i in range(n):
        P[1, i] += ay[i] / count[i]


@njit(**NB)
def xpbd_springs(P, PREV, W, si, sj, rest, comp, ci, cj, lam, h, beta):
    """World._xpbd_springs: one Jacobi XPBD pass (mass splitting, Rayleigh damping) over every spring, in place on P
    and the multipliers lam, in the numpy reference's expression order."""
    m = si.size
    nxs, nys, ash, bsh = np.empty(m), np.empty(m), np.empty(m), np.empty(m)
    hh = h * h
    for k in range(m):
        i, j = si[k], sj[k]
        dx, dy = P[0, j] - P[0, i], P[1, j] - P[1, i]
        dist = max(math.hypot(dx, dy), 1e-9)
        nx, ny = dx / dist, dy / dist
        wi, wj = W[i], W[j]
        at = comp[k] * (wi + wj) / hh
        gamma = at * beta * h
        rel = nx * ((P[0, j] - PREV[0, j]) - (P[0, i] - PREV[0, i])) + \
            ny * ((P[1, j] - PREV[1, j]) - (P[1, i] - PREV[1, i]))
        num = -(dist - rest[k]) - at * lam[k] - gamma * rel
        den = (1.0 + gamma) * (ci[k] * wi + cj[k] * wj) + at
        dl = num / den if den > 0 else 0.0
        lam[k] += dl
        nxs[k], nys[k] = nx, ny
        ash[k], bsh[k] = -wi * dl, wj * dl
    n = P.shape[1]
    ax, ay = np.zeros(n), np.zeros(n)
    for k in range(m):  # bincount order: i's shares, then j's
        ax[si[k]] += ash[k] * nxs[k]
    for k in range(m):
        ax[sj[k]] += bsh[k] * nxs[k]
    for k in range(m):
        ay[si[k]] += ash[k] * nys[k]
    for k in range(m):
        ay[sj[k]] += bsh[k] * nys[k]
    for i in range(n):
        P[0, i] += ax[i]
    for i in range(n):
        P[1, i] += ay[i]


@njit(**NB)
def shape_match(P, idx, rest, stiff):
    """RigidBody.solve_constraints for one same-size group: idx (B, n) point ids, rest (B, 2, n), stiff (B,)."""
    B, n = idx.shape
    tx, ty, t2 = np.empty(n), np.empty(n), np.empty(n)
    for b in range(B):
        for k in range(n):
            tx[k], ty[k] = P[0, idx[b, k]], P[1, idx[b, k]]
        # centre of mass: a plain sequential sum, because numpy's P[:, idx] gather is not C-contiguous (strides
        # (8, 8n, 16)) and its mean then reduces sequentially; the products below are contiguous -> pairwise
        comx, comy = tx[0], ty[0]
        for k in range(1, n):
            comx += tx[k]
            comy += ty[k]
        comx /= n
        comy /= n
        for k in range(n):
            tx[k], ty[k] = tx[k] - comx, ty[k] - comy
        for k in range(n):
            t2[k] = tx[k] * rest[b, 0, k]
        a00 = _pw(t2, n)
        for k in range(n):
            t2[k] = tx[k] * rest[b, 1, k]
        a01 = _pw(t2, n)
        for k in range(n):
            t2[k] = ty[k] * rest[b, 0, k]
        a10 = _pw(t2, n)
        for k in range(n):
            t2[k] = ty[k] * rest[b, 1, k]
        a11 = _pw(t2, n)
        # best-fit rotation straight from the covariance, no trig (as in the numpy reference)
        num, den = a10 - a01, a00 + a11
        rr = math.hypot(num, den)
        if rr > 0:
            c, s = den / rr, num / rr
        else:
            c, s = 1.0, 0.0
        for k in range(n):
            g = idx[b, k]
            gx = (c * rest[b, 0, k] - s * rest[b, 1, k]) + comx
            gy = (s * rest[b, 0, k] + c * rest[b, 1, k]) + comy
            P[0, g] += stiff[b] * (gx - P[0, g])
            P[1, g] += stiff[b] * (gy - P[1, g])


@njit(**NB)
def rope_pass(P, W, ia, ib, anchors, rest, stiff):
    """Rope.solve_constraints for every rope, in order (Gauss-Seidel). ia / ib: global point index of each end, or
    -1 for a fixed anchor at anchors[rope, end]."""
    for r in range(ia.size):
        a, b = ia[r], ib[r]
        if a >= 0:
            pax, pay, wa = P[0, a], P[1, a], W[a]
        else:
            pax, pay, wa = anchors[r, 0, 0], anchors[r, 0, 1], 0.0
        if b >= 0:
            pbx, pby, wb = P[0, b], P[1, b], W[b]
        else:
            pbx, pby, wb = anchors[r, 1, 0], anchors[r, 1, 1], 0.0
        dx, dy = pbx - pax, pby - pay
        dist = math.hypot(dx, dy)
        if dist <= rest[r] or dist < 1e-9 or wa + wb <= 0:
            continue  # slack (or nothing can move)
        f = stiff[r] * (dist - rest[r]) / (dist * (wa + wb))
        cx, cy = f * dx, f * dy
        if a >= 0 and wa > 0:
            P[0, a] += wa * cx
            P[1, a] += wa * cy
        if b >= 0 and wb > 0:
            P[0, b] += -wb * cx
            P[1, b] += -wb * cy


# ================================================================ broad phase (World._cell_pairs + _filter_pairs)
@njit(**NB)
def cell_pairs(plo, phi, elo, ehi, ea, eb, point_body, edge_body, body_fixed, body_self, point_local, point_ring,
               edge_local0, edge_local1, contact, max_cells):
    """Filtered (point, edge) pairs whose cell boxes share a grid cell, sorted by (point, edge), no duplicates.

    The cell -> edges table is a counting sort in edge order (each cell's list comes out sorted by edge id): a dense
    grid over the points' cell range when that has at most max_cells cells, else a sorted list of the occupied cells
    with binary search (the world box walls run far off screen). Only cells inside the points' range are registered:
    no point can share any other cell. A point spanning several cells (swept CCD boxes) merges its cells' edge lists
    with a per-edge stamp (dedupe) and an insertion sort."""
    N, E = plo.shape[1], elo.shape[1]
    empty = np.empty(0, np.int64)
    if N == 0 or E == 0:
        return empty, empty
    x0, y0, x1, y1 = plo[0].min(), plo[1].min(), phi[0].max(), phi[1].max()
    rows = y1 - y0 + 1
    dense = (x1 - x0 + 1) * rows <= max_cells
    if dense:
        ncell = (x1 - x0 + 1) * rows
        start = np.zeros(ncell + 1, np.int64)
        for e in range(E):
            for cx in range(max(elo[0, e], x0), min(ehi[0, e], x1) + 1):
                for cy in range(max(elo[1, e], y0), min(ehi[1, e], y1) + 1):
                    start[(cx - x0) * rows + (cy - y0) + 1] += 1
        for c in range(ncell):
            start[c + 1] += start[c]
        slots = np.empty(start[ncell], np.int64)
        cur = start[:-1].copy()
        for e in range(E):
            for cx in range(max(elo[0, e], x0), min(ehi[0, e], x1) + 1):
                for cy in range(max(elo[1, e], y0), min(ehi[1, e], y1) + 1):
                    c = (cx - x0) * rows + (cy - y0)
                    slots[cur[c]] = e
                    cur[c] += 1
        ukeys = empty
    else:
        T = 0
        for e in range(E):
            T += (max(0, min(ehi[0, e], x1) - max(elo[0, e], x0) + 1)
                  * max(0, min(ehi[1, e], y1) - max(elo[1, e], y0) + 1))
        comp = np.empty(T, np.int64)
        t = 0
        for e in range(E):
            for cx in range(max(elo[0, e], x0), min(ehi[0, e], x1) + 1):
                for cy in range(max(elo[1, e], y0), min(ehi[1, e], y1) + 1):
                    comp[t] = ((cx - x0) * rows + (cy - y0)) * E + e
                    t += 1
        comp.sort()  # by (cell, edge id)
        slots = np.empty(T, np.int64)
        ukeys = np.empty(T, np.int64)
        start = np.empty(T + 1, np.int64)
        nu = 0
        for t in range(T):
            k = comp[t] // E
            slots[t] = comp[t] - k * E
            if nu == 0 or ukeys[nu - 1] != k:
                ukeys[nu] = k
                start[nu] = t
                nu += 1
        start[nu] = T
        ukeys = ukeys[:nu]
    nuk = ukeys.size
    # upper bound on emitted pairs (cell list lengths summed over each point's cells): allocate once
    cap = 0
    for i in range(N):
        for cx in range(plo[0, i], phi[0, i] + 1):
            for cy in range(plo[1, i], phi[1, i] + 1):
                c = (cx - x0) * rows + (cy - y0)
                if not dense:
                    u = np.searchsorted(ukeys, c)
                    if u == nuk or ukeys[u] != c:
                        continue
                    c = u
                cap += start[c + 1] - start[c]
    oi, oe = np.empty(cap, np.int64), np.empty(cap, np.int64)
    m = 0
    stamp = np.full(E, -1, np.int64)  # last point that collected each edge
    buf = np.empty(max(cap, 1), np.int64)
    for i in range(N):
        if plo[0, i] == phi[0, i] and plo[1, i] == phi[1, i]:  # one cell: its list is sorted, no duplicates
            c = (plo[0, i] - x0) * rows + (plo[1, i] - y0)
            if not dense:
                u = np.searchsorted(ukeys, c)
                if u == nuk or ukeys[u] != c:
                    continue
                c = u
            nb = start[c + 1] - start[c]
            buf[:nb] = slots[start[c]:start[c + 1]]
        else:
            nb = 0
            for cx in range(plo[0, i], phi[0, i] + 1):
                for cy in range(plo[1, i], phi[1, i] + 1):
                    c = (cx - x0) * rows + (cy - y0)
                    if not dense:
                        u = np.searchsorted(ukeys, c)
                        if u == nuk or ukeys[u] != c:
                            continue
                        c = u
                    for s in range(start[c], start[c + 1]):
                        e = slots[s]
                        if stamp[e] != i:
                            stamp[e] = i
                            q = nb  # insertion into the sorted run buf[0:nb]
                            while q > 0 and buf[q - 1] > e:
                                buf[q] = buf[q - 1]
                                q -= 1
                            buf[q] = e
                            nb += 1
        # World._filter_pairs, inlined by hand (a helper taking a dozen arrays costs ~4x in this loop)
        pb, li, n = point_body[i], point_local[i], point_ring[i]
        for q in range(nb):
            e = buf[q]
            ebd = edge_body[e]
            if pb != ebd:
                ok = not (body_fixed[pb] and body_fixed[ebd])
            elif not body_self[pb]:
                ok = False
            elif contact:  # not the point's own or its neighbors' edges: cyclic distance > 1 to both ends
                x = li - edge_local0[e]
                if x < 0:
                    x += n
                ok = not (x == 0 or x == 1 or x == n - 1)
                if ok:
                    x = li - edge_local1[e]
                    if x < 0:
                        x += n
                    ok = not (x == 0 or x == 1 or x == n - 1)
            else:
                ok = i != ea[e] and i != eb[e]
            if ok:
                oi[m] = i
                oe[m] = e
                m += 1
    return oi[:m].copy(), oe[:m].copy()


@njit(**NB)
def contact_cells(P, ea, eb, cell, pad):
    """World._candidates' cell ranges: point cells (2, N), padded edge box lo / hi (2, E)."""
    N, E = P.shape[1], ea.size
    pc = np.empty((2, N), np.int64)
    for i in range(N):
        pc[0, i] = np.int64(math.floor(P[0, i] / cell))
        pc[1, i] = np.int64(math.floor(P[1, i] / cell))
    lo, hi = np.empty((2, E), np.int64), np.empty((2, E), np.int64)
    for e in range(E):
        for k in range(2):
            a, b = P[k, ea[e]], P[k, eb[e]]
            lo[k, e] = np.int64(math.floor((min(a, b) - pad) / cell))
            hi[k, e] = np.int64(math.floor((max(a, b) + pad) / cell))
    return pc, lo, hi


@njit(**NB)
def _fmin(a, b):
    """np.minimum of two floats (nan if either is nan)."""
    return a if (a <= b or a != a) else b


@njit(**NB)
def _fmax(a, b):
    """np.maximum of two floats (nan if either is nan)."""
    return a if (a >= b or a != a) else b


@njit(**NB)
def span_cells(plo, phi, ea, eb, pad, cell):
    """The cell ranges World._span_candidates floors from World._span_boxes, given its point boxes (float (2, N)):
    point lo / hi (2, N) and lo / hi (2, E) of the edge boxes around them grown by `pad` (2 contact_radius), as
    np.floor(box / cell) in int64, with the same float operations as the numpy expressions."""
    N, E = plo.shape[1], ea.size
    pl, ph = np.empty((2, N), np.int64), np.empty((2, N), np.int64)
    el, eh = np.empty((2, E), np.int64), np.empty((2, E), np.int64)
    for k in range(2):
        for i in range(N):
            pl[k, i] = np.int64(math.floor(plo[k, i] / cell))
            ph[k, i] = np.int64(math.floor(phi[k, i] / cell))
        for e in range(E):
            el[k, e] = np.int64(math.floor((_fmin(plo[k, ea[e]], plo[k, eb[e]]) - pad) / cell))
            eh[k, e] = np.int64(math.floor((_fmax(phi[k, ea[e]], phi[k, eb[e]]) + pad) / cell))
    return pl, ph, el, eh


@njit(**NB)
def ccd_boxes(X0, D, ea, eb):
    """World._ccd_boxes: point lo / hi (2, N), edge lo / hi (2, E, grown by 1.0 px)."""
    N, E = X0.shape[1], ea.size
    plo, phi, elo, ehi = np.empty((2, N)), np.empty((2, N)), np.empty((2, E)), np.empty((2, E))
    for k in range(2):
        for i in range(N):
            a = X0[k, i]
            b = a + D[k, i]
            plo[k, i], phi[k, i] = min(a, b), max(a, b)
        for e in range(E):
            a0, b0 = X0[k, ea[e]], X0[k, eb[e]]
            a1, b1 = a0 + D[k, ea[e]], b0 + D[k, eb[e]]
            elo[k, e] = min(min(a0, b0), min(a1, b1)) - 1.0
            ehi[k, e] = max(max(a0, b0), max(a1, b1)) + 1.0
    return plo, phi, elo, ehi


@njit(**NB)
def floor_cells(box, cell):
    """np.floor(box / cell).astype(int)."""
    out = np.empty(box.shape, np.int64)
    for k in range(box.shape[0]):
        for i in range(box.shape[1]):
            out[k, i] = np.int64(math.floor(box[k, i] / cell))
    return out


# ================================================================ CCD narrow phase + impulses
@njit(**NB)
def _tau_hit(rr, e0x, e0y, dex, dey, q0x, q0y, dqx, dqy, tau2):
    """The robust acceptance test of World._ccd_detect for root rr: rr in [0, 1] and the point within ccd_tau of
    the segment (s clamped) at rr. Returns (accepted, s)."""
    if not (rr >= 0 and rr <= 1):
        return False, 0.0
    ex, ey = e0x + rr * dex, e0y + rr * dey
    qx, qy = q0x + rr * dqx, q0y + rr * dqy
    s = min(max((qx * ex + qy * ey) / max(ex * ex + ey * ey, 1e-300), 0.0), 1.0)
    ox, oy = qx - s * ex, qy - s * ey
    return ox * ox + oy * oy <= tau2, s


@njit(**NB)
def ccd_narrow(X0, D, ea, eb, pts, eids, gap, tau, offx, offy):
    """World._ccd_detect after the broad phase: the swept-box pre-reject inline, then the quadratic per pair
    (textbook roots, verified; the float64-robust solve where they find nothing). Returns (pts, edge_ids, t, s,
    side, offsets (2, m)) of the hits, in pair order. offx / offy: per pair image offsets added to the point
    (periodic axis), or empty for none (the returned offsets are then empty too)."""
    m = pts.size
    has_off = offx.size > 0
    tau2 = tau * tau
    hp, he = np.empty(m, np.int64), np.empty(m, np.int64)
    ht, hs, hside = np.empty(m), np.empty(m), np.empty(m)
    ho = np.empty((2, m if has_off else 0))
    nh = 0
    for k in range(m):
        ip = pts[k]
        e = eids[k]
        ia = ea[e]
        ib = eb[e]
        ok = True  # pre-reject: the point's and the edge's swept boxes must overlap
        for c in range(2):
            p0 = X0[c, ip]
            if has_off:
                if c == 0:
                    p0 += offx[k]
                else:
                    p0 += offy[k]
            p1 = p0 + D[c, ip]
            a0, b0 = X0[c, ia], X0[c, ib]
            a1, b1 = a0 + D[c, ia], b0 + D[c, ib]
            if not (min(p0, p1) <= max(max(a0, b0), max(a1, b1)) + 1.0
                    and max(p0, p1) >= min(min(a0, b0), min(a1, b1)) - 1.0):
                ok = False
        if not ok:
            continue
        e0x, e0y = X0[0, ib] - X0[0, ia], X0[1, ib] - X0[1, ia]
        dex, dey = D[0, ib] - D[0, ia], D[1, ib] - D[1, ia]
        if has_off:
            q0x, q0y = (X0[0, ip] + offx[k]) - X0[0, ia], (X0[1, ip] + offy[k]) - X0[1, ia]
        else:
            q0x, q0y = X0[0, ip] - X0[0, ia], X0[1, ip] - X0[1, ia]
        dqx, dqy = D[0, ip] - D[0, ia], D[1, ip] - D[1, ia]
        c0 = e0x * q0y - e0y * q0x
        c1 = (e0x * dqy - e0y * dqx) + (dex * q0y - dey * q0x)
        c2 = dex * dqy - dey * dqx
        side = (1.0 if c0 > 0 else -1.0) if c0 != 0 else 1.0
        disc = c1 * c1 - 4 * c2 * c0
        if disc < 0 and disc > -1e-9 * (c1 * c1 + abs(4 * c2 * c0)):
            disc = 0.0
        sq = math.sqrt(disc) if disc >= 0 else np.nan
        if abs(c2) > 1e-12:
            r1, r2 = (-c1 - sq) / (2 * c2), (-c1 + sq) / (2 * c2)
        else:
            r1, r2 = -c0 / c1, np.nan
        th, sh = np.inf, 0.0
        for rr in (r1, r2):
            ex, ey = e0x + rr * dex, e0y + rr * dey
            qx, qy = q0x + rr * dqx, q0y + rr * dqy
            s = (qx * ex + qy * ey) / max(ex * ex + ey * ey, 1e-12)
            if rr >= 0 and rr <= 1 and s >= 0 and s <= 1 and rr < th:
                th, sh = rr, s
        if th != np.inf:  # verify: the point must be on the segment (within tau) at the root
            ex, ey = e0x + th * dex, e0y + th * dey
            ox, oy = (q0x + th * dqx) - sh * ex, (q0y + th * dqy) - sh * ey
            if ox * ox + oy * oy > tau2:
                th, sh = np.inf, 0.0
        if th == np.inf:  # float64-robust solve
            qq = -0.5 * (c1 + (sq if c1 >= 0 else -sq))
            if c2 != 0:
                ra, rb = qq / c2, c0 / qq
            elif c1 != 0:
                ra, rb = -c0 / c1, np.nan
            else:  # collinear all substep: the times the point passes either end of the segment
                w = dqx * e0x + dqy * e0y
                ra = -(q0x * e0x + q0y * e0y) / w
                rb = -((q0x - e0x) * e0x + (q0y - e0y) * e0y) / (w - (dex * e0x + dey * e0y))
            for rr in (ra, rb):
                if rr < th:
                    acc, s = _tau_hit(rr, e0x, e0y, dex, dey, q0x, q0y, dqx, dqy, tau2)
                    if acc:
                        th, sh = rr, s
        if th == np.inf:  # ending within gap of the edge counts as a hit at t = 1
            e1x, e1y, q1x, q1y = e0x + dex, e0y + dey, q0x + dqx, q0y + dqy
            len2 = max(e1x * e1x + e1y * e1y, 1e-12)
            s1 = min(max((q1x * e1x + q1y * e1y) / len2, 0.0), 1.0)
            if math.hypot(q1x - s1 * e1x, q1y - s1 * e1y) < gap:
                th, sh = 1.0, s1
        if th != np.inf:
            hp[nh], he[nh], ht[nh], hs[nh], hside[nh] = ip, e, th, sh, side
            if has_off:
                ho[0, nh], ho[1, nh] = offx[k], offy[k]
            nh += 1
    no = nh if has_off else 0
    return (hp[:nh].copy(), he[:nh].copy(), ht[:nh].copy(), hs[:nh].copy(), hside[:nh].copy(), ho[:, :no].copy())


@njit(**NB)
def impulse_pass(V, W, X0, ea, eb, pts, eids, ts, ss, sides, order, h, push, ox, oy):
    """World._ccd_impulses: one Gauss-Seidel pass over the hits in `order` (earliest first), editing V. ox / oy: per
    hit image offsets of the point (periodic axis), or empty for none."""
    has_off = ox.size > 0
    for q in range(order.size):
        k = order[q]
        i, ia, ib, t, s = pts[k], ea[eids[k]], eb[eids[k]], ts[k], ss[k]
        xix, xiy = X0[0, i], X0[1, i]
        if has_off:
            xix += ox[k]
            xiy += oy[k]
        e0x, e0y = X0[0, ib] - X0[0, ia], X0[1, ib] - X0[1, ia]
        u = min(max(((xix - X0[0, ia]) * e0x + (xiy - X0[1, ia]) * e0y) / max(e0x * e0x + e0y * e0y, 1e-12),
                    0.0), 1.0)
        sepx, sepy = xix - (X0[0, ia] + u * e0x), xiy - (X0[1, ia] + u * e0y)
        sl = math.hypot(sepx, sepy)
        if sl > 1e-9:
            nx, ny = sepx / sl, sepy / sl
        else:
            ex = (X0[0, ib] + t * h * V[0, ib]) - (X0[0, ia] + t * h * V[0, ia])
            ey = (X0[1, ib] + t * h * V[1, ib]) - (X0[1, ia] + t * h * V[1, ia])
            el = max(math.hypot(ex, ey), 1e-12)
            nx, ny = sides[k] * -ey / el, sides[k] * ex / el
        vrx = V[0, i] - ((1 - s) * V[0, ia] + s * V[0, ib])
        vry = V[1, i] - ((1 - s) * V[1, ia] + s * V[1, ib])
        epx, epy = xix + h * V[0, i], xiy + h * V[1, i]
        ecx = (1 - s) * (X0[0, ia] + h * V[0, ia]) + s * (X0[0, ib] + h * V[0, ib])
        ecy = (1 - s) * (X0[1, ia] + h * V[1, ia]) + s * (X0[1, ib] + h * V[1, ib])
        dvn = max(max((push - ((epx - ecx) * nx + (epy - ecy) * ny)) / h, -(vrx * nx + vry * ny)), 0.0)
        denom = W[i] + W[ia] * ((1 - s) * (1 - s)) + W[ib] * (s * s)
        if dvn == 0.0 or denom < 1e-12:
            continue
        jx, jy = dvn / denom * nx, dvn / denom * ny
        V[0, i] += W[i] * jx
        V[1, i] += W[i] * jy
        V[0, ia] -= W[ia] * (1 - s) * jx
        V[1, ia] -= W[ia] * (1 - s) * jy
        V[0, ib] -= W[ib] * s * jx
        V[1, ib] -= W[ib] * s * jy


# ================================================================ World._bounce (opt-in restitution)
@njit(**NB)
def _outline_gap(P, a0, a1, ea, eb, f0, f1):
    """solver._outline_gap: points a0..a1 against edges f0..f1."""
    m = np.inf
    for e in range(f0, f1):
        qx, qy = P[0, ea[e]], P[1, ea[e]]
        ex, ey = P[0, eb[e]] - qx, P[1, eb[e]] - qy
        l2 = max(ex * ex + ey * ey, 1e-12)
        for i in range(a0, a1):
            dx, dy = P[0, i] - qx, P[1, i] - qy
            t = min(max((dx * ex + dy * ey) / l2, 0.0), 1.0)
            ox, oy = dx - t * ex, dy - t * ey
            v = ox * ox + oy * oy
            if v < m:
                m = v
    return math.sqrt(m)


@njit(**NB)
def bounce_pass(P, V, W, offsets, eoffsets, ea, eb, bidx, rest, r, h):
    """World._bounce. bidx: World.bounce_bodies, rest: their restitution; offsets / eoffsets: first point / edge of
    each body (plus the total)."""
    sel, sq = np.empty(bidx.size, np.int64), np.empty(bidx.size, np.int64)
    nb = 0
    for q in range(bidx.size):
        k = bidx[q]
        ok = True
        for i in range(offsets[k], offsets[k + 1]):
            if not W[i] > 0:
                ok = False
        if ok:
            sel[nb], sq[nb] = k, q
            nb += 1
    if nb < 2:
        return
    mass, com, vel, reach = np.empty(nb), np.empty((nb, 2)), np.empty((nb, 2)), np.empty(nb)
    for q in range(nb):
        a0 = offsets[sel[q]]
        n = offsets[sel[q] + 1] - a0
        tmp = np.empty(n)
        for i in range(n):
            tmp[i] = 1.0 / W[a0 + i]
        mass[q] = _pw(tmp, n)
        for c in range(2):
            for i in range(n):
                tmp[i] = P[c, a0 + i] / W[a0 + i]
            com[q, c] = _pw(tmp, n) / mass[q]
            for i in range(n):
                tmp[i] = V[c, a0 + i] / W[a0 + i]
            vel[q, c] = _pw(tmp, n) / mass[q]
        rr = -np.inf
        for i in range(n):
            d = math.hypot(P[0, a0 + i] - com[q, 0], P[1, a0 + i] - com[q, 1])
            if d > rr:
                rr = d
        reach[q] = rr
    vmax = -np.inf
    for q in range(nb):
        vmax = max(vmax, math.hypot(vel[q, 0], vel[q, 1]))
    far = 1.1 * r + 4.0 * h * vmax
    npair = nb * (nb - 1) // 2
    pi, pj = np.empty(npair, np.int64), np.empty(npair, np.int64)
    pn, pg, pe, pw = np.empty((npair, 2)), np.empty(npair), np.empty(npair), np.empty(npair)
    m = 0
    for i in range(nb):
        for j in range(i + 1, nb):
            dx, dy = com[j, 0] - com[i, 0], com[j, 1] - com[i, 1]
            dist = math.hypot(dx, dy)
            if dist < 1e-9 or dist > reach[i] + reach[j] + far:
                continue  # bounding circles too far apart to touch this substep
            ki, kj = sel[i], sel[j]
            gap = min(_outline_gap(P, offsets[ki], offsets[ki + 1], ea, eb, eoffsets[kj], eoffsets[kj + 1]),
                      _outline_gap(P, offsets[kj], offsets[kj + 1], ea, eb, eoffsets[ki], eoffsets[ki + 1]))
            if gap < far:
                pi[m], pj[m], pn[m, 0], pn[m, 1], pg[m] = i, j, dx / dist, dy / dist, gap
                pe[m], pw[m] = min(rest[sq[i]], rest[sq[j]]), 1.0 / mass[i] + 1.0 / mass[j]
                m += 1
    if m == 0:
        return
    before = vel.copy()
    for _ in range(4 * m + 4):
        hit = False
        for p in range(m):
            i, j = pi[p], pj[p]
            closing = pn[p, 0] * (vel[i, 0] - vel[j, 0]) + pn[p, 1] * (vel[i, 1] - vel[j, 1])
            if closing > 1e-9 and pg[p] - h * closing < 1.1 * r:
                f = (1.0 + pe[p]) * closing / pw[p]
                jx, jy = f * pn[p, 0], f * pn[p, 1]
                vel[i, 0], vel[i, 1] = vel[i, 0] - jx / mass[i], vel[i, 1] - jy / mass[i]
                vel[j, 0], vel[j, 1] = vel[j, 0] + jx / mass[j], vel[j, 1] + jy / mass[j]
                hit = True
        if not hit:
            break
    for q in range(nb):
        if vel[q, 0] != before[q, 0] or vel[q, 1] != before[q, 1]:
            dvx, dvy = vel[q, 0] - before[q, 0], vel[q, 1] - before[q, 1]
            for i in range(offsets[sel[q]], offsets[sel[q] + 1]):
                V[0, i] += dvx
                V[1, i] += dvy


# ================================================================ numpy-identical sums
@njit(**NB)
def _pw_leaf(a, lo, n):
    """numpy's pairwise_sum for a run of at most 128: 8 accumulators, then the remainder in order."""
    if n < 8:
        res = -0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(**NB)
def _pw(a, n):
    """np.sum(a[:n]) bit for bit (numpy's pairwise summation: halves split at multiples of 8 down to runs of 128).
    Iterative with an explicit stack: see the module docstring about recursion."""
    if n <= 128:
        return _pw_leaf(a, 0, n)
    t_lo, t_n, t_st = np.empty(128, np.int64), np.empty(128, np.int64), np.empty(128, np.int64)
    vals = np.empty(128)
    t_lo[0], t_n[0], t_st[0] = 0, n, 0
    sp, vp = 1, 0
    while sp > 0:
        sp -= 1
        clo, cn, st = t_lo[sp], t_n[sp], t_st[sp]
        if cn <= 128:
            vals[vp] = _pw_leaf(a, clo, cn)
            vp += 1
        elif st == 0:  # split: come back to add the halves (st = 1) after both are summed
            n2 = cn // 2
            n2 -= n2 % 8
            t_lo[sp], t_n[sp], t_st[sp] = clo, cn, 1
            t_lo[sp + 1], t_n[sp + 1], t_st[sp + 1] = clo + n2, cn - n2, 0
            t_lo[sp + 2], t_n[sp + 2], t_st[sp + 2] = clo, n2, 0
            sp += 3
        else:
            vals[vp - 2] = vals[vp - 2] + vals[vp - 1]
            vp -= 1
    return vals[0]


def warmup():
    """Compile (or load from the on-disk cache) every kernel with small arrays of the real types, so no first-use
    compile stalls a frame mid-simulation (seconds the first time after an edit of this file, ~0.1 s cached)."""
    global _warm
    _warm = True
    P = np.zeros((2, 4))
    W = np.ones(4)
    i2, i1 = np.array([0, 1], np.int64), np.array([1, 2], np.int64)
    e, f = np.empty(0, np.int64), np.empty(0)
    moved_max(P, P)
    contact_pass(P, P.copy(), P.copy(), W, i2, i1, np.ones(2), e, e, 3.0, 0.5, 3.0, True, f, f)
    springs(P, W, i2, i1, np.ones(2), np.ones(2), np.ones(4))
    xpbd_springs(P, P.copy(), W, i2, i1, np.ones(2), np.ones(2), np.ones(2), np.ones(2), np.zeros(2), 0.004, 10.0)
    shape_match(P, np.zeros((1, 4), np.int64), np.zeros((1, 2, 4)), np.ones(1))
    rope_pass(P, W, e, e, np.zeros((0, 2, 2)), f, f)
    pc, ec, b = np.zeros((2, 4), np.int64), np.zeros((2, 2), np.int64), np.zeros(4, np.bool_)
    cell_pairs(pc, pc, ec, ec, i2, i1, np.zeros(4, np.int64), np.zeros(2, np.int64), b, b, np.zeros(4, np.int64),
               np.ones(4, np.int64), i2, i1, True, 100)
    contact_cells(P, i2, i1, 24.0, 6.0)
    span_cells(P, P, i2, i1, 6.0, 24.0)
    floor_cells(ccd_boxes(P, P, i2, i1)[0], 24.0)
    ccd_narrow(P, P, i2, i1, e, e, 0.01, 1e-6, f, f)
    impulse_pass(P.copy(), W, P, i2, i1, e, e, f, f, f, e, 0.004, 0.25, f, f)
    img = (P, P, np.zeros((2, 2)), np.zeros((2, 2)), np.ones(4, np.bool_), np.ones(2, np.bool_), np.zeros((1, 2)), 24.0,
           np.array([0, 0, 1, 1], np.int64), np.zeros(2, np.int64), b)
    image_merge_kernel(np.array([0, 1], np.int64), np.zeros(2, np.int64), 2, *img)
    image_pair_kernel(*img)
    image_extras_kernel(2, *img)
    bounce_pass(P, P.copy(), W, np.array([0, 2, 4], np.int64), np.array([0, 1, 2], np.int64), i2, i1, e, f, 3.0, 0.004)
