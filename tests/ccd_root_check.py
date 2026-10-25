"""CCD root-solve robustness check (research/02 F3, build #5). Run from the repo root:

    .venv/bin/python3 tests/ccd_root_check.py [cases_per_family] [seed]

1. World level: a rigid triangle whose vertex path runs (to float64 rounding) through a vertex of a fixed
   triangle and ends 0.5 px inside it, with a tiny rotation (a nearly translating edge). Before the robust solve
   World._resolve_ccd let these through: the root's s rounded to just outside [0, 1] on both edges at the vertex.
2. Narrow phase: adversarial point / moving-edge families scored against a 50-digit mpmath ground truth. A false
   negative (FN) is a real touch World._ccd_detect misses; a far false positive is a reported hit whose true
   distance stays > 1e-3 px and that doesn't end within ccd_gap.
3. The numpy and numba (EnvConfig.jit) narrow phases return bit-identical results on every case.
Exit status 1 if anything fails.
"""
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from entities import EnvConfig, FixedBody, RigidBody  # noqa: E402
from solver import World  # noqa: E402
import solver_jit  # noqa: E402

EnvConfig.backend = "cpu"
WORLD_SEEDS = (1736, 6905, 10012)  # tunnelled through main 0564f6a (found by scanning seeds 0..19999)


# ---------------------------------------------------------------- 1. World level
def _rot(x, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([c * x[0] - s * x[1], s * x[0] + c * x[1]])


def vertex_case(seed):
    """(start, end) of triangle A and fixed triangle B: A's vertex 0 passes through B's vertex 0 at time t*."""
    rng = np.random.default_rng(seed)
    v = np.round(rng.uniform(100, 1000, 2), 0)
    ang, half, L = rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 1.2), rng.uniform(10, 40)
    b1 = v + L * np.array([math.cos(ang - half), math.sin(ang - half)])
    b2 = v + L * np.array([math.cos(ang + half), math.sin(ang + half)])
    d_in = np.array([math.cos(ang), math.sin(ang)])
    t_star = rng.uniform(0.1, 0.9)
    p1 = v + 0.5 * d_in
    p0 = (v - t_star * p1) / (1 - t_star)
    sp = rng.uniform(0.3, 1.0)
    u1 = 8 * np.array([math.cos(ang + math.pi - sp), math.sin(ang + math.pi - sp)])
    u2 = 8 * np.array([math.cos(ang + math.pi + sp), math.sin(ang + math.pi + sp)])
    th = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-9, -5)
    end = np.stack([p1, p1 + u1, p1 + u2], axis=1)
    start = np.stack([p0, p0 + _rot(u1, -th), p0 + _rot(u2, -th)], axis=1)
    return start, end, np.stack([v, b1, b2], axis=1)


def _inside(pt, tri):
    a, b, c = tri.T

    def cr(o, u, w):
        return (u[0] - o[0]) * (w[1] - o[1]) - (u[1] - o[1]) * (w[0] - o[0])
    d = (cr(a, b, pt), cr(b, c, pt), cr(c, a, pt))
    return all(x > 0 for x in d) or all(x < 0 for x in d)


def world_tunnels(seed, h=1 / 240):
    start, end, tri = vertex_case(seed)
    w = World()
    w.add(FixedBody(tri.copy()))
    w.add(RigidBody(start.copy()))
    w.PREV[:] = w.P
    w.V[:] = 0.0
    w.V[:, 3:] = (end - start) / h
    w._resolve_ccd(h)
    P = w.PREV + w.V * h
    return _inside(P[:, 3], P[:, :3])


# ---------------------------------------------------------------- 2. narrow phase vs mpmath
FAMILIES = ("random", "endpoint", "endpoint_rest", "near_linear_endpoint", "far_coords", "grazing", "collapsing",
            "collinear", "resting")


def family(name, rng, n):
    """(p0, dp, a0, da, b0, db), each (2, n): point p and edge (a, b) start positions and substep displacements."""
    p0, a0 = rng.uniform(-20, 20, (2, n)), rng.uniform(-20, 20, (2, n))
    b0 = a0 + rng.uniform(-15, 15, (2, n))
    dp, da, db = (rng.uniform(-10, 10, (2, n)) for _ in range(3))

    def through_end(scale):  # the point's path passes (almost) exactly through an end of the edge at time t
        t, which = rng.uniform(0.05, 0.95, n), rng.integers(0, 2, n)
        end_t = np.where(which, b0 + t * db, a0 + t * da) + rng.normal(0, 1, (2, n)) * 10.0 ** rng.uniform(*scale, n)
        return end_t - t * dp

    if name == "endpoint":
        p0 = through_end((-15, -9))
    elif name == "endpoint_rest":  # resting-scale motion
        dp, da, db = (rng.uniform(-0.05, 0.05, (2, n)) for _ in range(3))
        p0 = through_end((-16, -10))
    elif name in ("near_linear_endpoint", "far_coords"):  # nearly translating edge (c2 << c1)
        da = rng.uniform(-10, 10, (2, n))
        db = da + rng.normal(0, 1, (2, n)) * 10.0 ** rng.uniform(-9, -4, n)
        if name == "far_coords":
            a0, b0 = a0 + 1000.0, b0 + 1000.0
        p0 = through_end((-15, -9) if name == "near_linear_endpoint" else (-12, -8))
    elif name == "grazing":  # tangential touch near mid-edge
        t, s = rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)
        e_t = (b0 + t * db) - (a0 + t * da)
        nrm = np.array([-e_t[1], e_t[0]]) / np.hypot(*e_t)
        dp = (1 - s) * da + s * db + nrm * rng.normal(0, 1, n) * 10.0 ** rng.uniform(-8, -2, n)
        p0 = (a0 + t * da) + s * e_t - t * dp + nrm * 10.0 ** rng.uniform(-12, -6, n) * rng.choice([-1, 1], n)
    elif name == "collapsing":  # the edge shrinks to ~zero length mid-substep, the point is far away
        t = rng.uniform(0.1, 0.9, n)
        b0 = a0 + t * da + rng.normal(0, 1, (2, n)) * 10.0 ** rng.uniform(-14, -4, n) - t * db
        p0 = a0 + rng.uniform(-20, 20, (2, n))
    elif name == "collinear":  # axis-aligned, exactly collinear: the point slides along the edge's line into it
        y = np.round(rng.uniform(-20, 20, n))
        ax = np.round(rng.uniform(-20, 20, n))
        a0 = np.stack([ax, y])
        b0 = np.stack([ax + np.round(rng.uniform(2, 15, n)), y])
        da = db = np.stack([np.round(rng.uniform(-4, 4, n)), np.zeros(n)])
        px = np.where(rng.integers(0, 2, n) == 1, ax - np.round(rng.uniform(1, 6, n)), b0[0] + np.round(rng.uniform(1, 6, n)))
        p0 = np.stack([px, y])
        dp = np.stack([np.round(rng.uniform(-30, 30, n)), np.zeros(n)])
        swap = rng.integers(0, 2, n) == 1  # half of them vertical
        for arr in (p0, dp, a0, da, b0, db):
            arr[:, swap] = arr[::-1, swap]
    elif name == "resting":  # tiny motions, point 0.005..1 px from the edge
        s, e = rng.uniform(-0.1, 1.1, n), b0 - a0
        nrm = np.array([-e[1], e[0]]) / np.hypot(*e)
        p0 = a0 + s * e + nrm * 10.0 ** rng.uniform(-2.3, 0, n) * rng.choice([-1, 1], n)
        dp, da, db = (rng.normal(0, 1, (2, n)) * 10.0 ** rng.uniform(-6, -1, n) for _ in range(3))
    return p0, dp, a0, da, b0, db


def truth(p0, dp, a0, da, b0, db, gap):
    """(touch, ends within gap, min distance) at 50 digits."""
    import mpmath as mp
    mp.mp.dps = 50

    def M(v):
        return [mp.mpf(float(x)) for x in v]
    p0, dp, a0, da, b0, db = map(M, (p0, dp, a0, da, b0, db))
    e0, de = [b0[k] - a0[k] for k in range(2)], [db[k] - da[k] for k in range(2)]
    q0, dq = [p0[k] - a0[k] for k in range(2)], [dp[k] - da[k] for k in range(2)]

    def cr(u, v):
        return u[0] * v[1] - u[1] * v[0]
    c0, c1, c2 = cr(e0, q0), cr(e0, dq) + cr(de, q0), cr(de, dq)

    def segdist(t):
        e = [e0[k] + t * de[k] for k in range(2)]
        q = [q0[k] + t * dq[k] for k in range(2)]
        l2 = e[0] ** 2 + e[1] ** 2
        s = min(max((q[0] * e[0] + q[1] * e[1]) / l2 if l2 > 0 else mp.mpf(0), mp.mpf(0)), mp.mpf(1))
        return mp.sqrt((q[0] - s * e[0]) ** 2 + (q[1] - s * e[1]) ** 2)
    roots = []
    if c2 != 0:
        disc = c1 * c1 - 4 * c2 * c0
        if disc >= 0:
            sq = mp.sqrt(disc)
            roots = [(-c1 - sq) / (2 * c2), (-c1 + sq) / (2 * c2)]
    elif c1 != 0:
        roots = [-c0 / c1]
    elif c0 == 0:  # collinear all substep: the touch starts where the point reaches an end (or sample densely)
        roots = [mp.mpf(k) / 512 for k in range(513)]
        for num, den in ((-(q0[0] * e0[0] + q0[1] * e0[1]), dq[0] * e0[0] + dq[1] * e0[1]),
                         (-((q0[0] - e0[0]) * e0[0] + (q0[1] - e0[1]) * e0[1]),
                          (dq[0] - de[0]) * e0[0] + (dq[1] - de[1]) * e0[1])):
            if den != 0:
                roots.append(num / den)
    touch = any(0 <= r <= 1 and segdist(r) < mp.mpf(10) ** -30 for r in roots)
    ts = [mp.mpf(k) / 40 for k in range(41)] + [r for r in roots if 0 <= r <= 1]
    return touch, segdist(mp.mpf(1)) < gap, float(min(segdist(t) for t in ts))


def stub(X0):
    """A World with just PREV and edges: point 3k against edge (3k+1, 3k+2)."""
    n = X0.shape[1] // 3
    w = World.__new__(World)
    w.backend, w.PREV = None, X0
    w.edges = np.array([np.arange(n) * 3 + 1, np.arange(n) * 3 + 2])
    w._jit_topology = (w.edges, np.ascontiguousarray(w.edges[0]), np.ascontiguousarray(w.edges[1]))
    return w, (np.arange(n) * 3, np.arange(n))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    failed = False
    jit_ok = solver_jit.enabled()
    for jit in ((False, True) if jit_ok else (False,)):
        EnvConfig.jit = jit
        bad = [s for s in WORLD_SEEDS if world_tunnels(s)]
        failed |= bool(bad)
        print(f"world: jit={jit} tunnelled {len(bad)}/{len(WORLD_SEEDS)} {bad}")
    rng = np.random.default_rng(seed)
    print(f"narrow phase vs 50-digit truth, {n} cases per family, seed {seed}")
    for fam in FAMILIES:
        p0, dp, a0, da, b0, db = family(fam, rng, n)
        X0, D = np.empty((2, 3 * n)), np.empty((2, 3 * n))
        X0[:, 0::3], X0[:, 1::3], X0[:, 2::3] = p0, a0, b0
        D[:, 0::3], D[:, 1::3], D[:, 2::3] = dp, da, db
        w, pairs = stub(X0)
        EnvConfig.jit = False
        res = World._ccd_detect(w, D, pairs)
        same = True
        if jit_ok:
            EnvConfig.jit = True
            res_j = World._ccd_detect(w, D, pairs)
            same = all(np.array_equal(x, y) for x, y in zip(res, res_j))
            failed |= not same
        hit = np.zeros(n, bool)
        hit[res[1]] = True
        touches = fn = fp_far = 0
        for k in range(n):
            touch, ends_close, dmin = truth(p0[:, k], dp[:, k], a0[:, k], da[:, k], b0[:, k], db[:, k],
                                            EnvConfig.ccd_gap)
            touches += touch
            fn += touch and not hit[k]
            fp_far += hit[k] and not touch and not ends_close and dmin > 1e-3
        failed |= fn > 0 or fp_far > 0
        print(f"  {fam:22s} touches {touches:4d}  hits {int(hit.sum()):4d}  FN {fn:4d}  far FP {fp_far:3d}  "
              f"numpy==numba {same}")
    print("FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
