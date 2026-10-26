"""Apple GPU backend (EnvConfig.backend = "gpu") using MLX custom Metal kernels.

Metal GPUs have no float64, so the work is split by what float32 / int32 can do without weakening the
no-intersection guarantee:

GPU:
- Broad phases: contact candidates always, CCD pairs only with GpuBackend.gpu_ccd_broadphase. Cell ranges are still computed on the CPU in float64 (O(points),
  the same floor() as World._candidates / World._ccd_cells), so the GPU only does integer work: a hash grid over
  point cells (count, prefix sum, scatter) and one thread per edge that walks its cells and emits the (point, edge)
  pairs whose cell ranges overlap, with World._filter_pairs' body rules. The pair set is exactly the CPU one.
- PBD iterations: spring Jacobi solves (XPBD with the multipliers kept on the GPU across a substep's passes, or PBD,
  as EnvConfig.xpbd), rigid shape matching and proximity contacts (float32, atomic accumulation).
  They run on substep-local displacements x = P - PREV, so rounding is relative to the motion, not to world
  coordinates; differences are formed as (PREV_j - PREV_i) + (x_j - x_i). The result is only a proposed motion.

CPU, float64, unchanged World code: predict / drag (O(points)), the exact CCD narrow phase and its pre-reject,
Gauss-Seidel impulses and rigid impact zones. The guarantee rests on those plus a broad phase that never drops a
pair, which holds because the GPU pair set equals the CPU one.

Scenes with ropes, or with bodies that override predict / solve_constraints, run their PBD part on the CPU
(pbd_substep returns False).

GPU time goes to world.profiler as gpu-tagged sections that end in a device sync (see profiler.py).
"""
from contextlib import nullcontext

import numpy as np

import mlx.core as mx

from entities import EnvConfig

_TG = 256  # threads per threadgroup
_NO_SECTION = nullcontext()

_HEADER = """
inline uint cell_hash(int cx, int cy, uint mask) {
    return ((uint(cx) * 73856093u) ^ (uint(cy) * 19349663u)) & mask;
}
inline int pmod(int a, int n) { int r = a % n; return r < 0 ? r + n : r; }
"""

# ---- hash grid over point cell ranges. params: [N, mask]
_GRID_COUNT = """
    uint i = thread_position_in_grid.x;
    int N = params[0]; uint mask = uint(params[1]);
    if (i >= uint(N)) return;
    for (int cx = plo[i]; cx <= phi[i]; cx++)
        for (int cy = plo[N + i]; cy <= phi[N + i]; cy++)
            atomic_fetch_add_explicit(&bcount[cell_hash(cx, cy, mask)], 1, memory_order_relaxed);
"""

_GRID_FILL = """
    uint i = thread_position_in_grid.x;
    int N = params[0]; uint mask = uint(params[1]);
    if (i >= uint(N)) return;
    for (int cx = plo[i]; cx <= phi[i]; cx++)
        for (int cy = plo[N + i]; cy <= phi[N + i]; cy++) {
            uint b = cell_hash(cx, cy, mask);
            int s = bstart[b] + atomic_fetch_add_explicit(&cursor[b], 1, memory_order_relaxed);
            atomic_store_explicit(&slot_pt[s], int(i), memory_order_relaxed);
            atomic_store_explicit(&slot_cx[s], cx, memory_order_relaxed);
            atomic_store_explicit(&slot_cy[s], cy, memory_order_relaxed);
        }
"""

# ---- pairs: one thread per edge. params: [E, N, mask, cap, mode (0 = contact rules, 1 = CCD rules)]
_PAIRS = """
    uint e = thread_position_in_grid.x;
    int E = params[0], N = params[1]; uint mask = uint(params[2]); int cap = params[3], mode = params[4];
    if (e >= uint(E)) return;
    int ia = edges[e], ib = edges[E + e], eb = edge_body[e];
    int ex0 = elo[e], ex1 = ehi[e], ey0 = elo[E + e], ey1 = ehi[E + e];
    for (int cx = ex0; cx <= ex1; cx++)
        for (int cy = ey0; cy <= ey1; cy++) {
            uint b = cell_hash(cx, cy, mask);
            int s1 = bstart[b] + bcount[b];
            for (int k = bstart[b]; k < s1; k++) {
                if (slot_cx[k] != cx || slot_cy[k] != cy) continue;  // hash collision with another cell
                int i = slot_pt[k];
                if (cx != max(plo[i], ex0) || cy != max(plo[N + i], ey0)) continue;  // report once, at low corner
                int pb = point_body[i];
                bool ok;
                if (pb != eb) ok = !(body_fixed[pb] && body_fixed[eb]);
                else if (!body_self[pb]) ok = false;
                else if (mode == 0) {  // contacts: skip the point's own and its neighbors' edges
                    int n = body_counts[pb], base = offsets[pb], li = i - base;
                    int va = ia - base, vb = ib - base;
                    int ga = min(pmod(li - va, n), pmod(va - li, n)), gb = min(pmod(li - vb, n), pmod(vb - li, n));
                    ok = ga > 1 && gb > 1;
                } else ok = (i != ia && i != ib);  // CCD: skip only the point's own edges
                if (!ok) continue;
                int slot = atomic_fetch_add_explicit(&npairs[0], 1, memory_order_relaxed);
                if (slot < cap) {
                    atomic_store_explicit(&out_pt[slot], i, memory_order_relaxed);
                    atomic_store_explicit(&out_e[slot], int(e), memory_order_relaxed);
                }
            }
        }
"""

# ---- constraints, accumulate: threads [0, S) are springs, [S, S + R) are rigid-body points.
# params: [N, S, R, xpbd], fparams: [h, spring_damping]. xpbd = 1: XPBD springs as World._xpbd_springs (mass-splitting
# Jacobi, Rayleigh damping), multipliers read from lam and written to lamout; 0: PBD springs (averaged in con_apply)
_CON_ACC = """
    uint t = thread_position_in_grid.x;
    int N = params[0], S = params[1], R = params[2];
    if (t < uint(S) && params[3] != 0) {
        int i = si[t], j = sj[t];
        float2 d = float2(prev[j] - prev[i], prev[N + j] - prev[N + i]) + float2(x[j] - x[i], x[N + j] - x[N + i]);
        float dist = max(length(d), 1e-9f);
        float2 n = d / dist;
        float wi = w[i], wj = w[j];
        float h = fparams[0];
        float at = scomp[t] * (wi + wj) / (h * h);
        float gamma = at * fparams[1] * h;
        float rel = dot(n, float2(x[j] - x[i], x[N + j] - x[N + i]));  // x = motion since the substep's start
        float den = (1.0f + gamma) * (scount[i] * wi + scount[j] * wj) + at;
        float dl = den > 0.0f ? (-(dist - rest_len[t]) - at * lam[t] - gamma * rel) / den : 0.0f;
        atomic_store_explicit(&lamout[t], lam[t] + dl, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[i], -wi * dl * n.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[N + i], -wi * dl * n.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[j], wj * dl * n.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[N + j], wj * dl * n.y, memory_order_relaxed);
    } else if (t < uint(S)) {
        int i = si[t], j = sj[t];
        float2 d = float2(prev[j] - prev[i], prev[N + j] - prev[N + i]) + float2(x[j] - x[i], x[N + j] - x[N + i]);
        float dist = max(length(d), 1e-9f);
        float wi = w[i], wj = w[j];
        float2 corr = sstiff[t] * (dist - rest_len[t]) / (dist * (wi + wj)) * d;
        atomic_fetch_add_explicit(&acc[i], wi * corr.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[N + i], wi * corr.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[j], -wj * corr.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&acc[N + j], -wj * corr.y, memory_order_relaxed);
    } else if (t < uint(S + R)) {
        int r = int(t) - S, g = rpt[r], k = rbody[r], an = ranchor[k];
        float2 q = float2(prev[g] - prev[an], prev[N + g] - prev[N + an]) + float2(x[g] - x[an], x[N + g] - x[N + an]);
        float rx = rrest[r], ry = rrest[R + r];
        atomic_fetch_add_explicit(&rsum[6 * k + 0], q.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&rsum[6 * k + 1], q.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&rsum[6 * k + 2], q.x * rx, memory_order_relaxed);
        atomic_fetch_add_explicit(&rsum[6 * k + 3], q.x * ry, memory_order_relaxed);
        atomic_fetch_add_explicit(&rsum[6 * k + 4], q.y * rx, memory_order_relaxed);
        atomic_fetch_add_explicit(&rsum[6 * k + 5], q.y * ry, memory_order_relaxed);
    }
"""

# ---- constraints, apply: one thread per point. kind 2 = rigid (move toward goal), else the springs' summed XPBD
# corrections (params[3] = 1) or their Jacobi average (PBD; acc is 0 for points without springs). Rigid covariance from raw sums: sum (q - com) rest^T = sum q rest^T -
# com (sum rest)^T, with q relative to the body's first point so the numbers stay small.
_CON_APPLY = """
    uint i = thread_position_in_grid.x;
    int N = params[0], R = params[2];
    if (i >= uint(N)) return;
    float2 xi = float2(x[i], x[N + i]);
    if (kinds[i] == 2) {
        int r = rslot[i], k = rbody[r], an = ranchor[k];
        float n = rn[k];
        float2 com = float2(rsum[6 * k], rsum[6 * k + 1]) / n;
        float sx = rrestsum[2 * k], sy = rrestsum[2 * k + 1];
        float a00 = rsum[6 * k + 2] - com.x * sx, a01 = rsum[6 * k + 3] - com.x * sy;
        float a10 = rsum[6 * k + 4] - com.y * sx, a11 = rsum[6 * k + 5] - com.y * sy;
        float th = atan2(a10 - a01, a00 + a11);
        float c = cos(th), s = sin(th);
        float rx = rrest[r], ry = rrest[R + r];
        float2 goal = float2(c * rx - s * ry, s * rx + c * ry) + com;
        float2 q = float2(prev[i] - prev[an], prev[N + i] - prev[N + an]) + float2(xi.x - x[an], xi.y - x[N + an]);
        xi += rstiff[k] * (goal - q);
    } else {
        xi += float2(acc[i], acc[N + i]) / (params[3] != 0 ? 1.0f : scount[i]);
    }
    xout[i] = xi.x;
    xout[N + i] = xi.y;
"""

# ---- contacts, accumulate: one thread per candidate pair, same math as World._solve_contacts.
# params: [N, cap], fparams: [radius, friction]
_CONTACT_ACC = """
    uint t = thread_position_in_grid.x;
    int N = params[0], cap = params[1];
    if (int(t) >= min(npairs[0], cap)) return;
    int ip = cpt[t], ed = ce[t];
    int ia = edges_a[ed], ib = edges_b[ed];
    float r = fparams[0], mu = fparams[1];
    float2 e0 = float2(prev[ib] - prev[ia], prev[N + ib] - prev[N + ia]);
    float2 q0 = float2(prev[ip] - prev[ia], prev[N + ip] - prev[N + ia]);
    float2 pa = q0 + float2(x[ip] - x[ia], x[N + ip] - x[N + ia]);
    float2 e = e0 + float2(x[ib] - x[ia], x[N + ib] - x[N + ia]);
    float len2 = max(dot(e, e), 1e-12f);
    float tt = clamp(dot(pa, e) / len2, 0.0f, 1.0f);
    float2 d = pa - tt * e;
    float dist = length(d);
    if (!(dist < r)) return;
    // edge normal facing the side the point started on: the substep start (sref = 0), or inside a multi-substep CCD
    // span the span start (sref = span start - prev), the last CCD-checked state
    float2 se = e0 + float2(sref[ib] - sref[ia], sref[N + ib] - sref[N + ia]);
    float2 sq = q0 + float2(sref[ip] - sref[ia], sref[N + ip] - sref[N + ia]);
    float cross0 = se.x * sq.y - se.y * sq.x;
    float side = cross0 != 0.0f ? sign(cross0) : -esign[ed];
    float2 n_edge = -side * float2(e.y, -e.x) / sqrt(len2);
    bool inside = dot(d, n_edge) < 0.0f;
    float2 n = (inside || dist < 1e-9f) ? n_edge : d / max(dist, 1e-9f);
    float depth = r - dot(d, n);
    float wp = w[ip], wa = w[ia], wb = w[ib];
    float denom = wp + wa * (1.0f - tt) * (1.0f - tt) + wb * tt * tt;
    if (!(denom > 1e-12f)) return;
    float2 xa = float2(x[ia], x[N + ia]);
    float2 rel = float2(x[ip], x[N + ip]) - (xa + tt * (float2(x[ib], x[N + ib]) - xa));
    float2 tang = rel - dot(rel, n) * n;
    float fric = min(1.0f, mu * depth / max(length(tang), 1e-9f));
    float2 lam = (depth * n - tang * fric) / denom;
    float2 dp = wp * lam, da = -wa * (1.0f - tt) * lam, db = -wb * tt * lam;
    atomic_fetch_add_explicit(&acc[ip], dp.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc[N + ip], dp.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc[ia], da.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc[N + ia], da.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc[ib], db.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&acc[N + ib], db.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&cnt[ip], 1.0f, memory_order_relaxed);
    atomic_fetch_add_explicit(&cnt[ia], 1.0f, memory_order_relaxed);
    atomic_fetch_add_explicit(&cnt[ib], 1.0f, memory_order_relaxed);
"""

_CONTACT_APPLY = """
    uint i = thread_position_in_grid.x;
    int N = params[0];
    if (i >= uint(N)) return;
    float c = max(cnt[i], 1.0f);
    xout[i] = x[i] + acc[i] / c;
    xout[N + i] = x[N + i] + acc[N + i] / c;
"""

_K = {}


def _kernels():
    if not _K:
        def k(name, inputs, outputs, source, atomic):
            _K[name] = mx.fast.metal_kernel(name="sb_" + name, input_names=inputs, output_names=outputs,
                                            source=source, header=_HEADER, atomic_outputs=atomic)
        k("grid_count", ["plo", "phi", "params"], ["bcount"], _GRID_COUNT, True)
        k("grid_fill", ["plo", "phi", "bstart", "params"], ["cursor", "slot_pt", "slot_cx", "slot_cy"], _GRID_FILL, True)
        k("pairs", ["elo", "ehi", "plo", "bstart", "bcount", "slot_pt", "slot_cx", "slot_cy", "edges", "edge_body",
                    "point_body", "body_fixed", "body_self", "body_counts", "offsets", "params"],
          ["npairs", "out_pt", "out_e"], _PAIRS, True)
        k("con_acc", ["prev", "x", "w", "si", "sj", "rest_len", "sstiff", "scomp", "scount", "lam", "rpt", "rbody",
                      "ranchor", "rrest", "params", "fparams"], ["acc", "rsum", "lamout"], _CON_ACC, True)
        k("con_apply", ["prev", "x", "acc", "scount", "kinds", "rslot", "rbody", "ranchor", "rn", "rsum", "rrest",
                        "rrestsum", "rstiff", "params"], ["xout"], _CON_APPLY, False)
        k("contact_acc", ["prev", "sref", "x", "w", "cpt", "ce", "npairs", "edges_a", "edges_b", "esign", "params", "fparams"],
          ["acc", "cnt"], _CONTACT_ACC, True)
        k("contact_apply", ["x", "acc", "cnt", "params"], ["xout"], _CONTACT_APPLY, False)
    return _K


def _grid(n):
    return (max(int(n), 1), 1, 1)


def _i32(a):
    return mx.array(np.ascontiguousarray(a, dtype=np.int32))


def _f32(a):
    return mx.array(np.ascontiguousarray(a, dtype=np.float32))


def _pow2(n):
    return 1 << max(10, int(np.ceil(np.log2(max(n, 1)))))


class GpuTopology:
    """Static per-world arrays on the GPU, from the World's packed arrays (body membership, springs, rigid shapes)."""

    def __init__(self, world):
        n_pts = world.P.shape[1]
        self.n_pts, self.n_edges = n_pts, world.edges.shape[1]
        self.edges = _i32(world.edges.ravel())
        self.edges_a, self.edges_b = _i32(world.edges[0]), _i32(world.edges[1])
        self.edge_body, self.point_body = _i32(world.edge_body), _i32(world.point_body)
        self.body_fixed, self.body_self = _i32(world.body_fixed), _i32(world.body_self)
        self.body_counts, self.offsets = _i32(world.body_counts), _i32(world.offsets[:-1])
        self.esign = _f32(world.edge_sign)

        self.n_springs = world.springs.shape[1]
        self.si, self.sj = _i32(world.springs[0]), _i32(world.springs[1])
        self.rest_len, self.sstiff = _f32(world.spring_rest), _f32(world.spring_stiffness)
        self.scount = _f32(world.spring_count)
        self.scomp = _f32(world.spring_comp)  # XPBD compliance per unit inverse mass (solver.spring_compliance)

        kinds = np.zeros(n_pts, dtype=np.int32)
        rslot = np.full(n_pts, -1, dtype=np.int32)
        rpt, rbody, rrest, ranchor, rn, rstiff, rrestsum = [], [], [], [], [], [], []
        for idx, rest, stiffness in world.rigid_groups:  # idx (B, n), rest (B, 2, n), stiffness (B,)
            for b in range(idx.shape[0]):
                k = len(ranchor)
                rslot[idx[b]] = sum(map(len, rpt)) + np.arange(idx.shape[1])
                kinds[idx[b]] = 2
                rpt.append(idx[b])
                rbody.append(np.full(idx.shape[1], k))
                rrest.append(rest[b])
                ranchor.append(idx[b, 0])
                rn.append(idx.shape[1])
                rstiff.append(stiffness[b])
                rrestsum.append(rest[b].astype(np.float32).sum(axis=1, dtype=np.float32))
        self.n_rigid = len(ranchor)
        self.n_rigid_pts = int(sum(map(len, rpt)))
        if rpt:
            self.rpt, self.rbody = _i32(np.concatenate(rpt)), _i32(np.concatenate(rbody))
            self.rrest = _f32(np.concatenate(rrest, axis=1).ravel())
            self.ranchor, self.rn, self.rstiff = _i32(ranchor), _f32(rn), _f32(rstiff)
            self.rrestsum = _f32(np.concatenate(rrestsum))
        else:
            zi, zf = mx.zeros((1,), dtype=mx.int32), mx.zeros((1,), dtype=mx.float32)
            self.rpt = self.rbody = self.ranchor = zi
            self.rrest = self.rn = self.rstiff = self.rrestsum = zf
        self.rslot, self.kinds = _i32(rslot), _i32(kinds)


class GpuBackend:
    """Hooks called by World when EnvConfig.backend == "gpu": pbd_substep(h) and ccd_pairs(cells)."""

    # The CCD broad phase on the GPU gives the same pairs, but measured slower than World's vectorized CPU version
    # at every size tried (one extra device sync per call, plus pair download and sort), so it's off by default.
    gpu_ccd_broadphase = False

    def __init__(self, world):
        self.world = world
        self.gtopo = None  # built on first use (World rebuilds the backend whenever a body is added)
        self.contact_cap = 4096  # pair buffer sizes; grown (and the work redone) when a dispatch overflows them
        self.ccd_cap = 4096
        self.stats = {"pbd_retries": 0, "ccd_retries": 0, "cpu_pbd_substeps": 0}

    def _topology(self):
        if self.gtopo is None:
            self.gtopo = GpuTopology(self.world)
        return self.gtopo

    def _gpu_section(self, name):
        prof = self.world.profiler
        return _NO_SECTION if prof is None else prof.section(name, device="gpu")

    def _pairs(self, plo, phi, elo, ehi, mode, cap):
        """Lazy MLX graph for the (point, edge) pairs whose integer cell ranges overlap: (npairs, out_pt, out_e)."""
        g, K = self._topology(), _kernels()
        N, E = g.n_pts, g.n_edges
        entries = int(np.sum((phi[0] - plo[0] + 1) * (phi[1] - plo[1] + 1)))
        T = _pow2(2 * entries)  # hash table buckets
        params = _i32([N, T - 1])
        plo_g, phi_g = _i32(plo.ravel()), _i32(phi.ravel())
        tg = (_TG, 1, 1)
        bcount, = K["grid_count"](inputs=[plo_g, phi_g, params], grid=_grid(N), threadgroup=tg,
                                  output_shapes=[(T,)], output_dtypes=[mx.int32], init_value=0)
        bstart = mx.cumsum(bcount) - bcount
        _, slot_pt, slot_cx, slot_cy = K["grid_fill"](
            inputs=[plo_g, phi_g, bstart, params], grid=_grid(N), threadgroup=tg,
            output_shapes=[(T,), (entries,), (entries,), (entries,)], output_dtypes=[mx.int32] * 4, init_value=0)
        return K["pairs"](
            inputs=[_i32(elo.ravel()), _i32(ehi.ravel()), plo_g, bstart, bcount, slot_pt, slot_cx, slot_cy,
                    g.edges, g.edge_body, g.point_body, g.body_fixed, g.body_self, g.body_counts, g.offsets,
                    _i32([E, N, T - 1, cap, mode])],
            grid=_grid(E), threadgroup=tg, output_shapes=[(1,), (cap,), (cap,)], output_dtypes=[mx.int32] * 3,
            init_value=0)

    # ---------------- CCD broad phase
    def ccd_pairs(self, cells):
        """World._ccd_pairs: same pairs, same (point, edge) order. cells = World._ccd_cells(D) (float64 floors).
        None when gpu_ccd_broadphase is off (World then uses its CPU code)."""
        w = self.world
        n_edges = w.edges.shape[1]
        if not self.gpu_ccd_broadphase or n_edges == 0:
            return None
        plo, phi, elo, ehi = cells
        with self._gpu_section("gpu pairs"):
            while True:
                npairs, out_pt, out_e = self._pairs(plo, phi, elo, ehi, 1, self.ccd_cap)
                mx.eval(npairs, out_pt, out_e)
                n = int(npairs.item())
                if n <= self.ccd_cap:
                    break
                self.stats["ccd_retries"] += 1
                self.ccd_cap = _pow2(int(n * 1.5))
        code = np.sort(np.array(out_pt[:n]).astype(np.int64) * n_edges + np.array(out_e[:n]))
        return np.divmod(code, n_edges)

    # ---------------- PBD part of a substep
    def pbd_substep(self, h):
        """World._pbd_substep with the candidates and iterations on the GPU. False if this scene needs the CPU."""
        w = self.world
        if w.ropes or w.solo_bodies or not w.batched_predict or w.edges.shape[1] == 0:
            self.stats["cpu_pbd_substeps"] += 1
            return False
        sec = w._section
        with sec("predict+grab"):  # Body.predict for every body, as World._pbd_substep (float64)
            w.V += EnvConfig.gravity[:, None] * h * (w.W > 0)
            w.PREV[:] = w.P
            w.P += w.V * h
            w._apply_grab(h)
        g, K = self._topology(), _kernels()
        N = g.n_pts
        cell, pad = EnvConfig.cell_size, 2 * EnvConfig.contact_radius
        span = w._span_contacts  # inside a multi-substep CCD span: (GPU pairs or None, point box lo, hi)
        with sec("upload"):
            if w._span_start is None:
                # contact candidate cell ranges, exactly as World._candidates computes them
                a, b = w.P[:, w.edges[0]], w.P[:, w.edges[1]]
                elo = np.floor((np.minimum(a, b) - pad) / cell).astype(np.int64)
                ehi = np.floor((np.maximum(a, b) + pad) / cell).astype(np.int64)
                pc = np.floor(w.P / cell).astype(np.int64)
                cells = (pc, pc, elo, ehi)
                w.contact_builds += 1
                sref = mx.zeros((2 * N,), dtype=mx.float32)
            else:
                # World._span_candidates on the GPU: the span's swept-box pairs, rebuilt when a point leaves its box
                if span is None or (w.P < span[1]).any() or (w.P > span[2]).any():
                    boxes = w._span_boxes()
                    cells = [np.floor(box / cell).astype(np.int64) for box in boxes]
                    w._span_contacts = span = (None, boxes[0], boxes[1])
                    w.contact_builds += 1
                sref = _f32((w._span_start - w.PREV).ravel())
            prev, inv_mass = _f32(w.PREV.ravel()), _f32(w.W)  # W re-sent each substep in case it was edited
            x0 = _f32((w.P - w.PREV).ravel())
        S, R = g.n_springs, g.n_rigid_pts
        cparams = _i32([N, S, R, 1 if EnvConfig.xpbd else 0])
        fparams = _f32([EnvConfig.contact_radius, EnvConfig.friction])
        sparams = _f32([h, EnvConfig.spring_damping])
        tg = (_TG, 1, 1)
        with self._gpu_section("gpu pbd"):
            while True:
                cap = self.contact_cap
                if span is not None and span[0] is not None:  # pairs this span already built (and fit in cap)
                    npairs, cpt, ce = span[0]
                else:
                    npairs, cpt, ce = self._pairs(*cells, 0, cap)
                kparams = _i32([N, cap])
                x = x0
                lam = mx.zeros((max(S, 1),), dtype=mx.float32)  # XPBD multipliers, fresh every substep
                for _ in range(EnvConfig.iterations):
                    if S + R:
                        acc, rsum, lam = K["con_acc"](
                            inputs=[prev, x, inv_mass, g.si, g.sj, g.rest_len, g.sstiff, g.scomp, g.scount, lam, g.rpt,
                                    g.rbody, g.ranchor, g.rrest, cparams, sparams],
                            grid=_grid(S + R), threadgroup=tg,
                            output_shapes=[(2 * N,), (6 * max(g.n_rigid, 1),), (max(S, 1),)],
                            output_dtypes=[mx.float32] * 3, init_value=0)
                        x, = K["con_apply"](
                            inputs=[prev, x, acc, g.scount, g.kinds, g.rslot, g.rbody, g.ranchor, g.rn, rsum,
                                    g.rrest, g.rrestsum, g.rstiff, cparams],
                            grid=_grid(N), threadgroup=tg, output_shapes=[(2 * N,)], output_dtypes=[mx.float32])
                    acc, cnt = K["contact_acc"](
                        inputs=[prev, sref, x, inv_mass, cpt, ce, npairs, g.edges_a, g.edges_b, g.esign, kparams, fparams],
                        grid=_grid(cap), threadgroup=tg, output_shapes=[(2 * N,), (N,)],
                        output_dtypes=[mx.float32] * 2, init_value=0)
                    x, = K["contact_apply"](inputs=[x, acc, cnt, kparams], grid=_grid(N), threadgroup=tg,
                                            output_shapes=[(2 * N,)], output_dtypes=[mx.float32])
                mx.eval(x, npairs)
                n = int(npairs.item())
                if n <= cap:
                    if span is not None:
                        w._span_contacts = (npairs, cpt, ce), span[1], span[2]
                    break
                self.stats["pbd_retries"] += 1
                self.contact_cap = _pow2(int(n * 1.5))
        # P = PREV + x in float64; points nothing moved (fixed bodies) get x == 0 exactly, so they stay bit-exact
        w.P[:] = w.PREV + np.array(x).astype(np.float64).reshape(2, N)
        return True
