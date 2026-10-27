"""Offscreen harness for gl_glass (workstream G): a synthetic G-buffer, material table, lit scene and SDF built on the
CPU from analytic shapes, run through GlassEffect / CausticsEffect on a standalone moderngl context. Run from the repo
root:

    .venv/bin/python3 tests/glass_harness.py [out_dir] [--perf N]

Writes PNGs into out_dir (default <tmp>/glass_out): the scene before / after (no SDF, with SDF, without a floor
texture), 2x crops of the glass, mirror and fluid areas, the fluid filter on / off, caustics. --perf N times N frames
per stage at 900x584. tests/glass_check.py imports the builders for its assertions.

Stand-ins that swap for the real pieces at integration:
- material table: materials.pack (F's real layout)
- G-buffer layout per research/lighting_contract.md (normal.rg = outward bevel normal, b = edge distance px / 255)
- scene_lit: albedo x (ambient + one point light with bevel shading), a stand-in for L's compose
- sdf: exact polygon signed distances near each shape, a bounding-circle lower bound elsewhere (S's sdf texture)
- gbuf["behind"]: the floor without bodies (what F's optional `behind` copy of the base pass would hold)
"""
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moderngl  # noqa: E402

import gl_glass  # noqa: E402
import lights  # noqa: E402
import materials  # noqa: E402
from materials import Material  # noqa: E402

SIZE = (900, 584)
ZOOM = 0.5  # render px per world unit (fluid blur sigma and light positions scale with it)
LENS = Material(kind=materials.GLASS, ior=1.5, transparency=0.9, reflectivity=0.06, specular=0.9, roughness=0.05,
                bevel_px=40.0, profile=materials.DOME, absorption=0.004)
LIGHT_POS, LIGHT_RADIUS, AMBIENT = (600.0, 90.0), 700.0, 0.55


def pack(m):
    return np.asarray(materials.pack(m), np.float32).reshape(materials.TEXELS, 4)


@dataclass
class Shape:
    poly: np.ndarray  # (k, 2) px
    material: Material
    color: tuple  # linear rgb 0..1


def ngon(cx, cy, r, n, angle=0.0):
    a = angle + 2 * np.pi * np.arange(n) / n
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def rounded_rect(cx, cy, hw, hh, r, per=8):
    pts = []
    for (sx, sy), a0 in (((1, 1), 0), ((-1, 1), 90), ((-1, -1), 180), ((1, -1), 270)):
        ox, oy = cx + sx * (hw - r), cy + sy * (hh - r)
        for k in range(per + 1):
            a = math.radians(a0 + 90 * k / per)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return np.array(pts)


def poly_sdf(x, y, poly):
    """Exact signed distance (negative inside) from points (x, y) to a closed polygon."""
    best = np.full(x.shape, np.inf, np.float64)
    inside = np.zeros(x.shape, bool)
    a = poly
    b = np.roll(poly, -1, axis=0)
    for (ax, ay), (bx, by) in zip(a, b):
        ex, ey = bx - ax, by - ay
        t = np.clip(((x - ax) * ex + (y - ay) * ey) / max(ex * ex + ey * ey, 1e-12), 0.0, 1.0)
        dx, dy = x - (ax + t * ex), y - (ay + t * ey)
        best = np.minimum(best, dx * dx + dy * dy)
        cond = (ay > y) != (by > y)
        xint = ax + (y - ay) * (bx - ax) / np.where(by - ay == 0, 1e-12, by - ay)
        inside ^= cond & (x < xint)
    d = np.sqrt(best)
    return np.where(inside, -d, d)


def profile_slope(d, m):
    """dh/dd of the bevel height (height scale = bevel_px): BEVEL 1 inside the bevel, DOME circular, PILLOW smooth."""
    b = max(m.bevel_px, 1e-3)
    t = np.clip(d / b, 0.0, 1.0)
    if m.profile == materials.DOME:
        s = (1 - t) / np.sqrt(np.maximum(1 - (1 - t) ** 2, 1e-3))
    elif m.profile == materials.PILLOW:
        s = 6 * t * (1 - t)
    else:
        s = np.ones_like(t)
    return np.where(d < b, s, 0.0)


def checker(size, cell=16, a=(0.92, 0.9, 0.82), b=(0.25, 0.3, 0.42)):
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    k = ((x // cell) + (y // cell)) % 2
    return np.where(k[..., None] == 0, np.array(a), np.array(b)).astype(np.float64)


def build(shapes, floor, size=SIZE, light=True, margin=48):
    """CPU G-buffer + scene for shapes drawn in order over `floor` (h, w, 3). Returns a dict of numpy arrays."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    albedo = floor.copy()
    nrm = np.zeros((h, w, 2))
    dist = np.zeros((h, w))
    mat = np.zeros((h, w), np.uint16)
    mats = [None]  # id 0: background
    sdf = np.full((h, w), np.inf)
    for s in shapes:  # bounding-circle lower bound everywhere
        c = s.poly.mean(axis=0)
        r = np.sqrt(((s.poly - c) ** 2).sum(axis=1)).max()
        sdf = np.minimum(sdf, np.hypot(xs - c[0], ys - c[1]) - r)
    for s in shapes:
        if s.material not in mats:
            mats.append(s.material)
        mid = mats.index(s.material)
        lo = np.maximum(np.floor(s.poly.min(axis=0) - margin).astype(int), 0)
        hi = np.minimum(np.ceil(s.poly.max(axis=0) + margin).astype(int), [w, h])
        sl = (slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        sd = poly_sdf(xs[sl], ys[sl], s.poly)
        sdf[sl] = np.where(np.isfinite(sd), np.minimum(sdf[sl], sd), sdf[sl])
        ins = sd < 0
        d = np.maximum(-sd, 0.0)
        gy, gx = np.gradient(np.where(ins, d, 0.0))
        gl = np.hypot(gx, gy) + 1e-9
        out = np.stack([-gx / gl, -gy / gl], axis=-1)
        hs = profile_slope(d, s.material) * 1.0
        tilt = (hs / np.sqrt(1 + hs * hs))[..., None]
        sub_n, sub_d, sub_m, sub_a = nrm[sl], dist[sl], mat[sl], albedo[sl]
        sub_n[ins] = (out * tilt)[ins]
        sub_d[ins] = d[ins]
        sub_m[ins] = mid
        col = np.array(s.color)
        m = s.material
        col = col * (1 - m.tint_strength + m.tint_strength * np.array(m.tint))
        sub_a[ins] = col
    # stand-in for L: ambient + a point light with pseudo height, bevel-shaded
    if light:
        lx, ly = LIGHT_POS
        vx, vy, vz = lx - xs, ly - ys, np.full_like(xs, 80.0)
        vl = np.sqrt(vx * vx + vy * vy + vz * vz)
        n2 = nrm
        nz = np.sqrt(np.maximum(1 - (n2 ** 2).sum(-1), 0))
        ndl = np.maximum((n2[..., 0] * vx + n2[..., 1] * vy + nz * vz) / vl, 0)
        fall = np.clip(1 - np.hypot(vx, vy) / LIGHT_RADIUS, 0, 1) ** 2
        lit = AMBIENT + 0.9 * fall * ndl
        scene = np.clip(albedo * lit[..., None], 0, 1)
    else:
        scene = albedo.copy()
    return dict(albedo=albedo, normal=nrm, dist=dist, mat=mat, mats=mats, sdf=sdf.astype(np.float32), scene=scene,
                floor=floor, size=size)


def u8(a):
    return np.clip(np.round(a * 255), 0, 255).astype(np.uint8)


def rgba8(rgb):
    h, w = rgb.shape[:2]
    out = np.full((h, w, 4), 255, np.uint8)
    out[..., :3] = u8(rgb)
    return out


class Harness:
    """Uploads a build() result and runs the effects. gw is a stand-in with only .tex / .size / .ctx (so GlassEffect
    takes the floor from gbuf["behind"] or, without it, inpaints)."""

    def __init__(self, ctx, data, behind=True, sdf=True, period=None, world=None):
        self.ctx, self.data = ctx, data
        size = data["size"]
        tex = lambda arr, comps, dtype="f1": self._tex(arr, size, comps, dtype)
        normal = np.zeros(data["normal"].shape[:2] + (4,), np.uint8)
        normal[..., :2] = u8(data["normal"] * 0.5 + 0.5)
        normal[..., 2] = np.clip(np.round(data["dist"]), 0, 255).astype(np.uint8)
        normal[..., 3] = np.where(data["mat"] > 0, 7, 0)
        self.gbuf = {"albedo": tex(rgba8(data["albedo"]), 4), "normal": tex(normal, 4),
                     "mat": tex(data["mat"].astype(np.uint16), 1, "u2")}
        if behind:
            self.gbuf["behind"] = tex(rgba8(data["floor"]), 4)
        table = np.stack([pack(m) if m is not None else np.zeros((materials.TEXELS, 4), np.float32)
                          for m in data["mats"]])  # (n, 8, 4)
        self.material_tex = ctx.texture((materials.TEXELS, len(data["mats"])), 4, table.astype(np.float32).tobytes(),
                                        dtype="f4")
        self.material_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.scene_bytes = rgba8(data["scene"])
        self.scene_lit = tex(self.scene_bytes, 4)
        self.sdf = tex(data["sdf"], 1, "f4") if sdf else None
        self.gw = SimpleNamespace(ctx=ctx, size=size, tex=tex(self.scene_bytes, 4))
        self.world = world if world is not None else SimpleNamespace(
            lights=[lights.Light(pos=tuple(c / ZOOM for c in LIGHT_POS), radius=LIGHT_RADIUS / ZOOM, intensity=1.0)],
            ambient=(AMBIENT,) * 3)
        self.period = period

    def _tex(self, arr, size, comps, dtype):
        t = self.ctx.texture(size, comps, np.ascontiguousarray(arr).tobytes(), dtype=dtype)
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t

    def frame(self):
        shared = {"scene_lit": self.scene_lit}
        if self.sdf is not None:
            shared["sdf"] = self.sdf
        return SimpleNamespace(size=self.data["size"], clip=self.data["size"], zoom=ZOOM, period=self.period,
                               world=self.world, gbuf=self.gbuf, material_tex=self.material_tex,
                               n_materials=len(self.data["mats"]), shared=shared)

    def reset(self):
        self.gw.tex.write(self.scene_bytes.tobytes())

    def read(self, tex=None):
        t = tex if tex is not None else self.gw.tex
        w, h = t.size
        if t.dtype == "f2":
            return np.frombuffer(t.read(), np.float16).reshape(h, w, t.components).astype(np.float32)
        return np.frombuffer(t.read(), np.uint8).reshape(h, w, t.components)

    def run(self, effects, frame=None):
        self.reset()
        frame = frame or self.frame()
        for e in sorted(effects, key=lambda e: e.order):
            e.run(self.gw, frame)
        return frame


def demo_shapes(seed=0):
    rng = np.random.default_rng(seed)
    P = materials.PRESETS
    shapes = [
        Shape(ngon(130, 150, 62, 64), P["glass"], (0.85, 0.9, 0.95)),
        Shape(ngon(310, 150, 75, 64), LENS, (0.8, 0.9, 1.0)),
        Shape(rounded_rect(490, 150, 55, 55, 18), P["jelly"], (0.35, 0.85, 0.4)),
        Shape(ngon(700, 160, 58, 64), P["mirror"], (0.8, 0.8, 0.85)),
        Shape(np.array([[690, 30], [760, 30], [725, 85]], float), P["matte"], (0.9, 0.2, 0.15)),
        Shape(rounded_rect(830, 260, 30, 30, 4), P["glossy"], (0.95, 0.65, 0.1)),
        Shape(rounded_rect(130, 400, 55, 40, 6), P["matte"], (0.85, 0.15, 0.2)),
        Shape(np.array([[260, 480], [360, 480], [310, 340]], float), P["glossy"], (0.2, 0.45, 0.95)),
    ]
    lo, hi = np.array([18, 60, 130]) / 255, np.array([90, 200, 240]) / 255
    R, r = 9.0, 7.5  # hex packing radius / octagon circumradius: 3 px gaps, like the fluid preset at zoom 0.5
    for row in range(8):
        cy = 560 - R - row * math.sqrt(3) * R
        for k in range(22 - (row % 2)):
            cx = 470 + R + (R if row % 2 else 0) + k * 2 * R
            if row >= 6 and k > 14:
                continue
            shapes.append(Shape(ngon(cx, cy, r * rng.uniform(0.85, 1.0), 8, rng.uniform(0, np.pi / 4)), P["water"],
                                tuple(lo + rng.uniform() * (hi - lo))))
    return shapes


def save_png(path, rgb_u8, scale=1):
    from PIL import Image
    img = Image.fromarray(np.ascontiguousarray(rgb_u8[..., :3]))
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)


def crop(a, x0, y0, x1, y1):
    return a[y0:y1, x0:x1]


def perf(ctx, data, n):
    out = {}
    for label, sdf_on, caust in (("no sdf", False, False), ("sdf trace", True, False), ("sdf + caustics", True, True)):
        h = Harness(ctx, data, sdf=sdf_on)
        caus, glass = gl_glass.make(ctx, data["size"])
        caus.enabled = caust
        caus.profile = glass.profile = True
        acc = {}
        for i in range(n + 2):
            h.run([caus, glass])
            if i < 2:
                continue  # warm-up (shader compiles)
            for k, v in list(glass.timings.items()) + [("caustics " + k, v) for k, v in caus.timings.items()]:
                acc.setdefault(k, []).append(v)
        caus.profile = glass.profile = False
        q, tot = ctx.query(time=True), []
        frame = h.frame()
        for _ in range(n):
            h.reset()
            frame.shared = h.frame().shared
            with q:
                for e in (caus, glass):
                    e.run(h.gw, frame)
            tot.append(q.elapsed / 1e6)
        total = float(np.median(tot))
        out[label] = ({k: float(np.median(v)) for k, v in acc.items()}, total)
        caus.release()
        glass.release()
    return out


def main(argv):
    out_dir = (argv[1] if len(argv) > 1 and not argv[1].startswith("--")
               else os.path.join(tempfile.gettempdir(), "glass_out"))
    n_perf = int(argv[argv.index("--perf") + 1]) if "--perf" in argv else 0
    os.makedirs(out_dir, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=410)
    data = build(demo_shapes(), checker(SIZE))
    pth = lambda name: os.path.join(out_dir, name)
    save_png(pth("scene_before.png"), rgba8(data["scene"]))
    results = {}
    for label, kw, fluid in (("nosdf", dict(sdf=False), True), ("sdf", dict(sdf=True), True),
                             ("inpaint", dict(sdf=False, behind=False), True), ("nofluid", dict(sdf=False), False)):
        h = Harness(ctx, data, **kw)
        caus, glass = gl_glass.make(ctx, SIZE)
        glass.fluid_filter = fluid
        h.run([caus, glass])
        results[label] = h.read()
        save_png(pth(f"after_{label}.png"), results[label])
        caus.release()
        glass.release()
    before = rgba8(data["scene"])
    for name, box in (("glass", (60, 70, 580, 230)), ("mirror", (620, 20, 880, 300)),
                      ("fluid", (460, 400, 880, 584))):
        save_png(pth(f"crop_{name}_before.png"), crop(before, *box), 2)
        for label in ("nosdf", "sdf") if name != "fluid" else ("nofluid", "nosdf"):
            save_png(pth(f"crop_{name}_{label}.png"), crop(results[label], *box), 2)
    # caustics: a point light left of the glass; visualised as scene + albedo x caustics (what L's compose would add)
    cshapes = [Shape(ngon(330, 290, 70, 64), replace(LENS, bevel_px=70.0), (0.85, 0.92, 1.0)),
               Shape(rounded_rect(700, 290, 18, 120, 6), materials.PRESETS["mirror"], (0.8, 0.8, 0.85))]
    cdata = build(cshapes, checker(SIZE, 32, (0.5, 0.5, 0.5), (0.42, 0.42, 0.42)), light=False)
    cdata["scene"] = cdata["albedo"] * 0.45
    world = SimpleNamespace(lights=[lights.Light(pos=(90.0 / ZOOM, 290.0 / ZOOM), radius=900.0 / ZOOM, intensity=1.0,
                                                 falloff=1.0)])
    h = Harness(ctx, cdata, world=world)
    caus, glass = gl_glass.make(ctx, SIZE)
    caus.enabled = True
    frame = h.run([caus], h.frame())
    c = h.read(frame.shared["caustics"])[..., :3]
    vis = np.clip(cdata["scene"] + cdata["albedo"] * c * 0.5, 0, 1)
    save_png(pth("caustics.png"), u8(vis))
    save_png(pth("caustics_raw.png"), u8(np.clip(c * 0.25, 0, 1)))
    caus.release()
    glass.release()
    print("wrote", out_dir)
    if n_perf:
        for label, (stages, total) in perf(ctx, data, n_perf).items():
            print(f"{label:16s} GPU total {total:5.2f} ms/frame (median); stages:  " +
                  "  ".join(f"{k} {v:.2f}" for k, v in stages.items()))


if __name__ == "__main__":
    main(sys.argv)
