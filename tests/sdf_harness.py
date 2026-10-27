"""Offscreen harness for gl_sdf (SDF, soft shadows, radiance cascades GI) on a synthetic G-buffer. From the repo root:

    .venv/bin/python3 tests/sdf_harness.py [out_dir]

Builds a G-buffer per research/lighting_contract.md (albedo RGBA8, normal RGBA8 with flags in alpha, mat R16UI) from
a few polygons rasterised at pixel centres, runs gl_sdf's effects through a minimal FrameInfo stand-in, and writes
PNGs to out_dir (default <tmp>/sdf_harness, outside the repo): the SDF, a point light with soft shadows at two softness values, a periodic
case with an occluder across the seam, and GI from a glowing polygon. Prints GPU timings (GL time queries).

Until workstream F's shaders/material.glsl and materials.pack exist, the material table uses
tests/sdf_stub_material.glsl's provisional packing. tests/sdf_check.py imports the helpers here.
"""
import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import moderngl  # noqa: E402

import gl_sdf  # noqa: E402
import materials  # noqa: E402

SHADERS = os.path.join(ROOT, "shaders")
TESTS = os.path.join(ROOT, "tests")
SIZE = (900, 584)
OCCLUDER, CASTS, LIT = 1, 2, 4
SOLID = OCCLUDER | CASTS | LIT
HAVE_F_MATERIALS = os.path.exists(os.path.join(SHADERS, "material.glsl")) and hasattr(materials, "pack")


def _read(name):
    if name in _EXTRA:  # the harness-only lighting shaders below
        return _EXTRA[name]
    for d, n in ((SHADERS, name), (TESTS, "sdf_stub_" + name)):
        if os.path.exists(os.path.join(d, n)):
            with open(os.path.join(d, n)) as f:
                return f.read()
    raise FileNotFoundError(name)


@dataclass
class Frame:
    """A minimal stand-in for gl_effects.FrameInfo (the fields gl_sdf uses, plus the rest of the contract's)."""
    ctx: object
    size: tuple
    gbuf: dict
    material_tex: object
    n_materials: int = 1
    period: tuple = None
    clip: tuple = None
    zoom: float = 1.0
    world: object = None
    shared: dict = field(default_factory=dict)

    def __post_init__(self):
        self._vao = {}

    def load(self, vert, frag, includes=()):
        """Like gl_world._load: '#version 330 core' + world_common.glsl (so name clashes with it show up here) +
        includes + the file."""
        common = _read("world_common.glsl")
        head = "#version 330 core\n"
        inc = "".join(_read(n) + "\n" for n in includes)
        return self.ctx.program(vertex_shader=head + "#define VERTEX_SHADER\n" + common + "\n" + _read(vert),
                                fragment_shader=head + common + "\n" + inc + _read(frag))

    def fullscreen(self, prog):
        vao = self._vao.get(prog.glo)
        if vao is None:
            vao = self._vao[prog.glo] = self.ctx.vertex_array(prog, [])
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4)


def regular(cx, cy, r, n, rot=0.0):
    a = rot + np.arange(n) * 2 * np.pi / n
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def raster(pts, size):
    """(h, w) bool: pixel centres inside the polygon (even-odd)."""
    w, h = size
    x = np.arange(w) + 0.5
    y = (np.arange(h) + 0.5)[:, None]
    inside = np.zeros((h, w), bool)
    q = np.roll(pts, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(pts, q):
        if y0 == y1:
            continue
        cond = (y >= min(y0, y1)) & (y < max(y0, y1))
        xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (x < xc)
    return inside


def poly_sdf(pts, px, py):
    """Signed distance from points (px, py) to the polygon outline, negative inside (even-odd)."""
    d = np.full(px.shape, np.inf)
    inside = np.zeros(px.shape, bool)
    q = np.roll(pts, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(pts, q):
        ex, ey = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * ex + (py - y0) * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(px - x0 - t * ex, py - y0 - t * ey))
        if y0 != y1:
            cond = (py >= min(y0, y1)) & (py < max(y0, y1))
            inside ^= cond & (px < x0 + (py - y0) * ex / ey)
    return np.where(inside, -d, d)


@dataclass
class Poly:
    pts: np.ndarray
    color: tuple  # 0..1 RGB
    flags: int = SOLID
    mat: int = 1  # 1 plain, 2 glow (emissive)


def scene_default():
    return [
        Poly(regular(430, 300, 45, 4, np.pi / 4), (0.85, 0.3, 0.25)),  # a square
        Poly(regular(600, 200, 40, 3, -np.pi / 2), (0.3, 0.75, 0.35)),  # a triangle
        Poly(np.array([[250, 380], [520, 420], [515, 440], [245, 400]], float), (0.3, 0.45, 0.85)),  # a thin bar
        Poly(regular(680, 420, 50, 32), (0.85, 0.75, 0.3)),  # a disc
        Poly(regular(180, 200, 30, 6), (1.0, 0.6, 0.2), flags=OCCLUDER | LIT, mat=2),  # glow: no shadow
    ]


def material_table(ctx):
    """Row 0 unused, 1 matte, 2 glow (emissive)."""
    mats = [materials.Material(), materials.PRESETS["matte"], materials.PRESETS["glow"]]
    if HAVE_F_MATERIALS:
        rows = np.array([np.asarray(materials.pack(m), np.float32).reshape(8, 4) for m in mats])
    else:
        rows = np.zeros((len(mats), 8, 4), np.float32)
        for i, m in enumerate(mats):
            rows[i, 1] = (*m.emissive, m.emissive_strength)
    tex = ctx.texture((8, len(mats)), 4, rows.tobytes(), dtype="f4")
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return tex, len(mats)


def gbuffer(ctx, polys, size=SIZE, period=None, background=(0.45, 0.45, 0.5)):
    """The G-buffer textures (+ numpy copies) for polygons, each also drawn one period over across a seam."""
    w, h = size
    albedo = np.zeros((h, w, 4), np.uint8)
    albedo[..., :3] = np.round(np.array(background) * 255)
    albedo[..., 3] = 255
    normal = np.zeros((h, w, 4), np.uint8)
    normal[..., :2] = 128
    mat = np.zeros((h, w), np.uint16)
    for poly in polys:
        shifts = [(0.0, 0.0)]
        if period is not None:
            px, py = period
            shifts = [(sx, sy) for sx in ((0, -px, px) if px else (0,)) for sy in ((0, -py, py) if py else (0,))]
        for sx, sy in shifts:
            m = raster(poly.pts + (sx, sy), size)
            albedo[m, :3] = np.round(np.array(poly.color) * 255)
            normal[m, 3] = poly.flags
            mat[m] = poly.mat
    texs = {}
    for name, arr, comps, dtype in (("albedo", albedo, 4, "f1"), ("normal", normal, 4, "f1"), ("mat", mat, 1, "u2")):
        t = ctx.texture(size, comps, np.ascontiguousarray(arr).tobytes(), dtype=dtype)
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texs[name] = t
    return texs, {"albedo": albedo, "normal": normal, "mat": mat}


LIT_FRAG = """
uniform sampler2D u_sdf;
uniform sampler2D u_albedo;
uniform sampler2D u_gi;
uniform bool u_has_gi;
uniform bool u_shadows;
uniform vec2 u_light;
uniform float u_radius;
uniform float u_soft;
uniform vec2 u_per;
uniform float u_amb;
uniform float u_light_gain;
out vec4 f_color;
void main() {
    vec2 p = gl_FragCoord.xy;
    vec3 alb = texelFetch(u_albedo, ivec2(p), 0).rgb;
    vec2 dl = u_light - p;
    if (u_per.x > 0.0) dl.x -= u_per.x * floor(dl.x / u_per.x + 0.5);
    if (u_per.y > 0.0) dl.y -= u_per.y * floor(dl.y / u_per.y + 0.5);
    float L = length(dl);
    float att = pow(clamp(1.0 - L / u_radius, 0.0, 1.0), 2.0) * u_light_gain;
    float sh = (u_shadows && L > 0.5) ? sdf_shadow(u_sdf, p, dl / L, L, u_soft, u_per) : 1.0;
    vec3 col = alb * (u_amb + att * sh);
    if (u_has_gi) col += alb * texture(u_gi, p / vec2(textureSize(u_gi, 0))).rgb;
    if (L < 4.0) col = vec3(1.0, 1.0, 0.6);
    f_color = vec4(col, 1.0);
}
"""

SHADOW_ONLY_FRAG = """
uniform sampler2D u_sdf;
uniform vec2 u_light;
uniform float u_soft;
uniform vec2 u_per;
out vec4 f_color;
void main() {
    vec2 p = gl_FragCoord.xy;
    vec2 dl = u_light - p;
    if (u_per.x > 0.0) dl.x -= u_per.x * floor(dl.x / u_per.x + 0.5);
    if (u_per.y > 0.0) dl.y -= u_per.y * floor(dl.y / u_per.y + 0.5);
    float L = length(dl);
    f_color = vec4(L > 0.5 ? sdf_shadow(u_sdf, p, dl / L, L, u_soft, u_per) : 1.0);
}
"""


class Harness:
    def __init__(self, size=SIZE):
        self.ctx = moderngl.create_standalone_context(require=410)
        self.size = size
        self.effects = gl_sdf.make(self.ctx, size)
        self.mats, self.n_mats = material_table(self.ctx)
        self.out = self.ctx.texture(size, 4, dtype="f1")
        self.out_fbo = self.ctx.framebuffer([self.out])
        self.shadow_out = self.ctx.texture(size, 4, dtype="f4")
        self.shadow_fbo = self.ctx.framebuffer([self.shadow_out])
        self._lit = self._shadow = None

    def frame(self, polys, period=None):
        texs, arrays = gbuffer(self.ctx, polys, self.size, period)
        f = Frame(self.ctx, self.size, texs, self.mats, self.n_mats, period=period, clip=self.size)
        f.arrays = arrays
        return f

    def run(self, f):
        f.shared.clear()
        for e in sorted(self.effects, key=lambda e: e.order):
            e.run(None, f)
        return f

    def sdf_array(self, f):
        t = f.shared["sdf"]
        return np.frombuffer(t.read(), np.float32).reshape(self.size[1], self.size[0])

    def gi_array(self, f):
        t = f.shared["gi"]
        return np.frombuffer(t.read(), np.float16).reshape(self.size[1], self.size[0], 4).astype(np.float32)

    def lit(self, f, light, soft, radius=700.0, amb=0.15, shadows=True, gain=1.0, read=True):
        """(h, w, 3) uint8 of the albedo under one point light (+ GI when shared["gi"] exists)."""
        if self._lit is None:
            self._lit = f.load("world_base.vert", "sdf_harness_lit.frag", ["sdf_shadow.glsl"])
        p = self._lit
        f.shared["sdf"].use(0)
        f.gbuf["albedo"].use(1)
        has_gi = "gi" in f.shared
        (f.shared["gi"] if has_gi else f.shared["sdf"]).use(2)
        for name, v in (("u_sdf", 0), ("u_albedo", 1), ("u_gi", 2), ("u_has_gi", has_gi), ("u_shadows", shadows),
                        ("u_light", tuple(map(float, light))), ("u_radius", radius), ("u_soft", soft),
                        ("u_per", gl_sdf._period(f.period)), ("u_amb", amb), ("u_light_gain", gain)):
            if name in p:
                p[name].value = v
        self.out_fbo.use()
        f.fullscreen(p)
        if not read:
            return None
        return np.frombuffer(self.out.read(), np.uint8).reshape(self.size[1], self.size[0], 4)[..., :3]

    def shadow(self, f, light, soft, read=True):
        """(h, w) float shadow factor (1 lit) of every pixel toward a point light."""
        if self._shadow is None:
            self._shadow = f.load("world_base.vert", "sdf_harness_shadow.frag", ["sdf_shadow.glsl"])
        p = self._shadow
        f.shared["sdf"].use(0)
        for name, v in (("u_sdf", 0), ("u_light", tuple(map(float, light))), ("u_soft", soft),
                        ("u_per", gl_sdf._period(f.period))):
            if name in p:
                p[name].value = v
        self.shadow_fbo.use()
        f.fullscreen(p)
        if not read:
            return None
        return np.frombuffer(self.shadow_out.read(), np.float32).reshape(self.size[1], self.size[0], 4)[..., 0]

    def gpu_ms(self, fn, reps=30):
        """Median GPU time of fn() in ms (GL time queries)."""
        fn()
        times = []
        q = self.ctx.query(time=True)
        for _ in range(reps):
            with q:
                fn()
            times.append(q.elapsed / 1e6)
        return float(np.median(times))


_EXTRA = {"sdf_harness_lit.frag": LIT_FRAG, "sdf_harness_shadow.frag": SHADOW_ONLY_FRAG}


def sdf_png(sdf):
    """Signed distance -> RGB: blue outside, orange inside, banded every 8 px, white boundary."""
    a = np.abs(sdf)
    band = 0.75 + 0.25 * np.cos(a * 2 * np.pi / 8.0)
    fade = np.exp(-a / 150.0)[..., None]
    out_c = np.array([0.35, 0.6, 0.95])
    in_c = np.array([0.95, 0.55, 0.2])
    col = np.where(sdf[..., None] < 0, in_c, out_c) * (0.25 + 0.75 * fade) * band[..., None]
    col = np.where((a < 1.0)[..., None], 1.0, col)
    return (np.clip(col, 0, 1) * 255).astype(np.uint8)


def save(path, rgb):
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(rgb)).save(path)


def main(argv):
    out = argv[0] if argv else os.path.join(tempfile.gettempdir(), "sdf_harness")
    os.makedirs(out, exist_ok=True)
    hz = Harness()
    gi = hz.effects[1]
    ctx = hz.ctx
    light = (520.0, 120.0)

    f = hz.run(hz.frame(scene_default()))
    save(os.path.join(out, "sdf.png"), sdf_png(hz.sdf_array(f)))
    for soft in (2.0, 16.0):
        save(os.path.join(out, f"shadows_soft{int(soft)}.png"), hz.lit(f, light, soft))

    # close-up of a round body's terminator and shadow edges (4x nearest upscale): clean, no speckle
    for soft in (2.0, 16.0):
        img = hz.lit(f, (560.0, 330.0), soft)[340:500, 610:770]
        save(os.path.join(out, f"round_edges_soft{int(soft)}.png"), np.repeat(np.repeat(img, 3, 0), 3, 1))

    t_sdf = hz.gpu_ms(lambda: hz.effects[0].run(None, f))
    t_lit_on = hz.gpu_ms(lambda: hz.lit(f, light, 8.0, read=False))
    t_lit_far = hz.gpu_ms(lambda: hz.lit(f, (20.0, 20.0), 8.0, read=False))
    t_lit_off = hz.gpu_ms(lambda: hz.lit(f, light, 8.0, shadows=False, read=False))

    # periodic: an occluder straddling the left / right seam, light near the right edge
    per = SIZE
    seam = [Poly(np.array([[-40, 250], [40, 250], [40, 330], [-40, 330]], float), (0.8, 0.3, 0.6)),
            Poly(regular(450, 470, 35, 5), (0.3, 0.7, 0.7))]
    fp = hz.run(hz.frame(seam, period=per))
    save(os.path.join(out, "periodic_sdf.png"), sdf_png(hz.sdf_array(fp)))
    save(os.path.join(out, "periodic_shadows.png"), hz.lit(fp, (840.0, 290.0), 4.0))

    # GI: a glowing polygon among occluders, the point light dimmed
    gl_sdf.GI_ENABLED = True
    try:
        glow = scene_default()
        glow[-1] = Poly(regular(520, 300, 25, 6), (1.0, 0.55, 0.15), flags=OCCLUDER | LIT, mat=2)
        glow[0] = Poly(regular(400, 300, 30, 4, np.pi / 4), (0.85, 0.85, 0.85))
        fg = hz.run(hz.frame(glow))
        save(os.path.join(out, "gi.png"), hz.lit(fg, light, 8.0, amb=0.05, gain=0.0))
        g = hz.gi_array(fg)[..., :3]
        save(os.path.join(out, "gi_raw.png"), (np.clip(g / max(g.max(), 1e-6), 0, 1) ** 0.5 * 255).astype(np.uint8))
        t_gi = hz.gpu_ms(lambda: gi.run(None, fg))
    finally:
        gl_sdf.GI_ENABLED = False

    print(f"GL {ctx.info['GL_RENDERER']} {ctx.version_code}, {SIZE[0]}x{SIZE[1]}")
    print(f"SDF build (seed + {len(hz.effects[0].jfa.steps)} steps + distance): {t_sdf:.3f} ms")
    print(f"one point light, fullscreen lit pass: {t_lit_on:.3f} ms with shadows (light at {light}), "
          f"{t_lit_far:.3f} ms (corner light), {t_lit_off:.3f} ms without: shadow march ~"
          f"{max(t_lit_on, t_lit_far) - t_lit_off:.3f} ms per light")
    print(f"GI ({gl_sdf.N_CASCADES} cascades, grid {gi.grid}): {t_gi:.3f} ms")
    print(f"PNGs in {out}")
    for e in hz.effects:
        e.release()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
