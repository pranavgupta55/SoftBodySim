"""Offscreen harness for the lit renderer's effects (gl_lights for now), independent of GLWorld's G-buffer.

Builds a synthetic G-buffer per research/lighting_contract.md (albedo / normal+edge distance+flags / material id) from
a few analytic shapes with bevel / dome normals, a material table, an analytic SDF, a FrameInfo stand-in and a GLWorld
stand-in (just `tex`), then runs the effects and reads back GLWorld.tex. Run it to render the check PNGs and time the
lights:

    .venv/bin/python3 tests/lit_harness.py [out_dir]     # default out_dir: tests/out/lights

PROVISIONAL (swap at integration): pack_material / MATERIAL_GLSL stand in for materials.pack and
shaders/material.glsl (F), SDF_SHADOW_GLSL for shaders/sdf_shadow.glsl (S). The loader prefers the real files in
shaders/ when they exist, so once F and S land only pack_material needs replacing (by materials.pack).
"""
import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import moderngl
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import materials  # noqa: E402
from lights import Light  # noqa: E402

SHADER_DIR = os.path.join(ROOT, "shaders")
SIZE = (900, 584)

# ---------------------------------------------------------------- provisional stand-ins (F's / S's files)
MAT_TEXELS = 8


def pack_material(m):
    """Provisional materials.pack: 8 RGBA32F texels (32 floats) per material, every Material field."""
    f = [*m.tint, m.tint_strength,
         *m.emissive, m.emissive_strength,
         m.specular, m.roughness, m.bands, m.bevel_px,
         m.profile, m.kind, m.ior, m.reflectivity,
         m.transparency, m.absorption, m.pattern, m.pattern_scale,
         m.pattern_strength, float(m.casts_shadow), float(m.lit), 0.0]
    return np.array(f + [0.0] * (MAT_TEXELS * 4 - len(f)), np.float32)


MATERIAL_GLSL = """
struct Mat {
    vec3 tint; float tint_strength;
    vec3 emissive; float emissive_strength;
    float specular; float roughness; int bands; float bevel_px;
    int profile; int kind; float ior; float reflectivity;
    float transparency; float absorption; int pattern; float pattern_scale;
    float pattern_strength; bool casts_shadow; bool lit;
};
Mat mat_fetch(sampler2D table, uint id) {
    int y = int(id);
    vec4 t0 = texelFetch(table, ivec2(0, y), 0), t1 = texelFetch(table, ivec2(1, y), 0);
    vec4 t2 = texelFetch(table, ivec2(2, y), 0), t3 = texelFetch(table, ivec2(3, y), 0);
    vec4 t4 = texelFetch(table, ivec2(4, y), 0), t5 = texelFetch(table, ivec2(5, y), 0);
    return Mat(t0.rgb, t0.a, t1.rgb, t1.a, t2.x, t2.y, int(t2.z + 0.5), t2.w, int(t3.x + 0.5), int(t3.y + 0.5), t3.z,
               t3.w, t4.x, t4.y, int(t4.z + 0.5), t4.w, t5.x, t5.y > 0.5, t5.z > 0.5);
}
"""

SDF_SHADOW_GLSL = """
float sdf_at(sampler2D sdf, vec2 p, vec2 period) {
    vec2 q = p;
    for (int k = 0; k < 2; k++) {
        if (period[k] > 0.0) q[k] -= period[k] * floor(q[k] / period[k]);
    }
    ivec2 s = textureSize(sdf, 0);
    ivec2 c = clamp(ivec2(floor(q)), ivec2(0), s - 1);
    return texelFetch(sdf, c, 0).r;
}
float sdf_shadow(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period) {
    float t = 0.0;
    for (int i = 0; i < 64 && t < max_t; i++) {  // step out of the pixel's own body first
        float d = sdf_at(sdf, p + dir * t, period);
        if (d >= 0.0) break;
        t += max(-d, 1.0);
    }
    float t0 = t;  // the surface the light reaches: penumbrae widen from here
    t += 1.0;
    float res = 1.0;
    float k = max(softness_px, 0.5);
    for (int i = 0; i < 96 && t < max_t; i++) {
        float d = sdf_at(sdf, p + dir * t, period);
        if (d < 0.0) return 0.0;
        res = min(res, d * max_t / (k * (t - t0)));
        t += max(d, 1.0);
    }
    return clamp(res, 0.0, 1.0);
}
"""
PROVISIONAL_INCLUDES = {"material.glsl": MATERIAL_GLSL, "sdf_shadow.glsl": SDF_SHADOW_GLSL}


def _read_shader(name):
    path = os.path.join(SHADER_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return PROVISIONAL_INCLUDES[name]


def load(ctx, vert, frag, includes=()):
    """gl_world._load plus includes: #version, the stage define, world_common.glsl, the includes, the file."""
    common = _read_shader("world_common.glsl")
    inc = "\n".join(_read_shader(n) for n in includes)
    srcs = []
    for name, define, extra in ((vert, "#define VERTEX_SHADER\n", ""), (frag, "", inc)):
        srcs.append("#version 330 core\n" + define + common + "\n" + extra + "\n" + _read_shader(name))
    return ctx.program(vertex_shader=srcs[0], fragment_shader=srcs[1])


@dataclass
class FrameInfo:
    """Stand-in for gl_effects.FrameInfo (F)."""
    ctx: object
    size: tuple
    clip: tuple
    zoom: float
    period: object
    world: object
    gbuf: dict
    material_tex: object
    n_materials: int
    shared: dict = field(default_factory=dict)

    def fullscreen(self, prog):
        self.ctx.vertex_array(prog, []).render(moderngl.TRIANGLE_STRIP, vertices=4)

    def load(self, vert, frag, includes=()):
        return load(self.ctx, vert, frag, includes)


# ---------------------------------------------------------------- synthetic scene
MATS = ["matte", "glossy", "toon", "glow", "metal"]  # material ids 1.. (0 = background)
MAT_ID = {name: i + 1 for i, name in enumerate(MATS)}
BACKGROUND = (28, 30, 40)
OCCLUDER, CASTS, LIT = 1, 2, 4


def default_shapes():
    """(kind, centre, size, colour, material, flags): circles (size = radius) and boxes (size = half extents)."""
    return [
        ("box", (450, 545), (430, 30), (150, 140, 120), "matte", 7),
        ("circle", (200, 300), 70, (210, 80, 70), "matte", 7),
        ("circle", (450, 260), 80, (80, 150, 220), "glossy", 7),
        ("circle", (700, 300), 70, (110, 200, 110), "toon", 7),
        ("box", (330, 430), (50, 35), (220, 190, 80), "matte", 7),
        ("box", (600, 440), (60, 30), (170, 120, 200), "metal", 7),
        ("circle", (820, 120), 30, (255, 230, 150), "glow", 5),  # emits, casts no shadow
        ("box", (90, 120), (45, 30), (90, 200, 200), "matte", OCCLUDER | CASTS),  # lit flag clear
    ]


def build_scene(shapes, size=SIZE):
    """numpy G-buffer arrays (row 0 = top) and the SDF: dict(albedo, normal, mat, sdf)."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    albedo = np.zeros((h, w, 4), np.uint8)
    albedo[..., :3] = BACKGROUND
    albedo[..., 3] = 255
    normal = np.zeros((h, w, 4), np.uint8)
    normal[..., :2] = 128
    mat = np.zeros((h, w), np.uint16)
    sdf = np.full((h, w), 1e4, np.float32)
    for kind, c, s, col, mname, flags in shapes:
        m = materials.PRESETS[mname]
        dx, dy = xs - c[0], ys - c[1]
        if kind == "circle":
            r = np.hypot(dx, dy)
            sd = r - s  # signed distance, negative inside
            safe = np.maximum(r, 1e-6)
            ux, uy = dx / safe, dy / safe
        else:
            qx, qy = np.abs(dx) - s[0], np.abs(dy) - s[1]
            outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
            sd = outside + np.minimum(np.maximum(qx, qy), 0)
            xside = qx > qy
            ux, uy = np.where(xside, np.sign(dx), 0.0), np.where(xside, 0.0, np.sign(dy))
        inside = sd < 0
        edge = -sd
        t = np.clip(1.0 - edge / max(m.bevel_px * 2.5, 1.0), 0.0, 1.0)  # synthetic bevel: wider than the material's
        if m.profile == materials.DOME or kind == "circle":
            t = np.clip(1.0 - edge / (s if kind == "circle" else min(s)), 0.0, 1.0) ** 1.5
        tilt = 0.9 * t
        nx, ny = ux * tilt, uy * tilt
        albedo[inside, :3] = col
        normal[inside, 0] = np.round((nx[inside] * 0.5 + 0.5) * 255)
        normal[inside, 1] = np.round((ny[inside] * 0.5 + 0.5) * 255)
        normal[inside, 2] = np.clip(np.round(edge[inside]), 0, 255)
        normal[inside, 3] = flags
        mat[inside] = MAT_ID[mname]
        if flags & OCCLUDER and m.casts_shadow:
            sdf = np.minimum(sdf, sd)
    return {"albedo": albedo, "normal": normal, "mat": mat, "sdf": sdf}


class Harness:
    def __init__(self, size=SIZE, shapes=None, ctx=None):
        import gl_lights
        self.ctx = ctx or moderngl.create_standalone_context(require=410)
        self.size = size
        self.scene = build_scene(shapes or default_shapes(), size)
        ctx = self.ctx
        tex = {}
        for key, comps, dtype in (("albedo", 4, "u1"), ("normal", 4, "u1"), ("mat", 1, "u2")):
            tex[key] = ctx.texture(size, comps, np.ascontiguousarray(self.scene[key]).tobytes(), dtype=dtype)
            tex[key].filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.gbuf = tex
        table = np.zeros((len(MATS) + 1, MAT_TEXELS * 4), np.float32)
        table[0] = pack_material(materials.PRESETS[materials.DEFAULT])
        for name, i in MAT_ID.items():
            table[i] = pack_material(materials.PRESETS[name])
        self.material_tex = ctx.texture((MAT_TEXELS, len(table)), 4, table.tobytes(), dtype="f4")
        self.material_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.sdf_tex = ctx.texture(size, 1, self.scene["sdf"].tobytes(), dtype="f4")
        self.sdf_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.gw = SimpleNamespace(tex=ctx.texture(size, 4), size=size)
        self.gw.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.effects = sorted(gl_lights.make(ctx, size), key=lambda e: e.order)
        self.frame = None

    def run(self, lights, ambient=(0.25, 0.25, 0.3), sdf=True, period=None, zoom=1.0, shared=None, only=None,
            read=True):
        world = SimpleNamespace(lights=list(lights), ambient=ambient)
        sh = dict(shared or {})
        if sdf:
            sh["sdf"] = self.sdf_tex
        self.frame = FrameInfo(self.ctx, self.size, self.size, zoom, period, world, self.gbuf, self.material_tex,
                               len(MATS) + 1, sh)
        for e in self.effects:
            if only is None or e.order in only:
                e.run(self.gw, self.frame)
        return self.image() if read else None

    def image(self):
        """GLWorld.tex as (h, w, 3) uint8, row 0 = top."""
        w, h = self.size
        return np.frombuffer(self.gw.tex.read(), np.uint8).reshape(h, w, 4)[..., :3].copy()

    def read(self, key):
        """A shared RGBA16F / RGBA32F texture as (h, w, 4) float32."""
        w, h = self.size
        tex = self.frame.shared[key]
        dt = np.float32 if tex.dtype == "f4" else np.float16
        return np.frombuffer(tex.read(), dt).reshape(h, w, 4).astype(np.float32)

    def release(self):
        for e in self.effects:
            e.release()


# ---------------------------------------------------------------- demo scenes
def scenes():
    warm = (1.0, 0.92, 0.8)
    return {
        "no_lights": [],
        "point": [Light(pos=(450, 150), color=warm, intensity=2.0, radius=700, falloff=1.5, height=120)],
        "spot": [Light(kind="spot", pos=(120, 40), dir=(0.55, 0.83), color=warm, intensity=2.5, radius=800,
                       falloff=1.0, cone_inner=16, cone_outer=26, height=100)],
        "directional": [Light(kind="directional", dir=(0.5, 1.0), color=(0.9, 0.9, 1.0), intensity=0.9, height=60)],
        "colored": [Light(pos=(200, 150), color=(1.0, 0.2, 0.2), intensity=1.5, radius=420, height=60),
                    Light(pos=(450, 420), color=(0.2, 1.0, 0.3), intensity=1.5, radius=380, height=60),
                    Light(pos=(720, 150), color=(0.3, 0.4, 1.0), intensity=1.8, radius=420, height=60)],
        "toon": [Light(pos=(620, 160), color=warm, intensity=2.0, radius=600, falloff=1.0, height=100)],
    }


def save_png(img, path):
    from PIL import Image
    Image.fromarray(img).save(path)


def perf(h, counts=(1, 8, 32), frames=60):
    rng = np.random.default_rng(1)
    out = {}
    for n in counts:
        ls = [Light(pos=(float(rng.uniform(0, SIZE[0])), float(rng.uniform(0, SIZE[1]))), radius=300,
                    color=tuple(rng.uniform(0.3, 1.0, 3)), intensity=0.5) for _ in range(n)]
        for with_sdf in (False, True):
            h.run(ls, sdf=with_sdf, read=False)
            h.ctx.finish()
            t0 = time.perf_counter()
            for _ in range(frames):
                h.run(ls, sdf=with_sdf, read=False)
            h.ctx.finish()
            out[(n, with_sdf)] = (time.perf_counter() - t0) / frames * 1000.0
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "out", "lights")
    os.makedirs(out_dir, exist_ok=True)
    h = Harness()
    for name, ls in scenes().items():
        path = os.path.join(out_dir, f"{name}.png")
        save_png(h.run(ls), path)
        print("wrote", path)
    for (n, with_sdf), ms in perf(h).items():
        print(f"{n:3d} lights {'sdf' if with_sdf else 'no sdf':6s} {ms:6.2f} ms/frame (lights + compose)")
    h.release()


if __name__ == "__main__":
    main()
