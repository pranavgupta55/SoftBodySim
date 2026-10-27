"""Lights and compose for the lit renderer (effects in gl_effects' plugin API, see research/lighting_contract.md).

LightsEffect (order 200): every drawable light of frame.world (lights.sanitize: disabled / invalid ones dropped, the
rest clamped) in one instanced additive draw into shared["light"] (diffuse RGB) and shared["spec"] (specular RGB),
RGBA16F, RGBA32F while more than F4_MIN_LIGHTS lights draw (LIGHT_DTYPE). Lights go to the GPU in a std140
uniform block, 4 vec4 per light (pack_lights), MAX_LIGHTS per draw (more lights draw in batches). Point lights
cover the box of their radius, spot lights the box of their cone, directional lights the whole surface; light_accum
does the per-pixel work (N.L with the light at its pseudo height, Blinn-Phong, toon bands, shadows from
shared["sdf"] via sdf_shadow.glsl when there is one, minimum-image distances on periodic maps).

ComposeEffect (order 300): GLWorld.tex = tone(exposure x (albedo * (ambient + light + gi + caustics) + spec +
emissive)) on lit pixels (TONEMAP: a hue-preserving shoulder above TONE_KNEE plus a +-0.5/255 dither; off: a plain
clamp; exposure: world.exposure, 1 by default), albedo unchanged on unlit ones; then a copy into
shared["scene_lit"] (RGBA8) for refraction. Missing light / spec / gi / caustics read as black.
"""
import math

import moderngl
import numpy as np

import lights as lights_mod

MAX_LIGHTS = 256  # per draw: 4 vec4 each fill the 16 KB uniform block GL guarantees
KINDS = {lights_mod.POINT: 0, lights_mod.SPOT: 1, lights_mod.DIRECTIONAL: 2}
# Directional lights have no position: their elevation's tangent is Light.height / this (the default height, 60,
# gives 45 degrees), so the same height slider still raises or lowers them.
DIRECTIONAL_HEIGHT_REF = 60.0
# Diffuse on flat ground: 0 = plain N.L with the light at its pseudo height (a flat pixel d px away gets
# h / sqrt(d^2 + h^2), so pools fade fast and anything far off the light is near black); > 0 = "relief" shading: a flat
# pixel gets 1 (the falloff alone shapes the pool) and a tilted bevel 1 + DIFFUSE_RELIEF x (N.L - flat N.L), so faces
# toward the light brighten and faces away darken by how much they turn.
DIFFUSE_RELIEF = 0.8
# The light / spec targets: RGBA16F rounds every additive light to half precision (300 dim lights drift ~8% near 1),
# RGBA32F doesn't but costs ~1-1.5 ms a frame at 900x584 on penrose maps, so by default (None) they are RGBA32F only
# while more than F4_MIN_LIGHTS lights draw (reallocated when the count crosses it). "f2" / "f4" force one. Where
# float32 render targets fail they fall back to "f2".
LIGHT_DTYPE = None
F4_MIN_LIGHTS = 32
LIGHT_LEVELS = 0  # > 0: compose quantises the total light's luminance to this many levels per unit, Bayer dithered
MIN_SOFTNESS_PX = 0.75  # the smallest shadow softness passed to the shader: an anti-aliased edge for hard shadows
# compose's tone mapping: a hue-preserving shoulder above TONE_KNEE (the brightest channel rolls off smoothly towards
# 1 instead of clipping, so an orange light stays orange instead of going flat white) plus a +-0.5/255 dither against
# banding. Off: plain clamp, as before. world.exposure (if set) scales the lit colour before it.
TONEMAP = True
TONE_KNEE = 0.8


def pack_lights(lights, zoom):
    """(n, 16) float32: the drawable lights (lights.sanitize: disabled / invalid ones dropped, the rest clamped) in
    light_quad.vert / light_accum.frag's layout, positions in render px.
        [pos.x, pos.y, height (px; directional: elevation tangent), kind]
        [colour * intensity (rgb), radius px]
        [dir.x, dir.y, cos inner, cos outer]          dir: unit; spot axis / directional travel direction
        [falloff, softness px, casts shadows 0/1, 0]
    Softness is at least MIN_SOFTNESS_PX: a 0 (hard) shadow still gets an anti-aliased edge."""
    zoom = float(zoom) if math.isfinite(float(zoom)) and float(zoom) > 0 else 1.0
    rows = []
    for raw in lights:
        lt = lights_mod.sanitize(raw)
        if lt is None:
            continue
        kind = KINDS[lt.kind]
        dx, dy = lt.dir
        cos_o, cos_i = math.cos(math.radians(lt.cone_outer)), math.cos(math.radians(lt.cone_inner))
        cos_i = max(cos_i, cos_o + 1e-4)  # smoothstep needs edge0 < edge1
        height = lt.height / DIRECTIONAL_HEIGHT_REF if kind == 2 else lt.height * zoom
        r, g, b = (c * lt.intensity for c in lt.color)
        rows.append([lt.pos[0] * zoom, lt.pos[1] * zoom, height, kind,
                     r, g, b, lt.radius * zoom,
                     dx, dy, cos_i, cos_o,
                     lt.falloff, max(lt.softness * zoom, MIN_SOFTNESS_PX),
                     1.0 if lt.casts_shadows else 0.0, 0.0])
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.array(rows, np.float32).reshape(-1, 16)
    # float32 overflow (huge positions x zoom) -> the largest finite value, so no inf / NaN reaches the shader
    return np.nan_to_num(out, nan=0.0, posinf=np.finfo(np.float32).max / 4, neginf=-np.finfo(np.float32).max / 4)


def _period(frame):
    p = getattr(frame, "period", None)
    return (0.0, 0.0) if p is None else (float(p[0] or 0.0), float(p[1] or 0.0))


def exposure(world):
    """world.exposure (multiplies the lit colour before tone mapping): finite, in [0, 64]; 1 when unset or bad."""
    e = getattr(world, "exposure", None)
    try:
        e = 1.0 if e is None else float(e)
    except (TypeError, ValueError):
        return 1.0
    return min(max(e, 0.0), 64.0) if math.isfinite(e) else 1.0


def shadow_reach(size, frame):
    """(u_shadow_reach, u_shadow_fade_start) in px: how far a directional light's shadow march looks, and from where
    its shadows fade out. On a periodic map the march wraps, so it looks at least 1.5 periods (an occluder one period
    over still shades) and fades from half way, so a long shadow doesn't end on a hard line; otherwise the surface
    diagonal and no fade (0)."""
    diag = float(math.hypot(*size))
    p = _period(frame)
    if max(p) > 0:
        reach = max(diag, 1.5 * max(p))
        return reach, 0.5 * reach
    return diag, 0.0


def _restore(ctx, prev):
    ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
    if prev is not None:
        prev.use()


class _Black:
    """A 1x1 black RGBA16F texture standing in for a missing shared input."""

    def __init__(self, ctx):
        self.tex = ctx.texture((1, 1), 4, np.zeros(4, np.float16).tobytes(), dtype="f2")
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)


class LightsEffect:
    order = 200

    def __init__(self, ctx, size):
        self.ctx = ctx
        self.size = None
        self.progs = {}
        self.vaos = {}
        self.ubo = ctx.buffer(reserve=MAX_LIGHTS * 64)
        self.light = self.spec = self.fbo = None
        self.dtype = None  # what the targets are ("f2" / "f4")
        self.want = "f2"  # what was asked for (an "f4" that failed stays "f2" without retrying every frame)
        self._alloc(size)

    def _dtype(self, n_lights):
        """The target dtype for n lights: LIGHT_DTYPE if forced, else f4 past F4_MIN_LIGHTS (back to f2 at or below
        F4_MIN_LIGHTS * 3 / 4, so a count near the threshold doesn't reallocate every frame)."""
        if LIGHT_DTYPE in ("f2", "f4"):
            return LIGHT_DTYPE
        if n_lights > F4_MIN_LIGHTS:
            return "f4"
        return "f2" if n_lights <= F4_MIN_LIGHTS * 3 // 4 else self.want

    def _alloc(self, size, dtype="f2"):
        for o in (self.fbo, self.light, self.spec):
            if o is not None:
                o.release()
        self.light = self.spec = self.fbo = None
        self.size = tuple(size)
        self.want = dtype
        for dtype in dict.fromkeys((dtype, "f2")):  # fp32 sums don't drift over hundreds of lights
            try:
                self.light = self.ctx.texture(self.size, 4, dtype=dtype)
                self.spec = self.ctx.texture(self.size, 4, dtype=dtype)
                self.fbo = self.ctx.framebuffer([self.light, self.spec])
                self.dtype = dtype
                break
            except moderngl.Error:  # no float32 render targets here: half floats
                for o in (self.fbo, self.light, self.spec):
                    if o is not None:
                        o.release()
                self.light = self.spec = self.fbo = None
        for t in (self.light, self.spec):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _prog(self, frame, has_sdf):
        if has_sdf not in self.progs:
            shadow = "sdf_shadow.glsl" if has_sdf else "light_noshadow.glsl"
            prog = frame.load("light_quad.vert", "light_accum.frag", includes=("material.glsl", shadow))
            prog["Lights"].binding = 0
            self.progs[has_sdf] = prog
            self.vaos[has_sdf] = self.ctx.vertex_array(prog, [])
        return self.progs[has_sdf], self.vaos[has_sdf]

    def run(self, gw, frame):
        ctx = self.ctx
        data = pack_lights(lights_mod.of_world(frame.world)[0], float(frame.zoom))
        want = self._dtype(len(data))
        prev = ctx.fbo
        if tuple(frame.size) != self.size or want != self.want:
            if prev is self.fbo:  # (about to be released: nothing to restore)
                prev = None
            self._alloc(frame.size, want)
        frame.shared["light"], frame.shared["spec"] = self.light, self.spec
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        if len(data):
            sdf = frame.shared.get("sdf")
            prog, vao = self._prog(frame, sdf is not None)
            units = {"u_normal": frame.gbuf["normal"], "u_mat": frame.gbuf["mat"], "u_mat_table": frame.material_tex}
            if sdf is not None:
                units["u_sdf"] = sdf
            for unit, (name, tex) in enumerate(units.items()):
                tex.use(unit)
                if name in prog:
                    prog[name].value = unit
            reach, fade_start = shadow_reach(self.size, frame)
            values = {"u_res": tuple(float(s) for s in self.size), "u_period": _period(frame),
                      "u_shadow_reach": reach, "u_shadow_fade_start": fade_start, "u_relief": float(DIFFUSE_RELIEF)}
            for name, v in values.items():
                if name in prog:
                    prog[name].value = v
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ctx.blend_equation = moderngl.FUNC_ADD
            self.ubo.bind_to_uniform_block(0)
            for start in range(0, len(data), MAX_LIGHTS):
                chunk = data[start:start + MAX_LIGHTS]
                self.ubo.write(chunk.tobytes())
                vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(chunk))
            ctx.blend_func = moderngl.DEFAULT_BLENDING
        _restore(ctx, prev)

    def release(self):
        for o in [self.fbo, self.light, self.spec, self.ubo, *self.vaos.values(), *self.progs.values()]:
            o.release()
        self.progs, self.vaos = {}, {}


class ComposeEffect:
    order = 300

    def __init__(self, ctx, size):
        self.ctx = ctx
        self.prog = self.vao = None
        self.black = _Black(ctx)
        self.size = None
        self.scene_lit = None
        self.albedo_copy = None  # only when the G-buffer's albedo is GLWorld.tex itself
        self.targets = {}  # id(texture) -> (texture, framebuffer) writing GLWorld.tex
        self._alloc(size)

    def _alloc(self, size):
        for o in (self.scene_lit, self.albedo_copy):
            if o is not None:
                o.release()
        self.albedo_copy = None
        self.size = tuple(size)
        self.scene_lit = self.ctx.texture(self.size, 4)
        self.scene_lit.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _target(self, tex):
        key = id(tex)
        if key not in self.targets or self.targets[key][0] is not tex:
            for _, fbo in self.targets.values():
                fbo.release()
            self.targets = {key: (tex, self.ctx.framebuffer([tex]))}
        return self.targets[key][1]

    def run(self, gw, frame):
        ctx = self.ctx
        if tuple(frame.size) != self.size:
            self._alloc(frame.size)
        if self.prog is None:
            self.prog = frame.load("light_fs.vert", "compose.frag", includes=("material.glsl",))
            self.vao = ctx.vertex_array(self.prog, [])
        prev = ctx.fbo
        fbo = self._target(gw.tex)
        albedo = frame.gbuf["albedo"]
        if albedo is gw.tex:  # can't read and write one texture: compose from a copy
            if self.albedo_copy is None:
                self.albedo_copy = ctx.texture(self.size, 4)
                self.albedo_copy.filter = (moderngl.NEAREST, moderngl.NEAREST)
            ctx.copy_framebuffer(self.albedo_copy, fbo)
            albedo = self.albedo_copy
        sh, black = frame.shared, self.black.tex
        units = {"u_albedo": albedo, "u_normal": frame.gbuf["normal"], "u_mat": frame.gbuf["mat"],
                 "u_mat_table": frame.material_tex, "u_light": sh.get("light", black), "u_spec": sh.get("spec", black),
                 "u_gi": sh.get("gi", black), "u_caustics": sh.get("caustics", black)}
        prog = self.prog
        for unit, (name, tex) in enumerate(units.items()):
            tex.use(unit)
            if name in prog:
                prog[name].value = unit
        ambient = lights_mod.of_world(frame.world)[1]
        if "u_ambient" in prog:
            prog["u_ambient"].value = tuple(float(c) for c in tuple(ambient)[:3])
        values = {"u_levels": float(LIGHT_LEVELS), "u_tonemap": bool(TONEMAP),
                  "u_knee": min(max(float(TONE_KNEE), 0.0), 0.99), "u_exposure": exposure(frame.world)}
        for name, v in values.items():
            if name in prog:
                prog[name].value = v
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        fbo.use()
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        ctx.copy_framebuffer(self.scene_lit, fbo)
        sh["scene_lit"] = self.scene_lit
        _restore(ctx, prev)

    def release(self):
        objs = [self.black.tex, self.scene_lit, self.albedo_copy, self.vao, self.prog]
        objs += [fbo for _, fbo in self.targets.values()]
        for o in objs:
            if o is not None:
                o.release()
        self.targets = {}


def make(ctx, size):
    return [LightsEffect(ctx, size), ComposeEffect(ctx, size)]
