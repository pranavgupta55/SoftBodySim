"""Lights and compose for the lit renderer (effects in gl_effects' plugin API, see research/lighting_contract.md).

LightsEffect (order 200): every enabled light of lights.of_world(frame.world) in one instanced additive draw into
shared["light"] (diffuse RGB) and shared["spec"] (specular RGB), both RGBA16F. Lights go to the GPU in a std140
uniform block, 4 vec4 per light (pack_lights), MAX_LIGHTS per draw (more lights draw in batches). Point lights
cover the box of their radius, spot lights the box of their cone, directional lights the whole surface; light_accum
does the per-pixel work (N.L with the light at its pseudo height, Blinn-Phong, toon bands, shadows from
shared["sdf"] via sdf_shadow.glsl when there is one, minimum-image distances on periodic maps).

ComposeEffect (order 300): GLWorld.tex = albedo * (ambient + light + gi + caustics) + spec + emissive on lit pixels,
albedo unchanged on unlit ones; then a copy into shared["scene_lit"] (RGBA8) for refraction. Missing light / spec /
gi / caustics read as black.
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
LIGHT_LEVELS = 0  # > 0: compose quantises the total light to this many levels per unit, Bayer dithered (retro look)


def pack_lights(lights, zoom):
    """(n, 16) float32: the enabled lights in light_quad.vert / light_accum.frag's layout, positions in render px.
        [pos.x, pos.y, height (px; directional: elevation tangent), kind]
        [colour * intensity (rgb), radius px]
        [dir.x, dir.y, cos inner, cos outer]          dir: unit; spot axis / directional travel direction
        [falloff, softness px, casts shadows 0/1, 0]"""
    rows = []
    for lt in lights:
        if not lt.enabled or lt.kind not in KINDS:
            continue
        kind = KINDS[lt.kind]
        dx, dy = (float(v) for v in lt.dir)
        norm = math.hypot(dx, dy)
        dx, dy = (dx / norm, dy / norm) if norm > 0 else (0.0, 1.0)
        outer = min(max(float(lt.cone_outer), 0.0), 180.0)
        inner = min(max(float(lt.cone_inner), 0.0), outer)
        cos_o, cos_i = math.cos(math.radians(outer)), math.cos(math.radians(inner))
        cos_i = max(cos_i, cos_o + 1e-4)  # smoothstep needs edge0 < edge1
        height = (max(float(lt.height), 0.0) / DIRECTIONAL_HEIGHT_REF if kind == 2
                  else max(float(lt.height), 0.0) * zoom)
        r, g, b = (float(c) * float(lt.intensity) for c in tuple(lt.color)[:3])
        rows.append([float(lt.pos[0]) * zoom, float(lt.pos[1]) * zoom, height, kind,
                     r, g, b, max(float(lt.radius), 0.0) * zoom,
                     dx, dy, cos_i, cos_o,
                     max(float(lt.falloff), 0.0), max(float(lt.softness), 0.0) * zoom,
                     1.0 if lt.casts_shadows else 0.0, 0.0])
    return np.array(rows, np.float32).reshape(-1, 16)


def _period(frame):
    p = getattr(frame, "period", None)
    return (0.0, 0.0) if p is None else (float(p[0] or 0.0), float(p[1] or 0.0))


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
        self._alloc(size)

    def _alloc(self, size):
        if self.fbo is not None:
            for o in (self.fbo, self.light, self.spec):
                o.release()
        self.size = tuple(size)
        self.light = self.ctx.texture(self.size, 4, dtype="f2")
        self.spec = self.ctx.texture(self.size, 4, dtype="f2")
        for t in (self.light, self.spec):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = self.ctx.framebuffer([self.light, self.spec])

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
        if tuple(frame.size) != self.size:
            self._alloc(frame.size)
        prev = ctx.fbo
        frame.shared["light"], frame.shared["spec"] = self.light, self.spec
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        data = pack_lights(lights_mod.of_world(frame.world)[0], float(frame.zoom))
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
            values = {"u_res": tuple(float(s) for s in self.size), "u_period": _period(frame),
                      "u_shadow_reach": float(math.hypot(*self.size))}
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
        if "u_levels" in prog:
            prog["u_levels"].value = float(LIGHT_LEVELS)
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
