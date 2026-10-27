"""Glass, mirrors, fluid and caustics for the lit renderer (workstream G of research/lighting_contract.md).

make(ctx, size) -> [CausticsEffect (order 250, off unless CAUSTICS), GlassEffect (order 400)].

The viewer looks straight down at the plane. Two kinds of optics:
- Pseudo-3D (screen space): every body is a bump whose surface normal is (bevel n.xy, n.z = sqrt(1 - |n.xy|^2)) from
  the G-buffer. A glass / fluid pixel shows what lies behind the body, displaced by the bent view ray
  (REFRACT_DEPTH_PX, MIN_REFRACT_PX), absorbed toward the body's colour (Beer-Lambert over THICK_PER_EDGE x the edge
  distance), blended over the body's own lit colour by `transparency`, plus a Schlick Fresnel rim and a glint.
- In-plane (with S's SDF, SDF_TRACE): rays travelling in the plane, marched with sdf_at and the G-buffer normals
  (glass_common.march): mirror bounces, Snell refraction into and out of glass, total internal reflection. Used for
  what a mirror (or a glass rim, by Fresnel) reflects, and by the caustics.
What a mirror shows follows one convention: the eye looks down along VIEW3 = (VIEW_TILT, -1), i.e. tilted toward
-y (up the screen); a pixel reflects that ray about its 3D normal, and the reflected ray's in-plane part points at
what it sees, MIRROR_DEPTH_PX * |R.xy| / R.z px away (flat interior: a short way up the screen; tilted rim: far out
beside the body), starting from the body's nearest edge.

"Behind" a glass body: the G-buffer is one opaque layer, so scene_lit holds the glass body itself where it is. The
floor under it is redrawn: gbuf["behind"] if F provides one (an RGBA8 copy of the base pass: background, grid,
trails, before the bodies), else GLWorld's own base pass run again into a texture of ours (it is a fullscreen pass
whose uniforms and textures are still bound from this frame), else nothing. It is lit by the light ratio
scene_lit / albedo of the surrounding floor (background pixels), inpainted under the bodies by push-pull; with no
floor at all the surrounding scene colour itself is inpainted. Bodies hidden under a glass body (overlaps) aren't
recovered.

Fluid (FLUID_FILTER): FLUID-kind pixels are splatted into an R16F field, blurred (separable Gaussian, sigma
FLUID_SIGMA_WORLD x zoom px, wrapping on periodic axes) and thresholded at FLUID_THRESHOLD, so neighbouring fluid
bodies merge into one liquid surface (gaps bridged, corners rounded); the field's gradient is its normal and its
value its thickness, feeding the same resolve.

Caustics (CausticsEffect): each light shoots CAUSTIC_RAYS rays through the march; points sampled along the rays
after they were refracted / reflected are splatted additively into shared["caustics"] (RGBA16F), one blur pass.
"""
import os

import moderngl
import numpy as np

import lights
import materials

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")

# ---- pseudo-3D refraction / reflection (render px)
REFRACT_DEPTH_PX = 10.0  # pseudo distance from a glass body's top surface down to the floor it shows
MIN_REFRACT_PX = 2.0  # the displacement on the tilted rim is at least this, so it reads at the low render res
THICK_PER_EDGE = 2.0  # glass thickness (Beer-Lambert path) = this x the distance to the nearest edge
VIEW_TILT = (0.0, -0.45)  # in-plane part of the pseudo view direction (z = -1): the eye leans toward -y (up)
MIRROR_DEPTH_PX = 24.0  # how far a reflected ray reaches in the plane per unit of R.xy / R.z
MIRROR_SOFT_PX = 3.0  # a reflection landing on its own body fades from the scene beside it to the blurred surroundings
GLINT_DIR = (-0.5, -0.6, 0.62)  # direction toward the fixed key light for glints (up-left and above the plane)
GLINT_STRENGTH = 0.6  # x material.specular
RATIO_EPS = 0.03  # light ratio (scene + eps) / (albedo + eps): keeps dark floors from blowing up
# ---- in-plane march (glass_common.march)
SDF_TRACE = True  # march mirror / glass-rim reflections in the plane when S's sdf is there (~+0.3 ms at 900x584)
MARCH_STEPS = 96
MARCH_BOUNCES = 3
MARCH_MAX_T = 320.0  # px
STEP_MIN_PX, STEP_MAX_PX = 1.0, 8.0  # sdf step clamp in air (max: so glass bodies outside the sdf aren't skipped)
STEP_NOSDF_PX = 2.0  # air step without an sdf (caustics only)
STEP_MEDIUM_PX = 1.5  # step inside glass when the sdf doesn't cover it
# ---- fluid filter
FLUID_FILTER = True
FLUID_SIGMA_WORLD = 10.0  # blur sigma, world units (x zoom = px): bridges the fluid preset's 6 unit gaps, hex holes
FLUID_SIGMA_PX = (1.0, 8.0)  # clamp
FLUID_SMOOTH_SIGMA_WORLD = 6.0  # second blur, of the thresholded liquid: its rim curvature (normals, thickness)
FLUID_THRESHOLD = 0.35  # field >= this is liquid (hex-packed octagons cover ~57%: their field stays above ~0.38)
FLUID_NORMAL_GAIN = 8.0  # smoothed liquid gradient (per px) -> normal.xy
FLUID_THICK_PX = 16.0  # thickness where the smoothed liquid is 1
FLUID_SEARCH = 12  # px a bridge pixel looks around (8 directions) for a fluid body, for its material
# ---- caustics
CAUSTICS = False  # CausticsEffect runs only when this (or its .enabled) is True
CAUSTIC_RAYS = 2048  # per light
CAUSTIC_SAMPLES = 48  # points per ray
CAUSTIC_MAX_T = 600.0  # px of ray path
CAUSTIC_GAIN = 1.0
CAUSTIC_STEPS = 192  # march steps per caustic sample (rays cross the whole view)
PUSHPULL_MIN = 8  # push-pull stops halving at this size (px): fewer, cheaper passes; holes this coarse are rare
KINDS_EVERY = 10  # runs between rescans of the world's material kinds (skips the passes a world doesn't need)

_FS_VERT = """#version 330 core
void main() {
    gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0);
}
"""


def _glsl(v):
    if isinstance(v, (tuple, list)):
        return f"vec{len(v)}({', '.join(_glsl(x) for x in v)})"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    return repr(float(v))


def _defines(**over):
    d = dict(KIND_OPAQUE=materials.OPAQUE, KIND_GLASS=materials.GLASS, KIND_MIRROR=materials.MIRROR,
             KIND_FLUID=materials.FLUID, MARCH_STEPS=MARCH_STEPS, MARCH_BOUNCES=MARCH_BOUNCES,
             MARCH_MAX_T=float(MARCH_MAX_T), STEP_MIN_PX=STEP_MIN_PX, STEP_MAX_PX=STEP_MAX_PX,
             STEP_NOSDF_PX=STEP_NOSDF_PX, STEP_MEDIUM_PX=STEP_MEDIUM_PX, RATIO_EPS=RATIO_EPS,
             REFRACT_DEPTH_PX=REFRACT_DEPTH_PX, MIN_REFRACT_PX=MIN_REFRACT_PX, THICK_PER_EDGE=THICK_PER_EDGE,
             MIRROR_DEPTH_PX=MIRROR_DEPTH_PX, MIRROR_SOFT_PX=MIRROR_SOFT_PX, VIEW3=(VIEW_TILT[0], VIEW_TILT[1], -1.0),
             GLINT_DIR=GLINT_DIR, GLINT_STRENGTH=GLINT_STRENGTH, FLUID_NORMAL_GAIN=FLUID_NORMAL_GAIN, FLUID_THICK_PX=FLUID_THICK_PX,
             FLUID_SEARCH=FLUID_SEARCH)
    d.update(over)
    return "".join(f"#define {k} {_glsl(v)}\n" for k, v in d.items())


def _read(name):
    with open(os.path.join(SHADER_DIR, name)) as f:
        return f.read()


def material_include():
    """F's shaders/material.glsl (mat_fetch) once it exists, else the provisional stand-in."""
    real = os.path.exists(os.path.join(SHADER_DIR, "material.glsl"))
    return "material.glsl" if real else "glass_mat_provisional.glsl"


def sdf_include():
    """S's shaders/sdf_shadow.glsl (sdf_at) once it exists, else the stand-in."""
    return "sdf_shadow.glsl" if os.path.exists(os.path.join(SHADER_DIR, "sdf_shadow.glsl")) else "glass_sdf_stub.glsl"


def program(ctx, frag, vert=None, sdf=False, **defines):
    """Compile a G program: #version, the constants (`defines` override them), [USE_SDF + sdf_at], mat_fetch,
    glass_common.glsl, then the shader (a fullscreen triangle vertex shader unless `vert`)."""
    head = "#version 330 core\n" + _defines(**defines)
    if sdf:
        head += "#define USE_SDF\n" + _read(sdf_include()) + "\n"
    head += _read(material_include()) + "\n" + _read("glass_common.glsl") + "\n"
    return ctx.program(vertex_shader=head + _read(vert) if vert else _FS_VERT, fragment_shader=head + _read(frag))


def _set(prog, **values):
    for k, v in values.items():
        if k in prog:
            prog[k].value = v


class _Pool:
    """Textures / framebuffers by key, remade when their size changes."""

    def __init__(self, ctx):
        self.ctx, self.items = ctx, {}

    def tex(self, key, size, comps, dtype, linear=False):
        t = self.items.get(key)
        if t is None or t.size != tuple(size) or t.components != comps or t.dtype != dtype:
            if t is not None:
                for k in [k for k in self.items if isinstance(k, tuple) and k[:1] == ("fbo",) and t.glo in k[1:]]:
                    self.items.pop(k).release()
                t.release()
            t = self.ctx.texture(tuple(size), comps, dtype=dtype)
            f = moderngl.LINEAR if linear else moderngl.NEAREST
            t.filter = (f, f)
            t.repeat_x = t.repeat_y = False
            self.items[key] = t
        return t

    def fbo(self, *texs):
        key = ("fbo",) + tuple(t.glo for t in texs)
        f = self.items.get(key)
        if f is None:
            f = self.items[key] = self.ctx.framebuffer(list(texs))
        return f

    def release(self):
        for v in self.items.values():
            v.release()
        self.items = {}


class _Kinds:
    """Which material kinds a world uses (rescanned every KINDS_EVERY runs); None world = all of them."""

    def __init__(self):
        self.key, self.kinds, self.age = None, None, 0

    def of(self, world):
        if world is None or not hasattr(world, "bodies"):
            return {materials.OPAQUE, materials.GLASS, materials.MIRROR, materials.FLUID}
        key = (id(world), len(world.bodies))
        self.age += 1
        if key != self.key or self.age >= KINDS_EVERY:
            self.key, self.age = key, 0
            self.kinds = {materials.of_body(b).kind for b in world.bodies}
        return self.kinds


def _common(prog, frame, sdf_tex=None):
    """Bind the G-buffer, material table and sdf to units 0..4 and set the shared uniforms."""
    gb = frame.gbuf
    gb["albedo"].use(0)
    gb["normal"].use(1)
    gb["mat"].use(2)
    frame.material_tex.use(3)
    if sdf_tex is not None:
        sdf_tex.use(4)
    period = tuple(float(p or 0.0) for p in (frame.period or (0.0, 0.0)))
    _set(prog, u_albedo=0, u_normal=1, u_mat=2, u_table=3, u_sdf=4, u_clip=tuple(int(c) for c in frame.clip),
         u_period=period)


def _restore(fbo):
    """Re-bind the framebuffer that was current before an effect ran (if it still exists)."""
    if fbo is not None and not isinstance(fbo.mglo, moderngl.InvalidObject):
        fbo.use()


class _Timer:
    """GPU ms per stage (GL time-elapsed queries, reused from `queries`) when `on`; free otherwise."""

    def __init__(self, ctx, on, queries):
        self.ctx, self.on, self.ms, self.queries, self.i, self.q = ctx, on, {}, queries, 0, None
        self._start()

    def _start(self):
        if self.on:
            if self.i == len(self.queries):
                self.queries.append(self.ctx.query(time=True))
            self.q = self.queries[self.i]
            self.i += 1
            self.q.__enter__()

    def _stop(self):
        self.q.__exit__(None, None, None)
        self.q, q = None, self.q
        return q.elapsed / 1e6

    def lap(self, name):
        if self.on:
            self.ms[name] = self.ms.get(name, 0.0) + self._stop()
            self._start()

    def done(self):
        if self.on and self.q is not None:
            self._stop()
        return self.ms


class GlassEffect:
    """Order 400: the glass / fluid / mirror resolve over the composed scene, written into GLWorld.tex."""
    order = 400

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.pool = _Pool(ctx)
        self.progs = {}
        self.vaos = {}
        self.kinds = _Kinds()
        self.profile = False  # True: GPU ms per stage in self.timings (time-elapsed queries)
        self.timings = {}
        self.queries = []
        self.sdf_trace = SDF_TRACE
        self.fluid_filter = FLUID_FILTER

    def _prog(self, frag, sdf=False, **defines):
        key = (frag, sdf, tuple(sorted(defines.items())))
        if key not in self.progs:
            p = self.progs[key] = program(self.ctx, frag, sdf=sdf, **defines)
            self.vaos[key] = self.ctx.vertex_array(p, [])
        return self.progs[key], self.vaos[key]

    def _draw(self, frag, fbo, sdf=False, **defines):
        prog, vao = self._prog(frag, sdf, **defines)
        fbo.use()
        vao.render(moderngl.TRIANGLES, vertices=3)

    def _floor(self, gw, frame):
        """The unlit floor under the bodies: F's gbuf["behind"], else GLWorld's base pass redrawn, else None."""
        behind = frame.gbuf.get("behind")
        if behind is not None:
            return behind
        progs, vao = getattr(gw, "progs", None), getattr(gw, "base_vao", None)
        if not progs or "base" not in progs or vao is None or not hasattr(gw, "grid_tex"):
            return None
        tex = self.pool.tex("floor", gw.size, 4, "f1")
        for unit, name in enumerate(("x", "y", "cells")):
            gw.grid_tex[name].use(unit)
        gw.trail_tex.use(3)
        self.pool.fbo(tex).use()
        vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        return tex

    def _fluid(self, frame):
        """Half-res fluid field passes -> (field, smoothed liquid, colour) textures, all LINEAR."""
        clip = tuple(int(c) for c in frame.clip)
        hs = ((clip[0] + 1) // 2, (clip[1] + 1) // 2)
        hp = tuple(float(p or 0.0) / 2 for p in (frame.period or (0.0, 0.0)))
        t = lambda name, comps: self.pool.tex(name, hs, comps, "f2", linear=True)
        a, b = (t("field_a", 1), t("fcol_a", 4)), (t("field_b", 1), t("fcol_b", 4))
        c, d = t("field_c", 1), t("field_d", 1)
        prog, _ = self._prog("fluid_splat.frag")
        _common(prog, frame)
        self._draw("fluid_splat.frag", self.pool.fbo(*a))
        zoom = float(frame.zoom)
        sigma = float(np.clip(FLUID_SIGMA_WORLD * zoom, *FLUID_SIGMA_PX)) / 2
        smooth = float(np.clip(FLUID_SMOOTH_SIGMA_WORLD * zoom, *FLUID_SIGMA_PX)) / 2
        for field_only, passes in ((False, ((a, b, (1, 0), sigma, -1.0), (b, a, (0, 1), sigma, -1.0))),
                                   (True, ((a[0], c, (1, 0), smooth, FLUID_THRESHOLD), (c, d, (0, 1), smooth, -1.0)))):
            defs = {"FIELD_ONLY": 1} if field_only else {}
            prog, _ = self._prog("fluid_blur.frag", **defs)
            _common(prog, frame)
            for src, dst, axis, sg, st in passes:
                if field_only:
                    src.use(5)
                else:
                    src[0].use(5)
                    src[1].use(6)
                _set(prog, u_src=5, u_src_col=6, u_dir=axis, u_sigma=sg, u_step=float(st), u_hsize=hs, u_hperiod=hp)
                self._draw("fluid_blur.frag", self.pool.fbo(*(dst if not field_only else (dst,))), **defs)
        return a[0], d, a[1]  # field (thresholded in the resolve), smoothed liquid (normal, thickness), colour

    def _pushpull(self, frame, scene):
        w, h = ((int(c) + 1) // 2 for c in frame.clip)
        sizes = []  # quarter res first (glass_pp_down's u_first), then halving
        while True:
            w, h = (w + 1) // 2, (h + 1) // 2
            sizes.append((w, h))
            if max(w, h) <= PUSHPULL_MIN:
                break
        own = [(self.pool.tex(("pr", k), s, 4, "f2"), self.pool.tex(("pc", k), s, 4, "f2"))
               for k, s in enumerate(sizes)]
        prog, _ = self._prog("glass_pp_down.frag")
        _common(prog, frame)
        scene.use(5)
        _set(prog, u_scene=5, u_prev_ratio=6, u_prev_col=7)
        for k in range(len(sizes)):
            _set(prog, u_first=k == 0)
            if k:
                own[k - 1][0].use(6)
                own[k - 1][1].use(7)
            else:  # unused on the first level: bound to something that isn't the target
                scene.use(6)
                scene.use(7)
            self._draw("glass_pp_down.frag", self.pool.fbo(*own[k]))
        prog, _ = self._prog("glass_pp_up.frag")
        _set(prog, u_own_ratio=5, u_own_col=6, u_coarse_ratio=7, u_coarse_col=8)
        filled = None
        for k in range(len(sizes) - 1, -1, -1):
            out = (self.pool.tex(("fr", k), sizes[k], 4, "f2", linear=True),
                   self.pool.tex(("fc", k), sizes[k], 4, "f2", linear=True))
            own[k][0].use(5)
            own[k][1].use(6)
            coarse = filled if filled is not None else own[k]
            coarse[0].use(7)
            coarse[1].use(8)
            _set(prog, u_has_coarse=filled is not None)
            self._draw("glass_pp_up.frag", self.pool.fbo(*out))
            filled = out
        return filled

    def _out_fbo(self, gw):
        return self.pool.fbo(gw.tex)

    def run(self, gw, frame):
        ctx = self.ctx
        kinds = self.kinds.of(getattr(frame, "world", None))
        if not kinds & {materials.GLASS, materials.MIRROR, materials.FLUID}:
            return
        prev = ctx.fbo
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        timer = _Timer(ctx, self.profile, self.queries)
        shared = frame.shared
        out = self._out_fbo(gw)
        scene = shared.get("scene_lit")
        if scene is None:  # no compose module: the resolve reads a copy of whatever is in GLWorld.tex
            scene = self.pool.tex("scene_copy", gw.size, 4, "f1")
            ctx.copy_framebuffer(scene, out)
        floor = self._floor(gw, frame)
        timer.lap("floor")
        sdf = shared.get("sdf") if self.sdf_trace else None
        field = None
        if self.fluid_filter and materials.FLUID in kinds:
            field = self._fluid(frame)
            timer.lap("fluid")
        fill_ratio, fill_col = self._pushpull(frame, scene)
        timer.lap("pushpull")
        prog, _ = self._prog("glass_resolve.frag", sdf=sdf is not None)
        _common(prog, frame, sdf)
        scene.use(5)
        (floor if floor is not None else scene).use(6)
        fill_ratio.use(7)
        fill_col.use(8)
        (field[0] if field is not None else fill_ratio).use(9)
        (field[2] if field is not None else fill_ratio).use(11)
        (field[1] if field is not None else fill_ratio).use(12)
        spec = shared.get("spec")
        (spec if spec is not None else scene).use(10)
        _set(prog, u_scene=5, u_behind=6, u_behind_on=floor is not None, u_fill_ratio=7, u_fill_col=8, u_field=9,
             u_fluid_col=11, u_fluid_smooth=12,
             u_fluid_on=field is not None, u_fluid_t=float(FLUID_THRESHOLD), u_spec=10, u_spec_on=spec is not None)
        self._draw("glass_resolve.frag", out, sdf=sdf is not None)
        timer.lap("resolve")
        self.timings = timer.done()
        _restore(prev)

    def release(self):
        for v in self.vaos.values():
            v.release()
        for p in self.progs.values():
            p.release()
        self.vaos, self.progs = {}, {}
        self.pool.release()


class CausticsEffect:
    """Order 250 (optional): light focused by glass / bent by mirrors -> shared["caustics"] (RGBA16F), which L's
    compose adds to the diffuse light. Runs only when `enabled` (CAUSTICS) and the world has glass or mirrors."""
    order = 250

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.pool = _Pool(ctx)
        self.enabled = CAUSTICS
        self.kinds = _Kinds()
        self.progs, self.vaos = {}, {}
        self.profile = False
        self.timings = {}
        self.queries = []

    def _prog(self, name, sdf):
        key = (name, sdf)
        if key not in self.progs:
            if name == "rays":
                p = program(self.ctx, "caustics_splat.frag", vert="caustics_rays.vert", sdf=sdf,
                            MARCH_STEPS=CAUSTIC_STEPS)
            else:
                p = program(self.ctx, "caustics_blur.frag", sdf=sdf)
            self.progs[key] = p
            self.vaos[key] = self.ctx.vertex_array(p, [])
        return self.progs[key], self.vaos[key]

    def run(self, gw, frame):
        if not self.enabled:
            return
        world = getattr(frame, "world", None)
        kinds = self.kinds.of(world)
        if not kinds & {materials.GLASS, materials.MIRROR, materials.FLUID}:
            return
        ls = [lt for lt in lights.of_world(world)[0] if lt.enabled]
        if not ls:
            return
        ctx = self.ctx
        prev = ctx.fbo
        timer = _Timer(ctx, self.profile, self.queries)
        sdf = frame.shared.get("sdf")
        raw = self.pool.tex("raw", self.size, 4, "f2")
        res = self.pool.tex("caustics", self.size, 4, "f2")
        fbo = self.pool.fbo(raw)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        prog, vao = self._prog("rays", sdf is not None)
        _common(prog, frame, sdf)
        zoom = float(frame.zoom)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for lt in ls:
            kind = {lights.POINT: 0, lights.SPOT: 1, lights.DIRECTIONAL: 2}.get(lt.kind, 0)
            _set(prog, u_kind=kind, u_light_pos=tuple(float(c) * zoom for c in lt.pos),
                 u_light_dir=tuple(float(c) for c in lt.dir), u_cone=float(np.radians(lt.cone_outer)),
                 u_light_col=tuple(float(c) * lt.intensity * CAUSTIC_GAIN for c in lt.color),
                 u_radius_px=max(float(lt.radius) * zoom, 1.0), u_falloff=float(lt.falloff),
                 u_rays=CAUSTIC_RAYS, u_samples=CAUSTIC_SAMPLES, u_max_t=float(CAUSTIC_MAX_T),
                 u_size=tuple(float(s) for s in self.size))
            vao.render(moderngl.POINTS, vertices=CAUSTIC_RAYS * CAUSTIC_SAMPLES)
        ctx.disable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        timer.lap("rays")
        prog, vao = self._prog("blur", False)
        _common(prog, frame)
        raw.use(5)
        _set(prog, u_src=5)
        self.pool.fbo(res).use()
        vao.render(moderngl.TRIANGLES, vertices=3)
        timer.lap("blur")
        frame.shared["caustics"] = res
        self.timings = timer.done()
        _restore(prev)

    def release(self):
        for v in self.vaos.values():
            v.release()
        for p in self.progs.values():
            p.release()
        self.vaos, self.progs = {}, {}
        self.pool.release()


def make(ctx, size):
    return [CausticsEffect(ctx, size), GlassEffect(ctx, size)]
