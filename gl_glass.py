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

"Behind" a glass body: the G-buffer is one opaque layer, so scene_lit holds the glass body itself where it is. What
lies under it: gbuf["behind"] (F's pass A: background, grid, trails and every opaque body, with behind_nrm /
behind_mat) lit directly at each pixel like compose does (glass_behind.frag: albedo x (ambient + light + gi +
caustics) + emissive, tone mapped) when L's shared["light"] is there. Otherwise (no light pass, or an older G-buffer
without behind_nrm) the floor is gbuf["behind"] or GLWorld's own base pass run again into a texture of ours (a
fullscreen pass whose uniforms and textures are still bound from this frame), lit by the light ratio scene_lit /
albedo of the surrounding floor (background pixels) inpainted under the bodies by push-pull; with no floor at all the
surrounding scene colour itself is inpainted. The push-pull pyramid also gives mirrors their blurred surroundings. On
a periodic axis whose period is the view (PERIOD_WRAP_PX) the pyramid is toroidal, so it has no seam there.

Fluid (FLUID_FILTER): FLUID-kind pixels are splatted into an R16F field, blurred (separable Gaussian, sigma
FLUID_SIGMA_WORLD x zoom px, wrapping on periodic axes) and thresholded at FLUID_THRESHOLD, so neighbouring fluid
bodies merge into one liquid surface (gaps bridged, corners rounded); the field's gradient is its normal and its
value its thickness, feeding the same resolve. The liquid is lit by the light ratio scene_lit / albedo of the fluid
bodies' own interior pixels, splatted and blurred with the colour (so a spot over the pool lights it, not the shadowed
floor in the gaps). Liquid pixels are never dropped: a bridge pixel that finds no fluid body nearby takes the
world's first FLUID material. A fluid body the filter would erase only because it is small and alone (no liquid
around it at all) stays as itself. Filled views only (frame.filled): in the points view a fluid body's points and
springs resolve like any see-through pixels, so the mesh stays visible instead of melting into one blob.

Caustics (CausticsEffect): each light shoots CAUSTIC_RAYS evenly spaced rays, each marched once in a geometry shader
(caustics_rays.geom); the straight pieces of a ray's path in open air after it was refracted (/ reflected) are drawn
as additive lines into shared["caustics"] (RGBA16F), then a separable blur. Continuous lines, not points sampled along
the rays: no dotted / grainy rays, and deterministic, so stable from frame to frame. Filled views only (frame.filled).
The lights are lights.active's (sanitised); a point / spot light that reaches no glass / mirror / fluid body is
skipped, at most CAUSTIC_MAX_LIGHTS draw, and past two lights each shoots fewer rays (CAUSTIC_RAY_BUDGET).
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
BEHIND_DIRECT = True  # light F's behind layer at each pixel (glass_behind.frag); False: the push-pull light ratio
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
# A FLUID pixel under the threshold stays as its own body (bevel normal, own colour) when the smoothed liquid there is
# below this: a small drop or thin stream far from any liquid, which the blur alone would erase (while its shadow
# stays). Next to liquid it is still cut away, so a pool's rim stays smooth.
FLUID_ISOLATED = 0.02
FLUID_IOR_MAX = 2.0  # the liquid's ior is clamped to [1, this] (high ior on a wobbly surface only makes black crowns)
# ---- caustics
CAUSTICS = False  # CausticsEffect runs only when this (its .enabled) or the world's `caustics` attribute is True
CAUSTIC_RAYS = 4096  # per light
CAUSTIC_SEG_PX = 32.0  # a ray's line is split into pieces this long (the light's falloff is linear along each)
CAUSTIC_MAX_VERTS = 64  # geometry shader output per ray (a strip per air piece; 2 per piece on periodic maps)
# Caustic rays refract on smoothed rim normals (a pre-pass, caustics_normals.frag; glass_common.rim_normal), not the
# sdf gradient, whose pixel stairs (+-10 degrees and more) fan the light into streaks: the G-buffer bevel normals of
# the rim, averaged over +-CAUSTIC_NORMAL_SMOOTH_PX along the outline (smooths polygon facets too). 0 px: off.
CAUSTIC_NORMAL_SMOOTH_PX = 8.0
CAUSTIC_NORMAL_TAPS = 4  # samples each side for that average
CAUSTIC_NORMAL_DEPTH = 3  # px a tap looks inward past the outline for a rim pixel with a bevel normal
CAUSTIC_NORMAL_CORNER_DEG = 35.0  # taps more than this off the hit's own normal are left out (corners stay sharp)
CAUSTIC_RIM_BAND = 3.0  # rim pixels: within this many px of the body's edge
CAUSTIC_BLUR_PX = 3.0  # sigma of the separable blur over the deposited lines
CAUSTIC_MAX_T = 600.0  # px of ray path
CAUSTIC_GAIN = 1.0
CAUSTIC_MIRRORS = False  # also deposit light only mirrors bent (a flat mirror wall = a second, streaky light source)
CAUSTIC_STEPS = 192  # march steps per caustic ray (rays cross the whole view)
CAUSTIC_MAX_LIGHTS = 8  # at most this many lights cast caustics (the brightest), after culling
CAUSTIC_RAY_BUDGET = 2 * CAUSTIC_RAYS  # rays over all lights: past 2 lights each gets fewer (spacing-normalised)
CAUSTIC_MIN_RAYS = CAUSTIC_RAYS // 4  # but never fewer than this per light
PUSHPULL_MIN = 8  # push-pull stops halving at this size (px): fewer, cheaper passes; holes this coarse are rare
PERIOD_WRAP_PX = 2.0  # a periodic axis whose period is within this of the view's size wraps the push-pull pyramid
KINDS_EVERY = 10  # runs between rescans of the world's material kinds (skips the passes a world doesn't need)

_FS_VERT = """#version 330 core
void main() {
    gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0);
}
"""


def _glsl(v):
    if isinstance(v, (tuple, list)):
        return f"vec{len(v)}({', '.join(_glsl(x) for x in v)})"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


def _defines(**over):
    d = dict(MARCH_STEPS=MARCH_STEPS, MARCH_BOUNCES=MARCH_BOUNCES,
             MARCH_MAX_T=float(MARCH_MAX_T), STEP_MIN_PX=STEP_MIN_PX, STEP_MAX_PX=STEP_MAX_PX,
             STEP_NOSDF_PX=STEP_NOSDF_PX, STEP_MEDIUM_PX=STEP_MEDIUM_PX, RATIO_EPS=RATIO_EPS,
             REFRACT_DEPTH_PX=REFRACT_DEPTH_PX, MIN_REFRACT_PX=MIN_REFRACT_PX, THICK_PER_EDGE=THICK_PER_EDGE,
             MIRROR_DEPTH_PX=MIRROR_DEPTH_PX, MIRROR_SOFT_PX=MIRROR_SOFT_PX, VIEW3=(VIEW_TILT[0], VIEW_TILT[1], -1.0),
             GLINT_DIR=GLINT_DIR, GLINT_STRENGTH=GLINT_STRENGTH, FLUID_NORMAL_GAIN=FLUID_NORMAL_GAIN,
             FLUID_THICK_PX=FLUID_THICK_PX,
             FLUID_SEARCH=FLUID_SEARCH, FLUID_ISOLATED=float(FLUID_ISOLATED), FLUID_IOR_MAX=float(FLUID_IOR_MAX))
    d.update(over)
    return "".join(f"#define {k} {_glsl(v)}\n" for k, v in d.items())


def _read(name):
    with open(os.path.join(SHADER_DIR, name)) as f:
        return f.read()


_POINT_VERT = """#version 330 core
void main() {
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""


def program(ctx, frag, vert=None, sdf=False, geom=None, **defines):
    """Compile a G program: #version, the constants (`defines` override them), [USE_SDF + S's sdf_shadow.glsl],
    F's material.glsl (Mat, mat_fetch, KIND_*), glass_common.glsl, then the shader (a fullscreen triangle vertex
    shader unless `vert`; with a `geom` geometry shader, a bare point vertex shader). Own loader rather than
    FrameInfo.load: it needs the #defines and builds programs in make() before any frame exists."""
    head = "#version 330 core\n" + _defines(**defines)
    if sdf:
        head += "#define USE_SDF\n" + _read("sdf_shadow.glsl") + "\n"
    head += _read("material.glsl") + "\n" + _read("glass_common.glsl") + "\n"
    if geom:
        return ctx.program(vertex_shader=_POINT_VERT, geometry_shader=head + _read(geom),
                           fragment_shader=head + _read(frag))
    return ctx.program(vertex_shader=head + _read(vert) if vert else _FS_VERT, fragment_shader=head + _read(frag))


def _set(prog, **values):
    for k, v in values.items():
        if k in prog:
            prog[k].value = v


class _Pool:
    """Textures / framebuffers by key, remade when their size changes."""

    def __init__(self, ctx):
        self.ctx, self.items = ctx, {}

    def tex(self, key, size, comps, dtype, linear=False, repeat=(False, False)):
        """The texture `key` (made or remade to fit); `repeat` = (x, y) wrap modes, set when they change."""
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
        if (t.repeat_x, t.repeat_y) != tuple(bool(r) for r in repeat):
            t.repeat_x, t.repeat_y = (bool(r) for r in repeat)
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

    def of(self, world, gw=None):
        """gw: the GLWorld that drew the frame: its material table (gl_world _materials, rebuilt whenever a body's
        material changes) already lists every kind in use, so no rescan is needed."""
        if world is None or not hasattr(world, "bodies"):
            return {materials.OPAQUE, materials.GLASS, materials.MIRROR, materials.FLUID}
        mk = getattr(gw, "mat_kinds", None)
        if mk is not None and getattr(gw, "_mat_sig", None) is not None:
            if mk is not self.key:
                self.key, self.kinds = mk, {int(k) for k in mk[1:]}
            return self.kinds
        if isinstance(self.key, np.ndarray):
            self.key = None
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
         u_period=period, u_sdf_origin=tuple(frame.shared.get("sdf_origin", (0.0, 0.0))))


def _wrap(frame):
    """(x, y): whether the push-pull pyramid wraps on that axis: periodic, and the view shows exactly one period."""
    period = frame.period or (0.0, 0.0)
    return tuple(bool(float(p or 0.0) > 0.0 and abs(float(p) - float(c)) <= PERIOD_WRAP_PX)
                 for p, c in zip(period, frame.clip))


def _tone():
    """compose's tone mapping settings (L's gl_lights), for the lit behind layer: (on, knee)."""
    try:
        import gl_lights
    except ImportError:
        return False, 0.8
    return bool(getattr(gl_lights, "TONEMAP", False)), min(max(float(getattr(gl_lights, "TONE_KNEE", 0.8)), 0.0), 0.99)


def _exposure(world):
    try:
        import gl_lights
        return float(gl_lights.exposure(world))
    except (ImportError, AttributeError):
        return 1.0


def _first_fluid_id(gw, frame):
    """The material id of the first FLUID material in the table (0 if none): GLWorld's mat_kinds, else the table
    texture read back (harnesses)."""
    kinds = getattr(gw, "mat_kinds", None)
    if kinds is None:
        try:
            tab = np.frombuffer(frame.material_tex.read(), np.float32).reshape(-1, materials.TEXELS, 4)
            kinds = [-1] + [int(round(float(row[3][1]))) for row in tab[1:]]  # (materials.pack: kind = texel 3 .y)
        except Exception:  # noqa: BLE001 (an unreadable table: a search miss just stays unresolved)
            return 0
    hit = np.flatnonzero(np.asarray(kinds) == materials.FLUID)
    return int(hit[0]) if len(hit) else 0


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

    def _black(self):
        """A 1x1 black texture standing in for a missing shared light / gi / caustics."""
        t = self.pool.items.get("black")
        if t is None:
            t = self.pool.items["black"] = self.ctx.texture((1, 1), 4, np.zeros(4, np.float16).tobytes(), dtype="f2")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return t

    def _behind_lit(self, frame):
        """gbuf["behind"] lit at each pixel like compose (glass_behind.frag) -> an RGBA16F texture, or None when there
        is no behind layer (no see-through body drawn this frame) or no light pass (L's shared["light"]): then the
        resolve lights the floor by the push-pull light ratio, as before."""
        gb, sh = frame.gbuf, frame.shared
        if not BEHIND_DIRECT or sh.get("light") is None or any(
                gb.get(k) is None for k in ("behind", "behind_nrm", "behind_mat")):
            return None
        tex = self.pool.tex("behind_lit", tuple(gb["behind"].size), 4, "f2")
        prog, _ = self._prog("glass_behind.frag")
        _common(prog, frame)
        black = self._black()
        units = dict(u_behind=gb["behind"], u_behind_nrm=gb["behind_nrm"], u_behind_mat=gb["behind_mat"],
                     u_light=sh["light"], u_gi=sh.get("gi", black), u_caustics=sh.get("caustics", black))
        for unit, (name, t) in enumerate(units.items(), 5):
            t.use(unit)
            _set(prog, **{name: unit})
        world = getattr(frame, "world", None)
        tonemap, knee = _tone()
        _set(prog, u_ambient=tuple(float(c) for c in lights.of_world(world)[1]), u_tonemap=tonemap, u_knee=knee,
             u_exposure=_exposure(world))
        self._draw("glass_behind.frag", self.pool.fbo(tex))
        return tex

    def _fluid(self, frame, scene, spec):
        """Half-res fluid field passes -> (field, smoothed liquid, colour, light ratio) textures, all LINEAR."""
        clip = tuple(int(c) for c in frame.clip)
        hs = ((clip[0] + 1) // 2, (clip[1] + 1) // 2)
        hp = tuple(float(p or 0.0) / 2 for p in (frame.period or (0.0, 0.0)))
        t = lambda name, comps: self.pool.tex(name, hs, comps, "f2", linear=True)
        a = (t("field_a", 1), t("fcol_a", 4), t("fratio_a", 4))
        b = (t("field_b", 1), t("fcol_b", 4), t("fratio_b", 4))
        c, d = t("field_c", 1), t("field_d", 1)
        prog, _ = self._prog("fluid_splat.frag")
        _common(prog, frame)
        scene.use(5)
        (spec if spec is not None else scene).use(6)
        _set(prog, u_scene=5, u_spec=6, u_spec_on=spec is not None)
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
                    for unit, s in enumerate(src, 5):
                        s.use(unit)
                _set(prog, u_src=5, u_src_col=6, u_src_ratio=7, u_dir=axis, u_sigma=sg, u_step=float(st), u_hsize=hs,
                     u_hperiod=hp)
                self._draw("fluid_blur.frag", self.pool.fbo(*(dst if not field_only else (dst,))), **defs)
        # field (thresholded in the resolve), smoothed liquid (normal, thickness), colour, light ratio
        return a[0], d, a[1], a[2]

    def _pushpull(self, frame, scene, wrap):
        """Push-pull inpainting pyramid -> the quarter-res (light ratio, colour) fill, LINEAR. Toroidal on the axes
        where `wrap` (_wrap): the pull pass wraps its source texels, the push pass's bilinear taps repeat."""
        w, h = ((int(c) + 1) // 2 for c in frame.clip)
        sizes = []  # quarter res first (glass_pp_down's u_first), then halving
        while True:
            w, h = (w + 1) // 2, (h + 1) // 2
            sizes.append((w, h))
            if max(w, h) <= PUSHPULL_MIN:
                break
        own = [(self.pool.tex(("pr", k), s, 4, "f2", repeat=wrap), self.pool.tex(("pc", k), s, 4, "f2", repeat=wrap))
               for k, s in enumerate(sizes)]
        prog, _ = self._prog("glass_pp_down.frag")
        _common(prog, frame)
        scene.use(5)
        _set(prog, u_scene=5, u_prev_ratio=6, u_prev_col=7, u_wrap=wrap)
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
        _set(prog, u_own_ratio=5, u_own_col=6, u_coarse_ratio=7, u_coarse_col=8, u_wrap=wrap)
        filled = None
        for k in range(len(sizes) - 1, -1, -1):
            out = (self.pool.tex(("fr", k), sizes[k], 4, "f2", linear=True, repeat=wrap),
                   self.pool.tex(("fc", k), sizes[k], 4, "f2", linear=True, repeat=wrap))
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
        kinds = self.kinds.of(getattr(frame, "world", None), gw)
        if not kinds & {materials.GLASS, materials.MIRROR, materials.FLUID}:
            return
        # the points view draws every body with its opaque twin (gl_world) and has no liquid: the resolve would
        # discard every pixel, so the push-pull and behind passes before it are skipped too
        if not getattr(frame, "filled", True):
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
        direct = self._behind_lit(frame)
        raw = self._floor(gw, frame)  # (gbuf["behind"] itself when there is one: nothing drawn)
        floor = direct if direct is not None else raw
        timer.lap("floor")
        sdf = shared.get("sdf") if self.sdf_trace else None
        spec = shared.get("spec")
        field = None
        # the points view shows each body's points and springs: no liquid merged over (and between) them
        if self.fluid_filter and materials.FLUID in kinds and getattr(frame, "filled", True):
            field = self._fluid(frame, scene, spec)
            timer.lap("fluid")
        wrap = _wrap(frame)
        fill_ratio, fill_col = self._pushpull(frame, scene, wrap)
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
        (field[3] if field is not None else fill_ratio).use(13)
        (raw if raw is not None else scene).use(14)
        (spec if spec is not None else scene).use(10)
        emit_gain = _exposure(getattr(frame, "world", None))
        _set(prog, u_scene=5, u_behind=6, u_behind_on=floor is not None, u_behind_direct=direct is not None,
             u_floor=14,
             u_fill_ratio=7, u_fill_col=8, u_field=9, u_fluid_col=11, u_fluid_smooth=12, u_fluid_ratio=13,
             u_fluid_on=field is not None, u_fluid_t=float(FLUID_THRESHOLD), u_spec=10, u_spec_on=spec is not None,
             u_wrap=wrap, u_emit_gain=float(emit_gain),
             u_fluid_mat=_first_fluid_id(gw, frame) if field is not None else 0)
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
    compose adds to the diffuse light. Runs only when `enabled` (CAUSTICS) or world.caustics, and the world has glass
    or mirrors."""
    order = 250

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.pool = _Pool(ctx)
        self.enabled = CAUSTICS
        self.kinds = _Kinds()
        self._optical_key, self._optical = None, None  # per body: glass / mirror / fluid (_boxes)
        self.progs, self.vaos = {}, {}
        self.profile = False
        self.timings = {}
        self.queries = []

    def _prog(self, name, sdf):
        key = (name, sdf)
        if key not in self.progs:
            if name == "rays":
                rim = {"USE_RIM_TEX": 1} if self._smooth() else {}
                p = program(self.ctx, "caustics_splat.frag", geom="caustics_rays.geom", sdf=sdf,
                            MARCH_STEPS=CAUSTIC_STEPS, CAUSTIC_MIRRORS=CAUSTIC_MIRRORS, CAUSTIC_PIECES=1,
                            CAUSTIC_SEG_PX=float(CAUSTIC_SEG_PX), CAUSTIC_MAX_VERTS=CAUSTIC_MAX_VERTS, **rim)
            elif name == "normals":
                p = program(self.ctx, "caustics_normals.frag", sdf=sdf, RIM_SMOOTH_PX=float(CAUSTIC_NORMAL_SMOOTH_PX),
                            RIM_TAPS=CAUSTIC_NORMAL_TAPS, RIM_DEPTH=CAUSTIC_NORMAL_DEPTH,
                            RIM_CORNER=float(np.cos(np.radians(CAUSTIC_NORMAL_CORNER_DEG))),
                            RIM_BAND=float(CAUSTIC_RIM_BAND))
            else:
                p = program(self.ctx, "caustics_blur.frag", sdf=sdf, CAUSTIC_BLUR_PX=CAUSTIC_BLUR_PX)
            self.progs[key] = p
            self.vaos[key] = self.ctx.vertex_array(p, [])
        return self.progs[key], self.vaos[key]

    @staticmethod
    def _smooth():
        return CAUSTIC_NORMAL_SMOOTH_PX > 0 and CAUSTIC_NORMAL_TAPS > 0

    def _boxes(self, world, gw=None):
        """(lo, hi) world-unit AABBs (n, 2) of the world's glass / mirror / fluid bodies, or None when unknown (no
        point array: then no light is culled). Body kinds are rescanned with the kinds (KINDS_EVERY runs)."""
        bodies = getattr(world, "bodies", None)
        P, offsets = getattr(world, "P", None), getattr(world, "offsets", None)
        if not bodies or P is None or offsets is None or len(offsets) != len(bodies):
            return None
        bm, mk = getattr(gw, "body_mat", None), getattr(gw, "mat_kinds", None)
        key = (id(world), len(bodies))
        if bm is not None and mk is not None and len(bm) == len(bodies):  # gl_world's per-body material ids
            if self._optical_key is not bm:  # (a new array whenever a body's material changes)
                self._optical_key = bm
                self._optical = np.isin(mk[bm], (materials.GLASS, materials.MIRROR, materials.FLUID))
        elif isinstance(self._optical_key, np.ndarray) or self._optical_key != key or self.kinds.age == 0:
            self._optical_key = key
            optic = {materials.GLASS, materials.MIRROR, materials.FLUID}
            self._optical = np.array([materials.of_body(b).kind in optic for b in bodies], bool)
        if not self._optical.any():
            return np.zeros((0, 2)), np.zeros((0, 2))
        P = np.asarray(P, np.float64)
        starts = np.asarray(offsets, np.int64)
        lo = np.minimum.reduceat(P, starts, axis=1).T[self._optical]
        hi = np.maximum.reduceat(P, starts, axis=1).T[self._optical]
        return lo, hi

    def _lights(self, world, gw=None):
        """The lights that cast caustics: lights.active (sanitised, enabled); a point or spot light whose radius
        reaches no glass / mirror / fluid body's box is skipped (directional lights never); the CAUSTIC_MAX_LIGHTS
        brightest of the rest."""
        ls = lights.active(world)
        boxes = self._boxes(world, gw) if any(lt.kind != lights.DIRECTIONAL for lt in ls) else None
        if boxes is not None:
            lo, hi = boxes
            centre, half = (lo + hi) / 2, (hi - lo) / 2
            period = getattr(world, "period", None)
            period = np.array([float(p or 0.0) for p in period] if period is not None else [0.0, 0.0])
            keep = []
            for lt in ls:
                if lt.kind == lights.DIRECTIONAL:
                    keep.append(lt)
                    continue
                d = np.asarray(lt.pos, np.float64) - centre
                d = np.where(period > 0, d - period * np.round(d / np.where(period > 0, period, 1.0)), d)
                gap = np.hypot(*np.maximum(np.abs(d) - half, 0.0).T) if len(d) else np.zeros(0)
                if len(gap) and gap.min() <= float(lt.radius):
                    keep.append(lt)
            ls = keep
        if len(ls) > CAUSTIC_MAX_LIGHTS:
            ls = sorted(ls, key=lambda lt: -float(lt.intensity) * max(lt.color))[:CAUSTIC_MAX_LIGHTS]
        return ls

    def run(self, gw, frame):
        world = getattr(frame, "world", None)
        if not (self.enabled or getattr(world, "caustics", False)):
            return
        if not getattr(frame, "filled", True):  # points view: a wireframe of springs focuses no light (its 1 px
            return  # lines, refracting on flat normals, would only fan the rays into streaks)
        kinds = self.kinds.of(world, gw)
        if not kinds & {materials.GLASS, materials.MIRROR, materials.FLUID}:
            return
        ls = self._lights(world, gw)
        if not ls:
            return
        n_rays = int(min(CAUSTIC_RAYS, max(CAUSTIC_MIN_RAYS, CAUSTIC_RAY_BUDGET // len(ls))))
        ctx = self.ctx
        prev = ctx.fbo
        timer = _Timer(ctx, self.profile, self.queries)
        sdf = frame.shared.get("sdf")
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        rim = None
        if self._smooth():  # the smoothed rim normals the march refracts on
            rim = self.pool.tex("rim", self.size, 2, "f2")
            prog, vao = self._prog("normals", sdf is not None)
            _common(prog, frame, sdf)
            self.pool.fbo(rim).use()
            vao.render(moderngl.TRIANGLES, vertices=3)
            timer.lap("normals")
        raw = self.pool.tex("raw", self.size, 4, "f2")
        res = self.pool.tex("caustics", self.size, 4, "f2")
        fbo = self.pool.fbo(raw)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        prog, vao = self._prog("rays", sdf is not None)
        _common(prog, frame, sdf)
        if rim is not None:
            rim.use(13)
            _set(prog, u_rim=13)
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
                 u_rays=n_rays, u_max_t=float(CAUSTIC_MAX_T),
                 u_size=tuple(float(s) for s in self.size))
            vao.render(moderngl.POINTS, vertices=n_rays)
        ctx.disable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        timer.lap("rays")
        prog, vao = self._prog("blur", False)
        _common(prog, frame)
        tmp = self.pool.tex("tmp", self.size, 4, "f2")
        for src, dst, axis in ((raw, tmp, (1, 0)), (tmp, res, (0, 1))):
            src.use(5)
            _set(prog, u_src=5, u_dir=axis)
            self.pool.fbo(dst).use()
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
