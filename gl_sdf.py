"""Signed distance field, soft shadows and global illumination for the lit renderer (research/lighting_contract.md).

make(ctx, size) -> [SdfEffect (order 100), GiEffect (order 150)], gl_effects' plugin API.

SdfEffect publishes frame.shared["sdf"]: R32F, the render size, LINEAR filtered, the signed distance in render px
from each pixel centre to the nearest boundary of a shadow-casting occluder (G-buffer `normal` flags bit0 and bit1
both set), negative inside, minimum-image across frame.period. Built by jump flooding in fragment ping-pong passes
(GL 4.1, no compute): a seed pass (pixels on either side of a boundary store their own coordinates, RG16I),
ceil(log2(max(w, h))) step passes with steps 2^(k-1) .. 1 plus one more step-1 pass (JFA+1), then a distance pass.
Accuracy: within ~0.7 px of the true distance to the rasterised outline. shaders/sdf_shadow.glsl has the
shadow march lights use (sdf_shadow / sdf_at). shared["sdf"] holds every shadow caster, see-through ones included
(glass refraction and caustics march it to find bodies).

Translucent shadows: see-through casters (glass / fluid with transparency > 0: G-buffer FLAG_TRANSLUCENT, bit4)
let light through. On frames that have them (a scan of the world's materials, redone when gl_world's material
signature changes), in the filled view, with a light that casts shadows and gl_lights.tr_scale > 0 (caustics off,
or gl_lights.CAUSTIC_SHARE < 1), SdfEffect also publishes, all at the render size:
  shared["sdf_opaque"] (R32F, LINEAR): the same field over the opaque casters only;
  shared["sdf_tr"] (R32F, LINEAR): over the translucent casters only;
  shared["sdf_tint"] (RGBA8, LINEAR): per texel, the transmittance of the nearest translucent caster
    (material.glsl mat_transmittance: transparency x Beer-Lambert toward its colour);
from a split pass (sdf_tr.frag) and one more jump flood over both masks (jfa2_*.frag). gl_lights then shadows with sdf_opaque x sdf_shadow_tr.
Frames without them cost nothing extra and render exactly as before.

GiEffect (radiance cascades, an experiment) publishes frame.shared["gi"]: RGBA16F at the render size, the light
emissive bodies throw on everything around them (occluded by shadow casters), to add to the diffuse light. It
runs while GI_ENABLED is True (module flag, default False, read every frame) or the world has `gi` set (a per-world
switch, saved in scenes; the Lights card's GI button toggles it), and then costs: a half-res scene pass, its own
half-res jump flood (emitters + occluders), N_CASCADES cascade passes and an upsampling pass. GI_GAIN, times the
world's own `gi_gain` (if any; saved in scenes, set by a map's "lighting" block), scales the result (gi_gain():
clamped to [0, MAX_GAIN], a non-finite gain counts as 1; GI_EMIT_GAIN likewise). Unlit materials emit nothing.
On periodic maps the GI wraps seamlessly: the probe grid runs two top-cascade probe spacings past the period, and
cascade merges and the upsampling take any footprint that would leave the grid one period over.
Translucent casters (FLAG_TRANSLUCENT) are partial occluders for GI: rc_scene stores their transmittance in the
scene texel (a = 1 + T) and a cascade ray entering one goes on through it with its radiance x T (rc_cascade).

Periodic maps: frame.period (render px; an axis with 0 doesn't repeat) is assumed to start at pixel 0; lookups
wrap into [0, period) and distances are minimum-image.
"""
import math

import moderngl

import lights as lights_mod
import materials

GI_ENABLED = False  # radiance cascades GI on / off (the UI may toggle this; read every frame)
GI_GAIN = 1.0  # multiplies shared["gi"]
GI_EMIT_GAIN = 1.0  # multiplies the emission the cascades start from
N_CASCADES = 6
GI_L0 = 1.0  # cascade 0's ray interval, half-res texels (cascade i's is 4^i times longer)

FULLSCREEN_VERT = "world_base.vert"  # gl_world's fullscreen triangle strip from gl_VertexID


def _period(period, scale=1.0):
    """frame.period (or None) -> a vec2 uniform value, 0 on axes that don't repeat."""
    if period is None:
        return 0.0, 0.0
    return tuple(max(float(p or 0.0), 0.0) / scale for p in period)


MAX_GAIN = 50.0  # GI gains are clamped to [0, MAX_GAIN]


def _gain(v):
    """A gain as a float in [0, MAX_GAIN]; 1 when it isn't a finite number."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    return min(max(v, 0.0), MAX_GAIN) if math.isfinite(v) else 1.0


def gi_gain(world):
    """The effective GI gain: GI_GAIN x world.gi_gain (1 when unset), each made finite, the product in
    [0, MAX_GAIN]."""
    g = getattr(world, "gi_gain", None)
    return min(_gain(GI_GAIN) * (1.0 if g is None else _gain(g)), MAX_GAIN)


def _set(prog, name, value):
    if name in prog:
        prog[name].value = value


def _tex(ctx, size, comps, dtype, linear=False):
    t = ctx.texture(size, comps, dtype=dtype)
    f = moderngl.LINEAR if linear else moderngl.NEAREST
    t.filter = (f, f)
    t.repeat_x = t.repeat_y = False
    return t


class JumpFlood:
    """A signed distance field of a mask texture, by jump flooding. build() returns `dist` (R32F, the mask's size)."""

    def __init__(self, ctx, size, linear):
        self.ctx, self.size = ctx, tuple(size)
        self.seeds = [_tex(ctx, self.size, 2, "i2") for _ in range(2)]
        self.seed_fbos = [ctx.framebuffer([t]) for t in self.seeds]
        self.dist = _tex(ctx, self.size, 1, "f4", linear=linear)
        self.dist_fbo = ctx.framebuffer([self.dist])
        n = max(1, math.ceil(math.log2(max(self.size))))
        self.steps = [1 << k for k in range(n - 1, -1, -1)] + [1]

    def build(self, frame, progs, src, mode, period):
        """src: the mask texture (unit 0); mode: sdf_common.glsl's sdf_solid mode; period: vec2 in texels."""
        seed, step, dist = progs["seed"], progs["step"], progs["dist"]
        src.use(0)
        _set(seed, "u_src", 0)
        _set(seed, "u_mode", mode)
        _set(seed, "u_period", period)
        self.seed_fbos[0].use()
        frame.fullscreen(seed)
        cur = 0
        _set(step, "u_seeds", 0)
        _set(step, "u_period", period)
        for s in self.steps:
            self.seeds[cur].use(0)
            _set(step, "u_step", s)
            self.seed_fbos[1 - cur].use()
            frame.fullscreen(step)
            cur = 1 - cur
        self.seeds[cur].use(0)
        src.use(1)
        _set(dist, "u_seeds", 0)
        _set(dist, "u_src", 1)
        _set(dist, "u_mode", mode)
        _set(dist, "u_period", period)
        self.dist_fbo.use()
        frame.fullscreen(dist)
        return self.dist

    def release(self):
        for o in self.seed_fbos + self.seeds + [self.dist_fbo, self.dist]:
            o.release()


class SplitFlood:
    """SdfEffect's translucent fields: the split pass (sdf_tr.frag: translucent casters' transmittance, opaque
    casters) and one jump flood for both masks at once (jfa2_*.frag, RGBA16I seeds: the passes' texel fetches are
    shared). build() -> (opaque distance, translucent distance, tint), all LINEAR at the render size."""

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.tr_src, self.opq_src = _tex(ctx, self.size, 4, "f1"), _tex(ctx, self.size, 4, "f1")
        self.split_fbo = ctx.framebuffer([self.tr_src, self.opq_src])
        self.seeds = [_tex(ctx, self.size, 4, "i2") for _ in range(2)]
        self.seed_fbos = [ctx.framebuffer([t]) for t in self.seeds]
        self.opq = _tex(ctx, self.size, 1, "f4", linear=True)
        self.tr = _tex(ctx, self.size, 1, "f4", linear=True)
        self.tint = _tex(ctx, self.size, 4, "f1", linear=True)
        self.dist_fbo = ctx.framebuffer([self.opq, self.tr, self.tint])
        n = max(1, math.ceil(math.log2(max(self.size))))
        self.steps = [1 << k for k in range(n - 1, -1, -1)] + [1]

    @staticmethod
    def progs(frame):
        inc = ["sdf_common.glsl"]
        return {"split": frame.load(FULLSCREEN_VERT, "sdf_tr.frag", ["material.glsl"]),
                "seed": frame.load(FULLSCREEN_VERT, "jfa2_seed.frag", inc),
                "step": frame.load(FULLSCREEN_VERT, "jfa2_step.frag", inc),
                "dist": frame.load(FULLSCREEN_VERT, "jfa2_dist.frag", inc)}

    def build(self, frame, progs, period):
        g = frame.gbuf
        sp = progs["split"]
        for unit, (name, tex) in enumerate((("u_normal", g["normal"]), ("u_albedo", g["albedo"]),
                                            ("u_mat", g["mat"]), ("u_mats", frame.material_tex))):
            tex.use(unit)
            _set(sp, name, unit)
        self.split_fbo.use()
        frame.fullscreen(sp)
        seed, step, dist = progs["seed"], progs["step"], progs["dist"]
        self.opq_src.use(0)
        self.tr_src.use(1)
        _set(seed, "u_src0", 0)
        _set(seed, "u_src1", 1)
        _set(seed, "u_period", period)
        self.seed_fbos[0].use()
        frame.fullscreen(seed)
        cur = 0
        _set(step, "u_seeds", 0)
        _set(step, "u_period", period)
        for s in self.steps:
            self.seeds[cur].use(0)
            _set(step, "u_step", s)
            self.seed_fbos[1 - cur].use()
            frame.fullscreen(step)
            cur = 1 - cur
        self.seeds[cur].use(0)
        self.opq_src.use(1)
        self.tr_src.use(2)
        _set(dist, "u_seeds", 0)
        _set(dist, "u_src0", 1)
        _set(dist, "u_src1", 2)
        _set(dist, "u_period", period)
        self.dist_fbo.use()
        frame.fullscreen(dist)
        return self.opq, self.tr, self.tint

    def release(self):
        for o in [self.split_fbo, self.tr_src, self.opq_src, *self.seed_fbos, *self.seeds, self.dist_fbo,
                  self.opq, self.tr, self.tint]:
            o.release()


def _jfa_progs(frame):
    inc = ["sdf_common.glsl"]
    return {"seed": frame.load(FULLSCREEN_VERT, "jfa_seed.frag", inc),
            "step": frame.load(FULLSCREEN_VERT, "jfa_step.frag", inc),
            "dist": frame.load(FULLSCREEN_VERT, "sdf_dist.frag", inc)}


def translucent_caster(m):
    """Does a Material cast a see-through shadow (material.glsl mat_flags' FLAG_TRANSLUCENT)?"""
    try:
        return bool(m.casts_shadow) and m.kind in (materials.GLASS, materials.FLUID) and float(m.transparency) > 0
    except (TypeError, ValueError, AttributeError):
        return False


class _Translucent:
    """Has a world any translucent shadow caster? Rescanned when gw's material signature (gl_world._materials'
    _mat_sig, replaced whenever a body or its material changes) is a new object, or every frame without one."""

    def __init__(self):
        self.sig, self.world, self.value = None, None, False

    def of(self, gw, world):
        if world is None:
            return False
        sig = getattr(gw, "_mat_sig", None)
        if sig is not None and sig is self.sig and world is self.world:
            return self.value
        self.value = any(translucent_caster(materials.of_body(b)) for b in getattr(world, "bodies", ()))
        self.sig, self.world = sig, world
        return self.value


def _shadow_lights(world):
    """Is any light drawn that casts shadows?"""
    try:
        return any(lt.casts_shadows for lt in (lights_mod.sanitize(x) for x in lights_mod.of_world(world)[0])
                   if lt is not None)
    except Exception:  # (a broken lights list: no translucent pass, plain shadows as before)
        return False


class SdfEffect:
    """shared["sdf"] from the G-buffer's shadow-casting occluders; with translucent casters on screen also
    shared["sdf_opaque"], shared["sdf_tr"] and shared["sdf_tint"] (module docs)."""
    order = 100

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.jfa = None
        self.progs = None
        self.translucent = _Translucent()
        self.split = None  # SplitFlood, made on the first frame that needs it
        self.split_progs = None

    def _wants_tr(self, gw, frame):
        world = getattr(frame, "world", None)
        if world is None or not getattr(frame, "filled", True):  # (points view: no see-through silhouettes)
            return False
        if not self.translucent.of(gw, world) or not _shadow_lights(world):
            return False
        import gl_lights  # (lazily: keeps the two modules' import order free)
        return gl_lights.tr_scale(gw, frame) > 0.0


    def run(self, gw, frame):
        size = tuple(frame.size)
        if self.progs is None:
            self.progs = _jfa_progs(frame)
        if self.jfa is None or self.jfa.size != size:
            if self.jfa is not None:
                self.jfa.release()
            self.jfa = JumpFlood(self.ctx, size, linear=True)
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        period = _period(frame.period)
        frame.shared["sdf"] = self.jfa.build(frame, self.progs, frame.gbuf["normal"], 0, period)
        if not self._wants_tr(gw, frame):
            return
        if self.split_progs is None:
            self.split_progs = SplitFlood.progs(frame)
        if self.split is None or self.split.size != size:
            if self.split is not None:
                self.split.release()
            self.split = SplitFlood(self.ctx, size)
        sh = frame.shared
        sh["sdf_opaque"], sh["sdf_tr"], sh["sdf_tint"] = self.split.build(frame, self.split_progs, period)

    def release(self):
        for o in (self.jfa, self.split):
            if o is not None:
                o.release()
        for p in [*(self.progs or {}).values(), *(self.split_progs or {}).values()]:
            p.release()
        self.jfa = self.progs = self.split = self.split_progs = None


class GiEffect:
    """shared["gi"] by radiance cascades, while GI_ENABLED or world.gi."""
    order = 150

    def __init__(self, ctx, size):
        self.ctx, self.size, self.period = ctx, None, None
        self.progs = None
        self.textures = []
        self.stats = {}

    def _alloc(self, size, period):
        for o in self.textures:
            o.release()
        ctx = self.ctx
        m = 1 << (N_CASCADES - 1)
        half = ((size[0] + 1) // 2, (size[1] + 1) // 2)
        # E: half res padded to a multiple of 2^(n-1); on a periodic axis at least 2 top-cascade probe spacings more
        # than the period, so every probe near a seam finds its upper cascade's 2x2 footprint whole one period over
        # (probes past the period are valid: their marches wrap) and rc_out can wrap into [0.5, period + 0.5)
        need = [max(h, int(math.ceil(p)) + 2 * m) if p > 0 else h for h, p in zip(half, period)]
        self.grid = tuple(-(-n // m) * m for n in need)
        self.scene = _tex(ctx, self.grid, 4, "f2")
        self.scene_fbo = ctx.framebuffer([self.scene])
        self.casc = [_tex(ctx, self.grid, 4, "f2", linear=True) for _ in range(2)]
        self.casc_fbos = [ctx.framebuffer([t]) for t in self.casc]
        self.out = _tex(ctx, size, 4, "f2", linear=True)
        self.out_fbo = ctx.framebuffer([self.out])
        self.jfa = JumpFlood(ctx, self.grid, linear=False)
        self.textures = [self.scene_fbo, self.scene, *self.casc_fbos, *self.casc, self.out_fbo, self.out]
        self.size, self.period = size, period

    def run(self, gw, frame):
        world = getattr(frame, "world", None)
        if not (GI_ENABLED or getattr(world, "gi", False)):
            return
        size = tuple(frame.size)
        if self.progs is None:
            inc = ["sdf_common.glsl"]
            self.progs = dict(_jfa_progs(frame),
                              scene=frame.load(FULLSCREEN_VERT, "rc_scene.frag", ["material.glsl", *inc]),
                              cascade=frame.load(FULLSCREEN_VERT, "rc_cascade.frag", inc),
                              out=frame.load(FULLSCREEN_VERT, "rc_out.frag", inc))
        period = _period(frame.period, 2.0)
        if self.size != size or self.period != period:
            if self.size is not None:
                self.jfa.release()
            self._alloc(size, period)
        ctx = self.ctx
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        p = self.progs
        # scene at half res
        g = frame.gbuf
        for unit, tex in enumerate((g["normal"], g["albedo"], g["mat"], frame.material_tex)):
            tex.use(unit)
        for unit, name in enumerate(("u_normal", "u_albedo", "u_mat", "u_mats")):
            _set(p["scene"], name, unit)
        _set(p["scene"], "u_emit_gain", _gain(GI_EMIT_GAIN))
        self.scene_fbo.use()
        frame.fullscreen(p["scene"])
        dist = self.jfa.build(frame, p, self.scene, 1, period)
        # cascades, top down, ping-ponging between two textures
        c = p["cascade"]
        self.scene.use(0)
        dist.use(1)
        _set(c, "u_scene", 0)
        _set(c, "u_dist", 1)
        _set(c, "u_upper", 2)
        _set(c, "u_l0", float(GI_L0))
        _set(c, "u_period", period)
        cur = 0
        for i in range(N_CASCADES - 1, -1, -1):
            self.casc[1 - cur].use(2)  # the cascade above (unused for the top one)
            _set(c, "u_has_upper", i < N_CASCADES - 1)
            _set(c, "u_index", i)
            self.casc_fbos[cur].use()
            frame.fullscreen(c)
            cur = 1 - cur
        top = self.casc[1 - cur]  # cascade 0
        o = p["out"]
        top.use(0)
        dist.use(1)
        g["mat"].use(2)
        _set(o, "u_c0", 0)
        _set(o, "u_dist", 1)
        _set(o, "u_mat", 2)
        _set(o, "u_period", period)
        _set(o, "u_gain", gi_gain(world))
        self.out_fbo.use()
        frame.fullscreen(o)
        frame.shared["gi"] = self.out

    def release(self):
        for o in self.textures:
            o.release()
        if self.size is not None:
            self.jfa.release()
        for p in (self.progs or {}).values():
            p.release()
        self.textures, self.progs, self.size, self.period = [], None, None, None


def make(ctx, size):
    return [SdfEffect(ctx, size), GiEffect(ctx, size)]
