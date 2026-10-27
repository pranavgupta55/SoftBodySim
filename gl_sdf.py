"""Signed distance field, soft shadows and global illumination for the lit renderer (research/lighting_contract.md).

make(ctx, size) -> [SdfEffect (order 100), GiEffect (order 150)], gl_effects' plugin API.

SdfEffect publishes frame.shared["sdf"]: R32F, the render size, LINEAR filtered, the signed distance in render px
from each pixel centre to the nearest boundary of a shadow-casting occluder (G-buffer `normal` flags bit0 and bit1
both set), negative inside, minimum-image across frame.period. Built by jump flooding in fragment ping-pong passes
(GL 4.1, no compute): a seed pass (pixels on either side of a boundary store their own coordinates, RG16I),
ceil(log2(max(w, h))) step passes with steps 2^(k-1) .. 1 plus one more step-1 pass (JFA+1), then a distance pass.
Accuracy: within ~0.7 px of the true distance to the rasterised outline. shaders/sdf_shadow.glsl has the
shadow march lights use (sdf_shadow / sdf_at).

GiEffect (radiance cascades, an experiment) publishes frame.shared["gi"]: RGBA16F at the render size, the light
emissive bodies throw on everything around them (occluded by shadow casters), to add to the diffuse light. It
runs only while GI_ENABLED is True (module flag, default False; the UI toggles gl_sdf.GI_ENABLED, read every
frame), and then costs: a half-res scene pass, its own half-res jump flood (emitters + occluders), N_CASCADES
cascade passes and an upsampling pass. GI_GAIN scales the result.

Periodic maps: frame.period (render px; an axis with 0 doesn't repeat) is assumed to start at pixel 0; lookups
wrap into [0, period) and distances are minimum-image.
"""
import math

import moderngl

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


def _jfa_progs(frame):
    inc = ["sdf_common.glsl"]
    return {"seed": frame.load(FULLSCREEN_VERT, "jfa_seed.frag", inc),
            "step": frame.load(FULLSCREEN_VERT, "jfa_step.frag", inc),
            "dist": frame.load(FULLSCREEN_VERT, "sdf_dist.frag", inc)}


class SdfEffect:
    """shared["sdf"] from the G-buffer's shadow-casting occluders."""
    order = 100

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.jfa = None
        self.progs = None

    def run(self, gw, frame):
        size = tuple(frame.size)
        if self.progs is None:
            self.progs = _jfa_progs(frame)
        if self.jfa is None or self.jfa.size != size:
            if self.jfa is not None:
                self.jfa.release()
            self.jfa = JumpFlood(self.ctx, size, linear=True)
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        frame.shared["sdf"] = self.jfa.build(frame, self.progs, frame.gbuf["normal"], 0, _period(frame.period))

    def release(self):
        if self.jfa is not None:
            self.jfa.release()
        for p in (self.progs or {}).values():
            p.release()
        self.jfa = self.progs = None


class GiEffect:
    """shared["gi"] by radiance cascades, while GI_ENABLED."""
    order = 150

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, None
        self.progs = None
        self.textures = []
        self.stats = {}

    def _alloc(self, size):
        for o in self.textures:
            o.release()
        ctx = self.ctx
        m = 1 << (N_CASCADES - 1)
        half = ((size[0] + 1) // 2, (size[1] + 1) // 2)
        self.grid = (-(-half[0] // m) * m, -(-half[1] // m) * m)  # E: half res padded to a multiple of 2^(n-1)
        self.scene = _tex(ctx, self.grid, 4, "f2")
        self.scene_fbo = ctx.framebuffer([self.scene])
        self.casc = [_tex(ctx, self.grid, 4, "f2", linear=True) for _ in range(2)]
        self.casc_fbos = [ctx.framebuffer([t]) for t in self.casc]
        self.out = _tex(ctx, size, 4, "f2", linear=True)
        self.out_fbo = ctx.framebuffer([self.out])
        self.jfa = JumpFlood(ctx, self.grid, linear=False)
        self.textures = [self.scene_fbo, self.scene, *self.casc_fbos, *self.casc, self.out_fbo, self.out]
        self.size = size

    def run(self, gw, frame):
        if not GI_ENABLED:
            return
        size = tuple(frame.size)
        if self.progs is None:
            inc = ["sdf_common.glsl"]
            self.progs = dict(_jfa_progs(frame),
                              scene=frame.load(FULLSCREEN_VERT, "rc_scene.frag", ["material.glsl", *inc]),
                              cascade=frame.load(FULLSCREEN_VERT, "rc_cascade.frag", inc),
                              out=frame.load(FULLSCREEN_VERT, "rc_out.frag", inc))
        if self.size != size:
            if self.size is not None:
                self.jfa.release()
            self._alloc(size)
        ctx = self.ctx
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        period = _period(frame.period, 2.0)
        p = self.progs
        # scene at half res
        g = frame.gbuf
        for unit, tex in enumerate((g["normal"], g["albedo"], g["mat"], frame.material_tex)):
            tex.use(unit)
        for unit, name in enumerate(("u_normal", "u_albedo", "u_mat", "u_mats")):
            _set(p["scene"], name, unit)
        _set(p["scene"], "u_emit_gain", float(GI_EMIT_GAIN))
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
        _set(o, "u_c0", 0)
        _set(o, "u_dist", 1)
        _set(o, "u_period", period)
        _set(o, "u_gain", float(GI_GAIN))
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
        self.textures, self.progs, self.size = [], None, None


def make(ctx, size):
    return [SdfEffect(ctx, size), GiEffect(ctx, size)]
