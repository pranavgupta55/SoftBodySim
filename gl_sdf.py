"""Signed distance field, soft shadows and global illumination for the lit renderer (research/lighting_contract.md).

make(ctx, size) -> [SdfEffect (order 100), GiEffect (order 150)], gl_effects' plugin API.

SdfEffect publishes frame.shared["sdf"]: R32F, the render size (plus the lit-render margin, below), LINEAR filtered,
the signed distance in render px
from each pixel centre to the nearest boundary of a shadow-casting occluder (G-buffer `normal` flags bit0 and bit1
both set), negative inside, minimum-image across frame.period. Built by jump flooding in fragment ping-pong passes
(GL 4.1, no compute): a seed pass (pixels on either side of a boundary store their own coordinates, RG16I),
ceil(log2(max(w, h))) step passes with steps 2^(k-1) .. 1 plus one more step-1 pass (JFA+1), then a distance pass.
Accuracy: within ~0.7 px of the true distance to the rasterised outline. shaders/sdf_shadow.glsl has the
shadow march lights use (sdf_shadow / sdf_at). shared["sdf"] holds every shadow caster, see-through ones included
(glass refraction and caustics march it to find bodies).

Translucent shadows: see-through casters (glass / fluid with transparency > 0: G-buffer FLAG_TRANSLUCENT, bit4)
let light through. On frames that have them (a scan of the world's materials, redone when gl_world's material
signature changes), in the filled view (or the points view when gl_world drew the silhouettes' transmittance,
gbuf["tr"], which then stands in for their material), with a light that casts shadows and gl_lights.tr_scale > 0 (caustics off,
or gl_lights.CAUSTIC_SHARE < 1), SdfEffect also publishes, all at the render size:
  shared["sdf_opaque"] (R32F, LINEAR): the same field over the opaque casters only;
  shared["sdf_tr"] (R32F, LINEAR): over the translucent casters only;
  shared["sdf_tint"] (RGBA8, LINEAR): per texel, the transmittance of the nearest translucent caster
    (material.glsl mat_transmittance: transparency x Beer-Lambert toward its colour);
from a split pass (sdf_tr.frag) and one jump flood over both masks (jfa2_*.frag). gl_lights then shadows with sdf_opaque x sdf_shadow_tr.
On those frames shared["sdf"] is min(sdf_opaque, sdf_tr) from the same flood (no flood of its own): the same field
outside the casters, and inside one where it touches a caster of the other kind, the distance to its own edge.

Frames nothing reads the fields on (SdfEffect._wanted: no light casting shadows, and the points view or no glass /
mirror / fluid material) skip the flood and publish no "sdf"; the points view with translucent fields publishes no
"sdf" either (lights shadow with sdf_opaque / sdf_tr there, and the glass / caustics effects that read "sdf" don't
run in it).

Lit-render margin (gl_world.LIT_MARGIN): when the frame has FrameInfo.occ (a G-buffer padded around the view, with
the bodies just off screen drawn in it), every field above is built over it instead of the view-sized G-buffer, and
shared["sdf_origin"] = FrameInfo.origin: render px p is texel p + origin of the fields (0, 0 without a margin).
sdf_shadow.glsl's sdf_at / sdf_tint_at add it (uniform u_sdf_origin, which every program sampling the fields sets
from shared["sdf_origin"]). So walls just past the view's edge shade it. GI stays view-sized: a ray between two
points of the (convex) view never leaves it, so off-screen occluders could only block off-screen emitters, which
the GI doesn't have.

GiEffect (radiance cascades, an experiment) publishes frame.shared["gi"]: RGBA16F at the render size, the light
emissive bodies throw on everything around them (occluded by shadow casters), to add to the diffuse light. It
runs while GI_ENABLED is True (module flag, default False, read every frame) or the world has `gi` set (a per-world
switch, saved in scenes; the Lights card's GI button toggles it), and then costs: a half-res scene pass (+ its medium's mip
levels), its own half-res jump flood (emitters + opaque occluders), N_CASCADES cascade passes and an upsampling pass. GI_GAIN, times the
world's own `gi_gain` (if any; saved in scenes, set by a map's "lighting" block), scales the result (gi_gain():
clamped to [0, MAX_GAIN], a non-finite gain counts as 1; GI_EMIT_GAIN likewise). Unlit materials emit nothing.
On periodic maps the GI wraps seamlessly: the probe grid runs two top-cascade probe spacings past the period, and
cascade merges and the upsampling take any footprint that would leave the grid one period over.
The top GI_FAR_CASCADES cascades (the far field) are redone a band of 1 / GI_FAR_EVERY of their rows per frame and
reused in between; a still scene renders exactly the same, and any whole-scene change redoes them at once
(GiEffect._far_key).
Translucent casters (FLAG_TRANSLUCENT) are a participating medium for GI (after DexRT, arxiv.org/abs/2511.08498):
rc_scene writes their extinction and emission per half-res texel (GI_GLASS_CHORD), mipmapped each frame, and leaves
them out of the opaque scene the GI's jump flood seeds from; a cascade ray steps by the distance to opaque texels and
each step multiplies its visibility by exp(-sigma x step), sigma read at the mip level of the step's length capped per
cascade (GI_GLASS_LOD_CAPS). A ray that runs out of steps crosses the rest of its interval in one coarse look.

Periodic maps: frame.period (render px; an axis with 0 doesn't repeat) is assumed to start at pixel 0; lookups
wrap into [0, period) and distances are minimum-image.
"""
import math
from types import SimpleNamespace

import moderngl
import numpy as np

import lights as lights_mod
import materials

# Reuse SdfEffect's fields from the last frame while every shadow caster is a fixed body and nothing they depend on
# changed (_StaticSet: the casters' points, materials and colours, the world, view mode, zoom, size,
# period). Not exact: a body that casts no shadow but is drawn over a caster's edge (a glowing shape resting on a
# glass tile) un-flags those px of the caster mask every frame it is there; the reused fields keep the mask of the
# frame they were built on. On penrose/evolved 1 halloween that is at most 6/255 on ~0.01% of px.
SDF_CACHE = True
# The floods' first step (0: the full flood, ceil(log2(size)) step passes; see _steps). A px farther than 2 x start
# from the boundary then reads that bound instead of its distance: safe for marching, but not the same pixels. Off:
# halloween's GI flood at 32 saved ~0.3 ms for max 1/255 on 0.07% of px; the full-res one at 64 changed the lighting
# demo's shadows by up to 40/255 (a sparse map: its far px march in shorter steps) and saved nothing.
SDF_JFA_START = 0
GI_JFA_START = 0
GI_ENABLED = False  # radiance cascades GI on / off (the UI may toggle this; read every frame)
GI_GAIN = 1.0  # multiplies shared["gi"]
GI_EMIT_GAIN = 1.0  # multiplies the emission the cascades start from
N_CASCADES = 6
# The far field: the top GI_FAR_CASCADES cascades (the longest rays: most of the GI's cost on maps full of
# translucent glass, whose rays march on through every body) are redone a band of 1 / GI_FAR_EVERY of their rows
# (scissored) per frame, so every frame costs the same and each row is redone every GI_FAR_EVERY frames. A still
# scene renders exactly as with every cascade every frame; moving emitters' light from more than
# GI_L0 x (4^(N - k) - 1) / 3 half-res texels away (k = GI_FAR_CASCADES: 21 texels = 42 render px) lags up to
# GI_FAR_EVERY - 1 frames. Anything that changes the whole scene (GiEffect._far_key: world swap or map load,
# materials / palette, view mode, zoom, size; GI or lighting switched off and on) redoes them whole at once.
# GI_FAR_EVERY 1 (or GI_FAR_CASCADES 0): every cascade every frame.
GI_FAR_CASCADES = 3
GI_FAR_EVERY = 2
# See-through bodies in the GI: a participating medium (rc_scene's mipmapped glass texture). A px of transmittance T
# has extinction -ln T / GI_GLASS_CHORD per world unit, so a ray crossing GI_GLASS_CHORD of it keeps T (the
# reference: Cauchy's mean chord pi A / P of the penrose maps' rhombi, edge 20.16: 15.1 thick, 9.3 thin, 12.9 for
# their 1.618 : 1 mix). A body of about that size passes what one crossing passed when GI counted a factor T per body;
# a thick slab now passes less (T^(chord / GI_GLASS_CHORD)) and a thin sliver more, and two touching bodies count
# twice (they counted once). Cascade i reads the medium at mip level min(log2(step), GI_GLASS_LOD_CAPS[i]) (its last
# entry for any cascade past the list). On the penrose glow / halloween maps the clusters of touching tiles came out
# darker inside and the room ~18% dimmer: their gi_gain went up 1.28x to match the mean; GI_GLASS_CHORD 26 at the old
# gains is the closest match to how they looked before (mean luma -0.3, the clusters less dark).
GI_GLASS_CHORD = 13.0
GI_GLASS_LOD_CAPS = (0, 0, 1, 2, 3, 4)  # ((1, 1, 2, 3, 4, 5): -0.37 ms, but a 1-chord wall passed 0.51, 2 chords 0.22)
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

    def build(self, frame, progs, src, mode, period, start=0):
        """src: the mask texture (unit 0); mode: sdf_common.glsl's sdf_solid mode; period: vec2 in texels.
        start: the first (largest) step, 0 = the full flood (_steps)."""
        seed, step, dist = progs["seed"], progs["step"], progs["dist"]
        steps, far = _steps(self.steps, start)
        src.use(0)
        _set(seed, "u_src", 0)
        _set(seed, "u_mode", mode)
        _set(seed, "u_period", period)
        self.seed_fbos[0].use()
        frame.fullscreen(seed)
        cur = 0
        _set(step, "u_seeds", 0)
        _set(step, "u_period", period)
        for s in steps:
            self.seeds[cur].use(0)
            _set(step, "u_step", s)
            self.seed_fbos[1 - cur].use()
            frame.fullscreen(step)
            cur = 1 - cur
        self.seeds[cur].use(0)
        src.use(1)
        _set(dist, "u_far", far)
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
    shared). build() -> (opaque distance, translucent distance, tint), all LINEAR at the render size; `union` then
    holds min(opaque, translucent): shared["sdf"] on these frames (no flood of its own)."""

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.tr_src, self.opq_src = _tex(ctx, self.size, 4, "f1"), _tex(ctx, self.size, 4, "f1")
        self.split_fbo = ctx.framebuffer([self.tr_src, self.opq_src])
        self.seeds = [_tex(ctx, self.size, 4, "i2") for _ in range(2)]
        self.seed_fbos = [ctx.framebuffer([t]) for t in self.seeds]
        self.opq = _tex(ctx, self.size, 1, "f4", linear=True)
        self.tr = _tex(ctx, self.size, 1, "f4", linear=True)
        self.tint = _tex(ctx, self.size, 4, "f1", linear=True)
        self.union = _tex(ctx, self.size, 1, "f4", linear=True)
        self.dist_fbo = ctx.framebuffer([self.opq, self.tr, self.tint, self.union])
        n = max(1, math.ceil(math.log2(max(self.size))))
        self.steps = [1 << k for k in range(n - 1, -1, -1)] + [1]

    @staticmethod
    def progs(frame):
        inc = ["sdf_common.glsl"]
        return {"split": frame.load(FULLSCREEN_VERT, "sdf_tr.frag", ["material.glsl"]),
                "seed": frame.load(FULLSCREEN_VERT, "jfa2_seed.frag", inc),
                "step": frame.load(FULLSCREEN_VERT, "jfa2_step.frag", inc),
                "dist": frame.load(FULLSCREEN_VERT, "jfa2_dist.frag", inc)}

    def build(self, frame, progs, period, g=None, origin=(0, 0)):
        """g: the G-buffer textures to split (frame.gbuf, or the margin's padded frame.occ, render px p at texel
        p + origin). The points view's frame.gbuf["tr"], when there is one, gives its silhouettes' transmittance."""
        g = frame.gbuf if g is None else g
        tr = frame.gbuf.get("tr")
        sp = progs["split"]
        for unit, (name, tex) in enumerate((("u_normal", g["normal"]), ("u_albedo", g["albedo"]),
                                            ("u_mat", g["mat"]), ("u_mats", frame.material_tex),
                                            ("u_tr", tr if tr is not None else g["albedo"]))):
            tex.use(unit)
            _set(sp, name, unit)
        _set(sp, "u_has_tr", tr is not None)
        _set(sp, "u_tr_origin", tuple(int(o) for o in origin))
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
                  self.opq, self.tr, self.tint, self.union]:
            o.release()


SDF_FAR = 1e4  # sdf_dist.frag's distance with no seed at all


def _steps(full, start):
    """The step passes of a flood starting at step `start` (a power of two; 0 or more than the full flood's first:
    the full flood), and the distance to write where no seed arrived. Every px with a seed within Chebyshev distance
    R = 2 start (the steps' sum, with the extra step 1) of it gets one (some seed, the nearest unless the flood
    errs), so a px left without one is at least R + 0.5 from the boundary: that bound is safe for sphere tracing."""
    if start <= 0 or start >= full[0]:
        return full, SDF_FAR
    return [s for s in full[:-1] if s <= start] + [1], 2.0 * start + 0.5


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


class _StaticSet:
    """Reuse across frames of work that depends on one kind of body only (a bool per material id on gl_world, e.g.
    mat_casts: shadow casters): key() is (key, the points of those bodies) when every one of them is a fixed body (so
    only an edit, a material, colour or view change or a world swap changes them), else None; same() says whether the
    last remembered frame had it. The key holds the world, its topology and material signature (gl_world: replaced on
    any change), the palette, the view mode and outlines, plus the caller's `extra`; the points are compared too (an
    editor drag of a fixed body). Other bodies are not in it: a moving one drawn over one of these changes what the
    G-buffer shows of it by the px it covers, which the reused work then misses."""

    def __init__(self, attr):
        self.attr = attr
        self.key, self.pos = None, None
        self._fixed_of = (None, None, None)  # (topology, material signature, the bodies' point indices or None)

    def of(self, gw, frame, extra):
        world = frame.world
        topo = getattr(getattr(gw, "packer", None), "topo", None)
        mats, kind = getattr(gw, "body_mat", None), getattr(gw, self.attr, None)
        bodies = getattr(world, "bodies", None)
        if topo is None or mats is None or kind is None or bodies is None or len(mats) != len(bodies) \
                or getattr(world, "P", None) is None:
            return None
        if getattr(world, "ropes", None) and len(kind) > 1 and kind[1]:  # (ropes draw in the default material)
            return None
        sig = getattr(gw, "_mat_sig", None)
        if self._fixed_of[0] is not topo or self._fixed_of[1] is not sig:
            fixed = np.array([bool(getattr(b, "fixed", False)) for b in bodies], bool)
            mine = kind[np.clip(mats, 0, len(kind) - 1)]
            idx = None
            if not (mine & ~fixed).any():
                pb = getattr(world, "point_body", None)
                idx = None if pb is None or len(pb) != world.P.shape[1] else np.flatnonzero(mine[pb])
            self._fixed_of = (topo, sig, idx)
        idx = self._fixed_of[2]
        if idx is None:
            return None
        hooks = tuple(tuple(float(v) for v in getattr(h, "pos", ())) for h in getattr(world, "hooks", ()) or ())
        key = (world, topo, sig, getattr(world, "palette", None), bool(getattr(frame, "filled", True)),
               getattr(gw, "last_outlines", None), float(frame.zoom), tuple(frame.clip), hooks, "tr" in frame.gbuf,
               *extra)
        return key, world.P[:, idx]

    def same(self, k):
        """Is k (from of()) the remembered frame's?"""
        if k is None or self.key is None or len(k[0]) != len(self.key):
            return False
        return (all(a is b or (type(a) is type(b) and a == b) for a, b in zip(k[0], self.key))
                and k[1].shape == self.pos.shape and np.array_equal(k[1], self.pos))

    def remember(self, k):
        self.key, self.pos = (None, None) if k is None else (k[0], k[1].copy())

    def forget(self):
        self.remember(None)
        self._fixed_of = (None, None, None)


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
        # by size (the margin's padding varies as off-screen casters move): the last few, so a body moving in and
        # out of the margin doesn't reallocate every frame
        self.jfas, self.splits = {}, {}
        self.stats = {}
        # the static cache (SDF_CACHE): every shadow caster fixed and unchanged since the last frame, which
        # published _pub
        self.static, self._pub = _StaticSet("mat_casts"), None

    CACHE = 4

    def _cached(self, cache, size, make):
        obj = cache.pop(size, None)
        if obj is None:
            obj = make()
            while len(cache) >= self.CACHE:
                cache.pop(next(iter(cache))).release()
        cache[size] = obj  # (most recent last)
        return obj

    def _wants_tr(self, gw, frame):
        world = getattr(frame, "world", None)
        if world is None:
            return False
        # the points view has see-through silhouettes only when gl_world drew their transmittance (gbuf["tr"])
        if not getattr(frame, "filled", True) and "tr" not in frame.gbuf:
            return False
        if not self.translucent.of(gw, world) or not _shadow_lights(world):
            return False
        import gl_lights  # (lazily: keeps the two modules' import order free)
        return gl_lights.tr_scale(gw, frame) > 0.0


    def wants_occ(self, gw, world, filled):
        """gl_world: draw the lit-render margin this frame? Only when the fields get built (it is all they use)."""
        return self._wanted(gw, SimpleNamespace(world=world, filled=filled))

    @staticmethod
    def _wanted(gw, frame):
        """Does anything read the fields this frame: a light that casts shadows (gl_lights; also all caustics need),
        or, in the filled view, glass / mirror / fluid materials (gl_glass's resolve marches shared["sdf"]; in the
        points view every body has an opaque twin and the resolve has nothing to do). No: the frame goes without
        (lights then use their no-shadow variant, which is what a light without shadows gets anyway)."""
        world = getattr(frame, "world", None)
        if world is None or _shadow_lights(world):
            return True
        if not getattr(frame, "filled", True):
            return False
        kinds = getattr(gw, "mat_kinds", None)
        if kinds is None:
            return True
        return bool(set(int(k) for k in kinds) & {materials.GLASS, materials.MIRROR, materials.FLUID})

    def run(self, gw, frame):
        if not self._wanted(gw, frame):
            return
        # the lit-render margin (gl_world.LIT_MARGIN): the fields cover the padded G-buffer, render px p at texel
        # p + origin (published as shared["sdf_origin"] for sdf_shadow.glsl's u_sdf_origin)
        src = getattr(frame, "occ", None) or frame.gbuf
        origin = tuple(float(o) for o in getattr(frame, "origin", (0, 0))) if src is not frame.gbuf else (0.0, 0.0)
        size = tuple(src["normal"].size)
        if self.progs is None:
            self.progs = _jfa_progs(frame)
        self.jfa = self._cached(self.jfas, size, lambda: JumpFlood(self.ctx, size, linear=True))
        self.ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        period = _period(frame.period)
        frame.shared["sdf_origin"] = origin
        tr = self._wants_tr(gw, frame)
        key = self.static.of(gw, frame, (size, origin, period, tr)) if SDF_CACHE else None
        hit = self.static.same(key) and self._pub is not None
        self.stats["cached"] = hit
        if hit:
            frame.shared.update(self._pub)
            return
        self.static.remember(key)
        self._pub = None
        before = set(frame.shared)
        # with translucent fields shared["sdf"] is their union (SplitFlood), no flood of its own; the points view
        # publishes none (lights shadow with the split fields alone, and glass / caustics are filled-view effects)
        if not tr:
            frame.shared["sdf"] = self.jfa.build(frame, self.progs, src["normal"], 0, period,
                                                    int(SDF_JFA_START))
        if tr:
            self._split(frame, size, period, src, origin)
            if getattr(frame, "filled", True):
                frame.shared["sdf"] = self.split.union
        if key is not None:
            self._pub = {k: frame.shared[k] for k in set(frame.shared) - before}

    def _split(self, frame, size, period, src, origin):
        if self.split_progs is None:
            self.split_progs = SplitFlood.progs(frame)
        self.split = self._cached(self.splits, size, lambda: SplitFlood(self.ctx, size))
        sh = frame.shared
        sh["sdf_opaque"], sh["sdf_tr"], sh["sdf_tint"] = self.split.build(frame, self.split_progs, period, src,
                                                                          origin)

    def release(self):
        for o in [*self.jfas.values(), *self.splits.values()]:
            o.release()
        for p in [*(self.progs or {}).values(), *(self.split_progs or {}).values()]:
            p.release()
        self.jfas, self.splits = {}, {}
        self.jfa = self.progs = self.split = self.split_progs = None
        self.static.forget()
        self._pub = None


class GiEffect:
    """shared["gi"] by radiance cascades, while GI_ENABLED or world.gi."""
    order = 150

    def __init__(self, ctx, size):
        self.ctx, self.size, self.period = ctx, None, None
        self.progs = None
        self.textures = []
        self.stats = {}
        self.far, self.far_fbos = [], []  # the far cascades' own textures (GI_FAR_CASCADES), top down
        self._far_last_key, self._far_last_no, self._far_age = None, None, 0

    @staticmethod
    def _far_key(gw, frame, period):
        """What the far cascades depend on besides the bodies' motion: a change redoes them at once. The world
        (swap / map load: another World or a repacked one), its materials and colours (gl_world's material
        signature, replaced on any change; the palette and background), the view mode, zoom and view size, the
        grid and the emission gain. (Lighting toggled off and on: GLWorld.frame_no skips, see run.)"""
        world = getattr(frame, "world", None)
        packer = getattr(gw, "packer", None)
        return (id(world), id(getattr(packer, "topo", None)), id(getattr(gw, "_mat_sig", None)),
                getattr(world, "palette", None), tuple(getattr(world, "background", None) or ()),
                bool(getattr(frame, "filled", True)), getattr(gw, "last_outlines", None), float(frame.zoom),
                tuple(frame.clip), tuple(frame.size), period, _gain(GI_EMIT_GAIN))

    def _alloc_far(self, n):
        for o in self.far_fbos + self.far:
            o.release()
        self.far = [_tex(self.ctx, self.grid, 4, "f2", linear=True) for _ in range(n)]
        self.far_fbos = [self.ctx.framebuffer([t]) for t in self.far]

    def _alloc(self, size, period):
        for o in self.textures:
            o.release()
        self._alloc_far(0)
        ctx = self.ctx
        m = 1 << (N_CASCADES - 1)
        half = ((size[0] + 1) // 2, (size[1] + 1) // 2)
        # E: half res padded to a multiple of 2^(n-1); on a periodic axis at least 2 top-cascade probe spacings more
        # than the period, so every probe near a seam finds its upper cascade's 2x2 footprint whole one period over
        # (probes past the period are valid: their marches wrap) and rc_out can wrap into [0.5, period + 0.5)
        need = [max(h, int(math.ceil(p)) + 2 * m) if p > 0 else h for h, p in zip(half, period)]
        self.grid = tuple(-(-n // m) * m for n in need)
        self.scene = _tex(ctx, self.grid, 4, "f2")
        # the see-through medium (rc_scene): extinction and emission, mipmapped each frame; bilinear within the
        # nearest mip level (trilinear changed at most 1/255 for +0.9 ms on halloween)
        self.sigma = _tex(ctx, self.grid, 1, "f2", linear=True)
        self.glow = _tex(ctx, self.grid, 4, "f2", linear=True)
        for t in (self.sigma, self.glow):
            t.filter = (moderngl.LINEAR_MIPMAP_NEAREST, moderngl.LINEAR)
            t.build_mipmaps()
        self.scene_fbo = ctx.framebuffer([self.scene, self.sigma, self.glow])
        self.casc = [_tex(ctx, self.grid, 4, "f2", linear=True) for _ in range(2)]
        self.casc_fbos = [ctx.framebuffer([t]) for t in self.casc]
        self.out = _tex(ctx, size, 4, "f2", linear=True)
        self.out_fbo = ctx.framebuffer([self.out])
        self.jfa = JumpFlood(ctx, self.grid, linear=False)
        self.textures = [self.scene_fbo, self.scene, self.sigma, self.glow, *self.casc_fbos, *self.casc, self.out_fbo, self.out]
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
        tr = g.get("tr")  # (the points view's silhouette transmittance, gl_world _occluder_fill)
        for unit, tex in enumerate((g["normal"], g["albedo"], g["mat"], frame.material_tex,
                                    tr if tr is not None else g["albedo"])):
            tex.use(unit)
        for unit, name in enumerate(("u_normal", "u_albedo", "u_mat", "u_mats", "u_tr")):
            _set(p["scene"], name, unit)
        _set(p["scene"], "u_has_tr", tr is not None)
        _set(p["scene"], "u_emit_gain", _gain(GI_EMIT_GAIN))
        _set(p["scene"], "u_chord", max(float(GI_GLASS_CHORD) * float(frame.zoom) / 2.0, 1e-3))  # (world -> texels)
        _set(p["scene"], "u_period", period)
        # (every frame: keeping the medium while all see-through bodies are fixed saved 0.15 ms of the pass's 0.7 on
        # halloween, most of it being the px classification the opaque scene needs anyway, for up to 12/255 where a
        # moving body had covered a tile's edge)
        self.scene_fbo.use()
        frame.fullscreen(p["scene"])
        self.sigma.build_mipmaps()
        self.glow.build_mipmaps()
        dist = self.jfa.build(frame, p, self.scene, 1, period, int(GI_JFA_START))
        # cascades, top down, ping-ponging between two textures
        c = p["cascade"]
        self.scene.use(0)
        dist.use(1)
        self.sigma.use(3)
        self.glow.use(4)
        _set(c, "u_scene", 0)
        _set(c, "u_dist", 1)
        _set(c, "u_upper", 2)
        _set(c, "u_sigma", 3)
        _set(c, "u_glow", 4)
        _set(c, "u_l0", float(GI_L0))
        _set(c, "u_period", period)
        # (medium lookups wrap into [m, period + m): their mip footprint stays on the grid, which runs 2 m past it)
        _set(c, "u_wrap0", float(1 << (N_CASCADES - 1)))
        caps = tuple(GI_GLASS_LOD_CAPS) or (0,)
        # the far cascades (GI_FAR_CASCADES, the top ones): kept in their own textures, a band of 1 / GI_FAR_EVERY of
        # their rows redone per frame (each cascade's band reads the one above as it stands: half fresh), all of them
        # when anything that changes the whole scene changed (_far_key) or a frame was skipped
        n_far = min(max(int(GI_FAR_CASCADES), 0), N_CASCADES - 1) if int(GI_FAR_EVERY) > 1 else 0
        far = list(range(N_CASCADES - 1, N_CASCADES - 1 - n_far, -1))  # top down
        # band: the rows redone this frame, (y0, y1), or None for all of them
        band, full = None, True
        if n_far:
            every = int(GI_FAR_EVERY)
            key = self._far_key(gw, frame, period)
            no = getattr(gw, "frame_no", None)
            if len(self.far) != n_far:
                self._alloc_far(n_far)
            elif (key == self._far_last_key and no is not None and self._far_last_no is not None
                  and no == self._far_last_no + 1):
                full = False
            self._far_age = 0 if full else (self._far_age + 1) % every
            self._far_last_key, self._far_last_no = key, no
            if not full:
                h = self.grid[1]
                band = (h * self._far_age // every, h * (self._far_age + 1) // every)
        self.stats["far_refreshed"] = bool(n_far) and full
        self.stats["far_band"] = band
        upper = None
        for k, i in enumerate(far):
            (upper if upper is not None else self.casc[0]).use(2)
            _set(c, "u_has_upper", i < N_CASCADES - 1)
            _set(c, "u_index", i)
            _set(c, "u_lod_cap", float(caps[min(i, len(caps) - 1)]))
            fbo = self.far_fbos[k]
            if band is not None:
                fbo.scissor = (0, band[0], self.grid[0], band[1] - band[0])
            fbo.use()
            frame.fullscreen(c)
            if band is not None:
                fbo.scissor = None
                fbo.use()  # (scissor test back off for the passes after)
            upper = self.far[k]
        cur = 0
        for i in range(N_CASCADES - 1 - n_far, -1, -1):
            # the cascade above (unused for the top one)
            (upper if upper is not None and i == N_CASCADES - 1 - n_far else self.casc[1 - cur]).use(2)
            _set(c, "u_has_upper", i < N_CASCADES - 1)
            _set(c, "u_index", i)
            _set(c, "u_lod_cap", float(caps[min(i, len(caps) - 1)]))
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
        self._alloc_far(0)
        if self.size is not None:
            self.jfa.release()
        for p in (self.progs or {}).values():
            p.release()
        self.textures, self.progs, self.size, self.period = [], None, None, None


def make(ctx, size):
    return [SdfEffect(ctx, size), GiEffect(ctx, size)]
