"""Effect plugins of the lit renderer (research/lighting_contract.md, "Effect plugin API").

With lighting on, GLWorld draws the world into its G-buffer (GLWorld.gbuf: albedo, normal, mat) and then runs the
effects in ascending `order`; each reads the G-buffer and what earlier effects published in `frame.shared`, and the
compose effect writes GLWorld.tex. Translucent overlays (grab lines) are drawn after the last effect. With no effect
modules present, lighting on shows the albedo as is.

An effect module (one of EFFECT_MODULES, importable from the repo root) provides `make(ctx, size) -> list[Effect]`;
an Effect is duck-typed:
    order: int
    run(gw, frame)   # gw: gl_world.GLWorld, frame: FrameInfo
    release()
Effects own their textures / FBOs and leave blending and depth testing off when they return.

Shader loading (FrameInfo.load / GLWorld.load): `load(vert, frag, includes=())` compiles shaders/<vert> and
shaders/<frag> as GLSL 330 core, with each file in `includes` (from shaders/, e.g. "material.glsl",
"world_common.glsl", "sdf_shadow.glsl") pasted in order right after the `#version` line of both stages, and
`#define VERTEX_SHADER` before them in the vertex stage. vert=None uses world_base.vert: a fullscreen triangle strip
from gl_VertexID with no outputs (fragment shaders address pixels with ivec2(gl_FragCoord.xy); row 0 = the top
row, same as every G-buffer texture). world_common.glsl is NOT included unless listed (it declares the uniforms
u_size and u_clip, to_clip, unpack_rgb, floordiv / ceildiv). FrameInfo.fullscreen(prog) draws that strip (4 vertices)
with prog into whatever framebuffer is bound.

Additions beyond the contract (optional to use): FrameInfo.gw (the GLWorld), FrameInfo.ctx (the moderngl context)
and FrameInfo.fbo (GLWorld.fbo: GLWorld.tex + the shared depth buffer, what compose / glass draw into), and
FrameInfo.filled (the view mode: False in the points + springs view).
Albedo note: with lighting on, gbuf["albedo"] is the unlit colour with the material's tint and procedural pattern
applied (identical to the unlit pixels for a material with tint_strength 0 and no pattern).
"""
import importlib
from dataclasses import dataclass, field

EFFECT_MODULES = ["gl_sdf", "gl_lights", "gl_glass"]


def make_effects(ctx, size):
    """Every effect from the EFFECT_MODULES that exist (a missing module is skipped; an error inside one is raised),
    sorted by order."""
    effects = []
    for name in EFFECT_MODULES:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name == name:
                continue
            raise
        effects.extend(mod.make(ctx, size))
    return sorted(effects, key=lambda e: e.order)


@dataclass
class FrameInfo:
    """One lit frame, handed to every Effect.run. Positions are render px (world units x zoom), row 0 = top.

    size: GLWorld.size; clip: (w, h) of the world view (pixels at or past it are outside the world, like
    world_common.glsl's u_clip); zoom: render px per world unit; period: (px_w, px_h) of a periodic world (0 on an
    axis that doesn't loop), or None; world: the solver World (lights.of_world(world), materials.of_body(body));
    gbuf: {"albedo", "normal", "mat"} textures (see the contract); material_tex: the material table (RGBA32F,
    materials.TEXELS wide, a row per id); n_materials: rows in it (id 0, the empty row, included); shared: textures
    effects publish for later ones, fresh each frame."""
    size: tuple
    clip: tuple
    zoom: float
    period: tuple
    world: object
    gbuf: dict
    material_tex: object
    n_materials: int
    shared: dict = field(default_factory=dict)
    gw: object = None  # the GLWorld (helpers below)
    ctx: object = None
    fbo: object = None
    filled: bool = True  # the view mode: False = points + springs (V's first mode), True = filled shapes

    def fullscreen(self, prog):
        """Draw a fullscreen triangle strip with prog into the bound framebuffer."""
        self.gw.fullscreen(prog)

    def load(self, vert, frag, includes=()):
        """Compile shaders/<vert> (None: world_base.vert) + shaders/<frag> with includes (see the module docs)."""
        return self.gw.load(vert, frag, includes)
