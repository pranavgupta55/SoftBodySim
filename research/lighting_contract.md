# Lit renderer: shared contract

Five parallel workstreams build lighting, shadows, materials, glass/fluid and UI on one pipeline. This file pins
the interfaces so they merge cleanly. Data models already exist: `materials.py` (Material, PRESETS, resolve,
of_body) and `lights.py` (Light, of_world). Extend them only additively; never rename fields.

## Constraints
- macOS OpenGL 4.1 core, GLSL 330: no compute shaders, no SSBOs. Fragment / vertex passes, FBOs, MRT, UBOs,
  texelFetch, instancing only.
- Everything runs in GLWorld's low-res space: `GLWorld.size` (render surface, ~900x584), row 0 = top, render px =
  world units x zoom. No camera pan. Periodic maps: `frame.period` (render px) or None.
- **Lighting off must be byte-identical to today's output** (same `GLWorld.tex` bytes). Lighting on is opt-in:
  `ViewConfig.lighting` (default False) or a world with `world.lighting = True` (the demo world sets it).

## Pipeline (owner F builds the skeleton in gl_world.py)
```
lighting off: base + bodies -> GLWorld.tex (unchanged path)
lighting on : base + bodies -> G-buffer (MRT, shares GLWorld.depth)
              effects in ascending `order`, each reading/writing frame.shared:
                100 SDF (S)            shared["sdf"]
                150 GI  (S, optional)  shared["gi"]
                200 lights (L)         shared["light"], shared["spec"]
                250 caustics (G, opt.) shared["caustics"]
                300 compose (L)        -> GLWorld.tex, and copies it to shared["scene_lit"]
                400 glass/fluid (G)    reads scene_lit + G-buffer -> writes GLWorld.tex
              overlays (translucent grab lines, editor previews) drawn after, by F
```

## G-buffer (F) — textures on `GLWorld.gbuf` (dict), all GLWorld.size, NEAREST
| key | format | location | contents |
|---|---|---|---|
| `albedo` | RGBA8 | 0 | the colour the unlit renderer would draw (same pixels as lighting off), a = 1 |
| `normal` | RGBA8 | 1 | rg = normal.xy * 0.5 + 0.5 (z implied, +y down), b = distance to the body's nearest edge in px / 255 (clamped), a = flags / 255: bit0 occluder, bit1 casts shadow, bit2 lit, bit3 no emission (a filled body's black border) |
| `mat` | R16UI ("u2", 1 comp) | 2 | material id (0 = background / none) |

Background, grid, trails write normal (0.5, 0.5, 0, 4 / 255) (flat, lit, not an occluder), mat 0; past the clip
(outside the world view) flags 0, so no effect lights it and compose keeps the plain background there. Outline
lines / points / stamps of a body write that body's material id with lit flag per material; outlines (`lit=False`
usage) may use a flat normal.

## Material table (F)
`GLWorld.material_tex`: RGBA32F, width 8 texels per material, one row per material id (id 0 = unused/default).
F documents the exact texel packing in `materials.py` (`pack(material) -> 8x4 floats`) and provides
`shaders/material.glsl`: `struct Mat {...}; Mat mat_fetch(usampler2D/sampler2D table, uint id);` covering every
Material field. Other shaders include it via gl_world's `_load(..., includes=[...])` helper (F adds includes support:
source text concatenation after the `#version` line).

## Effect plugin API (F defines; L, S, G implement)
New file `gl_effects.py` (F): `EFFECT_MODULES = ["gl_sdf", "gl_lights", "gl_glass"]`;
`make_effects(ctx, size)` imports each that exists and calls `module.make(ctx, size) -> list[Effect]`, sorted by
`order`. GLWorld calls each effect's `run(gw, frame)` in order when lighting is on, `release()` on teardown.
```
class Effect:                       # duck-typed
    order: int
    def run(self, gw, frame): ...   # gw: GLWorld; frame: FrameInfo
    def release(self): ...
@dataclass FrameInfo:               # (F, in gl_effects.py)
    size, clip, zoom, period        # period: (px_w, px_h) or None
    world                           # the solver World (lights via lights.of_world, materials via body.material)
    gbuf: dict                      # albedo, normal, mat textures
    material_tex                    # material table texture
    n_materials: int
    shared: dict                    # textures effects publish for later ones (keys above); reset per frame
    fullscreen(prog)                # helper: draw a fullscreen triangle strip with prog into the current fbo
    load(vert, frag, includes=())   # helper: compile from shaders/ with includes
```
Effects own their own textures/FBOs and must restore `gw.fbo`-independent state (blend off, depth test off) after
running. If an expected shared texture is missing (module absent), degrade gracefully: no SDF -> no shadows; no
lights -> compose = albedo x ambient + emissive.

## Shared textures
- `sdf` (S): R32F, signed distance in render px to the nearest occluder boundary (negative inside occluders),
  minimum-image across `period`. `shaders/sdf_shadow.glsl` (S) defines
  `float sdf_shadow(sampler2D sdf, vec2 p, vec2 dir, float max_t, float softness_px, vec2 period)` (1 = lit) and
  `float sdf_at(sampler2D sdf, vec2 p, vec2 period)`. L includes sdf_shadow.glsl only when `"sdf" in shared`
  (otherwise a stub returning 1.0, defined in L's own source).
- `gi` (S, optional): RGBA16F irradiance from emissive surfaces / bounce (radiance cascades); compose adds it.
- `light` / `spec` (L): RGBA16F diffuse and specular light per pixel (already including shadows, bands, ambient
  excluded).
- `caustics` (G, optional): RGBA16F light focused through glass; compose adds it to `light`.
- `scene_lit` (L): RGBA8 copy of the composed lit scene, for refraction.

## Data and UI (U)
- `Body.material` (preset name) / `Body.material_over` (dict): plain attributes, getattr-defaulted everywhere
  (bodies without them use materials.DEFAULT). scene_io saves / loads them; `world.lights` (Light.to_dict list),
  `world.ambient`, `world.lighting` saved / loaded too.
- ViewConfig.lighting toggled by **L** (key free today). Sidebar cards: Lights (add point / spot / directional,
  select, drag with the mouse, sliders) and Material (presets + sliders for picked bodies, undo via
  before_world_change). Showcase preset `"lighting demo"` in scene.PRESETS.

## Ownership (to keep merges clean)
- F: gl_world.py, gl_effects.py, shaders/world_*.frag/.vert, shaders/material.glsl, materials.py `pack()`,
  procedural patterns + rest-frame UVs (Material.pattern*).
- L: gl_lights.py, shaders/light_*.{vert,frag}, shaders/compose.frag, lights.py (additive).
- S: gl_sdf.py, shaders/jfa_*.frag, shaders/sdf*.glsl/frag, shaders/rc_*.frag.
- G: gl_glass.py, shaders/glass_*.frag, shaders/fluid_*.{vert,frag}, shaders/caustics_*.{vert,frag}.
- U: editor.py / selection.py / palettes.py / scene_io.py / scene.py (demo preset) / main_opengl.py (key, toggle,
  passing world into GLWorld.render if needed) / entities.py material attributes.
Touch others' files only for one-line hooks, and say so in your report.
