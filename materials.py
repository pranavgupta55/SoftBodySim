"""Surface materials for the lit renderer (gl_world's G-buffer and effects): what a body looks like under lights.

A body has `material` (a PRESETS name, default DEFAULT) and `material_over` (a dict of Material fields overriding
it). resolve() gives the effective Material. The GPU side (a material table texture indexed by material id, see
gl_world) packs resolved materials; bodies with identical resolved materials share one id. With lighting off,
materials change nothing: the world draws exactly as before.
"""
from dataclasses import dataclass, fields, replace

OPAQUE, GLASS, MIRROR, FLUID = 0, 1, 2, 3  # Material.kind
BEVEL, DOME, PILLOW = 0, 1, 2  # Material.profile: height h(d) over the distance d to the nearest edge
PATTERN_NONE, PATTERN_NOISE, PATTERN_STRIPES, PATTERN_CHECKER, PATTERN_DOTS = 0, 1, 2, 3, 4  # Material.pattern


@dataclass(frozen=True)
class Material:
    """All colours are linear 0..1 RGB. Defaults are a plain matte surface."""
    tint: tuple = (1.0, 1.0, 1.0)  # multiplied into the body's own colour (fill_col), blended by tint_strength
    tint_strength: float = 0.0
    emissive: tuple = (1.0, 1.0, 1.0)  # light the surface gives off itself, added after lighting (ignores shadows)
    emissive_strength: float = 0.0
    specular: float = 0.0  # highlight strength 0..1
    roughness: float = 0.6  # 0 = tight highlight, 1 = broad (shininess = exp2(10 * (1 - roughness)))
    bands: int = 0  # toon quantisation of diffuse / specular: 0 = smooth, n = n steps
    bevel_px: float = 3.0  # render px over which the surface curves at the edges (normals from edge distance)
    profile: int = BEVEL
    kind: int = OPAQUE
    ior: float = 1.0  # index of refraction (glass / fluid)
    reflectivity: float = 0.0  # 0..1 share of reflected light (mirror / glass Fresnel base)
    transparency: float = 0.0  # 0..1 (glass / fluid): how much of what's behind shows through
    absorption: float = 0.0  # per-px absorption of the tint colour through the body's thickness (Beer-Lambert)
    pattern: int = PATTERN_NONE  # procedural surface texture in the body's rest frame (stretches with it)
    pattern_scale: float = 6.0  # render px per pattern cell
    pattern_strength: float = 0.0  # 0..1 how much the pattern darkens / modulates the albedo
    casts_shadow: bool = True
    lit: bool = True  # False: drawn at its plain colour (outlines, UI-like bodies)


DEFAULT = "matte"
PRESETS = {
    "matte": Material(),
    "glossy": Material(specular=0.6, roughness=0.25),
    "jelly": Material(specular=0.8, roughness=0.15, bevel_px=6.0, profile=PILLOW, transparency=0.25, kind=GLASS,
                      ior=1.15, absorption=0.04),
    "metal": Material(specular=1.0, roughness=0.2, reflectivity=0.5, bevel_px=2.0, kind=MIRROR, tint_strength=0.3,
                      tint=(0.85, 0.85, 0.9)),
    "stone": Material(roughness=0.9, pattern=PATTERN_NOISE, pattern_strength=0.35, pattern_scale=4.0),
    "glow": Material(emissive_strength=1.0, casts_shadow=False),
    "glass": Material(kind=GLASS, ior=1.5, transparency=0.85, reflectivity=0.08, specular=0.9, roughness=0.05,
                      bevel_px=4.0, absorption=0.01),
    "mirror": Material(kind=MIRROR, reflectivity=0.95, specular=1.0, roughness=0.02, bevel_px=1.0),
    "water": Material(kind=FLUID, ior=1.33, transparency=0.7, reflectivity=0.05, specular=0.7, roughness=0.1,
                      bevel_px=5.0, profile=DOME, absorption=0.03, tint=(0.4, 0.7, 1.0), tint_strength=0.5),
    "toon": Material(specular=0.7, roughness=0.3, bands=3),
}
FIELDS = {f.name for f in fields(Material)}


def resolve(preset=None, overrides=None):
    """The effective Material of a preset name (None: DEFAULT; unknown names fall back to it) plus overrides."""
    base = PRESETS.get(preset or DEFAULT, PRESETS[DEFAULT])
    if not overrides:
        return base
    return replace(base, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in FIELDS})


def of_body(body):
    """A body's effective Material (its `material` / `material_over`, both optional)."""
    return resolve(getattr(body, "material", None), getattr(body, "material_over", None))
