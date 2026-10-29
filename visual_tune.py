"""The Look card: live sliders for a lit map's main visual settings, saved per map to a local overrides file.

Shown in the sidebar (main_opengl.py) while lighting is on and the active world is a map with a "lighting" block
(scene.build_diamonds_scene sets world.map_key, e.g. "penrose/evolved 1 halloween", and world.map_lighting). Each
slider writes straight to the world, so the next frame shows it (no rebuild, physics untouched):
  room:   GI gain (world.gi_gain), exposure (world.exposure), highlight knee (world.tone_knee,
          gl_lights.tone_knee: where bright colours start rolling off toward white), ambient (world.ambient: the
          map's ambient hue times this, its brightest channel),
          floor lightness / saturation (world.palette_background, palettes.lit_background; the background is redone
          from the current palette's)
  shapes (palette role "hexagon"): glow (emissive_strength in material_over), albedo (body.emissive_albedo: the grey
          a glowing body is lit as, palettes.sync_emissive / glow_tint; also scales the light it throws in the GI)
  tiles (palette role "diamond"): clear (transparency), glow (emissive_strength), reflect (reflectivity),
          absorb (absorption), IOR (ior: how far the floor seen through them is bent), rim (world.gi_edge,
          gl_sdf: see-through tiles lit by the light just outside their cluster's edge, as opaque ones are; 1 = the
          transparency-0 look at any transparency; the filled views only)
Material fields go into each body's material_over, which palettes.sync_emissive leaves alone on a palette switch
(it only rewrites emissive / tint / tint_strength), so the values survive the Palette card; the view modes don't
touch materials at all. The card reads its values back from the world, so it always shows what is drawn.

Click a slider's label to reset that slider to the map's value (an accent mark left of it shows while it differs);
"reset all" resets every slider; "save" writes the values that differ from the map's to OVERRIDES_PATH
(local/visual_overrides.json, gitignored), keyed by map_key. A world built from that map gets them on top of its
lighting block when it first becomes active (attach, called by main_opengl's loop). The map JSON is never written;
a missing or corrupt file means no overrides. The card's header says "map" (the map's own values), "saved" (the
saved overrides) or "edited" (changed since the last save / load).
"""
import json
import math
import os

import pygame

import gl_lights
import lights as L
import materials as M
import palettes
import ui_style as ui
from editor import CARD_TITLE_H
from lighting_ui import BTN_H, GAP, ROW_H, SMALL, VALUE_W, _Card, lighting_on
from ui_style import HUD

OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local", "visual_overrides.json")
LABEL_W = 50  # base px: this card's label column (its labels are longer than the lighting cards')

# (key, label, low, high, step)
SLIDERS = [("gi_gain", "GI gain", 0.0, 20.0, 0.1),
           ("exposure", "exposure", 0.25, 3.0, 0.01),
           ("tone_knee", "knee", 0.2, 0.99, 0.01),
           ("ambient", "ambient", 0.0, 0.5, 0.005),
           ("floor_lightness", "floor lit", 0.0, 1.0, 0.01),
           ("floor_saturation", "floor sat", 0.0, 1.0, 0.01),
           ("shape_glow", "shape glow", 0.0, 4.0, 0.05),
           ("shape_albedo", "shape alb", 0.0, 1.0, 0.01),
           ("tile_clear", "tile clear", 0.0, 1.0, 0.01),
           ("tile_glow", "tile glow", 0.0, 0.5, 0.005),
           ("tile_reflect", "tile refl", 0.0, 1.0, 0.01),
           ("tile_absorb", "tile abs", 0.0, 0.2, 0.002),
           ("tile_rim", "tile rim", 0.0, 1.0, 0.01),
           ("tile_ior", "tile IOR", 1.0, 2.5, 0.01)]
SPECS = {s[0]: s for s in SLIDERS}
ROLE_FIELDS = {"shape_glow": ("hexagon", "emissive_strength"), "tile_clear": ("diamond", "transparency"),
               "tile_glow": ("diamond", "emissive_strength"), "tile_reflect": ("diamond", "reflectivity"),
               "tile_absorb": ("diamond", "absorption"), "tile_ior": ("diamond", "ior")}
EPS = 1e-6


# ---------------- reading and writing the world
def _role_bodies(world, role):
    return [b for b in world.bodies if getattr(b, "palette_role", None) == role]


def _glowing_shapes(world):
    """The shapes whose colour sync_emissive keeps as a glow tint (the albedo slider's bodies)."""
    return [b for b in _role_bodies(world, "hexagon")
            if getattr(b, "emissive_fill", 0) and not getattr(b, "emissive_keep_colour", 0)]


def get(world, key):
    """The world's current value for a slider, or None when it doesn't apply to this world."""
    if key == "gi_gain":
        g = getattr(world, "gi_gain", None)
        return 1.0 if g is None else float(g)
    if key == "exposure":
        e = getattr(world, "exposure", None)
        return 1.0 if e is None else float(e)
    if key == "tone_knee":
        return gl_lights.tone_knee(world)
    if key == "tile_rim":
        return float(getattr(world, "gi_edge", None) or 0.0) if _role_bodies(world, "diamond") else None
    if key == "ambient":
        return float(max(getattr(world, "ambient", None) or L.DEFAULT_AMBIENT))
    if key in ("floor_lightness", "floor_saturation"):
        rule = getattr(world, "palette_background", None)
        if not isinstance(rule, dict) or getattr(world, "palette", None) not in palettes.PALETTES:
            return None
        return float(rule.get(key[6:], 0.4 if key == "floor_lightness" else 0.3))
    if key == "shape_albedo":
        shapes = _glowing_shapes(world)
        if not shapes:
            return None
        a = getattr(shapes[0], "emissive_albedo", None)
        return palettes.GLOW_ALBEDO if a is None else float(a)
    role, field = ROLE_FIELDS[key]
    bodies = _role_bodies(world, role)
    return float(getattr(M.of_body(bodies[0]), field)) if bodies else None


def set_value(world, key, value):
    """Write one slider's value into the world (live: the next frame draws it)."""
    value = float(value)
    if key == "gi_gain":
        world.gi_gain = value
    elif key == "exposure":
        world.exposure = value
    elif key == "tone_knee":
        world.tone_knee = value
    elif key == "tile_rim":
        world.gi_edge = value
    elif key == "ambient":
        amb = tuple(getattr(world, "ambient", None) or L.DEFAULT_AMBIENT)
        base = getattr(world, "look_ambient_hue", None)
        if base is None:  # the map's ambient hue (kept: an ambient of 0 has none)
            m = max(amb)
            base = tuple(c / m for c in amb) if m > 0 else (1.0, 1.0, 1.0)
            world.look_ambient_hue = base
        world.ambient = tuple(round(c * value, 5) for c in base)
    elif key in ("floor_lightness", "floor_saturation"):
        rule = getattr(world, "palette_background", None)
        if not isinstance(rule, dict) or getattr(world, "palette", None) not in palettes.PALETTES:
            return
        rule = dict(rule)
        rule[key[6:]] = value
        world.palette_background = rule
        world.background = palettes.lit_background(palettes.PALETTES[world.palette]["background"], **rule)
    elif key == "shape_albedo":
        for b in _glowing_shapes(world):
            b.emissive_albedo = value
            palettes.sync_emissive(b)
    else:
        role, field = ROLE_FIELDS[key]
        for b in _role_bodies(world, role):
            over = dict(getattr(b, "material_over", None) or {})
            over[field] = value
            b.material_over = over  # (a new dict: the renderer's material table notices the assignment)


def values(world):
    """{key: value} of every slider that applies to this world."""
    out = {}
    for key, *_ in SLIDERS:
        v = get(world, key)
        if v is not None:
            out[key] = v
    return out


def apply(world, vals):
    """Write {key: value} into the world (unknown keys and non-finite values skipped, the rest clamped to range)."""
    for key, v in (vals or {}).items():
        spec = SPECS.get(key)
        if spec is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        if get(world, key) is None:
            continue
        set_value(world, key, min(max(float(v), spec[2]), spec[3]))


# ---------------- the overrides file
def load_all(path=None):
    """{map_key: {key: value}} from the overrides file; {} when it is missing or unreadable."""
    path = path or OVERRIDES_PATH
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, dict)}


def load(map_key, path=None):
    return dict(load_all(path).get(map_key, {}))


def save(map_key, vals, path=None):
    """Store vals (those differing from the map's) for map_key; an empty one removes the map's entry."""
    path = path or OVERRIDES_PATH
    data = load_all(path)
    if vals:
        data[map_key] = {k: round(float(v), 6) for k, v in sorted(vals.items())}
    else:
        data.pop(map_key, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def attach(world, path=None):
    """Once per world: remember the map's own values (the reset targets), then apply the saved overrides on top.
    No-op for worlds that aren't lit maps (no map_key)."""
    if getattr(world, "map_key", None) is None or getattr(world, "look_defaults", None) is not None:
        return
    world.look_defaults = values(world)
    apply(world, load(world.map_key, path))
    world.look_saved = values(world)


def changed(world):
    """{key: value} of the sliders whose value differs from the map's own."""
    defaults = getattr(world, "look_defaults", None) or {}
    return {k: v for k, v in values(world).items() if k in defaults and abs(v - defaults[k]) > EPS}


# ---------------- the card
class LookCard(_Card):
    """The Look card (see the module docstring); get_editor() is the active world's editor."""
    title = "Look"

    def __init__(self, get_editor, path=None):
        super().__init__(get_editor)
        self.path = path

    def wanted(self, view):
        world = self.world
        return lighting_on(view, world) and getattr(world, "map_key", None) is not None

    def card_height(self, compact=False):
        px, pad, gap = HUD.px, HUD.CARD_PAD, HUD.px(GAP)
        return pad + px(CARD_TITLE_H) + len(SLIDERS) * px(ROW_H) + gap + px(BTN_H) + pad

    @staticmethod
    def _track(row):
        x0 = row.x + HUD.px(LABEL_W)
        return pygame.Rect(x0, row.y + HUD.px(3), row.right - HUD.px(VALUE_W) - x0, max(row.h - HUD.px(6), 3))

    def status(self):
        world = self.world
        saved = getattr(world, "look_saved", None)
        if saved is None:
            return ""
        now = values(world)
        if not all(abs(now.get(k, 0) - v) <= EPS for k, v in saved.items()):
            return "edited"  # (off what was last saved / loaded)
        return "saved" if changed(world) else "map"  # (the map's own values: nothing saved for it)

    def set_field(self, key, value):
        set_value(self.world, key, value)

    def reset(self, key=None):
        defaults = getattr(self.world, "look_defaults", None) or {}
        apply(self.world, defaults if key is None else {key: defaults.get(key)})

    def save(self):
        world = self.world
        save(world.map_key, changed(world), self.path)
        world.look_saved = values(world)

    def handle_event(self, event):
        """A click on a slider's label resets it; the rest as the lighting cards."""
        if (self.drag is None and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos)):
            for rect, spec in self.widgets():
                if spec[0] == "slider" and rect.collidepoint(event.pos) and event.pos[0] < self._track(rect).x - HUD.px(3):
                    if spec[3] is not None:
                        self.reset(spec[1][0])
                        self.version += 1
                    return True
        return super().handle_event(event)

    def widgets(self):
        px, pad, gap = HUD.px, HUD.CARD_PAD, HUD.px(GAP)
        world = self.world
        y = self.rect.y + pad + px(CARD_TITLE_H)
        out = []
        for row, spec in self._slider_rows(y, SLIDERS):
            v = get(world, spec[0])
            out.append((row, ("slider", spec, v, self.set_field if v is not None else None)))
        y += len(SLIDERS) * px(ROW_H) + gap
        edited = bool(changed(world))
        for rect, item in self._button_row(y, ["reset", "save"]):
            if item == "reset":
                out.append((rect, ("button", "reset all", False, not edited, self.reset)))
            else:
                out.append((rect, ("button", "save", False, self.status() != "edited", self.save)))
        return out

    def draw_card(self, surface, origin, mouse_px, compact=False):
        super().draw_card(surface, origin, mouse_px, compact)
        defaults = getattr(self.world, "look_defaults", None) or {}
        ox, oy = origin
        for row, spec in self.widgets():  # a mark left of each slider whose value is off the map's
            if spec[0] != "slider" or spec[2] is None:
                continue
            key = spec[1][0]
            if key in defaults and abs(spec[2] - defaults[key]) > EPS:
                r = row.move(-ox, -oy)
                surface.fill(ui.ACCENT, (r.x - HUD.px(4), r.y + HUD.px(3), HUD.px(2), r.h - HUD.px(6)))
