"""Sidebar cards for the lit renderer (research/lighting_contract.md): Material and Lights, plus the lights' gizmos.

Both are sidebar card providers (card_height / place / draw_card / handle_event, like palettes.PaletteCard; see
hud_layer.py), shown by main_opengl.py while lighting is on (L): the Lights card always then, the Material card while
bodies are picked (selection.py: paused, editor closed). They act on the active editor's world (`get_editor()`), and
every change is one world undo step (Editor.before_world_change: "material" / "lights"; a slider drag is one step).

Material card: a button per materials.PRESETS entry (sets the picked bodies' `material`, clearing their overrides),
"reset" (clears the overrides only), a slider per key Material field (writes `material_over[field]` on every picked
body; it shows the first body's effective value), and a colour grid (the editor's colours) for the tint or the
emissive (glow) colour. The last material set this way is what newly spawned bodies get (Editor.settings["material"]).

Lights card: "on" (L), add a point / spot / directional light at the view centre, step through the lights (< >),
delete the selected one, its shadows / on switches, the world's GI switch ("GI": world.gi, glowing bodies light the
room by radiance cascades, gl_sdf.GiEffect), sliders for the light's intensity, radius, falloff, softness, height and
spot cone, the world's ambient strength, and a colour grid for the light's or the ambient colour (hue only: the
brightness is the intensity / the ambient strength). Colours are the swatch's RGB / 255 (scaled so the brightest
channel is 1), used as given (no sRGB decoding).

Gizmos (LightsCard.draw_world, on the world-space preview layer while lighting is on): each light as a small icon in
its colour (grey when off): a disc (point), a disc with its cone (spot), a sun with an arrow (directional; its gizmo
sits at its `pos`, which the renderer ignores). The selected light gets a white ring, its radius circle, the full cone
and an aim handle. Mouse (LightsCard.handle_world_event, in the world, paused or not): press on a gizmo selects it
and drags it, press on the selected light's aim handle turns a spot / directional light; Delete / Backspace removes
the selected light when no bodies are picked.
"""
import math

import pygame

import lights as L
import materials as M
import ui_style as ui
from ui_style import HUD
from editor import CARD_TITLE_H

BTN_H = 13  # base px: a small button row
ROW_H = 11  # one slider row
GAP = 2
SWATCH = 8  # colour grid cell (base px; cells stretch to fill the row)
SWATCH_COLS = 15  # cells per row (at the card's base width, 150 - 2 x 5 px)
SMALL = 9  # font size of the slider labels and small buttons
LABEL_W = 40  # slider label column
VALUE_W = 26  # slider value column
GIZMO_PX = 4  # UI px radius of a light's icon
PICK_PX = 7  # UI px: a press this close to a light's icon (or the selected light's aim handle) grabs it
AIM_PX = 26  # UI px from a spot / directional light to its aim handle

# (Material field, label, low, high, step)
MATERIAL_SLIDERS = [("specular", "spec", 0.0, 1.0, 0.05), ("roughness", "rough", 0.0, 1.0, 0.05),
                    ("bands", "bands", 0, 6, 1), ("bevel_px", "bevel", 0.0, 12.0, 0.5),
                    ("emissive_strength", "glow", 0.0, 3.0, 0.1), ("tint_strength", "tint", 0.0, 1.0, 0.05),
                    ("transparency", "clear", 0.0, 1.0, 0.05), ("ior", "ior", 1.0, 2.5, 0.05),
                    ("reflectivity", "reflect", 0.0, 1.0, 0.05), ("absorption", "absorb", 0.0, 0.2, 0.005),
                    ("pattern", "pattern", 0, 4, 1), ("pattern_strength", "pat str", 0.0, 1.0, 0.05),
                    ("pattern_scale", "pat size", 1.0, 20.0, 0.5)]
PATTERN_NAMES = ("none", "noise", "stripes", "checker", "dots")
# (Light field, label, low, high, step)
LIGHT_SLIDERS = [("intensity", "power", 0.0, 4.0, 0.05), ("radius", "radius", 20.0, 2000.0, 10.0),
                 ("falloff", "falloff", 0.25, 4.0, 0.05), ("softness", "soft", 0.0, 40.0, 0.5),
                 ("height", "height", 0.0, 400.0, 5.0), ("cone_inner", "inner", 1.0, 89.0, 1.0),
                 ("cone_outer", "outer", 1.0, 90.0, 1.0)]
AMBIENT_SLIDER = ("ambient", "ambient", 0.0, 1.0, 0.02)
ADD_KINDS = ((L.POINT, "+point"), (L.SPOT, "+spot"), (L.DIRECTIONAL, "+sun"))


def quantize(frac, lo, hi, step):
    v = lo + min(max(frac, 0.0), 1.0) * (hi - lo)
    v = lo + round((v - lo) / step) * step
    v = min(max(v, lo), hi)
    return int(round(v)) if isinstance(step, int) else round(v, 6)


def hue(rgb):
    """A swatch colour (0..255) as linear 0..1 RGB with its brightest channel at 1 (black stays black)."""
    m = max(rgb)
    return tuple(c / m for c in rgb) if m > 0 else (0.0, 0.0, 0.0)


def lighting_on(view, world):
    """Is the lit renderer on for this world? ViewConfig.lighting, which main_opengl keeps equal to world.lighting."""
    return bool(getattr(view, "lighting", False) or getattr(world, "lighting", False))


class _Card:
    """Shared plumbing: a list of widgets (rect, spec) in native screen px, a slider drag, a version for redraws."""
    title = ""

    def __init__(self, get_editor):
        self.get_editor = get_editor
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.drag = None  # {"spec": slider spec, "track": rect} while a slider is held
        self.version = 0  # bumped on every change (main_opengl's sidebar redraw key)

    @property
    def editor(self):
        return self.get_editor()

    @property
    def world(self):
        return self.get_editor().world

    def place(self, rect):
        self.rect = pygame.Rect(rect)

    # ---------------- layout helpers (native px)
    @staticmethod
    def _swatch_rows(n):
        return -(-n // SWATCH_COLS)

    def _grid_height(self, n):
        rows = self._swatch_rows(n)
        return rows * HUD.px(SWATCH) + (rows - 1) * HUD.px(1)

    def _button_row(self, y, items):
        """[(rect, item)] for buttons sharing a row at y, equal widths."""
        pad, gap = HUD.CARD_PAD, HUD.px(GAP)
        x0, w = self.rect.x + pad, self.rect.w - 2 * pad
        n = len(items)
        bw = (w - (n - 1) * gap) // n
        out = []
        for k, item in enumerate(items):
            x = x0 + k * (bw + gap)
            out.append((pygame.Rect(x, y, bw if k < n - 1 else x0 + w - x, HUD.px(BTN_H)), item))
        return out

    def _swatches(self, y):
        """[(rect, (name, rgb))] for the editor's colours from row y."""
        colors = self.editor.colors
        pad, gap, sh = HUD.CARD_PAD, HUD.px(1), HUD.px(SWATCH)
        x0, w = self.rect.x + pad, self.rect.w - 2 * pad
        sw = (w - (SWATCH_COLS - 1) * gap) / SWATCH_COLS
        return [(pygame.Rect(round(x0 + (k % SWATCH_COLS) * (sw + gap)), y + (k // SWATCH_COLS) * (sh + gap),
                             max(round(sw), 1), sh), item) for k, item in enumerate(colors)]

    def _slider_rows(self, y, specs):
        """[(row rect, spec)] for slider rows from y."""
        pad, rh = HUD.CARD_PAD, HUD.px(ROW_H)
        return [(pygame.Rect(self.rect.x + pad, y + k * rh, self.rect.w - 2 * pad, rh), spec)
                for k, spec in enumerate(specs)]

    @staticmethod
    def _track(row):
        x0 = row.x + HUD.px(LABEL_W)
        return pygame.Rect(x0, row.y + HUD.px(3), row.right - HUD.px(VALUE_W) - x0, max(row.h - HUD.px(6), 3))

    # ---------------- events
    def handle_event(self, event):
        """Clicks on the card, and a slider drag wherever the mouse goes. True if used."""
        if self.drag is not None:
            if event.type == pygame.MOUSEMOTION:
                self._drag_to(event.pos[0])
                return True
            if event.type == pygame.MOUSEBUTTONUP:
                self.drag = None
                self.version += 1
                return True
        if event.type != pygame.MOUSEBUTTONDOWN or not self.rect.collidepoint(event.pos):
            return False
        if event.button != 1:
            return True
        for rect, spec in self.widgets():
            if spec[0] == "slider":
                track = self._track(rect)
                grab = pygame.Rect(track.x - HUD.px(3), rect.y, track.w + 2 * HUD.px(3), rect.h)
                if grab.collidepoint(event.pos):
                    if spec[2] is not None and spec[3] is not None:
                        self.drag = {"spec": spec, "track": track, "pushed": False}
                        self._drag_to(event.pos[0])
                    break
            elif rect.collidepoint(event.pos):
                action = spec[-1]
                if action is not None:
                    action()
                    self.version += 1
                break
        return True

    def _drag_to(self, x):
        d = self.drag
        _, (field, _label, lo, hi, step), _value, setter = d["spec"]
        value = quantize((x - d["track"].x) / max(d["track"].w, 1), lo, hi, step)
        if not d["pushed"]:  # one undo step per drag
            d["pushed"] = True
            self.before_change()
        setter(field, value)
        self.version += 1

    def before_change(self):
        pass

    # ---------------- drawing
    def _draw_slider(self, surface, row, spec, mouse_px, ox, oy, fmt=None):
        _, (field, label, lo, hi, step), value, setter = spec
        r = row.move(-ox, -oy)
        enabled = setter is not None and value is not None
        col = ui.TEXT if enabled else ui.TEXT_DIM
        HUD.text(surface, label, (r.x, r.centery), col, "regular", HUD.px(SMALL), valign="middle")
        track = self._track(row).move(-ox, -oy)
        pygame.draw.rect(surface, ui.BLACK, track)
        inner = track.inflate(-2, -2)
        pygame.draw.rect(surface, ui.BUTTON_DISABLED, inner)
        if enabled:
            frac = (value - lo) / (hi - lo) if hi > lo else 0.0
            frac = min(max(frac, 0.0), 1.0)
            pygame.draw.rect(surface, ui.BUTTON_SELECTED, (inner.x, inner.y, int(inner.w * frac), inner.h))
            kx = track.x + int(track.w * frac)
            hovered = self._track(row).inflate(HUD.px(6), 0).collidepoint(mouse_px) or (
                self.drag is not None and self.drag["spec"][1][0] == field)
            pygame.draw.rect(surface, ui.WHITE if hovered else ui.TEXT, (kx - HUD.px(1), r.y + HUD.px(1), HUD.px(3),
                                                                          r.h - HUD.px(2)))
            text = fmt(value) if fmt else (f"{value:d}" if isinstance(step, int) else
                                           f"{value:.0f}" if step >= 1 else f"{value:.3f}" if step < 0.01 else
                                           f"{value:.2f}")
            HUD.text(surface, text, (r.right, r.centery), ui.WHITE, "bold", HUD.px(SMALL), align="right",
                     valign="middle")

    def _draw_buttons(self, surface, widgets, mouse_px, ox, oy):
        for rect, spec in widgets:
            kind = spec[0]
            local = rect.move(-ox, -oy)
            if kind == "button":
                _, label, selected, disabled, _action = spec
                HUD.draw_button(surface, local, label, rect.collidepoint(mouse_px) and not disabled, selected, disabled,
                                HUD.px(SMALL))
            elif kind == "chip":  # a button with a colour chip on its left
                _, label, selected, rgb01, _action = spec
                HUD.draw_button(surface, local, None, rect.collidepoint(mouse_px), selected)
                chip = pygame.Rect(local.x + HUD.px(3), local.y + HUD.px(3), local.h - HUD.px(6), local.h - HUD.px(6))
                surface.fill(tuple(int(255 * min(max(c, 0.0), 1.0)) for c in rgb01), chip)
                pygame.draw.rect(surface, ui.BLACK, chip, 1)
                HUD.text(surface, label, (chip.right + HUD.px(3), local.centery), ui.WHITE, "bold", HUD.px(SMALL),
                         valign="middle")
            elif kind == "swatch":
                _, rgb, _action = spec
                surface.fill(rgb, local)
                if rect.collidepoint(mouse_px):
                    pygame.draw.rect(surface, ui.WHITE, local.inflate(2, 2), 1)

    def draw_card(self, surface, origin, mouse_px, compact=False):
        ox, oy = origin
        r = self.rect.move(-ox, -oy)
        HUD.draw_frame(surface, r)
        pad, px = HUD.CARD_PAD, HUD.px
        HUD.text(surface, self.title, (r.x + pad, r.y + pad - px(1)), ui.WHITE, "bold", HUD.TITLE_SIZE)
        status = self.status()
        if status:
            HUD.text(surface, status, (r.right - pad, r.y + pad), ui.ACCENT, "bold", align="right")
        widgets = self.widgets()
        self._draw_buttons(surface, [w for w in widgets if w[1][0] != "slider"], mouse_px, ox, oy)
        for row, spec in widgets:
            if spec[0] == "slider":
                self._draw_slider(surface, row, spec, mouse_px, ox, oy, self.slider_format(spec[1][0]))

    def status(self):
        return ""

    def slider_format(self, field):
        return None


class MaterialCard(_Card):
    """The Material card for the active editor's picked bodies (see the module docstring)."""
    title = "Material"

    def bodies(self):
        """The picked bodies (chains' boxes included), in world order; [] when nothing is picked."""
        ed = self.editor
        if not hasattr(ed, "_live_picked"):
            return []
        return ed._gather(ed._live_picked(), carry=False)[0]

    def wanted(self, view):
        """Shown while lighting is on and bodies are picked."""
        return lighting_on(view, self.world) and bool(self.bodies())

    def card_height(self, compact=False):
        px, pad, gap = HUD.px, HUD.CARD_PAD, HUD.px(GAP)
        preset_rows = -(-(len(M.PRESETS) + 1) // 3)
        return (pad + px(CARD_TITLE_H) + preset_rows * px(BTN_H) + (preset_rows - 1) * gap + gap
                + len(MATERIAL_SLIDERS) * px(ROW_H) + gap + px(BTN_H) + gap + self._grid_height(len(self.editor.colors))
                + pad)

    def __init__(self, get_editor):
        super().__init__(get_editor)
        self.target = "tint"  # what the colour grid sets: "tint" or "emissive"

    def status(self):
        n = len(self.bodies())
        return f"{n} bod{'y' if n == 1 else 'ies'}"

    def slider_format(self, field):
        return (lambda v: PATTERN_NAMES[int(v)] if 0 <= int(v) < len(PATTERN_NAMES) else str(v)) if field == "pattern" else None

    def before_change(self):
        self.editor.before_world_change("material")

    def _remember(self):
        """New bodies spawned by the editor get the first picked body's material (Editor.settings["material"])."""
        bodies = self.bodies()
        settings = getattr(self.editor, "settings", None)
        if bodies and isinstance(settings, dict):
            b = bodies[0]
            settings["material"] = {"preset": getattr(b, "material", None),
                                    "over": dict(getattr(b, "material_over", None) or {})}

    def set_preset(self, name):
        bodies = self.bodies()
        if not bodies:
            return
        self.before_change()
        for b in bodies:
            b.material, b.material_over = name, {}
        self._remember()

    def reset_overrides(self):
        bodies = self.bodies()
        if not bodies:
            return
        self.before_change()
        for b in bodies:
            b.material_over = {}
        self._remember()

    def set_field(self, field, value):
        """Override one Material field on every picked body (the undo step is the caller's: a drag pushes once)."""
        for b in self.bodies():
            over = dict(getattr(b, "material_over", None) or {})
            over[field] = value
            b.material_over = over
        self._remember()

    def set_color(self, rgb):
        """The colour grid: the tint or emissive colour of every picked body; a strength of 0 is raised so it shows."""
        bodies = self.bodies()
        if not bodies:
            return
        self.before_change()
        col = tuple(round(c / 255, 4) for c in rgb)
        strength = "tint_strength" if self.target == "tint" else "emissive_strength"
        for b in bodies:
            over = dict(getattr(b, "material_over", None) or {})
            over[self.target] = col
            if getattr(M.of_body(b), strength) <= 0:
                over[strength] = 0.5 if self.target == "tint" else 1.0
            b.material_over = over
        self._remember()

    def widgets(self):
        px, pad, gap = HUD.px, HUD.CARD_PAD, HUD.px(GAP)
        bodies = self.bodies()
        first = bodies[0] if bodies else None
        mat = M.of_body(first) if first is not None else None
        current = getattr(first, "material", None) or M.DEFAULT if first is not None else None
        out = []
        y = self.rect.y + pad + px(CARD_TITLE_H)
        items = list(M.PRESETS) + [None]
        for start in range(0, len(items), 3):
            row = items[start:start + 3]
            for rect, name in self._button_row(y, row + [False] * (3 - len(row))):
                if name is False:
                    continue
                if name is None:
                    over = bool(getattr(first, "material_over", None))
                    out.append((rect, ("button", "reset", False, not over, self.reset_overrides)))
                else:
                    out.append((rect, ("button", name, name == current, False, lambda n=name: self.set_preset(n))))
            y += px(BTN_H) + gap
        for row, spec in self._slider_rows(y, MATERIAL_SLIDERS):
            value = getattr(mat, spec[0]) if mat is not None else None
            if isinstance(value, bool):
                value = int(value)
            out.append((row, ("slider", spec, value, self.set_field if mat is not None else None)))
        y += len(MATERIAL_SLIDERS) * px(ROW_H) + gap
        for rect, target in self._button_row(y, ["tint", "emissive"]):
            col = getattr(mat, target) if mat is not None else (1.0, 1.0, 1.0)
            out.append((rect, ("chip", "tint" if target == "tint" else "glow colour", self.target == target, col,
                               lambda t=target: setattr(self, "target", t))))
        y += px(BTN_H) + gap
        for rect, (_name, rgb) in self._swatches(y):
            out.append((rect, ("swatch", rgb, lambda c=rgb: self.set_color(c))))
        return out


class LightsCard(_Card):
    """The Lights card and the lights' gizmos and mouse handling (see the module docstring)."""
    title = "Lights"

    def __init__(self, get_editor, view=None, toggle_lighting=None):
        """view: the ViewConfig (its `lighting`); toggle_lighting(): what L does (main_opengl's)."""
        super().__init__(get_editor)
        self.view, self.toggle_lighting = view, toggle_lighting
        self._selected = None  # (world, index into world.lights)
        self.target = "light"  # what the colour grid sets: "light" or "ambient"
        self.gizmo_drag = None  # {"index", "mode": "move" / "aim", "pushed", "grab": offset}

    def wanted(self, view):
        return lighting_on(view, self.world)

    def card_height(self, compact=False):
        px, pad, gap = HUD.px, HUD.CARD_PAD, HUD.px(GAP)
        return (pad + px(CARD_TITLE_H) + 3 * (px(BTN_H) + gap) + (len(LIGHT_SLIDERS) + 1) * px(ROW_H) + gap
                + px(BTN_H) + gap + self._grid_height(len(self.editor.colors)) + pad)

    # ---------------- the world's lights
    def lights(self):
        return list(getattr(self.world, "lights", None) or [])

    @property
    def selected(self):
        """Index of the selected light in the active world's lights, or None."""
        sel = self._selected
        if sel is None or sel[0] is not self.world or sel[1] >= len(self.lights()):
            return None
        return sel[1]

    def select(self, index):
        self._selected = None if index is None else (self.world, index)
        self.version += 1

    def light(self):
        k = self.selected
        return None if k is None else self.lights()[k]

    def before_change(self):
        self.editor.before_world_change("lights")

    def ambient(self):
        return tuple(getattr(self.world, "ambient", None) or L.DEFAULT_AMBIENT)

    def add(self, kind):
        """A new light of that kind at the view centre (world units), selected."""
        ed, world = self.editor, self.world
        self.before_change()
        if getattr(world, "lights", None) is None:
            world.lights = []
        centre = (ed.world_w / 2, ed.world_h / 2)
        if kind == L.SPOT:
            light = L.Light(kind=kind, pos=centre, dir=(0.0, 1.0), intensity=1.5)
        elif kind == L.DIRECTIONAL:
            n = math.hypot(0.35, 1.0)
            light = L.Light(kind=kind, pos=centre, dir=(0.35 / n, 1.0 / n), intensity=0.3, height=200.0)
        else:
            light = L.Light(kind=kind, pos=centre)
        world.lights.append(light)
        self.select(len(world.lights) - 1)

    def delete(self):
        k = self.selected
        if k is None:
            return
        self.before_change()
        del self.world.lights[k]
        self.select(None)

    def step_selection(self, d):
        n = len(self.lights())
        if n:
            k = self.selected
            self.select((d if d > 0 else n - 1) % n if k is None else (k + d) % n)

    def toggle_field(self, field):
        light = self.light()
        if light is not None:
            self.before_change()
            setattr(light, field, not getattr(light, field))

    def toggle_gi(self):
        """The world's GI switch (world.gi, saved with the scene; gl_sdf.GiEffect also runs while GI_ENABLED)."""
        self.world.gi = not bool(getattr(self.world, "gi", False))

    def set_field(self, field, value):
        if field == "ambient":
            amb = self.ambient()
            h = hue(tuple(c * 255 for c in amb)) if max(amb) > 0 else (1.0, 1.0, 1.0)
            self.world.ambient = tuple(round(c * value, 4) for c in h)
            return
        light = self.light()
        if light is None:
            return
        setattr(light, field, float(value))
        if field == "cone_inner" and light.cone_outer < light.cone_inner:  # inner never past outer
            light.cone_outer = float(value)
        elif field == "cone_outer" and light.cone_inner > light.cone_outer:
            light.cone_inner = float(value)

    def set_color(self, rgb):
        if self.target == "ambient":
            self.before_change()
            strength = max(self.ambient())
            self.world.ambient = tuple(round(c * strength, 4) for c in hue(rgb))
            return
        light = self.light()
        if light is not None:
            self.before_change()
            light.color = tuple(round(c, 4) for c in hue(rgb))

    def status(self):
        return "on" if lighting_on(self.view, self.world) else "off"

    def widgets(self):
        px, pad, gap = HUD.px, HUD.CARD_PAD, HUD.px(GAP)
        lights, k = self.lights(), self.selected
        light = self.light()
        out = []
        y = self.rect.y + pad + px(CARD_TITLE_H)
        for rect, (kind, label) in self._button_row(y, ADD_KINDS):
            out.append((rect, ("button", label, False, False, lambda kd=kind: self.add(kd))))
        y += px(BTN_H) + gap
        name = "none" if light is None else f"{'sun' if light.kind == L.DIRECTIONAL else light.kind} {k + 1}/{len(lights)}"
        row = self._button_row(y, ["<", "name", ">", "del"])
        for rect, item in row:
            if item == "<":
                out.append((rect, ("button", "<", False, not lights, lambda: self.step_selection(-1))))
            elif item == ">":
                out.append((rect, ("button", ">", False, not lights, lambda: self.step_selection(1))))
            elif item == "del":
                out.append((rect, ("button", "del", False, light is None, self.delete)))
            else:
                out.append((rect, ("button", name, light is not None, False, None)))
        y += px(BTN_H) + gap
        for rect, field in self._button_row(y, ["casts_shadows", "enabled", "gi", "lighting"]):
            if field == "gi":  # the world's radiance-cascade GI (gl_sdf.GiEffect): glowing bodies light the room
                on = bool(getattr(self.world, "gi", False))
                out.append((rect, ("button", "GI", on, False, self.toggle_gi)))
            elif field == "lighting":
                on = lighting_on(self.view, self.world)
                out.append((rect, ("button", "[L] on" if on else "[L] off", on, self.toggle_lighting is None,
                                   self.toggle_lighting)))
            else:
                label = "shadows" if field == "casts_shadows" else "enabled"
                out.append((rect, ("button", label, light is not None and getattr(light, field), light is None,
                                   lambda f=field: self.toggle_field(f))))
        y += px(BTN_H) + gap
        specs = LIGHT_SLIDERS + [AMBIENT_SLIDER]
        for row, spec in self._slider_rows(y, specs):
            field = spec[0]
            if field == "ambient":
                out.append((row, ("slider", spec, max(self.ambient()), self.set_field)))
                continue
            usable = light is not None and (field not in ("cone_inner", "cone_outer") or light.kind == L.SPOT) and (
                field not in ("radius", "falloff") or light.kind != L.DIRECTIONAL)
            out.append((row, ("slider", spec, getattr(light, field) if usable else None,
                              self.set_field if usable else None)))
        y += len(specs) * px(ROW_H) + gap
        for rect, target in self._button_row(y, ["light", "ambient"]):
            col = self.ambient() if target == "ambient" else (light.color if light is not None else (0.5, 0.5, 0.5))
            if target == "ambient" and max(col) > 0:
                col = hue(tuple(c * 255 for c in col))
            out.append((rect, ("chip", "light colour" if target == "light" else "ambient", self.target == target,
                               col, lambda t=target: setattr(self, "target", t))))
        y += px(BTN_H) + gap
        for rect, (_name, rgb) in self._swatches(y):
            out.append((rect, ("swatch", rgb, lambda c=rgb: self.set_color(c))))
        return out

    # ---------------- gizmos: mouse (world events) and drawing
    def _aim_point(self, light):
        """The aim handle of a spot / directional light (world units), or None."""
        if light.kind == L.POINT:
            return None
        z = self.editor.zoom
        dx, dy = light.dir
        n = math.hypot(dx, dy) or 1.0
        return light.pos[0] + dx / n * AIM_PX / z, light.pos[1] + dy / n * AIM_PX / z

    def _gizmo_at(self, wp):
        """(index, "move" / "aim") of the gizmo under wp (world units), or None. The selected light's aim handle
        first, then the nearest icon."""
        z, lights = self.editor.zoom, self.lights()
        k = self.selected
        if k is not None:
            aim = self._aim_point(lights[k])
            if aim is not None and math.hypot(wp[0] - aim[0], wp[1] - aim[1]) * z <= PICK_PX:
                return k, "aim"
        best = None
        for i, light in enumerate(lights):
            d = math.hypot(wp[0] - light.pos[0], wp[1] - light.pos[1]) * z
            if d <= PICK_PX and (best is None or d < best[0]):
                best = (d, i)
        return None if best is None else (best[1], "move")

    def handle_world_event(self, event):
        """Mouse in the world while lighting is on: gizmo picks and drags; Delete removes the selected light when no
        bodies are picked. True if used."""
        ed = self.editor
        if not lighting_on(self.view, self.world) or ed.active or ed.scenes.state is not None:
            self.gizmo_drag = None
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and self.selected is not None and not (
                    hasattr(ed, "_live_picked") and ed._live_picked()):
                self.delete()
                return True
            return False
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return False
        wp = ed.to_world(ed.to_ui(event.pos))
        g = self.gizmo_drag
        if g is not None:
            if event.type == pygame.MOUSEMOTION:
                self._gizmo_drag_to(wp)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.gizmo_drag = None
                self.version += 1
            return True
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1 or ed.sidebar.collidepoint(event.pos):
            return False
        hit = self._gizmo_at(wp)
        if hit is None:
            return False
        k, mode = hit
        self.select(k)
        pos = self.lights()[k].pos
        self.gizmo_drag = {"index": k, "mode": mode, "pushed": False, "grab": (pos[0] - wp[0], pos[1] - wp[1])}
        return True

    def _gizmo_drag_to(self, wp):
        g = self.gizmo_drag
        lights = self.lights()
        if g["index"] >= len(lights):
            self.gizmo_drag = None
            return
        if not g["pushed"]:  # one undo step per drag, taken only once it moves
            g["pushed"] = True
            self.before_change()
            lights = self.lights()
        light = lights[g["index"]]
        if g["mode"] == "move":
            light.pos = (float(wp[0] + g["grab"][0]), float(wp[1] + g["grab"][1]))
        else:
            dx, dy = wp[0] - light.pos[0], wp[1] - light.pos[1]
            n = math.hypot(dx, dy)
            if n > 1e-6:
                light.dir = (float(dx / n), float(dy / n))
        self.version += 1

    def gizmos_shown(self):
        return lighting_on(self.view, self.world)

    def draw_world(self, surface):
        """Every light's gizmo on the world-space preview layer (render px = world units x zoom)."""
        if not self.gizmos_shown():
            return
        z, sel = self.editor.zoom, self.selected
        for i, light in enumerate(self.lights()):
            col = tuple(int(255 * min(max(c, 0.0), 1.0)) for c in light.color) if light.enabled else (110, 110, 110)
            x, y = light.pos[0] * z, light.pos[1] * z
            c = (round(x), round(y))
            dx, dy = light.dir
            n = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n, dy / n
            selected = i == sel
            if light.kind == L.SPOT:
                reach = light.radius * z if selected else 3 * AIM_PX
                for half, width in ((light.cone_outer, 1),) + (((light.cone_inner, 1),) if selected else ()):
                    for s in (-1, 1):
                        t = math.radians(half) * s
                        ex = dx * math.cos(t) - dy * math.sin(t)
                        ey = dx * math.sin(t) + dy * math.cos(t)
                        pygame.draw.line(surface, col, c, (round(x + ex * reach), round(y + ey * reach)), width)
            elif light.kind == L.DIRECTIONAL:
                for k in range(8):  # sun rays
                    a = k * math.pi / 4
                    pygame.draw.line(surface, col, (round(x + math.cos(a) * (GIZMO_PX + 2)), round(y + math.sin(a) * (GIZMO_PX + 2))),
                                     (round(x + math.cos(a) * (GIZMO_PX + 4)), round(y + math.sin(a) * (GIZMO_PX + 4))))
                tip = (round(x + dx * AIM_PX), round(y + dy * AIM_PX))
                pygame.draw.line(surface, col, c, tip)
            if selected and light.kind == L.POINT:
                pygame.draw.circle(surface, col, c, max(int(light.radius * z), GIZMO_PX + 2), 1)
            pygame.draw.circle(surface, ui.BLACK, c, GIZMO_PX + 1)
            pygame.draw.circle(surface, col, c, GIZMO_PX)
            if selected:
                pygame.draw.circle(surface, ui.WHITE, c, GIZMO_PX + 3, 1)
                aim = self._aim_point(light)
                if aim is not None:
                    a = (round(aim[0] * z), round(aim[1] * z))
                    pygame.draw.rect(surface, ui.BLACK, (a[0] - 3, a[1] - 3, 7, 7))
                    pygame.draw.rect(surface, ui.WHITE, (a[0] - 2, a[1] - 2, 5, 5))
