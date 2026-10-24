"""Gravity selector: a small native-resolution card in the bottom-left corner with 8 diamond buttons placed radially
around a centre button. Clicking a diamond (or pressing the numpad key in the same spot) points gravity that way,
keeping its magnitude; the centre button (numpad 5) turns gravity off, and clicking it again turns it back on in the
last direction.

Gravity is EnvConfig.gravity (px/s^2, +y down). Both physics backends read it every substep (solver.World and
gpu_backend.GpuBackend add EnvConfig.gravity * h to the velocities), so it's changed in place and takes effect on the
next step, running or paused.

Drawn on its own small per-pixel-alpha surface (`surface`, at `rect` in native screen px) in the HUD style
(ui_style.HUD), which main_opengl.py blends over the frame like the HUD cards. The surface is only redrawn (and
`version` bumped, so the presenter re-uploads it) when the hover or the direction changes.
"""
import math

import numpy as np
import pygame

import ui_style as ui
from entities import EnvConfig
from ui_style import HUD

DEFAULT_MAGNITUDE = 500.0  # px/s^2: used when gravity starts at zero
# (name, unit direction in screen coords (+y down), numpad key, compass point); clockwise from up
_D = math.sqrt(0.5)
DIRECTIONS = [("up", (0.0, -1.0), pygame.K_KP8, "N"), ("up-right", (_D, -_D), pygame.K_KP9, "NE"),
              ("right", (1.0, 0.0), pygame.K_KP6, "E"), ("down-right", (_D, _D), pygame.K_KP3, "SE"),
              ("down", (0.0, 1.0), pygame.K_KP2, "S"), ("down-left", (-_D, _D), pygame.K_KP1, "SW"),
              ("left", (-1.0, 0.0), pygame.K_KP4, "W"), ("up-left", (-_D, -_D), pygame.K_KP7, "NW")]
ZERO_KEY = pygame.K_KP5
ZERO_NAME = "off"
KEY_DIRECTION = {k: i for i, (_, _, k, _) in enumerate(DIRECTIONS)}

# layout in base (UI-style) px; HUD.px scales them to native px
CARD_W = 72
TITLE_H = 16
DIAL_R = 28  # dial radius incl. the diamonds' outer tips
INNER = 8  # centre -> a diamond's inner tip
OUTER = 27  # centre -> its outer tip
HALF_W = 5  # half of a diamond's short axis (about half the long one, so the diagonal ones still read as diamonds)
CENTRE = 5  # the centre button's half diagonal
LINE_H = 13


def direction_index(g=None):
    """The DIRECTIONS index gravity `g` (default EnvConfig.gravity) points along, None if zero or between them."""
    g = EnvConfig.gravity if g is None else np.asarray(g, dtype=float)
    n = float(np.hypot(*g))
    if n == 0:
        return None
    for i, (_, d, _, _) in enumerate(DIRECTIONS):
        if abs(g[0] / n - d[0]) < 1e-6 and abs(g[1] / n - d[1]) < 1e-6:
            return i
    return None


def direction_name(g=None, compass=False):
    """'down', 'up-left', ... (compass: 'S', 'NW', ...), 'off' for zero gravity, 'custom' for any other direction."""
    g = EnvConfig.gravity if g is None else g
    i = direction_index(g)
    if i is not None:
        return DIRECTIONS[i][3 if compass else 0]
    return ZERO_NAME if not np.any(g) else "custom"


class GravityDial:
    def __init__(self, screen_size, ui_scale=2):
        """screen_size: native screen (window) px; ui_scale: native px per UI-surface px. The card sits in the
        bottom-left corner, its top-left on a multiple of ui_scale so the UI-res cursor drawn over it lines up."""
        px = HUD.px
        w = px(CARD_W)
        h = HUD.CARD_PAD + px(TITLE_H) + 2 * px(DIAL_R) + px(LINE_H) + HUD.CARD_PAD
        y = (screen_size[1] - HUD.CARD_MARGIN - h) // ui_scale * ui_scale
        self.rect = pygame.Rect(HUD.CARD_MARGIN // ui_scale * ui_scale, y, w, h)
        self.ui_scale = ui_scale
        self._top = pygame.Surface((-(-w // ui_scale), -(-h // ui_scale)), pygame.SRCALPHA)  # UI-res overlay
        self.surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self.visible = True  # the main loop hides it with the HUD (SPACE) and behind the editor's modal panels
        self.version = 0  # bumped whenever the surface is redrawn
        n = float(np.hypot(*EnvConfig.gravity))
        self.magnitude = n or DEFAULT_MAGNITUDE
        self.last = direction_index() if direction_index() is not None else 4  # where the centre button turns it back on
        self._drawn_key = None
        # dial centre, native screen px
        self.centre = np.array([self.rect.centerx, self.rect.y + HUD.CARD_PAD + px(TITLE_H) + px(DIAL_R)], dtype=float)

    # ---------------- State
    def set_direction(self, i):
        """Point gravity along DIRECTIONS[i] (None = zero gravity), keeping its magnitude."""
        n = float(np.hypot(*EnvConfig.gravity))
        if n:
            self.magnitude = n
        if i is None:
            EnvConfig.gravity[:] = 0.0
        else:
            EnvConfig.gravity[:] = np.array(DIRECTIONS[i][1]) * self.magnitude
            self.last = i

    def toggle_zero(self):
        """The centre button / numpad 5: gravity off, or back on in the last direction if it already is off."""
        self.set_direction(self.last if not np.any(EnvConfig.gravity) else None)

    # ---------------- Geometry (native screen px)
    def _diamond(self, i, inset=0.0):
        """DIRECTIONS[i]'s diamond: 4 vertices (inner tip, side, outer tip, side), shrunk by `inset` px per edge."""
        u = np.array(DIRECTIONS[i][1])
        v = np.array([-u[1], u[0]])
        a, b, hw = HUD.px(INNER), HUD.px(OUTER), HUD.px(HALF_W)
        half_l = (b - a) / 2
        mid = self.centre + u * (a + b) / 2
        k = self._shrink(half_l, hw, inset)
        return [mid - u * half_l * k, mid + v * hw * k, mid + u * half_l * k, mid - v * hw * k]

    def _centre_diamond(self, inset=0.0):
        r = HUD.px(CENTRE)
        k = self._shrink(r, r, inset)
        c = self.centre
        return [c + (0, -r * k), c + (r * k, 0), c + (0, r * k), c + (-r * k, 0)]

    @staticmethod
    def _shrink(half_l, half_w, inset):
        """Scale factor that moves a rhombus' edges `inset` px inwards (its edges are half_l*half_w/hyp from the centre)."""
        d = half_l * half_w / math.hypot(half_l, half_w)
        return max(d - inset, 0.0) / d

    @staticmethod
    def _inside(poly, p):
        """Is p inside (or on) the rhombus poly (inner tip, side, outer tip, side)?"""
        c = (poly[0] + poly[2]) / 2
        u, v = poly[2] - c, poly[1] - c
        q = np.asarray(p, dtype=float) - c
        return abs(q @ u) / (u @ u) + abs(q @ v) / (v @ v) <= 1.0

    def button_at(self, pos):
        """'zero', a DIRECTIONS index, or None for a screen position (native px)."""
        if not self.visible or not self.rect.collidepoint(pos):
            return None
        if self._inside(self._centre_diamond(), pos):
            return "zero"
        for i in range(len(DIRECTIONS)):
            if self._inside(self._diamond(i), pos):
                return i
        return None

    def button_center(self, which):
        """Screen px centre of a button ('zero' or a DIRECTIONS index), for tests."""
        poly = self._centre_diamond() if which == "zero" else self._diamond(which)
        c = (poly[0] + poly[2]) / 2
        return int(round(c[0])), int(round(c[1]))

    # ---------------- Events
    def handle_event(self, event):
        """Numpad keys (always) and left clicks on the card (while visible). True if used; clicks anywhere on the card
        are swallowed so they don't grab bodies or place points under it."""
        if event.type == pygame.KEYDOWN:
            if event.key in KEY_DIRECTION:
                self.set_direction(KEY_DIRECTION[event.key])
                return True
            if event.key == ZERO_KEY:
                self.toggle_zero()
                return True
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and self.visible and self.rect.collidepoint(event.pos):
            if event.button == 1:
                hit = self.button_at(event.pos)
                if hit == "zero":
                    self.toggle_zero()
                elif hit is not None:
                    self.set_direction(hit)
            return True
        return False

    # ---------------- Drawing
    def draw(self, mouse_px, overlay=None):
        """Redraw the surface if the hover or the direction changed. mouse_px: the mouse in native screen px.
        overlay(surface, origin): draws what goes over the HUD (the cursor) at UI px minus origin, as for
        HudLayer.overlay; while the mouse is near the card it's redrawn every frame with that on top."""
        hover = self.button_at(mouse_px)
        current = direction_index()
        near = overlay is not None and self.rect.inflate(8 * self.ui_scale, 8 * self.ui_scale).collidepoint(mouse_px)
        key = (hover, current, not np.any(EnvConfig.gravity), direction_name(), tuple(mouse_px) if near else None)
        if key == self._drawn_key:
            return
        self._drawn_key = key
        self.version += 1
        s, (ox, oy) = self.surface, self.rect.topleft
        s.fill((0, 0, 0, 0))
        local = pygame.Rect(0, 0, *self.rect.size)
        HUD.draw_frame(s, local)
        pad, px = HUD.CARD_PAD, HUD.px
        HUD.text(s, "Gravity", (pad, pad - px(1)), ui.WHITE, "bold", HUD.TITLE_SIZE)
        off = np.array([ox, oy], dtype=float)
        zero = not np.any(EnvConfig.gravity)
        for i in range(len(DIRECTIONS)):
            self._draw_rhombus(s, lambda inset, i=i: self._diamond(i, inset), off, hover == i, current == i)
        self._draw_rhombus(s, self._centre_diamond, off, hover == "zero", zero)
        HUD.text(s, direction_name(), (local.centerx, local.bottom - pad - px(LINE_H) // 2), ui.ACCENT, "bold",
                 align="center", valign="middle")
        if near:
            k = self.ui_scale
            self._top.fill((0, 0, 0, 0))
            overlay(self._top, (ox // k, oy // k))
            if self._top.get_bounding_rect().h:
                top = pygame.transform.scale(self._top, (self._top.get_width() * k, self._top.get_height() * k))
                s.blit(top, (0, 0))

    @staticmethod
    def _draw_rhombus(surface, shape, off, hovered, selected):
        """draw_button's look on a rhombus: black border, Endesga fill, a light inner outline on hover."""
        def poly(inset):
            return [tuple(np.round(p - off).astype(int)) for p in shape(inset)]
        border = HUD.px(1)
        pygame.draw.polygon(surface, ui.BLACK, poly(0))
        if hovered:
            pygame.draw.polygon(surface, ui.TEXT, poly(border))
            pygame.draw.polygon(surface, ui.button_fill(True, selected), poly(border + HUD.scale))
        else:
            pygame.draw.polygon(surface, ui.button_fill(False, selected), poly(border))
