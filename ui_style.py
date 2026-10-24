"""Shared look for the UI: sidebar cards, editor panels, buttons.

One pixel-art style (Endesga colors, solid black borders like the filled shape view, non-antialiased Montserrat text)
at two resolutions:

- `UI` (scale 1): the scaled-down UI surface (screenUI, real screen / scaleDownFactor), used by the editor's panels,
  hints and messages. The module-level names (CARD_PAD, BORDER, text(), draw_button(), ...) are UI's.
- `HUD`: the cards in the right-hand sidebar (Controls, Compute, Gravity, ...), drawn at native screen resolution on
  the sidebar layer (hud_layer.py). Every metric is a base (UI-style) length times HUD.scale, rounded to whole px.
  The sidebar picks that scale for the screen (HudLayer.fit): as large as its width allows (it's SIDEBAR_FRAC of
  the screen, but at least wide enough for HUD_MIN_SCALE) while all its cards still fit its height.
"""
import pygame

from fontDict import fonts

# Sidebar card scale bounds (native px per base px). Cards are drawn 1 native px per pixel; 1 is the smallest scale
# that's comfortably legible (11 px text), so the sidebar is at least wide enough for that. Below it (only if the cards
# can't fit the screen's height otherwise) the scale may drop to HUD_FLOOR_SCALE.
HUD_MIN_SCALE = 1.0
HUD_FLOOR_SCALE = 0.75
HUD_MAX_SCALE = 2.0
CARD_WIDTH_BASE = 150  # a sidebar card's width at scale 1 (base px), without the margins

# palette (Endesga)
BLACK = (0, 0, 0)
CARD_BG = (19, 19, 19)
PANEL_BG = (32, 36, 46)
SIDEBAR_BG = (24, 20, 37)  # Endesga's darkest (the sidebar behind the cards)
TEXT = (199, 207, 221)
TEXT_DIM = (150, 150, 150)
ACCENT = (255, 96, 141)  # debug_red
BUTTON = (66, 76, 110)  # grey_blue
BUTTON_HOVER = (70, 74, 124)  # sebastian_lague_purple
BUTTON_SELECTED = (137, 133, 181)  # sebastian_lague_light_purple
BUTTON_DISABLED = (50, 50, 50)  # greyVD
WHITE = (255, 255, 255)


def button_fill(hovered=False, selected=False, disabled=False):
    return BUTTON_DISABLED if disabled else BUTTON_SELECTED if selected else BUTTON_HOVER if hovered else BUTTON


class Style:
    """Card metrics, fonts and drawing helpers at one scale (px per base px; lengths rounded to whole px)."""

    def __init__(self, scale):
        self.set_scale(scale)

    def set_scale(self, scale):
        self.scale = scale
        px = self.px
        self.CARD_MARGIN = px(4)  # px from the surface edges
        self.CARD_GAP = px(4)  # px between stacked cards
        self.CARD_WIDTH = px(CARD_WIDTH_BASE)
        self.CARD_PAD = px(5)
        self.BORDER = px(2)  # same as the filled shape view's body borders (2 UI px)
        self.TEXT_SIZE = px(11)
        self.TITLE_SIZE = px(12)

    def px(self, v):
        """A base-px length at this scale, in whole px (at least 1 for a positive length)."""
        return max(round(v * self.scale), 1) if v > 0 else round(v * self.scale)

    def font(self, style="regular", size=None):
        return fonts[f"{style}{size or self.TEXT_SIZE}"]

    def draw_frame(self, surface, rect, fill=CARD_BG):
        """Solid fill with a solid black border drawn just inside the rect."""
        pygame.draw.rect(surface, fill, rect)
        pygame.draw.rect(surface, BLACK, rect, self.BORDER)

    def text(self, surface, string, pos, color=TEXT, style="regular", size=None, align="left", valign="top"):
        """Non-antialiased text; align is "left", "center" or "right" relative to pos[0], valign "top" or "middle"
        relative to pos[1]. Returns the text rect."""
        img = self.font(style, size).render(string, False, color)
        r = img.get_rect()
        key = {"left": "topleft", "center": "midtop", "right": "topright"}[align]
        if valign == "middle":
            key = {"topleft": "midleft", "midtop": "center", "topright": "midright"}[key]
        setattr(r, key, pos)
        surface.blit(img, r)
        return r

    def draw_button(self, surface, rect, label=None, hovered=False, selected=False, disabled=False, size=None):
        """A framed button: Endesga fill, black border, a light inner outline on hover, centered bold label."""
        self.draw_frame(surface, rect, button_fill(hovered, selected, disabled))
        if hovered and not disabled:
            pygame.draw.rect(surface, TEXT, rect.inflate(-2 * self.BORDER, -2 * self.BORDER), self.px(1))
        if label:
            self.text(surface, label, rect.center, TEXT_DIM if disabled else WHITE, "bold", size, "center", "middle")


UI = Style(1)  # the scaled-down UI surface
HUD = Style(HUD_MIN_SCALE)  # the sidebar's native-resolution cards; HudLayer.fit sets its scale for the screen


def sidebar_rect(screen_size, ui_scale, frac):
    """The native-px rect of the right-hand sidebar (full screen height): `frac` of the screen width, but at least
    wide enough for a card at HUD_MIN_SCALE. Its left edge is a multiple of ui_scale, so the world area (left of it)
    is a whole number of UI-surface px and UI pixels upscaled onto the sidebar line up with the UI surface."""
    w, h = screen_size
    min_w = round((CARD_WIDTH_BASE + 2 * 4) * HUD_MIN_SCALE)
    x = min(w - round(frac * w), w - min_w) // ui_scale * ui_scale
    return pygame.Rect(x, 0, w - x, h)


# UI-surface names (the editor's panels and hints)
CARD_MARGIN, CARD_GAP, CARD_WIDTH, CARD_PAD, BORDER = UI.CARD_MARGIN, UI.CARD_GAP, UI.CARD_WIDTH, UI.CARD_PAD, UI.BORDER
TEXT_SIZE, TITLE_SIZE = UI.TEXT_SIZE, UI.TITLE_SIZE
font, draw_frame, text, draw_button = UI.font, UI.draw_frame, UI.text, UI.draw_button
