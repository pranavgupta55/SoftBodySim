"""Shared look for the UI: HUD cards, editor panels, buttons.

One pixel-art style (Endesga colors, solid black borders like the filled shape view, non-antialiased Montserrat text)
at two resolutions:

- `UI` (scale 1): the scaled-down UI surface (screenUI, real screen / scaleDownFactor), used by the editor's panels,
  hints and messages. The module-level names (CARD_PAD, BORDER, text(), draw_button(), ...) are UI's.
- `HUD` (scale HUD_SCALE): the two info cards (Controls, Compute), drawn at native screen resolution on their own
  layer (hud_layer.py). Every metric is UI's times HUD_SCALE, so with HUD_SCALE == scaleDownFactor (2) the cards keep
  their old on-screen footprint but get sharper text and a finer chart.
"""
import pygame

from fontDict import fonts

# HUD card px per UI-style px. The cards are drawn at native resolution (1 native px per pixel), so 2 = the same
# on-screen size as UI drawn on the 2x scaled-down UI surface. Fonts, borders and all card metrics scale with it.
HUD_SCALE = 2

# palette (Endesga)
BLACK = (0, 0, 0)
CARD_BG = (19, 19, 19)
PANEL_BG = (32, 36, 46)
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
    """Card metrics, fonts and drawing helpers at one scale (px per base px). Cards stack down the top-right corner."""

    def __init__(self, scale):
        self.scale = s = scale
        self.CARD_MARGIN = 4 * s  # px from the surface edges
        self.CARD_GAP = 4 * s  # px between stacked cards
        self.CARD_WIDTH = 150 * s
        self.CARD_PAD = 5 * s
        self.BORDER = 2 * s  # same as the filled shape view's body borders (2 UI px)
        self.TEXT_SIZE = 11 * s
        self.TITLE_SIZE = 12 * s

    def px(self, v):
        """A base-px length at this scale."""
        return v * self.scale

    def font(self, style="regular", size=None):
        return fonts[f"{style}{size or self.TEXT_SIZE}"]

    def column_width(self):
        """Width of the card column including its margins."""
        return self.CARD_WIDTH + 2 * self.CARD_MARGIN

    def card_rect(self, surface_width, top, height):
        """A card in the top-right column, starting at `top` (use the previous card's rect.bottom + CARD_GAP)."""
        return pygame.Rect(surface_width - self.CARD_MARGIN - self.CARD_WIDTH, top, self.CARD_WIDTH, height)

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
            pygame.draw.rect(surface, TEXT, rect.inflate(-2 * self.BORDER, -2 * self.BORDER), self.scale)
        if label:
            self.text(surface, label, rect.center, TEXT_DIM if disabled else WHITE, "bold", size, "center", "middle")


UI = Style(1)  # the scaled-down UI surface
HUD = Style(HUD_SCALE)  # the native-resolution info cards


def hud_column(screen_size, ui_scale):
    """The native-px rect of the HUD card column (full screen height). Its left edge is a multiple of ui_scale, so
    UI-surface pixels upscaled into it line up exactly with the upscaled UI surface."""
    w, h = screen_size
    x = (w - HUD.column_width()) // ui_scale * ui_scale
    return pygame.Rect(x, 0, w - x, h)


# UI-surface names (the editor's panels and hints)
CARD_MARGIN, CARD_GAP, CARD_WIDTH, CARD_PAD, BORDER = UI.CARD_MARGIN, UI.CARD_GAP, UI.CARD_WIDTH, UI.CARD_PAD, UI.BORDER
TEXT_SIZE, TITLE_SIZE = UI.TEXT_SIZE, UI.TITLE_SIZE
font, card_rect, draw_frame, text, draw_button = UI.font, UI.card_rect, UI.draw_frame, UI.text, UI.draw_button
