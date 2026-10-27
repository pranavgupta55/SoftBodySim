"""Color palettes a preset world can switch between (the "diamonds" preset), and the sidebar card that picks one.

A palette gives the world's background, its walls, and a list of colors per role. A body takes part when it has a
`palette_role` ("wall", "diamond" or "hexagon") and a `palette_pick` (any int: which of the role's colors it gets,
modulo their count), so switching palettes recolors the world in place without touching the physics. The world
remembers its palette in `World.palette` (None: it has no palettes, and the card is hidden).

The card is a sidebar card provider (card_height / place / draw_card, see hud_layer.py), shown by main_opengl.py only
while the active world has a palette: a button per palette, its name and a strip of its colors, and a button that
hides the panel (H; main_opengl.py floats the sidebar over its fullscreen worlds).
"""
import pygame

import ui_style as ui
from ui_style import HUD
from editor import BTN_GAP, CARD_BTN_H, CARD_TITLE_H


def _hex(*codes):
    return [[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in codes]


# name -> {"background": rgb, "wall": [rgb], "diamond": [rgb, ...], "hexagon": [rgb, ...]}, in the card's order
PALETTES = {
    "dusk": {"background": _hex("26253e")[0], "wall": _hex("424f6a"),
             "diamond": _hex("9892a3", "424f6a", "3c2134"), "hexagon": _hex("6b1b2b", "772d38", "832f44")},
    "clouds": {"background": _hex("5c474e")[0], "wall": _hex("6c5355"),
               "diamond": _hex("826561", "94746a", "a48072"), "hexagon": _hex("b9917c", "ca9f85")},
    "forest": {"background": _hex("041e22")[0], "wall": _hex("163f36"),
               "diamond": _hex("225a52", "163f36"), "hexagon": _hex("86ad95")},
    "sunset": {"background": _hex("0a1624")[0], "wall": _hex("1f3c4e"),
               "diamond": _hex("1f3c4e", "6e3746", "b84f48"), "hexagon": _hex("f5ce8c", "e88a55")},
}
DEFAULT = "forest"


def apply_palette(world, name):
    """Recolor every body with a palette_role, and the background, from PALETTES[name]; remember it on the world."""
    pal = PALETTES[name]
    for body in world.bodies:
        role = getattr(body, "palette_role", None)
        if role in pal:
            cols = pal[role]
            col = list(cols[int(getattr(body, "palette_pick", 0)) % len(cols)])
            body.point_col, body.stick_col, body.fill_col = col, col, col
    world.background = list(pal["background"])
    world.palette = name


def palette_colors(name):
    """Every distinct color of a palette, background first (for the card's swatch strip)."""
    pal, seen = PALETTES[name], []
    for col in [pal["background"]] + pal["wall"] + pal["diamond"] + pal["hexagon"]:
        if col not in seen:
            seen.append(col)
    return seen


class PaletteCard:
    """The sidebar's Palette card for the active world (`get_world()`): a button per palette, then a hide-panel
    button that calls `on_hide()`."""

    def __init__(self, get_world, on_hide=None):
        self.get_world, self.on_hide = get_world, on_hide
        self.rect = pygame.Rect(0, 0, 0, 0)

    @staticmethod
    def card_height(compact=False):
        px, pad = HUD.px, HUD.CARD_PAD
        n = len(PALETTES) + 1  # + the hide-panel button
        return pad + px(CARD_TITLE_H) + n * px(CARD_BTN_H) + (n - 1) * px(BTN_GAP) + pad - px(1)

    def place(self, rect):
        self.rect = pygame.Rect(rect)

    def _buttons(self):
        """[(palette name, rect)], then (None, the hide-panel button's rect), in native screen px."""
        px, pad = HUD.px, HUD.CARD_PAD
        y, bh, gap = self.rect.y + pad + px(CARD_TITLE_H), px(CARD_BTN_H), px(BTN_GAP)
        return [(name, pygame.Rect(self.rect.x + pad, y + k * (bh + gap), self.rect.w - 2 * pad, bh))
                for k, name in enumerate(list(PALETTES) + [None])]

    def handle_event(self, event):
        """Left clicks on the card: pick that palette. True if the click was on the card."""
        if event.type != pygame.MOUSEBUTTONDOWN or not self.rect.collidepoint(event.pos):
            return False
        if event.button == 1:
            for name, rect in self._buttons():
                if rect.collidepoint(event.pos):
                    if name is None:
                        if self.on_hide is not None:
                            self.on_hide()
                    else:
                        apply_palette(self.get_world(), name)
        return True

    def draw_card(self, surface, origin, mouse_px, compact=False):
        ox, oy = origin
        r = self.rect.move(-ox, -oy)
        HUD.draw_frame(surface, r)
        px, pad = HUD.px, HUD.CARD_PAD
        HUD.text(surface, "Palette", (r.x + pad, r.y + pad - px(1)), ui.WHITE, "bold", HUD.TITLE_SIZE)
        current = getattr(self.get_world(), "palette", None)
        for name, rect in self._buttons():
            local, on = rect.move(-ox, -oy), name == current
            if name is None:
                HUD.draw_button(surface, local, "hide panel [h]", rect.collidepoint(mouse_px))
                continue
            HUD.draw_button(surface, local, None, rect.collidepoint(mouse_px), on)
            HUD.text(surface, name, (local.x + px(5), local.centery), ui.WHITE if on else ui.TEXT_DIM, "bold",
                     valign="middle")
            cols = palette_colors(name)
            sw, inset = px(6), px(4)  # the swatch strip, right-aligned inside the button
            x = local.right - inset - len(cols) * sw
            for k, col in enumerate(cols):
                surface.fill(col, (x + k * sw, local.y + inset, sw, local.h - 2 * inset))
            pygame.draw.rect(surface, ui.BLACK, (x - 1, local.y + inset - 1, len(cols) * sw + 2, local.h - 2 * inset + 2), 1)
