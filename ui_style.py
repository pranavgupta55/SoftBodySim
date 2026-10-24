"""Shared look for UI drawn on the scaled-down UI surface (HUD cards, editor panels, buttons).

Everything is in UI-surface pixels (the same resolution as the world view), so UI matches the game's pixel style:
Endesga colors, solid black 2px borders like the filled shape view, and non-antialiased Montserrat text.
"""
import pygame

from fontDict import fonts

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

# layout: cards stack down the top-right corner
CARD_MARGIN = 4  # px from the surface edges
CARD_GAP = 4  # px between stacked cards
CARD_WIDTH = 150
CARD_PAD = 5
BORDER = 2  # px, same as the filled shape view's body borders

TEXT_SIZE = 11
TITLE_SIZE = 12


def font(style="regular", size=TEXT_SIZE):
    return fonts[f"{style}{size}"]


def card_rect(surface_width, top, height):
    """A card in the top-right column, starting at `top` (use the previous card's rect.bottom + CARD_GAP)."""
    return pygame.Rect(surface_width - CARD_MARGIN - CARD_WIDTH, top, CARD_WIDTH, height)


def draw_frame(surface, rect, fill=CARD_BG):
    """Solid fill with a solid black border drawn just inside the rect."""
    pygame.draw.rect(surface, fill, rect)
    pygame.draw.rect(surface, BLACK, rect, BORDER)


def text(surface, string, pos, color=TEXT, style="regular", size=TEXT_SIZE, align="left"):
    """Non-antialiased text; align is "left", "center" or "right" relative to pos[0]. Returns the text rect."""
    img = font(style, size).render(string, False, color)
    r = img.get_rect()
    setattr(r, {"left": "topleft", "center": "midtop", "right": "topright"}[align], pos)
    surface.blit(img, r)
    return r
