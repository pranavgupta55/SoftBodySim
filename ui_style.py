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
BUTTON_DISABLED = (50, 50, 50)  # greyVD
WHITE = (255, 255, 255)

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


def text(surface, string, pos, color=TEXT, style="regular", size=TEXT_SIZE, align="left", valign="top"):
    """Non-antialiased text; align is "left", "center" or "right" relative to pos[0], valign "top" or "middle"
    relative to pos[1]. Returns the text rect."""
    img = font(style, size).render(string, False, color)
    r = img.get_rect()
    key = {"left": "topleft", "center": "midtop", "right": "topright"}[align]
    if valign == "middle":
        key = {"topleft": "midleft", "midtop": "center", "topright": "midright"}[key]
    setattr(r, key, pos)
    surface.blit(img, r)
    return r


def button_fill(hovered=False, selected=False, disabled=False):
    return BUTTON_DISABLED if disabled else BUTTON_SELECTED if selected else BUTTON_HOVER if hovered else BUTTON


def draw_button(surface, rect, label=None, hovered=False, selected=False, disabled=False, size=TEXT_SIZE):
    """A framed button: Endesga fill, black border, a 1px light inner outline on hover, centered bold label."""
    draw_frame(surface, rect, button_fill(hovered, selected, disabled))
    if hovered and not disabled:
        pygame.draw.rect(surface, TEXT, rect.inflate(-2 * BORDER, -2 * BORDER), 1)
    if label:
        text(surface, label, rect.center, TEXT_DIM if disabled else WHITE, "bold", size, "center", "middle")
