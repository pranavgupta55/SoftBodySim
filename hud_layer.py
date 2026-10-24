"""Native-resolution layer for the HUD cards (Controls + Compute), used by main_opengl.py.

The world and the editor's other UI are drawn on surfaces scaled down by scaleDownFactor and upscaled to the screen;
the two info cards instead go on this layer at native resolution (ui_style.HUD metrics), so their text and chart are
sharp. The layer is a small per-pixel-alpha surface covering just the card column (`rect`, native screen px, full
screen height); only its top `used` rows hold cards in a given frame.

A frame:
    hud.begin()                                   # clear last frame's rows
    card = editor.draw_info_card(hud.surface, mouse, hud.rect.topleft); hud.add(card)
    hud.add(perf_hud.draw(hud.surface, card.bottom + ui_style.HUD.CARD_GAP))
    hud.overlay(draw_top)                         # message + cursor over the cards (see overlay)
then main_opengl.py uploads those rows to a texture and draws them as a quad on top of its composite.
"""
import pygame

import ui_style as ui


class HudLayer:
    def __init__(self, screen_size, ui_scale):
        """screen_size: native screen (window) px; ui_scale: native px per UI-surface px (scaleDownFactor)."""
        self.ui_scale = ui_scale
        self.rect = ui.hud_column(screen_size, ui_scale)
        self.surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self.used = 0  # rows (from the top) drawn this frame; 0 = nothing to show
        self._cleared = 0  # rows to clear in begin()
        s = ui_scale
        self._top = pygame.Surface((-(-self.rect.w // s), -(-self.rect.h // s)), pygame.SRCALPHA)  # UI-res overlay

    @property
    def ui_origin(self):
        """The layer's top-left in UI-surface px."""
        return self.rect.x // self.ui_scale, self.rect.y // self.ui_scale

    def begin(self):
        if self._cleared:
            self.surface.fill((0, 0, 0, 0), (0, 0, self.rect.w, self._cleared))
        self.used = self._cleared = 0

    def add(self, card_rect):
        """Note a card drawn on the layer (layer px), so its rows get presented."""
        self.used = self._cleared = min(max(self.used, card_rect.bottom), self.rect.h)

    def overlay(self, draw):
        """Things drawn on the UI surface after the cards (the editor message, the cursor) must stay on top of them.
        `draw(surface, origin)` draws them at UI-surface px minus `origin`; here it draws into a UI-res surface over
        the cards' rows, which is upscaled (nearest, like the UI surface itself) onto the layer, so they look exactly
        as on the UI surface. Call it after the cards."""
        if not self.used:
            return
        s = self.ui_scale
        h = -(-self.used // s)
        area = pygame.Rect(0, 0, self._top.get_width(), h)
        self._top.fill((0, 0, 0, 0), area)
        draw(self._top, self.ui_origin)
        top = self._top.subsurface(area)
        if top.get_bounding_rect().h:  # nothing over the cards: skip the scale + blit
            self.surface.blit(pygame.transform.scale(top, (area.w * s, h * s)), (0, 0))
            self._cleared = min(h * s, self.rect.h)  # may reach a little past `used`: clear those rows too

