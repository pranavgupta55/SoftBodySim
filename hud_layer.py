"""The right-hand sidebar: a native-resolution layer holding every card, stacked top to bottom. Used by main_opengl.py.

The world and the editor's panels are drawn on surfaces scaled down by scaleDownFactor and upscaled to the screen;
the world fills the area left of the sidebar (scene.ViewConfig.sidebar_frac sets its width, see
ui_style.sidebar_rect). The sidebar is an opaque per-pixel-alpha surface covering the whole strip (`rect`, native
screen px, full screen height), drawn over the composite, so cards are sharp and never float over the world.

Cards are an ordered list of providers (`cards`), stacked from the top with ui_style.HUD metrics. A provider is any
object with:

    card_height(compact) -> int          its height in native px at the current ui_style.HUD scale; compact: its
                                         short form (providers without one ignore it), used only if all don't fit
    place(rect)                          where it goes this frame (native screen px); kept for hit-testing
    draw_card(surface, origin, mouse_px, compact)
                                         draw it at its placed rect minus origin (the layer's top-left)

and optionally `compactable = True` if card_height(True) is shorter, and `collapsed` (the user chose the short form,
e.g. F for the Compute card; ignored when fitting, so it never changes the scale). To add a card (e.g. a world selector), give it
those methods and add it to the list in main_opengl.py; route its clicks there too. Clicks anywhere else in the
sidebar are swallowed by the editor (it never passes them to the world).

`fit()` picks the HUD scale for the screen: the largest scale the sidebar's width allows (font sizes step 1 px at a
time) at which every card fits the height, trying full cards down to HUD_MIN_SCALE, then compact cards, then compact
cards down to HUD_FLOOR_SCALE. Card heights don't depend on the editor's state, so nothing moves between states; the
fit is redone only when the cards' natural heights change (e.g. a hotkey row added).

A frame:
    hud.draw(mouse_px)          # fit if needed, place the cards, draw the sidebar + cards
    hud.overlay(draw_top)       # message + cursor over the sidebar (see overlay)
then main_opengl.py uploads the layer to a texture and draws it as a quad on top of its composite.
"""
import pygame

import ui_style as ui


class HudLayer:
    def __init__(self, screen_size, ui_scale, frac, cards=()):
        """screen_size: native screen (window) px; ui_scale: native px per UI-surface px (scaleDownFactor); frac: the
        sidebar's share of the screen width; cards: the ordered card providers (can be set later)."""
        self.ui_scale = ui_scale
        self.rect = ui.sidebar_rect(screen_size, ui_scale, frac)
        self.surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self.used = self.rect.h  # rows (from the top) to present: all of them (the sidebar is opaque)
        self.cards = list(cards)
        self.compact = False  # cards drawn in their compact form (set by fit)
        self.placed = []  # (card, rect in native screen px) from the last layout
        self._fit_key = None
        s = ui_scale
        self._top = pygame.Surface((-(-self.rect.w // s), -(-self.rect.h // s)), pygame.SRCALPHA)  # UI-res overlay

    @property
    def ui_origin(self):
        """The layer's top-left in UI-surface px."""
        return self.rect.x // self.ui_scale, self.rect.y // self.ui_scale

    # ---------------- Fitting and layout
    def _stack_height(self, compact):
        h = ui.HUD
        heights = [c.card_height(compact) for c in self.cards]
        return 2 * h.CARD_MARGIN + sum(heights) + h.CARD_GAP * max(len(heights) - 1, 0)

    def _scales(self, lo):
        """Candidate HUD scales, largest first: the width's scale, then each 1 px smaller text size down to `lo`."""
        top = min(self.rect.w / (ui.CARD_WIDTH_BASE + 8), ui.HUD_MAX_SCALE)
        out = [top] + [n / 11 for n in range(int(11 * top), 0, -1) if lo - 1e-9 <= n / 11 < top] + ([lo] if lo < top else [])
        return out if top >= lo else [lo]

    def fit(self):
        """Choose the HUD scale and compact flag (see the module docstring). Cheap when nothing changed."""
        hud = ui.HUD
        saved = hud.scale
        # sized for the cards' full forms: a card the user collapsed (e.g. F) must not change the scale
        collapsed = [(c, c.collapsed) for c in self.cards if getattr(c, "collapsed", False)]
        for c, _ in collapsed:
            c.collapsed = False
        try:
            hud.set_scale(ui.HUD_MIN_SCALE)
            key = (self.rect.size, tuple(c.card_height(False) for c in self.cards),
                   tuple(c.card_height(True) for c in self.cards))
            if key == self._fit_key:
                hud.set_scale(saved)
                return
            self._fit_key = key
            compactable = any(getattr(c, "compactable", False) for c in self.cards)
            passes = [(False, ui.HUD_MIN_SCALE)] + ([(True, ui.HUD_MIN_SCALE)] if compactable else [])
            passes.append((compactable, ui.HUD_FLOOR_SCALE))
            for compact, lo in passes:
                for s in self._scales(lo):
                    hud.set_scale(s)
                    if self._stack_height(compact) <= self.rect.h:
                        self.compact = compact
                        return
            self.compact = compactable  # nothing fits (an absurd number of rows): smallest scale, runs off the bottom
        finally:
            for c, v in collapsed:
                c.collapsed = v

    def layout(self):
        """Place every card (native screen px), top to bottom; returns [(card, rect)]."""
        self.fit()
        h = ui.HUD
        x, w = self.rect.x + h.CARD_MARGIN, self.rect.w - 2 * h.CARD_MARGIN
        y, self.placed = self.rect.y + h.CARD_MARGIN, []
        for card in self.cards:
            r = pygame.Rect(x, y, w, card.card_height(self.compact))
            card.place(r)
            self.placed.append((card, r))
            y = r.bottom + h.CARD_GAP
        return self.placed

    # ---------------- Drawing
    def draw(self, mouse_px):
        """The sidebar background (a black rule on its left edge) and every card."""
        self.layout()
        s = self.surface
        s.fill(ui.SIDEBAR_BG)
        s.fill(ui.BLACK, (0, 0, ui.HUD.BORDER, self.rect.h))
        for card, _ in self.placed:
            card.draw_card(s, self.rect.topleft, mouse_px, self.compact)

    def overlay(self, draw):
        """Things drawn on the UI surface after the cards (the cursor; the message stays left of the sidebar) must
        stay on top of them. `draw(surface, origin)` draws them at UI-surface px minus `origin`; here it draws into a
        UI-res surface the size of the sidebar, which is upscaled (nearest, like the UI surface itself) onto the layer,
        so they look exactly as on the UI surface. Call it after draw()."""
        s = self.ui_scale
        self._top.fill((0, 0, 0, 0))
        draw(self._top, self.ui_origin)
        box = self._top.get_bounding_rect()
        if box.h:  # nothing over the sidebar (the mouse is over the world): skip the scale + blit
            part = self._top.subsurface(box)
            self.surface.blit(pygame.transform.scale(part, (box.w * s, box.h * s)), (box.x * s, box.y * s))
