"""The right-hand sidebar: a native-resolution layer holding every card, stacked top to bottom. Used by main_opengl.py.

The world and the editor's panels are drawn on surfaces scaled down by scaleDownFactor and upscaled to the screen;
the world fills the area left of the sidebar (scene.ViewConfig.sidebar_frac sets its width, see
ui_style.sidebar_rect). The sidebar is an opaque per-pixel-alpha surface covering the whole strip (`rect`, native
screen px, full screen height), drawn over the composite, so cards are sharp and never float over the world.

Cards are an ordered list of providers (`cards`), stacked from the top with ui_style.HUD metrics. A provider is any
object with:

    card_title                           its name (also the key its open / closed state is remembered under)
    card_height(compact) -> int          its height in native px at the current ui_style.HUD scale (compact is
                                         always False here; kept for the providers' signatures)
    place(rect)                          where it goes this frame (native screen px); kept for hit-testing
    draw_card(surface, origin, mouse_px, compact)
                                         draw it at its placed rect minus origin (the layer's top-left); it draws
                                         its own title row with ui_style.HUD.card_title (the open arrow + title)

and optionally header_status() -> str (shown right-aligned on its collapsed title row) and auto_open = True (opened
whenever it appears, e.g. the Material card). To add a card, give it those and add it to the list in main_opengl.py;
route its clicks there too.

Open / closed: every card has a header (its title row, with a disclosure arrow); a click on it opens or closes the
card (handle_event). A collapsed card is drawn here as just that row, and is placed far off-screen so none of its
widgets can take a click (its keyboard shortcuts still work). The state is kept per card title for the session
(DEFAULT_OPEN says which start open).

Scale and scrolling: `fit()` picks the HUD scale from the sidebar's width alone (the largest the width allows, at
least ui_style.HUD_MIN_SCALE: legible text), never shrinking it for height. When the open cards are taller than the
screen the sidebar scrolls: the mouse wheel over it (handle_event), or drag the thin scrollbar in its right margin.
The scroll has a slight momentum (the wheel adds velocity, which decays with time constant SCROLL_TAU) and rubber-band
ends (a little overscroll, at most OVERSCROLL_FRAC of the height, sprung back by a critically damped spring). The
motion is advanced in draw() from wall-clock time, so it is frame-rate independent; while it moves the cards' rects
change and the layer redraws, at rest it stops redrawing.

Clicks anywhere in the sidebar that aren't on an open card's body are swallowed here (the editor swallows the rest
of the sidebar's clicks too, so nothing reaches the world).

A frame:
    hud.draw(mouse_px, key)     # step the scroll, fit if needed, place the cards, draw (if anything changed)
    hud.overlay(draw_top)       # message + cursor over the sidebar (see overlay)
then main_opengl.py uploads the layer's changed rows (`dirty`) to a texture and draws it as a quad on top of its
composite.

Redrawing only on change: the cards cost several ms of pygame drawing and text per frame, but almost never change
from one frame to the next. draw() takes a `key`, a hashable summary from the caller of everything the cards show
(input events, the mouse when it is over the sidebar, the active world and editor state, plus a slow safety tick so
nothing the key misses stays stale for long). The sidebar is redrawn whole only when the key or the layout (which
includes the scroll offset and open cards) changes; otherwise only cards with a `refresh_hz` (live content: the
Compute card) are redrawn in place at that rate. `dirty` = the (top, bottom) rows changed this frame, or None, so the
presenter re-uploads just those. key=None redraws every frame (the tests drive cards directly, without a key).
"""
import math
import time

import numpy as np
import pygame

import ui_style as ui

DEFAULT_OPEN = {"Controls": True, "Worlds": True, "Material": True}  # cards not listed start collapsed
OFF_SCREEN = -1_000_000  # where collapsed cards are placed (no click can reach their widgets)

SCROLL_TAU = 0.2  # s: the scroll velocity's decay time constant (a flick coasts about velocity * TAU)
WHEEL_STEP = 60  # base px coasted per wheel notch (x HUD.scale)
MAX_SPEED = 5000  # base px / s
OVERSCROLL_FRAC = 0.08  # the rubber band's reach past either end, as a share of the sidebar's height
SPRING_W = 22.0  # 1/s: the critically damped spring pulling an overscroll back (settles in ~0.25 s)
REST_SPEED = 4.0  # px / s: slower than this (and within 0.5 px of an end when past it) is at rest
SUBSTEP = 1 / 240  # s: the motion is integrated in steps no longer than this
SCROLLBAR_W = 3  # base px
FOOTER_BTN_H = 20  # base px: the footer button (editor.CARD_BTN_H)


def card_title(card):
    return getattr(card, "card_title", None) or type(card).__name__


class HudLayer:
    def __init__(self, screen_size, ui_scale, frac, cards=()):
        """screen_size: native screen (window) px; ui_scale: native px per UI-surface px (scaleDownFactor); frac: the
        sidebar's share of the screen width; cards: the ordered card providers (can be set later)."""
        self.ui_scale = ui_scale
        self.rect = ui.sidebar_rect(screen_size, ui_scale, frac)
        self.surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self.used = self.rect.h  # rows (from the top) to present: all of them (the sidebar is opaque)
        self.cards = list(cards)
        self.compact = False  # (cards are never drawn compact: the sidebar scrolls instead)
        self.placed = []  # (card, rect in native screen px, open) from the last layout; collapsed: the header rect
        self.open = {}  # card title -> open, once toggled (else DEFAULT_OPEN)
        self.scroll = 0.0  # px the cards are scrolled up by (may be past either end while rubber-banding)
        self.velocity = 0.0  # px / s
        self.content_h = 0  # the stacked cards' height (with margins) at the last layout
        self._t = None  # the time of the last motion step (every draw stamps it, moving or not)
        self._drag = None  # scrollbar drag: the grab offset within the thumb (px)
        self._seen = set()  # titles of the cards in the last layout (auto_open cards open when they appear)
        self._fit_key = None
        self.dirty = None  # (top, bottom) layer rows changed by the last draw / overlay, or None
        self._key = self._layout_key = None  # what the layer was last drawn for (draw's key, the card layout)
        self._ticks = {}  # id(card) -> its refresh tick when last drawn (cards with refresh_hz)
        self._mouse = (0, 0)
        self._overlay_box = None
        # a button pinned under the cards, never scrolled or collapsed: (label, action(), enabled() -> bool) or None
        self.footer = None
        s = ui_scale
        self._top = pygame.Surface((-(-self.rect.w // s), -(-self.rect.h // s)), pygame.SRCALPHA)  # UI-res overlay

    @property
    def ui_origin(self):
        """The layer's top-left in UI-surface px."""
        return self.rect.x // self.ui_scale, self.rect.y // self.ui_scale

    # ---------------- Open / closed
    def is_open(self, card):
        t = card_title(card)
        return self.open.get(t, DEFAULT_OPEN.get(t, False))

    def set_open(self, card, value=True):
        self.open[card_title(card)] = bool(value)

    def toggle(self, card):
        """Open or close a card (its header was clicked). Opening one whose body runs off the bottom scrolls it into
        view (as much as fits with its header still on screen), by giving the scroll the velocity that coasts there."""
        opening = not self.is_open(card)
        self.set_open(card, opening)
        if not opening:
            return
        self.layout()
        r = next((r for c, r, _ in self.placed if c is card), None)
        if r is None:
            return
        m = ui.HUD.CARD_MARGIN
        need = min(r.bottom + m - (self.rect.y + self.view_h()), r.y - m - self.rect.y)
        if need > 0:
            self.velocity = (need + REST_SPEED * SCROLL_TAU) / SCROLL_TAU

    def state_key(self):
        """What the sidebar's look depends on here (main_opengl's sidebar_key): the open cards and the scroll."""
        return tuple(sorted(self.open.items())), round(self.scroll)

    # ---------------- Fitting and layout
    def fit(self):
        """The HUD scale: the largest the sidebar's width allows (at least HUD_MIN_SCALE). Cheap when nothing changed."""
        if self._fit_key == self.rect.size:
            return
        self._fit_key = self.rect.size
        scale = self.rect.w / (ui.CARD_WIDTH_BASE + 8)
        ui.HUD.set_scale(min(max(scale, ui.HUD_MIN_SCALE), ui.HUD_MAX_SCALE))

    def footer_h(self):
        """Height the footer takes from the bottom of the sidebar (0 without one)."""
        if self.footer is None:
            return 0
        return ui.HUD.px(FOOTER_BTN_H) + 2 * ui.HUD.CARD_MARGIN

    def view_h(self):
        """Height of the scrolling card area (the sidebar above the footer)."""
        return self.rect.h - self.footer_h()

    def footer_rect(self):
        """The footer button (native screen px), or None."""
        if self.footer is None:
            return None
        m = ui.HUD.CARD_MARGIN
        return pygame.Rect(self.rect.x + m, self.rect.bottom - m - ui.HUD.px(FOOTER_BTN_H), self.rect.w - 2 * m,
                           ui.HUD.px(FOOTER_BTN_H))

    def max_scroll(self):
        return max(self.content_h - self.view_h(), 0)

    def layout(self):
        """Place every card (native screen px), top to bottom, scrolled; returns [(card, rect, open)]."""
        self.fit()
        h = ui.HUD
        titles = {card_title(c) for c in self.cards}
        for card in self.cards:  # cards that just appeared and open by themselves
            if getattr(card, "auto_open", False) and card_title(card) not in self._seen:
                self.set_open(card)
        self._seen = titles
        x, w = self.rect.x + h.CARD_MARGIN, self.rect.w - 2 * h.CARD_MARGIN
        heights = [(c, self.is_open(c)) for c in self.cards]
        heights = [(c, o, c.card_height(False) if o else h.header_height()) for c, o in heights]
        self.content_h = 2 * h.CARD_MARGIN + sum(v for *_, v in heights) + h.CARD_GAP * max(len(heights) - 1, 0)
        y, self.placed = self.rect.y + h.CARD_MARGIN - round(self.scroll), []
        for card, is_open, ch in heights:
            r = pygame.Rect(x, y, w, ch)
            card.place(r if is_open else pygame.Rect(OFF_SCREEN, OFF_SCREEN, w, card.card_height(False)))
            self.placed.append((card, r, is_open))
            y = r.bottom + h.CARD_GAP
        return self.placed

    def header_rect(self, r, is_open):
        """The clickable header of a card placed at r."""
        return pygame.Rect(r.x, r.y, r.w, min(ui.HUD.header_hit_height(), r.h) if is_open else r.h)

    # ---------------- Scrolling
    @property
    def moving(self):
        return self.velocity != 0.0 or self.scroll != min(max(self.scroll, 0.0), self.max_scroll())

    def _cap(self):
        return OVERSCROLL_FRAC * self.view_h()

    def wheel(self, notches):
        """Wheel / trackpad input (notches up > 0): adds velocity; past an end, against a growing resistance."""
        step = WHEEL_STEP * ui.HUD.scale
        dv = -notches * step / SCROLL_TAU
        edge = min(max(self.scroll, 0.0), self.max_scroll())
        over = self.scroll - edge
        if over and (over > 0) == (dv > 0):  # pushing further out: the rubber band resists
            dv *= max(0.0, 1.0 - abs(over) / self._cap()) ** 2
        top =MAX_SPEED * ui.HUD.scale
        self.velocity = min(max(self.velocity + dv, -top), top)

    def step(self, now):
        """Advance the scroll's motion to time `now` (seconds)."""
        if self._t is None or not self.moving or self._drag is not None:
            self._t = now
            return
        dt = min(max(now - self._t, 0.0), 0.1)  # (a long stall doesn't fling it)
        self._t = now
        hi, cap = self.max_scroll(), self._cap()
        n = max(1, math.ceil(dt / SUBSTEP))
        h = dt / n
        decay = math.exp(-h / SCROLL_TAU)
        for _ in range(n):
            edge = min(max(self.scroll, 0.0), hi)
            x0, v0 = self.scroll - edge, self.velocity
            if x0 == 0.0:  # inside: exponential decay (exact)
                self.scroll += v0 * SCROLL_TAU * (1.0 - decay)
                self.velocity = v0 * decay
            else:  # past an end: x'' = -w^2 x - 2w x' (critically damped, exact)
                w = SPRING_W
                e = math.exp(-w * h)
                c = v0 + w * x0
                x, v = (x0 + c * h) * e, (v0 - w * c * h) * e
                if abs(x) > max(cap, abs(x0)):  # the band's reach
                    x, v = math.copysign(max(cap, abs(x0)), x), 0.0
                if (x > 0) != (x0 > 0) and x != 0.0:  # sprang back past the end: continue inside from the end
                    x = 0.0
                self.scroll, self.velocity = edge + x, v
        edge = min(max(self.scroll, 0.0), hi)
        if abs(self.velocity) < REST_SPEED:
            if self.scroll == edge:
                self.velocity = 0.0
            elif abs(self.scroll - edge) < 0.5:
                self.scroll, self.velocity = float(edge), 0.0

    def _thumb(self):
        """The scrollbar's thumb rect (native screen px), or None when everything fits."""
        hi = self.max_scroll()
        if hi <= 0:
            return None
        h, px = self.view_h(), ui.HUD.px
        m = ui.HUD.CARD_MARGIN
        track = h - 2 * m
        th = max(round(track * h / self.content_h), px(20))
        edge = min(max(self.scroll, 0.0), hi)
        over = abs(self.scroll - edge)
        th = max(round(th - over), px(8))  # squeezed while rubber-banding
        y = m + round((track - th) * edge / hi)
        bw = px(SCROLLBAR_W)
        return pygame.Rect(self.rect.right - m + (m - bw) // 2, self.rect.y + y, bw, th)

    def _bar_zone(self):
        """Where a click grabs the scrollbar: its margin plus a little of the cards' right border."""
        m = ui.HUD.CARD_MARGIN
        z = m + ui.HUD.px(3)
        return pygame.Rect(self.rect.right - z, self.rect.y, z, self.view_h())

    def _drag_to(self, y):
        thumb, hi, m = self._thumb(), self.max_scroll(), ui.HUD.CARD_MARGIN
        if thumb is None:
            return
        room = self.view_h() - 2 * m - thumb.h
        frac = (y - self._drag - self.rect.y - m) / room if room > 0 else 0.0
        self.scroll, self.velocity = min(max(frac, 0.0), 1.0) * hi, 0.0

    # ---------------- Events
    def handle_event(self, event, mouse_px):
        """The wheel over the sidebar (scroll), clicks on card headers (open / close), the scrollbar, and any other
        click in the sidebar that isn't on an open card's body (swallowed). mouse_px: the mouse now (wheel events
        carry no position). True if used."""
        if event.type == pygame.MOUSEWHEEL:
            if not self.rect.collidepoint(mouse_px):
                return False
            y = getattr(event, "precise_y", None)
            self.wheel(event.y if y is None else y)
            return True
        if self._drag is not None:
            if event.type == pygame.MOUSEMOTION:
                self._drag_to(event.pos[1])
                return True
            if event.type == pygame.MOUSEBUTTONUP:
                self._drag = None
                return True
        if event.type != pygame.MOUSEBUTTONDOWN or not self.rect.collidepoint(event.pos):
            return False
        if event.button in (4, 5):  # (pygame's old-style wheel buttons, alongside MOUSEWHEEL)
            return True
        if event.pos[1] >= self.rect.y + self.view_h():  # the footer strip: its button, nothing under it
            foot = self.footer_rect()
            if event.button == 1 and foot is not None and foot.collidepoint(event.pos) and self.footer[2]():
                self.footer[1]()
            return True
        thumb = self._thumb()
        if thumb is not None and event.button == 1 and self._bar_zone().collidepoint(event.pos):
            self._drag = event.pos[1] - thumb.y if thumb.top <= event.pos[1] < thumb.bottom else thumb.h // 2
            self._drag_to(event.pos[1])
            return True
        for card, r, is_open in self.placed:
            if self.header_rect(r, is_open).collidepoint(event.pos):
                if event.button == 1:
                    self.toggle(card)
                return True
            if is_open and r.collidepoint(event.pos):
                return False  # the card's own
        return True

    # ---------------- Drawing
    def draw(self, mouse_px, key=None, now=None):
        """The sidebar background (a black rule on its left edge) and every card, when anything changed (see the
        module docstring): key: hashable state the cards show, or None to redraw regardless; now: seconds, for the
        scroll's motion and refresh_hz cards (default time.perf_counter())."""
        now = time.perf_counter() if now is None else now
        self.step(now)
        self.layout()
        self._mouse = mouse_px
        layout_key = (tuple((id(c), tuple(r), o) for c, r, o in self.placed), ui.HUD.scale, self.content_h)
        if key is None or key != self._key or layout_key != self._layout_key:
            self._key, self._layout_key = key, layout_key
            self._redraw(now)
            return
        self.dirty = None
        for card, r, is_open in self.placed:  # live cards only, in place
            hz = getattr(card, "refresh_hz", None)
            tick = int(now * hz) if hz else None
            if tick is not None and tick != self._ticks.get(id(card)):
                self._ticks[id(card)] = tick
                local = r.move(-self.rect.x, -self.rect.y)
                if local.bottom <= 0 or local.top >= self.rect.h:
                    continue
                self.surface.fill(ui.SIDEBAR_BG, local)
                self._draw_card(card, r, is_open, mouse_px)
                self._mark(local.top, local.bottom)
                if local.bottom > self.view_h():
                    self._draw_footer()

    def _draw_card(self, card, r, is_open, mouse_px):
        if is_open:
            card.draw_card(self.surface, self.rect.topleft, mouse_px, False)
            return
        hud, s = ui.HUD, self.surface
        local = r.move(-self.rect.x, -self.rect.y)
        hover = r.collidepoint(mouse_px)
        hud.draw_frame(s, local, ui.BUTTON if hover else ui.CARD_BG)
        title = hud.card_title(s, local, card_title(card), False)
        status = card.header_status() if hasattr(card, "header_status") else ""
        if status:
            font = hud.font("bold")
            right = local.right - hud.CARD_PAD
            while len(status) > 1 and right - font.size(status)[0] < title.right + hud.px(6):
                status = status[:-1]
            if right - font.size(status)[0] >= title.right + hud.px(6):
                hud.text(s, status, (right, local.y + hud.CARD_PAD), ui.ACCENT, "bold", align="right")

    def _redraw(self, now):
        s = self.surface
        s.fill(ui.SIDEBAR_BG)
        s.fill(ui.BLACK, (0, 0, ui.HUD.BORDER, self.rect.h))
        for card, r, is_open in self.placed:
            if r.bottom > self.rect.y and r.top < self.rect.y + self.view_h():  # (scrolled out of view: not drawn)
                self._draw_card(card, r, is_open, self._mouse)
            hz = getattr(card, "refresh_hz", None)
            if hz:
                self._ticks[id(card)] = int(now * hz)
        self._draw_footer()
        thumb = self._thumb()
        if thumb is not None:
            s.fill(ui.BUTTON_SELECTED if self._drag is not None or self._bar_zone().collidepoint(self._mouse)
                   else ui.BUTTON, thumb.move(-self.rect.x, -self.rect.y))
        self.dirty = (0, self.rect.h)

    def _draw_footer(self):
        """The footer strip over whatever scrolled under it: the sidebar background and the pinned button."""
        foot = self.footer_rect()
        if foot is None:
            return
        top = self.view_h()
        self.surface.fill(ui.SIDEBAR_BG, (ui.HUD.BORDER, top, self.rect.w - ui.HUD.BORDER, self.rect.h - top))
        label, _, enabled = self.footer
        on = bool(enabled())
        ui.HUD.draw_button(self.surface, foot.move(-self.rect.x, -self.rect.y), label,
                           on and foot.collidepoint(self._mouse), disabled=not on)
        self._mark(top, self.rect.h)

    def _mark(self, top, bottom):
        top, bottom = max(top, 0), min(bottom, self.rect.h)
        if bottom > top:
            self.dirty = (top, bottom) if self.dirty is None else (min(self.dirty[0], top), max(self.dirty[1], bottom))

    def overlay(self, draw):
        """Things drawn on the UI surface after the cards (the cursor; the message stays left of the sidebar) must
        stay on top of them. `draw(surface, origin)` draws them at UI-surface px minus `origin`; here it draws into a
        UI-res surface the size of the sidebar, which is upscaled (nearest, like the UI surface itself) onto the layer,
        so they look exactly as on the UI surface. Call it after draw()."""
        s = self.ui_scale
        self._top.fill((0, 0, 0, 0))
        draw(self._top, self.ui_origin)
        box = bounding_rect(self._top)
        key = tuple(box) if box.h else None
        if key != self._overlay_box:  # it moved or went: the cards under the old one must be redrawn
            if self._overlay_box is not None:
                self._redraw(time.perf_counter())
            self._overlay_box = key
            if key is not None:
                self._mark(box.y * s, box.bottom * s)
        if box.h:  # nothing over the sidebar (the mouse is over the world): skip the scale + blit
            part = self._top.subsurface(box)
            self.surface.blit(pygame.transform.scale(part, (box.w * s, box.h * s)), (box.x * s, box.y * s))


def bounding_rect(surface):
    """Surface.get_bounding_rect for a cleared (all-zero) 32-bit surface: the box of its non-zero pixels, by a numpy
    scan of the raw buffer (several times faster)."""
    w, h = surface.get_size()
    if surface.get_pitch() != 4 * w:
        return surface.get_bounding_rect()
    buf = surface.get_buffer()
    used = np.frombuffer(buf, np.uint32).reshape(h, w) != 0
    del buf  # (the array keeps the buffer until it is freed below)
    rows, cols = np.flatnonzero(used.any(axis=1)), np.flatnonzero(used.any(axis=0))
    del used
    if not rows.size:
        return pygame.Rect(0, 0, 0, 0)
    return pygame.Rect(int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))
