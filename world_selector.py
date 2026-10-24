"""A session of several live worlds and the world selector panel (W, or the World button on the Controls card).

The session holds "Custom" (the player's own world: the default demo scene at launch, keeps every edit for the rest of
the session, never saved to disk, so it starts fresh on every launch) plus one world per scene.PRESETS entry (except
"default", which is what Custom starts as and resets to). Preset worlds are built the first time they're opened.

Only the active world is stepped; the others stay frozen exactly as they were left. Switching worlds pauses the one
being left and opens the other paused. Each world has its own Editor (editor.Editor), so its pause state, containment
cache and undo history (placement and any world-level history kept on the editor) stay with that world; a placement
or spawn menu that's in progress is cancelled when its world is left. UI preferences (HUD shown, compute card, spawn
settings, the peg overlay) are shared by every world's editor.

Global environment settings that a world may change (GLOBAL_ENV, e.g. EnvConfig.gravity, set by the gravity direction
selector) are kept per world: saved when a world is left and restored when it's opened. A newly built or reset world
starts from the launch defaults (then whatever its builder sets).

Looping (World.looping / World.set_looping, looping.py: a side without its wall wraps bodies to the opposite side) is
toggled per world from the panel. Re-adding a wall is deferred by the world while a body is beyond or straddles that
edge (World.pending_walls); the panel shows such a side in pink until the wall is back.

The panel is drawn on the scaled-down UI surface in the editor's pixel style (ui_style), over a dimmed scene; the sim
keeps running behind it. Its rects are UI-surface px.
"""
import time

import numpy as np
import pygame

import ui_style as ui
from entities import EnvConfig

SIDES = ("left", "right", "top", "bottom")
SIDE_LABELS = {"left": "L", "right": "R", "top": "T", "bottom": "B"}
GLOBAL_ENV = ("gravity",)  # EnvConfig attributes saved / restored per world
CUSTOM = "Custom"
CONFIRM_SECONDS = 3.0  # the Custom world's Reset asks for a second click within this long

# layout, UI px
PAD = 8
HEADER_H = 30
ROW_H = 18
ROW_GAP = 3
NAME_W = 132
TOGGLE_W = 18
RESET_W = 46
COL_GAP = 8
LINE_H = 13

_ENV_DEFAULTS = {name: np.array(getattr(EnvConfig, name), copy=True) for name in GLOBAL_ENV}


def env_snapshot():
    return {name: np.array(getattr(EnvConfig, name), copy=True) for name in GLOBAL_ENV}


def env_restore(env):
    """Put saved values back, in place where possible (other code may hold the EnvConfig arrays)."""
    for name, value in env.items():
        cur = getattr(EnvConfig, name)
        if isinstance(cur, np.ndarray) and cur.shape == np.shape(value):
            cur[...] = value
        else:
            setattr(EnvConfig, name, np.array(value, copy=True))


def looping_of(world):
    """{side: bool} for World.looping (all off for a world without one)."""
    lp = getattr(world, "looping", None)
    if isinstance(lp, dict):
        return {s: bool(lp.get(s, False)) for s in SIDES}
    return {s: False for s in SIDES}


def looping_pending(world, side):
    """Is this side's wall waiting to be re-added (looping just turned off, something still beyond the edge)?"""
    return side in getattr(world, "pending_walls", ())


class Slot:
    """One world of the session: its builder, and (once built) the world, its editor and its saved environment."""

    def __init__(self, name, builder, custom=False):
        self.name, self.builder, self.custom = name, builder, custom
        self.world = self.editor = None
        self.env = None  # GLOBAL_ENV values while the world isn't active

    @property
    def built(self):
        return self.world is not None

    def pending(self, side):
        """Is this side's wall re-add still waiting (see looping_pending)?"""
        return self.built and looping_pending(self.world, side)


class WorldSession:
    def __init__(self, presets, world_size, make_editor, profiler=None, custom_builder=None, start=None):
        """presets: scene.PRESETS (name -> builder(world_width, world_height) -> World); make_editor(world) -> Editor;
        profiler: attached to the active world (world.profiler); custom_builder: what Custom starts as and resets to
        (default: presets["default"]); start: a preset to open (running) at launch instead of Custom (main_opengl.py
        --scene)."""
        self.world_size, self.make_editor, self.profiler = world_size, make_editor, profiler
        self.slots = [Slot(CUSTOM, custom_builder or presets["default"], custom=True)]
        self.slots += [Slot(name, builder) for name, builder in presets.items() if name != "default"]
        self.active = 0
        self.open = False  # the panel
        self.confirm_until = 0.0  # the Custom Reset's confirmation window
        self._shared = None  # UI preferences shared by every world's editor (see _share_ui)
        self._build(self.slots[0])
        self._attach(self.slots[0])
        if start not in (None, "default"):
            self.switch(self.index(start))
        self.editor.paused = False  # the game starts running, as before

    # ---------------- The active world
    @property
    def slot(self):
        return self.slots[self.active]

    @property
    def world(self):
        return self.slot.world

    @property
    def editor(self):
        return self.slot.editor

    @property
    def name(self):
        return self.slot.name

    def index(self, name):
        return next(i for i, s in enumerate(self.slots) if s.name == name)

    # ---------------- Building, switching, resetting
    def _build(self, slot):
        """(Re)build a slot's world from its builder, from the launch environment defaults, with a fresh editor.
        The previous world (if any) is simply dropped."""
        if slot.world is not None:
            slot.world.profiler = None
        env_restore(_ENV_DEFAULTS)
        slot.world = slot.builder(*self.world_size)
        slot.env = env_snapshot()  # the builder may have changed them
        slot.editor = self.make_editor(slot.world)
        slot.editor.paused = True
        self._share_ui(slot.editor)

    def _share_ui(self, editor):
        """Give a world's editor the session's shared UI preferences (taken from the first editor)."""
        if self._shared is None:
            self._shared = editor
        else:
            src = self._shared
            editor.settings, editor.pegs = src.settings, src.pegs  # the same objects: one peg texture, one set of settings
            editor.show_card, editor.show_compute, editor.compute_card = src.show_card, src.show_compute, src.compute_card
        editor.world_selector = self

    def _carry_ui(self, src, dst):
        """Per-frame UI flags the main loop and the F key set on the active editor."""
        dst.show_card, dst.show_compute, dst.compute_card = src.show_card, src.show_compute, src.compute_card
        dst.gravity_card = getattr(src, "gravity_card", None)
        self._shared = dst

    def _detach(self, slot):
        """Leave a world: cancel any placement / spawn menu, pause it, drop the grab, stop profiling it, save its
        environment."""
        ed, world = slot.editor, slot.world
        if ed.active:
            ed.close()
        ed.paused = True
        ed.slider_drag = False
        world.release()
        world.profiler = None
        slot.env = env_snapshot()

    def _attach(self, slot):
        env_restore(slot.env)
        slot.world.profiler = self.profiler

    def switch(self, index):
        """Make slot `index` the active world, opened paused (built first if it's new). The world left is paused and
        frozen as it is. Returns True if the active world changed."""
        if index == self.active:
            return False
        old = self.slot
        self._detach(old)
        new = self.slots[index]
        if not new.built:
            self._build(new)
        self._carry_ui(old.editor, new.editor)
        new.editor.paused = True
        self.active = index
        self._attach(new)
        return True

    def reset(self, index):
        """Rebuild slot `index` from its builder (Custom: back to the default scene) and open it paused."""
        slot, prev = self.slots[index], self.slot
        self._detach(prev)
        prev_editor = prev.editor
        self._build(slot)
        self._carry_ui(prev_editor, slot.editor)
        self.active = index
        self._attach(slot)
        self.confirm_until = 0.0

    def toggle_looping(self, index, side):
        """Flip a world's looping on one side (building the world first if it's new)."""
        slot = self.slots[index]
        if not slot.built:
            env = env_snapshot()
            self._build(slot)
            env_restore(env)  # building restored the defaults; the active world keeps its own
        slot.world.set_looping(side, not looping_of(slot.world)[side])  # off may stay pending (pending_walls)

    @staticmethod
    def looping_shown(slot):
        """The toggles' states: World.looping (a pending side already reads off), all off for an unbuilt world."""
        return looping_of(slot.world) if slot.built else {s: False for s in SIDES}

    def ask_reset(self, index):
        """Reset button: presets reset at once; Custom asks for a second click within CONFIRM_SECONDS."""
        if self.slots[index].custom and time.time() > self.confirm_until:
            self.confirm_until = time.time() + CONFIRM_SECONDS
            return
        self.reset(index)

    def toggle(self):
        self.open = not self.open
        self.confirm_until = 0.0

    # ---------------- Layout (UI px)
    def _layout(self):
        """{"panel", "close", "rows": [(slot index, name rect, [(side, toggle rect)], reset rect)], "head_y",
        "foot_y"}, centered in the editor's free area (left of the HUD cards)."""
        w = PAD + NAME_W + COL_GAP + 4 * TOGGLE_W + 3 * ROW_GAP + COL_GAP + RESET_W + PAD
        n = len(self.slots)
        h = HEADER_H + LINE_H + n * (ROW_H + ROW_GAP) - ROW_GAP + PAD + 2 * LINE_H + PAD // 2
        area = self.editor._area()
        panel = pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), max(ui.CARD_MARGIN, area.centery - h // 2), w, h)
        head_y = panel.y + HEADER_H
        y0 = head_y + LINE_H
        rows = []
        for i in range(n):
            y = y0 + i * (ROW_H + ROW_GAP)
            name = pygame.Rect(panel.x + PAD, y, NAME_W, ROW_H)
            x = name.right + COL_GAP
            toggles = [(side, pygame.Rect(x + k * (TOGGLE_W + ROW_GAP), y, TOGGLE_W, ROW_H)) for k, side in enumerate(SIDES)]
            reset = pygame.Rect(toggles[-1][1].right + COL_GAP, y, RESET_W, ROW_H)
            rows.append((i, name, toggles, reset))
        close = pygame.Rect(panel.right - 6 - ROW_H, panel.y + 6, ROW_H, ROW_H)
        return {"panel": panel, "close": close, "rows": rows, "head_y": head_y,
                "foot_y": y0 + n * (ROW_H + ROW_GAP) - ROW_GAP + PAD}

    def _widgets(self):
        """(rect, action, draw spec) for every clickable thing on the panel."""
        lay = self._layout()
        ws = [(lay["close"], self.toggle, ("button", "x", False, False))]
        confirming = time.time() <= self.confirm_until
        for i, name, toggles, reset in lay["rows"]:
            slot = self.slots[i]
            ws.append((name, lambda i=i: self.switch(i), ("world", i)))
            shown = self.looping_shown(slot)
            for side, rect in toggles:
                ws.append((rect, lambda i=i, s=side: self.toggle_looping(i, s), ("loop", i, side, shown[side])))
            label = "Sure?" if slot.custom and confirming else "Reset"
            ws.append((reset, (lambda i=i: self.ask_reset(i)) if slot.built else None,
                       ("button", label, slot.custom and confirming, not slot.built)))
        return ws

    # ---------------- Events (mouse in real screen px)
    def handle_event(self, event):
        """W toggles the panel (Esc closes it). While it's open, clicks on it are its own; a click elsewhere (not on
        the HUD cards) closes it. Returns True if the event was used (the active world may have changed)."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w and not event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META):
                self.toggle()
                return True
            if self.open and event.key == pygame.K_ESCAPE:
                self.toggle()
                return True
            return False
        if not self.open or event.type != pygame.MOUSEBUTTONDOWN:
            return False
        ed = self.editor
        pos = ed.to_ui(event.pos)
        if self._layout()["panel"].collidepoint(pos):
            if event.button == 1:
                for rect, action, _ in self._widgets():
                    if rect.collidepoint(pos) and action is not None:
                        action()
                        break
            return True
        if any(w[0].collidepoint(event.pos) for w in ed._card_widgets()):
            return False  # the Controls / compute cards keep working (their World button closes the panel)
        self.toggle()
        return True

    # ---------------- Drawing (UI surface)
    def draw(self, surface, mouse_ui):
        if not self.open:
            return
        ed = self.editor
        surface.blit(ed._dim, (0, 0))
        lay = self._layout()
        panel = lay["panel"]
        ui.draw_frame(surface, panel, ui.PANEL_BG)
        ui.text(surface, "Worlds", (panel.centerx, panel.y + 6 + ROW_H // 2), ui.WHITE, "bold", ui.TITLE_SIZE, "center", "middle")
        rows = lay["rows"]
        _, name0, toggles0, reset0 = rows[0]
        hy = lay["head_y"] + LINE_H // 2 - 1
        ui.text(surface, "world", (name0.x + 2, hy), ui.TEXT_DIM, valign="middle")
        ui.text(surface, "looping", ((toggles0[0][1].x + toggles0[-1][1].right) // 2, hy), ui.TEXT_DIM, align="center", valign="middle")
        for rect, _, spec in self._widgets():
            hovered = rect.collidepoint(mouse_ui)
            kind = spec[0]
            if kind == "button":
                ui.draw_button(surface, rect, spec[1], hovered, spec[2], spec[3])
            elif kind == "world":
                self._draw_world_button(surface, rect, spec[1], hovered)
            elif kind == "loop":
                _, i, side, on = spec
                slot = self.slots[i]
                pending = slot.pending(side)
                ui.draw_button(surface, rect, None, hovered, on and not pending, False)
                col = ui.ACCENT if pending else ui.WHITE if on else ui.TEXT_DIM
                ui.text(surface, SIDE_LABELS[side], rect.center, col, "bold", align="center", valign="middle")
        fy = lay["foot_y"]
        pending = any(s.pending(side) for s in self.slots for side in SIDES)
        lines = ["click a world to open it (paused)  ·  W / Esc close"]
        lines.append("pink: wall returns once the edge is clear" if pending else "L R T B: loop out one side, in the other")
        for k, line in enumerate(lines):
            ui.text(surface, line, (panel.centerx, fy + k * LINE_H + LINE_H // 2), ui.ACCENT if (k and pending) else ui.TEXT_DIM,
                    align="center", valign="middle")

    def _draw_world_button(self, surface, rect, i, hovered):
        """The world's name, and its state on the right: running / paused / new (not built yet)."""
        slot = self.slots[i]
        active = i == self.active
        ui.draw_button(surface, rect, None, hovered, active, False)
        ui.text(surface, slot.name, (rect.x + 5, rect.centery), ui.WHITE, "bold", valign="middle")
        state = "new" if not slot.built else "paused" if slot.editor.paused else "running"
        ui.text(surface, state, (rect.right - 5, rect.centery), ui.WHITE if active else ui.TEXT_DIM, align="right", valign="middle")
