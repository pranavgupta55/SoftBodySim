"""A session of several live worlds: the sidebar's Worlds card and the world selector panel (W, [w] on the Controls
card, or a click on the Worlds card).

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

The Worlds card (the session is a sidebar card provider, see hud_layer.py) shows the active world: its name (a button
that opens the panel), whether it's running, and its looping sides as L R T B toggles; its compact form drops the
toggles. The full list (every world with its state, L R T B toggles and Reset) doesn't fit the sidebar on small
screens, so it's the panel: drawn on the scaled-down UI surface in the editor's pixel style (ui_style), centred in the
world area over a dimmed scene, while the sim keeps running behind it. Panel rects are UI-surface px, card rects
native screen px.
"""
import time

import numpy as np
import pygame

import ui_style as ui
from ui_style import HUD
from entities import EnvConfig
from editor import BTN_GAP, CARD_BTN_H, CARD_TITLE_H

SIDES = ("left", "right", "top", "bottom")
SIDE_LABELS = {"left": "L", "right": "R", "top": "T", "bottom": "B"}
GRID_SIZES = (8, 24, 96)  # broad phase cell sizes a world can pick (World.env["cell_size"]; default EnvConfig's)
GLOBAL_ENV = ("gravity",)  # EnvConfig attributes saved / restored per world
CUSTOM = "Custom"
CONFIRM_SECONDS = 3.0  # the Custom world's Reset asks for a second click within this long

# layout, UI px
PAD = 8
HEADER_H = 30
ROW_H = 18
ROW_GAP = 3
NAME_W = 160
TOGGLE_W = 18
RESET_W = 46
GRID_W = 22
COL_GAP = 8
LINE_H = 13
INDENT = 12  # a world inside a folder: its name button starts this far right
FOLDER_SEP = "/"  # a preset named "folder/name" is listed inside that folder (a collapsible row)
# UI attributes the main loop / keys set on the active editor, carried to the next one on a switch
CARRIED = ("show_compute", "overlays", "time", "_card_pos")

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


def grid_of(world):
    """The world's broad phase cell size: its own (World.env) or EnvConfig's."""
    return int(round(float(world.env.get("cell_size", EnvConfig.cell_size))))


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
        self.collapsed = set()  # folders closed on the panel (their worlds hidden; folders start open)
        self.confirm_until = 0.0  # the Custom Reset's confirmation window
        self._shared = None  # UI preferences shared by every world's editor (see _share_ui)
        self._cards = None  # the sidebar's card list (bind_cards): the active editor is swapped into it
        self.card_rect = pygame.Rect(0, 0, 0, 0)  # the Worlds card, native screen px (set by the sidebar, place)
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
            self._copy_ui(src, editor)
        editor.world_selector = self

    @staticmethod
    def _copy_ui(src, dst):
        for name in CARRIED:
            if hasattr(src, name):
                setattr(dst, name, getattr(src, name))

    def _carry_ui(self, src, dst):
        """Per-frame UI flags the main loop and the F key set on the active editor."""
        self._copy_ui(src, dst)
        self._shared = dst
        if self._cards is not None:  # the Controls card is the active editor
            self._cards[:] = [dst if c is src else c for c in self._cards]

    def bind_cards(self, cards):
        """The sidebar's card list (HudLayer.cards, holding the active editor as the Controls card): kept in sync."""
        self._cards = cards

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

    def replace_world(self, world, env=None):
        """Put a loaded world (e.g. a scene file opened with Ctrl/Cmd+O) into the active slot, opened paused with a
        fresh editor; env: its GLOBAL_ENV values (default: keep the current ones). The slot keeps its name, and its
        Reset still rebuilds it from its builder. The main loop must rebind world / editor afterwards."""
        slot = self.slot
        old_editor = slot.editor
        self._detach(slot)
        slot.world = world
        slot.editor = self.make_editor(world)
        self._share_ui(slot.editor)
        self._carry_ui(old_editor, slot.editor)
        slot.editor.paused = True
        if env is not None:
            slot.env = {name: np.array(value, copy=True) for name, value in env.items()}
        self._attach(slot)

    def toggle_looping(self, index, side):
        """Flip a world's looping on one side (building the world first if it's new)."""
        slot = self.slots[index]
        if not slot.built:
            env = env_snapshot()
            self._build(slot)
            env_restore(env)  # building restored the defaults; the active world keeps its own
        slot.world.set_looping(side, not looping_of(slot.world)[side])  # off may stay pending (pending_walls)

    def set_grid(self, index, size):
        """A world's broad phase cell size (building the world first if it's new). Results don't depend on it, only
        speed: small cells suit small bodies, big ones sparse scenes."""
        slot = self.slots[index]
        if not slot.built:
            env = env_snapshot()
            self._build(slot)
            env_restore(env)
        slot.world.env["cell_size"] = float(size)

    @staticmethod
    def grid_shown(slot):
        """The grid size buttons' state: the world's cell size (EnvConfig's default for an unbuilt world)."""
        return grid_of(slot.world) if slot.built else int(EnvConfig.cell_size)

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

    @staticmethod
    def folder_of(name):
        """The folder a world's name puts it in ("diamonds/evolved 1" -> "diamonds"), or None."""
        return name.split(FOLDER_SEP, 1)[0] if FOLDER_SEP in name else None

    def toggle_folder(self, folder):
        """Open / close a folder's row on the panel."""
        self.collapsed ^= {folder}

    def folder_open(self, folder):
        return folder not in self.collapsed

    def _entries(self):
        """The panel's rows top to bottom: ("folder", name) the first time a folder comes up, then ("world", slot
        index) for each world that's shown (one in a closed folder isn't)."""
        out, seen = [], set()
        for i, slot in enumerate(self.slots):
            folder = self.folder_of(slot.name)
            if folder is not None and folder not in seen:
                seen.add(folder)
                out.append(("folder", folder))
            if folder is None or self.folder_open(folder):
                out.append(("world", i))
        return out

    # ---------------- Layout (UI px)
    def _layout(self):
        """{"panel", "close", "rows": [(slot index, name rect, [(side, toggle rect)], reset rect, [(size, grid rect)])],
        "folders": [(folder, rect)], "head_y", "foot_y"}, centered in the editor's free area (left of the HUD cards).
        Rows are the shown worlds only (see _entries); a world in a folder has its name button indented."""
        nk = len(GRID_SIZES)
        w = (PAD + NAME_W + COL_GAP + 4 * TOGGLE_W + 3 * ROW_GAP + COL_GAP + nk * GRID_W + (nk - 1) * ROW_GAP + COL_GAP
             + RESET_W + PAD)
        entries = self._entries()
        n = len(entries)
        h = HEADER_H + LINE_H + n * (ROW_H + ROW_GAP) - ROW_GAP + PAD + 2 * LINE_H + PAD // 2
        area = self.editor._area()
        panel = pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), max(ui.CARD_MARGIN, area.centery - h // 2), w, h)
        head_y = panel.y + HEADER_H
        y0 = head_y + LINE_H
        rows, folders = [], []
        for row, (kind, what) in enumerate(entries):
            y = y0 + row * (ROW_H + ROW_GAP)
            if kind == "folder":
                folders.append((what, pygame.Rect(panel.x + PAD, y, NAME_W, ROW_H)))
                continue
            i = what
            indent = INDENT if self.folder_of(self.slots[i].name) is not None else 0
            name = pygame.Rect(panel.x + PAD + indent, y, NAME_W - indent, ROW_H)
            x = panel.x + PAD + NAME_W + COL_GAP
            toggles = [(side, pygame.Rect(x + k * (TOGGLE_W + ROW_GAP), y, TOGGLE_W, ROW_H)) for k, side in enumerate(SIDES)]
            gx = toggles[-1][1].right + COL_GAP
            grids = [(size, pygame.Rect(gx + k * (GRID_W + ROW_GAP), y, GRID_W, ROW_H)) for k, size in enumerate(GRID_SIZES)]
            reset = pygame.Rect(grids[-1][1].right + COL_GAP, y, RESET_W, ROW_H)
            rows.append((i, name, toggles, reset, grids))
        close = pygame.Rect(panel.right - 6 - ROW_H, panel.y + 6, ROW_H, ROW_H)
        return {"panel": panel, "close": close, "rows": rows, "folders": folders, "head_y": head_y,
                "foot_y": y0 + n * (ROW_H + ROW_GAP) - ROW_GAP + PAD}

    def _widgets(self):
        """(rect, action, draw spec) for every clickable thing on the panel."""
        lay = self._layout()
        ws = [(lay["close"], self.toggle, ("button", "x", False, False))]
        confirming = time.time() <= self.confirm_until
        for folder, rect in lay["folders"]:
            ws.append((rect, lambda f=folder: self.toggle_folder(f), ("folder", folder)))
        for i, name, toggles, reset, grids in lay["rows"]:
            slot = self.slots[i]
            ws.append((name, lambda i=i: self.switch(i), ("world", i)))
            shown = self.looping_shown(slot)
            for side, rect in toggles:
                ws.append((rect, lambda i=i, s=side: self.toggle_looping(i, s), ("loop", i, side, shown[side])))
            grid = self.grid_shown(slot)
            for size, rect in grids:
                ws.append((rect, lambda i=i, z=size: self.set_grid(i, z), ("button", str(size), size == grid, False)))
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
        if event.type == pygame.MOUSEBUTTONDOWN and self.card_rect.collidepoint(event.pos):
            if event.button == 1:
                for rect, action, _ in self._card_widgets():
                    if rect.collidepoint(event.pos) and action is not None:
                        action()
                        break
            return True  # the whole card swallows clicks
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
            return False  # the sidebar keeps working ([w] closes the panel)
        self.toggle()
        return True

    def on_panel(self, pos):
        """Is real screen px `pos` on the open panel?"""
        return self.open and self._layout()["panel"].collidepoint(self.editor.to_ui(pos))

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
        _, name0, toggles0, reset0, grids0 = rows[0]
        hy = lay["head_y"] + LINE_H // 2 - 1
        ui.text(surface, "world", (name0.x + 2, hy), ui.TEXT_DIM, valign="middle")
        ui.text(surface, "looping", ((toggles0[0][1].x + toggles0[-1][1].right) // 2, hy), ui.TEXT_DIM, align="center", valign="middle")
        ui.text(surface, "grid", ((grids0[0][1].x + grids0[-1][1].right) // 2, hy), ui.TEXT_DIM, align="center", valign="middle")
        for rect, _, spec in self._widgets():
            hovered = rect.collidepoint(mouse_ui)
            kind = spec[0]
            if kind == "button":
                ui.draw_button(surface, rect, spec[1], hovered, spec[2], spec[3])
            elif kind == "world":
                self._draw_world_button(surface, rect, spec[1], hovered)
            elif kind == "folder":
                self._draw_folder_button(surface, rect, spec[1], hovered)
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
        lines.append("pink: wall returns once the edge is clear" if pending
                     else "L R T B: loop out one side, in the other  ·  grid: collision cell size")
        for k, line in enumerate(lines):
            ui.text(surface, line, (panel.centerx, fy + k * LINE_H + LINE_H // 2), ui.ACCENT if (k and pending) else ui.TEXT_DIM,
                    align="center", valign="middle")

    def _draw_folder_button(self, surface, rect, folder, hovered):
        """A folder's row: open / closed marker, its name, and how many worlds it holds."""
        ui.draw_button(surface, rect, None, hovered, False, False)
        is_open = self.folder_open(folder)
        cx, cy = rect.x + 7, rect.centery
        tri = [(cx - 3, cy - 2), (cx + 3, cy - 2), (cx, cy + 2)] if is_open else [(cx - 2, cy - 3), (cx - 2, cy + 3), (cx + 2, cy)]
        pygame.draw.polygon(surface, ui.WHITE, tri)
        ui.text(surface, folder, (rect.x + 14, rect.centery), ui.WHITE, "bold", valign="middle")
        count = sum(self.folder_of(s.name) == folder for s in self.slots)
        ui.text(surface, f"{count} worlds", (rect.right - 5, rect.centery), ui.TEXT_DIM, align="right", valign="middle")

    def _draw_world_button(self, surface, rect, i, hovered):
        """The world's name, and its state on the right: running / paused / new (not built yet)."""
        slot = self.slots[i]
        active = i == self.active
        ui.draw_button(surface, rect, None, hovered, active, False)
        state = "new" if not slot.built else "paused" if slot.editor.paused else "running"
        room = rect.w - 15 - ui.font().size(state)[0]
        name = slot.name.split(FOLDER_SEP, 1)[-1]  # (the folder row above says which folder)
        while len(name) > 1 and ui.font("bold").size(name)[0] > room:  # cut long names to fit before the state
            name = name.rstrip(".")[:-1].rstrip() + ".."
        ui.text(surface, name, (rect.x + 5, rect.centery), ui.WHITE, "bold", valign="middle")
        ui.text(surface, state, (rect.right - 5, rect.centery), ui.WHITE if active else ui.TEXT_DIM, align="right", valign="middle")

    # ---------------- The Worlds card (a sidebar card provider, hud_layer.py; native screen px, HUD style)
    compactable = True

    @staticmethod
    def card_height(compact=False):
        """Title, the active world's button, and (not compact) its looping toggles and grid sizes. Independent of the
        state."""
        px, pad = HUD.px, HUD.CARD_PAD
        h = pad + px(CARD_TITLE_H) + px(CARD_BTN_H) + pad - px(1)
        return h if compact else h + 2 * (px(BTN_GAP) + px(CARD_BTN_H))

    def place(self, rect):
        self.card_rect = pygame.Rect(rect)

    def _card_layout(self):
        """{"name": the active world's button, "toggles": [(side, rect)], "grids": [(size, rect)] (both empty when
        compact), "loop_y" / "grid_y": their label rows}, native screen px."""
        r, px, pad = self.card_rect, HUD.px, HUD.CARD_PAD
        gap, bh = px(BTN_GAP), px(CARD_BTN_H)
        y = r.y + pad + px(CARD_TITLE_H)
        lay = {"name": pygame.Rect(r.x + pad, y, r.w - 2 * pad, bh), "toggles": [], "loop_y": None, "grids": [],
               "grid_y": None}
        if r.h >= self.card_height(False):
            y += bh + gap
            label_w = max(HUD.font().size(t)[0] for t in ("loop", "grid")) + px(6)
            x0 = r.x + pad + label_w
            tw = (r.right - pad - x0 - 3 * gap) // 4
            lay["toggles"] = [(side, pygame.Rect(x0 + k * (tw + gap), y, tw if k < 3 else r.right - pad - x0 - 3 * (tw + gap), bh))
                              for k, side in enumerate(SIDES)]
            lay["loop_y"] = y + bh // 2
            y += bh + gap
            n = len(GRID_SIZES)
            gw = (r.right - pad - x0 - (n - 1) * gap) // n
            lay["grids"] = [(size, pygame.Rect(x0 + k * (gw + gap), y, gw if k < n - 1 else r.right - pad - x0 - (n - 1) * (gw + gap), bh))
                            for k, size in enumerate(GRID_SIZES)]
            lay["grid_y"] = y + bh // 2
        return lay

    def _card_widgets(self):
        lay = self._card_layout()
        ws = [(lay["name"], self.toggle, ("name",))]
        lp = looping_of(self.world)
        ws += [(rect, lambda s=side: self.toggle_looping(self.active, s), ("loop", side, lp[side])) for side, rect in lay["toggles"]]
        grid = grid_of(self.world)
        ws += [(rect, lambda z=size: self.set_grid(self.active, z), ("grid", size, size == grid)) for size, rect in lay["grids"]]
        return ws

    def draw_card(self, surface, origin, mouse_px, compact=False):
        """Title + run state, the active world's name (opens the panel; lit while it's open), its looping toggles."""
        ox, oy = origin
        r = self.card_rect.move(-ox, -oy)
        HUD.draw_frame(surface, r)
        px, pad = HUD.px, HUD.CARD_PAD
        HUD.text(surface, "Worlds", (r.x + pad, r.y + pad - px(1)), ui.WHITE, "bold", HUD.TITLE_SIZE)
        HUD.text(surface, "paused" if self.editor.paused else "running", (r.right - pad, r.y + pad), ui.ACCENT, "bold",
                 align="right")
        lay = self._card_layout()
        slot = self.slot
        for rect, _, spec in self._card_widgets():
            local, hovered = rect.move(-ox, -oy), rect.collidepoint(mouse_px)
            if spec[0] == "name":
                HUD.draw_button(surface, local, None, hovered, self.open)
                room = local.w - 2 * px(5) - HUD.font().size("[w]")[0] - px(4)
                name = self.name
                while len(name) > 1 and HUD.font("bold").size(name)[0] > room:
                    name = name.rstrip(".")[:-1].rstrip() + ".."
                HUD.text(surface, name, (local.x + px(5), local.centery), ui.WHITE, "bold", valign="middle")
                HUD.text(surface, "[w]", (local.right - px(5), local.centery), ui.TEXT_DIM if not self.open else ui.WHITE,
                         align="right", valign="middle")
            elif spec[0] == "grid":
                _, size, on = spec
                HUD.draw_button(surface, local, None, hovered, on)
                HUD.text(surface, str(size), local.center, ui.WHITE if on else ui.TEXT_DIM, "bold", align="center",
                         valign="middle")
            else:
                _, side, on = spec
                pending = slot.pending(side)
                HUD.draw_button(surface, local, None, hovered, on and not pending)
                col = ui.ACCENT if pending else ui.WHITE if on else ui.TEXT_DIM
                HUD.text(surface, SIDE_LABELS[side], local.center, col, "bold", align="center", valign="middle")
        if lay["loop_y"] is not None:
            HUD.text(surface, "loop", (r.x + pad, lay["loop_y"] - oy), ui.TEXT, valign="middle")
            HUD.text(surface, "grid", (r.x + pad, lay["grid_y"] - oy), ui.TEXT, valign="middle")
