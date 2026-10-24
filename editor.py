"""In-game editor: pause/play, a spawn menu with per-body settings, and point placement with validation.

All UI (the info card, panels, buttons, hints, messages) is drawn on the scaled-down UI surface in the shared
ui_style (Endesga colors, solid black 2px borders, non-antialiased text), so it has the same pixel look as the world.
UI positions are UI-surface pixels: mouse events arrive in real screen pixels and are divided by the scale-down
factor; world units are UI pixels / zoom. Placement previews are drawn in world space on the game surfaces.
"""
import math
import time

import numpy as np
import pygame

import ui_style as ui
from entities import EnvConfig, FixedBody, Hook, RigidBody, Rope, SoftBody, polygon_points

KIND_TITLES = {"soft": "Soft body", "rigid": "Rigid body", "fixed": "Fixed body", "rope": "Rope", "hook": "Hook"}
POINT_COUNTS = {"soft": (3, 4, 5, 6, 7, 8, 12, 20), "rigid": (3, 4, 5, 6, 7, 8, 12, 20), "fixed": (2, 3, 4, 6, 8, 12)}
STRENGTH_RANGE = (0.1, 1.0)  # soft body spring strength and rope stiffness
ROPE_WIDTHS = (1, 2)
SNAP_UI_PX = 5  # a rope end clicked this close (UI px) to a body point or hook attaches to it
PEG_COL = (255, 255, 255)  # peg grid overlay shown while placing a hook (one pixel per peg, drawn over the scene)
PEG_ALPHA = 100
MESSAGE_SECONDS = 1.6
BAD_SPOT = "you cannot spawn a point there"
INCOMPLETE = "complete the new object to resume the sim"
MESSAGE_SIZE = 20  # bold, shrunk to fit narrow surfaces

# layout, in UI-surface pixels
PAD = 8  # panel padding
HEADER_H = 30  # panel title row
LABEL_H = 14  # a row label above its controls
ROW_GAP = 7
BTN_H = 18
BTN_GAP = 3
MODE_H = 22
COL_W = 232  # settings column of the config panel; the shape preview fills the rest
CONFIG_W = 400
LINE_H = 13  # one hotkey / hint line
CARD_TITLE_H = 16
CARD_BTN_H = 20
KEY_COL_W = 42  # hotkey name column on the info card


def max_skip_sets(n):
    """Largest skip_sets that still adds new springs (sets reach the 2nd .. (n // 2)th neighbor)."""
    return max(0, n // 2 - 1)


# ---------------- Geometry helpers (arrays are (2, k), one column per point)
def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def point_seg_dist(p, a, b):
    """Distance from point(s) p to segment(s) a -> b, broadcasting over columns."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=0) / np.maximum(np.sum(ab * ab, axis=0), 1e-12), 0.0, 1.0)
    return np.hypot(*(p - (a + t * ab)))


def seg_seg_dist(p, q, a, b):
    """Distance from segment p -> q to each segment a -> b (0 where they cross)."""
    p, q = p.reshape(2, 1), q.reshape(2, 1)
    d = np.minimum.reduce([point_seg_dist(p, a, b), point_seg_dist(q, a, b), point_seg_dist(a, p, q), point_seg_dist(b, p, q)])
    crosses = (_cross(q - p, a - p) * _cross(q - p, b - p) < 0) & (_cross(b - a, p - a) * _cross(b - a, q - a) < 0)
    return np.where(crosses, 0.0, d)


def point_in_poly(pt, poly):
    """Even-odd rule."""
    x, y = pt
    px, py = poly
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    straddle = (py > y) != (qy > y)
    x_cross = px + (y - py) * (qx - px) / np.where(straddle, qy - py, 1.0)
    return np.count_nonzero(straddle & (x < x_cross)) % 2 == 1


def skip_springs(n, sets):
    """Spring index pairs a SoftBody with n points and `sets` skip sets would get (for previews)."""
    pairs = set()
    for i in range(n):
        for k in range(1, sets + 2):
            j = (i + k) % n
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def peg_positions(lo, hi, spacing):
    """Peg coordinates along one axis: the multiples of spacing (aligned to the world origin) strictly between lo
    and hi, so pegs sit inside the world box and never on its walls."""
    first = math.floor(lo / spacing + 1e-9) + 1
    last = math.ceil(hi / spacing - 1e-9) - 1
    return np.arange(first, last + 1) * float(spacing)


class Editor:
    def __init__(self, world, palette, ui_size, world_size, scale, inset, zoom=1.0, view=None):
        """ui_size: the UI surface size (real screen / scale); scale: real screen px per UI px; zoom: UI px per world
        unit; view: an object with `show_grid` and `filled` (the G / V states listed on the info card)."""
        self.world, self.palette, self.scale, self.inset, self.zoom, self.view = world, palette, scale, inset, zoom, view
        self.W, self.H = ui_size
        self.world_w, self.world_h = world_size
        self.colors = [(name, tuple(col)) for name, col in vars(palette).items() if not name.startswith("_") and isinstance(col, list)]
        self.settings = {"soft": {"n": 6, "sets": 1, "strength": 0.1, "color": "network_green"},  # the hexagon
                         "rigid": {"n": 4, "color": "debug_red"},  # the square
                         "fixed": {"n": 4, "color": "grey"},
                         "rope": {"n": 2, "stiffness": 1.0, "width": 1, "color": "light_brown"},  # like the demo ropes
                         "hook": {"n": 1, "color": "greyL"}}  # one hook per Finish, like the other kinds
        self.paused = False
        self.active = False  # editor mode
        self.stage = None  # "menu" -> "config" -> "place"
        self.kind = None
        self.mode = None  # "regular", "custom" or "rope"
        self.points = []  # placed points (world coords) of the object being built
        self.snaps = []  # per point: (body, point_index) a rope end is attached to, or None (free point / anchor)
        self.undo_stack, self.redo_stack = [], []
        self.drag_idx = self.drag_from = None
        self.reg_start = None  # press position while dragging out a regular polygon
        self.slider_drag = False
        self.message, self.message_until = None, 0.0
        self.show_card = True  # the main loop mirrors its HUD toggle (SPACE) here: a hidden card takes no clicks
        self._dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._dim.fill((0, 0, 0, 140))
        self._peg_layer, self._peg_key = None, None  # cached peg grid overlay and what it was built for

    # ---------------- Small utilities
    def _color(self, name):
        return dict(self.colors)[name]

    def _say(self, text):
        self.message, self.message_until = text, time.time() + MESSAGE_SECONDS

    def _cfg(self):
        return self.settings[self.kind]

    def _complete(self):
        return len(self.points) == self._cfg()["n"]

    def to_ui(self, screen_pos):
        """Real screen pixels -> UI surface pixels."""
        return screen_pos[0] / self.scale, screen_pos[1] / self.scale

    def to_world(self, ui_pos):
        """UI surface pixels -> world units."""
        return np.array(ui_pos, dtype=float) / self.zoom

    # ---------------- Mode changes
    def toggle_pause(self):
        if self.active:
            if self.stage == "place":
                self.finish()
            else:
                self.close()
                self.paused = False
        else:
            self.paused = not self.paused
            if self.paused:
                self.world.release()

    def open(self):
        self.active, self.stage = True, "menu"

    def close(self):
        self.active, self.stage, self.kind, self.mode = False, None, None, None
        self._reset_object()

    def back(self):
        if self.stage == "place":
            self._reset_object()
            self.stage = "config"
        elif self.stage == "config":
            self.stage = "menu"
        else:
            self.close()

    def _reset_object(self):
        self.points, self.snaps, self.undo_stack, self.redo_stack = [], [], [], []
        self.drag_idx = self.drag_from = self.reg_start = None

    def choose_kind(self, kind):
        self.kind, self.stage = kind, "config"

    def start_place(self, mode):
        self._reset_object()
        self.mode, self.stage = mode, "place"

    def finish(self):
        if not self.points or not self._complete():
            self._say(INCOMPLETE)
            return
        cfg, col = self._cfg(), self._color(self._cfg()["color"])
        if self.kind == "hook":
            self.world.add_hook(Hook(self.points[0], col=col))
            self.close()
            self.paused = False
            return
        if self.kind == "rope":  # rest length = the ends' current distance, so it starts exactly taut
            ends = [snap if snap is not None else tuple(p) for p, snap in zip(self.points, self.snaps)]
            self.world.add_rope(Rope(ends[0], ends[1], stiffness=cfg["stiffness"], col=col, width=cfg["width"]))
            self.close()
            self.paused = False
            return
        pts = np.array(self.points, dtype=float).T
        style = dict(point_col=col, stick_col=col, fill_col=col)
        if self.kind == "soft":
            body = SoftBody(pts, stiffness=cfg["strength"], skip_sets=cfg["sets"], point_radius=3, **style)
        elif self.kind == "rigid":
            body = RigidBody(pts, point_radius=3, **style)
        else:
            body = FixedBody(pts, point_radius=2, stick_width=2 if pts.shape[1] == 2 else 1, **style)
        self.world.add(body)
        self.close()
        self.paused = False

    # ---------------- Undo / redo
    def _push(self, action):
        self.undo_stack.append(action)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        if action[0] == "add":
            self.points.pop()
            self.snaps.pop()
        elif action[0] == "poly":
            self.points, self.snaps = [], []
        else:  # move
            self.points[action[1]], self.snaps[action[1]] = action[2], action[4]
        self.redo_stack.append(action)

    def redo(self):
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        if action[0] == "add":
            self.points.append(action[1])
            self.snaps.append(action[2])
        elif action[0] == "poly":
            self.points, self.snaps = list(action[1]), [None] * len(action[1])
        else:
            self.points[action[1]], self.snaps[action[1]] = action[3], action[5]
        self.undo_stack.append(action)

    # ---------------- Placement validation
    def _valid(self, pts, complete):
        """Can these points (the object so far) exist without touching the world or themselves?"""
        r = EnvConfig.contact_radius
        new = np.array(pts, dtype=float).T
        k = new.shape[1]
        lo, hi_x, hi_y = self.inset + r, self.world_w - self.inset - r, self.world_h - self.inset - r
        if (new[0] < lo).any() or (new[0] > hi_x).any() or (new[1] < lo).any() or (new[1] > hi_y).any():
            return False

        world = self.world
        has_edges = bool(world.bodies) and world.edges.shape[1] > 0
        if has_edges:
            a, b = world.P[:, world.edges[0]], world.P[:, world.edges[1]]
        polygons = [body.points for body in world.bodies if body.edges.shape[1] >= 3]
        for i in range(k):
            p = new[:, i:i + 1]
            if has_edges and point_seg_dist(p, a, b).min() < r:
                return False
            if any(point_in_poly(new[:, i], poly) for poly in polygons):
                return False

        edges = [(i, i + 1) for i in range(k - 1)]
        if complete and k >= 3:
            edges.append((k - 1, 0))
        for i, j in edges:
            if has_edges and seg_seg_dist(new[:, i], new[:, j], a, b).min() < r:
                return False

        # the object itself: points spread out, edges not crossing each other
        for i in range(k):
            for j in range(i + 1, k):
                if np.hypot(*(new[:, i] - new[:, j])) < 2 * r:
                    return False
        for x, (i, j) in enumerate(edges):
            for (m, n) in edges[x + 1:]:
                if len({i, j, m, n}) == 4 and seg_seg_dist(new[:, i], new[:, j], new[:, m:m + 1], new[:, n:n + 1])[0] == 0.0:
                    return False

        if complete and k >= 3:
            x, y = new
            if abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < r * r:
                return False  # degenerate sliver
            if world.bodies and any(point_in_poly(world.P[:, c], new) for c in range(world.P.shape[1])):
                return False  # would swallow an existing body
        return True

    def _valid_rope(self, pts, snaps):
        """Rope ends: inside the world box, a free anchor not inside a body (attach to its point instead), and the
        two ends not the same point. Ropes don't collide, so the line itself may cross bodies."""
        lo, hi_x, hi_y = self.inset, self.world_w - self.inset, self.world_h - self.inset
        polygons = [body.points for body in self.world.bodies if body.edges.shape[1] >= 3]
        for p, snap in zip(pts, snaps):
            if not (lo < p[0] < hi_x and lo < p[1] < hi_y):
                return False
            if snap is None and any(point_in_poly(p, poly) for poly in polygons):
                return False
        if len(pts) == 2:
            (a, b), (sa, sb) = pts, snaps
            if (sa is not None and sb is not None and self._same_snap(sa, sb)) or np.hypot(*(np.asarray(a) - b)) < 1.0:
                return False
        return True

    @staticmethod
    def _same_snap(sa, sb):
        if isinstance(sa, Hook) or isinstance(sb, Hook):
            return sa is sb
        return sa[0] is sb[0] and sa[1] == sb[1]

    @staticmethod
    def _snap_pos(snap):
        """World position of a rope snap: a Hook or a (body, point_index)."""
        return snap.pos.copy() if isinstance(snap, Hook) else np.array(snap[0].points[:, snap[1]], dtype=float)

    def _snap_target(self, wp):
        """The nearest hook or (body, point_index) body point within SNAP_UI_PX of wp, or None."""
        best, best_d = None, SNAP_UI_PX / self.zoom
        for body in self.world.bodies:
            d = np.hypot(*(body.points - np.reshape(wp, (2, 1))))
            i = int(np.argmin(d))
            if d[i] <= best_d:
                best, best_d = (body, i), d[i]
        for hook in self.world.hooks:
            d = np.hypot(*(hook.pos - np.asarray(wp, dtype=float)))
            if d <= best_d:
                best, best_d = hook, d
        return best

    def _rope_end(self, wp):
        """Where a rope end at wp goes: (position, snap) — onto a body point or hook if one is close, else a free anchor."""
        snap = self._snap_target(wp)
        return (self._snap_pos(snap), snap) if snap is not None else (np.asarray(wp, dtype=float), None)

    # ---------------- Hooks and the peg grid (world units)
    def _pegs(self):
        """Peg x and y coordinates: EnvConfig.peg_spacing apart, aligned to the world origin, inside the world box."""
        s = EnvConfig.peg_spacing
        return (peg_positions(self.inset, self.world_w - self.inset, s),
                peg_positions(self.inset, self.world_h - self.inset, s))

    def _in_box(self, p):
        return self.inset < p[0] < self.world_w - self.inset and self.inset < p[1] < self.world_h - self.inset

    def _hook_spot(self, wp):
        """The peg a hook clicked at wp snaps to (the nearest one inside the world box), or None outside the box."""
        xs, ys = self._pegs()
        if not self._in_box(wp) or not (xs.size and ys.size):
            return None
        s = EnvConfig.peg_spacing
        ix = int(np.clip(round((wp[0] - xs[0]) / s), 0, xs.size - 1))
        iy = int(np.clip(round((wp[1] - ys[0]) / s), 0, ys.size - 1))
        return np.array([xs[ix], ys[iy]])

    def _valid_hook(self, p):
        """Inside the world box and not on a peg that already has a hook. Hooks don't collide: bodies are fine."""
        half = EnvConfig.peg_spacing / 2
        return self._in_box(p) and not any(np.hypot(*(hook.pos - p)) < half for hook in self.world.hooks)

    def _peg_px(self, p):
        """The surface pixel a peg (or hook) at world position p lands on (the overlay uses the same rounding)."""
        return int(math.floor(p[0] * self.zoom)), int(math.floor(p[1] * self.zoom))

    def _peg_overlay(self):
        """Translucent surface with one pixel per peg. Pre-rendered with a surfarray write and cached, so drawing it is
        one alpha blit; rebuilt only when the surface size, zoom, peg spacing or world box changes."""
        key = (self.W, self.H, self.zoom, EnvConfig.peg_spacing, self.inset, self.world_w, self.world_h)
        if key != self._peg_key:
            layer = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            layer.fill((*PEG_COL, 0))
            xs, ys = self._pegs()
            px, py = np.unique(np.floor(xs * self.zoom).astype(int)), np.unique(np.floor(ys * self.zoom).astype(int))
            px, py = px[(px >= 0) & (px < self.W)], py[(py >= 0) & (py < self.H)]
            alpha = pygame.surfarray.pixels_alpha(layer)  # indexed [x, y]
            alpha[np.ix_(px, py)] = PEG_ALPHA
            del alpha  # unlocks the surface
            self._peg_layer, self._peg_key = layer, key
        return self._peg_layer

    def _valid_now(self):
        """Is the object as placed so far valid?"""
        if self.mode == "rope":
            return self._valid_rope(self.points, self.snaps)
        if self.mode == "hook":
            return all(self._valid_hook(p) for p in self.points)
        return self._valid(self.points, self._complete())

    def _regular_points(self, start, end):
        n = self._cfg()["n"]
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        dist = np.hypot(*(end - start))
        if n == 2:  # a segment: press and release are its endpoints
            if dist < 4:
                return [start - (20.0, 0.0), start + (20.0, 0.0)]
            return [start, end]
        if dist < 4:  # plain click: default size, flat bottom edge
            radius, angle = 20.0, -math.pi / 2 + math.pi / n
        else:
            radius, angle = dist, math.atan2(*(end - start)[::-1])
        return list(polygon_points(start[0], start[1], radius, n, angle).T)

    def _near_point(self, wp):
        for i, p in enumerate(self.points):
            if np.hypot(*(np.asarray(p) - wp)) <= 6.0:
                return i
        return None

    # ---------------- Placement input (world coordinates)
    def _place_down(self, wp):
        i = self._near_point(wp)
        if i is not None:
            self.drag_idx, self.drag_from = i, np.array(self.points[i])
            return
        if self._complete():
            return
        if self.mode == "rope":
            pos, snap = self._rope_end(wp)
            if self._valid_rope(self.points + [pos], self.snaps + [snap]):
                self.points.append(pos)
                self.snaps.append(snap)
                self._push(("add", pos, snap))
            else:
                self._say(BAD_SPOT)
        elif self.mode == "hook":
            spot = self._hook_spot(wp)
            if spot is not None and self._valid_hook(spot):
                self.points.append(spot)
                self.snaps.append(None)
                self._push(("add", spot, None))
            else:
                self._say(BAD_SPOT)
        elif self.mode == "custom":
            candidate = self.points + [wp]
            if self._valid(candidate, len(candidate) == self._cfg()["n"]):
                self.points.append(wp)
                self.snaps.append(None)
                self._push(("add", wp, None))
            else:
                self._say(BAD_SPOT)
        else:
            self.reg_start = wp

    def _place_motion(self, wp):
        if self.drag_idx is not None:
            if self.mode == "hook":  # a dragged hook jumps from peg to peg
                spot = self._hook_spot(wp)
                if spot is not None:
                    self.points[self.drag_idx] = spot
            else:
                self.points[self.drag_idx] = wp

    def _place_up(self, wp):
        if self.drag_idx is not None:
            i, old, old_snap = self.drag_idx, self.drag_from, self.snaps[self.drag_idx]
            self.drag_idx = self.drag_from = None
            if self.mode == "rope":
                new, snap = self._rope_end(wp)
            elif self.mode == "hook":  # off the box there's no peg: keep wp so validation rejects it
                spot = self._hook_spot(wp)
                new, snap = (spot if spot is not None else np.asarray(wp, dtype=float)), None
            else:
                new, snap = wp, None
            self.points[i], self.snaps[i] = new, snap
            if np.hypot(*((new if self.mode == "hook" else wp) - old)) < 0.5:  # a hook on its own peg didn't move
                self.points[i], self.snaps[i] = old, old_snap
            elif self._valid_now():
                self._push(("move", i, old, new, old_snap, snap))
            else:
                self.points[i], self.snaps[i] = old, old_snap
                self._say(BAD_SPOT)
        elif self.reg_start is not None:
            pts = self._regular_points(self.reg_start, wp)
            self.reg_start = None
            if self._valid(pts, True):
                self.points, self.snaps = pts, [None] * len(pts)
                self._push(("poly", list(pts)))
            else:
                self._say(BAD_SPOT)

    # ---------------- Layout (UI surface pixels; rebuilt each call, shared by drawing and click handling)
    def _area(self):
        """The part of the UI surface left of the card column (all of it while the HUD is hidden)."""
        reserved = ui.CARD_WIDTH + 2 * ui.CARD_MARGIN if self.show_card else 0
        return pygame.Rect(0, 0, self.W - reserved, self.H)

    def _centered(self, w, h):
        area = self._area()
        return pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), max(ui.CARD_MARGIN, area.centery - h // 2), w, h)

    def _card_layout(self):
        pad = ui.CARD_PAD
        top = ui.CARD_MARGIN
        buttons_y = top + pad + CARD_TITLE_H
        bw = (ui.CARD_WIDTH - 2 * pad - BTN_GAP) // 2
        lines_y = buttons_y + CARD_BTN_H + pad
        if self._placing():  # Undo / Finish / Redo row under the pause / edit buttons
            lines_y += CARD_BTN_H + BTN_GAP
        rect = ui.card_rect(self.W, top, lines_y + len(self._hotkeys()) * LINE_H + pad - 1 - top)
        pause = pygame.Rect(rect.x + pad, buttons_y, bw, CARD_BTN_H)
        plus = pygame.Rect(pause.right + BTN_GAP, buttons_y, bw, CARD_BTN_H)
        return rect, pause, plus, lines_y

    def _placing(self):
        return self.active and self.stage == "place"

    def _hotkeys(self):
        """(key, what it does, current state or None, usable now) for every hotkey, as listed on the info card."""
        placing = self._placing()
        v = self.view
        grid = None if v is None else ("on" if v.show_grid else "off")
        look = None if v is None else ("filled" if v.filled else "points")
        return [("P", "pause", "paused" if self.paused else "running", True),
                ("G", "grid", grid, True),
                ("V", "view", look, True),
                ("SPACE", "HUD", "shown", True),
                ("Z", "undo", None, placing and bool(self.undo_stack)),
                ("X", "redo", None, placing and bool(self.redo_stack)),
                ("ESC", "back" if self.active else "quit", None, True),
                ("drag", "grab bodies", None, not self.paused)]

    def _menu_layout(self):
        n = len(KIND_TITLES)
        room = self._area().w - 2 * ui.CARD_MARGIN  # narrow surfaces shrink the tiles so the row stays clear of the card
        bw, bh = min(100, (room - (n + 1) * PAD) // n), 104
        panel = self._centered(n * (bw + PAD) + PAD, HEADER_H + bh + PAD)
        kinds = [(pygame.Rect(panel.x + PAD + i * (bw + PAD), panel.y + HEADER_H, bw, bh), kind)
                 for i, kind in enumerate(KIND_TITLES)]
        return panel, kinds

    def _config_layout(self):
        """Rects for the settings panel (built at the origin, then moved to the centered panel)."""
        cfg = self._cfg()
        lay = {"rows": []}
        x0, y = PAD, HEADER_H

        if self.kind == "rope":
            lay["rows"].append(("Stiffness", (x0, y)))
            y += LABEL_H
            lay["slider"] = pygame.Rect(x0 + 4, y + 4, 150, 6)
            y += 14 + ROW_GAP
            lay["rows"].append(("Width", (x0, y)))
            y += LABEL_H
            lay["widths"] = [(pygame.Rect(x0 + i * (30 + BTN_GAP), y, 30, BTN_H), w) for i, w in enumerate(ROPE_WIDTHS)]
            y += BTN_H + ROW_GAP
        elif self.kind == "hook":
            pass  # just a color: a hook is a single fixed point
        else:
            counts = POINT_COUNTS[self.kind]
            cols = 4 if len(counts) == 8 else 3
            lay["rows"].append(("Points", (x0, y)))
            y += LABEL_H
            lay["counts"] = [(pygame.Rect(x0 + (i % cols) * (30 + BTN_GAP), y + (i // cols) * (BTN_H + BTN_GAP), 30, BTN_H), n)
                             for i, n in enumerate(counts)]
            y += 2 * BTN_H + BTN_GAP + ROW_GAP

        if self.kind == "soft":
            options = range(max_skip_sets(cfg["n"]) + 1)
            bw = min(30, (COL_W - (len(options) - 1) * BTN_GAP) // len(options))
            lay["rows"].append(("Spring sets", (x0, y)))
            y += LABEL_H
            lay["sets"] = [(pygame.Rect(x0 + i * (bw + BTN_GAP), y, bw, BTN_H), s) for i, s in enumerate(options)]
            y += BTN_H + ROW_GAP
            lay["rows"].append(("Spring strength", (x0, y)))
            y += LABEL_H
            lay["slider"] = pygame.Rect(x0 + 4, y + 4, 150, 6)
            y += 14 + ROW_GAP

        sw = 14
        per_row = max(1, (COL_W + BTN_GAP) // (sw + BTN_GAP))
        lay["rows"].append(("Color", (x0, y)))
        y += LABEL_H
        lay["swatches"] = [(pygame.Rect(x0 + 2 + (i % per_row) * (sw + BTN_GAP), y + 2 + (i // per_row) * (sw + BTN_GAP), sw, sw), name)
                           for i, (name, _) in enumerate(self.colors)]
        y += 4 + ((len(self.colors) - 1) // per_row + 1) * (sw + BTN_GAP) - BTN_GAP + ROW_GAP + 2

        bw2 = (CONFIG_W - 3 * PAD) // 2
        if self.kind in ("rope", "hook"):
            lay["modes"] = [(pygame.Rect(PAD, y, CONFIG_W - 2 * PAD, MODE_H), self.kind, f"Place {self.kind}")]
        else:
            lay["modes"] = [(pygame.Rect(PAD, y, bw2, MODE_H), "regular", "Regular polygon"),
                            (pygame.Rect(2 * PAD + bw2, y, bw2, MODE_H), "custom", "Draw points")]
        px0 = PAD + COL_W + PAD
        lay["preview"] = pygame.Rect(px0, HEADER_H, CONFIG_W - PAD - px0, y - ROW_GAP - HEADER_H)
        panel = self._centered(CONFIG_W, y + MODE_H + PAD)

        dx, dy = panel.topleft
        lay["panel"] = panel
        lay["rows"] = [(label, (x + dx, ly + dy)) for label, (x, ly) in lay["rows"]]
        for key in ("counts", "sets", "widths", "swatches", "modes"):
            if key in lay:
                lay[key] = [(r.move(dx, dy),) + tuple(rest) for r, *rest in lay[key]]
        for key in ("slider", "preview"):
            if key in lay:
                lay[key] = lay[key].move(dx, dy)
        return lay

    def _header_buttons(self, panel, back=True):
        ws = [(pygame.Rect(panel.right - 6 - BTN_H, panel.y + 6, BTN_H, BTN_H), self.close, ("button", "x", False, False))]
        if back:
            ws.append((pygame.Rect(panel.x + 6, panel.y + 6, 44, BTN_H), self.back, ("button", "< Back", False, False)))
        return ws

    def _place_buttons(self):
        """Undo / Finish / Redo: a row in the info card, or under the hint box while the HUD is hidden."""
        if self.show_card:
            pause = self._card_layout()[1]
            x0, y, total = pause.x, pause.bottom + BTN_GAP, ui.CARD_WIDTH - 2 * ui.CARD_PAD
        else:
            box = self._hint_box()
            total = 3 * 60 + 2 * BTN_GAP
            x0, y = box.centerx - total // 2, box.bottom + BTN_GAP
        bw = (total - 2 * BTN_GAP) // 3
        rects = [pygame.Rect(x0 + i * (bw + BTN_GAP), y, bw if i < 2 else total - 2 * (bw + BTN_GAP), CARD_BTN_H) for i in range(3)]
        return [(rects[0], self.undo, ("button", "Undo", False, not self.undo_stack)),
                (rects[1], self.finish, ("button", "Finish", self._complete(), False)),
                (rects[2], self.redo, ("button", "Redo", False, not self.redo_stack))]

    def _card_widgets(self):
        if not self.show_card:
            return []
        rect, pause, plus, _ = self._card_layout()
        can_edit = self.paused and not self.active
        ws = [(pause, self.toggle_pause, ("pause",)), (plus, self.open if can_edit else None, ("plus", not can_edit))]
        if self._placing():
            ws += self._place_buttons()
        return ws + [(rect, None, ("blocker",))]  # the rest of the card swallows clicks (no grabs / points under it)

    def _panel_widgets(self):
        """(rect, action or None, draw spec) for the editor's panels and buttons (not the info card)."""
        if not self.active:
            return []
        ws = []
        if self.stage == "menu":
            panel, kinds = self._menu_layout()
            ws += [(rect, lambda k=kind: self.choose_kind(k), ("kind", kind)) for rect, kind in kinds]
            ws += self._header_buttons(panel, back=False)
        elif self.stage == "config":
            lay, cfg = self._config_layout(), self._cfg()
            ws += self._header_buttons(lay["panel"])
            for rect, w in lay.get("widths", []):
                ws.append((rect, lambda w=w: cfg.__setitem__("width", w), ("button", f"{w}px", cfg["width"] == w, False)))
            for rect, n in lay.get("counts", []):
                ws.append((rect, lambda n=n: self._set_count(n), ("button", str(n), cfg["n"] == n, False)))
            for rect, s in lay.get("sets", []):
                ws.append((rect, lambda s=s: cfg.__setitem__("sets", s), ("button", str(s), cfg["sets"] == s, False)))
            for rect, name in lay["swatches"]:
                ws.append((rect, lambda c=name: cfg.__setitem__("color", c), ("swatch", name, cfg["color"] == name)))
            for rect, mode, label in lay["modes"]:
                ws.append((rect, lambda m=mode: self.start_place(m), ("button", label, False, False)))
        elif self.stage == "place" and not self.show_card:
            ws += self._place_buttons()
        return ws

    def _widgets(self):
        """Everything clickable right now, in hit-test order (the info card first)."""
        return self._card_widgets() + self._panel_widgets()

    def _over_ui(self, pos):
        return any(rect.collidepoint(pos) for rect, _, _ in self._widgets())

    def _set_count(self, n):
        cfg = self._cfg()
        cfg["n"] = n
        if "sets" in cfg:
            cfg["sets"] = min(cfg["sets"], max_skip_sets(n))

    def _set_strength(self, x):
        rect = self._config_layout()["slider"]
        frac = min(max((x - rect.x) / rect.w, 0.0), 1.0)
        lo, hi = STRENGTH_RANGE
        self._cfg()[self._slider_key()] = round(lo + frac * (hi - lo), 1)

    def _slider_key(self):
        return "stiffness" if self.kind == "rope" else "strength"

    # ---------------- Events (mouse positions arrive in real screen pixels)
    def handle_event(self, event):
        """Returns True if the editor used the event (the main loop should then ignore it)."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.toggle_pause()
                return True
            if self.active:
                cmd = event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)
                if event.key == pygame.K_ESCAPE:
                    self.back()
                    return True
                if event.key == pygame.K_z:  # Z undo (Ctrl/Cmd+Shift+Z also redoes)
                    self.redo() if cmd and event.mod & pygame.KMOD_SHIFT else self.undo()
                    return True
                if event.key == pygame.K_x or (cmd and event.key == pygame.K_y):  # X redo (Ctrl/Cmd+Y too)
                    self.redo()
                    return True
            return False

        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return False
        pos = self.to_ui(event.pos)
        wp = self.to_world(pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button != 1:
                return self.paused
            for rect, action, _ in self._widgets():
                if rect.collidepoint(pos):
                    if action is not None:
                        action()
                    return True
            if self.active and self.stage == "config" and self.kind in ("soft", "rope"):
                if self._config_layout()["slider"].inflate(12, 14).collidepoint(pos):
                    self.slider_drag = True
                    self._set_strength(pos[0])
            elif self.active and self.stage == "place":
                self._place_down(wp)
            return self.paused
        if event.type == pygame.MOUSEMOTION:
            if self.slider_drag:
                self._set_strength(pos[0])
            elif self.active and self.stage == "place":
                self._place_motion(wp)
            return self.paused
        # MOUSEBUTTONUP
        self.slider_drag = False
        if self.active and self.stage == "place" and event.button == 1:
            self._place_up(wp)
        return self.paused

    # ---------------- Drawing: world-space preview (scaled-down game surfaces)
    def draw_world(self, surface, ghost_surface, mouse_ui):
        """Preview of the object being placed: grey ghost on the translucent layer, placed points on top.
        Before anything is placed the ghost follows the mouse (red where a click would be rejected)."""
        if not (self.active and self.stage == "place"):
            return
        wp = self.to_world(mouse_ui)
        z = self.zoom
        n, complete = self._cfg()["n"], self._complete()
        pts, valid = list(self.points), True
        if self.mode == "rope":
            self._draw_rope_preview(surface, ghost_surface, wp, self._over_ui(mouse_ui))
            return
        if self.mode == "hook":
            self._draw_hook_preview(surface, ghost_surface, wp, self._over_ui(mouse_ui))
            return
        if self._over_ui(mouse_ui) or self.drag_idx is not None or complete:
            pass  # no hover ghost while aiming at a button, dragging a point, or once the object is done
        elif self.mode == "custom":
            pts = pts + [wp]
            valid = self._valid(pts, len(pts) == n)
        elif self.mode == "regular":
            pts = self._regular_points(self.reg_start if self.reg_start is not None else wp, wp)
            valid = self._valid(pts, True)
        ghost = (200, 200, 200) if valid else tuple(self.palette.debug_red)
        if len(pts) >= 3:
            pygame.draw.polygon(ghost_surface, ghost, [tuple(np.asarray(p) * z) for p in pts])
        if len(pts) >= 2:
            pygame.draw.lines(surface, ghost, len(pts) >= 3, [tuple(np.asarray(p) * z) for p in pts], 1)  # opaque outline
        hover = self._near_point(wp)
        for i, p in enumerate(self.points):
            col = self.palette.white if i in (hover, self.drag_idx) else self.palette.greyL
            pygame.draw.circle(surface, col, tuple(np.asarray(p) * z), 2)

    def _draw_rope_preview(self, surface, ghost_surface, wp, over_ui):
        """Ghost line between the ends (the mouse is the next end); white ring on the body point an end would snap to."""
        z, pal = self.zoom, self.palette
        pts, snaps = list(self.points), list(self.snaps)
        target = None
        if not over_ui:
            if self.drag_idx is not None:
                pos, target = self._rope_end(wp)
                pts[self.drag_idx], snaps[self.drag_idx] = pos, target
            elif not self._complete() and self._near_point(wp) is None:
                pos, target = self._rope_end(wp)
                pts, snaps = pts + [pos], snaps + [target]
        valid = self._valid_rope(pts, snaps)
        if len(pts) == 2:
            ghost = (200, 200, 200) if valid else tuple(pal.debug_red)
            pygame.draw.line(ghost_surface, ghost, tuple(pts[0] * z), tuple(pts[1] * z), max(1, self._cfg()["width"]))
        if target is not None:
            ring = target.radius + 2 if isinstance(target, Hook) else 4
            pygame.draw.circle(surface, pal.white, tuple(self._snap_pos(target) * z), ring, 1)
        hover = self._near_point(wp)
        for i, (p, snap) in enumerate(zip(self.points, self.snaps)):
            col = pal.white if i in (hover, self.drag_idx) else pal.greyL
            if snap is None:  # free anchor: a small square
                pygame.draw.rect(surface, col, (int(p[0] * z) - 2, int(p[1] * z) - 2, 5, 5))
            else:
                pygame.draw.circle(surface, col, tuple(np.asarray(p) * z), 3, 1)

    def _draw_hook_preview(self, surface, ghost_surface, wp, over_ui):
        """Peg grid over the whole scene, the peg a click would use (white ring, ghost hook; red if taken), and the
        placed hook. The overlay goes on the world surface after the bodies, so it's under the ghost layer and UI."""
        surface.blit(self._peg_overlay(), (0, 0))
        z, pal = self.zoom, self.palette
        spot = None
        if not over_ui:
            if self.drag_idx is not None:
                spot = self.points[self.drag_idx]  # already snapped to a peg while dragging
            elif not self._complete() and self._near_point(wp) is None:
                spot = self._hook_spot(wp)
        if spot is not None:
            c = self._peg_px(spot)
            ghost = (200, 200, 200) if self._valid_hook(spot) else tuple(pal.debug_red)
            pygame.draw.circle(ghost_surface, ghost, c, 4)
            pygame.draw.circle(surface, pal.white, c, 6, 1)
        hover = self._near_point(wp)
        col = self._color(self._cfg()["color"])
        for i, p in enumerate(self.points):
            Hook(p, col=col).draw(surface, z, filled=True)
            if i in (hover, self.drag_idx):
                pygame.draw.circle(surface, pal.white, self._peg_px(p), 6, 1)
        if spot is not None:
            surface.set_at(self._peg_px(spot), pal.white)  # the target peg itself, on top

    # ---------------- Drawing: UI surface
    def draw_info_card(self, surface, mouse_ui):
        """The first card of the top-right column: pause/play and editor buttons plus every hotkey and its state.
        Returns the card's rect; the next card goes at rect.bottom + ui_style.CARD_GAP."""
        rect, _, _, lines_y = self._card_layout()
        ui.draw_frame(surface, rect)
        pad = ui.CARD_PAD
        ui.text(surface, "Controls", (rect.x + pad, rect.y + pad - 1), ui.WHITE, "bold", ui.TITLE_SIZE)
        for area, _, spec in self._card_widgets():
            if spec[0] != "blocker":
                self._draw_widget(surface, area, spec, area.collidepoint(mouse_ui))
        for i, (key, label, state, usable) in enumerate(self._hotkeys()):
            y = lines_y + i * LINE_H + LINE_H // 2
            dim = ui.TEXT_DIM if not usable else None
            ui.text(surface, key, (rect.x + pad, y), dim or ui.WHITE, "bold", valign="middle")
            ui.text(surface, label, (rect.x + pad + KEY_COL_W, y), dim or ui.TEXT, valign="middle")
            if state is not None:
                ui.text(surface, state, (rect.right - pad, y), dim or ui.ACCENT, "bold", align="right", valign="middle")
        return rect

    def draw_ui(self, surface, mouse_ui):
        """Panels, buttons and hints (not the info card or the message). Call before the HUD cards."""
        if self.active and self.stage in ("menu", "config"):
            surface.blit(self._dim, (0, 0))
            panel = self._menu_layout()[0] if self.stage == "menu" else self._config_layout()["panel"]
            ui.draw_frame(surface, panel, ui.PANEL_BG)
            title = "Spawn an object" if self.stage == "menu" else KIND_TITLES[self.kind]
            ui.text(surface, title, (panel.centerx, panel.y + 6 + BTN_H // 2), ui.WHITE, "bold", ui.TITLE_SIZE, "center", "middle")
            if self.stage == "config":
                self._draw_config_extras(surface)
        elif self.active and self.stage == "place":
            self._draw_hint(surface)
        for area, _, spec in self._panel_widgets():
            self._draw_widget(surface, area, spec, area.collidepoint(mouse_ui))

    def draw_message(self, surface):
        """The current message, centered on the surface in large bold debug_red. Call last (over the cards)."""
        if not (self.message and time.time() < self.message_until):
            return
        size = next((s for s in range(MESSAGE_SIZE, ui.TEXT_SIZE, -1) if ui.font("bold", s).size(self.message)[0] <= self.W - 16),
                    ui.TEXT_SIZE)
        cx, cy = self.W // 2, self.H // 2
        ui.text(surface, self.message, (cx + 1, cy + 1), ui.BLACK, "bold", size, "center", "middle")
        ui.text(surface, self.message, (cx, cy), tuple(self.palette.debug_red), "bold", size, "center", "middle")

    def _hint_lines(self):
        n = self._cfg()["n"]
        first = {"custom": "Click to place points in order",
                 "regular": "Press and drag to size and rotate, or click for the default size",
                 "rope": "Click both ends: on a body point or hook to attach, elsewhere to anchor",
                 "hook": "Click a peg to place the hook (it doesn't collide)"}[self.mode]
        what = {"rope": "an end", "hook": "the hook"}.get(self.mode, "a placed point")
        return [f"{first}  ({len(self.points)}/{n})", f"drag {what} to move it  ·  Z undo  ·  X redo  ·  Esc back"]

    def _hint_box(self):
        w = max(ui.font().size(line)[0] for line in self._hint_lines()) + 2 * ui.CARD_PAD
        area = self._area()
        return pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), ui.CARD_MARGIN, w, 2 * LINE_H + 2 * ui.CARD_PAD - 2)

    def _draw_hint(self, surface):
        lines, box = self._hint_lines(), self._hint_box()
        ui.draw_frame(surface, box)
        for i, line in enumerate(lines):
            ui.text(surface, line, (box.centerx, box.y + ui.CARD_PAD - 1 + i * LINE_H + LINE_H // 2),
                    ui.TEXT if i == 0 else ui.TEXT_DIM, align="center", valign="middle")

    def _draw_widget(self, surface, area, spec, hovered):
        kind = spec[0]
        if kind == "swatch":
            name, selected = spec[1], spec[2]
            ui.draw_frame(surface, area, self._color(name))
            if selected or hovered:
                pygame.draw.rect(surface, ui.WHITE if selected else ui.TEXT_DIM, area.inflate(4, 4), 1)
        elif kind == "button":
            ui.draw_button(surface, area, spec[1], hovered, spec[2], spec[3])
        elif kind == "pause":
            ui.draw_button(surface, area, None, hovered)
            self._draw_pause_icon(surface, area.x + 16, area.centery)
            ui.text(surface, "[p]", (area.x + 28, area.centery), ui.WHITE, "bold", valign="middle")
        elif kind == "plus":
            disabled = spec[1]
            ui.draw_button(surface, area, None, hovered, disabled=disabled)
            col = ui.TEXT_DIM if disabled else ui.WHITE
            cx, cy = area.x + 16, area.centery
            pygame.draw.rect(surface, col, (cx - 4, cy - 1, 9, 2))
            pygame.draw.rect(surface, col, (cx - 1, cy - 4, 2, 9))
            ui.text(surface, "edit", (area.x + 28, area.centery), col, "bold", valign="middle")
        elif kind == "kind":
            ui.draw_button(surface, area, None, hovered)
            self._draw_kind_icon(surface, spec[1], area)
            ui.text(surface, KIND_TITLES[spec[1]], (area.centerx, area.bottom - 14), ui.WHITE, "bold", align="center", valign="middle")

    def _draw_pause_icon(self, surface, cx, cy):
        if self.paused:  # paused: show play
            pygame.draw.polygon(surface, ui.WHITE, [(cx - 3, cy - 5), (cx - 3, cy + 5), (cx + 3, cy)])
        else:
            pygame.draw.rect(surface, ui.WHITE, (cx - 4, cy - 4, 3, 9))
            pygame.draw.rect(surface, ui.WHITE, (cx + 1, cy - 4, 3, 9))

    @staticmethod
    def _shape(surface, fill, pts, springs=()):
        """A filled polygon with a black 2px border, like the filled body view; springs as 1px black lines."""
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        pygame.draw.polygon(surface, fill, pts)
        for i, j in springs:
            pygame.draw.line(surface, ui.BLACK, pts[i], pts[j], 1)
        pygame.draw.polygon(surface, ui.BLACK, pts, ui.BORDER)

    def _draw_kind_icon(self, surface, kind, area):
        cx, cy, pal = area.centerx, area.y + 42, self.palette
        if kind == "soft":
            pts = polygon_points(cx, cy, 24, 6, -math.pi / 2 + math.pi / 6).T
            self._shape(surface, pal.network_green, pts, [(i, j) for i, j in skip_springs(6, 1) if (j - i) % 6 not in (1, 5)])
        elif kind == "rigid":
            ui.draw_frame(surface, pygame.Rect(cx - 18, cy - 18, 36, 36), pal.debug_red)
        elif kind == "fixed":
            ui.draw_frame(surface, pygame.Rect(cx - 30, cy - 5, 60, 10), pal.grey)
        elif kind == "hook":
            self._draw_hook_icon(surface, cx, cy - 8, pal.greyL)
        else:
            self._draw_rope_icon(surface, cx, cy, 30, pal.light_brown, 2)

    def _draw_hook_icon(self, surface, cx, cy, col, step=7):
        """A hook on a patch of the peg grid with a short rope and a weight hanging from it."""
        for i in range(-3, 4):
            for j in range(-2, 3):
                surface.fill(ui.TEXT_DIM, (cx + i * step, cy + j * step, 1, 1))
        drop = 3 * step
        pygame.draw.line(surface, self.palette.light_brown, (cx, cy), (cx, cy + drop), 2)
        ui.draw_frame(surface, pygame.Rect(cx - 6, cy + drop, 12, 12), self.palette.cream)
        pygame.draw.circle(surface, ui.BLACK, (cx, cy), 6)
        pygame.draw.circle(surface, col, (cx, cy), 4)
        pygame.draw.circle(surface, ui.BLACK, (cx, cy), 1)

    @staticmethod
    def _draw_rope_icon(surface, cx, cy, half, col, width):
        """A slack rope hanging between two black anchor blocks."""
        sag = half * 0.45
        pts = [(cx - half + 2 * half * t / 12, cy - sag / 2 + sag * (1 - (2 * t / 12 - 1) ** 2)) for t in range(13)]
        pygame.draw.lines(surface, col, False, pts, width)
        for x in (cx - half, cx + half):
            pygame.draw.rect(surface, ui.BLACK, (x - 3, cy - sag / 2 - 3, 6, 6))

    def _draw_config_extras(self, surface):
        """Row labels, the strength slider, and a live preview of the configured shape."""
        lay, cfg = self._config_layout(), self._cfg()
        for label, pos in lay["rows"]:
            ui.text(surface, label, pos, ui.TEXT)
        if "slider" in lay:
            track = lay["slider"]
            lo, hi = STRENGTH_RANGE
            value = cfg[self._slider_key()]
            frac = (value - lo) / (hi - lo)
            pygame.draw.rect(surface, ui.BLACK, track)
            inner = track.inflate(-2, -2)
            pygame.draw.rect(surface, ui.BUTTON_DISABLED, inner)
            pygame.draw.rect(surface, ui.BUTTON_SELECTED, (inner.x, inner.y, int(inner.w * frac), inner.h))
            kx = track.x + int(track.w * frac)
            ui.draw_frame(surface, pygame.Rect(kx - 4, track.centery - 7, 8, 14), ui.WHITE)
            ui.text(surface, f"{value:.1f}", (track.right + 10, track.centery), ui.WHITE, "bold", valign="middle")

        box = lay["preview"]
        ui.draw_frame(surface, box)
        n, col = cfg["n"], self._color(cfg["color"])
        r = min(box.w, box.h) * 0.34
        if self.kind == "rope":
            self._draw_rope_icon(surface, box.centerx, box.centery, int(r), col, cfg["width"])
        elif self.kind == "hook":
            self._draw_hook_icon(surface, box.centerx, box.centery - 9, col, step=6)  # icon spans cy - 12 .. cy + 30
        elif n == 2:
            a, b = (box.centerx - r, box.centery), (box.centerx + r, box.centery)
            pygame.draw.line(surface, ui.BLACK, a, b, 2 + 2 * ui.BORDER)
            pygame.draw.line(surface, col, a, b, 2)
        else:
            pts = polygon_points(box.centerx, box.centery, r, n, -math.pi / 2 + math.pi / n).T
            springs = [(i, j) for i, j in skip_springs(n, cfg["sets"]) if (j - i) % n not in (1, n - 1)] if self.kind == "soft" else ()
            self._shape(surface, col, pts, springs)
