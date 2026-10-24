"""In-game editor: pause/play, a spawn menu with per-body settings, and point placement with validation.

Several objects can be spawned in one pause: Finish adds the object and goes back to the spawn menu, still paused;
P / the play button resumes. Colliding kinds can be placed free, inside an existing object or around existing objects
(picked by clicking them first); validation then follows a containment hierarchy of the bodies (see _parents).

The panels, buttons, hints and messages are drawn on the scaled-down UI surface in the shared ui_style (Endesga
colors, solid black 2px borders, non-antialiased text), so they have the same pixel look as the world. Their positions
are UI-surface pixels: mouse events arrive in real screen pixels and are divided by the scale-down factor; world units
are UI pixels / zoom. Placement previews are drawn in world space on the game surfaces.

The info card ("Controls") is the exception: it is drawn at native resolution on the HUD layer (hud_layer.py) with
ui_style.HUD metrics, so its layout and hit-testing are in real screen pixels.
"""
import math
import time

import numpy as np
import pygame

import ui_style as ui
from ui_style import HUD
from peg_layer import PegLayer
from gravity_dial import direction_name
from entities import (Chain, EnvConfig, FixedBody, Hook, RigidBody, Rope, SoftBody, chain_box_points, chain_counts,
                      chain_layout, draw_border, polygon_points)

KIND_TITLES = {"soft": "Soft body", "rigid": "Rigid body", "fixed": "Fixed body", "rope": "Rope", "hook": "Hook",
               "chain": "Chain"}
MENU_COLS = 3  # spawn menu tiles per row
POINT_COUNTS = {"soft": (3, 4, 5, 6, 7, 8, 12, 20), "rigid": (3, 4, 5, 6, 7, 8, 12, 20), "fixed": (2, 3, 4, 6, 8, 12)}
STRENGTH_RANGE = (0.1, 1.0)  # soft body spring strength and rope stiffness
ROPE_WIDTHS = (1, 2)
CHAIN_MAX = 50  # chain segments (ropes + boxes)
# what a chain starts and ends with: rope...box and box...rope are mirror images (click the ends the other way round),
# so only three combinations are offered
CHAIN_ENDS = (("rope", "rope", "Rope-rope"), ("rope", "box", "Rope-box"), ("box", "box", "Box-box"))
CHAIN_MIN_ROPE = 2 * EnvConfig.contact_radius + 2  # world units: shortest rope a placed chain may have between pieces
SNAP_UI_PX = 5  # a rope end clicked this close (UI px) to a body point or hook attaches to it
PICK_UI_PX = 4  # an object clicked this close (UI px) to its outline is picked (inside / around selection)
HOOK_COLOR = "greyL"  # hooks have no settings panel: a 3 px marker in the demo's colour
PEG_COL = (255, 255, 255)  # peg grid overlay shown while placing a hook (see peg_layer.py), drawn over the scene
PEG_ALPHA = 150
PEG_DOT = 1  # peg dot size in native screen px (1 or 3)
# placement relations: free (top level, like before), inside one container, or around one or more objects. Only
# colliding kinds have them (ropes and hooks never collide); a chain is open, so it can't go around anything, and
# neither can a 2-point fixed segment
RELATIONS = {"soft": ("free", "inside", "around"), "rigid": ("free", "inside", "around"),
             "fixed": ("free", "inside", "around"), "chain": ("free", "inside")}
RELATION_HINTS = {"free": "on its own", "inside": "inside one object", "around": "around objects"}
HINT_FADE = 64  # opacity of the hint box while the mouse is over it
MESSAGE_SECONDS = 1.6
BAD_SPOT = "you cannot spawn a point there"
INCOMPLETE = "complete the new object to resume the sim"
FINISH_FIRST = "complete the new object first"
DIFFERENT_CONTAINERS = "pick objects in the same container"
MESSAGE_SIZE = 20  # bold, shrunk to fit narrow surfaces

# layout, in UI-surface pixels (the info card uses the CARD_* / LINE_H / BTN_GAP / KEY_COL_W ones times HUD_SCALE)
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


def chain_range(start, end):
    """Allowed segment counts (lo, hi, step 2) for a chain with these ends: odd when both ends are the same kind
    (3 .. 49), even when they differ (2 .. 50)."""
    return (3, CHAIN_MAX - 1) if start == end else (2, CHAIN_MAX)


def fit_chain_count(n, start, end):
    """The allowed count nearest n for these ends (one more when the parity is off, unless that's past the max)."""
    lo, hi = chain_range(start, end)
    if (n - lo) % 2:
        n = n + 1 if n + 1 <= hi else n - 1
    return min(max(n, lo), hi)


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


def segs_dist(p, q, a, b):
    """(m, k) distances between the segments p -> q (2, m) and a -> b (2, k); 0 where they cross."""
    p, q, a, b = p[:, :, None], q[:, :, None], a[:, None, :], b[:, None, :]
    d = np.minimum.reduce([point_seg_dist(p, a, b), point_seg_dist(q, a, b), point_seg_dist(a, p, q), point_seg_dist(b, p, q)])
    crosses = (_cross(q - p, a - p) * _cross(q - p, b - p) < 0) & (_cross(b - a, p - a) * _cross(b - a, q - a) < 0)
    return np.where(crosses, 0.0, d)


def points_in_poly(pts, poly):
    """Even-odd rule for every column of pts (2, m) at once: (m,) bools."""
    x, y = pts[0][:, None], pts[1][:, None]
    px, py = poly
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    straddle = (py > y) != (qy > y)
    x_cross = px + (y - py) * (qx - px) / np.where(straddle, qy - py, 1.0)
    return np.count_nonzero(straddle & (x < x_cross), axis=1) % 2 == 1


def point_in_poly(pt, poly):
    """Even-odd rule."""
    x, y = pt
    px, py = poly
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    straddle = (py > y) != (qy > y)
    x_cross = px + (y - py) * (qx - px) / np.where(straddle, qy - py, 1.0)
    return np.count_nonzero(straddle & (x < x_cross)) % 2 == 1


def poly_area(poly):
    x, y = poly
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def closed(body):
    """Does the body have an interior (a polygon, not a 2-point segment)? Only those can contain things."""
    return body.edges.shape[1] >= 3


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
    def __init__(self, world, palette, ui_size, world_size, scale, inset, zoom=1.0, view=None, screen_size=None):
        """ui_size: the UI surface size (real screen / scale); scale: real screen px per UI px; zoom: UI px per world
        unit; view: an object with `show_grid`, `filled` and `trails` (the G / V / T states listed on the info card); screen_size:
        the real screen (window) size the info card is laid out on (default ui_size * scale)."""
        self.world, self.palette, self.scale, self.inset, self.zoom, self.view = world, palette, scale, inset, zoom, view
        self.W, self.H = ui_size
        self.screen_w, self.screen_h = screen_size or (self.W * scale, self.H * scale)
        self.world_w, self.world_h = world_size
        self.colors = [(name, tuple(col)) for name, col in vars(palette).items() if not name.startswith("_") and isinstance(col, list)]
        self.settings = {"soft": {"n": 6, "sets": 1, "strength": 0.1, "color": "network_green"},  # the hexagon
                         "rigid": {"n": 4, "color": "debug_red"},  # the square
                         "fixed": {"n": 4, "color": "grey"},
                         "rope": {"n": 2, "stiffness": 1.0, "width": 1, "color": "light_brown"},  # like the demo ropes
                         "hook": {"n": 1},  # one hook per Finish, like the other kinds; no settings panel
                         # two ends to place; segments = ropes + boxes, odd for rope...rope (see chain_range)
                         "chain": {"n": 2, "segments": 3, "start": "rope", "end": "rope", "color": "grey",
                                   "rope_color": "light_brown"}}
        self.paused = False
        self.active = False  # editor mode
        self.stage = None  # "menu" -> "config" -> ("select" ->) "place"; Finish goes back to "menu"
        self.kind = None
        self.mode = None  # "regular", "custom", "rope", "hook" or "chain"
        self.relation = "free"  # "free", "inside" or "around" (chosen on the settings panel, reset per kind)
        self.selection = []  # the container (inside) or the objects to enclose (around), picked in the "select" stage
        self._tree_key, self._tree = None, None  # cached containment hierarchy (see _parents)
        self.points = []  # placed points (world coords) of the object being built
        self.snaps = []  # per point: (body, point_index) a rope end is attached to, or None (free point / anchor)
        self.undo_stack, self.redo_stack = [], []
        self.drag_idx = self.drag_from = None
        self.reg_start = None  # press position while dragging out a regular polygon
        self.slider_drag = False
        self.message, self.message_until = None, 0.0
        self.show_card = True  # the main loop mirrors its HUD toggle (SPACE) here: a hidden card takes no clicks
        self.show_compute = True  # F / the card's [f] button: the main loop draws the compute card only while set
        self.compute_card = None  # its rect (real screen px), set by the main loop when drawn: clicks on it are swallowed
        self.gravity_card = None  # the gravity selector's rect (screen px) while shown, set by the main loop: swallowed too
        self._dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self._dim.fill((0, 0, 0, 140))
        # the peg grid overlay, at native screen resolution (drawn by the main loop over the composite)
        self.pegs = PegLayer((self.screen_w, self.screen_h), PEG_DOT)

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
    def toggle_compute(self):
        self.show_compute = not self.show_compute

    def toggle_pause(self):
        """P / the play button. Outside the editor it pauses and resumes. In the editor it resumes (closing it): a
        complete object that wasn't finished yet is spawned first, a half-placed one blocks resuming with a message."""
        if self.active:
            if self.stage == "place" and self.points:
                if not self._complete():
                    self._say(INCOMPLETE)
                    return
                self._spawn()
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
        self.relation, self.selection = "free", []
        self._reset_object()

    def back(self):
        """Esc / < Back: place -> select (inside / around) -> settings -> menu -> closed. A hook has no settings
        panel, so its placement goes straight back to the menu."""
        if self.stage == "place":
            self._reset_object()
            self.stage = "menu" if self.kind == "hook" else "select" if self.relation != "free" else "config"
        elif self.stage == "select":
            self.selection, self.stage = [], "config"
        elif self.stage == "config":
            self.stage = "menu"
        else:
            self.close()

    def _reset_object(self):
        self.points, self.snaps, self.undo_stack, self.redo_stack = [], [], [], []
        self.drag_idx = self.drag_from = self.reg_start = None

    def choose_kind(self, kind):
        self.kind, self.stage, self.relation, self.selection = kind, "config", "free", []
        if kind == "hook":  # nothing to set: straight to placement
            self.start_place("hook")

    def set_relation(self, relation):
        self.relation, self.selection = relation, []

    def start_place(self, mode):
        """Start placing in `mode`; with an inside / around relation, the object(s) are picked first."""
        self._reset_object()
        self.mode = mode
        self.stage = "place" if self.relation == "free" else "select"

    def done_selecting(self):
        if not self.selection:
            self._say("click an object first")
            return
        self._reset_object()
        self.stage = "place"

    def finish(self):
        """Finish: spawn the object and go back to the spawn menu, still paused (P / the play button resumes)."""
        if not self.points or not self._complete():
            self._say(FINISH_FIRST)
            return
        self._spawn()
        self._reset_object()
        self.kind, self.mode, self.stage, self.relation, self.selection = None, None, "menu", "free", []

    def _spawn(self):
        """Add the complete object being placed to the world."""
        cfg = self._cfg()
        if self.kind == "hook":
            self.world.add_hook(Hook(self.points[0], col=self._color(HOOK_COLOR)))
            return
        col = self._color(cfg["color"])
        if self.kind == "rope":  # rest length = the ends' current distance, so it starts exactly taut
            ends = [snap if snap is not None else tuple(p) for p, snap in zip(self.points, self.snaps)]
            self.world.add_rope(Rope(ends[0], ends[1], stiffness=cfg["stiffness"], col=col, width=cfg["width"]))
            return
        if self.kind == "chain":  # boxes and ropes laid out between the ends; every rope starts exactly taut
            ends = [snap if snap is not None else tuple(p) for p, snap in zip(self.points, self.snaps)]
            self.world.add_chain(Chain(ends[0], ends[1], cfg["segments"], cfg["start"], cfg["end"], box_col=col,
                                       rope_col=self._color(cfg["rope_color"])))
            return
        pts = np.array(self.points, dtype=float).T
        style = dict(point_col=col, stick_col=col, fill_col=col)
        if self.kind == "soft":
            body = SoftBody(pts, stiffness=cfg["strength"], skip_sets=cfg["sets"], point_radius=3, **style)
        elif self.kind == "rigid":
            body = RigidBody(pts, point_radius=3, **style)
        else:
            body = FixedBody(pts, point_radius=2, stick_width=2 if pts.shape[1] == 2 else 1, **style)
        index = None
        if self.relation == "around":  # drawn under what it encloses: just before the first of those bodies
            enclosed = self._target()[1]
            index = min(self.world.bodies.index(b) for b in enclosed)
        self.world.add(body, index)

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

    # ---------------- Containment hierarchy (inside / around), world units
    def _is_wall(self, body):
        """One of the world box's walls: a fixed segment lying on the box outline (never picked or enclosed)."""
        if not body.fixed or body.points.shape[1] != 2:
            return False
        (x0, x1), (y0, y1) = body.points
        lo, hx, hy = self.inset, self.world_w - self.inset, self.world_h - self.inset
        return any(abs(a - v) < 1e-6 and abs(b - v) < 1e-6 for a, b, v in ((x0, x1, lo), (x0, x1, hx), (y0, y1, lo), (y0, y1, hy)))

    def _parents(self):
        """{body: the innermost closed body it lies inside, or None at the top level}. Point-in-polygon on whole
        bodies: B is inside C when every point of B is inside C's polygon; the state is intersection-free, so the
        bodies containing B are nested and the innermost is the one with the smallest area. Cached until a body is
        added or a point moves (the editor works while paused)."""
        world = self.world
        key = (tuple(map(id, world.bodies)), world.P.tobytes() if world.bodies else b"")
        if key != self._tree_key:
            polys = [c for c in world.bodies if closed(c)]
            lo = {id(b): b.points.min(axis=1) for b in world.bodies}
            hi = {id(b): b.points.max(axis=1) for b in world.bodies}
            area = {id(c): poly_area(c.points) for c in polys}
            tree = {}
            for b in world.bodies:
                best = None
                for c in polys:
                    if c is b or (lo[id(b)] < lo[id(c)]).any() or (hi[id(b)] > hi[id(c)]).any():
                        continue  # C's bounding box doesn't hold B's
                    if points_in_poly(b.points, c.points).all() and (best is None or area[id(c)] < area[id(best)]):
                        best = c
                tree[b] = best
            self._tree_key, self._tree = key, tree
        return self._tree

    def _ancestors(self, body):
        """body, its container, that one's container, ... (empty for None)."""
        if body is None:
            return []
        parents, out = self._parents(), []
        while body is not None:
            out.append(body)
            body = parents.get(body)
        return out

    def _with_contents(self, bodies):
        """The bodies plus everything nested inside them (at any depth)."""
        chosen = set(bodies)
        return {b for b in self.world.bodies if chosen.intersection(self._ancestors(b))}

    def _target(self):
        """(where the new object goes: its container or None for the top level, the bodies it must enclose)."""
        if self.relation == "inside" and self.selection:
            return self.selection[0], frozenset()
        if self.relation == "around" and self.selection:
            return self._parents()[self.selection[0]], frozenset(self._with_contents(self.selection))
        return None, frozenset()

    def _region_ok(self, new):
        """Are all the points (2, k) in the new object's region: inside its container and that one's containers, and
        inside no other body (a sibling, or anything else)? With no container that's simply "inside no body"."""
        must = set(self._ancestors(self._target()[0]))
        lo, hi = new.min(axis=1), new.max(axis=1)
        for c in self.world.bodies:
            if not closed(c):
                continue
            if c not in must and ((c.points.max(axis=1) < lo).any() or (c.points.min(axis=1) > hi).any()):
                continue  # bounding boxes apart: no point can be inside
            inside = points_in_poly(new, c.points)
            if (not inside.all()) if c in must else inside.any():
                return False
        return True

    def _encloses_ok(self, new):
        """The finished polygon (2, k) encloses exactly the bodies it must (the picked objects and their contents for
        "around", nothing otherwise) and cuts through none: an unpicked object that would end up inside is rejected,
        not added, so the result is always what was picked."""
        must = self._target()[1]
        lo, hi = new.min(axis=1), new.max(axis=1)
        for b in self.world.bodies:
            if b not in must and ((b.points.max(axis=1) < lo).any() or (b.points.min(axis=1) > hi).any()):
                continue
            inside = points_in_poly(b.points, new)
            if not (inside.all() if b in must else not inside.any()):
                return False
        return True

    def _pick(self, wp):
        """The object a click at wp picks for the current relation: the innermost one whose polygon holds wp, or one
        whose outline is within PICK_UI_PX (segments and chain boxes too). Walls, and for "inside" bodies without an
        interior, can't be picked. None if nothing is there."""
        reach = PICK_UI_PX / self.zoom
        best, best_area = None, math.inf
        p = np.asarray(wp, dtype=float).reshape(2, 1)
        for b in self.world.bodies:
            if self._is_wall(b) or (self.relation == "inside" and not closed(b)):
                continue
            a, c = b.points[:, b.edges[0]], b.points[:, b.edges[1]]
            hit = (closed(b) and point_in_poly(p[:, 0], b.points)) or point_seg_dist(p, a, c).min() <= reach
            area = poly_area(b.points) if closed(b) else 0.0
            if hit and area < best_area:
                best, best_area = b, area
        return best

    def _select(self, wp):
        """A click while picking: "inside" takes the container and moves on to placement; "around" toggles the object
        (picking a container of picked objects replaces them; they're enclosed with it anyway). Every picked object
        must be in the same container, or the new object couldn't reach around all of them without crossing it."""
        body = self._pick(wp)
        if body is None:
            self._say("click an object")
            return
        if self.relation == "inside":
            self.selection = [body]
            self.done_selecting()
            return
        if body in self.selection:
            self.selection.remove(body)
            return
        if any(s in self.selection for s in self._ancestors(body)[1:]):
            self._say("that is already inside a picked object")
            return
        keep = [s for s in self.selection if body not in self._ancestors(s)]
        parents = self._parents()
        if keep and parents[keep[0]] is not parents[body]:
            self._say(DIFFERENT_CONTAINERS)
            return
        self.selection = keep + [body]

    # ---------------- Placement validation
    def _valid(self, pts, complete):
        """Can these points (the object so far) exist without touching the world or themselves, in the region the
        relation puts them (top level / inside the container / around the picked objects)?"""
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
            if point_seg_dist(new[:, :, None], a[:, None, :], b[:, None, :]).min() < r:
                return False  # a point too close to an edge
        if not self._region_ok(new):
            return False

        edges = [(i, i + 1) for i in range(k - 1)]
        if complete and k >= 3:
            edges.append((k - 1, 0))
        if has_edges and edges:
            i, j = np.array(edges).T
            if segs_dist(new[:, i], new[:, j], a, b).min() < r:
                return False  # an edge too close to (or crossing) an existing edge

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
            if poly_area(new) < r * r:
                return False  # degenerate sliver
            if not self._encloses_ok(new):
                return False  # would swallow (or cut into) the wrong bodies
        elif self._target()[1]:
            return not complete  # "around" needs a closed polygon
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

    # ---------------- Chains (world units)
    def _end_kind(self, i):
        """What end i (0 = start, 1 = end) of the chain being placed is: "rope" or "box"."""
        return self._cfg()["start" if i == 0 else "end"]

    def _end_at(self, i, wp):
        """(position, snap) for end i of a rope or chain at wp: a rope end snaps to a body point or hook, a chain's
        box end is just placed (its box is centred there)."""
        if self.mode == "chain" and self._end_kind(i) == "box":
            return np.asarray(wp, dtype=float), None
        return self._rope_end(wp)

    def _chain_boxes(self, pts):
        """The boxes ((2, 6) each) and the rope length of the chain over the ends so far. With one end placed only a
        box end has a box (axis-aligned until the other end sets the direction) and the rope length is None."""
        cfg = self._cfg()
        if len(pts) < 2:
            if pts and self._end_kind(0) == "box":
                return [chain_box_points(pts[0], (1.0, 0.0), Chain.BOX_SIZE)], None
            return [], None
        centers, u, rope = chain_layout(pts[0], pts[1], cfg["segments"], cfg["start"], cfg["end"], Chain.BOX_SIZE)
        return [chain_box_points(c, u, Chain.BOX_SIZE) for c in centers], rope

    def _boxes_clear(self, boxes):
        """The rigid placement clearance rules for a chain's boxes, vectorized: inside the world box, no box point or
        edge within contact_radius of an existing edge, no box point inside a body and no body point inside a box."""
        if not boxes:
            return True
        r = EnvConfig.contact_radius
        pts = np.concatenate(boxes, axis=1)
        lo, hi_x, hi_y = self.inset + r, self.world_w - self.inset - r, self.world_h - self.inset - r
        if (pts[0] < lo).any() or (pts[0] > hi_x).any() or (pts[1] < lo).any() or (pts[1] > hi_y).any():
            return False
        world = self.world
        if not world.bodies or world.edges.shape[1] == 0:
            return True
        a, b = world.P[:, world.edges[0]], world.P[:, world.edges[1]]
        ends = np.concatenate([np.roll(box, -1, axis=1) for box in boxes], axis=1)
        # only world edges whose bounding box comes within r of some box's bounding box can be that close
        box_lo = np.array([box.min(axis=1) for box in boxes]).T - r  # (2, n_boxes)
        box_hi = np.array([box.max(axis=1) for box in boxes]).T + r
        e_lo, e_hi = np.minimum(a, b), np.maximum(a, b)
        near = ((e_lo[:, :, None] <= box_hi[:, None, :]) & (e_hi[:, :, None] >= box_lo[:, None, :])).all(axis=0).any(axis=1)
        if near.any() and segs_dist(pts, ends, a[:, near], b[:, near]).min() < r:  # also covers the box points
            return False
        if not self._region_ok(pts):  # no box point inside a body (or outside the container, for "inside")
            return False
        for box in boxes:  # boxes are convex: a world point is inside when it's left of (or right of) every edge
            e, rel = np.roll(box, -1, axis=1) - box, world.P[:, None, :] - box[:, :, None]
            side = e[0][:, None] * rel[1] - e[1][:, None] * rel[0]
            if ((side > 0).all(axis=0) | (side < 0).all(axis=0)).any():
                return False
        return True

    def _valid_chain(self, pts, snaps):
        """Rope ends follow the rope rules (in the box, a free anchor not inside a body, not both on one point), every
        box (end boxes too) follows the rigid clearance rules, and once both ends are down the line is long enough
        for every rope to be at least CHAIN_MIN_ROPE. The ropes themselves may cross bodies."""
        for i, (p, snap) in enumerate(zip(pts, snaps)):
            if self._end_kind(i) == "rope" and not self._valid_rope([p], [snap]):
                return False
        if len(pts) == 2:
            (sa, sb) = snaps
            if sa is not None and sb is not None and self._same_snap(sa, sb):
                return False
        boxes, rope = self._chain_boxes(pts)
        if rope is not None and rope < CHAIN_MIN_ROPE:
            return False
        return self._boxes_clear(boxes)

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
        """The world-surface pixel a peg (or hook) at world position p lands on (for the target / hover rings)."""
        return int(math.floor(p[0] * self.zoom)), int(math.floor(p[1] * self.zoom))

    def peg_screen_px(self, coords):
        """Native screen pixel (column or row index) of each peg coordinate (world units): the pixel holding the
        point world * zoom * scale, the same mapping as everything else on screen, so a dot sits on its peg."""
        return np.floor(np.asarray(coords, dtype=float) * self.zoom * self.scale + 1e-6).astype(int)

    def peg_layer(self):
        """The native-res peg overlay (peg_layer.PegLayer) while a hook is being placed, else None. Its dots are at
        the very pegs _hook_spot snaps to (_pegs, mapped by peg_screen_px); it's rebuilt only when the screen size,
        zoom, peg spacing or world box changes (layer.version then changes, so the main loop re-uploads it)."""
        if not (self.active and self.stage == "place" and self.mode == "hook"):
            return None
        key = (self.screen_w, self.screen_h, self.zoom, self.scale, EnvConfig.peg_spacing, self.inset, self.world_w, self.world_h)
        self.pegs.update(key, lambda: tuple(self.peg_screen_px(c) for c in self._pegs()))
        return self.pegs

    def _valid_now(self):
        """Is the object as placed so far valid?"""
        if self.mode == "rope":
            return self._valid_rope(self.points, self.snaps)
        if self.mode == "chain":
            return self._valid_chain(self.points, self.snaps)
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
        if self.mode in ("rope", "chain"):
            pos, snap = self._end_at(len(self.points), wp)
            valid = self._valid_rope if self.mode == "rope" else self._valid_chain
            if valid(self.points + [pos], self.snaps + [snap]):
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
            if self.mode in ("rope", "chain"):
                new, snap = self._end_at(i, wp)
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
        """The part of the UI surface left of the card column (all of it while the HUD is hidden). The column is
        laid out in real screen px; its left margin's edge is converted to UI px (rounded down)."""
        if not self.show_card:
            return pygame.Rect(0, 0, self.W, self.H)
        column_left = self.screen_w - HUD.column_width()
        return pygame.Rect(0, 0, min(self.W, column_left // self.scale), self.H)

    def _centered(self, w, h):
        area = self._area()
        return pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), max(ui.CARD_MARGIN, area.centery - h // 2), w, h)

    def _card_layout(self):
        """The info card in real screen px (HUD style): (card, [p] button, edit button, [f] button, hotkey list y)."""
        px, pad = HUD.px, HUD.CARD_PAD
        gap, bh = px(BTN_GAP), px(CARD_BTN_H)
        top = HUD.CARD_MARGIN
        buttons_y = top + pad + px(CARD_TITLE_H)
        lines_y = buttons_y + bh + pad
        if self._row_shown():  # Undo / Finish / Redo (or Done) row under the pause / edit / compute buttons
            lines_y += bh + gap
        rect = HUD.card_rect(self.screen_w, top, lines_y + len(self._hotkeys()) * px(LINE_H) + pad - px(1) - top)
        bw = (rect.w - 2 * pad - 2 * gap) // 3
        pause = pygame.Rect(rect.x + pad, buttons_y, bw, bh)
        plus = pygame.Rect(pause.right + gap, buttons_y, bw, bh)
        compute = pygame.Rect(plus.right + gap, buttons_y, rect.right - pad - plus.right - gap, bh)
        return rect, pause, plus, compute, lines_y

    def _placing(self):
        return self.active and self.stage == "place"

    def _row_shown(self):
        """Is there a row of action buttons (Undo / Finish / Redo while placing, Done while picking objects to go
        around)?"""
        return self._placing() or (self.active and self.stage == "select" and self.relation == "around")

    def _relations(self):
        """The placement relations the current kind offers (none for ropes and hooks: they never collide)."""
        options = RELATIONS.get(self.kind, ())
        if self.kind == "fixed" and self._cfg()["n"] == 2:  # a segment has no inside to put anything in
            options = tuple(r for r in options if r != "around")
        return options

    def _hotkeys(self):
        """(key, what it does, current state or None, usable now) for every hotkey, as listed on the info card."""
        placing = self._placing()
        v = self.view
        grid = None if v is None else ("on" if v.show_grid else "off")
        look = None if v is None else ("filled" if v.filled else "points")
        trails = None if v is None else ("on" if getattr(v, "trails", False) else "off")
        return [("P", "pause", "paused" if self.paused else "running", True),
                ("G", "grid", grid, True),
                ("V", "view", look, True),
                ("T", "trails", trails, True),
                ("F", "compute", "shown" if self.show_compute else "hidden", True),
                ("SPACE", "HUD", "shown", True),
                ("KP1-9", "gravity", direction_name(compass=True), True),
                ("Z", "undo", None, placing and bool(self.undo_stack)),
                ("X", "redo", None, placing and bool(self.redo_stack)),
                ("ESC", "back" if self.active else "quit", None, True),
                ("drag", "grab bodies", None, not self.paused)]

    def _menu_layout(self):
        n, cols = len(KIND_TITLES), MENU_COLS
        rows = -(-n // cols)
        room = self._area().w - 2 * ui.CARD_MARGIN  # narrow surfaces shrink the tiles so the grid stays clear of the card
        bw = min(100, (room - (cols + 1) * PAD) // cols)
        bh = min(104, (self.H - 2 * ui.CARD_MARGIN - HEADER_H - rows * PAD) // rows)
        panel = self._centered(cols * (bw + PAD) + PAD, HEADER_H + rows * (bh + PAD))
        kinds = [(pygame.Rect(panel.x + PAD + (i % cols) * (bw + PAD), panel.y + HEADER_H + (i // cols) * (bh + PAD), bw, bh), kind)
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
        elif self.kind == "chain":
            lay["rows"].append(("Segments", (x0, y)))
            y += LABEL_H
            lay["slider"] = pygame.Rect(x0 + 4, y + 4, 150, 6)
            y += 14 + ROW_GAP
            bw = 64
            lay["rows"].append(("Ends", (x0, y)))
            y += LABEL_H
            lay["ends"] = [(pygame.Rect(x0 + i * (bw + BTN_GAP), y, bw, BTN_H), start, end, label)
                           for i, (start, end, label) in enumerate(CHAIN_ENDS)]
            y += BTN_H + ROW_GAP
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
            if len(options) > 1:  # a triangle has no springs past its sides: nothing to choose
                bw = min(30, (COL_W - (len(options) - 1) * BTN_GAP) // len(options))
                lay["rows"].append(("Spring sets", (x0, y)))
                y += LABEL_H
                lay["sets"] = [(pygame.Rect(x0 + i * (bw + BTN_GAP), y, bw, BTN_H), s) for i, s in enumerate(options)]
                y += BTN_H + ROW_GAP
            lay["rows"].append(("Spring strength", (x0, y)))
            y += LABEL_H
            lay["slider"] = pygame.Rect(x0 + 4, y + 4, 150, 6)
            y += 14 + ROW_GAP

        # color swatches: one grid per color setting (a chain has one for its boxes and one for its ropes); the
        # label row also names the hovered (or chosen) color, see _draw_config_extras
        sw = 14
        per_row = max(1, (COL_W + BTN_GAP) // (sw + BTN_GAP))
        grids = [("Box color", "color"), ("Rope color", "rope_color")] if self.kind == "chain" else [("Color", "color")]
        lay["swatches"], lay["color_rows"] = [], []
        for label, key in grids:
            lay["rows"].append((label, (x0, y)))
            lay["color_rows"].append(((x0 + COL_W, y), key))
            y += LABEL_H
            lay["swatches"] += [(pygame.Rect(x0 + 2 + (i % per_row) * (sw + BTN_GAP), y + 2 + (i // per_row) * (sw + BTN_GAP), sw, sw), name, key)
                                for i, (name, _) in enumerate(self.colors)]
            y += 4 + ((len(self.colors) - 1) // per_row + 1) * (sw + BTN_GAP) - BTN_GAP + ROW_GAP + 2

        bw2 = (CONFIG_W - 3 * PAD) // 2
        full = pygame.Rect(PAD, y, CONFIG_W - 2 * PAD, MODE_H)
        if self.kind in ("rope", "chain"):
            lay["modes"] = [(full, self.kind, f"Place {self.kind}")]
        elif self.kind == "fixed" and cfg["n"] == 2:  # a segment: click its two ends (no regular polygon to size)
            lay["modes"] = [(full, "custom", "Place segment")]
        else:
            lay["modes"] = [(pygame.Rect(PAD, y, bw2, MODE_H), "regular", "Regular polygon"),
                            (pygame.Rect(2 * PAD + bw2, y, bw2, MODE_H), "custom", "Draw points")]
        px0 = PAD + COL_W + PAD
        lay["preview"] = pygame.Rect(px0, HEADER_H, CONFIG_W - PAD - px0, y - ROW_GAP - HEADER_H)
        relations = self._relations()
        if relations:  # free / inside / around under the preview, with what the chosen one means above the buttons
            ry = y - ROW_GAP - BTN_H
            bw = (lay["preview"].w - 2 * BTN_GAP) // 3
            lay["relations"] = [(pygame.Rect(px0 + i * (bw + BTN_GAP), ry, bw, BTN_H), rel) for i, rel in enumerate(relations)]
            lay["relation_row"] = (px0, ry - LABEL_H)
            lay["preview"].h -= BTN_H + LABEL_H + ROW_GAP
        panel = self._centered(CONFIG_W, y + MODE_H + PAD)

        dx, dy = panel.topleft
        lay["panel"] = panel
        lay["rows"] = [(label, (x + dx, ly + dy)) for label, (x, ly) in lay["rows"]]
        lay["color_rows"] = [((x + dx, ly + dy), key) for (x, ly), key in lay["color_rows"]]
        if "relation_row" in lay:
            lay["relation_row"] = (lay["relation_row"][0] + dx, lay["relation_row"][1] + dy)
        for key in ("counts", "sets", "widths", "ends", "relations", "swatches", "modes"):
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
        """Undo / Finish / Redo while placing, or Done while picking objects to go around: a row in the info card
        (real screen px), or under the hint box while the HUD is hidden (UI px)."""
        if self.show_card:
            pause = self._card_layout()[1]
            gap, bh = HUD.px(BTN_GAP), HUD.px(CARD_BTN_H)
            x0, y, total = pause.x, pause.bottom + gap, HUD.CARD_WIDTH - 2 * HUD.CARD_PAD
        else:
            box = self._hint_box()
            gap, bh = BTN_GAP, CARD_BTN_H
            total = 3 * 60 + 2 * gap
            x0, y = box.centerx - total // 2, box.bottom + gap
        if self.stage == "select":
            label = f"Done ({len(self.selection)} picked)"
            return [(pygame.Rect(x0, y, total, bh), self.done_selecting, ("button", label, bool(self.selection), not self.selection))]
        bw = (total - 2 * gap) // 3
        rects = [pygame.Rect(x0 + i * (bw + gap), y, bw if i < 2 else total - 2 * (bw + gap), bh) for i in range(3)]
        return [(rects[0], self.undo, ("button", "Undo", False, not self.undo_stack)),
                (rects[1], self.finish, ("button", "Finish", self._complete(), False)),
                (rects[2], self.redo, ("button", "Redo", False, not self.redo_stack))]

    def _card_widgets(self):
        """(rect, action or None, draw spec) on the info card, in real screen px."""
        if not self.show_card:
            return []
        rect, pause, plus, compute, _ = self._card_layout()
        can_edit = self.paused and not self.active
        ws = [(pause, self.toggle_pause, ("pause",)), (plus, self.open if can_edit else None, ("plus", not can_edit)),
              (compute, self.toggle_compute, ("compute",))]
        if self._row_shown():
            ws += self._place_buttons()
        ws.append((rect, None, ("blocker",)))  # the rest of the card swallows clicks (no grabs / points under it)
        if self.show_compute and self.compute_card is not None:
            ws.append((self.compute_card, None, ("blocker",)))  # and so does the compute card
        if self.gravity_card is not None:
            ws.append((self.gravity_card, None, ("blocker",)))  # and the gravity selector (gravity_dial.py)
        return ws

    def _panel_widgets(self):
        """(rect, action or None, draw spec) for the editor's panels and buttons (not the info card), in UI px."""
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
            for rect, start, end, label in lay.get("ends", []):
                ws.append((rect, lambda a=start, b=end: self._set_chain_ends(a, b),
                           ("button", label, (cfg["start"], cfg["end"]) == (start, end), False)))
            for rect, rel in lay.get("relations", []):
                ws.append((rect, lambda r=rel: self.set_relation(r), ("button", rel.capitalize(), self.relation == rel, False)))
            for rect, name, key in lay["swatches"]:
                ws.append((rect, lambda c=name, k=key: cfg.__setitem__(k, c), ("swatch", name, cfg[key] == name)))
            for rect, mode, label in lay["modes"]:
                ws.append((rect, lambda m=mode: self.start_place(m), ("button", label, False, False)))
        elif self._row_shown() and not self.show_card:
            ws += self._place_buttons()
        return ws

    def _hit(self, screen_pos):
        """The clickable widget under a real screen position, (rect, action, spec), or None. The info card (screen px)
        comes first, then the panels (UI px)."""
        for widget in self._card_widgets():
            if widget[0].collidepoint(screen_pos):
                return widget
        ui_pos = self.to_ui(screen_pos)
        for widget in self._panel_widgets():
            if widget[0].collidepoint(ui_pos):
                return widget
        return None

    def _over_ui(self, ui_pos):
        return self._hit((ui_pos[0] * self.scale, ui_pos[1] * self.scale)) is not None

    def _set_count(self, n):
        cfg = self._cfg()
        cfg["n"] = n
        if "sets" in cfg:
            cfg["sets"] = min(cfg["sets"], max_skip_sets(n))
        if self.relation not in self._relations():  # e.g. "around" for a 2-point segment
            self.set_relation("free")

    def _set_chain_ends(self, start, end):
        """Pick what the chain starts and ends with; the segment count moves to the nearest count that fits."""
        cfg = self._cfg()
        cfg["start"], cfg["end"] = start, end
        cfg["segments"] = fit_chain_count(cfg["segments"], start, end)

    def _set_strength(self, x):
        """Slider drag at UI x: strength / stiffness in 0.1 steps, or a chain's segment count in steps of 2."""
        rect = self._config_layout()["slider"]
        frac = min(max((x - rect.x) / rect.w, 0.0), 1.0)
        key, lo, hi, step = self._slider_spec()
        if step == 2:
            self._cfg()[key] = lo + 2 * int(round(frac * (hi - lo) / 2))
        else:
            self._cfg()[key] = round(lo + frac * (hi - lo), 1)

    def _slider_spec(self):
        """(settings key, low, high, step) of the settings panel's slider."""
        if self.kind == "chain":
            cfg = self._cfg()
            return ("segments",) + chain_range(cfg["start"], cfg["end"]) + (2,)
        return ("stiffness" if self.kind == "rope" else "strength",) + STRENGTH_RANGE + (0.1,)

    # ---------------- Events (mouse positions arrive in real screen pixels)
    def handle_event(self, event):
        """Returns True if the editor used the event (the main loop should then ignore it)."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                self.toggle_pause()
                return True
            if event.key == pygame.K_f:
                self.toggle_compute()
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
            hit = self._hit(event.pos)
            if hit is not None:  # also swallows clicks on the card's body (no grabs / points under it)
                if hit[1] is not None:
                    hit[1]()
                return True
            if self.active and self.stage == "config":
                slider = self._config_layout().get("slider")
                if slider is not None and slider.inflate(12, 14).collidepoint(pos):
                    self.slider_drag = True
                    self._set_strength(pos[0])
            elif self.active and self.stage == "select":
                self._select(wp)
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
        Before anything is placed the ghost follows the mouse (red where a click would be rejected). While picking
        objects (and while placing inside / around them) the picked objects are outlined, and the one under the
        mouse too while picking."""
        if not (self.active and self.stage in ("select", "place")):
            return
        wp = self.to_world(mouse_ui)
        self._draw_picks(surface, wp, self._over_ui(mouse_ui))
        if self.stage == "select":
            return
        z = self.zoom
        n, complete = self._cfg()["n"], self._complete()
        pts, valid = list(self.points), True
        if self.mode in ("rope", "chain"):
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

    def _draw_picks(self, surface, wp, over_ui):
        """Outlines over the bodies' borders: yellow for the picked container / objects, white for the object a click
        would pick (or unpick)."""
        pal = self.palette
        for body in self.selection:
            self._outline(surface, body, pal.yellow)
        if self.stage == "select" and not over_ui:
            body = self._pick(wp)
            if body is not None:
                self._outline(surface, body, pal.white)

    def _outline(self, surface, body, col):
        if closed(body):  # exactly over the filled view's black border
            loop = np.round(body.points[:, body.edges[0]] * self.zoom).astype(int)
            draw_border(surface, loop, col, ui.BORDER, body.winding)
        else:
            a, b = body.points.T * self.zoom
            pygame.draw.line(surface, col, a, b, 3)

    def _draw_rope_preview(self, surface, ghost_surface, wp, over_ui):
        """Ghost line between the ends (the mouse is the next end); white ring on the body point an end would snap to.
        Chains (same ends) show every box and rope of the chain as it would be laid out."""
        z, pal = self.zoom, self.palette
        pts, snaps = list(self.points), list(self.snaps)
        target = None
        if not over_ui:
            if self.drag_idx is not None:
                pos, target = self._end_at(self.drag_idx, wp)
                pts[self.drag_idx], snaps[self.drag_idx] = pos, target
            elif not self._complete() and self._near_point(wp) is None:
                pos, target = self._end_at(len(pts), wp)
                pts, snaps = pts + [pos], snaps + [target]
        chain = self.mode == "chain"
        valid = self._valid_chain(pts, snaps) if chain else self._valid_rope(pts, snaps)
        ghost = (200, 200, 200) if valid else tuple(pal.debug_red)
        if chain:
            self._draw_chain_ghost(surface, ghost_surface, pts, ghost)
        elif len(pts) == 2:
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

    def _draw_chain_ghost(self, surface, ghost_surface, pts, ghost):
        """The chain over the ends `pts` on the translucent layer: ropes as lines, boxes as filled squares."""
        z = self.zoom
        boxes, _ = self._chain_boxes(pts)
        if len(pts) == 2:
            cfg = self._cfg()
            # consecutive pairs of joints are the ropes: each runs from the chain's start (or a box's front midpoint)
            # to the next box's back midpoint (or the chain's end); end boxes have no rope on their outer side
            joints = [p for box in boxes for p in (box[:, 0], box[:, 3])]
            joints = joints[1:] if cfg["start"] == "box" else [pts[0]] + joints
            joints = joints[:-1] if cfg["end"] == "box" else joints + [pts[1]]
            for a, b in zip(joints[0::2], joints[1::2]):
                pygame.draw.line(ghost_surface, ghost, tuple(np.asarray(a) * z), tuple(np.asarray(b) * z), 1)
        for box in boxes:
            pygame.draw.polygon(ghost_surface, ghost, (box * z).T.tolist())

    def _draw_hook_preview(self, surface, ghost_surface, wp, over_ui):
        """The peg a click would use (white ring, ghost hook; red if taken) and the placed hook. The peg grid itself
        is the native-res overlay from peg_layer(), which the main loop draws over the scene (under the HUD)."""
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
        col = self._color(HOOK_COLOR)
        for i, p in enumerate(self.points):
            Hook(p, col=col).draw(surface, z, filled=True)
            if i in (hover, self.drag_idx):
                pygame.draw.circle(surface, pal.white, self._peg_px(p), 6, 1)

    # ---------------- Drawing: UI surface
    def draw_info_card(self, surface, mouse_px, origin=(0, 0)):
        """The first card of the top-right column, at native resolution (ui_style.HUD): pause/play, editor and compute
        toggle buttons plus every hotkey and its state. mouse_px: the mouse in real screen px; origin: the surface's
        top-left in real screen px (the HUD layer's rect.topleft, or (0, 0) for the screen itself). Returns the card's
        rect on `surface`; the next card goes at rect.bottom + ui_style.HUD.CARD_GAP."""
        ox, oy = origin
        screen_rect, *_, lines_y = self._card_layout()
        rect, lines_y = screen_rect.move(-ox, -oy), lines_y - oy
        HUD.draw_frame(surface, rect)
        px, pad, line_h = HUD.px, HUD.CARD_PAD, HUD.px(LINE_H)
        HUD.text(surface, "Controls", (rect.x + pad, rect.y + pad - px(1)), ui.WHITE, "bold", HUD.TITLE_SIZE)
        for area, _, spec in self._card_widgets():
            if spec[0] != "blocker":
                self._draw_widget(surface, area.move(-ox, -oy), spec, area.collidepoint(mouse_px), HUD)
        for i, (key, label, state, usable) in enumerate(self._hotkeys()):
            y = lines_y + i * line_h + line_h // 2
            dim = ui.TEXT_DIM if not usable else None
            HUD.text(surface, key, (rect.x + pad, y), dim or ui.WHITE, "bold", valign="middle")
            HUD.text(surface, label, (rect.x + pad + px(KEY_COL_W), y), dim or ui.TEXT, valign="middle")
            if state is not None:
                HUD.text(surface, state, (rect.right - pad, y), dim or ui.ACCENT, "bold", align="right", valign="middle")
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
                self._draw_config_extras(surface, mouse_ui)
        elif self.active and self.stage in ("select", "place"):
            self._draw_hint(surface, mouse_ui)
        for area, _, spec in self._panel_widgets():
            self._draw_widget(surface, area, spec, area.collidepoint(mouse_ui))

    def draw_message(self, surface, origin=(0, 0)):
        """The current message, centered on the UI surface in large bold debug_red. Call last (over the cards; see
        HudLayer.overlay). origin: the target surface's top-left in UI px."""
        if not (self.message and time.time() < self.message_until):
            return
        size = next((s for s in range(MESSAGE_SIZE, ui.TEXT_SIZE, -1) if ui.font("bold", s).size(self.message)[0] <= self.W - 16),
                    ui.TEXT_SIZE)
        cx, cy = self.W // 2 - origin[0], self.H // 2 - origin[1]
        ui.text(surface, self.message, (cx + 1, cy + 1), ui.BLACK, "bold", size, "center", "middle")
        ui.text(surface, self.message, (cx, cy), tuple(self.palette.debug_red), "bold", size, "center", "middle")

    def _hint_lines(self):
        if self.stage == "select":
            if self.relation == "inside":
                return ["Click the object to spawn the new one inside", "Esc back"]
            return [f"Click every object the new one goes around  ({len(self.selection)} picked)",
                    "click again to unpick  ·  then Done  ·  Esc back"]
        n = self._cfg()["n"]
        segment = self.kind == "fixed" and n == 2
        first = {"custom": "Click both ends of the segment" if segment else "Click to place points in order",
                 "regular": "Press and drag to size and rotate, or click for the default size",
                 "rope": "Click both ends: on a body point or hook to attach, elsewhere to anchor",
                 "hook": "Click a peg to place the hook (it doesn't collide)",
                 "chain": "Click both ends: a rope end attaches to a body point or hook"}[self.mode]
        what = {"rope": "an end", "chain": "an end", "hook": "the hook"}.get(self.mode, "a placed point")
        return [f"{first}  ({len(self.points)}/{n})", f"drag {what} to move it  ·  Z undo  ·  X redo  ·  Esc back"]

    def _hint_box(self):
        w = max(ui.font().size(line)[0] for line in self._hint_lines()) + 2 * ui.CARD_PAD
        area = self._area()
        return pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), ui.CARD_MARGIN, w, 2 * LINE_H + 2 * ui.CARD_PAD - 2)

    def _draw_hint(self, surface, mouse_ui=(-1, -1)):
        """The hint box at the top. It takes no clicks, so with the mouse over it it fades to HINT_FADE opacity to
        show the objects underneath."""
        lines, box = self._hint_lines(), self._hint_box()
        card = pygame.Surface(box.size, pygame.SRCALPHA)
        ui.draw_frame(card, card.get_rect())
        for i, line in enumerate(lines):
            ui.text(card, line, (box.w // 2, ui.CARD_PAD - 1 + i * LINE_H + LINE_H // 2),
                    ui.TEXT if i == 0 else ui.TEXT_DIM, align="center", valign="middle")
        if box.inflate(2 * PAD, 2 * PAD).collidepoint(mouse_ui):
            alpha = pygame.surfarray.pixels_alpha(card)
            alpha[...] = alpha.astype(np.uint16) * HINT_FADE // 255
            del alpha
        surface.blit(card, box.topleft)

    def _draw_widget(self, surface, area, spec, hovered, st=ui.UI):
        """st: the ui_style.Style the widget is drawn in (HUD for the info card's buttons)."""
        kind = spec[0]
        if kind == "swatch":
            name, selected = spec[1], spec[2]
            ui.draw_frame(surface, area, self._color(name))
            if selected or hovered:
                pygame.draw.rect(surface, ui.WHITE if selected else ui.TEXT_DIM, area.inflate(4, 4), 1)
        elif kind == "button":
            st.draw_button(surface, area, spec[1], hovered, spec[2], spec[3])
        elif kind == "pause":
            self._icon_button(surface, st, area, hovered, self._draw_pause_icon, "[p]")
        elif kind == "plus":
            self._icon_button(surface, st, area, hovered, self._draw_plus_icon, "edit", disabled=spec[1])
        elif kind == "compute":  # selected (lit) while the compute card is shown
            self._icon_button(surface, st, area, hovered, self._draw_chart_icon, "[f]", selected=self.show_compute)
        elif kind == "kind":
            ui.draw_button(surface, area, None, hovered)
            self._draw_kind_icon(surface, spec[1], area)
            ui.text(surface, KIND_TITLES[spec[1]], (area.centerx, area.bottom - 14), ui.WHITE, "bold", align="center", valign="middle")

    @staticmethod
    def _icon_button(surface, st, area, hovered, icon, label, disabled=False, selected=False):
        """A button with an icon (8 base px wide, drawn by icon(surface, st, cx, cy, col)) and a bold label, centered."""
        st.draw_button(surface, area, None, hovered, selected, disabled)
        col = ui.TEXT_DIM if disabled else ui.WHITE
        icon_w, gap = st.px(8), st.px(4)
        x = area.centerx - (icon_w + gap + st.font("bold").size(label)[0]) // 2
        icon(surface, st, x + icon_w // 2, area.centery, col)
        st.text(surface, label, (x + icon_w + gap, area.centery), col, "bold", valign="middle")

    def _draw_pause_icon(self, surface, st, cx, cy, col):
        px = st.px
        if self.paused:  # paused: show play
            pygame.draw.polygon(surface, col, [(cx - px(3), cy - px(5)), (cx - px(3), cy + px(5)), (cx + px(3), cy)])
        else:
            pygame.draw.rect(surface, col, (cx - px(4), cy - px(4), px(3), px(9)))
            pygame.draw.rect(surface, col, (cx + px(1), cy - px(4), px(3), px(9)))

    @staticmethod
    def _draw_plus_icon(surface, st, cx, cy, col):
        px = st.px
        pygame.draw.rect(surface, col, (cx - px(4), cy - px(1), px(9), px(2)))
        pygame.draw.rect(surface, col, (cx - px(1), cy - px(4), px(2), px(9)))

    @staticmethod
    def _draw_chart_icon(surface, st, cx, cy, col):
        """Three bars, like the compute card's chart."""
        px = st.px
        for i, h in enumerate((5, 9, 7)):
            pygame.draw.rect(surface, col, (cx - px(4) + i * px(3), cy + px(5) - px(h), px(2), px(h)))

    @staticmethod
    def _shape(surface, fill, pts, springs=()):
        """A filled polygon with a black 2px border, like the filled body view; springs as 1px black lines. The
        border is the body view's (entities.draw_border): pygame's thick polygon outline grows to one side only, so
        on some edges it left the fill showing outside the border."""
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        pygame.draw.polygon(surface, fill, pts)
        for i, j in springs:
            pygame.draw.line(surface, ui.BLACK, pts[i], pts[j], 1)
        draw_border(surface, np.array(pts).T, ui.BLACK, ui.BORDER)

    def _draw_chain_icon(self, surface, box, segments, start, end, box_col, rope_col, max_box=12, rope_width=1):
        """A chain hanging in a sag between its two ends inside the rect `box`: ropes as lines, boxes as bordered
        squares turned along the chain, a rope end as a black anchor block. Pieces shrink to fit long chains."""
        ropes, boxes = chain_counts(segments, start, end)
        halves = (start == "box") + (end == "box")  # an end box is centred on its end: half of it is past the ends

        def arc(span, sag, k=64):  # the sag curve (a parabola) between the ends, sampled; centred in the box
            t = np.linspace(0.0, 1.0, k)
            return np.array([box.centerx - span / 2 + span * t, box.centery - sag / 2 + sag * (1 - (2 * t - 1) ** 2)])

        def length(curve):
            return float(np.hypot(*np.diff(curve, axis=1)).sum())

        span_max, sag_max = box.w - 16, box.h - 24
        room = length(arc(span_max, sag_max))
        size, rope = max_box, max_box
        while boxes * size + ropes * rope - halves * size / 2 > room and (rope > 3 or size > 7):
            if rope > 3:
                rope -= 1
            else:
                size -= 1
        total = boxes * size + ropes * rope - halves * size / 2
        rope = min(rope, (room - boxes * size + halves * size / 2) / ropes)
        total = min(total, room)
        span = min(span_max, total * 0.75)
        lo, hi = 0.0, sag_max  # the sag that makes the curve as long as the chain
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if length(arc(span, mid)) < total else (lo, mid)
        curve = arc(span, lo, 200)
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(curve, axis=1)))])

        def at(s):
            return np.array([np.interp(s, dist, curve[0]), np.interp(s, dist, curve[1])])

        s = -size / 2 if start == "box" else 0.0
        squares = []
        for k in range(segments):
            if (k % 2 == 0) == (start == "box"):
                a, b = at(s), at(s + size)
                u = (b - a) / max(np.hypot(*(b - a)), 1e-9)
                c = (a + b) / 2 if 0 <= s and s + size <= dist[-1] else at(s + size / 2)
                squares.append(chain_box_points(c, u, size)[:, [1, 2, 4, 5]].T)  # the corners
                s += size
            else:
                pygame.draw.line(surface, rope_col, at(s), at(s + rope), rope_width)
                s += rope
        for end_kind, p in ((start, curve[:, 0]), (end, curve[:, -1])):
            if end_kind == "rope":
                pygame.draw.rect(surface, ui.BLACK, (int(p[0]) - 3, int(p[1]) - 3, 6, 6))
        for corners in squares:
            self._shape(surface, box_col, corners)

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
        elif kind == "chain":
            self._draw_chain_icon(surface, pygame.Rect(cx - 42, cy - 22, 84, 50), 7, "rope", "rope", pal.grey,
                                  pal.light_brown, max_box=9, rope_width=2)
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

    def _draw_config_extras(self, surface, mouse_ui=(-1, -1)):
        """Row labels, the slider, color names, and a live preview of the configured shape."""
        lay, cfg = self._config_layout(), self._cfg()
        for label, pos in lay["rows"]:
            ui.text(surface, label, pos, ui.TEXT)
        for pos, key in lay["color_rows"]:  # the hovered swatch's color name, else the chosen one's
            hovered = [name for rect, name, k in lay["swatches"] if k == key and rect.collidepoint(mouse_ui)]
            name = hovered[0] if hovered else cfg[key]
            ui.text(surface, name.replace("_", " "), pos, ui.WHITE if hovered else ui.TEXT_DIM, align="right")
        if "relation_row" in lay:  # what the hovered (or chosen) placement means
            hovered = [rel for rect, rel in lay["relations"] if rect.collidepoint(mouse_ui)]
            rel = hovered[0] if hovered else self.relation
            ui.text(surface, f"Spawn {RELATION_HINTS[rel]}", lay["relation_row"], ui.WHITE if hovered else ui.TEXT)
        if "slider" in lay:
            track = lay["slider"]
            key, lo, hi, step = self._slider_spec()
            value = cfg[key]
            frac = (value - lo) / (hi - lo)
            pygame.draw.rect(surface, ui.BLACK, track)
            inner = track.inflate(-2, -2)
            pygame.draw.rect(surface, ui.BUTTON_DISABLED, inner)
            pygame.draw.rect(surface, ui.BUTTON_SELECTED, (inner.x, inner.y, int(inner.w * frac), inner.h))
            kx = track.x + int(track.w * frac)
            ui.draw_frame(surface, pygame.Rect(kx - 4, track.centery - 7, 8, 14), ui.WHITE)
            label = f"{value:.1f}" if step < 1 else f"{value}  ({lo}-{hi})"
            ui.text(surface, label, (track.right + 10, track.centery), ui.WHITE, "bold", valign="middle")
            if self.kind == "chain":
                ropes, boxes = chain_counts(value, cfg["start"], cfg["end"])
                ui.text(surface, f"{ropes} ropes, {boxes} boxes", (track.x - 4 + COL_W, track.y - 4 - LABEL_H), ui.TEXT_DIM,
                        align="right")  # on the "Segments" label row

        box = lay["preview"]
        ui.draw_frame(surface, box)
        n, col = cfg["n"], self._color(cfg["color"])
        r = min(box.w, box.h) * 0.34
        if self.kind == "rope":
            self._draw_rope_icon(surface, box.centerx, box.centery, int(r), col, cfg["width"])
        elif self.kind == "chain":
            self._draw_chain_icon(surface, box.inflate(-8, -8), cfg["segments"], cfg["start"], cfg["end"], col,
                                  self._color(cfg["rope_color"]))
        elif n == 2:
            # a segment: a 2px bar framed in black on every side (a thick line's ends were left unbordered)
            half, b = int(r), ui.BORDER
            ui.draw_frame(surface, pygame.Rect(box.centerx - half - b, box.centery - 1 - b, 2 * half + 2 * b, 2 + 2 * b), col)
        else:
            pts = polygon_points(box.centerx, box.centery, r, n, -math.pi / 2 + math.pi / n).T
            springs = [(i, j) for i, j in skip_springs(n, cfg["sets"]) if (j - i) % n not in (1, n - 1)] if self.kind == "soft" else ()
            self._shape(surface, col, pts, springs)
