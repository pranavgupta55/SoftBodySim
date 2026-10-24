"""In-game editor: pause/play, a spawn menu with per-body settings, and point placement with validation.

UI (buttons, panels, messages) is drawn at full screen resolution so text stays crisp; placement previews are
drawn in world coordinates on the scaled-down game surfaces like everything else in the world.
"""
import math
import time

import numpy as np
import pygame

from entities import EnvConfig, FixedBody, RigidBody, SoftBody, polygon_points
from fontDict import fonts
from text import drawText

KIND_TITLES = {"soft": "Soft body", "rigid": "Rigid body", "fixed": "Fixed body"}
POINT_COUNTS = {"soft": (3, 4, 5, 6, 7, 8, 12, 20), "rigid": (3, 4, 5, 6, 7, 8, 12, 20), "fixed": (2, 3, 4, 6, 8, 12)}
STRENGTH_RANGE = (0.1, 1.0)
MESSAGE_SECONDS = 1.6
BAD_SPOT = "you cannot spawn a point there"
INCOMPLETE = "complete the new object to resume the sim"


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


class Editor:
    def __init__(self, world, palette, screen_size, world_size, scale, inset, zoom=1.0):
        self.world, self.palette, self.scale, self.inset, self.zoom = world, palette, scale, inset, zoom
        self.W, self.H = screen_size
        self.world_w, self.world_h = world_size
        self.u = self.H / 982  # UI scale: layouts are designed at 982px tall
        self.colors = [(name, tuple(col)) for name, col in vars(palette).items() if not name.startswith("_") and isinstance(col, list)]
        self.settings = {"soft": {"n": 6, "sets": 1, "strength": 0.1, "color": "network_green"},  # the hexagon
                         "rigid": {"n": 4, "color": "debug_red"},  # the square
                         "fixed": {"n": 4, "color": "grey"}}
        self.paused = False
        self.active = False  # editor mode
        self.stage = None  # "menu" -> "config" -> "place"
        self.kind = None
        self.mode = None  # "regular" or "custom"
        self.points = []  # placed points (world coords) of the object being built
        self.undo_stack, self.redo_stack = [], []
        self.drag_idx = self.drag_from = None
        self.reg_start = None  # press position while dragging out a regular polygon
        self.slider_drag = False
        self.message, self.message_until = None, 0.0

    # ---------------- Small utilities
    def _px(self, v):
        return int(round(v * self.u))

    def _font(self, style, size):
        return fonts[f"{style}{min(99, max(8, self._px(size)))}"]

    def _color(self, name):
        return dict(self.colors)[name]

    def _say(self, text):
        self.message, self.message_until = text, time.time() + MESSAGE_SECONDS

    def _cfg(self):
        return self.settings[self.kind]

    def _complete(self):
        return len(self.points) == self._cfg()["n"]

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
        self.points, self.undo_stack, self.redo_stack = [], [], []
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
        elif action[0] == "poly":
            self.points = []
        else:  # move
            self.points[action[1]] = action[2]
        self.redo_stack.append(action)

    def redo(self):
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        if action[0] == "add":
            self.points.append(action[1])
        elif action[0] == "poly":
            self.points = list(action[1])
        else:
            self.points[action[1]] = action[3]
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
        if self.mode == "custom":
            candidate = self.points + [wp]
            if self._valid(candidate, len(candidate) == self._cfg()["n"]):
                self.points.append(wp)
                self._push(("add", wp))
            else:
                self._say(BAD_SPOT)
        else:
            self.reg_start = wp

    def _place_motion(self, wp):
        if self.drag_idx is not None:
            self.points[self.drag_idx] = wp

    def _place_up(self, wp):
        if self.drag_idx is not None:
            i, old = self.drag_idx, self.drag_from
            self.drag_idx = self.drag_from = None
            if np.hypot(*(wp - old)) < 0.5:
                self.points[i] = old
            elif self._valid(self.points, self._complete()):
                self.points[i] = wp
                self._push(("move", i, old, wp))
            else:
                self.points[i] = old
                self._say(BAD_SPOT)
        elif self.reg_start is not None:
            pts = self._regular_points(self.reg_start, wp)
            self.reg_start = None
            if self._valid(pts, True):
                self.points = pts
                self._push(("poly", list(pts)))
            else:
                self._say(BAD_SPOT)

    # ---------------- Widgets (rebuilt each call; the same list drives drawing and clicking)
    def _pause_rect(self):
        s, m = self._px(56), self._px(16)
        return pygame.Rect(self.W - m - s, m, s, s)

    def _editor_button(self):
        s, m = self._px(56), self._px(16)
        return (m + s // 2, m + s // 2), s // 2

    def _panel(self, w_frac, h_frac):
        w, h = int(self.W * w_frac), int(self.H * h_frac)
        return pygame.Rect((self.W - w) // 2, (self.H - h) // 2, w, h)

    def _config_layout(self):
        """Rects for the settings panel, shared by drawing and click handling."""
        panel = self._panel(0.62, 0.8)
        pad, gap = self._px(24), self._px(8)
        col_w = int(panel.w * 0.62) - pad
        x0, y = panel.x + pad, panel.y + pad + self._px(48)
        label_h = self._px(26)
        lay = {"panel": panel, "rows": []}
        cfg = self._cfg()

        counts = POINT_COUNTS[self.kind]
        cols = 4 if len(counts) == 8 else 3
        cw, ch = min(self._px(72), (col_w - (cols - 1) * gap) // cols), self._px(40)
        lay["rows"].append(("Points", y))
        y += label_h
        lay["counts"] = [(pygame.Rect(x0 + (i % cols) * (cw + gap), y + (i // cols) * (ch + gap), cw, ch), n) for i, n in enumerate(counts)]
        y += 2 * ch + gap + pad

        if self.kind == "soft":
            options = range(max_skip_sets(cfg["n"]) + 1)
            bw = min(self._px(52), (col_w - (len(options) - 1) * gap) // len(options))
            lay["rows"].append(("Spring sets", y))
            y += label_h
            lay["sets"] = [(pygame.Rect(x0 + i * (bw + gap), y, bw, self._px(36)), s) for i, s in enumerate(options)]
            y += self._px(36) + pad
            lay["rows"].append(("Spring strength", y))
            y += label_h
            lay["slider"] = pygame.Rect(x0 + self._px(10), y + self._px(14), int(col_w * 0.75), self._px(6))
            y += self._px(34) + pad

        sw = self._px(30)
        per_row = max(1, (col_w + gap) // (sw + gap))
        lay["rows"].append(("Color", y))
        y += label_h
        lay["swatches"] = [(pygame.Rect(x0 + (i % per_row) * (sw + gap), y + (i // per_row) * (sw + gap), sw, sw), name)
                           for i, (name, _) in enumerate(self.colors)]

        bh = self._px(56)
        bw2 = (panel.w - 3 * pad) // 2
        by = panel.bottom - pad - bh
        lay["modes"] = [(pygame.Rect(panel.x + pad, by, bw2, bh), "regular", "Regular polygon"),
                        (pygame.Rect(panel.x + 2 * pad + bw2, by, bw2, bh), "custom", "Draw points")]
        px0 = panel.x + int(panel.w * 0.62) + pad // 2
        lay["preview"] = pygame.Rect(px0, panel.y + pad + self._px(48), panel.right - pad - px0, by - pad - (panel.y + pad + self._px(48)))
        return lay

    def _widgets(self):
        """(hit shape, action, draw spec) for everything clickable right now."""
        ws = [("rect", self._pause_rect(), self.toggle_pause, ("pause",))]
        if self.paused and not self.active:
            ws.append(("circle", self._editor_button(), self.open, ("plus",)))
        if not self.active:
            return ws
        if self.stage == "menu":
            panel = self._panel(0.6, 0.55)
            pad = self._px(28)
            bw = (panel.w - 4 * pad) // 3
            bh = int(panel.h * 0.55)
            by = panel.y + (panel.h - bh) // 2 + self._px(20)
            for i, kind in enumerate(("soft", "rigid", "fixed")):
                rect = pygame.Rect(panel.x + pad + i * (bw + pad), by, bw, bh)
                ws.append(("rect", rect, lambda k=kind: self.choose_kind(k), ("kind", kind)))
            ws.append(("rect", self._close_rect(panel), self.close, ("text", "x", False)))
        elif self.stage == "config":
            lay = self._config_layout()
            cfg = self._cfg()
            ws.append(("rect", self._close_rect(lay["panel"]), self.close, ("text", "x", False)))
            back = pygame.Rect(lay["panel"].x + self._px(16), lay["panel"].y + self._px(16), self._px(84), self._px(34))
            ws.append(("rect", back, self.back, ("text", "< Back", False)))
            for rect, n in lay["counts"]:
                ws.append(("rect", rect, lambda n=n: self._set_count(n), ("text", str(n), cfg["n"] == n)))
            for rect, s in lay.get("sets", []):
                ws.append(("rect", rect, lambda s=s: cfg.__setitem__("sets", s), ("text", str(s), cfg["sets"] == s)))
            for rect, name in lay["swatches"]:
                ws.append(("rect", rect, lambda c=name: cfg.__setitem__("color", c), ("swatch", name, cfg["color"] == name)))
            for rect, mode, label in lay["modes"]:
                ws.append(("rect", rect, lambda m=mode: self.start_place(m), ("text", label, False)))
        elif self.stage == "place":
            fw, fh, m = self._px(170), self._px(50), self._px(20)
            finish = pygame.Rect((self.W - fw) // 2, self.H - m - fh, fw, fh)
            ws.append(("rect", finish, self.finish, ("text", "Finish", self._complete())))
            sw = self._px(90)
            ws.append(("rect", pygame.Rect(finish.x - m // 2 - sw, finish.y, sw, fh), self.undo, ("text", "Undo", False)))
            ws.append(("rect", pygame.Rect(finish.right + m // 2, finish.y, sw, fh), self.redo, ("text", "Redo", False)))
        return ws

    def _close_rect(self, panel):
        s = self._px(34)
        return pygame.Rect(panel.right - self._px(16) - s, panel.y + self._px(16), s, s)

    def _set_count(self, n):
        cfg = self._cfg()
        cfg["n"] = n
        if "sets" in cfg:
            cfg["sets"] = min(cfg["sets"], max_skip_sets(n))

    def _set_strength(self, x):
        rect = self._config_layout()["slider"]
        frac = min(max((x - rect.x) / rect.w, 0.0), 1.0)
        lo, hi = STRENGTH_RANGE
        self._cfg()["strength"] = round(lo + frac * (hi - lo), 1)

    @staticmethod
    def _hit(shape, area, pos):
        if shape == "circle":
            (cx, cy), r = area
            return math.hypot(pos[0] - cx, pos[1] - cy) <= r
        return area.collidepoint(pos)

    # ---------------- Events (full-resolution screen coordinates)
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
                if cmd and event.key == pygame.K_z:
                    self.redo() if event.mod & pygame.KMOD_SHIFT else self.undo()
                    return True
                if cmd and event.key == pygame.K_y:
                    self.redo()
                    return True
            return False

        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return False
        pos = event.pos
        wp = np.array(pos, dtype=float) / (self.scale * self.zoom)  # screen pixels -> world units
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button != 1:
                return self.paused
            for shape, area, action, _ in self._widgets():
                if self._hit(shape, area, pos):
                    action()
                    return True
            if self.active and self.stage == "config" and self.kind == "soft":
                if self._config_layout()["slider"].inflate(self._px(24), self._px(28)).collidepoint(pos):
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
    def draw_world(self, surface, ghost_surface, mouse):
        """Preview of the object being placed: grey ghost on the translucent layer, placed points on top."""
        if not (self.active and self.stage == "place"):
            return
        wp = np.array(mouse, dtype=float) / (self.scale * self.zoom)
        z = self.zoom
        n, complete = self._cfg()["n"], self._complete()
        pts, valid = list(self.points), True
        over_ui = any(self._hit(shape, area, mouse) for shape, area, _, _ in self._widgets())
        if over_ui:
            pass  # no hover point while aiming at a button
        elif self.mode == "custom" and not complete and self.drag_idx is None:
            pts = pts + [wp]
            valid = self._valid(pts, len(pts) == n)
        elif self.mode == "regular" and self.reg_start is not None and not complete:
            pts = self._regular_points(self.reg_start, wp)
            valid = self._valid(pts, True)
        ghost = (200, 200, 200) if valid else tuple(self.palette.debug_red)
        if len(pts) >= 3:
            pygame.draw.polygon(ghost_surface, ghost, [tuple(np.asarray(p) * z) for p in pts])
        if len(pts) >= 2:
            pygame.draw.lines(ghost_surface, ghost, len(pts) >= 3, [tuple(np.asarray(p) * z) for p in pts], 1)
        hover = self._near_point(wp)
        for i, p in enumerate(self.points):
            col = self.palette.white if i in (hover, self.drag_idx) else self.palette.greyL
            pygame.draw.circle(surface, col, tuple(np.asarray(p) * z), 2)

    # ---------------- Drawing: full-resolution UI
    def draw_ui(self, screen, mouse):
        if self.active and self.stage in ("menu", "config"):
            dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 140))
            screen.blit(dim, (0, 0))
            panel = self._panel(0.6, 0.55) if self.stage == "menu" else self._config_layout()["panel"]
            pygame.draw.rect(screen, (24, 27, 35), panel, border_radius=self._px(14))
            pygame.draw.rect(screen, self.palette.grey_blue, panel, self._px(2), border_radius=self._px(14))
            title = "Spawn an object" if self.stage == "menu" else KIND_TITLES[self.kind]
            drawText(screen, self.palette.white, self._font("bold", 30), panel.centerx, panel.y + self._px(34), title,
                     justify="center", centeredVertically=True, antiAliasing=True)
            if self.stage == "config":
                self._draw_config_extras(screen)
        elif self.active and self.stage == "place":
            hint = ("Click to place points in order" if self.mode == "custom" else "Press and drag to size and rotate (click for default size)")
            hint += f"  ({len(self.points)}/{self._cfg()['n']})  ·  drag points to move  ·  Ctrl/Cmd+Z undo  ·  Esc back"
            drawText(screen, self.palette.very_light_blue, self._font("regular", 18), self.W // 2, self._px(40), hint,
                     justify="center", centeredVertically=True, antiAliasing=True)

        for shape, area, _, spec in self._widgets():
            self._draw_widget(screen, shape, area, spec, self._hit(shape, area, mouse))

        if self.message and time.time() < self.message_until:
            font = self._font("bold", 44)
            drawText(screen, (0, 0, 0), font, self.W // 2 + self._px(3), self.H // 2 + self._px(3), self.message,
                     justify="center", centeredVertically=True, antiAliasing=True)
            drawText(screen, self.palette.debug_red, font, self.W // 2, self.H // 2, self.message,
                     justify="center", centeredVertically=True, antiAliasing=True)

    def _draw_widget(self, screen, shape, area, spec, hovered):
        pal = self.palette
        base = pal.sebastian_lague_purple if hovered else pal.grey_blue
        if shape == "circle":
            (cx, cy), r = area
            pygame.draw.circle(screen, base, (cx, cy), r)
            pygame.draw.circle(screen, pal.very_light_blue, (cx, cy), r, self._px(2))
            arm, w = int(r * 0.5), max(2, self._px(5))
            pygame.draw.line(screen, pal.white, (cx - arm, cy), (cx + arm, cy), w)
            pygame.draw.line(screen, pal.white, (cx, cy - arm), (cx, cy + arm), w)
            return
        radius = self._px(10)
        kind = spec[0]
        if kind == "swatch":
            name, selected = spec[1], spec[2]
            pygame.draw.rect(screen, self._color(name), area, border_radius=self._px(6))
            if selected or hovered:
                pygame.draw.rect(screen, pal.white, area.inflate(self._px(6), self._px(6)), self._px(2), border_radius=self._px(8))
            return
        selected = kind == "text" and spec[2]
        pygame.draw.rect(screen, pal.sebastian_lague_light_purple if selected else base, area, border_radius=radius)
        pygame.draw.rect(screen, pal.very_light_blue, area, self._px(2), border_radius=radius)
        if kind == "pause":
            cx, cy = area.centerx, area.y + int(area.h * 0.4)
            s = int(area.h * 0.18)
            if self.paused:
                pygame.draw.polygon(screen, pal.white, [(cx - s, cy - s), (cx - s, cy + s), (cx + s, cy)])
            else:
                bw = max(2, s // 2)
                pygame.draw.rect(screen, pal.white, (cx - s, cy - s, bw, 2 * s))
                pygame.draw.rect(screen, pal.white, (cx + s - bw, cy - s, bw, 2 * s))
            drawText(screen, pal.white, self._font("bold", 14), area.centerx, area.y + int(area.h * 0.78), "[p]",
                     justify="center", centeredVertically=True, antiAliasing=True)
        elif kind == "kind":
            self._draw_kind_icon(screen, spec[1], area)
            drawText(screen, pal.white, self._font("bold", 22), area.centerx, area.bottom - self._px(30), KIND_TITLES[spec[1]],
                     justify="center", centeredVertically=True, antiAliasing=True)
        else:
            drawText(screen, pal.white, self._font("bold", 18), area.centerx, area.centery, spec[1],
                     justify="center", centeredVertically=True, antiAliasing=True)

    def _draw_kind_icon(self, screen, kind, area):
        cx, cy, r = area.centerx, area.y + int(area.h * 0.42), int(min(area.w, area.h) * 0.24)
        pal = self.palette
        if kind == "soft":
            pts = polygon_points(cx, cy, r, 6).T
            for i, j in skip_springs(6, 1):
                pygame.draw.line(screen, pal.very_light_blue, pts[i], pts[j], max(1, self._px(2)))
            for p in pts:
                pygame.draw.circle(screen, pal.network_green, p, self._px(6))
        elif kind == "rigid":
            s = int(r * 0.8)
            pygame.draw.rect(screen, pal.debug_red, (cx - s, cy - s, 2 * s, 2 * s))
            pygame.draw.rect(screen, (0, 0, 0), (cx - s, cy - s, 2 * s, 2 * s), self._px(3))
        else:
            pygame.draw.rect(screen, pal.grey, (cx - r, cy - self._px(8), 2 * r, self._px(16)))
            pygame.draw.rect(screen, (0, 0, 0), (cx - r, cy - self._px(8), 2 * r, self._px(16)), self._px(3))

    def _draw_config_extras(self, screen):
        """Row labels, the strength slider, and a live preview of the configured shape."""
        lay, cfg, pal = self._config_layout(), self._cfg(), self.palette
        label_font = self._font("regular", 17)
        for label, y in lay["rows"]:
            drawText(screen, pal.very_light_blue, label_font, lay["panel"].x + self._px(24), y, label, antiAliasing=True)
        if "slider" in lay:
            track = lay["slider"]
            lo, hi = STRENGTH_RANGE
            frac = (cfg["strength"] - lo) / (hi - lo)
            pygame.draw.rect(screen, pal.greyVD, track, border_radius=track.h // 2)
            pygame.draw.rect(screen, pal.sebastian_lague_light_purple, (track.x, track.y, int(track.w * frac), track.h), border_radius=track.h // 2)
            knob = (track.x + int(track.w * frac), track.centery)
            pygame.draw.circle(screen, pal.white, knob, self._px(11))
            drawText(screen, pal.white, self._font("bold", 18), track.right + self._px(20), track.centery, f"{cfg['strength']:.1f}",
                     centeredVertically=True, antiAliasing=True)

        box = lay["preview"]
        pygame.draw.rect(screen, pal.my_blue, box, border_radius=self._px(10))
        n, col = cfg["n"], self._color(cfg["color"])
        r = min(box.w, box.h) * 0.32
        if n == 2:
            a, b = (box.centerx - r, box.centery), (box.centerx + r, box.centery)
            pygame.draw.line(screen, col, a, b, self._px(4))
            pts = np.array([a, b])
        else:
            pts = polygon_points(box.centerx, box.centery, r, n, -math.pi / 2 + math.pi / n).T
            if self.kind == "soft":
                for i, j in skip_springs(n, cfg["sets"]):
                    pygame.draw.line(screen, pal.greyL, pts[i], pts[j], 1)
            pygame.draw.polygon(screen, col, pts.tolist(), max(1, self._px(3)))
        for p in pts:
            pygame.draw.circle(screen, col, p, self._px(5))
