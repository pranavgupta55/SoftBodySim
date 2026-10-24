"""Debug overlays for main_opengl.py, toggled with keys 1-4 and drawn on the world layer in world coordinates
(times the zoom), after the bodies:

1 velocity   one arrow per moving body from its centre (mean point velocity), plus a short tick per point that moves
             at least a pixel's worth. Length = speed x VEL_SECONDS, eased into a cap (L = cap * tanh(len / cap)) so
             fast bodies stay readable and slow ones still show
2 contacts   green dots: points within the contact radius of another edge after the last step (the proximity
             contacts, from the final positions, so it's the same for the CPU and GPU backends); orange rings: where
             continuous collision detection (CCD) caught a point crossing an edge during the last step, held for
             HOLD_STEPS physics steps so a single hit is visible
3 stress     every spring, rope and rigid-body edge coloured by length / rest length - 1: blue compressed, grey at
             rest, red stretched, full colour at +-STRESS_FULL. Ropes are tension-only: slack ropes are dim. Rigid
             edges show shape matching's error (normally tiny, so near grey; the legend gives the largest)
4 fallback   outlines on bodies whose points were merged into a rigid impact zone (CCD's last-resort fallback) in
             the last HOLD_STEPS physics steps

Cost: nothing while all four are off (World.debug stays None, nothing is drawn). Overlays 2 and 4 attach a
StepRecorder as World.debug; the solver then reports CCD hits and impact zone points to it (a few array copies per
substep, only when there are hits). Holds count physics steps, not seconds, so they stay put while paused and last
longer on screen in slow motion.
"""
import math

import numpy as np
import pygame

import ui_style as ui
from entities import EnvConfig

NAMES = ("velocity", "contacts", "stress", "fallback")
HOLD_STEPS = 30  # physics steps a CCD hit mark / fallback outline stays visible (0.5 s at 1x)
VEL_SECONDS = 0.15  # body arrow length = speed x this ...
VEL_CAP = 30  # ... eased into this many surface px
TICK_SECONDS = 0.05  # per-point ticks: same, shorter
TICK_CAP = 8
CONTACT_SLACK = 0.15  # a contact counts up to contact_radius * (1 + this): settled contacts sit right at the radius
STRESS_FULL = 0.15  # stretch / compression (fraction of rest length) drawn at full colour
STRESS_BINS = 7  # colour steps per side

VEL_COL = (254, 231, 97)  # Endesga yellow
TICK_COL = (228, 166, 114)  # tan
CONTACT_COL = (99, 199, 77)  # grass green
CCD_COL = (247, 118, 34)  # orange
FALLBACK_COL = (44, 232, 245)  # cyan
COMPRESSED_COL = (0, 153, 219)  # sky blue
NEUTRAL_COL = (192, 203, 220)  # pale steel
STRETCHED_COL = (228, 59, 68)  # red
SLACK_COL = (90, 105, 136)  # slate
SHADOW_COL = (19, 19, 19)


def _mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


# index STRESS_BINS = neutral; below compressed, above stretched
STRESS_PALETTE = ([_mix(NEUTRAL_COL, COMPRESSED_COL, (STRESS_BINS - k) / STRESS_BINS) for k in range(STRESS_BINS)]
                  + [NEUTRAL_COL]
                  + [_mix(NEUTRAL_COL, STRETCHED_COL, k / STRESS_BINS) for k in range(1, STRESS_BINS + 1)])


def stress_bins(ratio):
    """Palette indices for length / rest - 1 values (an array)."""
    t = np.clip(np.asarray(ratio) / STRESS_FULL, -1.0, 1.0)
    return np.rint(t * STRESS_BINS).astype(int) + STRESS_BINS


def eased(length, cap):
    """Monotonic soft clamp: ~length when short, approaches cap when long."""
    return cap * np.tanh(np.asarray(length) / cap)


class StepRecorder:
    """World.debug while overlay 2 or 4 is on: collects the CCD hits and impact zone bodies of one World.step."""

    def __init__(self):
        self.hits = []  # (2, k) positions (world units) where a point met an edge, one array per detection
        self.zone_bodies = set()  # movable bodies with points in an impact zone this step
        self.zone_substeps = 0  # substeps of this step that needed the impact zone fallback
        self._zones_before = 0

    def begin_step(self, world):
        self.hits = []
        self.zone_bodies = set()
        self.zone_substeps = 0
        self._zones_before = world.zone_count

    def ccd_hits(self, world, hits, D):
        pts, ts = hits[0], hits[2]
        if pts.size:
            self.hits.append(world.PREV[:, pts] + ts * D[:, pts])

    def zone_points(self, world, members):
        for k in np.unique(world.point_body[members]).tolist():
            body = world.bodies[k]
            if not body.fixed:
                self.zone_bodies.add(body)

    def end_step(self, world):
        self.zone_substeps = world.zone_count - self._zones_before


class DebugOverlays:
    def __init__(self):
        self.on = [False] * len(NAMES)
        self.recorder = StepRecorder()
        self.ccd_marks = []  # [(2, k) world positions, steps left]
        self.fallback = {}  # body -> steps left
        # numbers for the legend, from the last draw / step
        self.stats = {"contacts": 0, "ccd": 0, "zones": 0, "rigid_err": 0.0}

    # ---------------- state
    def toggle(self, k):
        """Key 1-4 -> k = 0-3."""
        self.on[k] = not self.on[k]
        if not self.on[1]:
            self.ccd_marks = []
        if not self.on[3]:
            self.fallback = {}

    def any(self):
        return any(self.on)

    def state(self):
        """Info card state: the numbers of the overlays that are on, e.g. "1 3", or "off"."""
        return " ".join(str(k + 1) for k, on in enumerate(self.on) if on) or "off"

    def attach(self, world):
        """Before stepping: give the world the recorder only while an overlay needs the solver's data."""
        world.debug = self.recorder if (self.on[1] or self.on[3]) else None

    def after_step(self, world):
        """After each World.step: age the held marks and take the step's CCD hits / fallback bodies."""
        rec = self.recorder
        if world.debug is not rec:
            return
        rec.end_step(world)
        if self.on[1]:
            self.ccd_marks = [(p, n - 1) for p, n in self.ccd_marks if n > 1]
            hits = [h for h in rec.hits if h.size]
            if hits:
                self.ccd_marks.append((np.concatenate(hits, axis=1), HOLD_STEPS))
            self.stats["ccd"] = sum(h.shape[1] for h in hits)
        if self.on[3]:
            self.fallback = {b: n - 1 for b, n in self.fallback.items() if n > 1}
            for b in rec.zone_bodies:
                self.fallback[b] = HOLD_STEPS
            self.stats["zones"] = rec.zone_substeps

    # ---------------- drawing (world layer)
    def draw(self, surface, world, zoom):
        if not any(self.on) or not world.bodies:
            return
        if self.on[2]:
            self._draw_stress(surface, world, zoom)
        if self.on[3]:
            self._draw_fallback(surface, world, zoom)
        if self.on[1]:
            self._draw_contacts(surface, world, zoom)
        if self.on[0]:
            self._draw_velocity(surface, world, zoom)

    @staticmethod
    def _lines(surface, cols, a, b, width=1):
        line = pygame.draw.line
        for col, p, q in zip(cols, a.T.tolist(), b.T.tolist()):
            line(surface, col, p, q, width)

    def _draw_stress(self, surface, world, zoom):
        P, pal = world.P, STRESS_PALETTE
        if world.springs.shape[1]:
            i, j = world.springs
            ratio = np.hypot(*(P[:, j] - P[:, i])) / np.maximum(world.spring_rest, 1e-9) - 1.0
            self._lines(surface, [pal[k] for k in stress_bins(ratio).tolist()], P[:, i] * zoom, P[:, j] * zoom)
        err = 0.0
        for idx, rest, _ in world.rigid_groups:  # (B, n) point ids, (B, 2, n) rest shapes
            pts = P[:, idx]  # (2, B, n)
            nxt = np.roll(pts, -1, axis=2)
            rest_len = np.hypot(*(np.roll(rest, -1, axis=2) - rest).transpose(1, 0, 2))
            ratio = np.hypot(*(nxt - pts)) / np.maximum(rest_len, 1e-9) - 1.0
            if ratio.size:
                err = max(err, float(np.abs(ratio).max()))
            self._lines(surface, [pal[k] for k in stress_bins(ratio.ravel()).tolist()],
                        pts.reshape(2, -1) * zoom, nxt.reshape(2, -1) * zoom)
        self.stats["rigid_err"] = err
        for r in world.ropes:
            a, b = r.end_pos(0), r.end_pos(1)
            ratio = float(np.hypot(*(b - a))) / max(r.rest_length, 1e-9) - 1.0
            col = pal[int(stress_bins(ratio))] if ratio > 0 else SLACK_COL
            pygame.draw.line(surface, col, a * zoom, b * zoom, max(r.width, 1))

    def _draw_fallback(self, surface, world, zoom):
        alive = set(world.bodies)
        for body in self.fallback:
            if body not in alive or body.points.shape[1] < 2:
                continue
            pts = (body.points * zoom).T.tolist()
            if len(pts) == 2:
                pygame.draw.line(surface, FALLBACK_COL, pts[0], pts[1], 2)
            else:
                pygame.draw.lines(surface, FALLBACK_COL, True, [pts[k] for k in body.edges[0].tolist()], 2)

    def contacts(self, world):
        """Points touching another edge (or a non-neighbouring edge of their own body) right now: unique point ids."""
        pts, eids = world._candidates()
        if pts.size == 0:
            return pts
        P = world.P
        a, b = P[:, world.edges[0, eids]], P[:, world.edges[1, eids]]
        p = P[:, pts]
        e = b - a
        t = np.clip(np.sum((p - a) * e, axis=0) / np.maximum(np.sum(e * e, axis=0), 1e-12), 0.0, 1.0)
        dist = np.hypot(*(p - a - t * e))
        return np.unique(pts[dist < EnvConfig.contact_radius * (1 + CONTACT_SLACK)])

    def _draw_contacts(self, surface, world, zoom):
        touching = self.contacts(world)
        self.stats["contacts"] = int(touching.size)
        fill = surface.fill
        for x, y in np.rint(world.P[:, touching] * zoom).astype(int).T.tolist():
            fill(CONTACT_COL, (x - 1, y - 1, 3, 3))
        circle = pygame.draw.circle
        for marks, _ in self.ccd_marks:
            for c in (marks * zoom).T.tolist():
                circle(surface, CCD_COL, c, 3, 1)

    def _draw_velocity(self, surface, world, zoom):
        P, V, W = world.P, world.V, world.W
        # per-point ticks, only where they'd be at least a pixel long
        speed = np.hypot(*V)
        tick = eased(speed * TICK_SECONDS * zoom, TICK_CAP)
        moving = np.flatnonzero((tick >= 1.0) & (W > 0))
        if moving.size:
            d = V[:, moving] * (tick[moving] / np.maximum(speed[moving], 1e-12))
            a = P[:, moving] * zoom
            self._lines(surface, [TICK_COL] * moving.size, a, a + d)
        # per-body arrows from the centre (mean of the points)
        counts = world.body_counts
        live = np.flatnonzero((counts > 0) & ~world.body_fixed)
        if live.size == 0:
            return
        pb, nb = world.point_body, counts.size

        def body_mean(X):
            return np.array([np.bincount(pb, X[k], minlength=nb)[live] for k in range(2)]) / counts[live]

        com, vel = body_mean(P), body_mean(V)
        speed = np.hypot(*vel)
        length = eased(speed * VEL_SECONDS * zoom, VEL_CAP)
        show = length >= 2.0
        if not show.any():
            return
        u = vel[:, show] / speed[show]
        a = com[:, show] * zoom
        b = a + u * length[show]
        head = np.minimum(4.0, length[show] * 0.5)
        side = np.array([-u[1], u[0]])
        h1 = b - u * head + side * head * 0.6
        h2 = b - u * head - side * head * 0.6
        line = pygame.draw.line
        for p, q, r, s in zip(a.T.tolist(), b.T.tolist(), h1.T.tolist(), h2.T.tolist()):
            line(surface, VEL_COL, p, q, 1)
            line(surface, VEL_COL, q, r, 1)
            line(surface, VEL_COL, q, s, 1)

    # ---------------- legend (UI layer, top left)
    LEGEND_SIZE = 10  # font px on the UI surface
    LEGEND_LINE = 12

    def legend_lines(self, world):
        st = self.stats
        return [("1", "velocity", "", "vel"),
                ("2", "contacts", f"{st['contacts']}  ccd {st['ccd']}", "contacts"),
                ("3", "stress", f"±{STRESS_FULL * 100:.0f}%  rigid {st['rigid_err'] * 100:.1f}%", "stress"),
                ("4", "fallback", f"{sum(1 for b in self.fallback if b in set(world.bodies))}  zones {world.zone_count}",
                 "fallback")]

    def draw_legend(self, surface, world, origin):
        """A small key for the overlays with its top-left corner at UI px `origin`. Drawn only while one is on;
        returns its rect (or None)."""
        if not any(self.on):
            return None
        st = ui.UI
        f, fb = st.font("regular", self.LEGEND_SIZE), st.font("bold", self.LEGEND_SIZE)
        lines = self.legend_lines(world)
        pad, sw, gap = 4, 14, 4  # padding, swatch column, column gap
        name_w = max(f.size(n)[0] for _, n, _, _ in lines)
        stat_w = max(f.size(s)[0] for _, _, s, _ in lines)
        key_w = fb.size("4")[0]
        w = pad + key_w + gap + sw + gap + name_w + gap + stat_w + pad
        h = pad + len(lines) * self.LEGEND_LINE + pad - 2
        rect = pygame.Rect(origin[0], origin[1], w, h)
        st.draw_frame(surface, rect, ui.CARD_BG)
        for k, (key, name, stat, kind) in enumerate(lines):
            on = self.on[k]
            y = rect.y + pad + k * self.LEGEND_LINE
            cy = y + self.LEGEND_LINE // 2 - 1
            x = rect.x + pad
            surface.blit(fb.render(key, False, ui.WHITE if on else ui.TEXT_DIM), (x, y))
            x += key_w + gap
            self._swatch(surface, kind, x, cy, sw, on)
            x += sw + gap
            surface.blit(f.render(name, False, ui.TEXT if on else ui.TEXT_DIM), (x, y))
            x += name_w + gap
            if on and stat:
                surface.blit(f.render(stat, False, ui.TEXT_DIM), (x, y))
        return rect

    @staticmethod
    def _swatch(surface, kind, x, cy, w, on):
        dim = (70, 74, 90)
        if kind == "vel":
            col = VEL_COL if on else dim
            pygame.draw.line(surface, col, (x, cy), (x + w - 2, cy))
            pygame.draw.line(surface, col, (x + w - 2, cy), (x + w - 5, cy - 3))
            pygame.draw.line(surface, col, (x + w - 2, cy), (x + w - 5, cy + 3))
        elif kind == "contacts":
            surface.fill(CONTACT_COL if on else dim, (x + 1, cy - 1, 3, 3))
            pygame.draw.circle(surface, CCD_COL if on else dim, (x + w - 4, cy), 3, 1)
        elif kind == "stress":
            for i in range(w):
                t = i / (w - 1) * 2 - 1
                col = STRESS_PALETTE[int(stress_bins(t * STRESS_FULL))] if on else dim
                surface.fill(col, (x + i, cy - 1, 1, 3))
        else:
            pygame.draw.rect(surface, FALLBACK_COL if on else dim, (x + 1, cy - 3, w - 2, 7), 1)
