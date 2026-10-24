"""Real-time compute card: FPS, frame time, CPU / GPU / idle split and a scrolling per-frame bar chart (last 3 s).

Drawn in the top-right card column (ui_style), under the info card. Timings come from a profiler.Profiler; see that
module for the timing and GPU-reporting API.

Bar chart: one stacked bar per frame, newest on the right, scrolling left. A bar's height is that frame's time
between frames, stacked bottom-up as CPU work (physics / draw / other), GPU time and idle wait, so it sums to the
frame time. The x axis is time (3 s across the graph), so at 60 fps a frame gets ~0.8 px: when several frames share
a pixel column the slowest one is drawn (spikes never disappear); at low fps a frame spans several columns. The y
scale is the smallest of SCALES_MS that fits the slowest frame in the window, with a reference line at 16.7 ms.
"""
from collections import deque

import pygame

import ui_style as ui

# stacked segments, bottom to top: (key, legend label, color)
SEGMENTS = (
    ("physics", "phys", (237, 171, 80)),  # cream
    ("draw", "draw", (191, 111, 74)),  # light_brown
    ("other", "other", (127, 36, 51)),  # lighter_maroon_red
    ("gpu", "GPU", (99, 199, 77)),  # green
    ("idle", "idle", (66, 76, 110)),  # grey_blue: muted, but distinct from the graph background
)
SEG_COL = {key: col for key, _, col in SEGMENTS}
DRAW_SECTIONS = ("draw", "ui", "present")  # top-level main-loop sections counted as "draw"
SCALES_MS = (100 / 3, 50, 200 / 3, 100, 200, 500, 1000)
REF_MS = 1000 / 60
GRAPH_H = 44
SMALL = 8  # font size for graph labels and legend


class PerfHUD:
    def __init__(self, profiler):
        self.prof = profiler
        self._text_cache = {}  # (string, color, style, size) -> rendered surface
        self._segs = deque()  # (frame end, segment seconds) for each frame in prof.frames, in the same order

    # ---------------- text (cached: most strings repeat frame to frame)
    def _text(self, surface, string, pos, color=ui.TEXT, style="regular", size=ui.TEXT_SIZE, align="left"):
        key = (string, color, style, size)
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) > 512:
                self._text_cache.clear()
            img = self._text_cache[key] = ui.font(style, size).render(string, False, color)
        r = img.get_rect()
        setattr(r, {"left": "topleft", "right": "topright"}[align], pos)
        surface.blit(img, r)
        return r

    @staticmethod
    def segments(f):
        """Seconds per stacked segment of one profiler Frame (same order as SEGMENTS); they sum to f.wall."""
        sec, gpu_in = f.sections, f.gpu_by_top
        physics = max(sec.get(("physics",), 0.0) - gpu_in.get("physics", 0.0), 0.0)
        draw = max(sum(sec.get((s,), 0.0) - gpu_in.get(s, 0.0) for s in DRAW_SECTIONS), 0.0)
        other = max(f.cpu - physics - draw, 0.0)
        return physics, draw, other, f.gpu, f.idle

    # ---------------- card
    def draw(self, surface, top):
        """Draw the card in the top-right column starting at y = `top`; returns its rect (for the next card)."""
        prof = self.prof
        st = prof.stats()
        line = ui.font().get_linesize()
        small_line = ui.font(size=SMALL).get_linesize()
        pad = ui.CARD_PAD
        height = pad + line + 1 + line + 3 * line + 3 + small_line + GRAPH_H + 1 + small_line + pad - 2
        rect = ui.card_rect(surface.get_width(), top, height)
        ui.draw_frame(surface, rect)
        x0, x1 = rect.left + pad, rect.right - pad
        y = rect.top + pad - 1

        # title + fps
        self._text(surface, "Compute", (x0, y), style="bold", size=ui.TITLE_SIZE)
        self._text(surface, f"{st['fps']:.0f} fps", (x1, y), color=ui.ACCENT, style="bold", size=ui.TITLE_SIZE, align="right")
        y += line + 1

        # time between frames: newest frame, then the window's mean and worst
        last = prof.last.wall * 1e3 if prof.last else 0.0
        self._text(surface, f"{last:.1f} ms", (x0, y))
        self._text(surface, f"avg {st['frame'] * 1e3:.1f}  max {st['frame_max'] * 1e3:.0f}", (x1, y), color=ui.TEXT_DIM, align="right")
        y += line

        # CPU / GPU / idle per frame (3 s means)
        frame = st["frame"] or 1.0
        for label, key, col in (("CPU", "cpu", SEG_COL["physics"]), ("GPU", "gpu", SEG_COL["gpu"]), ("idle", "idle", SEG_COL["idle"])):
            pygame.draw.rect(surface, col, (x0, y + 4, 6, 6))
            pygame.draw.rect(surface, ui.BLACK, (x0, y + 4, 6, 6), 1)
            self._text(surface, label, (x0 + 9, y))
            if key == "gpu" and not prof.gpu_seen:
                self._text(surface, "no GPU backend", (x1, y), color=ui.TEXT_DIM, align="right")
            else:
                self._text(surface, f"{st[key] / frame * 100:.0f}%", (x1, y), color=ui.TEXT_DIM, align="right")
                self._text(surface, f"{st[key] * 1e3:.1f} ms", (x1 - 26, y), align="right")
            y += line
        y += 3

        # legend
        lx = x0
        for _, label, col in SEGMENTS:
            surface.fill(ui.BLACK, (lx, y + 3, 5, 5))
            surface.fill(col, (lx + 1, y + 4, 3, 3))
            lx = self._text(surface, label, (lx + 7, y), color=ui.TEXT_DIM, size=SMALL).right + 4
        y += small_line

        # bar chart
        graph = pygame.Rect(x0, y, x1 - x0, GRAPH_H)
        self._bars(surface, graph, st)
        y = graph.bottom + 1
        self._text(surface, "past 3 s, newest right", (x0, y), color=ui.TEXT_DIM, size=SMALL)
        return rect

    def _bars(self, surface, graph, st):
        prof = self.prof
        surface.fill(ui.PANEL_BG, graph)
        top_ms = next((s for s in SCALES_MS if s >= st["frame_max"] * 1e3), SCALES_MS[-1])
        px_per_s = graph.height / (top_ms / 1e3)

        # segments are computed once per frame and kept while the frame is in the window
        segs, frames = self._segs, prof.frames
        while segs and (not frames or segs[0][0] < frames[0].end):
            segs.popleft()
        newest, new = (segs[-1][0] if segs else -1.0), []
        for f in reversed(frames):  # only the frames filed since the last draw
            if f.end <= newest:
                break
            new.append((f.end, self.segments(f)))
        segs.extend(reversed(new))

        # slowest frame per pixel column; x is time, the right edge is the newest frame's end
        if frames:
            n_cols = graph.width
            t_right = frames[-1].end
            col_scale = n_cols / prof.window
            best = [None] * n_cols
            for f, (_, s) in zip(frames, segs):
                c0 = int(n_cols - (t_right - f.end + f.wall) * col_scale)
                c1 = max(int(n_cols - (t_right - f.end) * col_scale), c0 + 1)
                for c in range(max(c0, 0), min(c1, n_cols)):
                    b = best[c]
                    if b is None or f.wall > b[0]:
                        best[c] = (f.wall, s)
            fill, bottom, left = surface.fill, graph.bottom, graph.left
            for c, b in enumerate(best):
                if b is None:
                    continue
                y_acc = 0.0  # stack in float seconds, round the edges so segments tile without gaps
                y_px = bottom
                for (key, _, col), sec in zip(SEGMENTS, b[1]):
                    y_acc += sec
                    y_new = max(bottom - round(y_acc * px_per_s), graph.top)
                    if y_new < y_px:
                        fill(col, (left + c, y_new, 1, y_px - y_new))
                        y_px = y_new
                if b[0] * 1e3 > top_ms:  # off the scale: mark the top pixel
                    fill(ui.ACCENT, (left + c, graph.top, 1, 1))

        # 60 fps reference line (dashed) and scale labels
        ref_y = graph.bottom - round(REF_MS / 1e3 * px_per_s)
        for x in range(graph.left, graph.right, 4):
            surface.fill(ui.TEXT, (x, ref_y, 2, 1))
        self._text(surface, f"{top_ms:.0f} ms", (graph.left + 2, graph.top), color=ui.TEXT_DIM, size=SMALL)
        self._text(surface, "16.7", (graph.right - 2, ref_y - small_h(SMALL)), color=ui.TEXT, size=SMALL, align="right")
        pygame.draw.rect(surface, ui.BLACK, graph.inflate(2, 2), 1)


def small_h(size):
    return ui.font(size=size).get_height()
