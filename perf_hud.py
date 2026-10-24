"""Real-time compute card: FPS, frame time, CPU / GPU / idle split and a scrolling per-frame bar chart (last 3 s).

A card in the right-hand sidebar (hud_layer.py: it implements the sidebar's card-provider methods), drawn at native
resolution with ui_style.HUD metrics (every length below is in base px, times HUD.scale). F (or the Controls card's
[f] button) collapses it to a short summary (title and fps, then the frame times), as does the sidebar when its cards
don't fit otherwise. Timings come from a profiler.Profiler; see
that module for the timing and GPU-reporting API.

Bar chart: one stacked bar per frame, newest on the right, scrolling left. A bar's height is that frame's time
between frames, stacked bottom-up as CPU work (physics / draw / other), GPU time and idle wait, so it sums to the
frame time. The x axis is time (3 s across the graph, one native px per column), so at 60 fps and a 280 px graph a
frame gets ~1.6 px: when several frames share a pixel column the slowest one is drawn (spikes never disappear); at
low fps a frame spans several columns. The y
scale is the smallest of SCALES_MS that fits the slowest frame in the window, with a reference line at 16.7 ms.
"""
from collections import deque

import pygame

import ui_style as ui

H = ui.HUD  # card style at native resolution

# stacked segments, bottom to top: (key, legend label, color)
SEGMENTS = (
    ("physics", "phys", (237, 171, 80)),  # cream
    ("draw", "draw", (191, 111, 74)),  # light_brown
    ("other", "other", (255, 96, 141)),  # debug_red: bright, since this segment is usually thin
    ("gpu", "GPU", (99, 199, 77)),  # green
    ("idle", "idle", (66, 76, 110)),  # grey_blue: muted, but distinct from the graph background
)
SEG_COL = {key: col for key, _, col in SEGMENTS}
DRAW_SECTIONS = ("draw", "ui", "present")  # top-level main-loop sections counted as "draw"
SCALES_MS = (100 / 3, 50, 200 / 3, 100, 200, 500, 1000)
REF_MS = 1000 / 60
GRAPH_H = 44  # base px
SMALL = 8  # base font size for graph labels and legend


def ms(v, unit=True):
    """A time given in ms, short enough that a slow frame's numbers still fit beside each other on the card: one
    decimal below 100 ms, whole ms below 1 s, else seconds ("2.5 s"). unit=False drops " ms" (the avg / max pair, whose
    unit is implied), but seconds keep their "s"."""
    if v >= 999.5:
        return f"{v / 1e3:.1f} s" if unit else f"{v / 1e3:.1f}s"
    text = f"{v:.1f}" if v < 99.95 else f"{v:.0f}"
    return f"{text} ms" if unit else text


def card_height(summary=False):
    """The card's height in native px at the current HUD scale: the full card, or the summary (its first two rows)."""
    px, pad, line = H.px, H.CARD_PAD, H.font().get_linesize()
    if summary:
        return pad + line + px(1) + line + pad - px(2)
    small_line = H.font(size=px(SMALL)).get_linesize()
    return pad + line + px(1) + line + 3 * line + px(3) + small_line + px(GRAPH_H) + px(1) + small_line + pad - px(2)


class PerfHUD:
    compactable = True  # sidebar card provider (hud_layer.py): has a shorter, summary form

    def __init__(self, profiler):
        self.prof = profiler
        self.collapsed = False  # shown as the summary by choice (F), whatever the sidebar's fit
        self.rect = None  # where the sidebar placed it (native screen px)
        self._text_cache = {}  # (string, color, style, size) -> rendered surface
        self._segs = deque()  # (frame end, segment seconds) for each frame in prof.frames, in the same order

    # ---------------- text (cached: most strings repeat frame to frame)
    def _text(self, surface, string, pos, color=ui.TEXT, style="regular", size=None, align="left"):
        size = size or H.TEXT_SIZE
        key = (string, color, style, size)
        img = self._text_cache.get(key)
        if img is None:
            if len(self._text_cache) > 512:
                self._text_cache.clear()
            img = self._text_cache[key] = H.font(style, size).render(string, False, color)
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

    # ---------------- sidebar card provider (hud_layer.py)
    def card_height(self, compact=False):
        return card_height(compact or self.collapsed)

    def place(self, rect):
        self.rect = rect

    def draw_card(self, surface, origin, mouse_px, compact=False):
        self.draw(surface, self.rect.move(-origin[0], -origin[1]), compact or self.collapsed)

    # ---------------- card
    def draw(self, surface, rect, summary=False):
        """Draw the card at `rect` (its height: card_height(summary)) on `surface`; summary: just the title row (fps)
        and the frame times. Sizes follow ui_style.HUD (native-resolution px). Returns the rect."""
        prof, px = self.prof, H.px
        st = prof.stats()
        small = px(SMALL)
        line = H.font().get_linesize()
        small_line = H.font(size=small).get_linesize()
        pad = H.CARD_PAD
        graph_h = px(GRAPH_H)
        rect = pygame.Rect(rect.x, rect.y, rect.w, card_height(summary))
        H.draw_frame(surface, rect)
        x0, x1 = rect.left + pad, rect.right - pad
        y = rect.top + pad - px(1)

        # title + fps
        self._text(surface, "Compute", (x0, y), style="bold", size=H.TITLE_SIZE)
        self._text(surface, f"{st['fps']:.0f} fps", (x1, y), color=ui.ACCENT, style="bold", size=H.TITLE_SIZE, align="right")
        y += line + px(1)

        # time between frames: newest frame, then the window's mean and worst
        last = prof.last.wall * 1e3 if prof.last else 0.0
        self._text(surface, ms(last), (x0, y))
        worst = st["frame_max"] * 1e3
        avg, worst = ms(st["frame"] * 1e3, False), (f"{worst:.0f}" if worst < 999.5 else ms(worst, False))
        self._text(surface, f"avg {avg}  max {worst}", (x1, y), color=ui.TEXT_DIM, align="right")
        y += line
        if summary:
            return rect

        # CPU / GPU / idle per frame (3 s means)
        frame = st["frame"] or 1.0
        for label, key, col in (("CPU", "cpu", SEG_COL["physics"]), ("GPU", "gpu", SEG_COL["gpu"]), ("idle", "idle", SEG_COL["idle"])):
            swatch = (x0, y + px(4), px(6), px(6))
            pygame.draw.rect(surface, col, swatch)
            pygame.draw.rect(surface, ui.BLACK, swatch, px(1))
            self._text(surface, label, (x0 + px(9), y))
            if key == "gpu" and not prof.gpu_seen:
                self._text(surface, "no GPU backend", (x1, y), color=ui.TEXT_DIM, align="right")
            else:
                self._text(surface, f"{st[key] / frame * 100:.0f}%", (x1, y), color=ui.TEXT_DIM, align="right")
                pct_w = H.font().size("100%")[0] + px(4)  # room for "100%" and a gap
                self._text(surface, ms(st[key] * 1e3), (x1 - pct_w, y), align="right")
            y += line
        y += px(3)

        # legend, spread across the card (native-size fonts are a little wider than the upscaled small ones were)
        font_small = H.font(size=small)
        widths = [px(7) + font_small.size(label)[0] for _, label, _ in SEGMENTS]
        spare = max(x1 - x0 - sum(widths), 0) / (len(SEGMENTS) - 1)
        lx = x0
        for (_, label, col), w in zip(SEGMENTS, widths):
            ix = round(lx)
            surface.fill(ui.BLACK, (ix, y + px(3), px(5), px(5)))
            surface.fill(col, (ix + px(1), y + px(4), px(3), px(3)))
            self._text(surface, label, (ix + px(7), y), color=ui.TEXT_DIM, size=small)
            lx += w + spare
        y += small_line

        # bar chart
        graph = pygame.Rect(x0, y, x1 - x0, graph_h)
        self._bars(surface, graph, st)
        y = graph.bottom + px(1)
        self._text(surface, "past 3 s, newest right", (x0, y), color=ui.TEXT_DIM, size=small)
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
                    fill(ui.ACCENT, (left + c, graph.top, 1, H.px(1)))

        # 60 fps reference line (dashed) and scale labels
        px, small = H.px, H.px(SMALL)
        ref_y = graph.bottom - round(REF_MS / 1e3 * px_per_s)
        for x in range(graph.left, graph.right, px(4)):
            surface.fill(ui.TEXT, (x, ref_y, px(2), px(1)))
        self._text(surface, f"{top_ms:.0f} ms", (graph.left + px(2), graph.top), color=ui.TEXT_DIM, size=small)
        self._text(surface, "16.7", (graph.right - px(2), ref_y - H.font(size=small).get_height()), color=ui.TEXT, size=small, align="right")
        pygame.draw.rect(surface, ui.BLACK, graph.inflate(2 * px(1), 2 * px(1)), px(1))
