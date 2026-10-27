"""The world layer on the GPU: background, broad phase grid (G), motion trails (T) and every body, rope and hook,
drawn by moderngl into a framebuffer the size of the render surface. main_opengl.py composites it under screen2
(which then only holds what is still drawn by pygame: debug overlays and the editor's world-space previews).

The look is pygame's, pixel for pixel: the shaders reproduce pygame's own rasterisation rules instead of OpenGL's.
    lines     pygame.draw.line's Bresenham (closed form) and its thick-line rule, on truncated endpoints
    polygons  pygame.draw.polygon's scanline fill (draw_fillpoly): each body's box is shaded and every pixel walks
              the body's edges, so concave and self-intersecting outlines fill exactly as pygame fills them
    circles   pygame.draw.circle's own pixels: each (radius, width) is drawn once by pygame into a mask atlas and
              stamped at the truncated centre
Draw order: World.draw draws bodies in list order (each one's fill, then its border; or springs, then points), then
ropes, then hooks, then the grab line. Every primitive carries a depth layer from that order (later = nearer, depth
test <=), so the whole world is three instanced draw calls (fills, lines, circles) whatever their interleaving.
The grid and the trails are one fullscreen base pass under the bodies, in whole 0-255 values like the CPU path.

Per frame the CPU work is numpy over the World's packed arrays (P, edges, offsets): one upload of truncated /
rounded pixel coordinates per primitive kind. Per-body topology (which points form each fill loop, spring lists) is
cached until the World repacks. Body colours, radii and widths are read every frame (palettes recolour in place).

Anything not handled here falls back to pygame: World.draw / draw_grid / Trails.draw keep working on any surface
(the --basic pipeline and the editor tests use them).
"""
import inspect
import os
from types import SimpleNamespace

import moderngl
import numpy as np
import pygame

from entities import Body, EnvConfig
import looping
from solver import World

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
VERT_ROW = 4096  # vertices per row of the fill vertex texture (world_fill.frag's VERT_ROW)
COORD_LIMIT = 1 << 14  # px; coordinates are clamped to +-this so the shaders' integer math can't overflow
# World.draw's filled view uses draw_filled's default border; draw_grid's default colours (main_opengl passes others
# for worlds with their own background)
_FILLED = inspect.signature(Body.draw_filled).parameters
BORDER_COL, BORDER_WIDTH = tuple(_FILLED["border_col"].default), _FILLED["border_width"].default
GRID_DEFAULTS = {k: tuple(v.default) for k, v in inspect.signature(World.draw_grid).parameters.items()
                 if k.endswith("_col")}


def _load(ctx, vert, frag):
    with open(os.path.join(SHADER_DIR, "world_common.glsl")) as f:
        common = f.read()
    srcs = []
    for name, define in ((vert, "#define VERTEX_SHADER\n"), (frag, "")):
        with open(os.path.join(SHADER_DIR, name)) as f:
            srcs.append("#version 330 core\n" + define + common + "\n" + f.read())
    return ctx.program(vertex_shader=srcs[0], fragment_shader=srcs[1])


def pack_rgb(cols):
    """(k,) packed 0xBBGGRR ints from a list of colours (lists, tuples or arrays; RGB or RGBA)."""
    if not len(cols):
        return np.zeros(0, np.int64)
    c = np.array([tuple(col)[:3] for col in cols], dtype=np.int64)
    return c[:, 0] | (c[:, 1] << 8) | (c[:, 2] << 16)


def _px(a):
    """Float pixel coordinates -> whole pixels the way pygame takes them (truncated toward zero), clamped (a NaN
    point goes off screen)."""
    return np.clip(np.nan_to_num(np.trunc(a), nan=-COORD_LIMIT), -COORD_LIMIT, COORD_LIMIT).astype(np.int64)


class _Topology:
    """Per-body index lists for drawing, valid until the World repacks (World._rebuild rebinds P and edges)."""

    def __init__(self, world):
        self.key = (world.P, world.edges, len(world.bodies))
        bodies = world.bodies
        counts = np.asarray(world.body_counts)
        offsets = world.offsets
        n_edges = np.array([b.edges.shape[1] for b in bodies], dtype=np.int64)
        eoff = np.concatenate([[0], np.cumsum(n_edges)])
        # draw_filled's three cases: a segment, a closed polygon (fill + border), anything else (points view)
        self.seg = counts == 2
        self.fill = ~self.seg & (n_edges >= 3)
        self.dot = ~self.seg & ~self.fill & (counts > 0)
        fb = np.flatnonzero(self.fill)
        lens = n_edges[fb]
        self.fill_bodies, self.loop_len = fb, lens
        self.loop_idx = (np.concatenate([world.edges[0, eoff[b]:eoff[b + 1]] for b in fb]) if fb.size
                         else np.zeros(0, np.int64))
        starts = np.cumsum(lens) - lens
        self.loop_start = starts
        self.loop_body = np.repeat(fb, lens)
        pos = np.arange(self.loop_idx.size)
        first, n = np.repeat(starts, lens), np.repeat(lens, lens)
        self.loop_next = np.where(pos - first + 1 < n, pos + 1, first)  # the next vertex around the same loop
        de = [b.draw_edges + off for b, off in zip(bodies, offsets[:-1])]
        self.springs = np.concatenate(de, axis=1).astype(np.int64) if de else np.zeros((2, 0), np.int64)
        self.spring_body = np.repeat(np.arange(len(bodies)), [d.shape[1] for d in de])
        self.point_body = np.asarray(world.point_body)
        self.seg_bodies = np.flatnonzero(self.seg)
        self.seg_first = offsets[self.seg_bodies]

    def valid(self, world):
        k = self.key
        return k[0] is world.P and k[1] is world.edges and k[2] == len(world.bodies)

    def subset(self, mask):
        """The fill / segment lists restricted to the bodies in `mask` (bool per body), for drawing their periodic
        copies (looping.image_copies): an object with the attributes _fills and _segments read."""
        keep = mask[self.fill_bodies]
        lens = self.loop_len[keep]
        starts = np.cumsum(lens) - lens
        pos = np.arange(int(lens.sum()))
        first, n = np.repeat(starts, lens), np.repeat(lens, lens)
        sel = mask[self.loop_body]
        seg = mask[self.seg_bodies]
        return SimpleNamespace(fill_bodies=self.fill_bodies[keep], loop_len=lens, loop_start=starts,
                               loop_idx=self.loop_idx[sel], loop_body=self.loop_body[sel],
                               loop_next=np.where(pos - first + 1 < n, pos + 1, first),
                               seg_bodies=self.seg_bodies[seg], seg_first=self.seg_first[seg])


class WorldPacker:
    """World -> instance arrays for the three draw calls (pure numpy, no GL: testable on its own).

    pack() returns dict(fills=(k, 8) int32 [start, count, colour, layer, minx, miny, maxx, maxy],
    verts=(m, 2) int32 fill vertices, lines=(k, 8) int32 [x1, y1, x2, y2, width, colour, layer, transparency 0..255 (0 opaque)],
    stamps=(k, 6) int64 [cx, cy, colour, layer, width, 0] with radii=(k,) float64)."""

    def __init__(self):
        self.topo = None

    def pack(self, world, filled, zoom, outlines=True):
        bodies = world.bodies
        lines, stamps, radii = [], [], []
        fills = np.zeros((0, 8), np.int32)
        verts = np.zeros((0, 2), np.int32)
        nb = len(bodies)
        if nb:
            if self.topo is None or not self.topo.valid(world):
                self.topo = _Topology(world)
            t = self.topo
            P = world.P * zoom
            body_layer = 2 * np.arange(nb, dtype=np.int64)  # +1 for a segment's inner line
            if filled:
                fill_c = pack_rgb([b.fill_col for b in bodies])
                if t.fill_bodies.size:
                    fills, verts = self._fills(t, P, fill_c, body_layer, lines, bodies, outlines)
                if t.seg_bodies.size:
                    self._segments(t, P, fill_c, body_layer, bodies, lines, stamps, radii, outlines)
                dots = t.dot
            else:
                dots = np.ones(nb, dtype=bool)  # the points view: Body.draw for every body
            if dots.any():
                self._points(t, P, dots, body_layer, bodies, lines, stamps, radii)
            # bodies sticking out past a periodic edge also draw one period over (World.draw does the same), on
            # their own layers, so a body straddling the seam shows on both sides
            groups = {}
            for k, s in looping.image_copies(world):
                groups.setdefault((float(s[0]), float(s[1])), []).append(k)
            for s, ks in groups.items():
                mask = np.zeros(nb, dtype=bool)
                mask[ks] = True
                Ps = (world.P + np.array(s)[:, None]) * zoom
                if filled:
                    sub = t.subset(mask)
                    if sub.fill_bodies.size:
                        f2, v2 = self._fills(sub, Ps, fill_c, body_layer, lines, bodies, outlines)
                        f2[:, 0] += len(verts)
                        fills, verts = np.concatenate([fills, f2]), np.concatenate([verts, v2])
                    if sub.seg_bodies.size:
                        self._segments(sub, Ps, fill_c, body_layer, bodies, lines, stamps, radii, outlines)
                    d = dots & mask
                else:
                    d = mask
                if d.any():
                    self._points(t, Ps, d, body_layer, bodies, lines, stamps, radii)
        layer = 2 * nb
        for r in world.ropes:
            if r.width >= 1:
                a, b = _px(r.end_pos(0) * zoom), _px(r.end_pos(1) * zoom)
                lines.append(np.array([[a[0], a[1], b[0], b[1], r.width, pack_rgb([r.col])[0], layer, 0]]))
            layer += 1
        black = pack_rgb([(0, 0, 0)])[0]  # Hook.draw's own black
        for hk in world.hooks:
            c, r, col = _px(hk.pos * zoom), hk.radius, pack_rgb([hk.col])[0]
            if filled and not outlines:  # Hook.draw without outlines: the coloured disc alone
                specs = ((r - 0.5, 0, col),)
            elif filled:  # Hook.draw: black disc, coloured disc, black hole
                specs = ((r + 1, 0, black), (r - 0.5, 0, col), (1, 0, black))
            else:  # a 1px ring around a dot
                specs = ((r, 1, col), (1, 0, col))
            for k, (rad, width, cc) in enumerate(specs):
                stamps.append(np.array([[c[0], c[1], cc, layer + k, width, 0]]))
                radii.append(np.array([float(rad)]))
            layer += 3
        if world.grabbed is not None and nb:
            a, b = _px(world.P[:, world.grabbed] * zoom), _px(np.asarray(world.grab_target) * zoom)
            lines.append(np.array([[a[0], a[1], b[0], b[1], 1, pack_rgb([(255, 255, 255)])[0], layer, 0]]))
        if getattr(world, "grab_set", None) is not None and nb:  # an area grab: each held point to its target
            idx, off, _, w = world.grab_set  # opacity 0.5 * w: the pull's strength, half max with dozens held
            a, b = _px(world.P[:, idx] * zoom), _px((np.asarray(world.grab_target)[:, None] + off) * zoom)
            k = len(idx)
            lines.append(np.stack([a[0], a[1], b[0], b[1], np.ones(k), np.full(k, pack_rgb([(255, 255, 255)])[0]),
                                   np.full(k, layer), np.round(255 * (1 - 0.5 * w))], axis=1))
        cat = lambda xs, w: np.concatenate(xs).astype(np.int64) if xs else np.zeros((0, w), np.int64)
        lines = cat(lines, 8)
        return dict(fills=fills, verts=verts, lines=lines[lines[:, 4] >= 1].astype(np.int32), stamps=cat(stamps, 6),
                    radii=np.concatenate(radii) if radii else np.zeros(0))

    @staticmethod
    def _fills(t, P, fill_c, body_layer, lines, bodies, outlines=True):
        """Closed polygons: draw_filled rounds the loop to whole pixels, fills it, then draw_border (entities.py):
        the 1px outline plus BORDER_WIDTH - 1 copies shifted 1px outward along each edge's minor axis."""
        loop = np.round(P[:, t.loop_idx])  # (2, m) whole pixels, rounded like draw_filled
        loop = np.clip(np.nan_to_num(loop, nan=-COORD_LIMIT), -COORD_LIMIT, COORD_LIMIT).astype(np.int64)
        fb, starts = t.fill_bodies, t.loop_start
        lo = np.minimum.reduceat(loop, starts, axis=1)
        hi = np.maximum.reduceat(loop, starts, axis=1)
        fills = np.stack([starts, t.loop_len, fill_c[fb], body_layer[fb], lo[0], lo[1], hi[0], hi[1]], axis=1)
        nxt = loop[:, t.loop_next]
        d = nxt - loop
        winding = np.array([b.winding for b in bodies])[t.loop_body]
        steep = np.abs(d[1]) >= np.abs(d[0])
        sx = np.where(steep, np.sign(winding * d[1]), 0).astype(np.int64)
        sy = np.where(steep, 0, np.sign(winding * -d[0])).astype(np.int64)
        col = np.full(loop.shape[1], pack_rgb([BORDER_COL])[0])
        layer = body_layer[t.loop_body]
        ones = np.ones(loop.shape[1], np.int64)
        for k in range(BORDER_WIDTH if outlines else 0):
            lines.append(np.stack([loop[0] + k * sx, loop[1] + k * sy, nxt[0] + k * sx, nxt[1] + k * sy, ones, col,
                                   layer, 0 * ones], axis=1))
        return fills.astype(np.int32), loop.T.astype(np.int32)

    @staticmethod
    def _segments(t, P, fill_c, body_layer, bodies, lines, stamps, radii, outlines=True):
        """Two-point bodies in the filled view: a thick border-coloured line with round ends, the fill line on top."""
        sb = t.seg_bodies
        a, b = _px(P[:, t.seg_first]), _px(P[:, t.seg_first + 1])
        sw = np.maximum(np.array([bodies[k].stick_width for k in sb], dtype=np.int64), 1)
        outer = sw + 2 * BORDER_WIDTH
        border = np.full(sb.size, pack_rgb([BORDER_COL])[0])
        layer, zero = body_layer[sb], np.zeros(sb.size, np.int64)
        if outlines:
            lines.append(np.stack([a[0], a[1], b[0], b[1], outer, border, layer, zero], axis=1))
            for end in (a, b):
                stamps.append(np.stack([end[0], end[1], border, layer, zero, zero], axis=1))
                radii.append(outer / 2)
        lines.append(np.stack([a[0], a[1], b[0], b[1], sw, fill_c[sb], layer + 1, zero], axis=1))

    @staticmethod
    def _points(t, P, which, body_layer, bodies, lines, stamps, radii):
        """Body.draw for the bodies in `which`: springs (draw_edges) in stick_col, then a disc per point."""
        stick_c = pack_rgb([b.stick_col for b in bodies])
        point_c = pack_rgb([b.point_col for b in bodies])
        sel = which[t.spring_body]
        if sel.any():
            i, j = t.springs[:, sel]
            sb = t.spring_body[sel]
            a, b = _px(P[:, i]), _px(P[:, j])
            sw = np.array([bd.stick_width for bd in bodies], dtype=np.int64)[sb]
            lines.append(np.stack([a[0], a[1], b[0], b[1], sw, stick_c[sb], body_layer[sb], 0 * sw], axis=1))
        sel = np.flatnonzero(which[t.point_body])
        if sel.size:
            pb = t.point_body[sel]
            c = _px(P[:, sel])
            zero = np.zeros(sel.size, np.int64)
            stamps.append(np.stack([c[0], c[1], point_c[pb], body_layer[pb], zero, zero], axis=1))
            radii.append(np.array([float(b.point_radius) for b in bodies])[pb])


class CircleAtlas:
    """pygame.draw.circle's pixels for each (radius, width) used, drawn once by pygame (centre at a whole pixel) and
    packed into one R8UI texture. masks[key] = (atlas x, atlas y, w, h, offset x, offset y from the centre)."""
    SIZE = 512

    def __init__(self, ctx):
        self.ctx = ctx
        self.masks = {}
        self.data = np.zeros((self.SIZE, self.SIZE), np.uint8)
        self.cursor = [0, 0, 0]  # x, y, row height
        self.tex = ctx.texture((self.SIZE, self.SIZE), 1, dtype="u1")
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.dirty = True

    def get(self, radius, width):
        key = (float(radius), int(width))
        m = self.masks.get(key)
        if m is None:
            m = self.masks[key] = self._add(*key)
        return m

    def _add(self, radius, width):
        r = int(abs(radius)) + 2
        c = r + 1
        s = pygame.Surface((2 * c + 1, 2 * c + 1))
        s.fill((0, 0, 0))
        pygame.draw.circle(s, (255, 255, 255), (c, c), radius, width)
        mask = (pygame.surfarray.array2d(s).T != 0).astype(np.uint8)  # (y, x)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None  # pygame draws nothing for this one
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        x, y, row = self.cursor
        if x + w > self.SIZE:
            x, y, row = 0, y + row, 0
        if y + h > self.SIZE:
            raise RuntimeError("circle atlas full")
        self.data[y:y + h, x:x + w] = mask[y0:y1, x0:x1]
        self.cursor = [x + w + 1, y, max(row, h + 1)]
        self.dirty = True
        return x, y, w, h, x0 - c, y0 - c

    def upload(self):
        if self.dirty:
            self.tex.write(self.data.tobytes())
            self.dirty = False


class _Stream:
    """A growable instance buffer and its vertex array for one program."""

    def __init__(self, ctx, prog, fmt, attrs):
        self.ctx, self.prog, self.fmt, self.attrs = ctx, prog, fmt, attrs
        self.buf, self.vao, self.cap = None, None, 0

    def write(self, data):
        data = np.ascontiguousarray(data, dtype=np.int32)
        if data.nbytes > self.cap:
            self.cap = max(data.nbytes, 2 * self.cap, 4096)
            if self.buf is not None:
                self.vao.release()
                self.buf.release()
            self.buf = self.ctx.buffer(reserve=self.cap, dynamic=True)
            self.vao = self.ctx.vertex_array(self.prog, [(self.buf, self.fmt, *self.attrs)])
        if data.nbytes:
            self.buf.write(data.tobytes())
        return len(data)


def grid_cells(world, w, h, zoom):
    """World.draw_grid's occupancy for a w x h px view: (cell size px, (cols, rows) int8: 0 empty, 1 an edge's padded
    box, 2 a point). Same numpy as draw_grid (which then fills cells one by one)."""
    cell = world.env.get("cell_size", EnvConfig.cell_size)
    cols, rows = int(np.ceil(w / zoom / cell)), int(np.ceil(h / zoom / cell))
    occupied = np.zeros((cols, rows), dtype=np.int8)
    if world.bodies and cols > 0 and rows > 0:
        pad = 2 * EnvConfig.contact_radius
        a, b = world.P[:, world.edges[0]], world.P[:, world.edges[1]]
        lo = np.clip(np.floor((np.minimum(a, b) - pad) / cell).astype(int), 0, [[cols], [rows]])
        hi = np.clip(np.floor((np.maximum(a, b) + pad) / cell).astype(int), -1, [[cols - 1], [rows - 1]])
        span = np.maximum(hi - lo + 1, 0)
        counts = span[0] * span[1]
        e = np.repeat(np.arange(counts.size), counts)
        k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        occupied[lo[0, e] + k % span[0, e], lo[1, e] + k // span[0, e]] = 1
        pc = np.floor(world.P / cell).astype(int)
        on = (pc[0] >= 0) & (pc[0] < cols) & (pc[1] >= 0) & (pc[1] < rows)
        occupied[pc[0, on], pc[1, on]] = 2
    return cell * zoom, occupied


def grid_axis(n_px, n_cells, cell_px):
    """(n_px, 2) int32 per pixel along one axis: (its cell, 1 if a grid line is on it). Cell i covers
    [floor(i * cell_px), floor((i + 1) * cell_px)); lines sit at floor(i * cell_px) for 0 < i < n_cells."""
    bounds = np.floor(np.arange(n_cells + 1) * cell_px).astype(np.int64)
    x = np.arange(n_px)
    cell = np.clip(np.searchsorted(bounds, x, "right") - 1, 0, max(n_cells - 1, 0))
    line = np.isin(x, bounds[1:n_cells])
    return np.stack([cell, line], axis=1).astype(np.int32)


class GLWorld:
    """The world layer: `tex` (RGBA8, the render surface's size, row 0 = top) after render()."""

    def __init__(self, ctx, size):
        self.ctx, self.size = ctx, tuple(size)
        self.tex = ctx.texture(self.size, 4)
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth = ctx.depth_renderbuffer(self.size)
        self.fbo = ctx.framebuffer([self.tex], self.depth)
        self.trail_tex = ctx.texture(self.size, 4)
        self.trail_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.trail_fbo = ctx.framebuffer([self.trail_tex])
        self.progs = {
            "base": _load(ctx, "world_base.vert", "world_base.frag"),
            "fill": _load(ctx, "world_fill.vert", "world_fill.frag"),
            "line": _load(ctx, "world_line.vert", "world_line.frag"),
            "stamp": _load(ctx, "world_stamp.vert", "world_stamp.frag"),
            "points": _load(ctx, "world_points.vert", "world_points.frag"),
        }
        for prog in self.progs.values():
            if "u_size" in prog:
                prog["u_size"].value = self.size
        self.base_vao = ctx.vertex_array(self.progs["base"], [])
        self.fills = _Stream(ctx, self.progs["fill"], "4i 4i/i", ("in_info", "in_box"))
        self.lines = _Stream(ctx, self.progs["line"], "4i 4i/i", ("in_ends", "in_style"))
        self.stamps = _Stream(ctx, self.progs["stamp"], "4i 4i 2i/i", ("in_pos", "in_mask", "in_style"))
        self.points = _Stream(ctx, self.progs["points"], "2i 4i", ("in_pos", "in_col"))
        self.verts_tex = None
        self.atlas = CircleAtlas(ctx)
        self.packer = WorldPacker()
        self.grid_key, self.grid_tex = None, {
            "x": ctx.texture((1, 1), 2, dtype="i4"), "y": ctx.texture((1, 1), 2, dtype="i4"),
            "cells": ctx.texture((1, 1), 1, dtype="u1")}
        for t in self.grid_tex.values():
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        base = self.progs["base"]
        for name, unit in (("u_xmap", 0), ("u_ymap", 1), ("u_cells", 2), ("u_trail", 3)):
            base[name].value = unit
        self.progs["fill"]["u_verts"].value = 0
        self.progs["stamp"]["u_atlas"].value = 0
        self.stats = {}

    # ---------------- passes
    def _set_clip(self, clip):
        for prog in self.progs.values():
            if "u_clip" in prog:
                prog["u_clip"].value = clip

    def _trails(self, trails, clip, zoom):
        px = trails.pixels(clip, zoom) if trails is not None else None
        if px is None:
            return False
        xs, ys, rgb, alpha = px
        n = self.points.write(np.column_stack([xs, ys, rgb, alpha]))
        self.trail_fbo.use()
        self.trail_fbo.clear(0.0, 0.0, 0.0, 0.0)
        if n:
            self.points.vao.render(moderngl.POINTS, vertices=n)
        return True

    def _grid(self, world, clip, zoom, colors):
        base = self.progs["base"]
        cell_px, occupied = grid_cells(world, clip[0], clip[1], zoom)
        cols, rows = occupied.shape
        key = (clip, cols, rows, cell_px)
        if key != self.grid_key:
            for t in self.grid_tex.values():
                t.release()
            self.grid_tex = {
                "x": self.ctx.texture((clip[0], 1), 2, grid_axis(clip[0], cols, cell_px).tobytes(), dtype="i4"),
                "y": self.ctx.texture((clip[1], 1), 2, grid_axis(clip[1], rows, cell_px).tobytes(), dtype="i4"),
                "cells": self.ctx.texture((cols, rows), 1, dtype="u1"),
            }
            for t in self.grid_tex.values():
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.grid_key = key
        self.grid_tex["cells"].write(np.ascontiguousarray(occupied.T).astype(np.uint8).tobytes())
        self.grid_tex["x"].use(0)
        self.grid_tex["y"].use(1)
        self.grid_tex["cells"].use(2)
        cols_ = dict(GRID_DEFAULTS, **(colors or {}))
        base["u_line_col"].value = tuple(float(c) for c in cols_["line_col"])
        base["u_edge_col"].value = tuple(float(c) for c in cols_["edge_cell_col"])
        base["u_point_col"].value = tuple(float(c) for c in cols_["point_cell_col"])

    def render(self, world, view_size, filled, zoom, background, grid=None, trails=None, outlines=True):
        """Draw the world into self.tex. view_size: the world view's size in px (drawing is clipped to it, like the
        subsurface the CPU path draws into); background: RGB; grid: None (off) or draw_grid colour overrides ({} for
        its defaults); trails: a trails.Trails (drawn when it has pixels) or None."""
        ctx = self.ctx
        prev = ctx.fbo if ctx.fbo is not None else ctx.screen
        clip = (int(min(view_size[0], self.size[0])), int(min(view_size[1], self.size[1])))
        self._set_clip(clip)
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        has_trails = self._trails(trails, clip, zoom)
        self.fbo.use()
        self.fbo.clear(depth=1.0)
        base = self.progs["base"]
        base["u_background"].value = tuple(float(c) for c in tuple(background)[:3])
        base["u_grid"].value = grid is not None
        if grid is not None:
            self._grid(world, clip, zoom, grid)
        else:
            for unit, name in enumerate(("x", "y", "cells")):  # keep every sampler bound to a texture of its type
                self.grid_tex[name].use(unit)
        base["u_trails"].value = has_trails
        self.trail_tex.use(3)
        self.base_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self._bodies(world, filled, zoom, outlines)
        if prev is not None:
            prev.use()

    def _bodies(self, world, filled, zoom, outlines=True):
        ctx = self.ctx
        data = self.packer.pack(world, filled, zoom, outlines)
        fills, verts, lines, stamps = data["fills"], data["verts"], data["lines"], data["stamps"]
        self.stats = {"fills": len(fills), "lines": len(lines), "stamps": len(stamps)}
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        if len(fills):
            rows = -(-len(verts) // VERT_ROW)
            if self.verts_tex is None or self.verts_tex.size[1] < rows:
                cap = max(rows, 2 * self.verts_tex.size[1]) if self.verts_tex is not None else rows
                if self.verts_tex is not None:
                    self.verts_tex.release()
                self.verts_tex = ctx.texture((VERT_ROW, cap), 2, dtype="i4")
                self.verts_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            buf = np.zeros((rows * VERT_ROW, 2), np.int32)
            buf[:len(verts)] = verts
            self.verts_tex.write(buf.tobytes(), viewport=(0, 0, VERT_ROW, rows))
            self.verts_tex.use(0)
            n = self.fills.write(fills)
            self.fills.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        if len(lines):
            n = self.lines.write(lines)
            ctx.enable(moderngl.BLEND)  # translucent grab lines; opaque ones (alpha 1) draw as before
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ZERO, moderngl.ONE
            self.lines.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
            ctx.disable(moderngl.BLEND)
        if len(stamps):
            inst = self._stamps(stamps, data["radii"])
            if len(inst):
                self.atlas.upload()
                self.atlas.tex.use(0)
                n = self.stamps.write(inst)
                self.stamps.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        ctx.disable(moderngl.DEPTH_TEST)

    def _stamps(self, stamps, radii):
        """(k, 10) [cx, cy, offx, offy, atlas x, atlas y, w, h, colour, layer] for the circles pygame would draw."""
        keys, inv = np.unique(np.column_stack([radii, stamps[:, 4]]), axis=0, return_inverse=True)
        table = np.zeros((len(keys), 6), np.int64)
        ok = np.zeros(len(keys), bool)
        for k, (rad, width) in enumerate(keys):
            m = self.atlas.get(rad, width)
            if m is not None:
                table[k], ok[k] = m, True
        inv = np.asarray(inv).reshape(-1)
        keep = ok[inv]
        m = table[inv[keep]]
        s = stamps[keep]
        return np.column_stack([s[:, 0], s[:, 1], m[:, 4], m[:, 5], m[:, 0], m[:, 1], m[:, 2], m[:, 3], s[:, 2], s[:, 3]])
